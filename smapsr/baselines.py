"""Training-free E1 baselines on the aligned grids (numpy; no model state).

All baselines map one day's coarse field (H, W; NaN = no retrieval) to the fine grid (H*f, W*f). They are
defined only where a coarse parent cell is valid: predictions are NaN elsewhere, and the evaluation set is
restricted to those pixels (the same "parent-valid" rule as the V1 monitoring metrics).
"""
import numpy as np
from scipy.signal import convolve2d

from smapsr.models import bilinear_kernel


def _stuff(a, factor):
    """Place `a` at the centres of factor x factor blocks of a zero grid (cell-centred layout of the aligned grids)."""
    out = np.zeros((a.shape[0] * factor, a.shape[1] * factor), dtype=np.float64)
    c = (factor - 1) // 2 if factor % 2 else factor // 2 - 1       # block-centre index (odd factor: exact centre)
    out[c::factor, c::factor] = a
    return out


def bilinear(coarse, factor):
    """Cell-centred bilinear interpolation, renormalised over valid coarse cells (normalised convolution).

    Equals plain bilinear interpolation wherever all contributing coarse cells are valid, and is the weighted mean
    of the valid ones otherwise. NaN where no valid coarse cell contributes. Same kernel as the V1 initial upsampler
    (`models.bilinear_kernel`), which makes an untrained V1 equal to this baseline when the input has no gaps."""
    if factor % 2 == 0:
        raise NotImplementedError("even factors: the aligned cache only uses 3 and 9 (odd)")
    k = bilinear_kernel(factor)
    valid = np.isfinite(coarse)
    num = convolve2d(_stuff(np.where(valid, coarse, 0.0), factor), k, mode="same")
    den = convolve2d(_stuff(valid.astype(np.float64), factor), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return out.astype(np.float32)


def block_constant(coarse, factor):
    """Each coarse value repeated over its factor x factor block (the 'no downscaling' reference)."""
    return np.kron(coarse, np.ones((factor, factor), dtype=coarse.dtype))


def parent_valid(coarse, factor):
    """Boolean fine-grid mask: the pixel's own coarse parent cell is valid."""
    return np.kron(np.isfinite(coarse), np.ones((factor, factor), dtype=bool))


class Climatology:
    """Training-period climatology of the labels and of the label-minus-bilinear anomaly.

    Accumulated over (coarse, fine) day pairs with `add`. Per fine pixel, over days where the label is valid and
    the pixel's parent is valid:
        label_mean  C = mean(L)            (the day-independent label climatology)
        anomaly     A = mean(L - B_t)      (B_t = bilinear of that day's coarse field)
    Pixels with fewer than `min_count` such days are undefined (NaN).
    `add(..., date=d)` also keeps that day's contributions, so `fields` can leave the day out (training inputs)."""

    def __init__(self, shape, factor, min_count=5):
        self.factor, self.min_count = factor, min_count
        self.n = np.zeros(shape, dtype=np.int32)
        self.sum_l = np.zeros(shape, dtype=np.float64)
        self.sum_d = np.zeros(shape, dtype=np.float64)
        self.contrib = {}

    def add(self, coarse, fine, date=None):
        b = bilinear(coarse, self.factor)
        ok = np.isfinite(fine) & np.isfinite(b)
        self.n += ok
        self.sum_l += np.where(ok, fine, 0.0)
        self.sum_d += np.where(ok, fine - b, 0.0)
        if date is not None:
            r, c = np.nonzero(ok)
            self.contrib[date] = (r.astype(np.int32), c.astype(np.int32), fine[ok].astype(np.float64), (fine - b)[ok].astype(np.float64))

    def fields(self, mean, std, rows=slice(None), cols=slice(None), leave_out=None):
        """Static input channels on a window of the fine grid: [(C - mean) / std, A / std, valid] (3, h, w), float32.

        C and A as above, 0 where undefined (valid = 0); mean/std are the model's normalisation (A is a difference,
        so it is only scaled). `leave_out=date` first removes that day's contribution (leave-one-out climatology,
        so a training label never reaches the model through its own input)."""
        n, sl, sd = self.n[rows, cols].astype(np.float64), self.sum_l[rows, cols].copy(), self.sum_d[rows, cols].copy()
        if leave_out is not None and leave_out in self.contrib:
            r, c, l, d = self.contrib[leave_out]
            r0, c0 = rows.start or 0, cols.start or 0
            inside = (r >= r0) & (r < r0 + n.shape[0]) & (c >= c0) & (c < c0 + n.shape[1])
            r, c = r[inside] - r0, c[inside] - c0
            np.subtract.at(n, (r, c), 1)
            np.subtract.at(sl, (r, c), l[inside])
            np.subtract.at(sd, (r, c), d[inside])
        ok = n >= self.min_count
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.where(ok, (sl / n - mean) / std, 0.0)
            A = np.where(ok, sd / n / std, 0.0)
        return np.stack([C, A, ok]).astype(np.float32)

    def save(self, path):
        np.savez_compressed(path, n=self.n, sum_l=self.sum_l, sum_d=self.sum_d, factor=self.factor, min_count=self.min_count)

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            c = cls(z["n"].shape, int(z["factor"]), int(z["min_count"]))
            c.n, c.sum_l, c.sum_d = z["n"], z["sum_l"], z["sum_d"]
        return c

    @property
    def covered(self):
        return self.n >= self.min_count

    @property
    def label_mean(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.covered, self.sum_l / self.n, np.nan).astype(np.float32)

    @property
    def anomaly(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.covered, self.sum_d / self.n, np.nan).astype(np.float32)

    def null_model(self, coarse):
        """Climatology null: bilinear + training time-mean anomaly; plain bilinear where the anomaly is undefined."""
        b = bilinear(coarse, self.factor)
        a = self.anomaly
        return b + np.where(np.isfinite(a), a, 0.0).astype(np.float32)
