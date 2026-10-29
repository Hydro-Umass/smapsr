"""Daily-pair crop sampler for the V1 protocol (amendment A1).

A sample is one date: the coarse SMAP-E field (9 km) and the same-day SPL2SMAP_S label, both read from the
aligned cache (every coarse cell is an exact factor x factor block of fine cells, so a crop of c coarse cells is
a crop of c * factor fine cells with no resampling). Differences from V0's `prepare_data`:

* labels are never gap-filled; the per-date validity mask travels with the sample and labels are zero-filled
  where invalid (NaN * 0 = NaN in JAX);
* the coarse input is not gap-filled either: invalid cells are set to 0 (the training mean, in normalised units)
  and a validity channel is returned next to the data channel;
* crops are random (no fixed geographic tile), drawn only where the crop holds enough labels *on the fine grid*
  and enough valid input cells, with probability proportional to the label count;
* normalisation (mean, std) comes from the valid coarse pixels of the dates given at construction and is applied
  unchanged to every other set (pass `norm=`), which is what a zero-shot protocol needs;
* optional static fine-grid channels (`set_static`): the training label climatology (`baselines.Climatology`:
  label mean C, label-minus-bilinear anomaly A, validity), left-one-out on the dates that built it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CLIP = (0.01, 0.65)     # m3/m3, as in V0
STATIC_CHANNELS = {"label_mean": 0, "anomaly": 1}   # index into Climatology.fields; validity (2) is always appended


def _read(path):
    with np.load(path) as z:
        return z["soil_moisture"].astype(np.float32)


def _window_sums(a, crop, stride):
    """Sum of `a` over every crop x crop window with origins on a stride grid (integral image)."""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    i = np.arange(0, a.shape[0] - crop + 1, stride)
    j = np.arange(0, a.shape[1] - crop + 1, stride)
    w = s[np.ix_(i + crop, j + crop)] - s[np.ix_(i, j + crop)] - s[np.ix_(i + crop, j)] + s[np.ix_(i, j)]
    return i, j, w


class DailyPairs:
    """Random-crop sampler over daily (coarse, fine) pairs.

    cache: directory holding <coarse>/ and <fine>/ folders of <YYYYMMDD>.npz (keys soil_moisture, mask).
    dates: candidate dates; those missing at either scale, or with no valid pixel at either scale, are skipped.
    crop: crop size in coarse cells (fine crop = crop * factor).
    min_label_frac / min_input_frac: a crop is eligible if at least this fraction of its fine pixels carries a label
        and of its coarse cells a valid input value.
    origin_stride: crop origins are taken every `origin_stride` coarse cells.
    norm: (mean, std) to use; computed from this set's valid coarse pixels when None.
    """

    def __init__(self, cache, dates, coarse="9km", fine="3km", factor=3, crop=32, min_label_frac=0.05,
                 min_input_frac=0.10, origin_stride=2, norm=None, clip=CLIP):
        self.factor, self.crop, self.clip = factor, crop, clip
        cache = Path(cache)
        self.dates, self.coarse, self.fine = [], [], []
        for d in pd.DatetimeIndex(dates):
            cp, fp = cache / coarse / f"{d:%Y%m%d}.npz", cache / fine / f"{d:%Y%m%d}.npz"
            if not (cp.exists() and fp.exists()):
                continue
            c, f = np.clip(_read(cp), *clip), np.clip(_read(fp), *clip)       # np.clip keeps NaN
            if f.shape != (c.shape[0] * factor, c.shape[1] * factor):
                raise ValueError(f"{d:%Y%m%d}: fine {f.shape} is not {factor}x coarse {c.shape}")
            if not (np.isfinite(c).any() and np.isfinite(f).any()):
                continue
            self.dates.append(d)
            self.coarse.append(c)
            self.fine.append(f)
        if not self.dates:
            raise ValueError("no usable dates")
        if norm is None:
            v = np.concatenate([c[np.isfinite(c)] for c in self.coarse]).astype(np.float64)
            norm = (float(v.mean()), float(v.std()))
        self.mean, self.std = norm
        self.clim, self.static = None, ()
        self._index_candidates(min_label_frac, min_input_frac, origin_stride)

    def set_static(self, clim, names):
        """Attach static channels `names` (keys of STATIC_CHANNELS; validity is appended whenever any is chosen).
        Dates that built `clim` (added with date=) see it with their own day left out."""
        bad = set(names) - set(STATIC_CHANNELS)
        if bad:
            raise ValueError(f"unknown static channels {bad}")
        self.clim, self.static = clim, tuple(names)

    @property
    def static_channels(self):
        return len(self.static) + 1 if self.static else 0

    def static_crop(self, k, rows, cols):
        if not self.static:
            return np.zeros((0, rows.stop - rows.start, cols.stop - cols.start), dtype=np.float32)
        f = self.clim.fields(self.mean, self.std, rows, cols, leave_out=self.dates[k])
        return f[[STATIC_CHANNELS[n] for n in self.static] + [2]]

    def _index_candidates(self, min_label_frac, min_input_frac, stride):
        F, cr = self.factor, self.crop
        di, ii, jj, ww = [], [], [], []
        for k, (c, f) in enumerate(zip(self.coarse, self.fine)):
            H, W = c.shape
            lab = np.isfinite(f).reshape(H, F, W, F).sum((1, 3))              # labelled fine pixels per coarse cell
            i, j, nl = _window_sums(lab, cr, stride)
            _, _, ni = _window_sums(np.isfinite(c), cr, stride)
            ok = (nl >= min_label_frac * (cr * F) ** 2) & (ni >= min_input_frac * cr ** 2)
            a, b = np.nonzero(ok)
            di.append(np.full(len(a), k, dtype=np.int32))
            ii.append(i[a].astype(np.int16))
            jj.append(j[b].astype(np.int16))
            ww.append(nl[a, b].astype(np.float32))
        self._d, self._i, self._j, self._w = map(np.concatenate, (di, ii, jj, ww))
        if len(self._w) == 0:
            raise ValueError("no eligible crop; relax min_label_frac / min_input_frac")
        self._cdf = np.cumsum(self._w, dtype=np.float64)                       # sampling weight = label count

    def __len__(self):
        return len(self.dates)

    @property
    def data_range(self):
        """Dynamic range of the clipped soil moisture in normalised units (SSIM data range)."""
        return (self.clip[1] - self.clip[0]) / self.std

    def summary(self):
        eligible_dates = np.unique(self._d).size
        return dict(n_dates=len(self.dates), n_dates_with_eligible_crop=int(eligible_dates), n_candidate_crops=int(len(self._w)),
                    first=str(self.dates[0].date()), last=str(self.dates[-1].date()), mean=self.mean, std=self.std,
                    crop_coarse=self.crop, crop_fine=self.crop * self.factor)

    def crop_at(self, k, i, j):
        """Normalised crop of date index k at coarse origin (i, j): x (2, c, c), y (1, cF, cF), mask (cF, cF),
        s (static_channels, cF, cF).

        x[0] is the coarse data (0 where invalid), x[1] its validity; y is zero-filled where mask == 0."""
        F, cr = self.factor, self.crop
        c = self.coarse[k][i:i + cr, j:j + cr]
        f = self.fine[k][i * F:(i + cr) * F, j * F:(j + cr) * F]
        cv, fv = np.isfinite(c), np.isfinite(f)
        x = np.stack([np.where(cv, (c - self.mean) / self.std, 0.0), cv.astype(np.float32)]).astype(np.float32)
        y = np.where(fv, (f - self.mean) / self.std, 0.0)[None].astype(np.float32)
        s = self.static_crop(k, slice(i * F, (i + cr) * F), slice(j * F, (j + cr) * F))
        return x, y, fv.astype(np.float32), s

    def sample(self, batch_size, rng):
        """Draw a batch (with replacement); returns x (B,2,c,c), y (B,1,cF,cF), mask (B,cF,cF), the dates and the
        static channels s (B, static_channels, cF, cF; zero channels when none are set)."""
        k = np.minimum(np.searchsorted(self._cdf, rng.random(batch_size) * self._cdf[-1], side="right"), len(self._w) - 1)
        xs, ys, ms, ss = zip(*(self.crop_at(self._d[n], int(self._i[n]), int(self._j[n])) for n in k))
        return np.stack(xs), np.stack(ys), np.stack(ms), [self.dates[self._d[n]] for n in k], np.stack(ss)

    def fixed(self, n, seed=0):
        """A reproducible batch of `n` crops (validation / monitoring)."""
        return self.sample(n, np.random.default_rng(seed))
