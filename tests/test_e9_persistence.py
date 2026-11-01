import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from e9_persistence_check import crossfit_beta, demean, gap_bin, pair_demean


def test_gap_bins():
    assert gap_bin(np.array([1, 3, 4, 6, 7, 12, 13, 24, 25, 90])).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_demean_is_per_pixel():
    v = demean(np.array([1.0, 3.0, 10.0, 14.0]), np.array([0, 0, 1, 1]))
    assert v.tolist() == [-1, 1, -2, 2]


def test_crossfit_beta_recovers_slope_without_using_own_fold():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    cell = np.repeat([0, 1], 2000)
    fold = rng.integers(0, 4, 4000)
    y = np.where(cell == 0, 0.5, 0.0) * x
    b = crossfit_beta(y, x, cell, fold)
    assert np.allclose(b[cell == 0], 0.5) and np.allclose(b[cell == 1], 0.0)
    # a fold whose own rows are the only signal contributes nothing to its own beta
    y2 = np.where(fold == 0, x, 0.0)
    assert np.allclose(crossfit_beta(y2, x, np.zeros(4000, dtype=int), fold)[fold == 0], 0.0)


def test_pair_demean_removes_the_short_series_bias_on_white_noise():
    rng = np.random.default_rng(0)
    n, npix = 6, 20000
    series = rng.normal(size=(npix, n))
    r = series[:, 1:].ravel()                                  # rows 1..5 of each pixel, rp = the scene before
    rp = series[:, :-1].ravel()
    pix = np.repeat(np.arange(npix), n - 1)
    naive = np.corrcoef(demean(r, pix), demean(rp, pix))[0, 1]
    a, b = pair_demean(r, rp, pix)
    assert naive < -0.1 and abs(np.corrcoef(a, b)[0, 1]) < 0.03
