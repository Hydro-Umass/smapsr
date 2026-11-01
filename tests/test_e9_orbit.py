import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from e9_orbit_split import assign_tracks, previous_rows, split_pair_corr, track_variance


def test_assign_tracks_groups_by_12_day_phase_and_satellite():
    t0 = pd.Timestamp("2020-03-01 07:00")
    times = [t0, t0 + pd.Timedelta(days=12, minutes=1), t0 + pd.Timedelta(days=5), t0 + pd.Timedelta(days=24), t0]
    tr = assign_tracks(times, ["1A", "1A", "1A", "1A", "1B"])
    assert tr[0] == tr[1] == tr[3] and tr[2] != tr[0] and tr[4] not in (tr[0], tr[2])


def test_track_variance_finds_offsets_and_is_near_zero_without_them():
    rng = np.random.default_rng(0)
    npix, ntrack, nobs = 400, 3, 12
    pix = np.repeat(np.arange(npix), ntrack * nobs)
    tr = np.tile(np.repeat(np.arange(ntrack), nobs), npix)
    cell = pix * ntrack + tr
    noise = rng.normal(size=pix.size)
    share, _ = track_variance(noise, pix, cell)
    assert abs(share) < 0.02
    offsets = rng.normal(size=npix * ntrack)
    share, _ = track_variance(noise + offsets[cell], pix, cell)
    assert 0.35 < share < 0.6       # a component of variance 2/3 (within pixel) against noise of variance 1


def test_previous_rows_same_and_different_track():
    pix = np.array([0, 0, 0, 0, 1])
    day = np.array([0, 5, 12, 17, 3])
    tr = np.array([7, 8, 7, 8, 7])
    same, diff = previous_rows(pix, day, tr)
    assert same.tolist() == [-1, -1, 0, 1, -1] and diff.tolist() == [-1, 0, 1, 2, -1]


def test_split_pair_corr_is_unbiased_on_white_noise():
    rng = np.random.default_rng(0)
    npix, n = 5000, 6
    r = rng.normal(size=npix * n)
    pix = np.repeat(np.arange(npix), n)
    a = np.arange(r.size)[np.arange(r.size) % n > 0]
    assert abs(split_pair_corr(r, pix, a, a - 1, rng)[0]) < 0.03
