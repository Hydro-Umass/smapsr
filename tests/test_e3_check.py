import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from e3_recursive_check import block_detail, lag_sums, radial_psd, second_stage_input, valid_windows


def test_block_detail_removes_the_block_mean_of_the_given_pixels_only():
    shape = (6, 6)
    idx = np.ravel_multi_index(([0, 0, 2, 3], [0, 1, 2, 3]), shape)   # two in block (0,0), one more there, one alone
    d = block_detail(idx, np.array([1.0, 2.0, 6.0, 9.0]), shape)
    assert np.allclose(d[:3], [-2, -1, 3]) and np.isnan(d[3])


def test_lag_sums_use_only_pairs_valid_at_both_ends():
    o = np.arange(16.0).reshape(4, 4)
    ev = np.ones_like(o, dtype=bool)
    ev[0, 0] = False
    s = lag_sums(o, o * 2, ev, lags=(1,))
    assert s[0, 0] == 24 - 2                # 12 + 12 pairs, two touch the masked cell
    assert np.isclose(s[0, 2], 4 * s[0, 1])


def test_second_stage_input_zeroes_gaps_and_normalises():
    y = np.array([[0.3, 0.5]], dtype=np.float32)
    x = second_stage_input(y, np.array([[True, False]]), 0.3, 0.1)
    assert x.shape == (2, 1, 2) and np.allclose(x[0], [[0, 0]]) and np.allclose(x[1], [[1, 0]])


def test_valid_windows_and_radial_psd():
    ev = np.zeros((8, 8), dtype=bool)
    ev[4:, 4:] = True
    assert valid_windows(ev, 4) == [(4, 4)]
    w = 32
    x = np.sin(2 * np.pi * 4 * np.arange(w) / w)[None] * np.ones((w, 1))
    assert radial_psd(x).argmax() == 3      # wavenumber 4 is bin index 3
