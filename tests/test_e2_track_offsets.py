import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from e2_track_offsets import composite_offset, fit_offsets


def moments(values, tracks, shape):
    """values, tracks: (rows, H, W) with track index per row."""
    n, s, ss = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for r in range(values.shape[0]):
        for i in range(shape[1]):
            for j in range(shape[2]):
                t = tracks[r, i, j]
                n[t, i, j] += 1
                s[t, i, j] += values[r, i, j]
                ss[t, i, j] += values[r, i, j] ** 2
    return n, s, ss


def synthetic(offset_sd, seed=0, rows=80, H=20, W=20, T=2, noise=0.1):
    rng = np.random.default_rng(seed)
    true = rng.normal(0, offset_sd, (T, H, W)) if offset_sd else np.zeros((T, H, W))
    true -= true.mean(0)
    tracks = np.tile((np.arange(rows) % T)[:, None, None], (1, H, W))
    base = rng.normal(0, 0.5, (H, W))                                        # pixel means must not leak into the offsets
    I, J = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    values = np.stack([base + true[tracks[r], I, J] + rng.normal(0, noise, (H, W)) for r in range(rows)])
    return true, *moments(values, tracks, (T, H, W))


def test_recovers_offsets_and_ignores_pixel_means():
    true, n, s, ss = synthetic(0.1)
    off, k, sigma2, tau2 = fit_offsets(n, s, ss)
    assert np.corrcoef(off.ravel(), true.ravel())[0, 1] > 0.9
    assert abs(np.sqrt(sigma2) - 0.1) < 0.01
    assert 0 < k < 5


def test_no_offsets_means_nearly_none_after_shrinkage():
    _, n, s, ss = synthetic(0)
    off, *_ = fit_offsets(n, s, ss)
    unshrunk = s[0] / n[0] - s.sum(0) / n.sum(0)
    assert np.sqrt((off ** 2).mean()) < 0.5 * np.sqrt((unshrunk ** 2).mean())


def test_shrinks_small_cells_more_and_drops_those_under_min_count():
    n = np.array([[[50.0]], [[50.0]], [[6.0]], [[4.0]]])
    means = np.array([[[0.1]], [[-0.1]], [[0.1]], [[9.0]]])
    s = n * means
    ss = n * (means ** 2 + 0.05 ** 2)
    # a second pixel set gives the variance components something to work with
    n2, s2, ss2 = np.tile(n, (1, 1, 30)), np.tile(s, (1, 1, 30)), np.tile(ss, (1, 1, 30))
    off, *_ = fit_offsets(n2, s2, ss2, min_count=5)
    assert off[3, 0, 0] == 0
    assert abs(off[0, 0, 0]) > 0 and off[2, 0, 0] != 0
    raw_big = 0.1 - (50 * 0.1 - 50 * 0.1 + 6 * 0.1) / 106
    assert abs(off[0, 0, 0]) <= abs(raw_big) + 1e-6


def test_all_one_track_gives_zero_offsets():
    n = np.array([[[20.0, 20.0]], [[0.0, 0.0]]])
    s = np.array([[[2.0, 4.0]], [[0.0, 0.0]]])
    off, *_ = fit_offsets(n, s, s * s / 20 + 1)
    assert np.all(off == 0)


def test_composite_offset_is_the_mean_of_its_scenes():
    off = np.array([[[0.1, 0.1]], [[0.3, 0.3]]])
    a, b = np.array([[True, True]]), np.array([[True, False]])
    d = composite_offset(off, [(a, 0), (b, 1)])
    assert np.allclose(d, [[0.2, 0.1]])
    assert np.all(composite_offset(off, []) == 0)
