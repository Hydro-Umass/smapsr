import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import e9_noise_ceiling as e9  # noqa: E402


def test_triple_collocation_recovers_errors():
    rng = np.random.default_rng(0)
    t = rng.normal(size=200_000)
    x = t + rng.normal(scale=0.5, size=t.size)
    y = 2 * t + rng.normal(scale=0.2, size=t.size)
    z = 0.5 * t + rng.normal(scale=0.3, size=t.size)
    r2, err = e9.triple_collocation(x, y, z)
    assert abs(err["label"] - 0.5) < 0.01 and abs(err["smap9km"] - 0.2) < 0.02 and abs(err["insitu"] - 0.3) < 0.01
    assert abs(r2["label"] - 1 / 1.25) < 0.01


def test_nugget_of_white_noise_on_smooth_field():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:200, 0:200]
    smooth = np.sin(xx / 15.0) + np.cos(yy / 20.0)
    field = smooth + rng.normal(scale=0.3, size=smooth.shape)
    g = e9.semivariogram_sums(field)
    for ax in (0, 1):
        gamma = g[ax, :, 1] / (2 * g[ax, :, 0])
        assert abs(e9.nugget(gamma) - 0.09) < 0.01


def test_semivariogram_ignores_nan():
    a = np.full((10, 10), np.nan)
    a[2:5, 2:5] = 1.0
    g = e9.semivariogram_sums(a)
    assert g[0, 0, 0] == 6 and g[0, 0, 1] == 0
