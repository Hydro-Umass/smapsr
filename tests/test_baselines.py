import jax.numpy as jnp
import jax.random as jr
import numpy as np

from smapsr.baselines import Climatology, bilinear, block_constant, parent_valid
from smapsr.models import Upsampler


def _field(h=8, w=9, seed=0):
    return np.random.default_rng(seed).uniform(0.1, 0.5, (h, w)).astype(np.float32)


def test_bilinear_matches_the_untrained_v1_upsampler_without_gaps():
    c = _field()
    up = Upsampler(2, (3,), key=jr.key(0))
    x = jnp.stack([jnp.asarray(c), jnp.ones_like(c)])
    ref = np.asarray(up(x))[0]
    got = bilinear(c, 3)
    assert got.shape == (24, 27)
    assert np.abs(got - ref)[2:-2, 2:-2].max() < 1e-5          # interior; edges differ by zero padding in the layer


def test_bilinear_reproduces_coarse_values_at_block_centres_and_constants():
    c = _field()
    assert np.allclose(bilinear(c, 3)[1::3, 1::3], c, atol=1e-6)
    assert np.allclose(bilinear(np.full((5, 5), 0.3, np.float32), 3), 0.3, atol=1e-6)


def test_bilinear_with_gaps_is_nan_only_far_from_valid_cells_and_never_extrapolates():
    c = np.full((7, 7), np.nan, np.float32)
    c[3, 3] = 0.4
    b = bilinear(c, 3)
    assert np.isfinite(b[9:12, 9:12]).all() and np.allclose(b[np.isfinite(b)], 0.4, atol=1e-6)
    assert np.isnan(b[0, 0]) and np.isnan(b[-1, -1])
    assert parent_valid(c, 3).sum() == 9


def test_block_constant():
    c = _field(2, 2)
    assert block_constant(c, 3).shape == (6, 6) and (block_constant(c, 3)[:3, :3] == c[0, 0]).all()


def test_climatology_recovers_a_fixed_offset_and_defines_pixels_by_count():
    rng = np.random.default_rng(1)
    off = np.zeros((12, 12), np.float32)
    off[:, :6] = 0.05                                          # fine-scale pattern the null should learn
    clim = Climatology((12, 12), 3, min_count=5)
    for _ in range(6):
        c = rng.uniform(0.2, 0.4, (4, 4)).astype(np.float32)
        clim.add(c, bilinear(c, 3) + off)
    assert clim.covered.all()
    assert np.allclose(clim.anomaly, off, atol=1e-5)
    c = rng.uniform(0.2, 0.4, (4, 4)).astype(np.float32)
    assert np.allclose(clim.null_model(c), bilinear(c, 3) + off, atol=1e-5)
    few = Climatology((12, 12), 3, min_count=7)
    for _ in range(6):
        few.add(c, bilinear(c, 3) + off)
    assert not few.covered.any() and np.isnan(few.anomaly).all()
    assert np.allclose(few.null_model(c), bilinear(c, 3), atol=1e-6)     # falls back to bilinear
