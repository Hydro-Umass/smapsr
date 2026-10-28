"""Tests of the V1 model, the per-date masked loss and the daily-pair crop sampler (synthetic data only)."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import pytest

from smapsr import evals
from smapsr.data import DailyPairs
from smapsr.losses import masked_daily_loss, masked_ssim
from smapsr.models import NeuralODEV1, Upsampler, bilinear_kernel, scale_embedding
from smapsr.train import train_v1


def n_params(model):
    return sum(a.size for a in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))


def bilinear_reference(c, f=3):
    """Cell-centred bilinear interpolation: fine pixel centres sit (r + 0.5) / f - 0.5 coarse cells from the origin."""
    H, W = c.shape
    ys, xs = (np.arange(f * H) + 0.5) / f - 0.5, (np.arange(f * W) + 0.5) / f - 0.5
    out = np.zeros((f * H, f * W), dtype=np.float32)
    for i, yy in enumerate(ys):
        for j, xx in enumerate(xs):
            y0, x0 = int(np.floor(yy)), int(np.floor(xx))
            fy, fx = yy - y0, xx - x0
            for dy, wy in ((0, 1 - fy), (1, fy)):
                for dx, wx in ((0, 1 - fx), (1, fx)):
                    if 0 <= y0 + dy < H and 0 <= x0 + dx < W:
                        out[i, j] += wy * wx * c[y0 + dy, x0 + dx]
    return out


# --- model ---------------------------------------------------------------------------------------

def test_bilinear_kernel_is_separable_triangle():
    k = bilinear_kernel(3)
    assert k.shape == (5, 5)
    assert np.allclose(k[2], [1 / 3, 2 / 3, 1, 2 / 3, 1 / 3])
    assert np.allclose(k, np.outer(k[2], k[2]))


@pytest.mark.parametrize("strides", [(3,), (3, 3)])
def test_upsampler_starts_as_bilinear_and_scales_the_grid(strides):
    h, w = 8, 9
    c = np.random.default_rng(0).normal(size=(h, w)).astype(np.float32)
    x = jnp.stack([jnp.asarray(c), jnp.ones((h, w))])                  # data channel + validity channel
    up = Upsampler(2, strides, key=jr.PRNGKey(0))
    y = np.asarray(up(x))
    f = int(np.prod(strides))
    assert y.shape == (1, h * f, w * f)
    ref = c
    for s in strides:
        ref = bilinear_reference(ref, s)
    # away from the edge (zero padding) the learned-at-init upsampler is exactly bilinear
    m = 3 * f
    assert np.abs(y[0, m:-m, m:-m] - ref[m:-m, m:-m]).max() < 1e-5


def test_validity_channel_does_not_change_the_initial_state():
    up = Upsampler(2, (3,), key=jr.PRNGKey(0))
    c = jnp.asarray(np.random.default_rng(1).normal(size=(6, 6)), jnp.float32)
    a = up(jnp.stack([c, jnp.ones((6, 6))]))
    b = up(jnp.stack([c, jnp.zeros((6, 6))]))
    assert float(jnp.abs(a - b).max()) == 0.0


def test_parameter_count_does_not_depend_on_the_grid_and_dynamics_start_at_zero():
    model = NeuralODEV1(2, (3,), width=8, depth=3, groups=4, n_steps=3, key=jr.PRNGKey(0))
    for hw in (10, 17):
        x = jnp.asarray(np.random.default_rng(2).normal(size=(2, hw, hw)), jnp.float32)
        u1 = model(x, 3.0)
        assert u1.shape == (1, 3 * hw, 3 * hw)
        assert float(jnp.abs(u1 - model.initial_state(x)).max()) == 0.0    # zero-initialised vector field: u(1) = u(0)
    assert n_params(model) == n_params(NeuralODEV1(2, (3,), width=8, depth=3, groups=4, n_steps=3, key=jr.PRNGKey(5)))


def test_scale_conditioning_reaches_the_dynamics():
    model = NeuralODEV1(2, (3,), width=8, depth=3, groups=4, n_steps=3, key=jr.PRNGKey(0))
    # break the zero initialisation so the vector field is non-trivial
    leaves, tree = jax.tree_util.tree_flatten(eqx.filter(model.func, eqx.is_inexact_array))
    keys = jr.split(jr.PRNGKey(7), len(leaves))
    func = jax.tree_util.tree_unflatten(tree, [0.3 * jr.normal(k, a.shape) for k, a in zip(keys, leaves)])
    model = eqx.tree_at(lambda m: m.func, model, eqx.combine(func, eqx.filter(model.func, lambda a: not eqx.is_inexact_array(a))))
    x = jnp.asarray(np.random.default_rng(3).normal(size=(2, 12, 12)), jnp.float32)
    assert float(jnp.abs(model(x, 3.0) - model.initial_state(x)).max()) > 1e-4
    assert float(jnp.abs(model(x, 3.0) - model(x, 1.0)).max()) > 1e-5


def test_scale_embedding_shape():
    assert scale_embedding(3.0, 4).shape == (9,)


def test_save_load_roundtrip(tmp_path):
    hyper = dict(in_channels=2, strides=[3], width=8, depth=3, groups=4, n_freq=4, n_steps=3, data_mean=0.3, data_std=0.14)
    model, _, _ = evals.make_v1(key=jr.PRNGKey(0), **hyper)
    evals.save(tmp_path / "m.eqx", hyper, model)
    loaded, mean, std = evals.load_v1(tmp_path / "m.eqx")
    x = jnp.asarray(np.random.default_rng(4).normal(size=(2, 9, 9)), jnp.float32)
    assert (mean, std) == (0.3, 0.14)
    assert float(jnp.abs(loaded(x, 3.0) - model(x, 3.0)).max()) == 0.0


# --- loss ----------------------------------------------------------------------------------------

def make_fields(seed=0, shape=(3, 40, 40), keep=0.4):
    rng = np.random.default_rng(seed)
    pred = jnp.asarray(rng.normal(size=shape), jnp.float32)
    y = pred + 0.3 * jnp.asarray(rng.normal(size=shape), jnp.float32)
    mask = jnp.asarray(rng.random(shape) < keep, jnp.float32)
    return pred, y, mask


def test_masked_daily_loss_ignores_nan_labels_and_masked_pixels():
    pred, y, mask = make_fields()
    y_nan = jnp.where(mask > 0, y, jnp.nan)
    loss, grad = jax.value_and_grad(lambda p: masked_daily_loss(p, y_nan, mask, data_range=4.0))(pred)
    assert np.isfinite(float(loss)) and bool(jnp.isfinite(grad).all())
    assert float(jnp.abs(grad * (1 - mask)).max()) < 1e-6                 # no gradient where there is no label
    assert float(loss) == pytest.approx(float(masked_daily_loss(pred, jnp.where(mask > 0, y, 0.0), mask, 4.0)))


def test_masked_daily_loss_skips_dates_without_labels():
    pred, y, mask = make_fields(shape=(2, 40, 40), keep=0.5)
    both = float(masked_daily_loss(pred[:1], y[:1], mask[:1], 4.0))
    padded = float(masked_daily_loss(pred, y, mask.at[1].set(0.0), 4.0))
    assert padded == pytest.approx(both, rel=1e-5)


def test_ssim_validity_threshold_changes_only_when_asked():
    pred, y, mask = make_fields(keep=0.15)
    default = masked_ssim(pred, y, mask)
    assert float(jnp.abs(default - masked_ssim(pred, y, mask, min_valid_fraction=0.0)).max()) == 0.0
    assert float(jnp.abs(default - masked_ssim(pred, y, mask, min_valid_fraction=0.5)).max()) > 0.0


# --- sampler -------------------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    """Synthetic aligned cache: 24x24 coarse, 72x72 fine, labels = block-constant coarse field + noise, with gaps."""
    rng = np.random.default_rng(0)
    (tmp_path / "9km").mkdir()
    (tmp_path / "3km").mkdir()
    for d in pd.date_range("2021-03-01", periods=6):
        c = rng.uniform(0.1, 0.5, (24, 24)).astype(np.float32)
        c[rng.random((24, 24)) < 0.3] = np.nan
        f = np.kron(c, np.ones((3, 3), np.float32))
        f[rng.random(f.shape) < 0.5] = np.nan
        np.savez(tmp_path / "9km" / f"{d:%Y%m%d}.npz", soil_moisture=c, mask=np.isfinite(c))
        np.savez(tmp_path / "3km" / f"{d:%Y%m%d}.npz", soil_moisture=f, mask=np.isfinite(f))
    # one date with a label file but no input file, one empty date
    np.savez(tmp_path / "3km" / "20210309.npz", soil_moisture=np.full((72, 72), 0.3, np.float32), mask=np.ones((72, 72), bool))
    np.savez(tmp_path / "9km" / "20210310.npz", soil_moisture=np.full((24, 24), np.nan, np.float32), mask=np.zeros((24, 24), bool))
    np.savez(tmp_path / "3km" / "20210310.npz", soil_moisture=np.full((72, 72), 0.3, np.float32), mask=np.ones((72, 72), bool))
    return tmp_path


def test_pairs_skip_unusable_dates_and_normalise_with_given_stats(cache):
    dates = pd.date_range("2021-03-01", "2021-03-12")
    train = DailyPairs(cache, dates[:4], crop=8, min_label_frac=0.1, min_input_frac=0.2)
    assert len(train) == 4
    other = DailyPairs(cache, dates, crop=8, min_label_frac=0.1, min_input_frac=0.2, norm=(train.mean, train.std))
    assert len(other) == 6                                                   # 03-09 (no input) and 03-10 (empty input) skipped
    assert (other.mean, other.std) == (train.mean, train.std)


def test_crops_are_aligned_masked_and_never_nan(cache):
    dp = DailyPairs(cache, pd.date_range("2021-03-01", periods=6), crop=8, min_label_frac=0.1, min_input_frac=0.2)
    x, y, m, dates = dp.sample(32, np.random.default_rng(1))
    assert x.shape == (32, 2, 8, 8) and y.shape == (32, 1, 24, 24) and m.shape == (32, 24, 24)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert (y[:, 0][m == 0] == 0).all()                                      # zero-filled where there is no label
    assert (x[:, 0][x[:, 1] == 0] == 0).all()                                # input gaps are the training mean (0), flagged in channel 1
    # every label pixel sits in its own coarse cell: the block-constant labels equal the coarse value after normalisation
    both = (m.reshape(32, 8, 3, 8, 3).mean((2, 4)) > 0) & (x[:, 1] > 0)
    blocks = y[:, 0].reshape(32, 8, 3, 8, 3)
    mb = m.reshape(32, 8, 3, 8, 3)
    mean_label = (blocks * mb).sum((2, 4)) / np.maximum(mb.sum((2, 4)), 1)
    assert np.allclose(mean_label[both], x[:, 0][both], atol=1e-5)


def test_eligibility_thresholds_and_label_coverage(cache):
    dp = DailyPairs(cache, pd.date_range("2021-03-01", periods=6), crop=8, min_label_frac=0.3, min_input_frac=0.5)
    _, _, m, _ = dp.sample(64, np.random.default_rng(2))
    x, _, _, _ = dp.sample(64, np.random.default_rng(2))
    assert (m.mean((1, 2)) >= 0.3 - 1e-9).all() and (x[:, 1].mean((1, 2)) >= 0.5 - 1e-9).all()
    with pytest.raises(ValueError):
        DailyPairs(cache, pd.date_range("2021-03-01", periods=6), crop=8, min_label_frac=0.99, min_input_frac=0.99)


def test_fixed_batches_are_reproducible(cache):
    dp = DailyPairs(cache, pd.date_range("2021-03-01", periods=6), crop=8, min_label_frac=0.1, min_input_frac=0.2)
    a, b = dp.fixed(8, seed=3), dp.fixed(8, seed=3)
    assert all(np.array_equal(p, q) for p, q in zip(a[:3], b[:3]))


# --- training smoke ------------------------------------------------------------------------------

def test_train_v1_smoke_runs_and_updates_parameters(cache):
    dp = DailyPairs(cache, pd.date_range("2021-03-01", periods=6), crop=8, min_label_frac=0.1, min_input_frac=0.2)
    model = NeuralODEV1(2, (3,), width=8, depth=2, groups=4, n_steps=2, key=jr.PRNGKey(0))
    seen = []
    trained, history = train_v1(model, dp, scale_km=3.0, steps=4, batch_size=2, seed=0, print_every=100,
                                eval_every=2, eval_fn=lambda m, step: seen.append(step) or dict(val=float(step)))
    assert len(history) == 4 and all(np.isfinite(h["loss"]) for h in history)
    assert seen == [2, 4]
    before = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    after = jax.tree_util.tree_leaves(eqx.filter(trained, eqx.is_inexact_array))
    assert any(float(jnp.abs(a - b).max()) > 0 for a, b in zip(before, after))


def test_kernel_size_changes_the_vector_field_only_and_roundtrips(tmp_path):
    from smapsr import evals
    import equinox as eqx
    m3 = NeuralODEV1(2, (3,), width=8, depth=2, groups=4, n_steps=2, kernel=3, key=jr.PRNGKey(0))
    m5 = NeuralODEV1(2, (3,), width=8, depth=2, groups=4, n_steps=2, kernel=5, key=jr.PRNGKey(0))
    assert m5.func.convs[1].weight.shape[-1] == 5 and m3.func.convs[1].weight.shape[-1] == 3
    assert m5.upsampler.layers[0].weight.shape == m3.upsampler.layers[0].weight.shape
    hyper = dict(in_channels=2, strides=[3], width=8, depth=2, groups=4, n_freq=4, n_steps=2, kernel=5, data_mean=0.3, data_std=0.1)
    evals.save(tmp_path / "m.eqx", hyper, m5)
    back, _, _ = evals.load_v1(tmp_path / "m.eqx")
    assert back.func.convs[0].weight.shape[-1] == 5
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(back, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(m5, eqx.is_array))))
