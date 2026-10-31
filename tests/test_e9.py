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


def _strata():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import e9_quality_strata as q
    return q


def test_composite_flags_count_only_scenes_with_a_valid_value():
    q = _strata()
    nan = np.nan
    sm1, sm2 = np.array([[0.2, nan, 0.3]]), np.array([[nan, 0.4, 0.5]])
    s1, s2 = np.array([[4, 65535, 0]], np.uint16), np.array([[1024, 512, 16]], np.uint16)
    r = np.zeros((1, 3), np.uint16)
    valid, surf, ret = q.composite_flags([(sm1, s1, r), (sm2, s2, r)])
    assert valid.tolist() == [[True, True, True]]
    assert surf.tolist() == [[4, 512, 16]]        # a scene's flag at its own NaN pixel is ignored (incl. the 65535 fill)
    assert ret.tolist() == [[0, 0, 0]]


def test_strata_masks_decode_the_documented_bits():
    q = _strata()
    surf = np.array([0, 1 << 2, 1 << 10, 1 << 9, 1 << 4, 1 << 11], np.uint16)
    ret = np.array([0, 1, 1 << 6, 1 << 7, 0, 0], np.uint16)
    m = q.strata_masks(surf, ret)
    assert set(m) == set(q.STRATA)
    assert m["coastal"].tolist() == [False, True, False, False, False, False]
    assert m["dense_veg"][2] and m["mountain"][3] and m["precip"][4] and m["edge"][5]
    assert m["recommended"].tolist() == [True, False, True, True, True, True]
    assert m["good67"].tolist() == [True, True, False, False, True, True]
    assert m["clean_surface"].tolist() == [True, False, False, False, True, True]
    assert m["clean_surface_good67"].tolist() == [True, False, False, False, True, False]


def test_overlap_sums_use_both_scenes_flags():
    q = _strata()
    nan = np.nan
    a, b = np.array([[0.2, 0.3, nan]]), np.array([[0.4, 0.3, 0.1]])
    z = np.zeros((1, 3), np.uint16)
    sb = np.array([[4, 0, 0]], np.uint16)                 # first pixel coastal in the second scene only
    out = q.overlap_sums([(a, z, z), (b, sb, z)])
    assert out["all"][0] == 2 and abs(out["all"][1] - 0.04) < 1e-9
    assert out["not_coastal"].tolist() == [1, 0.0]
    assert out["coastal"][0] == 1


def test_stratum_tables_remove_static_offsets_within_each_stratum():
    q = _strata()
    rng = np.random.default_rng(0)
    idx = np.arange(40)
    offset = rng.normal(size=40)
    days = []
    for k in range(8):
        day = rng.normal(size=40)
        f = offset + day + rng.normal(scale=0.5, size=40)
        surf = (np.arange(40) % 2 == 0).astype(np.uint16) * (1 << 2)    # even pixels coastal
        days.append((idx, f, dict(a=day, b=day + offset), surf, np.zeros(40, np.uint16)))
    res = {s: t for s, *t in q.stratum_tables(days, (4, 10), ["a", "b"], 5)}
    S, budget, _, pixels = res["coastal"]
    assert pixels == 20 and budget[:, 0].sum() == 8 * 20
    r = q.e1.pooled(S.sum(0))["R"]
    assert abs(r[0] - r[1]) < 1e-9                                       # a static per-pixel offset cannot move the gate
    S, budget, _, pixels = res["recommended"]
    assert pixels == 40
