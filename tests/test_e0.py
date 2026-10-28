"""Unit tests for the E0 readers, metrics and bootstrap (no data needed)."""
import numpy as np
import pandas as pd
import pytest

from smapsr import io
from smapsr import evals


def test_file_date_scene_and_daily():
    assert io.file_date("SMAP_L2_SM_SP_1AIWDV_20160819T180831_20160820T065937_176E36S_R17000_001.nc") == pd.Timestamp("2016-08-19")
    assert io.file_date("SMAP_L3_SM_P_E_20160101_R19240_001.nc") == pd.Timestamp("2016-01-01")
    assert io.file_date("SMAP_L2_SM_SP_20160821_daily.nc") == pd.Timestamp("2016-08-21")


def test_parse_scene_name():
    p = io.parse_scene_name("SMAP_L2_SM_SP_1BIWDV_20170112T175900_20170113T070000_172E43S_R17000_001.nc")
    assert p["satellite"] == "S1B"
    assert p["smap_time"] == pd.Timestamp("2017-01-12T17:59:00", tz="UTC")
    assert p["tile_lon"] == 172.0 and p["tile_lat"] == -43.0


def test_nearest_index():
    lat = -32.8 - 0.09 * np.arange(167)
    assert list(io.nearest_index(lat, [-32.8, -32.84, -32.86, lat[-1]])) == [0, 0, 1, 166]
    with pytest.raises(ValueError):
        io.nearest_index(lat, [-31.0])


def test_read_ground_dedup_sort_utc(tmp_path):
    f = tmp_path / "site.csv"
    f.write_text('"Time, GMT+12:00",Soil_Moisture\n'
                 "2019-07-16 00:00:00+00:00,0.30\n"
                 "2019-07-15 23:00:00+00:00,0.20\n"
                 "2019-07-16 00:00:00+00:00,0.40\n")
    s = io.read_ground(f)
    assert s.index.is_monotonic_increasing
    assert str(s.index.tz) == "UTC"
    assert s.index[0] == pd.Timestamp("2019-07-15 23:00", tz="UTC")
    assert len(s) == 2 and s.iloc[1] == pytest.approx(0.35)


def test_point_metrics_known_values():
    obs = np.array([0.1, 0.2, 0.3, 0.4, np.nan])
    pred = obs + 0.05
    m = evals.point_metrics(obs, pred)
    assert m["N"] == 4
    assert m["R"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(0.05)
    assert m["RMSE"] == pytest.approx(0.05)
    assert m["ubRMSE"] == pytest.approx(0.0, abs=1e-12)
    # consistent with the existing evals.ubrmse
    rng = np.random.default_rng(1)
    o, p = rng.random(50), rng.random(50)
    assert evals.point_metrics(o, p)["ubRMSE"] == pytest.approx(evals.ubrmse(o, p))


def test_date_blocks():
    d = pd.date_range("2017-01-01", periods=65, freq="D")
    b = evals.date_blocks(d, 30)
    assert b[0] == 0 and b[29] == 0 and b[30] == 1 and b[64] == 2


def test_block_bootstrap_covers_truth_and_keeps_blocks():
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 0.05, 400)
    blocks = np.repeat(np.arange(40), 10)
    boot = evals.block_bootstrap(lambda idx: dict(mean=x[idx].mean(), n=len(idx)), blocks, n_boot=300, seed=0)
    lo, hi = evals.ci(boot["mean"])
    assert lo < x.mean() < hi
    assert np.all(boot["n"] % 10 == 0)          # whole blocks only
    again = evals.block_bootstrap(lambda idx: dict(mean=x[idx].mean()), blocks, n_boot=300, seed=0)
    assert np.array_equal(boot["mean"], again["mean"])   # seeded


def test_paired_p():
    assert evals.paired_p(np.full(100, 0.1)) == 0.0
    assert evals.paired_p(np.linspace(-1, 1, 101)) == pytest.approx(1.0)


def test_spatial_corr_masks_nans():
    a = np.arange(200, dtype=float).reshape(10, 20)
    b = 2 * a + 1
    b[0, :5] = np.nan
    assert evals.spatial_corr(a, b, min_pixels=100) == pytest.approx(1.0)
    assert np.isnan(evals.spatial_corr(a, b, min_pixels=1000))


def test_buffer_indices():
    lat = -38.0 - 0.03 * np.arange(100)
    lon = 175.0 + 0.03 * np.arange(100)
    r0, c0 = io.buffer_indices(lat, lon, -39.0, 176.0, 0)
    assert len(r0) == 1 and lat[r0[0]] == pytest.approx(-39.0, abs=0.016)
    r5, c5 = io.buffer_indices(lat, lon, -39.0, 176.0, 5)
    assert (r0[0], c0[0]) in set(zip(r5, c5))
    dy = (lat[r5] + 39.0) * 111.32
    dx = (lon[c5] - 176.0) * 111.32 * np.cos(np.radians(-39.0))
    assert np.all(np.hypot(dy, dx) <= 5.0)
    # a 3-km grid holds roughly pi*5^2/(3.34*2.59) ~ 9 cells within 5 km
    assert 5 <= len(r5) <= 14
    # tiny radius still returns the nearest cell
    assert len(io.buffer_indices(lat, lon, -39.0, 176.0, 0.1)[0]) == 1
