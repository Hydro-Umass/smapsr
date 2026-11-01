import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from smap_download import DATE_RE, DEFAULT_PRODUCTS, PRODUCTS, REGIONS, process_layer, target_grid


@pytest.mark.parametrize("step", [0.36, 0.09, 0.03, 0.01])
def test_nz_grid_is_the_notebook_expression(step):
    lon, lat = target_grid(step, REGIONS["nz"])
    assert np.array_equal(lon[0], np.arange(164.7, 179.6, step))
    assert np.array_equal(lat[:, 0], np.arange(-32.8, -47.8, -step))


def test_default_products_and_radar_date_token():
    assert DEFAULT_PRODUCTS == ["SPL3SMP_E", "SPL3SMP", "SPL2SMAP_S"]
    assert DATE_RE.search("SMAP_L2_SM_A_01334_D_20150502T185747_R13080_001.nc").group(1) == "20150502"


def test_murrumbidgee_grids_nest_from_one_origin():
    bbox = REGIONS["murrumbidgee"]
    coarse_lon, coarse_lat = target_grid(0.09, bbox)
    for step, stride in ((0.03, 3), (0.01, 9)):
        lon, lat = target_grid(step, bbox)
        n = min(lon.shape[1] // stride, coarse_lon.shape[1])
        assert np.allclose(lon[0, ::stride][:n], coarse_lon[0][:n])
        n = min(lat.shape[0] // stride, coarse_lat.shape[0])
        assert np.allclose(lat[::stride, 0][:n], coarse_lat[:n, 0])
    assert coarse_lon.shape == (28, 45)


def radar_file(path, lats, lons, sm):
    with h5py.File(path, "w") as f:
        g = f.create_group("Soil_Moisture_Retrieval_Data")
        g["latitude"], g["longitude"], g["soil_moisture"] = lats.astype("f4"), lons.astype("f4"), sm.astype("f4")
        g["retrieval_qual_flag"] = np.full(lats.shape, 1, dtype="u2")
        g["surface_flag"] = np.full(lats.shape, 4, dtype="u2")
        g["distance_from_nadir"] = np.full(lats.shape, 250000.0, dtype="f4")


def test_radar_layer_reads_one_dimensional_arrays_and_keeps_gaps(tmp_path):
    bbox = REGIONS["murrumbidgee"]
    lon, lat = target_grid(0.03, bbox)
    sm = np.full(lat.shape, 0.25)
    sm[10, 20] = -9999.0
    path = tmp_path / "SMAP_L2_SM_A_01334_D_20150502T185747_R13080_001.h5"
    radar_file(path, lat.ravel(), lon.ravel(), sm.ravel())
    with h5py.File(path) as f:
        ds = process_layer(f, PRODUCTS["SPL2SMA"][0], path.name, "SPL2SMA", bbox)
    assert ds.soil_moisture.shape == lat.shape
    assert np.isnan(ds.soil_moisture.values[10, 20])
    assert np.isfinite(ds.soil_moisture.values).sum() == lat.size - 1
    assert np.nanmax(np.abs(ds.soil_moisture.values - 0.25)) < 1e-6
    assert ds.retrieval_qual_flag.values[0, 0] == 1 and ds.surface_flag.values[0, 0] == 4
    assert ds.distance_from_nadir.values[0, 0] == 250000.0
    assert ds.attrs["optional_missing"] == ""


def test_swath_outside_the_box_gives_no_output(tmp_path):
    path = tmp_path / "x.h5"
    radar_file(path, np.array([10.0, 10.1]), np.array([20.0, 20.1]), np.array([0.2, 0.3]))
    with h5py.File(path) as f:
        assert process_layer(f, PRODUCTS["SPL2SMA"][0], path.name, "SPL2SMA", REGIONS["murrumbidgee"]) is None
