import re
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as  xr
from rioxarray.merge import merge_arrays
from pyresample import kd_tree as prkdt, geometry as prgeom

# Date tokens differ per product: per-scene SPL2SMAP_S names carry the SMAP time
# (`_YYYYMMDDTHHMMSS_`, first match) followed by the Sentinel-1 time; L3 and daily
# composite names carry a bare `_YYYYMMDD_`.
SCENE_DATE_RE = re.compile(r"_(\d{8})T\d{6}_")
DAILY_DATE_RE = re.compile(r"_(\d{8})_")
SCENE_NAME_RE = re.compile(r"SMAP_L2_SM_SP_(1[AB])\w*?_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{3})E(\d{2})S_")

def file_date(path):
    """Date (Timestamp) encoded in a SMAP product filename (scene, L3 or daily composite)."""
    name = Path(path).name
    m = SCENE_DATE_RE.search(name) or DAILY_DATE_RE.search(name)
    if m is None:
        raise ValueError(f"no date token in {name}")
    return pd.Timestamp(m.group(1))

def list_product(directory, pattern="*.nc"):
    """Files of a gridded product indexed by date (one file per date)."""
    files = sorted(Path(directory).glob(pattern))
    s = pd.Series([str(f) for f in files], index=pd.DatetimeIndex([file_date(f) for f in files], name="time"))
    if s.index.has_duplicates:
        raise ValueError(f"{directory}: more than one file per date")
    return s.sort_index()

def parse_scene_name(path):
    """Satellite (1A/1B), SMAP and Sentinel-1 UTC times, and tile corner of an SPL2SMAP_S scene."""
    m = SCENE_NAME_RE.search(Path(path).name)
    if m is None:
        raise ValueError(f"not an SPL2SMAP_S scene name: {path}")
    sat, smap_t, s1_t, lon, lat = m.groups()
    return dict(satellite=f"S{sat}", smap_time=pd.Timestamp(smap_t, tz="UTC"),
                s1_time=pd.Timestamp(s1_t, tz="UTC"), tile_lon=float(lon), tile_lat=-float(lat))

def read_grid(path):
    """Read one gridded product file (raw L3, daily composite) as a (lat, lon) DataArray in m3/m3.

    Raw L3 files hold a single unnamed variable; daily composites name it `soil_moisture`."""
    with xr.open_dataset(path) as ds:
        name = "soil_moisture" if "soil_moisture" in ds else "__xarray_dataarray_variable__"
        da = ds[name].load()
    return da.rename("sm")

def nearest_index(coords, values):
    """Index of the nearest grid coordinate for each value (regular 1-D grid)."""
    coords = np.asarray(coords)
    step = coords[1] - coords[0]
    idx = np.rint((np.asarray(values) - coords[0]) / step).astype(int)
    if np.any((idx < 0) | (idx >= len(coords))):
        raise ValueError("value outside grid")
    return idx

def buffer_indices(lat, lon, site_lat, site_lon, radius_km):
    """Grid cells used for a site: the nearest cell plus every cell centred within radius_km.

    radius_km = 0 gives the nearest cell alone. Distances use a local equirectangular
    approximation (exact enough at <= tens of km). Returns (row, col) index arrays."""
    i0 = nearest_index(lat, [site_lat])[0]
    j0 = nearest_index(lon, [site_lon])[0]
    dy = (np.asarray(lat) - site_lat) * 111.32
    dx = (np.asarray(lon) - site_lon) * 111.32 * np.cos(np.radians(site_lat))
    d = np.hypot(dy[:, None], dx[None, :])
    inside = d <= radius_km
    inside[i0, j0] = True
    return np.nonzero(inside)

def read_sites(path):
    """Ground-site table (whitespace separated: name lon lat)."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=["site", "lon", "lat"])

def read_ground(path):
    """Read an in-situ soil moisture CSV as a UTC-indexed, time-sorted Series (m3/m3).

    The first column is the timestamp whatever its header; all timestamps are UTC
    (Tokoiti's "GMT+12:00" header is wrong, its +00:00 offsets are right). Rows sharing
    a timestamp (Puruki: separate probes; Balmoral, CPT861: a few) are averaged."""
    df = pd.read_csv(path)
    t = pd.to_datetime(df.iloc[:, 0], utc=True, format="mixed")
    s = pd.Series(df["Soil_Moisture"].to_numpy(dtype=float), index=pd.DatetimeIndex(t, name="time"))
    s = s.dropna().groupby(level=0).mean().sort_index()
    s.name = Path(path).stem
    return s

def regrid_nearest(lat, lon, sm, nz, roi):
    """Regrids swath data using nearest neighbor interpolation."""
    swath_def = prgeom.SwathDefinition(lons=lon, lats=lat)
    lons, lats = np.meshgrid(nz.x, nz.y)
    grid_def = prgeom.GridDefinition(lons=lons, lats=lats)
    g = prkdt.resample_nearest(swath_def, sm.data, grid_def, radius_of_influence=roi,
                               fill_value=None)
    out = xr.DataArray(g, dims=('y', 'x'), coords=dict(x=nz.x.data, y=nz.y.data))
    out = out.rio.write_crs(nz.rio.crs)
    return out

def read_smap(files, domain, res=9):
    """Read SMAP soil moisture observations from the SPL3SMP_E data product."""
    ars = []
    dt = []
    for f in files:
        x = xr.open_dataset(f, group="Soil_Moisture_Retrieval_Data_AM")
        lats = x.latitude.data
        lons = x.longitude.data
        sm = x.soil_moisture
        rsm = regrid_nearest(lats, lons, sm, domain, res*1000)
        ars.append(rsm)
        dt.append(f.split("_")[-4])
    dt = pd.to_datetime(dt)
    dt.name = "time"
    return xr.concat(ars, dim=dt).sortby('time').rename('sm')

def read_smap_sentinel(files, res, nz):
    """Read SMAP-Sentinel data."""
    dt = pd.to_datetime([f.split("_")[-4] for f in files])
    idx = pd.Series(range(len(dt)), dt).resample('1D').apply(lambda s: s.values)
    idx = idx.loc[idx.apply(len) > 0]
    def _read(fs):
        rsm = []
        for f in fs:
            x = xr.open_dataset(f, group=f"Soil_Moisture_Retrieval_Data_{res}km")
            lats = x[f'latitude_{res}km'].data
            lons = x[f'longitude_{res}km'].data
            sm = x[f'soil_moisture_{res}km']
            rsm.append(regrid_nearest(lats, lons, sm, nz, res*1000))
        return merge_arrays(rsm, nodata=np.nan)
    sm = idx.apply(lambda i: _read([files[j] for j in i]))
    return xr.concat(sm, dim=idx.index.rename('time')).rename('sm')
