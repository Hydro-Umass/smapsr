import pandas as pd
import numpy as np
import xarray as  xr
from rioxarray.merge import merge_arrays
from pyresample import kd_tree as prkdt, geometry as prgeom

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
