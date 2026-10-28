"""Download + regrid SMAP products onto the project lat/lon grids, keeping quality flags.

Successor to smap_data.ipynb (which produced the current data/ files but dropped every
variable except soil moisture). Same search box, same HDF5 groups, same grids
(np.arange(164.7, 179.6, step) x np.arange(-32.8, -47.8, -step)), same nearest-neighbour
regridding (radius of influence = resolution, epsilon 0.5) and the same `sm <= 0 -> NaN`
rule — so `soil_moisture` should match the existing files — plus, from the *same*
nearest swath cell:

  retrieval_qual_flag   (integer bit flags, regridded without decoding)
  surface_flag          (if present in the granule)
  tb_time_seconds       (L3 only, if present: per-pixel overpass time)

Bit meanings are NOT decoded here — check them in the product user guides before
filtering (bit 0 is the "retrieval recommended" bit in the SMAP L2/L3 products: 0 =
recommended). Run `--inspect` first: it prints the HDF5 tree of one granule per product so
the variable names below can be confirmed before a full run.

Run in AWS us-west-2 (where NSIDC's Earthdata Cloud buckets live): earthaccess.open then
streams only the datasets read, with no download step. Elsewhere it falls back to HTTPS.

Credentials: ~/.netrc (machine urs.earthdata.nasa.gov) or EARTHDATA_USERNAME/PASSWORD.

Usage:
    python smap_download.py --inspect
    python smap_download.py --products SPL3SMP_E SPL3SMP SPL2SMAP_S --workers 8
    python smap_download.py --products SPL2SMAP_S --start 2016-01-01 --end 2016-12-31

Output: <out-root>/{smap9km,smap36km,smap3km,smap1km}/<granule>.nc (one file per granule;
existing outputs are skipped, so an interrupted run can simply be restarted) and
<out-root>/download_log_<product>.csv (status per granule).
"""
import argparse
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from pyresample import geometry, kd_tree

BBOX = (164.7, -47.8, 179.6, -32.8)       # lon_min, lat_min, lon_max, lat_max (as in smap_data.ipynb)
MARGIN_DEG = 1.0                          # swath cells kept around the box before the kd-tree (> any radius)
DATE_RE = re.compile(r"_(\d{8})(?:T\d{6})?_")   # first date token: L3 `_YYYYMMDD_`, L2 SP SMAP time `_YYYYMMDDThhmmss_`

# product -> list of layers; each layer is one output grid read from one HDF5 group
PRODUCTS = {
    "SPL3SMP_E": [dict(out="smap9km", step=0.09, roi=9000, group="Soil_Moisture_Retrieval_Data_AM",
                       sm="soil_moisture", lat="latitude", lon="longitude", flag="retrieval_qual_flag",
                       optional=["surface_flag", "tb_time_seconds"])],
    "SPL3SMP": [dict(out="smap36km", step=0.36, roi=36000, group="Soil_Moisture_Retrieval_Data_AM",
                     sm="soil_moisture", lat="latitude", lon="longitude", flag="retrieval_qual_flag",
                     optional=["surface_flag", "tb_time_seconds"])],
    "SPL2SMAP_S": [dict(out=f"smap{r}km", step=0.01 * r, roi=1000 * r, group=f"Soil_Moisture_Retrieval_Data_{r}km",
                        sm=f"soil_moisture_{r}km", lat=f"latitude_{r}km", lon=f"longitude_{r}km",
                        flag=f"retrieval_qual_flag_{r}km", optional=[f"surface_flag_{r}km"])
                   for r in (3, 1)],
}


def target_grid(step):
    """Exactly the grid expression used by smap_data.ipynb."""
    lon, lat = np.meshgrid(np.arange(164.7, 179.6, step), np.arange(-32.8, -47.8, -step))
    return lon, lat


def regrid(lats, lons, fields, step, roi):
    """Nearest-neighbour regrid of several swath fields using ONE neighbour lookup.

    All fields are sampled from the same swath cell, so flags line up with soil moisture.
    `fields` maps name -> (array, fill value for grid cells with no neighbour).
    Returns (lat, lon, {name: grid array}) or None if no swath cell is near the box."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    keep = ((lats != -9999) & (lons != -9999) & np.isfinite(lats) & np.isfinite(lons)
            & (lats >= BBOX[1] - MARGIN_DEG) & (lats <= BBOX[3] + MARGIN_DEG)
            & (lons >= BBOX[0] - MARGIN_DEG) & (lons <= BBOX[2] + MARGIN_DEG))
    if not keep.any():
        return None
    swath = geometry.SwathDefinition(lons=lons[keep], lats=lats[keep])
    glon, glat = target_grid(step)
    grid = geometry.GridDefinition(lons=glon, lats=glat)
    vii, voi, idx, _ = kd_tree.get_neighbour_info(swath, grid, roi, neighbours=1, epsilon=0.5)
    out = {}
    for name, (arr, fill) in fields.items():
        out[name] = kd_tree.get_sample_from_neighbour_info("nn", grid.shape, np.asarray(arr)[keep], vii, voi, idx,
                                                           fill_value=fill)
    return glat[:, 0], glon[0, :], out


def int_fill(dtype):
    dtype = np.dtype(dtype)
    return np.iinfo(dtype).max if dtype.kind in "ui" else np.nan


def process_layer(h5, layer, granule_name, short_name):
    """Read one group of an open granule and return the regridded xr.Dataset (or None)."""
    g = h5[layer["group"]]
    missing = [k for k in ("sm", "lat", "lon", "flag") if layer[k] not in g]
    if missing:
        raise KeyError(f"{granule_name}: {layer['group']} lacks {[layer[k] for k in missing]}; "
                       f"available: {sorted(g.keys())}")
    sm = g[layer["sm"]][:]
    fields = {"soil_moisture": (sm, 0)}           # fill 0 then `<= 0 -> NaN`, as in the notebook
    flag = g[layer["flag"]][:]
    fields["retrieval_qual_flag"] = (flag, int_fill(flag.dtype))
    present = [v for v in layer["optional"] if v in g]
    for v in present:
        a = g[v][:]
        fields[v] = (a, int_fill(a.dtype))
    res = regrid(g[layer["lat"]][:], g[layer["lon"]][:], fields, layer["step"], layer["roi"])
    if res is None:
        return None
    lat, lon, out = res
    sm_g = out.pop("soil_moisture").astype(np.float32)
    sm_g[sm_g <= 0] = np.nan
    data = {"soil_moisture": (("lat", "lon"), sm_g, {"units": "m3/m3", "source": f"{layer['group']}/{layer['sm']}"})}
    for name, arr in out.items():
        src = layer["flag"] if name == "retrieval_qual_flag" else name
        attrs = {"source": f"{layer['group']}/{src}"}
        if arr.dtype.kind in "ui":
            attrs["_no_neighbour_fill"] = int(int_fill(arr.dtype))
        else:
            arr = arr.astype(np.float64)
            arr[arr == -9999] = np.nan
        data[name] = (("lat", "lon"), arr, attrs)
    return xr.Dataset(data, coords=dict(lat=lat, lon=lon),
                      attrs=dict(granule=granule_name, short_name=short_name, group=layer["group"],
                                 grid_step_deg=layer["step"], radius_of_influence_m=layer["roi"],
                                 regrid="pyresample nearest, epsilon=0.5, one neighbour lookup for all fields",
                                 optional_missing=",".join(sorted(set(layer["optional"]) - set(present))),
                                 generator="smapsr/scripts/smap_download.py"))


def granule_h5_link(granule):
    links = [u for u in granule.data_links(access="direct") if u.endswith(".h5")] or \
            [u for u in granule.data_links() if u.endswith(".h5")]
    return links[0]


def run_chunk(short_name, start, end, version, out_root, overwrite):
    """Search + process one date chunk (runs in a worker process). Returns log rows."""
    import earthaccess
    earthaccess.login(strategy="environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc")
    # Query one extra day, then keep granules whose filename date lies in [start, end]: chunks
    # neither miss boundary days nor process the same granule twice.
    q_end = str((pd.Timestamp(end) + pd.Timedelta(days=1)).date())
    kw = dict(short_name=short_name, bounding_box=BBOX, temporal=(start, q_end), cloud_hosted=True)
    if version:
        kw["version"] = version
    granules = earthaccess.search_data(**kw)
    layers = PRODUCTS[short_name]
    rows, todo = [], []
    for gr in granules:
        name = Path(granule_h5_link(gr)).name.replace(".h5", ".nc")
        m = DATE_RE.search(name)
        if m is None or not (start.replace("-", "") <= m.group(1) <= end.replace("-", "")):
            continue
        outs = [out_root / l["out"] / name for l in layers]
        if not overwrite and all(o.exists() for o in outs):
            rows.append(dict(granule=name, status="exists"))
        else:
            todo.append((gr, name, outs))
    if not todo:
        return rows
    handles = earthaccess.open([t[0] for t in todo], show_progress=False)
    for (gr, name, outs), fh in zip(todo, handles):
        try:
            with h5py.File(fh, "r") as h5:
                for layer, out in zip(layers, outs):
                    ds = process_layer(h5, layer, name.replace(".nc", ".h5"), short_name)
                    if ds is None:
                        rows.append(dict(granule=name, layer=layer["out"], status="no_overlap"))
                        continue
                    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
                    tmp = out.with_suffix(".nc.part")
                    ds.to_netcdf(tmp, encoding=enc)
                    tmp.rename(out)
                    rows.append(dict(granule=name, layer=layer["out"], status="ok"))
        except Exception as e:
            rows.append(dict(granule=name, status="error", error=f"{type(e).__name__}: {e}"))
        finally:
            fh.close()      # h5py does not close the fsspec handle; its read cache otherwise lives until the chunk ends
    return rows


def inspect(version):
    """Print the HDF5 tree of the first granule of each product."""
    import earthaccess
    earthaccess.login()
    for short_name, layers in PRODUCTS.items():
        kw = dict(short_name=short_name, bounding_box=BBOX, temporal=("2017-01-01", "2017-01-10"), cloud_hosted=True)
        if version:
            kw["version"] = version
        gr = earthaccess.search_data(**kw)
        print(f"\n=== {short_name}: {len(gr)} granules in 2017-01-01..10; first: {granule_h5_link(gr[0])}")
        with h5py.File(earthaccess.open(gr[:1])[0], "r") as h5:
            for layer in layers:
                g = h5[layer["group"]]
                print(f"--- {layer['group']}")
                for k in sorted(g.keys()):
                    d = g[k]
                    print(f"  {k:40s} {d.dtype!s:10s} {d.shape}  {d.attrs.get('long_name', b'')!r}")
                for k in [layer["sm"], layer["flag"], layer["lat"], layer["lon"], *layer["optional"]]:
                    print(f"  expected {k}: {'OK' if k in g else 'MISSING'}")
                if layer["flag"] in g:
                    for a, v in g[layer["flag"]].attrs.items():
                        print(f"    {layer['flag']}.{a} = {v!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=list(PRODUCTS))
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--version", default=None,
                    help="product version (default: latest in CMR; the current data are SPL3SMP_E v6, SPL3SMP v9, SPL2SMAP_S v3)")
    ap.add_argument("--out-root", default="data_qc", help="output root (keep it separate from data/)")
    ap.add_argument("--chunk", default="MS", help="pandas frequency of the date chunks handed to workers")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tasks-per-child", type=int, default=1,
                    help="chunks a worker process handles before being replaced (1 = fresh process per chunk; "
                         "needs Python >= 3.11, uses the spawn start method)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--inspect", action="store_true", help="print one granule's HDF5 layout per product and exit")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.version)
        return
    out_root = Path(args.out_root)
    edges = list(pd.date_range(args.start, args.end, freq=args.chunk))
    starts = [pd.Timestamp(args.start)] + [e for e in edges if e > pd.Timestamp(args.start)]
    chunks = [(s, min(n - pd.Timedelta(days=1), pd.Timestamp(args.end)))
              for s, n in zip(starts, starts[1:] + [pd.Timestamp(args.end) + pd.Timedelta(days=1)])]
    for short_name in args.products:
        for layer in PRODUCTS[short_name]:
            (out_root / layer["out"]).mkdir(parents=True, exist_ok=True)
        rows = []
        # A fresh process per chunk frees whatever a worker accumulates (earthaccess/fsspec block
        # caches, kd-tree buffers) — long-lived workers were OOM-killed, which breaks the whole pool.
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=args.tasks_per_child) as ex:
            futs = {ex.submit(run_chunk, short_name, str(s.date()), str(e.date()), args.version, out_root, args.overwrite): s
                    for s, e in chunks}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    r = f.result()
                except Exception:
                    r = [dict(granule=f"chunk {futs[f].date()}", status="error", error=traceback.format_exc(limit=1))]
                rows.extend(r)
                n_err = sum(x["status"] == "error" for x in r)
                print(f"[{short_name}] chunk {i}/{len(chunks)} {futs[f].date()}: {len(r)} rows, {n_err} errors", flush=True)
        log = pd.DataFrame(rows)
        log.to_csv(out_root / f"download_log_{short_name}.csv", index=False)
        print(f"[{short_name}] status counts:\n{log['status'].value_counts().to_string()}", flush=True)
        if (log["status"] == "error").any():
            print(f"[{short_name}] errors present — re-run the same command to retry them", file=sys.stderr)


if __name__ == "__main__":
    main()
