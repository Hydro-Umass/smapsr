"""Mosaic per-granule SPL2SMAP_S NetCDF files into one file per day.

The granules in data/smap3km/ and data/smap1km/ were already gridded onto a
shared regular lat/lon grid (via pyresample in smap_data.ipynb).  Every granule
for a given resolution shares the identical grid, so mosaicing is a simple
NaN-fill merge — no resampling or reprojection needed.

Overlapping valid pixels (same day, multiple overpasses) are reduced with
nanmean, the standard approach for daily composites.

Granules are classified as AM (descending, ~06 local solar) or PM (~18 local) from
the SMAP time in the filename and the tile longitude; `--pass AM` keeps only AM
granules (the SMAP-E input is the AM retrieval). If the granules carry
`retrieval_qual_flag` (smapsr/scripts/smap_download.py), a quality-screened composite is
written alongside the unscreened one: pixels whose flag has any bit of `--qc-mask`
set (default bit 0, "retrieval not recommended" — verify in the user guide) are
dropped before the nanmean.

Usage:
    python mosaic_daily.py                    # process both 3km and 1km (all passes, as before)
    python mosaic_daily.py --res 3            # only 3km
    python mosaic_daily.py --res 1            # only 1km
    python mosaic_daily.py --overwrite        # overwrite existing daily files
    python mosaic_daily.py --data-root data_qc --pass AM   # flagged granules, AM only
"""

import argparse
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_ROOT = "data"
RES_DIRS = {3: "smap3km", 1: "smap1km"}
OUT_DIRS = {3: "smap3km_daily", 1: "smap1km_daily"}

# SMAP_L2_SM_SP_{1AIWDV|1BIWDV}_<acq_date>THHMMSS_<...>_<lonElatS>_R<id>_00X.nc
DATE_RE = re.compile(r"_(\d{8})T\d{6}_")
SCENE_RE = re.compile(r"_(\d{8})T(\d{2})(\d{2})(\d{2})_\d{8}T\d{6}_(\d{3})E\d{2}S_")


def overpass(fname):
    """'AM' or 'PM' from the SMAP UTC time and tile longitude in a granule name.

    Offset from 06:00 local solar time on the morning after the UTC date: AM passes
    sit within ±3 h of it, PM passes ~12 h earlier (same rule as the E0 runner)."""
    m = SCENE_RE.search(os.path.basename(fname))
    if m is None:
        return "unknown"
    hh, mm, ss, lon = (int(x) for x in m.groups()[1:])
    utc_h = hh + mm / 60 + ss / 3600
    offset = utc_h - (30 - lon / 15)
    if abs(offset) <= 3:
        return "AM"
    if abs(offset + 12) <= 3:
        return "PM"
    return "unknown"


def group_by_date(files):
    """Group granule filenames by acquisition date (YYYYMMDD)."""
    groups = defaultdict(list)
    for f in files:
        m = DATE_RE.search(os.path.basename(f))
        if m:
            groups[m.group(1)].append(f)
    return dict(groups)


def mosaic_day(day_files, date_str, qc_mask=1):
    """Merge all granules for a single day into one Dataset.

    Granules share the same lat/lon grid, so we stack along a new dimension
    and reduce with nanmean.  Returns variables 'soil_moisture' and
    'coverage_count' (dims (lat, lon)), plus 'soil_moisture_qc' and
    'coverage_count_qc' when the granules carry 'retrieval_qual_flag'.
    """
    arrays, flags = [], []
    for f in day_files:
        ds = xr.open_dataset(f)
        name = "soil_moisture" if "soil_moisture" in ds else "__xarray_dataarray_variable__"
        arrays.append(ds[name].load())
        if "retrieval_qual_flag" in ds:
            flags.append(ds["retrieval_qual_flag"].load())
        ds.close()

    # Concat along a synthetic "granule" axis, then nanmean
    stacked = xr.concat(arrays, dim="granule")
    out = {"soil_moisture": stacked.reduce(np.nanmean, dim="granule"),
           # Count how many granules contributed to each pixel (useful metadata)
           "coverage_count": stacked.notnull().sum(dim="granule")}

    if flags and len(flags) == len(arrays):
        fl = xr.concat(flags, dim="granule")
        good = stacked.where((fl.astype("int64") & qc_mask) == 0)
        out["soil_moisture_qc"] = good.reduce(np.nanmean, dim="granule")
        out["coverage_count_qc"] = good.notnull().sum(dim="granule")

    return out, len(day_files)


def process_resolution(res, overwrite=False, data_root=DATA_ROOT, pass_=None, qc_mask=1):
    src_dir = os.path.join(data_root, RES_DIRS[res])
    out_dir = os.path.join(data_root, OUT_DIRS[res] + (f"_{pass_}" if pass_ else ""))
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir)
        if f.endswith(".nc")
    )
    n_all = len(files)
    if pass_:
        files = [f for f in files if overpass(f) == pass_]
    groups = group_by_date(files)
    print(f"\n[{res}km] {n_all} granules, {len(files)} kept (pass={pass_ or 'all'}) → {len(groups)} unique days")

    skipped = 0
    for date_str in tqdm(sorted(groups), desc=f"mosaic {res}km"):
        outpath = os.path.join(out_dir, f"SMAP_L2_SM_SP_{date_str}_daily.nc")
        if os.path.exists(outpath) and not overwrite:
            skipped += 1
            continue

        day_files = sorted(groups[date_str])
        out, n_gran = mosaic_day(day_files, date_str, qc_mask)

        attrs = {
            "product": "SPL2SMAP_S",
            "resolution_km": res,
            "date": date_str,
            "n_granules": n_gran,
            "overlap_strategy": "nanmean",
            "source_dir": RES_DIRS[res],
        }
        if pass_:
            attrs["pass"] = pass_
        if "soil_moisture_qc" in out:
            attrs["qc_mask"] = qc_mask
        ds = xr.Dataset(out, attrs=attrs)

        # Compression — these arrays are >90% NaN so deflate shrinks them a lot
        encoding = {
            v: {"zlib": True, "complevel": 4, "dtype": "float32" if v.startswith("soil") else "int16"}
            for v in ds.data_vars
        }
        ds.to_netcdf(outpath, encoding=encoding)
        ds.close()

    print(f"[{res}km] done.  Skipped {skipped} existing files.")


def main():
    ap = argparse.ArgumentParser(description="Mosaic SPL2SMAP_S granules into daily NetCDF files.")
    ap.add_argument("--res", type=int, choices=[1, 3], help="Process only this resolution.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing daily files.")
    ap.add_argument("--data-root", default=DATA_ROOT, help="Root holding smap{3,1}km/ (output goes beside them).")
    ap.add_argument("--pass", dest="pass_", choices=["AM", "PM"], default=None,
                    help="Keep only granules from this pass (output dirs get an _AM/_PM suffix).")
    ap.add_argument("--qc-mask", type=lambda x: int(x, 0), default=1,
                    help="Bits of retrieval_qual_flag that reject a pixel in the _qc composite (default 0x1).")
    args = ap.parse_args()

    resolutions = [args.res] if args.res else [3, 1]
    for res in resolutions:
        process_resolution(res, overwrite=args.overwrite, data_root=args.data_root,
                           pass_=args.pass_, qc_mask=args.qc_mask)


if __name__ == "__main__":
    main()