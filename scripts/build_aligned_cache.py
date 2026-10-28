"""Build the aligned per-date cache (<data-root>/cache/aligned/{9km,36km,3km,1km}/<YYYYMMDD>.npz).

Replaces the cache whose generator was never committed. The rule below was reverse-engineered and
checked bit-for-bit against the existing data/cache/aligned files (soil_moisture and mask, all four
scales) before being applied to data_qc.

  9 km, 36 km   the regridded L3 field as is (167x166, 42x42), one npz per date, including dates with
                no valid pixel (all NaN) so the calendar matches the L3 inventory.
  3 km, 1 km    the all-pass nanmean daily composite from mosaic_daily.py, laid on the *aligned* grid:
                every 9-km cell is an exact 3x3 / 9x9 block of fine cells, the 9-km centre at the block
                centre. The raw label grid starts at the same origin as the 9-km grid, so it is placed at
                index offset 1 (3 km) / 4 (1 km) of a NaN grid of 501x498 / 1503x1494; the raw rows and
                columns that fall off the south/east edge are dropped (1 km: one row, one column).

Keys: soil_moisture (float32, NaN = no data) and mask (= isfinite(soil_moisture)) — the same as the old
cache. Extras when the source carries them (data_qc does): retrieval_qual_flag (L3, uint16, 65535 where no
swath cell was near — bits NOT decoded here) and soil_moisture_qc (labels, composite after dropping pixels
whose flag has --qc-mask bits set; see mosaic_daily.py).

Needs the daily composites first:  python mosaic_daily.py --data-root data_qc [--pass AM]

--pass AM reads the AM-only composites (smap{3,1}km_daily_AM; days with no AM scene drop out) and writes the
whole cache to <src-root>/cache/aligned_AM/ (the 9/36-km L3 fields are AM already and are copied as is).

Usage:
    python smapsr/scripts/build_aligned_cache.py                                   # data_qc -> data_qc/cache/aligned
    python smapsr/scripts/build_aligned_cache.py --pass AM                  # AM-only labels -> data_qc/cache/aligned_AM
    python smapsr/scripts/build_aligned_cache.py --scales 3km 1km --start 2021-01-01 --end 2021-03-31
    python smapsr/scripts/build_aligned_cache.py --src-root data --out-root /scratch/x   # rebuild from the old data
"""
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

L3 = {"9km": ("smap9km", "SPL3SMP_E"), "36km": ("smap36km", "SPL3SMP")}
LABEL = {"3km": dict(src="smap3km_daily", offset=1, shape=(501, 498), raw=(500, 497)),
         "1km": dict(src="smap1km_daily", offset=4, shape=(1503, 1494), raw=(1500, 1491))}
L3_DATE = re.compile(r"_(\d{8})_")
DAILY_DATE = re.compile(r"_(\d{8})_daily\.nc$")
RAW_VAR = "__xarray_dataarray_variable__"          # unnamed variable of the pre-flag raw files


def sm_of(ds):
    return ds["soil_moisture" if "soil_moisture" in ds else RAW_VAR].values.squeeze().astype(np.float32)


def save_npz(path, **arrays):
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:                      # file object: np.savez would append ".npz" to the name
        np.savez_compressed(f, **arrays)
    tmp.rename(path)


def in_range(date, start, end):
    return (start is None or date >= start) and (end is None or date <= end)


def l3_dates(src, scale, start, end):
    """date -> file for every L3 file, plus the dates the download log marks no_overlap (no file written)."""
    folder, short_name = L3[scale]
    files = {}
    for f in sorted((src / folder).glob("*.nc")):
        d = L3_DATE.search(f.name).group(1)
        assert d not in files, f"two L3 files for {d}: {files[d].name}, {f.name}"
        files[d] = f
    log = src / f"download_log_{short_name}.csv"
    empty = set()
    if log.exists():
        df = pd.read_csv(log)
        empty = {L3_DATE.search(g).group(1) for g in df.loc[df["status"] == "no_overlap", "granule"]}
        empty -= set(files)
    dates = {d: files.get(d) for d in sorted(set(files) | empty)}
    return {d: f for d, f in dates.items() if in_range(d, start, end)}


def build_l3(src, out, scale, start, end, overwrite):
    dates = l3_dates(src, scale, start, end)
    out.mkdir(parents=True, exist_ok=True)
    first = next(f for f in dates.values() if f is not None)
    with xr.open_dataset(first) as ds:
        shape, has_flag = sm_of(ds).shape, "retrieval_qual_flag" in ds
    n_new = n_empty = 0
    for d, f in dates.items():
        path = out / f"{d}.npz"
        if path.exists() and not overwrite:
            continue
        if f is None:                               # no valid swath cell in the box that day
            sm, extra = np.full(shape, np.nan, np.float32), {}
            if has_flag:
                extra["retrieval_qual_flag"] = np.full(shape, np.iinfo(np.uint16).max, np.uint16)
            n_empty += 1
        else:
            with xr.open_dataset(f) as ds:
                sm = sm_of(ds)
                assert sm.shape == shape, f"{f.name}: {sm.shape} != {shape}"
                extra = {"retrieval_qual_flag": ds["retrieval_qual_flag"].values} if "retrieval_qual_flag" in ds else {}
        save_npz(path, soil_moisture=sm, mask=np.isfinite(sm), **extra)
        n_new += 1
    return dict(dates=len(dates), written=n_new, no_file_days=n_empty, grid=list(shape))


def place(x, shape, off):
    """Lay a raw-grid field onto the aligned grid (NaN outside); rows/cols past the edge are dropped."""
    a = np.full(shape, np.nan, x.dtype)
    h, w = min(x.shape[0], shape[0] - off), min(x.shape[1], shape[1] - off)
    a[off:off + h, off:off + w] = x[:h, :w]
    return a


def build_label(src, out, scale, start, end, overwrite, suffix=""):
    spec = LABEL[scale]
    files = {DAILY_DATE.search(f.name).group(1): f for f in sorted((src / (spec["src"] + suffix)).glob("*_daily.nc"))}
    files = {d: f for d, f in files.items() if in_range(d, start, end)}
    out.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for d, f in files.items():
        path = out / f"{d}.npz"
        if path.exists() and not overwrite:
            continue
        with xr.open_dataset(f) as ds:
            x = ds["soil_moisture"].values.astype(np.float32)
            assert x.shape == spec["raw"], f"{f.name}: {x.shape} != {spec['raw']}"
            sm = place(x, spec["shape"], spec["offset"])
            extra = {}
            if "soil_moisture_qc" in ds:
                extra["soil_moisture_qc"] = place(ds["soil_moisture_qc"].values.astype(np.float32), spec["shape"], spec["offset"])
        save_npz(path, soil_moisture=sm, mask=np.isfinite(sm), **extra)
        n_new += 1
    return dict(dates=len(files), written=n_new, grid=list(spec["shape"]), offset=spec["offset"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default="data_qc", help="holds smap9km/, smap36km/, smap{3,1}km_daily/, download logs")
    ap.add_argument("--out-root", default=None, help="default: <src-root>/cache/aligned")
    ap.add_argument("--scales", nargs="+", default=["9km", "36km", "3km", "1km"], choices=["9km", "36km", "3km", "1km"])
    ap.add_argument("--start", default=None, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--pass", dest="pass_", choices=["AM", "PM"], default=None,
                    help="use the single-pass label composites (smap{3,1}km_daily_<pass>) and write cache/aligned_<pass>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src_root)
    suffix = f"_{args.pass_}" if args.pass_ else ""
    out_root = Path(args.out_root) if args.out_root else src / "cache" / f"aligned{suffix}"
    start = args.start.replace("-", "") if args.start else None
    end = args.end.replace("-", "") if args.end else None
    summary = {}
    for scale in args.scales:
        if scale in L3:
            summary[scale] = build_l3(src, out_root / scale, scale, start, end, args.overwrite)
        else:
            summary[scale] = build_label(src, out_root / scale, scale, start, end, args.overwrite, suffix)
        print(scale, summary[scale], flush=True)
    prov = dict(generator="smapsr/scripts/build_aligned_cache.py", created=datetime.now().isoformat(timespec="seconds"),
                src_root=str(src), label_pass=args.pass_ or "all", start=args.start, end=args.end, scales=summary)
    (out_root / f"PROVENANCE_{datetime.now():%Y%m%dT%H%M%S}.json").write_text(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
