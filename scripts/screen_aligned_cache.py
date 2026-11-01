"""Aligned cache whose fine labels are the flag-screened composite: soil_moisture := soil_moisture_qc of a source cache
(built with mosaic_daily.py --qc-mask <mask>, build_aligned_cache.py). The coarse folder is linked, not copied.

Usage: python smapsr/scripts/screen_aligned_cache.py --src <cache> --out <cache> --fine 3km --coarse-from <cache>
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fine", default="3km")
    ap.add_argument("--coarse", default="9km")
    ap.add_argument("--coarse-from", required=True, help="cache holding the coarse folder to link")
    a = ap.parse_args()
    out = Path(a.out) / a.fine
    out.mkdir(parents=True, exist_ok=True)
    link = Path(a.out) / a.coarse
    if not link.exists():
        link.symlink_to((Path(a.coarse_from) / a.coarse).resolve())
    n_all, n_kept, n = 0, 0, 0
    for f in sorted((Path(a.src) / a.fine).glob("*.npz")):
        with np.load(f) as z:
            sm, qc = z["soil_moisture"], z["soil_moisture_qc"]
        n_all += int(np.isfinite(sm).sum())
        n_kept += int(np.isfinite(qc).sum())
        np.savez_compressed(out / f.name, soil_moisture=qc, mask=np.isfinite(qc))
        n += 1
    (Path(a.out) / f"PROVENANCE_{datetime.now():%Y%m%dT%H%M%S}.json").write_text(json.dumps(dict(args=vars(a), dates=n, valid_before=n_all, valid_after=n_kept), indent=2))
    print(f"{n} dates; valid pixel-days {n_all} -> {n_kept} ({n_kept / n_all:.1%})")


if __name__ == "__main__":
    main()
