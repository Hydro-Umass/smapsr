"""E2 track-corrected labels: cache whose fine labels have the pixel x track offset removed, estimated on the fit dates only.

The offset of a pixel on a track is the mean of label minus bilinear coarse on that pixel x track, minus the same mean over
the whole pixel. Bilinear coarse is subtracted first so that a track sampling a wetter season does not count as an offset.
Only pixel-days with one track enter the estimate; a composite of several scenes is corrected by the mean offset of its
scenes, as the composite is their mean. Offsets are shrunk by n / (n + k), with k = within-cell variance / between-cell
variance estimated from the fit dates, because a cell holds few rows. Cells under --min-count rows get no offset.

Dates outside --write are not written, so the 2016-2020 labels stay out of the cache. Writes <out>/<fine>/*.npz, links the
coarse folder, and saves offsets.npz and a provenance file next to them. The out-of-sample check compares the within-pixel
variance of label minus bilinear before and after correction on --check dates, which did not enter the offsets.

Usage (project root):  PYTHONPATH=smapsr python smapsr/experiments/e2_track_offsets.py --out data_qc/cache/aligned_AM_trackcorr
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e1_baselines as e1                                             # noqa: E402
import e9_noise_ceiling as e9                                         # noqa: E402
from build_aligned_cache import place                                 # noqa: E402
from e9_orbit_split import assign_tracks                              # noqa: E402
from smapsr.baselines import bilinear                                 # noqa: E402


def fit_offsets(n, s, ss, min_count=5):
    """Shrunk offsets (T, H, W) from per-cell count, sum and sum of squares of the residual (T, H, W); also the shrinkage
    constant k and the variance components. Cells under min_count are dropped before anything else."""
    keep = n >= min_count
    n, s, ss = np.where(keep, n, 0.0), np.where(keep, s, 0.0), np.where(keep, ss, 0.0)
    cn, ps = np.maximum(n, 1), s.sum(0)
    pn = n.sum(0)
    cm, pm = s / cn, ps / np.maximum(pn, 1)
    cells, pixels, rows = int(keep.sum()), int((pn > 0).sum()), n.sum()
    df_b, df_w = cells - pixels, rows - cells
    if df_b <= 0 or df_w <= 0:
        return np.zeros_like(cm, dtype=np.float32), np.inf, 0.0, 0.0
    sigma2 = (ss - s * cm).sum() / df_w
    n0 = (rows - (n ** 2 / np.maximum(pn, 1)).sum()) / df_b                  # effective cell size, nested in pixels
    tau2 = (((cm - pm) ** 2 * n).sum() / df_b - sigma2) / n0
    if tau2 <= 0:
        return np.zeros_like(cm, dtype=np.float32), np.inf, float(sigma2), float(tau2)
    k = sigma2 / tau2
    return ((cm - pm) * n / (n + k)).astype(np.float32), float(k), float(sigma2), float(tau2)


def composite_offset(offsets, scenes):
    """Offset of a composite: mean over its scenes (valid-mask field, track) of that track's offset, 0 where no scene."""
    total, count = np.zeros(offsets.shape[1:], dtype=np.float64), np.zeros(offsets.shape[1:], dtype=np.int32)
    for valid, track in scenes:
        total += np.where(valid, offsets[track], 0.0)
        count += valid
    return total / np.maximum(count, 1)


def scene_masks(paths, shape):
    out = []
    for path, track in paths:
        with xr.open_dataset(path) as ds:
            out.append((place(np.isfinite(ds["soil_moisture"].values).astype(np.float32), shape, 1) > 0, int(track)))
    return out


def within_pixel_var(v, pix):
    n = np.bincount(pix)
    return float(((v - (np.bincount(pix, v) / n)[pix]) ** 2).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data_qc/cache/aligned_AM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="data_qc/smap3km")
    ap.add_argument("--coarse", default="9km")
    ap.add_argument("--fine", default="3km")
    ap.add_argument("--factor", type=int, default=3)
    ap.add_argument("--fit", nargs=2, default=["2021-01-01", "2024-06-30"])
    ap.add_argument("--write", nargs=2, default=["2021-01-01", "2024-12-31"])
    ap.add_argument("--check", nargs=2, default=["2024-07-01", "2024-12-31"])
    ap.add_argument("--exclude-dates", nargs="*", default=["2022-04-26"])
    ap.add_argument("--exclude-buffer-days", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=5)
    a = ap.parse_args()

    groups = e9.scene_groups(a.scenes)
    rows = [(d, str(f), s1, sat) for d, g in groups.items() for f, _, s1, sat in g]
    table = pd.DataFrame(rows, columns=["date", "path", "s1_time", "sat"])
    table["track"] = assign_tracks(table.s1_time, table.sat)                 # acquisition times only, no label values
    n_tracks = int(table.track.max()) + 1
    excl = {pd.Timestamp(d) + pd.Timedelta(days=k) for d in a.exclude_dates for k in range(-a.exclude_buffer_days, a.exclude_buffer_days + 1)}

    def scenes_of(d, shape):
        g = table[table.date == d]
        return scene_masks(list(zip(g.path, g.track)), shape)

    fit_dates = [d for d in e1.pair_dates(a.cache, a.fine, *a.fit) if d not in excl]
    n = s = ss = None
    for d in fit_dates:
        c, f = e1.load(a.cache, a.coarse, d), e1.load(a.cache, a.fine, d)
        if c is None or f is None:
            continue
        if n is None:
            n, s, ss = (np.zeros((n_tracks, *f.shape)) for _ in range(3))
        r = f.astype(np.float64) - bilinear(c, a.factor)
        scenes = scenes_of(d, f.shape)
        count = sum(v.astype(np.int32) for v, _ in scenes) if scenes else np.zeros(f.shape, dtype=np.int32)
        tracks = np.full(f.shape, -1, dtype=np.int32)
        for v, t in scenes:
            tracks[v & (tracks == -1)] = t
            tracks[v & (tracks != t)] = -2
        i, j = np.nonzero(np.isfinite(r) & (count > 0) & (tracks >= 0))
        t = tracks[i, j]
        np.add.at(n, (t, i, j), 1)
        np.add.at(s, (t, i, j), r[i, j])
        np.add.at(ss, (t, i, j), r[i, j] ** 2)
    offsets, k, sigma2, tau2 = fit_offsets(n, s, ss, a.min_count)
    print(f"{len(fit_dates)} fit dates, {int((n >= a.min_count).sum())} cells, k = {k:.1f}, within sd {np.sqrt(sigma2):.4f}, between sd {np.sqrt(max(tau2, 0)):.4f}, "
          f"rms offset {np.sqrt((offsets[n >= a.min_count] ** 2).mean()):.4f}", flush=True)

    out = Path(a.out)
    (out / a.fine).mkdir(parents=True, exist_ok=True)
    link = out / a.coarse
    if not link.exists():
        link.symlink_to((Path(a.cache) / a.coarse).resolve())
    np.savez_compressed(out / "offsets.npz", offsets=offsets, n=n.astype(np.int32), k=k, sigma2=sigma2, tau2=tau2)
    chk0, chk1 = pd.Timestamp(a.check[0]), pd.Timestamp(a.check[1])
    before, after, cpix, valid, moved, outside = [], [], [], 0, 0, 0
    written = 0
    for d in e1.pair_dates(a.cache, a.fine, *a.write):
        with np.load(Path(a.cache) / a.fine / f"{d:%Y%m%d}.npz") as z:
            raw = z["soil_moisture"].astype(np.float32)
        delta = composite_offset(offsets, scenes_of(d, raw.shape))
        fixed = (raw - delta).astype(np.float32)                             # NaN stays NaN
        np.savez_compressed(out / a.fine / f"{d:%Y%m%d}.npz", soil_moisture=fixed, mask=np.isfinite(fixed))
        written += 1
        ok = np.isfinite(raw)
        valid += int(ok.sum())
        moved += int((ok & (delta != 0)).sum())
        outside += int((ok & ((fixed < e1.CLIP[0]) | (fixed > e1.CLIP[1]))).sum())
        if chk0 <= d <= chk1:
            b = bilinear(e1.load(a.cache, a.coarse, d), a.factor)
            i = np.flatnonzero(ok & np.isfinite(b))
            before.append(np.clip(raw, *e1.CLIP).ravel()[i] - b.ravel()[i])
            after.append(np.clip(fixed, *e1.CLIP).ravel()[i] - b.ravel()[i])
            cpix.append(i)
    _, pix = np.unique(np.concatenate(cpix), return_inverse=True)
    vb, va = within_pixel_var(np.concatenate(before).astype(np.float64), pix), within_pixel_var(np.concatenate(after).astype(np.float64), pix)
    info = dict(args=vars(a), n_fit_dates=len(fit_dates), n_written=written, n_tracks=n_tracks, cells=int((n >= a.min_count).sum()), k=k, within_cell_sd=float(np.sqrt(sigma2)),
                between_cell_sd=float(np.sqrt(max(tau2, 0))), rms_offset=float(np.sqrt((offsets[n >= a.min_count] ** 2).mean())), share_pixel_days_moved=moved / valid,
                share_outside_clip=outside / valid, check=dict(within_pixel_var_before=vb, within_pixel_var_after=va, reduction=1 - va / vb, pixel_days=int(pix.size)), git=e1.git_sha("smapsr"))
    (out / f"PROVENANCE_{datetime.now():%Y%m%dT%H%M%S}.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps({x: info[x] for x in ("n_written", "share_pixel_days_moved", "share_outside_clip", "check")}, indent=2))


if __name__ == "__main__":
    main()
