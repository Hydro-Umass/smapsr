"""E9 information audit: is part of the label's day-to-day variation tied to Sentinel-1 viewing geometry, and does the
day-specific gain survive within one geometry?

A track is a satellite plus the phase of the S1 acquisition time modulo 12 days (the repeat cycle), clustered with a
20-minute gap (`assign_tracks`); it stands in for the relative orbit, which the files do not record. Composite pixel-days
with one contributing track keep it; pixels with several tracks that day are dropped from this analysis.

Reports, on the same pixel-days:
  variance  share of within-pixel variance explained by pixel x track means, adjusted for chance (`track_variance`),
            for the label and, as a control, for bilinear coarse (the coarse field knows nothing of S1 geometry) and the
            model predictions; overall and per year
  gate      as in E1 (own pixel mean removed) and within geometry (own pixel x track mean removed), with the gain over
            the null on each; cells with >= min_count rows
  offset_R  correlation of label and prediction pixel x track offsets (cell mean minus pixel mean), weighted by row count
  pairs     correlation of label-minus-prediction residuals between a row and the same pixel's latest earlier scene of the
            same / a different track, by gap class. Rows are split at random into two sets, each pair must fall in both, and
            each residual loses the mean of its own set without itself, which avoids the short-series bias of a shared mean.

Usage (project root):  PYTHONPATH=smapsr python smapsr/experiments/e9_orbit_split.py --v1 <run dirs> [--v1-names ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e1_baselines as e1                                             # noqa: E402
import e9_noise_ceiling as e9                                         # noqa: E402
from build_aligned_cache import place                                 # noqa: E402
from smapsr import evals                                              # noqa: E402
from smapsr.baselines import bilinear, parent_valid                   # noqa: E402

REPEAT = pd.Timedelta(days=12)
GAPS = [(1, 6), (7, 10), (11, 13), (14, 18), (19, 26), (27, 10_000)]


def assign_tracks(times, sats, gap_minutes=20):
    """Integer track per scene: per satellite, phases of `times` modulo the 12-day repeat, split where consecutive
    phases differ by more than gap_minutes (circular, so a cluster across the wrap stays together)."""
    times, sats = pd.DatetimeIndex(times), np.asarray(sats)
    out = np.full(len(times), -1, dtype=np.int32)
    next_id = 0
    for sat in np.unique(sats):
        i = np.flatnonzero(sats == sat)
        phase = ((times[i] - pd.Timestamp("2015-01-01")) % REPEAT) / pd.Timedelta(minutes=1)
        phase = np.asarray(phase)
        order = np.argsort(phase)
        d = np.diff(phase[order], append=phase[order][0] + 12 * 1440)
        starts = np.flatnonzero(d > gap_minutes)                        # last scene of each cluster
        cluster = np.searchsorted(starts, np.arange(len(i)))            # scenes after the last split rejoin cluster 0
        cluster[cluster == len(starts)] = 0
        out[i[order]] = next_id + cluster
        next_id += len(starts)
    return out


def track_variance(v, pix, cell):
    """(share, total within-pixel variance): between-cell variance component within pixels (one-way ANOVA estimate, chance
    removed) over the within-pixel variance. pix and cell are dense ids; every cell lies within one pixel."""
    n, nc = np.bincount(pix), np.bincount(cell)
    mp, mc = np.bincount(pix, v) / n, np.bincount(cell, v) / nc
    pix_of_cell = np.zeros(nc.size, dtype=np.int64)
    pix_of_cell[cell] = pix
    ss_total = ((v - mp[pix]) ** 2).sum()
    ss_between = (nc * (mc - mp[pix_of_cell]) ** 2).sum()
    ss_within = ss_total - ss_between
    df_b, df_w = nc.size - n.size, v.size - nc.size
    comp = ss_between - df_b * ss_within / max(df_w, 1)
    return float(comp / ss_total), float(ss_total / v.size)


def previous_rows(pix, k, track):
    """For rows (any order): index of the same pixel's latest earlier row of the same / a different track, or -1."""
    order = np.lexsort((k, pix))
    same, diff = np.full(pix.size, -1, dtype=np.int64), np.full(pix.size, -1, dtype=np.int64)
    last, cur = {}, -1
    for i in order:
        if pix[i] != cur:
            last, cur = {}, pix[i]
        t = track[i]
        same[i] = last.get(t, -1)
        others = [r for tr, r in last.items() if tr != t]
        diff[i] = max(others, key=lambda r: k[r]) if others else -1
        last[t] = i
    return same, diff


def split_pair_corr(r, pix, a, b, rng):
    """corr(r[a], r[b]) for row pairs, each minus its pixel's mean over a separate random set of rows."""
    side = rng.integers(0, 2, r.size)
    use = side[a] != side[b]
    a, b = a[use], b[use]
    if a.size < 3:
        return np.nan, int(a.size)
    n, s = np.zeros((2, pix.max() + 1)), np.zeros((2, pix.max() + 1))
    np.add.at(n, (side, pix), 1)
    np.add.at(s, (side, pix), r)
    m = lambda i: (s[side[i], pix[i]] - r[i]) / np.maximum(n[side[i], pix[i]] - 1, 1)   # noqa: E731
    ok = (n[side[a], pix[a]] > 1) & (n[side[b], pix[b]] > 1)
    a, b = a[ok], b[ok]
    return float(np.corrcoef(r[a] - m(a), r[b] - m(b))[0, 1]), int(a.size)


def scene_track_table(a, start, end):
    groups = e9.scene_groups(a.scenes)
    rows = [(d, str(f), s1, sat) for d, g in groups.items() for f, _, s1, sat in g]
    t = pd.DataFrame(rows, columns=["date", "path", "s1_time", "sat"])
    t["track"] = assign_tracks(t.s1_time, t.sat)
    return t[(t.date >= pd.Timestamp(start)) & (t.date <= pd.Timestamp(end))], t


def track_field(paths, shape):
    """Aligned-grid int field: the track of the only contributing scene, -1 no scene, -2 several tracks."""
    out = np.full(shape, -1, dtype=np.int32)
    for path, tr in paths:
        with xr.open_dataset(path) as ds:
            v = place(np.isfinite(ds["soil_moisture"].values).astype(np.float32), shape, 1) > 0     # raw -> aligned, offset 1
        out[v & (out == -1)] = tr
        out[v & (out >= 0) & (out != tr)] = -2
    return out


def main():
    p = e1.build_parser()
    p.set_defaults(out_dir="results/e9_audit/orbit", v1_tile=0)
    p.add_argument("--scenes", default="data_qc/smap3km")
    p.add_argument("--v1-names", nargs="*", default=[], help="names for --v1 runs, in order (default v1_<n>)")
    a = p.parse_args()
    t0 = time.time()
    loaded = e1.load_v1_models(a.v1, a.v1_tile, a.v1_halo)
    names = a.v1_names or list(loaded)
    models = dict(zip(names, loaded.values()))
    clim, n_clim = e1.fit_climatology(a, e1.train_dates(a))
    tracks, all_tracks = scene_track_table(a, *a.test)
    print(f"{all_tracks.track.nunique()} tracks over {len(all_tracks)} AM scenes; climatology {n_clim} dates; {time.time() - t0:.0f} s", flush=True)
    methods = ["bilinear", "clim_null"] + names
    rows, used = [], []
    n_mixed = n_any = 0
    dates = e1.pair_dates(a.cache, a.fine, *a.test)
    dates = dates[:a.limit] if a.limit else dates
    shape = None
    for d in dates:
        c, f = e1.load(a.cache, a.coarse, d), e1.load(a.cache, a.fine, d)
        if c is None or f is None:
            continue
        g = tracks[tracks.date == d]
        shape = f.shape
        tf = track_field(list(zip(g.path, g.track)), f.shape)
        ev = np.isfinite(f) & parent_valid(c, a.factor) & clim.covered
        n_any += int(ev.sum())
        n_mixed += int((ev & (tf == -2)).sum())
        ev &= tf >= 0
        if not ev.any():
            continue
        preds = {"bilinear": bilinear(c, a.factor), "clim_null": clim.null_model(c)}
        for n, fn in models.items():
            preds[n] = fn(c)
        idx = np.flatnonzero(ev)
        rows.append(dict(k=np.full(idx.size, len(used), dtype=np.int32), day=np.full(idx.size, (d - dates[0]).days, dtype=np.int32), year=np.full(idx.size, d.year, dtype=np.int16),
                         pix=idx, track=tf.ravel()[idx], y=f.ravel()[idx], **{m: preds[m].ravel()[idx].astype(np.float32) for m in methods}))
        used.append(d)
        if len(used) % 100 == 0:
            print(f"  {len(used)}/{len(dates)} dates, {time.time() - t0:.0f} s", flush=True)
    t = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    ok = np.logical_and.reduce([np.isfinite(t[m]) for m in methods])
    t = {k: v[ok] for k, v in t.items()}
    cell_key = t["pix"].astype(np.int64) * 10_000 + t["track"]
    _, cell, nc = np.unique(cell_key, return_inverse=True, return_counts=True)
    keep = nc[cell] >= a.min_count
    t = {k: v[keep] for k, v in t.items()}
    _, pix = np.unique(t["pix"], return_inverse=True)
    _, cell = np.unique(t["pix"].astype(np.int64) * 10_000 + t["track"], return_inverse=True)
    y = t["y"].astype(np.float64)
    print(f"{y.size} pixel-days, {pix.max() + 1} pixels, {cell.max() + 1} pixel-track cells, {len(used)} dates, {n_mixed / max(n_any, 1):.1%} of evaluation pixel-days had several tracks; {time.time() - t0:.0f} s", flush=True)

    out = []
    # ---- variance
    for name, v in [("label", y)] + [(m, t[m].astype(np.float64)) for m in methods]:
        share, var = track_variance(v, pix, cell)
        out.append(dict(table="variance", method=name, year="all", value=share, within_pixel_var=var, n=int(y.size)))
        for yr in np.unique(t["year"]):
            s = t["year"] == yr
            _, pp = np.unique(pix[s], return_inverse=True)
            _, cc = np.unique(cell[s], return_inverse=True)
            if s.sum() > cc.max() + 1:
                sh, vr = track_variance(v[s], pp, cc)
                out.append(dict(table="variance", method=name, year=int(yr), value=sh, within_pixel_var=vr, n=int(s.sum())))
    # ---- offset_R
    nc, np_ = np.bincount(cell), np.bincount(pix)
    pix_of_cell = np.zeros(nc.size, dtype=np.int64)
    pix_of_cell[cell] = pix
    offset = lambda v: np.bincount(cell, v) / nc - (np.bincount(pix, v) / np_)[pix_of_cell]     # noqa: E731
    oy = offset(y)
    for m in methods:
        op = offset(t[m].astype(np.float64))
        c = np.cov(oy, op, aweights=nc)
        out.append(dict(table="offset_R", method=m, year="all", value=float(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])), n=int(nc.size)))
    # ---- gate, own pixel mean and own pixel x track mean, gain over the null
    D, M = len(used), len(methods)
    ids = dict(pixel=pix, geometry=cell)
    S = {kind: np.zeros((D, M, 6)) for kind in ids}
    for kind, g in ids.items():
        n_ = np.bincount(g)
        o = y - (np.bincount(g, y) / n_)[g]
        for m, name in enumerate(methods):
            v = t[name].astype(np.float64)
            q = v - (np.bincount(g, v) / n_)[g]
            for i, w in enumerate((np.ones_like(o), o, q, o * o, q * q, o * q)):
                S[kind][:, m, i] = np.bincount(t["k"], w, minlength=D)
    null = methods.index("clim_null")

    def stat(i):
        res = {}
        for kind in ids:
            r = e1.pooled(S[kind][i].sum(0))["R"]
            for m, name in enumerate(methods):
                res[f"{name}|{kind}"] = r[m]
                res[f"{name}-clim_null|{kind}"] = r[m] - r[null]
        return res

    point = stat(np.arange(D))
    boot = evals.block_bootstrap(stat, evals.date_blocks(used, a.block_days), n_boot=a.n_boot, seed=a.seed)
    for key, v in point.items():
        who, kind = key.split("|")
        lo, hi = evals.ci(boot[key])
        out.append(dict(table=f"gate_{kind}", method=who, year="all", value=v, ci_lo=lo, ci_hi=hi, p_paired=evals.paired_p(boot[key]) if "-" in who else np.nan, n=int(y.size)))
    # ---- pairs
    same, diff = previous_rows(pix, t["day"], t["track"])
    rng = np.random.default_rng(a.seed)
    for m in methods:
        r = y - t[m].astype(np.float64)
        for kind, prev in (("same_track", same), ("different_track", diff)):
            has = np.flatnonzero(prev >= 0)
            gap = t["day"][has] - t["day"][prev[has]]
            for lo, hi in GAPS:
                s = (gap >= lo) & (gap <= hi)
                rr, n = split_pair_corr(r, pix, has[s], prev[has[s]], rng)
                out.append(dict(table="pairs", method=m, year=kind, gap=f"{lo}-{hi}", value=rr, n=n))
    metrics = pd.DataFrame(out)
    run = Path(a.out_dir) / f"run_{pd.Timestamp.now():%Y%m%dT%H%M%S}"
    run.mkdir(parents=True)
    metrics.to_csv(run / "metrics.csv", index=False)
    all_tracks.drop(columns="path").groupby(["sat", "track"]).agg(n=("date", "size"), first=("date", "min"), last=("date", "max")).reset_index().to_csv(run / "tracks.csv", index=False)
    cfg = dict(args=vars(a), methods=methods, v1_names=names, n_dates=D, n_clim_dates=n_clim, pixel_days=int(y.size), pixels=int(pix.max() + 1), cells=int(cell.max() + 1),
               share_mixed_tracks=n_mixed / max(n_any, 1), n_tracks=int(all_tracks.track.nunique()), git=e1.git_sha("smapsr"), runtime_s=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    print(metrics[metrics.table.isin(["gate_pixel", "gate_geometry"]) & ~metrics.method.str.contains("-")].pivot_table(index="method", columns="table", values="value").round(4).to_string())
    print(metrics[(metrics.table == "variance") & (metrics.year == "all")][["method", "value", "within_pixel_var"]].round(4).to_string())
    print("wrote", run)


if __name__ == "__main__":
    main()
