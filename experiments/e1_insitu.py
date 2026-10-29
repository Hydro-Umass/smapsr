"""E1 in-situ reference: every E1 method vs. the NZ ground stations on the held-out period (2016-2020).

Same methods, climatology and V1 runs as `e1_baselines.py` (shared helpers), evaluated at the 7 stations with the E0
protocol: station value = mean of the method's valid 3-km cells centred within `--buffer-km` of the station (5 km is
the decided headline radius), in-situ value = mean within +-`--window-min` of the nominal overpass
(`e0_hygiene.nominal_overpass`, 06 h local solar time). Stations are never used for training or for the climatology.
Rows are matched on all test dates with a valid coarse field, not only on label-days. Two subsets:
  common      every model method (not the label) has a finite value, the in-situ mean exists;
  label_days  as common, plus the AM SPL2SMAP_S label itself has valid cells in the buffer -> the label is a method.
Pooled CIs: calendar-block bootstrap over dates, paired differences vs. bilinear and clim_null from the same resamples.
Per-site rows are point estimates (small N; no CIs).

Usage (project root):  PYTHONPATH=smapsr python smapsr/experiments/e1_insitu.py --v1 results/v1_train/run_*/ ...
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_baselines as e1                      # noqa: E402
import e0_hygiene as e0                        # noqa: E402
from smapsr import evals, io                   # noqa: E402
from smapsr.baselines import bilinear, block_constant  # noqa: E402

LAT0, LON0, STEP = -32.77, 164.67, 0.03        # aligned 3-km grid: raw 3-km origin shifted one cell N/W (AGENTS.md §5)


def main():
    p = e1.build_parser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--buffer-km", type=float, default=5.0)
    p.add_argument("--overpass-local-hour", type=float, default=6.0)
    p.add_argument("--window-min", type=float, default=90.0)
    p.add_argument("--min-insitu-samples", type=int, default=1)
    a = p.parse_args()
    t0 = time.time()
    v1 = e1.load_v1_models(a.v1, a.v1_tile, a.v1_halo)
    methods = e1.BASE_METHODS + list(v1)
    clim, n_train = e1.fit_climatology(a, e1.train_dates(a))
    C = clim.label_mean

    sites = io.read_sites(Path(a.data_dir) / "sites.txt")
    ground = {s["site"]: io.read_ground(Path(a.data_dir) / "ground" / f"{s['site'].lower()}.csv")
              for _, s in sites.iterrows() if (Path(a.data_dir) / "ground" / f"{s['site'].lower()}.csv").exists()}
    sites = sites[sites.site.isin(ground)].reset_index(drop=True)
    lat, lon = LAT0 - STEP * np.arange(501), LON0 + STEP * np.arange(498)
    cells = [io.buffer_indices(lat, lon, s["lat"], s["lon"], a.buffer_km) for _, s in sites.iterrows()]

    dates = [d for d in e1.pair_dates(a.cache, a.coarse, *a.test)]
    if a.limit:
        dates = dates[:a.limit]
    vals = {m: np.full((len(dates), len(sites)), np.nan) for m in methods + ["label"]}
    for k, d in enumerate(dates):
        c = e1.load(a.cache, a.coarse, d)
        if c is None or not np.isfinite(c).any():
            continue
        f = e1.load(a.cache, a.fine, d)
        preds = {"bilinear": bilinear(c, a.factor), "block_constant": block_constant(c, a.factor),
                 "clim_null": clim.null_model(c), "clim_only": C}
        for name, fn in v1.items():
            preds[name] = fn(c)
        if f is not None:
            preds["label"] = f
        for name, field in preds.items():
            for s, (r, q) in enumerate(cells):
                v = field[r, q]
                v = v[np.isfinite(v)]
                if v.size:
                    vals[name][k, s] = v.mean()
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(dates)} dates", flush=True)

    rows = []
    for s, (_, site) in enumerate(sites.iterrows()):
        centre = e0.nominal_overpass(dates, site["lon"], a.overpass_local_hour)
        m, n = e0.window_mean(ground[site["site"]], centre, a.window_min)
        df = pd.DataFrame(dict(date=dates, site=site["site"], insitu=m, insitu_n=n, **{k: v[:, s] for k, v in vals.items()}))
        rows.append(df[(df.insitu_n >= a.min_insitu_samples) & np.isfinite(df.insitu)])
    mu = pd.concat(rows, ignore_index=True)
    mu["year"] = mu.date.dt.year
    sub = {"common": np.isfinite(mu[methods].to_numpy()).all(1)}
    sub["label_days"] = sub["common"] & np.isfinite(mu["label"].to_numpy())
    print({k: int(v.sum()) for k, v in sub.items()}, f"{time.time() - t0:.0f} s", flush=True)

    out = []
    for sname, ok in sub.items():
        d = mu[ok].reset_index(drop=True)
        who = methods + (["label"] if sname == "label_days" else [])
        blocks = evals.date_blocks(d.date, a.block_days)

        def stat(idx, d=d, who=who):
            r = {}
            for m in who:
                pm = evals.point_metrics(d.insitu.to_numpy()[idx], d[m].to_numpy()[idx])
                for k in ("R", "RMSE", "ubRMSE", "bias"):
                    r[f"{m}|{k}"] = pm[k]
            for ref in ("bilinear", "clim_null"):
                for m in who:
                    if m != ref:
                        for k in ("R", "RMSE", "ubRMSE"):
                            r[f"{m}-{ref}|{k}"] = r[f"{m}|{k}"] - r[f"{ref}|{k}"]
            return r

        point = stat(np.arange(len(d)))
        boot = evals.block_bootstrap(stat, blocks, n_boot=a.n_boot, seed=a.seed)
        for key, v in point.items():
            who_, metric = key.split("|")
            lo, hi = evals.ci(boot[key])
            out.append(dict(subset=sname, scope="pooled", method=who_, metric=metric, value=v, ci_lo=lo, ci_hi=hi,
                            p_paired=evals.paired_p(boot[key]) if "-" in who_ else np.nan, N=len(d)))
        for site, g in d.groupby("site"):
            for m in who:
                pm = evals.point_metrics(g.insitu.to_numpy(), g[m].to_numpy())
                for k in ("R", "RMSE", "ubRMSE", "bias"):
                    out.append(dict(subset=sname, scope=site, method=m, metric=k, value=pm[k], N=pm["N"]))
    metrics = pd.DataFrame(out)
    run = Path(a.out_dir) / f"run_{pd.Timestamp.now():%Y%m%dT%H%M%S}_insitu"
    run.mkdir(parents=True)
    metrics.to_csv(run / "metrics.csv", index=False)
    mu.to_csv(run / "matchups.csv", index=False)
    cfg = dict(args=vars(a), methods=methods, n_train_dates=n_train, sites=sites.site.tolist(),
               n_cells_in_buffer=[len(c[0]) for c in cells], n_matchups={k: int(v.sum()) for k, v in sub.items()},
               per_site_n={k: int(g.shape[0]) for k, g in mu[sub["common"]].groupby("site")}, v1_runs=a.v1, v1_tile=a.v1_tile, v1_halo=a.v1_halo,
               grid="aligned 3 km, lat0 -32.77, lon0 164.67, step 0.03 (raw origin shifted one cell N/W)",
               git=e1.git_sha("smapsr"), runtime_s=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    show = metrics[(metrics.scope == "pooled") & ~metrics.method.str.contains("-") & metrics.metric.isin(["R", "RMSE", "ubRMSE", "bias"])]
    print(show.pivot_table(index=["subset", "method"], columns="metric", values="value").round(4).to_string())
    print("wrote", run)


if __name__ == "__main__":
    main()
