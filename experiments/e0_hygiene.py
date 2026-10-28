"""E0 validation hygiene (model-independent part).

Builds the evaluation protocol every later experiment uses and the reference-product
ceilings it is compared against (strategy §5, E0; AGENTS.md §8):

1. Label-day inventory per year and satellite (Sentinel-1B failure, Dec 2021) and the
   flipped train/test split (train 2021-2024 minus the SlimSAR window, test 2016-2020).
2. Per-date input coverage (9 km, 36 km) and per-pixel label density (3 km, 1 km).
3. SMAP products vs. the NZ ground stations on all overlapping dates (stations are
   independent of SMAP calibration), per site, pooled, and per year, with calendar-block
   bootstrap CIs, plus paired product-vs-product tests.
4. Daily vs. time-mean spatial correlation of the 9-km input (nearest-neighbour onto the
   3-km grid) against SPL2SMAP_S 3 km: how much a time-mean R flatters a field.

No model is evaluated here. Writes results/e0_hygiene/run_<timestamp>/.

Usage (from the project root, devenv shell active):
    PYTHONPATH=smapsr python smapsr/experiments/e0_hygiene.py
    PYTHONPATH=smapsr python smapsr/experiments/e0_hygiene.py --years 2017 --n-boot 50  # smoke
"""
import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from smapsr import io
from smapsr import evals

PRODUCTS = {
    # name: (directory under data/, glob, role)
    "SPL3SMP_36km": ("smap36km", "*.nc", "native anchor"),
    "SPL3SMP_E_9km": ("smap9km", "*.nc", "model input"),
    "SPL2SMAP_S_3km": ("smap3km_daily", "*_daily.nc", "label"),
    "SPL2SMAP_S_1km": ("smap1km_daily", "*_daily.nc", "label"),
}
METRICS = ["N", "R", "RMSE", "ubRMSE", "bias"]
UNITS = dict(N="count", R="-", RMSE="m3/m3", ubRMSE="m3/m3", bias="m3/m3")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="project data directory (sites.txt, ground/)")
    p.add_argument("--smap-dir", default=None,
                   help="root of the SMAP products (smap9km, smap36km, smap3km, smap{3,1}km_daily); default: --data-dir. "
                        "Use data_qc (primary since 2026-10-27)")
    p.add_argument("--out-dir", default="results/e0_hygiene", help="experiment results directory")
    p.add_argument("--train-start", default="2021-01-01")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--test-start", default="2016-01-01")
    p.add_argument("--test-end", default="2020-12-31")
    p.add_argument("--exclude-dates", nargs="*", default=["2022-04-26"],
                   help="dates removed from training (SlimSAR acquisitions)")
    p.add_argument("--exclude-buffer-days", type=int, default=4)
    p.add_argument("--overpass-local-hour", type=float, default=6.0,
                   help="nominal SMAP AM local solar time; in-situ is averaged around it")
    p.add_argument("--window-min", type=float, default=90.0,
                   help="half-width (minutes) of the in-situ averaging window around the overpass")
    p.add_argument("--buffer-km", type=float, nargs="+", default=[0, 3, 5, 10],
                   help="station buffer radii: product value = mean of valid cells centred within the radius "
                        "(plus the nearest cell); 0 = nearest cell only. Every radius is reported.")
    p.add_argument("--block-days", type=int, default=30, help="calendar block length for the bootstrap")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-pixels", type=int, default=100,
                   help="minimum joint-valid 3-km pixels for a daily spatial R")
    p.add_argument("--min-days", type=int, default=10,
                   help="minimum joint-valid days for a pixel to enter the time-mean field")
    p.add_argument("--years", type=int, nargs="*", default=None,
                   help="restrict to these years (smoke runs); default: all")
    return p.parse_args()


def git_state(repo):
    def run(*cmd):
        return subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True, text=True).stdout.strip()
    return dict(sha=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(run("status", "--porcelain")))


def assign_split(dates, args):
    """'test', 'train' or 'excluded' per date (flipped split, AGENTS.md §8)."""
    d = pd.DatetimeIndex(dates)
    out = np.full(len(d), "outside", dtype=object)
    out[(d >= args.test_start) & (d <= args.test_end)] = "test"
    out[(d >= args.train_start) & (d <= args.train_end)] = "train"
    for x in args.exclude_dates:
        c = pd.Timestamp(x)
        win = (d >= c - pd.Timedelta(days=args.exclude_buffer_days)) & (d <= c + pd.Timedelta(days=args.exclude_buffer_days))
        out[win & (out == "train")] = "excluded"
    return out


def nominal_overpass(dates, lon, local_hour):
    """UTC time of `local_hour` local solar time on the morning after each UTC date token.

    A file dated D (UTC) holds the descending pass at ~18 UTC on D, i.e. ~06 local on D+1."""
    return pd.DatetimeIndex(dates).tz_localize("UTC") + pd.to_timedelta(24 + local_hour - lon / 15.0, unit="h")


def window_mean(series, centres, half_width_min):
    """Mean and count of in-situ samples within ±half_width of each centre time."""
    # Compare in nanoseconds: pandas may store either index at a coarser unit.
    t = series.index.as_unit("ns").asi8
    v = series.to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(v)])
    hw = pd.Timedelta(minutes=half_width_min).value
    c = pd.DatetimeIndex(centres).as_unit("ns").asi8
    lo = np.searchsorted(t, c - hw, side="left")
    hi = np.searchsorted(t, c + hw, side="right")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    return m, n


def scan_products(args, files, sites, am_label_dates):
    """One pass over every product file: station pixels, coverage, yearly density, spatial R.

    The spatial R uses only label-days whose scenes are all AM passes (same pass as the input)."""
    station_vals = {}       # (product, buffer_km) -> DataFrame(date x site)
    buffers = []            # rows: product, site, buffer_km, n_cells
    coverage = []           # rows: product, date, n_valid
    density = {}            # product -> {year: count map}
    union = {}
    grids = {}
    input9 = {}             # 9-km fields kept for label dates (for the spatial R)
    input9_grid = None      # set while scanning the 9-km product (scanned before 3 km)
    label_dates = set(files["SPL2SMAP_S_3km"].index)
    daily_r = []
    tm = None               # time-mean accumulators per split

    for name in ["SPL3SMP_36km", "SPL3SMP_E_9km", "SPL2SMAP_S_3km", "SPL2SMAP_S_1km"]:
        fs = files[name]
        first = io.read_grid(fs.iloc[0])
        lat, lon = first["lat"].values, first["lon"].values
        grids[name] = dict(n_lat=len(lat), n_lon=len(lon), lat0=float(lat[0]), lon0=float(lon[0]),
                           dlat=float(lat[1] - lat[0]), dlon=float(lon[1] - lon[0]))
        cells = {}
        for r in args.buffer_km:
            for s, (_, site) in enumerate(sites.iterrows()):
                cells[(r, s)] = io.buffer_indices(lat, lon, site["lat"], site["lon"], r)
                buffers.append(dict(product=name, site=site["site"], buffer_km=r, n_cells=len(cells[(r, s)][0])))
        vals = {r: np.full((len(fs), len(sites)), np.nan, dtype=np.float32) for r in args.buffer_km}
        union[name] = np.zeros((len(lat), len(lon)), dtype=bool)
        density[name] = {}
        if name == "SPL2SMAP_S_3km":
            i9 = io.nearest_index(input9_grid["lat"], lat)
            j9 = io.nearest_index(input9_grid["lon"], lon)
            tm = {s: dict(sx=np.zeros((len(lat), len(lon))), sy=np.zeros((len(lat), len(lon))),
                          n=np.zeros((len(lat), len(lon)), dtype=np.int32)) for s in ("train", "test")}
        for k, (date, f) in enumerate(tqdm(fs.items(), total=len(fs), desc=name)):
            a = io.read_grid(f).values
            ok = np.isfinite(a)
            for (r, s), (ri, ci_) in cells.items():
                v = a[ri, ci_]
                v = v[np.isfinite(v)]
                if v.size:
                    vals[r][k, s] = v.mean()
            coverage.append((name, date, int(ok.sum())))
            union[name] |= ok
            dmap = density[name].setdefault(date.year, np.zeros(a.shape, dtype=np.int16))
            dmap += ok
            if name == "SPL3SMP_E_9km":
                if k == 0:
                    input9_grid = dict(lat=lat, lon=lon)
                if date in label_dates:
                    input9[date] = a
            if name == "SPL2SMAP_S_3km" and date in input9 and date in am_label_dates:
                x = input9[date][np.ix_(i9, j9)]
                split = assign_split([date], args)[0]
                both = np.isfinite(x) & ok
                daily_r.append(dict(date=date, split=split, n_pixels=int(both.sum()),
                                    R=evals.spatial_corr(x, a, args.min_pixels)))
                if split in tm:
                    acc = tm[split]
                    acc["sx"][both] += x[both]
                    acc["sy"][both] += a[both]
                    acc["n"][both] += 1
        for r in args.buffer_km:
            station_vals[(name, r)] = pd.DataFrame(vals[r], index=fs.index, columns=sites["site"].values)
    cov = pd.DataFrame(coverage, columns=["product", "date", "n_valid"])
    return station_vals, cov, density, union, grids, pd.DataFrame(daily_r), tm, pd.DataFrame(buffers)


def insitu_matchups(args, ground, sites, station_vals, splits_of):
    """Long table: product, buffer, site, date, product value, in-situ window mean, sample count."""
    rows = []
    for _, s in sites.iterrows():
        if s["site"] not in ground:
            continue
        g = ground[s["site"]]
        for (name, r), df in station_vals.items():
            dates = df.index
            centre = nominal_overpass(dates, s["lon"], args.overpass_local_hour)
            m, n = window_mean(g, centre, args.window_min)
            rows.append(pd.DataFrame(dict(product=name, buffer_km=r, site=s["site"], date=dates,
                                          product_sm=df[s["site"]].values,
                                          insitu_sm=m, insitu_n=n, overpass_utc=centre)))
    mu = pd.concat(rows, ignore_index=True)
    mu = mu[np.isfinite(mu["product_sm"]) & np.isfinite(mu["insitu_sm"])].reset_index(drop=True)
    mu["split"] = splits_of(mu["date"])
    mu["year"] = mu["date"].dt.year
    return mu


def metric_rows(args, mu):
    """Per product x buffer x (site | pooled) x (all | year) metrics with block-bootstrap CIs."""
    rows = []
    groups = []
    for (prod, r, site), g in mu.groupby(["product", "buffer_km", "site"]):
        groups.append((prod, r, f"insitu:{site}", "all", g))
        for y, gy in g.groupby("year"):
            groups.append((prod, r, f"insitu:{site}", f"year={y}", gy))
    for (prod, r), g in mu.groupby(["product", "buffer_km"]):
        groups.append((prod, r, "insitu:pooled", "all", g))
        for y, gy in g.groupby("year"):
            groups.append((prod, r, "insitu:pooled", f"year={y}", gy))
    for prod, r, ref, subset, g in tqdm(groups, desc="bootstrap"):
        obs, pred = g["insitu_sm"].to_numpy(), g["product_sm"].to_numpy()
        point = evals.point_metrics(obs, pred)
        boot = evals.block_bootstrap(lambda idx: evals.point_metrics(obs[idx], pred[idx]),
                                     evals.date_blocks(g["date"], args.block_days), args.n_boot, args.seed)
        for m in METRICS:
            lo, hi = (np.nan, np.nan) if m == "N" else evals.ci(boot[m])
            rows.append(dict(method=prod, reference=ref, metric=m, value=point[m], ci_low=lo, ci_high=hi,
                             unit=UNITS[m], subset=subset, n=point["N"], buffer_km=r))
    return pd.DataFrame(rows)


def paired_rows(args, mu):
    """Paired product-vs-product differences (A - B) on common (site, date) matchups, per buffer."""
    rows = []
    for r, mr in mu.groupby("buffer_km"):
        rows += [x | dict(buffer_km=r) for x in _paired_rows(args, mr)]
    return pd.DataFrame(rows)


def _paired_rows(args, mu):
    wide = mu.pivot_table(index=["site", "date"], columns="product", values="product_sm")
    obs = mu.drop_duplicates(["site", "date"]).set_index(["site", "date"])["insitu_sm"]
    rows = []
    for a, b in itertools.combinations(PRODUCTS, 2):
        if a not in wide or b not in wide:
            continue
        common = wide[[a, b]].dropna()
        for site in list(common.index.get_level_values("site").unique()) + ["pooled"]:
            c = common if site == "pooled" else common.xs(site, level="site", drop_level=False)
            if len(c) < 10:
                continue
            o = obs.loc[c.index].to_numpy()
            pa, pb = c[a].to_numpy(), c[b].to_numpy()

            def diff(idx):
                ma, mb = evals.point_metrics(o[idx], pa[idx]), evals.point_metrics(o[idx], pb[idx])
                return {m: ma[m] - mb[m] for m in ("R", "RMSE", "ubRMSE")} | {"absbias": abs(ma["bias"]) - abs(mb["bias"])}

            point = diff(np.arange(len(c)))
            boot = evals.block_bootstrap(diff, evals.date_blocks(c.index.get_level_values("date"), args.block_days),
                                         args.n_boot, args.seed)
            for m, v in point.items():
                lo, hi = evals.ci(boot[m])
                rows.append(dict(method_a=a, method_b=b, reference=f"insitu:{site}", metric=f"{m}(A)-{m}(B)",
                                 diff=v, ci_low=lo, ci_high=hi, p=evals.paired_p(boot[m]), n=len(c),
                                 unit="-" if m == "R" else "m3/m3"))
    return rows


def main():
    args = parse_args()
    t0 = time.time()
    root = Path(args.data_dir).resolve()
    smap_root = Path(args.smap_dir or args.data_dir).resolve()
    run_dir = Path(args.out_dir) / f"run_{time.strftime('%Y%m%dT%H%M%S')}"
    (run_dir / "figure_data").mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir()
    (run_dir / "inputs").mkdir()
    splits_of = lambda dates: assign_split(dates, args)

    # --- inventories -------------------------------------------------------------------
    files = {}
    for name, (d, pat, _) in PRODUCTS.items():
        fs = io.list_product(smap_root / d, pat)
        if args.years:
            fs = fs[fs.index.year.isin(args.years)]
        files[name] = fs
        fs.rename("path").to_frame().to_csv(run_dir / "inputs" / f"{name}.csv")
    scenes = pd.DataFrame([dict(name=p.name, **io.parse_scene_name(p)) for p in sorted((smap_root / "smap3km").glob("*.nc"))])
    scenes["date"] = pd.to_datetime([io.file_date(n) for n in scenes["name"]])
    if args.years:
        scenes = scenes[scenes["date"].dt.year.isin(args.years)]
    scenes["nominal_utc"] = [nominal_overpass([d], lon, args.overpass_local_hour)[0]
                             for d, lon in zip(scenes["date"], scenes["tile_lon"])]
    scenes["offset_min"] = (scenes["smap_time"] - scenes["nominal_utc"]).dt.total_seconds() / 60
    # SMAP time ~12 h before the AM nominal = the ~18 local PM (ascending) pass of the same UTC date.
    scenes["pass"] = np.where(scenes["offset_min"].abs() <= 180, "AM", np.where((scenes["offset_min"] + 720).abs() <= 180, "PM", "other"))
    scenes.to_csv(run_dir / "figure_data" / "scenes.csv", index=False)
    passes = scenes.groupby([scenes["date"].dt.year.rename("year"), "pass"]).size().unstack(fill_value=0)
    passes.to_csv(run_dir / "scene_passes_per_year.csv")
    am = scenes.loc[scenes["pass"] == "AM", "offset_min"]
    pd.Series(dict(n_scenes=len(scenes), n_am=len(am), am_offset_min_p05=am.quantile(0.05), am_offset_min_median=am.median(),
                   am_offset_min_p95=am.quantile(0.95))).to_csv(run_dir / "overpass_offsets.csv", header=["value"])

    label_days = files["SPL2SMAP_S_3km"].index
    per_day = scenes.groupby("date")["satellite"].agg(lambda s: set(s))
    ld = pd.DataFrame(dict(date=label_days, split=splits_of(label_days)))
    ld["year"] = ld["date"].dt.year
    ld["has_S1B"] = ld["date"].map(lambda d: "S1B" in per_day.get(d, set()))
    ld["only_S1A"] = ld["date"].map(lambda d: per_day.get(d, set()) == {"S1A"})
    day_pass = scenes.groupby("date")["pass"].agg(lambda s: set(s))
    ld["has_PM"] = ld["date"].map(lambda d: "PM" in day_pass.get(d, set()))
    ld["no_AM"] = ld["date"].map(lambda d: "AM" not in day_pass.get(d, set()))
    ld.to_csv(run_dir / "figure_data" / "label_days.csv", index=False)
    by_year = ld.groupby("year").agg(label_days=("date", "size"), label_days_with_S1B=("has_S1B", "sum"),
                                     label_days_S1A_only=("only_S1A", "sum"), label_days_with_PM_scene=("has_PM", "sum"),
                                     label_days_without_AM_scene=("no_AM", "sum"))
    sc = scenes.groupby([scenes["date"].dt.year, "satellite"]).size().unstack(fill_value=0)
    by_year = by_year.join(sc.add_prefix("scenes_")).fillna(0).astype(int)
    by_split = ld.groupby("split").size().rename("label_days")
    by_year.to_csv(run_dir / "label_days_per_year.csv")
    by_split.to_csv(run_dir / "label_days_per_split.csv")

    # --- ground stations ---------------------------------------------------------------
    sites = io.read_sites(root / "sites.txt")
    ground = {}
    gstats = []
    for _, s in sites.iterrows():
        f = root / "ground" / f"{s['site'].lower()}.csv"
        if not f.exists():
            gstats.append(dict(site=s["site"], file=None))
            continue
        raw = pd.read_csv(f)
        g = io.read_ground(f)
        ground[s["site"]] = g
        gstats.append(dict(site=s["site"], file=f.name, rows_raw=len(raw), timestamps=len(g),
                           duplicate_rows_averaged=len(raw) - len(g), start=g.index.min(), end=g.index.max(),
                           min=g.min(), max=g.max(), n_below_0p02=int((g < 0.02).sum())))
    gstats = pd.DataFrame(gstats)
    gstats.to_csv(run_dir / "ground_stations.csv", index=False)

    # --- product scan ------------------------------------------------------------------
    sites_used = sites[sites["site"].isin(list(ground))].reset_index(drop=True)
    am_label_dates = set(ld.loc[ld["date"].map(lambda d: day_pass.get(d, set()) == {"AM"}), "date"])
    station_vals, cov, density, union, grids, daily_r, tm, buffers = scan_products(args, files, sites_used, am_label_dates)
    buffers.to_csv(run_dir / "buffer_cells.csv", index=False)

    cov["frac_of_box"] = cov.apply(lambda r: r["n_valid"] / (grids[r["product"]]["n_lat"] * grids[r["product"]]["n_lon"]), axis=1)
    cov["frac_of_union"] = cov.apply(lambda r: r["n_valid"] / union[r["product"]].sum(), axis=1)
    cov["split"] = splits_of(cov["date"])
    cov.to_csv(run_dir / "figure_data" / "coverage_per_date.csv", index=False)
    cov["year"] = cov["date"].dt.year
    cov_year = cov.groupby(["product", "year"]).agg(
        n_dates=("date", "size"), median_frac_of_union=("frac_of_union", "median"),
        frac_days_lt2pct_box=("frac_of_box", lambda x: np.mean(x < 0.02)),
        frac_days_ge10pct_box=("frac_of_box", lambda x: np.mean(x >= 0.10)))
    cov_year.to_csv(run_dir / "coverage_per_year.csv")

    dens_rows = []
    for name, maps in density.items():
        u = union[name]
        for y, m in sorted(maps.items()):
            v = m[u]
            dens_rows.append(dict(product=name, year=y, union_pixels=int(u.sum()), pixels_never_valid_in_year=float(np.mean(v == 0)),
                                  p10=np.quantile(v, 0.1), median=np.median(v), p90=np.quantile(v, 0.9), max=int(v.max())))
    pd.DataFrame(dens_rows).to_csv(run_dir / "label_density_per_year.csv", index=False)
    np.savez_compressed(run_dir / "figure_data" / "density_3km.npz",
                        years=np.array(sorted(density["SPL2SMAP_S_3km"])),
                        counts=np.stack([density["SPL2SMAP_S_3km"][y] for y in sorted(density["SPL2SMAP_S_3km"])]),
                        **{f"grid_{k}": v for k, v in grids["SPL2SMAP_S_3km"].items()})

    # --- in-situ validation ------------------------------------------------------------
    mu = insitu_matchups(args, ground, sites_used, station_vals, splits_of)
    # In-situ is averaged around the AM overpass, so label composites containing PM scenes are dropped.
    is_label = mu["product"].str.startswith("SPL2SMAP_S")
    n_pm = int((is_label & ~mu["date"].isin(am_label_dates)).sum())
    mu = mu[~is_label | mu["date"].isin(am_label_dates)].reset_index(drop=True)
    mu.to_csv(run_dir / "insitu_matchups.csv", index=False)
    metrics = metric_rows(args, mu)
    paired = paired_rows(args, mu)
    paired.to_csv(run_dir / "paired_tests.csv", index=False)

    # --- daily vs time-mean spatial R ---------------------------------------------------
    daily_r.to_csv(run_dir / "figure_data" / "daily_spatial_r.csv", index=False)
    sr_rows = []
    ref = "SPL2SMAP_S_3km"
    meth = "SPL3SMP_E_9km_nearest"
    for split in ("test", "train"):
        d = daily_r[(daily_r["split"] == split) & daily_r["R"].notna()].reset_index(drop=True)
        if len(d) == 0:
            continue
        r = d["R"].to_numpy()
        stat = lambda idx: dict(median=np.median(r[idx]), q25=np.quantile(r[idx], 0.25), q75=np.quantile(r[idx], 0.75))
        boot = evals.block_bootstrap(stat, evals.date_blocks(d["date"], args.block_days), args.n_boot, args.seed)
        point = stat(np.arange(len(r)))
        for k, v in point.items():
            lo, hi = evals.ci(boot[k])
            sr_rows.append(dict(method=meth, reference=ref, metric=f"R_daily_{k}", value=v, ci_low=lo, ci_high=hi,
                                unit="-", subset=f"split={split}", n=len(r)))
        acc = tm[split]
        keep = acc["n"] >= args.min_days
        mx = acc["sx"][keep] / acc["n"][keep]
        my = acc["sy"][keep] / acc["n"][keep]
        sr_rows.append(dict(method=meth, reference=ref, metric="R_timemean", value=np.corrcoef(mx, my)[0, 1] if keep.sum() > 2 else np.nan,
                            ci_low=np.nan, ci_high=np.nan, unit="-", subset=f"split={split}", n=int(keep.sum())))
    metrics = pd.concat([metrics, pd.DataFrame(sr_rows)], ignore_index=True)
    metrics.to_csv(run_dir / "metrics.csv", index=False)

    # --- figures -----------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    by_year[["label_days_S1A_only", "label_days_with_S1B"]].plot.bar(stacked=True, ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("label-days")
    ax.set_title("SPL2SMAP_S label-days per year")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "label_days_per_year.png", dpi=150)
    by_year.to_csv(run_dir / "figure_data" / "label_days_per_year.csv")
    plt.close(fig)
    if len(daily_r):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for split, c in (("test", "#4C72B0"), ("train", "#DD8452")):
            d = daily_r[(daily_r["split"] == split)]["R"].dropna()
            if len(d):
                ax.hist(d, bins=40, range=(-0.5, 1), alpha=0.6, color=c, label=f"daily R ({split})")
                tmv = metrics[(metrics["metric"] == "R_timemean") & (metrics["subset"] == f"split={split}")]["value"]
                if len(tmv):
                    ax.axvline(tmv.iloc[0], color=c, ls="--", label=f"time-mean R ({split})")
        ax.set_xlabel("spatial R, SPL3SMP_E (nearest) vs SPL2SMAP_S 3 km")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / "daily_vs_timemean_r.png", dpi=150)
        plt.close(fig)

    # --- config + run summary ----------------------------------------------------------
    smapsr_repo = Path(__file__).resolve().parents[1]
    config = dict(experiment="e0_hygiene", script=str(Path(__file__).resolve().relative_to(smapsr_repo.parent)),
                  git=git_state(smapsr_repo), args=vars(args), data_dir=str(root), smap_dir=str(smap_root),
                  products={k: dict(dir=v[0], glob=v[1], role=v[2], n_files=len(files[k]),
                                    first=str(files[k].index.min().date()), last=str(files[k].index.max().date()),
                                    grid=grids.get(k)) for k, v in PRODUCTS.items()},
                  sites=sites.to_dict("records"), sites_without_data=sorted(set(sites["site"]) - set(ground)),
                  protocol=dict(
                      split="flipped: train 2021-2024 minus SlimSAR window, test 2016-2020 (AGENTS.md §8)",
                      insitu_matching=f"mean of in-situ samples within ±{args.window_min} min of {args.overpass_local_hour:g}:00 local solar time "
                                      "on the morning after the UTC date token (= ~18 UTC on that date); product value = mean of valid "
                                      f"cells centred within each buffer radius {args.buffer_km} km of the site (plus the nearest cell; "
                                      "0 km = nearest cell only); cell counts in buffer_cells.csv; one coordinate per site (data/sites.txt); "
                                      "all sensors at 10 cm (user)",
                      insitu_qc="duplicate timestamps averaged; no other filtering",
                      label_pass_filter=f"SPL2SMAP_S matchups and the spatial R use only label-days whose scenes are all AM "
                                        f"(scene SMAP time within ±3 h of nominal); {n_pm} label matchups dropped",
                      bootstrap=f"calendar blocks of {args.block_days} days resampled with replacement, {args.n_boot} replicates, percentile 95% CI",
                      paired_p="two-sided percentile bootstrap p for H0: difference = 0",
                      bias_sign="product - in-situ",
                      spatial_r=f"9-km input mapped to the raw 3-km grid by nearest coordinate; daily R needs >= {args.min_pixels} joint-valid pixels; "
                                f"time-mean fields use pixels with >= {args.min_days} joint-valid days"),
                  runtime_s=round(time.time() - t0, 1))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"wrote {run_dir} in {config['runtime_s']} s")


if __name__ == "__main__":
    main()
