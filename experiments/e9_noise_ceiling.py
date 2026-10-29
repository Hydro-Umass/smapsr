"""E9 information audit, part 1: label-noise ceiling for day-specific skill against SPL2SMAP_S (3 km, AM labels).

No model is involved. Question: how high can *any* prediction correlate with the labels on the E1 gate metric
(pixel-mean-removed anomaly R, `e1_baselines.py`), given that the labels are noisy? With label L = T + e, e white
and independent of everything predictable, R(pred, L) <= sqrt(1 - var(e_anom) / var(L_anom)) = the ceiling.

Three noise estimates, none needing truth:
  A  same-day scene overlaps: raw AM scenes of the same SMAP date that overlap. Pairs are classed by the Sentinel-1
     acquisition lag (same acquisition = adjacent frames; ~10 h; > 30 h). sigma_e^2 = mean squared difference / 2.
     All overlapping pairs share the SMAP radiometer pass, so the shared TB error is missed (tends to under-estimate
     the noise); overlaps sit at frame edges, which may be noisier than typical pixels (tends to over-estimate).
  B  variogram nugget of the daily anomaly fields on the E1 evaluation set (label- and parent-valid, climatology-
     covered, >= min_count test observations; anomaly = label - its own test-period pixel mean). The nugget is the
     linear extrapolation of the semivariogram at lags 1..3 cells to lag 0, per direction (N-S, E-W). It sees only
     the spatially white part of the noise (correlated errors, e.g. per S1 acquisition or per TB cell, are missed)
     and also holds real variability below the 3-km lag; with correlated errors present it under-estimates the noise,
     i.e. its ceiling reads as "if all label noise were white".
  Consistency check: the best R any method already achieves on the same set (E1) is a hard lower bound on the
  ceiling; a noise estimate implying a lower ceiling is not representative of the evaluation pixels.
  C  triple collocation at the stations, from an E1 in-situ run (`--matchups`): label (5-km buffer), SPL3SMP_E via
     bilinear, in-situ, on label-days; each series demeaned per site, then pooled. Label and 9-km product share the
     radiometer, so their error covariance inflates C_12 and the label's R^2 is over-estimated. This is a temporal
     R at the station scale, not the spatial gate metric; it is reported beside the ceiling, not combined with it.

Ceiling: R_max = sqrt(1 - N / V), V = pooled variance of the anomalies on the evaluation set, N = noise variance of
the anomalies = sigma_e^2 * mean(1 - 1/n_i) for A (the pixel mean carries noise too), = nugget for B. CIs: calendar-
block bootstrap over test dates, A and B resampled jointly; C has its own date-block bootstrap.

Usage (project root):  PYTHONPATH=smapsr python smapsr/experiments/e9_noise_ceiling.py \
    --matchups results/e1_baselines/run_20261028T220654_insitu/matchups.csv
"""
import itertools
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e1_baselines as e1                      # noqa: E402
from mosaic_daily import overpass              # noqa: E402
from smapsr import evals                       # noqa: E402
from smapsr.baselines import parent_valid      # noqa: E402

SCENE_RE = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_\d{3}E\d{2}S_")
LAG_CLASSES = [("same_acq", 0, 0.2), ("lag_10h", 0.2, 30), ("lag_gt30h", 30, np.inf)]
OVERLAP_SUBSETS = ["all", "interior", "good67", "interior_good67"]
LAGS = np.arange(1, 9)
FIT_LAGS = 3


def build_parser():
    p = e1.build_parser()
    p.set_defaults(out_dir="results/e9_audit/noise_ceiling")
    p.add_argument("--scenes", default="data_qc/smap3km", help="raw per-scene label files (part A)")
    p.add_argument("--matchups", default="", help="matchups.csv of an e1_insitu run (part C; skipped if empty)")
    p.add_argument("--tc-min-n", type=int, default=30, help="min triplets for a per-site TC estimate")
    return p


# ---------------------------------------------------------------- A: same-day scene overlaps
def scene_groups(directory):
    """{SMAP date: [(path, SMAP time, S1 time, satellite)]} for AM scenes."""
    g = defaultdict(list)
    for f in sorted(Path(directory).glob("*.nc")):
        if overpass(f.name) != "AM":
            continue
        m = SCENE_RE.search(f.name)
        g[pd.Timestamp(m.group(1)[:8])].append((f, pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2)), f.name[14:16]))
    return g


def overlap_rows(groups):
    """One row per (pair, subset) with the count and sums of the label difference; values clipped as in E1."""
    rows = []
    for date, v in groups.items():
        if len(v) < 2:
            continue
        sc = []
        for f, *_ in v:
            with xr.open_dataset(f) as ds:
                sc.append((np.clip(ds.soil_moisture.values.astype(np.float64), *e1.CLIP),
                           ((ds.surface_flag_3km.values >> 11) & 1).astype(bool),          # granule edge
                           (ds.retrieval_qual_flag.values & 0xC0) > 0))                    # bit 6|7
        for i, j in itertools.combinations(range(len(v)), 2):
            (a, ea, qa), (b, eb, qb) = sc[i], sc[j]
            ok = np.isfinite(a) & np.isfinite(b)
            if not ok.any():
                continue
            lag = abs((v[i][2] - v[j][2]).total_seconds()) / 3600
            interior, good = ~(ea | eb), ~(qa | qb)
            masks = dict(all=ok, interior=ok & interior, good67=ok & good, interior_good67=ok & interior & good)
            for sub, s in masks.items():
                if s.any():
                    d = (a - b)[s]
                    rows.append(dict(date=date, scene_i=v[i][0].name, scene_j=v[j][0].name, sats=v[i][3] + v[j][3],
                                     s1_lag_h=lag, lag_class=next(c for c, lo, hi in LAG_CLASSES if lo <= lag < hi),
                                     subset=sub, n=int(s.sum()), sum_d=d.sum(), sum_d2=(d * d).sum()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B: anomaly variance + variogram
def semivariogram_sums(a):
    """Per lag and direction: (count, sum of squared differences) of a field with NaN outside the evaluation set."""
    out = np.zeros((2, len(LAGS), 2))
    for k, h in enumerate(LAGS):
        for ax in (0, 1):
            x, y = (a[:-h], a[h:]) if ax == 0 else (a[:, :-h], a[:, h:])
            ok = np.isfinite(x) & np.isfinite(y)
            d = (x - y)[ok]
            out[ax, k] = ok.sum(), (d * d).sum()
    return out


def nugget(gamma):
    """Lag-0 intercept of a straight line through gamma(1..FIT_LAGS)."""
    return np.polyfit(LAGS[:FIT_LAGS], gamma[:FIT_LAGS], 1)[1]


def anomaly_budget(a, test, clim):
    """Per test date: [n, sum a^2, sum (1 - 1/n_i)], semivariogram sums, duplicate-neighbour counts (raw label, lag 1)."""
    covered = clim.covered
    fields, tm_sum, tm_n = {}, np.zeros(covered.shape), np.zeros(covered.shape, dtype=np.int32)
    for d in test:
        c, f = e1.load(a.cache, a.coarse, d), e1.load(a.cache, a.fine, d)
        if c is None or f is None:
            continue
        ev = np.isfinite(f) & parent_valid(c, a.factor) & covered
        if ev.any():
            fields[d] = np.where(ev, f, np.nan)
            tm_sum[ev] += f[ev]
            tm_n[ev] += 1
    ok_px = tm_n >= a.min_count
    o_bar = np.where(ok_px, tm_sum / np.maximum(tm_n, 1), np.nan)
    w = np.where(ok_px, 1 - 1 / np.maximum(tm_n, 1), np.nan)
    dates, budget, vg, dup = [], [], [], []
    for d, f in fields.items():
        an = np.where(ok_px, f - o_bar, np.nan)
        sel = np.isfinite(an)
        if not sel.any():
            continue
        dates.append(d)
        budget.append([sel.sum(), (an[sel] ** 2).sum(), w[sel].sum()])
        vg.append(semivariogram_sums(an))
        dup.append([(np.isfinite(f[:-1]) & np.isfinite(f[1:])).sum(), (f[:-1] == f[1:]).sum(),
                    (np.isfinite(f[:, :-1]) & np.isfinite(f[:, 1:])).sum(), (f[:, :-1] == f[:, 1:]).sum()])
    return dates, np.array(budget), np.array(vg), np.array(dup), int(ok_px.sum())


# ---------------------------------------------------------------- C: triple collocation
def triple_collocation(x, y, z):
    """Covariance-notation TC: R^2 of each series with the common signal, and error std in the series' own units."""
    C = np.cov(np.vstack([x, y, z]))
    r2 = dict(label=C[0, 1] * C[0, 2] / (C[0, 0] * C[1, 2]),
              smap9km=C[0, 1] * C[1, 2] / (C[1, 1] * C[0, 2]),
              insitu=C[0, 2] * C[1, 2] / (C[2, 2] * C[0, 1]))
    err = dict(label=C[0, 0] - C[0, 1] * C[0, 2] / C[1, 2],
               smap9km=C[1, 1] - C[0, 1] * C[1, 2] / C[0, 2],
               insitu=C[2, 2] - C[0, 2] * C[1, 2] / C[0, 1])
    return r2, {k: np.sqrt(v) if v > 0 else np.nan for k, v in err.items()}


def tc_rows(a):
    mu = pd.read_csv(a.matchups, parse_dates=["date"])
    mu = mu[np.isfinite(mu[["label", "bilinear", "insitu"]]).all(1)].reset_index(drop=True)
    cols = ["label", "bilinear", "insitu"]
    dm = mu.copy()
    dm[cols] = mu[cols] - mu.groupby("site")[cols].transform("mean")

    def stat(idx):
        d = dm.iloc[idx]
        r2, err = triple_collocation(*(d[c].to_numpy() for c in cols))
        out = {f"{k}|R_tc": np.sqrt(v) if v > 0 else np.nan for k, v in r2.items()}
        out.update({f"{k}|err_std_tc": v for k, v in err.items()})
        return out

    point = stat(np.arange(len(dm)))
    boot = evals.block_bootstrap(stat, evals.date_blocks(dm.date, a.block_days), n_boot=a.n_boot, seed=a.seed)
    rows = []
    for key, v in point.items():
        series, metric = key.split("|")
        lo, hi = evals.ci(boot[key])
        rows.append(dict(part="C", scope="pooled_site_demeaned", quantity=metric, series=series, value=v, ci_lo=lo, ci_hi=hi, N=len(dm)))
    for site, g in mu.groupby("site"):
        if len(g) < a.tc_min_n:
            continue
        r2, err = triple_collocation(*(g[c].to_numpy() for c in cols))
        for s in r2:
            rows.append(dict(part="C", scope=site, quantity="R_tc", series=s, value=np.sqrt(r2[s]) if r2[s] > 0 else np.nan, N=len(g)))
            rows.append(dict(part="C", scope=site, quantity="err_std_tc", series=s, value=err[s], N=len(g)))
    return rows, len(dm), mu.groupby("site").size().to_dict()


def main():
    a = build_parser().parse_args()
    t0 = time.time()
    train, test = e1.train_dates(a), e1.pair_dates(a.cache, a.fine, *a.test)
    if a.limit:
        train, test = train[:a.limit], test[:a.limit]
    clim, n_train = e1.fit_climatology(a, train)

    # ---- B
    dates, budget, vg, dup, n_px = anomaly_budget(a, test, clim)
    print(f"B: {len(dates)} test label-days, {int(budget[:, 0].sum())} pixel-days, {n_px} pixels; {time.time() - t0:.0f} s", flush=True)

    # ---- A
    groups = scene_groups(a.scenes)
    ov = overlap_rows(groups)
    ov["period"] = np.where(ov.date.between(*map(pd.Timestamp, a.test)), "test",
                            np.where(ov.date.between(*map(pd.Timestamp, a.train)), "train", "other"))
    print(f"A: {ov[ov.subset == 'all'].shape[0]} overlapping scene pairs on {ov.date.nunique()} dates; {time.time() - t0:.0f} s", flush=True)

    # per test date sums of the A pairs, aligned with the B dates (zeros where a date has no overlap)
    pos = {d: k for k, d in enumerate(dates)}
    keys = [(c, s) for c in [None] + [c for c, *_ in LAG_CLASSES] for s in OVERLAP_SUBSETS]
    A = np.zeros((len(dates), len(keys), 2))
    for (d, c, s), g in ov[ov.period == "test"].groupby(["date", "lag_class", "subset"]):
        if d in pos:
            for kk, (kc, ks) in enumerate(keys):
                if ks == s and kc in (None, c):
                    A[pos[d], kk] += g[["n", "sum_d2"]].sum().to_numpy()

    def stat(idx):
        n, s2, w = budget[idx].sum(0)
        V, k = s2 / n, w / n
        out = {"V": V, "k": k}
        g = vg[idx].sum(0)
        for ax, name in ((0, "NS"), (1, "EW")):
            gamma = g[ax, :, 1] / (2 * g[ax, :, 0])
            for h, gm in zip(LAGS, gamma):
                out[f"gamma_{name}_{h}"] = gm
            nug = nugget(gamma)
            out[f"nugget_{name}"] = nug
            out[f"ceiling_nugget_{name}"] = np.sqrt(1 - nug / V) if nug < V else 0.0
        for kk, (c, s) in enumerate(keys):
            m, ss = A[idx, kk].sum(0)
            sig2 = ss / (2 * m) if m else np.nan
            tag = f"{c or 'pooled'}_{s}"
            out[f"sigma_e_{tag}"] = np.sqrt(sig2)
            out[f"ceiling_overlap_{tag}"] = np.sqrt(max(1 - sig2 * k / V, 0.0)) if m else np.nan
        return out

    point = stat(np.arange(len(dates)))
    boot = evals.block_bootstrap(stat, evals.date_blocks(dates, a.block_days), n_boot=a.n_boot, seed=a.seed)
    rows = []
    for key, v in point.items():
        lo, hi = evals.ci(boot[key])
        unit = "(m3/m3)^2" if key == "V" or key.startswith(("gamma", "nugget")) else ("m3/m3" if key.startswith("sigma") else "-")
        part = "A" if "overlap" in key or key.startswith("sigma") else "B"
        rows.append(dict(part=part, scope="test", quantity=key, value=v, ci_lo=lo, ci_hi=hi, unit=unit))
    # A over every period (point estimates; train/other dates are context only, not in the ceiling)
    for (per, c, s), g in ov.groupby(["period", "lag_class", "subset"]):
        rows.append(dict(part="A", scope=per, quantity=f"sigma_e_{c}_{s}", value=np.sqrt(g.sum_d2.sum() / (2 * g.n.sum())),
                         unit="m3/m3", N=int(g.n.sum()), n_pairs=len(g)))
    dup_s = dup.sum(0)
    rows += [dict(part="B", scope="test", quantity="dup_frac_NS_lag1", value=dup_s[1] / dup_s[0], unit="-"),
             dict(part="B", scope="test", quantity="dup_frac_EW_lag1", value=dup_s[3] / dup_s[2], unit="-")]
    n_tc, tc_sites = 0, {}
    if a.matchups:
        tc, n_tc, tc_sites = tc_rows(a)
        rows += tc
    metrics = pd.DataFrame(rows)

    run = Path(a.out_dir) / f"run_{pd.Timestamp.now():%Y%m%dT%H%M%S}"
    (run / "figure_data").mkdir(parents=True)
    metrics.to_csv(run / "metrics.csv", index=False)
    ov.to_csv(run / "overlap_pairs.csv", index=False)
    g = vg.sum(0)
    pd.DataFrame(dict(lag_cells=np.tile(LAGS, 2), direction=np.repeat(["NS", "EW"], len(LAGS)),
                      lag_km=np.concatenate([LAGS * 3.34, LAGS * 0.03 * 111.32 * np.cos(np.radians(40))]),
                      n_pairs=g[:, :, 0].ravel(), gamma=(g[:, :, 1] / (2 * g[:, :, 0])).ravel(), V=point["V"])
                 ).to_csv(run / "figure_data" / "variogram.csv", index=False)
    cfg = dict(args=vars(a), n_train_dates=n_train, n_test_label_days=len(dates), n_eval_pixels=n_px,
               pooled_pixel_days=int(budget[:, 0].sum()), lags=LAGS.tolist(), nugget_fit_lags=FIT_LAGS,
               lag_classes_h=[(c, lo, hi if np.isfinite(hi) else None) for c, lo, hi in LAG_CLASSES],
               overlap_subsets=dict(all="both valid", interior="neither pixel has surface_flag_3km bit 11 (granule edge)",
                                    good67="neither pixel has retrieval_qual_flag bit 6 or 7", interior_good67="both"),
               n_scene_pairs={p: int(g_.shape[0]) for p, g_ in ov[ov.subset == "all"].groupby("period")},
               tc_matchups=a.matchups, tc_n=n_tc, tc_per_site_n=tc_sites, clip=e1.CLIP,
               evaluation_set="E1 anomaly_R_pixmean set: label- and parent-valid, climatology-covered, >= min_count test obs",
               git=e1.git_sha("smapsr"), runtime_s=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    show = metrics[metrics.quantity.str.match(r"(V|k|nugget|ceiling|sigma_e_pooled|R_tc|err_std)") & metrics.scope.isin(["test", "pooled_site_demeaned"])]
    print(show[["part", "quantity", "series", "value", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    print("wrote", run)


if __name__ == "__main__":
    main()
