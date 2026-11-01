"""Does the last fine-scale scene hold day-specific information the current prediction lacks, and does the coarse
SMAP series extend it across long gaps? Decides whether a real-time ODE has anything to work with. No training.

For each pixel-day of the block with an earlier label at that pixel (earlier within the block, because residuals of
training dates are in-sample):
  r_prev   the model's residual (label - prediction) at the pixel's previous scene
  gap      days since that scene
  dc       |bilinear coarse today - bilinear coarse at that scene| (net change since then)
  rise     largest rise of the bilinear coarse field since that scene (wetting that a net change can miss)

Reports, per base model: (1) correlation of today's residual with r_prev by exact gap and gap bin, by gap x |dc| and
gap x rise tercile (each residual minus its pixel's mean over separate other rows, see `pair_demean`);
(2) the base + beta * r_prev prediction, beta per gap bin (`_g`) or gap bin x |dc| tercile (`_gc`), fitted on the other
calendar blocks (not on training dates: models fit those, so their residuals are not representative); (3) model-free
persist (last label) and persist_dc (last label + coarse change); `_shuf` is a negative control, the `_g` procedure
with previous residuals shuffled in time within each pixel, which should gain nothing. The own-mean gate penalises
persist through its own short-series bias, so read persist on R and RMSE. Everything is scored on the same pixel-days, on the
gate metric as in E1. Bootstrap: calendar blocks over dates, with betas held fixed.

Defaults use the monitoring block only, which the models excluded from training.

Usage (project root):  PYTHONPATH=smapsr python smapsr/experiments/e9_persistence_check.py --v1 <name>:<run> ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_baselines as e1
from smapsr import evals
from smapsr.baselines import bilinear, parent_valid

GAP_EDGES = (1, 4, 7, 13, 25, 10_000)
GAP_NAMES = ("1-3", "4-6", "7-12", "13-24", "25+")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data_qc/cache/aligned_AM")
    p.add_argument("--out-dir", default="results/e9_audit/persistence")
    p.add_argument("--coarse", default="9km")
    p.add_argument("--fine", default="3km")
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--v1", nargs="+", help="<name>:<run dir>")
    p.add_argument("--train", nargs=2, default=["2021-01-01", "2024-06-30"], help="climatology for clim_null")
    p.add_argument("--test", nargs=2, default=["2024-07-01", "2024-12-31"])
    p.add_argument("--exclude-dates", nargs="*", default=["2022-04-26"])
    p.add_argument("--exclude-buffer-days", type=int, default=4)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--block-days", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    return p


def demean(values, pix):
    """values minus the mean over their own pixel; pix = dense pixel ids."""
    n = np.bincount(pix)
    return values - (np.bincount(pix, values) / n)[pix]


def pair_demean(r, rp, pix):
    """Row residual r and previous-scene residual rp, each minus its pixel's mean over a different set of other rows
    (rows alternate between the two sets). A mean containing the pair makes the two look anti-correlated, and one mean
    shared by both makes them look correlated; either would fake a gain. Noise in the means attenuates the result,
    so this errs towards no persistence."""
    order = np.argsort(pix, kind="stable")
    same = pix[order[1:]] == pix[order[:-1]]
    pos = np.zeros(r.size, dtype=np.int64)
    in_rows = np.zeros(r.size, dtype=bool)                              # rp is the pixel's previous row
    run = 0
    for i, sm in enumerate(same, start=1):
        run = run + 1 if sm else 0
        pos[order[i]] = run
    in_rows[order[1:]] = same & np.isclose(rp[order[1:]], r[order[:-1]], atol=1e-5)
    odd = pos % 2 == 1
    sums = np.stack([np.bincount(pix, np.where(odd, 0.0, r)), np.bincount(pix, np.where(odd, r, 0.0))])
    ns = np.stack([np.bincount(pix, ~odd), np.bincount(pix, odd)]).astype(np.float64)
    own, other = odd.astype(int), 1 - odd.astype(int)
    m_r = (sums[own, pix] - r) / np.maximum(ns[own, pix] - 1, 1)
    m_p = (sums[other, pix] - np.where(in_rows, rp, 0.0)) / np.maximum(ns[other, pix] - in_rows, 1)
    return r - m_r, rp - m_p


def gap_bin(gap):
    return np.searchsorted(GAP_EDGES, gap, side="right") - 1


def lag_corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and a.std() > 0 and b.std() > 0 else np.nan


def crossfit_beta(r, r_prev, cell, fold):
    """beta[i] = sum(r r_prev) / sum(r_prev^2) over rows of the same cell in other folds (0 when none)."""
    nc = cell.max() + 1
    num = np.zeros((fold.max() + 1, nc))
    den = np.zeros_like(num)
    np.add.at(num, (fold, cell), r * r_prev)
    np.add.at(den, (fold, cell), r_prev * r_prev)
    n, d = num.sum(0, keepdims=True) - num, den.sum(0, keepdims=True) - den
    beta = np.divide(n, d, out=np.zeros_like(n), where=d > 0)
    return beta[fold, cell]


def collect(a, models, clim):
    """Pixel-day rows with a previous scene at that pixel in the block; returns a dict of arrays + date list."""
    days = pd.date_range(*a.test)
    names = list(models) + ["clim_null", "bilinear"]
    state = None
    rows, used = [], []
    n_labels = 0
    for d in days:
        c = e1.load(a.cache, a.coarse, d)
        if c is None:
            continue
        bil = bilinear(c, a.factor)
        if state is not None:
            ok = np.isfinite(bil) & np.isfinite(state["bil"])
            state["rise"][ok] = np.maximum(state["rise"][ok], (bil - state["bil"])[ok])
        f = e1.load(a.cache, a.fine, d)
        if f is None:
            continue
        ev = np.isfinite(f) & parent_valid(c, a.factor) & clim.covered & np.isfinite(bil)
        if not ev.any() or (a.limit and len(used) >= a.limit):
            continue
        preds = {"clim_null": clim.null_model(c), "bilinear": bil}
        for name, fn in models.items():
            preds[name] = fn(c)
        if state is None:
            state = dict(date=np.full(f.shape, -1, dtype=np.int32), y=np.full(f.shape, np.nan, np.float32), bil=np.full(f.shape, np.nan, np.float32),
                         rise=np.zeros(f.shape, np.float32), r={m: np.full(f.shape, np.nan, np.float32) for m in names})
        idx = np.flatnonzero(ev & (state["date"] >= 0))
        k = (d - days[0]).days
        n_labels += 1
        if idx.size:
            row = dict(k=np.full(idx.size, len(used), dtype=np.int32), pix=idx, y=f.ravel()[idx], gap=k - state["date"].ravel()[idx],
                       dc=np.abs(bil - state["bil"]).ravel()[idx], rise=state["rise"].ravel()[idx], y_prev=state["y"].ravel()[idx], bil_prev=state["bil"].ravel()[idx], bil=bil.ravel()[idx])
            for m in names:
                row[m] = preds[m].ravel()[idx].astype(np.float32)
                row[f"rp_{m}"] = state["r"][m].ravel()[idx]
            rows.append(row)
            used.append(d)
        state["date"][ev] = k
        state["y"][ev] = f[ev]
        state["bil"][ev] = bil[ev]
        state["rise"][ev] = 0
        for m in names:
            state["r"][m][ev] = (f - preds[m])[ev]
        print(f"  {d.date()}: {idx.size} rows", flush=True) if len(used) % 25 == 0 and idx.size else None
    out = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    return out, used, names, n_labels


def main():
    a = build_parser().parse_args()
    t0 = time.time()
    specs = [s.split(":") for s in a.v1]
    loaded = e1.load_v1_models([r for _, r in specs], tile=0)
    models = {n: loaded[f"v1_{i}"] for i, (n, _) in enumerate(specs)}
    clim, n_clim = e1.fit_climatology(a, e1.train_dates(a))
    print(f"climatology from {n_clim} dates, {time.time() - t0:.0f} s", flush=True)
    t, used, names, n_labels = collect(a, models, clim)
    # the same pixel-days for every method; dense pixel ids; pixels with enough rows
    ok = np.isfinite(t["dc"]) & np.isfinite(t["y_prev"])
    for m in names:
        ok &= np.isfinite(t[m]) & np.isfinite(t[f"rp_{m}"])
    t = {k: v[ok] for k, v in t.items()}
    _, pix, cnt = np.unique(t["pix"], return_inverse=True, return_counts=True)
    keep = cnt[pix] >= a.min_count
    t = {k: v[keep] for k, v in t.items()}
    _, pix, cnt = np.unique(t["pix"], return_inverse=True, return_counts=True)
    print(f"{keep.sum()} pixel-days, {cnt.size} pixels, {len(used)} dates; {time.time() - t0:.0f} s", flush=True)

    y = t["y"].astype(np.float64)
    gb = gap_bin(t["gap"])
    q_dc, q_rise = np.quantile(t["dc"], [1 / 3, 2 / 3]), np.quantile(t["rise"], [1 / 3, 2 / 3])
    dc_t, rise_t = np.searchsorted(q_dc, t["dc"]), np.searchsorted(q_rise, t["rise"])
    dates = np.asarray(used, dtype="datetime64[D]")
    fold = evals.date_blocks(used, a.block_days)[t["k"]]
    # ---- (1) decay of residual correlation
    lag = []
    for m in names:
        r, rp = pair_demean(y - t[m], t[f"rp_{m}"].astype(np.float64), pix)
        for gap in np.unique(t["gap"]):
            s = t["gap"] == gap
            lag.append(dict(method=m, group="exact_gap", gap=int(gap), tercile="", n=int(s.sum()), r=lag_corr(r[s], rp[s])))
        for g, gname in enumerate(GAP_NAMES):
            s = gb == g
            lag.append(dict(method=m, group="gap_bin", gap=gname, tercile="", n=int(s.sum()), r=lag_corr(r[s], rp[s])))
            for lab, ter in (("dc", dc_t), ("rise", rise_t)):
                for q in range(3):
                    z = s & (ter == q)
                    lag.append(dict(method=m, group=f"gap_bin_x_{lab}", gap=gname, tercile=q, n=int(z.sum()), r=lag_corr(r[z], rp[z])))

    # ---- (2, 3) corrected predictions, all on the same rows
    pred = {m: t[m].astype(np.float64) for m in names}
    betas = []
    rng = np.random.default_rng(a.seed)
    for m in [n for n in names if n != "bilinear"]:
        r, rp = pair_demean(y - t[m], t[f"rp_{m}"].astype(np.float64), pix)
        shuffled = t[f"rp_{m}"].astype(np.float64).copy()               # negative control: previous residuals of the same pixel in random order
        perm = np.lexsort((rng.random(pix.size), pix))
        shuffled[np.argsort(pix, kind="stable")] = shuffled[perm]
        rs, rps = pair_demean(y - t[m], shuffled, pix)
        pred[m + "_shuf"] = t[m] + crossfit_beta(rs, rps, gb, fold) * shuffled
        for suffix, cell in (("_g", gb), ("_gc", gb * 3 + dc_t)):
            b = crossfit_beta(r, rp, cell, fold)
            pred[m + suffix] = t[m] + b * t[f"rp_{m}"]
            betas += [dict(method=m + suffix, cell=int(c), gap=GAP_NAMES[c if suffix == "_g" else c // 3], dc_tercile=-1 if suffix == "_g" else int(c % 3), mean_beta=float(b[cell == c].mean()), n=int((cell == c).sum())) for c in np.unique(cell)]
    pred["persist"] = t["y_prev"].astype(np.float64)
    pred["persist_dc"] = t["y_prev"].astype(np.float64) + t["bil"] - t["bil_prev"]
    methods = list(pred)
    D, M = len(used), len(methods)
    o_d = demean(y, pix)
    keys = ["all"] + list(GAP_NAMES)
    raw, gate = np.zeros((D, M, 6)), np.zeros((D, M, len(keys), 6))
    k = t["k"]

    def add(target, o, p, sel=None):
        kk, oo, pp = (k, o, p) if sel is None else (k[sel], o[sel], p[sel])
        for i, v in enumerate((np.ones_like(oo), oo, pp, oo * oo, pp * pp, oo * pp)):
            target[..., i] = np.bincount(kk, v, minlength=D)

    for m, name in enumerate(methods):
        add(raw[:, m], y, pred[name])
        p_d = demean(pred[name], pix)
        add(gate[:, m, 0], o_d, p_d)
        for g in range(len(GAP_NAMES)):
            add(gate[:, m, g + 1], o_d, p_d, gb == g)

    diffs = [(f"{n}{s}", n) for n in names if n != "bilinear" for s in ("_g", "_gc", "_shuf")] + [("persist_dc", "persist")]

    def stat(i):
        pr, out = e1.pooled(raw[i].sum(0)), {}
        for g, key in enumerate(keys):
            pg = e1.pooled(gate[i][:, :, g].sum(0))
            for m, name in enumerate(methods):
                out[f"{name}|{key}|gate"] = pg["R"][m]
        for m, name in enumerate(methods):
            out[f"{name}|all|R"], out[f"{name}|all|RMSE"] = pr["R"][m], pr["RMSE"][m]
        for x, ref in diffs:
            for key in keys:
                out[f"{x}-{ref}|{key}|gate"] = out[f"{x}|{key}|gate"] - out[f"{ref}|{key}|gate"]
            for met in ("R", "RMSE"):
                out[f"{x}-{ref}|all|{met}"] = out[f"{x}|all|{met}"] - out[f"{ref}|all|{met}"]
        return out

    point = stat(np.arange(D))
    boot = evals.block_bootstrap(stat, evals.date_blocks(used, a.block_days), n_boot=a.n_boot, seed=a.seed)
    rows = []
    for key, v in point.items():
        who, gap, metric = key.split("|")
        lo, hi = evals.ci(boot[key])
        rows.append(dict(method=who, gap=gap, metric=metric, value=v, ci_lo=lo, ci_hi=hi, p_paired=evals.paired_p(boot[key]) if "-" in who else np.nan))
    metrics = pd.DataFrame(rows)

    run = Path(a.out_dir) / f"run_{pd.Timestamp.now():%Y%m%dT%H%M%S}"
    run.mkdir(parents=True)
    metrics.to_csv(run / "metrics.csv", index=False)
    pd.DataFrame(lag).to_csv(run / "residual_lag_corr.csv", index=False)
    pd.DataFrame(betas).to_csv(run / "betas.csv", index=False)
    cfg = dict(args=vars(a), v1=specs, methods=methods, n_dates=D, n_label_days=n_labels, n_clim_dates=n_clim,
               pixel_days=int(y.size), pixels=int(cnt.size), median_rows_per_pixel=float(np.median(cnt)), gap_quantiles=dict(zip(("q10", "q50", "q90"), np.quantile(t["gap"], [0.1, 0.5, 0.9]).tolist())),
               gap_bin_counts={n: int((gb == g).sum()) for g, n in enumerate(GAP_NAMES)}, dc_tercile_edges=q_dc.tolist(), rise_tercile_edges=q_rise.tolist(),
               git=e1.git_sha("smapsr"), runtime_s=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    print(metrics[~metrics.method.str.contains("-") & (metrics.gap == "all")].pivot_table(index="method", columns="metric", values="value").round(4).to_string())
    print("wrote", run)


if __name__ == "__main__":
    main()
