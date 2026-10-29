"""E1 baselines, stage 1: training-free methods vs. SPL2SMAP_S labels on the held-out period (2016-2020).

Methods: bilinear (normalised over valid cells), block_constant (no downscaling), clim_null (bilinear + the
2021-2024 time-mean label-minus-bilinear anomaly = the "lookup table" null of strategy B1) and clim_only (the
2021-2024 label time-mean, no day information). Trained baselines (CNNs, discrete ResNet, EOF/CCA, ancillary RF,
external 1-km product) and the in-situ reference are later stages; V1 is added when it is trained for E1.

Protocol: aligned AM cache, labels unscreened, values clipped to [0.01, 0.65] as in training. Evaluation set per
date = label-valid pixels whose coarse parent cell is valid. Two subsets: "all" and "covered" (pixels where the
climatology has >= min_count training observations; clim_only exists only there). Metrics are pooled over pixel-days
(R, RMSE, uRMSE, bias), plus daily spatial R (median/IQR over dates with >= 100 pixels), the time-mean spatial R
and the day-specific anomaly R (anomalies taken relative to the training label climatology C, so RMSE is unchanged
and only the correlation isolates day-specific skill). CIs: moving calendar-block bootstrap over dates
(`--block-days`), paired differences against bilinear and against clim_null.

Usage (project root):  PYTHONPATH=smapsr python smapsr/experiments/e1_baselines.py
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from smapsr import evals
from smapsr.baselines import Climatology, bilinear, block_constant, parent_valid

CLIP = (0.01, 0.65)
BASE_METHODS = ["bilinear", "block_constant", "clim_null", "clim_only"]
SUBSETS = ["all", "covered"]
STAT = ["n", "so", "sp", "soo", "spp", "sop"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data_qc/cache/aligned_AM")
    p.add_argument("--out-dir", default="results/e1_baselines")
    p.add_argument("--coarse", default="9km")
    p.add_argument("--fine", default="3km")
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--train", nargs=2, default=["2021-01-01", "2024-12-31"])
    p.add_argument("--test", nargs=2, default=["2016-01-01", "2020-12-31"])
    p.add_argument("--exclude-dates", nargs="*", default=["2022-04-26"])
    p.add_argument("--exclude-buffer-days", type=int, default=4)
    p.add_argument("--min-count", type=int, default=5, help="min training observations per pixel for the climatology")
    p.add_argument("--min-pixels", type=int, default=100, help="min evaluation pixels for a daily spatial R")
    p.add_argument("--block-days", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--v1", nargs="*", default=[], help="train_v1 run directories (results/v1_train/run_*); each is evaluated as v1_<n>")
    p.add_argument("--v1-tile", type=int, default=32, help="V1 inference tile in coarse cells (= training crop; 0 = full grid, which GroupNorm penalises)")
    p.add_argument("--v1-halo", type=int, default=8, help="V1 inference halo in coarse cells around each tile")
    p.add_argument("--limit", type=int, default=0, help="debug: use only the first N dates of each period")
    return p


def parse_args():
    return build_parser().parse_args()


def load(cache, scale, date):
    path = Path(cache) / scale / f"{date:%Y%m%d}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return np.clip(z["soil_moisture"].astype(np.float32), *CLIP)


def pair_dates(cache, fine, start, end):
    ds = sorted(pd.to_datetime(p.stem, format="%Y%m%d") for p in (Path(cache) / fine).glob("*.npz"))
    return [d for d in ds if pd.Timestamp(start) <= d <= pd.Timestamp(end)]


def sums(o, p):
    return [o.size, o.sum(), p.sum(), (o * o).sum(), (p * p).sum(), (o * p).sum()]


def pooled(s):
    """Pooled metrics from stacked sufficient statistics s[..., 6] (summed over the leading axes by the caller)."""
    n, so, sp, soo, spp, sop = (s[..., i] for i in range(6))
    with np.errstate(invalid="ignore", divide="ignore"):
        bias = (sp - so) / n
        mse = (spp - 2 * sop + soo) / n
        cov = sop / n - so / n * sp / n
        r = cov / np.sqrt((soo / n - (so / n) ** 2) * (spp / n - (sp / n) ** 2))
    return dict(R=r, RMSE=np.sqrt(mse), uRMSE=np.sqrt(np.maximum(mse - bias**2, 0)), bias=bias)


def tiled(fwd, x, tile, halo, factor, s):
    """Predict (H*f, W*f) from x (C, h, w) and static fine-grid channels s (S, H*f, W*f) with tiles of `tile` coarse
    cells plus a `halo` on each side (zero input, validity and static channels outside the domain). Matches the
    training crop extent, which GroupNorm is sensitive to."""
    import jax.numpy as jnp
    _, h, w = x.shape
    out = np.zeros((h * factor, w * factor), dtype=np.float32)
    pad = np.pad(x, ((0, 0), (halo, halo), (halo, halo)))
    hf = halo * factor
    spad = np.pad(s, ((0, 0), (hf, hf), (hf, hf)))
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            ti, tj = min(tile, h - i), min(tile, w - j)
            st = spad[:, i * factor:(i + ti + 2 * halo) * factor, j * factor:(j + tj + 2 * halo) * factor]
            y = np.asarray(fwd(jnp.asarray(pad[:, i:i + ti + 2 * halo, j:j + tj + 2 * halo]), jnp.asarray(st)))[0]
            out[i * factor:(i + ti) * factor, j * factor:(j + tj) * factor] = y[halo * factor:(halo + ti) * factor, halo * factor:(halo + tj) * factor]
    return out


def load_v1_models(run_dirs, tile=32, halo=8):
    """{name: fn(coarse day in m3/m3, NaN gaps) -> fine field in m3/m3} for trained V1 runs (tiled inference unless tile=0).
    Runs trained with static channels use their own saved training climatology (models/static.npz), in full."""
    import jax
    import jax.numpy as jnp
    from smapsr.baselines import Climatology
    from smapsr.data import STATIC_CHANNELS
    out = {}
    for n, d in enumerate(run_dirs):
        cfg = json.loads((Path(d) / "config.json").read_text())
        model, mean, std = evals.load_v1(Path(d) / "models" / "model.eqx")
        scale, factor = float(cfg["args"]["scale_km"]), int(cfg["args"]["factor"])
        names = cfg["args"].get("static") or []
        fwd = jax.jit(lambda x, s, m=model: m(x, scale, s))
        s_full = None
        if names:
            f = Climatology.load(Path(d) / "models" / "static.npz").fields(mean, std)
            s_full = f[[STATIC_CHANNELS[k] for k in names] + [2]]

        def fn(c, fwd=fwd, mean=mean, std=std, s_full=s_full):
            v = np.isfinite(c)
            x = np.stack([np.where(v, (c - mean) / std, 0.0), v.astype(np.float32)]).astype(np.float32)
            s = s_full if s_full is not None else np.zeros((0, c.shape[0] * factor, c.shape[1] * factor), np.float32)
            if tile:
                return tiled(fwd, x, tile, halo, factor, s) * std + mean
            return np.asarray(fwd(jnp.asarray(x), jnp.asarray(s))[0]) * std + mean
        out[f"v1_{n}"] = fn
    return out


def fit_climatology(a, train):
    """Climatology (label mean, label-minus-bilinear anomaly) from the training dates; returns (Climatology, n_dates)."""
    clim, n = None, 0
    for d in train:
        c, f = load(a.cache, a.coarse, d), load(a.cache, a.fine, d)
        if c is None or f is None or not np.isfinite(f).any():
            continue
        if clim is None:
            clim = Climatology(f.shape, a.factor, a.min_count)
        clim.add(c, f)
        n += 1
    return clim, n


def train_dates(a):
    excl = [pd.Timestamp(d) + pd.Timedelta(days=k) for d in a.exclude_dates
            for k in range(-a.exclude_buffer_days, a.exclude_buffer_days + 1)]
    return [d for d in pair_dates(a.cache, a.fine, *a.train) if d not in excl]


def git_sha(repo):
    try:
        sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True).strip() != ""
        return sha + (" (dirty)" if dirty else "")
    except Exception:
        return "unknown"


def main():
    a = parse_args()
    t0 = time.time()
    v1 = load_v1_models(a.v1, a.v1_tile, a.v1_halo)
    methods = BASE_METHODS + list(v1)
    excl = [pd.Timestamp(d) + pd.Timedelta(days=k) for d in a.exclude_dates
            for k in range(-a.exclude_buffer_days, a.exclude_buffer_days + 1)]
    train = [d for d in pair_dates(a.cache, a.fine, *a.train) if d not in excl]
    test = pair_dates(a.cache, a.fine, *a.test)
    if a.limit:
        train, test = train[:a.limit], test[:a.limit]

    # ---- 1. climatology from the training period only
    clim, n_train = fit_climatology(a, train)
    C, covered = clim.label_mean, clim.covered
    print(f"climatology from {n_train} training dates; covered pixels {covered.sum()} of {covered.size}", flush=True)

    # ---- 2. evaluation: per-date sufficient statistics
    D, M, S = len(test), len(methods), len(SUBSETS)
    val = np.full((D, M, S, 6), np.nan)           # plain values
    anom = np.full((D, M, 6), np.nan)             # anomalies vs. C, covered subset only
    daily_r = np.full((D, M, S), np.nan)
    tm_o = np.zeros(covered.shape)                # time-mean accumulators (covered subset)
    tm_p = np.zeros((M,) + covered.shape)
    tm_n = np.zeros(covered.shape, dtype=np.int32)
    used = np.zeros(D, dtype=bool)

    def day(d):
        """(label, evaluation mask, {method: prediction}) for one test date, or None."""
        c, f = load(a.cache, a.coarse, d), load(a.cache, a.fine, d)
        if c is None or f is None:
            return None
        ev = np.isfinite(f) & parent_valid(c, a.factor)
        if not ev.any():
            return None
        preds = {"bilinear": bilinear(c, a.factor), "block_constant": block_constant(c, a.factor),
                 "clim_null": clim.null_model(c), "clim_only": C}
        for name, fn in v1.items():
            preds[name] = fn(c)
        return f, ev, preds

    for k, d in enumerate(test):
        got = day(d)
        if got is None:
            continue
        f, ev, preds = got
        used[k] = True
        for m, name in enumerate(methods):
            p = preds[name]
            for si, sub in enumerate(SUBSETS):
                sel = ev & (covered if sub == "covered" else True) & np.isfinite(p)
                if sel.any():
                    val[k, m, si] = sums(f[sel].astype(np.float64), p[sel].astype(np.float64))
                    daily_r[k, m, si] = evals.spatial_corr(np.where(sel, f, np.nan), np.where(sel, p, np.nan), a.min_pixels)
            sel = ev & covered & np.isfinite(p)
            if sel.any():
                anom[k, m] = sums((f - C)[sel].astype(np.float64), (p - C)[sel].astype(np.float64))
                tm_p[m][sel] += p[sel]
        sel = ev & covered
        tm_o[sel] += f[sel]
        tm_n[sel] += 1
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{D} test dates", flush=True)
    # second pass: anomalies relative to each series' own test-period pixel mean (separates day-specific skill
    # from pixel-level offsets; means are fixed from pass 1, not resampled in the bootstrap)
    ok_px = covered & (tm_n >= a.min_count)
    o_bar = np.where(ok_px, tm_o / np.maximum(tm_n, 1), np.nan)
    anom2 = np.full((D, M, 6), np.nan)
    for k, d in enumerate(test):
        if not used[k]:
            continue
        f, ev, preds = day(d)
        for m, name in enumerate(methods):
            p = preds[name]
            sel = ev & ok_px & np.isfinite(p)
            if sel.any():
                p_bar = tm_p[m] / np.maximum(tm_n, 1)
                anom2[k, m] = sums((f - o_bar)[sel].astype(np.float64), (p - p_bar)[sel].astype(np.float64))
    test_used = [d for d, u in zip(test, used) if u]
    val, anom, anom2, daily_r = val[used], anom[used], anom2[used], daily_r[used]
    print(f"{len(test_used)} test label-days with evaluation pixels; {time.time() - t0:.0f} s", flush=True)

    # ---- 3. point estimates + block-bootstrap CIs + paired differences
    blocks = evals.date_blocks(test_used, a.block_days)

    def stat(idx):
        out = {}
        for si, sub in enumerate(SUBSETS):
            pm = pooled(np.nansum(val[idx][:, :, si], axis=0))
            for m, name in enumerate(methods):
                for k in ("R", "RMSE", "uRMSE", "bias"):
                    out[f"{name}|{sub}|{k}"] = pm[k][m]
                out[f"{name}|{sub}|daily_R_median"] = np.nanmedian(daily_r[idx][:, m, si]) if np.isfinite(daily_r[idx][:, m, si]).any() else np.nan
        am = pooled(np.nansum(anom[idx], axis=0))
        am2 = pooled(np.nansum(anom2[idx], axis=0))
        for m, name in enumerate(methods):
            out[f"{name}|covered|anomaly_R"] = am["R"][m]
            out[f"{name}|covered|anomaly_R_pixmean"] = am2["R"][m]
        for ref in ("bilinear", "clim_null"):
            for name in methods:
                if name == ref:
                    continue
                for sub in SUBSETS:
                    for k in ("R", "RMSE", "daily_R_median"):
                        out[f"{name}-{ref}|{sub}|{k}"] = out[f"{name}|{sub}|{k}"] - out[f"{ref}|{sub}|{k}"]
                for k in ("anomaly_R", "anomaly_R_pixmean"):
                    out[f"{name}-{ref}|covered|{k}"] = out[f"{name}|covered|{k}"] - out[f"{ref}|covered|{k}"]
        return out

    point = stat(np.arange(len(test_used)))
    boot = evals.block_bootstrap(stat, blocks, n_boot=a.n_boot, seed=a.seed)

    rows = []
    for key, v in point.items():
        who, sub, metric = key.split("|")
        lo, hi = evals.ci(boot[key])
        paired = "-" in who
        unit = "m3/m3" if metric in ("RMSE", "uRMSE", "bias") else "-"
        rows.append(dict(method=who, reference="SPL2SMAP_S 3 km", subset=sub, metric=metric, value=v, ci_lo=lo, ci_hi=hi,
                         p_paired=evals.paired_p(boot[key]) if paired else np.nan, unit=unit))
    # daily-R IQR (point only) and time-mean spatial R (point only)
    tm_ok = tm_n >= a.min_count
    for m, name in enumerate(methods):
        d = daily_r[:, m, 1]
        if np.isfinite(d).any():
            q1, q3 = np.nanquantile(d, [0.25, 0.75])
            rows += [dict(method=name, reference="SPL2SMAP_S 3 km", subset="covered", metric="daily_R_q25", value=q1, unit="-"),
                     dict(method=name, reference="SPL2SMAP_S 3 km", subset="covered", metric="daily_R_q75", value=q3, unit="-")]
        if tm_ok.sum() > a.min_pixels:
            o = tm_o[tm_ok] / tm_n[tm_ok]
            p = tm_p[m][tm_ok] / tm_n[tm_ok]
            rows.append(dict(method=name, reference="SPL2SMAP_S 3 km", subset="covered", metric="time_mean_R",
                             value=evals.spatial_corr(o, p, a.min_pixels), unit="-"))
    metrics = pd.DataFrame(rows)

    # ---- 4. write the run
    run = Path(a.out_dir) / f"run_{pd.Timestamp.now():%Y%m%dT%H%M%S}"
    run.mkdir(parents=True)
    metrics.to_csv(run / "metrics.csv", index=False)
    pd.DataFrame(dict(date=test_used, **{f"n_{s}": val[:, 0, i, 0] for i, s in enumerate(SUBSETS)})).to_csv(run / "test_dates.csv", index=False)
    cfg = dict(args=vars(a), n_train_dates=n_train, n_test_label_days=len(test_used), covered_pixels=int(covered.sum()),
               total_pixels=int(covered.size), pooled_pixel_days={s: int(np.nansum(val[:, 0, i, 0])) for i, s in enumerate(SUBSETS)},
               time_mean_pixels=int(tm_ok.sum()), clip=CLIP, methods=methods, v1_runs=[str(x) for x in a.v1], v1_tile=a.v1_tile, v1_halo=a.v1_halo, git=git_sha("smapsr"), runtime_s=round(time.time() - t0, 1),
               train_dates=[str(d.date()) for d in train], note="trained baselines, in-situ reference, SSIM/PSNR and V1 are later stages")
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    show = metrics[metrics.metric.isin(["R", "RMSE", "uRMSE", "bias", "daily_R_median", "anomaly_R", "anomaly_R_pixmean", "time_mean_R"]) & ~metrics.method.str.contains("-")]
    print(show.pivot_table(index=["subset", "method"], columns="metric", values="value").round(4).to_string())
    print("wrote", run)


if __name__ == "__main__":
    main()
