"""E3 pre-check: how far do the existing 9->3 km models go at 1 km, on labels they never saw?

Sets the bar for a scale-flow NODE before any is built. No training. Methods at 1 km:
  <name>_twice  a no-clim 9->3 EDSR-lite applied again to its own 3-km output (assumes self-similarity; the only 3->1
                information is what the 9->3 model learned)
  <name>_bilin  a 9->3 model's output, bilinear to 1 km (interpolation bar; any 9->3 model, including +clim)
  bilinear_9to1 bilinear from the 9-km input
  clim_null_1km bilinear + time-mean 1-km label anomaly. Reference only: it uses 1-km training labels, the others don't.

Evaluation set per date: 1-km label-valid pixels with a valid 9-km parent, values clipped as in training. Metrics:
pooled R/RMSE/bias; gate = R after removing each series' own period pixel mean; detail_R = R of sub-3-km detail (value
minus the mean over the same valid pixels of its 3x3 block, blocks with >= 2 pixels), detail_gate the same with each
pixel's mean detail removed, detail_var_ratio = prediction/label detail variance. Texture: semivariogram on the same
pixel pairs, radial power spectrum on fully valid windows. CIs: calendar-block bootstrap over dates. 3-km gate of
every 9->3 model is reported too, against the 3-km label, as a pipeline check (bilinear 0.547 on the monitoring block).

Defaults use the monitoring block only, which all these models excluded from training. Do not point --test at
2016-2020 before the configuration is fixed.

Usage (project root):  PYTHONPATH=smapsr python smapsr/experiments/e3_recursive_check.py --twice <runs> --bilin <runs>
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
from smapsr.baselines import Climatology, bilinear, parent_valid

LAGS = (1, 2, 3, 4, 6, 9, 12, 18, 27, 54, 81)
METRICS = ("R", "RMSE", "bias", "gate", "detail_R", "detail_gate", "detail_var_ratio")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data_qc/cache/aligned_AM")
    p.add_argument("--out-dir", default="results/e3_scale/recursive_check")
    p.add_argument("--twice", nargs="*", default=[], help="no-clim 9->3 runs: <name>:<run dir>, scored both _twice and _bilin")
    p.add_argument("--bilin", nargs="*", default=[], help="9->3 runs scored _bilin only: <name>:<run dir>")
    p.add_argument("--train", nargs=2, default=["2021-01-01", "2024-06-30"], help="dates of the 1-km reference climatology")
    p.add_argument("--test", nargs=2, default=["2024-07-01", "2024-12-31"])
    p.add_argument("--exclude-dates", nargs="*", default=["2022-04-26"])
    p.add_argument("--exclude-buffer-days", type=int, default=4)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--window", type=int, default=64, help="spectrum window in 1-km cells")
    p.add_argument("--block-days", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    return p


def second_stage_input(y3, valid, mean, std):
    """(data, validity) channels for feeding a 3-km prediction back in as a coarse field; gaps as in training."""
    return np.stack([np.where(valid, (y3 - mean) / std, 0.0), valid.astype(np.float32)]).astype(np.float32)


def block_detail(idx, values, shape, block=3):
    """Value minus the mean over the given pixels of its block, on flat indices idx; NaN in blocks with one pixel."""
    r, c = np.unravel_index(idx, shape)
    b = (r // block) * (shape[1] // block + 1) + c // block
    _, inv, n = np.unique(b, return_inverse=True, return_counts=True)
    m = np.bincount(inv, values) / n
    return np.where(n[inv] >= 2, values - m[inv], np.nan)


def lag_sums(o, p, ev, lags=LAGS):
    """[pairs, sum (o_x - o_x+L)^2, sum (p_x - p_x+L)^2] per lag, N-S and E-W, on pairs valid at both ends."""
    out = np.zeros((len(lags), 3))
    ok = ev & np.isfinite(p)
    for k, L in enumerate(lags):
        for ax in (0, 1):
            a = [slice(None), slice(None)]
            b = [slice(None), slice(None)]
            a[ax], b[ax] = slice(None, -L), slice(L, None)
            a, b = tuple(a), tuple(b)
            m = ok[a] & ok[b]
            out[k] += m.sum(), ((o[a] - o[b])[m] ** 2).sum(), ((p[a] - p[b])[m] ** 2).sum()
    return out


def valid_windows(ev, w):
    """Origins of non-overlapping w x w windows that are fully inside the mask."""
    i, j, n = _sums(ev, w)
    a, b = np.nonzero(n == w * w)
    return list(zip(i[a].tolist(), j[b].tolist()))


def _sums(a, w):
    from smapsr.data import _window_sums
    return _window_sums(a, w, w)


def radial_psd(x):
    """Radially averaged power of a square window (mean removed, Hann taper) over integer wavenumber bins 1..w//2."""
    w = x.shape[0]
    h = np.hanning(w)
    P = np.abs(np.fft.fft2((x - x.mean()) * np.outer(h, h))) ** 2
    f = np.fft.fftfreq(w) * w
    k = np.rint(np.hypot(*np.meshgrid(f, f, indexing="ij"))).astype(int).ravel()
    s = np.bincount(k, P.ravel(), minlength=w)
    n = np.bincount(k, minlength=w)
    return s[1:w // 2 + 1] / np.maximum(n[1:w // 2 + 1], 1)


def loader(spec):
    """(name, 3-km prediction fn on the 9-km field, model forward on normalised x, mean, std)"""
    import jax
    name, run = spec.split(":")
    fn = e1.load_v1_models([run], tile=0)["v1_0"]
    model, mean, std = evals.load_v1(Path(run) / "models" / "model.eqx")
    return name, fn, jax.jit(lambda x: model(x, 3.0)), mean, std


def own_mean_anomaly(series, size, min_count):
    """[(idx, o, p)] -> per-date sufficient statistics of o and p, each with its own period pixel mean removed."""
    so, sp, n = np.zeros(size), np.zeros(size), np.zeros(size, dtype=np.int32)
    for idx, o, p in series:
        ok = np.isfinite(o) & np.isfinite(p)
        np.add.at(so, idx[ok], o[ok])
        np.add.at(sp, idx[ok], p[ok])
        np.add.at(n, idx[ok], 1)
    good = n >= min_count
    so, sp = so / np.maximum(n, 1), sp / np.maximum(n, 1)
    out = []
    for idx, o, p in series:
        ok = np.isfinite(o) & np.isfinite(p) & good[idx]
        out.append(e1.sums((o[ok] - so[idx[ok]]).astype(np.float64), (p[ok] - sp[idx[ok]]).astype(np.float64)) if ok.any() else [np.nan] * 6)
    return np.asarray(out, dtype=float)


def main():
    a = build_parser().parse_args()
    t0 = time.time()
    twice = [loader(s) for s in a.twice]
    only_bilin = [loader(s) for s in a.bilin]
    methods = ["bilinear_9to1", "clim_null_1km"] + [f"{m[0]}_bilin" for m in twice + only_bilin] + [f"{m[0]}_twice" for m in twice]
    excl = [pd.Timestamp(d) + pd.Timedelta(days=k) for d in a.exclude_dates for k in range(-a.exclude_buffer_days, a.exclude_buffer_days + 1)]

    clim = None
    n_clim = 0
    for d in e1.pair_dates(a.cache, "1km", *a.train):
        if d in excl:
            continue
        c, f = e1.load(a.cache, "9km", d), e1.load(a.cache, "1km", d)
        if c is None or f is None or not np.isfinite(f).any():
            continue
        clim = clim or Climatology(f.shape, 9, a.min_count)
        clim.add(c, f)
        n_clim += 1
    print(f"1-km reference climatology from {n_clim} dates; {time.time() - t0:.0f} s", flush=True)

    dates, per_date, gate3 = [], [], []
    vg = {m: np.zeros((len(LAGS), 3)) for m in methods}
    psd_o, psd_p, n_win = None, {m: 0.0 for m in methods}, 0
    test = e1.pair_dates(a.cache, "1km", *a.test)
    test = test[:a.limit] if a.limit else test
    shape = None
    for d in test:
        c, f, f3 = e1.load(a.cache, "9km", d), e1.load(a.cache, "1km", d), e1.load(a.cache, "3km", d)
        if c is None or f is None:
            continue
        ev = np.isfinite(f) & parent_valid(c, 9)
        if not ev.any():
            continue
        shape = f.shape
        pv3 = parent_valid(c, 3)
        preds = {"bilinear_9to1": bilinear(c, 9), "clim_null_1km": clim.null_model(c)}
        day3 = {"bilinear": bilinear(c, 3)}
        for name, fn, fwd, mean, std in twice + only_bilin:
            y3 = fn(c)
            day3[name] = y3
            preds[f"{name}_bilin"] = bilinear(np.where(pv3, y3, np.nan), 3)
        for name, fn, fwd, mean, std in twice:
            import jax.numpy as jnp
            x2 = second_stage_input(day3[name], pv3, mean, std)
            preds[f"{name}_twice"] = np.asarray(fwd(jnp.asarray(x2))[0]) * std + mean
        idx = np.flatnonzero(ev)
        o = f.ravel()[idx]
        per_date.append((idx, o, {m: preds[m].ravel()[idx] for m in methods}))
        ev3 = np.isfinite(f3) & pv3
        i3 = np.flatnonzero(ev3)
        gate3.append((i3, f3.ravel()[i3], {m: v.ravel()[i3] for m, v in day3.items()}))
        for m in methods:
            vg[m] += lag_sums(f, preds[m], ev)
        win = [(i, j) for i, j in valid_windows(ev, a.window)]
        for i, j in win:
            sl = (slice(i, i + a.window), slice(j, j + a.window))
            if not all(np.isfinite(preds[m][sl]).all() for m in methods):
                continue
            psd_o = (0.0 if psd_o is None else psd_o) + radial_psd(f[sl])
            for m in methods:
                psd_p[m] = psd_p[m] + radial_psd(preds[m][sl])
            n_win += 1
        dates.append(d)
        if len(dates) % 25 == 0:
            print(f"  {len(dates)}/{len(test)} dates, {time.time() - t0:.0f} s", flush=True)
    print(f"{len(dates)} dates, {n_win} spectrum windows; {time.time() - t0:.0f} s", flush=True)

    # per-date sufficient statistics, method by method
    size = shape[0] * shape[1]
    D, M = len(dates), len(methods)
    raw, gate, det, detg = (np.full((D, M, 6), np.nan) for _ in range(4))
    for m, name in enumerate(methods):
        ser, sdet = [], []
        for k, (idx, o, pr) in enumerate(per_date):
            p = pr[name]
            ok = np.isfinite(p)
            raw[k, m] = e1.sums(o[ok].astype(np.float64), p[ok].astype(np.float64))
            ser.append((idx, o, p))
            do, dp = block_detail(idx[ok], o[ok], shape), block_detail(idx[ok], p[ok], shape)
            good = np.isfinite(do)
            det[k, m] = e1.sums(do[good].astype(np.float64), dp[good].astype(np.float64)) if good.any() else [np.nan] * 6
            sdet.append((idx[ok][good], do[good], dp[good]))
        gate[:, m] = own_mean_anomaly(ser, size, a.min_count)
        detg[:, m] = own_mean_anomaly(sdet, size, a.min_count)

    def stat(i):
        pr, pg, pd_, pdg = (e1.pooled(np.nansum(x[i], axis=0)) for x in (raw, gate, det, detg))
        n, so, sp, soo, spp, _ = (np.nansum(det[i], axis=0)[..., q] for q in range(6))
        ratio = (spp / n - (sp / n) ** 2) / (soo / n - (so / n) ** 2)
        out = {}
        for m, name in enumerate(methods):
            for k, v in dict(R=pr["R"], RMSE=pr["RMSE"], bias=pr["bias"], gate=pg["R"], detail_R=pd_["R"], detail_gate=pdg["R"], detail_var_ratio=ratio).items():
                out[f"{name}|{k}"] = v[m]
        for name, *_ in twice:
            for k in METRICS:
                out[f"{name}_twice-{name}_bilin|{k}"] = out[f"{name}_twice|{k}"] - out[f"{name}_bilin|{k}"]
        return out

    point = stat(np.arange(D))
    boot = evals.block_bootstrap(stat, evals.date_blocks(dates, a.block_days), n_boot=a.n_boot, seed=a.seed)
    rows = []
    for key, v in point.items():
        who, metric = key.split("|")
        lo, hi = evals.ci(boot[key])
        rows.append(dict(method=who, reference="SPL2SMAP_S 1 km", metric=metric, value=v, ci_lo=lo, ci_hi=hi,
                         p_paired=evals.paired_p(boot[key]) if "-" in who else np.nan))
    names3 = list(gate3[0][2])
    for name in names3:
        g = own_mean_anomaly([(i, o, pr[name]) for i, o, pr in gate3], gate3[0][1].size and 501 * 498, a.min_count)
        rows.append(dict(method=name, reference="SPL2SMAP_S 3 km", metric="gate", value=float(e1.pooled(np.nansum(g, axis=0))["R"])))
    metrics = pd.DataFrame(rows)

    run = Path(a.out_dir) / f"run_{pd.Timestamp.now():%Y%m%dT%H%M%S}"
    (run / "figure_data").mkdir(parents=True)
    metrics.to_csv(run / "metrics.csv", index=False)
    vrows = []
    for m in methods:
        for k, L in enumerate(LAGS):
            n, so2, sp2 = vg[m][k]
            vrows.append(dict(method=m, lag_km=L, pairs=int(n), gamma_label=so2 / (2 * n) if n else np.nan, gamma_pred=sp2 / (2 * n) if n else np.nan))
    pd.DataFrame(vrows).to_csv(run / "figure_data" / "variogram.csv", index=False)
    if n_win:
        k = np.arange(1, a.window // 2 + 1)
        pd.DataFrame([dict(method=m, wavenumber=int(q), wavelength_km=a.window / q, power_label=po / n_win, power_pred=pp / n_win)
                      for m in methods for q, po, pp in zip(k, psd_o, psd_p[m])]).to_csv(run / "figure_data" / "spectrum.csv", index=False)
    cfg = dict(args=vars(a), methods=methods, n_dates=len(dates), n_clim_dates=n_clim, n_spectrum_windows=n_win,
               pooled_pixel_days=int(np.nansum(raw[:, 0, 0])), git=e1.git_sha("smapsr"), runtime_s=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    show = metrics[~metrics.method.str.contains("-") & (metrics.reference == "SPL2SMAP_S 1 km")]
    print(show.pivot_table(index="method", columns="metric", values="value")[list(METRICS)].round(4).to_string())
    print(metrics[metrics.reference == "SPL2SMAP_S 3 km"][["method", "value"]].round(4).to_string())
    print("wrote", run)


if __name__ == "__main__":
    main()
