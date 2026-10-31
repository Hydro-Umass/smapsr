"""E9 information audit, part 2: gate metric and label-noise ceiling per label-quality stratum (3 km, AM labels).

Question: does day-specific skill, and the room left for it, depend on how clean the labels are? If both rise on clean
pixels, label noise is what limits New Zealand and cleaner regions should pay off (multi-region pilot). If they do not,
it is not label quality.

A stratum is a set of label pixel-days defined by the SPL2SMAP_S flags (`strata_masks`). A composite pixel-day carries
the OR of the flags of the AM scenes with a valid value there (`composite_flags`), so "clean" means clean in every
contributing scene. Each stratum is treated as its own evaluation set, with the E1 gate metric and the
`e9_noise_ceiling` ceilings recomputed inside it:
  gate    pooled R of anomalies, each series minus its own pixel mean *within the stratum* (>= min_count stratum
          observations), so a static per-pixel offset still cannot move it (bilinear = clim_null in every stratum);
  V       pooled label-anomaly variance within the stratum;
  B       variogram nugget of the stratum anomaly fields (neighbour pairs with both pixels in the stratum);
  A       same-day scene overlaps where both scenes satisfy the stratum, all lags pooled. sigma_e^2 = mean d^2 / 2;
          ceiling = sqrt(1 - sigma_e^2 k / V), k = mean(1 - 1/n_i) within the stratum.
Strata are different pixels as well as cleaner ones (dense vegetation and terrain flags are mostly static), so a
difference between strata is an association with geography plus label quality, not a clean effect of noise.

Usage (project root):  PYTHONPATH=smapsr python smapsr/experiments/e9_quality_strata.py --v1 <run dirs>
"""
import itertools
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
from smapsr.baselines import parent_valid                             # noqa: E402

STRATA = ["all", "recommended", "not_recommended", "good67", "bad67", "not_coastal", "coastal", "not_dense_veg", "dense_veg",
          "not_mountain", "mountain", "not_precip", "precip", "interior", "edge", "clean_surface", "clean_surface_good67"]
STRATA_DEF = dict(all="every evaluation pixel-day", recommended="retrieval_qual_flag bit 0 clear (overall quality good)",
                  good67="retrieval_qual_flag bits 6 and 7 clear", coastal="surface flag bit 2", dense_veg="surface flag bit 10",
                  mountain="surface flag bit 9", precip="surface flag bit 4", edge="surface flag bit 11 (granule edge)",
                  clean_surface="not coastal, not dense_veg, not mountain", clean_surface_good67="clean_surface and good67 and interior")


def build_parser():
    p = e1.build_parser()
    p.set_defaults(out_dir="results/e9_audit/quality_strata", v1_tile=0)
    p.add_argument("--scenes", default="data_qc/smap3km", help="raw per-scene label files (flags)")
    return p


def strata_masks(surf, ret):
    """{stratum: bool array} from integer surface and retrieval flags (any shape)."""
    bit = lambda x, b: ((x >> b) & 1).astype(bool)     # noqa: E731
    coastal, veg, mountain, precip, edge = (bit(surf, b) for b in (2, 10, 9, 4, 11))
    good67, rec = (ret & 0xC0) == 0, (ret & 1) == 0
    clean = ~(coastal | veg | mountain)
    return dict(all=np.ones(surf.shape, bool), recommended=rec, not_recommended=~rec, good67=good67, bad67=~good67,
                not_coastal=~coastal, coastal=coastal, not_dense_veg=~veg, dense_veg=veg, not_mountain=~mountain, mountain=mountain,
                not_precip=~precip, precip=precip, interior=~edge, edge=edge, clean_surface=clean,
                clean_surface_good67=clean & good67 & ~edge)


def composite_flags(scenes):
    """(valid, surface OR, retrieval OR) of [(soil moisture, surface flag, retrieval flag)]: flags count only where that
    scene has a valid value, as only those scenes enter the nanmean composite."""
    valid = np.zeros(scenes[0][0].shape, bool)
    surf, ret = np.zeros(valid.shape, np.uint16), np.zeros(valid.shape, np.uint16)
    for sm, s, r in scenes:
        v = np.isfinite(sm)
        valid |= v
        surf[v] |= s[v]
        ret[v] |= r[v]
    return valid, surf, ret


def overlap_sums(scenes):
    """{stratum: [n, sum d^2]} over all overlapping scene pairs of one date, both scenes' flags ORed."""
    out = {s: np.zeros(2) for s in STRATA}
    for (a, sa, ra), (b, sb, rb) in itertools.combinations(scenes, 2):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.any():
            d2 = (a[ok].astype(np.float64) - b[ok]) ** 2
            for s, m in strata_masks(sa[ok] | sb[ok], ra[ok] | rb[ok]).items():
                out[s] += m.sum(), d2[m].sum()
    return out


def stratum_tables(days, shape, names, min_count):
    """Per stratum and date, from days = [(eval pixel index, label, {method: prediction}, surf, ret)]: S (date, method, 6)
    anomaly sufficient statistics, budget (date, 3) [n, sum a^2, sum (1 - 1/n_i)], variogram sums; pixel and pixel-day counts."""
    size = int(np.prod(shape))
    masks = [strata_masks(surf, ret) for _, _, _, surf, ret in days]
    for s in STRATA:
        n_i = np.zeros(size, np.int64)
        sum_o, sum_p = np.zeros(size), np.zeros((len(names), size))
        for (idx, f, preds, _, _), mk in zip(days, masks):
            sel = idx[mk[s]]
            n_i[sel] += 1
            sum_o[sel] += f[mk[s]]
            for m, name in enumerate(names):
                sum_p[m, sel] += preds[name][mk[s]]
        ok = n_i >= min_count
        safe = np.maximum(n_i, 1)
        o_bar, p_bar, w = sum_o / safe, sum_p / safe, 1 - 1 / safe
        S, budget, vg = np.zeros((len(days), len(names), 6)), np.zeros((len(days), 3)), np.zeros((len(days), 2, len(e9.LAGS), 2))
        for k, ((idx, f, preds, _, _), mk) in enumerate(zip(days, masks)):
            keep = mk[s] & ok[idx]
            if not keep.any():
                continue
            i = idx[keep]
            an = f[keep].astype(np.float64) - o_bar[i]
            budget[k] = keep.sum(), (an * an).sum(), w[i].sum()
            for m, name in enumerate(names):
                S[k, m] = e1.sums(an, preds[name][keep].astype(np.float64) - p_bar[m, i])
            field = np.full(size, np.nan)
            field[i] = an
            vg[k] = e9.semivariogram_sums(field.reshape(shape))
        yield s, S, budget, vg, int(ok.sum())


def main():
    a = build_parser().parse_args()
    t0 = time.time()
    v1 = e1.load_v1_models(a.v1, a.v1_tile, a.v1_halo)
    names = ["bilinear", "clim_null"] + list(v1)
    train, test = e1.train_dates(a), e1.pair_dates(a.cache, a.fine, *a.test)
    if a.limit:
        train, test = train[:a.limit], test[:a.limit]
    clim, n_train = e1.fit_climatology(a, train)
    covered = clim.covered
    groups = e9.scene_groups(a.scenes)

    days, dates, A, mismatch, n_fill = [], [], [], 0, 0
    for d in test:
        c, f = e1.load(a.cache, a.coarse, d), e1.load(a.cache, a.fine, d)
        if c is None or f is None:
            continue
        ev = np.isfinite(f) & parent_valid(c, a.factor) & covered
        if not ev.any() or d not in groups:
            continue
        scenes = []
        for path, *_ in groups[d]:
            with xr.open_dataset(path) as ds:
                scenes.append((np.clip(ds.soil_moisture.values.astype(np.float32), *e1.CLIP),
                               ds.surface_flag_3km.values.astype(np.uint16), ds.retrieval_qual_flag.values.astype(np.uint16)))
        valid, surf, ret = composite_flags(scenes)
        n_fill += int(((surf == 65535) | (ret == 65535))[valid].sum())
        valid = place(valid.astype(np.float32), f.shape, 1) > 0      # raw 3-km grid -> aligned grid (offset 1, see build_aligned_cache)
        mismatch += int((valid != np.isfinite(f)).sum())
        surf, ret = (np.nan_to_num(place(x.astype(np.float64), f.shape, 1)).astype(np.uint16) for x in (surf, ret))
        idx = np.flatnonzero(ev)
        preds = dict(bilinear=e1.bilinear(c, a.factor), clim_null=clim.null_model(c), **{k: fn(c) for k, fn in v1.items()})
        days.append((idx, f.ravel()[idx], {k: p.ravel()[idx] for k, p in preds.items()}, surf.ravel()[idx], ret.ravel()[idx]))
        dates.append(d)
        ov = overlap_sums(scenes)
        A.append([ov[s] for s in STRATA])
        if len(dates) % 100 == 0:
            print(f"  {len(dates)} dates; {time.time() - t0:.0f} s", flush=True)
    A = np.array(A)                                                   # (date, stratum, 2)
    print(f"{len(dates)} test label-days; validity mismatch composite vs cache {mismatch} pixels; flag fill on valid {n_fill}; {time.time() - t0:.0f} s", flush=True)

    tables = {s: t for s, *t in stratum_tables(days, covered.shape, names, a.min_count)}
    print(f"strata done; {time.time() - t0:.0f} s", flush=True)
    null = names.index("clim_null")

    def stat(idx):
        out = {}
        for si, s in enumerate(STRATA):
            T, budget, vg, _ = tables[s]
            R = e1.pooled(T[idx].sum(0))["R"]
            for m, name in enumerate(names):
                out[f"{s}|gate|{name}"] = R[m]
                if m != null:
                    out[f"{s}|gate_minus_null|{name}"] = R[m] - R[null]
            n, s2, w = budget[idx].sum(0)
            V, k = (s2 / n, w / n) if n else (np.nan, np.nan)
            out[f"{s}|V|"] = V
            g = vg[idx].sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                gamma = g[:, :, 1] / (2 * g[:, :, 0])
            for ax, nm in ((0, "NS"), (1, "EW")):
                good = np.isfinite(gamma[ax, :e9.FIT_LAGS]).all()
                nug = e9.nugget(gamma[ax]) if good else np.nan
                out[f"{s}|nugget_{nm}|"] = nug
                out[f"{s}|ceiling_nugget_{nm}|"] = np.sqrt(1 - nug / V) if good and nug < V else (0.0 if good else np.nan)
            m_, ss = A[idx, si].sum(0)
            sig2 = ss / (2 * m_) if m_ else np.nan
            out[f"{s}|sigma_e_overlap|"] = np.sqrt(sig2)
            out[f"{s}|ceiling_overlap|"] = np.sqrt(max(1 - sig2 * k / V, 0.0)) if m_ and n else np.nan
        return out

    point = stat(np.arange(len(dates)))
    boot = evals.block_bootstrap(stat, evals.date_blocks(dates, a.block_days), n_boot=a.n_boot, seed=a.seed)
    rows = []
    for key, v in point.items():
        s, q, who = key.split("|")
        lo, hi = evals.ci(boot[key])
        rows.append(dict(stratum=s, quantity=q, method=who, value=v, ci_lo=lo, ci_hi=hi))
    metrics = pd.DataFrame(rows)
    counts = pd.DataFrame([dict(stratum=s, pixel_days=int(tables[s][1][:, 0].sum()), pixels=tables[s][3],
                                dates=int((tables[s][1][:, 0] > 0).sum()), overlap_pixel_pairs=int(A[:, si, 0].sum()))
                           for si, s in enumerate(STRATA)])

    run = Path(a.out_dir) / f"run_{pd.Timestamp.now():%Y%m%dT%H%M%S}"
    run.mkdir(parents=True)
    metrics.to_csv(run / "metrics.csv", index=False)
    counts.to_csv(run / "strata_counts.csv", index=False)
    cfg = dict(args=vars(a), methods=names, v1_runs=[str(x) for x in a.v1], v1_tile=a.v1_tile, strata=STRATA_DEF, n_train_dates=n_train,
               n_test_label_days=len(dates), composite_validity_mismatch_pixels=mismatch, flag_fill_on_valid_pixels=n_fill, clip=e1.CLIP,
               evaluation_set="E1 anomaly_R_pixmean set inside each stratum: label- and parent-valid, climatology-covered, >= min_count stratum observations",
               git=e1.git_sha("smapsr"), runtime_s=round(time.time() - t0, 1))
    (run / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    show = metrics.assign(col=metrics.quantity + metrics.method.map(lambda x: f":{x}" if x else "")).pivot(index="stratum", columns="col", values="value").reindex(STRATA)
    print(counts.to_string(index=False))
    print(show[[c for c in show if c.startswith(("gate:", "V", "ceiling"))]].round(4).to_string())
    print("wrote", run)


if __name__ == "__main__":
    main()
