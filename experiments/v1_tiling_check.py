"""Does inference extent matter? Full-grid vs crop-sized tiled inference of a trained V1 on the monitoring dates.

The vector field uses GroupNorm, whose statistics depend on the spatial extent of its input; training sees random
crops of `--tile` coarse cells while full-domain inference sees the whole grid. This script quantifies the difference
on label-valid, parent-valid pixels of the monitoring block (never the 2016-2020 test period).

Usage (project root): PYTHONPATH=smapsr python smapsr/experiments/v1_tiling_check.py --run results/v1_train/run_<ts>
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from smapsr import evals
from smapsr.baselines import bilinear, parent_valid

CLIP = (0.01, 0.65)


def tiled(fwd, x, tile, halo, factor):
    """Predict (1, H*f, W*f) from x (C, h, w) with tiles of `tile` coarse cells plus a `halo` on each side."""
    _, h, w = x.shape
    out = np.zeros((h * factor, w * factor), dtype=np.float32)
    pad = np.pad(x, ((0, 0), (halo, halo), (halo, halo)))                 # zero input + zero validity outside the domain
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            ti, tj = min(tile, h - i), min(tile, w - j)
            patch = pad[:, i:i + ti + 2 * halo, j:j + tj + 2 * halo]
            y = np.asarray(fwd(jnp.asarray(patch)))[0]
            out[i * factor:(i + ti) * factor, j * factor:(j + tj) * factor] = y[halo * factor:(halo + ti) * factor, halo * factor:(halo + tj) * factor]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--cache", default="data_qc/cache/aligned_AM")
    p.add_argument("--tile", type=int, default=32)
    p.add_argument("--halos", type=int, nargs="*", default=[0, 8])
    p.add_argument("--max-dates", type=int, default=0)
    a = p.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    model, mean, std = evals.load_v1(run / "models" / "model.eqx")
    scale, factor = float(cfg["args"]["scale_km"]), int(cfg["args"]["factor"])
    fwd = jax.jit(lambda x: model(x, scale))
    dates = pd.read_csv(run / "inputs" / "val_dates.csv").date.tolist()
    if a.max_dates:
        dates = dates[:a.max_dates]
    acc = {k: np.zeros(5) for k in ["bilinear", "full"] + [f"tiled_halo{h}" for h in a.halos]}     # n, sum o, sum p, sum (p-o)^2, sum o*p ... see below
    sq = {k: [0, 0.0] for k in acc}
    used = 0
    for d in dates:
        d = d.replace("-", "")
        cp, fp = Path(a.cache) / "9km" / f"{d}.npz", Path(a.cache) / "3km" / f"{d}.npz"
        if not (cp.exists() and fp.exists()):
            continue
        c = np.clip(np.load(cp)["soil_moisture"].astype(np.float32), *CLIP)
        f = np.clip(np.load(fp)["soil_moisture"].astype(np.float32), *CLIP)
        ev = np.isfinite(f) & parent_valid(c, factor)
        if not ev.any():
            continue
        v = np.isfinite(c)
        x = np.stack([np.where(v, (c - mean) / std, 0.0), v.astype(np.float32)]).astype(np.float32)
        preds = {"bilinear": bilinear(c, factor), "full": np.asarray(fwd(jnp.asarray(x)))[0] * std + mean}
        for h in a.halos:
            preds[f"tiled_halo{h}"] = tiled(fwd, x, a.tile, h, factor) * std + mean
        for k, pr in preds.items():
            sq[k][0] += int(ev.sum())
            sq[k][1] += float(((pr[ev] - f[ev]) ** 2).sum())
        used += 1
    res = {k: dict(n=n, RMSE=float(np.sqrt(s / n))) for k, (n, s) in sq.items()}
    res["full_minus_tiled"] = {k: res["full"]["RMSE"] - res[k]["RMSE"] for k in res if k.startswith("tiled")}
    out = dict(run=str(run), dates_used=used, tile=a.tile, halos=a.halos, rmse=res)
    (run / "tiling_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
