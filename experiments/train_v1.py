"""Train the V1 model (shared local operator, daily masked crops; amendments A1, section 4.5) at one scale.

Reads the aligned AM-only cache (`scripts/build_aligned_cache.py --pass AM`), trains on 2021-2024 minus the
SlimSAR window (and minus an optional validation block used only for monitoring), and never opens the 2016-2020
test period. Reports the validation error of the full model u(1), of the learned initial state u(0) and of plain
bilinear interpolation of the gap-filled input, all on label-valid pixels. Writes results/v1_train/run_<timestamp>/.

Usage (project root, devenv shell active):
    PYTHONPATH=smapsr python smapsr/experiments/train_v1.py --steps 2000
    PYTHONPATH=smapsr python smapsr/experiments/train_v1.py --steps 20 --batch-size 2 --crop 16 --width 8   # smoke
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd

from smapsr import evals
from smapsr.data import DailyPairs
from smapsr.losses import masked_daily_loss
from smapsr.models import NeuralODEV1
from smapsr.train import train_v1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data_qc/cache/aligned_AM", help="aligned cache with <coarse>/ and <fine>/ npz folders")
    p.add_argument("--out-dir", default="results/v1_train")
    p.add_argument("--coarse", default="9km")
    p.add_argument("--fine", default="3km")
    p.add_argument("--factor", type=int, default=3, help="fine/coarse grid ratio (3 for 9 km -> 3 km)")
    p.add_argument("--scale-km", type=float, default=3.0, help="target pixel size, enters through the scale embedding")
    p.add_argument("--train-start", default="2021-01-01")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--exclude-dates", nargs="*", default=["2022-04-26"], help="SlimSAR acquisitions, removed from training")
    p.add_argument("--exclude-buffer-days", type=int, default=4)
    p.add_argument("--val-start", default="2024-07-01", help="first day of the monitoring block inside the training period; 'none' trains on everything")
    p.add_argument("--crop", type=int, default=32, help="crop size in coarse cells")
    p.add_argument("--min-label-frac", type=float, default=0.05)
    p.add_argument("--min-input-frac", type=float, default=0.10)
    p.add_argument("--origin-stride", type=int, default=2)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--kernel", type=int, default=3, help="conv kernel size of the vector field (receptive field per evaluation = (depth + 1) * (kernel - 1) / 2 fine cells)")
    p.add_argument("--n-freq", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=10, help="solver steps over t in [0, 1] (about 6 n + 1 function evaluations)")
    p.add_argument("--steps", type=int, default=2000, help="optimiser updates (batches, not epochs)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--ssim-weight", type=float, default=0.5)
    p.add_argument("--min-valid-fraction", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--val-crops", type=int, default=64)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def git_state(repo):
    def run(*cmd):
        return subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True, text=True).stdout.strip()
    return dict(sha=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"), dirty=bool(run("status", "--porcelain")))


def split_dates(args):
    d = pd.date_range(args.train_start, args.train_end)
    for x in args.exclude_dates:
        c = pd.Timestamp(x)
        d = d[~((d >= c - pd.Timedelta(days=args.exclude_buffer_days)) & (d <= c + pd.Timedelta(days=args.exclude_buffer_days)))]
    if args.val_start.lower() == "none":
        return d, d[:0]
    v = pd.Timestamp(args.val_start)
    return d[d < v], d[d >= v]


def main():
    args = parse_args()
    t0 = time.time()
    repo = Path(__file__).resolve().parents[1]
    run_dir = Path(args.out_dir) / f"run_{time.strftime('%Y%m%dT%H%M%S')}"
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "inputs").mkdir()

    train_dates, val_dates = split_dates(args)
    assert train_dates.min() >= pd.Timestamp(args.train_start), "training dates leaked outside the training period"
    kw = dict(coarse=args.coarse, fine=args.fine, factor=args.factor, crop=args.crop, min_label_frac=args.min_label_frac,
              min_input_frac=args.min_input_frac, origin_stride=args.origin_stride)
    train = DailyPairs(args.cache, train_dates, **kw)
    val = DailyPairs(args.cache, val_dates, norm=(train.mean, train.std), **kw) if len(val_dates) else None
    pd.Series([f"{d:%Y-%m-%d}" for d in train.dates], name="date").to_csv(run_dir / "inputs" / "train_dates.csv", index=False)
    if val is not None:
        pd.Series([f"{d:%Y-%m-%d}" for d in val.dates], name="date").to_csv(run_dir / "inputs" / "val_dates.csv", index=False)
    print("train", train.summary(), "\nval", val.summary() if val else None, flush=True)

    strides = (args.factor,)
    hyper = dict(in_channels=2, strides=list(strides), width=args.width, depth=args.depth, groups=args.groups,
                 n_freq=args.n_freq, n_steps=args.n_steps, kernel=args.kernel)
    model = NeuralODEV1(2, strides, args.width, args.depth, args.groups, args.n_freq, args.n_steps, args.kernel, key=jr.PRNGKey(args.seed))
    reference = NeuralODEV1(2, strides, args.width, args.depth, args.groups, args.n_freq, args.n_steps, args.kernel, key=jr.PRNGKey(0))  # untrained = bilinear
    n_params = sum(a.size for a in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))

    eval_fn = None
    if val is not None:
        vx, vy, vm, _ = val.fixed(args.val_crops, seed=1234)
        vx, vy, vm = jnp.asarray(vx), jnp.asarray(vy), jnp.asarray(vm)
        parent = np.repeat(np.repeat(np.asarray(vx[:, 1]), args.factor, axis=1), args.factor, axis=2) > 0   # label pixel with a valid coarse parent

        @jax.jit
        def predict(m, x):
            return jax.vmap(m, in_axes=(0, None))(x, args.scale_km)[:, 0]
        bil = np.asarray(jax.jit(jax.vmap(reference.initial_state))(vx)[:, 0])

        def pooled(pred, mask):
            mm = np.asarray(vm) > 0
            mm = mm & parent if mask == "parent" else mm
            return evals.point_metrics(np.asarray(vy[:, 0])[mm], pred[mm])

        def eval_fn(model, step):
            u1 = np.asarray(predict(model, vx))
            u0 = np.asarray(jax.jit(jax.vmap(model.initial_state))(vx)[:, 0])
            loss = float(masked_daily_loss(jnp.asarray(u1), vy[:, 0], vm, val.data_range, args.ssim_weight, args.min_valid_fraction))
            out = dict(val_loss=loss)
            for name, pred in (("", u1), ("_u0", u0), ("_bilinear", bil)):
                for sub in ("all", "parent"):
                    r = pooled(pred, sub)
                    suffix = "" if sub == "all" else "_parentvalid"
                    out[f"val_rmse{name}{suffix}"] = r["RMSE"] * train.std      # m3/m3
                    out[f"val_R{name}{suffix}"] = r["R"]
            return out

    model, history = train_v1(model, train, scale_km=args.scale_km, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                              clip_norm=args.clip_norm, ssim_weight=args.ssim_weight, min_valid_fraction=args.min_valid_fraction,
                              seed=args.seed, print_every=args.print_every, eval_every=args.eval_every, eval_fn=eval_fn)
    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)
    evals.save(run_dir / "models" / "model.eqx", dict(hyper, data_mean=train.mean, data_std=train.std), model)

    config = dict(experiment="v1_train", script=str(Path(__file__).resolve().relative_to(repo.parent)), git=git_state(repo), args=vars(args),
                  jax=dict(version=jax.__version__, devices=[str(d) for d in jax.devices()]),
                  model=dict(class_="NeuralODEV1", hyperparameters=hyper, n_parameters=int(n_params), approx_nfe=6 * args.n_steps + 1,
                             saved_as="models/model.eqx (evals.load_v1)"),
                  data=dict(cache=str(Path(args.cache).resolve()), train=train.summary(), val=val.summary() if val else None,
                            normalisation="mean/std of the valid coarse (clipped) pixels of the training dates, applied unchanged to validation",
                            clip=list(train.clip), labels="never gap-filled; per-date masks; zero-filled where invalid",
                            input_gaps="set to 0 (training mean in normalised units) with a validity channel",
                            test_period="2016-2020 is never opened by this script"),
                  loss=f"masked_daily_loss: {1 - args.ssim_weight} * masked MSE + {args.ssim_weight} * (1 - masked SSIM), SSIM windows need >= {args.min_valid_fraction} valid fraction, normalised units",
                  optimiser=f"AdaBelief lr={args.lr}" + (f" with global-norm clipping {args.clip_norm}" if args.clip_norm else ""),
                  final=hist.iloc[-1].to_dict(), runtime_s=round(time.time() - t0, 1))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"wrote {run_dir} in {config['runtime_s']} s")


if __name__ == "__main__":
    main()
