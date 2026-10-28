"""Summarise a sweep of train_v1 runs: final monitoring metrics per config, mean and range over seeds.

Usage: python smapsr/experiments/e2_summarise.py results/e2_ablations/receptive_field [--baseline results/v1_train/run_A results/v1_train/run_B ...]
Writes <dir>/summary.csv (one row per run) and <dir>/summary_by_config.csv. Selection uses the monitoring block only.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def row(run):
    cfg = json.loads((run / "config.json").read_text())
    h = pd.read_csv(run / "history.csv").dropna(subset=["val_rmse"]).iloc[-1]
    a = cfg["args"]
    return dict(run=str(run), kernel=a.get("kernel", 3), depth=a["depth"], width=a["width"], n_steps=a["n_steps"], seed=a["seed"],
                params=cfg["model"]["n_parameters"], step=int(h["step"]), val_rmse=h["val_rmse"], val_R=h["val_R"],
                val_rmse_bilinear=h["val_rmse_bilinear"], val_rmse_u0=h["val_rmse_u0"], runtime_s=cfg["runtime_s"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep")
    p.add_argument("--baseline", nargs="*", default=[])
    a = p.parse_args()
    runs = [r for r in sorted(Path(a.sweep).glob("*/run_*")) + [Path(b) for b in a.baseline] if (r / "config.json").exists()]   # finished runs only
    df = pd.DataFrame([row(r) for r in runs])
    df["rf_radius_fine_cells"] = (df.depth + 1) * (df.kernel - 1) / 2
    df.to_csv(Path(a.sweep) / "summary.csv", index=False)
    g = df.groupby(["kernel", "depth"]).agg(n_seeds=("seed", "count"), params=("params", "first"), rf=("rf_radius_fine_cells", "first"),
                                            rmse_mean=("val_rmse", "mean"), rmse_min=("val_rmse", "min"), rmse_max=("val_rmse", "max"),
                                            R_mean=("val_R", "mean"), bilinear=("val_rmse_bilinear", "mean"), runtime_s=("runtime_s", "mean")).round(4)
    g.to_csv(Path(a.sweep) / "summary_by_config.csv")
    print(g.to_string())


if __name__ == "__main__":
    main()
