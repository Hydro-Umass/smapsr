#!/usr/bin/env bash
# E1 trained baselines at the final V1 budget (9,000 steps, SSIM weight 0.5), with and without the climatology channel,
# one seed per invocation. Same data, split, sampler, loss and optimiser as V1; only the architecture differs.
#   edsr:  EDSR-lite, 8 residual blocks, width 32, kernel 3 (168k parameters with the channel; V1 162k), one forward pass.
#   euler: same-f discrete ResNet: V1's vector field (k7/d4, pixel norm) with explicit Euler, 61 steps (= V1's 61 evaluations).
# V1 references: results/e2_ablations/norm/pixel (clim), results/e2_ablations/final_control/noclim (no clim).
# Usage (project root): bash smapsr/experiments/e1_trained_baselines.sh <seed>
set -euo pipefail
seed=$1
out=results/e1_baselines/trained
common=(--steps 9000 --seed "$seed" --eval-every 1000 --print-every 1000 --checkpoint-every 1000)
clim=(--static label_mean anomaly)
edsr=(--arch edsr --depth 8 --kernel 3)
euler=(--kernel 7 --depth 4 --norm pixel --solver euler --n-steps 61)
run() { PYTHONPATH=smapsr python smapsr/experiments/train_v1.py "${common[@]}" --out-dir "$out/$1/seed$seed" "${@:2}"; }
run edsr_clim "${edsr[@]}" "${clim[@]}"
run edsr_noclim "${edsr[@]}"
run euler_clim "${euler[@]}" "${clim[@]}"
run euler_noclim "${euler[@]}"
