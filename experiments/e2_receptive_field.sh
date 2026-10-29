#!/usr/bin/env bash
# E2 receptive-field sweep: kernel x depth of the V1 vector field, one seed per invocation.
# Receptive-field radius per function evaluation = (depth + 1) * (kernel - 1) / 2 fine cells.
# Usage (project root): [CONFIGS="3 2;5 4"] bash smapsr/experiments/e2_receptive_field.sh <seed>   (CONFIGS = "kernel depth" pairs, ";"-separated)
# Baseline (kernel 3, depth 4) is results/v1_train/run_20261028T{105017,110731,112433} (seeds 0-2, same settings).
set -euo pipefail
seed=$1
IFS=";" read -ra cfgs <<< "${CONFIGS:-3 2;5 4;7 4;3 8}"
for cfg in "${cfgs[@]}"; do
  set -- $cfg
  PYTHONPATH=smapsr python smapsr/experiments/train_v1.py --kernel "$1" --depth "$2" --steps 3000 --seed "$seed" \
    --eval-every 500 --print-every 500 --out-dir "results/e2_ablations/receptive_field/seed${seed}_k$1_d$2"
done
