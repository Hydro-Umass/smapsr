#!/usr/bin/env bash
# E2 capacity sweep on V1+clim (k7/d4, static label_mean + anomaly), one seed per invocation.
#   steps: one long run (STEPS, default 12000) with the monitoring block evaluated every 1000 steps. The learning
#          rate is constant (no schedule), so the history at step n is what a run stopped at n would show.
#   loss:  SSIM weight w of the convex weighting (loss = (1 - w) MSE + w (1 - SSIM)), 3,000 steps;
#          w = 0.5 is results/e2_ablations/static_clim/seed<seed>_k7_d4 (not re-run).
# Usage (project root): [STEPS=12000] [WEIGHTS="0.0 0.25 0.75"] bash smapsr/experiments/e2_capacity.sh <seed>
set -euo pipefail
seed=$1
out=results/e2_ablations/capacity
common=(--kernel 7 --depth 4 --static label_mean anomaly --seed "$seed" --eval-every 1000 --print-every 500)
PYTHONPATH=smapsr python smapsr/experiments/train_v1.py "${common[@]}" --steps "${STEPS:-12000}" \
  --out-dir "$out/steps/seed${seed}"
for w in ${WEIGHTS:-0.0 0.25 0.75}; do
  PYTHONPATH=smapsr python smapsr/experiments/train_v1.py "${common[@]}" --steps 3000 --ssim-weight "$w" \
    --out-dir "$out/ssim_weight/seed${seed}_w$w"
done
