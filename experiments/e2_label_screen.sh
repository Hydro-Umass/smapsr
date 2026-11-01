#!/usr/bin/env bash
# E2 label screen: EDSR-lite (+clim and no clim, as in e1_trained_baselines.sh) trained on labels with L2 bits 6|7 dropped
# (cache from mosaic_daily.py --qc-mask 0xC0, build_aligned_cache.py, screen_aligned_cache.py), one seed per invocation.
# Usage (project root): bash smapsr/experiments/e2_label_screen.sh <seed>
set -euo pipefail
seed=$1
out=results/e2_ablations/label_screen
common=(--cache data_qc/cache/aligned_AM_qcC0 --arch edsr --depth 8 --kernel 3 --steps 9000 --seed "$seed" --eval-every 1000 --print-every 1000)
PYTHONPATH=smapsr python smapsr/experiments/train_v1.py "${common[@]}" --static label_mean anomaly --out-dir "$out/edsr_clim/seed$seed"
PYTHONPATH=smapsr python smapsr/experiments/train_v1.py "${common[@]}" --out-dir "$out/edsr_noclim/seed$seed"
