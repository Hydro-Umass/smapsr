#!/usr/bin/env bash
# E2 track-corrected labels: EDSR-lite (+clim and no clim, as in e1_trained_baselines.sh) trained on labels with the
# pixel x track offset removed (cache from e2_track_offsets.py), one seed per invocation.
# Usage (project root): bash smapsr/experiments/e2_track_corrected.sh <seed>
set -euo pipefail
seed=$1
out=results/e2_ablations/track_corrected
common=(--cache data_qc/cache/aligned_AM_trackcorr --arch edsr --depth 8 --kernel 3 --steps 9000 --seed "$seed" --eval-every 1000 --print-every 1000)
PYTHONPATH=smapsr python smapsr/experiments/train_v1.py "${common[@]}" --static label_mean anomaly --out-dir "$out/edsr_clim/seed$seed"
PYTHONPATH=smapsr python smapsr/experiments/train_v1.py "${common[@]}" --out-dir "$out/edsr_noclim/seed$seed"
