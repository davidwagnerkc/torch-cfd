#!/usr/bin/env bash
# Generate the 2dk_re_sweep_bi_vort dataset on this machine: forced 2D Kolmogorov DNS at
# 512^2, saved 64^2 with the divergence-free bilinear down-sample (vorticity -> reconstruct).
# train/test at Re {1000,2000,4000,8000}; val at Re 1000..8000 step 250. Sequential on one GPU.
#
#   bash scripts/gen_bi_vort.sh        # ~1 h generate + aggregate, logs to $LOG
set -u
cd /home/david-wagner/git/torch-cfd || exit 1
PY=.venv/bin/python
GROUP=2dk_re_sweep_bi_vort
ROOT=/scratch/dwcgt/$GROUP
LOG=/scratch/dwcgt/gen_bi_vort.log

{
  echo "=== generation start $(date) ==="
  fail=0; n=0
  for f in scripts/conf/$GROUP/*.yaml; do
    cfg=$(basename "$f" .yaml); n=$((n+1))
    echo ">>> [$n/37] $cfg  ($(date +%H:%M:%S))"
    $PY scripts/create_dataset.py --config-name "$GROUP/$cfg" \
        no_tqdm=true hydra.run.dir="/tmp/hg/$cfg" hydra.output_subdir=null \
        || { echo "!!! FAILED: $cfg"; fail=$((fail+1)); }
  done
  echo "=== generation done: $fail failure(s). aggregating $(date) ==="
  $PY scripts/aggregate_dataset.py "$ROOT"
  echo "=== aggregate done $(date) ==="
  du -sh "$ROOT" 2>/dev/null
  echo "per-(split,Re) .pt files: $(ls "$ROOT"/*/*.pt 2>/dev/null | wc -l)"
  echo "=== ALL DONE $(date) (failures: $fail) ==="
} 2>&1 | tee "$LOG"
