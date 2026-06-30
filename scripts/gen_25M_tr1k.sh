#!/usr/bin/env bash
# Generate 2dk_25M_tr1k (Re1k, 45s warmup, spectral, 55s, 25.05M windows). Train-only; reuses the
# 2dk_spectral_w45 val_Re1000. Single-node by default; to split across both nodes pass disjoint
# (nsamp, seed): e.g. alpha "1600 1700000000", beta "1600 1700001600", then combine raw shards
# into one $ROOT/train before the aggregate step.
set -u; cd /home/david-wagner/git/torch-cfd || exit 1
PY=.venv/bin/python; G=2dk_25M_tr1k; ROOT=/scratch/dwcgt/$G; LOG=/scratch/dwcgt/bivort_logs/gen_${G}.log
NSAMP="${1:-3200}"; SEED="${2:-1700000000}"; DO_AGG="${3:-1}"
{ echo "=== gen $G nsamp=$NSAMP seed=$SEED start $(date) ==="
  $PY scripts/create_dataset.py --config-name $G/train_Re1000 num_samples=$NSAMP seed=$SEED \
      no_tqdm=true hydra.run.dir=/tmp/hg/${G}_train hydra.output_subdir=null || echo "FAIL train"
  if [ "$DO_AGG" = 1 ]; then
    echo "=== partition-aggregating $(date) ==="; $PY scripts/aggregate_partitioned.py $ROOT 8
    mkdir -p $ROOT/val
    ln -sf /scratch/dwcgt/2dk_spectral_w45/val/val_Re1000.pt $ROOT/val/val_Re1000.pt   # reuse w45 val
    du -sh $ROOT
  fi
  echo "=== GEN $G [nsamp=$NSAMP] DONE $(date) ==="
} 2>&1 | tee -a $LOG
