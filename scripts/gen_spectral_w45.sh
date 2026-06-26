#!/usr/bin/env bash
# Generate 2dk_spectral_w45 for this machine's Re share, then partition-aggregate + drop val shards.
#   alpha: bash scripts/gen_spectral_w45.sh "1000 8000"   |   beta: ... "2000 4000"
set -u; cd /home/david-wagner/git/torch-cfd || exit 1
PY=.venv/bin/python; G=2dk_spectral_w45; ROOT=/scratch/dwcgt/$G; LOG=/scratch/dwcgt/bivort_logs/gen_${G}.log
RES="$1"
{ echo "=== gen $G [$RES] start $(date) ==="
  for re in $RES; do echo ">>> val_Re$re ($(date +%H:%M:%S))"
    $PY scripts/create_dataset.py --config-name $G/val_Re$re no_tqdm=true hydra.run.dir=/tmp/hg/${G}_val$re hydra.output_subdir=null || echo "FAIL val_Re$re"; done
  for re in $RES; do echo ">>> train_Re$re 320traj ($(date +%H:%M:%S))"
    $PY scripts/create_dataset.py --config-name $G/train_Re$re no_tqdm=true hydra.run.dir=/tmp/hg/${G}_train$re hydra.output_subdir=null || echo "FAIL train_Re$re"; done
  echo "=== partition-aggregating $(date) ==="; $PY scripts/aggregate_partitioned.py $ROOT 8
  rm -f $ROOT/val/*.npy $ROOT/val/*.json
  du -sh $ROOT; echo "=== GEN $G [$RES] DONE $(date) ==="
} 2>&1 | tee $LOG
