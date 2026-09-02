#!/usr/bin/env bash
# Generate 2dk_re125_w45 (Re=125, standard w45 recipe, saved at 128^2): val (128 x 20 s) then
# train (128 x 55 s), aggregate train into 2 partitions of 4 shards (train_Re125_p{0,1}.pt, shards
# deleted as they are absorbed) and val into val_Re125.pt, drop val shards. ic512/ is preserved.
#   bash scripts/gen_re125_w45.sh
set -u; cd /home/david-wagner/git/torch-cfd || exit 1
PY=.venv/bin/python; G=2dk_re125_w45; ROOT=/scratch/dwcgt/$G; mkdir -p /scratch/dwcgt/bivort_logs; LOG=/scratch/dwcgt/bivort_logs/gen_${G}.log
{ echo "=== gen $G start $(date) ==="
  echo ">>> val_Re125 ($(date +%H:%M:%S))"
  $PY scripts/create_dataset.py --config-name $G/val_Re125 no_tqdm=true hydra.run.dir=/tmp/hg/${G}_val125 hydra.output_subdir=null || echo "FAIL val_Re125"
  echo ">>> train_Re125 128traj ($(date +%H:%M:%S))"
  $PY scripts/create_dataset.py --config-name $G/train_Re125 no_tqdm=true hydra.run.dir=/tmp/hg/${G}_train125 hydra.output_subdir=null || echo "FAIL train_Re125"
  echo "=== partition-aggregating (4 shards/partition) $(date) ==="; $PY scripts/aggregate_partitioned.py $ROOT 4
  rm -f $ROOT/val/*.npy $ROOT/val/*.json
  ls -la $ROOT/train $ROOT/val; du -sh $ROOT; echo "=== GEN $G DONE $(date) ==="
} 2>&1 | tee $LOG
