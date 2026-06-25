#!/usr/bin/env bash
# Generate 2dk_no_repeat: tr1k zero-repeat train (1280 x 55s, warmup 42s, spectral) + 4 val anchors.
#   bash scripts/gen_no_repeat.sh   (~6 h on one GPU; val first, then the long train, then aggregate)
set -u; cd /home/david-wagner/git/torch-cfd || exit 1
PY=.venv/bin/python; G=2dk_no_repeat; ROOT=/scratch/dwcgt/$G; LOG=/scratch/dwcgt/bivort_logs/gen_no_repeat.log
{ echo "=== gen start $(date) ==="
  for re in 1000 2000 4000 8000; do echo ">>> val_Re$re ($(date +%H:%M:%S))"
    $PY scripts/create_dataset.py --config-name $G/val_Re$re no_tqdm=true hydra.run.dir=/tmp/hg/${G}_val$re hydra.output_subdir=null || echo "FAIL val_Re$re"; done
  echo ">>> train_Re1000 1280traj ($(date +%H:%M:%S))"
  $PY scripts/create_dataset.py --config-name $G/train_Re1000 no_tqdm=true hydra.run.dir=/tmp/hg/${G}_train hydra.output_subdir=null || echo "FAIL train"
  echo "=== aggregating $(date) ==="; $PY scripts/aggregate_dataset.py $ROOT; du -sh $ROOT; echo "=== ALL DONE $(date) ==="
} 2>&1 | tee $LOG
