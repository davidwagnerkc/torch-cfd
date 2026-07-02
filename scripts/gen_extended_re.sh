#!/usr/bin/env bash
# Generate 2dk_extended_re (VAL-ONLY wide-Re extension of 2dk_spectral_w45) for this machine's Re share,
# then aggregate shards -> val_Re{re}.pt and drop the shard npy/json (ic512/ subdir is preserved).
#   alpha: bash scripts/gen_extended_re.sh "500 595 707 841 1189 1414 1682 2378 2828"
#   beta:  bash scripts/gen_extended_re.sh "3364 4757 5657 6727 9514 11314 13454 16000"
# Grid: Re_j = 500*2^(j/4), j=0..20, minus the existing w45 anchors {1000,2000,4000,8000} (reused, not regenerated).
# No train split, no test split (for now). 128 traj x 20s per Re, 45s warmup, 512^2 spectral, spectral downsample.
set -u; cd /home/david-wagner/git/torch-cfd || exit 1
PY=.venv/bin/python; G=2dk_extended_re; ROOT=/scratch/dwcgt/$G; LOG=/scratch/dwcgt/bivort_logs/gen_${G}.log
RES="$1"
{ echo "=== gen $G [$RES] start $(date) ==="
  for re in $RES; do echo ">>> val_Re$re ($(date +%H:%M:%S))"
    $PY scripts/create_dataset.py --config-name $G/val_Re$re no_tqdm=true hydra.run.dir=/tmp/hg/${G}_val$re hydra.output_subdir=null || echo "FAIL val_Re$re"; done
  echo "=== aggregating $(date) ==="; $PY scripts/aggregate_dataset.py $ROOT
  rm -f $ROOT/val/*.npy $ROOT/val/*.json
  du -sh $ROOT; echo "=== GEN $G [$RES] DONE $(date) ==="
} 2>&1 | tee -a $LOG
