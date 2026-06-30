#!/bin/bash
# pipeline_25m.sh -- master driver (runs on alpha). Chains: wait-for-gen -> merge -> aggregate ->
# val symlinks -> mirror to beta -> launch base ladder (alpha) + refiner ladder (beta). Unattended.
set -u
B=david-wagner@5090-beta.tail0e606b.ts.net
ROOT=/scratch/dwcgt/2dk_25M_tr1k
W45=/scratch/dwcgt/2dk_spectral_w45
LOG=/scratch/dwcgt/bivort_logs/pipeline_25m.log
TCFD=/home/david-wagner/git/torch-cfd
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ssh_b(){ timeout "${1}" ssh -o BatchMode=yes -o ConnectTimeout=15 "$B" "$2"; }

log "===== 25M pipeline driver start ====="

# PHASE A -- wait for gen on BOTH nodes
log "A: waiting for gen done markers (poll 120s)..."
while [ ! -f "$ROOT/.gen_done_alpha" ]; do sleep 120; done
log "  alpha gen done"
until ssh_b 15 "[ -f $ROOT/.gen_done_beta ]"; do sleep 120; done
log "  beta gen done. alpha shards=$(ls $ROOT/train/*.npy 2>/dev/null|wc -l) ic512=$(ls $ROOT/train/ic512/*.npy 2>/dev/null|wc -l)"

# PHASE B -- merge beta's shards (100-199) + ic512 into alpha
log "B: rsync beta train/ -> alpha ..."
if rsync -a "$B:$ROOT/train/" "$ROOT/train/" >>"$LOG" 2>&1; then
  log "  merged. alpha shards now=$(ls $ROOT/train/*.npy 2>/dev/null|wc -l)"
else log "  !!! MERGE FAILED -- abort"; exit 1; fi

# PHASE C -- partition-aggregate (deletes raw shards as it goes; ic512/ untouched)
log "C: aggregate_partitioned ..."
cd "$TCFD" || exit 1
if .venv/bin/python scripts/aggregate_partitioned.py "$ROOT" 8 >>"$LOG" 2>&1; then
  log "  aggregated. partitions=$(ls $ROOT/train/train_Re1000_p*.pt 2>/dev/null|wc -l) size=$(du -sh $ROOT 2>/dev/null|cut -f1)"
else log "  !!! AGGREGATE FAILED -- abort"; exit 1; fi

# PHASE D -- symlink the w45 4-Re val (Re1k in-dist + 2k/4k/8k OOD)
log "D: symlinking w45 4-Re val ..."
mkdir -p "$ROOT/val"
for re in 1000 2000 4000 8000; do ln -sf "$W45/val/val_Re$re.pt" "$ROOT/val/val_Re$re.pt"; done
log "  val: $(ls $ROOT/val/ | tr '\n' ' ')"

# PHASE E -- mirror full dataset to beta (clear beta's stale raw shards first to free disk)
log "E: mirror dataset -> beta ..."
ssh_b 60 "rm -f $ROOT/train/*.npy; mkdir -p $ROOT"
if rsync -a "$ROOT/" "$B:$ROOT/" >>"$LOG" 2>&1; then
  log "  mirrored. beta partitions=$(ssh_b 20 "ls $ROOT/train/train_Re1000_p*.pt 2>/dev/null|wc -l")"
else log "  !! MIRROR FAILED -- refiner ladder (beta) will be skipped; base ladder still runs"; MIRROR_OK=0; fi

# PHASE F -- launch ladders: base on alpha, refiner on beta (parallel)
log "F: launching base ladder (alpha) ..."
setsid nohup bash "$TCFD/scripts/ladder_25m.sh" base false </dev/null >/dev/null 2>&1 &
if [ "${MIRROR_OK:-1}" = 1 ]; then
  log "F: launching refiner ladder (beta) ..."
  ssh_b 25 "setsid nohup bash $TCFD/scripts/ladder_25m.sh refiner true </dev/null >/dev/null 2>&1 &"
fi
log "===== driver done: ladders running (base@alpha, refiner@beta). Tail ladder_25m_*.log ====="
