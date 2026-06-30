#!/bin/bash
# pipeline_25m.sh -- master driver (runs on alpha). Bulk rsync goes over the 5GbE DIRECT ETHERNET
# (beta=10.42.0.1), NOT tailscale. Control SSH (markers/rm/launch) stays on tailscale. Before the
# first ethernet rsync it WAITS for beta's real sshd (needs 'bash ~/git/5090/direct-link.sh' on beta).
set -u
B=david-wagner@5090-beta.tail0e606b.ts.net     # tailscale: control commands
ETH=david-wagner@10.42.0.1                       # 5GbE direct: bulk data only
RS=(-a -e "ssh -o BatchMode=yes -o ConnectTimeout=10")
ROOT=/scratch/dwcgt/2dk_25M_tr1k
W45=/scratch/dwcgt/2dk_spectral_w45
LOG=/scratch/dwcgt/bivort_logs/pipeline_25m.log
TCFD=/home/david-wagner/git/torch-cfd
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ssh_b(){ timeout "$1" ssh -o BatchMode=yes -o ConnectTimeout=15 "$B" "$2"; }

log "===== 25M pipeline driver start (bulk rsync over 5GbE 10.42.0.1) ====="

# PHASE A -- wait for gen on BOTH nodes (local marker + beta marker over tailscale)
log "A: waiting for gen done markers..."
while [ ! -f "$ROOT/.gen_done_alpha" ]; do sleep 120; done; log "  alpha gen done"
until ssh_b 15 "[ -f $ROOT/.gen_done_beta ]"; do sleep 120; done; log "  beta gen done"

# PHASE A2 -- wait for beta's real sshd on the ethernet IP (needed for bulk rsync)
log "A2: waiting for beta ethernet sshd @10.42.0.1:22 -- run 'bash ~/git/5090/direct-link.sh' on beta"
until ssh -o BatchMode=yes -o ConnectTimeout=5 "$ETH" true 2>/dev/null; do sleep 60; done
log "  ethernet sshd up -> proceeding"

# PHASE B -- merge beta's shards (100-199) + ic512 into alpha, over ethernet
log "B: rsync (5GbE) beta train/ -> alpha ..."
if rsync "${RS[@]}" "$ETH:$ROOT/train/" "$ROOT/train/" >>"$LOG" 2>&1; then
  log "  merged. alpha shards=$(ls $ROOT/train/*.npy 2>/dev/null|wc -l) ic512=$(ls $ROOT/train/ic512/*.npy 2>/dev/null|wc -l)"
else log "  !!! MERGE FAILED -- abort"; exit 1; fi

# PHASE C -- partition-aggregate (deletes raw shards as it goes; ic512/ untouched)
log "C: aggregate_partitioned ..."
cd "$TCFD" || exit 1
if .venv/bin/python scripts/aggregate_partitioned.py "$ROOT" 8 >>"$LOG" 2>&1; then
  log "  aggregated. partitions=$(ls $ROOT/train/train_Re1000_p*.pt 2>/dev/null|wc -l) size=$(du -sh $ROOT 2>/dev/null|cut -f1)"
else log "  !!! AGGREGATE FAILED -- abort"; exit 1; fi

# PHASE D -- symlink the w45 4-Re val
log "D: symlinking w45 4-Re val ..."
mkdir -p "$ROOT/val"
for re in 1000 2000 4000 8000; do ln -sf "$W45/val/val_Re$re.pt" "$ROOT/val/val_Re$re.pt"; done
log "  val: $(ls $ROOT/val/ | tr '\n' ' ')"

# PHASE E -- mirror full dataset to beta over ethernet (clear beta's stale raw shards first)
log "E: mirror dataset -> beta (5GbE) ..."
ssh_b 60 "rm -f $ROOT/train/*.npy; mkdir -p $ROOT"
if rsync "${RS[@]}" "$ROOT/" "$ETH:$ROOT/" >>"$LOG" 2>&1; then
  log "  mirrored. beta partitions=$(ssh_b 20 "ls $ROOT/train/train_Re1000_p*.pt 2>/dev/null|wc -l")"; MIRROR_OK=1
else log "  !! MIRROR FAILED -- refiner ladder skipped; base ladder still runs"; MIRROR_OK=0; fi

# PHASE F -- launch ladders: base@alpha, refiner@beta (parallel)
log "F: launching base ladder (alpha) ..."
setsid nohup bash "$TCFD/scripts/ladder_25m.sh" base false </dev/null >/dev/null 2>&1 &
if [ "${MIRROR_OK:-0}" = 1 ]; then
  log "F: launching refiner ladder (beta) ..."
  ssh_b 25 "setsid nohup bash $TCFD/scripts/ladder_25m.sh refiner true </dev/null >/dev/null 2>&1 &"
fi
log "===== driver done: ladders running (base@alpha, refiner@beta). =====
"
