#!/bin/bash
# ladder_queue_next.sh <dataset: sub1M|25M> <wait_seed> <next_seed>
# Wait for the ladder_<ds>_base_s<wait_seed> chain to END (COMPLETE or FAILED marker), then launch the
# next seed's ladder. Detached-queue helper for overnight n=3 -> n=4 sequencing.
set -u
DS="$1"; WS="$2"; NS="$3"
WLOG=/scratch/dwcgt/bivort_logs/ladder_${DS}_base_s${WS}.log
i=0
until grep -qE "COMPLETE|FAILED" "$WLOG" 2>/dev/null; do
  sleep 60; i=$((i+1))
  [ $i -gt 600 ] && { echo "QUEUE TIMEOUT waiting on s${WS}" >> "$WLOG"; exit 1; }
done
echo "[queue] s${WS} ended $(date) -> launching s${NS}" >> "$WLOG"
sleep 30
bash /home/david-wagner/git/torch-cfd/scripts/ladder_seed.sh "$DS" "$NS"
