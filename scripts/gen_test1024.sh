#!/usr/bin/env bash
# Generate the official 1024-trajectory Re1000 test set (2dk_spectral_w45/test).
#   bash scripts/gen_test1024.sh [FIRST_RANK] [LAST_RANK]        # default 0 63
#
# One PROCESS PER SHARD, selected with the RANK env var rather than rank_start/rank_count.
# That matters for provenance: the `ranks = [int(RANK)]` branch of create_dataset.py never
# touches the config, so all 64 shards record a byte-identical `config` block (rank_start=0,
# rank_count=0). Driving it with rank_start/rank_count instead would stamp each shard with the
# range that produced it, and the 8 released partitions would then disagree on fields that say
# nothing about the data. Cost of a process per shard is one torch.compile (~25 s) against
# ~140 s of solve -- worth it for identical provenance and trivial restart.
#
# RESTART IS SAFE AND IS THE POINT. create_dataset.py preallocates the memmap at full size
# before solving, so an interrupted shard leaves a complete-LOOKING .npy with no sidecar .json
# (the .json is written only after a successful flush). This script therefore treats the .json
# as the done-marker: .npy+.json -> skip; .npy without .json -> truncated, regenerate with
# force_rerun=true. Re-running the whole range is idempotent.
set -u
cd /home/david-wagner/git/torch-cfd || exit 1
PY=.venv/bin/python
CONF=2dk_spectral_w45/test_Re1000
OUT=/scratch/dwcgt/2dk_spectral_w45/test
LOGD=/scratch/dwcgt/logs/test1024
FIRST=${1:-0}; LAST=${2:-63}
mkdir -p "$LOGD"

echo "=== gen test1024 ranks $FIRST..$LAST start $(date -Is) ==="
for r in $(seq "$FIRST" "$LAST"); do
  npy="$OUT/$r-Re1000-ds-16-2852-2-64-64.npy"
  jsn="$OUT/$r-Re1000-ds-16-2852-2-64-64.json"
  extra=""
  if [ -f "$npy" ] && [ -f "$jsn" ]; then
    echo ">>> rank $r: already complete, skip"; continue
  elif [ -f "$npy" ]; then
    echo ">>> rank $r: .npy without .json (truncated) -> regenerating"; extra="force_rerun=true"
  fi
  t0=$(date +%s)
  echo ">>> rank $r start $(date -Is)"
  # no_tqdm=true to match how val_Re1000 was generated (its recorded config has no_tqdm: true).
  RANK=$r $PY scripts/create_dataset.py --config-name $CONF no_tqdm=true $extra \
      hydra.run.dir=/tmp/hg/test1024_r$r hydra.output_subdir=null \
      >> "$LOGD/shard$r.log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -ne 0 ] || [ ! -f "$jsn" ]; then
    echo "!!! rank $r FAILED rc=$rc after $((t1-t0))s -- see $LOGD/shard$r.log"; tail -5 "$LOGD/shard$r.log"
    exit 1
  fi
  echo ">>> rank $r done in $((t1-t0))s | free: $(df -h /scratch | awk 'NR==2{print $4}')"
done
echo "=== gen test1024 ranks $FIRST..$LAST DONE $(date -Is) ==="
ls "$OUT"/*.json | wc -l | xargs echo "complete shards:"
