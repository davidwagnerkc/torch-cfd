#!/bin/bash
# gen_node.sh <rank_start> <rank_count> <node_tag>
# Generate this node's contiguous rank share of 2dk_25M_tr1k (Re1k, 45s warmup spectral, 55s,
# + 512^2 post-warmup IC). Writes a .gen_done_<tag> marker for the orchestrator to poll.
cd /home/david-wagner/git/torch-cfd || exit 1
mkdir -p /scratch/dwcgt/2dk_25M_tr1k /scratch/dwcgt/bivort_logs
.venv/bin/python scripts/create_dataset.py --config-name 2dk_25M_tr1k/train_Re1000 \
  rank_start="$1" rank_count="$2" no_tqdm=true \
  hydra.run.dir=/tmp/hg/gen_"$3" hydra.output_subdir=null \
  > /scratch/dwcgt/bivort_logs/gen25m_"$3".log 2>&1
rc=$?
echo "GEN_DONE tag=$3 rc=$rc $(date)" >> /scratch/dwcgt/bivort_logs/gen25m_"$3".log
[ $rc -eq 0 ] && touch /scratch/dwcgt/2dk_25M_tr1k/.gen_done_"$3"
