#!/bin/bash
# ladder_25m.sh <model_tag: base|refiner> <refiner: false|true>
# WSD stable-end resume chain on 2dk_25M_tr1k: 1M -> 10M -> 25M -> 50M -> 100M, 1M-param,
# lr 8e-4 / bs64, 4-Re val (Re1k in-dist + 2k/4k/8k OOD via the symlinked w45 val). n=1.
set -u
cd /home/david-wagner/git/research || exit 1
MODEL="$1"; REFINER="$2"
RUN=/scratch/dwcgt/run/simple
LOG=/scratch/dwcgt/bivort_logs/ladder_25m_${MODEL}.log
RESUME=null
echo "==== LADDER $MODEL (refiner=$REFINER) start $(date) ====" >> "$LOG"
for bs in "1Msamp:1e6" "10Msamp:1e7" "25Msamp:2.5e7" "50Msamp:5e7" "100Msamp:1e8"; do
  tag=${bs%%:*}; n=${bs##*:}; name=${MODEL}1m-tr1k-25M-${tag}
  echo "[$(date)] === $name  n_samples=$n  resume=$RESUME ===" >> "$LOG"
  uv run python -m research.cli.train_simple \
    name="$name" model=unet_1m data=re_sweep_bi_vort \
    data.root=/scratch/dwcgt/2dk_25M_tr1k data.train_re_values='[1000]' \
    data.re_values='[1000,2000,4000,8000]' data.batch_size=64 optim.lr=8e-4 \
    train.n_samples="$n" train.compile=true train.val_steps=20000 \
    train.resume_ckpt="$RESUME" model.refiner="$REFINER" seed=0 \
    wandb.project=2dk_25M_tr1k >> "$LOG" 2>&1
  rc=$?
  echo "[$(date)] $name rc=$rc" >> "$LOG"
  [ $rc -ne 0 ] && { echo "!!! LADDER $MODEL FAILED at $name (rc=$rc) -- stopping chain" >> "$LOG"; exit 1; }
  RESUME="$RUN/$name/stable-end.ckpt"
done
echo "==== LADDER $MODEL COMPLETE $(date) ====" >> "$LOG"
