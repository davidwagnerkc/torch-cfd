#!/bin/bash
# ladder_vort.sh <seed>
# VORTICITY experiment (research/VORTICITY.md): omega-output 1M UNet on the sub1M omega twin root.
# WSD budget ladder 1M -> 10M ONLY (pause for analysis before 25M/50M). Canonical recipe otherwise.
set -u
cd /home/david-wagner/git/research || exit 1
SEED="$1"
RUN=/scratch/dwcgt/run/simple
LOG=/scratch/dwcgt/bivort_logs/ladder_vort_s${SEED}.log
RESUME=null
echo "==== LADDER vort sub1M s$SEED start $(date) ====" >> "$LOG"
for bs in "1Msamp:1e6" "10Msamp:1e7"; do
  tag=${bs%%:*}; n=${bs##*:}; name=base1m-vort-tr1k-sub1M-${tag}-s${SEED}
  echo "[$(date)] === $name  n_samples=$n  resume=$RESUME ===" >> "$LOG"
  .venv/bin/python -m research.cli.train_simple \
    name="$name" model=unet_1m data=re_sweep_bi_vort \
    data.root=/scratch/dwcgt/2dk_sub1M_tr1k_vort +data.field=vorticity \
    model.unet.in_channels=1 model.unet.out_channels=1 \
    data.train_re_values='[1000]' data.re_values='[1000,2000,4000,8000]' \
    data.batch_size=64 optim.lr=8e-4 train.n_samples="$n" train.compile=true \
    train.val_steps=20000 train.resume_ckpt="$RESUME" seed="$SEED" \
    wandb.project=2dk_sub1M_tr1k >> "$LOG" 2>&1
  rc=$?
  echo "[$(date)] $name rc=$rc" >> "$LOG"
  [ $rc -ne 0 ] && { echo "!!! LADDER vort s$SEED FAILED at $name (rc=$rc)" >> "$LOG"; exit 1; }
  RESUME="$RUN/$name/stable-end.ckpt"
done
echo "==== LADDER vort s$SEED COMPLETE $(date) ====" >> "$LOG"
