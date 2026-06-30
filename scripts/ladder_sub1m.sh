#!/bin/bash
# ladder_sub1m.sh <model_tag: base|refiner> <refiner: false|true>
# Same WSD budget sweep as ladder_25m.sh, but on a 1M-UNIQUE subset (1 partition = 128 traj = 1,001,856
# windows) of 2dk_25M_tr1k. The repetition study's LOW-diversity arm: budgets 1M..100M = 1..100 epochs on
# 1M unique. Pairs with the 25M-unique ladder at matched budgets to isolate the effect of data diversity.
set -u
cd /home/david-wagner/git/research || exit 1
export PATH="$HOME/.local/bin:$PATH"   # uv lives in ~/.local/bin (non-login ssh shell drops it -> rc=127 on beta)
MODEL="$1"; REFINER="$2"
RUN=/scratch/dwcgt/run/simple
LOG=/scratch/dwcgt/bivort_logs/ladder_sub1m_${MODEL}.log
RESUME=null
echo "==== LADDER sub1M $MODEL (refiner=$REFINER) start $(date) ====" >> "$LOG"
for bs in "1Msamp:1e6" "10Msamp:1e7" "25Msamp:2.5e7" "50Msamp:5e7" "100Msamp:1e8"; do
  tag=${bs%%:*}; n=${bs##*:}; name=${MODEL}1m-tr1k-sub1M-${tag}
  echo "[$(date)] === $name  n_samples=$n  resume=$RESUME ===" >> "$LOG"
  uv run python -m research.cli.train_simple \
    name="$name" model=unet_1m data=re_sweep_bi_vort \
    data.root=/scratch/dwcgt/2dk_sub1M_tr1k data.train_re_values='[1000]' \
    data.re_values='[1000,2000,4000,8000]' data.batch_size=64 optim.lr=8e-4 \
    train.n_samples="$n" train.compile=true train.val_steps=20000 \
    train.resume_ckpt="$RESUME" model.refiner="$REFINER" seed=0 \
    wandb.project=2dk_sub1M_tr1k >> "$LOG" 2>&1
  rc=$?
  echo "[$(date)] $name rc=$rc" >> "$LOG"
  [ $rc -ne 0 ] && { echo "!!! LADDER sub1M $MODEL FAILED at $name (rc=$rc) -- stopping chain" >> "$LOG"; exit 1; }
  RESUME="$RUN/$name/stable-end.ckpt"
done
echo "==== LADDER sub1M $MODEL COMPLETE $(date) ====" >> "$LOG"
