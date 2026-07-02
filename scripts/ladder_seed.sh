#!/bin/bash
# ladder_seed.sh <dataset: sub1M|25M> <seed>
# Seed replica of the base WSD budget ladder (1M -> 10M -> 25M -> 50M samples, stable-end resume chain),
# for the n>=3 seed grid: {base} x {1M,10M,25M,50M budget} x {1M-uniq, 25M-uniq}. Base only (refiner seed
# spread is ~0.04 at 10M -- base is the bottleneck; see 1M_vs_25M_RE_SWEEP.md).
# Names follow the existing convention: base1m-tr1k-<ds>-<tag>-s<seed>.
set -u
cd /home/david-wagner/git/research || exit 1
DS="$1"; SEED="$2"
case "$DS" in
  sub1M) ROOT=/scratch/dwcgt/2dk_sub1M_tr1k; PROJ=2dk_sub1M_tr1k ;;
  25M)   ROOT=/scratch/dwcgt/2dk_25M_tr1k;   PROJ=2dk_25M_tr1k ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
RUN=/scratch/dwcgt/run/simple
LOG=/scratch/dwcgt/bivort_logs/ladder_${DS}_base_s${SEED}.log
RESUME=null
echo "==== LADDER $DS base s$SEED start $(date) ====" >> "$LOG"
for bs in "1Msamp:1e6" "10Msamp:1e7" "25Msamp:2.5e7" "50Msamp:5e7"; do
  tag=${bs%%:*}; n=${bs##*:}; name=base1m-tr1k-${DS}-${tag}-s${SEED}
  echo "[$(date)] === $name  n_samples=$n  resume=$RESUME ===" >> "$LOG"
  .venv/bin/python -m research.cli.train_simple \
    name="$name" model=unet_1m data=re_sweep_bi_vort \
    data.root="$ROOT" data.train_re_values='[1000]' \
    data.re_values='[1000,2000,4000,8000]' data.batch_size=64 optim.lr=8e-4 \
    train.n_samples="$n" train.compile=true train.val_steps=20000 \
    train.resume_ckpt="$RESUME" model.refiner=false seed="$SEED" \
    wandb.project="$PROJ" >> "$LOG" 2>&1
  rc=$?
  echo "[$(date)] $name rc=$rc" >> "$LOG"
  [ $rc -ne 0 ] && { echo "!!! LADDER $DS base s$SEED FAILED at $name (rc=$rc)" >> "$LOG"; exit 1; }
  RESUME="$RUN/$name/stable-end.ckpt"
done
echo "==== LADDER $DS base s$SEED COMPLETE $(date) ====" >> "$LOG"
