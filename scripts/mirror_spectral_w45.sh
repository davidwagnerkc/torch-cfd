#!/usr/bin/env bash
# Mirror 2dk_spectral_w45 so BOTH machines hold all 4 Re. RUN ON BETA.
# --ignore-existing: never overwrite a machine's own freshly-genned partitions; only fill gaps.
set -u; A=david-wagner@10.42.0.2; D=/scratch/dwcgt/2dk_spectral_w45/
F=(-a --ignore-existing --include='*/' --include='*.pt' --exclude='*')
echo "[mirror] pull alpha->beta $(date)"; rsync "${F[@]}" $A:$D $D
echo "[mirror] push beta->alpha $(date)"; rsync "${F[@]}" $D $A:$D
echo "[mirror] done $(date); local:"; ls $D/train/*.pt 2>/dev/null | wc -l | xargs echo "  train .pt:"
