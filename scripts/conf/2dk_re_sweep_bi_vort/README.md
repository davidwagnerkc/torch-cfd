# `2dk_re_sweep_bi_vort` — Kolmogorov 2D Re sweep, divergence-free bilinear

Same forced 2D Navier–Stokes (Kolmogorov flow, periodic `[0, 2π]²`, `512²` spectral, saved
`64²`) as [`2dk_re_sweep`](../2dk_re_sweep/README.md), with two changes:

1. **Divergence-free down-sample.** `bilinear` now down-samples the **vorticity** and
   reconstructs velocity from it (`_downsample_bilinear` in `create_dataset.py`), so the saved
   field is divergence-free to machine precision — the old velocity-bilinear left ~4 % (Re1000)
   → ~9 % (Re8000) spurious divergence. It is still a local filter, so the spectrum is aliased;
   `spectral` remains the alias-free option. See `scripts/downsample_eval.ipynb` for the full
   comparison against 512² DNS.
2. **Dense val Re sweep.** The `val` split now spans **Re ∈ {1000, 1250, …, 8000}** (step 250,
   29 values) for Re-generalization. `train` / `test` stay at Re {1000, 2000, 4000, 8000}.

| split | trajectory | samples / Re | Re values | shape per Re |
|-------|-----------|--------------|-----------|--------------|
| test  | 20 s | 16 | 1000, 2000, 4000, 8000 | `(16, 2852, 2, 64, 64)` |
| val   | 20 s | 16 | 1000 … 8000 step 250 (29) | `(16, 2852, 2, 64, 64)` |
| train | 55 s | 32 | 1000, 2000, 4000, 8000 | `(32, 7843, 2, 64, 64)` |

Seeds: train/test reuse the `2dk_re_sweep` seeds (identical underlying flows — only the
down-sample differs, for a clean A/B vs the aliased dataset); val uses fresh seeds
`2_000_000_000 + i*1000`. All 656 trajectories verified collision-free.

## Generate

```bash
bash scripts/gen_bi_vort.sh        # ~1 h on one RTX 5090; logs to /scratch/dwcgt/gen_bi_vort.log
```

Per-config helpers: `--config-name 2dk_re_sweep_bi_vort/val_Re1250 mode=dry_run` (resolved plan)
or `mode=estimate` (wall-time benchmark).

## Output — `/scratch/dwcgt/2dk_re_sweep_bi_vort/`

Per (split, Re): Re-tagged `.npy` shards + JSON sidecar + aggregated `.pt`. The `.pt` schema is
identical to `2dk_re_sweep` (see that README): `velocity` is `(N, T, 2, H, W)` f32, `u` = ch 0
(row axis), `v` = ch 1 (col axis) — now divergence-free.
