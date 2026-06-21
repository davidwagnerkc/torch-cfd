# `2dk_re_sweep` — Kolmogorov 2D Reynolds-number sweep

Forced 2D Navier–Stokes (Kolmogorov flow) on a periodic `[0, 2π]²` domain,
simulated spectrally at `512²` (RK4 + Crank–Nicolson, dealiased) and saved at
`64²` velocity fields. A sweep over **Re ∈ {1000, 2000, 4000, 8000}**, with
`test` / `val` / `train` splits kept in separate per-Re `.pt` files.

- Solver: `NavierStokes2DSpectral` (drag 0.1, Kolmogorov forcing, peak wavenumber 4, `max_velocity` 7)
- Time step: CFL-stable `dt ≈ 8.77e-4 s`; a snapshot every 8 steps → `snapshot_dt ≈ 7.0125e-3 s`
- Saved field: `[u, v]` velocity, downsampled `512 → 64` (**bilinear**); `u` = channel 0 (row axis), `v` = channel 1 (col axis)
- Warmup 4.5 s (discarded) before recording in every split

| split | trajectory | samples / Re | snapshots (T) | shape per Re |
|-------|-----------|--------------|---------------|--------------|
| test  | 20 s | 16 | 2852 | `(16, 2852, 2, 64, 64)` |
| val   | 20 s | 16 | 2852 | `(16, 2852, 2, 64, 64)` |
| train | 55 s | 32 | 7843 | `(32, 7843, 2, 64, 64)` |

Each trajectory has a **globally unique seed** (`config seed + rank*batch_size + i`);
all 256 trajectories across the sweep were verified collision-free.

## Inputs — config group (this folder)

Hydra configs (each composes `/base_config` = `DataGenConfig`), one per `(split, Re)`:

```
scripts/conf/2dk_re_sweep
├── test_Re1k.yaml   ├── val_Re1k.yaml   ├── train_Re1k.yaml
├── test_Re2k.yaml   ├── val_Re2k.yaml   ├── train_Re2k.yaml
├── test_Re4k.yaml   ├── val_Re4k.yaml   ├── train_Re4k.yaml
└── test_Re8k.yaml   └── val_Re8k.yaml   └── train_Re8k.yaml
```

## How this dataset was produced

Single RTX 5090, **sequential** loop (no parallel queueing), run in the background
and logged to `/scratch/dwcgt/gen_sweep.log` (~29 min generate + ~1.5 min aggregate).
From the repo root:

```bash
# 1) generate all 12 shards-sets  (each --config-name runs its split/Re config)
for split in test val train; do
  for re in Re1k Re2k Re4k Re8k; do
    .venv/bin/python scripts/create_dataset.py \
      --config-name 2dk_re_sweep/${split}_${re} \
      no_tqdm=true hydra.run.dir=/tmp/hg/${split}_${re} hydra.output_subdir=null
  done
done

# 2) aggregate shards -> one .pt per (split, Re)
.venv/bin/python scripts/aggregate_dataset.py /scratch/dwcgt/2dk_re_sweep
```

Per-config helpers (any single config):

```bash
# inspect the resolved plan without running
.venv/bin/python scripts/create_dataset.py --config-name 2dk_re_sweep/train_Re8k mode=dry_run
# benchmark wall-time before committing
.venv/bin/python scripts/create_dataset.py --config-name 2dk_re_sweep/train_Re8k mode=estimate
# spectral (divergence-free) downsample instead of bilinear
.venv/bin/python scripts/create_dataset.py --config-name 2dk_re_sweep/test_Re8k downsample_method=spectral
```

`create_dataset.py` refuses to overwrite existing shards (pass `force_rerun=true`
to override). For parallel generation, set `RANK` per process (train configs are 2
shards each → `RANK=0`/`RANK=1`).

## Outputs — `/scratch/dwcgt/2dk_re_sweep/` (84 GB)

Per split: Re-tagged shards (`{rank}-Re{Re}-ds-{N}-{T}-{C}-{H}-{W}.npy`) + a JSON
sidecar holding the exact generating config, and the aggregated per-Re `.pt`.

```
/scratch/dwcgt/2dk_re_sweep
├── test
│   ├── 0-Re1000-ds-16-2852-2-64-64.{npy,json}   (1.4G + sidecar)   … Re2000/4000/8000
│   └── test_Re1000.pt … test_Re8000.pt          (1.4G each)
├── val
│   ├── 0-Re1000-ds-16-2852-2-64-64.{npy,json}   (1.4G + sidecar)   … Re2000/4000/8000
│   └── val_Re1000.pt … val_Re8000.pt            (1.4G each)
└── train
    ├── {0,1}-Re1000-ds-16-7843-2-64-64.{npy,json} (3.8G each + sidecar) … Re2000/4000/8000
    └── train_Re1000.pt … train_Re8000.pt        (7.7G each)
```

### `.pt` contents (`torch.load(..., mmap=True)`)

| key | type | notes |
|-----|------|-------|
| `velocity` | `(N, T, 2, H, W)` f32 | `[u, v]`; `u`=ch0 (row axis), `v`=ch1 (col axis) |
| `time` | `(T,)` f64 | seconds, `arange(T) * snapshot_dt` |
| `snapshot_dt` | float | seconds between snapshots (`≈7.0125e-3`) |
| `Re` | `(N,)` f32 | Reynolds per sample (constant within a per-Re file) |
| `velocity_std_residual` | scalar | std of `v[:,16:] - v[:,:-16]` (normalization) |
| `residual_dt` | float | `16 * snapshot_dt` |
| `config` | dict | representative generating config |
| `shards` | list[dict] | per-shard provenance (config + resolved plan incl. seed) |

## Calculated statistics (test split, mean over 16 samples & time)

| Re | mean KE | mean enstrophy | peak \|u\| | div/vort (bilinear) | high-k tail `E(28–32)/E(4)` |
|----|---------|----------------|-----------|---------------------|------------------------------|
| 1000 | 1.04 | 9.35 | 6.25 | 0.038 | 5.4e-4 |
| 2000 | 1.13 | 10.28 | 6.51 | 0.057 | 9.7e-4 |
| 4000 | 1.16 | 11.15 | 6.55 | 0.074 | 1.5e-3 |
| 8000 | 1.19 | 12.08 | 7.97 | 0.086 | 1.8e-3 |

Reproduce / extend the analysis with `scripts/inspect_dataset.ipynb` (point
`DATASET` at any `.pt`) or `scripts/dataset_viz.py`.

### Notes

- **Downsample = bilinear.** The saved coarse field carries ~4% (Re1000) → ~9% (Re8000)
  spurious divergence from bilinear interpolation; it grows with Re. A divergence-free
  alternative is available via `downsample_method=spectral` (spectral truncation).
- **Re8000** is stable and physical but the highest-Re corner: peak `|u|` (7.97) slightly
  exceeds the `max_velocity=7` CFL assumption (effective Courant ≈ 0.57, still < 1), and the
  finest structures approach the 64-grid scale. Usable, but it's where the bilinear artifact
  is largest — sanity-check its spectrum for your task.
