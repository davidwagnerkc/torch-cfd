"""Concatenate the per-rank shards written by create_dataset.py into one tensor
per split, reading generation metadata (snapshot dt, config) from the sidecar
JSON each shard saves -- nothing about the time grid is hardcoded here anymore.

    python aggregate_dataset.py [DATASET_ROOT]   # default /scratch/dwcgt/create_dataset
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Snapshot lag used to estimate the residual std for normalization. This is a
# modeling choice (a fixed lookahead), not a generation parameter, so it stays
# here; its physical horizon is RESIDUAL_STRIDE * snapshot_dt seconds.
RESIDUAL_STRIDE = 16


def main(ds_root: Path):
    for split_dir in sorted(p for p in ds_root.iterdir() if p.is_dir()):
        shards, metas = [], []
        for p in sorted(split_dir.glob("*.npy")):
            meta = json.loads(p.with_suffix(".json").read_text())
            N, T, C, H, W = meta["resolved"]["shape"]
            np_mm = np.memmap(p, mode="r", dtype=np.float32, shape=(N, T, C, H, W))
            shards.append(torch.from_numpy(np_mm).clone())
            metas.append(meta)

        if not shards:
            continue

        # all shards in a split must share the same time grid
        snapshot_dts = {m["resolved"]["snapshot_dt"] for m in metas}
        assert len(snapshot_dts) == 1, f"{split_dir}: mixed snapshot_dt {snapshot_dts}"
        snapshot_dt = snapshot_dts.pop()

        ds = torch.cat(shards, dim=0)
        T = ds.shape[1]
        residuals = ds[:, RESIDUAL_STRIDE:] - ds[:, :-RESIDUAL_STRIDE]
        time = torch.arange(T, dtype=torch.float64) * snapshot_dt

        ds_write = {
            "velocity": ds,
            "time": time,
            "snapshot_dt": snapshot_dt,
            "velocity_std_residual": residuals.std(),
            "residual_dt": RESIDUAL_STRIDE * snapshot_dt,
            "config": metas[0]["config"],          # shared across shards
            "shards": metas,                        # full per-shard provenance
        }
        out = split_dir / f"{split_dir.name}.pt"
        torch.save(ds_write, out)
        print(f"{split_dir.name}: {tuple(ds.shape)} | snapshot_dt={snapshot_dt:.6e} -> {out}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/scratch/dwcgt/create_dataset")
    main(root)
