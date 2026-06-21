"""Concatenate the per-rank shards written by create_dataset.py into tensors,
reading generation metadata (snapshot dt, config, Re) from the sidecar JSON each
shard saves -- nothing about the time grid is hardcoded here.

    python aggregate_dataset.py [DATASET_ROOT]   # default /scratch/dwcgt/create_dataset

One .pt is written per (split, Re): the Reynolds numbers are kept separate, e.g.
test/test_Re1000.pt, test/test_Re2000.pt, ...  Shards are streamed into a
preallocated tensor and the residual std is accumulated online, so peak RAM is
~one (split, Re)'s velocity rather than double that.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Snapshot lag used to estimate the residual std for normalization. This is a
# modeling choice (a fixed lookahead), not a generation parameter, so it stays
# here; its physical horizon is RESIDUAL_STRIDE * snapshot_dt seconds.
RESIDUAL_STRIDE = 16


def aggregate(items, snapshot_dt, out):
    """Stream a list of (path, meta) shards (all one Re) into a single .pt."""
    shapes = [tuple(m["resolved"]["shape"]) for _, m in items]
    T, C, H, W = shapes[0][1:]
    assert all(s[1:] == (T, C, H, W) for s in shapes), f"{out}: mixed shard shapes {shapes}"
    N = sum(s[0] for s in shapes)

    velocity = torch.empty((N, T, C, H, W), dtype=torch.float32)
    Re = torch.empty(N, dtype=torch.float32)
    r_sum = r_sumsq = 0.0  # online residual-std accumulators (float64)
    r_cnt = 0
    off = 0
    for (p, m), shp in zip(items, shapes):
        n = shp[0]
        arr = np.memmap(p, mode="r", dtype=np.float32, shape=shp)
        velocity[off:off + n].copy_(torch.from_numpy(arr))
        Re[off:off + n] = float(m["config"]["Re"])
        r = velocity[off:off + n, RESIDUAL_STRIDE:] - velocity[off:off + n, :-RESIDUAL_STRIDE]
        r_sum += r.sum(dtype=torch.float64).item()
        r_sumsq += r.square().sum(dtype=torch.float64).item()
        r_cnt += r.numel()
        off += n
        del arr, r

    std_residual = max(0.0, r_sumsq / r_cnt - (r_sum / r_cnt) ** 2) ** 0.5
    ds_write = {
        "velocity": velocity,
        "time": torch.arange(T, dtype=torch.float64) * snapshot_dt,
        "snapshot_dt": snapshot_dt,
        "Re": Re,                                   # (N,) constant within a per-Re .pt
        "velocity_std_residual": torch.tensor(std_residual),
        "residual_dt": RESIDUAL_STRIDE * snapshot_dt,
        "config": items[0][1]["config"],
        "shards": [m for _, m in items],            # full per-shard provenance
    }
    torch.save(ds_write, out)
    return N


def main(ds_root: Path):
    for split_dir in sorted(p for p in ds_root.iterdir() if p.is_dir()):
        paths = sorted(split_dir.glob("*.npy"))
        if not paths:
            continue
        metas = [json.loads(p.with_suffix(".json").read_text()) for p in paths]

        snapshot_dts = {m["resolved"]["snapshot_dt"] for m in metas}
        assert len(snapshot_dts) == 1, f"{split_dir}: mixed snapshot_dt {snapshot_dts}"
        snapshot_dt = snapshot_dts.pop()

        by_re = defaultdict(list)
        for p, m in zip(paths, metas):
            by_re[int(m["config"]["Re"])].append((p, m))

        for re_val in sorted(by_re):
            out = split_dir / f"{split_dir.name}_Re{re_val}.pt"
            n = aggregate(by_re[re_val], snapshot_dt, out)
            print(f"{out.name}: N={n} shape={(n, *by_re[re_val][0][1]['resolved']['shape'][1:])} "
                  f"| snapshot_dt={snapshot_dt:.6e} -> {out}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/scratch/dwcgt/create_dataset")
    main(root)
