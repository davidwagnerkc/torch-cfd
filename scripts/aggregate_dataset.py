import torch
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    # create_dataset.sh produces shards of generated dataset in npy format. This concatenates into
    # tensor shape expected at training along with writing dict with some metadata for dataset
    for split in ("train", "val"):
        ds_dir = Path(f"/scratch/dwcgt/create_dataset/{split}/")
        shards = []
        for p in ds_dir.glob("*npy"):
            N, T, C, H, W = map(int, p.name.split("ds-")[-1].split(".npy")[0].split("-"))
            np_mm = np.memmap(
                p,
                mode="r",
                dtype=np.float32,
                shape=(N, T, C, H, W),
            )
            shards.append(torch.from_numpy(np_mm).clone())
        ds = torch.cat(shards, dim=0)
        # TODO: Assumes 0.007012483601762931 between snapshots
        residuals = ds[:, 16:] - ds[:, :-16]
        std_residual = residuals.std()
        time = torch.tensor([t_idx * 0.007012483601762931 for t_idx in range(T)], dtype=torch.float64)
        ds_write = {
            "velocity_std_residual": std_residual,
            "time": time,
            "velocity": ds,
        }
        torch.save(ds_write, ds_dir / f"{split}.pt")
