from dataclasses import dataclass
from datetime import datetime
import os

from tqdm import tqdm

import torch
import torch.fft as fft
import torch.nn.functional as F
from torch_cfd.finite_differences import curl_2d
from torch_cfd.forcings import KolmogorovForcing

from torch_cfd.grids import Grid
from torch_cfd.initial_conditions import filtered_velocity_field, log_normal_density
from torch_cfd.spectral import (
    vorticity_to_velocity,
    stable_time_step, 
    NavierStokes2DSpectral,
    RK4CrankNicolsonStepper,
)
from fno.data_gen.trajectories import get_trajectory_imex
from torch_cfd import boundaries, grids

import torch
import numpy as np
from time import perf_counter as pc
from pathlib import Path

# TODO: Maybe bring this back at some point
@dataclass()
class DataGenConfig:
    num_samples: int = 16
    batch_size: int = 16
    grid_size: int = 256
    subsample: int = 4
    visc: float = 1e-3
    dt: float = 1e-3
    stable_dt: float = None
    time: float = 10
    time_warmup: float = 4.5
    num_steps: int = 100
    diam: str = "2*torch.pi"
    Re: int = None
    scale: float = 1.0
    seed: int = 42
    peak_wavenumber: int = 4
    force_rerun: bool = False
    max_velocity: int = 5
    double: bool = False
    filename: str = None
    no_cuda: bool = False
    no_tqdm: bool = False
    demo: bool = False


@torch.compile(mode="reduce-overhead")
def spatial_downsample(vort_hat, nse, ns):
    (u_hat, v_hat), _ = vorticity_to_velocity(nse.grid, vort_hat, (nse.kx, nse.ky))
    u, v = fft.irfft2(u_hat), fft.irfft2(v_hat)
    velocity = torch.cat([u, v], dim=1)
    velocity = F.interpolate(velocity, size=(ns, ns), mode="bilinear")
    return velocity

@torch.inference_mode()
def generate(batch_size, n, T, T_warmup, rank, device):
    # batch_size = 8

    # NSE
    Re = 1000 
    viscosity = 1 / Re
    scale = 1 
    peak_wavenumber = 4 
    max_velocity = 7 

    # Space Time
    # n = 512
    diam = 2 * torch.pi 
    dt = float("inf")
    dx = diam / n
    dt = stable_time_step(dx, dt, max_velocity, viscosity=viscosity)
    # T = 80 
    traj_steps = int(T / dt)
    # T_warmup = 41 
    warmup_steps = int(T_warmup / dt)
    print(f"Using stable dt={dt:.4e}, total steps={warmup_steps + traj_steps}")

    # Downsampling
    ns = 64
    record_every_iters = int(0.007012483601762931 / dt)

    random_state = 0 + rank * batch_size
    print(f"Rank {rank} using random seed {random_state}")
    print(f"Will generate {warmup_steps} warmup steps ({T_warmup}s) and {traj_steps} trajectory steps ({T}s), saving snapshots every {record_every_iters} iters for {traj_steps // record_every_iters} total in trajectory")

    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    dtype = torch.float32
    cdtype = torch.complex64
    torch.set_default_dtype(dtype)
    
    grid = Grid(shape=(n, n), domain=((0, diam), (0, diam)), device=device)
    
    forcing_fn = KolmogorovForcing(
        grid=grid,
        scale=scale,
        wave_number=peak_wavenumber,
        swap_xy=False,
    )
    
    nse = NavierStokes2DSpectral(
        viscosity=viscosity,
        grid=grid,
        drag=0.1,
        smooth=True,
        forcing_fn=forcing_fn,
        step_fn=RK4CrankNicolsonStepper(),
    ).to(device)
    nse = torch.compile(nse, mode="reduce-overhead")
    
    ws = []
    for i in range(batch_size):
        print(f"Generating ic with seed {random_state + i}")
        uv = filtered_velocity_field(
            grid, 
            max_velocity, 
            peak_wavenumber,
            iterations=3,  # 1 enough with fp64
            random_state=random_state + i,
            batch_size=1, 
            spectral_density=log_normal_density,
            device=device,
        )
        w = curl_2d(uv)
        ws.append(w.data)
    vort_init = torch.stack(ws)
    vort_hat = fft.rfft2(vort_init).to(device)

    total_steps = 0
    for _ in tqdm(range(warmup_steps)):
        vort_hat = nse(vort_hat, dt).clone()
        total_steps += 1

    for _ in tqdm(range(traj_steps // record_every_iters)):
        for _ in range(record_every_iters):
            vort_hat = nse(vort_hat, dt).clone()
            total_steps += 1
        u = spatial_downsample(vort_hat, nse, ns)
        yield u, total_steps

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ.get("RANK", 0))
    ds_name = "testing"
    # ds_split = "train"
    ds_split = "val"
    batch_size = 8
    n = 512  # 512
    # traj_time = 80
    traj_time = 20
    warmup_time = 0
    T = int(traj_time / 0.007012483601762931)
    N, T, C, H, W = batch_size, T, 2, 64, 64
    mmap_path = f"/scratch/dwcgt/{ds_name}/{ds_split}/{rank}-ds-{N}-{T}-{C}-{H}-{W}.npy"
    if Path(mmap_path).exists():
        print(f"Dataset at {mmap_path} already exists, skipping generation")
        exit(0)
    Path(mmap_path).parent.mkdir(parents=True, exist_ok=True)
    np_mm = np.memmap(mmap_path, mode="write", dtype=np.float32, shape=(N, T, C, H, W))
    for t_idx, (u, step_idx) in enumerate(generate(batch_size, n, traj_time, warmup_time, rank, device)):
        np_mm[:, t_idx, :, :, :] = u.cpu().numpy()
    np_mm.flush()
    del np_mm
