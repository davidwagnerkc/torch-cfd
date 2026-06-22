"""Generate a Kolmogorov 2D NSE dataset shard.

Usage:
    python create_dataset.py --config-name 2dk_Re1k_test
    python create_dataset.py --config-name 2dk_Re1k_test batch_size=8   # key=value overrides

Config is a structured `DataGenConfig` registered with Hydra's ConfigStore, so the
dataclass is the single source of base defaults and each `conf/*.yaml` only overrides
the fields it cares about. Overrides on the CLI use Hydra's `key=value` grammar.

Parallel vs sequential:
    Set RANK in the environment to have this process generate exactly one shard
    (one batch of `batch_size` samples) -- launch `num_samples // batch_size`
    ranks in parallel. With RANK unset, a single process generates every shard
    sequentially. Either way the per-shard seed is `seed + rank * batch_size`,
    so give each split (train/val/test) a disjoint `seed` base.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import torch
import torch.fft as fft
import torch.nn.functional as F

from torch_cfd.finite_differences import curl_2d
from torch_cfd.forcings import KolmogorovForcing
from torch_cfd.grids import Grid
from torch_cfd.initial_conditions import filtered_velocity_field, log_normal_density
from torch_cfd.spectral import (
    NavierStokes2DSpectral,
    RK4CrankNicolsonStepper,
    stable_time_step,
    vorticity_to_velocity,
)


@dataclass
class DataGenConfig:
    """Raw, user-facing config. Derived quantities live in ResolvedConfig."""

    # what / where
    split: str = "test"              # train | val | test (used in the output path)
    dataset_name: str = "create_dataset"
    out_dir: str = "/scratch/dwcgt"

    # sampling
    num_samples: int = 16            # total samples across all ranks
    batch_size: int = 16             # samples per rank / shard
    seed: int = 0                    # base seed for this split; keep splits disjoint

    # physics
    Re: float = 1000.0
    grid_size: int = 512             # simulation grid (n x n)
    subsample: int = 8               # saved grid = grid_size // subsample
    downsample_method: str = "bilinear"  # bilinear | spectral -- both divergence-free; spectral also alias-free
    diam: str = "2*torch.pi"
    max_velocity: float = 7.0
    scale: float = 1.0
    peak_wavenumber: int = 4
    drag: float = 0.1

    # time (simulation seconds)
    time: float = 20.0               # trajectory length, recorded
    time_warmup: float = 4.5         # warmup length, discarded
    record_every_steps: int = 8      # save a snapshot every N solver steps

    # runtime
    mode: str = "generate"           # generate | dry_run | estimate
    force_rerun: bool = False
    no_cuda: bool = False
    no_tqdm: bool = False


@dataclass
class ResolvedConfig:
    """A fully resolved plan: `generate` consumes this and never recomputes shapes."""

    cfg: DataGenConfig
    device: torch.device
    diam: float
    viscosity: float
    dx: float
    dt: float                        # CFL-stable solver step
    ns: int                          # saved spatial resolution
    warmup_steps: int
    traj_steps: int
    num_snapshots: int
    snapshot_dt: float               # sim-seconds between saved snapshots (= record_every_steps * dt)
    shape: tuple                     # (batch_size, num_snapshots, 2, ns, ns)


cs = ConfigStore.instance()
cs.store(name="base_config", node=DataGenConfig)


def resolve_config(cfg: DictConfig, device: torch.device) -> ResolvedConfig:
    """Pre-run step: resolve physics + CFL dt and derive the output shape.

    This is where the previously-hardcoded snapshot interval (0.0070124836...,
    i.e. 8 * dt for the n=512/Re=1000 reference) is computed instead of baked in.
    """
    assert cfg.batch_size <= cfg.num_samples, "batch_size must be <= num_samples"
    assert cfg.num_samples % cfg.batch_size == 0, "num_samples must be divisible by batch_size"
    assert cfg.downsample_method in DOWNSAMPLERS, \
        f"downsample_method must be one of {sorted(DOWNSAMPLERS)}, got {cfg.downsample_method!r}"

    diam = eval(cfg.diam) if isinstance(cfg.diam, str) else cfg.diam
    viscosity = 1.0 / cfg.Re
    dx = diam / cfg.grid_size
    dt = stable_time_step(dx, float("inf"), cfg.max_velocity, viscosity=viscosity)

    ns = cfg.grid_size // cfg.subsample
    warmup_steps = int(cfg.time_warmup / dt)
    traj_steps = int(cfg.time / dt)
    num_snapshots = traj_steps // cfg.record_every_steps
    snapshot_dt = cfg.record_every_steps * dt
    shape = (cfg.batch_size, num_snapshots, 2, ns, ns)

    return ResolvedConfig(
        cfg=cfg,
        device=device,
        diam=diam,
        viscosity=viscosity,
        dx=dx,
        dt=dt,
        ns=ns,
        warmup_steps=warmup_steps,
        traj_steps=traj_steps,
        num_snapshots=num_snapshots,
        snapshot_dt=snapshot_dt,
        shape=shape,
    )


def shard_path(resolved: ResolvedConfig, rank: int) -> Path:
    cfg = resolved.cfg
    N, T, C, H, W = resolved.shape
    # Re is in the name so a multi-Re sweep sharing one dataset_name coexists in
    # the same split dir (identical shape across Re would otherwise collide).
    return (
        Path(cfg.out_dir)
        / cfg.dataset_name
        / cfg.split
        / f"{rank}-Re{int(cfg.Re)}-ds-{N}-{T}-{C}-{H}-{W}.npy"
    )


def shard_metadata(resolved: ResolvedConfig, rank: int) -> dict:
    """The exact config + derived plan for one shard, saved next to the .npy."""
    cfg = resolved.cfg
    return {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "resolved": {
            "rank": rank,
            "random_state": cfg.seed + rank * cfg.batch_size,
            "diam": resolved.diam,
            "viscosity": resolved.viscosity,
            "dx": resolved.dx,
            "dt": resolved.dt,
            "ns": resolved.ns,
            "warmup_steps": resolved.warmup_steps,
            "traj_steps": resolved.traj_steps,
            "num_snapshots": resolved.num_snapshots,
            "snapshot_dt": resolved.snapshot_dt,
            "shape": list(resolved.shape),
        },
    }


def _velocity_from_vorticity(omega):
    """Reconstruct velocity from a (coarse) scalar vorticity via the streamfunction:
    psi_hat = omega_hat / |k|^2, then u = d psi/dx, v = -d psi/dy. div(u) is the divergence of
    a curl == 0 *identically*, so any velocity built this way is divergence-free regardless of
    how omega was produced. Integer wavenumbers on the 2*pi box; same (channel 0 = row-paired)
    convention as vorticity_to_velocity, so it matches the existing saved-field layout."""
    h, w = omega.shape[-2:]
    kr = fft.fftfreq(h, d=1.0 / h, device=omega.device).view(1, 1, -1, 1)
    kc = fft.rfftfreq(w, d=1.0 / w, device=omega.device).view(1, 1, 1, -1)
    k2 = kr ** 2 + kc ** 2
    k2[..., 0, 0] = 1.0                       # avoid 0/0 at the mean mode (zeroed just below)
    psi = fft.rfft2(omega) / k2
    psi[..., 0, 0] = 0.0                       # mean velocity is undetermined by vorticity -> 0
    u = fft.irfft2(1j * kc * psi, s=(h, w))
    v = fft.irfft2(-1j * kr * psi, s=(h, w))
    return torch.cat([u, v], dim=1)


@torch.compile(mode="reduce-overhead")
def _downsample_bilinear(vort_hat, nse, ns):
    """Bilinear down-sampling applied to VORTICITY, then velocity reconstructed from the coarse
    vorticity. The reconstruction makes the saved field divergence-free (div of a curl == 0),
    even though bilinear still aliases the vorticity spectrum -- divergence-free and alias-free
    are independent axes (see downsample_eval.ipynb). Fast; use `spectral` if you also need a
    faithful spectrum or a super-resolution target."""
    omega = fft.irfft2(vort_hat)
    omega = F.interpolate(omega, size=(ns, ns), mode="bilinear")
    return _velocity_from_vorticity(omega)


def _truncate_to_grid(field_hat, ns):
    """rfft2 spectrum on an n x n grid -> real field on an ns x ns grid, keeping
    only |k| < ns/2. Rescaled by (ns/n)^2 for the smaller inverse-FFT norm.

    On a 2*pi box the wavenumber for a given mode index is the same integer on
    both grids, so a divergence-free spectrum stays divergence-free after truncation.
    The Nyquist band (k = +/- ns/2) is dropped: keeping only one of each conjugate
    pair would otherwise reintroduce a small divergence at the grid scale.
    """
    n = field_hat.shape[-2]
    half, nsh = ns // 2, ns // 2 + 1
    low = field_hat[..., : half + 1, :nsh]          # ky = 0 .. ns/2
    high = field_hat[..., n - half + 1 :, :nsh]     # ky = -(ns/2 - 1) .. -1
    coarse = torch.cat([low, high], dim=-2) * (ns / n) ** 2
    keep_row = (torch.arange(ns, device=coarse.device) != half).view(ns, 1)
    keep_col = (torch.arange(nsh, device=coarse.device) != half).view(1, nsh)
    return fft.irfft2(coarse * keep_row * keep_col, s=(ns, ns))


@torch.compile(mode="reduce-overhead")
def _downsample_spectral(vort_hat, nse, ns):
    """Ideal low-pass: truncate the velocity spectrum to the coarse grid's modes.
    No aliasing, and the saved field stays divergence-free (unlike bilinear)."""
    (u_hat, v_hat), _ = vorticity_to_velocity(nse.grid, vort_hat, (nse.kx, nse.ky))
    return torch.cat([_truncate_to_grid(u_hat, ns), _truncate_to_grid(v_hat, ns)], dim=1)


DOWNSAMPLERS = {"bilinear": _downsample_bilinear, "spectral": _downsample_spectral}


def build_simulation(resolved: ResolvedConfig, rank: int):
    """Shared, expensive setup: grid, forcing, compiled solver, initial vort_hat.

    Used by both `generate` and the runtime estimator so the two measure/run the
    exact same thing. Returns (nse, vort_hat) where vort_hat is pre-warmup.
    """
    cfg = resolved.cfg
    device = resolved.device
    n = cfg.grid_size
    random_state = cfg.seed + rank * cfg.batch_size

    grid = Grid(shape=(n, n), domain=((0, resolved.diam), (0, resolved.diam)), device=device)

    forcing_fn = KolmogorovForcing(
        grid=grid,
        scale=cfg.scale,
        wave_number=cfg.peak_wavenumber,
        swap_xy=False,
    )

    nse = NavierStokes2DSpectral(
        viscosity=resolved.viscosity,
        grid=grid,
        drag=cfg.drag,
        smooth=True,
        forcing_fn=forcing_fn,
        step_fn=RK4CrankNicolsonStepper(),
    ).to(device)
    nse = torch.compile(nse, mode="reduce-overhead")

    ws = []
    for i in range(cfg.batch_size):
        uv = filtered_velocity_field(
            grid,
            cfg.max_velocity,
            cfg.peak_wavenumber,
            iterations=3,  # 1 enough with fp64
            random_state=random_state + i,
            batch_size=1,
            spectral_density=log_normal_density,
            device=device,
        )
        ws.append(curl_2d(uv).data)
    vort_hat = fft.rfft2(torch.stack(ws)).to(device)
    return nse, vort_hat


@torch.inference_mode()
def generate(resolved: ResolvedConfig, rank: int):
    """Yield (snapshot_index, velocity) for one shard. Shape is dictated by `resolved`."""
    cfg = resolved.cfg
    dt = resolved.dt

    random_state = cfg.seed + rank * cfg.batch_size
    print(
        f"Rank {rank}: seeds {random_state}..{random_state + cfg.batch_size - 1} | "
        f"dt={dt:.4e} | snapshot_dt={resolved.snapshot_dt:.4e} | "
        f"{resolved.warmup_steps} warmup + {resolved.traj_steps} traj steps -> "
        f"{resolved.num_snapshots} snapshots | shape={resolved.shape}"
    )

    nse, vort_hat = build_simulation(resolved, rank)
    downsample = DOWNSAMPLERS[cfg.downsample_method]

    for _ in tqdm(range(resolved.warmup_steps), disable=cfg.no_tqdm, desc="warmup"):
        vort_hat = nse(vort_hat, dt).clone()

    for t_idx in tqdm(range(resolved.num_snapshots), disable=cfg.no_tqdm, desc="trajectory"):
        for _ in range(cfg.record_every_steps):
            vort_hat = nse(vort_hat, dt).clone()
        yield t_idx, downsample(vort_hat, nse, ns=resolved.ns)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not cfg.no_cuda) else "cpu"
    )
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.set_default_dtype(torch.float32)

    resolved = resolve_config(cfg, device)

    num_batches = cfg.num_samples // cfg.batch_size
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        ranks = [int(rank_env)]  # parallel: this process owns one shard
    else:
        ranks = list(range(num_batches))  # sequential: one process, all shards

    if cfg.mode == "dry_run":
        print("# composed config")
        print(OmegaConf.to_yaml(cfg))
        print("# resolved plan")
        print(json.dumps(shard_metadata(resolved, ranks[0])["resolved"], indent=2))
        print("# shards")
        for rank in ranks:
            seed = cfg.seed + rank * cfg.batch_size
            print(f"  rank {rank}: seeds {seed}..{seed + cfg.batch_size - 1} -> {shard_path(resolved, rank)}")
        return

    if cfg.mode == "estimate":
        from estimate import estimate_runtime  # lazy: keeps benchmark code out of the gen path

        estimate_runtime(
            resolved, ranks,
            build_simulation=build_simulation,
            spatial_downsample=DOWNSAMPLERS[cfg.downsample_method],
        )
        return

    if cfg.mode != "generate":
        raise ValueError(f"unknown mode {cfg.mode!r} (expected generate | dry_run | estimate)")

    # Fail early before any compute: never overwrite existing shards.
    targets = [shard_path(resolved, rank) for rank in ranks]
    existing = [p for p in targets if p.exists()]
    if existing and not cfg.force_rerun:
        listing = "\n".join(f"  {p}" for p in existing)
        raise SystemExit(
            f"Refusing to overwrite {len(existing)} existing shard(s):\n{listing}\n"
            f"Pass force_rerun=true to overwrite."
        )

    print(
        f"Split '{cfg.split}': {num_batches} shard(s) of {cfg.batch_size} samples | "
        f"generating rank(s) {ranks} on {device}"
    )

    for rank, path in zip(ranks, targets):
        path.parent.mkdir(parents=True, exist_ok=True)
        np_mm = np.memmap(path, mode="write", dtype=np.float32, shape=resolved.shape)
        for t_idx, u in generate(resolved, rank):
            np_mm[:, t_idx] = u.cpu().numpy()
        np_mm.flush()
        del np_mm
        # config used to generate this shard, saved alongside it
        path.with_suffix(".json").write_text(json.dumps(shard_metadata(resolved, rank), indent=2))
        print(f"Wrote shard {path}")


if __name__ == "__main__":
    main()
