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
import math
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
    rank_start: int = 0              # split-gen: first rank this process owns (shards named by rank -> disjoint per node)
    rank_count: int = 0              # split-gen: number of ranks to generate here (0 = all num_samples//batch_size)

    # physics
    Re: float = 1000.0
    grid_size: int = 512             # simulation grid (n x n)
    subsample: int = 8               # saved grid = grid_size // subsample
    downsample_method: str = "bilinear"  # bilinear | spectral | box | gaussian | macface[_collocated]
    # all but macface* are divergence-free; only spectral is alias-free.
    # box/gaussian/spectral = the three canonical LES filters (top-hat / Gaussian / sharp cutoff).
    # MULTI-WRITER (separate code path, see generate_multi/_write_multi). When non-empty, ONE
    # 512^2 solve feeds every listed operator and each gets its own dataset dir
    # "<dataset_name>_<method>". The downsample costs 0.2% of a shard (measured), so this is a
    # ~4x saving over running the operators as separate jobs -- and it guarantees they see
    # literally the same field rather than relying on solver determinism.
    # Empty (default) -> the original single-writer path, unchanged.
    downsample_methods: tuple = ()
    diam: str = "2*torch.pi"
    max_velocity: float = 7.0
    scale: float = 1.0
    peak_wavenumber: int = 4
    drag: float = 0.1

    # time (simulation seconds)
    time: float = 20.0               # trajectory length, recorded
    time_warmup: float = 4.5         # warmup length, discarded
    record_every_steps: int = 8      # save a snapshot every N solver steps
    save_ic512: bool = True          # also save the post-warmup full-res (grid^2) velocity IC per shard (-> train/ic512/)

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


def _downsample_box(vort_hat, nse, ns):
    """Top-hat (box) filter: r x r block average of VORTICITY, then velocity reconstructed from
    the coarse vorticity (so the saved field is divergence-free, like `bilinear`/`spectral`).

    One of the three canonical LES filters (sharp spectral cutoff / Gaussian / top-hat; Pope
    ch. 13). Also exactly the finite-volume cell average and what most CV pipelines mean by
    "downsample". ISOTROPIC -- it averages equally on both axes -- which is the contrast with
    `macface`, whose box average acts on the TANGENTIAL axis only and decimates the normal one.
    box-vs-macface at matched attenuation therefore isolates ANISOTROPY.

    Block-average-then-decimate is exactly "box filter then subsample", so this aliases (unlike
    `spectral`); that is the point of having it.
    """
    omega = fft.irfft2(vort_hat)
    n = omega.shape[-1]
    r = n // ns
    omega = omega.reshape(*omega.shape[:-2], ns, r, ns, r).mean(dim=(-3, -1))
    return _velocity_from_vorticity(omega)


def _downsample_gaussian(vort_hat, nse, ns):
    """Gaussian filter then decimate, on VORTICITY, velocity reconstructed as above.

    The second of the canonical LES filters. Width is set by the standard second-moment match
    to a box filter of the same nominal width Delta = r cells: sigma = r / sqrt(12). That makes
    `gaussian` and `box` directly comparable -- same nominal filter width, different kernel
    shape (smooth roll-off vs sinc ringing) -- rather than introducing a free parameter.

    Applied in Fourier on the FINE grid: for sigma in units of grid cells on an N-point periodic
    domain the transfer is exp(-2 pi^2 sigma^2 |k|^2 / N^2). Decimation afterwards still folds
    whatever survives above the coarse Nyquist, so this aliases too -- less than box, more than
    spectral.
    """
    n = vort_hat.shape[-2]
    r = n // ns
    sigma = r / math.sqrt(12.0)
    kr = fft.fftfreq(n, d=1.0 / n, device=vort_hat.device).view(1, 1, -1, 1)
    kc = fft.rfftfreq(n, d=1.0 / n, device=vort_hat.device).view(1, 1, 1, -1)
    H = torch.exp(-2.0 * math.pi ** 2 * sigma ** 2 * (kr ** 2 + kc ** 2) / n ** 2)
    omega = fft.irfft2(vort_hat * H, s=(n, n))
    omega = omega[..., ::r, ::r]
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


def _macface(u, factor):
    """jax-cfd `resize.downsample_staggered_velocity_component`, per velocity component.

    Component 0 (u) has normal axis -2 (= x, our convention), component 1 (v) normal axis -1.
    Decimate along the component's OWN (normal) axis at offset factor-1 with NO filter, then
    block-average `factor` cells along the tangential axis. Its stated purpose upstream is
    preserving the discrete MAC divergence, not anti-aliasing.
    """
    out = []
    for j in (0, 1):
        x = u[..., j:j + 1, :, :]
        dax, tax = (-2, -1) if j == 0 else (-1, -2)
        x = x.index_select(dax, torch.arange(factor - 1, x.shape[dax], factor, device=x.device))
        x = x.unfold(tax, factor, factor).mean(-1)
        out.append(x)
    return torch.cat(out, dim=-3)


def _to_mac_faces(u_hat, v_hat, n):
    """Half-cell spectral shift of a COLLOCATED velocity onto MAC faces: u -> +dx/2 in x,
    v -> +dy/2 in y. On a 2*pi box with integer wavenumbers a shift of +dx/2 = +pi/n is the
    phase exp(i*pi*k/n). Exact, and it makes `_macface` preserve the MAC discrete divergence
    the way it does in the jax-cfd pipeline."""
    kr = fft.fftfreq(n, d=1.0 / n, device=u_hat.device).view(1, 1, -1, 1)
    kc = fft.rfftfreq(n, d=1.0 / n, device=u_hat.device).view(1, 1, 1, -1)
    return u_hat * torch.exp(1j * torch.pi * kr / n), v_hat * torch.exp(1j * torch.pi * kc / n)


def _downsample_macface(vort_hat, nse, ns, stagger=True):
    """Reproduce the jax-cfd / TSM-PDE FVM coarse-graining ON SPECTRAL DATA.

    This is the parameter-free causal test of SOLVER_DIFFICULTY.md: the FVM 64^2 benchmark
    differs from ours because of the coarse-graining OPERATOR, so coarsening the same spectral
    flow this way should move its difficulty toward the FVM benchmark's.

    In closed form the operator is an ANISOTROPIC tangential low-pass -- a non-folding mode is
    attenuated by the Dirichlet kernel sin(pi k_t r/N)/(r sin(pi k_t/N)) in the TANGENTIAL
    wavenumber only -- plus ~19% aliased content in the near-Nyquist band.

    ⚠ The resulting field is NOT spectrally divergence-free (that is the point: neither is the
    FVM target). Score models trained on it RAW, or with proj_convention=mac -- the spectral
    Leray projection is invalid here, exactly as for `original_dataset` (V2_RESULTS §fvm).
    """
    (u_hat, v_hat), _ = vorticity_to_velocity(nse.grid, vort_hat, (nse.kx, nse.ky))
    n = vort_hat.shape[-2]
    if stagger:
        u_hat, v_hat = _to_mac_faces(u_hat, v_hat, n)
    u = torch.cat([fft.irfft2(u_hat, s=(n, n)), fft.irfft2(v_hat, s=(n, n))], dim=1)
    return _macface(u, n // ns)


def _downsample_macface_collocated(vort_hat, nse, ns):
    return _downsample_macface(vort_hat, nse, ns, stagger=False)


DOWNSAMPLERS = {"bilinear": _downsample_bilinear, "spectral": _downsample_spectral,
                "macface": _downsample_macface, "macface_collocated": _downsample_macface_collocated,
                "box": _downsample_box, "gaussian": _downsample_gaussian}


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

    if cfg.save_ic512:  # post-warmup stationary state at full (grid^2) resolution -> restart/extend without re-warming
        ic = downsample(vort_hat, nse, ns=cfg.grid_size).detach().cpu().numpy()  # (batch, 2, grid, grid) velocity
        ic_dir = shard_path(resolved, rank).parent / "ic512"
        ic_dir.mkdir(parents=True, exist_ok=True)
        ic_path = ic_dir / f"{rank}-Re{int(cfg.Re)}-ic512-{cfg.batch_size}-2-{cfg.grid_size}-{cfg.grid_size}.npy"
        np.save(ic_path, ic)
        print(f"  saved post-warmup {cfg.grid_size}^2 velocity IC -> {ic_path.name}", flush=True)

    for t_idx in tqdm(range(resolved.num_snapshots), disable=cfg.no_tqdm, desc="trajectory"):
        for _ in range(cfg.record_every_steps):
            vort_hat = nse(vort_hat, dt).clone()
        yield t_idx, downsample(vort_hat, nse, ns=resolved.ns)


def generate_multi(resolved: ResolvedConfig, rank: int, methods):
    """Like `generate`, but yields (snapshot_index, {method: velocity}) from ONE solve.

    Deliberately a separate function from `generate` rather than a generalisation of it: the
    single-operator path is what produced every existing corpus, and it stays byte-identical.
    """
    cfg = resolved.cfg
    dt = resolved.dt
    random_state = cfg.seed + rank * cfg.batch_size
    print(
        f"Rank {rank}: seeds {random_state}..{random_state + cfg.batch_size - 1} | "
        f"dt={dt:.4e} | snapshot_dt={resolved.snapshot_dt:.4e} | "
        f"{resolved.warmup_steps} warmup + {resolved.num_snapshots * cfg.record_every_steps} "
        f"traj steps -> {resolved.num_snapshots} snapshots | shape={resolved.shape} | "
        f"methods={list(methods)}",
        flush=True,
    )
    nse, vort_hat = build_simulation(resolved, rank)
    fns = {m: DOWNSAMPLERS[m] for m in methods}

    for _ in tqdm(range(resolved.warmup_steps), disable=cfg.no_tqdm, desc="warmup"):
        vort_hat = nse(vort_hat, dt).clone()

    if cfg.save_ic512:   # one shared post-warmup 512^2 IC (identical for every method)
        torch.compiler.cudagraph_mark_step_begin()
        ic = DOWNSAMPLERS["spectral"](vort_hat, nse, ns=cfg.grid_size).clone().detach().cpu().numpy()
        for m in methods:
            ic_dir = Path(cfg.out_dir) / f"{cfg.dataset_name}_{m}" / cfg.split / "ic512"
            ic_dir.mkdir(parents=True, exist_ok=True)
            np.save(ic_dir / f"{rank}-Re{int(cfg.Re)}-ic512-{cfg.batch_size}-2-"
                             f"{cfg.grid_size}-{cfg.grid_size}.npy", ic)

    for t_idx in tqdm(range(resolved.num_snapshots), disable=cfg.no_tqdm, desc="trajectory"):
        for _ in range(cfg.record_every_steps):
            vort_hat = nse(vort_hat, dt).clone()
        # .clone() is REQUIRED: _downsample_spectral is torch.compile(reduce-overhead), whose
        # CUDA-graph output buffer is reused on the next invocation. Holding several operators'
        # outputs live at once (and calling spectral at two different ns) otherwise raises
        # "accessing tensor output of CUDAGraphs that has been overwritten".
        torch.compiler.cudagraph_mark_step_begin()
        yield t_idx, {m: fn(vort_hat, nse, ns=resolved.ns).clone() for m, fn in fns.items()}


def _multi_shard_path(resolved: ResolvedConfig, rank: int, method: str) -> Path:
    cfg = resolved.cfg
    N, T, C, H, W = resolved.shape
    return (Path(cfg.out_dir) / f"{cfg.dataset_name}_{method}" / cfg.split
            / f"{rank}-Re{int(cfg.Re)}-ds-{N}-{T}-{C}-{H}-{W}.npy")


def _write_multi(resolved: ResolvedConfig, ranks, methods, force_rerun: bool):
    """Multi-operator writer. One memmap per (rank, method); all fed from one solve."""
    cfg = resolved.cfg
    targets = {(r, m): _multi_shard_path(resolved, r, m) for r in ranks for m in methods}
    existing = [p for p in targets.values() if p.exists()]
    if existing and not force_rerun:
        listing = "\n".join(f"  {p}" for p in existing)
        raise SystemExit(f"Refusing to overwrite {len(existing)} existing shard(s):\n{listing}\n"
                         f"Pass force_rerun=true to overwrite.")
    for rank in ranks:
        mms = {}
        for m in methods:
            path = targets[(rank, m)]
            path.parent.mkdir(parents=True, exist_ok=True)
            mms[m] = np.memmap(path, mode="write", dtype=np.float32, shape=resolved.shape)
        for t_idx, us in generate_multi(resolved, rank, methods):
            for m, u in us.items():
                mms[m][:, t_idx] = u.cpu().numpy()
        for m in methods:
            mms[m].flush()
            del mms[m]
            meta = shard_metadata(resolved, rank)
            meta["config"]["downsample_method"] = m       # record the ACTUAL operator per shard
            meta["multi_writer"] = {"methods": list(methods), "shared_solve": True}
            targets[(rank, m)].with_suffix(".json").write_text(json.dumps(meta, indent=2))
            print(f"Wrote shard {targets[(rank, m)]}", flush=True)


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
    elif cfg.rank_count:
        ranks = list(range(cfg.rank_start, cfg.rank_start + cfg.rank_count))  # split-gen: a contiguous rank subset (this node's share)
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

    methods = tuple(cfg.get("downsample_methods", ()) or ())
    if methods:
        for m in methods:
            assert m in DOWNSAMPLERS, f"unknown downsample method {m!r}"
        _write_multi(resolved, ranks, methods, cfg.force_rerun)
        return

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
