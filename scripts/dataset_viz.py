"""Analysis + rendering helpers for inspecting a generated dataset.

The dataset is the aggregated `.pt` written by aggregate_dataset.py:
    velocity (N, T, 2, H, W) float32 on a periodic [0, 2*pi]^2 grid,
    plus time / snapshot_dt / config metadata.

Everything here is pure functions over that tensor so the notebook stays a thin
presentation layer (and so it's reusable from a script). Spectral derivatives
are used throughout because the domain is periodic, so vorticity / divergence /
energy spectrum are exact for the saved field.

Loads with mmap and only materializes the frames it needs -- the full trajectory
is ~1.4 GB, but inspection only ever touches a temporally-subsampled slice.
"""

from pathlib import Path

import numpy as np
import torch
import torch.fft as fft

TWO_PI = 2.0 * np.pi


def load(path, *, device=None):
    """mmap-load the dataset dict; velocity stays lazy until indexed."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    data["_device"] = torch.device(device)
    return data


def _kgrid(h, w, device, L=TWO_PI):
    """rfft2 wavenumbers: kx along the last axis (W), ky along H. Integers on a 2*pi box."""
    ky = TWO_PI * fft.fftfreq(h, d=L / h, device=device)
    kx = TWO_PI * fft.rfftfreq(w, d=L / w, device=device)
    return kx.view(1, 1, 1, -1), ky.view(1, 1, -1, 1)


def vorticity(u, v, L=TWO_PI):
    """omega = dv/dx - du/dy for u, v of shape (..., H, W)."""
    h, w = u.shape[-2:]
    kx, ky = _kgrid(h, w, u.device, L)
    uh, vh = fft.rfft2(u), fft.rfft2(v)
    return fft.irfft2(1j * kx * vh - 1j * ky * uh, s=(h, w))


def divergence(u, v, L=TWO_PI):
    """div = du/dx + dv/dy; ~0 for an incompressible field (downsampling adds a little)."""
    h, w = u.shape[-2:]
    kx, ky = _kgrid(h, w, u.device, L)
    uh, vh = fft.rfft2(u), fft.rfft2(v)
    return fft.irfft2(1j * kx * uh + 1j * ky * vh, s=(h, w))


def energy_spectrum(u, v, L=TWO_PI):
    """Radially-averaged kinetic-energy spectrum E(k). Sums over the leading dims.

    Normalized so that sum_k E(k) == mean-over-space 1/2 (u^2 + v^2) (Parseval).
    Returns (k, E) as 1-D numpy arrays for k = 0 .. min(H, W)//2.
    """
    h, w = u.shape[-2:]
    uh, vh = fft.fft2(u), fft.fft2(v)
    power = 0.5 * (uh.abs() ** 2 + vh.abs() ** 2) / (h * w) ** 2
    power = power.reshape(-1, h, w).mean(0)  # average over samples/frames

    ky = TWO_PI * fft.fftfreq(h, d=L / h, device=u.device)
    kx = TWO_PI * fft.fftfreq(w, d=L / w, device=u.device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    kbin = torch.sqrt(KX**2 + KY**2).round().long()

    kmax = min(h, w) // 2
    E = torch.zeros(kmax + 1, device=u.device)
    E.index_add_(0, kbin.flatten().clamp(max=kmax), power.flatten())
    return np.arange(kmax + 1), E.cpu().numpy()


def frame_indices(n_steps, snapshot_dt, frame_seconds):
    """Frame indices spaced ~frame_seconds apart in sim-time (keeps the gif short)."""
    stride = max(1, round(frame_seconds / snapshot_dt))
    return np.arange(0, n_steps, stride)


def _uv(velocity, idx, device):
    """Materialize velocity[:, idx] as (N, F, 2, H, W) float32 on device."""
    uv = velocity[:, torch.as_tensor(idx)].to(device=device, dtype=torch.float32)
    return uv[:, :, 0], uv[:, :, 1]  # u, v -> (N, F, H, W)


def vorticity_frames(data, frame_seconds=1.0):
    """(omega, times) for the gif: omega (N, F, H, W) numpy, times in seconds."""
    velocity, device = data["velocity"], data["_device"]
    snapshot_dt = float(data["snapshot_dt"])
    idx = frame_indices(velocity.shape[1], snapshot_dt, frame_seconds)
    with torch.inference_mode():
        u, v = _uv(velocity, idx, device)
        w = vorticity(u, v).cpu().numpy()
    return w, idx * snapshot_dt


def stats(data, max_points=320, chunk=64):
    """Per-time bulk diagnostics, averaged over the 16 samples and space.

    Returns a dict of 1-D numpy arrays vs `t` (seconds): kinetic energy, enstrophy,
    peak speed (CFL-relevant), and RMS divergence (incompressibility check).
    """
    velocity, device = data["velocity"], data["_device"]
    n, T = velocity.shape[0], velocity.shape[1]
    snapshot_dt = float(data["snapshot_dt"])
    idx = np.arange(0, T, max(1, T // max_points))

    ke, ens, peak, div_rms = [], [], [], []
    with torch.inference_mode():
        for c in range(0, len(idx), chunk):
            u, v = _uv(velocity, idx[c : c + chunk], device)
            sp2 = u * u + v * v
            w = vorticity(u, v)
            d = divergence(u, v)
            ke.append((0.5 * sp2.mean((0, 2, 3))).cpu().numpy())
            ens.append((0.5 * (w * w).mean((0, 2, 3))).cpu().numpy())
            peak.append(sp2.amax((0, 2, 3)).sqrt().cpu().numpy())
            div_rms.append((d * d).mean((0, 2, 3)).sqrt().cpu().numpy())
    return {
        "t": idx * snapshot_dt,
        "kinetic_energy": np.concatenate(ke),
        "enstrophy": np.concatenate(ens),
        "peak_speed": np.concatenate(peak),
        "divergence_rms": np.concatenate(div_rms),
        "n_samples": n,
    }


def mean_spectrum(data, last_seconds=5.0):
    """E(k) averaged over all samples and the final `last_seconds` of the trajectory."""
    velocity, device = data["velocity"], data["_device"]
    snapshot_dt = float(data["snapshot_dt"])
    T = velocity.shape[1]
    n_last = max(1, round(last_seconds / snapshot_dt))
    idx = np.arange(max(0, T - n_last), T, max(1, n_last // 40))  # ~40 frames
    with torch.inference_mode():
        u, v = _uv(velocity, idx, device)
        return energy_spectrum(u, v)


# ---------------------------------------------------------------------------
# rendering (matplotlib); imported lazily so the analysis fns stay dependency-light
# ---------------------------------------------------------------------------

def _cmap():
    try:
        import seaborn as sns
        return sns.cm.icefire  # repo convention
    except Exception:
        return "RdBu_r"


def make_gif(omega, times, out_path, fps=5, max_samples=16):
    """4x4 grid of vorticity animated over `times`; writes a gif, returns its path."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = min(max_samples, omega.shape[0])
    omega = omega[:n]
    F = omega.shape[1]
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    vmax = float(np.percentile(np.abs(omega), 99))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.0, nrow * 2.0))
    axes = np.atleast_1d(axes).ravel()
    ims = []
    for i, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        if i < n:
            ims.append(ax.imshow(omega[i, 0], cmap=_cmap(), vmin=-vmax, vmax=vmax,
                                 interpolation="bilinear", animated=True))
            ax.set_title(f"#{i}", fontsize=8)
        else:
            ax.axis("off")
    title = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def update(f):
        for i, im in enumerate(ims):
            im.set_data(omega[i, f])
        title.set_text(f"vorticity   t = {times[f]:5.2f} s   (frame {f + 1}/{F})")
        return ims

    anim = FuncAnimation(fig, update, frames=F, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def plot_stats(s, out_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    panels = [
        ("kinetic_energy", "kinetic energy  ½⟨u²+v²⟩", False),
        ("enstrophy", "enstrophy  ½⟨ω²⟩", False),
        ("peak_speed", "peak speed  max|u|  (CFL)", False),
        ("divergence_rms", "RMS divergence ∇·u  (bilinear-downsample artifact)", True),
    ]
    for ax, (key, label, logy) in zip(axes.ravel(), panels):
        ax.plot(s["t"], s[key], lw=1.2)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("t (s)")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, ls="--", lw=0.4, alpha=0.6)
    fig.suptitle(f"Physics over time (mean of {s['n_samples']} samples)", y=1.0)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    return fig


def plot_spectrum(k, E, out_path=None, ref_slope=-3.0):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    m = (k >= 1) & (E > 0)
    ax.loglog(k[m], E[m], lw=1.4, label="E(k)")
    kk = k[m]
    if len(kk):
        ref = E[m][len(kk) // 4] * (kk / kk[len(kk) // 4]) ** ref_slope
        ax.loglog(kk, ref, "k--", lw=0.8, label=f"$k^{{{ref_slope:.0f}}}$ ref")
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("Kinetic energy spectrum")
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    return fig


def summary(data, s):
    """Compact text summary (config + headline stats)."""
    cfg = data.get("config", {})
    v = data["velocity"]
    return {
        "shape (N,T,C,H,W)": tuple(v.shape),
        "snapshot_dt (s)": round(float(data["snapshot_dt"]), 6),
        "duration (s)": round(float(data["time"][-1]), 3),
        "Re": cfg.get("Re"),
        "grid_size (sim)": cfg.get("grid_size"),
        "saved grid": f'{v.shape[-2]}x{v.shape[-1]}',
        "seed": cfg.get("seed"),
        "mean kinetic energy": round(float(s["kinetic_energy"].mean()), 4),
        "mean enstrophy": round(float(s["enstrophy"].mean()), 4),
        "peak speed (max)": round(float(s["peak_speed"].max()), 4),
        "divergence RMS (median)": round(float(np.median(s["divergence_rms"])), 4),
        # ~1 means the saved field is far from incompressible (bilinear downsample)
        "divergence/vorticity": round(
            float(np.median(s["divergence_rms"]) / np.sqrt(2 * np.median(s["enstrophy"]))), 3
        ),
    }
