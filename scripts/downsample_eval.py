"""Data-only fidelity analysis of coarse-graining (down-sample) methods.

Generates a full-resolution (512^2) Kolmogorov DNS trajectory with the project's own
solver, then compares several 512 -> 64 down-sampling methods against that DNS as
ground truth. No model training is involved -- every metric is a pure function of the
fields, reusing `dataset_viz` for spectra / divergence / vorticity.

Methods compared (all 512 -> 64):
    velocity_bilinear         current default: F.interpolate on velocity (aliases, NOT div-free)
    vorticity_bilinear_recon  bilinear on vorticity, then reconstruct velocity (aliases, div-free)
    velocity_spectral         ideal sharp low-pass on velocity = _downsample_spectral (exact, div-free)
    vorticity_spectral_recon  sharp low-pass on vorticity, then reconstruct (== velocity_spectral)
    velocity_smooth_spectral  exponential roll-off then truncate (no alias, no ring, attenuates)

`velocity_spectral` is the orthogonal projection P_ns(W) -- the L2-optimal band-limited
field -- so it also serves as the reference for resolved-band field error (its error is
0 by construction, which is the correct statement that it is L2-optimal).
"""

import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F

import dataset_viz as dv
from create_dataset import (
    DataGenConfig,
    resolve_config,
    build_simulation,
    _truncate_to_grid,
    DOWNSAMPLERS,
)
from torch_cfd.spectral import vorticity_to_velocity

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# reconstruction + down-samplers
# ---------------------------------------------------------------------------

def velocity_from_vorticity(omega, L=TWO_PI):
    """Velocity from scalar vorticity via the streamfunction, in dataset_viz's k-convention.
    div(u) is the divergence of a curl == 0 *identically* for any omega -- this is what
    makes any 'reconstruct' path divergence-free regardless of how omega was down-sampled."""
    h, w = omega.shape[-2:]
    kr, kc = dv._kgrid(h, w, omega.device, L)
    oh = fft.rfft2(omega)
    k2 = (kr ** 2 + kc ** 2).clone()
    k2[..., 0, 0] = 1.0
    psi = oh / k2
    psi[..., 0, 0] = 0.0
    return fft.irfft2(1j * kc * psi, s=(h, w)), fft.irfft2(-1j * kr * psi, s=(h, w))


def _smooth_spectral(u_hat, v_hat, ns, n, lo=0.5, hi=1.0):
    """Raised-cosine (Hann) taper over [lo, hi] * Nyquist of the velocity spectrum, then truncate.
    A scalar k-filter keeps k.u_hat = 0, so the field stays divergence-free. Preserves the low band
    fully and smoothly rolls off the top: no aliasing and no Gibbs ring, at the cost of attenuating
    the upper resolved modes -- one illustrative LES-style smooth filter (the taper is tunable)."""
    kr, kc = dv._kgrid(n, n, u_hat.device)
    kmag = torch.sqrt(kr ** 2 + kc ** 2)
    klo, khi = lo * ns / 2, hi * ns / 2
    t = ((kmag - klo) / (khi - klo)).clamp(0.0, 1.0)
    g = 0.5 * (1.0 + torch.cos(np.pi * t))
    return _truncate_to_grid(u_hat * g, ns), _truncate_to_grid(v_hat * g, ns)


def _split(field):
    """(B, 2, H, W) cat([u, v]) -> (u, v) each (B, 1, H, W)."""
    return field[:, 0:1], field[:, 1:2]


def downsample_methods(vort_hat, nse, ns, n):
    """Return {name: (u, v)} on the ns x ns grid, plus the full-res DNS (u, v, omega)."""
    (u_hat, v_hat), _ = vorticity_to_velocity(nse.grid, vort_hat, (nse.kx, nse.ky))
    u_f, v_f = fft.irfft2(u_hat), fft.irfft2(v_hat)          # ground-truth velocity (n x n)
    omega_f = fft.irfft2(vort_hat)                            # ground-truth vorticity (n x n)

    om_bil = F.interpolate(omega_f, size=(ns, ns), mode="bilinear")
    om_spc = _truncate_to_grid(fft.rfft2(omega_f), ns)

    # velocity_bilinear is the *old* current-default (bilinear on velocity, aliased + NOT div-free).
    # Built directly here: create_dataset's DOWNSAMPLERS["bilinear"] is now the div-free
    # vorticity->reconstruct version, which appears below as vorticity_bilinear_recon.
    vel_bil = F.interpolate(torch.cat([u_f, v_f], dim=1), size=(ns, ns), mode="bilinear")
    # _downsample_spectral is torch.compile'd (reduce-overhead reuses CUDA-graph output buffers
    # across calls) -- clone before the next compiled call can overwrite it.
    vel_spc = DOWNSAMPLERS["spectral"](vort_hat, nse, ns).clone()

    methods = {
        "velocity_bilinear":        _split(vel_bil),
        "vorticity_bilinear_recon": velocity_from_vorticity(om_bil),
        "velocity_spectral":        _split(vel_spc),
        "vorticity_spectral_recon": velocity_from_vorticity(om_spc),
        "velocity_smooth_spectral": _smooth_spectral(u_hat, v_hat, ns, n),
    }
    return methods, (u_f.clone(), v_f.clone(), omega_f.clone())


REFERENCE = "velocity_spectral"   # P_ns(W): the L2-optimal band-limited field


# ---------------------------------------------------------------------------
# metrics (pure functions over fields)
# ---------------------------------------------------------------------------

def _div_ratio(u, v):
    d, w = dv.divergence(u, v), dv.vorticity(u, v)
    return (d.pow(2).mean().sqrt() / w.pow(2).mean().sqrt()).item()


def _rel_l2(u, v, u_ref, v_ref):
    num = ((u - u_ref) ** 2 + (v - v_ref) ** 2).sum()
    den = (u_ref ** 2 + v_ref ** 2).sum()
    return (num / den).sqrt().item()


def _corr(u, v, u_ref, v_ref):
    a = torch.cat([u.flatten(), v.flatten()])
    b = torch.cat([u_ref.flatten(), v_ref.flatten()])
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


# ---------------------------------------------------------------------------
# driver: generate DNS + accumulate metrics
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_analysis(Re, *, n_snapshots=48, stride_steps=256, batch_size=4, ns=64, n=512,
                 warmup_scale=1.0, seed=0, device=None, progress=True):
    """Generate one 512^2 DNS run at `Re` and compare the down-samplers against it.

    Returns a dict of averaged spectra + per-method scalar metrics, plus one snapshot's
    vorticity fields (sample 0) for visualization.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    cfg = DataGenConfig(Re=Re, grid_size=n, subsample=n // ns,
                        batch_size=batch_size, num_samples=batch_size, seed=seed, no_tqdm=True)
    resolved = resolve_config(cfg, device)
    nse, vort_hat = build_simulation(resolved, rank=0)
    dt = resolved.dt
    warmup = int(resolved.warmup_steps * warmup_scale)

    for _ in range(warmup):
        vort_hat = nse(vort_hat, dt).clone()

    names = ["velocity_bilinear", "vorticity_bilinear_recon",
             "velocity_spectral", "vorticity_spectral_recon", "velocity_smooth_spectral"]
    E_dns = None
    E = {nm: None for nm in names}
    acc = {nm: {"div": [], "relL2": [], "corr": [], "vmin": [], "vmax": []} for nm in names}
    dns_div, dns_vmax = [], []
    fields = {}

    for s in range(n_snapshots):
        for _ in range(stride_steps):
            vort_hat = nse(vort_hat, dt).clone()
        methods, (u_f, v_f, omega_f) = downsample_methods(vort_hat, nse, ns, n)

        k_dns, e_dns = dv.energy_spectrum(u_f, v_f)
        E_dns = e_dns if E_dns is None else E_dns + e_dns
        dns_div.append(_div_ratio(u_f, v_f))
        dns_vmax.append(dv.vorticity(u_f, v_f).abs().amax().item())

        u_ref, v_ref = methods[REFERENCE]
        for nm in names:
            u, v = methods[nm]
            k, e = dv.energy_spectrum(u, v)
            E[nm] = e if E[nm] is None else E[nm] + e
            w = dv.vorticity(u, v)
            acc[nm]["div"].append(_div_ratio(u, v))
            acc[nm]["relL2"].append(_rel_l2(u, v, u_ref, v_ref))
            acc[nm]["corr"].append(_corr(u, v, u_ref, v_ref))
            acc[nm]["vmin"].append(w.amin().item())
            acc[nm]["vmax"].append(w.amax().item())

        if s == n_snapshots - 1:                       # keep one snapshot for visuals (sample 0)
            fields["dns_omega"] = omega_f[0, 0].cpu().numpy()
            fields["methods_omega"] = {nm: dv.vorticity(*methods[nm])[0, 0].cpu().numpy()
                                       for nm in names}
        if progress:
            print(f"  Re{Re}: snapshot {s + 1}/{n_snapshots}", end="\r")

    E_dns = E_dns / n_snapshots
    nyq = ns // 2
    irreducible_frac = float(E_dns[nyq + 1:].sum() / E_dns[1:].sum())   # energy lost above coarse Nyquist

    methods_out = {}
    for nm in names:
        a = acc[nm]
        methods_out[nm] = {
            "E": E[nm] / n_snapshots,
            "div_ratio": float(np.mean(a["div"])),
            "relL2_vs_ref": float(np.mean(a["relL2"])),
            "corr_vs_ref": float(np.mean(a["corr"])),
            "vmin": float(np.min(a["vmin"])),
            "vmax": float(np.max(a["vmax"])),
        }
    return {
        "Re": Re, "ns": ns, "n": n, "dt": dt, "snapshot_dt": resolved.snapshot_dt,
        "n_snapshots": n_snapshots, "batch_size": batch_size, "reference": REFERENCE,
        "k_dns": k_dns, "E_dns": E_dns, "k": k,
        "irreducible_frac": irreducible_frac,
        "dns_div_ratio": float(np.mean(dns_div)), "dns_vmax": float(np.max(dns_vmax)),
        "methods": methods_out, "fields": fields,
        "method_order": names,
    }
