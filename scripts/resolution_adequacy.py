"""T4: is 512^2 adequate for Re=1000 Kolmogorov flow, and where does it break?

Judged on SMALL-SCALE diagnostics, deliberately NOT the energy spectrum -- the energy
spectrum is low-k dominated and forgiving, and it is exactly what produced the earlier wrong
conclusion "512 is good to Re=40k" (memory: downsampling-method-irrelevant-this-regime).

Reported, on the FULL simulation grid (no downsampling), on a stationary window:
  * palinstrophy   P = sum k^4 E(k)          (grid-scale sensitive; 4th moment of k)
  * enstrophy      Z = sum k^2 E(k)
  * vorticity flatness  <w^4>/<w^2>^2        (3.0 = Gaussian; small-scale intermittency)
  * fraction of enstrophy / palinstrophy above k = 32, 64, 128
  * E(k), k^2 E(k), k^4 E(k) tails, and the resolution-matched comparison: the SAME bands
    measured at 512^2 and 1024^2. If 512^2 is adequate, every band up to its own reliable
    range must agree with 1024^2.

Uses the ORIGINAL torch-cfd spectral solver (`torch_cfd.spectral.NavierStokes2DSpectral`,
RK4-CrankNicolson, 2/3-rule smoothing) with the exact `2dk_spectral_w45` physics
(Re, scale 1.0, wavenumber 4, drag 0.1, max_velocity 7.0, diam 2pi, 45 s warmup).
NOT the JAX port -- deliberately, so our own port cannot be a confound.

  python scripts/resolution_adequacy.py --grids 512,1024 --Re 1000 --batch 4
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.fft as fft

from torch_cfd.finite_differences import curl_2d
from torch_cfd.forcings import KolmogorovForcing
from torch_cfd.grids import Grid
from torch_cfd.initial_conditions import filtered_velocity_field, log_normal_density
from torch_cfd.spectral import NavierStokes2DSpectral, RK4CrankNicolsonStepper, stable_time_step


def build(n, Re, batch, seed, device, diam=2 * np.pi, max_velocity=7.0, scale=1.0,
          peak_wavenumber=4, drag=0.1):
    grid = Grid(shape=(n, n), domain=((0, diam), (0, diam)), device=device)
    forcing = KolmogorovForcing(grid=grid, scale=scale, wave_number=peak_wavenumber, swap_xy=False)
    nse = NavierStokes2DSpectral(viscosity=1.0 / Re, grid=grid, drag=drag, smooth=True,
                                 forcing_fn=forcing, step_fn=RK4CrankNicolsonStepper()).to(device)
    nse = torch.compile(nse, mode="reduce-overhead")
    ws = []
    for i in range(batch):
        uv = filtered_velocity_field(grid, max_velocity, peak_wavenumber, iterations=3,
                                     random_state=seed + i, batch_size=1,
                                     spectral_density=log_normal_density, device=device)
        ws.append(curl_2d(uv).data)
    return nse, fft.rfft2(torch.stack(ws)).to(device)


def shell_moments(w_hat, n, nbin):
    """From vorticity_hat on an n x n grid: shell KE spectrum E(k) (density-normalised),
    k=1..nbin. E = |w_hat|^2 / (2 k^2) since w = curl u => |w_k|^2 = k^2 |u_k|^2."""
    kr = fft.fftfreq(n, 1.0 / n, device=w_hat.device).view(-1, 1)
    kc = fft.rfftfreq(n, 1.0 / n, device=w_hat.device).view(1, -1)
    k2 = kr ** 2 + kc ** 2
    kb = k2.sqrt().round().long()
    # rfft2 holds half the plane; double the interior columns to recover full-plane power
    wgt = torch.full_like(kc, 2.0).expand(n, kc.shape[-1]).clone()
    wgt[:, 0] = 1.0
    if n % 2 == 0:
        wgt[:, -1] = 1.0
    p = (w_hat.abs() ** 2) * wgt / (n ** 4)          # |w_k|^2 density, unnormalised-fft -> /n^4
    e = torch.where(k2 > 0, 0.5 * p / k2.clamp_min(1e-30), torch.zeros_like(p))
    keep = (kb >= 1) & (kb <= nbin)
    out = torch.zeros(w_hat.shape[0], nbin + 1, dtype=torch.float64, device=w_hat.device)
    out.index_add_(-1, kb[keep].reshape(-1), e.double()[..., keep].reshape(w_hat.shape[0], -1))
    return out[:, 1:]


def exact_moments(w_hat, n, kcuts=(32, 64, 128, 256)):
    """Scalars computed with the TRUE |k| (no shell rounding), so they are exact:
      <w^2> (2*enstrophy), <|grad w|^2> (2*palinstrophy), flatness, and the fraction of
      each above every k in `kcuts`."""
    kr = fft.fftfreq(n, 1.0 / n, device=w_hat.device).view(-1, 1)
    kc = fft.rfftfreq(n, 1.0 / n, device=w_hat.device).view(1, -1)
    k2 = kr ** 2 + kc ** 2
    wgt = torch.full_like(kc, 2.0).expand(n, kc.shape[-1]).clone()
    wgt[:, 0] = 1.0
    if n % 2 == 0:
        wgt[:, -1] = 1.0
    p = ((w_hat.abs() ** 2) * wgt / (n ** 4)).double()      # per-mode contribution to <w^2>
    w = fft.irfft2(w_hat, s=(n, n)).double()
    m2, m4 = (w ** 2).mean((-1, -2)), (w ** 4).mean((-1, -2))
    out = dict(m2=m2, m4=m4, flatness=m4 / m2 ** 2,
               ens=p.sum((-1, -2)), pal=(p * k2).sum((-1, -2)))
    out["pal_over_ens2"] = out["pal"] / out["ens"] ** 2
    for kk in kcuts:
        m = k2.sqrt() > kk
        out[f"ens_above{kk}"] = (p * m).sum((-1, -2)) / out["ens"]
        out[f"pal_above{kk}"] = (p * k2 * m).sum((-1, -2)) / out["pal"]
    return {k: v.cpu().numpy() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="512,1024")
    ap.add_argument("--Re", type=float, default=1000.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1700000000)
    ap.add_argument("--warmup", type=float, default=45.0)
    ap.add_argument("--window", type=float, default=5.0, help="stationary window (sim seconds) to average over")
    ap.add_argument("--n_snap", type=int, default=20)
    ap.add_argument("--out", default="/scratch/dwcgt/resolution_adequacy.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.set_default_dtype(torch.float32)
    res = {}

    for n in (int(x) for x in a.grids.split(",")):
        dx = 2 * np.pi / n
        dt = stable_time_step(dx, float("inf"), 7.0, viscosity=1.0 / a.Re)
        wsteps = int(a.warmup / dt)
        every = max(1, int(a.window / dt / a.n_snap))
        nbin = int(np.ceil(np.sqrt(2) * n / 2))   # to the box CORNER, so sum k^2 E == 0.5<w^2> exactly
        print(f"\n=== n={n}  dt={dt:.4e}  warmup {wsteps} steps ({a.warmup}s)  "
              f"{a.n_snap} snapshots every {every} steps ({every*dt:.4f}s) ===", flush=True)
        nse, w = build(n, a.Re, a.batch, a.seed, dev)
        t0 = time.time()
        for i in range(wsteps):
            w = nse(w, dt).clone()
            if i % 20000 == 0:
                print(f"  warmup {i}/{wsteps}  {time.time()-t0:.0f}s", flush=True)
        acc = torch.zeros(a.batch, nbin, dtype=torch.float64, device=dev)
        mom = []
        for s in range(a.n_snap):
            for _ in range(every):
                w = nse(w, dt).clone()
            acc += shell_moments(w, n, nbin)
            mom.append(exact_moments(w, n))
        E = (acc / a.n_snap).cpu().numpy()                     # (batch, nbin)
        S = {k: np.stack([m[k] for m in mom]) for k in mom[0]}        # (n_snap, batch) raw samples
        M = {k: v.mean(0) for k, v in S.items()}                      # time-mean, per IC
        # error bar: sem over ICs of the time-mean (ICs are independent; snapshots are not)
        res[n] = dict(E=E.tolist(), dt=dt, warmup_steps=wsteps, every=every, batch=a.batch,
                      moments={k: v.tolist() for k, v in M.items()},
                      moments_sem={k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in M.items()},
                      wall_s=time.time() - t0)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)
        del nse, w
        torch.cuda.empty_cache()

    # ---------- report ----------
    print("\n\n================ RESOLUTION ADEQUACY, Re=%g ================" % a.Re)
    ns = sorted(res)
    print(f"{'quantity':34s} " + " ".join(f"{'n=' + str(n):>14s}" for n in ns))
    rowfmt = lambda nm, f: print(f"{nm:34s} " + " ".join(f"{f(n):14.6g}" for n in ns))

    def Em(n):
        return np.asarray(res[n]["E"]).mean(0)

    def tot(n, p):
        E = Em(n)
        k = np.arange(1, len(E) + 1)
        return float((k ** p * E).sum())

    def M(n, k):
        return float(np.mean(res[n]["moments"][k]))

    def pm(n, k, scale=1.0):
        return f"{M(n, k)*scale:.5g} +- {res[n]['moments_sem'][k]*scale:.3g}"

    print("  --- exact scalars (TRUE |k|, no shell rounding); value +- sem over ICs ---")
    prow = lambda nm, k, sc=1.0: print(f"{nm:34s} " + " ".join(f"{pm(n, k, sc):>22s}" for n in ns))
    prow("enstrophy  <w^2>/2", "ens", 0.5)
    prow("PALINSTROPHY  <|grad w|^2>/2", "pal", 0.5)
    prow("vorticity flatness <w4>/<w2>^2", "flatness")
    prow("palinstrophy / enstrophy^2", "pal_over_ens2")
    for kc in (32, 64, 128, 256):
        prow(f"frac enstrophy above k={kc}", f"ens_above{kc}")
        prow(f"frac palinstrophy above k={kc}", f"pal_above{kc}")
    print("  --- shell-binned (for the band table below) ---")
    rowfmt("KE  sum E(k)", lambda n: tot(n, 0))
    rowfmt("enstrophy  sum k^2 E(k)", lambda n: tot(n, 2))
    rowfmt("palinstrophy sum k^4 E(k)", lambda n: tot(n, 4))
    print(f"\n{'band':12s} " + " ".join(f"{'k^4E n=' + str(n):>14s}" for n in ns) + f" {'ratio':>10s}")
    for lo, hi in ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 512)):
        vals = []
        for n in ns:
            E = Em(n)
            k = np.arange(1, len(E) + 1)
            m = (k >= lo) & (k <= hi)
            vals.append(float((k ** 4 * E)[m].sum()) if m.any() else float("nan"))
        r = vals[-1] / vals[0] if len(vals) > 1 and vals[0] and np.isfinite(vals[0]) else float("nan")
        print(f"k {lo:3d}-{hi:3d}   " + " ".join(f"{v:14.6g}" for v in vals) + f" {r:10.4f}")
    print("\n  'ratio' = coarsest-grid value / finest-grid value per band. A grid is adequate in a")
    print("  band when the ratio is ~1; >1 means the coarse grid PILES UP energy there (aliasing /")
    print("  missing dissipation), <1 means it is over-damped.")

    Path(a.out).write_text(json.dumps(res))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
