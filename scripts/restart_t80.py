"""Machine-epsilon twin: restart the 512^2 solver from the saved post-warmup ic512 states and record the
64^2 eval-cadence frames (178 frames @ 0.1122 s), to be scored against the STORED val trajectories.

The only perturbation is the fp32 velocity round-trip (~1 ulp) + batch-layout rounding, so the measured t80
is the decorrelation horizon of floating-point error itself: expected ~ ln(1/eps)/lambda(Re).

The velocity->vorticity inversion is done convention-proof: vorticity_to_velocity is linear and diagonal in k
(u_hat = A(k)*w_hat, v_hat = B(k)*w_hat), so A,B are extracted numerically with a ones probe and inverted by
least squares -- no assumptions about wavenumber scaling/axis conventions; validated by round-trip error.

Run inside torch-cfd: .venv/bin/python scripts/restart_t80.py --re 1189
Writes /scratch/dwcgt/evaluations/restart_t80/restart_frames_Re{re}.pt  (178, 128, 2, 64, 64 fp32, ~373 MB).
"""
import argparse, glob, time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.fft as fft
from hydra import compose, initialize_config_dir
from torch_cfd.spectral import vorticity_to_velocity

sys.path.insert(0, str(Path(__file__).parent))
from create_dataset import build_simulation, resolve_config, DOWNSAMPLERS  # exact gen-time components

ap = argparse.ArgumentParser()
ap.add_argument("--re", type=int, required=True)
ap.add_argument("--out-dir", default="/scratch/dwcgt/evaluations/restart_t80")
a = ap.parse_args()

DEV = torch.device("cuda")
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "ieee"
torch.set_default_dtype(torch.float32)

with initialize_config_dir(config_dir=str(Path(__file__).parent / "conf"), version_base=None):
    cfg = compose(config_name=f"2dk_extended_re/val_Re{a.re}")
resolved = resolve_config(cfg, DEV)
nse, _ = build_simulation(resolved, rank=0)            # exact solver; discard its fresh ICs
ds_spectral = DOWNSAMPLERS["spectral"]

# --- load all 8 ic512 shards -> (128, 2, 512, 512) fp32 velocity ---
ics = sorted(glob.glob(f"/scratch/dwcgt/2dk_extended_re/val/ic512/*-Re{a.re}-ic512-*.npy"),
             key=lambda p: int(Path(p).name.split("-")[0]))
assert len(ics) == 8, f"expected 8 ic512 shards for Re{a.re}, found {len(ics)}"
u = torch.from_numpy(np.concatenate([np.load(p) for p in ics], axis=0)).to(DEV)   # (128,2,512,512)

# --- velocity -> solver vorticity state, convention-proof (numeric diagonal-operator inversion) ---
probe = torch.ones(1, 1, 512, 257, dtype=torch.complex64, device=DEV)
(A, B), _ = vorticity_to_velocity(nse.grid, probe, (nse.kx, nse.ky))               # u_hat = A*w_hat, v_hat = B*w_hat
den = (A.abs() ** 2 + B.abs() ** 2)
uh, vh = fft.rfft2(u[:, :1]), fft.rfft2(u[:, 1:])
vort_hat = torch.where(den > 1e-12, (A.conj() * uh + B.conj() * vh) / den.clamp(min=1e-12),
                       torch.zeros_like(uh))                                       # (128,1,512,257)
(u2h, v2h), _ = vorticity_to_velocity(nse.grid, vort_hat, (nse.kx, nse.ky))
u2 = torch.cat([fft.irfft2(u2h), fft.irfft2(v2h)], dim=1)
rt = float((u2 - u).norm() / u.norm())
print(f"Re{a.re}: round-trip rel err = {rt:.3e} (expect ~1e-7; this IS the twin epsilon)", flush=True)
assert rt < 1e-4, "inversion failed -- investigate before running"

# --- advance and record at eval cadence: stored eval frames = snapshots 16,32,... = solver steps 136+128m ---
dt = resolved.dt
frames = torch.empty(178, 128, 2, 64, 64)
t0 = time.time()
with torch.inference_mode():
    for _ in range(136):
        vort_hat = nse(vort_hat, dt).clone()
    frames[0] = ds_spectral(vort_hat, nse, ns=64).cpu()
    for mrec in range(1, 178):
        for _ in range(128):
            vort_hat = nse(vort_hat, dt).clone()
        frames[mrec] = ds_spectral(vort_hat, nse, ns=64).cpu()
        if mrec % 30 == 0:
            print(f"  frame {mrec}/178  ({time.time()-t0:.0f}s)", flush=True)
Path(a.out_dir).mkdir(parents=True, exist_ok=True)
out = f"{a.out_dir}/restart_frames_Re{a.re}.pt"
torch.save({"frames": frames, "re": a.re, "roundtrip_relerr": rt, "dt": dt}, out)
print(f"DONE_RESTART Re{a.re} {time.time()-t0:.0f}s -> {out}", flush=True)
