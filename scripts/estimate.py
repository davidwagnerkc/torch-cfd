"""Quick runtime estimate for a dataset generation run (`mode=estimate`).

Deliberately separate from the generation hot path: this module imports nothing
from create_dataset -- main() injects the two simulation primitives it needs
(`build_simulation`, `spatial_downsample`). So the generation code stays clean and
the benchmark logic stays here.

Why two measurements: warmup is pure solver stepping, while the trajectory loop
also downsamples once every `record_every_steps` steps, so its per-step cost is
higher. We time each independently, account for the one-time torch.compile cost,
then project total wall time. Throughput is measured at steady state (after a
burn-in that triggers compilation / cudagraph capture) with a CUDA sync at every
window boundary, so the numbers reflect real GPU work, not async launch latency.
"""

from time import perf_counter as pc

import torch


def _steady_throughput(step, *, chunk, measure_seconds, sync):
    """Iterations/second of `step`, measured over a wall-time budget.

    Call only after `step` has been burned in. Syncs each chunk so the elapsed
    time bounds the GPU queue and reflects completed (not just launched) work.
    """
    iters, elapsed = 0, 0.0
    while elapsed < measure_seconds:
        t0 = pc()
        for _ in range(chunk):
            step()
        sync()
        elapsed += pc() - t0
        iters += chunk
    return iters / elapsed


def _fmt(seconds):
    if seconds < 90:
        return f"{seconds:5.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:5.1f} min"
    return f"{seconds / 3600:5.2f} hr"


def estimate_runtime(
    resolved,
    ranks,
    *,
    build_simulation,
    spatial_downsample,
    chunk=16,
    measure_seconds=2.0,
    burn_in=8,
):
    cfg = resolved.cfg
    dt = resolved.dt
    sync = torch.cuda.synchronize if resolved.device.type == "cuda" else (lambda: None)
    state = {}  # holds vort_hat so the step closures can advance it in place

    with torch.inference_mode():
        # --- setup: grid, ICs, solver build (per-shard cost) ---
        t0 = pc()
        nse, state["w"] = build_simulation(resolved, ranks[0])
        sync()
        setup_s = pc() - t0

        def step_solver():  # one warmup/solver step
            state["w"] = nse(state["w"], dt).clone()

        def step_record():  # one recorded snapshot: N steps + a downsample
            for _ in range(cfg.record_every_steps):
                state["w"] = nse(state["w"], dt).clone()
            spatial_downsample(state["w"], nse, ns=resolved.ns)

        # --- one-time compile cost (first call traces + compiles each graph) ---
        t0 = pc(); step_solver(); sync(); compile_solver_s = pc() - t0
        t0 = pc(); step_record(); sync(); compile_record_s = pc() - t0
        compile_s = compile_solver_s + compile_record_s

        # --- steady-state throughput (burn in, then measure) ---
        for _ in range(burn_in): step_solver()
        sync()
        warmup_it_s = _steady_throughput(step_solver, chunk=chunk, measure_seconds=measure_seconds, sync=sync)

        for _ in range(burn_in): step_record()
        sync()
        snap_s = _steady_throughput(step_record, chunk=chunk, measure_seconds=measure_seconds, sync=sync)

    traj_it_s = snap_s * cfg.record_every_steps
    warmup_s = resolved.warmup_steps / warmup_it_s
    traj_s = resolved.num_snapshots / snap_s
    per_shard_s = setup_s + warmup_s + traj_s

    num_batches = cfg.num_samples // cfg.batch_size
    n_here = len(ranks)
    # compile is amortized once per process; setup/warmup/traj recur per shard
    wall_here_s = compile_s + n_here * per_shard_s
    wall_all_seq_s = compile_s + num_batches * per_shard_s

    dev = torch.cuda.get_device_name(0) if resolved.device.type == "cuda" else "cpu"
    print(f"\n=== Runtime estimate: split '{cfg.split}' | grid {cfg.grid_size} | Re {cfg.Re} | batch {cfg.batch_size} | {dev} ===")
    print(f"  solver throughput   : {warmup_it_s:9.1f} it/s")
    print(f"  trajectory throughput: {traj_it_s:8.1f} it/s  ({snap_s:.1f} snapshots/s, downsample every {cfg.record_every_steps} steps)")
    print(f"  setup (grid + ICs)  : {_fmt(setup_s)} / shard")
    print(f"  compile (one-time)  : {_fmt(compile_s)}")
    print(f"  warmup  {cfg.time_warmup:>5}s    : {resolved.warmup_steps:>7} steps     -> {_fmt(warmup_s)} / shard")
    print(f"  traj    {cfg.time:>5}s    : {resolved.num_snapshots:>7} snapshots -> {_fmt(traj_s)} / shard")
    print(f"  per-shard total     : {_fmt(per_shard_s)}")
    print(f"  this run ({n_here} shard{'s' if n_here != 1 else ''})    : {_fmt(wall_here_s)}")
    print(f"  all {num_batches} shards (seq) : {_fmt(wall_all_seq_s)}\n")

    return {
        "warmup_it_s": warmup_it_s,
        "traj_it_s": traj_it_s,
        "snap_s": snap_s,
        "setup_s": setup_s,
        "compile_s": compile_s,
        "warmup_s": warmup_s,
        "traj_s": traj_s,
        "per_shard_s": per_shard_s,
        "wall_here_s": wall_here_s,
        "wall_all_seq_s": wall_all_seq_s,
    }
