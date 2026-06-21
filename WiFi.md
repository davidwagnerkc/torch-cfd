# Fast machine-to-machine sync (5090-alpha ↔ 5090-beta)

Runbook from syncing the `2dk_re_sweep` dataset (~45 GB of `.pt`) between the two
desktops. **Symptom:** transfer crawled at ~60 Mbps over Tailscale/WiFi (~90 min
for 45 GB) despite Google Fiber 500 Mbps.

## What we found (diagnosis)

| check | result |
|-------|--------|
| WiFi PHY rate | **270 Mbps** (5 GHz, ch 149, signal ~65%) — link has headroom |
| single rsync over `tailscale ssh` | ~5 MB/s (40 Mbps) |
| 8 parallel rsync streams | ~7 MB/s — **did not scale** → shared bottleneck, not per-stream |
| direct LAN ssh (`192.168.86.x:22`) | **refused** — beta only reachable through the tunnel |
| Tailscale WireGuard type | **userspace** (`tun` device, wireguard-go) |

**Conclusion:** the cap is Tailscale's *userspace WireGuard* tunnel — all streams
funnel through one userspace process, so parallelism doesn't help and the 270 Mbps
WiFi sits idle. Not WiFi, not crypto-per-stream, not latency.

## Fixes — what to do next time (ranked)

1. **Move less.** The dataset dir is 84 GB but half is redundant: the `.npy` shards
   are intermediate, the `.pt` files (~45 GB) are the trained-on artifact. Sync only
   `*.pt` (+ tiny `*.json` sidecars + README):
   ```bash
   rsync -ah --info=progress2 --include='*/' --include='*.pt' --include='*.json' \
     --include='README.md' --exclude='*' \
     /scratch/dwcgt/2dk_re_sweep/ user@host:/scratch/dwcgt/2dk_re_sweep/
   ```

2. **Regenerate instead of transfer** (often the winner). Seeds are committed and
   deterministic; on the mirror box: `git pull` then run the README's generation loop
   (~30 min, zero network). Caveat: not guaranteed bit-identical across machines
   (GPU reduction nondeterminism) — fine for separate runs, transfer if you need exact bytes.

3. **Bypass the WireGuard tunnel** for bulk transfer (the real fix for the cap):
   - **Best: kernel WireGuard.** Userspace wireguard-go is the bottleneck. If the kernel
     `wireguard` module is available, Tailscale uses it automatically and the tunnel does
     hundreds of Mbps+. Check: `ip -d link show tailscale0` (`tun` = userspace = slow).
   - **Or go around Tailscale on the LAN.** Both boxes are on `192.168.86.x`. Enable sshd on
     the LAN interface on the target (regular `sshd`, not just Tailscale SSH — that's why
     `192.168.86.x:22` was refused), then rsync to the **LAN IP** directly (plain TCP, no WG).
     For a one-off, an `nc` / `rsync --daemon` listener on the LAN works too (unencrypted,
     fine on a home LAN).

4. **Wire ethernet.** Helps if the cap is WiFi; helps *less* if it's the WireGuard tunnel
   (traffic still goes through it). Best combined with #3 (ethernet + LAN bypass = gigabit,
   ~7 min for 45 GB). Note: alpha's `enp7s0` is currently DOWN (on WiFi `wlp8s0`).

5. **External SSD sneakernet.** Only worth it if USB3/NVMe (~400+ MB/s); a USB2 SSD
   (~35 MB/s) is slower than WiFi.

**Skip Cloudflare R2 for same-LAN sync** — you'd pay the slow link twice (upload + download).
Keep R2 for backup or non-co-located machines.

## Diagnostics cheat-sheet

```bash
nmcli -t -f active,ssid,chan,rate,signal dev wifi | grep '^yes'   # WiFi PHY rate + band
tailscale ping <host>                                             # direct vs DERP relay, latency
tailscale status | grep <host>                                    # direct/relay + peer LAN addr
ip -d link show tailscale0 | grep -oE 'tun|wireguard'             # userspace(tun) vs kernel WG
ip -br addr | grep enp                                            # is ethernet up?
# live transfer rate: du the dest twice over an interval
ssh user@host 'du -sb /path' ; sleep 30 ; ssh user@host 'du -sb /path'
```

## Gotchas hit

- `pkill -f "rsync …"` **matched its own shell** (the pattern was in the command line) and
  killed the parent. Use `pkill -x rsync` (exact process name) to target rsync only.
- Parallel rsync (`ls */*.pt | xargs -P8 -I{} rsync -a --partial {} host:/dest/{}`) is the
  right tool when the bottleneck is per-stream — but here it didn't help, which is itself the
  clue that the tunnel (not the stream) was the limit.
