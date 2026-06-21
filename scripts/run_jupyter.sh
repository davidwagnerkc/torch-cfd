#!/usr/bin/env bash
# Launch JupyterLab reachable from another machine over Tailscale.
#
# Binds the Tailscale IPv4 by default, so the printed URL works as-is from your
# MacBook (and Jupyter is exposed only on your tailnet, not every interface like
# 0.0.0.0 would be). Note: the system hostname here resolves to 127.0.1.1
# (loopback), so binding to it would only listen on localhost.
#
# Usage: ./run_jupyter.sh [PORT]       # PORT defaults to 8888 (or $PORT)
#        IP=0.0.0.0 ./run_jupyter.sh   # bind all interfaces instead
set -euo pipefail

PORT="${1:-${PORT:-8888}}"

# Resolve paths from this script's location so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
JUPYTER="$REPO_ROOT/.venv/bin/jupyter"

if [[ ! -x "$JUPYTER" ]]; then
    echo "jupyter not found at $JUPYTER -- run 'uv sync' in $REPO_ROOT first." >&2
    exit 1
fi

# Bind address: explicit $IP wins, else the Tailscale IPv4, else all interfaces.
IP="${IP:-$(tailscale ip -4 2>/dev/null | head -1 || true)}"
if [[ -z "$IP" ]]; then
    echo "No Tailscale IP found; falling back to 0.0.0.0 (all interfaces)." >&2
    IP="0.0.0.0"
fi

echo "Launching JupyterLab on http://$IP:$PORT/  (root: $SCRIPT_DIR)"
cd "$SCRIPT_DIR"
exec "$JUPYTER" lab --no-browser --ip="$IP" --port="$PORT"
