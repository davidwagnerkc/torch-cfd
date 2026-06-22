#!/usr/bin/env bash
# Launch JupyterLab bound to all interfaces and print a Tailscale-reachable URL.
set -euo pipefail

PORT="${1:-8888}"

# Resolve the Tailscale IP at runtime so this keeps working if it changes.
TS_IP="$(tailscale ip -4 2>/dev/null | head -n1)"
if [[ -z "${TS_IP}" ]]; then
  echo "warning: could not get Tailscale IP; falling back to hostname display" >&2
  DISPLAY_URL=""
else
  DISPLAY_URL="--ServerApp.custom_display_url=http://${TS_IP}:${PORT}"
fi

exec jupyter lab \
  --ip=0.0.0.0 \
  --no-browser \
  --port="${PORT}" \
  ${DISPLAY_URL}
