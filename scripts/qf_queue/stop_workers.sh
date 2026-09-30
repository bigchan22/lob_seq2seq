#!/usr/bin/env bash
set -euo pipefail
if command -v tmux >/dev/null; then for s in qf_queue_gpu0 qf_queue_gpu1 qf_queue_coordinator; do tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true; done
else
 for f in "$(dirname "$0")"/*.pid; do [ -f "$f" ] || continue; pid=$(cat "$f"); [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'lob_forecasting.experiments.queue_cli' && kill "$pid" || true; done
fi
