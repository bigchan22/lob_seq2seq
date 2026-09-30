#!/usr/bin/env bash
set -euo pipefail
for s in qf_disc_gpu0 qf_disc_gpu1 qf_disc_coordinator; do tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true; done
