#!/usr/bin/env bash
set -euo pipefail
PY=/home/hosung/miniconda3/envs/lob/bin/python3.10
: "${QF_DISC_ROOT:?export QF_DISC_ROOT to the Stage-2 artifact root}"
launch() { local name=$1 cmd=$2; if command -v tmux >/dev/null; then tmux has-session -t "$name" 2>/dev/null || tmux new-session -d -s "$name" "$cmd"; else nohup bash -lc "$cmd" >"$QF_DISC_ROOT/${name}.log" 2>&1 & echo $! >"$QF_DISC_ROOT/${name}.pid"; fi; }
launch qf_disc_gpu0 "cd '$PWD' && export QF_DISC_ROOT='$QF_DISC_ROOT' CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0; '$PY' -m lob_forecasting.experiments.discrimination_cli worker --gpu 0"
launch qf_disc_gpu1 "cd '$PWD' && export QF_DISC_ROOT='$QF_DISC_ROOT' CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1; '$PY' -m lob_forecasting.experiments.discrimination_cli worker --gpu 1"
launch qf_disc_coordinator "cd '$PWD' && export QF_DISC_ROOT='$QF_DISC_ROOT' CUDA_VISIBLE_DEVICES=''; '$PY' -m lob_forecasting.experiments.discrimination_cli coordinator"
