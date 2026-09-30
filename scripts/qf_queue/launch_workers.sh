#!/usr/bin/env bash
set -euo pipefail
PY=/home/hosung/miniconda3/envs/lob/bin/python3.10
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
launch='CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='
if command -v tmux >/dev/null; then
  tmux has-session -t qf_queue_gpu0 2>/dev/null || tmux new-session -d -s qf_queue_gpu0 "$launch'0' $PY -m lob_forecasting.experiments.queue_cli worker --gpu 0"
  tmux has-session -t qf_queue_gpu1 2>/dev/null || tmux new-session -d -s qf_queue_gpu1 "$launch'1' $PY -m lob_forecasting.experiments.queue_cli worker --gpu 1"
  tmux has-session -t qf_queue_coordinator 2>/dev/null || tmux new-session -d -s qf_queue_coordinator "$PY -m lob_forecasting.experiments.queue_cli coordinator"
else
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup "$PY" -m lob_forecasting.experiments.queue_cli worker --gpu 0 > artifacts/queue/gpu0.log 2>&1 & echo $! > scripts/qf_queue/gpu0.pid
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 nohup "$PY" -m lob_forecasting.experiments.queue_cli worker --gpu 1 > artifacts/queue/gpu1.log 2>&1 & echo $! > scripts/qf_queue/gpu1.pid
  nohup "$PY" -m lob_forecasting.experiments.queue_cli coordinator > artifacts/queue/coordinator.log 2>&1 & echo $! > scripts/qf_queue/coordinator.pid
fi
