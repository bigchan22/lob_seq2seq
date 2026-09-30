#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
/home/hosung/miniconda3/envs/lob/bin/python3.10 -m lob_forecasting.experiments.queue_cli status
