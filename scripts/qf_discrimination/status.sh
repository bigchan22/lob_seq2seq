#!/usr/bin/env bash
set -euo pipefail
PY=/home/hosung/miniconda3/envs/lob/bin/python3.10
: "${QF_DISC_ROOT:?export QF_DISC_ROOT}"
"$PY" -m lob_forecasting.experiments.discrimination_cli status
df -h "$QF_DISC_ROOT"
