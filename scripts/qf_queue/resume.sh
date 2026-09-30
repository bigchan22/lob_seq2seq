#!/usr/bin/env bash
cd "$(dirname "$0")/../.." && exec /home/hosung/miniconda3/envs/lob/bin/python3.10 -m lob_forecasting.experiments.queue_cli resume
