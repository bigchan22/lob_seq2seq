#!/usr/bin/env bash
set -euo pipefail

cd /home/hosung/lob_seq2seq_predictor

OUT_DIR="results/single_asset_ablation_nofuture_valtest_cuda_20260713"
mkdir -p "$OUT_DIR"

/home/hosung/miniconda3/envs/lob/bin/python main_single_asset_ablation.py \
  --epochs 500 \
  --output-dir "$OUT_DIR" \
  --no-progress \
  --log-every 25 \
  --device cuda:2 \
  > "$OUT_DIR/run.log" 2>&1
