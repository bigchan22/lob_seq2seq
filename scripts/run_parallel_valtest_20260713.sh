#!/usr/bin/env bash
set -euo pipefail

cd /home/hosung/lob_seq2seq_predictor

OUT_DIR="results/parallel_asset_attention_nofuture_valtest_cuda_20260713"
mkdir -p "$OUT_DIR"

/home/hosung/miniconda3/envs/lob/bin/python main_parallel_asset_attention.py \
  --epochs 500 \
  --output-dir "$OUT_DIR" \
  --no-progress \
  --log-every 10 \
  --device cuda:1 \
  > "$OUT_DIR/run.log" 2>&1
