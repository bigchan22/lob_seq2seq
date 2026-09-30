#!/usr/bin/env bash
set -euo pipefail

cd /home/hosung/lob_seq2seq_predictor

OUT_DIR="results/classical_baselines_nofuture_valtest_20260713"
mkdir -p "$OUT_DIR"

/home/hosung/miniconda3/envs/lob/bin/python main_classical_baselines.py \
  --output-dir "$OUT_DIR" \
  --models majority persistence stratified logistic random_forest mlp \
  --max-iter 300 \
  --n-jobs 8 \
  > "$OUT_DIR/run.log" 2>&1
