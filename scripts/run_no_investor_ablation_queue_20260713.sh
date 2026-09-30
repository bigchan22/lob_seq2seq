#!/usr/bin/env bash
set -euo pipefail

cd /home/hosung/lob_seq2seq_predictor

PARALLEL_OUT="results/parallel_asset_attention_no_investor_valtest_cuda_20260713"
SINGLE_OUT="results/single_asset_ablation_no_investor_valtest_cuda_20260713"

mkdir -p "$PARALLEL_OUT" "$SINGLE_OUT"

echo "[simulation 1] no-investor asset-aware parallel model"
/home/hosung/miniconda3/envs/lob/bin/python main_parallel_asset_attention.py \
  --epochs 500 \
  --output-dir "$PARALLEL_OUT" \
  --no-include-investor \
  --no-progress \
  --log-every 10 \
  --device cuda:1 \
  > "$PARALLEL_OUT/run.log" 2>&1

echo "[simulation 2] no-investor 27 single-asset models"
/home/hosung/miniconda3/envs/lob/bin/python main_single_asset_ablation.py \
  --epochs 500 \
  --output-dir "$SINGLE_OUT" \
  --no-include-investor \
  --no-progress \
  --log-every 25 \
  --device cuda:1 \
  > "$SINGLE_OUT/run.log" 2>&1

echo "no-investor ablation queue complete"
