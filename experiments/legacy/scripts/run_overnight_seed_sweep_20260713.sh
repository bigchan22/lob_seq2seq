#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/hosung/miniconda3/envs/lob/bin/python
ROOT=/home/hosung/lob_seq2seq_predictor
cd "$ROOT"

mkdir -p results/overnight_logs_20260713

run_parallel() {
  local seed=$1
  local setting=$2
  local device=$3
  local out_dir=$4
  shift 4

  "$PYTHON" main_parallel_asset_attention.py \
    --epochs 500 \
    --output-dir "$out_dir" \
    --seed "$seed" \
    --no-progress \
    --log-every 10 \
    --device "$device" \
    "$@"
}

run_single() {
  local seed=$1
  local setting=$2
  local device=$3
  local out_dir=$4
  shift 4

  "$PYTHON" main_single_asset_ablation.py \
    --epochs 500 \
    --output-dir "$out_dir" \
    --seed "$seed" \
    --no-progress \
    --log-every 10 \
    --device "$device" \
    "$@"
}

(
  run_parallel 7 no_investor cuda:1 results/parallel_asset_attention_no_investor_valtest_cuda_20260713_seed7 --no-include-investor
  run_parallel 123 no_investor cuda:1 results/parallel_asset_attention_no_investor_valtest_cuda_20260713_seed123 --no-include-investor
) > results/overnight_logs_20260713/parallel_no_investor_seeds.log 2>&1 &

(
  run_parallel 7 investor_lag1 cuda:2 results/parallel_asset_attention_investor_lag1_valtest_cuda_20260713_seed7 --investor-lag-steps 1
  run_parallel 123 investor_lag1 cuda:2 results/parallel_asset_attention_investor_lag1_valtest_cuda_20260713_seed123 --investor-lag-steps 1
) > results/overnight_logs_20260713/parallel_investor_lag1_seeds.log 2>&1 &

(
  run_single 7 no_investor cuda:3 results/single_asset_ablation_no_investor_valtest_cuda_20260713_seed7 --no-include-investor
  run_single 7 investor_lag1 cuda:3 results/single_asset_ablation_investor_lag1_valtest_cuda_20260713_seed7 --investor-lag-steps 1
  run_single 123 no_investor cuda:3 results/single_asset_ablation_no_investor_valtest_cuda_20260713_seed123 --no-include-investor
  run_single 123 investor_lag1 cuda:3 results/single_asset_ablation_investor_lag1_valtest_cuda_20260713_seed123 --investor-lag-steps 1
) > results/overnight_logs_20260713/single_clean_seeds.log 2>&1 &

wait
