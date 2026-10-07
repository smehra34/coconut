#!/bin/bash

set -u

cd /users/smehra/looped_latent_reasoning/coconut

run_benchmark() {
  local config=$1
  local port=$2
  echo "BENCHMARK_START ${config}"
  if python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    --master-port="${port}" \
    run.py "${config}"; then
    echo "BENCHMARK_PASS ${config}"
  else
    echo "BENCHMARK_FAIL ${config}"
  fi
}

run_benchmark args/benchmark_qwen3_stage3_bs8.yaml 29541
run_benchmark args/benchmark_qwen3_stage3_bs16.yaml 29542
run_benchmark args/benchmark_qwen3_stage3_bs32.yaml 29543
run_benchmark args/benchmark_ouro_stage3_bs8.yaml 29544
run_benchmark args/benchmark_ouro_stage3_bs16.yaml 29545
run_benchmark args/benchmark_ouro_stage3_bs32.yaml 29546
