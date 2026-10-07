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

run_benchmark args/benchmark_qwen3_stage0_bs32.yaml 29571
run_benchmark args/benchmark_ouro_stage0_bs32.yaml 29572
