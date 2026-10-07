#!/bin/bash

set -euo pipefail

cd /users/smehra/looped_latent_reasoning/coconut

run_smoke() {
  local config=$1
  local port=$2
  echo "Running ${config}"
  python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    --master-port="${port}" \
    run.py "${config}"
}

run_smoke args/smoke_qwen3_1.7b_coconut.yaml 29533
run_smoke args/smoke_ouro_1.4b_r1_coconut.yaml 29534
