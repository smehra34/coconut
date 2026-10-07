#!/bin/bash

set -u

readonly REPO_DIR=/users/smehra/looped_latent_reasoning/coconut-safe-optimizations
readonly CONFIG_DIR=${REPO_DIR}/logs/benchmark-configs
mkdir -p "${CONFIG_DIR}"
cd "${REPO_DIR}"

make_config() {
  local base_config=$1
  local output_config=$2
  local run_name=$3
  local fused_optimizer=$4
  local attention_backend=$5
  python - "${base_config}" "${output_config}" "${run_name}" \
    "${fused_optimizer}" "${attention_backend}" <<'PY'
import sys
import yaml

base_path, output_path, name, fused, attention = sys.argv[1:]
with open(base_path) as source:
    config = yaml.safe_load(source)
config.update(
    {
        "name": name,
        "gradient_checkpointing": False,
        "distributed_strategy": "fsdp",
        "selective_loss_logits": False,
        "fused_optimizer": fused == "true",
        "attn_implementation": attention,
        "profile_training": False,
    }
)
with open(output_path, "w") as destination:
    yaml.safe_dump(config, destination, sort_keys=False)
PY
}

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

for model in qwen3 ouro; do
  base_config="args/benchmark_${model}_stage3_bs32.yaml"
  for variant in baseline fused flash eager; do
    fused=false
    attention=sdpa
    if [[ "${variant}" == fused ]]; then
      fused=true
    elif [[ "${variant}" == flash ]]; then
      attention=flash_attention_2
    elif [[ "${variant}" == eager ]]; then
      attention=eager
    fi
    make_config \
      "${base_config}" \
      "${CONFIG_DIR}/${model}-${variant}.yaml" \
      "${model}-safe-opt-${variant}" \
      "${fused}" \
      "${attention}"
  done
done

run_benchmark "${CONFIG_DIR}/qwen3-baseline.yaml" 29601
run_benchmark "${CONFIG_DIR}/qwen3-fused.yaml" 29602
run_benchmark "${CONFIG_DIR}/qwen3-flash.yaml" 29603
run_benchmark "${CONFIG_DIR}/qwen3-eager.yaml" 29604
run_benchmark "${CONFIG_DIR}/ouro-baseline.yaml" 29605
run_benchmark "${CONFIG_DIR}/ouro-fused.yaml" 29606
run_benchmark "${CONFIG_DIR}/ouro-flash.yaml" 29607
run_benchmark "${CONFIG_DIR}/ouro-eager.yaml" 29608
