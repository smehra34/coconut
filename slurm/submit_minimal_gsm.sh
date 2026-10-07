#!/bin/bash

set -euo pipefail

readonly REPO_DIR=/users/smehra/looped_latent_reasoning/coconut
readonly JOB_SCRIPT=${REPO_DIR}/slurm/train_gsm.slurm
readonly STAGE0_TIME=${STAGE0_TIME:-11:59:00}
readonly BRANCH_TIME=${BRANCH_TIME:-11:59:00}

cd "${REPO_DIR}"
mkdir -p logs/slurm checkpoints

submit_stage0() {
  local job_name=$1
  local config=$2
  sbatch --parsable \
    --job-name="${job_name}" \
    --time="${STAGE0_TIME}" \
    --export="ALL,CONFIG=${config},NPROC_PER_NODE=4" \
    "${JOB_SCRIPT}"
}

submit_branch() {
  local job_name=$1
  local config=$2
  local dependency=$3
  sbatch --parsable \
    --job-name="${job_name}" \
    --time="${BRANCH_TIME}" \
    --dependency="afterok:${dependency}" \
    --export="ALL,CONFIG=${config},NPROC_PER_NODE=4" \
    "${JOB_SCRIPT}"
}

qwen_stage0=$(submit_stage0 \
  gsm-qwen-stage0 args/gsm_qwen3_1.7b_stage0_cot.yaml)
ouro_stage0=$(submit_stage0 \
  gsm-ouro-r1-stage0 args/gsm_ouro_1.4b_r1_stage0_cot.yaml)

# Some federated Slurm installations append ";cluster" to --parsable output.
# Dependencies require the numeric job ID only.
qwen_stage0=${qwen_stage0%%;*}
ouro_stage0=${ouro_stage0%%;*}

qwen_cot=$(submit_branch \
  gsm-qwen-cot args/gsm_qwen3_1.7b_cot.yaml "${qwen_stage0}")
qwen_coconut=$(submit_branch \
  gsm-qwen-coconut args/gsm_qwen3_1.7b_coconut.yaml "${qwen_stage0}")
ouro_cot=$(submit_branch \
  gsm-ouro-r1-cot args/gsm_ouro_1.4b_r1_cot.yaml "${ouro_stage0}")
ouro_coconut=$(submit_branch \
  gsm-ouro-r1-coconut args/gsm_ouro_1.4b_r1_coconut.yaml "${ouro_stage0}")

qwen_cot=${qwen_cot%%;*}
qwen_coconut=${qwen_coconut%%;*}
ouro_cot=${ouro_cot%%;*}
ouro_coconut=${ouro_coconut%%;*}

printf '%-22s %s\n' \
  qwen_stage0 "${qwen_stage0}" \
  qwen_cot "${qwen_cot}" \
  qwen_coconut "${qwen_coconut}" \
  ouro_r1_stage0 "${ouro_stage0}" \
  ouro_r1_cot "${ouro_cot}" \
  ouro_r1_coconut "${ouro_coconut}"

echo
echo "Monitor with:"
echo "  squeue -j ${qwen_stage0},${qwen_cot},${qwen_coconut},${ouro_stage0},${ouro_cot},${ouro_coconut}"
