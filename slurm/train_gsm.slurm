#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:59:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/users/smehra/looped_latent_reasoning/coconut/logs/slurm/%x-%j.out
#SBATCH --error=/users/smehra/looped_latent_reasoning/coconut/logs/slurm/%x-%j.err

set -euo pipefail

readonly REPO_DIR=/users/smehra/looped_latent_reasoning/coconut
: "${CONFIG:?Submit with --export=ALL,CONFIG=args/<config>.yaml}"

cd "${REPO_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config does not exist: ${REPO_DIR}/${CONFIG}" >&2
  exit 2
fi
if [[ ! -s data/gsm_train.json || ! -s data/gsm_valid.json ]]; then
  echo "Prepared GSM8K data is missing; see preprocessing/gsm_icot.bash" >&2
  exit 2
fi

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} host=${MASTER_ADDR} config=${CONFIG} gpus=${NPROC_PER_NODE}"

srun --ntasks=1 --ntasks-per-node=1 \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --kill-on-bad-exit=1 \
  --environment=test-env \
  bash -c '
    python -m torch.distributed.run \
      --nnodes=1 \
      --nproc-per-node="${NPROC_PER_NODE}" \
      --node-rank=0 \
      --master-addr="${MASTER_ADDR}" \
      --master-port="${MASTER_PORT}" \
      run.py "${CONFIG}"
  '
