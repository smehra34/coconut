#!/bin/bash
#SBATCH --job-name=coconut-safe-opt-test
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=11:59:00
#SBATCH --output=/users/smehra/looped_latent_reasoning/coconut-safe-optimizations/logs/slurm/%x-%j.out
#SBATCH --error=/users/smehra/looped_latent_reasoning/coconut-safe-optimizations/logs/slurm/%x-%j.err

set -euo pipefail

readonly REPO_DIR=/users/smehra/looped_latent_reasoning/coconut-safe-optimizations
cd "${REPO_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

srun --ntasks=1 --ntasks-per-node=1 \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --kill-on-bad-exit=1 \
  --environment=test-env \
  bash -c '
    python -m unittest tests.test_coconut_optimizations tests.test_stage_cache
    python tests/generation_cache_smoke.py --model all --max-new-tokens 32
  '
