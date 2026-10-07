#!/bin/bash
#SBATCH --job-name=coconut-safe-opt-bench
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=11:59:00
#SBATCH --output=/users/smehra/looped_latent_reasoning/coconut-safe-optimizations/logs/slurm/%x-%j.out
#SBATCH --error=/users/smehra/looped_latent_reasoning/coconut-safe-optimizations/logs/slurm/%x-%j.err

set -euo pipefail

readonly REPO_DIR=/users/smehra/looped_latent_reasoning/coconut-safe-optimizations
cd "${REPO_DIR}"

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

srun --ntasks=1 --ntasks-per-node=1 \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --kill-on-bad-exit=1 \
  --environment=test-env \
  bash slurm/run_safe_optimization_benchmarks.sh
