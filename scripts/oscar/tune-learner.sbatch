#!/bin/bash
#SBATCH --job-name=policy-tune
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=tune-%j.log
set -euo pipefail
module load python/3.12.8-zr3c
: "${POLICY_REPO:?Set POLICY_REPO to your checkout}"
: "${POLICY_POOL:?Set POLICY_POOL to the training pool}"
cd "$POLICY_REPO"
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/pokemon-policy-compile-${SLURM_JOB_ID}"
mkdir -p runs/learner-tuning
out="runs/learner-tuning/job-${SLURM_JOB_ID}"
args=(--pool "$POLICY_POOL" --out "$out" --device cuda --envs "${POLICY_ENVS:-512}"
      --threads "$OMP_NUM_THREADS" --steps 64 --repeats 3 --profile --include-compile)
if [[ -n "${POLICY_CHECKPOINT:-}" ]]; then
  args+=(--checkpoint "$POLICY_CHECKPOINT")
fi
if [[ -n "${POLICY_LEAGUE_RUN:-}" ]]; then
  args+=(--league-run "$POLICY_LEAGUE_RUN")
fi
nvidia-smi
python -m pokemon_ml.tune_learner "${args[@]}"
