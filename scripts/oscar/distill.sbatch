#!/bin/bash
#SBATCH --job-name=distill
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=distill-%j.log

# A fresh, bigger network taught to play like a trained one (docs/distillation.md).
# Extra arguments go to pokemon_ml.distill.
#
#   DISTILL_OUT=runs/distill/large sbatch $SLURM_OPTS scripts/oscar/distill.sbatch
set -euo pipefail
module load python/3.12.8-zr3c
: "${POLICY_REPO:?Set POLICY_REPO to your checkout}"
: "${DISTILL_OUT:?Set DISTILL_OUT to a new directory for the student}"
cd "$POLICY_REPO"
source .venv/bin/activate
DISTILL_TEACHER="${DISTILL_TEACHER:-runs/plateau-comparison-15342/source-checkpoint.pt}"
DISTILL_POOL="${DISTILL_POOL:-teams/multi-tournament-warmstart/train.json}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
case "$DISTILL_POOL" in
  *eval*) echo "refusing an evaluation pool for training: $DISTILL_POOL" >&2; exit 2 ;;
esac
[[ -f "$DISTILL_TEACHER" ]] || { echo "teacher not found: $DISTILL_TEACHER" >&2; exit 2; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m pokemon_ml.distill --teacher "$DISTILL_TEACHER" --model-size "${DISTILL_SIZE:-large}" \
  --pool "$DISTILL_POOL" --out "$DISTILL_OUT" --hours "${DISTILL_HOURS:-3.5}" \
  --steps "${DISTILL_STEPS:-200000}" --threads "${SLURM_CPUS_PER_TASK:-8}" --device cuda "$@"
