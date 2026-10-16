#!/bin/bash
#SBATCH --job-name=expert-data
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=expert-data-%j.log

# Games of the learner with search against a league, recorded for expert
# iteration (docs/expert-iteration.md). Extra arguments go to
# pokemon_ml.expert_data.
#
#   EXPERT_OUT=runs/expert/round1-smoke EXPERT_GAMES=200 \
#   sbatch $SLURM_OPTS --time=01:00:00 scripts/oscar/expert-data.sbatch
set -euo pipefail
module load python/3.12.8-zr3c
: "${POLICY_REPO:?Set POLICY_REPO to your checkout}"
: "${EXPERT_OUT:?Set EXPERT_OUT to a new directory for the data}"
cd "$POLICY_REPO"
source .venv/bin/activate
EXPERT_CHECKPOINT="${EXPERT_CHECKPOINT:-runs/plateau-comparison-15342/source-checkpoint.pt}"
EXPERT_POOL="${EXPERT_POOL:-teams/multi-tournament-warmstart/train.json}"
# The league: TARGET[+search]:weight, separated by spaces. Frozen checkpoints by path.
EXPERT_LEAGUE="${EXPERT_LEAGUE:-self+search:0.3 self:0.2 runs/learning-rate-12450-ab/reference/low-lr-update-13450.pt:0.2 runs/louisville-small/checkpoint.pt:0.15 greedy:0.15}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
league=()
for spec in $EXPERT_LEAGUE; do
  target="${spec%%:*}"
  target="${target%+search}"
  case "$target" in
    self|random|greedy|lookahead) ;;
    *) [[ -f "$target" ]] || { echo "opponent checkpoint not found: $target" >&2; exit 2; } ;;
  esac
  league+=(--opponent "$spec")
done
case "$EXPERT_POOL" in
  *eval*) echo "refusing an evaluation pool for training data: $EXPERT_POOL" >&2; exit 2 ;;
esac

python -m pokemon_ml.expert_data --checkpoint "$EXPERT_CHECKPOINT" --pool "$EXPERT_POOL" \
  --out "$EXPERT_OUT" --games "${EXPERT_GAMES:-10000}" --envs "${EXPERT_ENVS:-64}" \
  --threads "${SLURM_CPUS_PER_TASK:-8}" --device cuda "${league[@]}" "$@"
