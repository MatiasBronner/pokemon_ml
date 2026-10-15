#!/bin/bash
#SBATCH --job-name=pokemon-search
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=search-%j.log

# The same checkpoint with one-turn search against itself without it
# (docs/search.md). Supply partition/account/QOS through sbatch options for
# your allocation. Activate your verified Python/Rust environment first and
# rebuild the extension (maturin develop --release) on this branch.
#
#   SEARCH_CHECKPOINT=runs/multi-tournament-small-fp32/snapshots/015342.pt \
#   SEARCH_POOL=teams/split/eval.json SEARCH_OUT=runs/search/15342-vs-self-seed1.json \
#   sbatch scripts/oscar/search-eval.sbatch --seed 9502347
set -euo pipefail
: "${POLICY_REPO:?Set POLICY_REPO to your checkout}"
: "${SEARCH_CHECKPOINT:?Set SEARCH_CHECKPOINT to the checkpoint that searches}"
: "${SEARCH_POOL:?Set SEARCH_POOL to the evaluation pool}"
: "${SEARCH_OUT:?Set SEARCH_OUT to the report path}"
cd "$POLICY_REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
python -m pokemon_ml.search --checkpoint "$SEARCH_CHECKPOINT" \
  --opponent "${SEARCH_OPPONENT:-self}" --pool "$SEARCH_POOL" --out "$SEARCH_OUT" \
  --games "${SEARCH_GAMES:-3000}" --envs "${SEARCH_ENVS:-64}" --open-sheets 1.0 \
  --threads "${SLURM_CPUS_PER_TASK:-8}" --device cuda "$@"
