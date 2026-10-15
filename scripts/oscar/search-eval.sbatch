#!/bin/bash
#SBATCH --job-name=pokemon-search
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=search-%j.log

# The same checkpoint with one-turn search against itself without it
# (docs/search.md). Rebuild the extension on the login node first
# (maturin develop --release). Extra arguments go to pokemon_ml.search.
#
#   SEARCH_OUT=runs/search/15342-self-seed9502347.json \
#   sbatch --partition=gpu --account=default --qos=norm-gpu --constraint=geforce3090 \
#     scripts/oscar/search-eval.sbatch --seed 9502347
set -euo pipefail
module load python/3.12.8-zr3c
: "${POLICY_REPO:?Set POLICY_REPO to your checkout}"
: "${SEARCH_OUT:?Set SEARCH_OUT to the report path}"
cd "$POLICY_REPO"
source .venv/bin/activate
SEARCH_CHECKPOINT="${SEARCH_CHECKPOINT:-runs/plateau-comparison-15342/source-checkpoint.pt}"
SEARCH_POOL="${SEARCH_POOL:-teams/multi-tournament-warmstart/eval-current.json}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python - "$SEARCH_CHECKPOINT" <<'EOF'
import sys, torch
from pokemon_ml import search  # fails here if the extension predates search
saved = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"checkpoint {sys.argv[1]}: update {saved.get('update')}, cuda {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("no GPU visible to PyTorch")
EOF

mkdir -p "$(dirname "$SEARCH_OUT")"
python -m pokemon_ml.search --checkpoint "$SEARCH_CHECKPOINT" \
  --opponent "${SEARCH_OPPONENT:-self}" --pool "$SEARCH_POOL" --out "$SEARCH_OUT" \
  --games "${SEARCH_GAMES:-3000}" --envs "${SEARCH_ENVS:-64}" --open-sheets 1.0 \
  --threads "${SLURM_CPUS_PER_TASK:-8}" --device cuda "$@"
