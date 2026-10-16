#!/bin/bash
#SBATCH --job-name=expert-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=expert-train-%A_%a.log

# Expert iteration's training step, one learning rate per array task
# (docs/expert-iteration.md). Extra arguments go to pokemon_ml.expert_train.
#
#   EXPERT_DATA=runs/expert/round1 EXPERT_TRAIN_OUT=runs/expert/round1-train \
#   sbatch $SLURM_OPTS scripts/oscar/expert-train.sbatch
set -euo pipefail
module load python/3.12.8-zr3c
: "${POLICY_REPO:?Set POLICY_REPO to your checkout}"
: "${EXPERT_DATA:?Set EXPERT_DATA to the data directories, separated by spaces}"
: "${EXPERT_TRAIN_OUT:?Set EXPERT_TRAIN_OUT to a directory for the runs}"
cd "$POLICY_REPO"
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

read -r -a lrs <<< "${EXPERT_LRS:-3e-5 1e-4 3e-4}"
lr="${lrs[${SLURM_ARRAY_TASK_ID:-0}]}"
data=()
for d in $EXPERT_DATA; do
  [[ -f "$d/manifest.json" ]] || { echo "no manifest.json in $d: is the data finished?" >&2; exit 2; }
  data+=(--data "$d")
done
if [[ -n "${EXPERT_SIZE:-}" ]]; then
  start=(--model-size "$EXPERT_SIZE")        # a fresh network of this size
  name="$EXPERT_SIZE-lr$lr"
else
  start=(--init "${EXPERT_INIT:-runs/plateau-comparison-15342/source-checkpoint.pt}")
  name="lr$lr"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m pokemon_ml.expert_train "${data[@]}" "${start[@]}" --out "$EXPERT_TRAIN_OUT/$name" \
  --lr "$lr" --device cuda "$@"
