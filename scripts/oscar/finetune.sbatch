#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=finetune-%j.log

# Self-play (PPO) from a distilled network, with the network it was distilled
# from always in the league as a fixed opponent (docs/distillation.md). Submit it
# again with the same FINETUNE_RUN to go on where it stopped. Extra arguments go to
# pokemon_ml.train.
#
#   FINETUNE_INIT=runs/distill/large/best.pt FINETUNE_RUN=runs/finetune/large \
#   sbatch $SLURM_OPTS scripts/oscar/finetune.sbatch
set -euo pipefail
module load python/3.12.8-zr3c
: "${POLICY_REPO:?Set POLICY_REPO to your checkout}"
: "${FINETUNE_RUN:?Set FINETUNE_RUN to a directory for the run}"
cd "$POLICY_REPO"
source .venv/bin/activate
FINETUNE_FIXED="${FINETUNE_FIXED:-runs/plateau-comparison-15342/source-checkpoint.pt:0.2}"
FINETUNE_POOL="${FINETUNE_POOL:-teams/multi-tournament-warmstart/train.json}"
FINETUNE_EVAL_POOL="${FINETUNE_EVAL_POOL:-teams/multi-tournament-warmstart/eval-current.json}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

start=()
if [[ -f "$FINETUNE_RUN/checkpoint.pt" ]]; then
  echo "resuming $FINETUNE_RUN"
else
  : "${FINETUNE_INIT:?Set FINETUNE_INIT to the distilled checkpoint for a new run}"
  mkdir -p "$FINETUNE_RUN"
  start=(--init-checkpoint "$FINETUNE_INIT")
fi
fixed=()
for spec in $FINETUNE_FIXED; do fixed+=(--league-fixed "$spec"); done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
policy_pid=""
trap 'if [[ -n "$policy_pid" ]]; then kill -USR1 "$policy_pid" 2>/dev/null || true; fi' USR1 TERM INT
python -m pokemon_ml.train --pool "$FINETUNE_POOL" --eval-pool "$FINETUNE_EVAL_POOL" --run "$FINETUNE_RUN" \
  "${start[@]}" --league-current "${FINETUNE_CURRENT:-0.4}" --league-frozen "${FINETUNE_FROZEN:-0.3}" \
  --league-scripted "${FINETUNE_SCRIPTED:-0.1}" "${fixed[@]}" --device cuda \
  --threads "${SLURM_CPUS_PER_TASK:-8}" --envs "${FINETUNE_ENVS:-512}" --updates "${FINETUNE_UPDATES:-100000}" \
  --hours "${FINETUNE_HOURS:-7.75}" --save-every 10 --no-lookahead-eval "$@" &
policy_pid=$!
while true; do
  wait "$policy_pid" && status=0 || status=$?
  if ! kill -0 "$policy_pid" 2>/dev/null; then exit "$status"; fi
done
