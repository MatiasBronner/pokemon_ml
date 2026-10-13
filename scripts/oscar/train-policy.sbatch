#!/bin/bash
#SBATCH --job-name=pokemon-policy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=policy-%j.log

# Supply Oscar partition/account/QOS through sbatch options appropriate for
# your actual allocation. Activate your verified Python/Rust environment first.
set -euo pipefail
: "${POLICY_REPO:?Set POLICY_REPO to your checkout}"
: "${POLICY_POOL:?Set POLICY_POOL to your training pool}"
: "${POLICY_EVAL_POOL:?Set POLICY_EVAL_POOL to your held-out pool}"
: "${POLICY_RUN:?Set POLICY_RUN to persistent run storage}"
cd "$POLICY_REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Forward the batch warning to Python. Finish a safe update and save before exit.
policy_pid=""
trap 'if [[ -n "$policy_pid" ]]; then kill -USR1 "$policy_pid" 2>/dev/null || true; fi' USR1 TERM INT
python -m pokemon_ml.train --pool "$POLICY_POOL" --eval-pool "$POLICY_EVAL_POOL" \
  --run "$POLICY_RUN" --model-size "${POLICY_SIZE:-small}" --device cuda \
  --threads "${SLURM_CPUS_PER_TASK:-4}" --envs "${POLICY_ENVS:-512}" \
  --updates "${POLICY_UPDATES:-10000}" --hours "${POLICY_HOURS:-2.75}" \
  --save-every 10 --no-lookahead-eval "$@" &
policy_pid=$!
# wait can be interrupted by the signal trap; continue waiting for checkpoint completion.
while true; do
  wait "$policy_pid" && policy_status=0 || policy_status=$?
  if ! kill -0 "$policy_pid" 2>/dev/null; then exit "$policy_status"; fi
done
