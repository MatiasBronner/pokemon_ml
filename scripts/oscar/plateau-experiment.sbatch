#!/bin/bash
#SBATCH --job-name=policy-plateau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=policy-plateau-%j.log

set -euo pipefail
module load python/3.12.8-zr3c
: "${POLICY_REPO:?Set POLICY_REPO}"
: "${POLICY_EXPERIMENT:?Set POLICY_EXPERIMENT}"
: "${POLICY_ARM:=control}"
cd "$POLICY_REPO"
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TORCHINDUCTOR_COMPILE_THREADS=1
if [[ "${POLICY_ACTION:-train}" == evaluate ]]; then
  exec python scripts/plateau-experiment.py evaluate --experiment "$POLICY_EXPERIMENT" --arm "$POLICY_ARM"
fi
policy_pid=""
stopping=0
trap 'stopping=1; if [[ -n "$policy_pid" ]]; then kill -USR1 "$policy_pid" 2>/dev/null || true; fi' USR1 TERM INT
python scripts/plateau-experiment.py train --experiment "$POLICY_EXPERIMENT" --arm "$POLICY_ARM" &
policy_pid=$!
while true; do
  wait "$policy_pid" && status=0 || status=$?
  if ! kill -0 "$policy_pid" 2>/dev/null; then break; fi
done
policy_pid=""
# An interrupted training run is resumable, not a completed experiment.
[[ "$stopping" == 0 ]] || exit 3
[[ "$status" == 0 ]] || exit "$status"
python scripts/plateau-experiment.py evaluate --experiment "$POLICY_EXPERIMENT" --arm "$POLICY_ARM"
