#!/bin/bash
#SBATCH --job-name=lr-ab
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --qos=norm-gpu
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=lr-ab-%j.log

set -euo pipefail
module load python/3.12.8-zr3c
: "${POLICY_REPO:?Set POLICY_REPO to your checkout}"
: "${POLICY_EXPERIMENT:?Set POLICY_EXPERIMENT to the prepared experiment directory}"
cd "$POLICY_REPO"
source "$POLICY_REPO/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TORCHINDUCTOR_COMPILE_THREADS=1
nvidia-smi
policy_pid=""
trap 'if [[ -n "$policy_pid" ]]; then kill -USR1 "$policy_pid" 2>/dev/null || true; fi; exit_requested=1' USR1 TERM INT
exit_requested=0
# Both arms share this GPU. Settings/RNG/optimizer come from the same source.
python scripts/evaluate-league-experiment.py --experiment "$POLICY_EXPERIMENT" --arm source --games 3000
for arm in control low-lr; do
  [[ "$exit_requested" == 0 ]] || break
  echo "Starting learning-rate arm: $arm"
  python scripts/run-learning-rate-arm.py --experiment "$POLICY_EXPERIMENT" --arm "$arm" &
  policy_pid=$!
  while true; do
    wait "$policy_pid" && status=0 || status=$?
    if ! kill -0 "$policy_pid" 2>/dev/null; then break; fi
  done
  policy_pid=""
  [[ "$status" == 0 ]] || exit "$status"
  [[ "$exit_requested" == 0 ]] || break
  python - "$POLICY_EXPERIMENT" "$arm" <<'PY'
import json, sys
from pathlib import Path
import torch
root, arm = Path(sys.argv[1]), sys.argv[2]
manifest = json.loads((root / 'experiment.json').read_text())
saved = torch.load(root / arm / 'checkpoint.pt', map_location='cpu', weights_only=False)
if saved['update'] != manifest['target_update']:
    raise SystemExit(f"Incomplete arm {arm}: {saved['update']} instead of {manifest['target_update']}; resume before comparing")
print(f"Completed {arm}: update={saved['update']}, lr={saved['ppo_config']['lr']}", flush=True)
PY
  python scripts/evaluate-league-experiment.py --experiment "$POLICY_EXPERIMENT" --arm "$arm" --games 3000 --include-source
done

if [[ "$exit_requested" == 0 ]]; then
  for match_seed in 10702347 10802347; do
    echo "Evaluating low-lr directly against control, seed $match_seed"
    python scripts/evaluate-league-experiment.py \
      --experiment "$POLICY_EXPERIMENT" --arm low-lr --control-only \
      --games 3000 --seed "$match_seed"
  done
fi
