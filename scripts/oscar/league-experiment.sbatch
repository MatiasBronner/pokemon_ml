#!/bin/bash
#SBATCH --job-name=league-ab
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --qos=norm-gpu
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=league-ab-%j.log

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
python scripts/evaluate-league-experiment.py --experiment "$POLICY_EXPERIMENT" --arm source
for arm in legacy broad; do
  [[ "$exit_requested" == 0 ]] || break
  echo "Starting league arm: $arm"
  python - "$POLICY_EXPERIMENT" "$arm" <<'PY' &
import json
import sys
from pathlib import Path
from pokemon_ml.train import main

root, arm = Path(sys.argv[1]), sys.argv[2]
manifest = json.loads((root / 'experiment.json').read_text())
a = manifest['source_args']
args = ['--run', str(root / arm), '--pool', a['pool'],
        '--updates', str(manifest['target_update']), '--hours', '1',
        '--device', 'cuda', '--precision', 'fp32', '--attention', 'sdpa',
        '--compile-encoder', '--deferred-metrics', '--no-fused-adam',
        '--league-selection', arm, '--league-pool-size', '12', '--league-cache-size', '32']
for key in ('eval_pool', 'envs', 'threads', 'seed', 'eval_seed', 'open_sheets',
            'max_turns', 'eval_every', 'eval_games', 'save_every', 'snapshot_every',
            'league_current', 'league_frozen', 'league_scripted', 'archetypes'):
    if a.get(key) is not None:
        args.extend(['--' + key.replace('_', '-'), str(a[key])])
for key in ('no_vary', 'no_lookahead_eval'):
    if a.get(key):
        args.append('--' + key.replace('_', '-'))
main(args)
PY
  policy_pid=$!
  while true; do
    wait "$policy_pid" && status=0 || status=$?
    if ! kill -0 "$policy_pid" 2>/dev/null; then break; fi
  done
  policy_pid=""
  [[ "$status" == 0 ]] || exit "$status"
  [[ "$exit_requested" == 0 ]] || break
  python scripts/evaluate-league-experiment.py --experiment "$POLICY_EXPERIMENT" --arm "$arm"
done
