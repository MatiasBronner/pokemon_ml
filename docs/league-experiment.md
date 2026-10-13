# Small-model league experiment

The diagnostic run improved broadly but latest vs update 4125 scored about 50%
across 6000 games. This experiment tests historical opponent diversity without
changing architecture, speed features, PPO settings or the 50/40/10 league mix.

Legacy selection keeps latest two, middle and oldest frozen snapshots. Broad
selection keeps latest two plus ten evenly spaced snapshots across the rest of
the registered history. Half of frozen draws go to the latest two; half go to
the historical pool. With four distinct legacy candidates this preserves the
50% recent / 50% historical balance. Selection stays fixed within an episode.
No performance-based selection is enabled: first measure the effect of diversity
and gather opponent-specific evidence, rather than change two things at once.

Terminal wins/losses/ties are recorded by opponent before assignments reset.
`league_results` in each update is a delta, not cumulative; checkpoint league
`results` is cumulative. Current self-play records one assigned learner side per
battle, not both zero-sum outcomes. These scores are from evolving policies and
sampled training teams; they are diagnostics, not fixed evaluation results.
Old checkpoints load without results or selection fields. An explicit
`--league-selection` can change a resumed run; omitted flags restore saved
selection. Use cloned runs for deliberate changes. The cache-size option controls
inference memory, not opponent probabilities; both experiment arms use 32 to avoid
cache churn when historical candidates are broadened. Defaults remain legacy/4.

## Oscar commands

After merging the change, on the login node with the existing environment active:

```bash
cd /users/mcbronne/pokemon_ml
git switch main
git pull --ff-only
module load python/3.12.8-zr3c
source .venv/bin/activate
python scripts/prepare-league-experiment.py \
  --source runs/multi-tournament-small-fp32 \
  --out runs/league-diversity-ab --updates 1000
export POLICY_REPO=/users/mcbronne/pokemon_ml
export POLICY_EXPERIMENT="$POLICY_REPO/runs/league-diversity-ab"
sbatch scripts/oscar/league-experiment.sbatch
```

The source must be finished, not actively writing checkpoints. The prepared
experiment freezes a source checkpoint, copies mutable state/logs, and hard-links
immutable snapshots (copies when hard-linking is unsupported). It refuses an
existing destination. Neither arm changes source weights or history. Both restore
identical optimizer/RNG/league state, then use their own future snapshots.

A three-hour job runs both arms sequentially on the same RTX 3090. Each arm aims
for 1000 additional updates, with a one-hour training-time cap. A fixed panel
(first, middle, three-quarter and latest registered source snapshots) plus greedy
is evaluated before training and after each arm: 1500 games per opponent, common
seed 9702347, current-format held-out pool. Outputs are under
`runs/league-diversity-ab/evaluation/{source,legacy,broad}`. The source evaluation
uses the frozen copy, not a live external checkpoint. Models remain FP32 SDPA
compiled, with deferred metrics and non-fused Adam in both arms.

If either arm hits its hour cap before the target, compare checkpoints at a common
update and account for learner decisions as well as wall time. Self-play opponent
mix affects the number of eligible learner rows, and inference overhead can make
the broad league slower. Short results establish neither a plateau nor saturation.
Use direct broad-vs-legacy matches and repeat promising panel results with another
seed before adopting a new default. Initial 1500-game intervals are exploratory;
small per-team/per-opponent samples should not drive adaptive sampling yet.

The job wrapper is syntax-checked locally. Training/clone/resume tests run on CPU;
the full CUDA experiment and its throughput must be measured on Oscar. No Oscar
jobs are submitted by the implementation.
