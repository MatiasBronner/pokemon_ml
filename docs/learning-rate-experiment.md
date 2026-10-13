# Small-model learning-rate experiment

The original small-model run plateaued around updates 4125–6132 on tested
head-to-head benchmarks. A broader historical league did not produce a detectable
improvement and substantially reduced throughput. This next experiment tests a
lower PPO learning rate, not a change of model size, speed features or opponent
selection. It is a hypothesis; prior logs do not establish that LR caused the
plateau or predict that reducing it will help.

## Design

Prepare two independent clones of the finished source at update 6132. Both retain
weights, optimizer moments, RNG state, counters and registered snapshot history:

- `control`: learning rate 0.0003.
- `low-lr`: learning rate 0.0001.

Both use legacy league selection, the existing 50/40/10 mix, unchanged PPO settings
otherwise, speed features, weighted dataset and held-out pool. Both use FP32 SDPA
compiled execution, deferred metrics, non-fused Adam and cache size 32. The cache
setting matches the previous controlled experiment and is identical across arms;
it changes capacity for caching opponents, not their selection probabilities.

Target 1000 additional updates per arm, i.e. update 7132 from source 6132. There
are no separate arm time caps. A four-hour Slurm allocation runs both arms
sequentially on one RTX 3090. Training may terminate safely at the scheduler
warning; the wrapper does not label an incomplete arm as an equal-update result.
If interrupted, resubmit the same wrapper against the existing experiment to
continue remaining updates. Do not rerun preparation, which refuses an existing
output. Completed arms may be reevaluated on resubmission. Common initial RNG does
not imply identical trajectories after policies diverge, nor identical eligible
learner decisions, so record decisions and GPU time in addition to updates.

## Run on Oscar

After merging, on the login node:

```bash
cd /users/mcbronne/pokemon_ml
git switch main
git pull --ff-only
module load python/3.12.8-zr3c
source .venv/bin/activate

python scripts/prepare-league-experiment.py \
  --source runs/multi-tournament-small-fp32 \
  --out runs/learning-rate-ab --experiment learning-rate \
  --expected-update 6132 --updates 1000

export POLICY_REPO=/users/mcbronne/pokemon_ml
export POLICY_EXPERIMENT="$POLICY_REPO/runs/learning-rate-ab"
LR_JOB=$(sbatch --parsable scripts/oscar/learning-rate-experiment.sbatch)
echo "$LR_JOB"
squeue -j "$LR_JOB"
```

The preparation guard checks source update and learning rate (0.0003), avoiding an
accidental change of baseline. The source must not be actively training. The
original checkpoint is preserved, mutable state is copied, and immutable
snapshots are linked or copied. No new Rust build is needed for these Python-only
experiment scripts if the earlier weighted-pool extension is already installed.

Monitor `lr-ab-JOBID.log`; accounting is available via
`sacct -j JOBID --format=JobID,State,ExitCode,Elapsed`.

## Evaluation and interpretation

The source and both arms are evaluated against the same frozen historical panel
and greedy, 3000 games per opponent with seed 9702347. Each continuation is also
evaluated directly against the immutable source checkpoint. JSON outputs are
under `runs/learning-rate-ab/evaluation/{source,control,low-lr}`; `source.json` in
each arm directory is its direct matchup against the source.

Review exact final update, learner decisions, games and elapsed training time
before comparing. A greedy-only gain does not establish a stronger general policy.
Promising direct-source and panel results should be confirmed with a second seed
and a direct low-LR-vs-control matchup. Multiple panel comparisons and team-level
small samples warrant caution. Do not adopt low LR by default on a single small
score difference. If neither arm improves, preserve the source and investigate
other hypotheses rather than repeat unchanged long runs.

Local validation exercises actual CPU training/resume, optimizer LR restoration,
source protection and the experiment's real evaluation subprocess including turn
limits and direct-source matches. The scheduler wrapper is syntax-checked. Full
CUDA runtime, duration and learning outcomes must be verified on Oscar; no cluster
jobs were launched while implementing these scripts.
