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

The preparation guard checks source update and that control LR matches the source, avoiding an
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

## Reported outcome, October 13, 2026

Both arms reached update 7132. Low-LR scored 55.3% [53.5%, 57.0%] against
the frozen source; control scored 50.3% [48.5%, 52.1%], each across 3000 games.
Low-LR also improved across the historical panel. It is the leading candidate,
pending direct low-LR-vs-control and second-seed source comparisons. Full results
and limitations are in [the training findings](training-findings-2026-10-13.md#controlled-learning-rate-experiment).


## Second reduction after update 12450

The longer 0.0001 continuation saved update 12450 with 33,952,065 cumulative games.
Update 9450 scored 54.1% against 7132 across two 3000-game seeds. Final 12450 scored
52.0% against 9450 across two further seeds, suggesting modest progress over that
longer interval. Against 11450, however, it scored 51.3% and 48.77% (combined
50.03% over 6000 games). No detectable advantage over the last 1000 updates was
found on this benchmark. This supports testing another reduction, not assuming
zero progress everywhere or that LR is the sole cause.

The new experiment compares source LR **0.0001** with **0.00003**, both restoring
update 12450 and targeting **13450**. The original run remains intact. Preparation
now accepts configurable rates; control must equal the checkpoint's saved rate,
and the lower rate must be finite, positive and strictly below it. Original
experiment commands retain their defaults. The wrapper additionally runs direct
low-LR-vs-control matches with seeds 10702347 and 10802347, 3000 games each, after
both arms complete. It inherits the source's evaluation settings for those matches.

After merging, on Oscar's login node:

```bash
cd /users/mcbronne/pokemon_ml
git switch main
git pull --ff-only
module load python/3.12.8-zr3c
source .venv/bin/activate

python scripts/prepare-league-experiment.py \
  --source runs/learning-rate-ab/low-lr \
  --out runs/learning-rate-12450-ab --experiment learning-rate \
  --expected-update 12450 --updates 1000 \
  --control-lr 0.0001 --low-lr 0.00003

export POLICY_REPO=/users/mcbronne/pokemon_ml
export POLICY_EXPERIMENT="$POLICY_REPO/runs/learning-rate-12450-ab"
LR_JOB=$(sbatch --parsable scripts/oscar/learning-rate-experiment.sbatch)
echo "$LR_JOB"
```

### Reported completion and outcome

The initial job 7379274 stopped low-LR at update 13151 at its saving warning, even
though scheduler accounting reported a clean exit. Job 7392303 resumed it to 13450
and completed in 00:41:44 with state COMPLETED and exit 0:0. Both arms therefore
finished their matched 1000 additional updates. The resume used a new allocation.

Low-LR at 0.00003 scored **54.4% [52.6%, 56.2%]** against the frozen source and
**85.9% [84.6%, 87.1%]** against greedy, 3000 games each. Direct comparisons
against the 0.0001 control scored **53.7% [51.9%, 55.5%]** and
**54.4% [52.6%, 56.2%]** over two 3000-game seeds, with zero timeouts. Their
combined score was 54.1%. These support adopting 0.00003 for this continuation.

Preserve update-13450 `runs/learning-rate-12450-ab/low-lr/checkpoint.pt` before
longer training, retaining its optimizer and legacy league. Keep source/control
artifacts too. The reduction's success does not imply further automatic reductions
or unlimited progress; independent training seeds and long-term behavior remain
untested. The [consolidated training report](training-findings-2026-10-13.md)
contains the full latest panel, earlier continuation checks and limitations.
