# OSCAR: update-15342 plateau comparison

The user reports that the finished update-15342 policy showed no detectable
advantage over update 13450 across 6000 games. That supplies the plateau diagnosis;
**skip additional diagnostic training**. Those evaluation artifacts are not present
in this cloud checkout, so this is a user-reported result, not an independent rerun.

Prepare all three continuations from the same preserved update-15342 checkpoint:

| Arm | Learning rate | Historical opponent selection |
|---|---|---|
| control | Constant 0.00003 | Legacy |
| decay | Exponential 0.00003 to 0.00001 | Legacy |
| targeted | Constant 0.00003 | Difficulty weighted |

All retain **50% current / 40% frozen / 10% scripted** opponents. This replaces
the proposed 60/30/10 experiment; do not change proportions in these runs.

## Prepare on OSCAR

After merging the implementation PR, update the OSCAR checkout with
`git switch main` and `git pull --ff-only` before running these commands.

On the login node, activate the existing environment and set the actual finished
source directory (it must contain config.json, checkpoint.pt and snapshots/):

```bash
cd /users/mcbronne/pokemon_ml
module load python/3.12.8-zr3c
source .venv/bin/activate
export POLICY_REPO="$PWD"
export POLICY_SOURCE=/absolute/path/to/finished-update-15342-run
export POLICY_EXPERIMENT="$PWD/runs/plateau-comparison-15342"
python scripts/plateau-experiment.py prepare \
  --source "$POLICY_SOURCE" --expected-update 15342 \
  --updates 1000 --stage compare --experiment "$POLICY_EXPERIMENT"
```

Preparation requires source LR 0.00003, legacy selection and 50/40/10. Evaluation
requires the source to specify its actual held-out pool. Preparation verifies the exact source update, copies mutable
state, preserves optimizer/RNG/counters, and links or copies immutable snapshots.
The source must not be actively training. It refuses an existing destination and
records a hash of the preserved source checkpoint. Keep artifacts on persistent
storage outside scratch-purge risk. All arms target **update 16342**.

## Submit matched continuations

The following allocation options match the project's prior LR wrapper. Replace
partition/account/QOS/constraint if your OSCAR allocation requires different ones.
The wrapper requests one GPU, four CPUs, 32 GB, four hours and a five-minute
checkpoint warning; it activates the existing venv.

```bash
unset POLICY_ACTION
export POLICY_ARM=control
CONTROL_JOB=$(sbatch --parsable --partition=gpu --account=default --qos=norm-gpu \
  --constraint=geforce3090 scripts/oscar/plateau-experiment.sbatch)
for arm in decay targeted; do
  POLICY_ARM="$arm" sbatch --dependency="afterok:$CONTROL_JOB" \
    --partition=gpu --account=default --qos=norm-gpu --constraint=geforce3090 \
    scripts/oscar/plateau-experiment.sbatch
done
```

Control must finish before direct-control evaluations of the other arms. If it
is interrupted, resubmit it and submit dependents with the successful replacement
job ID; old dependencies do not migrate. Never run two jobs writing the same arm.
Use `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed` and inspect logs.
A clean scheduler exit alone does not establish completed training: require update
16342 and all evaluation reports.

An interrupted training job saves at an update boundary and exits nonzero.
Resubmit the same job; do not prepare again. Live battles restart on resume, so
continuation is not bit-identical to uninterrupted training. If only evaluation
needs rerunning, set POLICY_ACTION=evaluate and submit that arm's wrapper; unset
it before subsequent training. Evaluations can exceed the remaining allocation;
rerunning evaluation rewrites that arm's reports without training.

## What changes and what stays fixed

Decay uses absolute start/end updates stored with the checkpoint. It cannot restart
on resubmission and stays at the floor after update 16342. ppo_config.lr retains
the initial rate; each log row and optimizer state record the actual rate.
Changing a saved schedule is rejected.

Targeted uses the same bounded candidates as legacy: latest two, middle and oldest.
For each candidate, p=(wins + 0.5*ties + 25)/(games + 50), with weight proportional
to p*(1-p), mixed with 20% uniform sampling. It favors competitive matches while
retaining an opponent sampling floor and shrinking sparse estimates. Selection
uses cumulative training outcomes only, never held-out data, and opponents remain
fixed within an episode. Cumulative results can become stale. This is a modest
PFSP-style experiment, not a full AlphaStar league or exploiter implementation.

All arms keep architecture, rewards, PPO coefficients and team sampling unchanged.
Execution matches prior controlled LR jobs: FP32 SDPA compiled CUDA, deferred
metrics, non-fused Adam and opponent cache 32. Common initial RNG does not imply
identical later trajectories. Jobs request the same GPU model but may use different
physical cards. No 60/30/10, search, exploiters or architecture changes are included.

## Evaluation and interpretation

Each arm is evaluated on the source's held-out pool against the frozen source,
greedy and four historical panel members: 3000 games per opponent for each seed
10702347 and 10802347. Decay and targeted also play completed control directly.
Outputs preserve both seeds under evaluation/ARM/. Require the expected completed
game counts and zero unexplained timeouts. Compare direct-control, source and
historical scores, actual additional learner decisions, training seconds and wall
seconds printed by the runner. Equal updates are not equal samples or GPU hours.

Predeclare a useful effect size (for example, two percentage points against
control). Around 6000 independent games at 50%, uncertainty is roughly 1.3 points;
team-level and training-seed uncertainty can be larger. These are evaluation seeds,
not independent training seeds. Do not adopt a method on a single small panel gain.
Confirm a promising result with an independent training seed before calling it robust.

## Local validation and limits

The previous implementation run passed 49 Python tests with one optional
live-server test skipped. The three focused tests also passed after final changes,
including real CPU training, partial-schedule resume, source preservation, targeted
sampling and evaluation subprocesses writing six reports. The Slurm wrapper passed
`bash -n`. These local CPU results do not establish CUDA performance or
improved policy strength. No OSCAR
jobs have been submitted from the cloud machine. The optional diagnose stage remains
available in the CLI for future experiments; it is not part of this run.
