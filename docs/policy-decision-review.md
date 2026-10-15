# Small-policy decision and representation review

Status: code review and diagnostic tooling, October 15, 2026. No trained Oscar
checkpoint or detailed battle traces were available in this workspace. The
findings below distinguish verified representation properties from hypotheses
about the observed plateau. No policy architecture, training default, reward,
league default or observation contract has been changed by this review.

## Completed plateau experiment

All arms restored source update 15342 and reached 16342, retaining the small
model, speed features, weighted teams, PPO settings otherwise and 50/40/10 mix.
Control used constant LR 3e-5 and legacy selection. Decay exponentially reduced
LR to 1e-5 with legacy selection. Targeted kept LR 3e-5 and weighted the same
legacy candidates by competitive difficulty, using shrinkage and a uniform floor.

Control job 7409369 completed 0:0 in 02:59:11. Decay job 7415315 completed 0:0 in
02:56:42; targeted 7415316 completed 0:0 in 02:56:36. Expected reports were present:
12 control, 14 decay, 14 targeted. All reported evaluations used 3000 games per
opponent for each of seeds 10702347 and 10802347 and had zero timeouts. Two-seed
mean scores (6000 games per comparison):

| Opponent | Control | Decay | Targeted |
|---|---:|---:|---:|
| Source update 15342 | 48.93% | 50.42% | 49.80% |
| Completed control | — | 50.67% | 50.80% |
| Greedy | 83.88% | 85.77% | 84.25% |
| Snapshot 000000 | 74.32% | 76.43% | 74.17% |
| Snapshot 007650 | 56.52% | 58.62% | 57.60% |
| Snapshot 011500 | 54.05% | 55.78% | 54.62% |
| Snapshot 015325 | 49.48% | 51.57% | 49.27% |

Direct-source and direct-control comparisons do not establish an improvement.
Decay's better greedy/older-opponent results do not establish a generally stronger
policy. This is one training continuation per method, not independent training
seed replication. Preserve every arm; retain source 15342 with constant 3e-5 and
legacy selection as the established reference pending decision-level evidence.

Training exposure and timer deltas printed by the runner:

| Arm | Learner decisions | Collection/update seconds | Run wall seconds |
|---|---:|---:|---:|
| Control | 41,200,365 | 4,062.17 | 10,419.59 |
| Decay | 41,016,082 | 4,032.20 | 10,252.37 |
| Targeted | 41,235,346 | 4,045.13 | 10,257.65 |

Approximately 61% of control wall time was outside the collection/update timing
regions. Periodic evaluation, compilation, saving and other overhead must be
profiled separately before assigning causes. This is an efficiency issue distinct
from a policy-learning plateau: removing observer features is not justified by
these numbers. Reducing evaluation cadence would change monitoring frequency,
not automatically improve learning per decision. Do not silently alter cadence
inside a controlled quality comparison.

## Code findings

### 1. Move identity and PP binding is lost for benched Pokémon

`src/obs.rs` retains four ordered move IDs and four corresponding PP values per
Pokémon. The observation therefore distinguishes which move has which PP.
`Model._encode` in `python/pokemon_ml/model.py` sums the four move embeddings into
a Pokémon token, while a separate linear projection consumes ordered numeric PP.
There is no per-move PP/identity pairing for benched Pokémon.

A focused test constructs two observations that swap a benched Pokémon's first
two move IDs while keeping unequal PP values in their original numeric slots.
These represent different move-to-PP associations, but the encoder outputs match
within float roundoff. This is a verified lossy representation, not an engine
mechanics bug or evidence that the observer omitted the information. Active
Pokémon also have explicit move tokens with move features; the demonstrated
collision concerns a Pokémon absent from active slots.

Potential remedy: encode each known move together with its corresponding PP
before pooling, or supply per-move tokens for the bench. Cost and weight migration
need measurement. PP exhaustion is often less important in short battles, so do
not assume this explains the broad plateau or prioritize a fix without traces.

### 2. Preview scoring has a specific additive constraint

With `joint_preview=False`, each of 90 preview logits is a lead-pair score plus a
reserve-pair score. Four compatible lead/reserve combinations therefore obey
`score(L1,B1)+score(L2,B2)-score(L1,B2)-score(L2,B1)=0` at a fixed encoded state.
A test verifies this identity. The transformer has already contextualized all
Pokémon, so preview does see opposing teams and team context; this is a head-level
restriction, not complete absence of synergy information.

Potential remedy: the existing `joint_preview` option introduces a nonlinear
joint lead/reserve scorer. It is disabled in the trained reference. Merely enabling
it is an architecture change and cannot be loaded as a strict full resume of the
existing checkpoint. It needs a controlled new experiment or explicit, validated
weight migration. Preview errors must first be separated from turn-play errors.

### 3. Doubles actions are already coordinated autoregressively

The policy samples `P(action0 | observation)`, then
`P(action1 | observation, action0)`. Legal second-action masks also depend on the
first choice. This factorization can represent a joint action distribution; it
is incorrect to describe the baseline as two independent policies.

The first action is currently supplied to the second head as an action-number
embedding. `semantic_actions=False` means its move identity, target representation
and selected switch-in are not additionally gathered into that conditioning
vector. They remain available indirectly through state tokens. Explicit semantic
conditioning may improve generalization across move-slot/team permutations, but
its absence is an inductive-bias hypothesis, not proven information loss of the
entire observation or a demonstrated strategic error.

### 4. No explicit sequence model, but the observer retains history summaries

`history=0` disables the optional public decision-history tokens. The input still
contains public reveals, timers, last visible attempted moves, and inferred speed
information. It is not correct to say all past information is absent. We have
not established whether information beyond these summaries is needed for the
observed failures. Hidden/opponent auxiliary prediction and prediction feedback
were also disabled; zero auxiliary losses are not failed training objectives.

### 5. Existing traces were insufficient for explaining alternatives

Old traces store observations, actions, value and outcome, but not legal masks or
alternative-action probabilities. New opt-in `--trace-policy-details` enriches
only traced evaluation decisions with human-readable chosen actions, their model
probabilities, top-five alternatives, public active HP/species and compact packed
legal masks. Slot-1 alternatives explicitly condition on the chosen slot-0 action;
they are not a ranked list of globally best joint actions. Original sampled
log-probabilities are retained; diagnostic probabilities are recomputed and can
have backend/batch-size numerical differences. Neither indicates optimality.

No simulator-private opponent state, actual opponent choice, future damage or
counterfactual reward is attached to the acting policy. Existing raw trace fields
remain actor-view observations. If configured, public history tensors are included.
Legal masks can be recovered with NumPy unpackbits using little bit order and the
recorded bit count. Default evaluation and all training avoid the extra encoding
work; no checkpoint parameters or observation contract change.

## Collect traces on Oscar

After merging this review/tooling PR, update main on the login node and activate
the existing venv. Rust source did not change, so a current extension needs no
rebuild. Keep training stopped. Use the immutable
`runs/plateau-comparison-15342/source-checkpoint.pt` rather than a mutable run head.
Request one debug GPU and run the following inside its allocated shell:

```bash
module load python/3.12.8-zr3c
cd /users/mcbronne/pokemon_ml
source .venv/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p runs/policy-decision-audit-15342

for RIVAL in greedy control; do
  if [[ "$RIVAL" == greedy ]]; then
    OPPONENT=greedy
  else
    OPPONENT=runs/plateau-comparison-15342/control/checkpoint.pt
  fi
  python -m pokemon_ml.evaluation \
    --checkpoint runs/plateau-comparison-15342/source-checkpoint.pt \
    --opponent "$OPPONENT" \
    --pool teams/multi-tournament-warmstart/eval-current.json \
    --games 128 --envs 64 --threads 4 \
    --seed 11202347 --device cuda \
    --trace-games 48 --trace-policy-details \
    --out "runs/policy-decision-audit-15342/vs-$RIVAL.json" || break
  python scripts/summarize-policy-traces.py \
    "runs/policy-decision-audit-15342/vs-$RIVAL.json" \
    --out "runs/policy-decision-audit-15342/vs-$RIVAL.md" || break
done
```

These are qualitative diagnostic games, not a new high-confidence strength
benchmark. The 48 trace slots are chosen before outcomes, not loss-selected; loss
counts can be small, especially against greedy. Reports include all outcomes.
Choose additional fixed seeds if necessary, retaining the first samples and
recording why more were collected. Do not adapt training weights to held-out team
identities based on these traces. Repeatedly used held-out teams are now validation
material, not an untouched final test set.

Package outputs on the login node, then download the archive from your Mac:

```bash
cd /users/mcbronne/pokemon_ml
tar -czf ~/policy-decision-audit-15342.tar.gz runs/policy-decision-audit-15342
```

```bash
scp mcbronne@ssh.ccv.brown.edu:/users/mcbronne/policy-decision-audit-15342.tar.gz ~/Desktop/
```

## How to inspect actual decisions

Review losses and matched successful examples. Separate preview composition/lead
choices, target selection, switches, protection/resource use and possible temporal
inference errors. Compare chosen probabilities with legal alternatives, public
board state and subsequent observations. Low-probability actions are expected
from a sampled policy and are not mistakes solely because they lose. Terminal
outcomes do not identify which earlier action caused a loss. We lack both players'
full choices and counterfactual outcomes here; do not assert exact damage or a
mechanic from these traces when hidden information leaves alternatives possible.
Use Showdown's mechanics as the reference whenever interaction semantics are
uncertain. Add a targeted reproducible test only after identifying a concrete issue.

Rank candidate changes by repeated evidence, expected learning impact and inference
cost. Keep architecture/feature hypotheses separate from measured simulator bugs.
Do not simultaneously change capacity, history, semantic conditioning and preview
heads. This review does not launch a new training arm or adopt a new policy.

## Validation and limits

CPU tests verify the bench-PP collision, preview additive constraint, mask packing,
action probabilities against the sampled log probability, no additional RNG
consumption and unchanged sampled evaluation outcomes with diagnostics enabled.
A real evaluation subprocess with the new CLI flag is rendered by the standalone
summarizer. Full CUDA trace execution and trained-checkpoint behavior still need
measurement on Oscar. Static code review cannot explain particular battle losses
without the requested artifacts.
