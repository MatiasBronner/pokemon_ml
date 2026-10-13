# Small-policy training findings through October 13, 2026

The small doubles policy learned useful strategy and benefited from training on
additional tournaments. Subsequent continuations show no measurable head-to-head
improvement on the tested held-out current-format distribution. A broader frozen
league did not improve these benchmarks and reduced throughput substantially.
These results establish diminishing returns under the tested settings, not a
hard capacity limit or proof that the small architecture cannot improve further.

This report records the Oscar runs and evaluation outputs reported during the
project. It does not imply independent reruns of every measurement. Scores are
wins plus half of draws, divided by games; reported intervals are the evaluator's
approximate 95% Wilson intervals. All head-to-head and panel evaluations listed
below reported zero timeouts. Rounded percentages may differ slightly from the
underlying JSON.

## Model and training setup

- Small transformer: 922,759 parameters; width 128, three layers, four attention
  heads, feed-forward width 256, dropout zero.
- Observation size: 1,220 floats plus the existing categorical features and legal
  action masks. Speed inference enabled throughout; no speed ablation was run.
- Semantic action encoding, joint-preview encoding, history, opponent prediction,
  hidden prediction and prediction feedback were disabled in these experiments.
- PPO: 512 parallel games, rollout 64 steps, two epochs, minibatch 4,096,
  learning rate 0.0003, clip 0.2, GAE lambda 0.95, value coefficient 0.5,
  entropy coefficient 0.01, uniform coefficient 0.001, gradient norm cap 0.5.
- League mix: 50% current self-play, 40% frozen snapshots, 10% scripted opponents.
  Assignments remain stable within each battle. Frozen/scripted opponent rows are
  excluded from PPO learner updates.
- Training and evaluation use the local Rust engine. No Showdown server calls are
  needed in this loop. Reference-engine parity work preceded these experiments;
  this document is not an exhaustive simulator-correctness claim.
- Oscar: one GPU, four CPU cores; PyTorch 2.11.0+cu128. GPU assignments differed
  across early runs. Controlled league experiments used the same RTX 3090
  allocation for both arms.

## Data and sampling

The first Louisville pool contained 176 legal teams, split into 148 training and
28 evaluation teams. Missing stat spreads were filled heuristically. Grouped
splits keep identical six-species compositions, including their move/item/spread
variants, on one side of the train/evaluation boundary.

The expanded pool contained:

| Tournament | Accepted teams |
|---|---:|
| Baltimore 2027 | 186 |
| Recife 2027 | 72 |
| Brisbane 2027 | 139 |
| Frankfurt 2027 | 175 |
| Worlds 2026 | 423 |
| Total | 995 |

Worlds used an older format; accepted teams were validated against the engine's
current target format. Current-format events receive 90% of sampling mass, Worlds
10%. Events receive equal mass within a group. Within each event, a 50/50 blend of
uniform entry sampling and uniform species-composition sampling reduces duplicate
composition dominance while retaining empirical frequencies. Replacement donors
use the same source weights, subject to legality checks.

Training variation remained enabled: 15% chance of attempting one Pokémon swap,
5% chance of two, and a 25% per-Pokémon chance of redistributing stat points.
Spreads are initially guessed during pool creation; training perturbs the saved
spreads rather than recomputing the heuristic each battle. Evaluation disables
these mutations but honors the pool's sampling weights. Junior/senior sections
were included; no Masters-only filter was applied in the reported mixture.

The initial expanded split was 789 training / 206 evaluation entries (139
current-format and 67 Worlds). For the warm-start experiment it was regenerated
with Louisville training compositions excluded from evaluation: **852 training /
143 evaluation entries**. The latter split supplied the primary current-format
and separate Worlds benchmarks. Composition holdout does not imply entirely
unseen species or absence of related five-of-six lineups.

## Louisville training and learner optimization

Three early jobs saved at approximately updates 476, 895 and 1,311. At update 1,311
the original run had counted 3,805,404 games. Early periodic evaluations were noisy
because they used only 300 games against each scripted opponent. Internal ratings
are fitted within a run and are not Showdown Elo; ratings from independently
initialized histories cannot be compared directly.

A larger Louisville evaluation found:

| Snapshot | Score against greedy, 3,000 games |
|---|---:|
| 000000 | 2.6% |
| 000100 | 65.8% |
| 000475 | 74.3% |
| 000625 | 74.5% |
| 000775 | 76.4% |
| 000875 | 75.0% |

The latest update-895 model scored 58.4% against update 475, 55.7% against 625,
and 52.3% against 775, each over 3,000 games. Head-to-head comparisons revealed
progress that the greedy benchmark partly obscured.

On an RTX 3090, learner tuning measured:

| Passing runtime candidate | Learner decisions/second |
|---|---:|
| Uncompiled FP32 SDPA, minibatch 4,096 | 7,482 |
| BF16 SDPA, minibatch 4,096 | 8,253 |
| BF16 SDPA, minibatch 16,384 | 8,512 |
| Compiled FP32 SDPA, minibatch 4,096 | 11,014 |

Compiled FP32 was about 47% faster than passing uncompiled FP32 SDPA on the same
card. Original attention-auto baseline candidates and compiled BF16 failed the
collection/update consistency gate; their root causes were not established by
these outputs. This does not prove earlier training was invalid. The tuner
reported no speedup against its failed original baseline.

Compiled FP32 validation had likelihood errors around 4–5e-6 and value
consistency errors around 2–3e-7. Cold validation took 67 seconds and the warmup
update 37 seconds. Compiled FP32 SDPA, deferred metrics and non-fused Adam became
the tested continuation configuration. Observed earlier-run comparisons of
6,500 versus 10,000 decisions/second also include hardware differences and should
not be treated as isolated compiler benchmarks.

A short compiled Louisville continuation completed in 20:30. Its checkpoint used
for transfer was update **1,554**, with **4,503,567** counted games.

## Expanded-data warm start

The policy and value weights were transferred from the compiled Louisville
checkpoint into `runs/multi-tournament-small-fp32`. Optimizer, league, RNG and
counters started fresh; source hashes/provenance were recorded. An update-zero
snapshot in this run is therefore already trained, not randomly initialized.

The longer run saved at **update 6,132**, with **16,955,995 games** counted in the
new experiment. These counters exclude the source run's previous games. Sustained
learner throughput was about 10,700 decisions/second. The following 3,000-game
held-out evaluations show useful transfer and further learning:

| Benchmark | Initial transferred policy | Final policy |
|---|---:|---:|
| Current-format teams versus greedy | 71.9% | 77.6% |
| Worlds teams versus greedy | 73.6% | 81.3% |

Final versus transferred initial scored **63.2% [61.5%, 64.9%]**. A separate-seed
comparison against the preserved Louisville source scored **63.8% [62.1%, 65.5%]**.
Final versus update 3,000 scored **52.0% [50.2%, 53.8%]**, a small advantage that
needed confirmation rather than a strong claim of continued progress.

Across 52 current-format compositions, 36 improved against greedy, 13 worsened,
and three were unchanged. Equal-composition mean scores rose from 71.6% to 77.8%,
so improvement was not solely due to frequently sampled teams. Per-composition
samples were often only a few dozen games; apparent regressions require focused
confirmation and battle traces. Reports had no traces and archetypes were
unlabelled, preventing a reliable causal explanation of particular decisions.

## Plateau checks

Using a common seed, final versus Louisville scored 63.6%, while update 4,125
versus Louisville scored 63.3%. Their 0.23-percentage-point difference did not
establish improvement.

Direct final-versus-4,125 comparisons were:

| Seed | Games | Final score | 95% interval |
|---|---:|---:|---:|
| 9502347 | 3,000 | 50.03% | 48.25–51.82% |
| 9602347 | 3,000 | 50.53% | 48.74–52.32% |

The combined score was 50.28%. These comparisons support little detectable
progress from update 4,125 to 6,132 against this opponent on this evaluation
distribution. They do not prove that every matchup or team ceased improving.

Training diagnostics compared updates 1–500 with 5,633–6,132:

| Metric | Early mean | Late mean |
|---|---:|---:|
| Entropy | 1.460 | 1.394 |
| PPO clipped fraction | 0.179 | 0.137 |
| KL | 0.0207 | 0.0158 |
| Value explained variance | 0.420 | 0.431 |
| Value loss | 0.225 | 0.222 |
| Gradient norm before clipping | 1.047 | 0.880 |

There was no obvious entropy collapse, rising PPO clipping or exploding-gradient
pattern. Value learning was largely flat. The reported norm is before the 0.5
clip and should not be mistaken for the applied gradient norm. Disabled auxiliary
tasks reported zero losses. League counts matched the intended 50/40/10 mix.

Against greedy, current-format value ECE increased from 0.095 to 0.132, while Brier
score improved slightly from 0.157 to 0.150. Worlds ECE increased from 0.084 to
0.153. Binned predictions underestimated success against greedy. This can reflect
training against a different opponent distribution; it does not by itself prove
a defective value head. Decision-level calibration samples within battles are
correlated, so these are diagnostic summaries rather than independent samples.

## Controlled league-diversity experiment

Experiment directory: `runs/league-diversity-ab`. Both arms cloned update 6,132's
weights, optimizer, RNG state, registered snapshot history and counters. Mutable
files were copied; immutable snapshots were shared through hard links where
supported. The source checkpoint remained unchanged.

- **Legacy:** latest two, middle and oldest frozen opponents.
- **Broad:** latest two plus up to ten evenly spaced historical opponents.
  Half the frozen probability stayed on the latest two; the other half was
  spread over historical opponents.
- Both retained the same architecture, features, PPO settings and 50/40/10 mix,
  and used a 32-model cache. No outcome-adaptive opponent weighting was enabled.
- A three-hour allocation ran arms sequentially on one RTX 3090, targeting 1,000
  additional updates per arm with a one-hour training cap per arm.
- Outcomes by opponent were logged before episode assignments reset. These are
  evolving-policy training diagnostics, not fixed-opponent evaluations.

Job 7351121 failed before training because the evaluation driver passed an
unsupported `--max-turns` option. The evaluator CLI was fixed and an end-to-end
CPU test added. Job 7351264 subsequently ran both arms and their evaluations.
The exact legacy final update and scheduler exit status were not included in the
reported final summary; they should be read from the retained checkpoint/accounting
records before claiming equal-update or exact-walltime comparisons.

Broad stopped at **update 6,646**, 514 additional updates, with 18,356,409 cumulative
counted games. Its sustained throughput was about **6,700 decisions/second** versus
legacy's **11,400**, approximately **41% lower throughput**. Broad added 1,400,414
counted games beyond the source. Because the time caps yielded different training
progress, this experiment informs GPU-time efficiency rather than isolating
quality at equal update or learner-decision counts.

Fixed-panel evaluations used 1,500 games per opponent, common seed 9702347:

| Opponent | Source | Legacy | Broad |
|---|---:|---:|---:|
| Initial warm-start snapshot 000000 | 64.9% | 62.0% | 57.4% |
| Snapshot 003075 | 55.9% | 55.1% | 50.9% |
| Snapshot 004600 | 50.5% | 50.3% | 48.7% |
| Snapshot 006125 | 50.8% | 51.4% | 49.9% |
| Greedy | 76.9% | 80.1% | 75.9% |

Legacy improved its greedy score without a corresponding clear gain against
recent historical opponents. Broad showed no improvement across this panel.
Some older-opponent scores declined, but multiple comparisons and finite samples
limit claims of individual regressions. The sampling distribution changed and
inference involved more opponent groups; profiling would be needed to attribute
the throughput loss precisely rather than assume cache loading alone caused it.

Final direct comparisons against the frozen starting checkpoint used 3,000 games
per arm, common seed 9802347:

| Continuation | Score | 95% interval |
|---|---:|---:|
| Legacy | 49.97% | 48.18–51.75% |
| Broad | 49.43% | 47.65–51.22% |

Neither continuation showed detectable improvement over the source. Broad is not
proven weaker by these direct matches, but did not earn its throughput cost.

## Decisions and remaining uncertainty

1. Preserve the source update-6,132 small checkpoint as the established baseline.
   Keep both continuation checkpoints for comparison; retain legacy selection as
   the default. Do not adopt this broad configuration on the evidence available.
2. Keep speed inference enabled. No controlled speed ablation has been performed,
   and these results do not implicate it in the plateau.
3. Do not claim a model-capacity ceiling. Medium/large networks have not been
   trained in this experiment; the user chose to investigate small-model training
   first. A medium model remains a possible future capacity experiment.
4. Avoid judging progress solely by 300-game ratings or greedy performance. Use
   direct source-checkpoint comparisons, fixed historical panels, repeated seeds,
   and separate current-format/Worlds reports. Compare learner decisions and GPU
   hours as well as updates and game counters.
5. A controlled lower-learning-rate continuation has been proposed, not run.
   Logs do not establish that learning rate causes the plateau or that lowering
   it will improve results. Keep other settings fixed if testing that hypothesis.
6. Outcome-based opponent weighting, focused battle-trace review and labelled
   archetype evaluation remain untested directions. Do not tune on held-out team
   identities or treat small per-team samples as confirmed strategic failures.

Useful artifacts are `config.json`, `log.jsonl`, checkpoints and snapshots under
`runs/multi-tournament-small-fp32` and `runs/league-diversity-ab`, plus the latter's
`experiment.json` and `evaluation/{source,legacy,broad}` directories. Keep these
outside scratch-purge risk. Setup and reproduction details are in
[policy-training.md](policy-training.md), [tournament-sampling.md](tournament-sampling.md),
[learner-tuning.md](learner-tuning.md), and [league-experiment.md](league-experiment.md).
