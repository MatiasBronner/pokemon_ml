# Multi-team doubles training

The [implementation plan](policy-training-plan.md) was written before the changes. Training keeps the existing tournament scraper, guessed spreads and randomized team sampler. Rust battle rules and Speed inference are unchanged. Rust changes add timeout and completed-episode diagnostics for evaluation.

## First run

Build the current Python extension in your virtual environment (`maturin develop --release`) before training; otherwise Python may use an older Rust binary. Use the project's existing Rust setup and dependencies. Split a tournament pool before choosing model settings:

```bash
python -m pokemon_ml.team_splits --pool teams/2027-frankfurt.json --out teams/split --seed 23
python -m pokemon_ml.train \
  --pool teams/split/train.json --eval-pool teams/split/eval.json \
  --run runs/policy-small --model-size small --device cuda
```

Small is the default. Baseline counts with the current dex are 922,759 / 3,716,103 / 10,350,343 parameters for `--model-size small`, `medium`, `large`. Explicit `--width`, `--layers`, `--heads`, `--ff` overrides remain available. An existing run resumes its saved architecture; incompatible model or feature flags require a new run directory.

Training hours exclude evaluation; `wall_hours` includes evaluation and startup evaluation. The split groups identical six-species compositions, ignoring roster order, spreads, items and moves. It does **not** group all teams sharing five species: those can remain across splits. Inspect your pool for broader near-duplicate leakage and use an explicit held-out-archetype pool when testing that harder generalization problem. Training rejects species-composition overlap when `--eval-pool` is given. Without it, evaluation is on the training distribution and must not be called held-out evaluation.

## Independent experiments

All architecture experiments are off by default. Speed inference and the training league are on by default.

| Option | Experiment |
|---|---|
| `--no-speed` | Mask resolver Speed ranges, possible Speed items and cross-field order features. Keep known stats, dex base Speed, revealed items, and public weather/boosts. The resolver still runs: this measures information value, not its CPU cost. |
| `--semantic-actions` | Condition the second slot on the first action's user, selected move/target or switching Pokémon, plus the action embedding. |
| `--joint-preview` | Jointly score the lead and reserve pairs instead of adding independent pair scores. |
| `--history 16` | Add 16 past public **decision snapshots**, with phase, sheet mode, turn, active species/last attempted moves, HP, status and boosts. This is not Jaxcalibur's exact event log. |
| `--opponent-prediction` | Supervise two marginal next-action distributions, including move identity, target, Mega, switch destination and pass. Not a full correlated joint-opponent distribution. |
| `--opponent-prediction --prediction-feedback` | Separately test feeding the model's own predicted distributions back into its policy features. |
| `--hidden-prediction` | Supervise unrevealed items/abilities and a multi-label move set, using balanced positive/negative move loss. |

Auxiliary loss coefficients are `--opponent-coef` and `--hidden-coef` (0.05 each initially). Labels come from the opposite side's **own** observation and chosen actions; the actor never receives those private labels. Item/ability loss covers unknown public IDs; ambiguous apparent species are excluded. Move targets are set-valued; known moves are excluded. Preview and non-decision actions are excluded from opponent-action loss. Labels reflect the simulator's assigned/guessed spreads and sets, not verified real-world facts.

History is shared between training, evaluation and the Showdown client. Each player has its own history, each battle room has separate state, and completed episodes reset it. Waiting/non-decision observations do not add history tokens in training. Disabled inference cannot be bypassed through history (history does not contain resolver features). No search or damage calculator was added.

## Training league

Defaults are `--league-current 0.5 --league-frozen 0.4 --league-scripted 0.1`, sampled at the start of an episode. Self-play learns from both sides; frozen/scripted matches learn only from the assigned learner. PPO never updates from a frozen/scripted opponent's actions. Wait requests preserve learner credit until its next decision or terminal result.

Frozen candidates include the latest two snapshots, a middle snapshot and the oldest. A bounded four-model cache limits device memory; candidates are a simple diversity heuristic, not a learned opponent-selection algorithm. Scripted training opponents are random and greedy. The existing privileged lookahead player is used only as a clearly labelled evaluation challenge. Use `--no-lookahead-eval` to omit its expensive CPU evaluations during training; the Oscar template does so. It remains available through the standalone evaluation command. Frozen models do not update during an episode. Current-policy opponents update between rollouts, as in ordinary self-play.

`--snapshot-every 25` controls opponent snapshots independently of `--eval-every 25`. Retain `snapshots/` when moving or resuming a run. Old checkpoints are not automatically pruned; plan persistent storage. `--league-current 1 --league-frozen 0 --league-scripted 0` is the no-league ablation.

## What is measured

`log.jsonl` records valid learner decisions, simulation/transfers/inference/scripted-opponent/update time, full game-step throughput, learner decisions/sec, PPO loss, nonnegative approximate KL, clipping fraction, gradient norm, explained variance, auxiliary losses and classification accuracies, phase-specific entropy and entropy normalized by the legal joint-action count. Phase 0/1/2 means preview/move/replacement. Normalized sampled conditional entropy is a diagnostic, not an exact enumeration of joint policy entropy.

Episode metrics distinguish ties from timeout caps and report mean completed-game turns. League assignment counts are cumulative episodes, not fractions of PPO samples. CUDA runs record peak allocated/reserved memory and GPU utilization where supported; unavailable utilization is null. GPU behavior has not been validated on this CPU-only development machine.

Checkpoint evaluations save `evaluation/<checkpoint>-vs-<opponent>.json`: actual completed games, wins/losses/draws, fractional outcome score, approximate Wilson 95% interval, per-composition results, matchup results, optional archetypes, value calibration (Brier/ECE/bins), episode length and timeouts. Draws score half; value calibration concerns expected outcome score, not necessarily a binary win probability. Decisions from one battle are correlated. `ratings.json` includes the frozen-opponent matchup matrix and relative Elo; these ratings are not Showdown ladder Elo.

Fixed evaluation seeds and no team variation improve comparisons. Evaluation counts each initial episode in a vector batch once, waits for that batch, and recreates the next batch with an incremented seed. It reports exactly the requested number of completed games; auto-reset episodes in finished slots are ignored. This fixes initial team pairings across checkpoints with different battle lengths. Unfinished episodes and extra simultaneous finishes beyond the requested count are excluded. `archetype` labels in pool teams are used if supplied; otherwise groups are `unlabelled`. An explicit `--archetypes` JSON maps numerical composition keys from evaluation reports to labels. Archetypes are not automatically guessed.

```bash
python -m pokemon_ml.evaluation \
  --checkpoint runs/policy-small/checkpoint.pt \
  --opponent greedy --pool teams/split/eval.json \
  --games 1000 --seed 9000001 --device cuda \
  --out runs/policy-small/held-out.json --trace-games 2
```

Use another checkpoint path for `--opponent` to compare policies. Reports retain up to `--trace-games` completed learner trajectories (observations, actions, values), suitable for tactical inspection. To assess cycles, inspect the matrix against the same frozen opponents over time. Test several training seeds before choosing an architecture.

The existing Showdown client supports `--results` JSONL and `--records` public battle recordings. It now supports experimental histories, validates checkpoint contracts, records value predictions and writes a results summary with confidence intervals/calibration beside `--results`. Run against a local Showdown server or an explicitly chosen opponent to validate transfer; server games require the user's server/account and are not automatically launched by training.

## Hardware and experiment benchmarks

```bash
python -m pokemon_ml.benchmark --pool teams/split/train.json \
  --sizes small medium --envs 256 512 1024 --threads 2 4 \
  --device cuda --out runs/hardware-benchmark.json
python -m pokemon_ml.experiments \
  --pool teams/split/train.json --eval-pool teams/split/eval.json \
  --out runs/ablations --seeds 1 2 3 \
  --experiments baseline no-speed opponent-prediction history --sizes small
```

The experiment runner writes commands by default; add `--execute` to run sequentially. Forward training options such as `--device cuda` and `--envs 512`. Use `--hours` for allocated wall-clock comparisons or `--updates` for update-count comparisons, then check actual learner decisions: league mixtures and episode lengths alter samples per update. Compare held-out scores at matched decision counts and wall-clock budgets, not just update number. Benchmark environment results include random action selection and are labelled accordingly. Full-loop figures include PPO, unlike raw Rust simulator numbers.

## Oscar jobs and restart behavior

`scripts/oscar/train-policy.sbatch` is a template, not a verified Oscar allocation request. Set `POLICY_REPO`, `POLICY_POOL`, `POLICY_EVAL_POOL`, `POLICY_RUN`, activate your environment, and supply the partition/account/QOS permitted by your allocation. It requests one GPU and four CPU workers; adjust resource/time settings after checking your actual free-tier limits. No guessed partition or GPU model is hardcoded.

```bash
sbatch scripts/oscar/train-policy.sbatch
```

Checkpoints are atomic and include model/optimizer configuration, model state, optimizer state, Python/NumPy/Torch CPU/CUDA RNGs, pool fingerprints, dex/layout/encoding contract, league RNG/manifest/counters, and progress counters. SIGINT/SIGTERM/SIGUSR1 request stopping at a safe update boundary; the Slurm template forwards its warning to Python. A long evaluation/update can exceed the warning window: schedule enough checkpoint margin and retain periodic saves.

**Resume restarts unfinished episodes.** The Rust environment does not serialize live vectorized battles. Its new episode seed is derived from the saved update; no claim of bit-identical uninterrupted trajectory continuation is made. GPU determinism and optimizer reproducibility across different hardware are also not guaranteed. Existing baseline snapshots without a contract remain loadable for compatibility; newly written files enforce the contract. Only load trusted PyTorch checkpoint files.

## Development validation

[Recorded checks](policy-training-validation.json): 28 Python tests passed, one live-server test skipped, and 10 focused Rust environment tests passed. Additional experimental regression and CLI smoke runs check collection/evaluation likelihoods, learner-only PPO, history/reset/privacy, matched initial matchups, split leakage, calibration, resume manifests, and positive benchmark timings. CPU smoke training is not evidence of competitive strength.
