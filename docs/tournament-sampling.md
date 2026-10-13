# Training across tournaments

Use one raw file per tournament: scraping to the same filename overwrites it.
Build each event's pool separately with `teampool`. This validates against the
engine's current regulation and fills missing stats using its existing heuristics.
An older tournament's rules being a subset does not exempt its teams from validation.

For example, on your Mac from the repository root:

```bash
for EVENT in 2027-baltimore 2027-recife 2027-brisbane 2026-worlds; do
  cargo run --release --locked --bin teampool -- \
    "teams/raw/$EVENT.json" --out "teams/$EVENT.json"
done

python3 python/pokemon_ml/team_mix.py \
  --current teams/2027-baltimore.json teams/2027-recife.json teams/2027-brisbane.json \
  --worlds teams/2026-worlds.json \
  --worlds-fraction 0.1 --composition-balance 0.5 \
  --out teams/multi-tournament.json

python3 python/pokemon_ml/team_splits.py \
  --pool teams/multi-tournament.json --out teams/multi-tournament-split --seed 23
```

Add Louisville or other validated current-format event files to `--current`.
Both commands also work as `python -m pokemon_ml.team_mix` / `team_splits` when installed.

## Sampling

The default mixture assigns 90% of team draws to current-format events and 10%
to Worlds. Events receive equal mass within their group, regardless of size.
Within each event, half of its mass samples entries uniformly; half samples
species-composition groups uniformly, then entries within that composition.
Thus popular compositions remain more frequent, but repeated compositions cannot
consume the entire event budget. Move, item, nature and spread variants remain
available. Divisions are retained; filter unwanted junior/senior entries before
combining if the desired distribution is Masters only.

The tool writes explicit per-team `sampling_weight`, `sampling_group`, `event`
and `source` fields. Rust precomputes cumulative weights once and uses a binary
search per draw. Both players' starting teams and replacement donors use these
weights. Illegal donor replacements are retried as before: accepted replacement
rates can differ from requested source masses because legality conditions them.
A Worlds-origin team can receive a current-origin replacement and vice versa.
Stat perturbations are unchanged.

`--composition-balance 0` preserves the empirical entry frequencies within each
event; `1` gives equal mass to each distinct composition. Group fractions are
renormalized over available groups. Missing groups after a split therefore cannot
receive their nominal share. Zero-weight teams are never drawn. Invalid weights,
zero total mass and inconsistent pool formats are rejected. Duplicate input
source/event pairs are rejected; repeated entries within an event remain entries.
Legacy pools without weights retain uniform sampling and the existing RNG path.

## Evaluation and continuation

Splits group identical species multisets across all events, so variants cannot
cross the train/evaluation boundary. Weights are recomputed separately after the
split to preserve the intended mixture rather than inherit distorted subset mass.
The splitter writes `train.json`, `eval.json`, and, when present,
`eval-current.json` and `eval-worlds.json`. Use `eval-current.json` as the primary
benchmark and evaluate Worlds separately; `eval.json` measures the mixed population.
Evaluation disables team mutations but honors the supplied pool's weights.

Keep the running experiment's pool unchanged. A new dataset must use a new run:
existing resume provenance checks deliberately reject changing pool hashes. Use the explicit warm-start mechanism below to transfer weights to a new dataset;
ordinary resume continues to reject changed pool hashes. Record the pool
and split files with checkpoints. On Oscar rebuild the Rust Python extension with
`maturin develop --release --locked` before using weighted pools; the old extension
would silently ignore the new JSON weight fields.

## Warm-start a trained policy on this dataset

`pokemon_ml.train --init-checkpoint PATH --run NEW_EMPTY_DIRECTORY` loads only
model weights and the source model architecture. It checks the observation/dex
contract and rejects conflicting architecture options or an existing nonempty
run. Optimizer, PPO defaults (unless explicitly overridden), league, RNG and
counters start fresh. Both policy and value heads are transferred. Source path,
SHA256, source update/game counters and provenance are recorded in the new
config and checkpoints. Subsequent launches omit `--init-checkpoint` and resume
normally, retaining that initialization record. Runtime flags are chosen for the
new run rather than inherited implicitly. Never copy the old checkpoint into the
new directory: that invokes full resume instead of warm start.

Before warm starting, regenerate the split with repeatable `--seen-pool` options:

```bash
python3 python/pokemon_ml/team_splits.py \
  --pool teams/multi-tournament.json \
  --seen-pool teams/louisville-split/train.json \
  --out teams/multi-tournament-warmstart --seed 23
```

Previously trained species compositions stay in training, even if they appear
under another tournament or with different moves/items/spreads. Evaluation is
sampled only from the remaining unseen compositions, and mixture weights are
recomputed after splitting. This protects composition-level holdout, not novel
individual species or absence of related five-of-six lineups. If no unseen
compositions remain, splitting fails rather than silently report a held-out
benchmark. Primary evaluation should use the resulting `eval-current.json`.
Supply all earlier training pools when a policy has trained on multiple datasets.
The splitter records excluded compositions and input hashes for review.

For an eight-hour Oscar allocation allow a saving margin, e.g. `POLICY_HOURS=7.75`,
and forward `--init-checkpoint` only for the first launch into a new run. On later
launches omit it. The standard script forwards additional trainer options.
