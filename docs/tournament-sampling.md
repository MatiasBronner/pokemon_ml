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
existing resume provenance checks deliberately reject changing pool hashes. These
changes do not introduce a warm-start mechanism for switching a trained model to
a new dataset; they prepare weighted pools for new training runs. Record the pool
and split files with checkpoints. On Oscar rebuild the Rust Python extension with
`maturin develop --release --locked` before using weighted pools; the old extension
would silently ignore the new JSON weight fields.
