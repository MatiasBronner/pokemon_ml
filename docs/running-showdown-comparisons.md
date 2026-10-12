# Run Showdown comparisons locally and retain differences

The comparison runner generates battles in the pinned Pokémon Showdown simulator, then compares both observations (`followcheck`) and mechanics (`difftest --by-hand`) with the Rust engine. It saves every differing, unfollowed or unsupported battle separately, including its full recording and seeds.

## Setup

Use Git, current stable Rust/Cargo, Node.js **22 or newer**, npm, and Python **3.10 or newer**. Run these commands from the repository on Linux or macOS; on Windows, use WSL.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
bash scripts/setup-oracle.sh
```

The setup script fetches and builds Showdown commit `ad7ca5d51c03865e6829dbec3a7875df71480fb8`. The oracle currently uses `gen9championsvgc2026regmc`. Comparisons run locally; no Showdown server, account or Python extension installation is needed.

Start with a small run:

```bash
python3 scripts/compare-showdown.py \
  --battles 100 --seed 20261012 \
  --batch-size 50 --workers 2 \
  --out runs/showdown-smoke-20261012
```

Then run 50,000 complete battles:

```bash
python3 scripts/compare-showdown.py \
  --battles 50000 --seed 20261013 \
  --workers 4 \
  --out runs/showdown-50000-20261013
```

Use a **new base seed and output directory** for each fresh run. Batch seeds are spaced apart to avoid simply reusing the preceding run's batches when the base seed increases by one. Identical seeds, options and generator versions regenerate the same battles. The runner refuses to overwrite an existing directory.

The runner builds the current Rust code with `cargo build --locked --release` and freezes copies of both executables for the entire run. Their hashes and the engine/Showdown commits are recorded in `provenance.json`. `--no-build` explicitly opts into existing release executables; their hashes are still recorded, but their correspondence to current source is not established by a build.

## Results to send back

Upload the resulting **`differences.tar.gz`**. It contains:

- `failures.json`: every failing battle's generator seed, original generator battle ID, four-word simulator seed, sheet mode, options, first observation difference and mechanics result.
- `failures/<name>/case.jsonl`: complete teams, choices, states, battle logs and both players' private requests. These recordings are enough to replay without regenerating the battle.
- `failures/<name>/follow.txt` and `mechanics.txt`: detailed differences and the mechanics comparison.
- `provenance.json`, `manifest.json`, `summary.json`, `results.json` and `working-tree.patch`: the tested versions, generator hashes, batch settings, totals and any tracked local changes.

The reports and failure recordings are written as batches finish, so discoveries survive a later failed batch. `summary.json` updates during the run. A completed run exits **0** with no findings, **1** when differences or unsupported cases were saved, and **2** for an incomplete run/error. Unsupported battles are labeled separately from observation differences. If the run crashes before producing the final archive, send the existing `failures/` directory and JSON reports; the current batch recording is retained too.

You can send just one failure directory together with `provenance.json`. **Seeds alone are insufficient:** also preserve the original battle ID, generation options and versions. The four-word simulator seed alone does not recreate the randomly generated teams and choices.

## Replay a saved failure

From the repository, substitute the actual failure directory:

```bash
cargo build --locked --release --bin followcheck --bin difftest
target/release/followcheck runs/showdown-50000-20261013/failures/NAME/case.jsonl --list
target/release/difftest runs/showdown-50000-20261013/failures/NAME/case.jsonl --by-hand
```

Use `followcheck ... --battle ID` to print every scalar-feature difference for the recording's `id`. `followcheck` exits 1 for differing observations as well as follower errors. Both tools accept `--report output.json` for machine-readable per-battle results (`difftest` in mechanics mode).

To regenerate rather than replay, use the generator seed, **original** battle ID and options from `metadata.json`:

```bash
node oracle/gen_cases.js --seed GENERATOR_SEED --only ORIGINAL_ID \
  --log --requests --out /tmp/reproduction.jsonl GENERATION_OPTIONS
target/release/followcheck /tmp/reproduction.jsonl --list
```

The regenerated battle uses its original ID; the runner's saved recording uses a unique ID within its batch. Preserve the pinned source version when regenerating.

## Scope and resource use

With the default 1,000-battle batches, the mixed profile runs 30,000 general and 20,000 targeted battles in a 50,000-battle run, alternating open and closed sheets. The targeted groups cover types, Pressure, Encore/items, trapping, grounding, Speed ordering, ability changes and called moves. `--profile general` generates only general battles.

This is a mechanics stress corpus: some teams have injected abilities or duplicate species. The runner excludes multiple initially brought Illusion users per side and impossible closed-sheet rosters with non-native Illusion, matching the earlier validation scope. It replaces excluded and truncated candidates until it has the requested number of complete recordings. Comparison reports retain the existing hidden-information pass-over count; they do not claim those observations were checked.

Passing recordings are removed after comparison by default, keeping disk use bounded by active batches. `--keep-passing` retains every full recording and can consume substantial disk space. Lower `--workers` or `--batch-size` if memory or temporary disk space is tight. Failure recordings and reports are always retained.

## Fresh 200,000-battle runs after merging fixes

Update to merged main, then run the convenience script:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
python3 scripts/run-showdown-200000.py
```

The script selects and prints a random 32-bit base seed, builds the current code,
then runs 200,000 comparisons using the existing mixed profile. Results go to
`~/showdown-runs/showdown-SEED`, outside the checkout. Save the printed seed;
random selection makes accidental reuse unlikely, but does not guarantee that
no seed or individual battle has ever been generated before. To choose a seed
explicitly or save the run on the Desktop:

```bash
python3 scripts/run-showdown-200000.py --seed 1764029381 --out ~/Desktop/showdown-1764029381
```

Upload that directory's `differences.tar.gz` after completion. The script
preserves the comparison runner's exit codes (0 clean, 1 findings, 2 incomplete).
It never updates Git automatically or overwrites an existing run.

## Targeted observation regressions

To run 2,000 new battles around the retained Illusion, residual-weather,
Feint/guard, waking/Attract, Court Change, reflected healing, Instruct/Quick Claw,
Focus Punch, Future Sight/Red Card recoil, Ripen/Leppa and instructed sleep, type, Pressure, item/ability-swap, Mental Herb, trapping and called-move
issues, plus confusion/Instruct, Mega/Pressure, solar animations, Toxic replacements,
Hunger Switch/Illusion, called Speed Swap, Baton suppression and Encore/Quick Claw.
This covers all 27 themes with both closed and open sheets:

```bash
python3 scripts/compare-showdown.py --battles 2000 --batch-size 25 \
  --profile regressions --seed 1648207359 --workers 4 \
  --keep-passing --out runs/observation-regressions-1648207359
```

These are deliberately unusual stress teams, not a competitive training pool.
The exact uploaded failures are also retained in the follower integration tests;
the fresh random cases supplement those deterministic regressions.

## Fresh 300,000-battle run

After merging the Future Sight/Ripen/Instruct fixes:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
python3 scripts/compare-showdown.py --battles 300000 --seed 3184072951 --workers 4 --out ~/showdown-runs/showdown-3184072951
```

This uses a new base seed, builds current sources and preserves failure recordings
and provenance using the existing mixed profile. Upload its `differences.tar.gz`.

## 100,000 fresh games focused on past bug families

After merging the targeted-suite PR:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
python3 scripts/run-showdown-targeted-100000.py --seed 1379264805
```

The new `scripts/showdown-observation-stress.json` suite contains 40 themes:
27 historical regression families and 13 additional combinations. These include
confusion/Endure/Instruct, Pressure/grounding/ability exchange, solar PP/suppression,
Toxic/Baton/forced switches, Illusion/forms/Speed Swap, reflected healing/Encore,
called Speed Swap/promotion, hidden suppression/revelation, Mental Herb/Disable/
allied Skill Swap, Ripen/solar PP, sleep/confusion/priority, trapping/type changes,
and Future Sight/weather/guards. Each theme's `purpose` explains its target.

This generates new random teams, battle seeds and choices, rather than replaying
saved failures. It uses 400 batches of 250 battles: 2,500 per theme, split equally
between open and closed sheets. Successive pairs of closed/open cycles alternate
Mega probability between 0 and 1 (60,000 games with Megas disabled and 40,000
with all eligible native stone holders opting into Mega selection). Species that
have no Mega remain unchanged; forced items can also prevent a Mega Stone.
The ordinary random action policy still supplies varied attacks and switches.
There are no unthemed general batches. Targeting increases the chance of the
interactions; it does not guarantee that every game executes them. These stress
teams are deliberately unusual, rather than a legal competitive training pool.
Existing single-Illusion restrictions remain in effect.

The launcher builds the current checkout and checks both observations and mechanics.
It chooses and prints a random seed if `--seed` is omitted. A fresh seed reduces
accidental overlap but cannot guarantee every individual battle is globally new.
Results default to `~/showdown-runs/observation-stress-SEED`; use `--out` to change
that directory and `--workers` to change concurrency. Existing directories are
never overwritten. All differing recordings, seeds and reproduction commands are
saved automatically; passing recordings are discarded to limit disk use. Send
`differences.tar.gz` when the run completes. `manifest.json`, a copied suite and
its provenance hash record the exact batch settings.

To inspect the configuration without generating battles or building binaries:

```bash
python3 scripts/run-showdown-targeted-100000.py --plan-only
```

This suite was prepared with configuration and syntax checks only; its battles
were not run in the cloud.

## Focused follow-up to the 56-failure stress run

After merging the shared-pattern fixes:

```bash
git fetch origin &&
git switch main &&
git pull --ff-only origin main &&
python3 scripts/run-showdown-targeted-100000.py \
  --suite scripts/showdown-pattern-stress.json --seed 2947610837
```

This generates 100,000 new random stress battles with **20 themes**, all focused
on the latest failing families. Twelve themes retain relevant earlier coverage;
eight add focused variants: Encore/Quick Claw rerolls and unchanged choices,
Instruct/protection with a pending natural action, protection with a queued faint,
Court Change/guards/pivots, Receiver after Leech Seed or trapping damage,
called charges with Choice items, and terminal delayed-hit/recoil positions.
It checks both observations and mechanics, including position reconstruction.

Each theme receives 5,000 battles, equally split between closed and open sheets.
The 400 batches of 250 cover equal numbers with Mega probability 0 and 1.
Species and move sampling plus random choices provide fresh scenarios; not every
battle is guaranteed to execute its targeted interaction. Existing single-Illusion
restrictions are unchanged. These are stress teams, not a competitive training pool.

Results go to `~/showdown-runs/observation-stress-2947610837`. Every difference is
retained with its seeds and reproduction command in `differences.tar.gz`. To
select another random base seed on a subsequent run, omit `--seed`; a random seed
reduces overlap but does not guarantee globally unique individual battles.
`--plan-only` validates this suite without building or generating games.
