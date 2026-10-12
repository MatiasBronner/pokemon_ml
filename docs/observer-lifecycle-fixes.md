# Observer lifecycle corrections

This implements the [focused review](reviews/observer-lifecycle-review.md) against main `7f1f70d85211f1284e1a4a96f76c454d5bafe936`. The review is a historical baseline; its unresolved Speed cause is resolved below. Pokémon Showdown `ad7ca5d51c03865e6829dbec3a7875df71480fb8` supplies the reference recordings. Battle rules and RNG behavior are unchanged; simulator edits emit observation trace events.

## Corrections

- **Protection knowledge:** a shared fixed-size public-action tracker distinguishes natural actions from Instruct, carries history through replacement requests and position swaps, and resets knowledge on entry, Feint or expiry. Silently skipped fainted actions cannot prove whether failed protection cleared the counter. Both engine observations and the log follower now represent that uncertainty.
- **Failure ownership:** stat/ability failure lines no longer clear the preceding move's charge state. This fixes Intimidate/Inner Focus failures during replacements following Dig.
- **Target identity and Pressure PP:** own choices retain the targeted actor's identity. A tracking move or known Stalwart/Propeller Tail can follow that actor through Ally Switch when Showdown omits the target from the move line.
- **Residual weather:** Perish Song's `perish0` identifies the post-weather residual phase. Receiver starting weather after that faint no longer ages the new weather at the same upkeep.
- **Speed evidence:** Heal Block preventing an ordinary draining attack was mistaken for a reflected Heal Pulse action. Only reflectable moves can use that exception; the prevented attack now contributes its natural-action ordering evidence.

## Policy input contract

The observation width is unchanged. The Stall volatile flag and protection-failure probability use **-1 for unknown**, rather than an exact hidden engine value, when public history cannot determine them. Known values retain their previous encoding. Training and live inference must both accept this value. This is conservative uncertainty: stronger inference from private choices might recover some cases, but hidden opposing choices cannot always be recovered.

The retained scripted witness pair has identical player-0 public logs and private requests throughout, yet different actual opponent protection counters. Treating both as exactly known would leak information during training. The regression checks equal player inputs, unequal Showdown state, and matching engine/follower observations.

## Retained tests and reproducible generation

`tests/fixtures/observer_lifecycle_regressions.jsonl` retains representative uploaded failures and earlier charge-state coverage. `tests/fixtures/observer_scripted_public_knowledge.jsonl` retains 18 deliberately constructed cases: eight Perish Song/Receiver weather boundaries in each sheet mode, plus the indistinguishable protection pair. Weather cases cross Drought/Drizzle, weather rocks and source positions, after the original weather has expired. Both continuous and rebuilt mechanics are checked for these scripted cases.

Declarative scripts are in `oracle/scripts/observer_perish_receiver.json` and `oracle/scripts/observer_silent_queue.json`. The 24-theme `scripts/showdown-observer-edge-cases.json` covers the new failures and earlier uncommon families. No ordinary unrestricted random sweep was used for this work.

To generate a new targeted run locally:

```bash
python3 scripts/compare-showdown.py \
  --profile regressions \
  --themes-file scripts/showdown-observer-edge-cases.json \
  --battles 2400 --batch-size 100 --workers 4 \
  --seed 1569032847 \
  --out "$HOME/showdown-runs/observer-edge-1569032847"
```

Use a different base seed for additional runs. `failures.json` and `differences.tar.gz` retain the discovered differences and recordings. The full original 55 failure seeds and generation options remain in [the review evidence](reviews/observer-lifecycle-evidence.json).

## Verification

Final results are recorded in [observer-lifecycle-validation.json](observer-lifecycle-validation.json). Final code passes 7,200 targeted randomized battles (2,400 retained cases replayed plus 4,800 fresh cases), totaling 468,652 observations and 228,109 decisions, with zero observation differences. All 7,200 have zero mechanics divergences; the retained 2,400 also pass rebuilding from state at every decision. The first 2,400 randomized edge cases had no exact Perish Song/Receiver boundary despite including the relevant components; the scripted cases explicitly close that coverage gap. Random success alone is not proof that every rare interaction was exercised.

All 55 uploaded failures replay with zero observation differences: 4,159 observations and 2,030 decisions. Focused unit and saved-regression checks cover public uncertainty, renumbering, entry/reset, Mental Herb/Encore, allied Skill Swap, hidden Choice/Pressure, wait requests, Soak, Symbiosis, Endure and Quick Claw. All 40 tests across the public-action units and follow, parity and speed integration suites pass. Formatting and whitespace checks pass.

An additional 9,600 targeted candidate battles passed 621,557 observations and 302,581 mechanics decisions. That candidate preceded the final guard suppressing a terminal upkeep trace when Showdown has ended the battle; these are supporting results, not final-binary validation. No performance benchmark or universal correctness claim is made. Public-action tracking uses fixed arrays and the existing observation trace; no full hidden queue reconstruction is introduced.
