# Shared patterns from the 100,000-game observation stress run

The uploaded run used merged commit `16d6ed6`, not an older checkout. It found
56 failing battles among 100,000 games: 804 differing observations (719 actionable)
and 13 final-position reconstruction failures. All battle mechanics up to the
final export matched; those 13 reports were not damage or action-result mismatches.
The targeted teams expose unusual combinations more often than a general sweep.

| First failing symptom | Battles | Shared cause and correction |
| --- | ---: | --- |
| Speed bounds | 30 | 29 involve Encore and Quick Claw: keep the initial activation separate from one announced after an action replacement. If the privately chosen move already matches Encore, the original priority survives. One involves Instruct/Counter: Showdown inserts only Counter's setup callback, so the next ordinary move must not be labelled an instructed attack. |
| Protect/Endure counter | 8 | A failed protecting move consults the actual action queue. An instructed Pokémon can still have its own action pending; a Pokémon fainted this turn can still have a queued action. Track participation at this turn's start so older faints do not count on subsequent turns. |
| Weather timer | 3 | Receiver can inherit weather after Leech Seed or binding damage. Those residual events occur after the weather tick, so the new weather has not aged at the ensuing upkeep. |
| Court Change guard | 1 | Quick Guard and Wide Guard are excluded from Court Change's transfer list. Keep them on their original side, including observations at an intermediate pivot request. |
| Two-turn charge | 1 | A called Solar Beam can start charging while its caller is Choice-locked, then be refused on the second turn. Clear the charge for this pre-execution failure. An ordinary locked second turn failing for lack of a target retains the generic two-turn flag until upkeep; prior tests cover that distinction. |
| Terminal position rebuild | 13 | A delayed hit can end the battle before the attacker's Life Orb faint event runs. Zero HP and completed fainting are distinct states. Preserve that distinction during export/import and removal of bookkeeping. The terminal comparison already ignores the unused faint queue, and now also ignores its count once the battle has ended. |

These are new combinations of familiar event-ordering and public-information
families. The battle mechanics and observation dimensions are unchanged; fixes
are in the follower and position reconstruction. No comparisons were disabled.

## Showdown reference

Reference commit: `ad7ca5d51c03865e6829dbec3a7875df71480fb8`.
The implementation was checked against the local pinned source:

- `data/moves.ts`: Court Change's explicit transfer whitelist; Instruct schedules
  `queue.resolveAction(...)[0]`; Protect and Endure consult `queue.willAct()`.
- `sim/battle-queue.ts`: action resolution adds setup callbacks before the move,
  rolls fractional priority, and gives prioritized actions order 3. `willAct()`
  tests queued action kinds, not the actor's current HP.
- `data/conditions.ts`: a failed StallMove roll removes the protection counter.
- `sim/battle.ts`: generation 9 faints do not cancel all queued actions, and a
  winning faint can terminate processing before a subsequent queued faint.
- The recorded logs show Quick Claw activations before and after Encore, late
  Receiver/weather events, and refusal of the called charge. Complete seeds and
  generator options are retained for reproduction.

## Verification

All **56 uploaded recordings** match on the final code: **4,720 observations**,
**2,311 decisions**, zero observation differences, and matching mechanics with
both continuous replay and reconstruction at each decision.

The requested **2,000 fresh targeted battles** used seed **1830574926** and
`scripts/showdown-pattern-stress.json`, 80 batches of 25. Each of 20 themes gets
100 games, both sheet modes and both Mega settings. Final replay checks
**131,048 observations** and **63,690 mechanics decisions**, with zero differences,
failed/unsupported cases or mechanics divergences. The 332 hidden-action
pass-overs are counted separately. Both continuous mechanics replay and
reconstruction pass on all 2,000 games.

Initial candidate results retain the protection and charge variants caught while
validating. Final reports are separate and use the final binary. Fourteen
representative uploaded battles plus three targeted variants are retained in
`tests/fixtures/targeted_pattern_regressions.jsonl`; all 56 uploads are replayed
in the external validation artifacts. All **61 integration tests** in `follow`,
`parity`, `position`, `shown` and `speed` pass, including earlier failures.

[Results, seeds and final source/binary hashes](showdown-targeted-pattern-results.json).
Full artifacts remain in `/workspace/showdown-validation/targeted-100000-new`
and `/workspace/showdown-validation/patterns-2000-1830574926`. Old generated
recordings were losslessly gzip-compressed and hash-verified to make workspace
space; no uploaded recordings or seeds were discarded.

## Next local run: 100,000 fresh focused games

After merging:

```bash
git fetch origin &&
git switch main &&
git pull --ff-only origin main &&
python3 scripts/run-showdown-targeted-100000.py \
  --suite scripts/showdown-pattern-stress.json --seed 2947610837
```

This focuses exclusively on the latest failing families and related variations:
20 themes, 5,000 games each, equal closed/open sheets and Mega settings.
Results go to `~/showdown-runs/observation-stress-2947610837`; upload its
`differences.tar.gz`. The new base seed differs from the uploaded and cloud
verification runs, though random seeds do not guarantee globally unique games.
The 100,000-game follow-up was prepared but **not run in the cloud**.
