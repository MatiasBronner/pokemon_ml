# Focused observer lifecycle review

Review date: 2026-10-12. Baseline: merged PR #33, `7f1f70d85211f1284e1a4a96f76c454d5bafe936`; implementation commit `b452416`. This review changes documentation only. It does not approve, implement or benchmark fixes. It is a focused review of the five requested areas, not a complete correctness audit of the simulator.

## Decision summary

There is enough evidence to justify a focused correction of the observation lifecycle. There is not evidence here that the mechanics engine needs a rewrite. Four current failure families have concrete code-level explanations; the fifth (Speed inference) still needs a trace that identifies the exact comparison responsible.

The highest-impact gap is the follower's model of pending actions: 51 of 55 failing battles first differ on the protection counter. The other four first differ on charge state, PP, weather age and Speed. Fixing every symptom separately without specifying the lifecycle contract is likely to produce another round of related failures.

A crucial design question must be resolved before trying to reconstruct the entire queue: **can a player actually know the value from the log, its own requests and any agreed open sheets?** Hidden opposing choices and silent queue consumption can make exact reconstruction impossible. If a feature is not publicly determinable, the engine-side observation must use the same agreed uncertainty representation as the follower. Using hidden engine state to force follower agreement would not solve the training/deployment mismatch.

## Evidence and limits

The uploaded run contains 100,000 completed battles, all mechanically matched, 55 failing battles, 222 observation differences and 160 differences at decision points. The run used the merged fixes. A read-only replay of all 55 recordings reproduces all 222 differences across 4,159 observations and 2,030 decisions. No new battles were generated for this review.

First-mismatch grouping is not a count of independent defects or every affected feature. All 51 protection cases contain an Instruct event; 49 contain a faint in that decision's log. Two occur in a decision segment following a pivot replacement. Full turn context, rather than an isolated segment, matters for those two.

[Evidence index](observer-lifecycle-evidence.json) preserves the uploaded provenance, all failing keys, seeds, generator settings and first mismatches, with representative event sequences. Full recordings are under `/workspace/showdown-validation/explain-latest/failures/`.

Confidence labels below mean:

- **Confirmed:** the recording and the code show the specific erroneous transition.
- **Structural gap:** the representation lacks a distinction needed by confirmed cases; not a claim that every route using it fails.
- **Unresolved:** a mismatching feature is reproducible, but the exact responsible inference has not been demonstrated.

## 1. Action lifecycle and silent queue consumption — highest priority

**Confirmed failing family; structural gap.**

Engine: [`Battle::will_act`](../../src/battle.rs:990) inspects the actual queue. [`note_move_starts`](../../src/speed.rs:415) snapshots real pending move actions.

Follower: [`begin`](../../src/follow.rs:1205) increments a visible-action count, except for an instruction. Turn start sets `queued_this_turn` from field presence. [`-fail`](../../src/follow.rs:1865) estimates that an action remains if a participant started the turn present and has not visibly begun its natural action.

Missing distinction: **queued but not yet reached** versus **silently skipped and already consumed**. A fainted actor can have no subsequent log line when the turn loop passes its action. Its count can therefore remain zero even after that action has left the real queue.

Representative `0066-confusion-endure-instruct-battle-245`, step 2:

1. Simisage uses Endure successfully.
2. Tauros faints Vivillon before Vivillon's natural action.
3. Oranguru uses Instruct on Simisage.
4. Simisage's repeated Endure fails; Showdown preserves the counter.
5. The follower sees Vivillon's unresolved-looking bookkeeping and treats another action as pending, so it deletes the counter.

Showdown's Endure `onPrepareHit` uses `queue.willAct() && runEvent('StallMove', ...)`: failure because the queue is empty does not run the odds test. Failure of the odds test removes the counter. Both outcomes can use a `-fail` message, but they have different state consequences.

The prior correction retained a same-turn faint's queued action; it did not fully specify when that action becomes consumed. Adding another Boolean exception will not establish that distinction across Instruct, priority changes, switches and requests splitting a turn.

**Required review outcome before implementation:** define natural action, inserted instruction, setup-only action, promoted action, skipped action and replacement action. Establish which transitions are visible and which are only uncertain. Preserve the distinctions through intermediate requests; a request boundary is not necessarily a turn boundary.

**Acceptance cases:** the fainted action is still ahead of Instruct; it has already been skipped before Instruct; the target has/has not used its natural action; a setup-only Counter instruction; a pivot request splits the sequence; an old faint must not count next turn. Check both sides' observations and the protection odds feature, not only HP.

## 2. Failure ownership and move context — small, bounded correction

**Confirmed failing case; structural gap.**

[`Mover`](../../src/follow.rs:434) retains the preceding move and `cancel_charge`. It is cleared at a `turn` message, but can survive upkeep and a replacement request before the next `turn`.

The [`-fail` handler](../../src/follow.rs:1855) clears charging state if the message names the same position as that retained mover and its charge-cancellation flag is set. It does not first establish that the failure belongs to that move. Protocol `-fail` also reports blocked stat changes and ability interactions.

Representative `0201-encore-quick-claw-battle-86`:

- Stunfisk prepares Dig.
- During the following replacement, Mega Scrafty's Intimidate tries to lower Stunfisk's Attack.
- Inner Focus prevents that change and produces `-fail|Stunfisk|unboost|atk|[from] ability: Inner Focus`.
- The charge handler associates this with the old Dig context and clears the two-turn flag, before the next turn starts.

This is an over-broad effect of the earlier charge correction, not evidence that Dig's mechanics are wrong.

**Required contract:** classify a failure by event kind, owner and reason before applying move cleanup. “Same position as last mover” is insufficient. A context should have an explicit lifetime across switches, residuals and requests.

**Acceptance cases:** an actual refused second-turn charge; an ordinary second turn with no target; Intimidate/Inner Focus during the intervening replacement; unrelated failed boost/unboost against the same actor; the same events against a different actor. Failed unrelated events must leave charge flags and PP unchanged.

## 3. Identity, position and target selection — small, bounded correction

**Confirmed failing case; duplicated target semantics.**

The engine's [`get_target_ex`](../../src/battle.rs:748) distinguishes a target's identity from its current position. Stalwart and Propeller Tail can track the original target through Ally Switch.

The follower [`swap`](../../src/follow.rs:1548) moves spot state and own identities, but missing-target PP inference also uses chosen target locations in [`move` handling](../../src/follow.rs:1692). A charging move may omit its target from the move line, so this inference matters.

Representative `0235-instruct-last-protection-battle-12`, step 19:

- Appletun has Pressure; Musharna uses Ally Switch.
- Mega Skarmory has Stalwart and begins Fly, chosen against Appletun's original position.
- Stalwart tracks Appletun after the swap. Showdown charges two PP: 12 → 10.
- The follower treats the original position's new occupant as the target and records 11 PP.
- The charging request lists only Fly, without PP, so it cannot immediately correct the discrepancy.

**Required contract:** record selected target identity, selected position and effective target separately when the player knows them. Ability-based tracking, move-based tracking, redirection, replacement and target absence need explicit precedence. Do not infer an unrevealed opponent's selected target from engine internals.

**Acceptance cases:** Ally Switch with neither tracking ability, Stalwart, Propeller Tail and a tracking move; Pressure on the original target versus the other slot; full and missing-target log lines; charge-start and strike turns; replacement of the original target. Confirm private requests without PP do not silently become authoritative PP updates.

## 4. Residual phases and effect age — bounded design change

**Confirmed failing case; structural gap.**

The engine runs ordered effect callbacks. The follower uses the Boolean `ending` and a list of recognizable late events. The existing recognition covers poison/burn, Leech Seed and trapping damage in [`line`](../../src/follow.rs:1015), plus weather and future-move events elsewhere. [`upkeep`](../../src/follow.rs:2413) advances weather unless `weather_late` was set.

Representative `0285-residual-weather-battle-73`, step 12:

- Perish Song reaches zero and Garbodor faints.
- Receiver inherits its Drought; sunlight starts.
- This occurs after the weather tick, but the follower has not recognized a late residual phase from this sequence.
- The follower ages the weather immediately; the engine reports newly started weather.

The missing concept is **where in residual processing the battle currently is**, rather than a list of damage sources that happen to imply “late.” Weather age is the demonstrated failure. Terrain and other duration accounting use adjacent logic and deserve review, but this review does not establish new bugs in them.

**Required contract:** specify effect birth relative to its own tick. Infer phase only from public ordering evidence, and state what happens when earlier residual phases produced no visible message. A richer phase model is useful only if it handles silent phases too.

**Acceptance cases:** weather inherited after Perish Song, poison, Leech Seed, trapping and a delayed hit; weather set during a normal move or ordinary switch-in; weather expiry followed by a new setter; silent weather ticks; extended-duration items. Verify age at an intermediate replacement request as well as the next turn.

## 5. Public information and Speed inference — unresolved case, decision blocker for a universal fix

**Confirmed mismatch; exact root unresolved.**

`0208-reflected-heal-encore-battle-182`, step 16, combines Mega Evolution, Encore and Giga Drain blocked by Heal Block. Side 1's inferred opponent Speed upper bound is 0.475 in the engine observation and 0.555 in the follower. The event sequence is reproduced, but this review does not yet prove which comparison should be accepted or rejected.

The architectural risk is clear: the shared [`Speeds::compare`](../../src/speed.rs:879) algorithm receives different histories. The engine supplies real queue snapshots; the follower [`digest`](../../src/follow.rs:2941) reconstructs pending actions from later visible actions, their classifications, known own choices and event snapshots. Sharing the narrowing algorithm does not guarantee that its evidence is equivalent.

**Required diagnostic before fixing this case:** record the first comparison that changes the two beliefs differently, its two actors, queued versus executed move, priority/fraction, Encore timing, snapshot and whether each input was publicly known. Determine whether the follower omitted valid evidence or the engine used hidden evidence. The smaller engine bound is not automatically the correct public bound.

**Acceptance cases:** opponent hidden original choice differs versus matches an Encore move; unchanged own choice; blocked healing with/without reflection; Mega replacement of an ability; repeated or promoted actions. If two histories have identical information for the observing player, their policy observations must agree even when hidden choices differ.

## What testing currently proves, and what it misses

[`check_follow`](../../src/replay.rs:1385) compares the engine's observation with the follower reading actual Showdown logs and requests. This is valuable: it finds training/deployment mismatches and checks legal actions. It is not an independent mathematical oracle for every inferred feature. Inferred Speed and protection probability are not simply fields Showdown sends in a request. Agreement of both implementations can still be wrong if they share an incorrect inference assumption.

The runner separately checks mechanics against recorded Showdown state. Zero mechanical divergences in the latest run support mechanical fidelity for those sampled games; they do not prove every mechanic in the codebase is correct.

The theme validator checks supported species/moves/abilities/items. The generator samples teams and actions. Neither currently requires evidence that a particular interaction boundary actually occurred. “2,000 targeted games passed” is therefore not “every lifecycle transition was tested.” The latest run demonstrates that limitation.

Before another fix is considered complete, require:

1. A small deterministic transition matrix that asserts the exact relevant observation before and after the event, including the negative/unrelated cases above.
2. Diagnostics that expose why a feature changed, especially the action identity and evidence used for a Speed bound or PP deduction.
3. A coverage ledger of transitions actually executed, not just battle/theme counts.
4. An observability check for any feature that depends on hidden choices or silent queue events.
5. The existing recorded cases and fresh randomized comparisons as additional verification.

Do not remove difficult comparisons or equate “unknown” with a definite zero to obtain a clean run.

## Recommended scope if you choose to proceed

| Work | Relative scope | Decision value |
| --- | --- | --- |
| Classify failure events and bound move-context lifetime | Small | Directly addresses the charge regression and guards against unrelated-event cleanup. |
| Correct target identity versus location for missing-target PP | Small | One concrete case, with finite tracking/redirection variants to check. |
| Specify residual effect birth/tick accounting | Medium | Replaces a recurring damage-source exception pattern. |
| Specify pending-action lifecycle and public uncertainty | Largest/first design priority | Addresses the 51-case family; may require an observation-contract decision rather than a precise reconstructed queue. |
| Trace the single Speed discrepancy | Bounded investigation; fix scope unknown | Establishes which evidence path is incorrect before changing inference. |

A full engine rewrite is not justified by this evidence. A staged observer correction with explicit contracts is justified. The small fixes can be reviewed separately, but they should include their negative-case matrices rather than only the new saved example.

Preserve the purpose of the Rust engine: fast training. Diagnostic traces should be opt-in and absent from the production hot path. Any shared representation should be measured before introducing allocations or replaying a full public log for every observation. No performance costs have been measured in this review.

## Questions this review leaves open

- Can the protection repetition feature always be recovered from the permitted public information? A witness-pair test is needed where hidden queue ordering differs but the player's visible information is the same.
- What uncertainty representation should the policy use if it cannot? The current protection feature is a definite value, not an explicit uncertainty distribution. Changing that contract may affect observation layout and training compatibility; that is a decision, not an automatic patch.
- Which exact comparison causes the current Speed mismatch?
- Are residual birth/tick rules consistent for terrain and other adjacent timers? This is a review item, not a newly confirmed failure.

**Enough to begin the bounded fixes and the lifecycle design review: yes. Enough to promise the whole observation system is correct afterward: no.** The completeness criteria above are intended to replace that promise with reviewable evidence.
