# Search and expert iteration: review so far

As of 2026-10-17. Results from Oscar; every match is 3,000 games on `teams/multi-tournament-warmstart/eval-current.json` with open sheets.

## Summary

One-turn search at play time is the clear win so far: the update-15342 network with search beats itself without search 70–72% of the time, up from 50%. Expert iteration round 1 produced a small gain with search (about 52% against the old network with search) but made the raw policy worse (about 45% without search). The main open question is why imitating the search hurts the policy on its own, and that has to be answered before round 2.

## What was built

All of this is merged to main (PRs #49–#57) and runs on Oscar through the scripts in `scripts/oscar/`.

| Piece | What it does | PRs |
| --- | --- | --- |
| One-turn search | Takes the policy's top 8 move pairs and 8 opponent replies, plays each pair out one turn in 4 sampled worlds, scores the results with the value net, solves the resulting zero-sum game and samples from the mixed strategy | #49 |
| Hidden-information fixes | Each sampled world redraws what a player can't see: the opponent's bench, stat spreads, exact HP within the shown bar, hidden countdowns (sleep, confusion, rampage, binding) and transformed Pokémon. Each fix is checked by a test that fails if the truth leaks | #50, #51 |
| Exploration | AlphaZero-style randomness: extra candidates beyond the policy's top picks, optionally chosen by noisy odds. It only adds candidates and never removes the policy's top ones | #52, #53 |
| Anchor (piKL) | Pulls the solved strategy toward the policy's own odds, with strength λ | #53 |
| Batched search | Searches many games at once on the GPU | #54 |
| New defaults | 4 dice rolls per cell, anchor λ = 0.03 | #55 |
| Expert-iteration data | Plays the network with search against a weighted league of frozen opponents and records the search's strategy and the game outcome for each turn | #56 |
| Expert-iteration trainer | Trains the policy toward the search's strategy (cross-entropy) and the value toward outcomes (0.5 × squared error), holding out whole games for evaluation | #57 |

## What worked: search

Search adds about 20 points over the same network without it, and the anchor at λ = 0.03 was the biggest single improvement. Every row is update 15342 with search against itself without search (about ±1.8 points at 95%).

| Setting | Score | Note |
| --- | --- | --- |
| Anchor 0.03, dice 4 (current default) | 70.4%, 71.6% | Two seeds |
| Anchor 0.03 + exploration | 71.4%, 71.1% | Exploration costs nothing here |
| Anchor 0.01 | 68.6% | |
| Anchor 0.1 | 68.3% | |
| Privileged (sees hidden information) | 68.0% | Without the anchor; an upper bound on what the leak fixes cost |
| Dice 4, no anchor | 65.1% | |
| Plain search | 64.5% | Two seeds |
| Explore 2 (no anchor) | 63.7% | |
| Explore replies 2 (no anchor) | 62.8% | |
| Anchor 0.3 | 60.0% | Too strong a pull back to the policy |
| Old exploration, all settings together | 58.5% | Replaced in #53 |

- The leak fixes cost only about 3.5 points compared with seeing everything (64.5% vs 68.0%), so the honest search isn't badly handicapped.
- Batched search takes 0.0154 s per search on a 3090, against 0.10 s before, which is what made 50,000-game data rounds affordable.

## Expert iteration round 1

Round 1 trained the network toward the search's choices. With search, the new network edges out the old one (about 52%). Without search, it loses (about 45%).

**Data.** 50,000 games, 505,327 turns (455,327 of them searched), 3.7 hours on one A5500, with exploration on (`--explore 2 --noise 0.25`). Scores of the learner with search against each league member:

| League member | Weight | Score |
| --- | --- | --- |
| Itself with search | 0.3 | 51.6% |
| Itself without search | 0.2 | 70.3% |
| Update 13450 | 0.2 | 68.3% |
| Louisville | 0.15 | 81.0% |
| Greedy | 0.15 | 88.2% |

**Training.** Every run started from update 15342. Held-out metrics at the end (baseline = 15342 before training):

| Run | KL to the search | Agreement | Value explained |
| --- | --- | --- | --- |
| Baseline | 1.119 | 0.633 | 0.351 |
| lr 3e-5, 2 passes | 0.770 | 0.583 | 0.383 |
| lr 1e-4, 2 passes | 0.724 | 0.590 | 0.383 |
| lr 3e-4, 2 passes | 0.691 | 0.599 | 0.383 |
| lr 3e-4, 8 passes | 0.671 | 0.609 | 0.381 |

Agreement is how often the policy's top move pair is the search's top pair. The best checkpoint was the last step in every run, so none of them overfit.

**Play.** New network against 15342:

| Run | With search on both sides | Without search on either side |
| --- | --- | --- |
| lr 3e-5, 2 passes | 50.9% | 44.6% [42.8, 46.4] |
| lr 1e-4, 2 passes | 50.3% | 43.9% [42.1, 45.6] |
| lr 3e-4, 2 passes | 51.9% [50.1, 53.7] and 52.8% [51.0, 54.5] (two seeds) | 44.8% [43.0, 46.6] |
| lr 3e-4, 8 passes | 51.3% [49.5, 53.1] | 46.8% [45.0, 48.6] |

## What has been weird

1. **The raw policy got worse while getting closer to the search.** Every learning rate lands at about 45% without search, even 3e-5, which barely moves the weights. So the learning rate is probably not the cause; the training targets are the likelier suspect.
2. **Agreement fell while KL fell.** The policy matches the search's whole distribution better but picks the search's top move less often (0.633 → 0.58–0.61). This fits the policy getting flatter: copying broad mixed strategies spreads probability over more moves. Evaluation samples from the policy rather than taking its top move, so a flatter policy plays weak moves more often. This is a hypothesis; nobody has measured the policy's entropy yet.
3. **Value explained is the same for every run.** It reaches 0.381–0.383 at every learning rate and for 2 or 8 passes, identical to the 4th decimal. I checked locally that this isn't a bug. The likely reason is that win/loss outcomes are noisy targets, so the value quickly reaches a ceiling.
4. **More training barely helps.** 8 passes moved KL from 0.691 to 0.671 and didn't change play with search. The round 1 data is mostly learned already.
5. **The search-vs-search gain is small.** About 52% against the old network, with the 95% interval barely above 50%. It's unclear whether the gain comes from the new policy or from the slightly better value.
6. **Exploration costs depend on the anchor.** Without the anchor, extra candidates cost 1–2 points (63.7%, 62.8% vs 64.5%). With it they cost nothing (71.4%, 71.1% vs 70.4%, 71.6%).

## Open questions

- Is the new policy actually flatter? We can measure its entropy against the old policy's on held-out positions.
- Would the new policy win without search if it played its top move instead of sampling? If yes, flatness is the problem, not move quality.
- Is the search's strategy broad because the position calls for mixing, or because of noise from the dice, the 4 sampled worlds and the value estimates?
- Does exploration in the data make the targets worse? Round 1 data was generated with exploration on (2 extra candidates, noise 0.25), so the search's strategy, and the training target, can put weight on randomly added moves. In play, exploration with the anchor cost nothing (71.4%, 71.1% vs 70.4%, 71.6%). That doesn't show whether the targets it produces are good to learn from. The trainer can test this on existing data by dropping the explored candidates from the target, since the data records which ones they were.
- Does the round 1 gain with search come from the new policy or the new value? Pairing the new policy with the old value, and the reverse, would tell.
- Is the small network the bottleneck? A KL that stops falling at about 0.67 could mean it can't represent the search's choices.
- Answered: the 8-pass network loses a little less without search (46.8% vs 44.8%) and is no better with search (51.3% vs 51.9% on the same seed).
- Answered: the 1e-4 run scores 43.9% without search, the same as 3e-5 and 3e-4. The learning rate isn't the cause.

## Possible next course of action

Diagnose before building: the steps below need no new code or only small trainer and evaluation options, and they decide what round 2 should change.

1. **Finish the round 1 picture.** Done. All four networks score 43.9–46.8% without search and 50.3–52.8% with it.
2. **Measure flatness.** Compare the old and new policies' entropy on held-out positions, and play policy-only matches using the top move instead of sampling. If the new policy wins when it plays its top move, the problem is flatness, not move quality.
3. **Split policy from value.** Search with the new policy and old value, then the reverse, against the old network with search. This shows which half produced the 52%.
4. **Test capacity.** Train fresh medium and large networks on the round 1 data. If large gets clearly lower KL and higher value explained, capacity was the limit.
5. **Then pick a round 2 fix based on what 2–4 show:**
   - Too flat → sharpen the targets (a temperature below 1, or train on the search's top move), drop explored candidates from the targets, or reduce the noise behind them (more worlds and dice when generating data).
   - Move quality got worse → add a pull toward the old policy (a KL term) during training.
   - KL stuck and the policy limited → move to the bigger network.
   - Value is the gain → consider blending the search's own value estimate into the value target to reduce outcome noise.
6. **Round 2 data** only after the fix has been chosen, using the winner of round 1 as both the learner and a new league member.

Later items, not started: a round-robin panel of checkpoints, scenario tests and event tables, a Showdown client with search, and sampling sets for closed team sheets.
