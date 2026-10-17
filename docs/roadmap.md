# Roadmap: from self-play to beating pro players

As of 2026-10-17. The goal is an agent that beats professional VGC players. The
biggest gap is not the network but measurement: every result so far is bot against
bot, and the agent has never been measured against a human. Pros will find
weaknesses that self-play never tests, so a human benchmark has to come early, not
last.

## Where we stand

| What exists | What is missing for beating pros |
| --- | --- |
| A Rust engine checked against Showdown (`docs/running-showdown-comparisons.md`) | Any measurement against human players |
| Update 15342 plus one-turn Nash search: 70–72% against the same network without search (`docs/search.md`) | Search in the Showdown client, which plays policy-only today |
| An expert-iteration pipeline; round 1 gave about +2 points with search (`docs/expert-iteration-round1-review.md`) | A model of how humans play, for the opponent's replies inside the search |
| A large network (10.4M parameters) distilled from 15342, then self-play with 15342 kept in the league (`docs/distillation.md`) | Closed-team-sheet play: the Showdown ladder hides sets, and the search assumes open sheets |
| A Showdown client that can play on the ladder (`pokemon_ml.showdown`) | Robustness: nothing in training hunts for the agent's exploitable habits |

## Phase 1: a stronger network

Now, a few weeks.

1. **Distil, then self-play.** Distil 15342 into the large network
   (`scripts/oscar/distill.sbatch`). Check it against 15342 without search;
   about 50% means it has caught up. Then run PPO self-play from it, with 15342
   fixed in the league (`scripts/oscar/finetune.sbatch`).
2. **Expert iteration with the large network as the learner.** Use the current
   learner for each round's data, and 15342 with search as a frozen league member.
3. **Better value targets.** Round 1 trained the value on game outcomes only, which
   are noisy: a 70%-winning position still ends as a loss 30% of the time. Value
   explained stalled at about 0.38. Options, cheapest first:
   - **Blend in the search's value.** The target becomes a mix of the outcome and
     the search's equilibrium value. The data already records it (the `value`
     field). Test 100% outcome, 50/50 and mostly search value on existing data, and
     judge by search-vs-search play, not by value explained.
   - **Bootstrap from a few turns later** (an n-step return).
   - **A deeper search, for training data only.**

   Full MCTS with UCT fits VGC poorly:
   - both players move at once, so every node needs a game-theoretic solve;
   - hidden information causes strategy fusion in deep trees over sampled worlds;
   - each extra turn of depth multiplies the cost by about 100;
   - the leaves still rely on the value network.

**Milestone:** the large network with search beats 15342 with search clearly (above
55% over 3,000 games).

## Phase 2: a human benchmark

Start in parallel with Phase 1. This is the most important step.

1. **Search in the Showdown client.** The client already turns Showdown's messages
   into the network's observations (`Follower`, `src/follow.rs`). Search also needs
   the engine's own game state, rebuilt from those messages, so it can sample worlds
   the same way it does in self-play. This is the main new engineering. Without it,
   the bot plays in front of people about 20 points weaker than it can.
2. **Ladder games.** `pokemon_ml.showdown --ladder` already exists. The ladder
   rating against real players becomes the number we track, next to self-play
   results.
3. **Closed team sheets.** The ladder hides opponents' sets. The search must sample
   items, moves and spreads from usage statistics, as it already samples stat
   spreads. Official VGC tournaments use open team sheets, so this matters for the
   ladder, not for tournament play.

**Milestone:** a first ladder rating, even with the small network plus search. It is
the baseline every later improvement is measured against.

## Phase 3: learn from humans

Once Phase 2 shows where the agent loses.

1. **Human replays.** Showdown publishes replays of the format. A human-like
   policy trained on them gives:
   - a much better model of the opponent's replies inside the search, which
     today assumes the opponent plays like our bot;
   - a prior for Team Preview and leads.
2. **Opponent modelling.** Strong players follow meta patterns: common leads,
   Protect timings, Tera timing. A search that expects human replies can punish
   them; one that assumes a perfect opponent plays too defensively. The piKL anchor
   (`docs/search.md`) is the same tool Diplomacy agents used to stay close to human
   play while improving on it.

## Phase 4: robustness

Before serious pro testing.

1. **Exploiter agents,** as in AlphaStar. Separate networks are trained only to
   beat the current agent, and whatever they find, the main agent then trains
   against. Pros are exploiters in effect, so this is how the agent prepares for
   them.
2. **Scenario tests and event tables.** These are checks for known bot blunders:
   wasted Protects, poor switches into spread moves, Tera timing, speed control.
   They catch rare situations that overall win rates hide.

## Phase 5: tournament conditions

1. **Team Preview against a known opponent team.** In Bring 6 Pick 4 this is
   decisive, and it deserves its own search.
2. **Time limits.** Search at 0.015 s per decision is far inside Showdown's
   timers, which leaves room for a deeper search at play time.
3. **Team selection.** A pro brings a strong, prepared team; the agent should be
   tested on teams of that quality.

## Order of work after distillation

1. **Now, already built:** test the distilled student against 15342, then run PPO
   self-play from it with 15342 fixed.
2. **Next new engineering:** search in the Showdown client.
3. **Small:** a value-blend option for expert iteration.
4. **Baseline:** a first ladder rating with the current best network plus search.

## Milestones and expectations

1. **Beat existing VGC bots.**
2. **Reach a high ladder rating.**
3. **Reach the top of the ladder.**
4. **Win against pros in arranged matches,** over enough games to be meaningful.

AI has beaten top humans in poker, Diplomacy and StarCraft. Like VGC, those are
games with hidden information and simultaneous moves. Each took large compute,
human data and serious robustness work. VGC also has a lot of luck (critical hits,
accuracy, speed ties), so even an agent better than the pros will lose many single
games, and proving it is stronger needs many matches.
