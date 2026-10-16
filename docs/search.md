# Search at play time

Until now the bot played straight from its policy network: one forward pass,
one sampled pair of actions. `pokemon_ml.search` adds what the strongest
Showdown bots are described as doing: look one turn ahead, score every
combination of both sides' choices with the value network, and play the
mixed Nash equilibrium of that grid. Training is unchanged. Search runs on
any existing checkpoint, so it can be tested on the preserved update-15342
policy without retraining.

## How a decision is made

At a battle decision where the searching side has more than one legal pair
of actions and the team sheets are open:

1. **Candidates.** The policy's `--candidates` most likely pairs (default 8),
   from the searching side's own observation.
2. **Worlds.** `--worlds` copies of the game (default 4). In each copy,
   what the searching side has not been shown is drawn again
   (`Game::determinize` in `src/env.rs`):
   - which registered Pokémon wait unseen on the bench, drawn from those not
     yet seen;
   - every opposing Pokémon's stat points. Speed is drawn from what the
     move order still allows (`Speeds::belief`), and the rest follows the
     `guess_spread` rule of thumb. HP is set so the bar shows the same
     percentage and colour as before.

   Each copy rolls its own dice, never the real game's.
3. **Replies.** In each world, the other side's `--replies` most likely pairs
   (default 8), from the same network looking from that side of that world.
4. **Payoffs.** Every (candidate, reply) pair is played one turn in the
   engine, `--dice` times (default 2). The value network scores each
   resulting position from the searching side's view; a finished game
   scores its result (+1, 0 or −1). The default settings give 8 × 8 × 4 × 2
   = 512 positions per decision, evaluated in one batch.
5. **Equilibrium.** The grids are solved as one zero-sum game (`solve`):
   - The searching side picks one mixed strategy, because it does not know
     which world is real.
   - The other side picks a best reply in each world, because it knows its
     own team.
   - The solver is regret matching+ with linear averaging, and reports its
     remaining exploitability (`gap`).

   The pair played is drawn from the searching side's mixed strategy, or,
   with `--pick best`, the pair it plays most often.

The policy still decides Team Preview, forced or single-option decisions,
and (unless `--closed-sheets`) games with closed team sheets.

## Exploration, for games to learn from

By default, the search only considers the policy's favourite pairs. That is
right for play. It is wrong for producing training data: a move the policy
ranks below 8th is never tested, so a network trained on the search would
keep the policy's blind spots. Like AlphaZero, the search can explore
(`choose` in `search.py`):

| Setting | What it does | AlphaZero's equivalent |
|---|---|---|
| `--noise 0.25` | Mixes 25% Dirichlet noise into the policy's odds before the candidate pairs are picked, so lower-ranked pairs sometimes make the cut. | Dirichlet noise on the root priors (ε = 0.25) |
| `--noise-alpha` | The noise's concentration. The default, 0, means 10 divided by the number of legal pairs: lower values put the noise on fewer pairs. | α scaled to the typical number of legal moves |
| `--explore 2` | Adds 2 candidate pairs drawn uniformly from the legal pairs left out. | — |
| `--explore-replies 2` | The same for the opponent's replies in each world, so the data also covers unexpected opponent moves. | — |

Explored pairs are scored and solved like any other. The noise and
`--noise-alpha` apply to the opponent's replies too.

**Exploring costs strength, not just compute.** The first version of this
section said a bad explored pair would simply get no weight. On Oscar,
update 15342 with search against itself without search (seed 9502347,
3,000 games each) said otherwise:

| Search | Score | 95% interval | Weight on explored pairs |
|---|---|---|---|
| Plain (two seeds) | 64.5% | ≈63.3–65.7% | — |
| `--noise 0.25 --explore 2 --explore-replies 2` | 58.5% | 56.8–60.3% | 19.6% |

The equilibrium gave explored pairs about a fifth of its weight and lost six
points. The likely cause is the optimiser's curse. The payoffs are value
network estimates, and noisy ones. The more options the search weighs, the
likelier one of them looks good by luck alone. The moves exploration brings
in are the ones the policy rarely plays, and so the ones the value network
has seen least. Do not generate training data with exploration on its own:
pair it with `--anchor` (next section), and check its score first.

The report adds:
- `mean_explored_weight`: the equilibrium's average weight on pairs that
  only exploration brought in;
- `played_explored_pair`: how often such a pair was played.

If the weight is clearly above 0, the policy is missing moves the search
finds.

**All of it is off by default. Keep it off for evaluation and real games.**

## Holding the search to the policy (piKL)

`--anchor λ` changes what the equilibrium maximises: payoff, less λ times how
far each side's strategy strays from its policy's own odds over the pairs
searched (KL divergence). This is piKL, from Meta's Diplomacy work. The
searching side is held to its policy's odds. The opponent, in each world, is
held to the opponent policy's odds.

The effect is that a pair the policy rates low gets weight only if the value
network finds it clearly better, not slightly better by luck. λ is in units
of win probability:

- At λ = 0.03, three points of extra value multiply a pair's odds by e.
- λ = 0 is the plain equilibrium, and the default.
- A very large λ plays the policy's own odds.

The anchored game is solved by magnetic mirror descent with optimistic steps
(`solve` in `search.py`). On grids of this size its gap reaches 1e-9 or less
for λ ≥ 0.01 in 1,000 iterations. Below 0.01 it converges more slowly, and
the gap in the report shows it.

The tests check:
- the closed-form answer against a single reply (prior × exp(payoff / λ));
- convergence across λ;
- the move from the plain equilibrium (small λ) to the policy (large λ);
- that a one-point edge on a pair the policy doubts is resisted, while a
  clear one is taken;
- that a very strong anchor makes the search play the policy's own
  favourite, which checks the policy's odds actually reach the solver.

A sweep, without and with exploration:

```bash
for lam in 0.01 0.03 0.1 0.3; do
  SEARCH_OUT=runs/search/15342-self-anchor$lam.json \
    sbatch $SLURM_OPTS --time=02:00:00 scripts/oscar/search-eval.sbatch --seed 9502347 --anchor $lam
done
for lam in 0.03 0.1; do
  SEARCH_OUT=runs/search/15342-self-anchor$lam-explore.json \
    sbatch $SLURM_OPTS --time=02:00:00 scripts/oscar/search-eval.sbatch --seed 9502347 \
    --anchor $lam --noise 0.25 --explore 2 --explore-replies 2
done
```

What to look for:
- **Some λ above 64.5% without exploration:** part of the plain search's
  noise was costing it. Use that λ from then on.
- **Exploration that no longer loses points at some λ:** that is the
  setting to generate training data with.

## What is and is not hidden

A redrawn copy has to be drawn from what the searching side has seen, and
from nothing else. What `Game::determinize` draws again:

| Hidden from a player | How it is drawn |
|---|---|
| The opposing bench | Each unseen spot gets a registered Pokémon not yet seen. |
| Opposing stat points | Speed comes from what move order allows (`Speeds::belief`). The other stats follow the `guess_spread` rule of thumb. HP points are moved as little as needed for the shown bar to be possible, borrowing from Attack and the defences; every amount from 0 to 32 is tried, so a fit is always found without falling back on the real spread. |
| Exact opposing HP | Uniformly among every HP that shows the same percentage and bar colour, transformed Pokémon included (Transform keeps a Pokémon's own HP). |
| What a transformed opponent copied | If it turned into a teammate, it takes that teammate's redrawn stats, not the real ones. If it turned into one of the searching side's Pokémon, it copied stats that side already knows. |
| Sleep length (2 turns one time in three, else 3) | Only lengths longer than it has already lasted, as often as the game would draw them. Rest's fixed 3 turns are kept. |
| Confusion (2–5 attempts), rampage (2–3 turns), binding (5–6 turns) | The same way. |

The four countdowns are redrawn for **both** sides. Showdown doesn't tell a
player its own Pokémon's sleep length either. To know how long a countdown
has visibly lasted, the engine records what each one was drawn at
(`Pokemon::drawn`, never shown, exported with positions). Likewise, a
transformed Pokémon records whom it copied (`Pokemon::copied_from`).

These tests check it:

- **The copy looks the same** (`tests/search.rs`). At every decision of
  1,500 random games, the searching side's observation of a redrawn copy is
  identical to its observation of the real game, number for number.
- **The copy depends on nothing hidden** (`tests/search.rs`). The real game
  and a different possible truth are both redrawn with the same random
  numbers. The results must be identical, and must stay identical when
  played three turns further with the same dice and choices. 59,932
  positions were checked.
- **The second truth is changed by hand** (the unit test in `src/env.rs`).
  The check above makes its second truth with `determinize` itself, so it
  cannot catch something `determinize` forgets to redraw. Here the second
  truth is changed independently: other countdown lengths, HP moved within
  its bar, other attack, defence and Speed points, and a different unseen
  bench Pokémon. That is 192,337 hidden values over 24,656 positions.
- **Targeted cases** (also in `src/env.rs`). Random games rarely produce
  these:
  - A Ditto transformed into a foe (Imposter) or into its own teammate (the
    move Transform), with its HP and its teammate's stats changed by hand.
  - An opposing Gholdengo whose guessed spread leaves too few HP points to
    show its bar (such as 37/185, which is 20% on a red bar), with two truths
    that differ only in hidden Special Attack.
  - A property check over every species: fitting HP to a bar never needs
    the real spread.

To check the tests themselves, each fix was undone in turn:
- HP scaled from the truth;
- countdowns copied, on both sides or on the searching side only;
- the real attack and defence points, or the real Speed;
- the real bench;
- a transformed Pokémon's real HP;
- a teammate's real copied stats;
- the real spread kept when the guess cannot show the bar.

The tests caught every one.

Two of those leaks were found by review after the first fix was merged. The
random-game tests had passed over 2,000 games because both cases are rare in
random play: transformed Pokémon, and a guess that can't show a "picky" bar
such as 20% on red, which needs a max HP divisible by 5. That is why the
targeted tests exist.

What the search still knows that a player would not:

- **With closed sheets**, the opponent's unrevealed moves, items and
  abilities are the real ones. That is why closed-sheet games fall back to
  the policy unless `--closed-sheets` is given. Sampling those sets is the
  next step (roadmap item 3).
- **On a team that registered an Illusion Pokémon**, the bench is not
  redrawn, because disguises depend on where teammates stand. Stat points
  still are.
- **Axe Kick's confusion** lasts 3–5 attempts, not 2–5. The redraw ignores
  the difference, so it may draw a length the game could not. That is a
  wrong guess, not a leak.
- **A position written by hand** has no record of what its countdowns were
  drawn at, or of whom a transformed Pokémon copied. Those countdowns, and
  the copied stats, are kept as written.

Natures and items on open sheets are given by the sheet, so they are
legitimately known. PP and stat stages are public: every move used and every
stat change is shown.

`--privileged` searches the real game without redrawing anything. Compare it
with the honest search to see how much hidden information is worth.

## Running it on Oscar

Every run plays one checkpoint with search against a copy of itself without
search (`--opponent self`). Searching sides alternate between player 1 and
player 2, and each slot's first game counts once, as in `evaluation.py`.
The job script defaults to the preserved
`runs/plateau-comparison-15342/source-checkpoint.pt` and
`teams/multi-tournament-warmstart/eval-current.json`. `SEARCH_CHECKPOINT` and
`SEARCH_POOL` override them.

On the login node, update the checkout and rebuild the extension, since the
Rust side changed:

```bash
cd /users/mcbronne/pokemon_ml
git switch main && git pull --ff-only
module load python/3.12.8-zr3c
source .venv/bin/activate
maturin develop --release
python -m pytest python/tests/test_search.py -q
export POLICY_REPO=$PWD
SLURM_OPTS="--partition=gpu --account=default --qos=norm-gpu --constraint=geforce3090"
```

A smoke test first (40 games; it also reports `seconds_per_search` for sizing the real runs):

```bash
SEARCH_GAMES=40 SEARCH_OUT=runs/search/smoke.json \
  sbatch $SLURM_OPTS --time=01:00:00 scripts/oscar/search-eval.sbatch --seed 1
```

Then the comparison:

```bash
# The main comparison: two seeds, 3,000 games each.
SEARCH_OUT=runs/search/15342-self-seed9502347.json sbatch $SLURM_OPTS scripts/oscar/search-eval.sbatch --seed 9502347
SEARCH_OUT=runs/search/15342-self-seed9602347.json sbatch $SLURM_OPTS scripts/oscar/search-eval.sbatch --seed 9602347

# How much the remaining hidden information is worth (an upper bound, not a fair player).
SEARCH_OUT=runs/search/15342-self-privileged.json sbatch $SLURM_OPTS scripts/oscar/search-eval.sbatch --seed 9502347 --privileged

# Cheaper and wider settings, to see where the gain comes from.
SEARCH_OUT=runs/search/15342-self-w1d1.json sbatch $SLURM_OPTS scripts/oscar/search-eval.sbatch --seed 9502347 --worlds 1 --dice 1
SEARCH_OUT=runs/search/15342-self-c12r12.json sbatch $SLURM_OPTS scripts/oscar/search-eval.sbatch --seed 9502347 --candidates 12 --replies 12
```

Set `SEARCH_OPPONENT=greedy` or a checkpoint path to play something other
than itself. `SEARCH_GAMES` and `SEARCH_ENVS` override the defaults of 3,000
and 64.

The report (`SEARCH_OUT`) contains:

- the score with a 95% Wilson interval, as in the other evaluations;
- `searched_decisions` and `seconds_per_search`;
- `mean_equilibrium_gap`, which should stay near 0;
- `played_policy_top_pair`: how often search played the policy's own first
  choice;
- `mean_mixed`: how far from a single pair the equilibrium is;
- `mean_shift_from_policy`: the total-variation distance between the
  equilibrium and the policy's own odds over the candidates.

**Speed has not been measured on a GPU.** On two CPU threads in a container,
a 260k-parameter network took about 0.34 s per decision at the default
settings (512 positions) and 0.14 s with
`--candidates 6 --replies 6 --worlds 2 --dice 1`. The engine part is cheap,
about 22 µs a decision per core. Most of the cost is the value network's
batch, which a GPU should make much smaller. Check `seconds_per_search` in
the first report before sizing `--time`. The 8-hour default is a guess.

## What would be worth doing next

1. **If search helps:** play the Showdown client with it, which needs
   positions rebuilt from `Follower` instead of `VecEnv`. Also train the
   value network on searched games, since under search it is the whole
   decision.
2. **If it does not:** the value network is the suspect. Its explained
   variance stayed around 0.42 through training, and its calibration
   worsened (`docs/training-findings-2026-10-13.md`). The privileged run
   separates "the value network is wrong" from "the redrawn worlds are
   wrong".
3. Sample the opponent's unrevealed sets so closed-sheet games can be
   searched too.
