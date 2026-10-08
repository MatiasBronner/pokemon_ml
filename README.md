# vgc-engine

A fast battle engine for Pokémon Champions doubles (VGC), written in Rust and
checked decision by decision against Pokémon Showdown. It is meant as the
simulator underneath a self-play bot: small copyable state, no allocation in
the turn loop, and a harness that proves each mechanic matches Showdown before
anything is trained on it.

**Status: a verified core, not yet a full simulator.** The turn loop, damage,
status conditions, switching and 231 of the 510 moves Champions Pokémon can
learn are modelled. Abilities, held items, Mega Evolution, weather, terrain
and field effects are not, so real tournament teams cannot be played yet. See
[What is and is not modelled](#what-is-and-is-not-modelled).

## Quick start

```sh
cargo test --release                       # includes 16 recorded Showdown battles
cargo run --release --example random_battle
```

```rust
use vgc_engine::{Battle, PokemonSet, Request};

// The four Pokémon each side picked at team preview; the first two lead.
let p1 = [PokemonSet::from_names("Garchomp", &["Earthquake", "Protect"], "Jolly", [0, 32, 0, 0, 2, 32])?, /* ... */];
let mut battle = Battle::new([&p1, &p2], [1, 2, 3, 4])?;   // seed: Showdown's four 16-bit words

while !battle.ended {
    // battle.request says what is being asked: Request::Move or Request::Switch.
    let c1 = pick(battle.joint_choices(0));   // every legal pair of slot choices for side 0
    let c2 = pick(battle.joint_choices(1));
    battle.choose([c1, c2])?;
}
println!("{:?}", battle.winner);              // Some(0), Some(1) or None for a tie
```

`Battle` is `Copy`: cloning a position for search is a plain memory copy.
Choices use Showdown's conventions (`Choice::to_showdown` prints `move 2 1`,
`switch 3`, `pass`), so they can be sent to a Showdown server unchanged.

## How it is checked

`oracle/gen_cases.js` plays random battles in Pokémon Showdown's own simulator
under `[Gen 9 Champions] VGC 2026 Reg M-C` and records, at every decision, the
choices made, the choices that were legal, and the full state afterwards
including the RNG seed. `difftest` replays each battle here from the same seed
and compares.

That only works because the engine draws random numbers in exactly the order
Showdown does. Showdown consumes its RNG in places with no in-game meaning
(re-resolving targets each time it re-sorts the action queue, shuffling speed
ties on every internal update pass), and the engine reproduces those draws
too. The benefit is that any mechanical mistake desynchronises the two RNG
streams and shows up within a turn or two.

Results for the code in this repository, against Showdown commit `ad7ca5d`
(8 October 2026):

| Check | Battles | Decisions | Diverged |
|---|---|---|---|
| State, RNG seed and legal choices after every decision | 10,000 | 185,823 | 0 |
| Every individual RNG draw (range and value, in order) | 1,500 | 27,288 | 0 |
| 1,000-turn limit (both sides only ever switch) | 6 | 6,000 | 0 |

The random teams are drawn from all 292 bringable species with at least four
modelled moves; each battle features one modelled move on a lead so all 231
are exercised (least-used move: 67 uses). About a third of battles are mirror
matches or single-species teams, so that speed ties are constant. A few
percent have no attacking moves at all, so they are decided by Struggle.

To confirm the comparison has teeth, twelve deliberate bugs were injected one
at a time (paralysis odds, spread-move modifier, toxic ramp, freeze timer,
drain rounding, Struggle recoil, the last-to-faint tiebreak, a skipped tie
shuffle, and others). Every one was caught.

What this does **not** establish: agreement with the cartridge games where
they differ from Showdown, or anything about mechanics outside the modelled
set. Both simulators ran with every Pokémon given no ability and no item.

### Running it yourself

```sh
scripts/setup-oracle.sh            # clone and build the pinned Showdown commit (needs Node 22+)
scripts/fuzz.sh 2000               # 2,000 fresh battles, random seed
TRACE=1 scripts/fuzz.sh 300 42     # also compare every RNG draw
```

When a battle diverges, `difftest` prints the fields that differ. Re-record
that one battle with tracing to see where the RNG streams part:

```sh
node oracle/gen_cases.js --seed 42 --only 17 --trace --out one.jsonl
cargo run --release --features trace --bin difftest -- one.jsonl
```

The report lines up the engine's draws (each labelled with its purpose)
against Showdown's (each labelled with its call stack) and marks the first one
that differs, followed by Showdown's log for the turn.

## What is and is not modelled

Modelled, and verified as above:

- Champions stat formula, natures, level 50, Champions PP values
- Action order: priority, speed, speed ties, switches before moves, dynamic
  re-sorting after each action
- Targeting in doubles: chosen targets, spread moves and their 0.75 modifier,
  retargeting when a foe has fainted, moves aimed at an ally
- Damage: critical hits, damage rolls, STAB, type chart and immunities, burn
- Accuracy and evasion stages, stat stages
- Burn, paralysis, poison, toxic, sleep and freeze with the Champions rules
  (1-in-8 full paralysis, sleep for 2 or 3 turns, freeze for at most 3)
- Flinching, Protect with its consecutive-use counter
- Secondary effects, self stat changes, draining, recoil, healing, multi-hit
  moves, stat-override moves (Body Press, Foul Play, Psyshock)
- Switching, fainting, replacements, PP, Struggle, win and tie conditions,
  the 1,000-turn limit

Not modelled yet. `Battle::new` returns `Error::Unsupported` for a move it
does not model rather than guessing:

- **Abilities and held items** (every Pokémon behaves as if it had neither)
- **Mega Evolution**
- Weather, terrain, Trick Room, Tailwind, screens, hazards
- Fake Out, Follow Me / Rage Powder, Helping Hand, Wide Guard, Quick Guard
- Pivoting moves (U-turn, Parting Shot), forced switches (Roar)
- Two-turn and recharge moves, Substitute, Encore, Taunt, Disable, confusion
- Team preview itself: the engine starts from the four Pokémon picked

`oracle/coverage.json` lists every unmodelled move and the Showdown feature
blocking it. The largest groups are moves with their own lingering condition
(75), moves with a scripted `onHit` (63) or `onTry` (30), and variable base
power (29).

## Layout

```
src/battle.rs      the turn loop; a function-by-function port of Showdown's sim
src/state.rs       fixed-size state: Battle, Side, Pokemon, the action queue
src/data.rs        move/species/type definitions
src/tables.rs      generated from Showdown's data (do not edit)
src/rng.rs         Showdown's Gen5RNG
src/replay.rs      replays a recorded battle and reports the first difference
src/trace.rs       optional RNG/action trace (feature `trace`)
src/bin/difftest.rs, src/bin/bench.rs
oracle/gen_data.js   Showdown data  -> src/tables.rs, pool.json, coverage.json
oracle/gen_cases.js  Showdown battles -> recorded cases (JSON lines)
tests/parity.rs    fixture of recorded battles, choice-validation checks
```

After changing which moves are supported (`oracle/lib.js`) or updating
Showdown, run `node oracle/gen_data.js` to regenerate the tables.

## Speed

`cargo run --release --bin bench -- tests/fixtures/showdown_cases.jsonl`
plays random battles to completion. On one core of a 2.1 GHz cloud Xeon it
runs about 20,000 battles (360,000 decisions) per second. Treat that as an
upper bound: the mechanics still to come will slow it down, and nothing has
been optimised yet (legal-move enumeration still allocates).

## Suggested order for what comes next

1. **Abilities and items as event hooks**, starting with the ones on most
   teams (Intimidate, Focus Sash, Sitrus Berry, Choice items, Life Orb). This
   is the largest structural addition: Showdown's handler ordering and its
   tie-breaking draws have to be reproduced for each event.
2. **Mega Evolution** (new action type, forme change mid-turn).
3. **Fake Out, Tailwind, Trick Room, weather and terrain.**
4. **Redirection, Helping Hand, Wide Guard, pivoting moves.**
5. **Building a `Battle` from an arbitrary mid-battle state**, which a bot
   needs to search from a live game, and sampling hidden information into it.
6. Python bindings and batched stepping for training.

Each step can be driven by the same loop: widen the supported set in
`oracle/lib.js`, regenerate, run `scripts/fuzz.sh`, and fix what diverges.

## Provenance

The tables in `src/tables.rs` and the behaviour in `src/battle.rs` are derived
from [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) (MIT
licence). Showdown is the reference for correctness here, not the cartridge
games; Smogon's own research into Champions mechanics is still in progress.
