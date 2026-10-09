# vgc-engine

A fast battle engine for Pokémon Champions doubles (VGC), written in Rust and
checked decision by decision against Pokémon Showdown. It is meant as the
simulator underneath a self-play bot: copyable fixed-size state, no allocation
on the common paths of the turn loop, and a harness that proves each mechanic
matches Showdown before anything is trained on it.

**Status: moves, abilities, held items and Mega Evolution; not yet a full
simulator.** Modelled so far: the turn loop, damage, status conditions,
switching, Mega Evolution, 238 of the 510 moves Champions Pokémon can learn,
181 of the 225 abilities and 79 of the 85 held items. Weather, terrain and the
other field effects are not, and neither are staples such as Fake Out,
Tailwind and Trick Room, so most real tournament teams still cannot be played.
See [What is and is not modelled](#what-is-and-is-not-modelled).

## Quick start

```sh
cargo test --release                       # includes recorded Showdown battles
cargo run --release --example random_battle
```

```rust
use vgc_engine::{Battle, PokemonSet};

// The four Pokémon each side picked at team preview; the first two lead.
let p1 = [
    PokemonSet::from_names("Arcanine", &["Flare Blitz", "Protect"], "Adamant", [32, 32, 0, 0, 2, 0])?
        .ability("Intimidate")?
        .item("Sitrus Berry")?,
    /* ... */
];
let mut battle = Battle::new([&p1, &p2], [1, 2, 3, 4])?;   // seed: Showdown's four 16-bit words

while !battle.ended {
    // battle.request says what is being asked: Request::Move or Request::Switch.
    let c1 = pick(battle.joint_choices(0));   // every legal pair of slot choices for side 0
    let c2 = pick(battle.joint_choices(1));
    battle.choose([c1, c2])?;
}
println!("{:?}", battle.winner);              // Some(0), Some(1) or None for a tie
```

`Battle` is `Copy`: cloning a position for search is a plain memory copy
(about 5.2 kB). Choices use Showdown's conventions (`Choice::to_showdown`
prints `move 2 1`, `move 1 2 mega`, `switch 3`, `pass`), so they can be sent
to a Showdown server unchanged. A Pokémon holding its Mega Stone is offered
every move a second time with `mega: true`; a side can Mega Evolve once. `Battle::new` returns `Error::Unsupported` naming the first
move, ability or item it does not model, rather than guessing.

A set built with `from_names` has no ability and no item until you add them,
and takes the species' fixed gender or male. The engine does not check that a
species can legally have an ability, move or item; neither does Showdown's
simulator (its team validator does that separately).

## How it is checked

`oracle/gen_cases.js` plays random battles in Pokémon Showdown's own simulator
under `[Gen 9 Champions] VGC 2026 Reg M-C` and records, at every decision, the
choices made, the choices that were legal, and the full state afterwards
including the RNG seed. `difftest` replays each battle here from the same seed
and compares.

That only works because the engine draws random numbers in exactly the order
Showdown does. Showdown consumes its RNG in places with no in-game meaning: it
re-resolves targets each time it re-sorts the action queue, and it sorts the
handlers of every event and shuffles the ones that tie in speed. The engine
reproduces those draws too. The benefit is that any mechanical mistake
desynchronises the two RNG streams and shows up within a turn or two.

Results for the code in this repository, against Showdown commit `ad7ca5d`
(8 October 2026):

| Check | Battles | Decisions | Diverged |
|---|---|---|---|
| Mega Evolution with abilities and items (about 3,500 Mega Evolutions, 71 different Megas) | 5,500 | 101,656 | 0 |
| Abilities and items, no Mega Stones | 21,000 | 393,381 | 0 |
| Items but no abilities | 5,000 | 93,086 | 0 |
| Neither (how the engine's first version was checked) | 15,200 | 270,196 | 0 |
| Comparing every individual RNG draw as well | 1,400 | 27,412 | 0 |
| 1,000-turn limit (both sides only ever switch) | 8 | 8,000 | 0 |

Each row compares state, RNG seed and legal choices after every decision.

How the battles are made up:

- Teams are drawn from the 292 bringable species with at least four modelled
  moves. Each battle puts one modelled move, one modelled ability and one
  modelled item on a lead, cycling through all of them, so every one gets its
  share.
- Abilities are the species' own half the time and any modelled ability
  otherwise, which exercises abilities on bodies and movesets their real
  owners lack. About 80% of Pokémon hold an item.
- A species with a Mega Stone holds it half the time, and Mega Evolves at its
  first chance half the time, so Megas arrive early and late. A few Pokémon
  hold a stone they cannot use.
- About 30% of battles are "themed": every Pokémon draws from the same one to
  three abilities and items, so effects meet themselves and each other
  (Intimidate into Defiant, two Lightning Rods, Unnerve against berries) far
  more often than uniform sampling would manage.
- About a third are mirror matches or single-species teams, so that speed
  ties, and with them tie-breaking draws, are constant. A few percent have no
  attacking moves and are decided by Struggle.

Two further checks:

- **Legal choices.** The recorder lists legal choices from the Pokémon's real
  state rather than from the request Showdown sends the player, because the
  request deliberately hides some things (a Shadow Tag trap not yet revealed).
  `gen_cases.js --check-legal` confirms that list against Showdown's own
  validation by submitting every conceivable choice: 500 battles, 9,248
  decisions, no disagreement.
- **Does the comparison have teeth?** `scripts/mutation_test.py` injects one
  small bug at a time (Life Orb's multiplier off by 1/4096, Intimidate
  lowering by two stages, Mold Breaker ignored, Sitrus Berry restoring a third) and
  replays recorded battles. All 126 injected bugs that can make a difference
  were caught. Two more went unnoticed and turned out not to be bugs: one
  changed nothing observable, the other only matters for abilities that are
  not modelled yet.

What this does **not** establish: agreement with the cartridge games where
they differ from Showdown, or anything about mechanics outside the modelled
set. Coverage of rare interactions is also uneven: every modelled ability and
item was brought into hundreds of battles, but a rare interaction, such as
Corrosion actually mattering (a Poison- or Steel-type being poisoned), comes
up only a few times in several thousand.

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
that differs, followed by Showdown's log for the turn. To hammer on particular
effects, `gen_cases.js --abilities intimidate,defiant --items whiteherb` makes
every battle a themed one over just those.

## How abilities and items work

The engine has a port of Showdown's event system (`src/events.rs`): `runEvent`,
`singleEvent`, `eachEvent` and `fieldEvent`, with the same handler ordering
(order, priority, speed, sub-order) and the same suppression rules (Mold
Breaker, Klutz, a status that has since changed).

- **Which effect listens to which event, with what priority, is generated**
  from Showdown's data into `src/tables.rs`. This matters beyond convenience:
  a handler that does nothing still changes the RNG stream if it ties with
  another one, so the set of handlers has to be exactly Showdown's.
- **What each callback does is written by hand** in `src/abilities.rs`,
  `src/items.rs` and `src/conditions.rs`, one match arm per Showdown callback,
  with the callback's name and parameters in a comment above it. A callback
  listed in the tables but missing a body panics with its name, and the
  generator refuses any callback for an event the engine does not know.

To add an effect: put its id in the supported list in `oracle/lib.js`, run
`node oracle/gen_data.js`, write the bodies, and fuzz.

## What is and is not modelled

Modelled, and verified as above:

- Champions stat formula, natures, level 50, Champions PP values
- Action order: priority, speed, speed ties, switches before moves, dynamic
  re-sorting after each action, fractional priority (Quick Claw, Stall)
- Targeting in doubles: chosen targets, spread moves and their 0.75 modifier,
  retargeting when a foe has fainted, moves aimed at an ally, redirection by
  Lightning Rod
- Damage: critical hits, damage rolls, STAB, type chart and immunities, burn,
  and every modifier event abilities and items hook into
- Accuracy and evasion stages, stat stages
- Burn, paralysis, poison, toxic, sleep and freeze with the Champions rules
  (1-in-8 full paralysis, sleep for 2 or 3 turns, freeze for at most 3)
- Confusion, flinching, Protect with its consecutive-use counter
- Secondary effects, self stat changes, draining, recoil, healing, multi-hit
  moves, stat-override moves (Body Press, Foul Play, Psyshock)
- Switching, fainting, replacements, PP, Struggle, trapping, disabled moves,
  win and tie conditions, the 1,000-turn limit
- **181 abilities**, including Intimidate and everything that answers it,
  the absorbing and contact abilities, Mold Breaker, Prankster, Magic Bounce,
  Parental Bond, Trace, Protean, Unaware, Sheer Force, Shadow Tag
- **Mega Evolution**: 72 of the 82 Megas (the Mega Stones themselves are all
  accepted as held items)
- **79 held items**: every other item except the six listed below. Items
  Showdown marks as unavailable in Champions (Choice Band, Choice Specs,
  Assault Vest among them) are left out; Choice Scarf is the only Choice item

Not modelled yet:

- Weather, terrain, Trick Room, Tailwind, screens, hazards
- **10 Megas** whose ability needs weather or terrain (Charizard Y,
  Tyranitar, Garchomp, ...); `Battle::new` refuses the Pokémon holding the stone
- Fake Out, Follow Me / Rage Powder, Helping Hand, Wide Guard, Quick Guard
- Pivoting moves (U-turn, Parting Shot), forced switches (Roar)
- Two-turn and recharge moves, Substitute, Encore, Taunt, Disable
- Team preview itself: the engine starts from the four Pokémon picked
- **44 abilities**: 29 that need weather or terrain (Drizzle, Swift Swim,
  Grassy Surge, ...), 10 that change forme or disguise the Pokémon (Stance
  Change, Disguise, Illusion, Imposter, ...), and Cursed Body, Cute Charm,
  Electromorphosis, Toxic Debris and Emergency Exit, which need a mechanic
  from the list above
- **6 items**: Eject Button, Red Card and the four terrain seeds

`oracle/coverage.json` has the full lists with the mechanic each one waits
for. It also lists the *dormant parts* of modelled effects: Mental Herb cures
Taunt and Encore, which do not exist yet; Damp Rock extends rain; Aroma Veil
blocks six conditions of which none is modelled. These effects are exact for
every battle the engine accepts, but each has to be revisited when the
missing mechanic arrives, and the list says which.

For moves, `coverage.json` lists every unmodelled one and the Showdown feature
blocking it. The largest groups are moves with their own lingering condition
(75), moves with a scripted `onHit` (63) or `onTry` (30), and variable base
power (29).

## Layout

```
src/battle.rs      the simulator core: state changes, the queue, the turn loop
src/moves.rs       using a move, from "can it move" to damage and secondaries
src/events.rs      Showdown's event system: handler discovery, ordering, dispatch
src/abilities.rs   ability callbacks        (one arm per Showdown callback)
src/items.rs       item callbacks
src/conditions.rs  status and volatile-condition callbacks
src/choice.rs      Battle::new, legal choices, submitting choices
src/state.rs       fixed-size state: Battle, Side, Pokemon, the action queue
src/data.rs        data definitions; src/tables.rs is generated (do not edit)
src/rng.rs         Showdown's Gen5RNG
src/replay.rs      replays a recorded battle and reports the first difference
src/trace.rs       optional RNG/action trace (feature `trace`)
src/bin/difftest.rs, src/bin/bench.rs
oracle/lib.js        which moves, abilities and items are modelled
oracle/gen_data.js   Showdown data  -> src/tables.rs, pool.json, coverage.json
oracle/gen_cases.js  Showdown battles -> recorded cases (JSON lines)
scripts/             setup-oracle.sh, fuzz.sh, mutation_test.py
tests/parity.rs    fixture of recorded battles, choice-validation checks
tests/effects.rs   a few abilities and items checked directly, as API examples
```

Everything in `src/battle.rs`, `moves.rs` and `events.rs` is a
function-by-function port of Showdown's `sim/` (comments name the function
mirrored). After changing what is supported (`oracle/lib.js`) or updating
Showdown, run `node oracle/gen_data.js` to regenerate the tables.

## Speed

`cargo run --release --bin bench -- cases.jsonl` plays random battles to
completion using the teams in a case file. On one core of a 2.1 GHz cloud
Xeon:

| Teams | Battles/s | Decisions/s |
|---|---|---|
| No abilities or items | about 10,000 | about 180,000 |
| Random abilities and items | about 7,000 | about 130,000 |

The event system roughly halved the speed of the first version, which ran
about 20,000 battles per second with nothing to dispatch. Nothing has been
tuned beyond skipping events that nobody in the battle listens to; caching
each Pokémon's listener set is the likely next step when speed starts to
matter.

## Suggested order for what comes next

1. **Weather and terrain**, which unlock 29 abilities, the terrain seeds and
   the last 10 Megas, then **Trick Room and Tailwind**.
2. **Fake Out, redirection (Follow Me, Rage Powder), Helping Hand, Wide
   Guard**, and the remaining volatile conditions (Substitute, Taunt, Encore,
   Disable).
3. **Pivoting and forced switches**, with Emergency Exit, Eject Button and
   Red Card.
4. The remaining moves and the forme-changing abilities.
5. **Building a `Battle` from an arbitrary mid-battle state**, which a bot
   needs to search from a live game, and sampling hidden information into it.
6. Python bindings and batched stepping for training.

Each step can be driven by the same loop: widen the supported set in
`oracle/lib.js`, regenerate, run `scripts/fuzz.sh`, and fix what diverges.

## Provenance

The tables in `src/tables.rs` and the behaviour of the simulator are derived
from [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) (MIT
licence). Showdown is the reference for correctness here, not the cartridge
games; Smogon's own research into Champions mechanics is still in progress.
