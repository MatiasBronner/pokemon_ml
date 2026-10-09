# vgc-engine

A fast battle engine for Pokémon Champions doubles (VGC), written in Rust and
checked decision by decision against Pokémon Showdown. It is meant as the
simulator underneath a self-play bot: copyable fixed-size state, no allocation
on the common paths of the turn loop, and a harness that proves each mechanic
matches Showdown before anything is trained on it.

**Status: every move, ability and held item a Champions Pokémon can have is
modelled**: all 510 moves in the Champions learnsets, 225 abilities, 85 held
items and 82 Mega Evolutions, each checked against Showdown as described
below. Any team that is legal in the format can be played, from turn 1 or
[from any position in the middle of a battle](#starting-from-the-middle-of-a-battle).

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
    // battle.request says what is being asked: Request::Move at the start of a
    // turn, Request::Switch when someone has to be replaced. The latter can come
    // in the middle of a turn (after U-turn or Eject Button) as well as at its end.
    let c1 = pick(battle.joint_choices(0));   // every legal pair of slot choices for side 0
    let c2 = pick(battle.joint_choices(1));
    battle.choose([c1, c2])?;
}
println!("{:?}", battle.winner);              // Some(0), Some(1) or None for a tie
```

`Battle` is `Copy`: cloning a position for search is a plain memory copy
(about 8 kB). Choices use Showdown's conventions (`Choice::to_showdown`
prints `move 2 1`, `move 1 2 mega`, `switch 3`, `pass`), so they can be sent
to a Showdown server unchanged. A Pokémon holding its Mega Stone is offered
every move a second time with `mega: true`; a side can Mega Evolve once.
`Battle::new` returns `Error::Unsupported`, naming the culprit, for anything
that does not exist in Champions (Tinted Lens, a forme of a
species the game lacks) rather than guessing how it would behave.

A set built with `from_names` has no ability and no item until you add them,
and takes the species' fixed gender or male. The engine does not check that a
species can legally have an ability, move or item; neither does Showdown's
simulator (its team validator does that separately).

## Starting from the middle of a battle

A bot searching from a live game does not start at turn 1. `Battle::from_state`
builds a battle from a description of the position, and `Battle::to_state`
writes one down:

```rust
use vgc_engine::position::{BattleState, CondState, PokemonState, SideState};

let mon = |species: &str, moves: &[&str]| PokemonState::new(species, moves);
let mut incineroar = mon("Incineroar", &["Fake Out", "Flare Blitz", "Parting Shot", "Protect"]).ability("Intimidate");
incineroar.hp_percent = Some(38.0);
incineroar.boosts[0] = -1;
let garchomp = mon("Garchomp", &["Earthquake", "Rock Slide", "Protect"])
    .item("Choice Scarf")
    .volatile(CondState::choice_lock("Rock Slide"));
let mut ours = SideState::new(vec![incineroar, garchomp, mon("Milotic", &["Scald", "Recover"]).status("par")]);
ours.conditions.push(CondState::new("tailwind").turns(1));
let theirs = SideState::new(vec![/* the first two are on the field, the rest on the bench */]);

let state = BattleState {
    turn: 6,
    sides: [ours, theirs],
    weather: Some(CondState::new("raindance").turns(2)),
    ..Default::default()
};
let battle = Battle::from_state(&state)?;       // waiting for turn 6's choices
```

Every field has a default, so a description only says what differs from
"nothing has happened": HP (exact or as a percentage), status and its
counters, stat stages, PP, current item, ability, types and forme, volatile
conditions with their timers (Taunt, Encore, a substitute's HP, a Choice
lock), side conditions, weather, terrain and Trick Room with the turns left,
who has fainted. `BattleState` is plain serde data: `to_json` and `from_json`
carry it across a process boundary, which is how a Python bot will hand over
a position. `cargo run --release --example position` builds one and does a
one-ply search from it; `src/position.rs` documents every field.

Three kinds of position can be described:

- **The start of a turn** (`request: Move`, the default).
- **Replacing the fainted at the end of a turn** (`request: Switch`): mark
  them `fainted`.
- **A switch in the middle of a turn** (after U-turn, Eject Button,
  Emergency Exit): set `switch_flag` on the Pokémon leaving and list the
  moves still to come in `pending`.

What to know before relying on it:

- **Exported positions are exact.** `from_state(to_state(b))` continues as `b`
  would, random numbers included. An exported position carries the
  simulator's bookkeeping (which moves the coming request disables, cached
  speeds, the order effects started in) and says so with `prepared: true`.
- **Hand-written positions leave the bookkeeping out**, and `from_state`
  works it out: disabled moves, trapping and locked moves come out as the
  engine would have had them. Two things cannot be known from outside and get
  a neutral default: the order in which effects started, with a flag Showdown
  keeps per field condition (both only settle ties between simultaneous
  effects), and the state of the random number generator (give a `seed`).
- **After editing an exported position**, set `prepared = false` (or call
  `without_bookkeeping()`), or the stale bookkeeping is believed.
- **Hidden information is yours to fill in.** A position has no "unknown":
  every Pokémon needs a species, moves and stats. Sampling the opponent's
  unrevealed moves, items, spreads and bench from a prior is not here yet.
- A Pokémon in the middle of a battle is taken to have been on the field for
  a turn and to have moved (so Fake Out is not offered); say `move_actions:
  Some(0)` and `active_turns: Some(0)` for one that has just come in.

How this is checked is under [How it is checked](#how-it-is-checked).

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
| Everything modelled now | 51,000 | 1,210,052 | 0 |
| Recorded before two-turn moves, pivoting, forced switches and forme changes existed | 16,900 | 364,631 | 0 |
| Recorded before Fake Out, Substitute and the volatile conditions existed | 19,300 | 385,071 | 0 |
| Recorded before weather and the other field effects existed | 5,500 | 101,656 | 0 |
| Recorded before Mega Evolution | 21,000 | 393,381 | 0 |
| Items but no abilities | 5,000 | 93,086 | 0 |
| Neither (how the engine's first version was checked) | 15,200 | 270,196 | 0 |
| Comparing every individual RNG draw as well | 5,750 | 133,075 | 0 |
| 1,000-turn limit (both sides switch whenever they can) | 32 | 30,679 | 0 |

Each row compares state, RNG seed and legal choices after every decision.
State includes every Pokémon's species, stats, moves and PP (which change
under Transform), its volatile conditions with what they remember (a
substitute's HP, the move an Encore holds it to), the per-turn bookkeeping
Showdown keeps (who hit it last and for how much, whether it has moved), the
weather, terrain, pseudo-weathers and each side's conditions with their
remaining turns. The
older rows are battles recorded at earlier stages and replayed with the
current code.

How the battles are made up:

- Teams are drawn from all 293 species that can be brought. Each battle puts
  one move, one ability and one item on a lead, cycling through all of them,
  so every one gets its share.
- Abilities are the species' own half the time and any ability at all
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
- Some effects only matter in combinations that random teams almost never
  produce: Aurora Veil on a side that also has Reflect up, or a Mega Stone
  being stolen. `gen_cases.js --moves`, `--species`, `--abilities` and
  `--items` build batches around such a combination; about 30,000 of the
  battles in the first three rows are of this kind.

Two further checks:

- **Legal choices.** The recorder lists legal choices from the Pokémon's real
  state rather than from the request Showdown sends the player, because the
  request deliberately hides some things (a Shadow Tag trap not yet revealed).
  `gen_cases.js --check-legal` confirms that list against Showdown's own
  validation by submitting every conceivable choice: 3,000 battles, 68,337
  decisions, no disagreement. (One oddity this turned up: when a foe's
  Imprison has sealed every move of a side's last active Pokémon, Showdown
  still lists the moves, and wants the forced Struggle spelled as a use of
  the first one, target included. `legal_choices` spells it that way.)
- **Positions.** `difftest --by-hand` replays the recorded battles and at
  every decision writes the position down (through JSON), builds a battle
  back from it and requires the two to be the same data, field for field,
  apart from scratch space. It then carries on with the rebuilt battle, which
  must keep matching Showdown to the end, so every one of those decisions was
  also the start of a battle begun in the middle. That holds for all of
  them: about 144,000 battles, 3.06 million decisions. At each
  decision it also strips the bookkeeping from the position, rebuilds from
  that, and requires what `from_state` works out (disabled moves, trapping,
  locked moves, who is to be replaced) to be what the engine had. Seven
  faults injected into the export and rebuild code (a counter not restored, a
  source forgotten, the event masks left empty) were all noticed. For
  positions nobody would write on purpose, `tests/position.rs` throws random
  conditions onto recorded positions; of 60,000, `from_state` refused about
  one in six (a Choice lock that names no move) and every one it accepted
  could be played on without the engine tripping.
- **Does the comparison have teeth?** `scripts/mutation_test.py` injects one
  small bug at a time (Life Orb's multiplier off by 1/4096, Intimidate
  lowering by two stages, Mold Breaker ignored, Sitrus Berry restoring a third) and
  replays recorded battles. A second mode switches off one callback at a time
  (one ability's reaction to one event, one move's script). All 413
  hand-written bugs and all 602 switched-off callbacks are caught. Another 15
  callbacks, and eight hand-written bugs that were tried, change nothing
  that can be observed in Champions (Ripen doubling the stat changes of
  berries, where no berry in the game changes stats); the script lists each
  with its reason. Random battles are not enough for this: with 3,000
  general battles, 86 of the 1,015 bugs were only caught by a batch built
  around the effect, such as Sleep Talk on a Pokémon that also knows Rest
  and Meteor Beam, or Dragon Darts into a Protect beside a Berserk Pokémon
  at just over half HP. `scripts/targeted.sh` records all such batches.

What this does **not** establish: agreement with the cartridge games where
they differ from Showdown. And coverage of rare interactions is uneven. Every
move, ability and item was brought into hundreds of battles, but an effect
that singles out one particular move or ability only shows when the two meet.
Five such cases were wrong at some point while this was being written and got
past thousands of random battles: Reckless boosting High Jump Kick, Sheer
Force boosting Electro Shot, Cud Chew ignoring a berry eaten with Bug Bite,
Magician not stealing after Fling, and a frozen Pokémon that has lost its
Fire type getting no thaw from Burn Up. They were found by reading Showdown's
source for every place that names a newly modelled move, and by batches
built around the pairing. Both are now routine for anything added, and the
mutation tests above say which pairings the recorded battles cover; but it is
the kind of error most likely to remain.

### Running it yourself

```sh
scripts/setup-oracle.sh            # clone and build the pinned Showdown commit (needs Node 22+)
scripts/fuzz.sh 2000               # 2,000 fresh battles, random seed
TRACE=1 scripts/fuzz.sh 300 42     # also compare every RNG draw
REBUILD=1 scripts/fuzz.sh 2000     # also rebuild the battle from its position at every decision
```

The mutation test needs recorded battles to replay: the batches built around
particular effects, and a few thousand general ones.

```sh
scripts/targeted.sh corpus                                              # 55 batches, about 17,000 battles
node oracle/gen_cases.js --n 3000 --seed 9500 --out corpus/general.jsonl
scripts/mutation_test.py corpus/*.jsonl --handlers                      # slow: a rebuild per injected bug
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

## How abilities, items and conditions work

The engine has a port of Showdown's event system (`src/events.rs`): `runEvent`,
`singleEvent`, `eachEvent` and `fieldEvent`, with the same handler ordering
(order, priority, speed, sub-order) and the same suppression rules (Mold
Breaker, Klutz, Cloud Nine, a status that has since changed). Effects can sit
on a Pokémon, on one of a side's positions, on a side, or on the whole field.

- **Which effect listens to which event, with what priority, is generated**
  from Showdown's data into `src/tables.rs`. This matters beyond convenience:
  a handler that does nothing still changes the RNG stream if it ties with
  another one, so the set of handlers has to be exactly Showdown's.
- **What each callback does is written by hand** in `src/abilities.rs`,
  `src/items.rs`, `src/conditions.rs` and `src/movecbs.rs`, one match arm per
  Showdown callback,
  with the callback's name and parameters in a comment above it. A callback
  listed in the tables but missing a body panics with its name, and the
  generator refuses any callback for an event the engine does not know.

When Showdown's Champions data gains an effect: run `node oracle/gen_data.js`.
It marks as unsupported any move that uses a property or event the engine has
not seen (and says which); for the rest, write the bodies of the callbacks
that now panic, and fuzz.

## What is and is not modelled

Modelled, and verified as above: everything that can happen in a battle
between two legal Champions teams.

- Champions stat formula, natures, level 50, Champions PP values
- Action order: priority, speed, speed ties, switches before moves, dynamic
  re-sorting after each action, fractional priority (Quick Claw, Stall),
  moves that reorder the queue (After You, Quash, Round, Instruct)
- Targeting in doubles: chosen targets, spread moves and their 0.75 modifier,
  retargeting when a foe has fainted, moves aimed at an ally, redirection
  (Follow Me, Rage Powder, Lightning Rod), Dragon Darts, Ally Switch
- Damage: critical hits, damage rolls, STAB, type chart and immunities, burn,
  and every modifier event abilities and items hook into
- Burn, paralysis, poison, toxic, sleep and freeze with the Champions rules
  (1-in-8 full paralysis, sleep for 2 or 3 turns, freeze for at most 3)
- **All 510 moves** in the Champions learnsets. By kind:
  - Protect and its relatives, Fake Out, Helping Hand, Wide Guard, Quick
    Guard, Feint, Endure
  - Substitute, Taunt, Encore, Disable, Torment, Imprison, Yawn, Leech Seed,
    Perish Song, Destiny Bond, Curse, the binding and trapping moves and the
    other lingering conditions
  - Two-turn moves (Fly, Solar Beam, Electro Shot), recharge moves, rampages
    (Outrage), Focus Punch, Counter, Mirror Coat, Metal Burst
  - **Pivoting** (U-turn, Volt Switch, Parting Shot, Chilly Reception, Baton
    Pass, Shed Tail) and **forced switches** (Roar, Whirlwind, Dragon Tail)
  - Moves whose power depends on the battle (Gyro Ball, Eruption, Acrobatics,
    Stored Power, Last Respects, ...)
  - Moves that tamper with items, abilities and types (Knock Off, Trick,
    Fling, Skill Swap, Entrainment, Gastro Acid, Soak, ...)
  - Moves that call other moves (Copycat, Sleep Talk), Future Sight, Healing
    Wish, Revival Blessing, Transform
- **Weather and terrain**, Trick Room, Gravity, Magic Room, Wonder Room,
  Tailwind, the screens, Safeguard, the entry hazards and the moves that
  clear them, Wish
- **All 225 abilities**, including the ones that switch a Pokémon out
  mid-turn (Emergency Exit), change its forme (Stance Change, Disguise, Zero
  to Hero, Hunger Switch), or disguise it (Illusion, Imposter)
- **Mega Evolution**: all 82 Megas
- **All 85 held items**, Eject Button and Red Card included. Items Showdown
  marks as unavailable in Champions (Choice Band, Choice Specs, Assault Vest
  among them) are left out; Choice Scarf is the only Choice item
- Switching, fainting, replacements in the middle of a turn and at its end,
  PP, Struggle, trapping, disabled moves, win and tie conditions, the
  1,000-turn limit

Not modelled:

- **Hidden information.** A position
  ([above](#starting-from-the-middle-of-a-battle)) has to say everything
  about both sides. Sampling what a player has not been shown is the next
  piece of work.
- Team preview: the engine starts from the four Pokémon each side picked, in
  the order picked.
- Anything Champions does not have: Terastallization, Z-moves, Dynamax, and
  the species, moves, abilities and items outside the format. A few abilities
  are legal to put on a Pokémon but belong to species or mechanics the game
  lacks (Ice Face, Gulp Missile, Shields Down, Battle Bond, Tera Shell, the
  four Embody Aspects); they do nothing, exactly as in Showdown.
  `oracle/coverage.json` lists these under `dormant_parts`.
- The battle log. The engine tracks state, not messages, so abilities that
  only announce something (Frisk, Anticipation) have no visible effect;
  Forewarn still makes its random draw.

### Showdown behaviour worth knowing about

The engine reproduces Showdown, not the cartridge. A few places where
Showdown's behaviour is surprising, all confirmed against its source and
reproduced here:

- A move that is vetoed while being set up (a sound move chosen before Throat
  Chop landed, a flying move under Gravity) still draws one random target
  before it fails.
- Metal Burst and Comeuppance return 1.5 times the damage taken without
  rounding, so a substitute can be left with half a hit point. The engine
  stores substitute HP in halves for this reason.
- Curse chosen by a Pokémon that is not a Ghost targets the user from the
  moment it is chosen, even if the Pokémon becomes a Ghost before moving.
- A forced switch (Roar) draws a random number to pick the replacement even
  when only one Pokémon can come in.
- With Revival Blessing and a forced replacement pending on the same side,
  Showdown counts available switches in slot order, so "revive with the left
  slot, replace the right" can be rejected where the mirror image is
  accepted. `legal_choices` applies the same rule.
- A Pokémon revived into its own active slot comes back through a switch
  queued at the end of the turn's remaining actions.

## Layout

```
src/battle.rs      the simulator core: state changes, the queue, the turn loop
src/moves.rs       using a move, from "can it move" to damage and secondaries
src/events.rs      Showdown's event system: handler discovery, ordering, dispatch
src/abilities.rs   ability callbacks        (one arm per Showdown callback)
src/items.rs       item callbacks
src/conditions.rs  callbacks of statuses, volatiles, weather, terrain and side conditions
src/movecbs.rs     script callbacks of moves (onTry, onHit, basePowerCallback, ...)
src/choice.rs      Battle::new, legal choices, submitting choices
src/position.rs    a position as data: Battle::to_state, Battle::from_state, JSON
src/state.rs       fixed-size state: Battle, Side, Pokemon, the action queue
src/data.rs        data definitions; src/tables.rs is generated (do not edit)
src/rng.rs         Showdown's Gen5RNG
src/replay.rs      replays a recorded battle and reports the first difference
src/trace.rs       optional RNG/action trace (feature `trace`)
src/bin/difftest.rs, src/bin/bench.rs
oracle/lib.js        what counts as modelled (the move properties and events the engine knows)
oracle/gen_data.js   Showdown data  -> src/tables.rs, pool.json, coverage.json
oracle/gen_cases.js  Showdown battles -> recorded cases (JSON lines)
scripts/             setup-oracle.sh, fuzz.sh, targeted.sh, mutation_test.py
tests/parity.rs    fixture of recorded battles, choice-validation checks
tests/effects.rs   a few abilities and items checked directly, as API examples
tests/position.rs  positions written by hand: defaults, timers, switches in the middle of a turn
```

Everything in `src/battle.rs`, `moves.rs` and `events.rs` is a
function-by-function port of Showdown's `sim/` (comments name the function
mirrored). After updating Showdown, run `node oracle/gen_data.js` to
regenerate the tables.

## Speed

`cargo run --release --bin bench -- cases.jsonl` plays random battles to
completion using the teams in a case file. On one core of a 2.1 GHz cloud
Xeon:

| Teams | Battles/s | Decisions/s |
|---|---|---|
| No abilities or items | about 7,100 | about 127,000 |
| Random abilities and items | about 4,300 | about 82,000 |
| The same with weather, terrain and the other field effects in play | about 4,200 | about 84,000 |
| The same with Substitute, Encore and the other volatile conditions in play | about 3,600 | about 79,000 |
| The same with every move in play (two-turn moves, pivoting, forme changes) | about 3,500 | about 80,000 |

Nothing has been tuned beyond skipping events that nobody in the battle
listens to, and it shows: the last batch of mechanics cost 15 to 30% on the
same teams (the first row was about 10,000 before it), from a larger state
and more bookkeeping per action rather than from any one hot spot, and the
event system had already halved the speed of the first version, which ran
about 20,000 battles per second with nothing to dispatch. Caching each
Pokémon's listener set and slimming the per-move scratch state are the
likely first steps when speed starts to matter.

## What comes next

1. **Sampling hidden information** into a position: the opponent's
   unrevealed moves, items, abilities, spreads and bench, drawn from a prior
   (usage statistics, or the bot's own model) and consistent with what has
   been seen.
2. **Python bindings and batched stepping** for training.
3. **Speed.** The per-Pokémon listener cache described above, then
   profiling.
4. Keeping up with Showdown: `scripts/setup-oracle.sh` pins a commit; after
   moving the pin, `node oracle/gen_data.js` regenerates the tables, a
   missing callback body panics with its name, and `scripts/fuzz.sh` finds
   behaviour changes.
