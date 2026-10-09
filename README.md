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
[from any position in the middle of a battle](#starting-from-the-middle-of-a-battle),
and the engine keeps track of
[what each player has been shown](#what-each-side-has-been-shown) of the
other's team.

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
and takes the species' fixed gender or male. `Battle::new` does not ask
whether a species can legally have an ability, move or item; neither does
Showdown's simulator. That is a separate question, answered per regulation:
see [Legal teams and regulations](#legal-teams-and-regulations).

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

## Legal teams and regulations

What a regulation allows is data, one file per regulation in `formats/`,
and `format::Format` checks teams against it:

```rust
use vgc_engine::format::Format;

let format = Format::current();                     // Reg M-C, the one the engine is built for
for problem in format.check_team(&team) {           // all six, as registered
    println!("{problem}");                          // "Garchomp cannot learn Moonblast"
}
let rule = format.rule(garchomp.species).unwrap();  // its legal moves, abilities, genders, Mega Stones
let random = format.random_team(&mut rng);          // six legal Pokémon, drawn uniformly
```

The file lists every species that can be brought with its legal moves,
abilities and genders, the items anyone may hold, and the team rules (for
Reg M-C: bring six and pick four, no two Pokémon with the same Pokédex number,
no item held twice, at most 32 stat points in a stat and 66 in all).

**Nothing in the file is written by hand.** `oracle/gen_format.js` asks
Showdown's own team validator, one question at a time: is this species
accepted, and with this move, this ability, this item? So the file is what
Showdown accepts, by construction, and a new regulation needs no rules
transcribed.

**Moving to another regulation:**

```sh
node oracle/gen_format.js gen9championsvgc2026regmb     # any format id Showdown knows
scripts/check-teams.sh 5000 1 gen9championsvgc2026regmb # verdicts against Showdown's, both ways
```

```rust
let reg_mb = Format::from_json(&std::fs::read_to_string("formats/gen9championsvgc2026regmb.json")?)?;
```

The file also records whether the regulation changes anything the simulator
computes with, as opposed to what may be brought. Reg M-B, the previous
regulation, is in the repository as the example: 29 fewer species, 18 fewer
items, and two moves with different PP (Wish and Strength Sap), which the file
lists under `differences` and `Format::simulated_exactly` reports. Teams can
be checked against such a regulation, but battles under it would be off by
those differences until the engine's tables are regenerated from that data.

For a regulation newer than the pinned Showdown: move the pin
(`scripts/setup-oracle.sh`), regenerate the tables (`node oracle/gen_data.js`;
a new move or ability stops the build by name until its callback is written,
and `scripts/fuzz.sh` checks it), set the format id in `oracle/lib.js` and the
file name in `Format::current`, and generate its file. `Format::from_json`
refuses a file that names anything the engine's tables lack.

What the validator does not do, where Showdown's does:

- **It reports, it does not repair.** Showdown accepts a Mega written as the
  species (and turns it into the ordinary forme holding its stone) and
  corrects an impossible gender. Here both are violations.
- **Levels.** Everything is level 50. Showdown also refuses a Pokémon whose
  stated level is too low for it to have evolved; there is no level to state
  here.
- **Nicknames**, and Showdown's reminder that a Pokémon with no stat points at
  all and a Serious nature looks unfinished. (Teams written out for Showdown
  use Hardy for a neutral nature, for that reason.)
- **Picking four of the six.** `Battle::new` takes the Pokémon picked; team
  preview is not modelled.

## Teams from tournaments

Training needs teams people actually play. `scripts/scrape_teams.py` collects
the team sheets of a tournament from its Victory Road page, and `teampool`
turns them into a pool of teams the engine can play:

```sh
python3 scripts/scrape_teams.py https://victoryroad.pro/2027-frankfurt/     # -> teams/raw/2027-frankfurt.json
cargo run --release --bin teampool -- teams/raw/2027-frankfurt.json --play 1000   # -> teams/2027-frankfurt.json
```

The scraper reads the results tables, follows each row's team-sheet link
(to vrpastes.com) and saves where the team placed, who played it and the text
of its sheet. It fetches one page a second and keeps what it has fetched, so
it can be stopped and run again. `teampool` reads each sheet
(`teams::read_sheet`), checks the team against the regulation, writes the
legal ones out and says which it left out and why; `--section` and `--top`
narrow it to a division or to the best-placed, and `--play` plays battles
between teams of the pool as a check that every one runs.

```rust
use vgc_engine::teams::Pool;

let pool = Pool::from_json(&std::fs::read_to_string("teams/2027-frankfurt.json")?)?;
let (ours, theirs) = (pool.teams[0].sets()?, pool.teams[7].sets()?);
let battle = Battle::with_rosters([&ours, &theirs], [&[0, 1, 2, 3], &[0, 2, 4, 5]], true, seed)?;
```

Two things to know about what comes out:

- **The stat points are guesses.** An open team sheet gives species, item,
  ability, moves and nature, and never how the 66 stat points are spent.
  `teams::guess_spread` spends them by rule of thumb from the nature and the
  moves (32 in each of two stats, 2 in a third: attack and Speed for a Jolly
  Garchomp, HP and Special Defense for a Careful Incineroar). Every team it
  did this to is marked `spreads_guessed`. Real spreads are finer, and
  finding better ones is a job for later; a paste that does give stat points
  is taken at its word.
- **The scraper has not been run against the live sites from where it was
  written**, which had no route to them. It is written not to depend on how
  the paste site lays out a sheet: it gathers every short line of text the
  page carries, in its markup and in its scripts, and `read_sheet` picks the
  team out by the names it knows. `scripts/check_scraper.py` checks that on a
  made-up tournament whose sheets are laid out five different ways. If the
  site delivers its sheets only after the page has loaded, the scraper will
  report teams left out for having no sheet, and will need to be taught where
  they come from; the pages it fetched are kept in `teams/raw/cache` to look at.

## What each side has been shown

The engine knows every move, item and ability on both sides. A player does
not, and neither may a policy that is going to play real games.
`Battle::shown` reports what the battle has made public about one side:

```rust
let theirs = battle.shown(1);                // side 1 as everyone watching knows it
for mon in theirs.active.iter().flatten() {  // on the field; `bench` has those seen and withdrawn
    mon.species;    // "garchomp", and "garchompmega" once it has Mega Evolved
    mon.hp;         // 73: a whole percentage, the way Showdown shows a foe's HP
    mon.status;     // "par"
    mon.moves;      // ["earthquake", "protect"]: the ones it has used
    mon.item;       // Some("lifeorb") once the orb has hurt it; None until an item is shown
    mon.ability;    // Some("roughskin") once that has gone off
}
theirs.unseen;      // how many of the Pokémon they brought have not appeared yet
```

What a player knows is then their own side in full and `shown` of the other;
`shown` of their own side is what the opponent knows about them.

"Shown" means what Pokémon Showdown's battle log says, line for line:

- **Moves**: used, or named some other way: stopped by Taunt or Disable
  (`cant`), a Focus Punch tightening its focus, read by Forewarn, drained by
  Spite, topped up by a Leppa Berry. A move borrowed with Copycat is not the
  Pokémon's own and is not counted; one picked by Sleep Talk is. Nor are the
  moves of a Pokémon that has transformed.
- **Items**: named when they act (Life Orb, Leftovers, Rocky Helmet, an Air
  Balloon on entry), when they are used up or eaten, and when they are
  knocked off, stolen, swapped or found by Frisk. An item that has gone is
  remembered as lost. A Mega Evolution names its stone.
- **Abilities**: announced on entry (Intimidate, Drizzle) or named when they
  do something. Replacements are followed (Skill Swap, Mummy, Trace, Worry
  Seed, Transform), along with the ability the Pokémon returns to when it
  leaves the field. A Mega's ability is known from the Mega.
- **HP and status** as shown to the opponent; for a Pokémon on the bench, as
  they were when it left.
- **Illusion**: what a disguised Pokémon shows is credited to the Pokémon it
  passes for, and moves over to the real one if the disguise breaks. Until
  then the record says that it cannot be trusted (see below).

Nothing is worked out. Less damage than expected does not reveal an Assault
Vest, moving first does not reveal a Choice Scarf, and a Frisk that finds
nothing does not reveal an empty hand; that kind of inference is left to
whoever uses the record. Other limits: PP is not counted, and a disguise that
is never broken is never corrected, only marked as doubtful. Stat stages, volatile conditions, weather
and side conditions are public and are in the `Battle` itself, not repeated
here, though their timers are not public (the turns a Reflect has left give
away a Light Clay); packing all of it into one view for a policy is part of
designing the observation, which is the next piece of work.

What has been shown is part of a position: `to_state` writes it down for
every Pokémon (`PokemonState::shown`), with the teams as registered
(`SideState::roster`) and whether the sheets are open, and `from_state`
takes it back, so a search that starts in the middle of a battle keeps it. A
position written by hand that says nothing about it starts with nothing
shown but who is on the field, and takes each side to have registered the
Pokémon it lists.

`observer::Observer` is the same bookkeeping done from the other end: it
reads Showdown's log, as a program playing on Showdown receives it, and
returns the same `ShownSide`. The engine is checked against it (see
[How it is checked](#how-it-is-checked)), and it is the reader a Showdown
client will need.

### Team Preview and open team sheets

Some things are known before the first turn. Team Preview shows each player
the species the other registered, and Showdown's Champions formats add open
team sheets: both players are asked in a best-of-one (the sheets open if
both agree), and in a best-of-three they are open regardless. A sheet gives
every registered Pokémon's item, ability, moves and nature. It never gives
stat points, and it does not say which four of the six are brought.

`Battle::new` takes the four Pokémon each side brought and assumes nothing
more. `Battle::with_rosters` takes the teams as registered, which of them
each side brings, and whether the sheets are open:

```rust
let battle = Battle::with_rosters([&ours, &theirs], [&[0, 2, 3, 5], &[1, 2, 4, 5]], true, seed)?;
let theirs = battle.shown(1);
theirs.roster;                 // their six, in the order registered
theirs.roster[2].species;      // Team Preview shows this much, sheets open or not
theirs.roster[2].sheet;        // open sheets: Some(item, ability, moves, nature)
let lead = theirs.active[0].as_ref().unwrap();
lead.listed;                   // Some(2): the entry of the roster it appears to be
theirs.sheet_of(lead);         // that entry's sheet
lead.item;                     // still None: it has not shown its item, whatever the sheet says
```

The battle plays the same either way; only what is known from the start
differs. The sheet and the record are kept apart on purpose. The sheet says
what a Pokémon came with, the record what the battle has shown of it since
(the Life Orb on the sheet has been knocked off; the ability has been
swapped), and an observation can use both.

### Illusion

A Zoroark comes in looking like a team-mate, so on a team that has one, a
Pokémon that appears need not be what it appears to be, and neither its
sheet nor what was noted under its name can be taken at face value.

- `ShownSide::illusion` lists the registered Pokémon that have Illusion.
  With open sheets that is what the sheets say. With closed sheets it is the
  Pokémon whose species can have Illusion under the regulation, which Team
  Preview shows; this assumes the team is a legal one.
- `ShownMon::maybe_disguise` is true for every Pokémon that may be the
  Illusion Pokémon in disguise, or under whose name the Illusion Pokémon may
  have acted earlier. It is false only when that is ruled out: the team has
  no Illusion Pokémon, or that Pokémon has fainted, or it stood on the field
  as itself beside the one in question.
- When a disguise breaks, the Pokémon is who it is from then on, and what it
  showed in disguise moves to its own record.

The rule never calls a disguised Pokémon genuine (this is checked against
the truth: see below), and it stops there. It does not reason from the
sheets, as a player would on seeing "Garchomp" use a move only Zoroark's
sheet lists. Both the sheet and what was shown are in the record for a model
to draw that conclusion itself, like every other inference.

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
| Fresh battles, after the last fix (Healing Wish, below) | 20,000 | 447,681 | 0 |
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

Further checks:

- **Legal choices.** The recorder lists legal choices from the Pokémon's real
  state rather than from the request Showdown sends the player, because the
  request deliberately hides some things (a Shadow Tag trap not yet revealed).
  `gen_cases.js --check-legal` confirms that list against Showdown's own
  validation by submitting every conceivable choice: 3,000 battles, 68,337
  decisions, no disagreement. (One oddity this turned up: when a foe's
  Imprison has sealed every move of a side's last active Pokémon, Showdown
  still lists the moves, and wants the forced Struggle spelled as a use of
  the first one, target included. `legal_choices` spells it that way.)
- **Legality.** `scripts/check-teams.sh` compares the team validator with
  Showdown's in both directions. Showdown judges random teams, most of them
  legal or one step from it (a move the Pokémon cannot learn, an ability from
  another species, an item held twice, a 33rd stat point, a second forme of
  the same Pokémon, five Pokémon); the engine gave the same verdict on all
  80,000 (40,000 for each of the two regulations, about 43% of them legal).
  And the engine makes random legal teams, which Showdown must accept: all
  80,000 were. The teams are built from Showdown's data
  directly, not from the regulation file, so the file is not checked against
  itself. Fourteen faults injected into the validator (a rule skipped, a
  limit off by one) were all noticed.
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
- **What has been shown.** Here the engine is checked against a second
  program rather than against Showdown's state, because Showdown keeps no
  record of what a player knows; it only sends the log. `observer` reads
  that log and keeps its own account, and `difftest --shown` requires the
  engine's account to equal it, for both sides, at every decision of battles
  recorded with their logs, their teams as registered and open team sheets
  (`gen_cases.js --log --open-sheets`): 24,250 battles, 636,029 decisions,
  no difference. Every battle is checked twice, once as recorded and once as
  if the sheets had stayed closed (`--closed-sheets`, which keeps them from
  both the engine and the reader). The two are written from opposite ends,
  the engine at the place each mechanic happens and the reader from the text
  alone, so a rule missing from either shows up. Four things keep the pair
  from being wrong together:
  - *Who a line is about.* A line naming an ability or item mentions one or
    two Pokémon and Showdown has no single rule for which of them has it.
    `gen_cases.js --holders` asked Showdown, on every such line of about
    20,000 battles, which Pokémon really had the thing; the reader's table
    was drawn up from the answers.
  - *Truth.* Everything the reader believes about a Pokémon's moves, item
    and ability is compared with the real Pokémon: in battles without
    Illusion for every record, and in battles with it for the records not
    marked as possibly a disguise's. Nothing it believed was false.
  - *Who is who.* The sheets the engine gives out are compared with the ones
    Showdown sends, and the entry a Pokémon is said to appear as with the
    Pokémon it is. On teams that brought an Illusion Pokémon, a Pokémon
    stood on the field in disguise at 7,218 decisions and was marked as a
    possible disguise at every one of them. Pokémon that were themselves
    were known to be at 21,951 and left in doubt at 28,226. (With closed
    sheets, on teams where only Pokémon that may have Illusion had it:
    2,341 in disguise, all marked; 7,280 known; 10,012 in doubt.) Most of
    those teams are ones the recorder builds to be hard, with several
    Illusion Pokémon or five Zoroark, where everything stays in doubt; on
    ordinary teams with one Zoroark about half are known.
  - *Coverage.* The engine records something in 146 places. Every one was
    reached in those battles, and 140 were at some point the first to reveal
    something, so that taking any of them out would have shown as a
    difference; the other six can only repeat what an earlier line said
    (the move a Leppa Berry tops up has been used already). Every move, item and
    ability in the game was shown somewhere, except four abilities: Battle
    Bond, Ice Face and Shields Down, which belong to formes the game lacks,
    and Stance Change, which no line names.

  `difftest --shown --by-hand` also carries the record, the rosters and the
  sheets through an exported position at every decision of the same battles,
  with no difference. Fresh
  battles were the real test, and are the reason not to read "no difference"
  as "finished". The first 50,000 played after the rules had settled turned
  up four gaps in them, each of which now has a batch of its own in
  `scripts/targeted.sh`: Big Pecks and Clear Body blocking Octolock's drops
  in silence; Symbiosis handing over an item between the two lines a
  damage-halving berry gets; Poison Touch poisoning after Wandering Spirit
  had already replaced it; and a Fling whose line names no item, because the
  Lum Berry being thrown was eaten on the way. The 40,000 after that turned
  up none, and the 20,000 after those one: an Uproar borrowed with Copycat
  and then disabled, whose `cant` line made it look like the Pokémon's own.
  That one, and a like case with Choice items that had been known and left
  alone, are now read correctly (a move just borrowed is not taken for the
  Pokémon's own by the next line that names it). The 20,000 after that,
  checked with the sheets open and closed, turned up nothing.
  Eight faults injected into the rules for team sheets and Illusion were
  all noticed, six by the recorded battles and two by `tests/shown.rs`:
  where the engine and the reader share a function, comparing them cannot
  catch a fault in it, and only the check against the truth or a test can.
- **A line of play, written out.** Random battles say that the mechanics
  agree, not that a particular plan works. `gen_cases.js --script` plays
  battles from a file of teams and choices in Showdown and records them like
  any other, so the engine can be held to a line of play.
  `scripts/perish_trap.json` has four, around the perish trap: Perish Song
  beside a Mega Gengar whose Shadow Tag keeps the other side in for three
  turns, with the singers leaving on the third and fourth; the ways out (a
  Ghost, a Shed Shell, U-turn, Soundproof); Mean Look letting go when its
  user leaves; and two against two with nobody able to leave, where all four
  faint at once and Showdown gives the battle to the side whose Pokémon went
  last. The engine matches each at every decision, legal choices included
  (confirmed against Showdown's own validation with `--check-legal`), and
  `tests/perish_trap.rs` walks through them. A batch of 600 random battles
  on the same theme agrees too (17,625 decisions): in it 739 Pokémon fainted
  to the count, 242 of them held in with somewhere to switch to, 78 of
  those on a turn when a foe left the field, and 57 battles ended with both
  sides' counts running out together.
- **Does the comparison have teeth?** `scripts/mutation_test.py` injects one
  small bug at a time (Life Orb's multiplier off by 1/4096, Intimidate
  lowering by two stages, Mold Breaker ignored, Sitrus Berry restoring a third) and
  replays recorded battles. A second mode switches off one callback at a time
  (one ability's reaction to one event, one move's script). All 415
  hand-written bugs and all 602 switched-off callbacks are caught. Another 15
  callbacks, and eight hand-written bugs that were tried, change nothing
  that can be observed in Champions (Ripen doubling the stat changes of
  berries, where no berry in the game changes stats); the script lists each
  with its reason. Random battles are not enough for this: with 3,000
  general battles, 88 of the 1,017 bugs were only caught by a batch built
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
the kind of error most likely to remain. A sixth turned up later, in 70,000
fresh battles played to check what each side has been shown: a Healing Wish
left waiting on a position stops working for Pokémon that switch in once
Ally Switch has moved a healthy one onto it. That one is a slip of
Showdown's (it is in the list further down) and took two battles in those
70,000 to show.

### Running it yourself

```sh
scripts/setup-oracle.sh            # clone and build the pinned Showdown commit (needs Node 22+)
scripts/fuzz.sh 2000               # 2,000 fresh battles, random seed
TRACE=1 scripts/fuzz.sh 300 42     # also compare every RNG draw
REBUILD=1 scripts/fuzz.sh 2000     # also rebuild the battle from its position at every decision
SHOWN=1 scripts/fuzz.sh 2000       # also check what each side has been shown, with open and with closed sheets
node oracle/gen_cases.js --script scripts/perish_trap.json --check-legal --out trap.jsonl   # battles written out
```

The mutation test needs recorded battles to replay: the batches built around
particular effects, and a few thousand general ones.

```sh
scripts/targeted.sh corpus                                              # 67 batches, about 21,000 battles
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

`SHOWN=1 scripts/targeted.sh corpus` records the same batches with their
logs and open team sheets (about 4 GB) and runs `difftest --shown` on each,
with the sheets open and as if they had stayed closed. When that reports a
difference it names the field, the two values and the first battle and
decision where it happened.

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
- The battle log as text. The engine tracks state, not messages. What the
  messages reveal is tracked (see
  [What each side has been shown](#what-each-side-has-been-shown)): Frisk
  shows the items it finds and Forewarn the move it picks, and Anticipation
  shows itself.

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
- Healing Wish waits on its position until a Pokémon it can help arrives.
  If Ally Switch moves a healthy Pokémon onto that position first, the wish
  stops working for Pokémon that switch in there, and only heals one that
  another Ally Switch brings. (Running the wish's handler for the swap
  leaves a Pokémon recorded as the wish's target, and the code that handles
  switch-ins then takes the wish for a condition of that Pokémon which it no
  longer has.)

And in what the log gives away:

- In Champions, Regenerator and Natural Cure are written into the log when
  a Pokémon is withdrawn, as lines the client does not display but every
  program reading the log receives. Both abilities count as shown the first
  time they act; a human opponent would not have noticed.
- Stance Change and Hunger Switch change the Pokémon's forme without the
  line naming the ability, unlike in Showdown's other formats.
- Big Pecks, Clear Body and White Smoke say nothing when the drop they block
  is Octolock's.
- A move borrowed with Copycat can carry on over the following turns
  (Uproar, Fly), and if something then stops it, the line that says so reads
  as if the move were the Pokémon's own: `|cant|p1a: A|Disable|Uproar`, or,
  when a Choice item is what stops it, a `move` line like that of any move
  that failed. `shown` passes over the first line that names a move just
  borrowed.
- When two allies swap abilities (Skill Swap on a partner), the log does not
  name the abilities. When the swap was a Wandering Spirit's, set off by its
  own partner hitting it, that much can still be read from the line, since
  no move called Skill Swap was used; `shown` does read it.
- An ability can act once more after it has been replaced, when it had
  already begun: Poison Touch poisons through the same hit whose contact
  swapped it for Wandering Spirit or Mummy, and the line names it as the
  attacker's. `shown` goes by the replacement.

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
src/format.rs      a regulation as data: Format, the team validator, random legal teams
src/teams.rs       teams from outside: reading team sheets, guessing stat points, a pool of teams
src/shown.rs       what the battle has shown of each Pokémon: Battle::shown
src/observer.rs    the same, read from Showdown's log
src/state.rs       fixed-size state: Battle, Side, Pokemon, the action queue
src/data.rs        data definitions; src/tables.rs is generated (do not edit)
src/rng.rs         Showdown's Gen5RNG
src/replay.rs      replays a recorded battle and reports the first difference
src/trace.rs       optional RNG/action trace (feature `trace`)
src/bin/difftest.rs, src/bin/bench.rs, src/bin/teamcheck.rs, src/bin/teampool.rs
oracle/lib.js        what counts as modelled (the move properties and events the engine knows)
oracle/gen_data.js   Showdown data  -> src/tables.rs, pool.json, coverage.json
oracle/gen_cases.js  Showdown battles -> recorded cases (JSON lines)
oracle/gen_format.js Showdown's team validator -> formats/<id>.json
oracle/gen_teams.js  random teams with Showdown's verdict on each
formats/             what each regulation allows (generated)
scripts/             setup-oracle.sh, fuzz.sh, targeted.sh, check-teams.sh, mutation_test.py,
                     perish_trap.json (battles written out for gen_cases.js --script),
                     scrape_teams.py (a tournament's team sheets), check_scraper.py
teams/               pools of teams (teampool's output); teams/raw/ is what the scraper fetched
tests/parity.rs    fixture of recorded battles, choice-validation checks
tests/effects.rs   a few abilities and items checked directly, as API examples
tests/position.rs  positions written by hand: defaults, timers, switches in the middle of a turn
tests/format.rs    legal and illegal teams, a second regulation, fixtures of teams judged by Showdown
tests/shown.rs     what each side has been shown, as examples; a fixture of battles recorded with their logs
tests/perish_trap.rs  the perish trap, step by step, on battles played out in Showdown from a script
tests/teams.rs     team sheets read, stat points guessed, a pool saved, loaded and played
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

Keeping the record of what each side has been shown costs nothing that can
be measured. Nothing has been tuned beyond skipping events that nobody in the battle
listens to, and it shows: the last batch of mechanics cost 15 to 30% on the
same teams (the first row was about 10,000 before it), from a larger state
and more bookkeeping per action rather than from any one hot spot, and the
event system had already halved the speed of the first version, which ran
about 20,000 battles per second with nothing to dispatch. Caching each
Pokémon's listener set and slimming the per-move scratch state are the
likely first steps when speed starts to matter.

## What comes next

1. **Team preview**: bringing six and picking four, as a decision the engine
   asks for. (The teams as registered are already there:
   `Battle::with_rosters`.)
2. **The observation and the action space**: one view of the battle for a
   policy (its own side, `shown` of the other, the public field) and the
   encoding of choices. What goes into it depends on the model that will
   read it.
3. **Python bindings and batched stepping** for training.
4. **Sampling hidden information** into a position: the opponent's
   unrevealed moves, items, abilities, spreads and bench, drawn from a prior
   (usage statistics, or the bot's own model) and consistent with what has
   been shown.
5. **A Showdown client** that feeds the log to `observer` and plays the
   policy's choices.
6. **Searching for teams**: the regulation file gives the space of legal
   teams and steps that stay inside it; scoring a team needs a pool of
   opponents and a policy to play it, so this follows the bot.
7. **Speed.** The per-Pokémon listener cache described above, then
   profiling.
8. Keeping up with Showdown: `scripts/setup-oracle.sh` pins a commit; after
   moving the pin, `node oracle/gen_data.js` regenerates the tables, a
   missing callback body panics with its name, and `scripts/fuzz.sh` finds
   behaviour changes.
