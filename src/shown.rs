//! What a battle has shown about each Pokémon.
//!
//! The engine knows everything: every move, item, ability and exact HP on
//! both sides. A player does not. This module keeps, for every Pokémon, what
//! everyone watching the battle has been told about it, and
//! [`Battle::shown`] hands that out:
//!
//! ```
//! use vgc_engine::{Battle, Choice, PokemonSet};
//!
//! let set = |species: &str, moves: &[&str]| PokemonSet::from_names(species, moves, "Hardy", [0; 6]).unwrap();
//! let ours = [
//!     set("Incineroar", &["Fake Out", "Flare Blitz"]).ability("Intimidate").unwrap(),
//!     set("Garchomp", &["Earthquake", "Protect"]).item("Life Orb").unwrap(),
//!     set("Milotic", &["Scald"]),
//! ];
//! let theirs = [set("Sylveon", &["Hyper Voice"]), set("Arcanine", &["Protect"]), set("Snorlax", &["Body Slam"])];
//! let mut battle = Battle::new([&ours, &theirs], [1, 2, 3, 4]).unwrap();
//!
//! // Before anyone has moved: the leads are on the field, a third Pokémon has
//! // not been seen, and the only thing learned is the ability that announced itself.
//! let shown = battle.shown(0);
//! assert_eq!(shown.unseen, 1);
//! let incineroar = shown.active[0].as_ref().unwrap();
//! assert_eq!(incineroar.ability.as_deref(), Some("intimidate"));
//! assert!(incineroar.moves.is_empty());
//! let garchomp = shown.active[1].as_ref().unwrap();
//! assert_eq!((garchomp.item.as_deref(), garchomp.ability.as_deref()), (None, None));
//!
//! // Garchomp uses Earthquake: now that move is known, and so is the Life Orb that hurt it.
//! let protect = Choice::Move { slot: 0, target: 0, mega: false };
//! let fake_out = Choice::Move { slot: 0, target: 1, mega: false };
//! battle.choose([[fake_out, protect], [protect, protect]]).unwrap();
//! let shown = battle.shown(0);
//! let garchomp = shown.active[1].as_ref().unwrap();
//! assert_eq!(garchomp.moves, ["earthquake"]);
//! assert_eq!(garchomp.item.as_deref(), Some("lifeorb"));
//! ```
//!
//! "Shown" means exactly what Pokémon Showdown's battle log says, and the
//! engine is checked against a reader of that log ([`crate::observer`]) at
//! every decision of every recorded battle. What follows from that:
//!
//! * Nothing is worked out. A move that did less damage than it should have
//!   does not reveal an Assault Vest, a Pokémon that outsped another does not
//!   reveal a Choice Scarf, and a Frisk that found nothing does not reveal an
//!   empty hand. Only what the log names counts.
//! * Everything the log names counts, including lines a player's screen does
//!   not dwell on: in Champions a Regenerator heals with a line of its own
//!   when it leaves, and Natural Cure says what it cured.
//! * Under Illusion, what the disguised Pokémon shows is credited to the
//!   Pokémon it is disguised as, since that is who everyone thinks it is.
//!   When the disguise breaks, what was shown since it came in moves over to
//!   the Pokémon it really is. A disguise that is never broken is never
//!   corrected.
//! * HP is the percentage Showdown reports (rounded down, never 0 for a
//!   Pokémon that has not fainted), with the colour of the bar.
//! * One thing the log says is untrue, and is repeated here. A Pokémon locked
//!   in by a Choice item that is made to use another move fails with a line
//!   like that of any move that failed. If that was the second turn of a
//!   move borrowed with Copycat, it is credited with a move it does not have.
//!
//! What is public anyway is not repeated here: stat stages, volatile
//! conditions, the weather and each side's conditions are in the [`Battle`]
//! itself. (Their timers are not public, though: the turns a Reflect has
//! left give away a Light Clay.) PP is not tracked. And with Open Team
//! Sheets, which Showdown's Champions formats offer and force in
//! best-of-three, species, items, abilities and moves are on the table from
//! the start; what stays hidden then is stat points and which four of the
//! six were brought.

use serde::{Deserialize, Serialize};

use crate::data::*;
use crate::state::*;

/// The most moves remembered for one Pokémon. Four are its own; more can pile
/// up on a Pokémon that an Illusion user has passed itself off as.
pub(crate) const SHOWN_MOVES: usize = 8;
/// "Not shown" for an ability.
pub(crate) const UNKNOWN: u16 = u16::MAX;

/// The colour of an HP bar: green above half, yellow above a fifth, red below.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Bar {
    #[default]
    Green,
    Yellow,
    Red,
}

/// What has been shown of a held item.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ItemShown {
    Unknown,
    Holds(u16),
    /// Holds nothing now, having lost this.
    Lost(u16),
}

/// What everyone has been shown about one Pokémon (as it appears: see the module notes on Illusion).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Shown {
    /// 0 until it has appeared; then 1 for the first Pokémon its side showed, 2 for the second, and so on.
    pub seen: u8,
    /// The species it was when last on the field under its own name (a Mega stays a Mega).
    pub species: u16,
    pub fainted: bool,
    /// HP and status as they were when it was last on the field. (While it is on the field they are read off the Pokémon.)
    pub hp: u8,
    pub bar: Bar,
    pub status: Status,
    pub moves: [u16; SHOWN_MOVES],
    pub n_moves: u8,
    pub item: ItemShown,
    /// The ability it has now and the one it returns to on leaving the field; `UNKNOWN` if not shown.
    pub ability: u16,
    pub base_ability: u16,
    /// Its ability has been changed since it came in, so the next one shown is not the one it returns to.
    pub ability_changed: bool,
}

pub(crate) const NOTHING_SHOWN: Shown = Shown {
    seen: 0,
    species: NO_SPECIES,
    fainted: false,
    hp: 100,
    bar: Bar::Green,
    status: Status::None,
    moves: [NO_MOVE; SHOWN_MOVES],
    n_moves: 0,
    item: ItemShown::Unknown,
    ability: UNKNOWN,
    base_ability: UNKNOWN,
    ability_changed: false,
};

impl Shown {
    pub(crate) fn moves(&self) -> &[u16] {
        &self.moves[..self.n_moves as usize]
    }

    pub(crate) fn add_move(&mut self, id: u16) {
        if self.moves().contains(&id) || self.n_moves as usize == SHOWN_MOVES {
            return;
        }
        self.moves[self.n_moves as usize] = id;
        self.n_moves += 1;
    }

    /// A line credits it with `ability`.
    pub(crate) fn set_ability(&mut self, ability: u16) {
        // Unless its ability has been replaced since it came in, and by what is known: then
        // this is the old one still finishing what it had begun. (Poison Touch poisons on
        // the second hit of a Double Hit whose first swapped it for Wandering Spirit.)
        if self.ability_changed && self.ability != UNKNOWN && self.ability != ability {
            return;
        }
        self.ability = ability;
        if !self.ability_changed {
            self.base_ability = ability;
        }
    }

    /// The ability has been replaced; `ability` is the new one if that was shown.
    pub(crate) fn change_ability(&mut self, ability: u16) {
        self.ability = ability;
        self.ability_changed = true;
    }

    /// Leaving the field: the ability goes back to its own.
    pub(crate) fn leave(&mut self) {
        self.ability = self.base_ability;
        self.ability_changed = false;
    }

    /// What `self` has that `before` did not: what was learned in between.
    pub(crate) fn learned_since(&self, before: &Shown, into: &mut Shown) {
        for &m in self.moves() {
            if !before.moves().contains(&m) {
                into.add_move(m);
            }
        }
        if self.item != before.item {
            into.item = self.item;
        }
        if self.base_ability != before.base_ability {
            into.base_ability = self.base_ability;
        }
        if self.ability != before.base_ability || self.ability_changed {
            into.ability = self.ability;
            into.ability_changed = self.ability_changed;
        }
    }
}

/// What `-mega|pokemon|Species|Stone` says about the ability: a Mega has
/// just the one, and the stone and the species name the Mega. (Under
/// Illusion the species named is the disguise's; then the stone alone has to
/// do, and if it makes Megas with different abilities the line says nothing.)
pub(crate) fn mega_ability(stone: u16, named: &str) -> u16 {
    let pairs = ITEMS[stone as usize].mega;
    let fits = |from: u16| SPECIES[from as usize].base_species == named || SPECIES[from as usize].id == named;
    let any_fit = pairs.iter().any(|&(from, _)| fits(from));
    let mut ability = UNKNOWN;
    for &(from, to) in pairs {
        if any_fit && !fits(from) {
            continue;
        }
        let a = SPECIES[to as usize].ability0;
        if ability != UNKNOWN && ability != a {
            return UNKNOWN;
        }
        ability = a;
    }
    ability
}

/// HP as Showdown reports it to the other player in Champions: a whole
/// percentage, rounded down but never to 0, and the colour of the bar.
pub(crate) fn hp_shown(hp: u16, max: u16) -> (u8, Bar) {
    if hp == 0 || max == 0 {
        return (0, Bar::Red);
    }
    let (hp, max) = (hp as u32, max as u32);
    let pct = (100 * hp / max).max(1);
    let bar = if hp * 2 > max {
        Bar::Green
    } else if hp * 5 > max {
        Bar::Yellow
    } else {
        Bar::Red
    };
    (pct as u8, bar)
}

/// One Pokémon as everyone watching knows it.
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ShownMon {
    /// A handle that stays with this Pokémon for the whole battle: 1 for the
    /// first Pokémon its side showed, 2 for the second, and so on. (It says
    /// nothing about the order the team was brought in.)
    pub id: u8,
    /// The species on show, as an id: the Mega after Mega Evolution, the
    /// copied Pokémon after Transform, the disguise under Illusion.
    pub species: String,
    /// "M", "F" or "N".
    pub gender: String,
    /// HP as a whole percentage, rounded down; 0 only once fainted. For a
    /// Pokémon on the bench, as it was when it left the field.
    pub hp: u8,
    pub bar: Bar,
    /// "brn", "par", "psn", "tox", "slp", "frz" or empty.
    pub status: String,
    pub fainted: bool,
    /// Moves it has been seen to have, as ids, in the order they came to light.
    pub moves: Vec<String>,
    /// Its item if that has been shown: an id, or empty for "holds nothing". `None`: not shown.
    pub item: Option<String>,
    /// The item it is known to have lost (used, eaten, knocked off, stolen); empty if none.
    pub item_lost: String,
    /// The ability it has now, if shown.
    pub ability: Option<String>,
    /// The ability it returns to when it leaves the field, if shown.
    pub base_ability: Option<String>,
    /// It has taken another Pokémon's shape (and `species` is that Pokémon's).
    pub transformed: bool,
}

/// One side of the field as everyone watching knows it.
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ShownSide {
    /// Who is in each position on the field; `None` for a position whose Pokémon has fainted.
    pub active: Vec<Option<ShownMon>>,
    /// The Pokémon that have been on the field and are not now, in the order they first appeared.
    pub bench: Vec<ShownMon>,
    /// How many of the Pokémon brought have not appeared yet.
    pub unseen: u8,
    /// How many have not fainted.
    pub left: u8,
}

pub(crate) fn ability_name(a: u16) -> Option<String> {
    (a != UNKNOWN).then(|| if a == ab::NOABILITY { String::new() } else { ABILITIES[a as usize].id.to_string() })
}

impl ShownMon {
    pub(crate) fn from_record(rec: &Shown, species: u16, gender: Gender) -> ShownMon {
        let (item, item_lost) = match rec.item {
            ItemShown::Unknown => (None, String::new()),
            ItemShown::Holds(i) => (Some(ITEMS[i as usize].id.to_string()), String::new()),
            ItemShown::Lost(i) => (Some(String::new()), ITEMS[i as usize].id.to_string()),
        };
        ShownMon {
            id: rec.seen,
            species: SPECIES[species as usize].id.to_string(),
            gender: gender.id().to_string(),
            hp: rec.hp,
            bar: rec.bar,
            status: rec.status.id().to_string(),
            fainted: rec.fainted,
            moves: rec.moves().iter().map(|&m| MOVES[m as usize].id.to_string()).collect(),
            item,
            item_lost,
            ability: ability_name(rec.ability),
            base_ability: ability_name(rec.base_ability),
            transformed: false,
        }
    }
}

impl Battle {
    /// The team index of the Pokémon `r` passes for (itself, unless Illusion says otherwise).
    pub(crate) fn shown_as(&self, r: MonRef) -> usize {
        let m = self.mon(r);
        if m.illusion != 0 { m.illusion as usize - 1 } else { r.idx as usize }
    }

    /// Whether `r` is on the field and has not fainted, as far as the log has said.
    pub(crate) fn shown_live(&self, r: MonRef) -> bool {
        self.has_live(r)
    }

    /// Whether `r` is on the field with a record of its own for this stay.
    fn has_live(&self, r: MonRef) -> bool {
        let m = self.mon(r);
        m.is_active && m.live.seen != 0
    }

    /// The record that something shown about `r` right now is written to.
    pub(crate) fn shown_mut(&mut self, r: MonRef) -> &mut Shown {
        if self.has_live(r) {
            &mut self.mon_mut(r).live
        } else {
            let a = self.shown_as(r);
            &mut self.sides[r.side as usize].shown[a]
        }
    }

    /// Changes the record of what `r` has shown. (With the `trace` feature,
    /// counts the use of the place this was asked from: see `trace::shown_sites`.)
    #[track_caller]
    fn show(&mut self, r: MonRef, what: &'static str, change: impl FnOnce(&mut Shown)) {
        let rec = self.shown_mut(r);
        #[cfg(feature = "trace")]
        let before = *rec;
        change(rec);
        #[cfg(feature = "trace")]
        crate::trace::shown_site(std::panic::Location::caller(), what, before != *rec);
        let _ = what;
    }

    /// `r` has been seen to have this move. (Not while it is transformed:
    /// those are the moves of the Pokémon it copied.)
    #[track_caller]
    pub(crate) fn show_move(&mut self, r: MonRef, move_id: u16) {
        if move_id == NO_MOVE || move_id == crate::battle::struggle_id() || self.mon(r).transformed {
            return;
        }
        self.show(r, MOVES[move_id as usize].id, |rec| rec.add_move(move_id));
    }

    /// `r`'s ability has been named.
    #[track_caller]
    pub(crate) fn show_ability(&mut self, r: MonRef, ability: u16) {
        self.show(r, ABILITIES[ability as usize].id, |rec| rec.set_ability(ability));
    }

    /// The ability the running callback belongs to has been named, on its holder.
    #[track_caller]
    pub(crate) fn show_own_ability(&mut self) {
        if let (Eff::Ability(a), Some(Holder::Mon(h))) = (self.effect, self.effect_holder) {
            self.show_ability(h, a);
        }
    }

    /// `r`'s ability was replaced in plain sight; `ability` is the new one, or `UNKNOWN`.
    #[track_caller]
    pub(crate) fn show_ability_change(&mut self, r: MonRef, ability: u16) {
        self.show(r, "", |rec| rec.change_ability(ability));
    }

    /// A line names `item` as `r`'s own (`[from] item: X`, `item: X`).
    #[track_caller]
    pub(crate) fn show_item(&mut self, r: MonRef, item: u16) {
        self.show(r, ITEMS[item as usize].id, |rec| rec.name_item(item));
    }

    /// `r` has been shown to come by `item` (`-item`).
    #[track_caller]
    pub(crate) fn show_item_gain(&mut self, r: MonRef, item: u16) {
        if item != it::NONE {
            self.show(r, ITEMS[item as usize].id, |rec| rec.item = ItemShown::Holds(item));
        }
    }

    /// The line that says `item` has reached `r` comes after it was handed over,
    /// and something may have happened to it in between (a White Herb used at
    /// once has had its own `-enditem`): then that stands.
    #[track_caller]
    pub(crate) fn show_item_arrived(&mut self, r: MonRef, item: u16) {
        if self.mon(r).item == item {
            self.show_item_gain(r, item);
        }
    }

    /// `r` has been shown to lose `item`, and holds nothing now.
    #[track_caller]
    pub(crate) fn show_item_lost(&mut self, r: MonRef, item: u16) {
        if item != it::NONE {
            self.show(r, ITEMS[item as usize].id, |rec| rec.item = ItemShown::Lost(item));
        }
    }

    /// A log line with `[from] effect`, as Showdown's `damage`, `heal` and the
    /// status messages write them: the ability or item is shown on whoever has
    /// it. That is the holder of the running callback when the effect is its
    /// own; for an effect named from elsewhere, the Pokémon the line points at
    /// (`[of] source` if it has one, else its subject).
    #[track_caller]
    pub(crate) fn show_from(&mut self, effect: Eff, target: MonRef, of: Option<MonRef>) {
        let holder = match self.effect_holder {
            Some(Holder::Mon(h)) if effect == self.effect => h,
            _ => of.unwrap_or(target),
        };
        match effect {
            Eff::Ability(a) => self.show_ability(holder, a),
            Eff::Item(i) => self.show_item(holder, i),
            _ => {}
        }
    }

    /// The `switch` or `drag` line: `r` appears, as whoever it passes for.
    pub(crate) fn shown_enter(&mut self, r: MonRef) {
        let a = self.shown_as(r);
        let side = &mut self.sides[r.side as usize];
        if side.shown[a].seen == 0 {
            side.n_seen += 1;
            side.shown[a].seen = side.n_seen;
        }
        side.shown[a].species = side.team[a].species;
        let mut live = side.shown[a];
        live.fainted = false;
        live.leave();
        self.mon_mut(r).live = live;
    }

    /// HP and status of `r` as the log has them now.
    fn shown_condition(&self, r: MonRef) -> (u8, Bar, Status) {
        let m = self.mon(r);
        let (hp, bar) = hp_shown(m.hp, m.max_hp());
        (hp, bar, if m.hp == 0 { Status::None } else { m.status })
    }

    /// `r` leaves the field (called back, dragged out or fainted): what was
    /// shown during its stay is filed under the Pokémon it passed for.
    pub(crate) fn shown_leave(&mut self, r: MonRef, fainted: bool) {
        if !self.has_live(r) {
            return;
        }
        let (hp, bar, status) = self.shown_condition(r);
        let a = self.shown_as(r);
        let mut live = self.mon(r).live;
        (live.hp, live.bar, live.status, live.fainted) = (hp, bar, status, fainted);
        if fainted {
            (live.hp, live.bar, live.status) = (0, Bar::Red, Status::None);
        }
        live.leave();
        // Its place in the order of appearance belongs to the name, which another Pokémon may share.
        live.seen = self.sides[r.side as usize].shown[a].seen;
        self.sides[r.side as usize].shown[a] = live;
        self.mon_mut(r).live = NOTHING_SHOWN;
    }

    /// Illusion ends (`replace`): the Pokémon in this position is, and has
    /// been since it came in, someone else. `was` is the team index it passed for.
    pub(crate) fn shown_unmask(&mut self, r: MonRef, was: usize) {
        if !self.has_live(r) {
            return;
        }
        let side = &mut self.sides[r.side as usize];
        let live = side.team[r.idx as usize].live;
        let before = side.shown[was];
        if side.shown[r.idx as usize].seen == 0 {
            side.n_seen += 1;
            side.shown[r.idx as usize].seen = side.n_seen;
        }
        side.shown[r.idx as usize].species = side.team[r.idx as usize].species;
        let mut own = side.shown[r.idx as usize];
        own.fainted = false;
        own.leave();
        live.learned_since(&before, &mut own);
        side.team[r.idx as usize].live = own;
    }

    /// A benched (or fainted) Pokémon's HP or status changed in plain sight.
    pub(crate) fn shown_bench_update(&mut self, r: MonRef) {
        if self.has_live(r) {
            return;
        }
        let (hp, bar, status) = self.shown_condition(r);
        let fainted = self.mon(r).fainted;
        let a = self.shown_as(r);
        let side = &mut self.sides[r.side as usize];
        if side.shown[a].seen == 0 {
            side.n_seen += 1;
            side.shown[a].seen = side.n_seen;
        }
        let species = side.team[a].species;
        let rec = &mut side.shown[a];
        if rec.species == NO_SPECIES {
            rec.species = species;
        }
        (rec.hp, rec.bar, rec.status, rec.fainted) = (hp, bar, status, fainted);
    }

    /// `detailschange`: `r` has changed forme for good. (Under Illusion the line shows the disguise, unchanged.)
    pub(crate) fn shown_new_forme(&mut self, r: MonRef, species: u16) {
        if self.mon(r).illusion == 0 {
            self.shown_mut(r).species = species;
        }
    }

    /// What everyone watching has been shown about `side`'s Pokémon.
    pub fn shown(&self, side: usize) -> ShownSide {
        let s = &self.sides[side];
        let mut out = ShownSide { left: s.pokemon_left, ..Default::default() };
        let mut on_field = [false; MAX_TEAM];
        for pos in 0..ACTIVE.min(s.n as usize) {
            let r = MonRef { side: side as u8, idx: s.order[pos] };
            if !self.has_live(r) {
                out.active.push(None);
                continue;
            }
            let m = self.mon(r);
            let a = self.shown_as(r);
            on_field[a] = true;
            // A disguise is the Pokémon it copies as that one looked when the disguise came in.
            let species = if m.illusion != 0 { m.live.species } else { m.species };
            let mut mon = ShownMon::from_record(&m.live, species, s.team[a].gender);
            let (hp, bar, status) = self.shown_condition(r);
            (mon.hp, mon.bar, mon.status) = (hp, bar, status.id().to_string());
            mon.fainted = false;
            mon.transformed = m.transformed;
            out.active.push(Some(mon));
        }
        while out.active.len() < ACTIVE {
            out.active.push(None);
        }
        let mut bench = Vec::new();
        for (a, rec) in s.shown.iter().enumerate().take(s.n as usize) {
            if rec.seen == 0 {
                out.unseen += 1;
            } else if !on_field[a] {
                bench.push(ShownMon::from_record(rec, rec.species, s.team[a].gender));
            }
        }
        bench.sort_by_key(|m| m.id);
        out.bench = bench;
        out
    }
}
