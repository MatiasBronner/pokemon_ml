//! The core of the simulator: a port of the parts of Pokémon Showdown's
//! `sim/battle.ts`, `sim/pokemon.ts` and `sim/battle-queue.ts` that the
//! supported mechanics exercise, under the Champions mod. Using a move is in
//! `moves.rs`, the event system in `events.rs`, and the public entry points
//! (`Battle::new`, `choose`, legal choices) in `choice.rs`.
//!
//! The port is deliberately literal: it draws random numbers in the same
//! order Showdown does, including the draws Showdown makes for reasons that
//! have no in-game meaning (re-resolving targets when re-sorting the queue,
//! shuffling speed ties among the handlers of every event). That is what
//! makes it possible to replay a Showdown battle here from the same seed and
//! compare state after every decision. Comments name the Showdown function
//! being mirrored.

// The control flow follows Showdown's on purpose, so keep its shape even where
// clippy would fold branches or loops together.
#![allow(clippy::needless_range_loop, clippy::collapsible_if, clippy::collapsible_match, clippy::if_same_then_else)]

use crate::data::*;
use crate::state::*;
use crate::trace;

/// How a player describes one Pokémon to bring.
#[derive(Clone, Debug)]
pub struct PokemonSet {
    pub species: u16,
    pub moves: Vec<u16>,
    /// (raised, lowered) stat indices 1..=5, or (0, 0) for a neutral nature.
    pub nature: (u8, u8),
    /// Champions stat points: hp, atk, def, spa, spd, spe.
    pub stat_points: [u8; 6],
    /// Index into `data::ABILITIES`.
    pub ability: u16,
    /// Index into `data::ITEMS`; 0 is no item.
    pub item: u16,
    pub gender: Gender,
}

impl PokemonSet {
    /// Builds a set from Showdown names or ids, e.g.
    /// `PokemonSet::from_names("Garchomp", &["Earthquake", "Protect"], "Jolly", [0, 32, 0, 0, 2, 32])`.
    /// Stat points are hp, atk, def, spa, spd, spe. The set starts with no
    /// ability and no item; add them with [`PokemonSet::ability`] and
    /// [`PokemonSet::item`]. The gender is the species' fixed gender, or male
    /// if it can be either; change it with [`PokemonSet::gender`].
    pub fn from_names(
        species: &str,
        moves: &[&str],
        nature_name: &str,
        stat_points: [u8; 6],
    ) -> Result<PokemonSet, Error> {
        let species =
            species_id(&to_id(species)).ok_or_else(|| Error::BadTeam(format!("unknown species {species}")))?;
        let moves = moves
            .iter()
            .map(|m| move_id(&to_id(m)).ok_or_else(|| Error::BadTeam(format!("unknown move {m}"))))
            .collect::<Result<Vec<_>, _>>()?;
        let nature = nature(nature_name).ok_or_else(|| Error::BadTeam(format!("unknown nature {nature_name}")))?;
        let gender = SPECIES[species as usize].gender.unwrap_or(Gender::M);
        Ok(PokemonSet { species, moves, nature, stat_points, ability: ab::NOABILITY, item: it::NONE, gender })
    }

    /// Sets the ability by Showdown name or id.
    pub fn ability(mut self, name: &str) -> Result<PokemonSet, Error> {
        self.ability = ability_id(&to_id(name)).ok_or_else(|| Error::BadTeam(format!("unknown ability {name}")))?;
        Ok(self)
    }

    /// Sets the held item by Showdown name or id; the empty string is no item.
    pub fn item(mut self, name: &str) -> Result<PokemonSet, Error> {
        self.item = item_id(&to_id(name)).ok_or_else(|| Error::BadTeam(format!("unknown item {name}")))?;
        Ok(self)
    }

    pub fn gender(mut self, gender: Gender) -> PokemonSet {
        self.gender = gender;
        self
    }
}

#[derive(Debug)]
pub enum Error {
    /// The request uses something the engine does not model yet.
    Unsupported(String),
    BadTeam(String),
    BadChoice(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::BadTeam(s) => write!(f, "bad team: {s}"),
            Error::BadChoice(s) => write!(f, "bad choice: {s}"),
        }
    }
}

impl std::error::Error for Error {}

/// An entry of Showdown's `targets` arrays, which are overwritten with
/// `false` as targets drop out of a move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Tgt {
    Mon(MonRef),
    Gone,
}

pub(crate) const MAX_TARGETS: usize = 3;
pub(crate) type Targets = [Tgt; MAX_TARGETS];
pub(crate) type Damage = [Res; MAX_TARGETS];

/// `Battle#modify` with a modifier in 4096ths. Showdown truncates the
/// product to 32 bits before rounding.
pub(crate) fn modify(value: u32, modifier: u32) -> u32 {
    let product = (value as u64 * modifier as u64) as u32;
    ((product as u64 + 2047) / 4096) as u32
}

/// A stat after stage changes (the boost table in `Pokemon#calculateStat`).
pub(crate) fn boosted(stat: u32, stage: i32) -> u32 {
    let s = stage.clamp(-6, 6);
    if s >= 0 { stat * (2 + s as u32) / 2 } else { stat * 2 / (2 + (-s) as u32) }
}

/// `Math.round(a * num / den)` for non-negative integers.
pub(crate) fn round_div(a: u32, num: u32, den: u32) -> u32 {
    (2 * a * num + den) / (2 * den)
}

pub(crate) fn struggle_id() -> u16 {
    move_id("struggle").expect("struggle is always in the move table")
}

/// `Battle#comparePriority` for queued actions: order low to high, priority
/// high to low, speed high to low. Negative means `a` goes first.
pub(crate) fn cmp_action(a: &Action, b: &Action) -> i64 {
    let o = a.order as i64 - b.order as i64;
    if o != 0 {
        return o;
    }
    let p = b.priority as i64 - a.priority as i64;
    if p != 0 {
        return p;
    }
    b.speed as i64 - a.speed as i64
}

impl Battle {
    // ---------------------------------------------------------------- access

    pub fn mon(&self, r: MonRef) -> &Pokemon {
        &self.sides[r.side as usize].team[r.idx as usize]
    }
    pub(crate) fn mon_mut(&mut self, r: MonRef) -> &mut Pokemon {
        &mut self.sides[r.side as usize].team[r.idx as usize]
    }
    /// The Pokémon occupying an active slot (it may be fainted).
    pub fn active(&self, side: usize, pos: usize) -> MonRef {
        MonRef { side: side as u8, idx: self.sides[side].order[pos] }
    }
    /// The Pokémon at any position in a side's current order (active or benched).
    pub fn active_at(&self, side: usize, pos: usize) -> MonRef {
        MonRef { side: side as u8, idx: self.sides[side].order[pos] }
    }
    /// Whether an active slot's occupant has actually entered the battle.
    /// (Showdown's `side.active` holds nothing until the opening switch-ins.)
    pub(crate) fn in_play(&self, r: MonRef) -> bool {
        let m = self.mon(r);
        m.is_active || m.fainted
    }
    /// `Battle#getAllActive`.
    pub(crate) fn all_active(&self, include_fainted: bool) -> ([MonRef; 4], usize) {
        let mut out = [MonRef { side: 0, idx: 0 }; 4];
        let mut n = 0;
        for side in 0..2 {
            for pos in 0..ACTIVE {
                let r = self.active(side, pos);
                if self.in_play(r) && (include_fainted || !self.mon(r).fainted) {
                    out[n] = r;
                    n += 1;
                }
            }
        }
        (out, n)
    }
    /// `Side#allies()`: this side's active Pokémon that still have HP.
    pub(crate) fn allies_and_self(&self, side: usize) -> ([MonRef; ACTIVE], usize) {
        let mut out = [MonRef { side: 0, idx: 0 }; ACTIVE];
        let mut n = 0;
        for pos in 0..ACTIVE {
            let r = self.active(side, pos);
            if self.in_play(r) && self.mon(r).hp > 0 {
                out[n] = r;
                n += 1;
            }
        }
        (out, n)
    }
    /// `Pokemon#adjacentAllies` in doubles: the partner, if neither has fainted.
    pub(crate) fn adjacent_allies(&self, r: MonRef) -> ([MonRef; ACTIVE], usize) {
        let mut out = [r; ACTIVE];
        let mut n = 0;
        if self.mon(r).fainted {
            return (out, 0);
        }
        let (allies, k) = self.allies_and_self(r.side as usize);
        for &a in &allies[..k] {
            if a != r && !self.mon(a).fainted {
                out[n] = a;
                n += 1;
            }
        }
        (out, n)
    }
    pub(crate) fn is_ally(&self, a: MonRef, b: MonRef) -> bool {
        a.side == b.side
    }
    /// `Pokemon#getLocOf`.
    pub(crate) fn loc_of(&self, from: MonRef, target: MonRef) -> i8 {
        let p = self.mon(target).position as i8 + 1;
        if from.side == target.side { -p } else { p }
    }
    /// `Pokemon#getAtLoc`.
    pub(crate) fn at_loc(&self, from: MonRef, loc: i8) -> Option<MonRef> {
        let side = if loc < 0 { from.side as usize } else { 1 - from.side as usize };
        let pos = loc.unsigned_abs() as usize;
        if pos == 0 || pos > ACTIVE {
            return None;
        }
        let r = self.active(side, pos - 1);
        self.in_play(r).then_some(r)
    }
    /// `Battle#canSwitch`.
    pub(crate) fn can_switch(&self, side: usize) -> bool {
        let s = &self.sides[side];
        if s.pokemon_left == 0 {
            return false;
        }
        (ACTIVE..s.n as usize).any(|p| !s.team[s.order[p] as usize].fainted)
    }
    /// Index of a Pokémon's slot in per-move hit data.
    pub(crate) fn slot_index(&self, r: MonRef) -> usize {
        r.side as usize * 2 + (self.mon(r).position as usize).min(1)
    }

    // ------------------------------------------------------------------- rng

    pub(crate) fn rand(&mut self, n: u32, what: &'static str) -> u32 {
        let v = self.rng.below(n);
        trace::rng(what, n, v);
        v
    }
    pub(crate) fn chance(&mut self, num: u32, den: u32, what: &'static str) -> bool {
        self.rand(den, what) < num
    }
    /// `PRNG#random(from, to)`.
    pub(crate) fn rand_range(&mut self, from: usize, to: usize, what: &'static str) -> usize {
        let v = self.rng.range(from as u32, to as u32);
        trace::rng_range(what, from as u32, to as u32, v);
        v as usize
    }
    /// `PRNG#shuffle` on `items[start..end]`.
    fn shuffle<T>(&mut self, items: &mut [T], mut start: usize, end: usize, what: &'static str) {
        while start + 1 < end {
            let next = self.rand_range(start, end, what);
            if start != next {
                items.swap(start, next);
            }
            start += 1;
        }
    }
    /// `Battle#speedSort`: a selection sort that shuffles each group of ties.
    pub(crate) fn speed_sort<T: Copy>(&mut self, list: &mut [T], cmp: impl Fn(&T, &T) -> i64, what: &'static str) {
        if list.len() < 2 {
            return;
        }
        let mut sorted = 0;
        let mut next = [0usize; 64];
        while sorted + 1 < list.len() {
            next[0] = sorted;
            let mut n_next = 1;
            for i in sorted + 1..list.len() {
                let delta = cmp(&list[next[0]], &list[i]);
                if delta < 0 {
                    continue;
                }
                if delta > 0 {
                    next[0] = i;
                    n_next = 1;
                } else {
                    next[n_next] = i;
                    n_next += 1;
                }
            }
            for (i, &index) in next[..n_next].iter().enumerate() {
                if index != sorted + i {
                    list.swap(sorted + i, index);
                }
            }
            if n_next > 1 {
                self.shuffle(list, sorted, sorted + n_next, what);
            }
            sorted += n_next;
        }
    }

    // ----------------------------------------------------------------- stats

    /// `Pokemon#getStat` (stat index 1 = atk .. 5 = spe).
    pub(crate) fn get_stat(&mut self, r: MonRef, stat: usize, unboosted: bool, unmodified: bool) -> u32 {
        let mut value = self.mon(r).stats[stat] as u32;
        if !unboosted {
            let mut boosts = self.mon(r).boosts;
            if !unmodified {
                boosts = self.modify_boost(r, boosts);
            }
            value = boosted(value, boosts[stat - 1] as i32);
        }
        if !unmodified {
            const EVS: [Ev; 5] = [Ev::ModifyAtk, Ev::ModifyDef, Ev::ModifySpA, Ev::ModifySpD, Ev::ModifySpe];
            value = self.run_event(EVS[stat - 1], Some(r), None, Eff::None, Res::Num(value as i32)).num() as u32;
        }
        if stat == 5 && value > 10000 {
            value = 10000;
        }
        value
    }

    /// The `ModifyBoost` event on a table of stat stages.
    pub(crate) fn modify_boost(&mut self, r: MonRef, boosts: Boosts) -> Boosts {
        if self.event_mask & Ev::ModifyBoost.bit() == 0 {
            return boosts;
        }
        let mut e = Event::new(Ev::ModifyBoost, Some(r), None, Eff::None);
        e.boosts = boosts;
        self.run_event_ex(e, Res::Undef, false, false).1.boosts
    }

    /// `Pokemon#calculateStat`: a stat at a given stage, as seen by `stat_user`.
    pub(crate) fn calculate_stat(&mut self, r: MonRef, stat: usize, boost: i32, stat_user: MonRef) -> u32 {
        let value = self.mon(r).stats[stat] as u32;
        let mut boosts = [0i8; 7];
        boosts[stat - 1] = boost as i8;
        let boosts = self.modify_boost(stat_user, boosts);
        boosted(value, boosts[stat - 1] as i32)
    }

    /// `Pokemon#getActionSpeed` under the Champions mod (no Trick Room yet).
    pub(crate) fn action_speed(&mut self, r: MonRef) -> i32 {
        self.get_stat(r, 5, false, false) as i32
    }
    /// `Battle#updateSpeed`.
    pub(crate) fn update_speed(&mut self) {
        let (actives, n) = self.all_active(false);
        for &r in &actives[..n] {
            let s = self.action_speed(r);
            self.mon_mut(r).speed = s;
        }
    }

    // ----------------------------------------------------- types and immunity

    /// `Pokemon#hasType`.
    pub(crate) fn has_type(&self, r: MonRef, t: Type) -> bool {
        let m = self.mon(r);
        m.types[0] == t || m.types[1] == t
    }

    /// `dex.getImmunity` for a status-like immunity kind: whether the
    /// Pokémon's types leave it open to it.
    pub(crate) fn type_allows(&self, r: MonRef, kind: usize) -> bool {
        !self.mon(r).types.iter().any(|&t| t != Type::None && t != Type::Typeless && STATUS_IMMUNE[kind][t as usize])
    }

    /// `Pokemon#runStatusImmunity`: false means immune.
    pub(crate) fn run_status_immunity(&mut self, r: MonRef, imm: Imm) -> bool {
        if self.mon(r).fainted {
            return false;
        }
        let natural = match imm {
            Imm::Status(Status::Brn) => self.type_allows(r, 0),
            Imm::Status(Status::Par) => self.type_allows(r, 1),
            Imm::Status(Status::Psn) | Imm::Status(Status::Tox) => self.type_allows(r, 2),
            Imm::Status(Status::Slp) => self.type_allows(r, 3),
            Imm::Status(Status::Frz) => self.type_allows(r, 4),
            Imm::Status(Status::None) => return true,
            Imm::Powder => self.type_allows(r, 5),
            Imm::Trapped => self.type_allows(r, 6),
            Imm::Prankster => self.type_allows(r, 7),
            Imm::Vol(_) => true,
        };
        if !natural {
            return false;
        }
        if self.event_mask & Ev::Immunity.bit() == 0 {
            return true;
        }
        let mut e = Event::new(Ev::Immunity, Some(r), None, Eff::None);
        e.imm = Some(imm);
        self.run_event_ex(e, Res::Undef, false, false).0.truthy()
    }

    /// `Pokemon#isGrounded`. Showdown returns `null` for Levitate, which every
    /// caller treats like `false`.
    pub(crate) fn is_grounded(&mut self, r: MonRef, negate_immunity: bool) -> bool {
        let item = if self.ignoring_item(r) { it::NONE } else { self.mon(r).item };
        if item == it::IRONBALL {
            return true;
        }
        if !negate_immunity && self.has_type(r, Type::Flying) {
            return false;
        }
        if (self.has_ability(r, ab::LEVITATE) || self.has_ability(r, ab::EELEVATE))
            && !self.suppressing_ability(Some(r))
        {
            return false;
        }
        item != it::AIRBALLOON
    }

    /// `Pokemon#runImmunity` for a move: false means the move's type cannot hit.
    pub(crate) fn run_immunity(&mut self, target: MonRef, mi: u8) -> bool {
        let am = &self.am[mi as usize];
        let typ = am.typ;
        match am.ignore_immunity {
            IgnoreImm::All => return true,
            IgnoreImm::NormalFighting if matches!(typ, Type::Normal | Type::Fighting) => return true,
            _ => {}
        }
        if typ == Type::Typeless || typ == Type::None {
            return true;
        }
        // The NegateImmunity event has no listeners among modelled effects.
        if typ == Type::Ground {
            return self.is_grounded(target, false);
        }
        !self
            .mon(target)
            .types
            .iter()
            .any(|&t| t != Type::None && t != Type::Typeless && TYPE_CHART[typ as usize][t as usize] == 3)
    }

    /// `Pokemon#runEffectiveness`.
    pub(crate) fn run_effectiveness(&mut self, target: MonRef, mi: u8) -> i32 {
        let typ = self.am[mi as usize].typ;
        let types = self.mon(target).types;
        let mut total = 0;
        for &t in &types {
            if t == Type::None {
                continue;
            }
            let mut type_mod = 0;
            if typ != Type::Typeless && t != Type::Typeless {
                type_mod = match TYPE_CHART[typ as usize][t as usize] {
                    1 => 1,
                    2 => -1,
                    _ => 0,
                };
            }
            if self.event_mask & Ev::Effectiveness.bit() != 0 {
                let mut e = Event::new(Ev::Effectiveness, Some(target), None, Eff::Move(mi));
                e.typ = t;
                type_mod = self.run_event_ex(e, Res::Num(type_mod), false, false).0.num();
            }
            total += type_mod;
        }
        total
    }

    // ------------------------------------------------------------- targeting

    /// `Battle#validTargetLoc`.
    pub(crate) fn valid_target_loc(&self, loc: i8, source: MonRef, target: Target) -> bool {
        if loc == 0 {
            return true;
        }
        let num_slots = ACTIVE as i8;
        let source_loc = self.loc_of(source, source);
        if loc.abs() > num_slots {
            return false;
        }
        let is_self = source_loc == loc;
        let is_foe = loc > 0;
        let across = -(num_slots + 1 - loc);
        let is_adjacent = if loc > 0 { (across - source_loc).abs() <= 1 } else { (loc - source_loc).abs() == 1 };
        match target {
            Target::RandomNormal | Target::Scripted | Target::Normal => is_adjacent,
            Target::AdjacentAlly => is_adjacent && !is_foe,
            Target::AdjacentAllyOrSelf => (is_adjacent && !is_foe) || is_self,
            Target::AdjacentFoe => is_adjacent && is_foe,
            Target::Any => !is_self,
            _ => false,
        }
    }

    /// `Battle#getRandomTarget`. Note that `sample` draws even from a
    /// one-element list.
    pub(crate) fn get_random_target(&mut self, user: MonRef, target: Target) -> Option<MonRef> {
        match target {
            Target::User | Target::All | Target::AllySide | Target::AllyTeam | Target::AdjacentAllyOrSelf => Some(user),
            Target::AdjacentAlly => {
                let (adj, k) = self.adjacent_allies(user);
                if k == 0 { None } else { Some(adj[self.rand(k as u32, "random ally target") as usize]) }
            }
            _ => {
                let foe = 1 - user.side as usize;
                let (foes, n) = self.allies_and_self(foe);
                if n == 0 {
                    let r = self.active(foe, 0);
                    self.in_play(r).then_some(r)
                } else {
                    Some(foes[self.rand(n as u32, "random foe target") as usize])
                }
            }
        }
    }

    /// `Battle#getTarget`.
    pub(crate) fn get_target(
        &mut self,
        user: MonRef,
        target: Target,
        loc: i8,
        original: Option<MonRef>,
    ) -> Option<MonRef> {
        if self.has_ability(user, ab::STALWART) || self.has_ability(user, ab::PROPELLERTAIL) {
            if let Some(o) = original {
                if self.mon(o).is_active {
                    return Some(o);
                }
            }
        }
        let self_loc = self.loc_of(user, user);
        if matches!(target, Target::AdjacentAlly | Target::Any | Target::Normal) && loc == self_loc {
            return None;
        }
        if target != Target::RandomNormal && self.valid_target_loc(loc, user, target) {
            if let Some(t) = self.at_loc(user, loc) {
                if self.mon(t).fainted {
                    if t.side == user.side {
                        if target == Target::AdjacentAllyOrSelf {
                            return Some(user);
                        }
                        // A fainted ally is not retargeted.
                        return Some(t);
                    }
                } else {
                    return Some(t);
                }
            }
        }
        self.get_random_target(user, target)
    }

    // ----------------------------------------------------------------- queue

    /// `Battle#getActionSpeed`. For moves this re-resolves the target for the
    /// `ModifyPriority` hooks; the lookup can draw from the RNG, so it has to
    /// happen even when nothing listens.
    pub(crate) fn resolve_speed(&mut self, a: &mut Action) {
        let Some(r) = a.mon else {
            a.speed = 1;
            return;
        };
        if a.kind == ActKind::Move {
            let d = &MOVES[a.move_id as usize];
            let mut priority = d.priority as i32;
            let target = self.get_target(r, d.target, a.target_loc, None);
            if self.event_mask & Ev::ModifyPriority.bit() != 0 {
                // Handlers see the queued move itself; give it a scratch slot.
                let saved = self.am_len;
                let mi = self.new_am(a.move_id);
                self.am[mi as usize].prankster_boosted = a.prankster;
                priority = self.run_event(Ev::ModifyPriority, Some(r), target, Eff::Move(mi), Res::Num(priority)).num();
                a.prankster = self.am[mi as usize].prankster_boosted;
                self.am_len = saved;
            }
            a.priority = priority * 10 + a.frac as i32;
            a.move_priority = priority as i8;
        }
        a.speed = self.action_speed(r);
    }

    pub(crate) fn blank_action(kind: ActKind, order: u32) -> Action {
        Action {
            kind,
            order,
            priority: 0,
            speed: 1,
            mon: None,
            move_id: 0,
            target_loc: 0,
            switch_to: None,
            frac: 0,
            move_priority: 0,
            prankster: false,
            orig_target: None,
        }
    }

    /// `BattleQueue#resolveAction` for a move.
    pub(crate) fn resolve_move(&mut self, user: MonRef, move_id: u16, loc: i8) -> Action {
        let d = &MOVES[move_id as usize];
        let mut a = Battle::blank_action(ActKind::Move, 200);
        a.mon = Some(user);
        a.move_id = move_id;
        a.target_loc = loc;
        if self.event_mask & Ev::FractionalPriority.bit() != 0 {
            let saved = self.am_len;
            let mi = self.new_am(move_id);
            a.frac = self.run_event(Ev::FractionalPriority, Some(user), None, Eff::Move(mi), Res::Num(0)).num() as i8;
            self.am_len = saved;
        }
        if a.target_loc == 0 {
            if let Some(t) = self.get_random_target(user, d.target) {
                a.target_loc = self.loc_of(user, t);
            }
        }
        a.orig_target = self.at_loc(user, a.target_loc);
        self.resolve_speed(&mut a);
        a
    }

    /// `BattleQueue#resolveAction` for a switch chosen at a move or switch request.
    pub(crate) fn resolve_switch(&mut self, kind: ActKind, out: MonRef, incoming: MonRef) -> Action {
        let mut a = Battle::blank_action(kind, if kind == ActKind::InstaSwitch { 3 } else { 103 });
        a.mon = Some(out);
        a.switch_to = Some(incoming);
        self.mon_mut(out).switch_flag = false;
        self.resolve_speed(&mut a);
        a
    }

    /// `BattleQueue#insertChoice`: place an action where it would have sorted,
    /// picking a random spot among actions it ties with.
    pub(crate) fn insert_choice(&mut self, a: Action) {
        let mut first = None;
        let mut last = None;
        for (i, cur) in self.queue.as_slice().iter().enumerate() {
            let c = cmp_action(&a, cur);
            if c <= 0 && first.is_none() {
                first = Some(i);
            }
            if c < 0 {
                last = Some(i);
                break;
            }
        }
        match first {
            None => self.queue.push(a),
            Some(f) => {
                let l = last.unwrap_or(self.queue.len as usize);
                let index = if f == l { f } else { self.rand_range(f, l + 1, "insert choice tie") };
                self.queue.insert(index, a);
            }
        }
    }

    pub(crate) fn sort_queue(&mut self) {
        let mut q = self.queue;
        let n = q.len as usize;
        self.speed_sort(&mut q.items[..n], cmp_action, "queue sort tie");
        self.queue = q;
    }

    /// `BattleQueue#willAct`.
    pub(crate) fn will_act(&self) -> bool {
        self.queue.as_slice().iter().any(|a| matches!(a.kind, ActKind::Move | ActKind::Switch | ActKind::InstaSwitch))
    }

    /// `BattleQueue#willMove`.
    pub(crate) fn will_move(&self, r: MonRef) -> bool {
        if self.mon(r).fainted {
            return false;
        }
        self.queue.as_slice().iter().any(|a| a.kind == ActKind::Move && a.mon == Some(r))
    }

    // ------------------------------------------------------------- switching

    /// `Pokemon#clearVolatile` (the Champions version).
    pub(crate) fn clear_volatile(&mut self, r: MonRef, include_switch_flags: bool) {
        let m = self.mon_mut(r);
        m.boosts = [0; 7];
        m.ability = m.base_ability;
        m.volatiles.clear();
        if include_switch_flags {
            m.switch_flag = false;
        }
        // `setSpecies` restores the species' types and resets the cached speed to the raw stat.
        m.types = SPECIES[m.species as usize].types;
        m.speed = m.stats[5] as i32;
    }

    /// `BattleActions#switchIn`.
    pub(crate) fn switch_in(&mut self, incoming: MonRef, pos: usize, at_start: bool) -> bool {
        if self.mon(incoming).is_active {
            return false;
        }
        let side = incoming.side as usize;
        if !at_start {
            let old = self.active(side, pos);
            if self.mon(old).hp > 0 {
                self.run_event(Ev::BeforeSwitchOut, Some(old), None, Eff::None, Res::Undef);
                self.each_event(Ev::Update);
                if !self.run_event(Ev::SwitchOut, Some(old), None, Eff::None, Res::Undef).truthy() {
                    return false;
                }
                if self.mon(old).hp == 0 {
                    // Fainted before it could leave.
                    return false;
                }
                let (ability, item) = (self.mon(old).ability, self.mon(old).item);
                self.single_event(Ev::End, Eff::Ability(ability), Some(old), Some(old), None, Eff::None, Res::Undef);
                self.single_event(Ev::End, Eff::Item(item), Some(old), Some(old), None, Eff::None, Res::Undef);
                self.queue.cancel(old);
                self.clear_volatile(old, true);
            }
            let old_new_pos = self.mon(incoming).position;
            {
                let o = self.mon_mut(old);
                o.is_active = false;
                o.used_item_this_turn = false;
                o.position = old_new_pos;
                if o.fainted {
                    o.status = Status::None;
                }
            }
            self.mon_mut(incoming).position = pos as u8;
            self.sides[side].order[pos] = incoming.idx;
            self.sides[side].order[old_new_pos as usize] = old.idx;
        }
        {
            let m = self.mon_mut(incoming);
            m.is_active = true;
            m.active_turns = 0;
        }
        let st = self.new_state(true, incoming);
        self.mon_mut(incoming).ability_st = st;
        let has_item = self.mon(incoming).item != it::NONE;
        let st = self.new_state(has_item, incoming);
        self.mon_mut(incoming).item_st = st;
        self.run_event(Ev::BeforeSwitchIn, Some(incoming), None, Eff::None, Res::Undef);
        // `queue.insertChoice({choice: 'runSwitch', pokemon})`: the speed is
        // computed once to cache it and once more for the action.
        let s = self.action_speed(incoming);
        self.mon_mut(incoming).speed = s;
        let mut a = Battle::blank_action(ActKind::RunSwitch, 101);
        a.mon = Some(incoming);
        a.speed = self.action_speed(incoming);
        self.insert_choice(a);
        true
    }

    /// `BattleActions#runSwitch`.
    pub(crate) fn run_switch(&mut self, first: MonRef) {
        let mut switchers = [first; 4];
        let mut n = 1;
        while let Some(a) = self.queue.peek() {
            if a.kind != ActKind::RunSwitch {
                break;
            }
            switchers[n] = self.queue.shift().unwrap().mon.unwrap();
            n += 1;
        }
        // Showdown sorts everyone on the field (fainted included) by cached
        // speed here; switch-in handlers then use that fixed order.
        let (actives, k) = self.all_active(true);
        let mut keyed = [(actives[0], 0i32); 4];
        for i in 0..k {
            keyed[i] = (actives[i], self.mon(actives[i]).speed);
        }
        self.speed_sort(&mut keyed[..k], |a, b| b.1 as i64 - a.1 as i64, "switch-in speed tie");
        for i in 0..k {
            let r = keyed[i].0;
            self.speed_order[i] = r.side + 2 * self.mon(r).position;
        }
        self.n_speed_order = k as u8;
        self.field_event(Ev::SwitchIn, Some(&switchers[..n]));
    }

    // -------------------------------------------------------- damage, healing

    /// `Pokemon#faint`: queue the faint; it resolves in `faint_messages`.
    pub(crate) fn faint(&mut self, r: MonRef, source: Option<MonRef>, effect: Eff) -> u32 {
        let m = self.mon_mut(r);
        if m.fainted || m.faint_queued {
            return 0;
        }
        let d = m.hp as u32;
        m.hp = 0;
        m.switch_flag = false;
        m.faint_queued = true;
        self.faint_queue[self.n_faint as usize] = FaintEntry { target: r, source, effect };
        self.n_faint += 1;
        d
    }

    /// `Pokemon#damage`: returns the HP actually lost.
    pub(crate) fn damage_mon(&mut self, r: MonRef, d: i32, source: Option<MonRef>, effect: Eff) -> u32 {
        let hp = self.mon(r).hp as i32;
        if hp == 0 || d <= 0 {
            return 0;
        }
        if d >= hp {
            self.faint(r, source, effect);
            hp as u32
        } else {
            self.mon_mut(r).hp = (hp - d) as u16;
            d as u32
        }
    }

    /// The defaults Showdown fills in from the running event when a callback
    /// leaves `target`, `source` or `effect` out.
    fn event_defaults(
        &self,
        target: Option<MonRef>,
        source: Option<MonRef>,
        effect: Eff,
    ) -> (Option<MonRef>, Option<MonRef>, Eff) {
        (
            target.or(self.event.target),
            source.or(self.event.source),
            if effect == Eff::None { self.effect } else { effect },
        )
    }

    /// `Battle#damage`: one target, with the `Damage` event. Fractions must be
    /// floored by the caller (see `div1`).
    pub(crate) fn damage(&mut self, d: i32, target: Option<MonRef>, source: Option<MonRef>, effect: Eff) -> Res {
        let (target, source, effect) = self.event_defaults(target, source, effect);
        let mut dmg: Damage = [Res::Num(d), Res::Undef, Res::Undef];
        let tgts: Targets = [target.map_or(Tgt::Gone, Tgt::Mon), Tgt::Gone, Tgt::Gone];
        self.spread_damage(&mut dmg, &tgts, 1, source, effect);
        dmg[0]
    }

    /// `Battle#spreadDamage`.
    pub(crate) fn spread_damage(
        &mut self,
        damage: &mut Damage,
        targets: &Targets,
        n: usize,
        source: Option<MonRef>,
        effect: Eff,
    ) {
        for i in 0..n {
            let cur = damage[i];
            let Tgt::Mon(target) = targets[i] else {
                if cur.truthy() || cur == Res::Num(0) {
                    damage[i] = Res::Num(0);
                }
                continue;
            };
            let Res::Num(mut amount) = cur else {
                continue;
            };
            if self.mon(target).hp == 0 {
                damage[i] = Res::Num(0);
                continue;
            }
            if !self.mon(target).is_active {
                damage[i] = FALSE;
                continue;
            }
            if amount != 0 {
                amount = amount.max(1);
            }
            if effect != Eff::StruggleRecoil {
                let e = Event::new(Ev::Damage, Some(target), source, effect);
                let r = if self.event_mask & Ev::Damage.bit() != 0 || effect.is_move() {
                    self.run_event_ex(e, Res::Num(amount), true, false).0
                } else {
                    Res::Num(amount)
                };
                match r {
                    Res::Num(v) => amount = v,
                    other => {
                        damage[i] = other;
                        continue;
                    }
                }
            }
            if amount != 0 {
                amount = amount.max(1);
            }
            let dealt = self.damage_mon(target, amount, source, effect) as i32;
            damage[i] = Res::Num(dealt);
            if dealt != 0 {
                if let Eff::Move(mi) = effect {
                    let drain = self.am[mi as usize].d().drain;
                    if drain.0 > 0 {
                        if let Some(s) = source {
                            let amount = round_div(dealt as u32, drain.0 as u32, drain.1 as u32);
                            self.heal(amount as i32, Some(s), Some(target), Eff::Drain);
                        }
                    }
                }
            }
        }
    }

    /// `Battle#directDamage`: no `Damage` event (Struggle recoil).
    pub(crate) fn direct_damage(&mut self, d: i32, target: MonRef, source: Option<MonRef>, effect: Eff) -> u32 {
        if self.mon(target).hp == 0 || d == 0 {
            return 0;
        }
        self.damage_mon(target, d.max(1), source, effect)
    }

    /// `Battle#heal`. Fractions must already be truncated, with anything in
    /// (0, 1] raised to 1 (see `div1`).
    pub(crate) fn heal(&mut self, amount: i32, target: Option<MonRef>, source: Option<MonRef>, effect: Eff) -> Res {
        let (target, source, effect) = self.event_defaults(target, source, effect);
        let r = self.run_event(Ev::TryHeal, target, source, effect, Res::Num(amount));
        if !r.truthy() {
            return r;
        }
        let amount = r.num();
        let Some(t) = target else {
            return FALSE;
        };
        let m = self.mon_mut(t);
        if m.hp == 0 || !m.is_active || m.hp >= m.max_hp() || amount <= 0 {
            return FALSE;
        }
        let new = (m.hp as i32 + amount).min(m.max_hp() as i32);
        let healed = new - m.hp as i32;
        m.hp = new as u16;
        self.run_event(Ev::Heal, target, source, effect, Res::Num(healed));
        Res::Num(healed)
    }

    /// `Pokemon#heal` used directly (no events): returns the HP restored.
    pub(crate) fn heal_mon(&mut self, r: MonRef, d: i32) -> i32 {
        let m = self.mon_mut(r);
        if m.hp == 0 || d <= 0 || m.hp >= m.max_hp() {
            return 0;
        }
        let new = (m.hp as i32 + d).min(m.max_hp() as i32);
        let healed = new - m.hp as i32;
        m.hp = new as u16;
        healed
    }

    /// `Battle#faintMessages`.
    pub(crate) fn faint_messages(&mut self, last_first: bool, force_check: bool, mut check_win: bool) -> bool {
        if self.ended {
            return false;
        }
        let length = self.n_faint as usize;
        if length == 0 {
            return force_check && self.check_win(None);
        }
        if last_first {
            self.faint_queue[..length].rotate_right(1);
        }
        let mut last = None;
        // The queue can grow while it is being drained.
        while self.n_faint > 0 {
            let left = self.n_faint as usize;
            let fd = self.faint_queue[0];
            self.faint_queue.copy_within(1..left, 0);
            self.n_faint -= 1;
            last = Some(fd);
            let r = fd.target;
            if !self.mon(r).fainted
                && self.run_event(Ev::BeforeFaint, Some(r), fd.source, fd.effect, Res::Undef).truthy()
            {
                let side = &mut self.sides[r.side as usize];
                if side.pokemon_left > 0 {
                    side.pokemon_left -= 1;
                }
                if side.total_fainted < 100 {
                    side.total_fainted += 1;
                }
                self.run_event(Ev::Faint, Some(r), fd.source, fd.effect, Res::Undef);
                let (ability, item) = (self.mon(r).ability, self.mon(r).item);
                self.single_event(Ev::End, Eff::Ability(ability), Some(r), Some(r), None, Eff::None, Res::Undef);
                self.single_event(Ev::End, Eff::Item(item), Some(r), Some(r), None, Eff::None, Res::Undef);
                self.clear_volatile(r, false);
                let m = self.mon_mut(r);
                m.fainted = true;
                m.is_active = false;
                if self.n_faint as usize >= left {
                    check_win = true;
                }
            }
        }
        if check_win && self.check_win(last.map(|fd| fd.target)) {
            return true;
        }
        if let Some(fd) = last {
            self.run_event(Ev::AfterFaint, Some(fd.target), fd.source, fd.effect, Res::Num(length as i32));
        }
        false
    }

    /// `Battle#checkWin`. If both sides run out together, the side whose
    /// Pokémon fainted last wins.
    pub(crate) fn check_win(&mut self, last_fainted: Option<MonRef>) -> bool {
        if self.ended {
            return true;
        }
        let left = [self.sides[0].pokemon_left, self.sides[1].pokemon_left];
        if left[0] == 0 && left[1] == 0 {
            self.win(last_fainted.map(|r| r.side));
            return true;
        }
        for side in 0..2 {
            if left[1 - side] == 0 {
                self.win(Some(side as u8));
                return true;
            }
        }
        false
    }

    pub(crate) fn win(&mut self, side: Option<u8>) {
        if self.ended {
            return;
        }
        self.winner = side;
        self.ended = true;
        self.request = Request::None;
    }

    /// `Battle#checkFainted`.
    fn check_fainted(&mut self) {
        for side in 0..2 {
            for pos in 0..ACTIVE {
                let r = self.active(side, pos);
                let m = self.mon_mut(r);
                if m.fainted {
                    // Showdown parks the status at 'fnt' here, which has no handlers.
                    m.status = Status::None;
                    m.switch_flag = true;
                }
            }
        }
    }

    // ------------------------------------------------- status and volatiles

    /// `Pokemon#setStatus` for a non-empty status.
    pub(crate) fn set_status(&mut self, r: MonRef, status: Status, source: Option<MonRef>, source_effect: Eff) -> Res {
        debug_assert!(status != Status::None);
        if self.mon(r).hp == 0 || !self.mon(r).is_active {
            return FALSE;
        }
        let source = source.or(self.event.source).or(Some(r));
        let source_effect = if source_effect == Eff::None { self.effect } else { source_effect };
        if self.mon(r).status == status {
            return FALSE;
        }
        let corrosive =
            source.is_some_and(|s| self.has_ability(s, ab::CORROSION)) && matches!(status, Status::Tox | Status::Psn);
        if !corrosive && !self.run_status_immunity(r, Imm::Status(status)) {
            return FALSE;
        }
        let prev = (self.mon(r).status, self.mon(r).status_time, self.mon(r).tox_stage, self.mon(r).status_st);
        let mut e = Event::new(Ev::SetStatus, Some(r), source, source_effect);
        e.status = status;
        let result = if self.event_mask & Ev::SetStatus.bit() != 0 {
            self.run_event_ex(e, Res::Undef, false, false).0
        } else {
            TRUE
        };
        if !result.truthy() {
            return result;
        }
        let st = self.new_state(true, r);
        {
            let m = self.mon_mut(r);
            m.status = status;
            m.status_time = 0;
            m.tox_stage = 0;
            m.status_st = st;
        }
        if !self
            .single_event(Ev::Start, Eff::Status(status), Some(r), Some(r), source, source_effect, Res::Undef)
            .truthy()
        {
            let m = self.mon_mut(r);
            (m.status, m.status_time, m.tox_stage, m.status_st) = prev;
            return FALSE;
        }
        let mut e = Event::new(Ev::AfterSetStatus, Some(r), source, source_effect);
        e.status = status;
        if self.event_mask & Ev::AfterSetStatus.bit() != 0 && !self.run_event_ex(e, Res::Undef, false, false).0.truthy()
        {
            return FALSE;
        }
        TRUE
    }

    /// `Pokemon#trySetStatus`: fails if there is already a status.
    pub(crate) fn try_set_status(
        &mut self,
        r: MonRef,
        status: Status,
        source: Option<MonRef>,
        source_effect: Eff,
    ) -> Res {
        // Showdown calls `setStatus(this.status || status)`, which fails on an existing status.
        if self.mon(r).status != Status::None {
            if self.mon(r).hp == 0 || !self.mon(r).is_active {
                return FALSE;
            }
            return FALSE;
        }
        self.set_status(r, status, source, source_effect)
    }

    /// `Pokemon#cureStatus` / `clearStatus`.
    pub(crate) fn cure_status(&mut self, r: MonRef) -> bool {
        let m = self.mon_mut(r);
        if m.hp == 0 || m.status == Status::None {
            return false;
        }
        m.status = Status::None;
        m.status_time = 0;
        m.tox_stage = 0;
        let st = self.new_state(false, r);
        self.mon_mut(r).status_st = st;
        true
    }

    /// `Pokemon#addVolatile`.
    pub(crate) fn add_volatile(&mut self, r: MonRef, kind: VolKind, source: Option<MonRef>, source_effect: Eff) -> Res {
        if self.mon(r).hp == 0 {
            return FALSE;
        }
        let source = source.or(self.event.source).or(Some(r));
        let source_effect = if source_effect == Eff::None { self.effect } else { source_effect };
        if self.mon(r).volatiles.has(kind) {
            if !self.has_cb(Eff::Vol(kind), Ev::Restart) {
                return FALSE;
            }
            return self.single_event(Ev::Restart, Eff::Vol(kind), Some(r), Some(r), source, source_effect, Res::Undef);
        }
        if !self.run_status_immunity(r, Imm::Vol(kind)) {
            return FALSE;
        }
        if self.event_mask & Ev::TryAddVolatile.bit() != 0 {
            let mut e = Event::new(Ev::TryAddVolatile, Some(r), source, source_effect);
            e.vol = Some(kind);
            let result = self.run_event_ex(e, Res::Undef, false, false).0;
            if !result.truthy() {
                return result;
            }
        }
        let st = self.new_state(true, r);
        let v = Volatile { kind, duration: VOL_CONDS[kind as usize].duration, data: 0, st };
        self.mon_mut(r).volatiles.push(v);
        let result = self.single_event(Ev::Start, Eff::Vol(kind), Some(r), Some(r), source, source_effect, Res::Undef);
        if !result.truthy() {
            self.mon_mut(r).volatiles.remove(kind);
            return result;
        }
        TRUE
    }

    /// `Pokemon#removeVolatile`.
    pub(crate) fn remove_volatile(&mut self, r: MonRef, kind: VolKind) -> bool {
        if self.mon(r).hp == 0 || !self.mon(r).volatiles.has(kind) {
            return false;
        }
        self.single_event(Ev::End, Eff::Vol(kind), Some(r), Some(r), None, Eff::None, Res::Undef);
        self.mon_mut(r).volatiles.remove(kind);
        true
    }

    // ------------------------------------------------------------ stat stages

    /// `Battle#boost`. `order` indexes `BOOST_ORDERS`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn boost(
        &mut self,
        b: Boosts,
        order: u8,
        target: Option<MonRef>,
        source: Option<MonRef>,
        effect: Eff,
    ) -> Res {
        let (target, source, effect) = self.event_defaults(target, source, effect);
        let Some(r) = target else {
            return Res::Num(0);
        };
        if self.mon(r).hp == 0 {
            return Res::Num(0);
        }
        if !self.mon(r).is_active {
            return FALSE;
        }
        if self.sides[1 - r.side as usize].pokemon_left == 0 {
            return FALSE;
        }
        let mut b = b;
        if self.event_mask & Ev::ChangeBoost.bit() != 0 {
            let mut e = Event::new(Ev::ChangeBoost, Some(r), source, effect);
            e.boosts = b;
            b = self.run_event_ex(e, Res::Undef, false, false).1.boosts;
        }
        // getCappedBoost
        for k in 0..7 {
            let cur = self.mon(r).boosts[k];
            b[k] = (cur + b[k]).clamp(-6, 6) - cur;
        }
        if self.event_mask & Ev::TryBoost.bit() != 0 {
            let mut e = Event::new(Ev::TryBoost, Some(r), source, effect);
            e.boosts = b;
            b = self.run_event_ex(e, Res::Undef, false, false).1.boosts;
        }
        let mut success = false;
        for &k in &BOOST_ORDERS[order as usize] {
            let k = k as usize;
            if b[k] == 0 {
                continue;
            }
            let cur = self.mon(r).boosts[k];
            let new = (cur + b[k]).clamp(-6, 6);
            if new != cur {
                self.mon_mut(r).boosts[k] = new;
                success = true;
                if self.event_mask & Ev::AfterEachBoost.bit() != 0 {
                    let mut e = Event::new(Ev::AfterEachBoost, Some(r), source, effect);
                    e.boosts[k] = b[k];
                    self.run_event_ex(e, Res::Undef, false, false);
                }
            }
        }
        if self.event_mask & Ev::AfterBoost.bit() != 0 {
            let mut e = Event::new(Ev::AfterBoost, Some(r), source, effect);
            e.boosts = b;
            self.run_event_ex(e, Res::Undef, false, false);
        }
        if success { TRUE } else { Res::Null }
    }

    // --------------------------------------------------- abilities and items

    /// `Pokemon#ignoringAbility`.
    pub(crate) fn ignoring_ability(&self, r: MonRef) -> bool {
        let m = self.mon(r);
        if !m.is_active {
            return true;
        }
        // Gastro Acid and Neutralizing Gas are not modelled.
        false
    }

    /// `Pokemon#hasAbility`.
    pub(crate) fn has_ability(&self, r: MonRef, ability: u16) -> bool {
        self.mon(r).ability == ability && !self.ignoring_ability(r)
    }

    /// `Pokemon#ignoringItem`.
    pub(crate) fn ignoring_item(&self, r: MonRef) -> bool {
        let m = self.mon(r);
        if !m.is_active {
            return true;
        }
        ITEMS[m.item as usize].flags & IF_IGNORE_KLUTZ == 0 && self.has_ability(r, ab::KLUTZ)
    }

    /// `Pokemon#hasItem`.
    pub(crate) fn has_item(&self, r: MonRef, item: u16) -> bool {
        self.mon(r).item == item && !self.ignoring_item(r)
    }

    /// `Pokemon#eatItem`.
    pub(crate) fn eat_item(&mut self, r: MonRef, force: bool, source: Option<MonRef>, source_effect: Eff) -> bool {
        let item = self.mon(r).item;
        if item == it::NONE {
            return false;
        }
        if self.mon(r).hp == 0 || !self.mon(r).is_active {
            return false;
        }
        let source_effect = if source_effect == Eff::None { self.effect } else { source_effect };
        let source = source.or(self.event.target);
        if let Eff::Item(i) = source_effect {
            if i != item && source == Some(r) {
                // An item is telling us to eat it but we no longer hold it.
                return false;
            }
        }
        let mut e = Event::new(Ev::UseItem, Some(r), None, Eff::None);
        e.item = item;
        if !self.run_event_ex(e, Res::Undef, false, false).0.truthy() {
            return false;
        }
        if !force {
            let mut e = Event::new(Ev::TryEatItem, Some(r), None, Eff::None);
            e.item = item;
            if !self.run_event_ex(e, Res::Undef, false, false).0.truthy() {
                return false;
            }
        }
        self.single_event(Ev::Eat, Eff::Item(item), Some(r), Some(r), source, source_effect, Res::Undef);
        let mut e = Event::new(Ev::EatItem, Some(r), source, source_effect);
        e.item = item;
        self.run_event_ex(e, Res::Undef, false, false);
        {
            let m = self.mon_mut(r);
            m.last_item = item;
            m.item = it::NONE;
            // clearEffectState keeps the state object but blanks it.
            m.item_st.order = 0;
            m.item_st.a = 0;
            m.item_st.b = 0;
            m.used_item_this_turn = true;
            m.ate_berry = true;
        }
        let mut e = Event::new(Ev::AfterUseItem, Some(r), None, Eff::None);
        e.item = item;
        self.run_event_ex(e, Res::Undef, false, false);
        true
    }

    /// `Pokemon#useItem`.
    pub(crate) fn use_item(&mut self, r: MonRef, source: Option<MonRef>, source_effect: Eff) -> bool {
        let item = self.mon(r).item;
        let is_gem = ITEMS[item as usize].flags & IF_GEM != 0;
        if (self.mon(r).hp == 0 && !is_gem) || !self.mon(r).is_active {
            return false;
        }
        if item == it::NONE {
            return false;
        }
        let source_effect = if source_effect == Eff::None { self.effect } else { source_effect };
        let source = source.or(self.event.target);
        if let Eff::Item(i) = source_effect {
            if i != item && source == Some(r) {
                return false;
            }
        }
        let mut e = Event::new(Ev::UseItem, Some(r), None, Eff::None);
        e.item = item;
        if !self.run_event_ex(e, Res::Undef, false, false).0.truthy() {
            return false;
        }
        self.single_event(Ev::Use, Eff::Item(item), Some(r), Some(r), source, source_effect, Res::Undef);
        {
            let m = self.mon_mut(r);
            m.last_item = item;
            m.item = it::NONE;
            m.item_st.order = 0;
            m.item_st.a = 0;
            m.item_st.b = 0;
            m.used_item_this_turn = true;
        }
        let mut e = Event::new(Ev::AfterUseItem, Some(r), None, Eff::None);
        e.item = item;
        self.run_event_ex(e, Res::Undef, false, false);
        true
    }

    /// `Pokemon#takeItem`: returns the item taken (0 if none or blocked).
    pub(crate) fn take_item(&mut self, r: MonRef, source: Option<MonRef>) -> u16 {
        let source = source.or(Some(r));
        let item = self.mon(r).item;
        if item == it::NONE {
            return it::NONE;
        }
        let mut e = Event::new(Ev::TakeItem, Some(r), source, Eff::None);
        e.item = item;
        if !self.run_event_ex(e, Res::Undef, false, false).0.truthy() {
            return it::NONE;
        }
        {
            let m = self.mon_mut(r);
            m.item = it::NONE;
            m.item_st.order = 0;
            m.item_st.a = 0;
            m.item_st.b = 0;
        }
        self.single_event(Ev::End, Eff::Item(item), Some(r), Some(r), None, Eff::None, Res::Undef);
        item
    }

    /// `Pokemon#setItem`.
    pub(crate) fn set_item(&mut self, r: MonRef, item: u16, source: Option<MonRef>, effect: Eff) -> bool {
        if self.mon(r).hp == 0 || !self.mon(r).is_active {
            return false;
        }
        let old = self.mon(r).item;
        let st = self.new_state(item != it::NONE, r);
        {
            let m = self.mon_mut(r);
            m.item = item;
            m.item_st = st;
        }
        if old != it::NONE {
            self.single_event(Ev::End, Eff::Item(old), Some(r), Some(r), None, Eff::None, Res::Undef);
        }
        if item != it::NONE {
            self.single_event(Ev::Start, Eff::Item(item), Some(r), Some(r), source, effect, Res::Undef);
        }
        true
    }

    /// `Pokemon#setAbility`: returns whether the ability was changed.
    pub(crate) fn set_ability(&mut self, r: MonRef, ability: u16, source: Option<MonRef>, source_effect: Eff) -> bool {
        if self.mon(r).hp == 0 {
            return false;
        }
        let source_effect = if source_effect == Eff::None { self.effect } else { source_effect };
        let old = self.mon(r).ability;
        if (ABILITIES[ability as usize].flags | ABILITIES[old as usize].flags) & AF_CANTSUPPRESS != 0 {
            return false;
        }
        if !self.run_event(Ev::SetAbility, Some(r), source, source_effect, Res::Undef).truthy() {
            return false;
        }
        self.single_event(Ev::End, Eff::Ability(old), Some(r), Some(r), source, Eff::None, Res::Undef);
        let st = self.new_state(true, r);
        {
            let m = self.mon_mut(r);
            m.ability = ability;
            m.ability_st = st;
        }
        self.single_event(Ev::Start, Eff::Ability(ability), Some(r), Some(r), source, Eff::None, Res::Undef);
        true
    }

    // -------------------------------------------------------------- the turn

    /// `Battle#runAction`.
    fn run_action(&mut self, a: Action) {
        match a.kind {
            ActKind::Team => return,
            ActKind::Start => {
                for side in 0..2 {
                    for pos in 0..ACTIVE {
                        let r = self.active(side, pos);
                        self.switch_in(r, pos, true);
                    }
                }
            }
            ActKind::Move => {
                let r = a.mon.unwrap();
                if !self.mon(r).is_active || self.mon(r).fainted {
                    return;
                }
                trace::note(|| format!("move {} by p{}:{}", MOVES[a.move_id as usize].id, r.side + 1, r.idx));
                self.run_move(&a);
            }
            ActKind::Switch | ActKind::InstaSwitch => {
                let out = a.mon.unwrap();
                let pos = self.mon(out).position as usize;
                self.switch_in(a.switch_to.unwrap(), pos, false);
            }
            ActKind::RunSwitch => self.run_switch(a.mon.unwrap()),
            ActKind::BeforeTurn => self.each_event(Ev::BeforeTurn),
            ActKind::Residual => {
                self.clear_active_move(true);
                self.update_speed();
                self.field_event(Ev::Residual, None);
            }
        }

        self.clear_active_move(false);
        self.faint_messages(false, false, true);
        if self.ended {
            return;
        }

        let next = self.queue.peek().map(|x| x.kind);
        match next {
            None => self.check_fainted(),
            Some(ActKind::InstaSwitch) => return,
            _ => {}
        }
        if a.kind != ActKind::Start {
            self.each_event(Ev::Update);
        }

        let mut any = false;
        for side in 0..2 {
            let wants = (0..ACTIVE).any(|pos| self.mon(self.active(side, pos)).switch_flag);
            if wants && !self.can_switch(side) {
                for pos in 0..ACTIVE {
                    let r = self.active(side, pos);
                    self.mon_mut(r).switch_flag = false;
                }
            } else if wants {
                any = true;
            }
        }
        if any {
            self.request = Request::Switch;
            return;
        }

        if next == Some(ActKind::Move) {
            // Speed is dynamic since Gen 8: recompute and re-sort what is left.
            self.update_speed();
            let mut q = self.queue;
            for act in q.as_mut_slice() {
                if act.mon.is_some() {
                    self.resolve_speed(act);
                }
            }
            self.queue = q;
            self.sort_queue();
        }
    }

    /// `Battle#turnLoop`.
    pub(crate) fn turn_loop(&mut self) {
        self.request = Request::None;
        if !self.mid_turn {
            self.insert_choice(Battle::blank_action(ActKind::BeforeTurn, 4));
            self.queue.push(Battle::blank_action(ActKind::Residual, 300));
            self.mid_turn = true;
        }
        while let Some(a) = self.queue.shift() {
            // Moves of earlier actions are finished; their slots can be reused.
            self.am_len = 0;
            self.run_action(a);
            if self.request != Request::None || self.ended {
                return;
            }
        }
        self.end_turn();
        self.mid_turn = false;
        self.queue.clear();
    }

    /// `Battle#endTurn`.
    fn end_turn(&mut self) {
        self.turn += 1;
        for side in 0..2 {
            for pos in 0..ACTIVE {
                let r = self.active(side, pos);
                if !self.in_play(r) {
                    continue;
                }
                {
                    let turn = self.turn;
                    let m = self.mon_mut(r);
                    if turn != 1 {
                        m.used_item_this_turn = false;
                    }
                    for k in 0..m.n_moves as usize {
                        m.moves[k].disabled = false;
                    }
                }
                self.run_event(Ev::DisableMove, Some(r), None, Eff::None, Res::Undef);
                // (Moves that disable themselves are not modelled.)
                self.mon_mut(r).trapped = Trapped::No;
                self.run_event(Ev::TrapPokemon, Some(r), None, Eff::None, Res::Undef);
                if self.type_allows(r, 6) {
                    self.run_event(Ev::MaybeTrapPokemon, Some(r), None, Eff::None, Res::Undef);
                }
                if self.mon(r).fainted {
                    continue;
                }
                self.mon_mut(r).active_turns += 1;
            }
        }
        if self.turn > 1000 {
            // Showdown's hard turn limit: the battle is a tie.
            self.win(None);
            return;
        }
        self.request = Request::Move;
    }

    // ------------------------------------------------------- active move slots

    /// `dex.getActiveMove`: a fresh mutable copy of a move in the next slot.
    pub(crate) fn new_am(&mut self, move_id: u16) -> u8 {
        let mi = self.am_len;
        assert!((mi as usize) < AM_CAP, "too many nested moves");
        self.am[mi as usize] = ActiveMove::new(move_id);
        self.am_len += 1;
        mi
    }

    /// `Battle#setActiveMove`.
    pub(crate) fn set_active_move(&mut self, mi: Option<u8>, pokemon: Option<MonRef>, target: Option<MonRef>) {
        self.active_move = mi;
        self.active_pokemon = pokemon;
        self.active_target = target.or(pokemon);
    }

    /// `Battle#clearActiveMove`.
    pub(crate) fn clear_active_move(&mut self, _failed: bool) {
        self.active_move = None;
        self.active_pokemon = None;
        self.active_target = None;
    }
}

/// A fraction of something as Showdown's damage and heal functions end up
/// using it: floored, but at least 1 when the numerator is positive.
pub(crate) fn div1(num: u32, den: u32) -> i32 {
    if num == 0 { 0 } else { (num / den).max(1) as i32 }
}
