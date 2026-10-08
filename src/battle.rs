//! The turn loop. This is a port of the parts of Pokémon Showdown's simulator
//! (`sim/battle.ts`, `sim/battle-actions.ts`, `sim/battle-queue.ts`) that the
//! supported mechanics exercise, under the Champions mod.
//!
//! The port is deliberately literal: it draws random numbers in the same
//! order Showdown does, including the draws Showdown makes for reasons that
//! have no in-game meaning (re-resolving targets when re-sorting the queue,
//! shuffling speed ties in every `eachEvent`). That is what makes it possible
//! to replay a Showdown battle here from the same seed and compare state
//! after every decision. Comments name the Showdown function being mirrored.

// The control flow follows Showdown's on purpose, so keep its shape even where
// clippy would fold branches or loops together.
#![allow(clippy::needless_range_loop, clippy::collapsible_if, clippy::collapsible_match, clippy::if_same_then_else)]

use crate::data::*;
use crate::rng::Rng;
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
}

impl PokemonSet {
    /// Builds a set from Showdown names or ids, e.g.
    /// `PokemonSet::from_names("Garchomp", &["Earthquake", "Protect"], "Jolly", [0, 32, 0, 0, 2, 32])`.
    /// Stat points are hp, atk, def, spa, spd, spe.
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
        Ok(PokemonSet { species, moves, nature, stat_points })
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

/// Showdown callbacks return numbers, booleans, `undefined`, `null` or `''`
/// (`NOT_FAIL`), and the move pipeline branches on which one it got. This is
/// that value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Res {
    Undef,
    NotFail,
    Null,
    Bool(bool),
    Num(i32),
}

impl Res {
    fn truthy(self) -> bool {
        match self {
            Res::Bool(b) => b,
            Res::Num(n) => n != 0,
            _ => false,
        }
    }
    fn rank(self) -> u8 {
        match self {
            Res::Undef => 0,
            Res::NotFail => 1,
            Res::Null => 2,
            Res::Bool(_) => 3,
            Res::Num(_) => 4,
        }
    }
    /// "Counts as a hit": truthy, or exactly 0 damage.
    fn hit(self) -> bool {
        self.truthy() || self == Res::Num(0)
    }
}

/// `BattleActions#combineResults`.
fn combine(left: Res, right: Res) -> Res {
    if left.rank() > right.rank() {
        left
    } else if left.truthy() && !right.truthy() && right != Res::Num(0) {
        left
    } else if let (Res::Num(a), Res::Num(b)) = (left, right) {
        Res::Num(a + b)
    } else {
        right
    }
}

/// An entry of Showdown's `targets` arrays, which are overwritten with
/// `false` as targets drop out of a move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tgt {
    Mon(MonRef),
    Gone,
}

const MAX_TARGETS: usize = 3;
type Targets = [Tgt; MAX_TARGETS];
type Damage = [Res; MAX_TARGETS];

/// The per-use copy of a move (Showdown's `ActiveMove`).
struct ActiveMove {
    d: &'static MoveData,
    typ: Type,
    spread_hit: bool,
    self_dropped: bool,
    total_damage: i32,
}

/// What one application of a move does to its target: the move itself, one of
/// its secondaries, or its `self` block (Showdown's `HitEffect`).
#[derive(Clone, Copy)]
struct Effect {
    primary: bool,
    boosts: Option<Boosts>,
    status: Status,
    volatile: Option<VolKind>,
    heal: (u8, u8),
    self_boosts: Option<Boosts>,
    self_chance: u8,
    /// Protect's `onHit`: start or extend the stall counter.
    adds_stall: bool,
}

impl Effect {
    fn of_move(d: &MoveData) -> Effect {
        Effect {
            primary: true,
            boosts: d.boosts,
            status: d.status,
            volatile: if d.special == Special::Protect { Some(VolKind::Protect) } else { None },
            heal: d.heal,
            self_boosts: d.self_boosts,
            self_chance: d.self_chance,
            adds_stall: d.special == Special::Protect,
        }
    }
    fn of_secondary(s: &Secondary) -> Effect {
        Effect {
            primary: false,
            boosts: s.boosts,
            status: s.status,
            volatile: if s.flinch { Some(VolKind::Flinch) } else { None },
            heal: (0, 0),
            self_boosts: s.self_boosts,
            self_chance: 0,
            adds_stall: false,
        }
    }
    fn of_self(b: Boosts) -> Effect {
        Effect {
            primary: false,
            boosts: Some(b),
            status: Status::None,
            volatile: None,
            heal: (0, 0),
            self_boosts: None,
            self_chance: 0,
            adds_stall: false,
        }
    }
}

/// `Battle#modify` with a 4096-based modifier.
fn modify(value: u32, modifier: u32) -> u32 {
    (value * modifier + 2047) / 4096
}

/// Stat after stage changes (`Pokemon#calculateStat` / `getStat`).
fn boosted(stat: u32, stage: i8) -> u32 {
    let s = stage.clamp(-6, 6) as i32;
    if s >= 0 { stat * (2 + s as u32) / 2 } else { stat * 2 / (2 + (-s) as u32) }
}

/// `Math.round(a * num / den)` for non-negative integers.
fn round_div(a: u32, num: u32, den: u32) -> u32 {
    (2 * a * num + den) / (2 * den)
}

fn struggle_id() -> u16 {
    move_id("struggle").expect("struggle is always in the move table")
}

/// `Battle#comparePriority` for queued actions: order low to high, priority
/// high to low, speed high to low. Negative means `a` goes first.
fn cmp_action(a: &Action, b: &Action) -> i64 {
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
    fn mon_mut(&mut self, r: MonRef) -> &mut Pokemon {
        &mut self.sides[r.side as usize].team[r.idx as usize]
    }
    /// The Pokémon occupying an active slot (it may be fainted).
    pub fn active(&self, side: usize, pos: usize) -> MonRef {
        MonRef { side: side as u8, idx: self.sides[side].order[pos] }
    }
    /// `Battle#getAllActive`.
    fn all_active(&self, include_fainted: bool) -> ([MonRef; 4], usize) {
        let mut out = [MonRef { side: 0, idx: 0 }; 4];
        let mut n = 0;
        for side in 0..2 {
            for pos in 0..ACTIVE {
                let r = self.active(side, pos);
                if include_fainted || !self.mon(r).fainted {
                    out[n] = r;
                    n += 1;
                }
            }
        }
        (out, n)
    }
    /// `Side#allies()`: this side's active Pokémon that still have HP.
    fn living_active(&self, side: usize) -> ([MonRef; ACTIVE], usize) {
        let mut out = [MonRef { side: 0, idx: 0 }; ACTIVE];
        let mut n = 0;
        for pos in 0..ACTIVE {
            let r = self.active(side, pos);
            if self.mon(r).hp > 0 {
                out[n] = r;
                n += 1;
            }
        }
        (out, n)
    }
    /// `Pokemon#getLocOf`.
    fn loc_of(&self, from: MonRef, target: MonRef) -> i8 {
        let p = self.mon(target).position as i8 + 1;
        if from.side == target.side { -p } else { p }
    }
    /// `Pokemon#getAtLoc`.
    fn at_loc(&self, from: MonRef, loc: i8) -> Option<MonRef> {
        let side = if loc < 0 { from.side as usize } else { 1 - from.side as usize };
        let pos = loc.unsigned_abs() as usize;
        if pos == 0 || pos > ACTIVE {
            return None;
        }
        Some(self.active(side, pos - 1))
    }
    /// `Battle#canSwitch`.
    fn can_switch(&self, side: usize) -> bool {
        let s = &self.sides[side];
        if s.pokemon_left == 0 {
            return false;
        }
        (ACTIVE..s.n as usize).any(|p| !s.team[s.order[p] as usize].fainted)
    }

    // ------------------------------------------------------------------- rng

    fn rand(&mut self, n: u32, what: &'static str) -> u32 {
        let v = self.rng.below(n);
        trace::rng(what, n, v);
        v
    }
    fn chance(&mut self, num: u32, den: u32, what: &'static str) -> bool {
        self.rand(den, what) < num
    }
    /// `PRNG#random(from, to)`.
    fn rand_range(&mut self, from: usize, to: usize, what: &'static str) -> usize {
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
    fn speed_sort<T: Copy>(&mut self, list: &mut [T], cmp: impl Fn(&T, &T) -> i64, what: &'static str) {
        if list.len() < 2 {
            return;
        }
        let mut sorted = 0;
        let mut next = [0usize; QUEUE_CAP];
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
    /// `Battle#eachEvent` for events nothing listens to yet. Showdown still
    /// sorts the active Pokémon by cached speed first, which draws from the
    /// RNG whenever two of them are tied.
    fn each_event(&mut self, what: &'static str) {
        let (actives, n) = self.all_active(false);
        let mut keyed = [(actives[0], 0i32); 4];
        for i in 0..n {
            keyed[i] = (actives[i], self.mon(actives[i]).speed);
        }
        self.speed_sort(&mut keyed[..n], |a, b| b.1 as i64 - a.1 as i64, what);
    }

    // ----------------------------------------------------------------- speed

    /// `Pokemon#getActionSpeed` under the Champions mod (no Trick Room yet).
    fn action_speed(&self, r: MonRef) -> i32 {
        let m = self.mon(r);
        let mut spe = boosted(m.stats[5] as u32, m.boosts[SPE]);
        // Paralysis is an event handler on the Pokémon, and handlers only run
        // for Pokémon on the field.
        if m.is_active && m.status == Status::Par {
            spe = spe * 50 / 100;
        }
        spe.min(10000) as i32
    }
    /// `Battle#updateSpeed`.
    fn update_speed(&mut self) {
        let (actives, n) = self.all_active(false);
        for &r in &actives[..n] {
            let s = self.action_speed(r);
            self.mon_mut(r).speed = s;
        }
    }

    // ------------------------------------------------------------- targeting

    /// `Battle#validTargetLoc`.
    fn valid_target_loc(&self, loc: i8, source: MonRef, target: Target) -> bool {
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
    fn get_random_target(&mut self, user: MonRef, target: Target) -> Option<MonRef> {
        match target {
            Target::User | Target::All | Target::AllySide | Target::AllyTeam | Target::AdjacentAllyOrSelf => Some(user),
            Target::AdjacentAlly => {
                if self.mon(user).fainted {
                    return None;
                }
                let (allies, n) = self.living_active(user.side as usize);
                let mut adj = [user; ACTIVE];
                let mut k = 0;
                for &a in &allies[..n] {
                    if a != user && !self.mon(a).fainted {
                        adj[k] = a;
                        k += 1;
                    }
                }
                if k == 0 { None } else { Some(adj[self.rand(k as u32, "random ally target") as usize]) }
            }
            _ => {
                let foe = 1 - user.side as usize;
                let (foes, n) = self.living_active(foe);
                if n == 0 {
                    Some(self.active(foe, 0))
                } else {
                    Some(foes[self.rand(n as u32, "random foe target") as usize])
                }
            }
        }
    }

    /// `Battle#getTarget`.
    fn get_target(&mut self, user: MonRef, d: &MoveData, loc: i8) -> Option<MonRef> {
        let self_loc = self.loc_of(user, user);
        if matches!(d.target, Target::AdjacentAlly | Target::Any | Target::Normal) && loc == self_loc {
            return None;
        }
        if d.target != Target::RandomNormal && self.valid_target_loc(loc, user, d.target) {
            if let Some(t) = self.at_loc(user, loc) {
                if self.mon(t).fainted {
                    if t.side == user.side {
                        if d.target == Target::AdjacentAllyOrSelf {
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
        self.get_random_target(user, d.target)
    }

    /// `Pokemon#getMoveTargets`.
    fn get_move_targets(&mut self, user: MonRef, d: &MoveData, target: MonRef) -> ([MonRef; MAX_TARGETS], usize) {
        let mut out = [user; MAX_TARGETS];
        let mut n = 0;
        let own = user.side as usize;
        match d.target {
            Target::AllAdjacent | Target::AllAdjacentFoes => {
                if d.target == Target::AllAdjacent && !self.mon(user).fainted {
                    let (allies, k) = self.living_active(own);
                    for &a in &allies[..k] {
                        if a != user && !self.mon(a).fainted {
                            out[n] = a;
                            n += 1;
                        }
                    }
                }
                let (foes, k) = self.living_active(1 - own);
                for &f in &foes[..k] {
                    out[n] = f;
                    n += 1;
                }
            }
            Target::Allies => {
                let (allies, k) = self.living_active(own);
                for &a in &allies[..k] {
                    out[n] = a;
                    n += 1;
                }
            }
            _ => {
                let mut t = target;
                if self.mon(t).fainted && t.side != user.side {
                    // A foe that fainted earlier this turn: retarget.
                    match self.get_random_target(user, d.target) {
                        Some(p) => t = p,
                        None => return (out, 0),
                    }
                }
                if self.mon(t).fainted {
                    return (out, 0);
                }
                out[0] = t;
                n = 1;
            }
        }
        (out, n)
    }

    // ----------------------------------------------------------------- queue

    /// `Battle#getActionSpeed`. For moves this re-resolves the target purely
    /// for Showdown's `ModifyPriority` hooks; the result is unused here but
    /// the lookup can draw from the RNG, so it has to happen.
    fn resolve_speed(&mut self, a: &mut Action) {
        let Some(r) = a.mon else {
            a.speed = 1;
            return;
        };
        if a.kind == ActKind::Move {
            let d = &MOVES[a.move_id as usize];
            a.priority = d.priority as i32;
            self.get_target(r, d, a.target_loc);
        }
        a.speed = self.action_speed(r);
    }

    fn blank_action(kind: ActKind, order: u32) -> Action {
        Action { kind, order, priority: 0, speed: 1, mon: None, move_id: 0, target_loc: 0, switch_to: None }
    }

    /// `BattleQueue#resolveAction` for a move.
    fn resolve_move(&mut self, user: MonRef, move_id: u16, loc: i8) -> Action {
        let d = &MOVES[move_id as usize];
        let mut a = Battle::blank_action(ActKind::Move, 200);
        a.mon = Some(user);
        a.move_id = move_id;
        a.target_loc = loc;
        if a.target_loc == 0 {
            if let Some(t) = self.get_random_target(user, d.target) {
                a.target_loc = self.loc_of(user, t);
            }
        }
        self.resolve_speed(&mut a);
        a
    }

    /// `BattleQueue#resolveAction` for a switch chosen at a move or switch request.
    fn resolve_switch(&mut self, kind: ActKind, out: MonRef, incoming: MonRef) -> Action {
        let mut a = Battle::blank_action(kind, if kind == ActKind::InstaSwitch { 3 } else { 103 });
        a.mon = Some(out);
        a.switch_to = Some(incoming);
        self.mon_mut(out).switch_flag = false;
        self.resolve_speed(&mut a);
        a
    }

    /// `BattleQueue#insertChoice`: place an action where it would have sorted,
    /// picking a random spot among actions it ties with.
    fn insert_choice(&mut self, a: Action) {
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

    fn sort_queue(&mut self) {
        let mut q = self.queue;
        let n = q.len as usize;
        self.speed_sort(&mut q.items[..n], cmp_action, "queue sort tie");
        self.queue = q;
    }

    /// `BattleQueue#willAct`.
    fn will_act(&self) -> bool {
        self.queue.as_slice().iter().any(|a| matches!(a.kind, ActKind::Move | ActKind::Switch | ActKind::InstaSwitch))
    }

    // ------------------------------------------------------------- switching

    /// `Pokemon#clearVolatile`.
    fn clear_volatile(&mut self, r: MonRef, include_switch_flags: bool) {
        let m = self.mon_mut(r);
        m.boosts = [0; 7];
        m.volatiles.clear();
        if include_switch_flags {
            m.switch_flag = false;
        }
        // `setSpecies` resets the cached speed to the raw stat.
        m.speed = m.stats[5] as i32;
    }

    /// `BattleActions#switchIn`.
    fn switch_in(&mut self, incoming: MonRef, pos: usize, at_start: bool) {
        if self.mon(incoming).is_active {
            return;
        }
        let side = incoming.side as usize;
        if !at_start {
            let old = self.active(side, pos);
            if self.mon(old).hp > 0 {
                // BeforeSwitchOut has no listeners yet; the Update pass after it still runs.
                self.each_event("update tie (before switch out)");
                self.queue.cancel(old);
                self.clear_volatile(old, true);
            }
            let old_new_pos = self.mon(incoming).position;
            {
                let o = self.mon_mut(old);
                o.is_active = false;
                o.position = old_new_pos;
                if o.fainted {
                    o.status = Status::None;
                }
            }
            self.mon_mut(incoming).position = pos as u8;
            self.sides[side].order[pos] = incoming.idx;
            self.sides[side].order[old_new_pos as usize] = old.idx;
        }
        self.mon_mut(incoming).is_active = true;
        // `queue.insertChoice({choice: 'runSwitch', pokemon})`
        let s = self.action_speed(incoming);
        self.mon_mut(incoming).speed = s;
        let mut a = Battle::blank_action(ActKind::RunSwitch, 101);
        a.mon = Some(incoming);
        a.speed = s;
        self.insert_choice(a);
    }

    /// `BattleActions#runSwitch`.
    fn run_switch(&mut self, first: MonRef) {
        let mut switchers = [first; 4];
        let mut n = 1;
        while let Some(a) = self.queue.peek() {
            if a.kind != ActKind::RunSwitch {
                break;
            }
            switchers[n] = self.queue.shift().unwrap().mon.unwrap();
            n += 1;
        }
        // Showdown sorts everyone on the field (fainted included) by cached speed here.
        let (actives, k) = self.all_active(true);
        let mut keyed = [(actives[0], 0i32); 4];
        for i in 0..k {
            keyed[i] = (actives[i], self.mon(actives[i]).speed);
        }
        self.speed_sort(&mut keyed[..k], |a, b| b.1 as i64 - a.1 as i64, "switch-in speed tie");
        // fieldEvent('SwitchIn'): the only listener so far is toxic resetting its counter.
        for &r in &switchers[..n] {
            let m = self.mon_mut(r);
            if !m.fainted && m.status == Status::Tox {
                m.tox_stage = 0;
            }
        }
    }

    // ---------------------------------------------------------------- damage

    /// `Pokemon#faint`: queue the faint; it resolves in `faint_messages`.
    fn faint(&mut self, r: MonRef) {
        let m = self.mon_mut(r);
        if m.fainted || m.faint_queued {
            return;
        }
        m.hp = 0;
        m.switch_flag = false;
        m.faint_queued = true;
        self.faint_queue[self.n_faint as usize] = r;
        self.n_faint += 1;
    }

    /// `Pokemon#damage`: returns the HP actually lost.
    fn damage_mon(&mut self, r: MonRef, d: u32) -> u32 {
        let hp = self.mon(r).hp as u32;
        if hp == 0 || d == 0 {
            return 0;
        }
        if d >= hp {
            self.faint(r);
            hp
        } else {
            self.mon_mut(r).hp = (hp - d) as u16;
            d
        }
    }

    /// `Battle#damage` for effects other than a move's own damage (status,
    /// recoil): floor, at least 1, nothing if the target has no HP.
    fn effect_damage(&mut self, r: MonRef, d: u32) -> u32 {
        if self.mon(r).hp == 0 {
            return 0;
        }
        self.damage_mon(r, d.max(1))
    }

    /// `Battle#heal`.
    fn heal(&mut self, r: MonRef, amount: u32) -> Res {
        if amount == 0 {
            return Res::Num(0);
        }
        let m = self.mon_mut(r);
        if m.hp == 0 || !m.is_active || m.hp >= m.max_hp() {
            return Res::Bool(false);
        }
        let new = (m.hp as u32 + amount).min(m.max_hp() as u32);
        let healed = new - m.hp as u32;
        m.hp = new as u16;
        Res::Num(healed as i32)
    }

    /// `Battle#faintMessages`.
    fn faint_messages(&mut self, last_first: bool, force_check: bool, mut check_win: bool) -> bool {
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
        let mut i = 0;
        while i < self.n_faint as usize {
            let left = self.n_faint as usize - i;
            let r = self.faint_queue[i];
            i += 1;
            last = Some(r);
            if !self.mon(r).fainted {
                let side = &mut self.sides[r.side as usize];
                if side.pokemon_left > 0 {
                    side.pokemon_left -= 1;
                }
                self.clear_volatile(r, false);
                let m = self.mon_mut(r);
                m.fainted = true;
                m.is_active = false;
                if self.n_faint as usize - i >= left {
                    check_win = true;
                }
            }
        }
        self.n_faint = 0;
        if check_win && self.check_win(last) {
            return true;
        }
        false
    }

    /// `Battle#checkWin`. If both sides run out together, the side whose
    /// Pokémon fainted last wins.
    fn check_win(&mut self, last_fainted: Option<MonRef>) -> bool {
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

    fn win(&mut self, side: Option<u8>) {
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

    // ---------------------------------------------------------------- status

    fn status_immune(&self, r: MonRef, kind: usize) -> bool {
        self.mon(r).types.iter().any(|&t| t != Type::None && STATUS_IMMUNE[kind][t as usize])
    }

    /// `Pokemon#setStatus` for a non-empty status.
    fn set_status(&mut self, r: MonRef, status: Status) -> bool {
        let m = self.mon(r);
        if m.hp == 0 || !m.is_active || m.status == status || m.fainted {
            return false;
        }
        let kind = match status {
            Status::Brn => 0,
            Status::Par => 1,
            Status::Psn | Status::Tox => 2,
            Status::Slp => 3,
            Status::Frz => 4,
            Status::None => return false,
        };
        if self.status_immune(r, kind) {
            return false;
        }
        // onStart
        let time = match status {
            // Champions: asleep for 2 turns one time in three, otherwise 3.
            Status::Slp => [2u8, 3, 3][self.rand(3, "sleep turns") as usize],
            // Champions: frozen for at most 3 turns.
            Status::Frz => 3,
            _ => 0,
        };
        let m = self.mon_mut(r);
        m.status = status;
        m.status_time = time;
        m.tox_stage = 0;
        true
    }

    /// `Pokemon#trySetStatus`: fails if there is already a status.
    fn try_set_status(&mut self, r: MonRef, status: Status) -> bool {
        if self.mon(r).status != Status::None {
            return false;
        }
        self.set_status(r, status)
    }

    /// `Pokemon#cureStatus`.
    fn cure_status(&mut self, r: MonRef) -> bool {
        let m = self.mon_mut(r);
        if m.hp == 0 || m.status == Status::None {
            return false;
        }
        m.status = Status::None;
        m.status_time = 0;
        m.tox_stage = 0;
        true
    }

    /// `Pokemon#addVolatile`.
    fn add_volatile(&mut self, r: MonRef, kind: VolKind) -> bool {
        let m = self.mon_mut(r);
        if m.hp == 0 {
            return false;
        }
        if let Some(v) = m.volatiles.get_mut(kind) {
            return match kind {
                // stall.onRestart
                VolKind::Stall => {
                    if v.counter < 729 {
                        v.counter *= 3;
                    }
                    v.duration = 2;
                    true
                }
                _ => false,
            };
        }
        let v = match kind {
            VolKind::Protect => Volatile { kind, duration: 1, counter: 0 },
            VolKind::Flinch => Volatile { kind, duration: 1, counter: 0 },
            VolKind::Stall => Volatile { kind, duration: 2, counter: 3 },
        };
        m.volatiles.push(v);
        true
    }

    /// `Battle#boost`.
    fn boost(&mut self, b: Boosts, r: MonRef) -> Res {
        let foe_left = self.sides[1 - r.side as usize].pokemon_left;
        let m = self.mon_mut(r);
        if m.hp == 0 {
            return Res::Num(0);
        }
        if !m.is_active || foe_left == 0 {
            return Res::Bool(false);
        }
        let mut success = false;
        for k in 0..7 {
            if b[k] == 0 {
                continue;
            }
            let new = (m.boosts[k] + b[k]).clamp(-6, 6);
            if new != m.boosts[k] {
                m.boosts[k] = new;
                success = true;
            }
        }
        if success { Res::Bool(true) } else { Res::Null }
    }

    // ----------------------------------------------------------------- moves

    /// `Pokemon#runImmunity` for a move's type.
    fn run_immunity(&self, target: MonRef, typ: Type, ignore: bool) -> bool {
        if ignore || typ == Type::Typeless || typ == Type::None {
            return true;
        }
        let m = self.mon(target);
        if typ == Type::Ground {
            // isGrounded: only the Flying type lifts a Pokémon so far.
            return !m.has_type(Type::Flying);
        }
        !m.types.iter().any(|&t| t != Type::None && TYPE_CHART[typ as usize][t as usize] == 3)
    }

    /// The `BeforeMove` handlers, in Showdown's priority order: sleep and
    /// freeze (10), flinch (8), paralysis (1). Returns whether the move goes ahead.
    fn before_move(&mut self, user: MonRef, d: &MoveData) -> bool {
        match self.mon(user).status {
            Status::Slp => {
                let m = self.mon_mut(user);
                m.status_time = m.status_time.saturating_sub(1);
                if m.status_time == 0 {
                    self.cure_status(user);
                } else {
                    return false;
                }
            }
            Status::Frz => {
                if d.flags & F_DEFROST == 0 {
                    let m = self.mon_mut(user);
                    m.status_time = m.status_time.saturating_sub(1);
                    // The thaw roll is skipped once the timer has run out.
                    if m.status_time == 0 || self.chance(1, 4, "thaw") {
                        self.cure_status(user);
                    } else {
                        return false;
                    }
                }
            }
            _ => {}
        }
        if self.mon(user).volatiles.has(VolKind::Flinch) {
            return false;
        }
        // Champions: full paralysis is 1 in 8.
        if self.mon(user).status == Status::Par && self.chance(1, 8, "full paralysis") {
            return false;
        }
        true
    }

    /// `BattleActions#runMove`.
    fn run_move(&mut self, user: MonRef, move_id: u16, loc: i8) {
        let d = &MOVES[move_id as usize];
        let target = self.get_target(user, d, loc);
        if !self.before_move(user, d) {
            return;
        }
        if d.special != Special::Struggle {
            // deductPP: a move with no PP left fails here.
            let m = self.mon_mut(user);
            let n = m.n_moves as usize;
            match m.moves[..n].iter_mut().find(|s| s.id == move_id) {
                Some(s) if s.pp > 0 => s.pp -= 1,
                _ => return,
            }
        }
        self.use_move(user, d, target);
        self.faint_messages(false, false, true);
        self.check_win(None);
    }

    /// `BattleActions#useMoveInner`.
    fn use_move(&mut self, user: MonRef, d: &'static MoveData, target: Option<MonRef>) -> bool {
        let mut mv = ActiveMove { d, typ: d.typ, spread_hit: false, self_dropped: false, total_damage: 0 };
        let mut target = target;
        if matches!(d.target, Target::User | Target::Allies) {
            target = Some(user);
        }
        if d.special == Special::Struggle {
            // struggle.onModifyMove
            mv.typ = Type::Typeless;
        }
        // frz.onModifyMove: using a defrosting move thaws the user.
        if d.flags & F_DEFROST != 0 && self.mon(user).status == Status::Frz {
            self.cure_status(user);
        }
        if self.mon(user).fainted {
            return false;
        }
        let Some(target) = target else {
            return false;
        };
        let (targets, n) = self.get_move_targets(user, d, target);
        if n == 0 {
            return false;
        }
        let result = self.try_spread_move_hit(&targets[..n], user, &mut mv);
        if self.mon(user).hp == 0 {
            self.faint(user);
        }
        result
    }

    /// Showdown filters `targets` after each hit step by "truthy or 0".
    fn keep_hits(targets: &mut [MonRef; MAX_TARGETS], n: &mut usize, res: &Damage) {
        let mut w = 0;
        for i in 0..*n {
            if res[i].hit() {
                targets[w] = targets[i];
                w += 1;
            }
        }
        *n = w;
    }

    /// `BattleActions#trySpreadMoveHit` (this is also the path single-target moves take).
    fn try_spread_move_hit(&mut self, initial: &[MonRef], user: MonRef, mv: &mut ActiveMove) -> bool {
        let d = mv.d;
        let mut targets = [user; MAX_TARGETS];
        let mut n = initial.len();
        targets[..n].copy_from_slice(initial);
        if n > 1 {
            mv.spread_hit = true;
        }

        // PrepareHit: Protect fails if nothing is left to act, or on a bad stall roll.
        if d.special == Special::Protect && !(self.will_act() && self.stall_move(user)) {
            return false;
        }

        // Step 1, TryHit: the target's Protect.
        let mut res: Damage = [Res::Bool(true); MAX_TARGETS];
        for i in 0..n {
            if d.flags & F_PROTECT != 0 && self.mon(targets[i]).volatiles.has(VolKind::Protect) {
                res[i] = Res::NotFail;
            }
        }
        Battle::keep_hits(&mut targets, &mut n, &res);
        if n == 0 {
            return false;
        }

        // Step 2, type immunity.
        for i in 0..n {
            res[i] = Res::Bool(self.run_immunity(targets[i], mv.typ, d.ignore_immunity));
        }
        Battle::keep_hits(&mut targets, &mut n, &res);
        if n == 0 {
            return false;
        }

        // Step 3, move-specific immunity: powder moves against Grass types.
        for i in 0..n {
            let immune = d.flags & F_POWDER != 0 && targets[i] != user && self.status_immune(targets[i], 5);
            res[i] = Res::Bool(!immune);
        }
        Battle::keep_hits(&mut targets, &mut n, &res);
        if n == 0 {
            return false;
        }

        // Step 4, accuracy.
        for i in 0..n {
            let t = targets[i];
            let mut acc = if d.accuracy == 0 { None } else { Some(d.accuracy as i32) };
            if let Some(a) = acc.as_mut() {
                let mut stage = (self.mon(user).boosts[ACC] as i32).clamp(-6, 6);
                if !d.ignore_evasion {
                    stage = (stage - self.mon(t).boosts[EVA] as i32).clamp(-6, 6);
                }
                if stage > 0 {
                    *a = *a * (3 + stage) / 3;
                } else if stage < 0 {
                    *a = *a * 3 / (3 - stage);
                }
            }
            let toxic_from_poison = d.id == "toxic" && self.mon(user).has_type(Type::Poison);
            if toxic_from_poison || (d.target == Target::User && d.category == Category::Status) {
                acc = None;
            }
            res[i] = match acc {
                Some(a) => Res::Bool(self.chance(a as u32, 100, "accuracy")),
                None => Res::Bool(true),
            };
        }
        Battle::keep_hits(&mut targets, &mut n, &res);
        if n == 0 {
            return false;
        }

        // Step 7, the hits themselves.
        let dmg = self.move_hit_loop(&targets[..n], user, mv);
        Battle::keep_hits(&mut targets, &mut n, &dmg);
        n > 0
    }

    /// The `StallMove` event: only the stall counter listens.
    fn stall_move(&mut self, user: MonRef) -> bool {
        let Some(v) = self.mon(user).volatiles.get(VolKind::Stall) else {
            return true;
        };
        let counter = v.counter.max(1) as u32;
        let success = self.chance(1, counter, "consecutive protect");
        if !success {
            self.mon_mut(user).volatiles.remove(VolKind::Stall);
        }
        success
    }

    /// `hitStepMoveHitLoop` as overridden by the Champions mod.
    fn move_hit_loop(&mut self, targets: &[MonRef], user: MonRef, mv: &mut ActiveMove) -> Damage {
        let d = mv.d;
        let n = targets.len();
        let mut damage: Damage = [Res::Num(0); MAX_TARGETS];
        mv.total_damage = 0;
        let target_hits: u32 = match d.multihit {
            (0, 0) => 1,
            (2, 5) => {
                const HITS: [u32; 20] = [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5];
                HITS[self.rand(20, "multihit count") as usize]
            }
            (a, b) if a == b => a as u32,
            (a, b) => self.rand((b - a + 1) as u32, "multihit count") + a as u32,
        };
        let mut copy: Targets = [Tgt::Gone; MAX_TARGETS];
        let mut hit = 1;
        let mut broke_on_first = true;
        while hit <= target_hits {
            if damage[..n].contains(&Res::Bool(false)) {
                break;
            }
            if hit > 1 && self.mon(user).status == Status::Slp {
                break;
            }
            if targets.iter().all(|&t| self.mon(t).hp == 0) {
                break;
            }
            for i in 0..n {
                copy[i] = Tgt::Mon(targets[i]);
            }
            let move_damage = self.spread_move_hit(&mut copy, n, user, mv, Effect::of_move(d), false, false);
            if !move_damage[..n].iter().any(|&v| v != Res::Bool(false)) {
                break;
            }
            for i in 0..n {
                let dealt = match move_damage[i] {
                    Res::Num(x) => x,
                    _ => 0,
                };
                damage[i] = Res::Num(dealt);
                mv.total_damage += dealt;
            }
            self.each_event("update tie (after hit)");
            broke_on_first = false;
            if self.mon(user).hp == 0 && n == 1 {
                break;
            }
            hit += 1;
        }
        if broke_on_first {
            return [Res::Bool(false); MAX_TARGETS];
        }
        let user_down = self.mon(user).hp == 0;
        self.faint_messages(false, false, user_down);

        if mv.total_damage > 0 {
            self.apply_recoil(mv.total_damage as u32, d, user);
        }
        if !damage[..n].iter().any(|v| v.hit()) {
            return damage;
        }
        self.each_event("update tie (after move)");
        self.after_move_secondary(&copy[..n], d);
        damage
    }

    /// `BattleActions#applyRecoilDamage`.
    fn apply_recoil(&mut self, dealt: u32, d: &MoveData, user: MonRef) {
        if d.special == Special::Struggle {
            let r = round_div(self.mon(user).max_hp() as u32, 1, 4).max(1);
            self.effect_damage(user, r);
        } else if d.recoil.0 > 0 {
            let r = round_div(dealt, d.recoil.0 as u32, d.recoil.1 as u32).max(1);
            self.effect_damage(user, r);
        }
    }

    /// The `AfterMoveSecondary` event. Freeze listens on each target so that
    /// thawing moves can cure it; the listeners are speed-sorted like any others.
    fn after_move_secondary(&mut self, targets: &[Tgt], d: &MoveData) {
        let mut frozen = [(MonRef { side: 0, idx: 0 }, 0i32); MAX_TARGETS];
        let mut k = 0;
        for t in targets {
            if let Tgt::Mon(r) = *t {
                if self.mon(r).status == Status::Frz {
                    frozen[k] = (r, self.mon(r).speed);
                    k += 1;
                }
            }
        }
        self.speed_sort(&mut frozen[..k], |a, b| b.1 as i64 - a.1 as i64, "after-move handler tie");
        if d.thaws_target {
            for &(r, _) in &frozen[..k] {
                self.cure_status(r);
            }
        }
    }

    /// `BattleActions#spreadMoveHit` as overridden by the Champions mod.
    #[allow(clippy::too_many_arguments)]
    fn spread_move_hit(
        &mut self,
        targets: &mut Targets,
        n: usize,
        user: MonRef,
        mv: &mut ActiveMove,
        eff: Effect,
        is_secondary: bool,
        is_self: bool,
    ) -> Damage {
        let d = mv.d;
        let mut damage: Damage = [Res::Bool(true); MAX_TARGETS];

        // 1. damage calculation
        for i in 0..n {
            if let Tgt::Mon(t) = targets[i] {
                let cur = self.get_damage(user, t, mv, &eff);
                damage[i] = if cur == Res::Bool(false) || cur == Res::Null { Res::Bool(false) } else { cur };
            }
        }
        for i in 0..n {
            if damage[i] == Res::Bool(false) {
                targets[i] = Tgt::Gone;
            }
        }

        // 2. apply damage (`Battle#spreadDamage`)
        for i in 0..n {
            let Res::Num(amount) = damage[i] else {
                continue;
            };
            let Tgt::Mon(t) = targets[i] else {
                damage[i] = Res::Num(0);
                continue;
            };
            if self.mon(t).hp == 0 {
                damage[i] = Res::Num(0);
                continue;
            }
            if !self.mon(t).is_active {
                damage[i] = Res::Bool(false);
                continue;
            }
            let amount = if amount != 0 { amount.max(1) } else { 0 };
            let dealt = self.damage_mon(t, amount as u32);
            damage[i] = Res::Num(dealt as i32);
            if dealt > 0 && d.drain.0 > 0 {
                let heal = round_div(dealt, d.drain.0 as u32, d.drain.1 as u32);
                self.heal(user, heal);
            }
        }
        for i in 0..n {
            if damage[i] == Res::Bool(false) {
                targets[i] = Tgt::Gone;
            }
        }

        // 3. the effect's own boosts, healing, status and volatiles
        self.run_move_effects(&mut damage, targets, n, mv, &eff);
        for i in 0..n {
            if !damage[i].hit() {
                targets[i] = Tgt::Gone;
            }
        }

        // 4. stat changes to the user
        if let Some(b) = eff.self_boosts {
            if !mv.self_dropped {
                self.self_drops(targets, n, user, mv, b, eff.self_chance, is_secondary);
            }
        }

        // 5. secondary effects
        if eff.primary && !d.secondaries.is_empty() {
            for i in 0..n {
                let Tgt::Mon(t) = targets[i] else {
                    continue;
                };
                for sec in d.secondaries {
                    // The roll happens even when the secondary is guaranteed.
                    let roll = self.rand(100, "secondary effect");
                    if sec.chance == 0 || roll < sec.chance as u32 {
                        let mut one: Targets = [Tgt::Mon(t), Tgt::Gone, Tgt::Gone];
                        self.spread_move_hit(&mut one, 1, user, mv, Effect::of_secondary(sec), true, false);
                    }
                }
            }
        }

        // DamagingHit: a damaging Fire move thaws a frozen target.
        if !is_secondary && !is_self {
            for i in 0..n {
                if let (Res::Num(_), Tgt::Mon(t)) = (damage[i], targets[i]) {
                    if self.mon(t).status == Status::Frz && mv.typ == Type::Fire && d.category != Category::Status {
                        self.cure_status(t);
                    }
                }
            }
        }
        damage
    }

    /// `BattleActions#selfDrops`.
    #[allow(clippy::too_many_arguments)]
    fn self_drops(
        &mut self,
        targets: &Targets,
        n: usize,
        user: MonRef,
        mv: &mut ActiveMove,
        b: Boosts,
        chance: u8,
        is_secondary: bool,
    ) {
        for i in 0..n {
            if targets[i] == Tgt::Gone || mv.self_dropped {
                continue;
            }
            let mut apply = true;
            if !is_secondary {
                // Showdown rolls here even when there is no chance to check.
                let roll = self.rand(100, "self stat change");
                apply = chance == 0 || roll < chance as u32;
                if mv.d.multihit == (0, 0) {
                    mv.self_dropped = true;
                }
            }
            if apply {
                let mut one: Targets = [Tgt::Mon(user), Tgt::Gone, Tgt::Gone];
                self.spread_move_hit(&mut one, 1, user, mv, Effect::of_self(b), is_secondary, true);
            }
        }
    }

    /// `BattleActions#runMoveEffects`.
    fn run_move_effects(&mut self, damage: &mut Damage, targets: &Targets, n: usize, mv: &ActiveMove, eff: &Effect) {
        for i in 0..n {
            let Tgt::Mon(t) = targets[i] else {
                continue;
            };
            let mut did = Res::Undef;
            if let Some(b) = eff.boosts {
                if !self.mon(t).fainted {
                    did = combine(did, self.boost(b, t));
                }
            }
            if eff.heal.0 > 0 && !self.mon(t).fainted {
                let m = self.mon(t);
                if m.hp >= m.max_hp() {
                    damage[i] = combine(damage[i], Res::Bool(false));
                    continue;
                }
                let amount = round_div(m.max_hp() as u32, eff.heal.0 as u32, eff.heal.1 as u32);
                let healed = self.heal(t, amount);
                if !healed.hit() {
                    damage[i] = combine(damage[i], Res::Bool(false));
                    continue;
                }
                did = Res::Bool(true);
            }
            if eff.status != Status::None {
                let ok = self.try_set_status(t, eff.status);
                if !ok && mv.d.status != Status::None {
                    damage[i] = combine(damage[i], Res::Bool(false));
                    continue;
                }
                did = combine(did, Res::Bool(ok));
            }
            if let Some(v) = eff.volatile {
                did = combine(did, Res::Bool(self.add_volatile(t, v)));
            }
            if eff.adds_stall {
                self.add_volatile(t, VolKind::Stall);
                did = combine(did, Res::Bool(true));
            }
            if did == Res::Undef {
                did = Res::Bool(true);
            }
            damage[i] = combine(damage[i], if did == Res::Null { Res::Bool(false) } else { did });
        }
    }

    /// `BattleActions#getDamage` followed by the Champions `modifyDamage`.
    fn get_damage(&mut self, user: MonRef, target: MonRef, mv: &ActiveMove, eff: &Effect) -> Res {
        if !eff.primary {
            // Secondaries and self effects carry no base power.
            return Res::Undef;
        }
        let d = mv.d;
        if !self.run_immunity(target, mv.typ, d.ignore_immunity) {
            return Res::Bool(false);
        }
        if d.base_power == 0 {
            return Res::Undef;
        }

        const CRIT_MULT: [u32; 5] = [0, 24, 8, 2, 1];
        let crit_ratio = (d.crit_ratio as usize).min(4);
        let crit = if d.will_crit {
            true
        } else if crit_ratio > 0 {
            self.chance(1, CRIT_MULT[crit_ratio], "critical hit")
        } else {
            false
        };

        let physical = d.category == Category::Physical;
        let atk_stat = if d.off_stat != 0 {
            d.off_stat as usize
        } else if physical {
            1
        } else {
            3
        };
        let def_stat = if d.def_stat != 0 {
            d.def_stat as usize
        } else if physical {
            2
        } else {
            4
        };
        let attacker = if d.off_from_target { target } else { user };
        let mut atk_stage = self.mon(attacker).boosts[atk_stat - 1];
        let mut def_stage = self.mon(target).boosts[def_stat - 1];
        // A critical hit ignores the attacker's drops and the defender's boosts.
        if crit && atk_stage < 0 {
            atk_stage = 0;
        }
        if d.ignore_defensive || (crit && def_stage > 0) {
            def_stage = 0;
        }
        let attack = boosted(self.mon(attacker).stats[atk_stat] as u32, atk_stage);
        let defense = boosted(self.mon(target).stats[def_stat] as u32, def_stage);
        let level = self.mon(user).level as u32;
        let mut dmg = (2 * level / 5 + 2) * d.base_power as u32 * attack / defense / 50;

        // modifyDamage
        dmg += 2;
        if mv.spread_hit {
            dmg = modify(dmg, 3072);
        }
        if crit {
            dmg = dmg * 3 / 2;
        }
        let roll = self.rand(16, "damage roll");
        dmg = dmg * (100 - roll) / 100;
        if mv.typ != Type::Typeless && self.mon(user).has_type(mv.typ) {
            dmg = modify(dmg, 6144);
        }
        let mut type_mod = 0i32;
        if mv.typ != Type::Typeless {
            for &t in &self.mon(target).types {
                if t == Type::None {
                    continue;
                }
                match TYPE_CHART[mv.typ as usize][t as usize] {
                    1 => type_mod += 1,
                    2 => type_mod -= 1,
                    _ => {}
                }
            }
        }
        if type_mod > 0 {
            dmg <<= type_mod;
        } else {
            for _ in 0..-type_mod {
                dmg /= 2;
            }
        }
        if physical && self.mon(user).status == Status::Brn {
            dmg = modify(dmg, 2048);
        }
        if dmg == 0 {
            return Res::Num(1);
        }
        Res::Num((dmg & 0xFFFF) as i32)
    }

    // -------------------------------------------------------------- residual

    /// `Battle#fieldEvent('Residual')`: end-of-turn status damage and volatile
    /// timers, speed-sorted with ties shuffled.
    fn residual(&mut self) {
        #[derive(Clone, Copy)]
        enum Kind {
            Status(Status),
            Vol(VolKind),
        }
        #[derive(Clone, Copy)]
        struct Handler {
            order: u32,
            speed: i32,
            sub_order: u8,
            mon: MonRef,
            kind: Kind,
        }
        const NO_ORDER: u32 = u32::MAX;
        let blank = Handler {
            order: 0,
            speed: 0,
            sub_order: 0,
            mon: MonRef { side: 0, idx: 0 },
            kind: Kind::Vol(VolKind::Flinch),
        };
        let mut hs = [blank; 20];
        let mut n = 0;
        for side in 0..2 {
            for pos in 0..ACTIVE {
                let r = self.active(side, pos);
                let m = self.mon(r);
                let order = match m.status {
                    Status::Psn | Status::Tox => 9,
                    Status::Brn => 10,
                    _ => 0,
                };
                if order != 0 {
                    hs[n] = Handler { order, speed: m.speed, sub_order: 0, mon: r, kind: Kind::Status(m.status) };
                    n += 1;
                }
                for v in m.volatiles.as_slice() {
                    if v.duration > 0 {
                        hs[n] =
                            Handler { order: NO_ORDER, speed: m.speed, sub_order: 2, mon: r, kind: Kind::Vol(v.kind) };
                        n += 1;
                    }
                }
            }
        }
        self.speed_sort(
            &mut hs[..n],
            |a, b| {
                let o = a.order as i64 - b.order as i64;
                if o != 0 {
                    return o;
                }
                let s = b.speed as i64 - a.speed as i64;
                if s != 0 {
                    return s;
                }
                a.sub_order as i64 - b.sub_order as i64
            },
            "residual handler tie",
        );
        for h in &hs[..n] {
            let r = h.mon;
            if self.mon(r).fainted {
                continue;
            }
            match h.kind {
                Kind::Vol(kind) => {
                    let m = self.mon_mut(r);
                    let Some(v) = m.volatiles.get_mut(kind) else {
                        continue;
                    };
                    v.duration -= 1;
                    if v.duration == 0 {
                        // removeVolatile does nothing to a Pokémon with no HP.
                        if m.hp > 0 {
                            m.volatiles.remove(kind);
                        }
                        continue;
                    }
                }
                Kind::Status(status) => {
                    if self.mon(r).status != status {
                        continue;
                    }
                    let max = self.mon(r).max_hp() as u32;
                    match status {
                        Status::Brn => {
                            self.effect_damage(r, max / 16);
                        }
                        Status::Psn => {
                            self.effect_damage(r, max / 8);
                        }
                        Status::Tox => {
                            let m = self.mon_mut(r);
                            if m.tox_stage < 15 {
                                m.tox_stage += 1;
                            }
                            let stage = m.tox_stage as u32;
                            self.effect_damage(r, (max / 16).max(1) * stage);
                        }
                        _ => {}
                    }
                }
            }
            self.faint_messages(false, false, true);
            if self.ended {
                return;
            }
        }
    }

    // ------------------------------------------------------------- turn loop

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
                self.run_move(r, a.move_id, a.target_loc);
            }
            ActKind::Switch | ActKind::InstaSwitch => {
                let out = a.mon.unwrap();
                let pos = self.mon(out).position as usize;
                self.switch_in(a.switch_to.unwrap(), pos, false);
            }
            ActKind::RunSwitch => self.run_switch(a.mon.unwrap()),
            ActKind::BeforeTurn => self.each_event("before-turn tie"),
            ActKind::Residual => {
                self.update_speed();
                self.residual();
            }
        }

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
            self.each_event("update tie (after action)");
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
    fn turn_loop(&mut self) {
        self.request = Request::None;
        if !self.mid_turn {
            self.insert_choice(Battle::blank_action(ActKind::BeforeTurn, 4));
            self.queue.push(Battle::blank_action(ActKind::Residual, 300));
            self.mid_turn = true;
        }
        while let Some(a) = self.queue.shift() {
            self.run_action(a);
            if self.request != Request::None || self.ended {
                return;
            }
        }
        // endTurn
        self.turn += 1;
        self.mid_turn = false;
        self.queue.clear();
        if self.turn > 1000 {
            // Showdown's hard turn limit: the battle is a tie.
            self.win(None);
            return;
        }
        self.request = Request::Move;
    }

    // ---------------------------------------------------------- public entry

    /// Builds a battle from the four Pokémon each side picked at team preview,
    /// in the order picked (the first two lead), and plays the opening
    /// switch-ins. `seed` is Showdown's four 16-bit seed words.
    pub fn new(teams: [&[PokemonSet]; 2], seed: [u16; 4]) -> Result<Battle, Error> {
        let blank_slot = MoveSlot { id: 0, pp: 0, maxpp: 0 };
        let blank_mon = Pokemon {
            species: 0,
            types: [Type::None; 2],
            level: 50,
            stats: [0; 6],
            hp: 0,
            status: Status::None,
            status_time: 0,
            tox_stage: 0,
            boosts: [0; 7],
            moves: [blank_slot; MAX_MOVES],
            n_moves: 0,
            position: 0,
            is_active: false,
            fainted: true,
            faint_queued: false,
            switch_flag: false,
            speed: 0,
            volatiles: Volatiles::default(),
        };
        let blank_side = Side { team: [blank_mon; MAX_TEAM], n: 0, order: [0, 1, 2, 3, 4, 5], pokemon_left: 0 };
        let mut b = Battle {
            rng: Rng::from_words(seed),
            sides: [blank_side; 2],
            turn: 0,
            request: Request::None,
            ended: false,
            winner: None,
            queue: Queue::new(),
            faint_queue: [MonRef { side: 0, idx: 0 }; 8],
            n_faint: 0,
            mid_turn: true,
        };
        for (s, team) in teams.iter().enumerate() {
            if team.len() < ACTIVE || team.len() > MAX_TEAM {
                return Err(Error::BadTeam(format!(
                    "side {} brings {} Pokémon; need {ACTIVE} to {MAX_TEAM}",
                    s + 1,
                    team.len()
                )));
            }
            for (i, set) in team.iter().enumerate() {
                let sp = SPECIES.get(set.species as usize).ok_or_else(|| Error::BadTeam("unknown species".into()))?;
                if set.moves.is_empty() || set.moves.len() > MAX_MOVES {
                    return Err(Error::BadTeam(format!("{} needs 1 to {MAX_MOVES} moves", sp.name)));
                }
                if set.stat_points.iter().any(|&p| p > 32)
                    || set.stat_points.iter().map(|&p| p as u32).sum::<u32>() > 66
                {
                    return Err(Error::BadTeam(format!(
                        "{}: at most 32 stat points per stat and 66 in total",
                        sp.name
                    )));
                }
                let mut mon = blank_mon;
                mon.species = set.species;
                mon.types = sp.types;
                mon.fainted = false;
                mon.position = i as u8;
                // Champions stats: base + stat points + 75 for HP, + 20 otherwise.
                for k in 0..6 {
                    let base = sp.base[k] as u32 + set.stat_points[k] as u32;
                    let mut v = if k == 0 { base + 75 } else { base + 20 };
                    if k as u8 == set.nature.0 && set.nature.0 != set.nature.1 {
                        v = ((v * 110) & 0xFFFF) / 100;
                    } else if k as u8 == set.nature.1 && set.nature.0 != set.nature.1 {
                        v = ((v * 90) & 0xFFFF) / 100;
                    }
                    mon.stats[k] = v as u16;
                }
                mon.hp = mon.stats[0];
                mon.speed = mon.stats[5] as i32;
                for (k, &id) in set.moves.iter().enumerate() {
                    let d = MOVES.get(id as usize).ok_or_else(|| Error::BadTeam("unknown move".into()))?;
                    if !d.supported {
                        return Err(Error::Unsupported(format!("move {}", d.name)));
                    }
                    mon.moves[k] = MoveSlot { id, pp: d.pp, maxpp: d.pp };
                }
                mon.n_moves = set.moves.len() as u8;
                b.sides[s].team[i] = mon;
            }
            b.sides[s].n = team.len() as u8;
            b.sides[s].pokemon_left = team.len() as u8;
        }

        // Team preview picks are queued as 'team' actions and sorted like any
        // others, so equal-speed Pokémon picked in the same slot draw from the RNG.
        let mut picks = [Battle::blank_action(ActKind::Team, 1); 2 * MAX_TEAM];
        let mut n = 0;
        for s in 0..2 {
            for i in 0..b.sides[s].n as usize {
                let r = MonRef { side: s as u8, idx: i as u8 };
                picks[n].priority = -(i as i32);
                picks[n].speed = b.action_speed(r);
                n += 1;
            }
        }
        b.speed_sort(&mut picks[..n], cmp_action, "team preview tie");

        b.queue.push(Battle::blank_action(ActKind::Start, 2));
        b.turn_loop();
        Ok(b)
    }

    fn usable_moves(&self, r: MonRef) -> bool {
        let m = self.mon(r);
        m.moves[..m.n_moves as usize].iter().any(|s| s.pp > 0)
    }

    /// Every choice the given active slot may make at the current request.
    pub fn legal_choices(&self, side: usize, pos: usize) -> Vec<Choice> {
        let mut out = Vec::new();
        let r = self.active(side, pos);
        let m = self.mon(r);
        let s = &self.sides[side];
        let bench = |out: &mut Vec<Choice>| {
            for p in ACTIVE..s.n as usize {
                if !s.team[s.order[p] as usize].fainted {
                    out.push(Choice::Switch { to: p as u8 });
                }
            }
        };
        match self.request {
            Request::None => {}
            Request::Move => {
                if m.fainted {
                    out.push(Choice::Pass);
                    return out;
                }
                if !self.usable_moves(r) {
                    out.push(Choice::Move { slot: 0, target: 0 });
                } else {
                    for (i, slot) in m.moves[..m.n_moves as usize].iter().enumerate() {
                        if slot.pp == 0 {
                            continue;
                        }
                        let t = MOVES[slot.id as usize].target;
                        if t.is_chosen() {
                            for loc in [1i8, 2, -1, -2] {
                                if self.valid_target_loc(loc, r, t) {
                                    out.push(Choice::Move { slot: i as u8, target: loc });
                                }
                            }
                        } else {
                            out.push(Choice::Move { slot: i as u8, target: 0 });
                        }
                    }
                }
                bench(&mut out);
            }
            Request::Switch => {
                if !m.switch_flag {
                    out.push(Choice::Pass);
                    return out;
                }
                bench(&mut out);
                if out.len() < self.open_slots(side) {
                    // More empty slots than replacements: one of them stays empty.
                    out.push(Choice::Pass);
                }
            }
        }
        out
    }

    fn living_bench(&self, side: usize) -> usize {
        let s = &self.sides[side];
        (ACTIVE..s.n as usize).filter(|&p| !s.team[s.order[p] as usize].fainted).count()
    }

    fn open_slots(&self, side: usize) -> usize {
        (0..ACTIVE).filter(|&p| self.mon(self.active(side, p)).switch_flag).count()
    }

    /// Whether one slot's choice is legal on its own (the allocation-free
    /// counterpart of `legal_choices`).
    fn slot_ok(&self, side: usize, pos: usize, c: Choice) -> bool {
        let r = self.active(side, pos);
        let m = self.mon(r);
        let s = &self.sides[side];
        let bench_ok =
            |to: u8| (ACTIVE..s.n as usize).contains(&(to as usize)) && !s.team[s.order[to as usize] as usize].fainted;
        match self.request {
            Request::None => false,
            Request::Move => match c {
                Choice::Pass => m.fainted,
                _ if m.fainted => false,
                Choice::Switch { to } => bench_ok(to),
                Choice::Move { slot, target } => {
                    if !self.usable_moves(r) {
                        return slot == 0 && target == 0;
                    }
                    if slot >= m.n_moves || m.moves[slot as usize].pp == 0 {
                        return false;
                    }
                    let t = MOVES[m.moves[slot as usize].id as usize].target;
                    if t.is_chosen() { target != 0 && self.valid_target_loc(target, r, t) } else { target == 0 }
                }
            },
            Request::Switch => match c {
                Choice::Pass => !m.switch_flag || self.living_bench(side) < self.open_slots(side),
                Choice::Switch { to } => m.switch_flag && bench_ok(to),
                Choice::Move { .. } => false,
            },
        }
    }

    /// The constraints that tie a side's two slots together: no two slots
    /// switching to the same Pokémon, and at a switch request every open slot
    /// filled while replacements last.
    fn pair_ok(&self, side: usize, c: &[Choice; ACTIVE]) -> bool {
        if let (Choice::Switch { to: a }, Choice::Switch { to: b }) = (c[0], c[1]) {
            if a == b {
                return false;
            }
        }
        if self.request == Request::Switch {
            let passes =
                (0..ACTIVE).filter(|&p| c[p] == Choice::Pass && self.mon(self.active(side, p)).switch_flag).count();
            if passes != self.open_slots(side).saturating_sub(self.living_bench(side)) {
                return false;
            }
        }
        true
    }

    /// Whether a side's two slot choices are legal together.
    pub fn joint_ok(&self, side: usize, c: &[Choice; ACTIVE]) -> bool {
        (0..ACTIVE).all(|pos| self.slot_ok(side, pos, c[pos])) && self.pair_ok(side, c)
    }

    /// Every legal pair of slot choices for a side at the current request.
    pub fn joint_choices(&self, side: usize) -> Vec<[Choice; ACTIVE]> {
        let a = self.legal_choices(side, 0);
        let b = self.legal_choices(side, 1);
        let mut out = Vec::with_capacity(a.len() * b.len());
        for &x in &a {
            for &y in &b {
                let c = [x, y];
                if self.pair_ok(side, &c) {
                    out.push(c);
                }
            }
        }
        out
    }

    /// Submits both sides' choices and runs the battle to the next request (or the end).
    /// A side that is not being asked anything passes in both slots.
    pub fn choose(&mut self, choices: [[Choice; ACTIVE]; 2]) -> Result<(), Error> {
        if self.ended || self.request == Request::None {
            return Err(Error::BadChoice("the battle is not waiting for a choice".into()));
        }
        let request = self.request;
        for side in 0..2 {
            if !self.joint_ok(side, &choices[side]) {
                return Err(Error::BadChoice(format!("p{}: {:?} is not legal here", side + 1, choices[side])));
            }
        }

        // commitChoices
        self.update_speed();
        let old = self.queue;
        self.queue.clear();
        for side in 0..2 {
            for pos in 0..ACTIVE {
                let r = self.active(side, pos);
                match choices[side][pos] {
                    Choice::Pass => {}
                    Choice::Move { slot, target } => {
                        let id = if self.usable_moves(r) { self.mon(r).moves[slot as usize].id } else { struggle_id() };
                        let a = self.resolve_move(r, id, target);
                        self.queue.push(a);
                    }
                    Choice::Switch { to } => {
                        let incoming = self.active_at(side, to as usize);
                        let kind = if request == Request::Switch { ActKind::InstaSwitch } else { ActKind::Switch };
                        let a = self.resolve_switch(kind, r, incoming);
                        self.queue.push(a);
                    }
                }
            }
        }
        self.sort_queue();
        for a in old.as_slice() {
            self.queue.push(*a);
        }
        self.turn_loop();
        Ok(())
    }

    /// The Pokémon at any position in a side's current order (active or benched).
    pub fn active_at(&self, side: usize, pos: usize) -> MonRef {
        MonRef { side: side as u8, idx: self.sides[side].order[pos] }
    }
}
