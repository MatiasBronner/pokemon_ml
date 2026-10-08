//! Battle state. Everything here is plain fixed-size data so a `Battle` can be
//! copied cheaply (search wants thousands of copies per decision).

use crate::data::{Status, Type};
use crate::rng::Rng;

/// Largest team a side can bring into battle.
pub const MAX_TEAM: usize = 6;
/// Active Pokémon per side (doubles).
pub const ACTIVE: usize = 2;
pub const MAX_MOVES: usize = 4;
const MAX_VOLATILES: usize = 4;

/// A Pokémon identified by side and by its fixed index in that side's team.
/// The index never changes, unlike its field position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MonRef {
    pub side: u8,
    pub idx: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoveSlot {
    pub id: u16,
    pub pp: u8,
    pub maxpp: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VolKind {
    Protect,
    /// Showdown's `stall`: the consecutive-Protect counter.
    Stall,
    Flinch,
}

impl VolKind {
    pub fn id(self) -> &'static str {
        match self {
            VolKind::Protect => "protect",
            VolKind::Stall => "stall",
            VolKind::Flinch => "flinch",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Volatile {
    pub kind: VolKind,
    /// Turns left; 0 means the volatile has no duration.
    pub duration: u8,
    /// `stall` only: 1-in-`counter` chance the next protecting move works.
    pub counter: u16,
}

/// Volatile statuses in the order they were added. Showdown iterates them in
/// insertion order, and that order decides how speed ties between their
/// handlers are broken, so it is part of the state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Volatiles {
    len: u8,
    items: [Volatile; MAX_VOLATILES],
}

impl Default for Volatiles {
    fn default() -> Self {
        Volatiles { len: 0, items: [Volatile { kind: VolKind::Flinch, duration: 0, counter: 0 }; MAX_VOLATILES] }
    }
}

impl Volatiles {
    pub fn as_slice(&self) -> &[Volatile] {
        &self.items[..self.len as usize]
    }
    pub fn get(&self, kind: VolKind) -> Option<&Volatile> {
        self.as_slice().iter().find(|v| v.kind == kind)
    }
    pub fn get_mut(&mut self, kind: VolKind) -> Option<&mut Volatile> {
        let n = self.len as usize;
        self.items[..n].iter_mut().find(|v| v.kind == kind)
    }
    pub fn has(&self, kind: VolKind) -> bool {
        self.get(kind).is_some()
    }
    pub fn push(&mut self, v: Volatile) {
        debug_assert!(!self.has(v.kind));
        self.items[self.len as usize] = v;
        self.len += 1;
    }
    pub fn remove(&mut self, kind: VolKind) -> bool {
        let n = self.len as usize;
        match self.items[..n].iter().position(|v| v.kind == kind) {
            Some(i) => {
                self.items.copy_within(i + 1..n, i);
                self.len -= 1;
                true
            }
            None => false,
        }
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pokemon {
    pub species: u16,
    pub types: [Type; 2],
    pub level: u8,
    /// Unboosted stats: max HP, Atk, Def, SpA, SpD, Spe.
    pub stats: [u16; 6],
    pub hp: u16,
    pub status: Status,
    /// Sleep / freeze turn counter (Showdown's `statusState.time`).
    pub status_time: u8,
    /// Toxic damage stage (Showdown's `statusState.stage`).
    pub tox_stage: u8,
    /// Stat stages: atk, def, spa, spd, spe, accuracy, evasion.
    pub boosts: [i8; 7],
    pub moves: [MoveSlot; MAX_MOVES],
    pub n_moves: u8,
    /// Index into the side's current order; positions below `ACTIVE` are on the field.
    pub position: u8,
    pub is_active: bool,
    pub fainted: bool,
    pub faint_queued: bool,
    /// Must be replaced at the next switch request.
    pub switch_flag: bool,
    /// Speed as last cached by Showdown's `updateSpeed`; several orderings read
    /// this stale value rather than the live stat.
    pub speed: i32,
    pub volatiles: Volatiles,
}

impl Pokemon {
    pub fn max_hp(&self) -> u16 {
        self.stats[0]
    }
    pub fn has_type(&self, t: Type) -> bool {
        self.types[0] == t || self.types[1] == t
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Side {
    pub team: [Pokemon; MAX_TEAM],
    /// Number of Pokémon brought.
    pub n: u8,
    /// `order[position]` is the team index of the Pokémon at that position.
    pub order: [u8; MAX_TEAM],
    pub pokemon_left: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Request {
    /// The battle is running or over; nobody is being asked anything.
    None,
    /// Both sides choose moves or switches for the turn.
    Move,
    /// One or both sides replace fainted Pokémon.
    Switch,
}

/// One active slot's decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Choice {
    Pass,
    /// `slot` is the move slot (0-based). `target` is Showdown's target
    /// location: +1/+2 are the foe's slots, -1/-2 your own, 0 for moves that
    /// do not take a target.
    Move {
        slot: u8,
        target: i8,
    },
    /// Switch to the Pokémon currently at this position (0-based) in the side's order.
    Switch {
        to: u8,
    },
}

impl Choice {
    /// The choice in Showdown's notation: `move 2 1`, `move 3`, `switch 4`, `pass`.
    pub fn to_showdown(self) -> String {
        match self {
            Choice::Pass => "pass".into(),
            Choice::Move { slot, target: 0 } => format!("move {}", slot + 1),
            Choice::Move { slot, target } => format!("move {} {}", slot + 1, target),
            Choice::Switch { to } => format!("switch {}", to + 1),
        }
    }

    /// Parses one slot's choice from Showdown's notation.
    pub fn parse(s: &str) -> Option<Choice> {
        let mut parts = s.split_whitespace();
        let kind = parts.next()?;
        let mut num = || parts.next().and_then(|p| p.parse::<i32>().ok());
        match kind {
            "pass" => Some(Choice::Pass),
            "move" => {
                let slot = num()?;
                let target = num().unwrap_or(0);
                (1..=MAX_MOVES as i32)
                    .contains(&slot)
                    .then_some(Choice::Move { slot: (slot - 1) as u8, target: target as i8 })
            }
            "switch" => {
                let to = num()?;
                (1..=MAX_TEAM as i32).contains(&to).then_some(Choice::Switch { to: (to - 1) as u8 })
            }
            _ => None,
        }
    }

    /// Parses a side's choice, e.g. `move 1 2, switch 3`.
    pub fn parse_side(s: &str) -> Option<[Choice; ACTIVE]> {
        let mut out = [Choice::Pass; ACTIVE];
        let mut n = 0;
        for part in s.split(',') {
            if n == ACTIVE {
                return None;
            }
            out[n] = Choice::parse(part.trim())?;
            n += 1;
        }
        (n == ACTIVE).then_some(out)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ActKind {
    Team,
    Start,
    InstaSwitch,
    BeforeTurn,
    RunSwitch,
    Switch,
    Move,
    Residual,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Action {
    pub kind: ActKind,
    pub order: u32,
    pub priority: i32,
    pub speed: i32,
    pub mon: Option<MonRef>,
    /// Move id for `Move`.
    pub move_id: u16,
    pub target_loc: i8,
    /// Incoming Pokémon for `Switch` / `InstaSwitch`.
    pub switch_to: Option<MonRef>,
}

pub(crate) const QUEUE_CAP: usize = 16;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Queue {
    pub len: u8,
    pub items: [Action; QUEUE_CAP],
}

impl Queue {
    pub fn new() -> Self {
        let blank = Action {
            kind: ActKind::Residual,
            order: 0,
            priority: 0,
            speed: 0,
            mon: None,
            move_id: 0,
            target_loc: 0,
            switch_to: None,
        };
        Queue { len: 0, items: [blank; QUEUE_CAP] }
    }
    pub fn as_slice(&self) -> &[Action] {
        &self.items[..self.len as usize]
    }
    pub fn as_mut_slice(&mut self) -> &mut [Action] {
        let n = self.len as usize;
        &mut self.items[..n]
    }
    pub fn push(&mut self, a: Action) {
        self.items[self.len as usize] = a;
        self.len += 1;
    }
    pub fn insert(&mut self, at: usize, a: Action) {
        let n = self.len as usize;
        self.items.copy_within(at..n, at + 1);
        self.items[at] = a;
        self.len += 1;
    }
    pub fn shift(&mut self) -> Option<Action> {
        if self.len == 0 {
            return None;
        }
        let a = self.items[0];
        let n = self.len as usize;
        self.items.copy_within(1..n, 0);
        self.len -= 1;
        Some(a)
    }
    pub fn peek(&self) -> Option<&Action> {
        self.as_slice().first()
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
    /// Showdown's `cancelAction`: drop every queued action belonging to `mon`.
    pub fn cancel(&mut self, mon: MonRef) {
        let mut w = 0;
        for r in 0..self.len as usize {
            if self.items[r].mon != Some(mon) {
                self.items[w] = self.items[r];
                w += 1;
            }
        }
        self.len = w as u8;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Battle {
    pub rng: Rng,
    pub sides: [Side; 2],
    pub turn: u16,
    pub request: Request,
    pub ended: bool,
    /// Winning side once `ended`; `None` there means a tie.
    pub winner: Option<u8>,
    pub(crate) queue: Queue,
    pub(crate) faint_queue: [MonRef; 8],
    pub(crate) n_faint: u8,
    pub(crate) mid_turn: bool,
}
