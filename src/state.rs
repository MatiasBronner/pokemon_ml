//! Battle state. Everything here is plain fixed-size data so a `Battle` can be
//! copied cheaply (search wants thousands of copies per decision).

use crate::data::{
    Boosts, Category, Ev, Gender, MOVES, MoveData, Pseudo, Secondary, SideCond, SlotCond, Status, Target, Terrain,
    Type, VolKind, Weather,
};
use crate::rng::Rng;

/// Largest team a side can bring into battle.
pub const MAX_TEAM: usize = 6;
/// Active Pokémon per side (doubles).
pub const ACTIVE: usize = 2;
pub const MAX_MOVES: usize = 4;
/// Most volatile conditions one Pokémon can hold at once. Far more than any
/// real game state reaches; adding one beyond it fails like an immunity would.
pub const VOL_CAP: usize = 20;
/// "No position" in `Cond::source_slot`.
pub const NO_SLOT: u8 = u8::MAX;

/// "No species" in fields that hold an optional species index.
pub const NO_SPECIES: u16 = u16::MAX;

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
    /// Cannot be chosen this turn (Choice lock and the like). Recomputed every turn.
    pub disabled: bool,
}

/// Bookkeeping Showdown attaches to every effect instance (`EffectState`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct EffState {
    /// Showdown's `effectOrder`: a battle-wide counter value taken when the
    /// effect started on an active Pokémon, 0 otherwise. Breaks ties between
    /// switch-in and redirection handlers.
    pub order: u32,
    /// Identity of this instance. Showdown skips a queued handler when the
    /// state object it was collected for has since been replaced.
    pub uid: u16,
    /// Scratch values whose meaning depends on the effect (documented where used).
    pub a: i16,
    pub b: i16,
}

/// One instance of a condition: a volatile on a Pokémon, a condition on a
/// side or on one of its positions, a pseudo-weather, the weather or the
/// terrain. `K` is the enum naming the condition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cond<K: Copy> {
    pub kind: K,
    /// Turns left; 0 means the condition has no duration.
    pub duration: u8,
    /// Whatever the condition keeps in its state, documented where the
    /// condition is written (`conditions.rs`). For the first volatiles:
    /// `stall`: 1-in-`data` chance the next protecting move works.
    /// `choicelock`: the move table index the holder is locked into, plus one.
    /// `confusion`: turns of confusion left.
    /// `metronome`: the last move used (table index plus one); `st.a` counts consecutive uses.
    pub data: u16,
    /// The Pokémon that caused the condition (`effectState.source`).
    pub source: Option<MonRef>,
    /// Where that Pokémon stood at the time, as side * 2 + position (`effectState.sourceSlot`).
    pub source_slot: u8,
    /// Set once one of the condition's handlers has run inside an event.
    /// Showdown sorts a pseudo-weather's handlers differently from then on.
    pub(crate) targeted: bool,
    pub st: EffState,
}

impl<K: Copy> Cond<K> {
    pub(crate) const fn new(kind: K) -> Cond<K> {
        Cond {
            kind,
            duration: 0,
            data: 0,
            source: None,
            source_slot: NO_SLOT,
            targeted: false,
            st: EffState { order: 0, uid: 0, a: 0, b: 0 },
        }
    }
}

/// Conditions in the order they were added. Showdown iterates them in
/// insertion order, and that order decides how ties between their handlers
/// come out, so it is part of the state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CondList<K: Copy, const N: usize> {
    len: u8,
    items: [Cond<K>; N],
}

impl<K: Copy, const N: usize> CondList<K, N> {
    /// An empty list; `blank` only fills the unused storage.
    pub(crate) const fn new(blank: K) -> Self {
        CondList { len: 0, items: [Cond::new(blank); N] }
    }
}

impl<K: Copy + PartialEq, const N: usize> CondList<K, N> {
    pub fn as_slice(&self) -> &[Cond<K>] {
        &self.items[..self.len as usize]
    }
    pub fn get(&self, kind: K) -> Option<&Cond<K>> {
        self.as_slice().iter().find(|v| v.kind == kind)
    }
    pub fn get_mut(&mut self, kind: K) -> Option<&mut Cond<K>> {
        let n = self.len as usize;
        self.items[..n].iter_mut().find(|v| v.kind == kind)
    }
    pub fn has(&self, kind: K) -> bool {
        self.get(kind).is_some()
    }
    pub fn is_full(&self) -> bool {
        self.len as usize == N
    }
    pub(crate) fn push(&mut self, v: Cond<K>) {
        debug_assert!(!self.has(v.kind));
        self.items[self.len as usize] = v;
        self.len += 1;
    }
    pub(crate) fn remove(&mut self, kind: K) -> bool {
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
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }
    /// Remove every condition `pred` picks and return them, both lists keeping their order.
    pub(crate) fn take_where(&mut self, pred: impl Fn(K) -> bool) -> Self {
        let mut taken = Self { len: 0, items: self.items };
        let mut kept = 0;
        for i in 0..self.len as usize {
            let c = self.items[i];
            if pred(c.kind) {
                taken.items[taken.len as usize] = c;
                taken.len += 1;
            } else {
                self.items[kept] = c;
                kept += 1;
            }
        }
        self.len = kept as u8;
        taken
    }
}

/// A volatile condition on a Pokémon.
pub type Volatile = Cond<VolKind>;
/// The volatiles of the Pokémon in one active position. Only Pokémon on the
/// field have any, so they are stored per position rather than per Pokémon.
pub type Volatiles = CondList<VolKind, VOL_CAP>;
pub type SideConds = CondList<SideCond, { crate::data::N_SIDE_CONDS }>;
pub type SlotConds = CondList<SlotCond, { crate::data::N_SLOT_CONDS }>;
pub type PseudoWeathers = CondList<Pseudo, { crate::data::N_PSEUDO }>;

/// Conditions on the whole field (Showdown's `Field`).
#[derive(Clone, Copy, Debug)]
pub struct Field {
    /// `kind` is `Weather::None` when there is no weather.
    pub weather: Cond<Weather>,
    pub terrain: Cond<Terrain>,
    pub pseudo: PseudoWeathers,
}

/// `Pokemon#trapped`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trapped {
    No,
    Yes,
    /// Trapped by something the player has not been shown yet (Shadow Tag).
    Hidden,
}

#[derive(Clone, Copy, Debug)]
pub struct Pokemon {
    /// Current species (index into `data::SPECIES`); changes on Mega Evolution.
    pub species: u16,
    /// The species the Pokémon returns to when it leaves the field. Mega
    /// Evolution changes this too, so a Mega stays a Mega.
    pub base_species: u16,
    /// The Mega this Pokémon can still evolve into (`NO_SPECIES` if none).
    pub can_mega: u16,
    /// The set's nature and stat points, kept to recompute stats when the species changes.
    pub(crate) nature: (u8, u8),
    pub(crate) stat_points: [u8; 6],
    /// Current types; an effect can change them until the Pokémon leaves the field.
    pub types: [Type; 2],
    pub level: u8,
    pub gender: Gender,
    /// Unboosted stats: max HP, Atk, Def, SpA, SpD, Spe.
    pub stats: [u16; 6],
    pub hp: u16,
    pub status: Status,
    /// Sleep / freeze turn counter (Showdown's `statusState.time`).
    pub status_time: u8,
    /// Toxic damage stage (Showdown's `statusState.stage`).
    pub tox_stage: u8,
    pub status_st: EffState,
    /// Stat stages: atk, def, spa, spd, spe, accuracy, evasion.
    pub boosts: [i8; 7],
    pub moves: [MoveSlot; MAX_MOVES],
    pub n_moves: u8,
    /// Current ability; `base_ability` is restored when the Pokémon leaves the field.
    pub ability: u16,
    pub base_ability: u16,
    pub ability_st: EffState,
    /// Stat stages an ability is holding on to (Opportunist's pending copies).
    pub(crate) ability_boosts: [i8; 7],
    /// Supersweet Syrup has already gone off once this battle.
    pub(crate) syrup_triggered: bool,
    /// Whether anything has attacked this Pokémon since it came in, and the
    /// damage of the latest attack (Showdown's `getLastAttackedBy`).
    pub(crate) was_attacked: bool,
    pub(crate) last_attack_damage: i32,
    /// Held item; 0 is none.
    pub item: u16,
    pub item_st: EffState,
    /// The item most recently used up or eaten.
    pub last_item: u16,
    pub used_item_this_turn: bool,
    pub ate_berry: bool,
    /// Index into the side's current order; positions below `ACTIVE` are on the field.
    pub position: u8,
    pub is_active: bool,
    pub fainted: bool,
    pub faint_queued: bool,
    /// Must be replaced at the next switch request.
    pub switch_flag: bool,
    /// Cannot switch out this turn.
    pub trapped: Trapped,
    /// Full turns spent on the field since switching in.
    pub active_turns: u16,
    /// Showdown's `moveThisTurnResult` / `moveLastTurnResult`: `Undef` (did not
    /// move), `Null`, or a boolean for whether the move did anything.
    pub(crate) move_this_turn: Res,
    pub(crate) move_last_turn: Res,
    /// Speed as last cached by Showdown's `updateSpeed`; several orderings read
    /// this stale value rather than the live stat.
    pub speed: i32,
}

impl Pokemon {
    pub fn max_hp(&self) -> u16 {
        self.stats[0]
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
    /// How many of this side's Pokémon have fainted so far.
    pub total_fainted: u8,
    /// Conditions on the whole side (Tailwind, screens, hazards).
    pub conds: SideConds,
    /// Conditions on each active position (Wish).
    pub slot_conds: [SlotConds; ACTIVE],
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
    /// `mega`: Mega Evolve before moving.
    Move {
        slot: u8,
        target: i8,
        mega: bool,
    },
    /// Switch to the Pokémon currently at this position (0-based) in the side's order.
    Switch {
        to: u8,
    },
}

impl Choice {
    /// A move choice without Mega Evolution.
    pub fn mv(slot: u8, target: i8) -> Choice {
        Choice::Move { slot, target, mega: false }
    }

    /// The choice in Showdown's notation: `move 2 1`, `move 3`, `move 1 2 mega`, `switch 4`, `pass`.
    pub fn to_showdown(self) -> String {
        match self {
            Choice::Pass => "pass".into(),
            Choice::Move { slot, target, mega } => {
                let mut s = format!("move {}", slot + 1);
                if target != 0 {
                    s.push_str(&format!(" {target}"));
                }
                if mega {
                    s.push_str(" mega");
                }
                s
            }
            Choice::Switch { to } => format!("switch {}", to + 1),
        }
    }

    /// Parses one slot's choice from Showdown's notation.
    pub fn parse(s: &str) -> Option<Choice> {
        let mut parts: Vec<&str> = s.split_whitespace().collect();
        let mega = parts.last() == Some(&"mega");
        if mega {
            parts.pop();
        }
        let mut parts = parts.into_iter();
        let kind = parts.next()?;
        let mut num = || parts.next().and_then(|p| p.parse::<i32>().ok());
        if mega && kind != "move" {
            return None;
        }
        match kind {
            "pass" => Some(Choice::Pass),
            "move" => {
                let slot = num()?;
                let target = num().unwrap_or(0);
                (1..=MAX_MOVES as i32).contains(&slot).then_some(Choice::Move {
                    slot: (slot - 1) as u8,
                    target: target as i8,
                    mega,
                })
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
    MegaEvo,
    Move,
    Residual,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Action {
    pub kind: ActKind,
    pub order: u32,
    /// Priority in tenths: ten times the move's priority plus `frac`.
    pub priority: i32,
    pub speed: i32,
    pub mon: Option<MonRef>,
    /// Move id for `Move`.
    pub move_id: u16,
    pub target_loc: i8,
    /// Incoming Pokémon for `Switch` / `InstaSwitch`.
    pub switch_to: Option<MonRef>,
    /// Showdown's `fractionalPriority` in tenths (Quick Claw and the like).
    pub frac: i8,
    /// The move's priority after `ModifyPriority` (Showdown stores it on the queued move).
    pub move_priority: i8,
    /// Set once Prankster has raised this move's priority; it stays set.
    pub prankster: bool,
    /// The Pokémon in the targeted slot when the move was chosen.
    pub orig_target: Option<MonRef>,
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
            frac: 0,
            move_priority: 0,
            prankster: false,
            orig_target: None,
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
    /// Showdown's `cancelMove`: drop the queued move of `mon`.
    pub fn cancel_move(&mut self, mon: MonRef) -> bool {
        let n = self.len as usize;
        match self.items[..n].iter().position(|a| a.kind == ActKind::Move && a.mon == Some(mon)) {
            Some(i) => {
                self.items.copy_within(i + 1..n, i);
                self.len -= 1;
                true
            }
            None => false,
        }
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

/// Any effect that can cause something or own an event handler (Showdown's `Effect`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Eff {
    None,
    /// A move in use; the index is its slot in `Battle::am`.
    Move(u8),
    Status(Status),
    Vol(VolKind),
    Ability(u16),
    Item(u16),
    Weather(Weather),
    Terrain(Terrain),
    Pseudo(Pseudo),
    SideCond(SideCond),
    SlotCond(SlotCond),
    /// The pseudo-conditions Showdown names `recoil`, `drain` and `strugglerecoil`.
    Recoil,
    Drain,
    StruggleRecoil,
    /// The stand-in move Showdown uses for confusion self-damage: it counts
    /// as a move for effects that ask, but has no data of its own.
    Confused,
}

impl Eff {
    /// Showdown's `effect.effectType === 'Move'`.
    pub fn is_move(self) -> bool {
        matches!(self, Eff::Move(_) | Eff::Confused)
    }
}

/// What an effect sits on (Showdown's `effectHolder`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Holder {
    Mon(MonRef),
    Side(u8),
    Field,
}

/// Showdown callbacks return numbers, booleans, `undefined`, `null` or `''`
/// (`NOT_FAIL`), and the caller branches on which one it got. This is that value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Res {
    Undef,
    NotFail,
    Null,
    Bool(bool),
    Num(i32),
    Mon(MonRef),
}

pub(crate) const TRUE: Res = Res::Bool(true);
pub(crate) const FALSE: Res = Res::Bool(false);

impl Res {
    pub fn truthy(self) -> bool {
        match self {
            Res::Bool(b) => b,
            Res::Num(n) => n != 0,
            Res::Mon(_) => true,
            _ => false,
        }
    }
    fn rank(self) -> u8 {
        match self {
            Res::Undef => 0,
            Res::NotFail => 1,
            Res::Null => 2,
            Res::Bool(_) => 3,
            Res::Num(_) | Res::Mon(_) => 4,
        }
    }
    /// "Counts as a hit": truthy, or exactly 0 damage.
    pub fn hit(self) -> bool {
        self.truthy() || self == Res::Num(0)
    }
    pub fn num(self) -> i32 {
        match self {
            Res::Num(n) => n,
            _ => 0,
        }
    }
    /// `BattleActions#combineResults`, branch for branch.
    #[allow(clippy::if_same_then_else)]
    pub fn combine(self, right: Res) -> Res {
        let left = self;
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
}

/// Kinds of thing `Pokemon#runStatusImmunity` can be asked about.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Imm {
    Status(Status),
    Vol(VolKind),
    Powder,
    Trapped,
    /// Damage from a weather (only sandstorm deals any).
    Weather(Weather),
}

/// The event being run (Showdown's `battle.event`), including the values that
/// travel with it as its relay variable.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Event {
    /// `None` outside any event.
    pub id: Option<Ev>,
    pub target: Option<MonRef>,
    /// The side, for the few events whose target is a side rather than a Pokémon.
    pub target_side: Option<u8>,
    pub source: Option<MonRef>,
    /// The effect that caused the event (`sourceEffect`).
    pub effect: Eff,
    /// Accumulated `chainModify` multiplier in 4096ths.
    pub modifier: u32,
    /// The relay variable when it is a number or boolean.
    pub relay: Res,
    /// Relay for the stat-stage events.
    pub boosts: Boosts,
    /// Relay for `SetStatus` / `AfterSetStatus`.
    pub status: Status,
    /// Relay for `TryAddVolatile`.
    pub vol: Option<VolKind>,
    /// Relay for the item events.
    pub item: u16,
    /// Relay for `Immunity`.
    pub imm: Option<Imm>,
    /// The defending type an `Effectiveness` event is about.
    pub typ: Type,
    /// Relay for `ModifySecondaries`: bit i set = secondary i still applies.
    pub secs: u8,
}

impl Event {
    pub const NONE: Event = Event {
        id: None,
        target: None,
        target_side: None,
        source: None,
        effect: Eff::None,
        modifier: 4096,
        relay: Res::Undef,
        boosts: [0; 7],
        status: Status::None,
        vol: None,
        item: 0,
        imm: None,
        typ: Type::None,
        secs: 0,
    };
    pub fn new(id: Ev, target: Option<MonRef>, source: Option<MonRef>, effect: Eff) -> Event {
        Event { id: Some(id), target, source, effect, ..Event::NONE }
    }
}

/// Showdown's `move.ignoreImmunity`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum IgnoreImm {
    No,
    All,
    /// Scrappy: Normal and Fighting moves hit Ghost types.
    NormalFighting,
}

/// Per-target results of a move (`Pokemon#getMoveHitData`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct HitData {
    pub crit: bool,
    pub type_mod: i8,
    pub bypass_protect: bool,
}

pub(crate) const MAX_SECS: usize = 3;

/// The mutable per-use copy of a move (Showdown's `ActiveMove`). Handlers
/// change its type, flags, secondaries and so on while it is being used.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ActiveMove {
    pub id: u16,
    pub typ: Type,
    pub category: Category,
    pub base_power: u16,
    /// 0 = never checks accuracy.
    pub accuracy: u8,
    pub priority: i8,
    pub target: Target,
    pub flags: u32,
    pub crit_ratio: u8,
    /// `None` is Showdown's `undefined`: roll for a critical hit.
    pub will_crit: Option<bool>,
    pub multihit: (u8, u8),
    /// The stat changes the move makes to its target (Growth doubles its own in the sun).
    pub boosts: Option<Boosts>,
    pub secs: [Secondary; MAX_SECS],
    pub n_secs: u8,
    /// Whether `secondaries` still exists (Sheer Force deletes it).
    pub has_secs: bool,
    pub self_boosts: Option<Boosts>,
    pub self_chance: u8,
    pub ignore_ability: bool,
    pub ignore_immunity: IgnoreImm,
    pub ignore_evasion: bool,
    pub ignore_defensive: bool,
    /// `overrideOffensiveStat` (0 = the category's own); Wonder Room can flip it.
    pub off_stat: u8,
    pub has_sheer_force: bool,
    pub prankster_boosted: bool,
    pub tracks_target: bool,
    pub infiltrates: bool,
    pub has_bounced: bool,
    /// The ability that changed this move's type and boosts it (Pixilate and so on).
    pub type_changer_boosted: Eff,
    /// The effect that called this move, if it was not chosen directly.
    pub source_effect: Eff,
    pub spread_hit: bool,
    pub self_dropped: bool,
    pub total_damage: i32,
    pub hit: u8,
    pub last_hit: bool,
    /// Indexed by side * 2 + position.
    pub hit_data: [HitData; 4],
    /// Parental Bond made this a two-hit move (`multihitType`).
    pub parental_bond: bool,
    /// The Pokémon whose Fairy Aura boosts this move.
    pub aura_booster: Option<MonRef>,
    /// The targets the move ended up hitting (`hitTargets`), once known.
    pub hit_targets: [MonRef; 3],
    pub n_hit_targets: u8,
    pub has_hit_targets: bool,
}

impl ActiveMove {
    pub fn d(&self) -> &'static MoveData {
        &MOVES[self.id as usize]
    }
    /// `dex.getActiveMove`: a fresh copy of the move's data.
    pub fn new(id: u16) -> ActiveMove {
        let d = &MOVES[id as usize];
        let blank = Secondary {
            chance: 0,
            status: Status::None,
            boosts: None,
            volatile: None,
            self_boosts: None,
            on_hit: false,
        };
        let mut secs = [blank; MAX_SECS];
        for (i, s) in d.secondaries.iter().enumerate() {
            secs[i] = *s;
        }
        ActiveMove {
            id,
            typ: d.typ,
            category: d.category,
            base_power: d.base_power,
            accuracy: d.accuracy,
            priority: d.priority,
            target: d.target,
            flags: d.flags,
            crit_ratio: d.crit_ratio,
            will_crit: if d.will_crit { Some(true) } else { None },
            multihit: d.multihit,
            boosts: d.boosts,
            secs,
            n_secs: d.secondaries.len() as u8,
            has_secs: !d.secondaries.is_empty(),
            self_boosts: d.self_boosts,
            self_chance: d.self_chance,
            ignore_ability: false,
            // Showdown resolves the default (status moves ignore type immunity) when it loads the move.
            ignore_immunity: if d.ignore_immunity { IgnoreImm::All } else { IgnoreImm::No },
            ignore_evasion: d.ignore_evasion,
            ignore_defensive: d.ignore_defensive,
            off_stat: d.off_stat,
            has_sheer_force: false,
            prankster_boosted: false,
            tracks_target: false,
            infiltrates: false,
            has_bounced: false,
            type_changer_boosted: Eff::None,
            source_effect: Eff::None,
            spread_hit: false,
            self_dropped: false,
            total_damage: 0,
            hit: 0,
            last_hit: false,
            hit_data: [HitData { crit: false, type_mod: 0, bypass_protect: false }; 4],
            parental_bond: false,
            aura_booster: None,
            hit_targets: [MonRef { side: 0, idx: 0 }; 3],
            n_hit_targets: 0,
            has_hit_targets: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FaintEntry {
    pub target: MonRef,
    pub source: Option<MonRef>,
    pub effect: Eff,
}

/// Active-move slots kept per action; nested move use (a bounced move) takes another slot.
pub(crate) const AM_CAP: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct Battle {
    pub rng: Rng,
    pub sides: [Side; 2],
    pub field: Field,
    /// Volatile conditions, indexed by side and active position.
    pub(crate) vols: [[Volatiles; ACTIVE]; 2],
    pub turn: u16,
    pub request: Request,
    pub ended: bool,
    /// Winning side once `ended`; `None` there means a tie.
    pub winner: Option<u8>,
    pub(crate) queue: Queue,
    pub(crate) faint_queue: [FaintEntry; 8],
    pub(crate) n_faint: u8,
    pub(crate) mid_turn: bool,
    /// Showdown's `effectOrder` counter.
    pub(crate) effect_order: u32,
    pub(crate) next_uid: u16,
    /// Events some effect in this battle can listen to; the rest are skipped outright.
    pub(crate) event_mask: u128,
    /// The subset something listens to from the side (onAlly, onFoe, onAny, onSource).
    pub(crate) event_mask_pre: u128,
    pub(crate) event: Event,
    /// The effect whose handler is running (`battle.effect`) and the Pokémon it is on.
    pub(crate) effect: Eff,
    pub(crate) effect_holder: Option<Holder>,
    pub(crate) event_depth: u8,
    pub(crate) am: [ActiveMove; AM_CAP],
    pub(crate) am_len: u8,
    /// `battle.activeMove` as a slot in `am`, with `activePokemon` and `activeTarget`.
    pub(crate) active_move: Option<u8>,
    pub(crate) active_pokemon: Option<MonRef>,
    pub(crate) active_target: Option<MonRef>,
    /// Field positions (side + 2 * position) in the speed order fixed at the last switch-in.
    pub(crate) speed_order: [u8; 4],
    pub(crate) n_speed_order: u8,
}
