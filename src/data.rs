//! Static game data types. The tables themselves live in `tables.rs`, which is
//! generated from Pokémon Showdown's Champions data by `oracle/gen_data.js`.

pub use crate::tables::{
    ABILITIES, BOOST_ORDERS, ITEMS, MOVES, SPECIES, STATUS_CONDS, STATUS_IMMUNE, TYPE_CHART, VOL_CONDS, VolKind, ab, it,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Type {
    Normal = 0,
    Fighting,
    Flying,
    Poison,
    Ground,
    Rock,
    Bug,
    Ghost,
    Steel,
    Fire,
    Water,
    Grass,
    Electric,
    Psychic,
    Ice,
    Dragon,
    Dark,
    Fairy,
    /// Empty second type slot.
    None,
    /// Showdown's `???` type (Struggle): no STAB, neutral against everything.
    Typeless,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category {
    Physical,
    Special,
    Status,
}

/// Major status conditions. `None` is "healthy".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Status {
    None = 0,
    Brn,
    Par,
    Psn,
    Tox,
    Slp,
    Frz,
}

impl Status {
    pub fn id(self) -> &'static str {
        match self {
            Status::None => "",
            Status::Brn => "brn",
            Status::Par => "par",
            Status::Psn => "psn",
            Status::Tox => "tox",
            Status::Slp => "slp",
            Status::Frz => "frz",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Gender {
    /// Genderless.
    N = 0,
    M,
    F,
}

impl Gender {
    pub fn parse(s: &str) -> Option<Gender> {
        match s {
            "" | "N" => Some(Gender::N),
            "M" => Some(Gender::M),
            "F" => Some(Gender::F),
            _ => None,
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Gender::N => "N",
            Gender::M => "M",
            Gender::F => "F",
        }
    }
}

/// Showdown move target types.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    Normal,
    Any,
    AdjacentFoe,
    AllAdjacentFoes,
    AllAdjacent,
    /// Showdown's `self`.
    User,
    AdjacentAlly,
    AdjacentAllyOrSelf,
    Allies,
    RandomNormal,
    All,
    AllySide,
    FoeSide,
    AllyTeam,
    Scripted,
}

impl Target {
    /// Whether the player picks a target slot for this move in doubles
    /// (Showdown's `targetTypeChoices`).
    pub fn is_chosen(self) -> bool {
        matches!(
            self,
            Target::Normal | Target::Any | Target::AdjacentAlly | Target::AdjacentAllyOrSelf | Target::AdjacentFoe
        )
    }
}

/// Moves whose behaviour comes from script callbacks in Showdown and is
/// written by hand in the engine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Special {
    None,
    Protect,
    Struggle,
}

// Move flags (Showdown's `move.flags`). Only the ones some modelled effect reads.
pub const F_PROTECT: u32 = 1 << 0;
pub const F_POWDER: u32 = 1 << 1;
pub const F_DEFROST: u32 = 1 << 2;
pub const F_CONTACT: u32 = 1 << 3;
pub const F_SOUND: u32 = 1 << 4;
pub const F_HEAL: u32 = 1 << 5;
pub const F_PUNCH: u32 = 1 << 6;
pub const F_BITE: u32 = 1 << 7;
pub const F_BULLET: u32 = 1 << 8;
pub const F_PULSE: u32 = 1 << 9;
pub const F_SLICING: u32 = 1 << 10;
pub const F_WIND: u32 = 1 << 11;
pub const F_DANCE: u32 = 1 << 12;
pub const F_REFLECTABLE: u32 = 1 << 13;
pub const F_BYPASSSUB: u32 = 1 << 14;
pub const F_MIRROR: u32 = 1 << 15;
pub const F_SNATCH: u32 = 1 << 16;
pub const F_CHARGE: u32 = 1 << 17;
pub const F_RECHARGE: u32 = 1 << 18;
pub const F_FUTUREMOVE: u32 = 1 << 19;
pub const F_METRONOME: u32 = 1 << 20;
pub const F_NOPARENTALBOND: u32 = 1 << 21;
pub const F_FAILCOPYCAT: u32 = 1 << 22;

/// Stat stage changes in the order atk, def, spa, spd, spe, accuracy, evasion.
pub type Boosts = [i8; 7];

pub const ATK: usize = 0;
pub const DEF: usize = 1;
pub const SPA: usize = 2;
pub const SPD: usize = 3;
pub const SPE: usize = 4;
pub const ACC: usize = 5;
pub const EVA: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct Secondary {
    /// Percent chance; 0 means "always" (no `chance` in Showdown).
    pub chance: u16,
    pub status: Status,
    pub boosts: Option<Boosts>,
    pub volatile: Option<VolKind>,
    pub self_boosts: Option<Boosts>,
}

#[derive(Clone, Copy, Debug)]
pub struct MoveData {
    pub id: &'static str,
    pub name: &'static str,
    pub typ: Type,
    pub category: Category,
    pub base_power: u16,
    /// Percent accuracy; 0 means the move never checks accuracy.
    pub accuracy: u8,
    /// Maximum PP in a Champions battle.
    pub pp: u8,
    pub priority: i8,
    pub target: Target,
    pub crit_ratio: u8,
    pub will_crit: bool,
    pub flags: u32,
    pub boosts: Option<Boosts>,
    /// Index into `BOOST_ORDERS`: the order `boosts` is applied in.
    pub boost_order: u8,
    pub status: Status,
    pub volatile: Option<VolKind>,
    pub self_boosts: Option<Boosts>,
    pub self_chance: u8,
    pub secondaries: &'static [Secondary],
    pub drain: (u8, u8),
    pub recoil: (u8, u8),
    pub heal: (u8, u8),
    /// (min, max) hits; (0, 0) for ordinary single-hit moves.
    pub multihit: (u8, u8),
    /// Stat index (1 = atk .. 5 = spe) used for offence instead of the category default; 0 = default.
    pub off_stat: u8,
    /// Stat index used for defence instead of the category default; 0 = default.
    pub def_stat: u8,
    /// Foul Play: use the target's offensive stat.
    pub off_from_target: bool,
    pub ignore_defensive: bool,
    pub ignore_evasion: bool,
    pub thaws_target: bool,
    pub ignore_immunity: bool,
    pub special: Special,
    /// Whether the engine models every effect of this move.
    pub supported: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct SpeciesData {
    pub id: &'static str,
    pub name: &'static str,
    /// Id of the species this is a forme of (its own id for a base forme).
    pub base_species: &'static str,
    pub types: [Type; 2],
    /// Base stats in the order hp, atk, def, spa, spd, spe.
    pub base: [u8; 6],
    /// The gender every member of the species has, if it is fixed (`N` for
    /// genderless species); `None` if it can be either.
    pub gender: Option<Gender>,
}

// --------------------------------------------------------------------- events

/// Showdown event names. An effect's `onModifyAtk` callback is the handler for
/// `Ev::ModifyAtk` with prefix `Pre::On`; `onSourceModifyAtk` is the same event
/// with `Pre::Source`, and so on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Ev {
    // effect lifecycle
    Start,
    End,
    Restart,
    // switching
    BeforeSwitchIn,
    SwitchIn,
    BeforeSwitchOut,
    SwitchOut,
    // once per turn or per action
    BeforeTurn,
    Update,
    Residual,
    DisableMove,
    TrapPokemon,
    MaybeTrapPokemon,
    // action order
    ModifyPriority,
    FractionalPriority,
    // using a move
    BeforeMove,
    MoveAborted,
    ModifyType,
    ModifyMove,
    DeductPP,
    TryMove,
    PrepareHit,
    Invulnerability,
    TryHit,
    TryHitSide,
    HitProtect,
    ModifyAccuracy,
    Accuracy,
    TryPrimaryHit,
    ModifyCritRatio,
    CriticalHit,
    BasePower,
    ModifyAtk,
    ModifyDef,
    ModifySpA,
    ModifySpD,
    ModifySpe,
    ModifyBoost,
    ModifySTAB,
    Effectiveness,
    ModifyDamage,
    Damage,
    Hit,
    ModifySecondaries,
    DamagingHit,
    AfterMoveSecondary,
    AfterMoveSecondarySelf,
    AfterMove,
    RedirectTarget,
    StallMove,
    Flinch,
    // status, volatiles, stat stages, healing
    SetStatus,
    AfterSetStatus,
    TryAddVolatile,
    Immunity,
    ChangeBoost,
    TryBoost,
    AfterEachBoost,
    AfterBoost,
    TryHeal,
    Heal,
    // items
    UseItem,
    TryEatItem,
    Eat,
    EatItem,
    Use,
    AfterUseItem,
    TakeItem,
    // fainting
    BeforeFaint,
    Faint,
    AfterFaint,
    // Named by some modelled effect but never fired by a modelled mechanic yet.
    DragOut,
    EmergencyExit,
    AfterMega,
    AfterTerastallization,
    AfterSubDamage,
    TerrainChange,
    WeatherChange,
    Weather,
    ModifyWeight,
    WeatherModifyDamage,
    CheckShow,
    SetAbility,
    LockMove,
}

impl Ev {
    pub const fn bit(self) -> u128 {
        1u128 << (self as u8)
    }
}

/// Which Pokémon, relative to the event's target, a handler listens from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Pre {
    /// `onX`: the effect is on the event's target.
    On,
    /// `onAllyX`: the effect is on the target or its ally.
    Ally,
    /// `onFoeX`: the effect is on an opponent of the target.
    Foe,
    /// `onAnyX`: the effect is on any active Pokémon.
    Any,
    /// `onSourceX`: the effect is on the event's source.
    Source,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CbKind {
    /// An ordinary callback.
    Fn,
    /// Abilities and items without an `onSwitchIn` run their `onStart` during
    /// the SwitchIn event (Showdown's `getCallback`).
    StartAlias,
    /// No callback: the effect only takes part in Residual so that its
    /// duration counts down.
    DurationOnly,
}

/// One callback of an effect, with Showdown's ordering metadata.
#[derive(Clone, Copy, Debug)]
pub struct CbInfo {
    pub ev: Ev,
    pub pre: Pre,
    /// `onXOrder`; 0 means unset (sorts last).
    pub order: u32,
    /// `onXPriority`, times ten (a few are fractional).
    pub priority: i16,
    /// `onXSubOrder`, or Showdown's default for the effect type.
    pub sub_order: u8,
    pub kind: CbKind,
}

pub const AF_BREAKABLE: u8 = 1 << 0;
pub const AF_CANTSUPPRESS: u8 = 1 << 1;
pub const AF_NOTRANSFORM: u8 = 1 << 2;
pub const AF_FAILSKILLSWAP: u8 = 1 << 3;
pub const AF_FAILROLEPLAY: u8 = 1 << 4;
pub const AF_NORECEIVER: u8 = 1 << 5;
pub const AF_NOENTRAIN: u8 = 1 << 6;
pub const AF_NOTRACE: u8 = 1 << 7;

#[derive(Clone, Copy, Debug)]
pub struct AbilityData {
    pub id: &'static str,
    pub name: &'static str,
    pub flags: u8,
    pub supported: bool,
    pub cbs: &'static [CbInfo],
    /// Bit set of the events in `cbs`.
    pub events: u128,
}

pub const IF_BERRY: u8 = 1 << 0;
pub const IF_GEM: u8 = 1 << 1;
pub const IF_CHOICE: u8 = 1 << 2;
pub const IF_IGNORE_KLUTZ: u8 = 1 << 3;

#[derive(Clone, Copy, Debug)]
pub struct ItemData {
    pub id: &'static str,
    pub name: &'static str,
    pub flags: u8,
    pub supported: bool,
    pub cbs: &'static [CbInfo],
    pub events: u128,
}

/// A status or volatile condition.
#[derive(Clone, Copy, Debug)]
pub struct CondData {
    pub id: &'static str,
    /// Turns the condition lasts once added; 0 if it has no fixed duration.
    pub duration: u8,
    /// Can be added to a Pokémon with no HP left.
    pub affects_fainted: bool,
    pub cbs: &'static [CbInfo],
    pub events: u128,
}

/// Natures as (raised stat, lowered stat) using stat indices 1..=5; (0, 0) is neutral.
pub fn nature(name: &str) -> Option<(u8, u8)> {
    Some(match name.to_ascii_lowercase().as_str() {
        "hardy" | "docile" | "serious" | "bashful" | "quirky" => (0, 0),
        "lonely" => (1, 2),
        "brave" => (1, 5),
        "adamant" => (1, 3),
        "naughty" => (1, 4),
        "bold" => (2, 1),
        "relaxed" => (2, 5),
        "impish" => (2, 3),
        "lax" => (2, 4),
        "timid" => (5, 1),
        "hasty" => (5, 2),
        "jolly" => (5, 3),
        "naive" => (5, 4),
        "modest" => (3, 1),
        "mild" => (3, 2),
        "quiet" => (3, 5),
        "rash" => (3, 4),
        "calm" => (4, 1),
        "gentle" => (4, 2),
        "sassy" => (4, 5),
        "careful" => (4, 3),
        _ => return None,
    })
}

pub fn move_id(id: &str) -> Option<u16> {
    MOVES.binary_search_by(|m| m.id.cmp(id)).ok().map(|i| i as u16)
}

pub fn species_id(id: &str) -> Option<u16> {
    SPECIES.binary_search_by(|s| s.id.cmp(id)).ok().map(|i| i as u16)
}

pub fn ability_id(id: &str) -> Option<u16> {
    ABILITIES.binary_search_by(|a| a.id.cmp(id)).ok().map(|i| i as u16)
}

/// Item index by id; the empty id is "no item" (index 0).
pub fn item_id(id: &str) -> Option<u16> {
    ITEMS.binary_search_by(|a| a.id.cmp(id)).ok().map(|i| i as u16)
}

/// Showdown's `toID`: lower-case alphanumerics only.
pub fn to_id(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}
