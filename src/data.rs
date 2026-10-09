//! Static game data types. The tables themselves live in `tables.rs`, which is
//! generated from Pokémon Showdown's Champions data by `oracle/gen_data.js`.

pub use crate::tables::{
    ABILITIES, BOOST_ORDERS, ITEMS, MOVES, N_PSEUDO, N_SIDE_CONDS, N_SLOT_CONDS, N_VOLATILES, PSEUDO_CONDS, Pseudo,
    SIDE_CONDS, SLOT_CONDS, SPECIES, STATUS_CONDS, STATUS_IMMUNE, SideCond, SlotCond, TERRAIN_CONDS, TYPE_CHART,
    Terrain, VOL_CONDS, VolKind, WEATHER_CONDS, Weather, ab, it, mv,
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
pub const F_MUSTPRESSURE: u32 = 1 << 23;
pub const F_GRAVITY: u32 = 1 << 24;
pub const F_NONSKY: u32 = 1 << 25;
pub const F_MINIMIZE: u32 = 1 << 26;
pub const F_FAILENCORE: u32 = 1 << 27;
pub const F_NOSLEEPTALK: u32 = 1 << 28;
pub const F_FAILINSTRUCT: u32 = 1 << 29;
pub const F_FAILMIMIC: u32 = 1 << 30;
pub const F_CANTUSETWICE: u32 = 1 << 31;
// These two only matter to moves that are not in Champions (Assist, Me First).
pub const F_NOASSIST: u32 = 0;
pub const F_FAILMEFIRST: u32 = 0;

/// Stat stage changes in the order atk, def, spa, spd, spe, accuracy, evasion.
pub type Boosts = [i8; 7];

pub const ATK: usize = 0;
pub const DEF: usize = 1;
pub const SPA: usize = 2;
pub const SPD: usize = 3;
pub const SPE: usize = 4;
pub const ACC: usize = 5;
pub const EVA: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Secondary {
    /// Percent chance; 0 means "always" (no `chance` in Showdown).
    pub chance: u16,
    pub status: Status,
    pub boosts: Option<Boosts>,
    pub volatile: Option<VolKind>,
    pub self_boosts: Option<Boosts>,
    /// The secondary has a script callback of its own (written by hand, keyed by the move).
    pub on_hit: bool,
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
    /// The move's PP before any boost (what a copy made by Transform is capped against).
    pub base_pp: u8,
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
    /// Volatile the user gains (`self.volatileStatus`).
    pub self_volatile: Option<VolKind>,
    pub side_condition: Option<SideCond>,
    pub slot_condition: Option<SlotCond>,
    pub pseudo_weather: Option<Pseudo>,
    pub weather: Weather,
    pub terrain: Terrain,
    pub self_boosts: Option<Boosts>,
    pub self_chance: u8,
    /// The `self` block has a script callback of its own.
    pub self_on_hit: bool,
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
    /// A protecting move, whose success rate drops with consecutive use.
    pub stalling_move: bool,
    /// Removes Protect and its relatives from the target before hitting (Feint).
    pub breaks_protect: bool,
    /// Damage that ignores the usual formula.
    pub fixed_damage: FixedDamage,
    pub selfdestruct: SelfDestruct,
    /// Stat changes for the user once the whole move has gone through (Scale Shot).
    pub self_boost: Option<Boosts>,
    pub self_boost_order: u8,
    /// Costs the user half its HP, hit or miss (Steel Beam).
    pub mind_blown_recoil: bool,
    /// Goes for the Pokémon it was aimed at, wherever that has moved and whoever calls for attention.
    pub tracks_target: bool,
    /// Dragon Darts: one hit for each of two adjacent targets, or both hits for the one it can reach.
    pub smart_target: bool,
    /// Uses another move in its place (Copycat, Sleep Talk); effects that follow a Pokémon's moves look past it.
    pub calls_move: bool,
    /// Hurts the user when it misses (High Jump Kick); Reckless counts it as a recoil move.
    pub has_crash_damage: bool,
    /// Sheer Force boosts the move as it stands, taking nothing away (Electro Shot).
    pub sheer_force_boost: bool,
    pub self_switch: SelfSwitch,
    /// Drags the target out for a random teammate (Roar, Dragon Tail).
    pub force_switch: bool,
    /// Can be used while asleep.
    pub sleep_usable: bool,
    /// Each hit after the first checks accuracy again (Population Bomb).
    pub multiaccuracy: bool,
    pub ohko: Ohko,
    /// Struggle's recoil: a quarter of the user's max HP, not blockable.
    pub struggle_recoil: bool,
    /// Bit set of the events this move has a script callback of its own for
    /// (`onTry`, `onHit`, `basePowerCallback`, ...); the bodies are in `movecbs.rs`.
    pub events: u128,
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
    /// Index of the species' first ability: the one a forme change gives (a Mega's ability).
    pub ability0: u16,
    /// Weight in hectograms.
    pub weight_hg: u16,
    /// Not a Pokémon that can be in a Champions battle at all: a forme of a
    /// species the game does not have (Eiscue without its ice face).
    pub illegal: bool,
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
    // Lifecycle of conditions on the field and on a side.
    FieldStart,
    FieldEnd,
    FieldRestart,
    FieldResidual,
    SideStart,
    SideEnd,
    SideRestart,
    SideResidual,
    Copy,
    Swap,
    Type,
    // Script callbacks of moves (`onTry`, `basePowerCallback`, ...).
    Try,
    TryHitField,
    TryImmunity,
    HitField,
    HitSide,
    AfterHit,
    MoveFail,
    ModifyTarget,
    BasePowerCallback,
    DamageCallback,
    BeforeMoveCallback,
    BeforeTurnCallback,
    PriorityChargeCallback,
    // Not Showdown events: the `onHit` of a move's `self` block and of one of
    // its secondaries, which need a name to be filed under.
    SelfHit,
    SecondaryHit,
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
    /// Bit set of the events `cbs` listens to with prefix `On`.
    pub events: u128,
    /// Bit set of the events `cbs` listens to with any other prefix.
    pub events_pre: u128,
}

pub const IF_BERRY: u8 = 1 << 0;
pub const IF_GEM: u8 = 1 << 1;
pub const IF_CHOICE: u8 = 1 << 2;
pub const IF_IGNORE_KLUTZ: u8 = 1 << 3;
/// A Mega Stone that goes by exact forme: it cannot be taken from the formes it
/// evolves or from the Megas it makes (Floettite, Meowsticite).
pub const IF_MEGA_BY_FORME: u8 = 1 << 4;

#[derive(Clone, Copy, Debug)]
pub struct ItemData {
    pub id: &'static str,
    pub name: &'static str,
    pub flags: u8,
    pub supported: bool,
    pub cbs: &'static [CbInfo],
    pub events: u128,
    pub events_pre: u128,
    /// For a Mega Stone: (species that can use it, the Mega it becomes), as indices into `SPECIES`.
    pub mega: &'static [(u16, u16)],
    /// Stat changes the holder gets when the item is used up.
    pub boosts: Option<Boosts>,
    pub fling: Fling,
}

/// `item.fling`: what the item does when thrown with Fling.
#[derive(Clone, Copy, Debug)]
pub struct Fling {
    /// Fling's base power with this item; 0 for an item that cannot be thrown.
    pub power: u8,
    /// A status the item gives whatever it hits (Light Ball, Poison Barb).
    pub status: Status,
    /// King's Rock: the target flinches.
    pub flinch: bool,
    /// The item does something of its own to the target (the herbs).
    pub effect: bool,
}

/// `move.selfSwitch`: the user switches out after the move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SelfSwitch {
    No,
    Yes,
    /// Baton Pass: stat stages and most volatile conditions go to the replacement.
    CopyVolatile,
    /// Shed Tail: only the substitute does.
    ShedTail,
}

/// `move.damage`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FixedDamage {
    No,
    /// The user's level (Night Shade, Seismic Toss).
    Level,
}

/// `move.ohko`: a one-hit knockout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ohko {
    No,
    Yes,
    /// Sheer Cold: Ice types are immune, and it is less accurate from anything else.
    Ice,
}

/// `move.selfdestruct`: the user faints.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SelfDestruct {
    No,
    /// Before the move hits, whatever happens (Explosion).
    Always,
    /// Only if the move worked (Memento, Final Gambit).
    IfHit,
}

/// A status, a volatile, or a condition on a side, a position or the field.
#[derive(Clone, Copy, Debug)]
pub struct CondData {
    pub id: &'static str,
    /// Turns the condition lasts once added; 0 if it has no fixed duration.
    pub duration: u8,
    /// Can be added to a Pokémon with no HP left.
    pub affects_fainted: bool,
    /// Not passed on by Baton Pass.
    pub no_copy: bool,
    /// The duration is computed when the condition starts (`durationCallback`).
    pub duration_cb: bool,
    pub cbs: &'static [CbInfo],
    /// Bit set of the events `cbs` listens to with prefix `On`.
    pub events: u128,
    /// Bit set of the events `cbs` listens to with any other prefix.
    pub events_pre: u128,
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
