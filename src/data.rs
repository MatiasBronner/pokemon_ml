//! Static game data types. The tables themselves live in `tables.rs`, which is
//! generated from Pokémon Showdown's Champions data by `oracle/gen_data.js`.

pub use crate::tables::{MOVES, SPECIES, STATUS_IMMUNE, TYPE_CHART};

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

pub const F_PROTECT: u16 = 1 << 0;
pub const F_POWDER: u16 = 1 << 1;
pub const F_DEFROST: u16 = 1 << 2;
pub const F_CONTACT: u16 = 1 << 3;
pub const F_SOUND: u16 = 1 << 4;
pub const F_HEAL: u16 = 1 << 5;

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
    pub chance: u8,
    pub status: Status,
    pub boosts: Option<Boosts>,
    pub flinch: bool,
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
    pub flags: u16,
    pub boosts: Option<Boosts>,
    pub status: Status,
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
    pub types: [Type; 2],
    /// Base stats in the order hp, atk, def, spa, spd, spe.
    pub base: [u8; 6],
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

/// Showdown's `toID`: lower-case alphanumerics only.
pub fn to_id(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}
