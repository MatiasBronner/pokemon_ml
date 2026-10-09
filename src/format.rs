//! What a regulation allows: which Pokémon can be brought, with which moves,
//! abilities and items, and the rules a team has to meet.
//!
//! The simulator itself plays any team ([`Battle::new`](crate::Battle::new)
//! does not ask whether a Pokémon could really know a move). Legality is a
//! separate question with a separate answer per regulation, and the answer is
//! data: `formats/<id>.json`, written by `oracle/gen_format.js` from what
//! Pokémon Showdown's team validator accepts. This module reads such a file
//! and checks teams against it.
//!
//! ```
//! use vgc_engine::format::Format;
//! use vgc_engine::PokemonSet;
//!
//! let format = Format::current();
//! let garchomp = PokemonSet::from_names("Garchomp", &["Earthquake", "Moonblast"], "Jolly", [0, 32, 0, 0, 2, 32])
//!     .and_then(|set| set.ability("Rough Skin"))
//!     .unwrap();
//! let problems = format.check_set(&garchomp);
//! assert_eq!(problems.len(), 1);
//! assert_eq!(problems[0].to_string(), "Garchomp cannot learn Moonblast");
//! ```
//!
//! Moving to another regulation is a matter of generating its file and
//! loading it with [`Format::from_json`]; see the README.

use std::fmt;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::battle::{Error, PokemonSet};
use crate::data::*;
use crate::rng::Rng;

/// The rules a whole team has to meet.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct TeamRules {
    /// How many Pokémon a team has, at least and at most.
    pub min_size: u8,
    pub max_size: u8,
    /// How many of them are picked for a battle at team preview.
    pub picked: u8,
    /// The level everyone is set to.
    pub level: u8,
    /// No two Pokémon with the same Pokédex number.
    pub species_clause: bool,
    /// How many Pokémon may hold the same item; 0 for no limit.
    pub item_clause: u8,
    /// Stat points a Pokémon may have in all, and in one stat.
    pub stat_points_total: u16,
    pub stat_points_per_stat: u8,
}

/// What one species may be brought with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeciesRule {
    /// Index into `data::SPECIES`.
    pub species: u16,
    /// Pokédex number: the Species Clause counts formes of one Pokémon as the same.
    pub num: u16,
    /// Move, ability and item indices into the engine's tables, each sorted.
    pub moves: Vec<u16>,
    pub abilities: Vec<u16>,
    pub genders: Vec<Gender>,
    /// The Mega Stones that work for it.
    pub megas: Vec<u16>,
    /// Items it may hold beyond the ones anyone may.
    pub items: Vec<u16>,
}

/// A regulation.
#[derive(Clone, Debug)]
pub struct Format {
    /// Showdown's format id, e.g. `gen9championsvgc2026regmc`.
    pub id: String,
    pub name: String,
    /// The Showdown commit the file was generated from.
    pub showdown_commit: String,
    pub team: TeamRules,
    /// Where this regulation's data differs from the data the engine was
    /// built from in something the simulator computes with (a move's PP,
    /// say). Empty for the regulation the engine is built for. If it is not
    /// empty, teams can be checked but battles are not simulated exactly.
    pub differences: Vec<String>,
    species: Vec<SpeciesRule>,
    /// Position in `species`, plus one, by species index; 0 for a species that cannot be brought.
    by_species: Vec<u16>,
    /// Items anyone may hold, sorted.
    items: Vec<u16>,
}

/// One thing wrong with a team. `slot` counts the team's Pokémon from 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Violation {
    TeamSize {
        size: usize,
        min: u8,
        max: u8,
    },
    /// A name the engine's tables do not have at all.
    Unknown {
        slot: usize,
        what: &'static str,
        name: String,
    },
    Species {
        slot: usize,
        species: u16,
    },
    DuplicateSpecies {
        slot: usize,
        other: usize,
        species: u16,
    },
    NoMoves {
        slot: usize,
        species: u16,
    },
    TooManyMoves {
        slot: usize,
        species: u16,
        count: usize,
    },
    DuplicateMove {
        slot: usize,
        species: u16,
        move_id: u16,
    },
    Move {
        slot: usize,
        species: u16,
        move_id: u16,
    },
    Ability {
        slot: usize,
        species: u16,
        ability: u16,
    },
    Item {
        slot: usize,
        species: u16,
        item: u16,
    },
    /// More Pokémon hold this item than the Item Clause allows.
    DuplicateItem {
        item: u16,
        holders: usize,
        limit: u8,
    },
    Gender {
        slot: usize,
        species: u16,
        gender: Gender,
    },
    /// More than the cap in one stat (`stat` is 0..6: hp, atk, def, spa, spd, spe), or a negative number.
    StatPoints {
        slot: usize,
        species: u16,
        stat: usize,
        points: i32,
    },
    StatPointTotal {
        slot: usize,
        species: u16,
        total: i32,
        limit: u16,
    },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sp = |s: &u16| SPECIES[*s as usize].name;
        match self {
            Violation::TeamSize { size, min, max } if min == max => write!(f, "a team has {min} Pokémon, not {size}"),
            Violation::TeamSize { size, min, max } => write!(f, "a team has {min} to {max} Pokémon, not {size}"),
            Violation::Unknown { slot, what, name } => write!(f, "Pokémon {}: unknown {what} {name:?}", slot + 1),
            Violation::Species { species, .. } => write!(f, "{} cannot be brought", sp(species)),
            Violation::DuplicateSpecies { species, .. } => {
                write!(f, "two Pokémon with the same Pokédex number ({})", sp(species))
            }
            Violation::NoMoves { species, .. } => write!(f, "{} has no moves", sp(species)),
            Violation::TooManyMoves { species, count, .. } => write!(f, "{} has {count} moves", sp(species)),
            Violation::DuplicateMove { species, move_id, .. } => {
                write!(f, "{} has {} twice", sp(species), MOVES[*move_id as usize].name)
            }
            Violation::Move { species, move_id, .. } => {
                write!(f, "{} cannot learn {}", sp(species), MOVES[*move_id as usize].name)
            }
            Violation::Ability { species, ability, .. } if *ability == ab::NOABILITY => {
                write!(f, "{} has no ability", sp(species))
            }
            Violation::Ability { species, ability, .. } => {
                write!(f, "{} cannot have {}", sp(species), ABILITIES[*ability as usize].name)
            }
            Violation::Item { species, item, .. } => {
                write!(f, "{} cannot hold {}", sp(species), ITEMS[*item as usize].name)
            }
            Violation::DuplicateItem { item, holders, limit } => {
                write!(f, "{holders} Pokémon hold {}; the limit is {limit}", ITEMS[*item as usize].name)
            }
            Violation::Gender { species, gender, .. } => write!(f, "{} cannot be {}", sp(species), gender.id()),
            Violation::StatPoints { species, stat, points, .. } => {
                const STATS: [&str; 6] = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"];
                write!(f, "{} has {points} stat points in {}", sp(species), STATS[*stat])
            }
            Violation::StatPointTotal { species, total, limit, .. } => {
                write!(f, "{} has {total} stat points in all; the limit is {limit}", sp(species))
            }
        }
    }
}

/// A Pokémon as Showdown's team JSON has it: names rather than table
/// indices, and stat points in the `evs` field. This is what
/// [`showdown_team`] writes and [`Format::check_showdown_team`] reads.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ShowdownSet {
    #[serde(default, deserialize_with = "or_default")]
    pub name: String,
    pub species: String,
    #[serde(default, deserialize_with = "or_default")]
    pub item: String,
    #[serde(default, deserialize_with = "or_default")]
    pub ability: String,
    #[serde(default, deserialize_with = "or_default")]
    pub moves: Vec<String>,
    #[serde(default, deserialize_with = "or_default")]
    pub nature: String,
    #[serde(default, deserialize_with = "or_default")]
    pub gender: String,
    #[serde(default, deserialize_with = "or_default")]
    pub evs: ShowdownStats,
}

/// Showdown writes `null` for a field it has nothing to say about.
fn or_default<'de, D: serde::Deserializer<'de>, T: Deserialize<'de> + Default>(d: D) -> Result<T, D::Error> {
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ShowdownStats {
    #[serde(default)]
    pub hp: i32,
    #[serde(default)]
    pub atk: i32,
    #[serde(default)]
    pub def: i32,
    #[serde(default)]
    pub spa: i32,
    #[serde(default)]
    pub spd: i32,
    #[serde(default)]
    pub spe: i32,
}

const NEUTRAL: &str = "Hardy";
const NATURE_NAMES: [&str; 20] = [
    "Lonely", "Brave", "Adamant", "Naughty", "Bold", "Relaxed", "Impish", "Lax", "Timid", "Hasty", "Jolly", "Naive",
    "Modest", "Mild", "Quiet", "Rash", "Calm", "Gentle", "Sassy", "Careful",
];

impl ShowdownSet {
    /// A set in Showdown's terms. A neutral nature is written as Hardy
    /// (Showdown reads "Serious" with no stat points as a set its player forgot to finish).
    pub fn from_set(set: &PokemonSet) -> ShowdownSet {
        let p = set.stat_points.map(|v| v as i32);
        ShowdownSet {
            name: String::new(),
            species: SPECIES[set.species as usize].name.to_string(),
            item: ITEMS[set.item as usize].name.to_string(),
            ability: if set.ability == ab::NOABILITY {
                String::new()
            } else {
                ABILITIES[set.ability as usize].name.to_string()
            },
            moves: set.moves.iter().map(|&m| MOVES[m as usize].name.to_string()).collect(),
            nature: NATURE_NAMES.iter().copied().find(|n| nature(n) == Some(set.nature)).unwrap_or(NEUTRAL).to_string(),
            gender: set.gender.id().to_string(),
            evs: ShowdownStats { hp: p[0], atk: p[1], def: p[2], spa: p[3], spd: p[4], spe: p[5] },
        }
    }

    /// The engine's set, if every name is one the engine knows and the stat points fit in a byte.
    pub fn to_set(&self) -> Result<PokemonSet, Error> {
        let (candidate, problems) = Candidate::from_showdown(0, self);
        match problems.first() {
            Some(p) => Err(Error::BadTeam(p.to_string())),
            None => candidate
                .into_set()
                .ok_or_else(|| Error::BadTeam(format!("{}: stat points out of range", self.species))),
        }
    }
}

/// A team as the JSON Showdown takes (`Teams.pack(JSON.parse(...))` on its side).
pub fn showdown_team(team: &[PokemonSet]) -> String {
    let sets: Vec<ShowdownSet> = team.iter().map(ShowdownSet::from_set).collect();
    serde_json::to_string(&sets).expect("a team is always serialisable")
}

/// A set with whatever could be made of its names, so that one bad name does
/// not hide the other problems.
struct Candidate {
    species: Option<u16>,
    moves: Vec<Option<u16>>,
    ability: Option<u16>,
    item: Option<u16>,
    nature: Option<(u8, u8)>,
    gender: Option<Gender>,
    stat_points: [i32; 6],
}

impl Candidate {
    fn from_set(set: &PokemonSet) -> Candidate {
        Candidate {
            species: Some(set.species),
            moves: set.moves.iter().map(|&m| Some(m)).collect(),
            ability: Some(set.ability),
            item: Some(set.item),
            nature: Some(set.nature),
            gender: Some(set.gender),
            stat_points: set.stat_points.map(|p| p as i32),
        }
    }

    fn from_showdown(slot: usize, s: &ShowdownSet) -> (Candidate, Vec<Violation>) {
        let mut problems = Vec::new();
        let mut look = |what: &'static str, name: &str, found: Option<u16>| {
            if found.is_none() {
                problems.push(Violation::Unknown { slot, what, name: name.to_string() });
            }
            found
        };
        let species = look("species", &s.species, species_id(&to_id(&s.species)));
        let moves = s.moves.iter().map(|m| look("move", m, move_id(&to_id(m)))).collect();
        let ability = if s.ability.is_empty() {
            Some(ab::NOABILITY)
        } else {
            look("ability", &s.ability, ability_id(&to_id(&s.ability)))
        };
        let item = look("item", &s.item, item_id(&to_id(&s.item)));
        let nature_pair = if s.nature.is_empty() { Some((0, 0)) } else { nature(&s.nature) };
        if nature_pair.is_none() {
            problems.push(Violation::Unknown { slot, what: "nature", name: s.nature.clone() });
        }
        // An unspecified gender is the species' own, or male where it can be either.
        let gender = match (s.gender.as_str(), species) {
            ("", Some(sp)) => Some(SPECIES[sp as usize].gender.unwrap_or(Gender::M)),
            (g, _) => Gender::parse(g),
        };
        if gender.is_none() && species.is_some() {
            problems.push(Violation::Unknown { slot, what: "gender", name: s.gender.clone() });
        }
        let e = &s.evs;
        let candidate = Candidate {
            species,
            moves,
            ability,
            item,
            nature: nature_pair,
            gender,
            stat_points: [e.hp, e.atk, e.def, e.spa, e.spd, e.spe],
        };
        (candidate, problems)
    }

    fn into_set(self) -> Option<PokemonSet> {
        let mut stat_points = [0u8; 6];
        for (k, &p) in self.stat_points.iter().enumerate() {
            stat_points[k] = u8::try_from(p).ok()?;
        }
        Some(PokemonSet {
            species: self.species?,
            moves: self.moves.into_iter().collect::<Option<Vec<u16>>>()?,
            nature: self.nature?,
            stat_points,
            ability: self.ability?,
            item: self.item?,
            gender: self.gender?,
        })
    }
}

/// Looks the ids of a regulation file up in the engine's tables, keeping note of the ones that are not there.
#[derive(Default)]
struct Names {
    unknown: Vec<String>,
}

impl Names {
    fn found(&mut self, kind: &str, id: &str, index: Option<u16>) -> u16 {
        index.unwrap_or_else(|| {
            self.unknown.push(format!("{kind} {id}"));
            0
        })
    }
    fn species(&mut self, id: &str) -> u16 {
        self.found("species", id, species_id(id).filter(|&i| !SPECIES[i as usize].illegal))
    }
    fn mv(&mut self, id: &str) -> u16 {
        self.found("move", id, move_id(id).filter(|&i| MOVES[i as usize].supported))
    }
    fn ability(&mut self, id: &str) -> u16 {
        self.found("ability", id, ability_id(id).filter(|&i| ABILITIES[i as usize].supported))
    }
    fn item(&mut self, id: &str) -> u16 {
        self.found("item", id, item_id(id).filter(|&i| ITEMS[i as usize].supported))
    }
}

#[derive(Deserialize)]
struct FormatFile {
    id: String,
    name: String,
    showdown_commit: String,
    game_type: String,
    team: TeamRules,
    differences: Vec<String>,
    items: Vec<String>,
    species: Vec<SpeciesFile>,
}

#[derive(Deserialize)]
struct SpeciesFile {
    id: String,
    num: u16,
    moves: Vec<String>,
    abilities: Vec<String>,
    genders: Vec<String>,
    megas: Vec<String>,
    #[serde(default)]
    items: Vec<String>,
}

impl Format {
    /// The regulation this build of the engine was generated for.
    pub fn current() -> &'static Format {
        static CURRENT: OnceLock<Format> = OnceLock::new();
        CURRENT.get_or_init(|| {
            Format::from_json(include_str!("../formats/gen9championsvgc2026regmc.json"))
                .expect("the regulation file shipped with the engine loads")
        })
    }

    /// Reads a regulation file written by `oracle/gen_format.js`.
    ///
    /// Fails with [`Error::Unsupported`] if the file names anything this
    /// build's tables do not have: the engine has to be regenerated from the
    /// same Showdown data before it can play that regulation.
    pub fn from_json(text: &str) -> Result<Format, Error> {
        let file: FormatFile =
            serde_json::from_str(text).map_err(|e| Error::BadState(format!("regulation file: {e}")))?;
        if file.game_type != "doubles" {
            return Err(Error::Unsupported(format!("{} battles ({})", file.game_type, file.name)));
        }
        if file.team.level != 50 {
            return Err(Error::Unsupported(format!("level {} ({})", file.team.level, file.name)));
        }
        let mut names = Names::default();
        let mut items: Vec<u16> = file.items.iter().map(|i| names.item(i)).collect();
        items.sort_unstable();
        let mut species = Vec::with_capacity(file.species.len());
        for s in &file.species {
            let mut rule = SpeciesRule {
                species: names.species(&s.id),
                num: s.num,
                moves: s.moves.iter().map(|m| names.mv(m)).collect(),
                abilities: s.abilities.iter().map(|a| names.ability(a)).collect(),
                genders: Vec::new(),
                megas: s.megas.iter().map(|i| names.item(i)).collect(),
                items: s.items.iter().map(|i| names.item(i)).collect(),
            };
            for g in &s.genders {
                rule.genders
                    .push(Gender::parse(g).ok_or_else(|| Error::BadState(format!("regulation file: gender {g}")))?);
            }
            rule.moves.sort_unstable();
            rule.abilities.sort_unstable();
            rule.megas.sort_unstable();
            rule.items.sort_unstable();
            species.push(rule);
        }
        let mut unknown = names.unknown;
        if !unknown.is_empty() {
            unknown.sort();
            unknown.dedup();
            let shown = unknown.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
            let more = if unknown.len() > 8 { format!(" and {} more", unknown.len() - 8) } else { String::new() };
            return Err(Error::Unsupported(format!(
                "{}: this build of the engine does not have {shown}{more}; regenerate it from the same Showdown data",
                file.name
            )));
        }
        species.sort_by_key(|r| r.species);
        let mut by_species = vec![0u16; SPECIES.len()];
        for (k, r) in species.iter().enumerate() {
            by_species[r.species as usize] = k as u16 + 1;
        }
        Ok(Format {
            id: file.id,
            name: file.name,
            showdown_commit: file.showdown_commit,
            team: file.team,
            differences: file.differences,
            species,
            by_species,
            items,
        })
    }

    /// Whether battles under this regulation come out exactly as the engine plays them.
    pub fn simulated_exactly(&self) -> bool {
        self.differences.is_empty()
    }

    /// Every species that can be brought, in the order of the engine's species table.
    pub fn species(&self) -> &[SpeciesRule] {
        &self.species
    }

    /// What a species may be brought with; `None` if it cannot be brought.
    pub fn rule(&self, species: u16) -> Option<&SpeciesRule> {
        let k = *self.by_species.get(species as usize)?;
        k.checked_sub(1).map(|k| &self.species[k as usize])
    }

    /// The items anyone may hold (Mega Stones included: a Pokémon may hold a stone it cannot use).
    pub fn items(&self) -> &[u16] {
        &self.items
    }

    fn item_allowed(&self, rule: &SpeciesRule, item: u16) -> bool {
        item == it::NONE || self.items.binary_search(&item).is_ok() || rule.items.binary_search(&item).is_ok()
    }

    fn check_candidate(&self, slot: usize, c: &Candidate, out: &mut Vec<Violation>) {
        let Some(species) = c.species else {
            return;
        };
        let Some(rule) = self.rule(species) else {
            out.push(Violation::Species { slot, species });
            return;
        };
        match c.moves.len() {
            0 => out.push(Violation::NoMoves { slot, species }),
            1..=4 => {}
            count => out.push(Violation::TooManyMoves { slot, species, count }),
        }
        for (k, &m) in c.moves.iter().enumerate() {
            let Some(m) = m else {
                continue;
            };
            if c.moves[..k].contains(&Some(m)) {
                out.push(Violation::DuplicateMove { slot, species, move_id: m });
            } else if rule.moves.binary_search(&m).is_err() {
                out.push(Violation::Move { slot, species, move_id: m });
            }
        }
        if let Some(ability) = c.ability
            && rule.abilities.binary_search(&ability).is_err()
        {
            out.push(Violation::Ability { slot, species, ability });
        }
        if let Some(item) = c.item
            && !self.item_allowed(rule, item)
        {
            out.push(Violation::Item { slot, species, item });
        }
        if let Some(gender) = c.gender
            && !rule.genders.contains(&gender)
        {
            out.push(Violation::Gender { slot, species, gender });
        }
        for (stat, &points) in c.stat_points.iter().enumerate() {
            if points < 0 || points > self.team.stat_points_per_stat as i32 {
                out.push(Violation::StatPoints { slot, species, stat, points });
            }
        }
        let total: i32 = c.stat_points.iter().sum();
        if total > self.team.stat_points_total as i32 {
            out.push(Violation::StatPointTotal { slot, species, total, limit: self.team.stat_points_total });
        }
    }

    fn check_candidates(&self, team: &[Candidate], out: &mut Vec<Violation>) {
        let rules = &self.team;
        if team.len() < rules.min_size as usize || team.len() > rules.max_size as usize {
            out.push(Violation::TeamSize { size: team.len(), min: rules.min_size, max: rules.max_size });
        }
        for (slot, c) in team.iter().enumerate() {
            self.check_candidate(slot, c, out);
        }
        // The Pokédex number of anything the engine knows, allowed or not, so that
        // a duplicate is reported even next to another problem.
        let num = |c: &Candidate| c.species.and_then(|s| self.rule(s).map(|r| (r.num, s)));
        if rules.species_clause {
            for (slot, c) in team.iter().enumerate() {
                let Some((n, species)) = num(c) else {
                    continue;
                };
                if let Some(other) = team[..slot].iter().position(|o| num(o).is_some_and(|(m, _)| m == n)) {
                    out.push(Violation::DuplicateSpecies { slot, other, species });
                }
            }
        }
        if rules.item_clause > 0 {
            let mut held: Vec<u16> = team.iter().filter_map(|c| c.item).filter(|&i| i != it::NONE).collect();
            held.sort_unstable();
            let mut k = 0;
            while k < held.len() {
                let holders = held[k..].iter().take_while(|&&i| i == held[k]).count();
                if holders > rules.item_clause as usize {
                    out.push(Violation::DuplicateItem { item: held[k], holders, limit: rules.item_clause });
                }
                k += holders;
            }
        }
    }

    /// Everything wrong with one Pokémon on its own (as slot 0); empty if it can be brought.
    pub fn check_set(&self, set: &PokemonSet) -> Vec<Violation> {
        let mut out = Vec::new();
        self.check_candidate(0, &Candidate::from_set(set), &mut out);
        out
    }

    /// Everything wrong with a team as it is registered (all six, before any are picked);
    /// empty if it is legal.
    pub fn check_team(&self, team: &[PokemonSet]) -> Vec<Violation> {
        let candidates: Vec<Candidate> = team.iter().map(Candidate::from_set).collect();
        let mut out = Vec::new();
        self.check_candidates(&candidates, &mut out);
        out
    }

    pub fn is_legal(&self, team: &[PokemonSet]) -> bool {
        self.check_team(team).is_empty()
    }

    /// [`Format::check_team`] for a team in Showdown's JSON, which may name
    /// things the engine has never heard of; those are reported too.
    pub fn check_showdown_team(&self, team: &[ShowdownSet]) -> Vec<Violation> {
        let mut out = Vec::new();
        let mut candidates = Vec::with_capacity(team.len());
        for (slot, s) in team.iter().enumerate() {
            let (c, problems) = Candidate::from_showdown(slot, s);
            out.extend(problems);
            candidates.push(c);
        }
        self.check_candidates(&candidates, &mut out);
        out
    }

    /// A legal Pokémon of this species, everything about it drawn uniformly:
    /// up to four of its moves, one of its abilities, a nature, a spread
    /// that uses all the stat points. It holds no item (see [`Format::random_team`]).
    pub fn random_set(&self, rule: &SpeciesRule, rng: &mut Rng) -> PokemonSet {
        let mut moves = rule.moves.clone();
        let n = moves.len().min(4);
        rng.shuffle(&mut moves, 0, rule.moves.len());
        moves.truncate(n);
        // Twenty natures change something; one draw in twenty-one is neutral.
        let k = rng.below(NATURE_NAMES.len() as u32 + 1) as usize;
        let nature_pair = NATURE_NAMES.get(k).and_then(|n| nature(n)).unwrap_or((0, 0));
        let mut stat_points = [0u8; 6];
        let mut left = self.team.stat_points_total;
        while left > 0 {
            let stat = rng.below(6) as usize;
            let room = self.team.stat_points_per_stat - stat_points[stat];
            if room == 0 {
                continue;
            }
            let add = (1 + rng.below(room.min(16) as u32) as u16).min(left);
            stat_points[stat] += add as u8;
            left -= add;
        }
        PokemonSet {
            species: rule.species,
            moves,
            nature: nature_pair,
            stat_points,
            ability: rule.abilities[rng.below(rule.abilities.len() as u32) as usize],
            item: it::NONE,
            gender: rule.genders[rng.below(rule.genders.len() as u32) as usize],
        }
    }

    /// A legal team drawn at random: distinct species, each a [`Format::random_set`]
    /// holding a different item (its own Mega Stone a third of the time, if it has one).
    ///
    /// This samples the space of legal teams, not the teams people play: most
    /// of what it returns is bad. It is for testing, and a starting point for search.
    pub fn random_team(&self, rng: &mut Rng) -> Vec<PokemonSet> {
        let size = self.team.max_size as usize;
        let mut team: Vec<PokemonSet> = Vec::with_capacity(size);
        let mut nums: Vec<u16> = Vec::new();
        let mut megas = 0;
        while team.len() < size {
            let rule = &self.species[rng.below(self.species.len() as u32) as usize];
            if self.team.species_clause && nums.contains(&rule.num) {
                continue;
            }
            nums.push(rule.num);
            let mut set = self.random_set(rule, rng);
            let free = |item: u16, team: &[PokemonSet]| {
                self.team.item_clause == 0
                    || team.iter().filter(|s| s.item == item).count() < self.team.item_clause as usize
            };
            if !rule.megas.is_empty() && megas < 2 && rng.chance(1, 3) {
                let stone = rule.megas[rng.below(rule.megas.len() as u32) as usize];
                if free(stone, &team) {
                    set.item = stone;
                    megas += 1;
                }
            }
            // Otherwise an ordinary item nobody on the team holds yet (one try in eight, none).
            for _ in 0..20 {
                if set.item != it::NONE || rng.chance(1, 8) {
                    break;
                }
                let item = self.items[rng.below(self.items.len() as u32) as usize];
                if ITEMS[item as usize].mega.is_empty() && free(item, &team) {
                    set.item = item;
                }
            }
            team.push(set);
        }
        team
    }
}
