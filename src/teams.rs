//! Teams from outside: reading team sheets as people publish them, and
//! keeping a pool of teams to play.
//!
//! A published team sheet (an "open team sheet", as tournaments post them)
//! gives each Pokémon's species, item, ability and moves, and in Champions
//! its nature. It never gives the stat points. [`read_sheet`] reads one, in
//! Pokémon Showdown's export format or picked out of looser text, and
//! [`guess_spread`] fills in stat points with a plain rule of thumb, so that
//! the team can be played at all. Those points are a guess and are marked as
//! one ([`PoolTeam::spreads_guessed`]).
//!
//! ```
//! use vgc_engine::teams::read_sheet;
//!
//! let sheet = "\
//! Garchomp @ Life Orb
//! Ability: Rough Skin
//! Jolly Nature
//! - Earthquake
//! - Rock Slide
//! - Protect
//!
//! Milotic @ Leftovers
//! Ability: Competitive
//! - Scald
//! - Recover";
//! let team = read_sheet(sheet.lines());
//! assert_eq!(team.len(), 2);
//! assert_eq!((team[0].set.species.as_str(), team[0].set.item.as_str()), ("Garchomp", "Life Orb"));
//! assert_eq!(team[0].set.moves, ["Earthquake", "Rock Slide", "Protect"]);
//! assert!(team[0].nature_given && !team[1].nature_given && !team[0].spread_given);
//! ```

use serde::{Deserialize, Serialize};

use crate::battle::{Error, PokemonSet};
use crate::data::*;
use crate::format::{ShowdownSet, ShowdownStats};

/// One Pokémon read off a sheet, with what the sheet did and did not say.
#[derive(Clone, Debug, PartialEq)]
pub struct Read {
    /// Names as the engine's tables spell them. Stat points are 0 unless the sheet gave them.
    pub set: ShowdownSet,
    pub nature_given: bool,
    pub spread_given: bool,
}

const STAT_NAMES: [&str; 6] = ["hp", "atk", "def", "spa", "spd", "spe"];

fn nature_name(text: &str) -> Option<&'static str> {
    const NAMES: [&str; 25] = [
        "Hardy", "Lonely", "Brave", "Adamant", "Naughty", "Bold", "Docile", "Relaxed", "Impish", "Lax", "Timid",
        "Hasty", "Serious", "Jolly", "Naive", "Modest", "Mild", "Quiet", "Bashful", "Rash", "Calm", "Gentle", "Sassy",
        "Careful", "Quirky",
    ];
    NAMES.iter().copied().find(|n| n.eq_ignore_ascii_case(text.trim()))
}

/// The species a sheet means by `name`: a Mega is listed as the Pokémon that becomes it.
fn species_named(name: &str) -> Option<u16> {
    let sp = species_id(&to_id(name))?;
    let is_mega = ITEMS.iter().flat_map(|i| i.mega.iter()).any(|&(_, to)| to == sp);
    if is_mega { species_id(SPECIES[sp as usize].base_species) } else { Some(sp) }
}

/// `Nickname (Species) (F)`, `Species (M)` or `Species` to the species.
fn head_species(text: &str) -> Option<u16> {
    let mut text = text.trim();
    for gender in ["(M)", "(F)"] {
        if let Some(rest) = text.strip_suffix(gender) {
            text = rest.trim();
        }
    }
    if let (Some(open), true) = (text.rfind('('), text.ends_with(')'))
        && let Some(sp) = species_named(&text[open + 1..text.len() - 1])
    {
        return Some(sp);
    }
    species_named(text)
}

/// `32 HP / 2 Def / 32 Spe` to stat points.
fn stats(text: &str) -> Option<ShowdownStats> {
    let mut out = [0i32; 6];
    for part in text.split('/') {
        let mut words = part.split_whitespace();
        let value: i32 = words.next()?.parse().ok()?;
        let stat = to_id(words.next()?);
        let k = STAT_NAMES.iter().position(|s| *s == stat)?;
        out[k] = value;
    }
    Some(ShowdownStats { hp: out[0], atk: out[1], def: out[2], spa: out[3], spd: out[4], spe: out[5] })
}

/// Reads a team sheet.
///
/// It takes Showdown's export format, which is what paste sites show:
///
/// ```text
/// Garchomp @ Life Orb
/// Ability: Rough Skin
/// EVs: 32 Atk / 2 SpD / 32 Spe
/// Jolly Nature
/// - Earthquake
/// ```
///
/// Failing that, it picks a team out of whatever lines it is given by the
/// names it knows: a line that is a Pokémon's name starts a Pokémon, and
/// lines that are an item, an ability, a nature or a move are taken as that
/// Pokémon's. That is for pages that lay a team out some other way (or
/// deliver it as data inside a script); text around the team does no harm,
/// since it names nothing. A team given twice over is read once.
pub fn read_sheet<'a>(lines: impl Iterator<Item = &'a str> + Clone) -> Vec<Read> {
    let strict = read(lines.clone(), false);
    if !strict.is_empty() && strict.len() <= 6 && strict.iter().all(|r| !r.set.moves.is_empty()) {
        return strict;
    }
    let loose = read(lines, true);
    if loose.len() > strict.len() || strict.len() > 6 { loose } else { strict }
}

fn read<'a>(lines: impl Iterator<Item = &'a str>, loose: bool) -> Vec<Read> {
    let mut team: Vec<Read> = Vec::new();
    // The Pokémon the lines are about, if any; `None` while passing over a second copy of one already read.
    let mut at: Option<usize> = None;
    let start = |team: &mut Vec<Read>, species: u16, item: &str| -> Option<usize> {
        let name = SPECIES[species as usize].name;
        if let Some(i) = team.iter().position(|r| r.set.species == name) {
            // Named before. If it came to nothing then (a bare list of the team, say), this is
            // where it is described; if it was described, this is the same thing over again.
            if team[i].set.moves.is_empty() {
                if !item.is_empty() {
                    team[i].set.item = item.to_string();
                }
                return Some(i);
            }
            return None;
        }
        let set = ShowdownSet { species: name.to_string(), item: item.to_string(), ..Default::default() };
        team.push(Read { set, nature_given: false, spread_given: false });
        Some(team.len() - 1)
    };
    let item_named = |text: &str| -> Option<&'static str> {
        let id = to_id(text);
        (!id.is_empty()).then(|| item_id(&id)).flatten().map(|i| ITEMS[i as usize].name)
    };
    let ability_named = |text: &str| ability_id(&to_id(text)).map(|a| ABILITIES[a as usize].name);
    let move_named = |text: &str| move_id(&to_id(text)).map(|m| MOVES[m as usize].name);
    for line in lines {
        let line = line.trim().trim_matches('\u{200b}');
        if line.is_empty() {
            continue;
        }
        // `- Earthquake`
        if let Some(rest) = line.strip_prefix(['-', '–', '—', '•', '*']) {
            if let (Some(i), Some(name)) = (at, move_named(rest)) {
                let moves = &mut team[i].set.moves;
                if moves.len() < 4 && !moves.iter().any(|m| m == name) {
                    moves.push(name.to_string());
                }
            }
            continue;
        }
        // `@ Life Orb`, where the page put the item on a line of its own.
        if let Some(rest) = line.strip_prefix('@') {
            if let (Some(i), Some(name)) = (at, item_named(rest)) {
                team[i].set.item = name.to_string();
            }
            continue;
        }
        // `Ability: Rough Skin`, `EVs: ...`, and keys that say nothing needed here.
        if let Some((key, value)) = line.split_once(':') {
            let (key, value) = (to_id(key), value.trim());
            let known = match key.as_str() {
                "ability" => {
                    if let (Some(i), Some(name)) = (at, ability_named(value)) {
                        team[i].set.ability = name.to_string();
                    }
                    true
                }
                "item" | "helditem" => {
                    if let (Some(i), Some(name)) = (at, item_named(value)) {
                        team[i].set.item = name.to_string();
                    }
                    true
                }
                "nature" => {
                    if let (Some(i), Some(name)) = (at, nature_name(value)) {
                        team[i].set.nature = name.to_string();
                        team[i].nature_given = true;
                    }
                    true
                }
                "evs" | "sps" | "statpoints" => {
                    if let (Some(i), Some(points)) = (at, stats(value)) {
                        team[i].set.evs = points;
                        team[i].spread_given = true;
                    }
                    true
                }
                "level" | "shiny" | "teratype" | "ivs" | "happiness" | "gigantamax" | "dynamaxlevel" => true,
                _ => false,
            };
            if known {
                continue;
            }
        }
        // `Jolly Nature`
        if let Some(name) = line.strip_suffix("Nature").and_then(nature_name) {
            if let Some(i) = at {
                team[i].set.nature = name.to_string();
                team[i].nature_given = true;
            }
            continue;
        }
        // `Garchomp @ Life Orb`, `Rex (Garchomp) (M) @ Life Orb`, `Garchomp`
        let (head, item) = line.split_once(" @ ").unwrap_or((line, ""));
        if let Some(species) = head_species(head) {
            let item = item_named(item).unwrap_or("");
            at = start(&mut team, species, item);
            continue;
        }
        // Anything else is not Showdown's format. Read loosely, a line that is a name counts.
        if let (true, Some(i)) = (loose, at) {
            let set = &mut team[i].set;
            if let (true, Some(name)) = (set.item.is_empty(), item_named(line)) {
                set.item = name.to_string();
            } else if let (true, Some(name)) = (set.ability.is_empty(), ability_named(line)) {
                set.ability = name.to_string();
            } else if let (false, Some(name)) = (team[i].nature_given, nature_name(line)) {
                team[i].set.nature = name.to_string();
                team[i].nature_given = true;
            } else if let Some(name) = move_named(line) {
                let moves = &mut team[i].set.moves;
                if moves.len() < 4 && !moves.iter().any(|m| m == name) {
                    moves.push(name.to_string());
                }
            }
        }
    }
    // A Pokémon that was named and never described was not part of a sheet.
    team.retain(|r| !r.set.moves.is_empty() || !r.set.ability.is_empty());
    team
}

/// Gives a Pokémon stat points, and a nature if it has none, by rule of thumb.
///
/// A team sheet does not say how a Pokémon's 66 stat points are spent, and
/// the engine needs them. This puts 32 in each of two stats and 2 in a third:
///
/// * with a nature that raises Speed, or none given and a fast attacker: its
///   attacking stat and Speed, the rest in HP;
/// * with a nature that raises a defence: HP and that defence;
/// * otherwise an attacker gets HP and its attacking stat, and a Pokémon with
///   no attacking moves HP and Defense.
///
/// The attacking stat is the one more of its moves' power runs off. A nature
/// that is not given is chosen to match (Jolly or Timid for the fast, Adamant
/// or Modest for the rest, Brave or Quiet on a `trick_room` team, Bold for
/// supports). Real spreads are finer than this; it is a stand-in until they
/// are known or searched for.
pub fn guess_spread(read: &mut Read, trick_room: bool) {
    if read.spread_given {
        return;
    }
    let Some(species) = species_id(&to_id(&read.set.species)) else {
        return;
    };
    let base = SPECIES[species as usize].base;
    let (mut physical, mut special) = (0u32, 0u32);
    for m in &read.set.moves {
        if let Some(id) = move_id(&to_id(m)) {
            let data = &MOVES[id as usize];
            match data.category {
                Category::Physical => physical += data.base_power.max(40) as u32,
                Category::Special => special += data.base_power.max(40) as u32,
                Category::Status => {}
            }
        }
    }
    const HP: usize = 0;
    const ATK: usize = 1;
    const DEF: usize = 2;
    const SPA: usize = 3;
    const SPD: usize = 4;
    const SPE: usize = 5;
    let attacks = physical + special > 0;
    let offence = if physical > special || (physical == special && base[ATK] >= base[SPA]) { ATK } else { SPA };
    let (plus, minus) = if read.nature_given { nature(&read.set.nature).unwrap_or((0, 0)) } else { (0, 0) };
    let (plus, minus) = (plus as usize, minus as usize);
    let slow_team_member = trick_room && base[SPE] <= 60;
    let fast = if read.nature_given && (plus, minus) != (0, 0) {
        plus == SPE
    } else {
        attacks && base[SPE] >= 85 && !slow_team_member
    };
    let mut points = [0i32; 6];
    if fast && attacks {
        (points[offence], points[SPE], points[HP]) = (32, 32, 2);
    } else if fast {
        (points[HP], points[SPE], points[DEF]) = (32, 32, 2);
    } else if plus == DEF || plus == SPD {
        (points[HP], points[plus], points[if plus == DEF { SPD } else { DEF }]) = (32, 32, 2);
    } else if attacks {
        (points[HP], points[offence], points[DEF]) = (32, 32, 2);
    } else {
        (points[HP], points[DEF], points[SPD]) = (32, 32, 2);
    }
    let _ = minus;
    read.set.evs =
        ShowdownStats { hp: points[0], atk: points[1], def: points[2], spa: points[3], spd: points[4], spe: points[5] };
    if !read.nature_given {
        let nature = match (attacks, fast, slow_team_member, offence == ATK) {
            (true, true, _, true) => "Jolly",
            (true, true, _, false) => "Timid",
            (true, false, true, true) => "Brave",
            (true, false, true, false) => "Quiet",
            (true, false, false, true) => "Adamant",
            (true, false, false, false) => "Modest",
            (false, ..) => "Bold",
        };
        read.set.nature = nature.to_string();
    }
}

/// One team of a [`Pool`], and where it came from.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PoolTeam {
    pub player: String,
    /// Where it finished, if the source said.
    pub placing: Option<u32>,
    /// Wins and losses, as the source gave them ("11-2").
    pub record: String,
    pub country: String,
    /// The heading it stood under at the source (a division, or a stage of the tournament).
    pub section: String,
    pub url: String,
    /// The stat points are [`guess_spread`]'s, not the player's.
    pub spreads_guessed: bool,
    /// So are the natures.
    pub natures_guessed: bool,
    /// The team as registered: six Pokémon, by name, with stat points in `evs`.
    pub team: Vec<ShowdownSet>,
}

impl PoolTeam {
    /// The team as the engine takes it ([`crate::Battle::with_rosters`]).
    pub fn sets(&self) -> Result<Vec<PokemonSet>, Error> {
        self.team.iter().map(ShowdownSet::to_set).collect()
    }
}

/// A collection of teams to play: the teams of a tournament, say.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Pool {
    /// Where the teams were collected from, and what that page called itself.
    pub source: String,
    pub event: String,
    /// The regulation the teams were checked against, as Showdown's format id.
    pub format: String,
    pub teams: Vec<PoolTeam>,
}

impl Pool {
    pub fn from_json(text: &str) -> Result<Pool, Error> {
        serde_json::from_str(text).map_err(|e| Error::BadTeam(format!("unreadable team pool: {e}")))
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a pool is always serialisable")
    }
}
