//! Replays battles recorded from Pokémon Showdown by `oracle/gen_cases.js`
//! and reports the first place this engine disagrees. Used by the `difftest`
//! binary and by the fixture test in `tests/`.

use crate::data::{ABILITIES, Gender, ITEMS, MOVES, SPECIES, SideCond, SlotCond, Terrain, Type, VolKind, Weather};
use crate::position::BattleState;
use crate::state::{Cond, NO_SPECIES, Res, Trapped};
use crate::{Battle, Choice, Error, PokemonSet, Request, trace};
use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
pub struct SetJson {
    pub species: String,
    pub moves: Vec<String>,
    pub nature: String,
    pub sp: [u8; 6],
    /// Ability id; absent means no ability.
    #[serde(default)]
    pub ability: Option<String>,
    /// Item id; absent or empty means no item.
    #[serde(default)]
    pub item: Option<String>,
    /// "M", "F" or "N"; absent means genderless, which is what the recorder
    /// used before genders were recorded.
    #[serde(default)]
    pub gender: Option<String>,
}

impl SetJson {
    pub fn to_set(&self) -> Result<PokemonSet, String> {
        let moves: Vec<&str> = self.moves.iter().map(String::as_str).collect();
        let mut set =
            PokemonSet::from_names(&self.species, &moves, &self.nature, self.sp).map_err(|e| e.to_string())?;
        if let Some(a) = &self.ability {
            set = set.ability(a).map_err(|e| e.to_string())?;
        }
        if let Some(i) = &self.item {
            set = set.item(i).map_err(|e| e.to_string())?;
        }
        let g = self.gender.as_deref().unwrap_or("N");
        set = set.gender(Gender::parse(g).ok_or_else(|| format!("unknown gender {g}"))?);
        Ok(set)
    }
}

#[derive(Deserialize, PartialEq, Debug, Clone)]
pub struct MonSnap {
    pub idx: u8,
    pub species: String,
    pub hp: u16,
    pub maxhp: u16,
    pub stats: [u16; 6],
    pub status: String,
    pub time: u8,
    pub stage: u8,
    pub boosts: [i8; 7],
    pub pp: Vec<u8>,
    pub active: bool,
    pub fainted: bool,
    pub switch_flag: bool,
    pub speed: i32,
    pub vol: Vec<String>,
    // Recorded since abilities and items were modelled; absent in older files.
    #[serde(default)]
    pub ability: Option<String>,
    #[serde(default)]
    pub item: Option<String>,
    #[serde(default)]
    pub last_item: Option<String>,
    #[serde(default)]
    pub used_item: Option<bool>,
    #[serde(default)]
    pub ate_berry: Option<bool>,
    #[serde(default)]
    pub types: Option<Vec<String>>,
    /// 0 free, 1 trapped, 2 trapped by something not yet revealed.
    #[serde(default)]
    pub trapped: Option<u8>,
    #[serde(default)]
    pub disabled: Option<Vec<bool>>,
    #[serde(default)]
    pub ability_order: Option<u32>,
    #[serde(default)]
    pub item_order: Option<u32>,
    #[serde(default)]
    pub active_turns: Option<u16>,
    /// `moveThisTurnResult` and `moveLastTurnResult`, one letter each:
    /// u(ndefined), n(ull), t(rue), f(alse).
    #[serde(default)]
    pub move_result: Option<String>,
    #[serde(default)]
    pub base_species: Option<String>,
    /// Id of the Mega the Pokémon can still become; empty if none.
    #[serde(default)]
    pub can_mega: Option<String>,
    /// Id of the move last used since coming in; empty if none.
    #[serde(default)]
    pub last_move: Option<String>,
    #[serde(default)]
    pub move_actions: Option<u8>,
    #[serde(default)]
    pub newly_switched: Option<bool>,
    /// `hurtThisTurn` (0 if unhurt), `timesAttacked`, the two stat-change flags
    /// as "raised/lowered" letters, and what this turn's `attackedBy` entries
    /// amount to: "sources that did damage : slot and damage of the last foe to hit".
    #[serde(default)]
    pub hurt: Option<u16>,
    #[serde(default)]
    pub times_attacked: Option<u16>,
    #[serde(default)]
    pub stat_flags: Option<String>,
    #[serde(default)]
    pub attacked_by: Option<String>,
    /// The last hit from a foe still on the field, from this turn or an earlier one ("*").
    #[serde(default)]
    pub last_damaged: Option<String>,
    /// The ids of the moves it has now (another Pokémon's while it is transformed).
    #[serde(default)]
    pub moves: Option<String>,
    #[serde(default)]
    pub transformed: Option<bool>,
    /// Team index of the Pokémon an Illusion shows; -1 for none.
    #[serde(default)]
    pub illusion: Option<i8>,
    #[serde(default)]
    pub locked: Option<String>,
    /// The type added by Forest's Curse or Trick-or-Treat; empty if none.
    #[serde(default)]
    pub added_type: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct SideSnap {
    pub left: u8,
    pub mons: Vec<MonSnap>,
    /// Side conditions as `id:turns left:detail:effect order` (absent in older files).
    #[serde(default)]
    pub conds: Option<Vec<String>>,
    /// Slot conditions per active position, as `id:turns left:detail`.
    #[serde(default)]
    pub slots: Option<Vec<Vec<String>>>,
}

/// Weather, terrain and pseudo-weathers.
#[derive(Deserialize, PartialEq, Debug)]
pub struct FieldSnap {
    pub weather: String,
    pub weather_turns: u8,
    pub terrain: String,
    pub terrain_turns: u8,
    pub pseudo: Vec<String>,
}

/// Showdown's state at one decision point.
#[derive(Deserialize, Debug)]
pub struct Snap {
    pub turn: u16,
    pub ended: bool,
    /// 0 or 1 for a winner, -1 for a tie, absent while the battle is running.
    pub winner: Option<i8>,
    pub request: String,
    pub rng: [u16; 4],
    /// Showdown's `effectOrder` counter (absent in older files).
    #[serde(default)]
    pub effect_order: Option<u32>,
    /// `battle.lastMove` (absent in older files); empty before the first move.
    #[serde(default)]
    pub battle_last_move: Option<String>,
    #[serde(default)]
    pub field: Option<FieldSnap>,
    pub sides: [SideSnap; 2],
    /// Showdown's RNG draws during the step (only recorded with `--trace`).
    #[serde(default)]
    pub draws: Vec<String>,
    /// Showdown's battle log for the step (only recorded with `--trace`).
    #[serde(default)]
    pub log: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct Step {
    pub choices: [String; 2],
    /// Legal choices per side and slot before `choices` were made.
    pub legal: [[Vec<String>; 2]; 2],
    pub after: Snap,
}

#[derive(Deserialize, Debug)]
pub struct Case {
    pub id: u32,
    pub seed: [u16; 4],
    pub teams: [Vec<SetJson>; 2],
    pub initial: Snap,
    pub steps: Vec<Step>,
}

pub enum Outcome {
    /// Matched at every decision; carries the number of decisions checked.
    Pass(usize),
    /// The case uses something the engine does not model.
    Unsupported(String),
    /// Diverged; carries a human-readable report.
    Fail(Vec<String>),
}

fn mon_snap(b: &Battle, side: usize, pos: usize) -> MonSnap {
    let r = b.active_at(side, pos);
    let m = b.mon(r);
    MonSnap {
        idx: r.idx,
        species: SPECIES[m.species as usize].id.to_string(),
        hp: m.hp,
        maxhp: m.max_hp(),
        stats: m.stats,
        // Showdown parks a fainted Pokémon's status in ways that no longer matter.
        status: if m.fainted { String::new() } else { m.status.id().to_string() },
        time: if m.fainted { 0 } else { m.status_time },
        stage: if m.fainted { 0 } else { m.tox_stage },
        boosts: m.boosts,
        pp: m.moves[..m.n_moves as usize].iter().map(|s| s.pp).collect(),
        active: m.is_active,
        fainted: m.fainted,
        switch_flag: m.switch_flag,
        speed: m.speed,
        vol: b.vols(r).as_slice().iter().map(vol_snap).collect(),
        ability: Some(ABILITIES[m.ability as usize].id.to_string()),
        item: Some(ITEMS[m.item as usize].id.to_string()),
        last_item: Some(ITEMS[m.last_item as usize].id.to_string()),
        used_item: Some(m.used_item_this_turn),
        ate_berry: Some(m.ate_berry),
        types: Some(m.types.iter().filter(|&&t| t != Type::None).map(|&t| type_name(t)).collect()),
        trapped: Some(match m.trapped {
            Trapped::No => 0,
            Trapped::Yes => 1,
            Trapped::Hidden => 2,
        }),
        disabled: Some(m.moves[..m.n_moves as usize].iter().map(|s| s.disabled).collect()),
        ability_order: Some(m.ability_st.order),
        item_order: Some(m.item_st.order),
        active_turns: Some(m.active_turns),
        move_result: Some([m.move_this_turn, m.move_last_turn].iter().map(|r| result_code(*r)).collect()),
        base_species: Some(SPECIES[m.base_species as usize].id.to_string()),
        can_mega: Some(if m.can_mega == NO_SPECIES {
            String::new()
        } else {
            SPECIES[m.can_mega as usize].id.to_string()
        }),
        last_move: Some(if m.last_move == crate::state::NO_MOVE {
            String::new()
        } else {
            MOVES[m.last_move as usize].id.to_string()
        }),
        move_actions: Some(m.active_move_actions),
        newly_switched: Some(m.newly_switched),
        hurt: Some(m.hurt_this_turn),
        times_attacked: Some(m.times_attacked),
        stat_flags: Some(format!(
            "{}{}",
            if m.stats_raised_this_turn { 'r' } else { '-' },
            if m.stats_lowered_this_turn { 'l' } else { '-' }
        )),
        attacked_by: Some(format!(
            "{}:{}",
            m.hit_by_this_turn,
            match m.last_damaged_by() {
                Some(a) if a.this_turn => format!("{}/{}", slot_index(a.slot), a.damage),
                _ => String::new(),
            }
        )),
        last_damaged: Some(match m.last_damaged_by() {
            Some(a) => format!("{}/{}{}", slot_index(a.slot), a.damage, if a.this_turn { "" } else { "*" }),
            None => String::new(),
        }),
        moves: Some(
            m.moves[..m.n_moves as usize].iter().map(|s| MOVES[s.id as usize].id).collect::<Vec<_>>().join(","),
        ),
        transformed: Some(m.transformed),
        illusion: Some(m.illusion as i8 - 1),
        added_type: Some(if m.added_type == Type::None { String::new() } else { type_name(m.added_type) }),
        locked: Some(if m.locked_move == crate::state::NO_MOVE {
            String::new()
        } else {
            MOVES[m.locked_move as usize].id.to_string()
        }),
    }
}

/// A type as Showdown spells it.
fn type_name(t: Type) -> String {
    if t == Type::Typeless { "???".to_string() } else { format!("{t:?}") }
}

fn result_code(r: Res) -> char {
    match r {
        Res::Undef => 'u',
        Res::Null => 'n',
        Res::Bool(true) => 't',
        Res::Bool(false) => 'f',
        _ => '?',
    }
}

/// The state a side condition carries (the recorder writes the same thing).
fn side_detail(c: &Cond<SideCond>) -> String {
    match c.kind {
        // Layers.
        SideCond::Spikes | SideCond::Toxicspikes => c.data.to_string(),
        _ => "0".to_string(),
    }
}

/// A field slot as the recording prints it: -1 for "not on the field".
fn slot_index(slot: u8) -> i32 {
    if slot == crate::state::NO_SLOT { -1 } else { slot as i32 }
}

/// The state a slot condition carries.
fn slot_detail(c: &Cond<SlotCond>) -> String {
    match c.kind {
        // HP to restore / turn counter value when the wish was made.
        SlotCond::Wish => format!("{}/{}", c.data, c.st.a),
        // The turn counter value it lands at / who sent it (side, team index).
        SlotCond::Futuremove => {
            format!("{}/{}", c.st.a, c.source.map_or(String::new(), |s| format!("{}{}", s.side, s.idx)))
        }
        _ => "0".to_string(),
    }
}

/// A volatile as `id:turns left:detail`, where the detail is whatever state
/// the volatile carries (the recorder writes the same thing).
fn vol_snap(v: &crate::state::Volatile) -> String {
    let detail = match v.kind {
        VolKind::Stall | VolKind::Allyswitch => v.data.to_string(),
        VolKind::Confusion => v.data.to_string(),
        VolKind::Choicelock | VolKind::Encore | VolKind::Disable if v.data > 0 => {
            MOVES[v.data as usize - 1].id.to_string()
        }
        VolKind::Helpinghand => v.data.to_string(),
        // Kept in half points; printed the way JavaScript prints the number.
        VolKind::Substitute => format!("{}{}", v.data / 2, if v.data % 2 == 1 { ".5" } else { "" }),
        VolKind::Dragoncheer | VolKind::Partiallytrapped => v.data.to_string(),
        VolKind::Stockpile => format!("{}/{}/{}", v.data, v.st.a, v.st.b),
        VolKind::Leechseed => v.source_slot.to_string(),
        VolKind::Lockedmove if v.data > 0 => format!("{}/{}", MOVES[v.data as usize - 1].id, v.st.a),
        VolKind::Twoturnmove if v.data > 0 => MOVES[v.data as usize - 1].id.to_string(),
        VolKind::Fly
        | VolKind::Dig
        | VolKind::Dive
        | VolKind::Bounce
        | VolKind::Phantomforce
        | VolKind::Solarbeam
        | VolKind::Solarblade
        | VolKind::Skyattack
        | VolKind::Meteorbeam
        | VolKind::Electroshot => v.st.a.to_string(),
        VolKind::Metronome => {
            let last = if v.data > 0 { MOVES[v.data as usize - 1].id } else { "-" };
            format!("{last}/{}", v.st.a)
        }
        _ => "0".to_string(),
    };
    format!("{}:{}:{}", v.kind.id(), v.duration, detail)
}

/// Every field where the engine disagrees with Showdown's snapshot.
pub fn diff(b: &Battle, want: &Snap) -> Vec<String> {
    let mut out = Vec::new();
    macro_rules! check {
        ($name:expr, $got:expr, $want:expr) => {
            if $got != $want {
                out.push(format!("{}: engine {:?}, Showdown {:?}", $name, $got, $want));
            }
        };
    }
    check!("turn", b.turn, want.turn);
    check!("ended", b.ended, want.ended);
    let winner = if b.ended { Some(b.winner.map(|w| w as i8).unwrap_or(-1)) } else { None };
    check!("winner", winner, want.winner);
    let request = match b.request {
        Request::None => "",
        Request::Move => "move",
        Request::Switch => "switch",
    };
    check!("request", request, want.request.as_str());
    check!("rng seed", b.rng.words(), want.rng);
    if let Some(eo) = want.effect_order {
        check!("effect order counter", b.effect_order, eo);
    }
    if let Some(lm) = &want.battle_last_move {
        let ours = if b.last_move == crate::state::NO_MOVE { "" } else { MOVES[b.last_move as usize].id };
        check!("the battle's last move", ours, lm.as_str());
    }
    if let Some(wf) = &want.field {
        let f = &b.field;
        let got = FieldSnap {
            weather: f.weather.kind.id().to_string(),
            weather_turns: if f.weather.kind == Weather::None { 0 } else { f.weather.duration },
            terrain: f.terrain.kind.id().to_string(),
            terrain_turns: if f.terrain.kind == Terrain::None { 0 } else { f.terrain.duration },
            pseudo: f.pseudo.as_slice().iter().map(|c| format!("{}:{}", c.kind.id(), c.duration)).collect(),
        };
        check!("field", &got, wf);
    }
    for side in 0..2 {
        let ws = &want.sides[side];
        check!(format!("p{} pokemon left", side + 1), b.sides[side].pokemon_left, ws.left);
        if let Some(wc) = &ws.conds {
            let got: Vec<String> = b.sides[side]
                .conds
                .as_slice()
                .iter()
                .map(|c| format!("{}:{}:{}:{}", c.kind.id(), c.duration, side_detail(c), c.st.order))
                .collect();
            check!(format!("p{} side conditions", side + 1), &got, wc);
        }
        if let Some(wsl) = &ws.slots {
            let got: Vec<Vec<String>> = b.sides[side]
                .slot_conds
                .iter()
                .map(|l| {
                    l.as_slice().iter().map(|c| format!("{}:{}:{}", c.kind.id(), c.duration, slot_detail(c))).collect()
                })
                .collect();
            check!(format!("p{} slot conditions", side + 1), &got, wsl);
        }
        if b.sides[side].n as usize != ws.mons.len() {
            out.push(format!("p{} team size differs", side + 1));
            continue;
        }
        for (pos, w) in ws.mons.iter().enumerate() {
            let g = mon_snap(b, side, pos);
            let who = format!("p{} position {} ({})", side + 1, pos + 1, w.species);
            check!(format!("{who} team index"), g.idx, w.idx);
            check!(format!("{who} species"), &g.species, &w.species);
            check!(format!("{who} hp"), g.hp, w.hp);
            check!(format!("{who} max hp"), g.maxhp, w.maxhp);
            check!(format!("{who} stats"), g.stats, w.stats);
            check!(format!("{who} status"), &g.status, &w.status);
            check!(format!("{who} status timer"), g.time, w.time);
            check!(format!("{who} toxic stage"), g.stage, w.stage);
            check!(format!("{who} boosts"), g.boosts, w.boosts);
            check!(format!("{who} pp"), &g.pp, &w.pp);
            check!(format!("{who} active"), g.active, w.active);
            check!(format!("{who} fainted"), g.fainted, w.fainted);
            check!(format!("{who} switch flag"), g.switch_flag, w.switch_flag);
            check!(format!("{who} cached speed"), g.speed, w.speed);
            check!(format!("{who} volatiles"), &g.vol, &w.vol);
            // Fields older recordings do not carry are only compared when present.
            macro_rules! check_opt {
                ($name:expr, $field:ident) => {
                    if w.$field.is_some() {
                        check!(format!("{who} {}", $name), &g.$field, &w.$field);
                    }
                };
            }
            check_opt!("ability", ability);
            check_opt!("item", item);
            check_opt!("last item", last_item);
            check_opt!("used item this turn", used_item);
            check_opt!("ate berry", ate_berry);
            check_opt!("types", types);
            check_opt!("disabled moves", disabled);
            check_opt!("active turns", active_turns);
            check_opt!("base species", base_species);
            check_opt!("can mega evolve", can_mega);
            check_opt!("illusion", illusion);
            if w.active {
                check_opt!("move results", move_result);
                check_opt!("last move", last_move);
                check_opt!("move actions since switching in", move_actions);
                check_opt!("newly switched", newly_switched);
                check_opt!("hurt this turn", hurt);
                check_opt!("times attacked", times_attacked);
                check_opt!("stats raised/lowered this turn", stat_flags);
                check_opt!("attacked by", attacked_by);
                check_opt!("last damaged by", last_damaged);
                check_opt!("moves", moves);
                check_opt!("transformed", transformed);
                check_opt!("added type", added_type);
                check_opt!("trapped", trapped);
                check_opt!("ability effect order", ability_order);
                check_opt!("item effect order", item_order);
            }
        }
    }
    out
}

fn diff_legal(b: &Battle, want: &[[Vec<String>; 2]; 2]) -> Vec<String> {
    let mut out = Vec::new();
    for (side, slots) in want.iter().enumerate() {
        for (pos, expected) in slots.iter().enumerate() {
            let mut got: Vec<String> = b.legal_choices(side, pos).into_iter().map(Choice::to_showdown).collect();
            let mut exp = expected.clone();
            got.sort();
            exp.sort();
            if got != exp {
                out.push(format!(
                    "legal choices for p{} slot {}: engine {:?}, Showdown {:?}",
                    side + 1,
                    pos + 1,
                    got,
                    exp
                ));
            }
        }
    }
    out
}

/// The `rng(range)=value` part of a trace line, without its label.
fn drawn(line: &str) -> &str {
    line.split("  ").next().unwrap_or(line)
}

/// With tracing on both sides, checks that the engine drew the same ranges and
/// values in the same order as Showdown, not merely that the seeds agree afterwards.
fn draws_differ(snap: &Snap) -> Option<Vec<String>> {
    if !cfg!(feature = "trace") || snap.draws.is_empty() {
        trace::take();
        return None;
    }
    let ours = trace::take();
    let ours_rng: Vec<&String> = ours.iter().filter(|l| l.starts_with("rng(")).collect();
    let same =
        ours_rng.len() == snap.draws.len() && ours_rng.iter().zip(&snap.draws).all(|(a, b)| drawn(a) == drawn(b));
    if same {
        return None;
    }
    let mut out = vec![format!("RNG draws differ: engine {}, Showdown {}", ours_rng.len(), snap.draws.len())];
    let len = ours_rng.len().max(snap.draws.len());
    for j in 0..len {
        let a = ours_rng.get(j).map(|s| s.as_str()).unwrap_or("-");
        let b = snap.draws.get(j).map(|s| s.as_str()).unwrap_or("-");
        if drawn(a) != drawn(b) {
            out.push(format!("  #{j:<3} engine   {a}"));
            out.push(format!("       showdown {b}"));
        }
    }
    Some(out)
}

/// Lines up the engine's RNG draws against Showdown's for the failing step.
fn trace_report(snap: &Snap, out: &mut Vec<String>) {
    let ours = trace::take();
    if snap.draws.is_empty() || ours.is_empty() {
        out.push(
            "(to see where the RNG streams part: record with `gen_cases.js --trace` and build with `--features trace`)"
                .into(),
        );
        return;
    }
    let ours_rng: Vec<&String> = ours.iter().filter(|l| l.starts_with("rng(")).collect();
    out.push(format!("RNG draws this step: engine {}, Showdown {}", ours_rng.len(), snap.draws.len()));
    let len = ours_rng.len().max(snap.draws.len());
    let first_bad = (0..len).find(|&i| match (ours_rng.get(i), snap.draws.get(i)) {
        (Some(a), Some(b)) => drawn(a) != drawn(b),
        _ => true,
    });
    if let Some(i) = first_bad {
        out.push(format!("first differing draw is #{i}:"));
        for j in i.saturating_sub(5)..(i + 4).min(len) {
            out.push(format!("  #{j:<3} engine   {}", ours_rng.get(j).map(|s| s.as_str()).unwrap_or("-")));
            out.push(format!("       showdown {}", snap.draws.get(j).map(|s| s.as_str()).unwrap_or("-")));
        }
    }
    out.push("engine trace:".into());
    out.extend(ours.iter().map(|l| format!("  {l}")));
    out.push("Showdown log:".into());
    let noise = |l: &&String| l.is_empty() || l.starts_with("|t:|") || l.starts_with("|split") || l.contains("uhtml");
    out.extend(snap.log.iter().filter(|l| !noise(l)).map(|l| format!("  {l}")));
}

/// What else to do at every decision while replaying.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rebuild {
    /// Nothing: just replay.
    No,
    /// Write the position down, build a battle back from it, require the two
    /// to be the same position, and carry on with the rebuilt one.
    Exact,
    /// Also build a battle from the position stripped of the simulator's
    /// bookkeeping, as if it had been written by hand, and require what
    /// `from_state` works out to be what the engine had.
    ByHand,
}

/// A position with everything removed that a battle built from a hand-written
/// description may legitimately have differently: the random numbers, the
/// order effects started in and the cached speeds.
fn loose(st: &BattleState, at_move_request: bool) -> BattleState {
    let mut s = st.clone();
    s.seed = [0; 4];
    s.rng_calls = 0;
    s.effect_order = 0;
    s.speed_order.clear();
    let strip = |c: &mut crate::position::CondState| {
        c.order = 0;
        // (Whether a field condition's handlers have ever run: it changes how
        // Showdown sorts them among ties, and cannot be known from outside.)
        c.targeted = false;
    };
    s.weather.iter_mut().for_each(strip);
    s.terrain.iter_mut().for_each(strip);
    s.pseudo_weather.iter_mut().for_each(strip);
    for side in &mut s.sides {
        side.conditions.iter_mut().for_each(strip);
        side.slot_conditions.iter_mut().flatten().for_each(strip);
        for (place, m) in side.pokemon.iter_mut().enumerate() {
            m.speed = 0;
            m.status_extra.order = 0;
            m.ability_extra.order = 0;
            m.item_extra.order = 0;
            m.volatiles.iter_mut().for_each(strip);
            // (Leftover flags on moves it cannot use while transformed.)
            for mv in &mut m.base_moves {
                mv.disabled = false;
                mv.hidden = false;
            }
            if !at_move_request || place >= crate::ACTIVE || m.fainted {
                // Only a move request works these out; in between they are leftovers.
                m.trapped = Trapped::No;
                m.locked_move.clear();
                for mv in &mut m.moves {
                    mv.disabled = false;
                    mv.hidden = false;
                }
            }
        }
    }
    if s.ended {
        // (Nothing is going to continue.)
        s.mid_turn = None;
    }
    // The queue as a set: ties in it are settled by the random numbers.
    if st.ended {
        s.pending.clear();
    }
    for a in &mut s.pending {
        a.resolved = None;
    }
    s.pending.sort_by_key(|a| (a.kind.clone(), a.pokemon.map(|m| (m.side, m.pokemon)), a.move_id.clone()));
    s
}

/// The first place two positions differ, as JSON paths.
fn json_diff(path: &str, a: &serde_json::Value, b: &serde_json::Value, out: &mut Vec<String>) {
    use serde_json::Value;
    if a == b || out.len() >= 6 {
        return;
    }
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                json_diff(
                    &format!("{path}.{k}"),
                    x.get(k).unwrap_or(&Value::Null),
                    y.get(k).unwrap_or(&Value::Null),
                    out,
                );
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                json_diff(&format!("{path}[{i}]"), p, q, out);
            }
        }
        _ => out.push(format!("{path}: engine {a}, rebuilt {b}")),
    }
}

/// The rebuild checks at one decision. On success the battle to carry on with.
fn rebuild(b: &Battle, mode: Rebuild) -> Result<Battle, Vec<String>> {
    let st = b.to_state();
    let text = st.to_json();
    match BattleState::from_json(&text) {
        Ok(back) if back == st => {}
        Ok(_) => return Err(vec!["the position changes when written as JSON and read back".into()]),
        Err(e) => return Err(vec![format!("the position's own JSON does not read back: {e}")]),
    }
    let exact = Battle::from_state(&st).map_err(|e| vec![format!("from_state refused an exported position: {e}")])?;
    if let Some(d) = b.position_diff(&exact) {
        return Err(vec!["a battle rebuilt from the exported position differs:".into(), d]);
    }
    if mode == Rebuild::ByHand {
        let by_hand = Battle::from_state(&st.without_bookkeeping()).map_err(|e| {
            vec![
                format!("from_state refused the position without bookkeeping: {e}"),
                format!("pending: {}", serde_json::to_string(&st.pending).unwrap_or_default()),
            ]
        })?;
        let at_move = b.request == Request::Move;
        let (want, got) = (loose(&st, at_move), loose(&by_hand.to_state(), at_move));
        if want != got {
            let mut out = vec!["a battle built from the position without bookkeeping differs:".to_string()];
            let (x, y) = (serde_json::to_value(&want).unwrap(), serde_json::to_value(&got).unwrap());
            json_diff("", &x, &y, &mut out);
            return Err(out);
        }
        // (Working the bookkeeping out draws random numbers of its own; they are not the battle's.)
        trace::take();
    }
    Ok(exact)
}

/// Replays one recorded battle and compares state after every decision.
pub fn check_case(c: &Case) -> Outcome {
    check_case_with(c, Rebuild::No)
}

/// [`check_case`], optionally rebuilding the battle from its exported position at every decision.
pub fn check_case_with(c: &Case, mode: Rebuild) -> Outcome {
    let sets: Result<Vec<Vec<PokemonSet>>, String> =
        c.teams.iter().map(|t| t.iter().map(SetJson::to_set).collect()).collect();
    let sets = match sets {
        Ok(s) => s,
        Err(e) => return Outcome::Fail(vec![e]),
    };
    trace::take();
    let mut b = match Battle::new([&sets[0], &sets[1]], c.seed) {
        Ok(b) => b,
        Err(Error::Unsupported(what)) => return Outcome::Unsupported(what),
        Err(e) => return Outcome::Fail(vec![e.to_string()]),
    };
    let d = diff(&b, &c.initial);
    if !d.is_empty() {
        let mut out = vec!["state differs after the opening switch-ins".to_string()];
        out.extend(d);
        trace_report(&c.initial, &mut out);
        return Outcome::Fail(out);
    }
    if let Some(lines) = draws_differ(&c.initial) {
        let mut out = vec!["same state, different RNG draws during the opening switch-ins".to_string()];
        out.extend(lines);
        return Outcome::Fail(out);
    }
    for (i, step) in c.steps.iter().enumerate() {
        trace::take();
        let turn = b.turn;
        let header = |what: &str| {
            format!("{what} at decision {i} (turn {turn}), choices p1 [{}] p2 [{}]", step.choices[0], step.choices[1])
        };
        if mode != Rebuild::No {
            match rebuild(&b, mode) {
                Ok(rebuilt) => b = rebuilt,
                Err(mut out) => {
                    out.insert(0, header("rebuilding the position failed"));
                    return Outcome::Fail(out);
                }
            }
        }
        let dl = diff_legal(&b, &step.legal);
        if !dl.is_empty() {
            let mut out = vec![header("legal choices differ")];
            out.extend(dl);
            return Outcome::Fail(out);
        }
        let (Some(p1), Some(p2)) = (Choice::parse_side(&step.choices[0]), Choice::parse_side(&step.choices[1])) else {
            return Outcome::Fail(vec![header("unreadable choice")]);
        };
        if let Err(e) = b.choose([p1, p2]) {
            return Outcome::Fail(vec![header("engine rejected the choice"), e.to_string()]);
        }
        let d = diff(&b, &step.after);
        if !d.is_empty() {
            let mut out = vec![header("state differs")];
            out.extend(d);
            trace_report(&step.after, &mut out);
            return Outcome::Fail(out);
        }
        if let Some(lines) = draws_differ(&step.after) {
            let mut out = vec![header("same state, different RNG draws")];
            out.extend(lines);
            return Outcome::Fail(out);
        }
    }
    if mode != Rebuild::No {
        // The final position too (usually a finished battle).
        if let Err(mut out) = rebuild(&b, mode) {
            out.insert(0, "rebuilding the final position failed".to_string());
            return Outcome::Fail(out);
        }
    }
    Outcome::Pass(c.steps.len())
}
