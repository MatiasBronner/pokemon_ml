//! Replays battles recorded from Pokémon Showdown by `oracle/gen_cases.js`
//! and reports the first place this engine disagrees. Used by the `difftest`
//! binary and by the fixture test in `tests/`.

use crate::data::SPECIES;
use crate::{Battle, Choice, Error, PokemonSet, Request, trace};
use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
pub struct SetJson {
    pub species: String,
    pub moves: Vec<String>,
    pub nature: String,
    pub sp: [u8; 6],
}

impl SetJson {
    pub fn to_set(&self) -> Result<PokemonSet, String> {
        let moves: Vec<&str> = self.moves.iter().map(String::as_str).collect();
        PokemonSet::from_names(&self.species, &moves, &self.nature, self.sp).map_err(|e| e.to_string())
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
}

#[derive(Deserialize, Debug)]
pub struct SideSnap {
    pub left: u8,
    pub mons: Vec<MonSnap>,
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
        vol: m.volatiles.as_slice().iter().map(|v| format!("{}:{}:{}", v.kind.id(), v.duration, v.counter)).collect(),
    }
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
    for side in 0..2 {
        let ws = &want.sides[side];
        check!(format!("p{} pokemon left", side + 1), b.sides[side].pokemon_left, ws.left);
        if b.sides[side].n as usize != ws.mons.len() {
            out.push(format!("p{} team size differs", side + 1));
            continue;
        }
        for (pos, w) in ws.mons.iter().enumerate() {
            let g = mon_snap(b, side, pos);
            if &g == w {
                continue;
            }
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

/// Replays one recorded battle and compares state after every decision.
pub fn check_case(c: &Case) -> Outcome {
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
    Outcome::Pass(c.steps.len())
}
