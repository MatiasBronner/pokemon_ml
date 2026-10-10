//! How fast the training environment runs: games stepped in parallel with
//! random or greedy players, observations for both sides written at every decision.
//!
//!     envbench [POOL.json] [--envs N] [--steps N] [--threads N] [--greedy | --lookahead]
//!
//! Without a pool file it plays random legal teams.

use std::time::Instant;

use vgc_engine::env::{Baseline, Config, OBS_M, VecEnv};
use vgc_engine::format::{Format, ShowdownSet};
use vgc_engine::obs::{OBS_F, OBS_I};
use vgc_engine::rng::Rng;
use vgc_engine::teams::{Pool, PoolTeam};

fn main() {
    let mut config = Config { envs: 1024, ..Config::default() };
    let (mut steps, mut path, mut kind) = (200usize, None, Baseline::Random);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut number = |what: &str| {
            args.next().and_then(|n| n.parse::<usize>().ok()).unwrap_or_else(|| panic!("{what} takes a number"))
        };
        match a.as_str() {
            "--envs" => config.envs = number("--envs"),
            "--steps" => steps = number("--steps"),
            "--threads" => config.threads = number("--threads"),
            "--greedy" => kind = Baseline::Greedy,
            "--lookahead" => kind = Baseline::Lookahead,
            _ => path = Some(a),
        }
    }
    let pool = match &path {
        Some(p) => Pool::from_json(&std::fs::read_to_string(p).unwrap_or_else(|e| panic!("cannot read {p}: {e}")))
            .unwrap_or_else(|e| panic!("{p}: {e:?}")),
        None => {
            let format = Format::current();
            let mut rng = Rng::from_words([1, 2, 3, 4]);
            let team = |rng: &mut Rng| PoolTeam {
                team: format.random_team(rng).iter().map(ShowdownSet::from_set).collect(),
                ..PoolTeam::default()
            };
            Pool { format: format.id.clone(), teams: (0..200).map(|_| team(&mut rng)).collect(), ..Pool::default() }
        }
    };
    let mut env = VecEnv::new(&pool, config).unwrap_or_else(|e| panic!("{e:?}"));
    let n = env.len();
    let (mut f, mut i, mut m) = (vec![0f32; 2 * n * OBS_F], vec![0i16; 2 * n * OBS_I], vec![0u8; 2 * n * OBS_M]);
    let (mut reward, mut done, mut actions) = (vec![0f32; 2 * n], vec![0u8; n], vec![0i32; 4 * n]);
    env.observe(&mut f, &mut i, &mut m);
    let (mut choosing, mut stepping) = (0.0, 0.0);
    for _ in 0..steps {
        let t = Instant::now();
        env.baseline(kind, &mut actions);
        choosing += t.elapsed().as_secs_f64();
        let t = Instant::now();
        env.step(&actions, &mut f, &mut i, &mut m, &mut reward, &mut done).unwrap_or_else(|e| panic!("{e:?}"));
        stepping += t.elapsed().as_secs_f64();
    }
    let s = env.stats();
    println!(
        "{n} games at once on {} threads, {} teams, {kind:?} players: {:.0} decisions/s ({:.0} observations/s), {:.0} games/s; \
         {:.1} decisions and {:.1} turns a game, {:.1}% tied or out of turns",
        env.threads(),
        pool.teams.len(),
        s.decisions as f64 / stepping,
        2.0 * s.decisions as f64 / stepping,
        s.games as f64 / stepping,
        s.decisions as f64 / s.games.max(1) as f64,
        s.turns as f64 / s.games.max(1) as f64,
        100.0 * s.ties as f64 / s.games.max(1) as f64,
    );
    println!(
        "  stepping and observing takes {:.1} µs a decision; the {kind:?} players take another {:.1} µs to choose",
        1e6 * stepping / s.decisions as f64,
        1e6 * choosing / s.decisions as f64
    );
    println!(
        "  an observation is {} bytes: {OBS_F} features, {OBS_I} ids, {OBS_M} bytes of mask",
        4 * OBS_F + 2 * OBS_I + OBS_M
    );
}
