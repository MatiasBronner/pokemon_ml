//! Checks the engine's team validator against Pokémon Showdown's.
//!
//!     teamcheck teams.jsonl [--format formats/<id>.json] [--max-failures N]
//!         `teams.jsonl` comes from `oracle/gen_teams.js`: teams with Showdown's verdict.
//!         Every team must get the same verdict here.
//!     teamcheck --sample N [--seed S] [--format formats/<id>.json]
//!         Prints N random legal teams in Showdown's JSON, one to a line, for
//!         `oracle/gen_teams.js --judge` to put before Showdown's validator.

use std::io::{BufRead, BufReader, Write};

use serde::Deserialize;
use vgc_engine::format::{Format, ShowdownSet, showdown_team};
use vgc_engine::rng::Rng;

#[derive(Deserialize)]
struct Judged {
    id: usize,
    faults: Vec<String>,
    legal: bool,
    problems: Vec<String>,
    team: Vec<ShowdownSet>,
}

fn main() {
    let mut path = None;
    let mut format_path = None;
    let mut max_failures = 5usize;
    let mut sample = None;
    let mut seed = 1u16;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut number = |what: &str| {
            args.next().and_then(|n| n.parse::<usize>().ok()).unwrap_or_else(|| panic!("{what} takes a number"))
        };
        match a.as_str() {
            "--max-failures" => max_failures = number("--max-failures"),
            "--sample" => sample = Some(number("--sample")),
            "--seed" => seed = number("--seed") as u16,
            "--format" => format_path = args.next(),
            _ => path = Some(a),
        }
    }
    let loaded;
    let format = match &format_path {
        Some(p) => {
            let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("cannot read {p}: {e}"));
            loaded = Format::from_json(&text).unwrap_or_else(|e| panic!("{p}: {e}"));
            &loaded
        }
        None => Format::current(),
    };

    if let Some(n) = sample {
        let mut rng = Rng::from_words([seed, 2026, 10, 9]);
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        for _ in 0..n {
            let team = format.random_team(&mut rng);
            assert!(
                format.is_legal(&team),
                "the sampler made a team the validator refuses: {:?}",
                format.check_team(&team)
            );
            writeln!(out, "{}", showdown_team(&team)).expect("write error");
        }
        return;
    }

    let path =
        path.expect("usage: teamcheck teams.jsonl [--format file] | teamcheck --sample N [--seed S] [--format file]");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let (mut agreed, mut legal) = (0usize, 0usize);
    let mut failed = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.expect("read error");
        if line.trim().is_empty() {
            continue;
        }
        let case: Judged = serde_json::from_str(&line).expect("malformed line");
        let violations = format.check_showdown_team(&case.team);
        if violations.is_empty() == case.legal {
            agreed += 1;
            legal += case.legal as usize;
            continue;
        }
        failed.push(case.id);
        if failed.len() <= max_failures {
            println!(
                "team {} (made with: {}):",
                case.id,
                if case.faults.is_empty() { "nothing wrong".into() } else { case.faults.join(", ") }
            );
            println!("    Showdown: {}", if case.legal { "legal".into() } else { case.problems.join(" / ") });
            let ours: Vec<String> = violations.iter().map(|v| v.to_string()).collect();
            println!("    engine:   {}", if ours.is_empty() { "legal".into() } else { ours.join(" / ") });
            for s in &case.team {
                println!("      {} @ {} [{}] {:?} {:?}", s.species, s.item, s.ability, s.moves, s.evs);
            }
        }
    }
    println!(
        "{}: same verdict as Showdown on {agreed} teams ({legal} legal, {} not); {} differ",
        format.name,
        agreed - legal,
        failed.len()
    );
    if !failed.is_empty() {
        println!("differing team ids: {:?}", &failed[..failed.len().min(40)]);
        std::process::exit(1);
    }
}
