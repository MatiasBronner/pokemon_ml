//! Replays battles recorded from Pokémon Showdown and checks that this engine
//! reaches the same state, RNG seed and legal choices after every decision.
//!
//!     difftest cases.jsonl [--max-failures N] [--quiet] [--rebuild | --by-hand]
//!
//! `--rebuild` also writes the position down at every decision
//! (`Battle::to_state`), builds a battle back from it and carries on with
//! that one. `--by-hand` does the same and checks in addition that a battle
//! built from the position without the simulator's bookkeeping comes out the same.

use std::io::{BufRead, BufReader};
use vgc_engine::replay::{Case, Outcome, Rebuild, check_case_with};

fn main() {
    let mut path = None;
    let mut max_failures = 3usize;
    let mut quiet = false;
    let mut mode = Rebuild::No;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--max-failures" => {
                max_failures = args.next().and_then(|n| n.parse().ok()).expect("--max-failures takes a number")
            }
            "--quiet" => quiet = true,
            "--rebuild" => mode = Rebuild::Exact,
            "--by-hand" => mode = Rebuild::ByHand,
            _ => path = Some(a),
        }
    }
    let path = path.expect("usage: difftest cases.jsonl [--max-failures N] [--quiet] [--rebuild | --by-hand]");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let (mut passed, mut unsupported, mut decisions) = (0usize, 0usize, 0usize);
    let mut failed = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.expect("read error");
        if line.trim().is_empty() {
            continue;
        }
        let case: Case = serde_json::from_str(&line).expect("malformed case");
        match check_case_with(&case, mode) {
            Outcome::Pass(n) => {
                passed += 1;
                decisions += n;
            }
            Outcome::Unsupported(what) => {
                unsupported += 1;
                if !quiet {
                    println!("case {}: skipped, {what} is not modelled", case.id);
                }
            }
            Outcome::Fail(report) => {
                failed.push(case.id);
                if !quiet && failed.len() <= max_failures {
                    println!("case {}: {}", case.id, report[0]);
                    for l in &report[1..] {
                        println!("    {l}");
                    }
                }
            }
        }
    }
    println!(
        "{passed} battles matched Showdown at all {decisions} decisions; {} diverged; {unsupported} skipped as unsupported",
        failed.len()
    );
    if !failed.is_empty() {
        println!("diverging case ids: {:?}", &failed[..failed.len().min(40)]);
        std::process::exit(1);
    }
}
