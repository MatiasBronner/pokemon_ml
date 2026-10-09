//! Replays battles recorded from Pokémon Showdown and checks that this engine
//! reaches the same state, RNG seed and legal choices after every decision.
//!
//!     difftest cases.jsonl [--max-failures N] [--quiet] [--rebuild | --by-hand | --shown]
//!
//! `--shown` checks something else: that what the engine says each side has
//! been shown (`Battle::shown`) is, at every decision, what a reader of
//! Showdown's log makes of it. The battles must have been recorded with
//! their logs (`gen_cases.js --log`). It can be combined with `--rebuild` or
//! `--by-hand`.
//!
//! `--rebuild` also writes the position down at every decision
//! (`Battle::to_state`), builds a battle back from it and carries on with
//! that one. `--by-hand` does the same and checks in addition that a battle
//! built from the position without the simulator's bookkeeping comes out the same.

use std::io::{BufRead, BufReader};
use vgc_engine::replay::{Case, Outcome, Rebuild, ShownTally, check_case_with, check_shown_with};

fn main() {
    let mut path = None;
    let mut max_failures = 3usize;
    let mut quiet = false;
    let mut mode = Rebuild::No;
    let mut shown = false;
    let mut tally = ShownTally::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--max-failures" => {
                max_failures = args.next().and_then(|n| n.parse().ok()).expect("--max-failures takes a number")
            }
            "--quiet" => quiet = true,
            "--rebuild" => mode = Rebuild::Exact,
            "--by-hand" => mode = Rebuild::ByHand,
            "--shown" => shown = true,
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
        let outcome = if shown { check_shown_with(&case, mode, &mut tally) } else { check_case_with(&case, mode) };
        match outcome {
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
    if shown {
        let total = |m: &std::collections::BTreeMap<String, (usize, String)>| m.values().map(|v| v.0).sum::<usize>();
        println!(
            "{passed} battles, {} decisions: {} differences between the engine and the log, {} untrue beliefs; {} could not be followed; {unsupported} skipped as unsupported",
            tally.decisions,
            total(&tally.mismatches),
            total(&tally.untrue),
            failed.len()
        );
        let expected = "expected, a Choice item's failure line naming a borrowed move";
        for (title, map) in
            [("differences", &tally.mismatches), ("untrue beliefs", &tally.untrue), (expected, &tally.expected)]
        {
            if !quiet && !map.is_empty() {
                println!("{title}:");
                let mut rows: Vec<_> = map.iter().collect();
                rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.0));
                for (what, (n, at)) in rows.iter().take(max_failures.max(40)) {
                    println!("  {n:6}  {what}   (first: {at})");
                }
            }
        }
        // With the `trace` feature: which places in the engine recorded something, and what.
        for (file, line, what, uses, news) in vgc_engine::trace::shown_sites() {
            println!("site {file}:{line} {what} {uses} {news}");
        }
        if !failed.is_empty() || !tally.mismatches.is_empty() || !tally.untrue.is_empty() {
            std::process::exit(1);
        }
        return;
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
