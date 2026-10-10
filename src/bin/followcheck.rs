//! Checks the log follower against recorded Showdown battles.
//!
//!     followcheck CASES.jsonl... [--limit N] [--battle ID] [--slim OUT.jsonl]
//!
//! The battles must have been recorded with their logs and requests
//! (`oracle/gen_cases.js --log --requests`). Each is played in the
//! simulator, and beside it a follower for each side reads what Showdown
//! sent that side. Printed: how many observations were compared, how many
//! differed, and which features differed how often.

use std::io::{BufRead, BufReader};

use vgc_engine::replay::{FollowCase, FollowTally, Outcome, check_follow};

fn main() {
    let mut files = Vec::new();
    let mut limit = usize::MAX;
    let mut only = None;
    let mut slim = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--limit" {
            limit = args.next().and_then(|n| n.parse().ok()).expect("--limit takes a number");
        } else if arg == "--slim" {
            // Write the battles followed back out with only what this check reads.
            let path = args.next().expect("--slim takes a file to write");
            slim = Some(std::fs::File::create(&path).unwrap_or_else(|e| panic!("cannot write {path}: {e}")));
        } else if arg == "--battle" {
            // Every difference found in this one battle.
            only = Some(args.next().and_then(|n| n.parse::<u32>().ok()).expect("--battle takes a number"));
        } else {
            files.push(arg);
        }
    }
    if files.is_empty() {
        eprintln!("usage: followcheck CASES.jsonl... [--limit N] [--battle ID] [--slim OUT.jsonl]");
        std::process::exit(2);
    }
    let mut tally = FollowTally { verbose: only.is_some(), ..FollowTally::default() };
    let (mut battles, mut decisions, mut skipped, mut failed) = (0, 0, 0, Vec::new());
    'files: for path in &files {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        for line in BufReader::new(file).lines() {
            if battles + skipped + failed.len() >= limit {
                break 'files;
            }
            let line = line.expect("a readable line");
            let case: FollowCase = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{path}: {e}"));
            if only.is_some_and(|id| id != case.id) {
                continue;
            }
            match check_follow(&case, &mut tally) {
                Outcome::Pass(n) => {
                    battles += 1;
                    decisions += n;
                    if let Some(out) = slim.as_mut() {
                        use std::io::Write;
                        writeln!(out, "{}", serde_json::to_string(&case).expect("a case is serialisable"))
                            .expect("a writable file");
                    }
                }
                Outcome::Unsupported(_) => skipped += 1,
                Outcome::Fail(why) => failed.push(format!("battle {}: {}", case.id, why.join("; "))),
            }
        }
    }
    println!(
        "{battles} battles followed to the end ({decisions} decisions), {skipped} skipped, {} not followed",
        failed.len()
    );
    for why in failed.iter().take(20) {
        println!("  {why}");
    }
    println!(
        "{} observations compared, {} differ ({:.2}%); {} passed over (a move or a switch barred in a way the player is not shown)",
        tally.observations,
        tally.wrong,
        100.0 * tally.wrong as f64 / tally.observations.max(1) as f64,
        tally.hidden_disables
    );
    let mut rows: Vec<_> = tally.by_feature.iter().collect();
    rows.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (name, (n, example)) in rows {
        println!("{n:>8}  {name}\n          {example}");
    }
    if !failed.is_empty() {
        std::process::exit(1);
    }
}
