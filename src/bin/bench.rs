//! Rough throughput check: plays random battles to completion using teams
//! from a recorded case file.
//!
//!     bench cases.jsonl [battles]

use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use vgc_engine::data::{move_id, nature, species_id};
use vgc_engine::{ACTIVE, Battle, Choice, PokemonSet};

#[derive(Deserialize)]
struct SetJson {
    species: String,
    moves: Vec<String>,
    nature: String,
    sp: [u8; 6],
}

#[derive(Deserialize)]
struct Case {
    teams: [Vec<SetJson>; 2],
}

/// xorshift64*; deliberately separate from the battle's own RNG.
struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 33) as usize) % n
    }
}

fn pick_side(b: &Battle, side: usize, xs: &mut Xs) -> [Choice; ACTIVE] {
    let joint = b.joint_choices(side);
    joint[xs.below(joint.len())]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: bench cases.jsonl [battles]");
    let battles: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(100_000);
    let file = std::fs::File::open(path).expect("cannot open case file");
    let mut teams: Vec<[Vec<PokemonSet>; 2]> = Vec::new();
    for line in BufReader::new(file).lines().take(500) {
        let c: Case = serde_json::from_str(&line.unwrap()).unwrap();
        let conv = |t: &Vec<SetJson>| -> Vec<PokemonSet> {
            t.iter()
                .map(|s| PokemonSet {
                    species: species_id(&s.species).unwrap(),
                    moves: s.moves.iter().map(|m| move_id(m).unwrap()).collect(),
                    nature: nature(&s.nature).unwrap(),
                    stat_points: s.sp,
                })
                .collect()
        };
        teams.push([conv(&c.teams[0]), conv(&c.teams[1])]);
    }
    let mut xs = Xs(0x9E37_79B9_7F4A_7C15);
    let (mut turns, mut decisions, mut capped) = (0u64, 0u64, 0u64);
    let start = Instant::now();
    for i in 0..battles {
        let t = &teams[i % teams.len()];
        let seed = [xs.next() as u16, xs.next() as u16, xs.next() as u16, xs.next() as u16];
        let mut b = Battle::new([&t[0], &t[1]], seed).unwrap();
        while !b.ended {
            if b.turn > 300 {
                capped += 1;
                break;
            }
            let c = [pick_side(&b, 0, &mut xs), pick_side(&b, 1, &mut xs)];
            b.choose(c).unwrap();
            decisions += 1;
        }
        turns += b.turn as u64;
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{battles} random battles in {secs:.2}s on one core: {:.0} battles/s, {:.0} decisions/s, mean {:.1} turns ({capped} capped at 300 turns)",
        battles as f64 / secs,
        decisions as f64 / secs,
        turns as f64 / battles as f64
    );
}
