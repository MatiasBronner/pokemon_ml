//! What the order of moves shows about Speed (`vgc_engine::speed`): an
//! example worked through, and then the property that matters: over
//! thousands of games with every legal ability and item, what a side has
//! worked out about the other never rules out the truth, and what it says
//! about who moves first is never wrong.

#![allow(clippy::needless_range_loop)]

use vgc_engine::env::{Baseline, Game, preview_table};
use vgc_engine::format::{Format, ShowdownSet};
use vgc_engine::obs::{ACT_F, ACTIVES, FIELD_F, MON_F, MONS, OBS_F, OBS_I, ROSTER};
use vgc_engine::rng::Rng;
use vgc_engine::speed::First;
use vgc_engine::{MonRef, PokemonSet, Request};

fn set(species: &str, nature: &str, speed_points: u8, item: &str) -> PokemonSet {
    PokemonSet::from_names(species, &["Calm Mind"], nature, [0, 0, 0, 0, 0, speed_points]).unwrap().item(item).unwrap()
}

fn team(leads: [PokemonSet; 2]) -> Vec<PokemonSet> {
    let [a, b] = leads;
    vec![
        a,
        b,
        set("Arcanine", "Hardy", 0, ""),
        set("Kingambit", "Hardy", 0, ""),
        set("Rillaboom", "Hardy", 0, ""),
        set("Volcarona", "Hardy", 0, ""),
    ]
}

/// Milotic (Speed 101) and Sylveon (80) face a Garchomp at full Speed (169)
/// and a Snorlax that would be slow (90 at the very most) but for the Choice
/// Scarf nobody has seen. Everyone uses the same move.
#[test]
fn one_turn_of_moves_narrows_what_each_can_be() {
    let ours = team([set("Milotic", "Hardy", 0, ""), set("Sylveon", "Hardy", 0, "")]);
    let theirs = team([set("Garchomp", "Jolly", 32, ""), set("Snorlax", "Jolly", 32, "Choice Scarf")]);
    let (garchomp, snorlax) = (theirs[0].species, theirs[1].species);
    let mut game = Game::new([ours, theirs], false).unwrap();
    game.check_speeds();

    // At Team Preview: anything the rules allow, from an Iron Ball on the slowest build to a Choice Scarf on the fastest.
    assert_eq!(game.speeds().belief(1, 0).range(garchomp), (54, 253));
    assert_eq!(game.speeds().belief(1, 0).items(), [true, false, true, false]);

    game.act([[0, 0], [0, 0]], [1, 2, 3, 4]).unwrap();
    let b = game.battle().unwrap();
    assert_eq!((game.speeds().own(0, 0), game.speeds().own(0, 1)), (101, 80));
    assert_eq!((game.speeds().own(1, 0), game.speeds().own(1, 1)), (169, 135));
    // Nothing has moved yet, so nothing is known about who is faster.
    assert_eq!(game.speeds().first(b, 0, 0, 0), Some(First::Unknown));

    // One turn: Garchomp, Snorlax, Milotic, Sylveon.
    game.act([[0, 46], [0, 46]].map(|_| [0, 0]), [1, 2, 3, 4]).unwrap();
    let (b, speeds) = (game.battle().unwrap(), game.speeds());

    // Garchomp went before Milotic (101). Any Garchomp does that unless it holds an Iron Ball: so it does not.
    let seen = speeds.belief(1, 0);
    assert_eq!(seen.range(garchomp), (109, 253));
    assert_eq!(seen.items(), [true, false, false, false]);
    // Snorlax went before Milotic too, and no Snorlax reaches 101 by itself: it holds a Choice Scarf.
    let seen = speeds.belief(1, 1);
    assert_eq!(seen.items(), [true, true, false, false]);
    assert_eq!(seen.range(snorlax), (102, 135));
    // So next turn, with nothing changed, both of theirs go before both of ours, as far as our side can tell...
    for (mine, theirs) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        assert_eq!(speeds.first(b, 0, mine, theirs), Some(First::Theirs));
    }
    // ...and their side knows as much about ours: Milotic went after Snorlax (135), Sylveon after Milotic says nothing to them.
    let milotic = game.rosters()[0][0].species;
    assert_eq!(speeds.belief(0, 0).range(milotic).1, 135);
    assert_eq!(speeds.first(b, 1, 0, 0), Some(First::Mine));

    // The truth is among what is left, for all four.
    for (side, entry) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        let set = &game.rosters()[side][entry];
        assert!(speeds.belief(side, entry).has(set.nature, set.stat_points[5], set.item));
    }
}

/// With open team sheets the nature and the item are on the sheet, and only the stat points are left to find.
#[test]
fn an_open_sheet_leaves_only_the_stat_points() {
    let ours = team([set("Milotic", "Hardy", 0, ""), set("Sylveon", "Hardy", 0, "")]);
    let theirs = team([set("Garchomp", "Jolly", 32, ""), set("Snorlax", "Jolly", 32, "Choice Scarf")]);
    let (garchomp, snorlax) = (theirs[0].species, theirs[1].species);
    let game = Game::new([ours, theirs], true).unwrap();
    // Jolly Garchomp with no item: 134 with no points to 169 with all of them.
    assert_eq!(game.speeds().belief(1, 0).range(garchomp), (134, 169));
    assert_eq!(game.speeds().belief(1, 0).items(), [false; 4]);
    // Jolly Snorlax with a Choice Scarf.
    assert_eq!(game.speeds().belief(1, 1).range(snorlax), (82, 135));
    assert_eq!(game.speeds().belief(1, 1).items(), [true, true, false, false]);
}

fn random_teams(seed: u32) -> [Vec<PokemonSet>; 2] {
    let format = Format::current();
    let mut rng = Rng::from_words([seed as u16, (seed >> 16) as u16, 6, 7]);
    // Through Showdown's own form and back, as a pool hands teams out.
    let mut one = || -> Vec<PokemonSet> {
        format.random_team(&mut rng).iter().map(|s| ShowdownSet::from_set(s).to_set().unwrap()).collect()
    };
    [one(), one()]
}

#[derive(Default, Debug)]
struct Tally {
    games: u32,
    decisions: u32,
    /// Pairs of facing Pokémon at a decision, and of those the ones a side could call.
    matchups: u32,
    called: u32,
    /// Opposing Pokémon that were seen, and the width of what was left of their Speed at the end, against the width at the start.
    seen: u32,
    left: f64,
    scarves_proven: u32,
    narrowed: u32,
}

/// Plays one game with the tracker checking itself, and checks it from outside at every decision.
fn play(seed: u32, open: bool, kinds: [Baseline; 2], tally: &mut Tally) {
    let mut rng = Rng::from_words([seed as u16, (seed >> 16) as u16, 9, 9]);
    let mut game = Game::new(random_teams(seed), open).unwrap();
    game.check_speeds();
    let start: Vec<Vec<(u32, u32)>> = (0..2)
        .map(|side| (0..ROSTER).map(|j| game.speeds().belief(side, j).range(game.rosters()[side][j].species)).collect())
        .collect();
    let picks = [rng.below(90) as usize, rng.below(90) as usize];
    game.act([[picks[0], 0], [picks[1], 0]], [seed as u16, (seed >> 16) as u16, 2, 3]).unwrap();
    let brought = picks.map(|p| preview_table()[p]);
    let (mut f, mut i, mut m) = (vec![0f32; OBS_F], vec![0i16; OBS_I], vec![0u8; vgc_engine::env::OBS_M]);
    loop {
        let b = game.battle().unwrap();
        let speeds = game.speeds();
        // 1. Nothing a side has worked out rules out how the other's Pokémon were really built.
        for side in 0..2 {
            for (idx, &entry) in brought[side].iter().enumerate() {
                let set = &game.rosters()[side][entry as usize];
                let item = b.mon(MonRef { side: side as u8, idx: idx as u8 }).item;
                assert!(
                    speeds.belief(side, entry as usize).has(set.nature, set.stat_points[5], item),
                    "seed {seed} turn {}: the truth about side {side}'s {entry} was ruled out: {} registered with {}, now holding {}, {:?} with {} points\n{:?}\n{:#?}",
                    b.turn,
                    vgc_engine::data::SPECIES[set.species as usize].name,
                    vgc_engine::data::ITEMS[set.item as usize].name,
                    vgc_engine::data::ITEMS[item as usize].name,
                    set.nature,
                    set.stat_points[5],
                    speeds.belief(side, entry as usize),
                    b.shown(side)
                );
            }
        }
        if b.ended || b.turn > 80 {
            break;
        }
        // 2. Whenever a side says who goes first, it is right. (`own` is the engine's own figure for each.)
        if b.request == Request::Move {
            for view in 0..2 {
                for mine in 0..2 {
                    for theirs in 0..2 {
                        let Some(call) = speeds.first(b, view, mine, theirs) else { continue };
                        let (me, them) = (speeds.own(view, mine), speeds.own(1 - view, theirs));
                        tally.matchups += 1;
                        tally.called += (call != First::Unknown) as u32;
                        match call {
                            First::Mine => {
                                assert!(me > them, "seed {seed} turn {}: {me} called faster than {them}", b.turn)
                            }
                            First::Theirs => {
                                assert!(me < them, "seed {seed} turn {}: {me} called slower than {them}", b.turn)
                            }
                            First::Unknown => {}
                        }
                    }
                }
            }
        }
        // 3. The observation carries it without falling over.
        for view in 0..2 {
            game.observe(view, &mut f, &mut i, &mut m);
            assert!(f.iter().all(|v| v.is_finite() && (-1.01..=4.0).contains(v)));
        }
        tally.decisions += 1;
        let actions = [game.baseline(0, kinds[0], &mut rng), game.baseline(1, kinds[1], &mut rng)];
        game.act(actions, [1, 1, 1, 1]).unwrap();
    }
    tally.games += 1;
    let b = game.battle().unwrap();
    for side in 0..2 {
        let shown = b.shown(side);
        for mon in shown.active.iter().flatten().chain(&shown.bench) {
            let Some(j) = mon.listed else { continue };
            let belief = game.speeds().belief(side, j as usize);
            let species = game.rosters()[side][j as usize].species;
            let (lo, hi) = belief.range(species);
            let (lo0, hi0) = start[side][j as usize];
            tally.seen += 1;
            tally.left += (hi - lo) as f64 / (hi0 - lo0).max(1) as f64;
            tally.narrowed += ((lo, hi) != (lo0, hi0)) as u32;
            tally.scarves_proven += (belief.items()[1] && !open) as u32;
        }
    }
    let _ = (FIELD_F, MON_F, MONS, ACT_F, ACTIVES);
}

#[test]
fn the_truth_is_never_ruled_out_and_no_call_is_wrong() {
    // `SPEED_GAMES=300000 cargo test --release --test speed` for a longer look.
    let games = std::env::var("SPEED_GAMES").ok().and_then(|n| n.parse().ok()).unwrap_or(3000u32);
    let mut tally = Tally::default();
    for seed in 0..games {
        let kinds = match seed % 3 {
            0 => [Baseline::Random, Baseline::Random],
            1 => [Baseline::Greedy, Baseline::Random],
            _ => [Baseline::Greedy, Baseline::Greedy],
        };
        play(1000 + seed, seed % 2 == 0, kinds, &mut tally);
    }
    println!("{tally:?}");
    println!(
        "{} games, {} decisions: {:.0}% of matchups called; a Pokémon that was seen ends with {:.0}% of its range of Speed left ({:.0}% of them narrowed at all); {} Choice Scarves worked out",
        tally.games,
        tally.decisions,
        100.0 * tally.called as f64 / tally.matchups as f64,
        100.0 * tally.left / tally.seen as f64,
        100.0 * tally.narrowed as f64 / tally.seen as f64,
        tally.scarves_proven
    );
    // The check is worth something only if the tracker does something.
    assert!(tally.called * 5 > tally.matchups, "few matchups called: {tally:?}");
    assert!(tally.narrowed * 2 > tally.seen, "little narrowed: {tally:?}");
}
