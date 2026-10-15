//! What a search is handed (`Game::determinize`, `Game::reseed`): a copy of
//! the game in which what one side has not been shown of the other is drawn
//! anew. That side must not be able to tell the copy from the game, and the
//! copy must play on like any other game.

use vgc_engine::PokemonSet;
use vgc_engine::env::{Baseline, Game, OBS_M};
use vgc_engine::format::{Format, ShowdownSet};
use vgc_engine::obs::{OBS_F, OBS_I};
use vgc_engine::rng::Rng;
use vgc_engine::teams::PoolTeam;

fn teams(seed: u16) -> [Vec<PokemonSet>; 2] {
    let format = Format::current();
    let mut rng = Rng::from_words([seed, 1, 2, 3]);
    let mut team = || {
        let sets: Vec<ShowdownSet> = format.random_team(&mut rng).iter().map(ShowdownSet::from_set).collect();
        PoolTeam { team: sets, ..PoolTeam::default() }.sets().unwrap()
    };
    [team(), team()]
}

fn observation(game: &Game, view: usize) -> (Vec<f32>, Vec<i16>, Vec<u8>) {
    let (mut f, mut i, mut m) = (vec![0.0; OBS_F], vec![0; OBS_I], vec![0; OBS_M]);
    game.observe(view, &mut f, &mut i, &mut m);
    (f, i, m)
}

fn random_actions(game: &Game, rng: &mut Rng) -> [[usize; 2]; 2] {
    [game.baseline(0, Baseline::Random, rng), game.baseline(1, Baseline::Random, rng)]
}

fn finished(game: &Game) -> bool {
    game.battle().is_some_and(|b| b.ended || b.turn > 150)
}

/// Plays random games with open and closed sheets, and at every decision
/// redraws each side's view of the other: the observation that side is given
/// must not change by a single number, the copy must play on to the end,
/// and the redrawing must actually redraw something.
#[test]
fn a_redrawn_game_looks_the_same_to_the_side_that_searches_and_plays_on() {
    let games: u16 = std::env::var("SEARCH_GAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(120);
    let (mut decisions, mut benches_redrawn, mut stats_redrawn) = (0, 0, 0);
    for seed in 0..games {
        let mut rng = Rng::from_words([seed, 7, 7, 7]);
        let mut game = Game::new(teams(seed), seed % 2 == 0).unwrap();
        while !finished(&game) {
            if game.battle().is_some() {
                for view in 0..2 {
                    let mut copy = game.clone();
                    copy.determinize(view, &mut rng).unwrap();
                    assert_eq!(
                        observation(&game, view),
                        observation(&copy, view),
                        "game {seed}, turn {}: side {view} can tell the redrawn copy from the game",
                        game.battle().unwrap().turn
                    );
                    let (real, drawn) =
                        (&game.battle().unwrap().sides[1 - view], &copy.battle().unwrap().sides[1 - view]);
                    benches_redrawn +=
                        (0..real.n as usize).any(|i| real.team[i].species != drawn.team[i].species) as u32;
                    stats_redrawn += (0..real.n as usize).any(|i| real.team[i].stats != drawn.team[i].stats) as u32;
                    // Played on with dice of its own, the copy reaches an end like any game.
                    copy.reseed([seed, 1, view as u16, rng.below(65536) as u16]);
                    for _ in 0..3 {
                        if finished(&copy) {
                            break;
                        }
                        let actions = random_actions(&copy, &mut rng);
                        copy.act(actions, [1, 2, 3, 4]).unwrap();
                    }
                    decisions += 1;
                }
            }
            let actions = random_actions(&game, &mut rng);
            game.act(actions, [seed, 2, 3, 4]).unwrap();
        }
    }
    println!(
        "{decisions} redrawn views, {benches_redrawn} with a different bench, {stats_redrawn} with different stats"
    );
    assert!(benches_redrawn > decisions / 20, "the bench is hardly ever redrawn");
    assert!(stats_redrawn > decisions / 2, "stat points are hardly ever redrawn");
}

/// A copy given new dice does not roll the game's: played on with the same
/// choices, copies with different seeds come apart from each other.
#[test]
fn reseeded_copies_roll_their_own_dice() {
    let mut differ = 0;
    for seed in 0..40u16 {
        let mut rng = Rng::from_words([seed, 3, 3, 3]);
        let mut game = Game::new(teams(seed), true).unwrap();
        game.act([[0, 0], [0, 0]], [seed, 1, 1, 1]).unwrap();
        let actions = random_actions(&game, &mut rng);
        let outcomes: Vec<_> = (0..8u16)
            .map(|k| {
                let mut copy = game.clone();
                copy.reseed([k, k, k, k]);
                copy.act(actions, [0; 4]).unwrap();
                observation(&copy, 0).0
            })
            .collect();
        differ += outcomes.iter().any(|o| *o != outcomes[0]) as u32;
    }
    assert!(differ > 20, "reseeding hardly ever changes what happens ({differ} of 40)");
}

/// What a redrawn copy is drawn from must be what the searching side has seen,
/// and nothing else. A redrawn copy is itself a game that agrees with all of
/// that, so redrawing the real game and redrawing such a copy, with the same
/// random numbers, must come to the same thing: the same Pokémon, stats, HP,
/// countdowns and bench, and, played on with the same dice and the same
/// choices, the same turns. Anything hidden that slipped through (exact HP
/// within the bar, how long a sleep was drawn to last, the real stat points)
/// would tell the two apart.
#[test]
fn a_redrawn_copy_depends_on_nothing_the_searching_side_has_not_seen() {
    let games: u16 = std::env::var("SEARCH_GAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(120);
    let (mut compared, mut clocks) = (0, 0);
    for seed in 0..games {
        let mut rng = Rng::from_words([seed, 9, 9, 9]);
        // Open sheets: with closed ones the moves, items and abilities are not redrawn.
        let mut game = Game::new(teams(seed), true).unwrap();
        while !finished(&game) {
            if game.battle().is_some() {
                for view in 0..2 {
                    let mut other_truth = game.clone();
                    other_truth.determinize(view, &mut rng).unwrap();
                    let same = rng.next_u32() as u64;
                    let redraw = |g: &Game| {
                        let mut copy = g.clone();
                        copy.determinize(view, &mut Rng { seed: same, calls: 0 }).unwrap();
                        copy.reseed([seed, 4, 4, view as u16]);
                        copy
                    };
                    let (mut a, mut b) = (redraw(&game), redraw(&other_truth));
                    let turn = game.battle().unwrap().turn;
                    assert_eq!(
                        a.battle().unwrap().sides,
                        b.battle().unwrap().sides,
                        "game {seed}, turn {turn}, side {view}"
                    );
                    let sides = &game.battle().unwrap().sides;
                    clocks += sides.iter().flat_map(|s| &s.team[..s.n as usize]).any(|m| m.status_time > 0) as u32;
                    let mut dice = Rng::from_words([seed, turn, view as u16, 1]);
                    for _ in 0..3 {
                        if finished(&a) {
                            break;
                        }
                        let actions = random_actions(&a, &mut dice);
                        a.act(actions, [0; 4]).unwrap();
                        b.act(actions, [0; 4]).unwrap();
                        for side in 0..2 {
                            assert_eq!(
                                observation(&a, side),
                                observation(&b, side),
                                "game {seed}, turn {turn}: played on"
                            );
                        }
                        assert_eq!(
                            a.battle().unwrap().sides,
                            b.battle().unwrap().sides,
                            "game {seed}, turn {turn}: played on"
                        );
                    }
                    compared += 1;
                }
            }
            let actions = random_actions(&game, &mut rng);
            game.act(actions, [seed, 2, 3, 4]).unwrap();
        }
    }
    println!("{compared} positions redrawn from two different truths; {clocks} with a sleep or freeze counter running");
    assert!(compared > 1000);
}
