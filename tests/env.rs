//! The training environment (`vgc_engine::env`) and what it shows a model
//! (`vgc_engine::obs`): actions and choices agree, the masks say what is
//! legal, nothing hidden reaches the other side, and many games run at once.

use vgc_engine::data::{ABILITIES, ITEMS, MOVES, SPECIES, VolKind, ab, ability_id, item_id, move_id, species_id};
use vgc_engine::env::{Baseline, Config, Game, N_ACTIONS, N_PREVIEW, OBS_M, VecEnv, preview_table};
use vgc_engine::format::{Format, ShowdownSet};
use vgc_engine::obs::{
    ACT_F, ACT_VOLATILES, ACTIVES, FIELD_F, HIDDEN_VOLATILES, ID_BASE, ID_NONE, ID_UNKNOWN, MON_F, MON_IDS, MONS,
    OBS_F, OBS_I, Phase, ROSTER, item_table, layout, move_table, species_table, vocab,
};
use vgc_engine::rng::Rng;
use vgc_engine::teams::{Pool, PoolTeam, Variation};
use vgc_engine::{PokemonSet, Request};

fn pool(teams: usize, seed: u16) -> Pool {
    let format = Format::current();
    let mut rng = Rng::from_words([seed, 1, 2, 3]);
    let team = |k: usize, rng: &mut Rng| PoolTeam {
        player: format!("player {k}"),
        placing: Some(k as u32 + 1),
        record: String::new(),
        country: String::new(),
        section: String::new(),
        url: String::new(),
        spreads_guessed: false,
        natures_guessed: false,
        team: format.random_team(rng).iter().map(ShowdownSet::from_set).collect(),
    };
    Pool {
        source: String::new(),
        event: "made up".into(),
        format: format.id.clone(),
        teams: (0..teams).map(|k| team(k, &mut rng)).collect(),
    }
}

fn teams(seed: u16) -> [Vec<PokemonSet>; 2] {
    let p = pool(2, seed);
    [p.teams[0].sets().unwrap(), p.teams[1].sets().unwrap()]
}

struct Obs {
    f: Vec<f32>,
    i: Vec<i16>,
    m: Vec<u8>,
}

impl Obs {
    fn of(game: &Game, view: usize) -> Obs {
        let mut o = Obs { f: vec![9.0; OBS_F], i: vec![9; OBS_I], m: vec![9; OBS_M] };
        game.observe(view, &mut o.f, &mut o.i, &mut o.m);
        o
    }
    fn mon_feats(&self, token: usize) -> &[f32] {
        &self.f[FIELD_F + token * MON_F..FIELD_F + (token + 1) * MON_F]
    }
    fn mon_ids(&self, token: usize) -> &[i16] {
        &self.i[token * MON_IDS..(token + 1) * MON_IDS]
    }
    fn act_feats(&self, slot: usize) -> &[f32] {
        let at = FIELD_F + MONS * MON_F + slot * ACT_F;
        &self.f[at..at + ACT_F]
    }
    fn act_ids(&self, slot: usize) -> &[i16] {
        &self.i[MONS * MON_IDS + 2 * slot..MONS * MON_IDS + 2 * slot + 2]
    }
    fn info(&self) -> &[i16] {
        &self.i[OBS_I - 4..]
    }
}

/// Plays a game to its end with random legal actions, calling `at` at every decision.
fn play(game: &mut Game, rng: &mut Rng, mut at: impl FnMut(&Game)) {
    loop {
        at(game);
        let mut actions = [[0usize; 2]; 2];
        for (side, a) in actions.iter_mut().enumerate() {
            *a = game.baseline(side, Baseline::Random, rng);
        }
        let seed = [rng.below(65536) as u16, 2, 3, 4];
        game.act(actions, seed).unwrap();
        let b = game.battle().unwrap();
        if b.ended || b.turn > 150 {
            return;
        }
    }
}

#[test]
fn the_layout_adds_up() {
    let (f, i) = layout();
    let end = |parts: &[(&str, usize, Vec<usize>)]| {
        let (_, at, shape) = parts.last().unwrap();
        at + shape.iter().product::<usize>()
    };
    assert_eq!((end(&f), end(&i)), (OBS_F, OBS_I));
    assert_eq!(OBS_M, N_ACTIONS * (N_ACTIONS + 1));
    let [species, items, abilities, moves] = vocab();
    assert_eq!(species, SPECIES.len() + 2);
    assert_eq!((species_table().rows, item_table().rows, move_table().rows), (species, items, moves));
    assert_eq!(abilities, ABILITIES.len() + 2);
    for t in [species_table(), item_table(), move_table()] {
        assert_eq!(t.data.len(), t.rows * t.cols);
        assert!(t.data[..2 * t.cols].iter().all(|&v| v == 0.0), "the rows for nothing and unknown are blank");
        assert!(t.data.iter().all(|v| v.is_finite() && v.abs() <= 4.0));
    }
}

#[test]
fn team_preview_offers_every_way_to_bring_four() {
    let table = preview_table();
    let mut seen = std::collections::HashSet::new();
    for pick in table {
        let mut sorted = *pick;
        sorted.sort_unstable();
        assert!(sorted.windows(2).all(|w| w[0] < w[1]) && sorted[3] < 6, "{pick:?}");
        assert!(pick[0] < pick[1] && pick[2] < pick[3]);
        assert!(seen.insert(*pick));
    }
    assert_eq!(seen.len(), N_PREVIEW);

    // The pick decides who is brought and who leads.
    let mut game = Game::new(teams(5), true).unwrap();
    assert_eq!(game.phase(), Phase::Preview);
    let action = table.iter().position(|p| *p == [1, 4, 0, 5]).unwrap();
    let registered: Vec<u16> = game.rosters()[0].iter().map(|s| s.species).collect();
    game.act([[action, 0], [0, 0]], [1, 2, 3, 4]).unwrap();
    let b = game.battle().unwrap();
    let brought: Vec<u16> = (0..4).map(|p| b.mon(b.active_at(0, p)).species).collect();
    assert_eq!(brought, [registered[1], registered[4], registered[0], registered[5]]);
    assert_eq!(b.sides[0].n, 4);
    assert!(b.open_team_sheets());
}

#[test]
fn actions_and_choices_agree_and_the_masks_say_what_is_legal() {
    let mut rng = Rng::from_words([7, 7, 7, 7]);
    let (mut decisions, mut switches, mut megas, mut forced) = (0, 0, 0, 0);
    for seed in 0..60 {
        let mut game = Game::new(teams(100 + seed), seed % 2 == 0).unwrap();
        play(&mut game, &mut rng, |game| {
            let Some(b) = game.battle() else { return };
            decisions += 1;
            for side in 0..2 {
                let joint = b.joint_choices(side);
                let legal = game.legal(side);
                assert_eq!(joint.len(), legal.len());
                let mut mask = vec![0u8; OBS_M];
                assert_eq!(game.masks(side, &mut mask), joint.len());
                for (c, a) in joint.iter().zip(&legal) {
                    for pos in 0..2 {
                        // A number names one choice, and that choice has that number.
                        assert_eq!(game.choice_of(side, a[pos]), Some(c[pos]), "{:?} as {}", c[pos], a[pos]);
                        assert!(a[pos] < N_ACTIONS);
                    }
                    assert_eq!(mask[a[0]], 1);
                    assert_eq!(mask[N_ACTIONS * (1 + a[0]) + a[1]], 1);
                    switches += (40..46).contains(&a[0]) as usize;
                    megas += (a[0] < 40 && a[0] % 2 == 1) as usize;
                }
                // And nothing else is marked legal.
                let firsts: std::collections::HashSet<usize> = legal.iter().map(|a| a[0]).collect();
                assert_eq!(mask[..N_ACTIONS].iter().filter(|&&x| x == 1).count(), firsts.len());
                assert_eq!(mask[N_ACTIONS..].iter().filter(|&&x| x == 1).count(), legal.len());
                forced += (b.request == Request::Switch && legal.len() == 1) as usize;
            }
        });
    }
    assert!(decisions > 500 && switches > 500 && megas > 100 && forced > 20, "{decisions} {switches} {megas} {forced}");
}

/// The id a name has in an observation.
fn named<T>(name: &str, find: impl Fn(&str) -> Option<T>) -> i16
where
    T: Into<i64>,
{
    find(name).unwrap_or_else(|| panic!("no {name}")).into() as i16 + ID_BASE
}

#[test]
fn the_other_side_is_given_as_the_battle_has_shown_it() {
    let mut rng = Rng::from_words([3, 1, 4, 1]);
    let (mut checked, mut unknown_items, mut known_items, mut lost) = (0, 0, 0, 0);
    let (mut hidden_own, mut active_seen) = (0, 0);
    for seed in 0..80 {
        let open = seed % 2 == 1;
        let mut game = Game::new(teams(300 + seed), open).unwrap();
        play(&mut game, &mut rng, |game| {
            let Some(b) = game.battle() else { return };
            for view in 0..2 {
                let o = Obs::of(game, view);
                assert!(o.f.iter().all(|v| v.is_finite() && (-1.01..=4.0).contains(v)), "a feature out of range");
                let vocab = vocab();
                let shown = b.shown(1 - view);
                let on_show = shown.active.iter().flatten().chain(&shown.bench);
                for (j, listed) in shown.roster.iter().enumerate() {
                    let ids = o.mon_ids(ROSTER + j);
                    let feats = o.mon_feats(ROSTER + j);
                    let mon = on_show.clone().find(|m| m.listed == Some(j as u8));
                    let sheet = listed.sheet.as_ref();
                    assert_eq!(sheet.is_some(), open);
                    let species = mon.map_or(listed.species.as_str(), |m| m.species.as_str());
                    assert_eq!(ids[0], named(species, species_id), "species of {j}");
                    // Item: what the battle showed, else what the sheet says, else unknown.
                    let (item, was) = match (mon.and_then(|m| m.item.as_deref()), sheet) {
                        (Some(""), _) => (named("", |_| Some(0u16)), mon.unwrap().item_lost.as_str()),
                        (Some(held), _) => (named(held, item_id), ""),
                        (None, Some(s)) if s.item.is_empty() => (ID_BASE, ""),
                        (None, Some(s)) => (named(&s.item, item_id), ""),
                        (None, None) => (ID_UNKNOWN, ""),
                    };
                    assert_eq!(ids[1], item, "item of {j}");
                    assert_eq!(ids[2], if was.is_empty() { ID_NONE } else { named(was, item_id) });
                    let ability = match (mon.and_then(|m| m.ability.as_deref()), sheet) {
                        (Some(""), _) => ab::NOABILITY as i16 + ID_BASE,
                        (Some(a), _) => named(a, ability_id),
                        (None, Some(s)) => named(&s.ability, ability_id),
                        (None, None) => ID_UNKNOWN,
                    };
                    assert_eq!(ids[3], ability, "ability of {j}");
                    let moves: Vec<i16> = match (sheet, mon) {
                        (Some(s), _) => s.moves.iter().map(|m| named(m, move_id)).collect(),
                        (None, Some(m)) => m.moves.iter().take(4).map(|m| named(m, move_id)).collect(),
                        (None, None) => Vec::new(),
                    };
                    let rest = if open { ID_NONE } else { ID_UNKNOWN };
                    for k in 0..4 {
                        assert_eq!(ids[4 + k], moves.get(k).copied().unwrap_or(rest), "move {k} of {j}");
                    }
                    for (k, &id) in ids.iter().enumerate() {
                        let size = [vocab[0], vocab[1], vocab[1], vocab[2], vocab[3], vocab[3], vocab[3], vocab[3]][k];
                        assert!((0..size as i16).contains(&id));
                    }
                    // Seen, on the field, fainted, HP as the bar shows it, status.
                    let active = shown.active.iter().flatten().any(|m| m.listed == Some(j as u8));
                    assert_eq!((feats[0], feats[1]), (1.0, 0.0));
                    assert_eq!(feats[4], mon.is_some() as u8 as f32, "seen");
                    assert_eq!(feats[5], active as u8 as f32, "on the field");
                    assert_eq!(feats[6], mon.is_some_and(|m| m.fainted) as u8 as f32, "fainted");
                    assert_eq!(feats[7], mon.map_or(1.0, |m| m.hp as f32 / 100.0), "hp of {j}: {mon:?}\n{shown:#?}");
                    let status = mon.map_or(0, |m| {
                        ["", "brn", "par", "psn", "tox", "slp", "frz"].iter().position(|s| *s == m.status).unwrap()
                    });
                    assert_eq!(feats[8 + status], 1.0, "status");
                    // Stats are never given.
                    assert!(feats[16..23].iter().all(|&v| v == 0.0));
                    checked += 1;
                    unknown_items += (ids[1] == ID_UNKNOWN) as usize;
                    known_items += (ids[1] > ID_BASE && !open) as usize;
                    lost += (ids[2] != ID_NONE) as usize;
                }
                // The positions on the field point at the Pokémon standing there.
                for slot in 0..ACTIVES {
                    let token = o.act_ids(slot)[0];
                    if token == 0 {
                        assert!(o.act_feats(slot).iter().all(|&v| v == 0.0));
                        continue;
                    }
                    let token = token as usize - 1;
                    assert_eq!(token < ROSTER, slot < 2, "a side's positions hold its own Pokémon");
                    assert_eq!(o.mon_feats(token)[5], 1.0, "the Pokémon in a position is on the field");
                    active_seen += 1;
                    // What an unrevealed item or ability keeps is the holder's to know.
                    for kind in HIDDEN_VOLATILES {
                        let flag = o.act_feats(slot)[ACT_VOLATILES + kind as usize];
                        if slot >= 2 {
                            assert_eq!(flag, 0.0, "{kind:?} shown to the other side");
                        } else {
                            hidden_own += (flag == 1.0 && kind == VolKind::Choicelock) as usize;
                        }
                    }
                }
                let info = o.info();
                assert_eq!(info[0], game.phase() as i16);
                assert_eq!(info[1] as usize, game.legal(view).len());
            }
        });
    }
    // The check saw every case it is there for.
    assert!(
        checked > 10_000 && unknown_items > 2_000 && known_items > 300 && lost > 300,
        "{checked} {unknown_items} {known_items} {lost}"
    );
    assert!(active_seen > 5_000 && hidden_own > 5, "{active_seen} {hidden_own}");
    let _ = (ITEMS.len(), MOVES.len());
}

/// Two opponents that differ only in what Team Preview does not show give
/// the same observation at Team Preview with closed sheets, and a different
/// one with open sheets.
#[test]
fn team_preview_shows_species_and_no_more_unless_sheets_are_open() {
    let [ours, theirs] = teams(42);
    // The same six species with every hidden thing changed.
    let format = Format::current();
    let mut rng = Rng::from_words([9, 9, 9, 9]);
    let mut other = theirs.clone();
    for set in &mut other {
        let rule = format.rule(set.species).unwrap();
        for _ in 0..50 {
            let again = format.random_set(rule, &mut rng);
            if again.species == set.species && again.gender == set.gender && again != *set {
                *set = again;
                break;
            }
        }
    }
    assert_ne!(other, theirs);
    for open in [false, true] {
        let a = Obs::of(&Game::new([ours.clone(), theirs.clone()], open).unwrap(), 0);
        let b = Obs::of(&Game::new([ours.clone(), other.clone()], open).unwrap(), 0);
        assert_eq!(a.f == b.f && a.i == b.i, !open, "open sheets: {open}");
        // Its own team it sees in full either way.
        for j in 0..ROSTER {
            assert!([0, 1, 3, 4].iter().all(|&k| a.mon_ids(j)[k] >= ID_BASE));
            assert_eq!(a.mon_feats(j)[22], 1.0, "its own stats are known");
            let unknown = a.mon_ids(ROSTER + j)[1] == ID_UNKNOWN;
            assert_eq!(unknown, !open);
        }
        assert_eq!(a.info(), [Phase::Preview as i16, N_PREVIEW as i16, 0, 1]);
    }
}

#[test]
fn many_games_run_at_once_and_start_over() {
    let pool = pool(12, 1);
    let config = Config { envs: 64, seed: 5, max_turns: 60, threads: 2, ..Config::default() };
    let mut env = VecEnv::new(&pool, config.clone()).unwrap();
    let n = env.len();
    let (mut f, mut i, mut m) = (vec![0f32; 2 * n * OBS_F], vec![0i16; 2 * n * OBS_I], vec![0u8; 2 * n * OBS_M]);
    let (mut reward, mut done, mut actions) = (vec![0f32; 2 * n], vec![0u8; n], vec![0i32; 4 * n]);
    env.observe(&mut f, &mut i, &mut m);
    assert!((0..2 * n).all(|k| i[(k + 1) * OBS_I - 4] == Phase::Preview as i16));
    let (mut finished, mut total) = (0u64, 0.0f32);
    for step in 0..400 {
        env.baseline(if step % 2 == 0 { Baseline::Random } else { Baseline::Greedy }, &mut actions);
        // What a baseline gives is legal by the mask handed out with the observation.
        for k in 0..2 * n {
            let (info, mask) = (&i[(k + 1) * OBS_I - 4..(k + 1) * OBS_I], &m[k * OBS_M..(k + 1) * OBS_M]);
            let (a0, a1) = (actions[2 * k] as usize, actions[2 * k + 1] as usize);
            if info[0] == Phase::Preview as i16 {
                assert!(a0 < N_PREVIEW);
            } else {
                assert!(mask[a0] == 1 && mask[N_ACTIONS * (1 + a0) + a1] == 1, "game {} step {step}", k / 2);
            }
        }
        env.step(&actions, &mut f, &mut i, &mut m, &mut reward, &mut done).unwrap();
        for g in 0..n {
            let r = &reward[2 * g..2 * g + 2];
            if done[g] == 1 {
                finished += 1;
                assert!(r == [1.0, -1.0] || r == [-1.0, 1.0] || r == [0.0, 0.0]);
                // A new game has taken its place.
                assert_eq!(i[(2 * g + 1) * OBS_I - 4], Phase::Preview as i16);
            } else {
                assert_eq!(r, [0.0, 0.0]);
            }
            total += r[0] + r[1];
        }
    }
    let stats = env.stats();
    assert_eq!((stats.games, stats.decisions), (finished, 400 * n as u64));
    assert!(finished > 300 && total == 0.0, "{finished} games");
    assert!(stats.ties * 10 < stats.games, "{} of {} games tied or ran out of turns", stats.ties, stats.games);

    // The same seed plays the same games; an illegal action is refused.
    let mut run = |steps: usize| {
        let mut env = VecEnv::new(&pool, config.clone()).unwrap();
        let (mut f, mut i, mut m) = (vec![0f32; 2 * n * OBS_F], vec![0i16; 2 * n * OBS_I], vec![0u8; 2 * n * OBS_M]);
        for _ in 0..steps {
            env.baseline(Baseline::Random, &mut actions);
            env.step(&actions, &mut f, &mut i, &mut m, &mut reward, &mut done).unwrap();
        }
        (f, i, m)
    };
    assert!(run(50) == run(50));
    let mut env = VecEnv::new(&pool, config).unwrap();
    env.baseline(Baseline::Random, &mut actions);
    env.step(&actions, &mut f, &mut i, &mut m, &mut reward, &mut done).unwrap();
    actions.fill(46);
    assert!(env.step(&actions, &mut f, &mut i, &mut m, &mut reward, &mut done).is_err());
}

#[test]
fn the_greedy_player_beats_the_random_one() {
    let pool = pool(20, 3);
    let config = Config { envs: 128, seed: 9, variation: Variation::NONE, threads: 2, ..Config::default() };
    let mut env = VecEnv::new(&pool, config).unwrap();
    let n = env.len();
    let (mut f, mut i, mut m) = (vec![0f32; 2 * n * OBS_F], vec![0i16; 2 * n * OBS_I], vec![0u8; 2 * n * OBS_M]);
    let (mut reward, mut done) = (vec![0f32; 2 * n], vec![0u8; n]);
    let (mut greedy, mut random) = (vec![0i32; 4 * n], vec![0i32; 4 * n]);
    let (mut wins, mut games) = (0, 0);
    for _ in 0..600 {
        env.baseline(Baseline::Greedy, &mut greedy);
        env.baseline(Baseline::Random, &mut random);
        // The greedy player takes the first side of even games and the second of odd ones.
        for g in 0..n {
            let side = g % 2;
            random[4 * g + 2 * side..4 * g + 2 * side + 2]
                .copy_from_slice(&greedy[4 * g + 2 * side..4 * g + 2 * side + 2]);
        }
        env.step(&random, &mut f, &mut i, &mut m, &mut reward, &mut done).unwrap();
        for g in 0..n {
            if done[g] == 1 {
                games += 1;
                wins += (reward[2 * g + g % 2] > 0.0) as usize;
            }
        }
    }
    assert!(games > 1000 && wins * 100 > games * 80, "greedy won {wins} of {games}");
}

/// Rain from a Pelipper holding a Damp Rock lasts eight turns for five, and
/// nothing shows the rock. Until the fifth turn is over, the other side's
/// view is the same with the rock and without: it is told how long the rain
/// has been up, not how long it has left.
#[test]
fn a_hidden_item_that_extends_the_weather_does_not_show_in_its_timer() {
    let set = |species: &str, ability: &str, item: &str| {
        PokemonSet::from_names(species, &["Protect"], "Hardy", [0; 6])
            .unwrap()
            .ability(ability)
            .unwrap()
            .item(item)
            .unwrap()
    };
    let rest = |first: PokemonSet| {
        vec![
            first,
            set("Snorlax", "Thick Fat", ""),
            set("Garchomp", "Rough Skin", ""),
            set("Milotic", "Competitive", ""),
            set("Sylveon", "Pixilate", ""),
            set("Arcanine", "Intimidate", ""),
        ]
    };
    let game = |rock: bool| {
        let pelipper = set("Pelipper", "Drizzle", if rock { "Damp Rock" } else { "" });
        let mut g = Game::new([rest(pelipper), rest(set("Kingambit", "Defiant", ""))], false).unwrap();
        g.act([[0, 0], [0, 0]], [4, 3, 2, 1]).unwrap();
        g
    };
    let (mut with, mut without) = (game(true), game(false));
    // Where the field's features say how long the weather has been up.
    let elapsed = 3 + 1 + 1 + 5;
    for turn in 1..=7 {
        let (a, b) = (Obs::of(&with, 1), Obs::of(&without, 1));
        let raining = |o: &Obs| o.f[5 + 1] == 1.0;
        assert!(raining(&a), "turn {turn}");
        assert_eq!(a.f[elapsed], (turn - 1) as f32 / 8.0, "turn {turn}");
        if turn <= 5 {
            assert!(a.f == b.f && a.i == b.i, "the rock shows on turn {turn}");
        } else {
            // Without the rock the rain has stopped, which everyone sees.
            assert!(!raining(&b), "turn {turn}");
        }
        // The side holding the rock knows what it holds.
        assert_ne!(Obs::of(&with, 0).mon_ids(0)[1], Obs::of(&without, 0).mon_ids(0)[1]);
        for g in [&mut with, &mut without] {
            g.act([[0, 0], [0, 0]], [1, 1, 1, 1]).unwrap();
        }
    }
}
