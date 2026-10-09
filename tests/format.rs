//! Team legality: `Format` and its validator.
//!
//! The regulation files are written from what Pokémon Showdown's team
//! validator accepts, and `scripts/check-teams.sh` compares verdicts on tens
//! of thousands of teams. The fixtures here are a small sample of those.

use serde::Deserialize;
use vgc_engine::format::{Format, ShowdownSet, Violation, showdown_team};
use vgc_engine::rng::Rng;
use vgc_engine::{Battle, Error, Gender, PokemonSet};

const REG_MB: &str = include_str!("../formats/gen9championsvgc2026regmb.json");

fn set(species: &str, moves: &[&str]) -> PokemonSet {
    PokemonSet::from_names(species, moves, "Hardy", [0; 6]).unwrap()
}

/// Six Pokémon nothing is wrong with.
fn team() -> Vec<PokemonSet> {
    let build =
        |species, moves: &[&str], ability, item| set(species, moves).ability(ability).unwrap().item(item).unwrap();
    vec![
        build("Garchomp", &["Earthquake", "Rock Slide", "Dragon Claw", "Protect"], "Rough Skin", "Choice Scarf"),
        build("Sylveon", &["Hyper Voice", "Moonblast", "Calm Mind", "Protect"], "Pixilate", "Leftovers"),
        build("Arcanine", &["Flare Blitz", "Extreme Speed", "Will-O-Wisp", "Protect"], "Intimidate", "Sitrus Berry"),
        build("Milotic", &["Scald", "Ice Beam", "Recover", "Icy Wind"], "Competitive", "Rocky Helmet"),
        build("Charizard", &["Heat Wave", "Air Slash", "Protect"], "Blaze", "Charizardite Y"),
        build("Kangaskhan", &["Fake Out", "Double-Edge", "Sucker Punch"], "Scrappy", "Kangaskhanite").gender(Gender::F),
    ]
}

fn problems(format: &Format, team: &[PokemonSet]) -> Vec<String> {
    format.check_team(team).iter().map(|v| v.to_string()).collect()
}

#[test]
fn the_regulation_the_engine_is_built_for() {
    let f = Format::current();
    assert_eq!(f.id, "gen9championsvgc2026regmc");
    assert_eq!(f.species().len(), 293);
    let t = &f.team;
    assert_eq!((t.min_size, t.max_size, t.picked, t.level), (6, 6, 4, 50));
    assert!(t.species_clause);
    assert_eq!((t.item_clause, t.stat_points_total, t.stat_points_per_stat), (1, 66, 32));
    assert!(f.simulated_exactly(), "the engine's own regulation differs from the engine: {:?}", f.differences);
}

#[test]
fn a_legal_team_and_what_can_be_wrong_with_one() -> Result<(), Error> {
    let f = Format::current();
    assert_eq!(problems(f, &team()), Vec::<String>::new());

    let mut t = team();
    t[0].moves[3] = set("Sylveon", &["Moonblast"]).moves[0];
    assert_eq!(problems(f, &t), ["Garchomp cannot learn Moonblast"]);

    let mut t = team();
    t[1] = t[1].clone().ability("Intimidate")?;
    assert_eq!(problems(f, &t), ["Sylveon cannot have Intimidate"]);

    let mut t = team();
    t[1].ability = set("Sylveon", &["Protect"]).ability;
    assert_eq!(problems(f, &t), ["Sylveon has no ability"]);

    let mut t = team();
    t[3] = t[3].clone().item("Leftovers")?;
    assert_eq!(problems(f, &t), ["2 Pokémon hold Leftovers; the limit is 1"]);

    let mut t = team();
    t[2] = set("Ninetales", &["Flamethrower"]).ability("Drought")?;
    t[3] = set("Ninetales-Alola", &["Blizzard"]).ability("Snow Warning")?;
    assert_eq!(problems(f, &t), ["two Pokémon with the same Pokédex number (Ninetales-Alola)"]);

    let mut t = team();
    t[0].stat_points = [32, 33, 0, 0, 0, 2];
    assert_eq!(
        problems(f, &t),
        ["Garchomp has 33 stat points in Attack", "Garchomp has 67 stat points in all; the limit is 66"]
    );

    let mut t = team();
    t[5].gender = Gender::M;
    assert_eq!(problems(f, &t), ["Kangaskhan cannot be M"]);

    let mut t = team();
    let first = t[0].moves[0];
    t[0].moves.push(first);
    assert_eq!(problems(f, &t), ["Garchomp has 5 moves", "Garchomp has Earthquake twice"]);

    let mut t = team();
    t.pop();
    assert_eq!(problems(f, &t), ["a team has 6 Pokémon, not 5"]);

    // A Mega is not brought; it is evolved into.
    let mut t = team();
    t[4].species = set("Charizard-Mega-Y", &["Protect"]).species;
    assert_eq!(problems(f, &t), ["Charizard-Mega-Y cannot be brought"]);

    // A stone its holder cannot use is still an item it may hold.
    let mut t = team();
    t[1] = t[1].clone().item("Garchompite")?;
    assert_eq!(problems(f, &t), Vec::<String>::new());
    Ok(())
}

#[test]
fn one_set_on_its_own() {
    let f = Format::current();
    let ok = set("Incineroar", &["Fake Out", "Flare Blitz", "Parting Shot", "Protect"]).ability("Intimidate").unwrap();
    assert!(f.check_set(&ok).is_empty());
    let rule = f.rule(ok.species).expect("Incineroar can be brought");
    assert!(rule.moves.contains(&ok.moves[0]));
    assert_eq!(rule.genders, [Gender::M, Gender::F]);
    let mega = set("Charizard-Mega-X", &["Protect"]);
    assert!(f.rule(mega.species).is_none());
    assert!(matches!(f.check_set(&mega)[..], [Violation::Species { .. }]));
}

#[derive(Deserialize)]
struct Judged {
    id: usize,
    legal: bool,
    problems: Vec<String>,
    team: Vec<ShowdownSet>,
}

fn agrees_with_showdown(format: &Format, fixture: &str) {
    let (mut legal, mut refused) = (0, 0);
    for line in fixture.lines().filter(|l| !l.trim().is_empty()) {
        let case: Judged = serde_json::from_str(line).expect("malformed fixture line");
        let ours = format.check_showdown_team(&case.team);
        assert_eq!(
            ours.is_empty(),
            case.legal,
            "team {}: Showdown says {:?}, the engine {:?}",
            case.id,
            case.problems,
            ours.iter().map(|v| v.to_string()).collect::<Vec<_>>()
        );
        if case.legal {
            legal += 1;
        } else {
            refused += 1;
        }
    }
    assert!(legal > 100 && refused > 100, "{legal} legal, {refused} refused");
}

#[test]
fn verdicts_match_showdowns() {
    agrees_with_showdown(Format::current(), include_str!("fixtures/judged_teams_regmc.jsonl"));
}

#[test]
fn another_regulation_is_another_file() -> Result<(), Error> {
    let mb = Format::from_json(REG_MB)?;
    agrees_with_showdown(&mb, include_str!("fixtures/judged_teams_regmb.jsonl"));
    assert!(mb.species().len() < Format::current().species().len());
    assert!(mb.items().len() < Format::current().items().len());
    // The file says where its data differs from what the engine simulates.
    assert!(!mb.simulated_exactly());
    assert_eq!(mb.differences.len(), 2, "{:?}", mb.differences);
    assert!(mb.differences.iter().any(|d| d.starts_with("move wish: pp")));
    // And the two regulations really do disagree about some teams.
    let mc = Format::current();
    let only_now = mc.species().iter().find(|r| mb.rule(r.species).is_none()).expect("M-C added species");
    let newcomer = mc.random_set(only_now, &mut Rng::from_words([1, 2, 3, 4]));
    assert!(mc.check_set(&newcomer).is_empty());
    assert!(matches!(mb.check_set(&newcomer)[..], [Violation::Species { .. }]));
    Ok(())
}

#[test]
fn random_teams_are_legal_and_can_be_played() -> Result<(), Error> {
    let f = Format::current();
    let mut rng = Rng::from_words([20, 26, 10, 9]);
    let mut seen = std::collections::HashSet::new();
    for n in 0..500 {
        let team = f.random_team(&mut rng);
        assert_eq!(problems(f, &team), Vec::<String>::new());
        seen.extend(team.iter().map(|s| s.species));
        // Through Showdown's JSON and back it is the same team.
        let text = showdown_team(&team);
        let back: Vec<ShowdownSet> = serde_json::from_str(&text).unwrap();
        assert!(f.check_showdown_team(&back).is_empty());
        for (a, b) in team.iter().zip(&back) {
            let b = b.to_set()?;
            assert_eq!(
                (a.species, &a.moves, a.nature, a.stat_points, a.ability, a.item, a.gender),
                (b.species, &b.moves, b.nature, b.stat_points, b.ability, b.item, b.gender)
            );
        }
        // And the simulator takes the first four against the last four.
        if n % 10 == 0 {
            let b = Battle::new([&team[..4], &team[2..]], [n as u16, 1, 2, 3])?;
            assert!(!b.joint_choices(0).is_empty());
        }
    }
    assert!(seen.len() > 280, "only {} of the species came up", seen.len());
    Ok(())
}

#[test]
fn a_file_the_engine_cannot_play_is_refused() {
    let text =
        include_str!("../formats/gen9championsvgc2026regmc.json").replacen("\"garchomp\"", "\"garchompgalar\"", 1);
    match Format::from_json(&text) {
        Err(Error::Unsupported(what)) => assert!(what.contains("species garchompgalar"), "{what}"),
        other => panic!("expected the file to be refused, got {:?}", other.map(|f| f.id)),
    }
    assert!(Format::from_json("{").is_err());
}
