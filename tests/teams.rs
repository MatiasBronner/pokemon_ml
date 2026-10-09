//! Team sheets from outside (`vgc_engine::teams`): reading them, filling in
//! what a sheet leaves out, and keeping a pool. `scripts/check_scraper.py`
//! checks the whole path from a tournament's web page to a pool.

use vgc_engine::format::Format;
use vgc_engine::teams::{Pool, PoolTeam, guess_spread, read_sheet};
use vgc_engine::{Battle, Error};

/// An open team sheet as a paste site shows it: no stat points.
const SHEET: &str = "\
Garchomp @ Garchompite Z
Ability: Rough Skin
Level: 50
Jolly Nature
- Earthquake
- Rock Slide
- Dragon Claw
- Protect

Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
Careful Nature
- Fake Out
- Flare Blitz
- Parting Shot
- Snarl

Rillaboom (M) @ Assault Vest
Ability: Grassy Surge
Level: 50
Adamant Nature
- Grassy Glide
- Wood Hammer
- U-turn
- Fake Out

Milotic @ Leftovers
Ability: Competitive
Level: 50
Bold Nature
- Scald
- Icy Wind
- Recover
- Protect

Sylveon @ Throat Spray
Ability: Pixilate
Level: 50
Modest Nature
- Hyper Voice
- Moonblast
- Calm Mind
- Protect

Kommo-o @ Leppa Berry
Ability: Soundproof
Level: 50
Timid Nature
- Clanging Scales
- Aura Sphere
- Flamethrower
- Protect
";

#[test]
fn a_sheet_in_showdowns_format() {
    let team = read_sheet(SHEET.lines());
    let species: Vec<_> = team.iter().map(|r| r.set.species.as_str()).collect();
    assert_eq!(species, ["Garchomp", "Incineroar", "Rillaboom", "Milotic", "Sylveon", "Kommo-o"]);
    let garchomp = &team[0];
    assert_eq!((garchomp.set.item.as_str(), garchomp.set.ability.as_str()), ("Garchompite Z", "Rough Skin"));
    assert_eq!(garchomp.set.moves, ["Earthquake", "Rock Slide", "Dragon Claw", "Protect"]);
    assert_eq!(garchomp.set.nature, "Jolly");
    // A sheet gives the nature and never the stat points.
    assert!(team.iter().all(|r| r.nature_given && !r.spread_given));

    // Nicknames, genders and stat points, where a paste has them.
    let full =
        "Rex (Garchomp) (F) @ Life Orb\nAbility: Rough Skin\nEVs: 32 Atk / 2 SpD / 32 Spe\nJolly Nature\n- Earthquake";
    let rex = &read_sheet(full.lines())[0];
    assert_eq!((rex.set.species.as_str(), rex.set.item.as_str()), ("Garchomp", "Life Orb"));
    assert!(rex.spread_given);
    assert_eq!((rex.set.evs.atk, rex.set.evs.spd, rex.set.evs.spe, rex.set.evs.hp), (32, 2, 32, 0));
}

#[test]
fn a_sheet_laid_out_some_other_way() {
    // One fact to a line and no punctuation, as a page of cards reads once its markup is gone,
    // between lines that have nothing to do with the team.
    let lines = [
        "VR Pastes - Victory Road",
        "Beta",
        "Somebody's Regional Championships OTS",
        "Garchomp",
        "Life Orb",
        "Rough Skin",
        "Jolly",
        "Earthquake",
        "Rock Slide",
        "Protect",
        "Copy",
        "Milotic",
        "Leftovers",
        "Competitive",
        "Scald",
        "Recover",
        "Share",
    ];
    let team = read_sheet(lines.iter().copied());
    assert_eq!(team.len(), 2);
    assert_eq!((team[0].set.item.as_str(), team[0].set.ability.as_str()), ("Life Orb", "Rough Skin"));
    assert_eq!(team[0].set.moves, ["Earthquake", "Rock Slide", "Protect"]);
    assert!(team[0].nature_given && !team[1].nature_given);
    assert_eq!(team[1].set.moves, ["Scald", "Recover"]);

    // A strip of the six Pokémon first, and the whole sheet twice (a page and the copy in its
    // script): still one team.
    let twice = format!("Garchomp\nIncineroar\nRillaboom\nMilotic\nSylveon\nKommo-o\n{SHEET}\n{SHEET}");
    let team = read_sheet(twice.lines());
    assert_eq!(team.len(), 6);
    assert!(team.iter().all(|r| r.set.moves.len() == 4));
}

#[test]
fn stat_points_are_filled_in_by_rule_of_thumb() {
    let mut team = read_sheet(SHEET.lines());
    team.iter_mut().for_each(|r| guess_spread(r, false));
    let points = |i: usize| {
        let e = team[i].set.evs;
        [e.hp, e.atk, e.def, e.spa, e.spd, e.spe]
    };
    // Jolly: attack and speed. Careful: HP and Special Defense. Adamant: HP and attack.
    assert_eq!(points(0), [2, 32, 0, 0, 0, 32]);
    assert_eq!(points(1), [32, 0, 2, 0, 32, 0]);
    assert_eq!(points(2), [32, 32, 2, 0, 0, 0]);
    // Bold: HP and Defense. Modest: HP and special attack. Timid: special attack and speed.
    assert_eq!(points(3), [32, 0, 32, 0, 2, 0]);
    assert_eq!(points(4), [32, 0, 2, 32, 0, 0]);
    assert_eq!(points(5), [2, 0, 0, 32, 0, 32]);
    assert!(team.iter().all(|r| {
        r.set.evs.hp + r.set.evs.atk + r.set.evs.def + r.set.evs.spa + r.set.evs.spd + r.set.evs.spe == 66
    }));

    // With no nature on the sheet one is chosen to go with the points: a fast attacker is
    // Jolly, a slow one Adamant (Brave on a Trick Room team), a Pokémon with no attacks Bold.
    let mut bare = read_sheet(
        "Garchomp\n- Earthquake\n\nSnorlax\n- Body Slam\n\nSnorlax\n- Body Slam\n\nClefable\n- Follow Me".lines(),
    );
    assert_eq!(bare.len(), 3, "the second Snorlax is the first over again");
    guess_spread(&mut bare[0], false);
    guess_spread(&mut bare[1], false);
    assert_eq!((bare[0].set.nature.as_str(), bare[1].set.nature.as_str()), ("Jolly", "Adamant"));
    let mut slow = read_sheet("Snorlax\n- Body Slam".lines());
    guess_spread(&mut slow[0], true);
    assert_eq!(slow[0].set.nature, "Brave");
}

#[test]
fn a_pool_of_teams_can_be_saved_loaded_and_played() -> Result<(), Error> {
    let mut read = read_sheet(SHEET.lines());
    read.iter_mut().for_each(|r| guess_spread(r, false));
    let team: Vec<_> = read.into_iter().map(|r| r.set).collect();
    let format = Format::current();
    let problems = format.check_showdown_team(&team);
    assert!(problems.is_empty(), "{:?}", problems.iter().map(|p| p.to_string()).collect::<Vec<_>>());

    let entry =
        PoolTeam { player: "Somebody".into(), placing: Some(1), spreads_guessed: true, team, ..Default::default() };
    let pool =
        Pool { event: "A tournament".into(), format: format.id.clone(), teams: vec![entry], ..Default::default() };
    let back = Pool::from_json(&pool.to_json())?;
    assert_eq!(back, pool);

    // Six registered, four brought, sheets open: a mirror match.
    let sets = back.teams[0].sets()?;
    let battle = Battle::with_rosters([&sets, &sets], [&[0, 1, 2, 3], &[0, 1, 4, 5]], true, [1, 2, 3, 4])?;
    let shown = battle.shown(1);
    assert_eq!(shown.roster.len(), 6);
    assert_eq!(shown.roster[0].sheet.as_ref().unwrap().item, "garchompitez");
    Ok(())
}
