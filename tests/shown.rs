//! What each side has been shown: `Battle::shown`.
//!
//! The engine's bookkeeping of what the battle has revealed is checked
//! against a reader of Showdown's own log (`vgc_engine::observer`) at every
//! decision of the recorded battles; `the_log_reader_agrees_on_recorded_battles`
//! runs a small sample of those, and `scripts/targeted.sh` with `SHOWN=1` the
//! rest. The other tests are readable examples.

use vgc_engine::observer::{Observer, Who, holder};
use vgc_engine::position::{BattleState, PokemonState, ShownState, SideState};
use vgc_engine::replay::{Case, Outcome, ShownTally, check_shown};
use vgc_engine::shown::Bar;
use vgc_engine::{Battle, Choice, Error, PokemonSet, Request};

fn set(species: &str, moves: &[&str]) -> PokemonSet {
    PokemonSet::from_names(species, moves, "Hardy", [0; 6]).unwrap()
}

fn seed() -> [u16; 4] {
    [1, 2, 3, 4]
}

/// A move from this slot at that target (1 and 2 are the foes, 0 for a move that takes none).
fn mv(slot: u8, target: i8) -> Choice {
    Choice::Move { slot, target, mega: false }
}

/// Four Pokémon with no abilities or items; everyone's second move is Protect.
fn filler() -> Vec<PokemonSet> {
    vec![
        set("Snorlax", &["Body Slam", "Protect"]),
        set("Milotic", &["Scald", "Protect"]),
        set("Arcanine", &["Flamethrower", "Protect"]),
        set("Sylveon", &["Moonblast", "Protect"]),
    ]
}

const PROTECT: Choice = Choice::Move { slot: 1, target: 0, mega: false };

#[test]
fn nothing_is_known_at_the_start_but_who_is_on_the_field() -> Result<(), Error> {
    let b = Battle::new([&filler(), &filler()], seed())?;
    for side in 0..2 {
        let shown = b.shown(side);
        assert_eq!((shown.unseen, shown.left), (2, 4));
        assert!(shown.bench.is_empty());
        let species: Vec<_> = shown.active.iter().flatten().map(|m| (m.id, m.species.as_str())).collect();
        assert_eq!(species, [(1, "snorlax"), (2, "milotic")]);
        for mon in shown.active.iter().flatten() {
            assert!(mon.moves.is_empty());
            assert_eq!((&mon.item, &mon.ability, mon.hp, mon.bar), (&None, &None, 100, Bar::Green));
        }
    }
    Ok(())
}

#[test]
fn a_move_is_shown_by_using_it_and_stays_shown_on_the_bench() -> Result<(), Error> {
    let mut b = Battle::new([&filler(), &filler()], seed())?;
    // Snorlax uses Body Slam on the foe's Milotic; everyone else protects.
    b.choose([[mv(0, 2), PROTECT], [PROTECT, PROTECT]])?;
    let shown = b.shown(0);
    assert_eq!(shown.active[0].as_ref().unwrap().moves, ["bodyslam"]);
    assert_eq!(shown.active[1].as_ref().unwrap().moves, ["protect"]);
    // Their side: two Protects seen, and nothing about Snorlax's Body Slam.
    assert_eq!(b.shown(1).active[0].as_ref().unwrap().moves, ["protect"]);

    // Snorlax is called back for Arcanine. It is still known to have Body Slam.
    b.choose([[Choice::Switch { to: 2 }, mv(0, 1)], [PROTECT, PROTECT]])?;
    let shown = b.shown(0);
    assert_eq!(shown.active[0].as_ref().unwrap().species, "arcanine");
    assert_eq!(shown.active[0].as_ref().unwrap().id, 3, "the third Pokémon this side has shown");
    assert_eq!(shown.unseen, 1);
    let benched = &shown.bench[0];
    assert_eq!((benched.species.as_str(), benched.id), ("snorlax", 1));
    assert_eq!(benched.moves, ["bodyslam"]);
    Ok(())
}

#[test]
fn hp_is_shown_as_a_percentage_rounded_down() -> Result<(), Error> {
    let mut b = Battle::new([&filler(), &filler()], seed())?;
    b.choose([[mv(0, 1), mv(0, 1)], [PROTECT, mv(0, 1)]])?;
    for side in 0..2 {
        for pos in 0..2 {
            let m = b.mon(b.active(side, pos));
            let shown = b.shown(side).active[pos].clone().unwrap();
            let (hp, max) = (m.hp as u32, m.max_hp() as u32);
            assert_eq!(shown.hp as u32, (100 * hp / max).max(1));
            let bar = if 2 * hp > max {
                Bar::Green
            } else if 5 * hp > max {
                Bar::Yellow
            } else {
                Bar::Red
            };
            assert_eq!(shown.bar, bar);
        }
    }
    // Someone took a hit, so this is not vacuous.
    assert!(b.shown(0).active[0].as_ref().unwrap().hp < 100);
    Ok(())
}

#[test]
fn an_item_is_shown_when_it_does_something_and_when_it_goes() -> Result<(), Error> {
    let mut ours = filler();
    ours[0] = set("Garchomp", &["Dragon Claw", "Protect"]).item("Life Orb")?;
    ours[1] = set("Weavile", &["Knock Off", "Protect"]).item("Air Balloon")?;
    let mut theirs = filler();
    theirs[0] = set("Snorlax", &["Body Slam", "Protect"]).item("Leftovers")?;
    theirs[1] = set("Milotic", &["Scald", "Protect"]).item("Mystic Water")?;
    let mut b = Battle::new([&ours, &theirs], seed())?;
    // The balloon is announced as its holder comes in; nothing else is.
    let shown = b.shown(0);
    assert_eq!(shown.active[0].as_ref().unwrap().item, None);
    assert_eq!(shown.active[1].as_ref().unwrap().item.as_deref(), Some("airballoon"));
    assert_eq!(b.shown(1).active[0].as_ref().unwrap().item, None);

    // Dragon Claw into Snorlax (Life Orb recoil), Knock Off into Milotic (its item is named as it goes).
    b.choose([[mv(0, 1), mv(0, 2)], [PROTECT, mv(0, 2)]])?;
    assert_eq!(b.shown(0).active[0].as_ref().unwrap().item, None, "Snorlax protected: no hit, no recoil");
    let milotic = b.shown(1).active[1].clone().unwrap();
    assert_eq!((milotic.item.as_deref(), milotic.item_lost.as_str()), (Some(""), "mysticwater"));
    // Milotic's Scald popped the balloon.
    let weavile = b.shown(0).active[1].clone().unwrap();
    assert_eq!((weavile.item.as_deref(), weavile.item_lost.as_str()), (Some(""), "airballoon"));

    b.choose([[mv(0, 1), PROTECT], [mv(0, 1), PROTECT]])?;
    assert_eq!(b.shown(0).active[0].as_ref().unwrap().item.as_deref(), Some("lifeorb"));
    // Snorlax was hurt, so its Leftovers healed it at the end of the turn.
    assert_eq!(b.shown(1).active[0].as_ref().unwrap().item.as_deref(), Some("leftovers"));
    Ok(())
}

#[test]
fn an_ability_is_shown_when_it_is_named() -> Result<(), Error> {
    let mut ours = filler();
    ours[0] = set("Arcanine", &["Flamethrower", "Protect"]).ability("Intimidate")?;
    ours[1] = set("Garchomp", &["Earth Power", "Protect"]).ability("Rough Skin")?;
    let mut theirs = filler();
    theirs[0] = set("Snorlax", &["Body Slam", "Protect"]).ability("Thick Fat")?;
    theirs[1] = set("Rotom", &["Thunderbolt", "Protect"]).ability("Levitate")?;
    let mut b = Battle::new([&ours, &theirs], seed())?;
    let ability = |b: &Battle, side: usize, pos: usize| b.shown(side).active[pos].clone().unwrap().ability;
    assert_eq!(ability(&b, 0, 0).as_deref(), Some("intimidate"), "announced on entry");
    assert_eq!(ability(&b, 0, 1), None);
    assert_eq!((ability(&b, 1, 0), ability(&b, 1, 1)), (None, None));

    // Snorlax's Body Slam meets Rough Skin. Rotom protects itself from Earth Power, so its Levitate is not tested.
    b.choose([[PROTECT, mv(0, 2)], [mv(0, 2), PROTECT]])?;
    assert_eq!(ability(&b, 0, 1).as_deref(), Some("roughskin"));
    assert_eq!(ability(&b, 1, 1), None);
    // Earth Power again: it does not affect Rotom, and the game says why.
    b.choose([[PROTECT, mv(0, 2)], [PROTECT, mv(0, 1)]])?;
    assert_eq!(ability(&b, 1, 1).as_deref(), Some("levitate"));
    // Thick Fat halves the Flamethrower Snorlax takes without a word, and stays unknown.
    let before = b.shown(1).active[0].clone().unwrap().hp;
    b.choose([[mv(0, 1), PROTECT], [mv(0, 1), PROTECT]])?;
    assert!(b.shown(1).active[0].clone().unwrap().hp < before);
    assert_eq!(ability(&b, 1, 0), None);
    Ok(())
}

#[test]
fn a_disguise_takes_the_credit_until_it_is_broken() -> Result<(), Error> {
    // Zoroark comes in looking like the last Pokémon of its team.
    let ours = vec![
        set("Zoroark", &["Nasty Plot", "Night Daze"]).ability("Illusion")?,
        set("Milotic", &["Scald", "Protect"]),
        set("Garchomp", &["Earthquake", "Protect"]),
    ];
    let mut b = Battle::new([&ours, &filler()], seed())?;
    let shown = b.shown(0);
    assert_eq!(shown.active[0].as_ref().unwrap().species, "garchomp");
    assert_eq!(shown.unseen, 1, "two names seen, of three Pokémon");

    // What it does is put down to Garchomp...
    b.choose([[mv(0, 0), PROTECT], [PROTECT, PROTECT]])?;
    let fake = b.shown(0).active[0].clone().unwrap();
    assert_eq!((fake.species.as_str(), fake.id), ("garchomp", 1));
    assert_eq!(fake.moves, ["nastyplot"]);
    assert_eq!(fake.ability, None);

    // ...until a hit breaks the disguise. Then it was Zoroark all along, and Garchomp has shown nothing.
    b.choose([[mv(0, 0), PROTECT], [mv(0, 1), PROTECT]])?;
    let shown = b.shown(0);
    let zoroark = shown.active[0].clone().unwrap();
    assert_eq!((zoroark.species.as_str(), zoroark.id), ("zoroark", 3));
    assert_eq!(zoroark.moves, ["nastyplot"]);
    assert_eq!(zoroark.ability.as_deref(), Some("illusion"));
    let garchomp = &shown.bench[0];
    assert_eq!((garchomp.species.as_str(), garchomp.id, garchomp.hp), ("garchomp", 1, 100));
    assert!(garchomp.moves.is_empty());
    assert_eq!(shown.unseen, 0, "three names for three Pokémon");
    Ok(())
}

/// Plays the same choices in two battles and requires that both show the same at every decision.
fn shows_the_same(teams_a: [&[PokemonSet]; 2], teams_b: [&[PokemonSet]; 2]) -> Result<(), Error> {
    let mut a = Battle::new(teams_a, seed())?;
    let mut b = Battle::new(teams_b, seed())?;
    let mut decisions = 0;
    while !a.ended && decisions < 40 {
        for side in 0..2 {
            assert_eq!(a.shown(side), b.shown(side), "after {decisions} decisions");
        }
        // Always the first legal choice that is not the move slot the two battles differ in.
        let pick = |battle: &Battle, side: usize| {
            let all = battle.joint_choices(side);
            *all.iter().find(|c| !c.iter().any(|x| matches!(x, Choice::Move { slot: 3, .. }))).unwrap_or(&all[0])
        };
        let choices = [pick(&a, 0), pick(&a, 1)];
        assert_eq!(choices, [pick(&b, 0), pick(&b, 1)]);
        a.choose(choices)?;
        b.choose(choices)?;
        decisions += 1;
    }
    assert!(decisions > 5);
    Ok(())
}

#[test]
fn what_has_not_been_shown_does_not_show() -> Result<(), Error> {
    // Two battles that differ only in things that never come into play: a move
    // that is not used, an item and an ability that have nothing to act on.
    let team = |fourth: &str, item: &str, ability: &str| -> Result<Vec<PokemonSet>, Error> {
        Ok(vec![
            set("Snorlax", &["Body Slam", "Protect", "Crunch", fourth]).item(item)?.ability(ability)?,
            set("Milotic", &["Scald", "Protect", "Ice Beam", "Recover"]),
            set("Arcanine", &["Flamethrower", "Protect", "Crunch", "Extreme Speed"]),
            set("Sylveon", &["Moonblast", "Protect", "Hyper Voice", "Calm Mind"]),
        ])
    };
    let other = filler();
    let a = team("Earthquake", "Miracle Seed", "Immunity")?;
    let b = team("Fire Punch", "Mystic Water", "Gluttony")?;
    shows_the_same([&a, &other], [&b, &other])?;
    shows_the_same([&other, &a], [&other, &b])
}

#[test]
fn a_position_carries_what_has_been_shown() -> Result<(), Error> {
    let mut ours = filler();
    ours[0] = set("Arcanine", &["Flamethrower", "Protect"]).ability("Intimidate")?;
    let mut b = Battle::new([&ours, &filler()], seed())?;
    b.choose([[mv(0, 1), PROTECT], [mv(0, 1), PROTECT]])?;
    b.choose([[Choice::Switch { to: 2 }, mv(0, 1)], [mv(0, 1), mv(0, 1)]])?;
    let rebuilt = Battle::from_state(&BattleState::from_json(&b.to_state().to_json())?)?;
    for side in 0..2 {
        assert_eq!(b.shown(side), rebuilt.shown(side));
    }
    assert_eq!(rebuilt.shown(0).bench[0].ability.as_deref(), Some("intimidate"));

    // Written by hand: say what has been seen, or say nothing and only the Pokémon on the field have been.
    let mon = |species: &str, moves: &[&str]| PokemonState::new(species, moves);
    let mut garchomp = mon("Garchomp", &["Earthquake", "Dragon Claw", "Protect"]).item("Choice Scarf");
    garchomp.shown = Some(ShownState {
        moves: vec!["Dragon Claw".to_string()],
        item: Some("Choice Scarf".to_string()),
        ..Default::default()
    });
    let mut benched = mon("Rotom", &["Thunderbolt"]);
    benched.shown = Some(ShownState { id: 3, hp: Some(40), bar: Bar::Yellow, ..Default::default() });
    let side = SideState::new(vec![garchomp, mon("Milotic", &["Scald"]), benched, mon("Sylveon", &["Moonblast"])]);
    let state = BattleState {
        turn: 5,
        sides: [side, SideState::new(vec![mon("Snorlax", &["Body Slam"]), mon("Arcanine", &["Protect"])])],
        ..Default::default()
    };
    let b = Battle::from_state(&state)?;
    assert_eq!(b.request, Request::Move);
    let shown = b.shown(0);
    let garchomp = shown.active[0].clone().unwrap();
    assert_eq!((garchomp.moves, garchomp.item.as_deref()), (vec!["dragonclaw".to_string()], Some("choicescarf")));
    assert_eq!(shown.active[1].as_ref().unwrap().species, "milotic");
    assert_eq!((shown.bench.len(), shown.bench[0].species.as_str(), shown.bench[0].hp), (1, "rotom", 40));
    assert_eq!(shown.unseen, 1);
    Ok(())
}

#[test]
fn the_log_reader_agrees_on_recorded_battles() {
    // 30 battles recorded from Showdown with their logs (`gen_cases.js --log`), from
    // batches built around Illusion, Transform, items changing hands and abilities being
    // replaced. Three are here for what they caught: Big Pecks blocking Octolock's drop,
    // which Showdown does in silence; Symbiosis passing an item to a Pokémon whose berry has
    // just halved a hit, between the two lines that berry gets; and a Poison Touch that
    // poisons after Wandering Spirit has already taken its place.
    let text = include_str!("fixtures/shown_cases.jsonl");
    let mut tally = ShownTally::default();
    let mut battles = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let case: Case = serde_json::from_str(line).expect("malformed fixture line");
        match check_shown(&case, &mut tally) {
            Outcome::Pass(_) => battles += 1,
            Outcome::Unsupported(what) => panic!("case {}: {what} is not modelled", case.id),
            Outcome::Fail(report) => panic!("case {}: {}", case.id, report.join("\n")),
        }
    }
    assert!(battles >= 20 && tally.decisions > 200, "{battles} battles, {} decisions", tally.decisions);
    assert!(tally.mismatches.is_empty(), "the engine and the log disagree: {:?}", tally.mismatches);
    assert!(tally.untrue.is_empty(), "the log's reader believes something untrue: {:?}", tally.untrue);
    // The one untrue belief there is. In the last battle a Maushold holding a Choice Scarf
    // borrows Sky Attack with Copycat. On the second turn the scarf stops it, with
    // `|move|p1a: Maushold|Sky Attack||[still]`: a line like that of any move that failed.
    // Both the log's reader and the engine take Sky Attack for one of Maushold's moves.
    let expected: Vec<_> = tally.expected.keys().map(String::as_str).collect();
    assert_eq!(expected, ["choice lock: move skyattack: not one of its own"]);
}

#[test]
fn reading_a_log() {
    let mut seen = Observer::new();
    let log = [
        "|teamsize|p1|4",
        "|teamsize|p2|4",
        "|switch|p1a: Chomp|Garchomp, L50, M|100/100",
        "|switch|p1b: Mime|Mr. Rime, L50, F|100/100",
        "|switch|p2a: Zap|Rotom-Wash, L50|100/100",
        "|switch|p2b: Kitty|Incineroar, L50, M|100/100",
        "|-ability|p2b: Kitty|Intimidate|boost",
        "|-unboost|p1a: Chomp|atk|1",
        "|-unboost|p1b: Mime|atk|1",
        "|turn|1",
        "|move|p2b: Kitty|Fake Out|p1a: Chomp",
        "|-damage|p1a: Chomp|93/100",
        "|-damage|p2b: Kitty|88/100|[from] item: Rocky Helmet|[of] p1a: Chomp",
        "|cant|p1a: Chomp|flinch",
        "|move|p1b: Mime|Trick|p2a: Zap",
        "|-activate|p1b: Mime|move: Trick|[of] p2a: Zap",
        "|-item|p2a: Zap|Choice Scarf|[from] move: Trick",
        "|-item|p1b: Mime|Sitrus Berry|[from] move: Trick",
        "|move|p2a: Zap|Hydro Pump|p1a: Chomp|[miss]",
        "|-miss|p2a: Zap|p1a: Chomp",
        "|turn|2",
    ];
    seen.lines(&log).unwrap();
    let (ours, theirs) = (seen.shown(0), seen.shown(1));
    let chomp = ours.active[0].clone().unwrap();
    assert_eq!((chomp.species.as_str(), chomp.hp, chomp.item.as_deref()), ("garchomp", 93, Some("rockyhelmet")));
    assert_eq!(ours.active[1].as_ref().unwrap().item.as_deref(), Some("sitrusberry"));
    assert_eq!(ours.active[1].as_ref().unwrap().moves, ["trick"]);
    let zap = theirs.active[0].clone().unwrap();
    assert_eq!((zap.species.as_str(), zap.item.as_deref()), ("rotomwash", Some("choicescarf")));
    assert_eq!(zap.moves, ["hydropump"]);
    let kitty = theirs.active[1].clone().unwrap();
    assert_eq!((kitty.ability.as_deref(), kitty.hp, kitty.item), (Some("intimidate"), 88, None));
    assert_eq!((theirs.unseen, theirs.left), (2, 4));

    // Whose helmet, whose bell: the same shape of line means different things.
    assert_eq!(holder("-damage", true, true, "rockyhelmet"), Who::Of);
    assert_eq!(holder("-heal", true, true, "shellbell"), Who::Subject);
    assert_eq!(holder("-heal", true, false, "waterabsorb"), Who::Subject);
    assert_eq!(holder("-heal", true, false, "hospitality"), Who::Of);
}
