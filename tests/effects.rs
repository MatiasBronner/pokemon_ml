//! A few abilities and items checked directly, as readable examples of the
//! API. The real verification is the replay of recorded Showdown battles in
//! `parity.rs` and `scripts/fuzz.sh`; these only pin down outcomes that do not
//! depend on the random number generator.

use vgc_engine::data::ATK;
use vgc_engine::{Battle, Choice, Error, PokemonSet};

fn set(species: &str, moves: &[&str]) -> PokemonSet {
    PokemonSet::from_names(species, moves, "Hardy", [0; 6]).unwrap()
}

fn seed() -> [u16; 4] {
    [1, 2, 3, 4]
}

/// Four generic Pokémon with no abilities or items.
fn filler() -> Vec<PokemonSet> {
    vec![
        set("Snorlax", &["Body Slam", "Protect"]),
        set("Milotic", &["Scald", "Protect"]),
        set("Arcanine", &["Flamethrower", "Protect"]),
        set("Sylveon", &["Moonblast", "Protect"]),
    ]
}

#[test]
fn intimidate_lowers_both_foes_on_entry() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Arcanine", &["Flamethrower", "Protect"]).ability("Intimidate")?;
    let p2 = filler();
    let b = Battle::new([&p1, &p2], seed())?;
    for pos in 0..2 {
        assert_eq!(b.mon(b.active(1, pos)).boosts[ATK], -1);
        assert_eq!(b.mon(b.active(0, pos)).boosts[ATK], 0);
    }
    Ok(())
}

#[test]
fn clear_body_and_defiant_answer_intimidate() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Arcanine", &["Flamethrower", "Protect"]).ability("Intimidate")?;
    let mut p2 = filler();
    p2[0] = set("Snorlax", &["Body Slam", "Protect"]).ability("Clear Body")?;
    p2[1] = set("Milotic", &["Scald", "Protect"]).ability("Defiant")?;
    let b = Battle::new([&p1, &p2], seed())?;
    assert_eq!(b.mon(b.active(1, 0)).boosts[ATK], 0, "Clear Body blocks the drop");
    assert_eq!(b.mon(b.active(1, 1)).boosts[ATK], 1, "Defiant: -1, then +2");
    Ok(())
}

#[test]
fn choice_scarf_locks_the_holder_into_its_first_move() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Garchomp", &["Dragon Claw", "Rock Slide", "Protect"]).item("Choice Scarf")?;
    let p2 = filler();
    let mut b = Battle::new([&p1, &p2], seed())?;
    let before = b.legal_choices(0, 0);
    assert!(before.iter().any(|c| matches!(c, Choice::Move { slot: 1, .. })));

    // Use Rock Slide (slot 1); everyone else protects.
    let protect = Choice::Move { slot: 1, target: 0 };
    b.choose([[Choice::Move { slot: 1, target: 0 }, protect], [protect, protect]])?;
    if b.ended || b.mon(b.active(0, 0)).fainted {
        return Ok(());
    }
    let after = b.legal_choices(0, 0);
    let moves: Vec<u8> = after
        .iter()
        .filter_map(|c| match c {
            Choice::Move { slot, .. } => Some(*slot),
            _ => None,
        })
        .collect();
    assert!(!moves.is_empty() && moves.iter().all(|&s| s == 1), "only Rock Slide is offered, got {after:?}");
    assert!(after.iter().any(|c| matches!(c, Choice::Switch { .. })), "switching out is still allowed");
    Ok(())
}

#[test]
fn levitate_and_air_balloon_avoid_ground_moves() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Garchomp", &["Earthquake", "Protect"]);
    p1[1] = set("Rotom-Wash", &["Calm Mind", "Protect"]).ability("Levitate")?;
    let mut p2 = filler();
    p2[0] = set("Snorlax", &["Calm Mind", "Protect"]).item("Air Balloon")?;
    p2[1] = set("Milotic", &["Calm Mind", "Protect"]).ability("Levitate")?;
    let mut b = Battle::new([&p1, &p2], seed())?;
    // Earthquake hits everyone adjacent, its own partner included. The other
    // three only raise their stats, so any lost HP would be Earthquake's doing.
    let wait = Choice::Move { slot: 0, target: 0 };
    b.choose([[Choice::Move { slot: 0, target: 0 }, wait], [wait, wait]])?;
    for (side, pos) in [(0usize, 1usize), (1, 0), (1, 1)] {
        let m = b.mon(b.active(side, pos));
        assert_eq!(m.hp, m.max_hp(), "side {side} slot {pos} should not have been hit");
    }
    Ok(())
}

#[test]
fn effects_that_are_not_modelled_are_refused() {
    let p1 = filler();
    let mut p2 = filler();
    p2[0] = set("Pelipper", &["Scald", "Protect"]).ability("Drizzle").unwrap();
    match Battle::new([&p1, &p2], seed()) {
        Err(Error::Unsupported(what)) => assert!(what.contains("Drizzle")),
        other => panic!("expected an unsupported error, got {:?}", other.map(|_| ())),
    }
    let mut p2 = filler();
    p2[0] = set("Venusaur", &["Sludge Bomb", "Protect"]).item("Venusaurite").unwrap();
    assert!(matches!(Battle::new([&p1, &p2], seed()), Err(Error::Unsupported(_))));
}
