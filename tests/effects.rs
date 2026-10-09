//! A few abilities and items checked directly, as readable examples of the
//! API. The real verification is the replay of recorded Showdown battles in
//! `parity.rs` and `scripts/fuzz.sh`; these only pin down outcomes that do not
//! depend on the random number generator.

use vgc_engine::data::{ABILITIES, ATK, Pseudo, SPA, SPECIES, SideCond, Type, Weather};
use vgc_engine::{Battle, Choice, Error, PokemonSet, Request, VolKind};

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
    let protect = Choice::mv(1, 0);
    b.choose([[Choice::mv(1, 0), protect], [protect, protect]])?;
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
    let wait = Choice::mv(0, 0);
    b.choose([[Choice::mv(0, 0), wait], [wait, wait]])?;
    for (side, pos) in [(0usize, 1usize), (1, 0), (1, 1)] {
        let m = b.mon(b.active(side, pos));
        assert_eq!(m.hp, m.max_hp(), "side {side} slot {pos} should not have been hit");
    }
    Ok(())
}

/// Everything a Champions Pokémon can have is modelled; what is still refused
/// is a Pokémon the game does not have at all.
#[test]
fn a_forme_of_a_species_that_is_not_in_the_game_is_refused() {
    let p1 = filler();
    let mut p2 = filler();
    p2[0] = set("Eiscue-Noice", &["Ice Beam", "Protect"]);
    match Battle::new([&p1, &p2], seed()) {
        Err(Error::Unsupported(what)) => assert!(what.contains("Eiscue"), "{what}"),
        other => panic!("expected an unsupported error, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn weather_tailwind_and_trick_room_are_set_and_count_down() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Pelipper", &["Protect", "Scald"]).ability("Drizzle")?;
    p1[1] = set("Whimsicott", &["Tailwind", "Protect"]);
    let mut p2 = filler();
    p2[0] = set("Hatterene", &["Trick Room", "Protect"]);
    let mut b = Battle::new([&p1, &p2], seed())?;
    assert_eq!(b.field.weather.kind, Weather::Raindance, "Drizzle sets rain on entry");
    assert_eq!(b.field.weather.duration, 5);
    let whimsicott = b.active(0, 1);
    let speed = b.mon(whimsicott).speed;

    let protect = Choice::mv(1, 0);
    b.choose([[Choice::mv(0, 0), Choice::mv(0, 0)], [Choice::mv(0, 0), protect]])?;
    // Each has used up one of its turns by the end of the turn it was set in.
    assert_eq!(b.field.weather.duration, 4);
    assert_eq!(b.sides[0].conds.get(SideCond::Tailwind).map(|c| c.duration), Some(3));
    assert!(!b.sides[1].conds.has(SideCond::Tailwind));
    assert_eq!(b.field.pseudo.get(Pseudo::Trickroom).map(|c| c.duration), Some(4));

    // Rain and Trick Room end after their fifth turn, Tailwind after its fourth.
    // Nobody attacks, so nothing else can happen in the meantime.
    for turn in 2..=5 {
        b.choose([[Choice::mv(0, 0), protect], [protect, protect]])?;
        // `speed` is the value the turn order was decided with: doubled by
        // Tailwind in turns 2 to 4, and negated while Trick Room is up.
        let expected = if turn <= 4 { -speed * 2 } else { -speed };
        assert_eq!(b.mon(whimsicott).speed, expected, "turn {turn}");
    }
    assert_eq!(b.field.weather.kind, Weather::None);
    assert!(!b.sides[0].conds.has(SideCond::Tailwind));
    assert!(!b.field.pseudo.has(Pseudo::Trickroom));
    Ok(())
}

#[test]
fn mega_evolution_changes_stats_type_and_ability_once_per_side() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Charizard", &["Protect", "Flamethrower"]).ability("Blaze")?.item("Charizardite X")?;
    p1[1] = set("Venusaur", &["Protect", "Sludge Bomb"]).item("Venusaurite")?;
    let p2 = filler();
    let mut b = Battle::new([&p1, &p2], seed())?;
    let charizard = b.active(0, 0);
    let before = b.mon(charizard).stats;
    assert_eq!(SPECIES[b.mon(charizard).species as usize].name, "Charizard");

    // Both could Mega Evolve, but not in the same turn.
    let mega = Choice::Move { slot: 0, target: 0, mega: true };
    assert!(b.legal_choices(0, 0).contains(&mega));
    assert!(b.legal_choices(0, 1).contains(&mega));
    assert!(!b.joint_ok(0, &[mega, mega]));

    let protect = Choice::mv(1, 0);
    b.choose([[mega, Choice::mv(0, 0)], [protect, protect]])?;
    let m = b.mon(charizard);
    assert_eq!(SPECIES[m.species as usize].name, "Charizard-Mega-X");
    assert_eq!(m.types, [Type::Fire, Type::Dragon]);
    assert_eq!(ABILITIES[m.ability as usize].name, "Tough Claws");
    assert_eq!(m.stats[0], before[0], "max HP does not change");
    assert!(m.stats[ATK + 1] > before[ATK + 1]);
    // One Mega Evolution per side: Venusaur has lost its chance.
    assert!(!b.legal_choices(0, 1).contains(&mega));
    Ok(())
}

#[test]
fn fake_out_flinches_once_and_is_then_unavailable() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Incineroar", &["Fake Out", "Protect"]);
    let mut p2 = filler();
    p2[0] = set("Snorlax", &["Calm Mind", "Protect"]);
    let mut b = Battle::new([&p1, &p2], seed())?;
    let fake_out = |c: &Choice| matches!(c, Choice::Move { slot: 0, .. });
    assert!(b.legal_choices(0, 0).iter().any(fake_out));

    let protect = Choice::mv(1, 0);
    b.choose([[Choice::mv(0, 1), protect], [Choice::mv(0, 0), protect]])?;
    let snorlax = b.active(1, 0);
    assert!(b.mon(snorlax).hp < b.mon(snorlax).max_hp());
    assert_eq!(b.mon(snorlax).boosts[SPA], 0, "Snorlax flinched instead of using Calm Mind");
    assert!(!b.legal_choices(0, 0).iter().any(fake_out), "Fake Out only works on the first turn out");
    Ok(())
}

#[test]
fn follow_me_redirects_single_target_moves() -> Result<(), Error> {
    let p1 = filler();
    let mut p2 = filler();
    p2[0] = set("Snorlax", &["Calm Mind", "Protect"]);
    p2[1] = set("Clefable", &["Follow Me", "Protect"]);
    let mut b = Battle::new([&p1, &p2], seed())?;
    // Snorlax aims Body Slam at the foe on the left; Clefable, on the right, calls it over.
    let protect = Choice::mv(1, 0);
    b.choose([[Choice::mv(0, 1), protect], [Choice::mv(0, 0), Choice::mv(0, 0)]])?;
    let (left, clefable) = (b.active(1, 0), b.active(1, 1));
    assert_eq!(b.mon(left).hp, b.mon(left).max_hp());
    assert!(b.mon(clefable).hp < b.mon(clefable).max_hp());
    Ok(())
}

#[test]
fn a_substitute_costs_a_quarter_of_max_hp_and_blocks_status_moves() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Snorlax", &["Substitute", "Protect"]);
    let mut p2 = filler();
    p2[0] = set("Clefable", &["Charm", "Protect"]);
    let mut b = Battle::new([&p1, &p2], seed())?;
    let protect = Choice::mv(1, 0);
    b.choose([[Choice::mv(0, 0), protect], [protect, protect]])?;
    let snorlax = b.active(0, 0);
    let max = b.mon(snorlax).max_hp();
    assert_eq!(b.mon(snorlax).hp, max - max / 4);
    assert!(b.vols(snorlax).has(VolKind::Substitute));

    // Charm never misses, but it cannot get past the substitute.
    b.choose([[Choice::mv(0, 0), protect], [Choice::mv(0, 1), protect]])?;
    assert_eq!(b.mon(snorlax).boosts[ATK], 0);
    assert!(b.vols(snorlax).has(VolKind::Substitute));
    Ok(())
}

#[test]
fn taunt_takes_status_moves_off_the_menu() -> Result<(), Error> {
    let p1 = filler();
    let mut p2 = filler();
    p2[0] = set("Arcanine", &["Taunt", "Protect"]);
    let mut b = Battle::new([&p1, &p2], seed())?;
    let protect = Choice::mv(1, 0);
    b.choose([[Choice::mv(0, 1), protect], [Choice::mv(0, 1), protect]])?;
    if b.ended || b.mon(b.active(0, 0)).fainted {
        return Ok(());
    }
    // Snorlax knows Body Slam (slot 0) and Protect (slot 1).
    let slots: Vec<u8> = b
        .legal_choices(0, 0)
        .iter()
        .filter_map(|c| match c {
            Choice::Move { slot, .. } => Some(*slot),
            _ => None,
        })
        .collect();
    assert!(slots.contains(&0) && !slots.contains(&1), "only the attack is offered, got {slots:?}");
    Ok(())
}

/// A Pokémon locked into a status move by its Choice Scarf and then taunted can
/// only Struggle. Electrify turns its target's move Electric, but not Struggle.
#[test]
fn electrify_does_not_change_the_type_of_struggle() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Snorlax", &["Swords Dance", "Body Slam"]).item("Choice Scarf")?;
    let mut p2 = filler();
    // Two Ground types: an Electric Struggle would do nothing to either.
    // Hippowdon is slower than a scarfed Snorlax, so its Taunt lands after the Swords Dance.
    p2[0] = set("Garchomp", &["Protect", "Electrify"]);
    p2[1] = set("Hippowdon", &["Taunt", "Slack Off"]);
    let mut b = Battle::new([&p1, &p2], seed())?;
    let protect = Choice::mv(1, 0);
    b.choose([[Choice::mv(0, 0), protect], [Choice::mv(0, 0), Choice::mv(0, 1)]])?;
    let snorlax = b.active(0, 0);
    assert_eq!(b.legal_choices(0, 0).iter().filter(|c| matches!(c, Choice::Move { .. })).count(), 1);
    let before: u32 = (0..2).map(|pos| b.mon(b.active(1, pos)).hp as u32).sum();

    b.choose([[Choice::mv(0, 0), protect], [Choice::mv(1, 1), Choice::mv(1, 0)]])?;
    let after: u32 = (0..2).map(|pos| b.mon(b.active(1, pos)).hp as u32).sum();
    assert!(after < before, "Struggle hit a Ground type, so it was not Electric");
    assert!(b.mon(snorlax).hp < b.mon(snorlax).max_hp(), "and Snorlax took Struggle's recoil");
    Ok(())
}

/// U-turn stops the turn to ask its user who comes in; the rest of the turn follows.
#[test]
fn u_turn_asks_for_a_replacement_in_the_middle_of_the_turn() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Scizor", &["U-turn", "Protect"]);
    let p2 = filler();
    let mut b = Battle::new([&p1, &p2], seed())?;
    let scizor = b.active(0, 0);
    let protect = Choice::mv(1, 0);
    // (The foe it hits attacks Scizor's protected partner, so nothing else happens to Scizor.)
    b.choose([[Choice::mv(0, 1), protect], [Choice::mv(0, 2), protect]])?;
    assert_eq!(b.request, Request::Switch);
    assert_eq!(b.turn, 1, "the turn is not over yet");
    // The user's slot may bring in either benched Pokémon; its partner just passes.
    assert_eq!(b.legal_choices(0, 0), vec![Choice::Switch { to: 2 }, Choice::Switch { to: 3 }]);
    assert_eq!(b.legal_choices(0, 1), vec![Choice::Pass]);

    b.choose([[Choice::Switch { to: 2 }, Choice::Pass], [Choice::Pass, Choice::Pass]])?;
    assert_ne!(b.active(0, 0), scizor);
    assert!(!b.mon(scizor).is_active);
    assert_eq!((b.request, b.turn), (Request::Move, 2));
    Ok(())
}

/// Roar drags its target out for a teammate picked at random, with no request.
#[test]
fn roar_drags_in_a_random_teammate() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Arcanine", &["Roar", "Protect"]);
    let p2 = filler();
    let mut b = Battle::new([&p1, &p2], seed())?;
    let before = b.active(1, 0);
    let protect = Choice::mv(1, 0);
    b.choose([[Choice::mv(0, 1), protect], [Choice::mv(0, 1), protect]])?;
    assert_ne!(b.active(1, 0), before, "the target was replaced");
    assert!(!b.mon(before).is_active);
    assert_eq!(b.request, Request::Move);
    Ok(())
}

/// Aegislash attacks in Blade forme and goes back behind its shield for King's Shield.
#[test]
fn stance_change_follows_the_move_being_used() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Aegislash", &["Shadow Ball", "King's Shield"]).ability("Stance Change")?;
    let p2 = filler();
    let mut b = Battle::new([&p1, &p2], seed())?;
    let aegislash = b.active(0, 0);
    let shield = SPECIES[b.mon(aegislash).species as usize];
    let protect = Choice::mv(1, 0);
    b.choose([[Choice::mv(0, 1), protect], [protect, protect]])?;
    let blade = SPECIES[b.mon(aegislash).species as usize];
    assert_eq!(blade.name, "Aegislash-Blade");
    assert!(blade.base[ATK + 1] > shield.base[ATK + 1]);

    b.choose([[Choice::mv(1, 0), protect], [protect, protect]])?;
    assert_eq!(SPECIES[b.mon(aegislash).species as usize].name, "Aegislash");
    Ok(())
}

/// Ditto's Imposter copies the foe across from it as it enters: species,
/// stats other than HP, ability and moves, each move with at most 5 PP.
#[test]
fn imposter_transforms_into_the_foe_across_the_field() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Ditto", &["Transform"]).ability("Imposter")?;
    let mut p2 = filler();
    p2[1] = set("Garchomp", &["Earthquake", "Dragon Claw", "Protect"]).ability("Rough Skin")?;
    let b = Battle::new([&p1, &p2], seed())?;
    // Position 0 on one side faces position 1 on the other.
    let (ditto, garchomp) = (b.mon(b.active(0, 0)), b.mon(b.active(1, 1)));
    assert!(ditto.transformed);
    assert_eq!(ditto.species, garchomp.species);
    assert_eq!(ditto.ability, garchomp.ability);
    assert_eq!(ditto.stats[1..], garchomp.stats[1..]);
    assert_ne!(ditto.max_hp(), garchomp.max_hp(), "HP is not copied");
    assert_eq!(ditto.n_moves, 3);
    for k in 0..3 {
        assert_eq!(ditto.moves[k].id, garchomp.moves[k].id);
        assert_eq!(ditto.moves[k].pp, 5);
    }
    Ok(())
}

/// Mimikyu's Disguise takes the first hit: no damage from the move, then an eighth of its HP.
#[test]
fn disguise_takes_one_hit_and_then_breaks() -> Result<(), Error> {
    let mut p1 = filler();
    p1[1] = set("Milotic", &["Aqua Jet", "Protect"]);
    let mut p2 = filler();
    p2[0] = set("Mimikyu", &["Shadow Sneak", "Protect"]).ability("Disguise")?;
    let mut b = Battle::new([&p1, &p2], seed())?;
    let mimikyu = b.active(1, 0);
    let max = b.mon(mimikyu).max_hp();
    let protect = Choice::mv(1, 0);
    // Milotic's Aqua Jet is what the disguise soaks up. (Mimikyu's own Shadow Sneak
    // goes into Snorlax, a Normal type, and does nothing.)
    b.choose([[protect, Choice::mv(0, 1)], [Choice::mv(0, 1), protect]])?;
    assert_eq!(b.mon(mimikyu).hp, max - max / 8);
    assert_eq!(SPECIES[b.mon(mimikyu).species as usize].name, "Mimikyu-Busted");
    Ok(())
}

/// Future Sight does nothing on the turn it is used and lands at the end of the turn after next.
#[test]
fn future_sight_lands_two_turns_later() -> Result<(), Error> {
    let mut p1 = filler();
    p1[0] = set("Gardevoir", &["Future Sight", "Protect"]);
    let p2 = filler();
    let mut b = Battle::new([&p1, &p2], seed())?;
    let target = b.active(1, 0);
    let max = b.mon(target).max_hp();
    let protect = Choice::mv(1, 0);
    b.choose([[Choice::mv(0, 1), protect], [protect, protect]])?;
    assert_eq!(b.mon(target).hp, max, "nothing yet (and Protect would not have stopped it)");
    b.choose([[protect, protect], [Choice::mv(1, 0), protect]])?;
    assert_eq!(b.mon(target).hp, max);
    // Protecting on the turn it lands is no help either: the hit comes after the turn's moves.
    b.choose([[Choice::mv(1, 0), protect], [Choice::mv(1, 0), protect]])?;
    assert!(b.mon(target).hp < max);
    Ok(())
}

/// Search copies a `Battle` thousands of times per decision; keep it a small flat value.
#[test]
fn a_battle_is_a_small_copyable_value() {
    fn assert_copy<T: Copy>() {}
    assert_copy::<Battle>();
    let size = std::mem::size_of::<Battle>();
    println!("size of Battle: {size} bytes");
    assert!(size < 16 * 1024, "Battle has grown to {size} bytes");
}
