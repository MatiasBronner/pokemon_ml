//! The perish trap: Perish Song, with the other side kept from switching for
//! the three turns it takes.
//!
//! The four battles in `fixtures/perish_trap.jsonl` were played in Pokémon
//! Showdown from `scripts/perish_trap.json`, a script of teams and choices
//! (`gen_cases.js --script`), with every legal choice at every decision
//! confirmed by Showdown's own validation (`--check-legal`). The first test
//! holds the engine to them decision by decision. The others walk through
//! what happens in each, as examples.

use vgc_engine::replay::{Case, Outcome, Rebuild, ShownTally, check_case, check_case_with, check_shown};
use vgc_engine::{Battle, Choice, PokemonSet, Request};

fn scripted() -> Vec<Case> {
    let text = include_str!("fixtures/perish_trap.jsonl");
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).expect("malformed fixture")).collect()
}

/// The battle of `case` after the first `n` of its scripted choices.
fn after(case: &Case, n: usize) -> Battle {
    let sets = |side: usize| -> Vec<PokemonSet> {
        case.rosters.as_ref().unwrap()[side].iter().map(|set| set.to_set().unwrap()).collect()
    };
    let (ours, theirs) = (sets(0), sets(1));
    let picks = case.picks.as_ref().unwrap();
    let mut battle = Battle::with_rosters([&ours, &theirs], [&picks[0], &picks[1]], true, case.seed).unwrap();
    for step in &case.steps[..n] {
        let side = |s: usize| Choice::parse_side(&step.choices[s]).unwrap();
        battle.choose([side(0), side(1)]).unwrap();
    }
    battle
}

fn can_switch(battle: &Battle, side: usize, pos: usize) -> bool {
    battle.legal_choices(side, pos).iter().any(|c| matches!(c, Choice::Switch { .. }))
}

/// Who stands in a position, and the turns its perish count has left (`None`: it has none).
fn standing(battle: &Battle, side: usize, pos: usize) -> (String, Option<u8>) {
    let state = battle.to_state();
    let mon = &state.sides[side].pokemon[pos];
    let count = mon.volatiles.iter().find(|v| v.id == "perishsong").map(|v| v.turns.unwrap());
    (mon.species.clone(), count)
}

fn fainted(battle: &Battle, side: usize) -> Vec<String> {
    let state = battle.to_state();
    state.sides[side].pokemon.iter().filter(|m| m.fainted).map(|m| m.species.clone()).collect()
}

#[test]
fn showdown_plays_them_the_same_way() {
    let cases = scripted();
    assert_eq!(cases.len(), 4);
    let mut tally = ShownTally::default();
    for case in &cases {
        // State, random numbers and legal choices after every decision.
        assert!(matches!(check_case(case), Outcome::Pass(_)), "battle {} diverged from Showdown", case.id);
        // The same from a position rebuilt at every decision.
        assert!(matches!(check_case_with(case, Rebuild::ByHand), Outcome::Pass(_)), "battle {}, rebuilt", case.id);
        // And what each side has been shown, against Showdown's log.
        assert!(matches!(check_shown(case, &mut tally), Outcome::Pass(_)), "battle {}, what was shown", case.id);
    }
    assert!(tally.mismatches.is_empty() && tally.untrue.is_empty(), "{tally:?}");
}

/// Politoed sings beside a Gengar that Mega Evolves on the same turn. Mega
/// Gengar's Shadow Tag keeps Garchomp and Milotic in for three turns, Politoed
/// leaves on the third and Gengar on the fourth, after the other side has
/// had to choose without the option of switching.
#[test]
fn the_classic_trap() {
    let case = &scripted()[0];

    // Turn 1. Everyone on the field heard the song, Protect or not, and has three turns.
    let battle = after(case, 1);
    assert_eq!(standing(&battle, 0, 0), ("gengarmega".to_string(), Some(3)));
    assert_eq!(standing(&battle, 0, 1), ("politoed".to_string(), Some(3)));
    assert_eq!(standing(&battle, 1, 0), ("garchomp".to_string(), Some(3)));
    assert_eq!(standing(&battle, 1, 1), ("milotic".to_string(), Some(3)));
    // The singers may leave. The others may not, with two Pokémon waiting on the bench.
    assert!(can_switch(&battle, 0, 0) && can_switch(&battle, 0, 1));
    assert!(!can_switch(&battle, 1, 0) && !can_switch(&battle, 1, 1));

    // Turn 2: still held, two turns left.
    let battle = after(case, 2);
    assert_eq!(standing(&battle, 1, 0).1, Some(2));
    assert!(!can_switch(&battle, 1, 0) && !can_switch(&battle, 1, 1));

    // Turn 3: Politoed has gone, and its count with it. Snorlax came in clean.
    let battle = after(case, 3);
    assert_eq!(standing(&battle, 0, 1), ("snorlax".to_string(), None));
    assert_eq!(standing(&battle, 1, 0).1, Some(1));
    // The last choice the trapped side gets is made with Gengar still there.
    assert!(!can_switch(&battle, 1, 0) && !can_switch(&battle, 1, 1));

    // Turn 4: Gengar switches out before anyone moves. Too late for the other side:
    // at the end of the turn its two Pokémon faint, and nobody else does.
    let battle = after(case, 4);
    assert_eq!(battle.request, Request::Switch);
    assert_eq!(fainted(&battle, 1), ["garchomp", "milotic"]);
    assert!(fainted(&battle, 0).is_empty());
    assert_eq!(standing(&battle, 0, 0), ("arcanine".to_string(), None));
    // Gengar and Politoed sit on the bench, unharmed and with no count.
    let state = battle.to_state();
    let bench: Vec<_> = state.sides[0].pokemon[2..].iter().map(|m| (m.species.as_str(), m.volatiles.len())).collect();
    assert_eq!(bench, [("politoed", 0), ("gengarmega", 0)]);
}

/// The same trap against a side that has answers: a Ghost and a Shed Shell
/// walk out of Shadow Tag, U-turn carries a trapped Pokémon out, and
/// Soundproof does not hear the song. The singers, this time, stay to the end.
#[test]
fn the_ways_out() {
    let case = &scripted()[1];

    // Dragapult (a Ghost) and Milotic (holding a Shed Shell) have the count but can leave.
    let battle = after(case, 1);
    assert_eq!(standing(&battle, 1, 0), ("dragapult".to_string(), Some(3)));
    assert_eq!(standing(&battle, 1, 1), ("milotic".to_string(), Some(3)));
    assert!(can_switch(&battle, 1, 0) && can_switch(&battle, 1, 1));

    // They do. Scizor and Kommo-o have no count, having not been there, but are now held.
    let battle = after(case, 2);
    assert_eq!(standing(&battle, 1, 0), ("scizor".to_string(), None));
    assert_eq!(standing(&battle, 1, 1), ("kommoo".to_string(), None));
    assert!(!can_switch(&battle, 1, 0) && !can_switch(&battle, 1, 1));

    // Scizor's U-turn takes it out all the same: its side is asked for a replacement mid-turn.
    let battle = after(case, 3);
    assert_eq!(battle.request, Request::Switch);
    assert!(can_switch(&battle, 1, 0));
    // Dragapult comes back for it. Being a Ghost it could leave again; Kommo-o could not.
    let battle = after(case, 4);
    assert_eq!(standing(&battle, 1, 0), ("dragapult".to_string(), None));
    assert!(can_switch(&battle, 1, 0) && !can_switch(&battle, 1, 1));

    // A second song on turn 4. Kommo-o's Soundproof keeps it out; Dragapult, back in, gets a
    // new count. Gengar and Politoed are still on the field with the first one, which runs out.
    let battle = after(case, 5);
    assert_eq!(fainted(&battle, 0), ["gengarmega", "politoed"]);
    assert_eq!(standing(&battle, 1, 0), ("dragapult".to_string(), Some(3)));
    assert_eq!(standing(&battle, 1, 1), ("kommoo".to_string(), None));
}

/// Mean Look holds the one Pokémon it was used on, for as long as its user stays.
#[test]
fn mean_look_lets_go_when_its_user_leaves() {
    let case = &scripted()[2];

    // Umbreon used Mean Look on Garchomp; Azumarill sang. Garchomp is held, Milotic is not.
    let battle = after(case, 1);
    assert_eq!(standing(&battle, 1, 0), ("garchomp".to_string(), Some(3)));
    assert!(!can_switch(&battle, 1, 0));
    assert!(can_switch(&battle, 1, 1));

    // Still held on turn 3, when Umbreon switches out...
    let battle = after(case, 2);
    assert!(!can_switch(&battle, 1, 0));
    // ...and free the turn after, with one turn left on its count.
    let battle = after(case, 3);
    assert_eq!(standing(&battle, 1, 0), ("garchomp".to_string(), Some(1)));
    assert!(can_switch(&battle, 1, 0));

    // It leaves, as does Azumarill, and the song ends with nobody having fainted.
    let battle = after(case, 5);
    assert!(fainted(&battle, 0).is_empty() && fainted(&battle, 1).is_empty());
    for (side, pos) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        assert_eq!(standing(&battle, side, pos).1, None);
    }
}

/// Two against two and nobody can leave. All four faint at the end of the
/// fourth turn. Showdown takes them in order of speed and gives the battle to
/// the side whose Pokémon went last: here the slower pair, which were not
/// the singers.
#[test]
fn everyone_perishes_at_once() {
    let case = &scripted()[3];
    let battle = after(case, 3);
    for (side, pos) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        assert_eq!(standing(&battle, side, pos).1, Some(1));
        assert!(!can_switch(&battle, side, pos), "there is nobody to switch to");
    }
    let battle = after(case, 4);
    assert!(battle.ended);
    assert_eq!(battle.winner, Some(1));
}
