//! Starting a battle from a position, as a bot searching from a live game would.
//!
//!     cargo run --release --example position
//!
//! The position is written by hand here; `Battle::to_state` gives the same
//! kind of description of a battle in progress, and `BattleState::to_json` /
//! `from_json` carry one across a process boundary.

use vgc_engine::position::{BattleState, CondState, PokemonState, SideState};
use vgc_engine::{Battle, Choice, Error};

fn main() -> Result<(), Error> {
    let mon = |species: &str, moves: &[&str]| PokemonState::new(species, moves);

    // Turn 6. We are behind: Incineroar is hurt and has been Intimidated back,
    // our Tailwind has one turn left, and it is raining for two more.
    let mut incineroar = mon("Incineroar", &["Fake Out", "Flare Blitz", "Parting Shot", "Protect"])
        .ability("Intimidate")
        .nature("Adamant", [32, 32, 0, 0, 2, 0]);
    incineroar.hp_percent = Some(38.0);
    incineroar.boosts[0] = -1;
    incineroar.last_item = "sitrusberry".into(); // eaten earlier; it holds nothing now
    let garchomp = mon("Garchomp", &["Earthquake", "Rock Slide", "Dragon Claw", "Protect"])
        .ability("Rough Skin")
        .item("Choice Scarf")
        .nature("Jolly", [0, 32, 0, 0, 2, 32])
        .volatile(CondState::choice_lock("Rock Slide"));
    let mut ours = SideState::new(vec![
        incineroar,
        garchomp,
        mon("Rillaboom", &["Grassy Glide", "Wood Hammer", "Fake Out", "Protect"]).ability("Grassy Surge"),
        mon("Milotic", &["Scald", "Ice Beam", "Recover", "Protect"]).status("par"),
    ]);
    ours.conditions.push(CondState::new("tailwind").turns(1));

    let mut pelipper = mon("Pelipper", &["Hurricane", "Weather Ball", "Tailwind", "Protect"]).ability("Drizzle");
    pelipper.hp_percent = Some(71.0);
    let theirs = SideState::new(vec![
        pelipper,
        mon("Archaludon", &["Electro Shot", "Draco Meteor", "Flash Cannon", "Protect"]).ability("Stamina"),
        mon("Kingambit", &["Sucker Punch", "Kowtow Cleave", "Iron Head", "Protect"]).ability("Defiant"),
    ]);

    let state = BattleState {
        turn: 6,
        sides: [ours, theirs],
        weather: Some(CondState::new("raindance").turns(2)),
        ..Default::default()
    };
    let battle = Battle::from_state(&state)?;

    println!("turn {}, {:?} request", battle.turn, battle.request);
    for side in 0..2 {
        let legal = battle.joint_choices(side);
        println!("side {}: {} ways to choose, for instance {:?}", side + 1, legal.len(), show(legal[0]));
    }

    // A one-ply look: for each of our choices, how much HP the other side has
    // left on average against their random replies.
    let ours = battle.joint_choices(0);
    let theirs = battle.joint_choices(1);
    let mut best = (f64::MAX, ours[0]);
    for &mine in &ours {
        let mut left = 0.0;
        for (k, &reply) in theirs.iter().enumerate() {
            let mut b = battle; // a Battle is Copy: this is the whole cost of trying a line
            b.rng = vgc_engine::rng::Rng::from_words([k as u16, 7, 7, 7]);
            b.choose([mine, reply])?;
            left += hp_fraction(&b, 1);
        }
        let left = left / theirs.len() as f64;
        if left < best.0 {
            best = (left, mine);
        }
    }
    println!("leaves them the least HP ({:.0}%): {:?}", best.0 * 100.0, show(best.1));

    // The position as JSON, bookkeeping included, and back.
    let exported = battle.to_state();
    let text = exported.to_json();
    let again = Battle::from_state(&BattleState::from_json(&text)?)?;
    assert_eq!(again.to_state(), exported);
    println!("exported position: {} bytes of JSON", text.len());
    Ok(())
}

fn show(c: [Choice; 2]) -> String {
    c.map(|c| c.to_showdown()).join(", ")
}

fn hp_fraction(b: &Battle, side: usize) -> f64 {
    let s = &b.sides[side];
    let (hp, max) =
        s.team[..s.n as usize].iter().fold((0u32, 0u32), |(h, m), p| (h + p.hp as u32, m + p.max_hp() as u32));
    hp as f64 / max as f64
}
