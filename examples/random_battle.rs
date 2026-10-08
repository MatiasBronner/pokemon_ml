//! Plays one battle with both sides choosing uniformly at random.
//!
//!     cargo run --release --example random_battle

use vgc_engine::data::SPECIES;
use vgc_engine::{Battle, Error, PokemonSet};

fn main() -> Result<(), Error> {
    let set = PokemonSet::from_names;
    // The four Pokémon each side picked at team preview; the first two lead.
    let p1 = [
        set("Garchomp", &["Earthquake", "Rock Slide", "Dragon Claw", "Protect"], "Jolly", [0, 32, 0, 0, 2, 32])?,
        set("Sylveon", &["Hyper Voice", "Moonblast", "Calm Mind", "Protect"], "Modest", [32, 0, 2, 32, 0, 0])?,
        set("Arcanine", &["Flare Blitz", "Extreme Speed", "Will-O-Wisp", "Protect"], "Adamant", [32, 32, 0, 0, 2, 0])?,
        set("Milotic", &["Scald", "Ice Beam", "Recover", "Icy Wind"], "Bold", [32, 0, 32, 2, 0, 0])?,
    ];
    let p2 = [
        set("Dragonite", &["Extreme Speed", "Iron Head", "Dragon Dance", "Protect"], "Adamant", [32, 32, 0, 0, 2, 0])?,
        set("Gardevoir", &["Psychic", "Dazzling Gleam", "Thunder Wave", "Protect"], "Timid", [0, 0, 2, 32, 0, 32])?,
        set(
            "Excadrill",
            &["High Horsepower", "Iron Head", "Rock Slide", "Swords Dance"],
            "Jolly",
            [0, 32, 2, 0, 0, 32],
        )?,
        set("Rotom-Wash", &["Hydro Pump", "Thunderbolt", "Will-O-Wisp", "Protect"], "Calm", [32, 0, 0, 2, 32, 0])?,
    ];
    let mut battle = Battle::new([&p1, &p2], [2026, 10, 8, 1])?;

    // Any generator will do for the players; this one is xorshift.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut pick = |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };

    while !battle.ended {
        let mut choices = [[vgc_engine::Choice::Pass; 2]; 2];
        for (side, slot) in choices.iter_mut().enumerate() {
            let legal = battle.joint_choices(side);
            *slot = legal[pick(legal.len())];
        }
        let describe = |side: usize| choices[side].map(|c| c.to_showdown()).join(", ");
        println!("turn {:>2} ({:?}): p1 [{}]  p2 [{}]", battle.turn, battle.request, describe(0), describe(1));
        battle.choose(choices)?;
    }

    for side in 0..2 {
        let s = &battle.sides[side];
        let team: Vec<String> = (0..s.n as usize)
            .map(|pos| {
                let m = battle.mon(battle.active_at(side, pos));
                format!("{} {}/{}", SPECIES[m.species as usize].name, m.hp, m.max_hp())
            })
            .collect();
        println!("p{}: {}", side + 1, team.join(", "));
    }
    match battle.winner {
        Some(w) => println!("p{} wins on turn {}", w + 1, battle.turn),
        None => println!("tie on turn {}", battle.turn),
    }
    Ok(())
}
