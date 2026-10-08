//! A fast Pokémon Champions doubles battle engine, checked turn by turn
//! against Pokémon Showdown.
//!
//! Start with [`Battle::new`], read [`Battle::request`], and answer with
//! [`Battle::choose`]. [`Battle::legal_choices`] lists what each slot may do.

pub mod battle;
pub mod data;
pub mod replay;
#[rustfmt::skip]
mod tables;
pub mod rng;
pub mod state;
pub mod trace;

pub use battle::{Error, PokemonSet};
pub use state::{ACTIVE, Battle, Choice, MAX_TEAM, MonRef, Pokemon, Request, Side, VolKind};
