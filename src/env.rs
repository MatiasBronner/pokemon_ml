//! Games for a model to play: Team Preview and the battle as one run of
//! decisions, actions as numbers, and many games stepped at once.
//!
//! A [`Game`] starts at Team Preview, where each side picks which four of its
//! six Pokémon to bring and which two lead, and goes on as a [`Battle`]. At
//! every decision both sides give an action and get an observation
//! ([`crate::obs`]). [`VecEnv`] holds many games, plays teams from a
//! [`Pool`] against each other, starts a new game when one ends and fills
//! the observations of all of them in parallel.
//!
//! # Actions
//!
//! At Team Preview an action is one number below [`N_PREVIEW`] (90): which
//! pair leads, and which pair of the other four waits behind
//! ([`preview_table`]).
//!
//! In battle a side gives two numbers below [`N_ACTIONS`] (47), one for each
//! of its positions on the field:
//!
//! | action | meaning |
//! |---|---|
//! | `10 * move + 2 * target + mega` | use move 0 to 3. Target 0: the move takes none; 1, 2: the opponent's first and second position; 3, 4: one's own first and second. `mega` 1: Mega Evolve first. |
//! | `40 + j` | switch to the Pokémon registered `j`th, 0 to 5 |
//! | `46` | pass: nothing to do in this position |
//!
//! A switch names the Pokémon and not where it stands on the bench, so that
//! it means the same thing all game, and the same thing as the Pokémon's
//! token in the observation. What is legal comes with each observation as a
//! mask: `N_ACTIONS` bytes for the first position, then for each of its
//! actions `N_ACTIONS` bytes for the second position given the first (the
//! two cannot switch to the same Pokémon, or both Mega Evolve).

use rayon::prelude::*;

use crate::battle::{Error, PokemonSet, calc_stats, struggle_id};
use crate::data::*;
use crate::obs::{OBS_F, OBS_I, Phase, ROSTER, Timers, observe_battle, observe_preview};
use crate::rng::Rng;
use crate::speed::{self, Speeds};
use crate::state::*;
use crate::teams::{Pool, Sampler, Variation};

/// Actions for one position on the field.
pub const N_ACTIONS: usize = 47;
/// Ways to pick two leads and two more from six.
pub const N_PREVIEW: usize = 90;
/// Bytes of legal-action mask per observation.
pub const OBS_M: usize = N_ACTIONS + N_ACTIONS * N_ACTIONS;

const SWITCH: usize = 40;
const PASS: usize = 46;
const TARGETS: [i8; 5] = [0, 1, 2, -1, -2];

/// The four roster entries each Team Preview action brings: the two leads, then the two behind.
pub fn preview_table() -> &'static [[u8; 4]; N_PREVIEW] {
    static TABLE: std::sync::OnceLock<[[u8; 4]; N_PREVIEW]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut out = [[0u8; 4]; N_PREVIEW];
        let mut n = 0;
        for a in 0..6u8 {
            for b in a + 1..6 {
                let rest: Vec<u8> = (0..6).filter(|&x| x != a && x != b).collect();
                for (k, &c) in rest.iter().enumerate() {
                    for &d in &rest[k + 1..] {
                        out[n] = [a, b, c, d];
                        n += 1;
                    }
                }
            }
        }
        debug_assert_eq!(n, N_PREVIEW);
        out
    })
}

/// A player that needs no model, to measure one against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Baseline {
    /// Any legal action, each as likely as the next.
    Random,
    /// The strongest attack it has for the targets in front of it, going by
    /// power, type and accuracy alone. It switches only when it must, and
    /// brings a random four.
    Greedy,
    /// Tries its choices out in the simulator, one turn ahead, and keeps the
    /// one that leaves it best off: Pokémon and HP left, its own against the
    /// other side's. It plays each against what the greedy player would
    /// answer (twice) and against a random answer. It sees everything: the
    /// other side's stats, items and moves are the real ones, which no
    /// player has. Only the dice are not: each try rolls its own. A bar to
    /// clear, then, and no model of fair play. Replacements and Team Preview
    /// it picks at random.
    Lookahead,
}

/// One game: Team Preview, then the battle.
#[derive(Clone, Debug)]
pub struct Game {
    rosters: [Vec<PokemonSet>; 2],
    /// Each registered Pokémon's stats.
    stats: [[[u16; 6]; ROSTER]; 2],
    open: bool,
    battle: Option<Battle>,
    timers: Timers,
    speeds: Speeds,
    /// What the engine noted during the last step, for `speeds` (kept to reuse the space).
    events: Vec<speed::Event>,
}

impl Game {
    /// A game between two teams of six, at Team Preview. `open`: the players have each other's team sheets.
    pub fn new(rosters: [Vec<PokemonSet>; 2], open: bool) -> Result<Game, Error> {
        let mut stats = [[[0u16; 6]; ROSTER]; 2];
        for side in 0..2 {
            if rosters[side].len() != ROSTER {
                return Err(Error::BadTeam(format!(
                    "side {} registers {} Pokémon; a game is played with {ROSTER}",
                    side + 1,
                    rosters[side].len()
                )));
            }
            for (j, set) in rosters[side].iter().enumerate() {
                stats[side][j] = calc_stats(set.species, set.nature, set.stat_points);
            }
        }
        let speeds = Speeds::new([&rosters[0], &rosters[1]], open);
        Ok(Game { rosters, stats, open, battle: None, timers: Timers::default(), speeds, events: Vec::new() })
    }

    pub fn phase(&self) -> Phase {
        match &self.battle {
            None => Phase::Preview,
            Some(b) if b.request == Request::Switch => Phase::Switch,
            Some(_) => Phase::Move,
        }
    }

    /// What each side can tell of the other's Speed.
    pub fn speeds(&self) -> &Speeds {
        &self.speeds
    }

    /// Has [`Speeds`] check itself against the truth as the game goes, and panic on a difference: for tests.
    pub fn check_speeds(&mut self) {
        self.speeds.strict = true;
    }

    /// The battle, once Team Preview is over.
    pub fn battle(&self) -> Option<&Battle> {
        self.battle.as_ref()
    }

    pub fn rosters(&self) -> [&[PokemonSet]; 2] {
        [&self.rosters[0], &self.rosters[1]]
    }

    pub fn open_team_sheets(&self) -> bool {
        self.open
    }

    pub fn ended(&self) -> bool {
        self.battle.as_ref().is_some_and(|b| b.ended)
    }

    /// The side that won, once the game has ended; `None` for a tie.
    pub fn winner(&self) -> Option<usize> {
        self.battle.as_ref().and_then(|b| b.winner).map(|w| w as usize)
    }

    /// The number a choice goes by.
    pub fn action_of(&self, side: usize, choice: Choice) -> Option<usize> {
        action_of(self.battle.as_ref()?, side, choice)
    }

    /// The choice a number stands for, whether or not it is legal now.
    pub fn choice_of(&self, side: usize, action: usize) -> Option<Choice> {
        choice_of(self.battle.as_ref()?, side, action)
    }

    /// Every pair of actions `side` may give now.
    pub fn legal(&self, side: usize) -> Vec<[usize; 2]> {
        let Some(b) = &self.battle else {
            return (0..N_PREVIEW).map(|a| [a, 0]).collect();
        };
        let act = |c: Choice| self.action_of(side, c).expect("a legal choice has a number");
        b.joint_choices(side).into_iter().map(|c| [act(c[0]), act(c[1])]).collect()
    }

    /// Fills `mask` ([`OBS_M`] bytes) with what `side` may do and returns how
    /// many pairs of actions that is. At Team Preview the mask stays empty:
    /// every one of the [`N_PREVIEW`] picks is allowed.
    pub fn masks(&self, side: usize, mask: &mut [u8]) -> usize {
        assert_eq!(mask.len(), OBS_M, "a mask buffer of the wrong size");
        mask.fill(0);
        let Some(b) = &self.battle else {
            return N_PREVIEW;
        };
        masks_of(b, side, mask)
    }

    /// Team Preview ends: each side brings the Pokémon it names, as places in
    /// its registered team, in the order it wants them (the first two lead).
    /// `seed` seeds the battle that starts. ([`Game::act`] does this from a
    /// Team Preview action, which names the leads in the order registered.)
    pub fn start(&mut self, picks: [&[usize]; 2], seed: [u16; 4]) -> Result<(), Error> {
        let rosters = [&self.rosters[0][..], &self.rosters[1][..]];
        let b = speed::record(&mut self.events, || Battle::with_rosters(rosters, picks, self.open, seed))?;
        self.timers = Timers::default();
        self.timers.update(&b);
        self.speeds.digest(&self.events, &b);
        self.battle = Some(b);
        Ok(())
    }

    /// Both sides choose, in the battle's own terms. ([`Game::act`] does this from actions.)
    pub fn choose(&mut self, choices: [[Choice; ACTIVE]; 2]) -> Result<(), Error> {
        let b = self.battle.as_mut().ok_or_else(|| Error::BadChoice("the battle has not started".into()))?;
        speed::record(&mut self.events, || b.choose(choices))?;
        self.timers.update(b);
        self.speeds.digest(&self.events, b);
        Ok(())
    }

    /// Both sides act. At Team Preview only the first number of each side
    /// counts, and `seed` seeds the battle that starts.
    pub fn act(&mut self, actions: [[usize; 2]; 2], seed: [u16; 4]) -> Result<(), Error> {
        if self.battle.is_none() {
            let mut picks = [[0usize; 4]; 2];
            for side in 0..2 {
                let pick = preview_table().get(actions[side][0]).ok_or_else(|| {
                    Error::BadChoice(format!("p{}: {} is not a Team Preview action", side + 1, actions[side][0]))
                })?;
                picks[side] = pick.map(|j| j as usize);
            }
            return self.start([&picks[0], &picks[1]], seed);
        }
        let mut choices = [[Choice::Pass; ACTIVE]; 2];
        for side in 0..2 {
            for pos in 0..ACTIVE {
                choices[side][pos] = self.choice_of(side, actions[side][pos]).ok_or_else(|| {
                    Error::BadChoice(format!("p{}: {} is not an action", side + 1, actions[side][pos]))
                })?;
            }
        }
        self.choose(choices)
    }

    /// `view`'s side of the game, into buffers of [`OBS_F`], [`OBS_I`] and [`OBS_M`].
    pub fn observe(&self, view: usize, f: &mut [f32], i: &mut [i16], mask: &mut [u8]) {
        match &self.battle {
            None => observe_preview(self.rosters(), &self.speeds, &self.stats[view], self.open, view, f, i),
            Some(b) => observe_battle(b, &self.timers, &self.speeds, &self.stats[view], view, f, i),
        }
        let n = self.masks(view, mask);
        let info = &mut i[OBS_I - crate::obs::INFO..];
        info[1] = n.min(i16::MAX as usize) as i16;
        info[3] = (n > 1) as i16;
    }

    /// For search: redraws what `view` has not been shown of the other side,
    /// so that a copy of the game can be played on without knowing the truth.
    ///
    /// * Which Pokémon wait unseen on the bench: each place there is given
    ///   to a registered Pokémon not yet seen, drawn at random (the one
    ///   really there among them). Not on a team that registered an
    ///   Illusion Pokémon, whose disguises are tied to where its team-mates
    ///   stand.
    /// * Every Pokémon's stat points: Speed drawn from what the order of
    ///   moves still allows ([`Speeds::belief`]), the rest by the rule of
    ///   thumb of `teams::guess_spread`. HP keeps the percentage and the
    ///   colour of bar `view` was shown.
    ///
    /// Moves, items, abilities and natures are left as they are: with open
    /// team sheets `view` has been given them. With closed sheets they are
    /// still the truth, which a search should not be given.
    ///
    /// The random number generator is left alone; [`Game::reseed`] it too.
    pub fn determinize(&mut self, view: usize, rng: &mut Rng) -> Result<(), Error> {
        let opp = 1 - view;
        let Some(b) = self.battle.as_mut() else {
            return Ok(());
        };
        let illusion = self.rosters[opp].iter().any(|s| s.ability == ab::ILLUSION);
        if !illusion {
            let side = &b.sides[opp];
            let taken = side.taken();
            let places: Vec<usize> =
                (0..side.n as usize).filter(|&i| side.shown[i].seen == 0 && !side.team[i].is_active).collect();
            let mut free: Vec<usize> = (0..side.n_roster as usize).filter(|&j| taken & (1 << j) == 0).collect();
            if !places.is_empty() && free.len() >= places.len() {
                // The first places.len() of a shuffle.
                for k in 0..places.len() {
                    let pick = k + rng.below((free.len() - k) as u32) as usize;
                    free.swap(k, pick);
                }
                for &i in &places {
                    let roster = &mut b.sides[opp].roster;
                    if let Some(old) = roster.iter_mut().find(|l| l.brought == i as u8) {
                        old.brought = NOT_LISTED;
                    }
                }
                for (&i, &j) in places.iter().zip(&free) {
                    let position = b.sides[opp].team[i].position as usize;
                    let mon = b.new_mon(&self.rosters[opp][j], position)?;
                    b.sides[opp].team[i] = mon;
                    b.sides[opp].roster[j].brought = i as u8;
                }
            }
        }
        for i in 0..b.sides[opp].n as usize {
            let Some(j) = b.sides[opp].roster.iter().position(|l| l.brought == i as u8) else { continue };
            let m = &mut b.sides[opp].team[i];
            let guessed = guess_points(&self.rosters[opp][j], self.speeds.belief(opp, j), rng);
            // The bar shown, never the HP behind it.
            let shown = (m.hp > 0).then(|| crate::shown::hp_shown(m.hp, m.stats[0]));
            let (points, hp) = fit_hp(guessed, m.set_species, m.nature, shown, rng.next_u32());
            m.stat_points = points;
            // Its own HP, transformed or not (Transform copies everything but HP).
            m.stats[0] = calc_stats(m.set_species, m.nature, points)[0];
            m.hp = hp;
            if !m.transformed {
                let stats = calc_stats(m.species, m.nature, points);
                m.stats[1..].copy_from_slice(&stats[1..]);
            }
            m.speed = m.stats[5] as i32;
        }
        // A Pokémon that turned into a team-mate copied that team-mate's stats: the ones
        // just drawn, not the truth. (One that turned into the searching side's own
        // Pokémon copied stats that side knows.)
        for i in 0..b.sides[opp].n as usize {
            let m = b.sides[opp].team[i];
            if let Some(from) = m.copied_from.filter(|r| m.transformed && r.side as usize == opp) {
                let copied = b.sides[opp].team[from.idx as usize].stats;
                b.sides[opp].team[i].stats[1..].copy_from_slice(&copied[1..]);
            }
        }
        b.update_speed();
        // Countdowns nobody is shown, the searching side's own included.
        for side in 0..2 {
            redraw_clocks(b, side, rng);
        }
        Ok(())
    }

    /// Gives the battle a fresh random number generator, so that a copy played on does not roll the real game's dice.
    pub fn reseed(&mut self, seed: [u16; 4]) {
        if let Some(b) = self.battle.as_mut() {
            b.rng = Rng::from_words(seed);
        }
    }

    /// What a [`Baseline`] player does on `side`.
    pub fn baseline(&self, side: usize, kind: Baseline, rng: &mut Rng) -> [usize; 2] {
        let Some(b) = &self.battle else {
            return [rng.below(N_PREVIEW as u32) as usize, 0];
        };
        if kind == Baseline::Random || b.request != Request::Move {
            let legal = self.legal(side);
            return legal[rng.below(legal.len() as u32) as usize];
        }
        let act = |c: Choice| self.action_of(side, c).expect("a legal choice has a number");
        let best = if kind == Baseline::Lookahead { lookahead(b, side, rng) } else { greedy(b, side, rng) };
        [act(best[0]), act(best[1])]
    }
}

/// The roster entry of the Pokémon at a position of a side's current order.
fn entry_at(b: &Battle, side: usize, position: usize) -> Option<usize> {
    let s = &b.sides[side];
    let idx = *s.order.get(position)?;
    s.roster[..s.n_roster as usize].iter().position(|l| l.brought == idx)
}

/// The number a choice goes by in a battle.
pub(crate) fn action_of(b: &Battle, side: usize, choice: Choice) -> Option<usize> {
    Some(match choice {
        Choice::Pass => PASS,
        Choice::Move { slot, target, mega } => {
            let t = TARGETS.iter().position(|&x| x == target)?;
            10 * slot as usize + 2 * t + mega as usize
        }
        Choice::Switch { to } => SWITCH + entry_at(b, side, to as usize)?,
    })
}

/// The choice a number stands for in a battle, whether or not it is legal now.
pub(crate) fn choice_of(b: &Battle, side: usize, action: usize) -> Option<Choice> {
    Some(match action {
        PASS => Choice::Pass,
        a if a < SWITCH => Choice::Move { slot: (a / 10) as u8, target: TARGETS[a % 10 / 2], mega: a % 2 == 1 },
        a if a < PASS => {
            let s = &b.sides[side];
            let brought = s.roster.get(a - SWITCH)?.brought;
            if brought == NOT_LISTED {
                return None;
            }
            Choice::Switch { to: s.team[brought as usize].position }
        }
        _ => return None,
    })
}

/// Fills `mask` with what `side` may do in a battle and returns how many pairs of actions that is.
pub(crate) fn masks_of(b: &Battle, side: usize, mask: &mut [u8]) -> usize {
    assert_eq!(mask.len(), OBS_M, "a mask buffer of the wrong size");
    mask.fill(0);
    if b.ended {
        return 0;
    }
    let act = |c: Choice| action_of(b, side, c).expect("a legal choice has a number");
    let first: Vec<(Choice, usize)> = b.legal_choices(side, 0).into_iter().map(|c| (c, act(c))).collect();
    let second: Vec<(Choice, usize)> = b.legal_choices(side, 1).into_iter().map(|c| (c, act(c))).collect();
    let mut n = 0;
    for &(x, ax) in &first {
        for &(y, ay) in &second {
            if b.pair_ok(side, &[x, y]) {
                mask[ax] = 1;
                mask[N_ACTIONS * (1 + ax) + ay] = 1;
                n += 1;
            }
        }
    }
    n
}

/// What the greedy player does on a turn.
fn greedy(b: &Battle, side: usize, rng: &mut Rng) -> [Choice; ACTIVE] {
    let mut best = [Choice::Pass; ACTIVE];
    for pos in 0..ACTIVE {
        let mut top = f32::NEG_INFINITY;
        for c in b.legal_choices(side, pos) {
            let mut trial = best;
            trial[pos] = c;
            if pos == 1 && !b.pair_ok(side, &trial) {
                continue;
            }
            // A coin's worth of noise settles ties.
            let score = greedy_score(b, side, pos, c) + rng.below(1000) as f32 * 1e-4;
            if score > top {
                (top, best[pos]) = (score, c);
            }
        }
    }
    best
}

/// How well off `side` is: a point for each Pokémon it has left and up to another for that
/// one's HP, less the same for the other side. A battle that is over outweighs any of that.
fn standing(b: &Battle, side: usize) -> f32 {
    if b.ended {
        return match b.winner {
            Some(w) if w as usize == side => 100.0,
            Some(_) => -100.0,
            None => 0.0,
        };
    }
    let worth = |s: usize| -> f32 {
        let team = &b.sides[s].team[..b.sides[s].n as usize];
        team.iter().filter(|m| !m.fainted).map(|m| 1.0 + m.hp as f32 / m.max_hp().max(1) as f32).sum()
    };
    worth(side) - worth(1 - side)
}

/// What the look-ahead player does on a turn.
fn lookahead(b: &Battle, side: usize, rng: &mut Rng) -> [Choice; ACTIVE] {
    let foe = 1 - side;
    let anything = b.joint_choices(foe);
    let replies = [greedy(b, foe, rng), greedy(b, foe, rng), anything[rng.below(anything.len() as u32) as usize]];
    // How things stand after a turn of `mine` against each reply, with dice of the try's own.
    let value = |mine: [Choice; ACTIVE], rng: &mut Rng| -> f32 {
        let mut total = 0.0;
        for reply in replies {
            let mut after = *b;
            after.rng = Rng::from_words(seed_of(rng));
            let mut choices = [[Choice::Pass; ACTIVE]; 2];
            (choices[side], choices[foe]) = (mine, reply);
            if after.choose(choices).is_err() {
                return f32::NEG_INFINITY;
            }
            total += standing(&after, side);
        }
        total
    };
    // Each position's choices tried beside the other's greedy pick, to find the four worth pairing up.
    let base = greedy(b, side, rng);
    let mut short: [Vec<(f32, Choice)>; ACTIVE] = [Vec::new(), Vec::new()];
    for (pos, list) in short.iter_mut().enumerate() {
        for c in b.legal_choices(side, pos) {
            let mut trial = base;
            trial[pos] = c;
            if b.pair_ok(side, &trial) {
                list.push((value(trial, rng), c));
            }
        }
        list.sort_by(|x, y| y.0.total_cmp(&x.0));
        list.truncate(4);
    }
    let (mut best, mut top) = (base, f32::NEG_INFINITY);
    for &(_, x) in &short[0] {
        for &(_, y) in &short[1] {
            if b.pair_ok(side, &[x, y]) {
                let v = value([x, y], rng);
                if v > top {
                    (top, best) = (v, [x, y]);
                }
            }
        }
    }
    best
}

/// How much the greedy player likes a choice: roughly the damage a move promises.
fn greedy_score(b: &Battle, side: usize, pos: usize, choice: Choice) -> f32 {
    let Choice::Move { slot, target, mega } = choice else {
        return if choice == Choice::Pass { 0.0 } else { 0.5 };
    };
    let r = b.active(side, pos);
    let m = b.mon(r);
    let id = if m.locked_move != NO_MOVE {
        m.locked_move
    } else if b.usable_moves(r) {
        m.moves[slot as usize].id
    } else {
        struggle_id()
    };
    let d = &MOVES[id as usize];
    if d.category == Category::Status {
        return 1.0;
    }
    let (types, n) = b.get_types(r, false);
    let stab = if types[..n].contains(&d.typ) { 1.5 } else { 1.0 };
    let power = if d.base_power == 0 { 60.0 } else { d.base_power as f32 };
    let accuracy = if d.accuracy == 0 { 1.0 } else { d.accuracy as f32 / 100.0 };
    // How well the move's type does against whoever stands at a position, going by the
    // types that Pokémon is seen to have. Nothing for an empty position.
    let against = |s: usize, p: usize| -> f32 {
        let t = b.active(s, p);
        let tm = b.mon(t);
        if p >= b.sides[s].n as usize || !tm.is_active || tm.fainted {
            return 0.0;
        }
        let (seen, n) = if s != side && tm.illusion != 0 {
            let sp = SPECIES[tm.live.species as usize].types;
            ([sp[0], sp[1], Type::None], 2)
        } else {
            b.get_types(t, false)
        };
        let mut mult = 1.0;
        for &def in &seen[..n] {
            if (d.typ as usize) < 18 && (def as usize) < 18 {
                mult *= [1.0, 2.0, 0.5, 0.0][TYPE_CHART[d.typ as usize][def as usize] as usize];
            }
        }
        mult
    };
    let foe = 1 - side;
    let reach = match (target, d.target) {
        (1 | 2, _) => against(foe, target as usize - 1),
        // Never at a partner on purpose.
        (-1 | -2, _) => 0.0,
        (_, Target::AllAdjacentFoes) => 0.75 * (against(foe, 0) + against(foe, 1)),
        (_, Target::AllAdjacent) => 0.75 * (against(foe, 0) + against(foe, 1) - against(side, 1 - pos)),
        _ => 0.5 * (against(foe, 0) + against(foe, 1)),
    };
    2.0 + power * stab * accuracy * reach * if mega { 1.05 } else { 1.0 }
}

/// How a [`VecEnv`] plays.
#[derive(Clone, Debug)]
pub struct Config {
    /// Games running at once.
    pub envs: usize,
    pub seed: u64,
    /// The share of games played with open team sheets.
    pub open_sheets: f64,
    /// How the pool's teams are varied from game to game.
    pub variation: Variation,
    /// A game still going after this many turns is stopped and scored as a tie.
    pub max_turns: u16,
    /// Worker threads; 0 for one per core.
    pub threads: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config { envs: 256, seed: 1, open_sheets: 0.5, variation: Variation::default(), max_turns: 100, threads: 0 }
    }
}

/// Totals since a [`VecEnv`] was made.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Stats {
    /// Games finished.
    pub games: u64,
    /// Decisions taken (Team Preview counts as one).
    pub decisions: u64,
    /// Turns played in the games that finished.
    pub turns: u64,
    /// Games that ended in a tie or were stopped at the turn limit.
    pub ties: u64,
    /// Games stopped by the turn cap before the battle ended.
    pub timeouts: u64,
    /// Games the first side won.
    pub first_wins: u64,
}

impl Stats {
    fn add(self, o: Stats) -> Stats {
        Stats {
            games: self.games + o.games,
            decisions: self.decisions + o.decisions,
            turns: self.turns + o.turns,
            ties: self.ties + o.ties,
            timeouts: self.timeouts + o.timeouts,
            first_wins: self.first_wins + o.first_wins,
        }
    }
}

/// Many games at once. Observations are indexed game by game, the first
/// side's then the second's: observation `2 * game + side`.
pub struct VecEnv {
    games: Vec<Game>,
    rngs: Vec<Rng>,
    sampler: Sampler,
    config: Config,
    workers: rayon::ThreadPool,
    stats: Stats,
    last_episodes: Vec<[u64; 2]>,
}

fn fresh(sampler: &Sampler, config: &Config, rng: &mut Rng) -> Game {
    let a = sampler.sample(rng, &config.variation).team;
    let b = sampler.sample(rng, &config.variation).team;
    let open = config.open_sheets > 0.0 && (rng.below(1_000_000) as f64) < config.open_sheets * 1_000_000.0;
    Game::new([a, b], open).expect("the pool's teams have six Pokémon")
}

fn seed_of(rng: &mut Rng) -> [u16; 4] {
    [rng.below(65536) as u16, rng.below(65536) as u16, rng.below(65536) as u16, rng.below(65536) as u16]
}

impl VecEnv {
    /// `config.envs` games between teams of `pool`, all at Team Preview.
    pub fn new(pool: &Pool, config: Config) -> Result<VecEnv, Error> {
        let sampler = Sampler::new(pool)?;
        if let Some(short) = pool.teams.iter().find(|t| t.team.len() != ROSTER) {
            return Err(Error::BadTeam(format!(
                "{}'s team has {} Pokémon, not {ROSTER}",
                short.player,
                short.team.len()
            )));
        }
        let mut rngs: Vec<Rng> = (0..config.envs as u64)
            .map(|k| {
                // splitmix64, so that neighbouring seeds give unrelated games
                let mut z = config.seed.wrapping_add((k + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                Rng { seed: z ^ (z >> 31), calls: 0 }
            })
            .collect();
        let games = rngs.iter_mut().map(|rng| fresh(&sampler, &config, rng)).collect();
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .build()
            .map_err(|e| Error::BadTeam(format!("cannot start worker threads: {e}")))?;
        let last_episodes = vec![[0; 2]; config.envs];
        Ok(VecEnv { games, rngs, sampler, config, workers, stats: Stats::default(), last_episodes })
    }

    /// Completed episode turns and timeout flag from the last step; zero if not done.
    pub fn last_episodes(&self) -> &[[u64; 2]] {
        &self.last_episodes
    }

    pub fn len(&self) -> usize {
        self.games.len()
    }

    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }

    pub fn games(&self) -> &[Game] {
        &self.games
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    pub fn threads(&self) -> usize {
        self.workers.current_num_threads()
    }

    fn check(&self, f: &[f32], i: &[i16], mask: &[u8]) {
        let n = 2 * self.games.len();
        assert!(
            f.len() == n * OBS_F && i.len() == n * OBS_I && mask.len() == n * OBS_M,
            "observation buffers must hold {n} observations"
        );
    }

    /// Every game from both sides, as it stands.
    pub fn observe(&self, f: &mut [f32], i: &mut [i16], mask: &mut [u8]) {
        self.check(f, i, mask);
        self.workers.install(|| {
            (
                self.games.par_iter(),
                f.par_chunks_mut(2 * OBS_F),
                i.par_chunks_mut(2 * OBS_I),
                mask.par_chunks_mut(2 * OBS_M),
            )
                .into_par_iter()
                .with_min_len(8)
                .for_each(|(game, f, i, mask)| observe_both(game, f, i, mask));
        });
    }

    /// Both sides of every game act (`actions`: four numbers a game, the
    /// first side's two then the second's), and the observations of what
    /// follows are filled in. A game that ends gives its result in `reward`
    /// (two numbers a game: 1 for the winner, -1 for the loser, 0 each for a
    /// tie) and sets `done`; a new game has then already taken its place,
    /// and the observation is that game's Team Preview.
    pub fn step(
        &mut self,
        actions: &[i32],
        f: &mut [f32],
        i: &mut [i16],
        mask: &mut [u8],
        reward: &mut [f32],
        done: &mut [u8],
    ) -> Result<(), Error> {
        self.check(f, i, mask);
        let n = self.games.len();
        assert!(actions.len() == 4 * n && reward.len() == 2 * n && done.len() == n, "one set of actions a game");
        let (sampler, config) = (&self.sampler, &self.config);
        let stepped = self.workers.install(|| {
            (
                self.games.par_iter_mut(),
                self.rngs.par_iter_mut(),
                actions.par_chunks(4),
                f.par_chunks_mut(2 * OBS_F),
                i.par_chunks_mut(2 * OBS_I),
                mask.par_chunks_mut(2 * OBS_M),
                reward.par_chunks_mut(2),
                done.par_iter_mut(),
                self.last_episodes.par_iter_mut(),
            )
                .into_par_iter()
                .with_min_len(8)
                .map(|(game, rng, a, f, i, mask, reward, done, episode)| -> Result<Stats, Error> {
                    let number = |x: i32| usize::try_from(x).unwrap_or(usize::MAX);
                    let seed = seed_of(rng);
                    game.act([[number(a[0]), number(a[1])], [number(a[2]), number(a[3])]], seed)?;
                    let mut stats = Stats { decisions: 1, ..Stats::default() };
                    reward.fill(0.0);
                    *done = 0;
                    *episode = [0; 2];
                    let b = game.battle.as_ref().expect("a game that has been acted on has a battle");
                    if b.ended || b.turn > config.max_turns {
                        *done = 1;
                        stats.games = 1;
                        stats.timeouts = (!b.ended) as u64;
                        stats.turns = b.turn as u64;
                        *episode = [stats.turns, stats.timeouts];
                        match b.winner.filter(|_| b.ended) {
                            Some(w) => {
                                reward[w as usize] = 1.0;
                                reward[1 - w as usize] = -1.0;
                                stats.first_wins = (w == 0) as u64;
                            }
                            None => stats.ties = 1,
                        }
                        *game = fresh(sampler, config, rng);
                    }
                    observe_both(game, f, i, mask);
                    Ok(stats)
                })
                .try_reduce(Stats::default, |a, b| Ok(a.add(b)))
        })?;
        self.stats = self.stats.add(stepped);
        Ok(())
    }

    /// What a [`Baseline`] player would do on each side of every game (four numbers a game, as [`VecEnv::step`] takes them).
    pub fn baseline(&mut self, kind: Baseline, actions: &mut [i32]) {
        assert_eq!(actions.len(), 4 * self.games.len(), "one set of actions a game");
        self.workers.install(|| {
            (self.games.par_iter(), self.rngs.par_iter_mut(), actions.par_chunks_mut(4))
                .into_par_iter()
                .with_min_len(8)
                .for_each(|(game, rng, out)| {
                    for side in 0..2 {
                        let a = game.baseline(side, kind, rng);
                        out[2 * side] = a[0] as i32;
                        out[2 * side + 1] = a[1] as i32;
                    }
                });
        });
    }
}

/// Stat points for a Pokémon whose real ones are not known: Speed drawn from
/// what `belief` still allows with its nature and item, then the rule of thumb
/// of `teams::guess_spread` for the rest (an attacker's attacking stat and HP,
/// a support's HP, then Defense and Special Defense with what is left).
fn guess_points(set: &PokemonSet, belief: &speed::Belief, rng: &mut Rng) -> [u8; 6] {
    const HP: usize = 0;
    const ATK: usize = 1;
    const DEF: usize = 2;
    const SPA: usize = 3;
    const SPD: usize = 4;
    const SPE: usize = 5;
    let possible: Vec<u8> = (0..=32u8).filter(|&p| belief.has(set.nature, p, set.item)).collect();
    let speed =
        if possible.is_empty() { rng.below(33) as u8 } else { possible[rng.below(possible.len() as u32) as usize] };
    let (mut physical, mut special, mut attacks) = (0u32, 0u32, 0);
    for &m in &set.moves {
        let data = &MOVES[m as usize];
        match data.category {
            Category::Physical => physical += data.base_power.max(40) as u32,
            Category::Special => special += data.base_power.max(40) as u32,
            Category::Status => continue,
        }
        attacks += 1;
    }
    let base = SPECIES[set.species as usize].base;
    let offence = if physical > special || (physical == special && base[ATK] >= base[SPA]) { ATK } else { SPA };
    let mut points = [0u8; 6];
    points[SPE] = speed;
    let mut left = 66 - speed;
    let order: &[usize] = if attacks >= 2 { &[offence, HP, DEF, SPD] } else { &[HP, DEF, SPD] };
    for &k in order {
        points[k] = left.min(32);
        left -= points[k];
    }
    points
}

/// Stat points like `guessed`, with the HP points moved as little as needed for
/// the bar to be able to show `shown` (`None`: fainted), and an HP that shows it,
/// each such HP as likely as the next (picked by `pick`, 32 random bits).
///
/// It sees the bar, never the HP behind it or the real stat points, and always
/// finds a fit: the HP points the Pokémon really has are one, and every amount
/// from 0 to 32 is tried, nearest the guess first. Points HP takes beyond what is
/// spare come from Attack and the defences, the largest first; Speed is left as
/// guessed, since the order of moves has a say in it.
fn fit_hp(
    guessed: [u8; 6],
    species: u16,
    nature: (u8, u8),
    shown: Option<(u8, crate::shown::Bar)>,
    pick: u32,
) -> ([u8; 6], u16) {
    let with_hp = |hp_points: u8| {
        let mut p = guessed;
        p[0] = hp_points;
        let mut over = p.iter().map(|&x| x as i32).sum::<i32>() - 66;
        while over > 0 {
            let k = (1..=4).max_by_key(|&k| (p[k], std::cmp::Reverse(k))).unwrap();
            let take = (p[k] as i32).min(over);
            p[k] -= take as u8;
            over -= take;
        }
        p
    };
    let tries = (0..=32i32).flat_map(|d| [guessed[0] as i32 - d, guessed[0] as i32 + d]);
    for hp_points in tries.filter(|p| (0..=32).contains(p)) {
        let points = with_hp(hp_points as u8);
        let max = calc_stats(species, nature, points)[0];
        let Some(shown) = shown else { return (points, 0) };
        let fits: Vec<u16> = (1..=max).filter(|&h| crate::shown::hp_shown(h, max) == shown).collect();
        if !fits.is_empty() {
            return (points, fits[((pick as u64 * fits.len() as u64) >> 32) as usize]);
        }
    }
    // Not reached (see above). Should it be, the guess stands, still without the truth.
    let max = calc_stats(species, nature, guessed)[0];
    (guessed, if shown.is_some() { (((pick as u64 * max as u64) >> 32) as u16).max(1) } else { 0 })
}

/// Draws again, for each of `side`'s Pokémon, the countdowns nobody is shown:
/// how long its sleep (2 turns one time in three, otherwise 3), confusion (2
/// to 5 attempts), rampage (2 or 3 turns) and binding (5 or 6 turns) last.
/// What everyone has seen is kept: how long each has gone on so far
/// (`Pokemon::drawn` less what is left), so only lengths longer than that are
/// drawn, as often as the game would draw them. Rest's sleep, which is not
/// drawn, and a countdown whose start is not known (a position written by
/// hand) are left as they are. Exactly four draws a Pokémon, used or not.
fn redraw_clocks(b: &mut Battle, side: usize, rng: &mut Rng) {
    const SLEEP: &[(u8, u32)] = &[(2, 1), (3, 2)];
    const CONFUSION: &[(u8, u32)] = &[(2, 1), (3, 1), (4, 1), (5, 1)];
    const RAMPAGE: &[(u8, u32)] = &[(2, 1), (3, 1)];
    const BINDING: &[(u8, u32)] = &[(5, 1), (6, 1)];
    for i in 0..b.sides[side].n as usize {
        let r = MonRef { side: side as u8, idx: i as u8 };
        let draws = [rng.next_u32(), rng.next_u32(), rng.next_u32(), rng.next_u32()];
        let m = b.mon(r);
        let (drawn, active) = (m.drawn, m.is_active);
        if m.status == Status::Slp
            && let Some((start, left)) = redraw(drawn[SLEEP_CLOCK], m.status_time, SLEEP, draws[0])
        {
            let m = b.mon_mut(r);
            (m.drawn[SLEEP_CLOCK], m.status_time) = (start, left);
        }
        if !active {
            continue;
        }
        if let Some(v) = b.vol_mut(r, VolKind::Confusion)
            && let Some((start, left)) = redraw(drawn[CONFUSION_CLOCK], v.data.min(255) as u8, CONFUSION, draws[1])
        {
            v.data = left as u16;
            b.mon_mut(r).drawn[CONFUSION_CLOCK] = start;
        }
        if let Some(v) = b.vol_mut(r, VolKind::Lockedmove)
            && let Some((start, left)) = redraw(drawn[RAMPAGE_CLOCK], v.st.a.clamp(0, 255) as u8, RAMPAGE, draws[2])
        {
            v.st.a = left as i16;
            b.mon_mut(r).drawn[RAMPAGE_CLOCK] = start;
        }
        if let Some(v) = b.vol_mut(r, VolKind::Partiallytrapped)
            && let Some((start, left)) = redraw(drawn[BINDING_CLOCK], v.duration, BINDING, draws[3])
        {
            v.duration = left;
            b.mon_mut(r).drawn[BINDING_CLOCK] = start;
        }
    }
}

/// A countdown drawn at `start` with `left` to go has gone on `start - left`;
/// draws a new start among `lengths` (each with its weight) longer than that,
/// with `pick`, and returns it with what it leaves to go.
fn redraw(start: u8, left: u8, lengths: &[(u8, u32)], pick: u32) -> Option<(u8, u8)> {
    if start == 0 || left == 0 || left > start {
        return None;
    }
    let gone = start - left;
    let fits: Vec<(u8, u32)> = lengths.iter().copied().filter(|&(n, _)| n > gone).collect();
    let total: u32 = fits.iter().map(|&(_, w)| w).sum();
    let mut at = ((pick as u64 * total as u64) >> 32) as u32;
    for (n, w) in fits {
        if at < w {
            return Some((n, n - gone));
        }
        at -= w;
    }
    None
}

fn observe_both(game: &Game, f: &mut [f32], i: &mut [i16], mask: &mut [u8]) {
    let (f0, f1) = f.split_at_mut(OBS_F);
    let (i0, i1) = i.split_at_mut(OBS_I);
    let (m0, m1) = mask.split_at_mut(OBS_M);
    game.observe(0, f0, i0, m0);
    game.observe(1, f1, i1, m1);
}

#[cfg(test)]
mod hidden_truth {
    //! `tests/search.rs` checks that a redrawn copy depends on nothing hidden by
    //! redrawing two truths, but builds the second truth with `determinize`
    //! itself, which cannot catch what `determinize` forgets to redraw. Here the
    //! second truth is changed by hand instead, in everything the searching
    //! side has not been shown: countdowns given every other length the game
    //! could have drawn, the other side's HP moved within its bar, its stat
    //! points other than HP and Speed moved, and an unseen Pokémon on its bench
    //! swapped for one it did not bring. Anything copied rather than drawn
    //! again would show.

    use super::*;
    use crate::format::Format;

    /// Gives every running countdown on both sides another length the game
    /// could have drawn, keeping how long it has gone on. Returns how many it changed.
    fn other_lengths(g: &mut Game) -> usize {
        let b = g.battle.as_mut().unwrap();
        let mut changed = 0;
        let other = |start: u8, left: u8, lengths: &[u8]| -> Option<(u8, u8)> {
            if start == 0 || left == 0 || left > start {
                return None;
            }
            let gone = start - left;
            lengths.iter().find(|&&n| n > gone && n != start).map(|&n| (n, n - gone))
        };
        for side in 0..2 {
            for i in 0..b.sides[side].n as usize {
                let r = MonRef { side: side as u8, idx: i as u8 };
                let m = b.mon(r);
                let (drawn, active) = (m.drawn, m.is_active);
                if m.status == Status::Slp
                    && let Some((s, l)) = other(drawn[SLEEP_CLOCK], m.status_time, &[2, 3])
                {
                    let m = b.mon_mut(r);
                    (m.drawn[SLEEP_CLOCK], m.status_time) = (s, l);
                    changed += 1;
                }
                if !active {
                    continue;
                }
                if let Some(v) = b.vol_mut(r, VolKind::Confusion)
                    && let Some((s, l)) = other(drawn[CONFUSION_CLOCK], v.data as u8, &[2, 3, 4, 5])
                {
                    v.data = l as u16;
                    b.mon_mut(r).drawn[CONFUSION_CLOCK] = s;
                    changed += 1;
                }
                if let Some(v) = b.vol_mut(r, VolKind::Lockedmove)
                    && let Some((s, l)) = other(drawn[RAMPAGE_CLOCK], v.st.a as u8, &[2, 3])
                {
                    v.st.a = l as i16;
                    b.mon_mut(r).drawn[RAMPAGE_CLOCK] = s;
                    changed += 1;
                }
                if let Some(v) = b.vol_mut(r, VolKind::Partiallytrapped)
                    && let Some((s, l)) = other(drawn[BINDING_CLOCK], v.duration, &[5, 6])
                {
                    v.duration = l;
                    b.mon_mut(r).drawn[BINDING_CLOCK] = s;
                    changed += 1;
                }
            }
        }
        changed
    }

    /// Changes what `view` has not been shown of the other side, by hand. Returns how many things it changed.
    fn other_truth(g: &mut Game, view: usize, rng: &mut Rng) -> usize {
        let opp = 1 - view;
        let mut changed = other_lengths(g);
        let rosters = g.rosters.clone();
        let b = g.battle.as_mut().unwrap();
        let side = &b.sides[opp];
        let unseen = (0..side.n as usize).find(|&i| side.shown[i].seen == 0 && !side.team[i].is_active);
        let unbrought = (0..side.n_roster as usize).find(|&j| side.roster[j].brought == NOT_LISTED);
        let illusion = rosters[opp].iter().any(|s| s.ability == ab::ILLUSION);
        if let (Some(i), Some(j), false) = (unseen, unbrought, illusion) {
            let position = b.sides[opp].team[i].position as usize;
            let old = b.sides[opp].roster.iter().position(|l| l.brought == i as u8).unwrap();
            b.sides[opp].roster[old].brought = NOT_LISTED;
            b.sides[opp].team[i] = b.new_mon(&rosters[opp][j], position).unwrap();
            b.sides[opp].roster[j].brought = i as u8;
            changed += 1;
        }
        let brought: Vec<Option<usize>> = (0..b.sides[opp].n as usize)
            .map(|i| b.sides[opp].roster.iter().position(|l| l.brought == i as u8))
            .collect();
        for (i, m) in b.sides[opp].team[..b.sides[opp].n as usize].iter_mut().enumerate() {
            // Transformed, its stats other than HP are the ones it copied: its own points change nothing shown.
            let own_stats = !m.transformed;
            // Speed: any amount the order of moves still allows.
            if let Some(j) = brought[i].filter(|_| own_stats) {
                let set = &rosters[opp][j];
                let belief = g.speeds.belief(opp, j);
                let room = 66 - m.stat_points.iter().map(|&p| p as u32).sum::<u32>() + m.stat_points[5] as u32;
                let options: Vec<u8> =
                    (0..=32u8).filter(|&p| p as u32 <= room && belief.has(set.nature, p, set.item)).collect();
                if !options.is_empty() {
                    let p = options[rng.below(options.len() as u32) as usize];
                    changed += (p != m.stat_points[5]) as usize;
                    m.stat_points[5] = p;
                    m.stats[5] = calc_stats(m.species, m.nature, m.stat_points)[5];
                }
            }
            // Attack and the defences: shown by nothing but damage, which is not kept.
            let mut points = m.stat_points;
            let spare = 66 - points.iter().map(|&p| p as u32).sum::<u32>()
                + (points[1] + points[2] + points[3] + points[4]) as u32;
            for k in 1..=4 {
                points[k] = 0;
            }
            let mut left = spare;
            for k in [4, 3, 2, 1] {
                points[k] = (rng.below(33).min(left)) as u8;
                left -= points[k] as u32;
            }
            if points != m.stat_points {
                if own_stats {
                    let stats = calc_stats(m.species, m.nature, points);
                    m.stats[1..5].copy_from_slice(&stats[1..5]);
                }
                m.stat_points = points;
                changed += 1;
            }
            // HP: anywhere the bar shows the same.
            if m.hp > 0 {
                let shown = crate::shown::hp_shown(m.hp, m.stats[0]);
                let fits: Vec<u16> =
                    (1..=m.stats[0]).filter(|&h| crate::shown::hp_shown(h, m.stats[0]) == shown).collect();
                let h = fits[rng.below(fits.len() as u32) as usize];
                changed += (h != m.hp) as usize;
                m.hp = h;
            }
        }
        // A copy of a team-mate has that team-mate's stats, whatever they now are.
        for i in 0..b.sides[opp].n as usize {
            let m = b.sides[opp].team[i];
            if let Some(from) = m.copied_from.filter(|r| m.transformed && r.side as usize == opp) {
                let copied = b.sides[opp].team[from.idx as usize].stats;
                b.sides[opp].team[i].stats[1..].copy_from_slice(&copied[1..]);
            }
        }
        changed
    }

    /// Redraws `game` as `view` would, from `game` and from another truth made by
    /// hand, with the same random numbers; the two must come out the same.
    fn same_either_way(game: &Game, view: usize, rng: &mut Rng) -> usize {
        let mut other = game.clone();
        let changed = other_truth(&mut other, view, rng);
        let same = rng.next_u32() as u64;
        let redraw = |g: &Game| {
            let mut copy = g.clone();
            copy.determinize(view, &mut Rng { seed: same, calls: 0 }).unwrap();
            copy.battle.unwrap()
        };
        let (a, b) = (redraw(game), redraw(&other));
        assert_eq!(a.sides, b.sides, "turn {}, side {view}", a.turn);
        assert!(a.vols == b.vols, "turn {}, side {view}: volatiles", a.turn);
        changed
    }

    fn set(species: &str, moves: &[&str], ability: &str) -> PokemonSet {
        PokemonSet::from_names(species, moves, "Bold", [32, 0, 32, 0, 2, 0]).unwrap().ability(ability).unwrap()
    }

    /// A transformed Pokémon keeps its own HP: that must be drawn again within the bar
    /// like anyone's. One that turned into a team-mate copied the team-mate's hidden
    /// stats, so it must take the team-mate's redrawn ones.
    #[test]
    fn transformed_pokemon_are_redrawn_too() {
        let format = Format::current();
        let (mut into_foe, mut into_ally) = (0, 0);
        for seed in 0..400u16 {
            let mut rng = Rng::from_words([seed, 6, 6, 6]);
            let imposter = seed % 2 == 0;
            let mut theirs = format.random_team(&mut rng);
            theirs[0] = if imposter {
                set("Ditto", &["Transform"], "Imposter")
            } else {
                set("Ditto", &["Transform"], "Limber")
            };
            let ours = format.random_team(&mut rng);
            let mut game = Game::new([ours, theirs], true).unwrap();
            game.start([&[0, 1, 2, 3], &[0, 1, 2, 3]], [seed, 1, 1, 1]).unwrap();
            for _ in 0..4 {
                if game.ended() {
                    break;
                }
                let mut actions =
                    [game.baseline(0, Baseline::Random, &mut rng), game.baseline(1, Baseline::Random, &mut rng)];
                // Ditto's Transform at its own team-mate (target 4: own second position), when it may.
                if !imposter && game.legal(1).iter().any(|a| a[0] == 8) {
                    actions[1] = *game.legal(1).iter().find(|a| a[0] == 8).unwrap();
                }
                game.act(actions, [seed, 2, 2, 2]).unwrap();
                let ditto = game.battle.as_ref().unwrap().sides[1].team[0];
                if !ditto.transformed || ditto.hp == 0 {
                    continue;
                }
                match ditto.copied_from {
                    Some(r) if r.side == 1 => into_ally += 1,
                    Some(_) => into_foe += 1,
                    None => panic!("a transformed Pokémon that does not know whom it copied"),
                }
                same_either_way(&game, 0, &mut rng);
                // And the copy carries the stats its team-mate was given.
                let mut copy = game.clone();
                copy.determinize(0, &mut rng).unwrap();
                let side = &copy.battle.as_ref().unwrap().sides[1];
                if let Some(r) = side.team[0].copied_from.filter(|r| r.side == 1) {
                    assert_eq!(side.team[0].stats[1..], side.team[r.idx as usize].stats[1..]);
                }
            }
        }
        println!("{into_foe} positions transformed into a foe, {into_ally} into a team-mate");
        assert!(into_foe > 100 && into_ally > 20, "too few transformed positions ({into_foe}, {into_ally})");
    }

    /// The case found by review: an opposing Gholdengo whose guessed spread (Special
    /// Attack and Speed first) leaves too few points for any HP that shows its bar,
    /// such as 20% on a red bar, which only a maximum divisible by 5 can show. Two
    /// truths that differ in nothing but its hidden Special Attack must be redrawn
    /// the same, for every amount of HP points it may really have and every HP.
    #[test]
    fn a_spread_that_cannot_show_the_bar_does_not_fall_back_on_the_truth() {
        let format = Format::current();
        let mut rng = Rng::from_words([8, 8, 8, 8]);
        let mut theirs = format.random_team(&mut rng);
        theirs[0] = PokemonSet::from_names(
            "Gholdengo",
            &["Make It Rain", "Shadow Ball", "Nasty Plot", "Protect"],
            "Timid",
            [0, 0, 0, 32, 0, 32],
        )
        .unwrap()
        .ability("Good as Gold")
        .unwrap();
        let mut game = Game::new([format.random_team(&mut rng), theirs], true).unwrap();
        game.start([&[0, 1, 2, 3], &[0, 1, 2, 3]], [8, 3, 3, 3]).unwrap();
        let (mut checked, mut picky) = (0, 0);
        for speed in 0..=32u8 {
            let mut truth = game.clone();
            let m = &mut truth.battle.as_mut().unwrap().sides[1].team[0];
            m.stat_points = [(34 - speed).min(32), 0, 0, 32, 0, speed];
            m.stats = calc_stats(m.species, m.nature, m.stat_points);
            let max = m.stats[0];
            for hp in 1..=max {
                let shown = crate::shown::hp_shown(hp, max);
                // Only bars some maximum HP near this one cannot show.
                if (max - 3..=max + 3).all(|mx| (1..=mx).any(|h| crate::shown::hp_shown(h, mx) == shown)) {
                    continue;
                }
                picky += 1;
                for _ in 0..40 {
                    let mut a = truth.clone();
                    a.battle.as_mut().unwrap().sides[1].team[0].hp = hp;
                    let mut b = a.clone();
                    let m = &mut b.battle.as_mut().unwrap().sides[1].team[0];
                    m.stat_points[3] = 10;
                    m.stats[3] = calc_stats(m.species, m.nature, m.stat_points)[3];
                    let same = rng.next_u32() as u64;
                    let redraw = |g: &Game| {
                        let mut copy = g.clone();
                        copy.determinize(0, &mut Rng { seed: same, calls: 0 }).unwrap();
                        copy.battle.unwrap().sides
                    };
                    assert_eq!(redraw(&a), redraw(&b), "Gholdengo at {hp}/{max}");
                    checked += 1;
                }
            }
        }
        assert!(picky > 0 && checked > 100, "no bar that rules out a maximum HP ({picky})");
    }

    /// Fitting HP to the bar never needs the truth: for every species, whatever the
    /// guess, some HP points show the bar, and the HP found shows exactly it.
    #[test]
    fn hp_is_always_fitted_to_the_bar_without_the_truth() {
        let mut rng = Rng::from_words([7, 7, 7, 7]);
        let mut checked = 0;
        for species in 0..SPECIES.len() as u16 {
            for _ in 0..40 {
                let nature = (rng.below(6) as u8, rng.below(6) as u8);
                let random_points = |rng: &mut Rng| {
                    let mut p = [0u8; 6];
                    let mut left = 66u32;
                    for k in 0..6 {
                        p[k] = rng.below(33).min(left) as u8;
                        left -= p[k] as u32;
                    }
                    p
                };
                let (truth, guess) = (random_points(&mut rng), random_points(&mut rng));
                let max = calc_stats(species, nature, truth)[0];
                let hp = 1 + rng.below(max as u32) as u16;
                let shown = crate::shown::hp_shown(hp, max);
                let (points, new_hp) = fit_hp(guess, species, nature, Some(shown), rng.next_u32());
                let new_max = calc_stats(species, nature, points)[0];
                assert_eq!(crate::shown::hp_shown(new_hp, new_max), shown, "species {species}");
                assert!(points.iter().all(|&p| p <= 32) && points.iter().map(|&p| p as u32).sum::<u32>() <= 66);
                assert_eq!(points[5], guess[5], "Speed is left as guessed");
                checked += 1;
            }
        }
        assert!(checked > 10_000);
    }

    #[test]
    fn what_is_hidden_is_drawn_again_not_copied() {
        let format = Format::current();
        let (mut positions, mut changed) = (0, 0);
        for seed in 0..600u16 {
            let mut rng = Rng::from_words([seed, 5, 5, 5]);
            let teams = [format.random_team(&mut rng), format.random_team(&mut rng)];
            let mut game = Game::new(teams, true).unwrap();
            while !game.battle.as_ref().is_some_and(|b| b.ended || b.turn > 150) {
                if game.battle.is_some() {
                    for view in 0..2 {
                        let n = same_either_way(&game, view, &mut rng);
                        if n > 0 {
                            changed += n;
                            positions += 1;
                        }
                    }
                }
                let actions =
                    [game.baseline(0, Baseline::Random, &mut rng), game.baseline(1, Baseline::Random, &mut rng)];
                game.act(actions, [seed, 2, 3, 4]).unwrap();
            }
        }
        println!("{positions} positions with {changed} hidden things changed by hand");
        assert!(positions > 1000, "too few positions to tell ({positions})");
    }
}
