//! Script callbacks of moves (`onTry`, `onHit`, `basePowerCallback`, ...),
//! ported from Showdown's `data/moves.ts` with the Champions overrides.
//!
//! Which move has which callback comes from the generated move table
//! (`MoveData::events`); this file holds the bodies, one arm per callback,
//! named in the comment above it together with its parameter list. Inside a
//! callback `self.event` holds what Showdown passes as arguments.

#![allow(clippy::collapsible_if)]

use crate::battle::modify;
use crate::data::*;
use crate::state::*;

/// The entry hazards, in the order Showdown's moves clear them.
const HAZARDS: [SideCond; 4] = [SideCond::Spikes, SideCond::Toxicspikes, SideCond::Stealthrock, SideCond::Stickyweb];
const SCREENS: [SideCond; 3] = [SideCond::Reflect, SideCond::Lightscreen, SideCond::Auroraveil];

impl Battle {
    /// Rapid Spin and Mortal Spin's clean-up of the user's side:
    /// onAfterHit(target, pokemon, move) and onAfterSubDamage(damage, target, pokemon, move).
    fn spin_away(&mut self, mi: u8, pokemon: MonRef, need_hp: bool) {
        if self.am[mi as usize].has_sheer_force || (need_hp && self.mon(pokemon).hp == 0) {
            return;
        }
        if let Some(k) = VolKind::named("leechseed") {
            self.remove_volatile(pokemon, k);
        }
        for cond in HAZARDS {
            self.remove_side_condition(pokemon.side as usize, cond);
        }
        if let Some(k) = VolKind::named("partiallytrapped") {
            self.remove_volatile(pokemon, k);
        }
    }

    /// Runs one script callback of the move in slot `mi`.
    pub(crate) fn move_cb(&mut self, mi: u8, ev: Ev) -> Res {
        let e = self.event;
        let id = self.am[mi as usize].id;
        match (id, ev) {
            // ---- Protect, Detect and the other moves that fail more often when repeated
            // onPrepareHit(pokemon)
            (
                mv::PROTECT | mv::DETECT | mv::ENDURE | mv::BANEFULBUNKER | mv::KINGSSHIELD | mv::SPIKYSHIELD,
                Ev::PrepareHit,
            ) => {
                if !self.will_act() {
                    return FALSE;
                }
                self.run_event(Ev::StallMove, e.target, None, Eff::None, Res::Undef)
            }
            // onHit(pokemon)
            (
                mv::PROTECT | mv::DETECT | mv::ENDURE | mv::BANEFULBUNKER | mv::KINGSSHIELD | mv::SPIKYSHIELD,
                Ev::Hit,
            ) => {
                if let Some(t) = e.target {
                    self.add_volatile(t, VolKind::Stall, None, Eff::None);
                }
                Res::Undef
            }

            // ---- Wide Guard, Quick Guard
            // onTry(): fails if nothing else is left to act this turn.
            (mv::WIDEGUARD | mv::QUICKGUARD, Ev::Try) => Res::Bool(self.will_act()),
            // onHitSide(side, source): counts as a use of Protect for the user's next one.
            (mv::WIDEGUARD | mv::QUICKGUARD, Ev::HitSide) => {
                if let Some(source) = e.source {
                    self.add_volatile(source, VolKind::Stall, None, Eff::None);
                }
                Res::Undef
            }

            // ---- Fake Out, First Impression: only on the user's first turn out.
            // onTry(source)
            (mv::FAKEOUT | mv::FIRSTIMPRESSION, Ev::Try) => {
                if e.target.is_some_and(|source| self.mon(source).active_move_actions > 1) {
                    return FALSE;
                }
                Res::Undef
            }
            // onDisableMove(pokemon)
            (mv::FAKEOUT | mv::FIRSTIMPRESSION, Ev::DisableMove) => {
                if let Some(pokemon) = e.target {
                    if self.mon(pokemon).active_move_actions > 0 {
                        self.disable_slots_where(pokemon, |s| s.id == id);
                    }
                }
                Res::Undef
            }

            // ---- Follow Me, Rage Powder: onTry(source). Fail in singles only.
            (mv::FOLLOWME | mv::RAGEPOWDER, Ev::Try) => TRUE,

            // ---- Helping Hand: onTryHit(target). The ally must still have a move to make.
            (mv::HELPINGHAND, Ev::TryHit) => {
                if e.target.is_some_and(|t| !self.mon(t).newly_switched && !self.will_move(t)) {
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Disable: onTryHit(target). The target must have used a move.
            (mv::DISABLE, Ev::TryHit) => {
                if e.target.is_some_and(|t| matches!(self.mon(t).last_move, NO_MOVE | mv::STRUGGLE)) {
                    return FALSE;
                }
                Res::Undef
            }

            // ============================================ moves that set up a volatile
            // ---- Leech Seed: onTryImmunity(target). Grass types are immune.
            (mv::LEECHSEED, Ev::TryImmunity) => Res::Bool(e.target.is_some_and(|t| !self.has_type(t, Type::Grass))),

            // ---- Magnet Rise: onTry(source, target, move)
            (mv::MAGNETRISE, Ev::Try) => {
                if let Some(target) = e.source {
                    if self.has_vol_named(target, "smackdown") || self.has_vol_named(target, "ingrain") {
                        return FALSE;
                    }
                }
                if self.field.pseudo.has(Pseudo::Gravity) {
                    return Res::Null;
                }
                Res::Undef
            }

            // ---- No Retreat: onTry(source, target, move). Fails if already used;
            // a Pokémon trapped by something else only gets the boosts.
            (mv::NORETREAT, Ev::Try) => {
                let Some(source) = e.target else {
                    return Res::Undef;
                };
                if self.vols(source).has(VolKind::Noretreat) {
                    return FALSE;
                }
                if self.vols(source).has(VolKind::Trapped) {
                    self.am[mi as usize].no_volatile = true;
                }
                Res::Undef
            }

            // ---- Octolock: onTryImmunity(target). Ghosts cannot be trapped.
            (mv::OCTOLOCK, Ev::TryImmunity) => Res::Bool(e.target.is_some_and(|t| self.type_allows(t, 6))),

            // ---- Electrify: onTryHit(target). The target must still have a move to make.
            (mv::ELECTRIFY, Ev::TryHit) => {
                if e.target.is_some_and(|t| !self.will_move(t) && self.mon(t).active_turns > 0) {
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Gastro Acid: onTryHit(target). Some abilities cannot be suppressed.
            (mv::GASTROACID, Ev::TryHit) => {
                if e.target.is_some_and(|t| ABILITIES[self.mon(t).ability as usize].flags & AF_CANTSUPPRESS != 0) {
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Lock-On
            // onTryHit(target, source)
            (mv::LOCKON, Ev::TryHit) => {
                if e.source.is_some_and(|s| self.vols(s).has(VolKind::Lockon)) {
                    return FALSE;
                }
                Res::Undef
            }
            // onHit(target, source)
            (mv::LOCKON, Ev::Hit) => {
                if let (Some(target), Some(source)) = (e.target, e.source) {
                    self.add_volatile(source, VolKind::Lockon, Some(target), Eff::None);
                }
                Res::Undef
            }

            // ---- Perish Song: onHitField(target, source, move). Everyone who hears it.
            (mv::PERISHSONG, Ev::HitField) => {
                let me = Eff::Move(mi);
                let mut result = false;
                let (actives, n) = self.all_active(false);
                for &pokemon in &actives[..n] {
                    // Out of reach, or deaf to it (Soundproof answers `null`).
                    let unreachable = self.run_event(Ev::Invulnerability, Some(pokemon), e.source, me, Res::Undef)
                        == FALSE
                        || self.run_event(Ev::TryHit, Some(pokemon), e.source, me, Res::Undef) == Res::Null;
                    if unreachable {
                        result = true;
                    } else if !self.vols(pokemon).has(VolKind::Perishsong) {
                        self.add_volatile(pokemon, VolKind::Perishsong, None, Eff::None);
                        result = true;
                    }
                }
                if !result {
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Yawn: onTryHit(target). Not on something that cannot fall asleep.
            (mv::YAWN, Ev::TryHit) => {
                let Some(target) = e.target else {
                    return Res::Undef;
                };
                if self.mon(target).status != Status::None
                    || !self.run_status_immunity(target, Imm::Status(Status::Slp))
                {
                    return FALSE;
                }
                Res::Undef
            }

            // ---- Stockpile, Spit Up, Swallow
            // onTry(source): three layers at most.
            (mv::STOCKPILE, Ev::Try) => {
                if e.target.is_some_and(|s| self.vols(s).get(VolKind::Stockpile).is_some_and(|v| v.data >= 3)) {
                    return FALSE;
                }
                Res::Undef
            }
            // onTry(source): needs something stockpiled.
            (mv::SPITUP | mv::SWALLOW, Ev::Try) => {
                Res::Bool(e.target.is_some_and(|s| self.vols(s).has(VolKind::Stockpile)))
            }
            // basePowerCallback(pokemon)
            (mv::SPITUP, Ev::BasePowerCallback) => {
                match e.target.and_then(|p| self.vols(p).get(VolKind::Stockpile)).map(|v| v.data) {
                    Some(layers) if layers > 0 => Res::Num(layers as i32 * 100),
                    _ => FALSE,
                }
            }
            // onAfterMove(pokemon)
            (mv::SPITUP, Ev::AfterMove) => {
                if let Some(pokemon) = e.target {
                    self.remove_volatile(pokemon, VolKind::Stockpile);
                }
                Res::Undef
            }
            // onHit(pokemon): a quarter, half or all of its HP.
            (mv::SWALLOW, Ev::Hit) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                let layers = self.vols(pokemon).get(VolKind::Stockpile).map_or(1, |v| v.data.clamp(1, 3));
                let part = [1024, 2048, 4096][layers as usize - 1];
                let amount = crate::battle::modify(self.mon(pokemon).max_hp() as u32, part);
                let success = self.heal(amount as i32, None, None, Eff::None).truthy();
                self.remove_volatile(pokemon, VolKind::Stockpile);
                if success { TRUE } else { Res::NotFail }
            }

            // ---- Destiny Bond: onPrepareHit(pokemon). Fails if it is still in effect from last time.
            (mv::DESTINYBOND, Ev::PrepareHit) => {
                Res::Bool(e.target.is_some_and(|p| !self.remove_volatile(p, VolKind::Destinybond)))
            }

            // ---- Sparkling Aria: onAfterMove(source, target, move). Cures the burns of
            // those it hit; its secondary only marks them.
            (mv::SPARKLINGARIA, Ev::AfterMove) => {
                let Some(source) = e.target else {
                    return Res::Undef;
                };
                let am = self.am[mi as usize];
                if self.mon(source).fainted || !am.has_hit_targets || am.has_sheer_force {
                    let (actives, n) = self.all_active(true);
                    for &pokemon in &actives[..n] {
                        self.drop_vol(pokemon, VolKind::Sparklingaria);
                    }
                    return Res::Undef;
                }
                let n = am.n_hit_targets as usize;
                for &pokemon in &am.hit_targets[..n] {
                    if pokemon != source
                        && self.mon(pokemon).is_active
                        && (self.remove_volatile(pokemon, VolKind::Sparklingaria) || n > 1)
                        && self.mon(pokemon).status == Status::Brn
                    {
                        self.cure_status(pokemon);
                    }
                }
                Res::Undef
            }

            // ---- Block, Mean Look: onHit(target, source, move)
            (mv::BLOCK | mv::MEANLOOK, Ev::Hit) => match (e.target, e.source) {
                (Some(target), Some(source)) => self.add_trapped(target, source, Eff::Move(mi)),
                _ => Res::Undef,
            },
            // ---- Jaw Lock: onHit(target, source, move). Both are trapped.
            (mv::JAWLOCK, Ev::Hit) => {
                if let (Some(target), Some(source)) = (e.target, e.source) {
                    self.add_trapped(source, target, Eff::Move(mi));
                    self.add_trapped(target, source, Eff::Move(mi));
                }
                Res::Undef
            }
            // ---- Spirit Shackle: its secondary's onHit(target, source, move)
            (mv::SPIRITSHACKLE, Ev::SecondaryHit) => {
                if let (Some(target), Some(source)) = (e.target, e.source) {
                    if self.mon(source).is_active {
                        self.add_trapped(target, source, Eff::Move(mi));
                    }
                }
                Res::Undef
            }
            // ---- Throat Chop: its secondary's onHit(target)
            (mv::THROATCHOP, Ev::SecondaryHit) => {
                if let Some(target) = e.target {
                    self.add_volatile(target, VolKind::Throatchop, None, Eff::None);
                }
                Res::Undef
            }

            // ---- Substitute
            // onTryHit(source): fails with a substitute already up, or without the HP to make one.
            (mv::SUBSTITUTE, Ev::TryHit) => {
                let Some(source) = e.target else {
                    return Res::Undef;
                };
                let m = self.mon(source);
                if self.vols(source).has(VolKind::Substitute) || m.hp as u32 * 4 <= m.max_hp() as u32 || m.max_hp() == 1
                {
                    return Res::NotFail;
                }
                Res::Undef
            }
            // onHit(target): the substitute costs a quarter of the user's HP.
            (mv::SUBSTITUTE, Ev::Hit) => {
                if let Some(target) = e.target {
                    let cost = self.mon(target).max_hp() as i32 / 4;
                    self.direct_damage(cost, target, e.source, Eff::Move(mi));
                }
                Res::Undef
            }

            // ---- Attract: onTryImmunity(target, source)
            (mv::ATTRACT, Ev::TryImmunity) => match (e.target, e.source) {
                (Some(t), Some(s)) => {
                    Res::Bool(crate::conditions::opposite_genders(self.mon(t).gender, self.mon(s).gender))
                }
                _ => FALSE,
            },

            // ---- Aurora Veil: onTry(). Only in snow.
            (mv::AURORAVEIL, Ev::Try) => Res::Bool(self.is_weather(Weather::Snowscape)),

            // ============================================ weather-dependent moves
            // ---- Blizzard: onModifyMove(move). Never misses in snow.
            (mv::BLIZZARD, Ev::ModifyMove) => {
                if self.is_weather(Weather::Snowscape) {
                    self.am[mi as usize].accuracy = 0;
                }
                Res::Undef
            }
            // ---- Thunder, Hurricane: onModifyMove(move, pokemon, target)
            (mv::THUNDER | mv::HURRICANE, Ev::ModifyMove) => {
                match e.source.map(|t| self.effective_weather(t)) {
                    Some(Weather::Raindance) => self.am[mi as usize].accuracy = 0,
                    Some(Weather::Sunnyday) => self.am[mi as usize].accuracy = 50,
                    _ => {}
                }
                Res::Undef
            }
            // ---- Weather Ball
            // onModifyType(move, pokemon)
            (mv::WEATHERBALL, Ev::ModifyType) => {
                if let Some(pokemon) = e.target {
                    match self.effective_weather(pokemon) {
                        Weather::Sunnyday => self.am[mi as usize].typ = Type::Fire,
                        Weather::Raindance => self.am[mi as usize].typ = Type::Water,
                        Weather::Sandstorm => self.am[mi as usize].typ = Type::Rock,
                        Weather::Snowscape => self.am[mi as usize].typ = Type::Ice,
                        Weather::None => {}
                    }
                }
                Res::Undef
            }
            // onModifyMove(move, pokemon)
            (mv::WEATHERBALL, Ev::ModifyMove) => {
                if e.target.is_some_and(|p| self.effective_weather(p) != Weather::None) {
                    self.am[mi as usize].base_power *= 2;
                }
                Res::Undef
            }
            // ---- Growth: onModifyMove(move, pokemon). +2/+2 in the sun.
            (mv::GROWTH, Ev::ModifyMove) => {
                if e.target.is_some_and(|p| self.effective_weather(p) == Weather::Sunnyday) {
                    let mut b = [0i8; 7];
                    b[ATK] = 2;
                    b[SPA] = 2;
                    self.am[mi as usize].boosts = Some(b);
                }
                Res::Undef
            }
            // ---- Moonlight, Morning Sun, Synthesis: onHit(pokemon)
            (mv::MOONLIGHT | mv::MORNINGSUN | mv::SYNTHESIS, Ev::Hit) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                // 0.667, 0.25 and 0.5 as Showdown's `modify` sees them.
                let factor = match self.effective_weather(pokemon) {
                    Weather::Sunnyday => 2732,
                    Weather::None => 2048,
                    _ => 1024,
                };
                let amount = modify(self.mon(pokemon).max_hp() as u32, factor);
                if self.heal(amount as i32, None, None, Eff::None).truthy() { TRUE } else { Res::NotFail }
            }

            // ============================================ terrain-dependent moves
            // ---- Expanding Force
            // onBasePower(basePower, source)
            (mv::EXPANDINGFORCE, Ev::BasePower) => {
                if self.is_terrain(Terrain::Psychicterrain) && e.target.is_some_and(|s| self.is_grounded(s, false)) {
                    return self.chain_modify(3, 2);
                }
                Res::Undef
            }
            // onModifyMove(move, source, target): it becomes a spread move.
            (mv::EXPANDINGFORCE, Ev::ModifyMove) => {
                if self.is_terrain(Terrain::Psychicterrain) && e.target.is_some_and(|s| self.is_grounded(s, false)) {
                    self.am[mi as usize].target = Target::AllAdjacentFoes;
                }
                Res::Undef
            }
            // ---- Rising Voltage: basePowerCallback(source, target, move)
            (mv::RISINGVOLTAGE, Ev::BasePowerCallback) => {
                let bp = self.am[mi as usize].base_power as i32;
                if self.is_terrain(Terrain::Electricterrain) && e.source.is_some_and(|t| self.is_grounded(t, false)) {
                    return Res::Num(bp * 2);
                }
                Res::Num(bp)
            }
            // ---- Grassy Glide: onModifyPriority(priority, source, target, move)
            (mv::GRASSYGLIDE, Ev::ModifyPriority) => {
                if self.is_terrain(Terrain::Grassyterrain) && e.target.is_some_and(|s| self.is_grounded(s, false)) {
                    return Res::Num(e.relay.num() + 1);
                }
                Res::Undef
            }
            // ---- Terrain Pulse
            // onModifyType(move, pokemon)
            (mv::TERRAINPULSE, Ev::ModifyType) => {
                if !e.target.is_some_and(|p| self.is_grounded(p, false)) {
                    return Res::Undef;
                }
                match self.field.terrain.kind {
                    Terrain::Electricterrain => self.am[mi as usize].typ = Type::Electric,
                    Terrain::Grassyterrain => self.am[mi as usize].typ = Type::Grass,
                    Terrain::Mistyterrain => self.am[mi as usize].typ = Type::Fairy,
                    Terrain::Psychicterrain => self.am[mi as usize].typ = Type::Psychic,
                    Terrain::None => {}
                }
                Res::Undef
            }
            // onModifyMove(move, pokemon)
            (mv::TERRAINPULSE, Ev::ModifyMove) => {
                if self.field.terrain.kind != Terrain::None && e.target.is_some_and(|p| self.is_grounded(p, false)) {
                    self.am[mi as usize].base_power *= 2;
                }
                Res::Undef
            }
            // ---- Steel Roller: onTry() fails without a terrain; onHit() and onAfterSubDamage() remove it.
            (mv::STEELROLLER, Ev::Try) => Res::Bool(self.field.terrain.kind != Terrain::None),
            (mv::STEELROLLER, Ev::Hit | Ev::AfterSubDamage) => {
                self.clear_terrain();
                Res::Undef
            }
            // ---- Ice Spinner
            // onAfterHit(target, source)
            (mv::ICESPINNER, Ev::AfterHit) => {
                self.clear_terrain();
                Res::Undef
            }
            // onAfterSubDamage(damage, target, source)
            (mv::ICESPINNER, Ev::AfterSubDamage) => {
                if e.source.is_some_and(|s| self.mon(s).hp > 0) {
                    self.clear_terrain();
                }
                Res::Undef
            }

            // ============================================ screens and hazards
            // ---- Brick Break, Psychic Fangs: onTryHit(pokemon). The screens go before the hit.
            (mv::BRICKBREAK | mv::PSYCHICFANGS, Ev::TryHit) => {
                if let Some(target) = e.target {
                    for cond in SCREENS {
                        self.remove_side_condition(target.side as usize, cond);
                    }
                }
                Res::Undef
            }
            // ---- Defog: onHit(target, source, move)
            (mv::DEFOG, Ev::Hit) => {
                let (Some(target), Some(source)) = (e.target, e.source) else {
                    return Res::Undef;
                };
                let mut success = false;
                if !self.has_vol_named(target, "substitute") || self.am[mi as usize].infiltrates {
                    success = self.boost1(EVA, -1, None, None, Eff::None).truthy();
                }
                // The target's side loses its screens and Safeguard too; only a hazard
                // counts as the move having done something.
                for cond in SCREENS.into_iter().chain([SideCond::Safeguard]) {
                    self.remove_side_condition(target.side as usize, cond);
                }
                for side in [target.side as usize, source.side as usize] {
                    for cond in HAZARDS {
                        if self.remove_side_condition(side, cond) {
                            success = true;
                        }
                    }
                }
                self.clear_terrain();
                Res::Bool(success)
            }
            // ---- Rapid Spin, Mortal Spin
            // onAfterHit(target, pokemon, move)
            (mv::RAPIDSPIN | mv::MORTALSPIN, Ev::AfterHit) => {
                if let Some(pokemon) = e.source {
                    self.spin_away(mi, pokemon, false);
                }
                Res::Undef
            }
            // onAfterSubDamage(damage, target, pokemon, move)
            (mv::RAPIDSPIN | mv::MORTALSPIN, Ev::AfterSubDamage) => {
                if let Some(pokemon) = e.source {
                    self.spin_away(mi, pokemon, true);
                }
                Res::Undef
            }
            // ---- Tidy Up: onHit(pokemon)
            (mv::TIDYUP, Ev::Hit) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                let mut success = false;
                if let Some(k) = VolKind::named("substitute") {
                    let (actives, n) = self.all_active(false);
                    for &active in &actives[..n] {
                        if self.remove_volatile(active, k) {
                            success = true;
                        }
                    }
                }
                for side in [pokemon.side as usize, 1 - pokemon.side as usize] {
                    for cond in HAZARDS {
                        if self.remove_side_condition(side, cond) {
                            success = true;
                        }
                    }
                }
                let mut b = [0i8; 7];
                b[ATK] = 1;
                b[SPE] = 1;
                let boosted = self.boost(b, Some(pokemon), Some(pokemon), Eff::None).truthy();
                Res::Bool(boosted || success)
            }
            // ---- Court Change: onHitField(target, source). The two sides trade conditions.
            (mv::COURTCHANGE, Ev::HitField) => {
                let Some(source) = e.source else {
                    return Res::Undef;
                };
                // Everything a side can have except the one-turn guards.
                let swaps = |k: SideCond| {
                    matches!(
                        k,
                        SideCond::Tailwind
                            | SideCond::Reflect
                            | SideCond::Lightscreen
                            | SideCond::Auroraveil
                            | SideCond::Safeguard
                            | SideCond::Spikes
                            | SideCond::Toxicspikes
                            | SideCond::Stealthrock
                            | SideCond::Stickyweb
                    )
                };
                let own = source.side as usize;
                let from_own = self.sides[own].conds.take_where(swaps);
                let from_foe = self.sides[1 - own].conds.take_where(swaps);
                if from_own.as_slice().is_empty() && from_foe.as_slice().is_empty() {
                    return FALSE;
                }
                for c in from_own.as_slice() {
                    self.sides[1 - own].conds.push(*c);
                }
                for c in from_foe.as_slice() {
                    self.sides[own].conds.push(*c);
                }
                Res::Undef
            }
            // ---- Magnetic Flux: onHitSide(side, source, move)
            (mv::MAGNETICFLUX, Ev::HitSide) => {
                let Some(side) = e.target_side else {
                    return Res::Undef;
                };
                let (allies, n) = self.allies_and_self(side as usize);
                let mut any_target = false;
                let mut did = false;
                let mut b = [0i8; 7];
                b[DEF] = 1;
                b[SPD] = 1;
                for &ally in &allies[..n] {
                    if self.has_ability(ally, ab::PLUS) || self.has_ability(ally, ab::MINUS) {
                        any_target = true;
                        did = self.boost(b, Some(ally), e.source, Eff::Move(mi)).truthy() || did;
                    }
                }
                if !any_target {
                    return FALSE;
                }
                Res::Bool(did)
            }
            // ---- Haze: onHitField()
            (mv::HAZE, Ev::HitField) => {
                let (actives, n) = self.all_active(false);
                for &pokemon in &actives[..n] {
                    self.mon_mut(pokemon).boosts = [0; 7];
                }
                Res::Undef
            }

            // ---- Struggle: onModifyMove(move, pokemon, target)
            (mv::STRUGGLE, Ev::ModifyMove) => {
                self.am[mi as usize].typ = Type::Typeless;
                Res::Undef
            }

            _ => unreachable!("no body for move {} {ev:?}", MOVES[id as usize].id),
        }
    }
}
