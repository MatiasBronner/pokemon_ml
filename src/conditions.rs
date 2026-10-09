//! Event callbacks of status conditions and volatile conditions, ported from
//! Showdown's `data/conditions.ts` (with the Champions overrides) and from the
//! `condition` blocks of moves.

// The nesting mirrors Showdown's code; keep it rather than folding conditions together.
#![allow(clippy::collapsible_if, clippy::collapsible_match)]

use crate::battle::div1;
use crate::data::*;
use crate::state::*;

impl Battle {
    /// Callbacks of the major status conditions. `holder` has the status.
    pub(crate) fn status_cb(&mut self, status: Status, ev: Ev, holder: MonRef) -> Res {
        match (status, ev) {
            // Showdown's onStart for brn, par and psn only writes to the log.
            (Status::Brn | Status::Par | Status::Psn, Ev::Start) => Res::Undef,

            (Status::Brn, Ev::Residual) => {
                let d = div1(self.mon(holder).max_hp() as u32, 16);
                self.damage(d, None, None, Eff::None);
                Res::Undef
            }
            (Status::Psn, Ev::Residual) => {
                let d = div1(self.mon(holder).max_hp() as u32, 8);
                self.damage(d, None, None, Eff::None);
                Res::Undef
            }

            (Status::Tox, Ev::Start | Ev::SwitchIn) => {
                self.mon_mut(holder).tox_stage = 0;
                Res::Undef
            }
            (Status::Tox, Ev::Residual) => {
                let m = self.mon_mut(holder);
                if m.tox_stage < 15 {
                    m.tox_stage += 1;
                }
                let d = (m.max_hp() as i32 / 16).max(1) * m.tox_stage as i32;
                self.damage(d, None, None, Eff::None);
                Res::Undef
            }

            // Paralysis halves Speed after every other modifier.
            (Status::Par, Ev::ModifySpe) => {
                let mut spe = self.final_modify(self.event.relay.num() as u32);
                if !self.has_ability(holder, ab::QUICKFEET) {
                    spe = spe * 50 / 100;
                }
                Res::Num(spe as i32)
            }
            // Champions: full paralysis is 1 in 8.
            (Status::Par, Ev::BeforeMove) => {
                if self.chance(1, 8, "full paralysis") {
                    return FALSE;
                }
                Res::Undef
            }

            // Champions: asleep for 2 turns one time in three, otherwise 3.
            (Status::Slp, Ev::Start) => {
                let t = [2u8, 3, 3][self.rand(3, "sleep turns") as usize];
                self.mon_mut(holder).status_time = t;
                Res::Undef
            }
            (Status::Slp, Ev::BeforeMove) => {
                let early = self.has_ability(holder, ab::EARLYBIRD);
                let m = self.mon_mut(holder);
                if early {
                    m.status_time = m.status_time.saturating_sub(1);
                }
                m.status_time = m.status_time.saturating_sub(1);
                if m.status_time == 0 {
                    self.cure_status(holder);
                    return Res::Undef;
                }
                FALSE
            }

            // Champions: frozen for at most 3 turns.
            (Status::Frz, Ev::Start) => {
                self.mon_mut(holder).status_time = 3;
                Res::Undef
            }
            (Status::Frz, Ev::BeforeMove) => {
                if self.event_move_flags() & F_DEFROST != 0 {
                    return Res::Undef;
                }
                let m = self.mon_mut(holder);
                m.status_time = m.status_time.saturating_sub(1);
                // The thaw roll is skipped once the timer has run out.
                if m.status_time == 0 || self.chance(1, 4, "thaw") {
                    self.cure_status(holder);
                    return Res::Undef;
                }
                FALSE
            }
            // Using a defrosting move thaws the user.
            (Status::Frz, Ev::ModifyMove) => {
                if self.event_move_flags() & F_DEFROST != 0 {
                    self.cure_status(holder);
                }
                Res::Undef
            }
            (Status::Frz, Ev::AfterMoveSecondary) => {
                if let (Eff::Move(mi), Some(t)) = (self.event.effect, self.event.target) {
                    if self.am[mi as usize].d().thaws_target {
                        self.cure_status(t);
                    }
                }
                Res::Undef
            }
            // A damaging Fire move thaws its target.
            (Status::Frz, Ev::DamagingHit) => {
                if let (Eff::Move(mi), Some(t)) = (self.event.effect, self.event.target) {
                    let am = &self.am[mi as usize];
                    if am.typ == Type::Fire && am.category != Category::Status {
                        self.cure_status(t);
                    }
                }
                Res::Undef
            }
            _ => unreachable!("no body for {status:?} {ev:?}"),
        }
    }

    /// Flags of the move the running event is about (0 if it is not about a move).
    pub(crate) fn event_move_flags(&self) -> u32 {
        match self.event.effect {
            Eff::Move(mi) => self.am[mi as usize].flags,
            _ => 0,
        }
    }

    /// Callbacks of volatile conditions. `holder` has the volatile.
    pub(crate) fn vol_cb(&mut self, kind: VolKind, ev: Ev, pre: Pre, holder: MonRef) -> Res {
        debug_assert!(pre == Pre::On, "no body for {kind:?} {ev:?} {pre:?}");
        match (kind, ev) {
            (VolKind::Flinch, Ev::BeforeMove) => {
                self.run_event(Ev::Flinch, Some(holder), None, Eff::None, Res::Undef);
                FALSE
            }

            (VolKind::Protect, Ev::Start) => Res::Undef,
            (VolKind::Protect, Ev::TryHit) => {
                let (Eff::Move(mi), Some(source)) = (self.event.effect, self.event.source) else {
                    return Res::Undef;
                };
                let target = self.event.target.unwrap_or(holder);
                if self.bypasses_protect(mi, source, target, true) {
                    return Res::Undef;
                }
                Res::NotFail
            }

            // The consecutive-Protect counter.
            (VolKind::Stall, Ev::Start) => {
                if let Some(v) = self.vol_mut(holder, VolKind::Stall) {
                    v.data = 3;
                }
                Res::Undef
            }
            (VolKind::Stall, Ev::StallMove) => {
                let counter = self.vols(holder).get(VolKind::Stall).map_or(1, |v| v.data.max(1)) as u32;
                let success = self.chance(1, counter, "consecutive protect");
                if !success {
                    self.drop_vol(holder, VolKind::Stall);
                }
                Res::Bool(success)
            }
            (VolKind::Stall, Ev::Restart) => {
                if let Some(v) = self.vol_mut(holder, VolKind::Stall) {
                    if v.data < 729 {
                        v.data *= 3;
                    }
                    v.duration = 2;
                }
                Res::Undef
            }

            (VolKind::Confusion, Ev::Start) => {
                // Lasts 2 to 5 of the holder's move attempts.
                let t = self.rand_range(2, 6, "confusion turns");
                if let Some(v) = self.vol_mut(holder, VolKind::Confusion) {
                    v.data = t as u16;
                }
                Res::Undef
            }
            (VolKind::Confusion, Ev::End) => Res::Undef,
            (VolKind::Confusion, Ev::BeforeMove) => {
                let Some(v) = self.vol_mut(holder, VolKind::Confusion) else {
                    return Res::Undef;
                };
                v.data = v.data.saturating_sub(1);
                if v.data == 0 {
                    self.remove_volatile(holder, VolKind::Confusion);
                    return Res::Undef;
                }
                if !self.chance(33, 100, "confusion self-hit") {
                    return Res::Undef;
                }
                self.active_target = Some(holder);
                let d = self.confusion_damage(holder, 40);
                self.damage(d, Some(holder), Some(holder), Eff::Confused);
                FALSE
            }

            (VolKind::Choicelock, Ev::Start) => {
                let Some(mi) = self.active_move else {
                    return FALSE;
                };
                let am = &self.am[mi as usize];
                if am.has_bounced {
                    return FALSE;
                }
                let id = am.id;
                if let Some(v) = self.vol_mut(holder, VolKind::Choicelock) {
                    v.data = id + 1;
                }
                Res::Undef
            }
            (VolKind::Choicelock, Ev::BeforeMove) => {
                if ITEMS[self.mon(holder).item as usize].flags & IF_CHOICE == 0 {
                    self.remove_volatile(holder, VolKind::Choicelock);
                    return Res::Undef;
                }
                let locked = self.vols(holder).get(VolKind::Choicelock).map_or(0, |v| v.data);
                if let Eff::Move(mi) = self.event.effect {
                    let am = &self.am[mi as usize];
                    if !self.ignoring_item(holder) && am.id + 1 != locked && am.id != mv::STRUGGLE {
                        // Fails, and no PP is lost.
                        return FALSE;
                    }
                }
                Res::Undef
            }
            (VolKind::Choicelock, Ev::DisableMove) => {
                let locked = self.vols(holder).get(VolKind::Choicelock).map_or(0, |v| v.data);
                let m = self.mon(holder);
                let has_move = m.moves[..m.n_moves as usize].iter().any(|s| s.id + 1 == locked);
                if ITEMS[m.item as usize].flags & IF_CHOICE == 0 || !has_move {
                    self.remove_volatile(holder, VolKind::Choicelock);
                    return Res::Undef;
                }
                if self.ignoring_item(holder) {
                    return Res::Undef;
                }
                let m = self.mon_mut(holder);
                for k in 0..m.n_moves as usize {
                    if m.moves[k].id + 1 != locked {
                        m.moves[k].disabled = true;
                    }
                }
                Res::Undef
            }

            (VolKind::Gem, Ev::BasePower) => self.chain_modify(5325, 4096),

            // Flash Fire's boost, once a Fire move has been absorbed.
            (VolKind::Flashfire, Ev::Start | Ev::End) => Res::Undef,
            (VolKind::Flashfire, Ev::ModifyAtk | Ev::ModifySpA) => {
                let fire = matches!(self.event.effect, Eff::Move(mi) if self.am[mi as usize].typ == Type::Fire);
                if fire && self.has_ability(holder, ab::FLASHFIRE) {
                    return self.chain_modify(6144, 4096);
                }
                Res::Undef
            }

            // Unburden's doubled Speed while the holder has no item.
            (VolKind::Unburden, Ev::ModifySpe) => {
                if self.mon(holder).item == it::NONE && !self.ignoring_ability(holder) {
                    return self.chain_modify(2, 1);
                }
                Res::Undef
            }

            // The Metronome item's counter of consecutive uses of one move.
            (VolKind::Metronome, Ev::Start) => {
                if let Some(v) = self.vol_mut(holder, VolKind::Metronome) {
                    v.data = 0;
                    v.st.a = 0;
                }
                Res::Undef
            }
            (VolKind::Metronome, Ev::TryMove) => {
                if !self.has_item(holder, it::METRONOME) {
                    self.remove_volatile(holder, VolKind::Metronome);
                    return Res::Undef;
                }
                let Eff::Move(mi) = self.event.effect else {
                    return Res::Undef;
                };
                let id = self.am[mi as usize].id + 1;
                let last_ok = self.mon(holder).move_last_turn.truthy();
                if let Some(v) = self.vol_mut(holder, VolKind::Metronome) {
                    if v.data == id && last_ok {
                        v.st.a += 1;
                    } else {
                        v.st.a = 0;
                    }
                    v.data = id;
                }
                Res::Undef
            }
            (VolKind::Metronome, Ev::ModifyDamage) => {
                const MODS: [u32; 6] = [4096, 4915, 5734, 6553, 7372, 8192];
                let n = self.vols(holder).get(VolKind::Metronome).map_or(0, |v| v.st.a).clamp(0, 5);
                self.chain_modify(MODS[n as usize], 4096)
            }
            _ => unreachable!("no body for {kind:?} {ev:?}"),
        }
    }

    /// `durationCallback` of the conditions that have one: how long the
    /// condition lasts, given who it is on and who caused it.
    pub(crate) fn cond_duration(
        &mut self,
        eff: Eff,
        _target: Option<MonRef>,
        source: Option<MonRef>,
        _source_effect: Eff,
    ) -> u8 {
        let holds = |b: &Battle, item: u16| source.is_some_and(|s| b.has_item(s, item));
        // (The Persistent ability, which lengthens some of these, is not in Champions.)
        match eff {
            // durationCallback(source, effect): 8 turns with the matching rock.
            Eff::Weather(Weather::Raindance) => {
                if holds(self, it::DAMPROCK) {
                    8
                } else {
                    5
                }
            }
            Eff::Weather(Weather::Sunnyday) => {
                if holds(self, it::HEATROCK) {
                    8
                } else {
                    5
                }
            }
            Eff::Weather(Weather::Sandstorm) => {
                if holds(self, it::SMOOTHROCK) {
                    8
                } else {
                    5
                }
            }
            Eff::Weather(Weather::Snowscape) => {
                if holds(self, it::ICYROCK) {
                    8
                } else {
                    5
                }
            }
            Eff::Terrain(_) => {
                if holds(self, it::TERRAINEXTENDER) {
                    8
                } else {
                    5
                }
            }
            Eff::Pseudo(Pseudo::Trickroom | Pseudo::Gravity | Pseudo::Magicroom | Pseudo::Wonderroom) => 5,
            // durationCallback(target, source, effect)
            Eff::SideCond(SideCond::Tailwind) => 4,
            Eff::SideCond(SideCond::Safeguard) => 5,
            Eff::SideCond(SideCond::Reflect | SideCond::Lightscreen | SideCond::Auroraveil) => {
                if holds(self, it::LIGHTCLAY) { 8 } else { 5 }
            }
            _ => unreachable!("no durationCallback body for {eff:?}"),
        }
    }

    /// Callbacks of side conditions. `side` is the side the condition is on.
    /// `Cond::data` holds the layers of Spikes and Toxic Spikes.
    pub(crate) fn side_cb(&mut self, kind: SideCond, ev: Ev, pre: Pre, side: usize) -> Res {
        let e = self.event;
        match (kind, ev, pre) {
            // ---- Spikes, Toxic Spikes: layers
            // onSideStart(side)
            (SideCond::Spikes | SideCond::Toxicspikes, Ev::SideStart, Pre::On) => {
                if let Some(c) = self.sides[side].conds.get_mut(kind) {
                    c.data = 1;
                }
                Res::Undef
            }
            // onSideRestart(side)
            (SideCond::Spikes | SideCond::Toxicspikes, Ev::SideRestart, Pre::On) => {
                let max = if kind == SideCond::Spikes { 3 } else { 2 };
                match self.sides[side].conds.get_mut(kind) {
                    Some(c) if c.data < max => {
                        c.data += 1;
                        Res::Undef
                    }
                    _ => FALSE,
                }
            }
            // Every other onSideStart(side) and onSideEnd(side) only writes to the log.
            (_, Ev::SideStart | Ev::SideEnd, Pre::On) => Res::Undef,

            // ---- Tailwind: onModifySpe(spe, pokemon)
            (SideCond::Tailwind, Ev::ModifySpe, Pre::On) => self.chain_modify(2, 1),

            // ---- Reflect, Light Screen, Aurora Veil: onAnyModifyDamage(damage, source, target, move).
            // `e.target` is the attacker and `e.source` the Pokémon being hit.
            (SideCond::Reflect | SideCond::Lightscreen | SideCond::Auroraveil, Ev::ModifyDamage, Pre::Any) => {
                let (Some(attacker), Some(defender), Eff::Move(mi)) = (e.target, e.source, e.effect) else {
                    return Res::Undef;
                };
                if defender == attacker || defender.side as usize != side {
                    return Res::Undef;
                }
                let category = self.am[mi as usize].category;
                let covered = match kind {
                    SideCond::Reflect => category == Category::Physical,
                    SideCond::Lightscreen => category == Category::Special,
                    // Aurora Veil steps aside where the matching screen is also up.
                    _ => {
                        let conds = &self.sides[defender.side as usize].conds;
                        !(conds.has(SideCond::Reflect) && category == Category::Physical
                            || conds.has(SideCond::Lightscreen) && category == Category::Special)
                    }
                };
                if !covered {
                    return Res::Undef;
                }
                let am = &self.am[mi as usize];
                if !am.hit_data[self.slot_index(defender)].crit && !am.infiltrates {
                    // The doubles value; in singles it would be a half.
                    return self.chain_modify(2732, 4096);
                }
                Res::Undef
            }

            // ---- Safeguard
            // onSetStatus(status, target, source, effect)
            (SideCond::Safeguard, Ev::SetStatus, Pre::On) => {
                let (Some(target), Some(source)) = (e.target, e.source) else {
                    return Res::Undef;
                };
                if e.effect == Eff::None || self.eff_is_named(e.effect, "yawn") {
                    return Res::Undef;
                }
                if let Eff::Move(mi) = e.effect {
                    if self.am[mi as usize].infiltrates && !self.is_ally(target, source) {
                        return Res::Undef;
                    }
                }
                if target != source {
                    return Res::Null;
                }
                Res::Undef
            }
            // onTryAddVolatile(status, target, source, effect)
            (SideCond::Safeguard, Ev::TryAddVolatile, Pre::On) => {
                let (Some(target), Some(source)) = (e.target, e.source) else {
                    return Res::Undef;
                };
                if e.effect == Eff::None {
                    return Res::Undef;
                }
                if let Eff::Move(mi) = e.effect {
                    if self.am[mi as usize].infiltrates && !self.is_ally(target, source) {
                        return Res::Undef;
                    }
                }
                let blocked = e.vol.is_some_and(|v| v == VolKind::Confusion || v.id() == "yawn");
                if blocked && target != source {
                    return Res::Null;
                }
                Res::Undef
            }

            // ---- Entry hazards: onSwitchIn(pokemon), run for each Pokémon that comes in.
            (SideCond::Spikes, Ev::SwitchIn, Pre::On) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                if !self.is_grounded(pokemon, false) {
                    return Res::Undef;
                }
                let layers = self.sides[side].conds.get(kind).map_or(1, |c| c.data) as u32;
                let d = div1([0u32, 3, 4, 6][layers.min(3) as usize] * self.mon(pokemon).max_hp() as u32, 24);
                self.damage(d, None, None, Eff::None);
                Res::Undef
            }
            (SideCond::Toxicspikes, Ev::SwitchIn, Pre::On) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                if !self.is_grounded(pokemon, false) {
                    return Res::Undef;
                }
                if self.has_type(pokemon, Type::Poison) {
                    // A grounded Poison type soaks them up.
                    self.remove_side_condition(pokemon.side as usize, SideCond::Toxicspikes);
                } else if self.has_type(pokemon, Type::Steel) {
                } else {
                    let layers = self.sides[side].conds.get(kind).map_or(1, |c| c.data);
                    let status = if layers >= 2 { Status::Tox } else { Status::Psn };
                    // Showdown names the foe's first active slot as the source.
                    let source = self.active(1 - pokemon.side as usize, 0);
                    self.try_set_status(pokemon, status, Some(source), Eff::None);
                }
                Res::Undef
            }
            (SideCond::Stealthrock, Ev::SwitchIn, Pre::On) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                // The damage scales with how a Rock move would do against the Pokémon.
                let saved = self.am_len;
                let rock = self.new_am(mv::STEALTHROCK);
                let type_mod = self.run_effectiveness(pokemon, rock).clamp(-6, 6);
                self.am_len = saved;
                let hp = self.mon(pokemon).max_hp() as u32;
                let d = if type_mod >= 0 { div1(hp << type_mod, 8) } else { div1(hp, 8 << -type_mod) };
                self.damage(d, None, None, Eff::None);
                Res::Undef
            }
            (SideCond::Stickyweb, Ev::SwitchIn, Pre::On) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                if !self.is_grounded(pokemon, false) {
                    return Res::Undef;
                }
                // The drop counts as coming from the move Sticky Web, used by the
                // Pokémon in the foe's first slot.
                let source = self.active(1 - pokemon.side as usize, 0);
                let saved = self.am_len;
                let web = self.new_am(mv::STICKYWEB);
                self.boost1(SPE, -1, Some(pokemon), Some(source), Eff::Move(web));
                self.am_len = saved;
                Res::Undef
            }
            _ => unreachable!("no body for {kind:?} {ev:?} {pre:?}"),
        }
    }

    /// Whether an effect is the condition with this id (for effects the engine may not model yet).
    pub(crate) fn eff_is_named(&self, eff: Eff, id: &str) -> bool {
        match eff {
            Eff::Vol(v) => v.id() == id,
            Eff::Move(mi) => self.am[mi as usize].d().id == id,
            _ => false,
        }
    }

    /// Callbacks of slot conditions. `holder` is the Pokémon in the position.
    pub(crate) fn slot_cb(&mut self, kind: SlotCond, ev: Ev, holder: MonRef) -> Res {
        let e = self.event;
        let side = holder.side as usize;
        let pos = self.mon(holder).position as usize;
        match (kind, ev) {
            // ---- Wish. `data` is the HP it will restore, `st.a` the turn counter when it was made.
            // onStart(pokemon, source)
            (SlotCond::Wish, Ev::Start) => {
                let hp = e.source.map_or(0, |s| self.mon(s).max_hp() / 2);
                let turn = self.overflowed_turn_count();
                if let Some(c) = self.sides[side].slot_conds[pos].get_mut(kind) {
                    c.data = hp;
                    c.st.a = turn;
                }
                Res::Undef
            }
            // onResidual(target): comes true at the end of the next turn.
            (SlotCond::Wish, Ev::Residual) => {
                let Some(c) = self.sides[side].slot_conds[pos].get(kind) else {
                    return Res::Undef;
                };
                if self.overflowed_turn_count() <= c.st.a {
                    return Res::Undef;
                }
                // The position the wisher stood in, on this side.
                let made_at = if c.source_slot == NO_SLOT { pos } else { (c.source_slot & 1) as usize };
                self.remove_slot_condition(side, made_at, kind);
                Res::Undef
            }
            // onEnd(target): whoever stands there now is healed.
            (SlotCond::Wish, Ev::End) => {
                let Some(target) = e.target else {
                    return Res::Undef;
                };
                if !self.mon(target).fainted {
                    let hp = self.sides[side].slot_conds[pos].get(kind).map_or(0, |c| c.data);
                    self.heal(hp as i32, Some(target), Some(target), Eff::None);
                }
                Res::Undef
            }
            _ => unreachable!("no body for {kind:?} {ev:?}"),
        }
    }

    /// `Battle#getOverflowedTurnCount`: the turn counter as the games keep it, in one byte.
    pub(crate) fn overflowed_turn_count(&self) -> i16 {
        ((self.turn as i32 - 1) & 0xFF) as i16
    }

    /// Callbacks of pseudo-weathers.
    pub(crate) fn pseudo_cb(&mut self, kind: Pseudo, ev: Ev) -> Res {
        let e = self.event;
        match (kind, ev) {
            // ---- Trick Room, Magic Room, Wonder Room: using the move again ends the room.
            // onFieldRestart(target, source)
            (Pseudo::Trickroom | Pseudo::Magicroom | Pseudo::Wonderroom, Ev::FieldRestart) => {
                self.remove_pseudo_weather(kind);
                Res::Undef
            }
            // onFieldStart / onFieldEnd only write to the log.
            (Pseudo::Trickroom | Pseudo::Wonderroom | Pseudo::Fairylock, Ev::FieldStart) => Res::Undef,
            (_, Ev::FieldEnd) => Res::Undef,

            // ---- Magic Room: onFieldStart(target, source). Every held item stops working.
            (Pseudo::Magicroom, Ev::FieldStart) => {
                let (actives, n) = self.all_active(false);
                for &mon in &actives[..n] {
                    let item = self.mon(mon).item;
                    self.single_event(Ev::End, Eff::Item(item), Some(mon), Some(mon), None, Eff::None, Res::Undef);
                }
                Res::Undef
            }

            // ---- Wonder Room: onModifyMove(move, source, target). Body Press follows the swap.
            (Pseudo::Wonderroom, Ev::ModifyMove) => {
                if let Eff::Move(mi) = e.effect {
                    let am = &mut self.am[mi as usize];
                    am.off_stat = match am.off_stat {
                        2 => 4,
                        4 => 2,
                        s => s,
                    };
                }
                Res::Undef
            }

            // ---- Fairy Lock: onTrapPokemon(pokemon)
            (Pseudo::Fairylock, Ev::TrapPokemon) => {
                if let Some(pokemon) = e.target {
                    self.try_trap(pokemon, false);
                }
                Res::Undef
            }

            // ---- Gravity
            // onFieldStart(target, source): everyone comes down.
            (Pseudo::Gravity, Ev::FieldStart) => {
                let (actives, n) = self.all_active(false);
                for &pokemon in &actives[..n] {
                    let mut flying = false;
                    for id in ["bounce", "fly"] {
                        if let Some(k) = VolKind::named(id) {
                            flying |= self.remove_volatile(pokemon, k);
                        }
                    }
                    if flying {
                        self.cancel_move(pokemon);
                        if let Some(k) = VolKind::named("twoturnmove") {
                            self.remove_volatile(pokemon, k);
                        }
                    }
                    // (Sky Drop is not in Champions.)
                    for id in ["magnetrise", "telekinesis"] {
                        if let Some(k) = VolKind::named(id) {
                            self.drop_vol(pokemon, k);
                        }
                    }
                }
                Res::Undef
            }
            // onModifyAccuracy(accuracy)
            (Pseudo::Gravity, Ev::ModifyAccuracy) => {
                if !matches!(e.relay, Res::Num(_)) {
                    return Res::Undef;
                }
                self.chain_modify(6840, 4096)
            }
            // onDisableMove(pokemon)
            (Pseudo::Gravity, Ev::DisableMove) => {
                if let Some(pokemon) = e.target {
                    let m = self.mon_mut(pokemon);
                    for k in 0..m.n_moves as usize {
                        if MOVES[m.moves[k].id as usize].flags & F_GRAVITY != 0 {
                            m.moves[k].disabled = true;
                        }
                    }
                }
                Res::Undef
            }
            // onBeforeMove(pokemon, target, move) and onModifyMove(move, pokemon, target)
            (Pseudo::Gravity, Ev::BeforeMove | Ev::ModifyMove) => {
                if self.event_move_flags() & F_GRAVITY != 0 {
                    return FALSE;
                }
                Res::Undef
            }
            _ => unreachable!("no body for {kind:?} {ev:?}"),
        }
    }

    /// Callbacks of weather conditions.
    pub(crate) fn weather_cb(&mut self, kind: Weather, ev: Ev) -> Res {
        let e = self.event;
        let mtype = match e.effect {
            Eff::Move(mi) => Some(self.am[mi as usize].typ),
            _ => None,
        };
        match (kind, ev) {
            // onFieldStart(field, source, effect) and onFieldEnd() only write to the log.
            (_, Ev::FieldStart | Ev::FieldEnd) => Res::Undef,

            // onFieldResidual(): the weather's turn; everyone gets the Weather event.
            (Weather::Raindance | Weather::Sunnyday, Ev::FieldResidual) => {
                self.each_event(Ev::Weather);
                Res::Undef
            }
            (Weather::Sandstorm | Weather::Snowscape, Ev::FieldResidual) => {
                if self.is_weather(kind) {
                    self.each_event(Ev::Weather);
                }
                Res::Undef
            }

            // onWeatherModifyDamage(damage, attacker, defender, move)
            (Weather::Raindance, Ev::WeatherModifyDamage) => {
                let Some(defender) = e.source else {
                    return Res::Undef;
                };
                if self.effective_weather(defender) != Weather::Raindance {
                    return Res::Undef;
                }
                match mtype {
                    Some(Type::Water) => self.chain_modify(3, 2),
                    Some(Type::Fire) => self.chain_modify(1, 2),
                    _ => Res::Undef,
                }
            }
            (Weather::Sunnyday, Ev::WeatherModifyDamage) => self.sun_modify_damage(),

            // onImmunity(type, pokemon): nothing freezes in the sun.
            (Weather::Sunnyday, Ev::Immunity) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                if self.effective_weather(pokemon) != Weather::Sunnyday {
                    return Res::Undef;
                }
                if e.imm == Some(Imm::Status(Status::Frz)) {
                    return FALSE;
                }
                Res::Undef
            }

            // onModifySpD(spd, pokemon): Rock types in a sandstorm.
            (Weather::Sandstorm, Ev::ModifySpD) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                if self.has_type(pokemon, Type::Rock) && self.effective_weather(pokemon) == Weather::Sandstorm {
                    return Res::Num(crate::battle::modify(e.relay.num() as u32, 6144) as i32);
                }
                Res::Undef
            }
            // onModifyDef(def, pokemon): Ice types in snow.
            (Weather::Snowscape, Ev::ModifyDef) => {
                let Some(pokemon) = e.target else {
                    return Res::Undef;
                };
                if self.has_type(pokemon, Type::Ice) && self.effective_weather(pokemon) == Weather::Snowscape {
                    return Res::Num(crate::battle::modify(e.relay.num() as u32, 6144) as i32);
                }
                Res::Undef
            }

            // onWeather(target): sandstorm damage.
            (Weather::Sandstorm, Ev::Weather) => {
                if let Some(target) = e.target {
                    let d = div1(self.mon(target).max_hp() as u32, 16);
                    self.damage(d, None, None, Eff::None);
                }
                Res::Undef
            }
            _ => unreachable!("no body for {kind:?} {ev:?}"),
        }
    }

    /// Sunny Day's `onWeatherModifyDamage(damage, attacker, defender, move)`,
    /// which Mega Sol also calls as its own.
    pub(crate) fn sun_modify_damage(&mut self) -> Res {
        let e = self.event;
        let (Some(attacker), Some(defender), Eff::Move(mi)) = (e.target, e.source, e.effect) else {
            return Res::Undef;
        };
        if self.am[mi as usize].d().id == "hydrosteam" && self.effective_weather(attacker) == Weather::Sunnyday {
            return self.chain_modify(3, 2);
        }
        if self.effective_weather(defender) != Weather::Sunnyday {
            return Res::Undef;
        }
        match self.am[mi as usize].typ {
            Type::Fire => self.chain_modify(3, 2),
            Type::Water => self.chain_modify(1, 2),
            _ => Res::Undef,
        }
    }

    /// Callbacks of terrains.
    pub(crate) fn terrain_cb(&mut self, kind: Terrain, ev: Ev) -> Res {
        let e = self.event;
        let mtype = match e.effect {
            Eff::Move(mi) => Some(self.am[mi as usize].typ),
            _ => None,
        };
        // "On the ground and not in the middle of Fly or Dig."
        let grounded = |b: &mut Battle, r: Option<MonRef>| {
            r.is_some_and(|r| b.is_grounded(r, false) && !b.is_semi_invulnerable(r))
        };
        match (kind, ev) {
            // onFieldStart(field, source, effect) and onFieldEnd() only write to the log.
            (_, Ev::FieldStart | Ev::FieldEnd) => Res::Undef,

            // ---- Electric Terrain
            // onSetStatus(status, target, source, effect)
            (Terrain::Electricterrain, Ev::SetStatus) => {
                if e.status == Status::Slp && grounded(self, e.target) {
                    return FALSE;
                }
                Res::Undef
            }
            // onTryAddVolatile(status, target)
            (Terrain::Electricterrain, Ev::TryAddVolatile) => {
                if !grounded(self, e.target) {
                    return Res::Undef;
                }
                if e.vol.is_some_and(|v| v.id() == "yawn") {
                    return Res::Null;
                }
                Res::Undef
            }
            // onBasePower(basePower, attacker, defender, move)
            (Terrain::Electricterrain, Ev::BasePower) => {
                if mtype == Some(Type::Electric) && grounded(self, e.target) {
                    return self.chain_modify(5325, 4096);
                }
                Res::Undef
            }

            // ---- Grassy Terrain
            // onBasePower(basePower, attacker, defender, move)
            (Terrain::Grassyterrain, Ev::BasePower) => {
                let weakened = matches!(e.effect, Eff::Move(mi)
                    if matches!(self.am[mi as usize].d().id, "earthquake" | "bulldoze" | "magnitude"));
                if weakened && grounded(self, e.source) {
                    return self.chain_modify(1, 2);
                }
                // (The attacker only has to be on the ground, not out of Fly or Dig.)
                if mtype == Some(Type::Grass) && e.target.is_some_and(|a| self.is_grounded(a, false)) {
                    return self.chain_modify(5325, 4096);
                }
                Res::Undef
            }
            // onResidual(pokemon): run once for each active Pokémon.
            (Terrain::Grassyterrain, Ev::Residual) => {
                if let Some(pokemon) = e.target {
                    if grounded(self, Some(pokemon)) {
                        let amount = div1(self.mon(pokemon).max_hp() as u32, 16);
                        self.heal(amount, Some(pokemon), Some(pokemon), Eff::None);
                    }
                }
                Res::Undef
            }

            // ---- Misty Terrain
            // onSetStatus(status, target, source, effect)
            (Terrain::Mistyterrain, Ev::SetStatus) => {
                if !grounded(self, e.target) {
                    return Res::Undef;
                }
                FALSE
            }
            // onTryAddVolatile(status, target, source, effect)
            (Terrain::Mistyterrain, Ev::TryAddVolatile) => {
                if !grounded(self, e.target) {
                    return Res::Undef;
                }
                if e.vol == Some(VolKind::Confusion) {
                    return Res::Null;
                }
                Res::Undef
            }
            // onBasePower(basePower, attacker, defender, move)
            (Terrain::Mistyterrain, Ev::BasePower) => {
                if mtype == Some(Type::Dragon) && grounded(self, e.source) {
                    return self.chain_modify(1, 2);
                }
                Res::Undef
            }

            // ---- Psychic Terrain
            // onTryHit(target, source, effect): priority moves fail against grounded foes.
            (Terrain::Psychicterrain, Ev::TryHit) => {
                let (Some(target), Some(source), Eff::Move(mi)) = (e.target, e.source, e.effect) else {
                    return Res::Undef;
                };
                let am = &self.am[mi as usize];
                // (Showdown compares against 0.1 because Quick Claw-style boosts are fractions.)
                if am.priority <= 0 || am.target == Target::User {
                    return Res::Undef;
                }
                if self.is_semi_invulnerable(target) || self.is_ally(target, source) {
                    return Res::Undef;
                }
                if !self.is_grounded(target, false) {
                    return Res::Undef;
                }
                Res::Null
            }
            // onBasePower(basePower, attacker, defender, move)
            (Terrain::Psychicterrain, Ev::BasePower) => {
                if mtype == Some(Type::Psychic) && grounded(self, e.target) {
                    return self.chain_modify(5325, 4096);
                }
                Res::Undef
            }
            _ => unreachable!("no body for {kind:?} {ev:?}"),
        }
    }

    /// `BattleActions#getConfusionDamage`.
    fn confusion_damage(&mut self, r: MonRef, base_power: u32) -> i32 {
        let (atk_stage, def_stage) = (self.mon(r).boosts[ATK] as i32, self.mon(r).boosts[DEF] as i32);
        let attack = self.calculate_stat(r, 1, atk_stage, r);
        let defense = self.calculate_stat(r, 2, def_stage, r);
        let level = self.mon(r).level as u32;
        let base = (2 * level / 5 + 2).wrapping_mul(base_power).wrapping_mul(attack) / defense.max(1) / 50 + 2;
        let dmg = base & 0xFFFF;
        let roll = self.rand(16, "confusion damage roll");
        (dmg.wrapping_mul(100 - roll) / 100).max(1) as i32
    }
}
