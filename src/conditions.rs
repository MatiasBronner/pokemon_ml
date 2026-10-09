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
    pub(crate) fn vol_cb(&mut self, kind: VolKind, ev: Ev, holder: MonRef) -> Res {
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
                if let Some(v) = self.mon_mut(holder).volatiles.get_mut(VolKind::Stall) {
                    v.data = 3;
                }
                Res::Undef
            }
            (VolKind::Stall, Ev::StallMove) => {
                let counter = self.mon(holder).volatiles.get(VolKind::Stall).map_or(1, |v| v.data.max(1)) as u32;
                let success = self.chance(1, counter, "consecutive protect");
                if !success {
                    self.mon_mut(holder).volatiles.remove(VolKind::Stall);
                }
                Res::Bool(success)
            }
            (VolKind::Stall, Ev::Restart) => {
                if let Some(v) = self.mon_mut(holder).volatiles.get_mut(VolKind::Stall) {
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
                if let Some(v) = self.mon_mut(holder).volatiles.get_mut(VolKind::Confusion) {
                    v.data = t as u16;
                }
                Res::Undef
            }
            (VolKind::Confusion, Ev::End) => Res::Undef,
            (VolKind::Confusion, Ev::BeforeMove) => {
                let Some(v) = self.mon_mut(holder).volatiles.get_mut(VolKind::Confusion) else {
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
                if let Some(v) = self.mon_mut(holder).volatiles.get_mut(VolKind::Choicelock) {
                    v.data = id + 1;
                }
                Res::Undef
            }
            (VolKind::Choicelock, Ev::BeforeMove) => {
                if ITEMS[self.mon(holder).item as usize].flags & IF_CHOICE == 0 {
                    self.remove_volatile(holder, VolKind::Choicelock);
                    return Res::Undef;
                }
                let locked = self.mon(holder).volatiles.get(VolKind::Choicelock).map_or(0, |v| v.data);
                if let Eff::Move(mi) = self.event.effect {
                    let am = &self.am[mi as usize];
                    if !self.ignoring_item(holder) && am.id + 1 != locked && am.d().special != Special::Struggle {
                        // Fails, and no PP is lost.
                        return FALSE;
                    }
                }
                Res::Undef
            }
            (VolKind::Choicelock, Ev::DisableMove) => {
                let locked = self.mon(holder).volatiles.get(VolKind::Choicelock).map_or(0, |v| v.data);
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
                if let Some(v) = self.mon_mut(holder).volatiles.get_mut(VolKind::Metronome) {
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
                if let Some(v) = self.mon_mut(holder).volatiles.get_mut(VolKind::Metronome) {
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
                let n = self.mon(holder).volatiles.get(VolKind::Metronome).map_or(0, |v| v.st.a).clamp(0, 5);
                self.chain_modify(MODS[n as usize], 4096)
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
