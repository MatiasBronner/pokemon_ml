//! Event callbacks of status conditions and volatile conditions, ported from
//! Showdown's `data/conditions.ts` (with the Champions overrides) and from the
//! `condition` blocks of moves.

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
            #[allow(unreachable_patterns)]
            _ => unreachable!("no body for {kind:?} {ev:?}"),
        }
    }
}
