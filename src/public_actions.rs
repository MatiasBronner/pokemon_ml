//! Public protection knowledge. Silent skipped actions cannot reveal a failed
//! protection roll's cause, so both training and the log follower retain doubt.
use crate::state::{ACTIVE, MAX_TEAM, MonRef};

#[derive(Clone, Copy, Debug)]
pub(crate) enum PublicEvent {
    Turn([Option<MonRef>; 4]),
    Upkeep,
    Switch { r: MonRef, pos: usize },
    Swap { side: usize, first: usize, second: usize },
    Reset(MonRef),
    Attempt(MonRef),
    Success(MonRef),
    Failure { r: MonRef, visible: [Option<MonRef>; 4] },
}

impl PublicEvent {
    pub(crate) fn renumber(&mut self, mut f: impl FnMut(&mut MonRef)) {
        match self {
            Self::Turn(rs) => rs.iter_mut().flatten().for_each(&mut f),
            Self::Switch { r, .. } | Self::Reset(r) | Self::Attempt(r) | Self::Success(r) => f(r),
            Self::Failure { r, visible } => {
                f(r);
                visible.iter_mut().flatten().for_each(f);
            }
            _ => {}
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Protection {
    age: u8,
    uncertain: bool,
    before_age: u8,
    before_uncertain: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PublicActions {
    pending: [[bool; MAX_TEAM]; 2],
    slots: [Option<MonRef>; 4],
    protection: [[Protection; MAX_TEAM]; 2],
}

impl PublicActions {
    pub(crate) fn start(&mut self, r: MonRef, ordinary: bool) {
        if ordinary {
            self.pending[r.side as usize][r.idx as usize] = false;
        }
    }
    pub(crate) fn uncertain(&self, r: MonRef) -> bool {
        let p = self.protection[r.side as usize][r.idx as usize];
        p.uncertain && p.age > 0
    }
    pub(crate) fn renumber(&mut self, side: usize, moved: &[(usize, usize, bool)]) {
        let pending = self.pending[side];
        let protection = self.protection[side];
        for &(from, _, _) in moved {
            self.pending[side][from] = false;
            self.protection[side][from] = Protection::default();
        }
        for &(from, to, _) in moved {
            self.pending[side][to] = pending[from];
            self.protection[side][to] = protection[from];
        }
        for r in self.slots.iter_mut().flatten().filter(|r| r.side as usize == side) {
            if let Some((_, to, _)) = moved.iter().find(|&&(from, _, _)| from == r.idx as usize) {
                r.idx = *to as u8;
            }
        }
    }
    pub(crate) fn event(&mut self, event: PublicEvent) {
        match event {
            PublicEvent::Turn(slots) => {
                self.pending = [[false; MAX_TEAM]; 2];
                self.slots = slots;
                for r in slots.into_iter().flatten() {
                    self.pending[r.side as usize][r.idx as usize] = true;
                }
            }
            PublicEvent::Upkeep => {
                for p in self.protection.iter_mut().flatten() {
                    p.age = p.age.saturating_sub(1);
                    if p.age == 0 {
                        p.uncertain = false;
                    }
                }
            }
            PublicEvent::Switch { r, pos } => {
                if let Some(old) = self.slots[r.side as usize * ACTIVE + pos] {
                    self.pending[old.side as usize][old.idx as usize] = false;
                }
                self.slots[r.side as usize * ACTIVE + pos] = Some(r);
                self.pending[r.side as usize][r.idx as usize] = false;
                self.protection[r.side as usize][r.idx as usize] = Protection::default();
            }
            PublicEvent::Swap { side, first, second } => self.slots.swap(side * ACTIVE + first, side * ACTIVE + second),
            PublicEvent::Reset(r) => self.protection[r.side as usize][r.idx as usize] = Protection::default(),
            PublicEvent::Attempt(r) => {
                let p = &mut self.protection[r.side as usize][r.idx as usize];
                p.before_age = p.age;
                p.before_uncertain = p.uncertain;
            }
            PublicEvent::Success(r) => {
                let p = &mut self.protection[r.side as usize][r.idx as usize];
                p.age = 2;
            }
            PublicEvent::Failure { r, visible } => {
                let living_pending = visible.iter().flatten().any(|r| self.pending[r.side as usize][r.idx as usize]);
                let silent_pending = (0..2).any(|s| {
                    (0..MAX_TEAM)
                        .any(|i| self.pending[s][i] && !visible.contains(&Some(MonRef { side: s as u8, idx: i as u8 })))
                });
                let p = &mut self.protection[r.side as usize][r.idx as usize];
                (p.age, p.uncertain) = (p.before_age, p.before_uncertain);
                if living_pending {
                    *p = Protection::default();
                } else if silent_pending && p.age > 0 {
                    p.uncertain = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const A: MonRef = MonRef { side: 0, idx: 0 };
    const B: MonRef = MonRef { side: 1, idx: 0 };
    #[test]
    fn skipped_action_is_uncertain_but_a_visible_pending_action_resets_the_counter() {
        let mut p = PublicActions::default();
        p.event(PublicEvent::Turn([Some(A), Some(B), None, None]));
        p.start(A, true);
        p.event(PublicEvent::Attempt(A));
        p.event(PublicEvent::Success(A));
        p.event(PublicEvent::Attempt(A));
        p.event(PublicEvent::Failure { r: A, visible: [Some(A), None, None, None] });
        assert!(p.uncertain(A));
        let mut visible = p.clone();
        visible.event(PublicEvent::Attempt(A));
        visible.event(PublicEvent::Failure { r: A, visible: [Some(A), Some(B), None, None] });
        assert!(!visible.uncertain(A));
        p.event(PublicEvent::Upkeep);
        assert!(p.uncertain(A));
        p.event(PublicEvent::Upkeep);
        assert!(!p.uncertain(A));
    }
    #[test]
    fn an_inserted_instruction_does_not_consume_the_natural_action() {
        let mut p = PublicActions::default();
        p.event(PublicEvent::Turn([Some(A), Some(B), None, None]));
        p.start(A, false);
        p.event(PublicEvent::Attempt(A));
        p.event(PublicEvent::Success(A));
        p.event(PublicEvent::Attempt(A));
        p.event(PublicEvent::Failure { r: A, visible: [Some(A), None, None, None] });
        assert!(!p.uncertain(A));
    }
    #[test]
    fn public_history_does_not_turn_old_faints_into_pending_actions_and_resets_on_entry() {
        let mut p = PublicActions::default();
        p.event(PublicEvent::Turn([Some(A), Some(B), None, None]));
        p.start(A, true);
        p.event(PublicEvent::Attempt(A));
        p.event(PublicEvent::Success(A));
        p.event(PublicEvent::Attempt(A));
        p.event(PublicEvent::Failure { r: A, visible: [Some(A), None, None, None] });
        assert!(p.uncertain(A));
        p.event(PublicEvent::Upkeep);
        p.event(PublicEvent::Turn([Some(A), None, None, None]));
        p.start(A, true);
        p.event(PublicEvent::Attempt(A));
        p.event(PublicEvent::Failure { r: A, visible: [Some(A), None, None, None] });
        assert!(p.uncertain(A)); // An empty queue does not settle the old hidden counter.
        p.event(PublicEvent::Switch { r: A, pos: 0 });
        assert!(!p.uncertain(A));
    }
    #[test]
    fn breaking_protection_resolves_doubt_and_renumbering_preserves_it() {
        let mut p = PublicActions::default();
        p.event(PublicEvent::Turn([Some(A), Some(B), None, None]));
        p.start(A, true);
        p.event(PublicEvent::Attempt(A));
        p.event(PublicEvent::Success(A));
        p.event(PublicEvent::Attempt(A));
        p.event(PublicEvent::Failure { r: A, visible: [Some(A), None, None, None] });
        let renamed = MonRef { side: 0, idx: 2 };
        p.renumber(0, &[(0, 2, false)]);
        assert!(p.uncertain(renamed));
        assert!(!p.uncertain(A));
        p.event(PublicEvent::Reset(renamed));
        assert!(!p.uncertain(renamed));
    }
}
