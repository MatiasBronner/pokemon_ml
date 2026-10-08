//! Event callbacks of abilities, ported from Showdown's `data/abilities.ts`
//! with the Champions overrides.

use crate::data::*;
use crate::state::*;

impl Battle {
    /// Runs one ability callback. `holder` has the ability; the event's own
    /// target, source and relay value are in `self.event`.
    pub(crate) fn ability_cb(&mut self, ability: u16, ev: Ev, pre: Pre, holder: MonRef) -> Res {
        let _ = holder;
        unreachable!("no body for ability {} {ev:?} {pre:?}", ABILITIES[ability as usize].id)
    }
}
