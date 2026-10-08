//! Event callbacks of held items, ported from Showdown's `data/items.ts` with
//! the Champions overrides.

use crate::data::*;
use crate::state::*;

impl Battle {
    /// Runs one item callback. `holder` has the item; the event's own target,
    /// source and relay value are in `self.event`.
    pub(crate) fn item_cb(&mut self, item: u16, ev: Ev, pre: Pre, holder: MonRef) -> Res {
        let _ = holder;
        unreachable!("no body for item {} {ev:?} {pre:?}", ITEMS[item as usize].id)
    }
}
