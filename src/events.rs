//! Showdown's event system (`Battle#runEvent`, `singleEvent`, `eachEvent`,
//! `fieldEvent`, `findEventHandlers`, `resolvePriority`).
//!
//! Which effect listens to which event, and with what order and priority,
//! comes from tables generated out of Showdown's data (`tables.rs`). The
//! bodies of the callbacks are written by hand in `conditions.rs`,
//! `abilities.rs` and `items.rs`. Collecting handlers through the tables,
//! rather than hard-coding who can react where, matters for more than
//! convenience: Showdown sorts the handlers of every event and shuffles the
//! ones that tie, so a handler that does nothing still changes the random
//! number stream if it ties with another one.

// The nesting mirrors Showdown's code; keep it rather than folding conditions together.
#![allow(clippy::collapsible_if, clippy::collapsible_match)]

use crate::data::*;
use crate::state::*;
use crate::trace;

const INLINE_HANDLERS: usize = 12;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Handler {
    pub eff: Eff,
    pub holder: MonRef,
    /// The event whose body runs (`Start` for an `onStart` run at switch-in).
    pub ev: Ev,
    pub pre: Pre,
    pub has_cb: bool,
    pub order: u32,
    pub priority: i16,
    /// Holder's cached speed times four (switch-in handlers subtract their
    /// place in the speed order, which Showdown expresses in quarters).
    pub speed: i32,
    pub sub_order: u8,
    pub effect_order: u32,
    /// `abilityState.effectOrder` of the holder, used by `compareRedirectOrder`.
    pub ability_order: u32,
    pub uid: u16,
    /// Position of the handler's target in a multi-target event.
    pub index: u8,
}

const BLANK: Handler = Handler {
    eff: Eff::None,
    holder: MonRef { side: 0, idx: 0 },
    ev: Ev::Start,
    pre: Pre::On,
    has_cb: false,
    order: 0,
    priority: 0,
    speed: 0,
    sub_order: 0,
    effect_order: 0,
    ability_order: 0,
    uid: 0,
    index: 0,
};

/// The handlers collected for one event. Nearly every event has only a few,
/// so they live in a small inline array; the rare larger set spills to the heap.
pub(crate) struct HList {
    n: usize,
    inline: [Handler; INLINE_HANDLERS],
    spill: Vec<Handler>,
}

impl HList {
    fn new() -> HList {
        HList { n: 0, inline: [BLANK; INLINE_HANDLERS], spill: Vec::new() }
    }
    fn push(&mut self, h: Handler) {
        if self.spill.is_empty() && self.n < INLINE_HANDLERS {
            self.inline[self.n] = h;
        } else {
            if self.spill.is_empty() {
                self.spill.extend_from_slice(&self.inline[..self.n]);
            }
            self.spill.push(h);
        }
        self.n += 1;
    }
    fn insert_front(&mut self, h: Handler) {
        if self.spill.is_empty() && self.n < INLINE_HANDLERS {
            self.inline.copy_within(0..self.n, 1);
            self.inline[0] = h;
        } else {
            if self.spill.is_empty() {
                self.spill.extend_from_slice(&self.inline[..self.n]);
            }
            self.spill.insert(0, h);
        }
        self.n += 1;
    }
    fn as_mut_slice(&mut self) -> &mut [Handler] {
        if self.spill.is_empty() { &mut self.inline[..self.n] } else { &mut self.spill[..] }
    }
    fn get(&self, k: usize) -> Handler {
        if self.spill.is_empty() { self.inline[k] } else { self.spill[k] }
    }
}

/// `Battle#comparePriority` for handlers.
fn cmp_priority(a: &Handler, b: &Handler) -> i64 {
    let ord = |h: &Handler| if h.order == 0 { 1i64 << 32 } else { h.order as i64 };
    let o = ord(a) - ord(b);
    if o != 0 {
        return o;
    }
    let p = b.priority as i64 - a.priority as i64;
    if p != 0 {
        return p;
    }
    let s = b.speed as i64 - a.speed as i64;
    if s != 0 {
        return s;
    }
    let so = a.sub_order as i64 - b.sub_order as i64;
    if so != 0 {
        return so;
    }
    a.effect_order as i64 - b.effect_order as i64
}

/// `Battle.compareLeftToRightOrder`.
fn cmp_left_to_right(a: &Handler, b: &Handler) -> std::cmp::Ordering {
    let ord = |h: &Handler| if h.order == 0 { 1i64 << 32 } else { h.order as i64 };
    ord(a).cmp(&ord(b)).then(b.priority.cmp(&a.priority)).then(a.index.cmp(&b.index))
}

/// `Battle.compareRedirectOrder`.
fn cmp_redirect(a: &Handler, b: &Handler) -> std::cmp::Ordering {
    b.priority.cmp(&a.priority).then(b.speed.cmp(&a.speed)).then(a.ability_order.cmp(&b.ability_order))
}

fn find_cb(cbs: &'static [CbInfo], ev: Ev, pre: Pre) -> Option<&'static CbInfo> {
    cbs.iter().find(|c| c.ev == ev && c.pre == pre)
}

impl Battle {
    /// `Battle#initEffectState` for an effect starting on `holder`.
    pub(crate) fn new_state(&mut self, has_id: bool, holder: MonRef) -> EffState {
        let order = if has_id && self.mon(holder).is_active {
            let o = self.effect_order;
            self.effect_order += 1;
            o
        } else {
            0
        };
        self.next_uid = self.next_uid.wrapping_add(1);
        EffState { order, uid: self.next_uid, a: 0, b: 0 }
    }

    /// Whether `eff` has an unprefixed callback for `ev`.
    pub(crate) fn has_cb(&self, eff: Eff, ev: Ev) -> bool {
        match eff {
            Eff::Status(s) => find_cb(STATUS_CONDS[s as usize].cbs, ev, Pre::On).is_some_and(|c| c.kind == CbKind::Fn),
            Eff::Vol(v) => find_cb(VOL_CONDS[v as usize].cbs, ev, Pre::On).is_some_and(|c| c.kind == CbKind::Fn),
            Eff::Ability(a) => find_cb(ABILITIES[a as usize].cbs, ev, Pre::On).is_some_and(|c| c.kind == CbKind::Fn),
            Eff::Item(i) => find_cb(ITEMS[i as usize].cbs, ev, Pre::On).is_some_and(|c| c.kind == CbKind::Fn),
            Eff::Move(mi) => self.move_has_cb(mi, ev),
            _ => false,
        }
    }

    /// `Battle#resolvePriority`.
    fn resolve(&self, eff: Eff, cb: &CbInfo, holder: MonRef, st: EffState) -> Handler {
        let m = self.mon(holder);
        let mut speed = m.speed * 4;
        let mut effect_order = 0;
        if matches!(cb.ev, Ev::SwitchIn | Ev::BeforeSwitchIn) {
            // Speed ties between switch-in handlers were settled when the
            // Pokémon were sorted; use that fixed order.
            let fpv = holder.side + 2 * m.position;
            let idx = self.speed_order[..self.n_speed_order as usize].iter().position(|&v| v == fpv);
            speed -= idx.map_or(-1, |i| i as i32);
            effect_order = st.order;
        } else if cb.ev == Ev::RedirectTarget {
            effect_order = st.order;
        }
        Handler {
            eff,
            holder,
            ev: if cb.kind == CbKind::StartAlias { Ev::Start } else { cb.ev },
            pre: cb.pre,
            has_cb: cb.kind != CbKind::DurationOnly,
            order: cb.order,
            priority: cb.priority,
            speed,
            sub_order: cb.sub_order,
            effect_order,
            ability_order: m.ability_st.order,
            uid: st.uid,
            index: 0,
        }
    }

    /// `Battle#findPokemonEventHandlers`.
    fn find_pokemon_handlers(&self, mon: MonRef, ev: Ev, pre: Pre, get_duration: bool, out: &mut HList) {
        let m = self.mon(mon);
        let bit = ev.bit();
        let a = &ABILITIES[m.ability as usize];
        let i = &ITEMS[m.item as usize];
        if pre != Pre::On {
            // Only abilities and items listen from the side (onAlly, onFoe, onAny, onSource).
            if a.events_pre & bit != 0 {
                if let Some(cb) = find_cb(a.cbs, ev, pre) {
                    out.push(self.resolve(Eff::Ability(m.ability), cb, mon, m.ability_st));
                }
            }
            if i.events_pre & bit != 0 {
                if let Some(cb) = find_cb(i.cbs, ev, pre) {
                    out.push(self.resolve(Eff::Item(m.item), cb, mon, m.item_st));
                }
            }
            return;
        }
        if m.status != Status::None {
            let c = &STATUS_CONDS[m.status as usize];
            if c.events & bit != 0 {
                if let Some(cb) = find_cb(c.cbs, ev, pre) {
                    out.push(self.resolve(Eff::Status(m.status), cb, mon, m.status_st));
                }
            }
        }
        for v in m.volatiles.as_slice() {
            let c = &VOL_CONDS[v.kind as usize];
            if c.events & bit == 0 {
                continue;
            }
            if let Some(cb) = find_cb(c.cbs, ev, pre) {
                if cb.kind == CbKind::DurationOnly && !(get_duration && v.duration > 0) {
                    continue;
                }
                out.push(self.resolve(Eff::Vol(v.kind), cb, mon, v.st));
            }
        }
        if a.events & bit != 0 {
            if let Some(cb) = find_cb(a.cbs, ev, pre) {
                out.push(self.resolve(Eff::Ability(m.ability), cb, mon, m.ability_st));
            }
        }
        if i.events & bit != 0 {
            if let Some(cb) = find_cb(i.cbs, ev, pre) {
                out.push(self.resolve(Eff::Item(m.item), cb, mon, m.item_st));
            }
        }
    }

    /// `Battle#findEventHandlers` for a Pokémon target (or none).
    fn find_event_handlers(&self, target: Option<MonRef>, ev: Ev, source: Option<MonRef>, out: &mut HList) {
        // Events normally run through `eachEvent` never have prefixed handlers.
        let prefixed = !matches!(ev, Ev::BeforeTurn | Ev::Update | Ev::Weather | Ev::WeatherChange | Ev::TerrainChange)
            && self.event_mask_pre & ev.bit() != 0;
        if let Some(t) = target {
            if self.mon(t).is_active || source.is_some_and(|s| self.mon(s).is_active) {
                self.find_pokemon_handlers(t, ev, Pre::On, false, out);
                if prefixed {
                    let (allies, n) = self.allies_and_self(t.side as usize);
                    for &a in &allies[..n] {
                        self.find_pokemon_handlers(a, ev, Pre::Ally, false, out);
                        self.find_pokemon_handlers(a, ev, Pre::Any, false, out);
                    }
                    let (foes, n) = self.allies_and_self(1 - t.side as usize);
                    for &f in &foes[..n] {
                        self.find_pokemon_handlers(f, ev, Pre::Foe, false, out);
                        self.find_pokemon_handlers(f, ev, Pre::Any, false, out);
                    }
                }
            }
        }
        if let Some(s) = source {
            if prefixed {
                self.find_pokemon_handlers(s, ev, Pre::Source, false, out);
            }
        }
        // Side, field and format handlers: nothing modelled has any.
    }

    /// `Battle#suppressingAbility`: a Mold Breaker-style move is in flight
    /// against `target`.
    pub(crate) fn suppressing_ability(&self, target: Option<MonRef>) -> bool {
        let Some(ap) = self.active_pokemon else {
            return false;
        };
        let Some(mi) = self.active_move else {
            return false;
        };
        self.mon(ap).is_active && Some(ap) != target && self.am[mi as usize].ignore_ability
    }

    /// The suppression rules `runEvent` applies to each handler. Returns
    /// whether the handler is skipped.
    fn handler_suppressed(&self, h: &Handler, ev: Ev) -> bool {
        match h.eff {
            Eff::Status(s) => self.mon(h.holder).status != s,
            Eff::Ability(a) => {
                if ABILITIES[a as usize].flags & AF_BREAKABLE != 0 && self.suppressing_ability(Some(h.holder)) {
                    return true;
                }
                ev != Ev::End && self.ignoring_ability(h.holder)
            }
            Eff::Item(_) => !matches!(ev, Ev::Start | Ev::SwitchIn | Ev::TakeItem) && self.ignoring_item(h.holder),
            _ => false,
        }
    }

    fn call_handler(&mut self, h: &Handler) -> Res {
        let parent = (self.effect, self.effect_holder);
        self.effect = h.eff;
        self.effect_holder = Some(h.holder);
        let r = self.dispatch(h.eff, h.ev, h.pre, Some(h.holder));
        (self.effect, self.effect_holder) = parent;
        r
    }

    /// The hand-written body of one callback. `holder` is the Pokémon the
    /// effect is on (`this.effectState.target`).
    fn dispatch(&mut self, eff: Eff, ev: Ev, pre: Pre, holder: Option<MonRef>) -> Res {
        match (eff, holder) {
            (Eff::Move(mi), _) => self.move_cb(mi, ev),
            (Eff::Status(s), Some(h)) => self.status_cb(s, ev, h),
            (Eff::Vol(v), Some(h)) => self.vol_cb(v, ev, h),
            (Eff::Ability(a), Some(h)) => self.ability_cb(a, ev, pre, h),
            (Eff::Item(i), Some(h)) => self.item_cb(i, ev, pre, h),
            _ => Res::Undef,
        }
    }

    /// `Battle#runEvent` with a single target (or none).
    pub(crate) fn run_event(
        &mut self,
        ev: Ev,
        target: Option<MonRef>,
        source: Option<MonRef>,
        effect: Eff,
        relay: Res,
    ) -> Res {
        if self.event_mask & ev.bit() == 0 {
            return if relay == Res::Undef { TRUE } else { relay };
        }
        self.run_event_ex(Event::new(ev, target, source, effect), relay, false, false).0
    }

    /// Whether any effect in this battle listens to `ev` at all.
    pub(crate) fn listens(&self, ev: Ev) -> bool {
        self.event_mask & ev.bit() != 0
    }

    /// `runEvent` given a prepared event (for relay variables that are not
    /// plain numbers); returns the event as the handlers left it.
    pub(crate) fn run_event_ex(&mut self, e: Event, relay: Res, on_effect: bool, fast_exit: bool) -> (Res, Event) {
        let ev = e.id.expect("event without id");
        let own = on_effect && self.has_cb(e.effect, ev);
        if self.event_mask & ev.bit() == 0 && !own {
            return (if relay == Res::Undef { TRUE } else { relay }, e);
        }
        let mut hl = HList::new();
        if self.event_mask & ev.bit() != 0 {
            self.find_event_handlers(e.target, ev, e.source, &mut hl);
        }
        if own {
            // The causing effect's own callback goes first.
            let holder = e.target.expect("onEffect without a target");
            let m = self.mon(holder);
            let h = Handler {
                eff: e.effect,
                holder,
                ev,
                has_cb: true,
                speed: m.speed * 4,
                ability_order: m.ability_st.order,
                sub_order: match e.effect {
                    Eff::Ability(_) => 7,
                    Eff::Item(_) => 8,
                    Eff::Vol(_) => 2,
                    _ => 0,
                },
                ..BLANK
            };
            hl.insert_front(h);
        }
        let mut relays = [Res::Undef; 1];
        let r = self.run_handlers(e, ev, &mut hl, relay, fast_exit, None, &mut relays);
        (r.0, r.1)
    }

    /// `runEvent` with an array of targets. `relays` holds the per-target relay
    /// variables on entry (`Res::Undef` for "none": Showdown starts those at
    /// `true`) and the per-target results on return.
    pub(crate) fn run_event_multi(
        &mut self,
        ev: Ev,
        targets: &[MonRef],
        source: Option<MonRef>,
        effect: Eff,
        relays: &mut [Res],
        has_relay: bool,
    ) {
        if !has_relay {
            for r in relays.iter_mut() {
                *r = TRUE;
            }
        }
        if self.event_mask & ev.bit() == 0 {
            return;
        }
        let mut hl = HList::new();
        for (i, &t) in targets.iter().enumerate() {
            let start = hl.n;
            self.find_event_handlers(Some(t), ev, source, &mut hl);
            for h in &mut hl.as_mut_slice()[start..] {
                h.index = i as u8;
            }
        }
        let e = Event::new(ev, None, source, effect);
        let relay = if has_relay { Res::Num(0) } else { Res::Undef };
        self.run_handlers(e, ev, &mut hl, relay, false, Some(targets), relays);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_handlers(
        &mut self,
        e: Event,
        ev: Ev,
        hl: &mut HList,
        relay: Res,
        fast_exit: bool,
        targets: Option<&[MonRef]>,
        target_relays: &mut [Res],
    ) -> (Res, Event) {
        let n = hl.n;
        if n == 0 {
            // Nobody listens: the relay variable comes straight back.
            return (if relay == Res::Undef { TRUE } else { relay }, e);
        }
        if matches!(ev, Ev::Invulnerability | Ev::TryHit | Ev::DamagingHit) {
            hl.as_mut_slice().sort_by(cmp_left_to_right);
        } else if fast_exit {
            hl.as_mut_slice().sort_by(cmp_redirect);
        } else {
            self.speed_sort(hl.as_mut_slice(), cmp_priority, "event handler tie");
        }
        let has_relay = relay != Res::Undef;
        let mut relay = if has_relay { relay } else { TRUE };

        let parent = self.event;
        self.event = e;
        self.event.modifier = 4096;
        self.event_depth += 1;
        debug_assert!(self.event_depth < 12, "event stack too deep in {ev:?}");

        for k in 0..n {
            let h = hl.get(k);
            if let Some(ts) = targets {
                let i = h.index as usize;
                let cur = target_relays[i];
                if !cur.truthy() && !(cur == Res::Num(0) && ev == Ev::DamagingHit) {
                    continue;
                }
                self.event.target = Some(ts[i]);
                if has_relay {
                    relay = cur;
                }
            }
            if self.handler_suppressed(&h, ev) {
                continue;
            }
            self.event.relay = relay;
            let ret = if h.has_cb { self.call_handler(&h) } else { Res::Undef };
            if ret != Res::Undef {
                relay = ret;
                if !relay.truthy() || fast_exit {
                    if targets.is_some() {
                        target_relays[h.index as usize] = relay;
                        if target_relays.iter().all(|v| !v.truthy()) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        self.event_depth -= 1;
        if let Res::Num(v) = relay {
            if v >= 0 && !matches!(ev, Ev::ModifySTAB | Ev::FractionalPriority) {
                relay = Res::Num(crate::battle::modify(v as u32, self.event.modifier) as i32);
            }
        }
        let after = self.event;
        self.event = parent;
        (relay, after)
    }

    /// `Battle#priorityEvent`: stops at the first handler that returns anything.
    pub(crate) fn priority_event(
        &mut self,
        ev: Ev,
        target: Option<MonRef>,
        source: Option<MonRef>,
        effect: Eff,
        relay: Res,
    ) -> Res {
        if self.event_mask & ev.bit() == 0 {
            return if relay == Res::Undef { TRUE } else { relay };
        }
        self.run_event_ex(Event::new(ev, target, source, effect), relay, false, true).0
    }

    /// `Battle#singleEvent`: run one effect's own callback for an event.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn single_event(
        &mut self,
        ev: Ev,
        eff: Eff,
        holder: Option<MonRef>,
        target: Option<MonRef>,
        source: Option<MonRef>,
        source_effect: Eff,
        relay: Res,
    ) -> Res {
        self.single_event_ex(ev, ev, Pre::On, eff, holder, Event::new(ev, target, source, source_effect), relay, false)
    }

    /// `singleEvent` in full: `ev` is the event the suppression rules see,
    /// `body` the callback that runs (they differ when an `onStart` runs
    /// during SwitchIn), and `custom` says the caller supplied the callback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn single_event_ex(
        &mut self,
        ev: Ev,
        body: Ev,
        pre: Pre,
        eff: Eff,
        holder: Option<MonRef>,
        e: Event,
        relay: Res,
        custom: bool,
    ) -> Res {
        let relay_var = if relay == Res::Undef { TRUE } else { relay };
        if let (Eff::Status(s), Some(t)) = (eff, e.target) {
            if self.mon(t).status != s {
                return relay_var;
            }
        }
        match eff {
            Eff::Ability(a) => {
                if ev == Ev::SwitchIn
                    && ABILITIES[a as usize].flags & AF_BREAKABLE != 0
                    && self.suppressing_ability(e.target)
                {
                    return relay_var;
                }
                if ev != Ev::End && e.target.is_some_and(|t| self.ignoring_ability(t)) {
                    return relay_var;
                }
            }
            Eff::Item(_) => {
                if !matches!(ev, Ev::Start | Ev::TakeItem | Ev::SetAbility)
                    && e.target.is_some_and(|t| self.ignoring_item(t))
                {
                    return relay_var;
                }
            }
            _ => {}
        }
        if !custom && !self.has_cb(eff, body) {
            return relay_var;
        }
        let parent = (self.effect, self.effect_holder, self.event);
        self.effect = eff;
        self.effect_holder = holder;
        self.event = e;
        self.event.relay = relay_var;
        self.event_depth += 1;
        debug_assert!(self.event_depth < 12, "event stack too deep in single {ev:?}");
        let ret = self.dispatch(eff, body, pre, holder.or(e.target));
        self.event_depth -= 1;
        (self.effect, self.effect_holder, self.event) = parent;
        if ret == Res::Undef { relay_var } else { ret }
    }

    /// `Battle#eachEvent`: run an event on every active Pokémon in speed order.
    pub(crate) fn each_event(&mut self, ev: Ev) {
        let (actives, n) = self.all_active(false);
        let mut keyed = [(actives[0], 0i32); 4];
        for i in 0..n {
            keyed[i] = (actives[i], self.mon(actives[i]).speed);
        }
        self.speed_sort(&mut keyed[..n], |a, b| b.1 as i64 - a.1 as i64, trace::each_label(ev));
        if self.event_mask & ev.bit() == 0 {
            return;
        }
        let effect = self.effect;
        for &(r, _) in &keyed[..n] {
            self.run_event_ex(Event::new(ev, Some(r), None, effect), Res::Undef, false, false);
        }
    }

    /// `Battle#fieldEvent`, used for Residual and SwitchIn: every effect on
    /// the field takes its turn in one global order.
    pub(crate) fn field_event(&mut self, ev: Ev, targets: Option<&[MonRef]>) {
        let get_duration = ev == Ev::Residual;
        let mut hl = HList::new();
        for side in 0..2 {
            for pos in 0..ACTIVE {
                let active = self.active(side, pos);
                if !self.in_play(active) {
                    continue;
                }
                if ev == Ev::SwitchIn && self.event_mask_pre & ev.bit() != 0 {
                    self.find_pokemon_handlers(active, ev, Pre::Any, false, &mut hl);
                }
                if targets.is_some_and(|ts| !ts.contains(&active)) {
                    continue;
                }
                self.find_pokemon_handlers(active, ev, Pre::On, get_duration, &mut hl);
            }
        }
        let n = hl.n;
        self.speed_sort(
            hl.as_mut_slice(),
            cmp_priority,
            if get_duration { "residual handler tie" } else { "switch-in handler tie" },
        );
        for k in 0..n {
            let h = hl.get(k);
            if self.mon(h.holder).fainted {
                continue;
            }
            if ev == Ev::Residual {
                if let Eff::Vol(kind) = h.eff {
                    // Count the volatile's duration down; it ends at zero.
                    let m = self.mon_mut(h.holder);
                    if let Some(v) = m.volatiles.get_mut(kind) {
                        if v.st.uid == h.uid && v.duration > 0 {
                            v.duration -= 1;
                            if v.duration == 0 {
                                self.remove_volatile(h.holder, kind);
                                if self.ended {
                                    return;
                                }
                                continue;
                            }
                        }
                    }
                }
            }
            // The effect may have been removed or replaced by an earlier handler.
            let m = self.mon(h.holder);
            let current = match h.eff {
                Eff::Ability(_) => Some(m.ability_st.uid),
                Eff::Item(_) => Some(m.item_st.uid),
                Eff::Status(_) => Some(m.status_st.uid),
                Eff::Vol(kind) => m.volatiles.get(kind).map(|v| v.st.uid),
                _ => None,
            };
            if current != Some(h.uid) {
                continue;
            }
            if h.has_cb {
                let e = Event::new(ev, Some(h.holder), None, Eff::None);
                self.single_event_ex(ev, h.ev, h.pre, h.eff, Some(h.holder), e, Res::Undef, true);
            }
            self.faint_messages(false, false, true);
            if self.ended {
                return;
            }
        }
    }

    // ------------------------------------------------------------- modifiers

    /// `Battle#chainModify` with a fraction: multiply the running event's modifier.
    pub(crate) fn chain_modify(&mut self, num: u32, den: u32) -> Res {
        let next = num * 4096 / den;
        self.event.modifier = (self.event.modifier * next + 2048) >> 12;
        Res::Undef
    }

    /// `Battle#finalModify`: apply the accumulated modifier now and reset it.
    pub(crate) fn final_modify(&mut self, value: u32) -> u32 {
        let v = crate::battle::modify(value, self.event.modifier);
        self.event.modifier = 4096;
        v
    }
}
