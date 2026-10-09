//! Optional event trace used when hunting a divergence from Showdown. With
//! the `trace` feature off (the default) every call here compiles to nothing.

#[cfg(feature = "trace")]
thread_local! {
    static LOG: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(feature = "trace")]
thread_local! {
    #[allow(clippy::type_complexity)]
    static SHOWN: std::cell::RefCell<std::collections::BTreeMap<(&'static str, u32, &'static str), (u64, u64)>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

/// Counts one use of a place in the engine where something is recorded as
/// shown: where it is, what was shown (an ability, item or move id), and
/// whether that was news.
#[inline(always)]
#[allow(unused_variables)]
pub fn shown_site(at: &'static std::panic::Location<'static>, what: &'static str, news: bool) {
    #[cfg(feature = "trace")]
    SHOWN.with(|m| {
        let mut m = m.borrow_mut();
        let e = m.entry((at.file(), at.line(), what)).or_insert((0, 0));
        e.0 += 1;
        e.1 += news as u64;
    });
}

/// Every such place used so far: file, line, what was shown, how often, and
/// how often it was news. Empty without the `trace` feature.
pub fn shown_sites() -> Vec<(&'static str, u32, &'static str, u64, u64)> {
    #[cfg(feature = "trace")]
    {
        return SHOWN.with(|m| m.borrow().iter().map(|(&(f, l, w), &(n, news))| (f, l, w, n, news)).collect());
    }
    #[allow(unreachable_code)]
    Vec::new()
}

/// Records one RNG draw: what it was for, the range, and the value drawn.
#[inline(always)]
#[allow(unused_variables)]
pub fn rng(what: &'static str, n: u32, value: u32) {
    #[cfg(feature = "trace")]
    LOG.with(|l| l.borrow_mut().push(format!("rng({n})={value}  {what}")));
}

/// Records a draw made with `random(from, to)`, in the same form the
/// Showdown recorder uses so the two streams can be compared line by line.
#[inline(always)]
#[allow(unused_variables)]
pub fn rng_range(what: &'static str, from: u32, to: u32, value: u32) {
    #[cfg(feature = "trace")]
    LOG.with(|l| l.borrow_mut().push(format!("rng({from},{to})={value}  {what}")));
}

/// Records a free-form note. The closure only runs when tracing is on.
#[inline(always)]
#[allow(unused_variables)]
pub fn note(f: impl FnOnce() -> String) {
    #[cfg(feature = "trace")]
    LOG.with(|l| l.borrow_mut().push(f()));
}

/// Label for the speed-tie shuffle `eachEvent` makes before running `ev`.
pub(crate) fn each_label(ev: crate::data::Ev) -> &'static str {
    match ev {
        crate::data::Ev::Update => "update tie",
        crate::data::Ev::BeforeTurn => "before-turn tie",
        _ => "each-event tie",
    }
}

/// Drains everything recorded since the last call.
pub fn take() -> Vec<String> {
    #[cfg(feature = "trace")]
    {
        return LOG.with(|l| std::mem::take(&mut *l.borrow_mut()));
    }
    #[allow(unreachable_code)]
    Vec::new()
}
