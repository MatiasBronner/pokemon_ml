//! Optional event trace used when hunting a divergence from Showdown. With
//! the `trace` feature off (the default) every call here compiles to nothing.

#[cfg(feature = "trace")]
thread_local! {
    static LOG: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
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
