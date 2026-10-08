//! Pokémon Showdown's `Gen5RNG`: a 64-bit linear congruential generator that
//! hands out the upper 32 bits of its state. Matching it call-for-call is
//! what lets a battle here be compared against Showdown turn by turn.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rng {
    pub seed: u64,
    /// Number of values drawn so far (handy when comparing against Showdown).
    pub calls: u32,
}

impl Rng {
    /// Builds the generator from Showdown's `[a, b, c, d]` 16-bit seed words.
    pub fn from_words(w: [u16; 4]) -> Self {
        let seed = ((w[0] as u64) << 48) | ((w[1] as u64) << 32) | ((w[2] as u64) << 16) | w[3] as u64;
        Rng { seed, calls: 0 }
    }

    pub fn words(&self) -> [u16; 4] {
        [(self.seed >> 48) as u16, (self.seed >> 32) as u16, (self.seed >> 16) as u16, self.seed as u16]
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(0x5D58_8B65_6C07_8965).wrapping_add(0x0026_9EC3);
        self.calls = self.calls.wrapping_add(1);
        (self.seed >> 32) as u32
    }

    /// `PRNG#random(n)`: uniform integer in `[0, n)`. Always consumes one value.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        ((self.next_u32() as u64 * n as u64) >> 32) as u32
    }

    /// `PRNG#random(from, to)`: uniform integer in `[from, to)`.
    #[inline]
    pub fn range(&mut self, from: u32, to: u32) -> u32 {
        self.below(to - from) + from
    }

    /// `PRNG#randomChance(numerator, denominator)`.
    #[inline]
    pub fn chance(&mut self, numerator: u32, denominator: u32) -> bool {
        self.below(denominator) < numerator
    }

    /// `PRNG#shuffle(items, start, end)`: Fisher–Yates over `items[start..end]`.
    pub fn shuffle<T>(&mut self, items: &mut [T], mut start: usize, end: usize) {
        while start + 1 < end {
            let next = self.range(start as u32, end as u32) as usize;
            if start != next {
                items.swap(start, next);
            }
            start += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values produced by Showdown's `PRNG` with seed "1,2,3,4".
    #[test]
    fn matches_showdown_stream() {
        let mut r = Rng::from_words([1, 2, 3, 4]);
        let got: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
        assert_eq!(got, [2030470262, 3793892072, 2851743046, 574702432]);
    }

    #[test]
    fn matches_showdown_helpers() {
        let mut r = Rng::from_words([1, 2, 3, 4]);
        assert_eq!(r.below(16), 7);
        assert_eq!(r.below(100), 88);
        assert_eq!(r.range(2, 5), 3);
        assert!(!r.chance(1, 8));
        assert_eq!(r.words(), [8769, 17248, 37115, 18776]);
        assert_eq!(r.calls, 4);
    }

    #[test]
    fn matches_showdown_shuffle() {
        let mut r = Rng::from_words([65535, 65535, 65535, 65535]);
        let mut items = [0, 1, 2, 3, 4, 5, 6, 7];
        r.shuffle(&mut items, 2, 7);
        assert_eq!(items, [0, 1, 5, 6, 3, 4, 2, 7]);
        assert_eq!(r.words(), [54575, 34758, 50826, 47011]);
    }
}
