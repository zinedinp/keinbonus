//! Quantization: real signals to integer codes and back, at any bit depth
//! from 8 to 16, limited or full range (H.273 equations 20 to 31).
//!
//! | | luma (and every GBR, alpha signal) | colour difference |
//! |---|---|---|
//! | limited | `Round(2^(n-8) * (219 E + 16))` | `Round(2^(n-8) * (224 E + 128))` |
//! | full | `Round((2^n - 1) E)` | `Round((2^n - 1) E + 2^(n-1))` |
//!
//! Both clipped to `0..2^n`. In limited range the codes outside the
//! nominal span (foot- and headroom) are kept, not clipped: they carry
//! super-whites and below-blacks.

use crate::Error;
use crate::math::{clip3, round};

/// A bit depth from 8 to 16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BitDepth(u8);

impl BitDepth {
    pub const B8: Self = Self(8);
    pub const B10: Self = Self(10);
    pub const B12: Self = Self(12);
    pub const B14: Self = Self(14);
    pub const B16: Self = Self(16);

    /// # Errors
    ///
    /// [`Error::BitDepth`] outside 8..=16.
    pub const fn new(bits: u8) -> Result<Self, Error> {
        if bits >= 8 && bits <= 16 {
            Ok(Self(bits))
        } else {
            Err(Error::BitDepth(bits))
        }
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// The largest code, `2^n - 1`.
    #[must_use]
    pub const fn max(self) -> u16 {
        u16::MAX >> (16 - self.0)
    }

    fn scale(self) -> f64 {
        f64::from(1u32 << (self.0 - 8))
    }

    fn full(self) -> f64 {
        f64::from(self.max())
    }

    fn mid(self) -> f64 {
        f64::from(1u32 << (self.0 - 1))
    }
}

/// `VideoFullRangeFlag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Range {
    /// "Video", "TV", "narrow" range: black at 16, white at 235 (8 bits).
    Limited,
    /// "PC" range: black at 0, white at `2^n - 1`.
    Full,
}

impl Range {
    #[must_use]
    pub const fn from_flag(full_range: bool) -> Self {
        if full_range {
            Self::Full
        } else {
            Self::Limited
        }
    }
}

/// Which equation a signal takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Signal {
    /// 0..1: luma, the three signals of GBR, alpha.
    Luma,
    /// -0.5..0.5, centred on the middle code.
    Chroma,
}

/// The code of `e`.
#[must_use]
pub fn quantize(e: f64, depth: BitDepth, range: Range, signal: Signal) -> u16 {
    if e.is_nan() {
        return match signal {
            Signal::Luma => 0,
            Signal::Chroma => quantize(0.0, depth, range, signal),
        };
    }
    let code = match (range, signal) {
        (Range::Limited, Signal::Luma) => depth.scale() * (219.0 * e + 16.0),
        (Range::Limited, Signal::Chroma) => depth.scale() * (224.0 * e + 128.0),
        (Range::Full, Signal::Luma) => depth.full() * e,
        (Range::Full, Signal::Chroma) => depth.full() * e + depth.mid(),
    };
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "rounded and clipped to 0..=max"
    )]
    let code = clip3(0.0, depth.full(), round(code)) as u16;
    code
}

/// The signal a code stands for.
#[must_use]
pub fn dequantize(code: u16, depth: BitDepth, range: Range, signal: Signal) -> f64 {
    let d = f64::from(code);
    match (range, signal) {
        (Range::Limited, Signal::Luma) => (d / depth.scale() - 16.0) / 219.0,
        (Range::Limited, Signal::Chroma) => (d / depth.scale() - 128.0) / 224.0,
        (Range::Full, Signal::Luma) => d / depth.full(),
        (Range::Full, Signal::Chroma) => (d - depth.mid()) / depth.full(),
    }
}

/// A code at another bit depth: exact (a shift) in limited range going
/// up, rounded going down or in full range.
#[must_use]
pub fn requantize(code: u16, from: BitDepth, to: BitDepth, range: Range, signal: Signal) -> u16 {
    quantize(dequantize(code, from, range, signal), to, range, signal)
}

/// The codes of nominal black and white (luma) or of the colour
/// difference extremes (chroma).
#[must_use]
pub fn nominal(depth: BitDepth, range: Range, signal: Signal) -> (u16, u16) {
    let (lo, hi) = match signal {
        Signal::Luma => (0.0, 1.0),
        Signal::Chroma => (-0.5, 0.5),
    };
    (
        quantize(lo, depth, range, signal),
        quantize(hi, depth, range, signal),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: Signal = Signal::Luma;
    const C: Signal = Signal::Chroma;

    #[test]
    fn nominal_codes() {
        use Range::{Full, Limited};
        let d = |b| BitDepth::new(b).unwrap();
        assert_eq!(nominal(d(8), Limited, L), (16, 235));
        assert_eq!(nominal(d(8), Limited, C), (16, 240));
        assert_eq!(nominal(d(10), Limited, L), (64, 940));
        assert_eq!(nominal(d(10), Limited, C), (64, 960));
        assert_eq!(nominal(d(12), Limited, L), (256, 3760));
        assert_eq!(nominal(d(16), Limited, L), (4096, 60160));
        assert_eq!(nominal(d(8), Full, L), (0, 255));
        assert_eq!(nominal(d(10), Full, L), (0, 1023));
        // Full-range chroma: 0 at 2^(n-1); -0.5 lands on 1, +0.5 clips to max.
        assert_eq!(quantize(0.0, d(8), Full, C), 128);
        assert_eq!(nominal(d(8), Full, C), (1, 255));
        assert_eq!(quantize(0.0, d(10), Limited, C), 512);
        assert_eq!(BitDepth::B14.max(), 16383);
    }

    #[test]
    fn every_code_survives_dequantize_and_quantize() {
        for bits in 8..=16 {
            let depth = BitDepth::new(bits).unwrap();
            for range in [Range::Limited, Range::Full] {
                for signal in [L, C] {
                    for code in 0..=depth.max() {
                        let e = dequantize(code, depth, range, signal);
                        assert_eq!(
                            quantize(e, depth, range, signal),
                            code,
                            "{bits} {range:?} {signal:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn limited_range_bit_depths_are_shifts() {
        for code in 0..=255u16 {
            for to in [BitDepth::B10, BitDepth::B12, BitDepth::B16] {
                let shift = to.bits() - 8;
                for signal in [L, C] {
                    assert_eq!(
                        requantize(code, BitDepth::B8, to, Range::Limited, signal),
                        code << shift
                    );
                }
            }
        }
        // Down: rounded, halves up.
        assert_eq!(
            requantize(941, BitDepth::B10, BitDepth::B8, Range::Limited, L),
            235
        );
        assert_eq!(
            requantize(942, BitDepth::B10, BitDepth::B8, Range::Limited, L),
            236
        );
        // Full range scales by (2^m - 1) / (2^n - 1): 255 -> 1023, 128 -> 514.
        assert_eq!(
            requantize(255, BitDepth::B8, BitDepth::B10, Range::Full, L),
            1023
        );
        assert_eq!(
            requantize(128, BitDepth::B8, BitDepth::B10, Range::Full, L),
            514
        );
    }

    #[test]
    fn out_of_range_clips_and_nan_is_black() {
        assert_eq!(quantize(2.0, BitDepth::B8, Range::Limited, L), 255);
        assert_eq!(quantize(-1.0, BitDepth::B8, Range::Limited, L), 0);
        // Head- and footroom stay.
        assert_eq!(quantize(1.05, BitDepth::B8, Range::Limited, L), 246);
        assert_eq!(quantize(f64::NAN, BitDepth::B8, Range::Limited, L), 0);
        assert_eq!(quantize(f64::NAN, BitDepth::B8, Range::Limited, C), 128);
        assert_eq!(BitDepth::new(7), Err(Error::BitDepth(7)));
        assert_eq!(BitDepth::new(17), Err(Error::BitDepth(17)));
    }
}
