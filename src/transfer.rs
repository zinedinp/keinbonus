//! Transfer functions (H.273 Table 3): [`Curve::encode`] takes linear light
//! to the signal, [`Curve::decode`] back.
//!
//! What "linear" means differs, as in H.273:
//!
//! - Scene light, 0..1 (1 = reference white): BT.709 and the curves that
//!   reuse it, gamma 2.2/2.8, SMPTE 240M, linear, log, xvYCC, BT.1361, HLG.
//! - Display light, 0..1 (1 = 10 000 cd/m²): PQ.
//! - Display light, 0..1 (1 = 48 cd/m²): SMPTE 428.
//! - sRGB is defined display-side too; its 1 is the display's white.
//!
//! [`hdr`](crate::hdr) turns either into cd/m².
//!
//! Outside the domain H.273 states, each curve does what its standard
//! allows, else the least surprising thing, written on the variant: the
//! curves with a linear segment through 0 mirror (an odd function, as
//! xvYCC defines it), the log, PQ, HLG and SMPTE 428 curves clamp below 0.
//! Above 1 every curve continues its formula.

use crate::Error;
use crate::cicp::TransferCharacteristics as Tc;
use crate::math::{exp, ln, log10, pow, pow_odd, sqrt};

/// H.273's `alpha` of the BT.709 curve (to 15 places, where the segments
/// meet with matching slope).
pub const BT709_ALPHA: f64 = 1.099_296_826_809_442;
/// H.273's `beta` of the BT.709 curve.
pub const BT709_BETA: f64 = 0.018_053_968_510_807;

pub(crate) const PQ_M1: f64 = 2610.0 / 16384.0;
pub(crate) const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
pub(crate) const PQ_C1: f64 = 3424.0 / 4096.0;
pub(crate) const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
pub(crate) const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

/// BT.2100's HLG `a`.
pub const HLG_A: f64 = 0.178_832_77;
/// BT.2100's HLG `b = 1 - 4a`.
pub const HLG_B: f64 = 1.0 - 4.0 * HLG_A;
/// BT.2100's HLG `c = 0.5 - a * ln(4a)`, to 8 places as BT.2100 states it.
pub const HLG_C: f64 = 0.559_910_73;

/// A transfer function H.273 defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Curve {
    /// BT.709 (also BT.601, BT.2020 10- and 12-bit): `alpha * L^0.45 -
    /// (alpha - 1)` above `beta`, `4.5 * L` below. Mirrored below 0.
    Bt709,
    /// `L^(1/2.2)`, mirrored.
    Gamma22,
    /// `L^(1/2.8)`, mirrored.
    Gamma28,
    /// SMPTE 240M: `1.1115 * L^0.45 - 0.1115` above 0.0228, `4 * L`
    /// below. Mirrored.
    Smpte240,
    Linear,
    /// `1 + log10(L) / 2` down to 0.01, 0 below.
    Log100,
    /// `1 + log10(L) / 2.5` down to `sqrt(10) / 1000`, 0 below.
    Log316,
    /// IEC 61966-2-4: the BT.709 curve, mirrored below 0 by definition.
    Xvycc,
    /// BT.1361 extended gamut: BT.709 above 0, and below `-beta / 4`
    /// `-(alpha * (-4L)^0.45 - (alpha - 1)) / 4`, down to -0.25.
    Bt1361,
    /// IEC 61966-2-1: `1.055 * L^(1/2.4) - 0.055` above 0.003 130 8,
    /// `12.92 * L` below. Mirrored (as sYCC and extended sRGB use it).
    Srgb,
    /// SMPTE ST 2084. Clamped below 0.
    Pq,
    /// SMPTE ST 428-1: `(48 * L / 52.37)^(1/2.6)`. Clamped below 0.
    Smpte428,
    /// BT.2100 HLG OETF. Clamped below 0.
    Hlg,
}

impl Tc {
    /// The curve this code point stands for.
    ///
    /// # Errors
    ///
    /// [`Error::Unspecified`] for [`Unspecified`](Self::Unspecified).
    pub const fn curve(self) -> Result<Curve, Error> {
        Ok(match self {
            Self::Bt709 | Self::Bt601 | Self::Bt2020_10 | Self::Bt2020_12 => Curve::Bt709,
            Self::Unspecified => return Err(Error::Unspecified("transfer characteristics")),
            Self::Gamma22 => Curve::Gamma22,
            Self::Gamma28 => Curve::Gamma28,
            Self::Smpte240 => Curve::Smpte240,
            Self::Linear => Curve::Linear,
            Self::Log100 => Curve::Log100,
            Self::Log316 => Curve::Log316,
            Self::Iec61966_2_4 => Curve::Xvycc,
            Self::Bt1361 => Curve::Bt1361,
            Self::Srgb => Curve::Srgb,
            Self::Pq => Curve::Pq,
            Self::Smpte428 => Curve::Smpte428,
            Self::Hlg => Curve::Hlg,
        })
    }
}

fn bt709(l: f64) -> f64 {
    if l >= BT709_BETA {
        BT709_ALPHA * pow(l, 0.45) - (BT709_ALPHA - 1.0)
    } else {
        4.5 * l
    }
}

fn bt709_inverse(v: f64) -> f64 {
    if v >= 4.5 * BT709_BETA {
        pow((v + BT709_ALPHA - 1.0) / BT709_ALPHA, 1.0 / 0.45)
    } else {
        v / 4.5
    }
}

/// An odd function from its half for `x >= 0`.
fn mirrored(x: f64, half: impl Fn(f64) -> f64) -> f64 {
    half(x.abs()).copysign(x)
}

impl Curve {
    /// Linear light to the signal.
    #[must_use]
    pub fn encode(self, l: f64) -> f64 {
        match self {
            Self::Bt709 | Self::Xvycc => mirrored(l, bt709),
            Self::Gamma22 => pow_odd(l, 1.0 / 2.2),
            Self::Gamma28 => pow_odd(l, 1.0 / 2.8),
            Self::Smpte240 => mirrored(l, |l| {
                if l >= 0.0228 {
                    1.1115 * pow(l, 0.45) - 0.1115
                } else {
                    4.0 * l
                }
            }),
            Self::Linear => l,
            Self::Log100 => {
                if l >= 0.01 {
                    1.0 + log10(l) / 2.0
                } else {
                    0.0
                }
            }
            Self::Log316 => {
                if l >= sqrt(10.0) / 1000.0 {
                    1.0 + log10(l) / 2.5
                } else {
                    0.0
                }
            }
            Self::Bt1361 => {
                if l >= -BT709_BETA / 4.0 {
                    bt709(l)
                } else {
                    -(BT709_ALPHA * pow(-4.0 * l, 0.45) - (BT709_ALPHA - 1.0)) / 4.0
                }
            }
            Self::Srgb => mirrored(l, |l| {
                if l > 0.003_130_8 {
                    1.055 * pow(l, 1.0 / 2.4) - 0.055
                } else {
                    12.92 * l
                }
            }),
            Self::Pq => {
                let y = pow(l.max(0.0), PQ_M1);
                pow((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y), PQ_M2)
            }
            Self::Smpte428 => pow(48.0 * l.max(0.0) / 52.37, 1.0 / 2.6),
            Self::Hlg => {
                let e = l.max(0.0);
                if e <= 1.0 / 12.0 {
                    sqrt(3.0 * e)
                } else {
                    HLG_A * ln(12.0 * e - HLG_B) + HLG_C
                }
            }
        }
    }

    /// The signal back to linear light. Where [`encode`](Self::encode)
    /// clamps, the clamped range decodes to its lower end (0).
    #[must_use]
    pub fn decode(self, v: f64) -> f64 {
        match self {
            Self::Bt709 | Self::Xvycc => mirrored(v, bt709_inverse),
            Self::Gamma22 => pow_odd(v, 2.2),
            Self::Gamma28 => pow_odd(v, 2.8),
            Self::Smpte240 => mirrored(v, |v| {
                if v >= 4.0 * 0.0228 {
                    pow((v + 0.1115) / 1.1115, 1.0 / 0.45)
                } else {
                    v / 4.0
                }
            }),
            Self::Linear => v,
            Self::Log100 => {
                if v > 0.0 {
                    pow(10.0, (v - 1.0) * 2.0)
                } else {
                    0.0
                }
            }
            Self::Log316 => {
                if v > 0.0 {
                    pow(10.0, (v - 1.0) * 2.5)
                } else {
                    0.0
                }
            }
            Self::Bt1361 => {
                if v >= -4.5 * BT709_BETA / 4.0 {
                    bt709_inverse(v)
                } else {
                    -pow((-4.0 * v + BT709_ALPHA - 1.0) / BT709_ALPHA, 1.0 / 0.45) / 4.0
                }
            }
            Self::Srgb => mirrored(v, |v| {
                if v > 12.92 * 0.003_130_8 {
                    pow((v + 0.055) / 1.055, 2.4)
                } else {
                    v / 12.92
                }
            }),
            Self::Pq => {
                let e = pow(v.max(0.0), 1.0 / PQ_M2);
                pow((e - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * e), 1.0 / PQ_M1)
            }
            Self::Smpte428 => 52.37 / 48.0 * pow(v.max(0.0), 2.6),
            Self::Hlg => {
                let e = v.max(0.0);
                if e <= 0.5 {
                    e * e / 3.0
                } else {
                    (exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0
                }
            }
        }
    }

    /// Whether values below 0 carry colour (xvYCC, BT.1361): such signals
    /// are not clipped at 0 when converted into.
    #[must_use]
    pub const fn is_extended(self) -> bool {
        matches!(self, Self::Xvycc | Self::Bt1361)
    }

    /// Light above SDR reference white reaches the display.
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }

    /// Whether linear light is absolute display light (PQ, SMPTE 428),
    /// not relative to a white.
    #[must_use]
    pub const fn is_absolute(self) -> bool {
        matches!(self, Self::Pq | Self::Smpte428)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::close;

    const ALL: [Curve; 13] = [
        Curve::Bt709,
        Curve::Gamma22,
        Curve::Gamma28,
        Curve::Smpte240,
        Curve::Linear,
        Curve::Log100,
        Curve::Log316,
        Curve::Xvycc,
        Curve::Bt1361,
        Curve::Srgb,
        Curve::Pq,
        Curve::Smpte428,
        Curve::Hlg,
    ];

    #[test]
    fn every_curve_round_trips_on_its_domain() {
        for c in ALL {
            let low = match c {
                // Everything below the log floor encodes to 0.
                Curve::Log100 => 0.010_1,
                Curve::Log316 => 0.003_2,
                _ => 0.0,
            };
            for i in 0..=1000 {
                let l = low + (1.0 - low) * f64::from(i) / 1000.0;
                let back = c.decode(c.encode(l));
                assert!(close(back, l, 1e-12 * l.max(1.0)), "{c:?} {l} -> {back}");
            }
            let one = if c == Curve::Smpte428 {
                pow(48.0 / 52.37, 1.0 / 2.6)
            } else {
                1.0
            };
            // HLG's c has 8 places.
            let eps = if c == Curve::Hlg { 1e-7 } else { 1e-12 };
            assert!(close(c.encode(1.0), one, eps), "{c:?}");
        }
    }

    #[test]
    fn the_segments_meet() {
        // BT.709: both segments give 4.5 * beta at beta (H.273's alpha and
        // beta are chosen so).
        let at = BT709_ALPHA * pow(BT709_BETA, 0.45) - (BT709_ALPHA - 1.0);
        assert!(close(at, 4.5 * BT709_BETA, 1e-12));
        let eps = 1e-12;
        // How well each standard's published constants meet: BT.709 with
        // H.273's 15-place alpha and beta, sRGB's 0.003 130 8 to 5e-7,
        // HLG's 8-place c to 1e-8, SMPTE 240M's to 6e-5.
        for (c, x, gap) in [
            (Curve::Bt709, BT709_BETA, 1e-9),
            (Curve::Bt1361, -BT709_BETA / 4.0, 1e-9),
            (Curve::Hlg, 1.0 / 12.0, 1e-8),
            (Curve::Srgb, 0.003_130_8, 1e-6),
            (Curve::Smpte240, 0.0228, 1e-4),
        ] {
            let (below, above) = (c.encode(x - eps), c.encode(x + eps));
            assert!(close(below, above, gap), "{c:?}: {below} vs {above}");
        }
        // SMPTE 240M's published constants meet to 4 places only.
        assert!(close(
            1.1115 * pow(0.0228, 0.45) - 0.1115,
            4.0 * 0.0228,
            1e-4
        ));
    }

    #[test]
    fn reference_points() {
        // sRGB: 18 % grey.
        assert!(close(Curve::Srgb.encode(0.18), 0.461_356, 1e-6));
        // PQ (ST 2084): 100, 203 (BT.2408's 58 %), 1000, 10 000 cd/m².
        let pq = |nits: f64| Curve::Pq.encode(nits / 10_000.0);
        assert!(close(pq(100.0), 0.508_078, 1e-6));
        assert!(close(pq(203.0), 0.580_688, 1e-6));
        assert!(close(pq(1000.0), 0.751_827, 1e-6));
        assert!(close(pq(10_000.0), 1.0, 1e-12));
        assert!(close(pq(0.0), pow(PQ_C1, PQ_M2), 1e-18));
        // HLG: the segments meet at 1/12 -> 0.5; 1 -> 1.
        assert!(close(Curve::Hlg.encode(1.0 / 12.0), 0.5, 1e-15));
        assert!(close(Curve::Hlg.encode(1.0), 1.0, 1e-7));
        assert!(close(HLG_C, 0.5 - HLG_A * ln(4.0 * HLG_A), 1e-8));
        // BT.709: 18 % grey is 40.9 %.
        assert!(close(Curve::Bt709.encode(0.18), 0.409, 5e-4));
        assert!(close(Curve::Log100.encode(0.1), 0.5, 1e-15));
        assert!(close(Curve::Log316.encode(0.1), 0.6, 1e-15));
    }

    #[test]
    fn domains_outside_zero_to_one() {
        // Mirrored curves are odd.
        for c in [
            Curve::Bt709,
            Curve::Xvycc,
            Curve::Srgb,
            Curve::Gamma22,
            Curve::Smpte240,
        ] {
            assert!(close(c.encode(-0.3), -c.encode(0.3), 1e-15), "{c:?}");
            assert!(close(c.decode(c.encode(-0.3)), -0.3, 1e-12), "{c:?}");
        }
        // BT.1361 reaches -0.25 with the quarter-scale curve.
        assert!(close(Curve::Bt1361.encode(-0.25), -0.25, 1e-12));
        assert!(close(
            Curve::Bt1361.decode(Curve::Bt1361.encode(-0.1)),
            -0.1,
            1e-12
        ));
        // Clamped ones.
        for c in [Curve::Pq, Curve::Hlg, Curve::Smpte428, Curve::Log100] {
            assert_eq!(c.encode(-0.5), c.encode(0.0), "{c:?}");
        }
        assert_eq!(Curve::Log100.encode(0.005), 0.0);
        assert!(Tc::Unspecified.curve().is_err());
        assert_eq!(Tc::Bt2020_12.curve(), Ok(Curve::Bt709));
    }
}
