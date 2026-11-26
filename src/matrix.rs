//! Matrix coefficients: R'G'B' to luma and two colour differences, and back
//! (H.273 equations for `MatrixCoefficients`, BT.2100 for ICtCp).
//!
//! Signals are real numbers: `Y'` in 0..1, the colour differences in
//! -0.5..0.5 ([`Matrix::chroma_signed`]), except for
//! [`Matrix::Identity`], whose three signals are G', B', R' in 0..1.
//! [`quant`](crate::quant) makes codes of them.

use crate::Error;
use crate::cicp::{ColourPrimaries, MatrixCoefficients as Mc};
use crate::math::{self, Mat3};
use crate::transfer::Curve;

/// A matrix H.273 defines, with its constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Matrix {
    /// GBR: `[G', B', R']` as they are.
    Identity,
    /// Non-constant luminance: `Y' = KR R' + (1 - KR - KB) G' + KB B'`,
    /// `Cb = (B' - Y') / (2 (1 - KB))`, `Cr = (R' - Y') / (2 (1 - KR))`.
    Ncl { kr: f64, kb: f64 },
    /// `Y = (R + 2G + B) / 4`, `Cg = (-R + 2G - B) / 4`, `Co = (R - B) / 2`.
    YCgCo,
    /// Constant luminance: luma from linear light through `curve`, colour
    /// differences divided by twice `scale = [NB, PB, NR, PR]` per sign.
    Cl {
        kr: f64,
        kb: f64,
        curve: Curve,
        scale: [f64; 4],
    },
    /// BT.2100 ICtCp, with the PQ or the HLG constants.
    Ictcp { curve: Curve },
}

impl Mc {
    /// The matrix, for signals of `primaries` and `curve` (the
    /// constant-luminance matrices and ICtCp work on linear light, and the
    /// chroma-derived ones take their constants from the primaries).
    ///
    /// # Errors
    ///
    /// [`Error::Unspecified`], [`Error::Unsupported`] (YDzDx, IPT-C2,
    /// YCgCo-R), or [`Error::Invalid`] for ICtCp without PQ or HLG or
    /// outside BT.2020.
    pub fn matrix(self, primaries: ColourPrimaries, curve: Curve) -> Result<Matrix, Error> {
        let ncl = |kr, kb| Ok(Matrix::Ncl { kr, kb });
        let derived = || -> Result<(f64, f64), Error> {
            let [kr, _, kb] = primaries.primaries()?.luminance();
            Ok((kr, kb))
        };
        match self {
            Self::Identity => Ok(Matrix::Identity),
            Self::Bt709 => ncl(0.2126, 0.0722),
            Self::Unspecified => Err(Error::Unspecified("matrix coefficients")),
            Self::Fcc => ncl(0.30, 0.11),
            Self::Bt470Bg | Self::Bt601 => ncl(0.299, 0.114),
            Self::Smpte240 => ncl(0.212, 0.087),
            Self::YCgCo => Ok(Matrix::YCgCo),
            Self::Bt2020Ncl => ncl(0.2627, 0.0593),
            // BT.2020 Table 4 states the four denominators as numbers
            // (1.9404, 1.5816, 1.7184, 0.9936): normative, and up to 4e-4
            // away from what the curve gives (`cl_scale`).
            Self::Bt2020Cl => Ok(Matrix::Cl {
                kr: 0.2627,
                kb: 0.0593,
                curve,
                scale: BT2020_CL,
            }),
            Self::ChromaDerivedNcl => {
                let (kr, kb) = derived()?;
                ncl(kr, kb)
            }
            Self::ChromaDerivedCl => {
                let (kr, kb) = derived()?;
                Ok(Matrix::Cl {
                    kr,
                    kb,
                    curve,
                    scale: cl_scale(kr, kb, curve),
                })
            }
            Self::Ictcp => {
                if !matches!(curve, Curve::Pq | Curve::Hlg) {
                    Err(Error::Invalid("ICtCp needs PQ or HLG"))
                } else if primaries != ColourPrimaries::Bt2020 {
                    Err(Error::Invalid("ICtCp is defined on BT.2020"))
                } else {
                    Ok(Matrix::Ictcp { curve })
                }
            }
            Self::Smpte2085 => Err(Error::Unsupported("YDzDx (SMPTE 2085)")),
            Self::IptC2 => Err(Error::Unsupported("IPT-C2")),
            Self::YCgCoRe | Self::YCgCoRo => Err(Error::Unsupported("YCgCo-R")),
        }
    }
}

/// BT.2100: linear BT.2020 RGB to LMS.
const LMS: Mat3 = [
    [1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0],
    [683.0 / 4096.0, 2951.0 / 4096.0, 462.0 / 4096.0],
    [99.0 / 4096.0, 309.0 / 4096.0, 3688.0 / 4096.0],
];

/// BT.2100: L'M'S' to ICtCp, PQ.
const ICTCP_PQ: Mat3 = [
    [0.5, 0.5, 0.0],
    [6610.0 / 4096.0, -13613.0 / 4096.0, 7003.0 / 4096.0],
    [17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0],
];

/// BT.2100: L'M'S' to ICtCp, HLG.
const ICTCP_HLG: Mat3 = [
    [0.5, 0.5, 0.0],
    [3625.0 / 4096.0, -7465.0 / 4096.0, 3840.0 / 4096.0],
    [9500.0 / 4096.0, -9212.0 / 4096.0, -288.0 / 4096.0],
];

const LMS_INVERSE: Mat3 = math::inverse_of(&LMS);
const ICTCP_PQ_INVERSE: Mat3 = math::inverse_of(&ICTCP_PQ);
const ICTCP_HLG_INVERSE: Mat3 = math::inverse_of(&ICTCP_HLG);

fn ictcp_matrix(curve: Curve) -> &'static Mat3 {
    if curve == Curve::Hlg {
        &ICTCP_HLG
    } else {
        &ICTCP_PQ
    }
}

/// BT.2020's `[NB, PB, NR, PR]`: half its Table 4 denominators.
pub const BT2020_CL: [f64; 4] = [0.9702, 0.7908, 0.8592, 0.4968];

/// The constant-luminance scale factors `(NB, PB, NR, PR)`: the most
/// negative and most positive `B' - Y'` and `R' - Y'`.
#[must_use]
pub fn cl_scale(kr: f64, kb: f64, curve: Curve) -> [f64; 4] {
    [
        curve.encode(1.0 - kb),
        1.0 - curve.encode(kb),
        curve.encode(1.0 - kr),
        1.0 - curve.encode(kr),
    ]
}

impl Matrix {
    /// Whether the second and third signals are colour differences
    /// (-0.5..0.5, coded around the middle code), not colours (0..1).
    #[must_use]
    pub const fn chroma_signed(&self) -> bool {
        !matches!(self, Self::Identity)
    }

    /// The matrix as a linear map, `(to_ycbcr, to_rgb)`, for the matrices
    /// that are one (GBR, NCL, YCgCo); `None` for constant luminance and
    /// ICtCp, which go through linear light.
    #[must_use]
    pub fn linear(&self) -> Option<(Mat3, Mat3)> {
        if !matches!(self, Self::Identity | Self::Ncl { .. } | Self::YCgCo) {
            return None;
        }
        let column = |f: &dyn Fn([f64; 3]) -> [f64; 3]| {
            let (x, y, z) = (f([1.0, 0.0, 0.0]), f([0.0, 1.0, 0.0]), f([0.0, 0.0, 1.0]));
            [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
        };
        Some((column(&|v| self.to_ycbcr(v)), column(&|v| self.to_rgb(v))))
    }

    /// `[R', G', B']` to `[Y', Cb, Cr]` (or `[G', B', R']`, `[Y, Cg, Co]`,
    /// `[I, Ct, Cp]`).
    #[must_use]
    pub fn to_ycbcr(&self, [r, g, b]: [f64; 3]) -> [f64; 3] {
        match *self {
            Self::Identity => [g, b, r],
            Self::Ncl { kr, kb } => {
                let y = kr * r + (1.0 - kr - kb) * g + kb * b;
                [
                    y,
                    (b - y) / (2.0 * (1.0 - kb)),
                    (r - y) / (2.0 * (1.0 - kr)),
                ]
            }
            Self::YCgCo => [
                0.25 * r + 0.5 * g + 0.25 * b,
                -0.25 * r + 0.5 * g - 0.25 * b,
                0.5 * r - 0.5 * b,
            ],
            Self::Cl {
                kr,
                kb,
                curve,
                scale,
            } => {
                let [nb, pb, nr, pr] = scale;
                let lin = [r, g, b].map(|v| curve.decode(v));
                let y = curve.encode(kr * lin[0] + (1.0 - kr - kb) * lin[1] + kb * lin[2]);
                let (db, dr) = (b - y, r - y);
                [
                    y,
                    db / (2.0 * if db <= 0.0 { nb } else { pb }),
                    dr / (2.0 * if dr <= 0.0 { nr } else { pr }),
                ]
            }
            Self::Ictcp { curve } => {
                let lin = [r, g, b].map(|v| curve.decode(v));
                let lms = math::apply(&LMS, lin).map(|v| curve.encode(v));
                math::apply(ictcp_matrix(curve), lms)
            }
        }
    }

    /// `[Y', Cb, Cr]` back to `[R', G', B']`.
    #[must_use]
    pub fn to_rgb(&self, [y, cb, cr]: [f64; 3]) -> [f64; 3] {
        match *self {
            Self::Identity => [cr, y, cb],
            Self::Ncl { kr, kb } => {
                let r = y + 2.0 * (1.0 - kr) * cr;
                let b = y + 2.0 * (1.0 - kb) * cb;
                [r, (y - kr * r - kb * b) / (1.0 - kr - kb), b]
            }
            Self::YCgCo => {
                let t = y - cb;
                [t + cr, y + cb, t - cr]
            }
            Self::Cl {
                kr,
                kb,
                curve,
                scale,
            } => {
                let [nb, pb, nr, pr] = scale;
                let b = y + 2.0 * cb * if cb <= 0.0 { nb } else { pb };
                let r = y + 2.0 * cr * if cr <= 0.0 { nr } else { pr };
                let (yl, rl, bl) = (curve.decode(y), curve.decode(r), curve.decode(b));
                let g = (yl - kr * rl - kb * bl) / (1.0 - kr - kb);
                [r, curve.encode(g), b]
            }
            Self::Ictcp { curve } => {
                let back = if curve == Curve::Hlg {
                    &ICTCP_HLG_INVERSE
                } else {
                    &ICTCP_PQ_INVERSE
                };
                let lms = math::apply(back, [y, cb, cr]).map(|v| curve.decode(v));
                let rgb = math::apply(&LMS_INVERSE, lms);
                rgb.map(|v| curve.encode(v))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::close;

    fn m(c: Mc, p: ColourPrimaries, t: Curve) -> Matrix {
        c.matrix(p, t).unwrap()
    }

    fn near(a: [f64; 3], b: [f64; 3], eps: f64) -> bool {
        a.iter().zip(b).all(|(&x, y)| close(x, y, eps))
    }

    const BT709: Matrix = Matrix::Ncl {
        kr: 0.2126,
        kb: 0.0722,
    };

    #[test]
    fn ncl_white_black_and_red() {
        assert!(near(BT709.to_ycbcr([1.0; 3]), [1.0, 0.0, 0.0], 1e-15));
        assert!(near(BT709.to_ycbcr([0.0; 3]), [0.0; 3], 1e-15));
        // Red: Cr reaches 0.5 exactly, Cb is -KR / (2 (1 - KB)).
        let red = BT709.to_ycbcr([1.0, 0.0, 0.0]);
        assert!(near(red, [0.2126, -0.2126 / 1.8556, 0.5], 1e-15));
        assert!(near(
            BT709.to_ycbcr([0.0, 0.0, 1.0]),
            [0.0722, 0.5, -0.0722 / 1.5748],
            1e-15
        ));
    }

    #[test]
    fn every_matrix_round_trips() {
        let cases = [
            m(Mc::Identity, ColourPrimaries::Bt709, Curve::Srgb),
            BT709,
            m(Mc::Bt601, ColourPrimaries::Bt601, Curve::Bt709),
            m(Mc::Fcc, ColourPrimaries::Bt470M, Curve::Gamma22),
            m(Mc::Smpte240, ColourPrimaries::Smpte240, Curve::Smpte240),
            m(Mc::YCgCo, ColourPrimaries::Bt709, Curve::Bt709),
            m(Mc::Bt2020Ncl, ColourPrimaries::Bt2020, Curve::Bt709),
            m(Mc::Bt2020Cl, ColourPrimaries::Bt2020, Curve::Bt709),
            m(Mc::ChromaDerivedNcl, ColourPrimaries::Smpte432, Curve::Srgb),
            m(Mc::ChromaDerivedCl, ColourPrimaries::Smpte432, Curve::Srgb),
            m(Mc::Ictcp, ColourPrimaries::Bt2020, Curve::Pq),
            m(Mc::Ictcp, ColourPrimaries::Bt2020, Curve::Hlg),
        ];
        let steps = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
        for matrix in cases {
            for &r in &steps {
                for &g in &steps {
                    for &b in &steps {
                        let ycc = matrix.to_ycbcr([r, g, b]);
                        let (lo, hi) = if matrix.chroma_signed() {
                            (-0.5, 0.5)
                        } else {
                            (0.0, 1.0)
                        };
                        // BT.2020's rounded CL denominators overshoot 0.5 by 1.2e-4.
                        let slack = if matches!(
                            matrix,
                            Matrix::Cl {
                                scale: BT2020_CL,
                                ..
                            }
                        ) {
                            2.5e-4
                        } else {
                            1e-9
                        };
                        let inside = |v: f64| v >= lo - slack && v <= hi + slack;
                        assert!(inside(ycc[1]) && inside(ycc[2]), "{matrix:?} {ycc:?}");
                        assert!(
                            ycc[0] >= -1e-9 && ycc[0] <= 1.0 + 1e-9,
                            "{matrix:?} {ycc:?}"
                        );
                        let back = matrix.to_rgb(ycc);
                        // PQ and HLG are infinitely steep at 0 (and PQ's 0 is c1^m2 = 7.3e-7):
                        // 1e-18 of linear rounding noise is 1e-6 of signal near black,
                        // a thousandth of a 10-bit code.
                        let eps = if matches!(matrix, Matrix::Ictcp { .. }) {
                            1e-5
                        } else {
                            1e-9
                        };
                        assert!(
                            near(back, [r, g, b], eps),
                            "{matrix:?} {:?} -> {back:?}",
                            [r, g, b]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bt2020_cl_scales_are_bt2020s() {
        // BT.2020 Table 4: 1.9404, 1.5816, 1.7184, 0.9936 (twice N and P);
        // what the curve gives is within 4e-4 of them.
        let derived = cl_scale(0.2627, 0.0593, Curve::Bt709);
        for (got, want) in derived.iter().zip(BT2020_CL) {
            assert!(close(*got, want, 4e-4), "{got} vs {want}");
        }
        let Matrix::Cl { scale, .. } = m(Mc::Bt2020Cl, ColourPrimaries::Bt2020, Curve::Bt709)
        else {
            panic!()
        };
        assert_eq!(scale, BT2020_CL);
        // With BT.2020's numbers the extremes are 0.5 to within 2.5e-4;
        // the chroma-derived matrix reaches -0.5 and 0.5 exactly.
        let bt2020 = m(Mc::Bt2020Cl, ColourPrimaries::Bt2020, Curve::Bt709);
        assert!(close(bt2020.to_ycbcr([0.0, 0.0, 1.0])[1], 0.5, 2.5e-4));
        let cl = m(Mc::ChromaDerivedCl, ColourPrimaries::Bt2020, Curve::Bt709);
        assert!(close(cl.to_ycbcr([0.0, 0.0, 1.0])[1], 0.5, 1e-12));
        assert!(close(cl.to_ycbcr([1.0, 1.0, 0.0])[1], -0.5, 1e-12));
        assert!(close(cl.to_ycbcr([1.0, 0.0, 0.0])[2], 0.5, 1e-12));
        assert!(close(cl.to_ycbcr([0.0, 1.0, 1.0])[2], -0.5, 1e-12));
    }

    #[test]
    fn ictcp_white_is_neutral() {
        for curve in [Curve::Pq, Curve::Hlg] {
            let ictcp = m(Mc::Ictcp, ColourPrimaries::Bt2020, curve);
            let white = curve.encode(0.01);
            let [i, ct, cp] = ictcp.to_ycbcr([white; 3]);
            assert!(close(i, white, 1e-12) && close(ct, 0.0, 1e-12) && close(cp, 0.0, 1e-12));
        }
        for row in LMS {
            assert!(close(row.iter().sum(), 1.0, 1e-15));
        }
    }

    #[test]
    fn derived_and_refused() {
        let Matrix::Ncl { kr, kb } = m(Mc::ChromaDerivedNcl, ColourPrimaries::Bt709, Curve::Bt709)
        else {
            panic!()
        };
        assert!(close(kr, 0.2126, 1e-4) && close(kb, 0.0722, 1e-4));
        assert!(Mc::Ictcp.matrix(ColourPrimaries::Bt709, Curve::Pq).is_err());
        assert!(
            Mc::Ictcp
                .matrix(ColourPrimaries::Bt2020, Curve::Bt709)
                .is_err()
        );
        assert_eq!(
            Mc::Smpte2085.matrix(ColourPrimaries::Bt709, Curve::Bt709),
            Err(Error::Unsupported("YDzDx (SMPTE 2085)"))
        );
        assert!(
            Mc::Unspecified
                .matrix(ColourPrimaries::Bt709, Curve::Bt709)
                .is_err()
        );
        let y = m(Mc::YCgCo, ColourPrimaries::Bt709, Curve::Bt709);
        assert!(near(y.to_ycbcr([1.0, 0.0, 0.0]), [0.25, -0.25, 0.5], 1e-15));
    }
}
