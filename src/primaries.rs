//! Colour primaries: the chromaticities of H.273 Table 2, the matrices to
//! and from CIE 1931 XYZ, the matrix between two sets of primaries, and
//! chromatic adaptation between white points.

use crate::Error;
use crate::cicp::ColourPrimaries;
use crate::math::{self, Mat3};

/// A point in CIE 1931 xy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Xy {
    pub x: f64,
    pub y: f64,
}

#[must_use]
pub const fn xy(x: f64, y: f64) -> Xy {
    Xy { x, y }
}

impl Xy {
    /// XYZ of this chromaticity at `Y = 1`; `None` at `y = 0`.
    #[must_use]
    pub fn xyz(self) -> Option<[f64; 3]> {
        (self.y != 0.0).then(|| [self.x / self.y, 1.0, (1.0 - self.x - self.y) / self.y])
    }
}

/// CIE standard illuminant D65.
pub const D65: Xy = xy(0.3127, 0.3290);
/// CIE illuminant C (BT.470 System M, film).
pub const C: Xy = xy(0.310, 0.316);
/// The DCI theatre white (SMPTE RP 431-2).
pub const DCI: Xy = xy(0.314, 0.351);
/// The equal-energy illuminant E.
pub const E: Xy = xy(1.0 / 3.0, 1.0 / 3.0);

/// Red, green and blue primaries and the white point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Chromaticities {
    pub red: Xy,
    pub green: Xy,
    pub blue: Xy,
    pub white: Xy,
}

const fn rgbw(r: (f64, f64), g: (f64, f64), b: (f64, f64), white: Xy) -> Chromaticities {
    Chromaticities {
        red: xy(r.0, r.1),
        green: xy(g.0, g.1),
        blue: xy(b.0, b.1),
        white,
    }
}

impl ColourPrimaries {
    /// H.273 Table 2.
    ///
    /// # Errors
    ///
    /// [`Error::Unspecified`] for [`Unspecified`](Self::Unspecified).
    pub const fn chromaticities(self) -> Result<Chromaticities, Error> {
        Ok(match self {
            Self::Bt709 => rgbw((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), D65),
            Self::Unspecified => return Err(Error::Unspecified("colour primaries")),
            Self::Bt470M => rgbw((0.67, 0.33), (0.21, 0.71), (0.14, 0.08), C),
            Self::Bt470Bg => rgbw((0.64, 0.33), (0.29, 0.60), (0.15, 0.06), D65),
            Self::Bt601 | Self::Smpte240 => {
                rgbw((0.630, 0.340), (0.310, 0.595), (0.155, 0.070), D65)
            }
            Self::Film => rgbw((0.681, 0.319), (0.243, 0.692), (0.145, 0.049), C),
            Self::Bt2020 => rgbw((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), D65),
            Self::Smpte428 => rgbw((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), E),
            Self::Smpte431 => rgbw((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), DCI),
            Self::Smpte432 => rgbw((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), D65),
            Self::Ebu3213 => rgbw((0.630, 0.340), (0.295, 0.605), (0.155, 0.077), D65),
        })
    }

    /// The primaries with their XYZ matrices.
    ///
    /// # Errors
    ///
    /// [`Error::Unspecified`] for [`Unspecified`](Self::Unspecified).
    pub fn primaries(self) -> Result<Primaries, Error> {
        if self == Self::Smpte428 {
            // R, G, B are X, Y, Z: the generic derivation divides by the
            // y = 0 of red and blue.
            return Ok(Primaries {
                chromaticities: self.chromaticities()?,
                to_xyz: math::IDENTITY,
                from_xyz: math::IDENTITY,
            });
        }
        Primaries::new(self.chromaticities()?)
    }
}

/// A set of primaries with its matrices to and from XYZ (white at `Y = 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Primaries {
    pub chromaticities: Chromaticities,
    to_xyz: Mat3,
    from_xyz: Mat3,
}

impl Primaries {
    /// # Errors
    ///
    /// [`Error::Invalid`] when a chromaticity has `y = 0` or the three
    /// primaries lie on one line.
    pub fn new(c: Chromaticities) -> Result<Self, Error> {
        let bad = Error::Invalid("degenerate primaries");
        let (r, g, b, w) = (
            c.red.xyz().ok_or(bad)?,
            c.green.xyz().ok_or(bad)?,
            c.blue.xyz().ok_or(bad)?,
            c.white.xyz().ok_or(bad)?,
        );
        let m = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
        let s = math::apply(&math::invert(&m).ok_or(bad)?, w);
        let to_xyz = math::mul(&m, &math::diag(s));
        let from_xyz = math::invert(&to_xyz).ok_or(bad)?;
        Ok(Self {
            chromaticities: c,
            to_xyz,
            from_xyz,
        })
    }

    /// Linear RGB to XYZ (RGB white to the white point at `Y = 1`).
    #[must_use]
    pub const fn to_xyz(&self) -> Mat3 {
        self.to_xyz
    }

    #[must_use]
    pub const fn from_xyz(&self) -> Mat3 {
        self.from_xyz
    }

    /// The luminance of each primary: `(KR, KG, KB)`, the row `Y` of
    /// [`to_xyz`](Self::to_xyz), summing to 1.
    #[must_use]
    pub const fn luminance(&self) -> [f64; 3] {
        self.to_xyz[1]
    }

    /// Linear RGB in `self` to linear RGB in `to`. Where the white points
    /// differ, `adaptation` says whether white stays white
    /// ([`Adaptation::Bradford`]) or keeps its XYZ (`None`, absolute
    /// colorimetry). Colours outside `to` come out below 0 or above 1.
    #[must_use]
    pub fn conversion(&self, to: &Self, adaptation: Adaptation) -> Mat3 {
        let adapt = match adaptation {
            Adaptation::None => math::IDENTITY,
            Adaptation::Bradford => bradford(self.chromaticities.white, to.chromaticities.white),
        };
        math::mul(&to.from_xyz, &math::mul(&adapt, &self.to_xyz))
    }
}

/// What happens to white when the white points differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Adaptation {
    /// White becomes white: the Bradford transform (ICC v4, ISO 22028).
    #[default]
    Bradford,
    /// XYZ is kept: the source white is shown as it measured.
    None,
}

/// The Bradford cone response matrix.
const BRADFORD: Mat3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

const BRADFORD_INVERSE: Mat3 = math::inverse_of(&BRADFORD);

/// XYZ under white `from` to XYZ under white `to`, by von Kries scaling in
/// the Bradford cone space.
#[must_use]
pub fn bradford(from: Xy, to: Xy) -> Mat3 {
    if from == to {
        return math::IDENTITY;
    }
    let (Some(src), Some(dst)) = (from.xyz(), to.xyz()) else {
        return math::IDENTITY;
    };
    let (s, d) = (math::apply(&BRADFORD, src), math::apply(&BRADFORD, dst));
    let scale = math::diag([d[0] / s[0], d[1] / s[1], d[2] / s[2]]);
    math::mul(&BRADFORD_INVERSE, &math::mul(&scale, &BRADFORD))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{close, mat_close};

    fn p(c: ColourPrimaries) -> Primaries {
        c.primaries().unwrap()
    }

    #[test]
    fn bt709_is_the_srgb_matrix() {
        // IEC 61966-2-1, to four places.
        let srgb = [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ];
        assert!(mat_close(&p(ColourPrimaries::Bt709).to_xyz(), &srgb, 1e-4));
    }

    #[test]
    fn luminance_rows_are_the_matrix_constants() {
        let k = |c| p(c).luminance();
        let [kr, _, kb] = k(ColourPrimaries::Bt709);
        assert!(close(kr, 0.2126, 1e-4) && close(kb, 0.0722, 1e-4));
        let [kr, _, kb] = k(ColourPrimaries::Bt2020);
        assert!(close(kr, 0.2627, 1e-4) && close(kb, 0.0593, 1e-4));
        for c in ColourPrimaries::ALL {
            if let Ok(prim) = c.primaries() {
                assert!(close(prim.luminance().iter().sum(), 1.0, 1e-12), "{c}");
            }
        }
    }

    #[test]
    fn bt709_to_bt2020_is_bt2087() {
        // BT.2087-0, equation 3 (four places).
        let want = [
            [0.6274, 0.3293, 0.0433],
            [0.0691, 0.9195, 0.0114],
            [0.0164, 0.0880, 0.8956],
        ];
        let m =
            p(ColourPrimaries::Bt709).conversion(&p(ColourPrimaries::Bt2020), Adaptation::Bradford);
        assert!(mat_close(&m, &want, 1e-4), "{m:?}");
        let back =
            p(ColourPrimaries::Bt2020).conversion(&p(ColourPrimaries::Bt709), Adaptation::Bradford);
        assert!(mat_close(&math::mul(&back, &m), &math::IDENTITY, 1e-12));
    }

    #[test]
    fn adaptation_keeps_white_white_and_none_keeps_xyz() {
        let dci = p(ColourPrimaries::Smpte431);
        let p3 = p(ColourPrimaries::Smpte432);
        let white = math::apply(&dci.conversion(&p3, Adaptation::Bradford), [1.0; 3]);
        assert!(white.iter().all(|&v| close(v, 1.0, 1e-12)), "{white:?}");
        let absolute = dci.conversion(&p3, Adaptation::None);
        let xyz_in = math::apply(&dci.to_xyz(), [1.0; 3]);
        let xyz_out = math::apply(&p3.to_xyz(), math::apply(&absolute, [1.0; 3]));
        assert!(xyz_in.iter().zip(xyz_out).all(|(&a, b)| close(a, b, 1e-12)));
        assert_eq!(bradford(D65, D65), math::IDENTITY);
        // Bradford maps the source white onto the target white.
        let m = bradford(C, D65);
        let c = math::apply(&m, C.xyz().unwrap());
        let d = D65.xyz().unwrap();
        assert!(c.iter().zip(d).all(|(&a, b)| close(a, b, 1e-12)));
    }

    #[test]
    fn xyz_primaries_are_the_identity() {
        let xyz = p(ColourPrimaries::Smpte428);
        assert_eq!(xyz.to_xyz(), math::IDENTITY);
        assert!(ColourPrimaries::Unspecified.primaries().is_err());
        let line = rgbw((0.1, 0.1), (0.2, 0.2), (0.3, 0.3), D65);
        assert!(Primaries::new(line).is_err());
    }
}
