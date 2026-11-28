//! Tone mapping between SDR and HDR display light as a matched pair:
//! Report ITU-R BT.2446-1 Method C (§ 6). The inverse (SDR to HDR, § 6.2)
//! is the mathematical inverse of the tone mapping (§ 6.1) without its
//! optional chroma correction (§ 6.1.8), so a round trip is exact.
//!
//! Everything is linear BT.2020 display light in cd/m²: SDR as a 100 cd/m²
//! BT.1886 display shows it, HDR as a 1 000 cd/m² one does (the report's
//! HLG input through its EOTF, or any PQ signal through [`to_nits`]).
//!
//! [`to_nits`]: super::to_nits

use crate::cicp::ColourPrimaries;
use crate::math::{self, Mat3, exp, ln};

/// Method C's parameters: the tone curve (equation 5) and the crosstalk
/// matrix's `alpha` (equation 2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MethodC {
    /// The gain of the linear part.
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub k4: f64,
    /// The SDR luminance of the inflection point, cd/m² (80 % SDR).
    pub sdr_inflection: f64,
    /// The crosstalk matrix's de-saturation, `0..=0.33`.
    pub alpha: f64,
}

impl MethodC {
    /// Equation (10): skin tones at 50 % HLG and 70 % SDR, the inflection
    /// at 80 % SDR (58.5 cd/m²), HDR Reference White at 96 % SDR. The
    /// report gives the values rounded: they meet equations (7) and (8) to
    /// about 0.3 %, which [`derive`](Self::derive) shows. No crosstalk.
    pub const REPORT: Self = Self {
        k1: 0.838_02,
        k2: 15.099_68,
        k3: 0.742_04,
        k4: 78.994_39,
        sdr_inflection: 58.5,
        alpha: 0.0,
    };

    /// The parameters for other levels: gain `k1`, the inflection at
    /// `sdr_inflection` cd/m², and HDR Reference White (`hdr_white`)
    /// tone-mapped to `sdr_white`, all display light. `k3` solves
    /// equation (9), `k2` and `k4` follow from (7) and (8). `None` when no
    /// `k3` below 1 meets the condition.
    #[must_use]
    pub fn derive(k1: f64, sdr_inflection: f64, hdr_white: f64, sdr_white: f64) -> Option<Self> {
        let ip = sdr_inflection / k1;
        if k1 <= 0.0 || hdr_white <= ip || sdr_white <= sdr_inflection {
            return None;
        }
        // The SDR rise from the inflection to white the log part gives,
        // minus the one wanted: falls as k3 rises toward 1.
        let excess = |k3: f64| {
            let k2 = k1 * (1.0 - k3) * ip;
            k2 * (ln(hdr_white / ip - k3) - ln(1.0 - k3)) - (sdr_white - sdr_inflection)
        };
        let (mut low, mut high) = (-1.0e6, 1.0 - 1.0e-12);
        if excess(low) < 0.0 || excess(high) > 0.0 {
            return None;
        }
        for _ in 0..200 {
            let mid = f64::midpoint(low, high);
            if excess(mid) > 0.0 {
                low = mid;
            } else {
                high = mid;
            }
        }
        let k3 = f64::midpoint(low, high);
        let k2 = k1 * (1.0 - k3) * ip;
        Some(Self {
            k1,
            k2,
            k3,
            k4: sdr_inflection - k2 * ln(1.0 - k3),
            sdr_inflection,
            alpha: 0.0,
        })
    }

    /// Equation (6): the HDR luminance of the inflection point.
    #[must_use]
    pub fn hdr_inflection(&self) -> f64 {
        self.sdr_inflection / self.k1
    }

    /// Equation (5): HDR luminance to SDR luminance, cd/m².
    #[must_use]
    pub fn tone_map(&self, y_hdr: f64) -> f64 {
        let ip = self.hdr_inflection();
        if y_hdr < ip {
            self.k1 * y_hdr
        } else {
            self.k2 * ln(y_hdr / ip - self.k3) + self.k4
        }
    }

    /// Equation (5) solved for the HDR luminance (§ 6.2).
    #[must_use]
    pub fn inverse_tone_map(&self, y_sdr: f64) -> f64 {
        if y_sdr < self.sdr_inflection {
            y_sdr / self.k1
        } else {
            self.hdr_inflection() * (exp((y_sdr - self.k4) / self.k2) + self.k3)
        }
    }

    /// Equation (2).
    #[must_use]
    pub fn crosstalk(&self) -> Mat3 {
        let (a, d) = (self.alpha, 1.0 - 2.0 * self.alpha);
        [[d, a, a], [a, d, a], [a, a, d]]
    }

    /// Equation (14).
    #[must_use]
    pub fn inverse_crosstalk(&self) -> Mat3 {
        let a = self.alpha;
        let s = 1.0 / (1.0 - 3.0 * a);
        let (d, o) = (s * (1.0 - a), -s * a);
        [[d, o, o], [o, d, o], [o, o, d]]
    }

    /// § 6.1 without § 6.1.8: HDR display light to SDR display light.
    #[must_use]
    pub fn hdr_to_sdr(&self, rgb: [f64; 3]) -> [f64; 3] {
        self.through(rgb, |y| self.tone_map(y))
    }

    /// § 6.2: SDR display light to HDR display light.
    #[must_use]
    pub fn sdr_to_hdr(&self, rgb: [f64; 3]) -> [f64; 3] {
        self.through(rgb, |y| self.inverse_tone_map(y))
    }

    /// Crosstalk, Yxy (equations 3 and 4), `curve` on Y, back to RGB
    /// (11 to 13), inverse crosstalk: the same steps both ways.
    fn through(&self, rgb: [f64; 3], curve: impl Fn(f64) -> f64) -> [f64; 3] {
        let (to_xyz, from_xyz) = bt2020();
        let xyz = math::apply(&to_xyz, math::apply(&self.crosstalk(), rgb));
        let sum = xyz[0] + xyz[1] + xyz[2];
        if sum <= 0.0 || xyz[1] <= 0.0 {
            return [0.0; 3];
        }
        let (x, y) = (xyz[0] / sum, xyz[1] / sum);
        let mapped = curve(xyz[1]);
        let scale = mapped / y;
        let out = [x * scale, mapped, (1.0 - x - y) * scale];
        math::apply(&self.inverse_crosstalk(), math::apply(&from_xyz, out))
    }
}

/// BT.2020 to XYZ and back: equation (3) and (13) are these rounded to four
/// places.
fn bt2020() -> (Mat3, Mat3) {
    match ColourPrimaries::Bt2020.primaries() {
        Ok(p) => (p.to_xyz(), p.from_xyz()),
        Err(_) => (math::IDENTITY, math::IDENTITY),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::close;

    #[test]
    fn the_reports_parameters_are_its_conditions_rounded() {
        // 70 % SDR and 50 % HLG (BT.1886 at 100, HLG at 1 000 with gamma
        // 1.2): the gain.
        let sdr = 100.0 * math::pow(0.7, 2.4);
        let from_hlg = 50.697_028;
        assert!(close(sdr / from_hlg, MethodC::REPORT.k1, 1e-5));
        // 75 % HLG to 96 % SDR, the inflection at 80 % SDR.
        let white_sdr = 100.0 * math::pow(0.96, 2.4);
        let derived = MethodC::derive(MethodC::REPORT.k1, 58.5, 203.152, white_sdr).unwrap();
        let near = |a: f64, b: f64| (a - b).abs() / b < 3e-3;
        assert!(near(derived.k2, MethodC::REPORT.k2), "{derived:?}");
        assert!(near(derived.k3, MethodC::REPORT.k3), "{derived:?}");
        assert!(near(derived.k4, MethodC::REPORT.k4), "{derived:?}");
        // The derived curve is continuous at the inflection, in value and slope.
        let ip = derived.hdr_inflection();
        assert!(close(derived.tone_map(ip), 58.5, 1e-9));
        let slope = (derived.tone_map(ip + 1e-6) - derived.tone_map(ip)) / 1e-6;
        assert!(close(slope, derived.k1, 1e-5), "{slope}");
        assert!(close(derived.tone_map(203.152), white_sdr, 1e-6));
    }

    #[test]
    fn the_inverse_undoes_the_tone_map() {
        let c = MethodC::REPORT;
        for y in [0.0, 1.0, 40.0, 69.8, 69.81, 100.0, 203.0, 500.0, 1000.0] {
            assert!(close(c.inverse_tone_map(c.tone_map(y)), y, 1e-9), "{y}");
        }
        // SDR white of a 100 cd/m² display goes to about 332 cd/m².
        let top = c.inverse_tone_map(100.0);
        assert!((330.0..335.0).contains(&top), "{top}");
    }

    #[test]
    fn colours_round_trip_with_and_without_crosstalk() {
        for alpha in [0.0, 0.1, 0.33] {
            let c = MethodC { alpha, ..MethodC::REPORT };
            let id = math::mul(&c.inverse_crosstalk(), &c.crosstalk());
            assert!(id.iter().flatten().zip(math::IDENTITY.iter().flatten()).all(|(&a, &b)| close(a, b, 1e-12)));
            for rgb in [[100.0, 100.0, 100.0], [80.0, 20.0, 5.0], [1.0, 60.0, 30.0], [0.0; 3]] {
                let back = c.hdr_to_sdr(c.sdr_to_hdr(rgb));
                assert!(back.iter().zip(rgb).all(|(&a, b)| close(a, b, 1e-9)), "{alpha} {rgb:?} {back:?}");
            }
        }
        // Grey stays grey: only luminance changes.
        let grey = MethodC::REPORT.sdr_to_hdr([100.0; 3]);
        assert!(close(grey[0], grey[1], 1e-9) && close(grey[1], grey[2], 1e-9));
    }

    #[test]
    fn equation_3_is_bt2020_rounded() {
        let (to_xyz, _) = bt2020();
        let report = [[0.6370, 0.1446, 0.1689], [0.2627, 0.6780, 0.0593], [0.0, 0.0281, 1.0610]];
        assert!(to_xyz.iter().flatten().zip(report.iter().flatten()).all(|(&a, &b)| close(a, b, 5e-5)));
    }
}
