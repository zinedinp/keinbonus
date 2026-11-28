//! HDR processing: signals as absolute display light (cd/m², "nits"), the
//! HLG OOTF, and BT.2390 tone mapping. The HDR formats themselves (HDR10
//! and its static metadata, HDR10+, HLG, Dolby Vision) are in [`formats`].
//!
//! # Display light
//!
//! [`to_nits`] reads a signal as a reference display shows it, so that two
//! signals with the same nits look the same (display-referred conversion,
//! as BT.2408 and BT.2087's display-light method):
//!
//! | curve | display light |
//! |---|---|
//! | PQ | `10 000 * EOTF(V)`, absolute |
//! | SMPTE 428 | `48 * decode(V)`, absolute |
//! | HLG | the BT.2100 OOTF on scene light, peak [`Luminance::hlg_peak`] |
//! | BT.709 family, xvYCC, BT.1361 | BT.1886, `sdr_white * V^2.4` (black 0) |
//! | gamma 2.2, 2.8 | `sdr_white * V^gamma` |
//! | sRGB, SMPTE 240M, linear, log | `sdr_white * decode(V)` |
//!
//! `sdr_white` is where SDR white sits: 203 cd/m² when SDR goes into or
//! out of HDR (BT.2408), 100 for a classic SDR reference display.

use crate::math::{self, log10, pow, pow_odd};

pub mod formats;
#[cfg(feature = "display")]
pub mod itm;
#[cfg(feature = "display")]
pub mod shader;

use crate::transfer::Curve;
pub use formats::{Av1Mastering, ContentLight, HevcMastering, MasteringDisplay};

/// The luminances the conversions between SDR and HDR need.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Luminance {
    /// SDR reference white, cd/m².
    pub sdr_white: f64,
    /// The HLG display's peak, cd/m²: sets the OOTF's system gamma.
    pub hlg_peak: f64,
    /// The peak a PQ signal is made for (tone mapping into PQ), cd/m².
    pub pq_peak: f64,
}

impl Default for Luminance {
    /// BT.2408: 203 cd/m² SDR white, a 1000 cd/m² HLG display, PQ
    /// unlimited.
    fn default() -> Self {
        Self {
            sdr_white: 203.0,
            hlg_peak: 1000.0,
            pq_peak: 10_000.0,
        }
    }
}

/// BT.2100's HLG system gamma for a display of `peak` cd/m²:
/// `1.2 + 0.42 log10(peak / 1000)`.
#[must_use]
pub fn hlg_system_gamma(peak: f64) -> f64 {
    1.2 + 0.42 * log10(peak / 1000.0)
}

/// The BT.2100 HLG OOTF: scene light (0..1 per component) to display light
/// in cd/m², black 0. `luma` are the luminance weights of the primaries
/// (`Primaries::luminance`, BT.2020's for BT.2100).
#[must_use]
pub fn hlg_ootf(scene: [f64; 3], peak: f64, luma: [f64; 3]) -> [f64; 3] {
    let gamma = hlg_system_gamma(peak);
    let ys = dot(luma, scene).max(0.0);
    if ys == 0.0 {
        return [0.0; 3];
    }
    let gain = peak * pow(ys, gamma - 1.0);
    scene.map(|e| gain * e)
}

/// [`hlg_ootf`] back.
#[must_use]
pub fn hlg_inverse_ootf(display: [f64; 3], peak: f64, luma: [f64; 3]) -> [f64; 3] {
    let gamma = hlg_system_gamma(peak);
    let yd = dot(luma, display).max(0.0);
    if yd == 0.0 {
        return [0.0; 3];
    }
    let ys = pow(yd / peak, 1.0 / gamma);
    let gain = peak * pow(ys, gamma - 1.0);
    display.map(|f| f / gain)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A signal (`[R', G', B']` through `curve`) as display light in cd/m².
#[must_use]
pub fn to_nits(curve: Curve, signal: [f64; 3], lum: &Luminance, luma: [f64; 3]) -> [f64; 3] {
    let w = lum.sdr_white;
    match curve {
        Curve::Pq => signal.map(|v| 10_000.0 * curve.decode(v)),
        Curve::Smpte428 => signal.map(|v| 48.0 * curve.decode(v)),
        Curve::Hlg => hlg_ootf(signal.map(|v| curve.decode(v)), lum.hlg_peak, luma),
        Curve::Bt709 | Curve::Xvycc | Curve::Bt1361 => signal.map(|v| w * pow_odd(v, 2.4)),
        Curve::Gamma22 => signal.map(|v| w * pow_odd(v, 2.2)),
        Curve::Gamma28 => signal.map(|v| w * pow_odd(v, 2.8)),
        Curve::Srgb | Curve::Smpte240 | Curve::Linear | Curve::Log100 | Curve::Log316 => {
            signal.map(|v| w * curve.decode(v))
        }
    }
}

/// [`to_nits`] back.
#[must_use]
pub fn from_nits(curve: Curve, nits: [f64; 3], lum: &Luminance, luma: [f64; 3]) -> [f64; 3] {
    let w = lum.sdr_white;
    match curve {
        Curve::Pq => nits.map(|l| curve.encode(l / 10_000.0)),
        Curve::Smpte428 => nits.map(|l| curve.encode(l / 48.0)),
        Curve::Hlg => hlg_inverse_ootf(nits, lum.hlg_peak, luma).map(|e| curve.encode(e)),
        Curve::Bt709 | Curve::Xvycc | Curve::Bt1361 => nits.map(|l| pow_odd(l / w, 1.0 / 2.4)),
        Curve::Gamma22 => nits.map(|l| pow_odd(l / w, 1.0 / 2.2)),
        Curve::Gamma28 => nits.map(|l| pow_odd(l / w, 1.0 / 2.8)),
        Curve::Srgb | Curve::Smpte240 | Curve::Linear | Curve::Log100 | Curve::Log316 => {
            nits.map(|l| curve.encode(l / w))
        }
    }
}

/// The light of a full white signal through `curve`: where a target
/// clips, and the peak tone mapping aims for.
#[must_use]
pub fn peak(curve: Curve, lum: &Luminance) -> f64 {
    match curve {
        Curve::Pq => lum.pq_peak,
        _ => to_nits(curve, [1.0; 3], lum, [0.2627, 0.6780, 0.0593])[1],
    }
}

/// How light the target can't show is brought into its range.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ToneMap {
    /// Light is kept and clipped at the target's peak.
    #[default]
    Clip,
    /// BT.2390 EETF, per component in PQ, from a source mastered between
    /// `source_black` and `source_peak` cd/m² to the target's black (0)
    /// and [`peak`].
    Bt2390 { source_black: f64, source_peak: f64 },
}

/// BT.2390's EETF between two display ranges, in cd/m².
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eetf {
    pq_black: f64,
    pq_range: f64,
    ks: f64,
    max_lum: f64,
    min_lum: f64,
}

impl Eetf {
    /// From a source display (black, peak) to a target display (black,
    /// peak), all in cd/m².
    #[must_use]
    pub fn new(source_black: f64, source_peak: f64, target_black: f64, target_peak: f64) -> Self {
        let pq = |l: f64| Curve::Pq.encode(l / 10_000.0);
        let (pb, pw) = (pq(source_black), pq(source_peak));
        let range = pw - pb;
        let max_lum = (pq(target_peak) - pb) / range;
        Self {
            pq_black: pb,
            pq_range: range,
            ks: 1.5 * max_lum - 0.5,
            max_lum,
            min_lum: (pq(target_black) - pb) / range,
        }
    }

    /// One PQ signal through the EETF.
    #[must_use]
    pub fn apply_pq(&self, e: f64) -> f64 {
        if self.max_lum >= 1.0 || self.pq_range <= 0.0 {
            return e;
        }
        let e1 = math::clip3(0.0, 1.0, (e - self.pq_black) / self.pq_range);
        let e2 = if e1 < self.ks {
            e1
        } else {
            let t = (e1 - self.ks) / (1.0 - self.ks);
            let (t2, t3) = (t * t, t * t * t);
            (2.0 * t3 - 3.0 * t2 + 1.0) * self.ks
                + (t3 - 2.0 * t2 + t) * (1.0 - self.ks)
                + (-2.0 * t3 + 3.0 * t2) * self.max_lum
        };
        let e3 = e2 + self.min_lum * pow(1.0 - e2, 4.0);
        e3 * self.pq_range + self.pq_black
    }

    /// Display light through the EETF, per component.
    #[must_use]
    pub fn apply_nits(&self, nits: [f64; 3]) -> [f64; 3] {
        nits.map(|l| 10_000.0 * Curve::Pq.decode(self.apply_pq(Curve::Pq.encode(l / 10_000.0))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::close;

    const BT2020: [f64; 3] = [0.2627, 0.6780, 0.0593];

    #[test]
    fn bt2408_reference_white_is_58_percent_pq_and_75_percent_hlg() {
        let lum = Luminance::default();
        let white = 203.0;
        let pq = from_nits(Curve::Pq, [white; 3], &lum, BT2020)[0];
        let hlg = from_nits(Curve::Hlg, [white; 3], &lum, BT2020)[0];
        assert!(close(pq, 0.58, 0.001), "{pq}");
        assert!(close(hlg, 0.75, 0.001), "{hlg}");
        // And SDR white sits there.
        assert!(close(
            to_nits(Curve::Bt709, [1.0; 3], &lum, BT2020)[0],
            203.0,
            1e-12
        ));
    }

    #[test]
    fn hlg_ootf_has_bt2100s_gamma_and_inverts() {
        assert!(close(hlg_system_gamma(1000.0), 1.2, 1e-15));
        assert!(close(
            hlg_system_gamma(2000.0),
            1.2 + 0.42 * log10(2.0),
            1e-15
        ));
        // A white scene reaches the peak.
        let d = hlg_ootf([1.0; 3], 1000.0, BT2020);
        assert!(d.iter().all(|&v| close(v, 1000.0, 1e-9)));
        for scene in [[0.1, 0.5, 0.9], [0.0, 0.0, 0.2], [0.7, 0.7, 0.7]] {
            for peak in [400.0, 1000.0, 2000.0] {
                let back = hlg_inverse_ootf(hlg_ootf(scene, peak, BT2020), peak, BT2020);
                assert!(
                    back.iter().zip(scene).all(|(&a, b)| close(a, b, 1e-12)),
                    "{scene:?} {peak}"
                );
            }
        }
        assert_eq!(hlg_ootf([0.0; 3], 1000.0, BT2020), [0.0; 3]);
    }

    #[test]
    fn nits_round_trip_through_every_curve() {
        let lum = Luminance::default();
        for curve in [
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
        ] {
            let signal = [0.2, 0.5, 0.9];
            let back = from_nits(curve, to_nits(curve, signal, &lum, BT2020), &lum, BT2020);
            assert!(
                back.iter().zip(signal).all(|(&a, b)| close(a, b, 1e-12)),
                "{curve:?} {back:?}"
            );
        }
        // BT.2100 states c to 8 places: decode(1) is 1 to about 1e-7.
        assert!(close(peak(Curve::Hlg, &lum), 1000.0, 1e-3));
        assert!(close(peak(Curve::Srgb, &lum), 203.0, 1e-12));
        assert!(close(peak(Curve::Smpte428, &lum), 52.37, 1e-12));
        assert_eq!(peak(Curve::Pq, &lum), 10_000.0);
    }

    #[test]
    fn eetf_maps_peak_to_peak_and_leaves_the_low_end() {
        let eetf = Eetf::new(0.0, 4000.0, 0.0, 1000.0);
        let out = |nits: f64| eetf.apply_nits([nits; 3])[0];
        assert!(close(out(4000.0), 1000.0, 1e-9));
        // Above the source peak: held at the target peak.
        assert!(close(out(10_000.0), 1000.0, 1e-9));
        // Below the knee: unchanged.
        assert!(close(out(100.0), 100.0, 1e-9));
        assert!(close(out(0.0), 0.0, 1e-12));
        // Monotone.
        let mut last = 0.0;
        for i in 0..=400 {
            let v = out(f64::from(i) * 10.0);
            assert!(v >= last - 1e-12);
            last = v;
        }
        // A brighter target leaves everything as is.
        assert_eq!(Eetf::new(0.0, 1000.0, 0.0, 4000.0).apply_pq(0.6), 0.6);
        // A raised target black lifts black to it.
        let lifted = Eetf::new(0.0, 1000.0, 0.1, 500.0);
        assert!(close(lifted.apply_nits([0.0; 3])[0], 0.1, 1e-9));
    }
}
