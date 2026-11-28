//! The constants a GPU shader needs to run keinbonus's curves and maps in
//! `f32`, laid out for std140 uniform blocks (matrices as three `vec4`
//! rows). The shader is to be checked against the `f64` reference.

use super::Eetf;
use super::itm::MethodC;
use crate::math::Mat3;
use crate::transfer::{HLG_A, HLG_B, HLG_C, PQ_C1, PQ_C2, PQ_C3, PQ_M1, PQ_M2};

/// SMPTE ST 2084's constants: `m1, m2, c1, c2, c3`.
#[allow(clippy::cast_possible_truncation, reason = "the shader's f32")]
pub const PQ: [f32; 5] = [
    PQ_M1 as f32,
    PQ_M2 as f32,
    PQ_C1 as f32,
    PQ_C2 as f32,
    PQ_C3 as f32,
];

/// BT.2100 HLG's constants: `a, b, c`.
#[allow(clippy::cast_possible_truncation, reason = "the shader's f32")]
pub const HLG: [f32; 3] = [HLG_A as f32, HLG_B as f32, HLG_C as f32];

/// A 3x3 matrix as three std140 `vec4` rows (the fourth component 0).
#[must_use]
#[allow(clippy::cast_possible_truncation, reason = "the shader's f32")]
pub fn rows(m: &Mat3) -> [[f32; 4]; 3] {
    m.map(|r| [r[0] as f32, r[1] as f32, r[2] as f32, 0.0])
}

/// [`Eetf`]'s values in PQ: `pq_black, pq_range, ks, max_lum, min_lum`.
#[must_use]
#[allow(clippy::cast_possible_truncation, reason = "the shader's f32")]
pub fn eetf(e: &Eetf) -> [f32; 5] {
    [
        e.pq_black as f32,
        e.pq_range as f32,
        e.ks as f32,
        e.max_lum as f32,
        e.min_lum as f32,
    ]
}

/// [`MethodC`]'s curve: `k1, k2, k3, k4`, then the SDR and HDR
/// inflection points.
#[must_use]
#[allow(clippy::cast_possible_truncation, reason = "the shader's f32")]
pub fn method_c(c: &MethodC) -> [f32; 6] {
    [
        c.k1 as f32,
        c.k2 as f32,
        c.k3 as f32,
        c.k4 as f32,
        c.sdr_inflection as f32,
        c.hdr_inflection() as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::IDENTITY;
    use crate::transfer::Curve;

    #[test]
    fn the_f32_pq_matches_the_reference() {
        // The shader's formula, in f32, against `Curve::Pq`.
        let [m1, m2, c1, c2, c3] = PQ;
        for i in 0..=100u8 {
            let l = f64::from(i) / 100.0;
            let lf = f32::from(i) / 100.0;
            let p = lf.powf(m1);
            let e = ((c1 + c2 * p) / (1.0 + c3 * p)).powf(m2);
            assert!((f64::from(e) - Curve::Pq.encode(l)).abs() < 1e-5, "{l}");
        }
        assert_eq!(rows(&IDENTITY)[1], [0.0, 1.0, 0.0, 0.0]);
        let e = eetf(&Eetf::new(0.0, 1000.0, 0.0, 600.0));
        assert!(e[3] < 1.0 && e[1] > 0.0);
        assert_eq!(method_c(&MethodC::REPORT)[4], 58.5);
    }
}
