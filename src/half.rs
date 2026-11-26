//! IEEE 754 binary16 (half), as FP16 swapchains and textures (scRGB,
//! `R16G16B16A16_SFLOAT`) store colour: from `f64` rounded to nearest, ties
//! to even, in one step (through `f32` would round twice), and back exactly.

/// The half nearest `x` (ties to even); overflow to infinity, NaN stays a
/// (quiet) NaN.
#[must_use]
pub fn from_f64(x: f64) -> u16 {
    let bits = x.to_bits();
    #[allow(clippy::cast_possible_truncation, reason = "the sign bit")]
    let sign = ((bits >> 48) & 0x8000) as u16;
    #[allow(clippy::cast_possible_truncation, reason = "11 bits")]
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let man = bits & ((1 << 52) - 1);
    if exp == 0x7ff {
        return sign | 0x7c00 | if man == 0 { 0 } else { 0x200 };
    }
    if exp == 0 {
        // f64 subnormals are far below the smallest half.
        return sign;
    }
    let e = exp - 1023;
    if e > 15 {
        return sign | 0x7c00;
    }
    let m = man | (1 << 52);
    // Normal halves keep 11 significant bits; below 2^-14 the half is
    // subnormal, with a fixed exponent of -24 for its last bit.
    let shift = if e >= -14 { 42 } else { 42 + (-14 - e) };
    if shift > 63 {
        return sign;
    }
    #[allow(clippy::cast_sign_loss, reason = "42..=63")]
    let shift = shift as u32;
    let mut q = m >> shift;
    let rem = m & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    if rem > half || (rem == half && q & 1 == 1) {
        q += 1;
    }
    let out = if e >= -14 {
        // q is 1024..=2048; 2048 carries into the exponent.
        #[allow(clippy::cast_sign_loss, reason = "e + 15 is 1..=30")]
        let biased = (e + 15) as u64;
        (biased << 10) + q - 1024
    } else {
        q
    };
    if out >= 0x7c00 {
        return sign | 0x7c00;
    }
    #[allow(clippy::cast_possible_truncation, reason = "below 0x7c00")]
    let out = out as u16;
    sign | out
}

/// The value of half `h`, exactly.
#[must_use]
pub fn to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f64::from(h & 0x3ff);
    sign * match exp {
        0 => man * libm::exp2(-24.0),
        31 if man == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1024.0 + man) * libm::exp2(f64::from(exp - 25)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_half_round_trips() {
        for h in 0..=u16::MAX {
            let x = to_f64(h);
            if x.is_nan() {
                assert!(to_f64(from_f64(x)).is_nan());
            } else {
                assert_eq!(from_f64(x), h, "{h:#06x} = {x}");
            }
        }
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the f32 detour is the point"
    )]
    fn rounding_is_to_nearest_even() {
        let p = |e: i32| libm::exp2(f64::from(e));
        assert_eq!(from_f64(1.0), 0x3c00);
        // Halfway between 1 and 1 + 2^-10: to 1 (even).
        assert_eq!(from_f64(1.0 + p(-11)), 0x3c00);
        // Halfway between 1 + 2^-10 and 1 + 2^-9: to 1 + 2^-9 (even).
        assert_eq!(from_f64(1.0 + 3.0 * p(-11)), 0x3c02);
        // Just above halfway: up.
        assert_eq!(from_f64(1.0 + p(-11) + p(-40)), 0x3c01);
        // Largest finite, and the overflow threshold 65520.
        assert_eq!(from_f64(65504.0), 0x7bff);
        assert_eq!(from_f64(65519.99), 0x7bff);
        assert_eq!(from_f64(65520.0), 0x7c00);
        // Subnormals: 2^-24 is the smallest; half of it ties to 0.
        assert_eq!(from_f64(p(-24)), 0x0001);
        assert_eq!(from_f64(p(-25)), 0x0000);
        assert_eq!(from_f64(p(-25) * 1.000_001), 0x0001);
        assert_eq!(from_f64(1.5 * p(-24)), 0x0002);
        // The largest subnormal rounds up into the smallest normal.
        assert_eq!(from_f64(p(-14) - p(-26)), 0x0400);
        assert_eq!(from_f64(-2.0), 0xc000);
        assert_eq!(from_f64(-0.0), 0x8000);
        assert_eq!(from_f64(f64::INFINITY), 0x7c00);
        assert_eq!(from_f64(1e-300), 0);
        // Double rounding through f32 would get this one wrong: it is
        // just above a half's midpoint, but rounds to the midpoint in f32.
        let x = 1.0 + p(-11) + p(-30);
        assert_eq!(from_f64(x), 0x3c01);
        assert_eq!(from_f64(f64::from(x as f32)), 0x3c00);
    }
}
