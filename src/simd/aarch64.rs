//! AArch64: Advanced SIMD (NEON), through miraculix's `Neon` token.
//!
//! NEON is mandatory on AArch64 and enabled for every AArch64 target, so
//! the token's methods inline without a `target_feature` wrapper.
//!
//! - The matrix 4 pixels at a time, the vertical chroma passes 4 columns
//!   at a time: explicit NEON (`mul`, `add`, `sshr`, `sshl` with a
//!   negative count for the variable arithmetic shift, `smin`, `smax`).
//! - The horizontal chroma passes and the conversions to and from codes:
//!   the scalar row loops of [`fast`](crate::fast), which the compiler
//!   vectorizes with NEON anyway.

#![allow(
    clippy::too_many_arguments,
    reason = "planes and their geometry, one kernel each"
)]

use miraculix::aarch64::ops::neon::neon::Neon;

use super::Level;
use crate::fast::{self, Affine, DownPlan, LOW, UpPlan};

/// The proof the kernels take.
pub(super) type Token = Neon;

pub(super) const LEVEL: Level = Level::Neon;

pub(super) fn warm_up() {
    miraculix::aarch64::warm_up();
}

pub(super) fn detect() -> Option<Token> {
    Neon::detect()
}

/// Four lanes from `s[i..]`.
#[inline]
fn load(s: &[i32], i: usize) -> [i32; 4] {
    let mut v = [0; 4];
    v.copy_from_slice(&s[i..i + 4]);
    v
}

/// One output row of [`Affine`] for four pixels.
#[inline]
fn row(t: Neon, k: &Affine, r: usize, x: [[i32; 4]; 3]) -> [i32; 4] {
    let (h, l) = (k.high[r], k.low[r]);
    let mut high = [k.c[r]; 4];
    let mut low = [k.c_low[r]; 4];
    for j in 0..3 {
        high = t.add_i32x4(high, t.mul_i32x4([h[j]; 4], x[j]));
        low = t.add_i32x4(low, t.mul_i32x4([l[j]; 4], x[j]));
    }
    #[allow(clippy::cast_possible_wrap, reason = "LOW is 13")]
    let sum = t.add_i32x4(high, t.srai_i32x4::<{ LOW as i32 }>(low));
    // `sshl` by a negative count: arithmetic right shift (shifts are
    // below 32, where it is exact).
    #[allow(clippy::cast_possible_wrap, reason = "shifts are below 32")]
    let v = t.shl_i32x4(sum, [-(k.shift[r] as i32); 4]);
    t.max_i32x4(t.min_i32x4(v, [k.max[r]; 4]), [k.min[r]; 4])
}

pub(super) fn affine(
    t: Neon,
    k: &Affine,
    x0: &[i32],
    x1: &[i32],
    x2: &[i32],
    o0: &mut [i32],
    o1: &mut [i32],
    o2: &mut [i32],
) {
    let n = x0.len();
    let body = n / 4 * 4;
    let mut i = 0;
    while i < body {
        let x = [load(x0, i), load(x1, i), load(x2, i)];
        o0[i..i + 4].copy_from_slice(&row(t, k, 0, x));
        o1[i..i + 4].copy_from_slice(&row(t, k, 1, x));
        o2[i..i + 4].copy_from_slice(&row(t, k, 2, x));
        i += 4;
    }
    while i < n {
        let v = k.pixel([x0[i], x1[i], x2[i]]);
        (o0[i], o1[i], o2[i]) = (v[0], v[1], v[2]);
        i += 1;
    }
}

pub(super) fn up_v(t: Neon, a: &[i32], b: &[i32], wa: i32, wb: i32, out: &mut [i32]) {
    let n = out.len();
    let body = n / 4 * 4;
    let mut x = 0;
    while x < body {
        let v = t.add_i32x4(
            t.mul_i32x4([wa; 4], load(a, x)),
            t.mul_i32x4([wb; 4], load(b, x)),
        );
        out[x..x + 4].copy_from_slice(&v);
        x += 4;
    }
    while x < n {
        out[x] = wa * a[x] + wb * b[x];
        x += 1;
    }
}

pub(super) fn down_v(t: Neon, r: [&[i32]; 3], w: [i32; 3], out: &mut [i32]) {
    let n = out.len();
    let body = n / 4 * 4;
    let mut x = 0;
    while x < body {
        let mut v = t.mul_i32x4([w[0]; 4], load(r[0], x));
        v = t.add_i32x4(v, t.mul_i32x4([w[1]; 4], load(r[1], x)));
        v = t.add_i32x4(v, t.mul_i32x4([w[2]; 4], load(r[2], x)));
        out[x..x + 4].copy_from_slice(&v);
        x += 4;
    }
    while x < n {
        out[x] = w[0] * r[0][x] + w[1] * r[1][x] + w[2] * r[2][x];
        x += 1;
    }
}

pub(super) fn widen(_: Neon, src: &[u16], shift: u32, out: &mut [i32]) {
    fast::widen_row(src, shift, out);
}

pub(super) fn narrow(_: Neon, src: &[i32], max: u16, out: &mut [u16]) {
    fast::narrow_row(src, max, out);
}

pub(super) fn up_h(_: Neon, src: &[u16], plan: &UpPlan, out: &mut [i32]) {
    fast::up_h_row(src, plan, out);
}

pub(super) fn down_h(_: Neon, src: &[i32], plan: &DownPlan, frac: u32, max: u16, out: &mut [u16]) {
    fast::down_h_row(src, plan, frac, max, out);
}
