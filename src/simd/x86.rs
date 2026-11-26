//! x86 and x86-64: AVX2 (`vpmulld`, `vpsravd`, ...), through miraculix's
//! tokens and `avx2_fn!`, which compiles each kernel as one
//! `target_feature` function.
//!
//! - The matrix 8 pixels at a time, the vertical chroma passes 8 columns
//!   at a time: explicit AVX2.
//! - The horizontal chroma passes and the conversions to and from codes:
//!   the scalar row loops of [`fast`](crate::fast), compiled here with AVX2
//!   enabled (the same code, so the same results).

#![allow(
    clippy::too_many_arguments,
    reason = "planes and their geometry, one target_feature function each"
)]

use miraculix::x86::ops::avx::avx2::Avx2;

use super::Level;
use crate::fast::{self, Affine, DownPlan, LOW, UpPlan};

/// The proof the kernels take.
pub(super) type Token = Avx2;

pub(super) const LEVEL: Level = Level::Avx2;

pub(super) fn warm_up() {
    miraculix::x86::warm_up();
}

pub(super) fn detect() -> Option<Token> {
    Avx2::detect()
}

/// Eight lanes from `s[i..]`.
#[allow(clippy::inline_always, reason = "inside the target_feature kernels")]
#[inline(always)]
fn load(s: &[i32], i: usize) -> [i32; 8] {
    let mut v = [0; 8];
    v.copy_from_slice(&s[i..i + 8]);
    v
}

miraculix::avx2_fn! {
    /// One output row of [`Affine`] for eight pixels.
    fn row(t: Avx2, k: &Affine, r: usize, x: [[i32; 8]; 3]) -> [i32; 8] {
        let (h, l) = (k.high[r], k.low[r]);
        let mut high = [k.c[r]; 8];
        let mut low = [k.c_low[r]; 8];
        let mut j = 0;
        while j < 3 {
            high = t.add_i32x8(high, t.mul_i32x8([h[j]; 8], x[j]));
            low = t.add_i32x8(low, t.mul_i32x8([l[j]; 8], x[j]));
            j += 1;
        }
        let sum = t.add_i32x8(high, t.sra_i32x8::<LOW>(low));
        #[allow(clippy::cast_possible_wrap, reason = "shifts are below 32")]
        let v = t.srav_i32x8(sum, [k.shift[r] as i32; 8]);
        t.max_i32x8(t.min_i32x8(v, [k.max[r]; 8]), [k.min[r]; 8])
    }
}

miraculix::avx2_fn! {
    pub(super) fn affine(t: Avx2, k: &Affine, x0: &[i32], x1: &[i32], x2: &[i32], o0: &mut [i32], o1: &mut [i32], o2: &mut [i32]) {
        let n = x0.len();
        let body = n / 8 * 8;
        let mut i = 0;
        while i < body {
            let x = [load(x0, i), load(x1, i), load(x2, i)];
            o0[i..i + 8].copy_from_slice(&row(t, k, 0, x));
            o1[i..i + 8].copy_from_slice(&row(t, k, 1, x));
            o2[i..i + 8].copy_from_slice(&row(t, k, 2, x));
            i += 8;
        }
        while i < n {
            let v = k.pixel([x0[i], x1[i], x2[i]]);
            (o0[i], o1[i], o2[i]) = (v[0], v[1], v[2]);
            i += 1;
        }
    }
}

miraculix::avx2_fn! {
    pub(super) fn up_v(t: Avx2, a: &[i32], b: &[i32], wa: i32, wb: i32, out: &mut [i32]) {
        let n = out.len();
        let body = n / 8 * 8;
        let mut x = 0;
        while x < body {
            let v = t.add_i32x8(t.mul_i32x8([wa; 8], load(a, x)), t.mul_i32x8([wb; 8], load(b, x)));
            out[x..x + 8].copy_from_slice(&v);
            x += 8;
        }
        while x < n {
            out[x] = wa * a[x] + wb * b[x];
            x += 1;
        }
    }
}

miraculix::avx2_fn! {
    pub(super) fn down_v(t: Avx2, r: [&[i32]; 3], w: [i32; 3], out: &mut [i32]) {
        let n = out.len();
        let body = n / 8 * 8;
        let mut x = 0;
        while x < body {
            let mut v = t.mul_i32x8([w[0]; 8], load(r[0], x));
            v = t.add_i32x8(v, t.mul_i32x8([w[1]; 8], load(r[1], x)));
            v = t.add_i32x8(v, t.mul_i32x8([w[2]; 8], load(r[2], x)));
            out[x..x + 8].copy_from_slice(&v);
            x += 8;
        }
        while x < n {
            out[x] = w[0] * r[0][x] + w[1] * r[1][x] + w[2] * r[2][x];
            x += 1;
        }
    }
}

miraculix::avx2_fn! {
    pub(super) fn widen(t: Avx2, src: &[u16], shift: u32, out: &mut [i32]) {
        let _ = t;
        fast::widen_row(src, shift, out);
    }
}

miraculix::avx2_fn! {
    pub(super) fn narrow(t: Avx2, src: &[i32], max: u16, out: &mut [u16]) {
        let _ = t;
        fast::narrow_row(src, max, out);
    }
}

miraculix::avx2_fn! {
    pub(super) fn up_h(t: Avx2, src: &[u16], plan: &UpPlan, out: &mut [i32]) {
        let _ = t;
        fast::up_h_row(src, plan, out);
    }
}

miraculix::avx2_fn! {
    pub(super) fn down_h(t: Avx2, src: &[i32], plan: &DownPlan, frac: u32, max: u16, out: &mut [u16]) {
        let _ = t;
        fast::down_h_row(src, plan, frac, max, out);
    }
}
