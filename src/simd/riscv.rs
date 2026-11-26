//! RISC-V: the V extension (RVV 1.0), as miraculix reaches it: inline
//! assembly strip-mined with `vsetvli`, so it is correct for any VLEN, and
//! a `Rvv` token from miraculix's detection as the proof.
//!
//! - The matrix one output row at a time (eleven scalars: three high and
//!   three low coefficients, two offsets, shift, clip bounds), the vertical
//!   chroma passes: RVV (`vmacc.vx`, `vsra`, `vmin`/`vmax`), 32-bit lanes,
//!   LMUL 4 or 8. Multiplies keep the low 32 bits, as the scalar wrapping
//!   arithmetic does.
//! - The horizontal chroma passes and the conversions to and from codes:
//!   the scalar row loops of [`fast`](crate::fast).
//!
//! Built for riscv32 and riscv64 here; the equality tests against the
//! scalar kernels run on a RISC-V machine with V.

#![allow(
    clippy::too_many_arguments,
    reason = "planes and their geometry, one kernel each"
)]

use core::arch::asm;

use miraculix::riscv::ops::rvv::Rvv;

use super::Level;
use crate::fast::{self, Affine, DownPlan, LOW, UpPlan};

/// The proof the kernels take.
pub(super) type Token = Rvv;

pub(super) const LEVEL: Level = Level::Rvv;

pub(super) fn warm_up() {
    miraculix::riscv::warm_up();
}

pub(super) fn detect() -> Option<Token> {
    Rvv::detect()
}

/// `asm!` with every vector register clobbered (miraculix's `rvv_asm!`).
macro_rules! rvv_asm {
    ($($tt:tt)*) => {
        asm!($($tt)*,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
            out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
            out("v30") _, out("v31") _,
            options(nostack))
    };
}

/// One output row `r` of [`Affine`] over `out.len()` pixels.
fn affine_row(_: Rvv, k: &Affine, r: usize, x0: &[i32], x1: &[i32], x2: &[i32], out: &mut [i32]) {
    let n = out.len();
    assert!(x0.len() == n && x1.len() == n && x2.len() == n);
    if n == 0 {
        return;
    }
    let ([h0, h1, h2], [l0, l1, l2]) = (k.high[r], k.low[r]);
    #[allow(clippy::cast_possible_wrap, reason = "shifts are below 32")]
    let shift = k.shift[r] as i32;
    let (mut p0, mut p1, mut p2, mut po, mut n) =
        (x0.as_ptr(), x1.as_ptr(), x2.as_ptr(), out.as_mut_ptr(), n);
    // SAFETY: the token proves V; the four slices hold `n` i32s each, the
    // loop reads `n` from each input and writes `n` to `out`.
    unsafe {
        rvv_asm!(
            ".option push",
            ".option arch, +v",
            "2:",
            "vsetvli {vl}, {n}, e32, m4, ta, ma",
            "vle32.v v4, ({p0})",
            "vle32.v v8, ({p1})",
            "vle32.v v12, ({p2})",
            // high = c + h0 x0 + h1 x1 + h2 x2
            "vmv.v.x v16, {c}",
            "vmacc.vx v16, {h0}, v4",
            "vmacc.vx v16, {h1}, v8",
            "vmacc.vx v16, {h2}, v12",
            // low = c_low + l0 x0 + l1 x1 + l2 x2
            "vmv.v.x v20, {cl}",
            "vmacc.vx v20, {l0}, v4",
            "vmacc.vx v20, {l1}, v8",
            "vmacc.vx v20, {l2}, v12",
            // (high + (low >> LOW)) >> shift, clamped
            "vsra.vi v20, v20, {low}",
            "vadd.vv v16, v16, v20",
            "vsra.vx v16, v16, {sh}",
            "vmin.vx v16, v16, {hi}",
            "vmax.vx v16, v16, {lo}",
            "vse32.v v16, ({po})",
            "sub {n}, {n}, {vl}",
            "slli {t}, {vl}, 2",
            "add {p0}, {p0}, {t}",
            "add {p1}, {p1}, {t}",
            "add {p2}, {p2}, {t}",
            "add {po}, {po}, {t}",
            "bnez {n}, 2b",
            ".option pop",
            p0 = inout(reg) p0, p1 = inout(reg) p1, p2 = inout(reg) p2, po = inout(reg) po, n = inout(reg) n,
            c = in(reg) k.c[r], cl = in(reg) k.c_low[r],
            h0 = in(reg) h0, h1 = in(reg) h1, h2 = in(reg) h2,
            l0 = in(reg) l0, l1 = in(reg) l1, l2 = in(reg) l2,
            sh = in(reg) shift, hi = in(reg) k.max[r], lo = in(reg) k.min[r],
            low = const LOW,
            vl = out(reg) _, t = out(reg) _
        );
    }
    let _ = (p0, p1, p2, po, n);
}

pub(super) fn affine(
    t: Rvv,
    k: &Affine,
    x0: &[i32],
    x1: &[i32],
    x2: &[i32],
    o0: &mut [i32],
    o1: &mut [i32],
    o2: &mut [i32],
) {
    affine_row(t, k, 0, x0, x1, x2, o0);
    affine_row(t, k, 1, x0, x1, x2, o1);
    affine_row(t, k, 2, x0, x1, x2, o2);
}

pub(super) fn up_v(_: Rvv, a: &[i32], b: &[i32], wa: i32, wb: i32, out: &mut [i32]) {
    let n = out.len();
    assert!(a.len() >= n && b.len() >= n);
    if n == 0 {
        return;
    }
    let (mut pa, mut pb, mut po, mut n) = (a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n);
    // SAFETY: the token proves V; reads `n` i32s of `a` and `b`, writes
    // `n` to `out`.
    unsafe {
        rvv_asm!(
            ".option push",
            ".option arch, +v",
            "2:",
            "vsetvli {vl}, {n}, e32, m8, ta, ma",
            "vle32.v v8, ({pa})",
            "vle32.v v16, ({pb})",
            "vmul.vx v8, v8, {wa}",
            "vmacc.vx v8, {wb}, v16",
            "vse32.v v8, ({po})",
            "sub {n}, {n}, {vl}",
            "slli {t}, {vl}, 2",
            "add {pa}, {pa}, {t}",
            "add {pb}, {pb}, {t}",
            "add {po}, {po}, {t}",
            "bnez {n}, 2b",
            ".option pop",
            pa = inout(reg) pa, pb = inout(reg) pb, po = inout(reg) po, n = inout(reg) n,
            wa = in(reg) wa, wb = in(reg) wb, vl = out(reg) _, t = out(reg) _
        );
    }
    let _ = (pa, pb, po, n);
}

pub(super) fn down_v(_: Rvv, r: [&[i32]; 3], w: [i32; 3], out: &mut [i32]) {
    let n = out.len();
    assert!(r.iter().all(|row| row.len() >= n));
    if n == 0 {
        return;
    }
    let (mut p0, mut p1, mut p2, mut po, mut n) = (
        r[0].as_ptr(),
        r[1].as_ptr(),
        r[2].as_ptr(),
        out.as_mut_ptr(),
        n,
    );
    // SAFETY: the token proves V; reads `n` i32s of each row, writes `n` to
    // `out`.
    unsafe {
        rvv_asm!(
            ".option push",
            ".option arch, +v",
            "2:",
            "vsetvli {vl}, {n}, e32, m4, ta, ma",
            "vle32.v v4, ({p0})",
            "vle32.v v8, ({p1})",
            "vle32.v v12, ({p2})",
            "vmul.vx v16, v4, {w0}",
            "vmacc.vx v16, {w1}, v8",
            "vmacc.vx v16, {w2}, v12",
            "vse32.v v16, ({po})",
            "sub {n}, {n}, {vl}",
            "slli {t}, {vl}, 2",
            "add {p0}, {p0}, {t}",
            "add {p1}, {p1}, {t}",
            "add {p2}, {p2}, {t}",
            "add {po}, {po}, {t}",
            "bnez {n}, 2b",
            ".option pop",
            p0 = inout(reg) p0, p1 = inout(reg) p1, p2 = inout(reg) p2, po = inout(reg) po, n = inout(reg) n,
            w0 = in(reg) w[0], w1 = in(reg) w[1], w2 = in(reg) w[2], vl = out(reg) _, t = out(reg) _
        );
    }
    let _ = (p0, p1, p2, po, n);
}

pub(super) fn widen(_: Rvv, src: &[u16], shift: u32, out: &mut [i32]) {
    fast::widen_row(src, shift, out);
}

pub(super) fn narrow(_: Rvv, src: &[i32], max: u16, out: &mut [u16]) {
    fast::narrow_row(src, max, out);
}

pub(super) fn up_h(_: Rvv, src: &[u16], plan: &UpPlan, out: &mut [i32]) {
    fast::up_h_row(src, plan, out);
}

pub(super) fn down_h(_: Rvv, src: &[i32], plan: &DownPlan, frac: u32, max: u16, out: &mut [u16]) {
    fast::down_h_row(src, plan, frac, max, out);
}
