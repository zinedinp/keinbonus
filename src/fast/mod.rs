//! Fixed-point fast paths, scalar: the definition the SIMD kernels of
//! [`simd`](crate::simd) (feature `simd`) reproduce bit for bit.
//!
//! [`Pipeline`] covers every conversion that is affine on the codes: both
//! sides share curve and primaries and use a linear matrix (GBR, NCL,
//! YCgCo), at bit depths up to [`MAX_BITS`]. That is RGB to YCbCr and
//! back, any matrix to any other of those, range and bit depth changes,
//! and any chroma format to any other.
//!
//! How close it is to the [reference](crate::convert):
//!
//! - Chroma up- and downsampling is exact: the weights are quarters, so
//!   integer sums give the reference's values, rounded the same way.
//! - The colour matrix runs with each coefficient split into a high and
//!   a low `i32` part ([`Affine`]), about 2^-30 of the output scale
//!   together: a code differs from the reference only where the exact
//!   value lies that close to a rounding boundary, and then by 1.
//! - Into a subsampled format the matrix output is kept to 1/256 of a
//!   code before the downsampling sum, which rounds once more: again at
//!   most 1 off, near boundaries.
//!
//! The tests measure both against the reference.

// The row kernels are `#[inline(always)]` on purpose: `simd` wraps the
// same loops in AVX2 `target_feature` functions, and only inlined do they
// get compiled with AVX2 there.
#![allow(
    clippy::inline_always,
    reason = "the row loops inline into the SIMD target_feature wrappers"
)]

pub mod light;

use alloc::vec;
use alloc::vec::Vec;

use crate::chroma::{ChromaFormat, ChromaLocation, Upsample};
use crate::convert::{Converter, Format, Options};
use crate::image::Image;
use crate::math::{self, Mat3};
use crate::quant::{BitDepth, Range, Signal};
use crate::{Colour, Error};

/// The most fractional bits of a coefficient's high part; each output row
/// takes as many as its sums allow in `i32` (at least [`MIN_FRAC`]).
pub const MAX_FRAC: u32 = 24;
/// The fewest: below, the fast path refuses.
pub const MIN_FRAC: u32 = 12;
/// Bits of a coefficient's low part: the low sums of three inputs up to
/// 2^16 stay below 2^31.
pub const LOW: u32 = 13;
/// The deepest codes the fast paths take (inputs up to 2^16 with
/// [`FINE`]).
pub const MAX_BITS: u8 = 12;
/// Fractional bits of the planes between the stages (chroma upsampled to
/// full resolution, inputs of the matrix).
pub const FINE: u32 = 4;
/// Fractional bits of colour differences between the matrix and the
/// downsampling.
pub const FINE_OUT: u32 = 8;

/// A 3x3 matrix with offsets in two-part fixed point, `i32` throughout.
///
/// Each coefficient is `a = hi * 2^LOW + lo` at `FRAC + LOW` fractional
/// bits, `0 <= lo < 2^LOW`. With `H = c + sum hi_j x_j` and
/// `L = c_lo + sum lo_j x_j` (inputs `x >= 0`), the exact sum is
/// `H * 2^LOW + L`, and since `L >> LOW` takes all of `L` that reaches
/// `H`'s scale,
///
/// `out = floor((H * 2^LOW + L) / 2^(LOW + shift)) = (H + (L >> LOW)) >> shift`,
///
/// clamped to `lo..=hi`. [`Affine::new`] makes sure both sums fit, so the
/// wrapping arithmetic used gives the exact values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Affine {
    pub high: [[i32; 3]; 3],
    pub low: [[i32; 3]; 3],
    /// High part of the offsets, the rounding half included.
    pub c: [i32; 3],
    pub c_low: [i32; 3],
    pub shift: [u32; 3],
    pub min: [i32; 3],
    pub max: [i32; 3],
}

impl Affine {
    /// The fixed-point form of `out = a * in + c` (real numbers, inputs
    /// and outputs in their own units), each output kept to `out_frac[k]`
    /// fractional bits and clipped to `0..=hi[k]` where given. Inputs lie
    /// in `0..=in_max[j]` (at most 2^16). Each row takes the most bits
    /// ([`MAX_FRAC`] down to [`MIN_FRAC`]) for which its sums over that box
    /// stay in `i32`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when not even [`MIN_FRAC`] bits fit, or an
    /// input exceeds 2^16.
    pub fn new(
        a: &Mat3,
        c: [f64; 3],
        in_max: [i64; 3],
        out_frac: [u32; 3],
        hi: [Option<i32>; 3],
    ) -> Result<Self, Error> {
        if in_max.iter().any(|&m| !(0..=1 << 16).contains(&m)) {
            return Err(Error::Unsupported("fixed point range"));
        }
        let mut fixed = Self {
            high: [[0; 3]; 3],
            low: [[0; 3]; 3],
            c: [0; 3],
            c_low: [0; 3],
            shift: [0; 3],
            min: [0; 3],
            max: [0; 3],
        };
        for k in 0..3 {
            let r = (MIN_FRAC.max(out_frac[k] + 1)..=MAX_FRAC)
                .rev()
                .find_map(|frac| Row::new(&a[k], c[k], in_max, frac, frac - out_frac[k]))
                .ok_or(Error::Unsupported("fixed point range"))?;
            (
                fixed.high[k],
                fixed.low[k],
                fixed.c[k],
                fixed.c_low[k],
                fixed.shift[k],
            ) = (r.high, r.low, r.c, r.c_low, r.shift);
            (fixed.min[k], fixed.max[k]) = match hi[k] {
                Some(h) => (0, h),
                None => (i32::MIN, i32::MAX),
            };
        }
        Ok(fixed)
    }

    /// One pixel.
    #[must_use]
    #[inline]
    pub const fn pixel(&self, x: [i32; 3]) -> [i32; 3] {
        let mut out = [0; 3];
        let mut k = 0;
        while k < 3 {
            let (h, l) = (self.high[k], self.low[k]);
            let high = self.c[k]
                .wrapping_add(h[0].wrapping_mul(x[0]))
                .wrapping_add(h[1].wrapping_mul(x[1]))
                .wrapping_add(h[2].wrapping_mul(x[2]));
            let low = self.c_low[k]
                .wrapping_add(l[0].wrapping_mul(x[0]))
                .wrapping_add(l[1].wrapping_mul(x[1]))
                .wrapping_add(l[2].wrapping_mul(x[2]));
            let v = high.wrapping_add(low >> LOW) >> self.shift[k];
            out[k] = if v < self.min[k] {
                self.min[k]
            } else if v > self.max[k] {
                self.max[k]
            } else {
                v
            };
            k += 1;
        }
        out
    }
}

/// One output row in two-part fixed point.
struct Row {
    high: [i32; 3],
    low: [i32; 3],
    c: i32,
    c_low: i32,
    shift: u32,
}

impl Row {
    /// At `frac` high-part bits, if both sums fit in `i32` over the box.
    fn new(a: &[f64; 3], c: f64, in_max: [i64; 3], frac: u32, shift: u32) -> Option<Self> {
        let one = f64::from(1u32 << frac) * f64::from(1u32 << LOW);
        let split = |v: f64| -> Option<(i32, i32)> {
            if v.abs() >= 9.0e15 {
                return None;
            }
            #[allow(clippy::cast_possible_truncation, reason = "below 2^53")]
            let v = math::round(v * one) as i64;
            let (h, l) = (v >> LOW, v & ((1 << LOW) - 1));
            Some((i32::try_from(h).ok()?, i32::try_from(l).ok()?))
        };
        let (c_high, c_low) = split(c)?;
        let c = c_high.checked_add(1 << (shift - 1))?;
        let (mut hi_min, mut hi_max) = (i64::from(c), i64::from(c));
        let mut low_max = i64::from(c_low);
        let (mut high, mut low) = ([0; 3], [0; 3]);
        for j in 0..3 {
            (high[j], low[j]) = split(a[j])?;
            let reach = i64::from(high[j]) * in_max[j];
            if reach < 0 {
                hi_min += reach;
            } else {
                hi_max += reach;
            }
            low_max += i64::from(low[j]) * in_max[j];
        }
        let fits = |v: i64| (i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&v);
        (fits(low_max) && fits(hi_min) && fits(hi_max + (low_max >> LOW))).then_some(Self {
            high,
            low,
            c,
            c_low,
            shift,
        })
    }
}

/// [`Affine::pixel`] over planes of equal length.
///
/// # Panics
///
/// When the planes differ in length.
pub fn affine(k: &Affine, input: [&[i32]; 3], out: [&mut [i32]; 3]) {
    let n = input[0].len();
    assert!(input.iter().all(|p| p.len() == n) && out.iter().all(|p| p.len() == n));
    let [o0, o1, o2] = out;
    for i in 0..n {
        let v = k.pixel([input[0][i], input[1][i], input[2][i]]);
        (o0[i], o1[i], o2[i]) = (v[0], v[1], v[2]);
    }
}

/// Twice the chroma offset of `location` along each axis of `format` (0,
/// 1 or 2: on a sample, between two, on the next).
fn offsets2(format: ChromaFormat, location: ChromaLocation) -> (i64, i64) {
    let (ox, oy) = location.offset(format);
    #[allow(clippy::cast_possible_truncation, reason = "0, 1 or 2")]
    let twice = |o: f64| (o * 2.0) as i64;
    (twice(ox), twice(oy))
}

/// The downsampling taps of output `i`, weights in quarters (sum 4).
#[must_use]
pub fn down_taps(i: usize, factor: usize, o2: i64, n: usize) -> [(usize, i32); 3] {
    let i = i64::try_from(i).unwrap_or(i64::MAX);
    let at = |k: i64| clamp(k, n);
    if factor == 1 {
        return [(at(i), 4), (0, 0), (0, 0)];
    }
    let c = 2 * i;
    match o2 {
        0 => [(at(c - 1), 1), (at(c), 2), (at(c + 1), 1)],
        1 => [(at(c), 2), (at(c + 1), 2), (0, 0)],
        _ => [(at(c), 1), (at(c + 1), 2), (at(c + 2), 1)],
    }
}

/// The upsampling taps of full-resolution sample `x`, weights in quarters
/// (sum 4).
#[must_use]
pub fn up_taps(x: usize, factor: usize, o2: i64, n: usize, filter: Upsample) -> [(usize, i32); 2] {
    let x = i64::try_from(x).unwrap_or(i64::MAX);
    // The chroma coordinate in quarters: 4 (x - o) / factor.
    let u4 = if factor == 1 { 4 * x } else { 2 * x - o2 };
    match filter {
        Upsample::Nearest => [(clamp((u4 + 2).div_euclid(4), n), 4), (0, 0)],
        Upsample::Linear => {
            let i0 = u4.div_euclid(4);
            #[allow(clippy::cast_possible_truncation, reason = "0..4")]
            let t = u4.rem_euclid(4) as i32;
            [(clamp(i0, n), 4 - t), (clamp(i0 + 1, n), t)]
        }
    }
}

fn clamp(k: i64, n: usize) -> usize {
    usize::try_from(k).map_or(0, |k| k.min(n.saturating_sub(1)))
}

/// The interior of a factor-2 pass: tap offsets relative to the input
/// index of pair `i` (`i` for upsampling, outputs `2i + p` for parity `p`;
/// `2i` for downsampling, output `i`) and the pairs `i` where the taps
/// repeat unclamped. There the loop has constant offsets and weights,
/// which compilers vectorize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Interior<const T: usize, const P: usize> {
    pub rel: [[(isize, i32); T]; P],
    pub range: core::ops::Range<usize>,
}

impl<const T: usize, const P: usize> Interior<T, P> {
    /// From the taps of every output (`P` outputs per pair: 2 when
    /// upsampling, 1 when downsampling); `None` without an interior.
    fn new(taps: &[[(usize, i32); T]], inputs: usize) -> Option<Self> {
        let pairs = taps.len() / P;
        let mid = pairs / 2;
        if mid < 2 {
            return None;
        }
        let step = if P == 2 { 1 } else { 2 };
        let mut rel = [[(0isize, 0i32); T]; P];
        for (p, r) in rel.iter_mut().enumerate() {
            for (t, &(k, w)) in r.iter_mut().zip(&taps[P * mid + p]) {
                *t = (
                    isize::try_from(k).ok()? - isize::try_from(step * mid).ok()?,
                    w,
                );
            }
            // A tap of weight 0 reads the first tap's sample: its own
            // index is a placeholder.
            let first = r[0].0;
            for t in r.iter_mut().skip(1) {
                if t.1 == 0 {
                    t.0 = first;
                }
            }
        }
        let lo_off = rel.iter().flatten().map(|t| t.0).min()?;
        let hi_off = rel.iter().flatten().map(|t| t.0).max()?;
        // step * i + lo_off >= 0 and step * i + hi_off <= inputs - 1.
        let lo = usize::try_from((-lo_off).max(0)).ok()?.div_ceil(step);
        let top = isize::try_from(inputs).ok()? - 1 - hi_off;
        if top < 0 {
            return None;
        }
        let hi = (usize::try_from(top).ok()? / step + 1).min(pairs);
        let interior = Self { rel, range: lo..hi };
        // The repeating taps must be the taps there.
        let fits = interior.range.clone().all(|i| {
            (0..P).all(|p| {
                taps[P * i + p]
                    .iter()
                    .zip(&interior.rel[p])
                    .all(|(&(k, w), &(d, rw))| {
                        w == rw && (w == 0 || (step * i).checked_add_signed(d) == Some(k))
                    })
            })
        });
        (lo < hi && fits).then_some(interior)
    }
}

/// The taps of every output column and row of an upsampling.
pub struct UpPlan {
    pub(crate) cw: usize,
    pub(crate) ch: usize,
    pub(crate) x: Vec<[(usize, i32); 2]>,
    pub(crate) y: Vec<[(usize, i32); 2]>,
    pub(crate) inner: Option<Interior<2, 2>>,
}

impl UpPlan {
    /// # Errors
    ///
    /// [`Error::Size`] when `plane` isn't the subsampled size.
    pub fn new(
        plane: usize,
        width: usize,
        height: usize,
        format: ChromaFormat,
        location: ChromaLocation,
        filter: Upsample,
    ) -> Result<Self, Error> {
        let (sx, sy) = format.factors();
        let (cw, ch) = format.plane(width, height);
        if plane != cw * ch || width == 0 || height == 0 {
            return Err(Error::Size);
        }
        let (ox2, oy2) = offsets2(format, location);
        let x: Vec<_> = (0..width)
            .map(|x| up_taps(x, sx, ox2, cw, filter))
            .collect();
        let inner = if sx == 2 { Interior::new(&x, cw) } else { None };
        Ok(Self {
            cw,
            ch,
            x,
            y: (0..height)
                .map(|y| up_taps(y, sy, oy2, ch, filter))
                .collect(),
            inner,
        })
    }
}

/// The taps of every output column and row of a downsampling.
pub struct DownPlan {
    pub(crate) cw: usize,
    pub(crate) ch: usize,
    pub(crate) x: Vec<[(usize, i32); 3]>,
    pub(crate) y: Vec<[(usize, i32); 3]>,
    pub(crate) inner: Option<Interior<3, 1>>,
}

impl DownPlan {
    /// # Errors
    ///
    /// [`Error::Size`] when `plane` isn't `width * height`.
    pub fn new(
        plane: usize,
        width: usize,
        height: usize,
        format: ChromaFormat,
        location: ChromaLocation,
    ) -> Result<Self, Error> {
        if plane != width * height || width == 0 || height == 0 {
            return Err(Error::Size);
        }
        let (sx, sy) = format.factors();
        let (cw, ch) = format.plane(width, height);
        let (ox2, oy2) = offsets2(format, location);
        let x: Vec<_> = (0..cw).map(|i| down_taps(i, sx, ox2, width)).collect();
        let inner = if sx == 2 {
            Interior::new(&x, width)
        } else {
            None
        };
        Ok(Self {
            cw,
            ch,
            x,
            y: (0..ch).map(|j| down_taps(j, sy, oy2, height)).collect(),
            inner,
        })
    }
}

/// Codes to `i32` with `shift` more bits.
#[inline(always)]
pub(crate) fn widen_row(src: &[u16], shift: u32, out: &mut [i32]) {
    for (o, &c) in out.iter_mut().zip(src) {
        *o = i32::from(c) << shift;
    }
}

/// Values clipped to `0..=max` as codes.
#[inline(always)]
pub(crate) fn narrow_row(src: &[i32], max: u16, out: &mut [u16]) {
    for (o, &v) in out.iter_mut().zip(src) {
        *o = narrow(v, max);
    }
}

/// One chroma row (`cw` samples) to full width, weights in quarters.
#[inline(always)]
pub(crate) fn up_h_row(src: &[u16], plan: &UpPlan, out: &mut [i32]) {
    let width = plan.x.len();
    let edge = |out: &mut [i32], xs: core::ops::Range<usize>| {
        for x in xs {
            let [(a, wa), (b, wb)] = plan.x[x];
            out[x] = wa * i32::from(src[a]) + wb * i32::from(src[b]);
        }
    };
    match &plan.inner {
        Some(Interior {
            rel: [[(a0, w0), (b0, v0)], [(a1, w1), (b1, v1)]],
            range,
        }) => {
            edge(out, 0..2 * range.start);
            let (a0, b0, a1, b1, w0, v0, w1, v1) = (*a0, *b0, *a1, *b1, *w0, *v0, *w1, *v1);
            for i in range.clone() {
                let at = |d: isize| i32::from(src[i.wrapping_add_signed(d)]);
                out[2 * i] = w0 * at(a0) + v0 * at(b0);
                out[2 * i + 1] = w1 * at(a1) + v1 * at(b1);
            }
            edge(out, 2 * range.end..width);
        }
        None => edge(out, 0..width),
    }
}

/// Two full-width rows to one: `wa a + wb b`.
#[inline(always)]
pub(crate) fn up_v_row(a: &[i32], b: &[i32], wa: i32, wb: i32, out: &mut [i32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = wa * x + wb * y;
    }
}

/// Three full-width rows to one: `w0 r0 + w1 r1 + w2 r2`.
#[inline(always)]
pub(crate) fn down_v_row(r: [&[i32]; 3], w: [i32; 3], out: &mut [i32]) {
    for (((o, &x), &y), &z) in out.iter_mut().zip(r[0]).zip(r[1]).zip(r[2]) {
        *o = w[0] * x + w[1] * y + w[2] * z;
    }
}

/// A vertically filtered full-width row to `cw` codes: horizontal taps,
/// rounding (weights sum to 16 over both passes, `frac` more bits in the
/// values) and clipping to `0..=max`.
#[inline(always)]
pub(crate) fn down_h_row(src: &[i32], plan: &DownPlan, frac: u32, max: u16, out: &mut [u16]) {
    let shift = frac + 4;
    let half = 1i32 << (shift - 1);
    let edge = |out: &mut [u16], is: core::ops::Range<usize>| {
        for i in is {
            let [(a, wa), (b, wb), (c, wc)] = plan.x[i];
            out[i] = narrow(
                (half + wa * src[a] + wb * src[b] + wc * src[c]) >> shift,
                max,
            );
        }
    };
    match &plan.inner {
        Some(Interior {
            rel: [[(a, wa), (b, wb), (c, wc)]],
            range,
        }) => {
            edge(out, 0..range.start);
            let (a, b, c, wa, wb, wc) = (*a, *b, *c, *wa, *wb, *wc);
            for i in range.clone() {
                let at = |d: isize| src[(2 * i).wrapping_add_signed(d)];
                out[i] = narrow((half + wa * at(a) + wb * at(b) + wc * at(c)) >> shift, max);
            }
            edge(out, range.end..plan.cw);
        }
        None => edge(out, 0..plan.cw),
    }
}

/// A chroma plane of `format` upsampled to `width` x `height`, as codes
/// with [`FINE`] fractional bits: exact (no rounding).
///
/// # Errors
///
/// [`Error::Size`] when `plane` doesn't have the subsampled size.
pub fn upsample(
    plane: &[u16],
    width: usize,
    height: usize,
    format: ChromaFormat,
    location: ChromaLocation,
    filter: Upsample,
) -> Result<Vec<i32>, Error> {
    upsample_with(&Scalar, plane, width, height, format, location, filter)
}

pub(crate) fn upsample_with(
    k: &dyn Kernels,
    plane: &[u16],
    width: usize,
    height: usize,
    format: ChromaFormat,
    location: ChromaLocation,
    filter: Upsample,
) -> Result<Vec<i32>, Error> {
    let plan = UpPlan::new(plane.len(), width, height, format, location, filter)?;
    let mut rows = vec![0i32; width * plan.ch];
    for (row, src) in rows
        .chunks_exact_mut(width)
        .zip(plane.chunks_exact(plan.cw))
    {
        k.up_h(src, &plan, row);
    }
    let mut out = vec![0i32; width * height];
    for (o, &[(a, wa), (b, wb)]) in out.chunks_exact_mut(width).zip(&plan.y) {
        k.up_v(
            &rows[a * width..][..width],
            &rows[b * width..][..width],
            wa,
            wb,
            o,
        );
    }
    Ok(out)
}

/// A full-resolution plane with `frac` fractional bits downsampled for
/// `format` and rounded (H.273 `Round`) to codes in `0..=max`. Vertical
/// first, then horizontal: exact integer sums, so the order gives the
/// reference's values.
///
/// # Errors
///
/// [`Error::Size`] when `plane` isn't `width * height` long.
pub fn downsample(
    plane: &[i32],
    width: usize,
    height: usize,
    format: ChromaFormat,
    location: ChromaLocation,
    frac: u32,
    max: u16,
) -> Result<Vec<u16>, Error> {
    downsample_with(&Scalar, plane, width, height, format, location, frac, max)
}

#[allow(clippy::too_many_arguments, reason = "the plane and its geometry")]
pub(crate) fn downsample_with(
    k: &dyn Kernels,
    plane: &[i32],
    width: usize,
    height: usize,
    format: ChromaFormat,
    location: ChromaLocation,
    frac: u32,
    max: u16,
) -> Result<Vec<u16>, Error> {
    let plan = DownPlan::new(plane.len(), width, height, format, location)?;
    let mut row = vec![0i32; width];
    let mut out = vec![0u16; plan.cw * plan.ch];
    for (o, &[(a, wa), (b, wb), (c, wc)]) in out.chunks_exact_mut(plan.cw).zip(&plan.y) {
        let at = |r: usize| &plane[r * width..][..width];
        k.down_v([at(a), at(b), at(c)], [wa, wb, wc], &mut row);
        k.down_h(&row, &plan, frac, max, o);
    }
    Ok(out)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to 0..=max"
)]
#[inline(always)]
pub(crate) fn narrow(v: i32, max: u16) -> u16 {
    v.clamp(0, i32::from(max)) as u16
}

/// `(scale, offset)` of dequantization: `E = scale * D + offset`.
fn dequant_affine(depth: BitDepth, range: Range, signal: Signal) -> (f64, f64) {
    let n = depth.bits();
    let full = f64::from(depth.max());
    let s = f64::from(1u32 << (n - 8));
    match (range, signal) {
        (Range::Limited, Signal::Luma) => (1.0 / (219.0 * s), -16.0 / 219.0),
        (Range::Limited, Signal::Chroma) => (1.0 / (224.0 * s), -128.0 / 224.0),
        (Range::Full, Signal::Luma) => (1.0 / full, 0.0),
        (Range::Full, Signal::Chroma) => (1.0 / full, -f64::from(1u32 << (n - 1)) / full),
    }
}

/// `(scale, offset)` of quantization before rounding: `D = scale * E + offset`.
pub(crate) fn quant_affine(depth: BitDepth, range: Range, signal: Signal) -> (f64, f64) {
    let n = depth.bits();
    let s = f64::from(1u32 << (n - 8));
    match (range, signal) {
        (Range::Limited, Signal::Luma) => (219.0 * s, 16.0 * s),
        (Range::Limited, Signal::Chroma) => (224.0 * s, 128.0 * s),
        (Range::Full, Signal::Luma) => (f64::from(depth.max()), 0.0),
        (Range::Full, Signal::Chroma) => (f64::from(depth.max()), f64::from(1u32 << (n - 1))),
    }
}

fn signals(c: &Colour) -> [Signal; 3] {
    let chroma = if c.matrix.chroma_signed() {
        Signal::Chroma
    } else {
        Signal::Luma
    };
    [Signal::Luma, chroma, chroma]
}

/// The fast form of [`convert::convert`](crate::convert::convert) for
/// affine conversions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pipeline {
    from: Format,
    to: Format,
    upsample: Upsample,
    /// Inputs with [`FINE`] fractional bits; luma (and, into 4:4:4, every
    /// output) as clipped codes, colour differences into subsampled
    /// formats with [`FINE_OUT`] bits, unclipped.
    pub affine: Affine,
}

impl Pipeline {
    /// # Errors
    ///
    /// [`Error::Unsupported`] when the conversion isn't affine (other curve
    /// or primaries, constant luminance, ICtCp), a bit depth is above
    /// [`MAX_BITS`], or a side is monochrome; what [`Converter::new`]
    /// refuses.
    pub fn new(from: &Format, to: &Format, options: &Options) -> Result<Self, Error> {
        let conv = Converter::new(&from.cicp, &to.cicp, options)?;
        let unsupported = Error::Unsupported("not an affine conversion");
        if !conv.same_light() || from.depth.bits() > MAX_BITS || to.depth.bits() > MAX_BITS {
            return Err(unsupported);
        }
        if from.chroma == ChromaFormat::Monochrome || to.chroma == ChromaFormat::Monochrome {
            return Err(Error::Unsupported("monochrome"));
        }
        let (Some((_, to_rgb)), Some((to_ycc, _))) =
            (conv.from().matrix.linear(), conv.to().matrix.linear())
        else {
            return Err(unsupported);
        };
        let m = math::mul(&to_ycc, &to_rgb);
        let (src, dst) = (signals(conv.from()), signals(conv.to()));
        let fine = f64::from(1u32 << FINE);
        let deq = [0, 1, 2].map(|j| dequant_affine(from.depth, conv.from().range, src[j]));
        let q = [0, 1, 2].map(|k| quant_affine(to.depth, conv.to().range, dst[k]));
        // out = diag(q.scale) M (diag(deq.scale) in / fine + deq.offset) + q.offset.
        let t = math::apply(&m, deq.map(|d| d.1));
        let mut a = [[0.0; 3]; 3];
        let mut c = [0.0; 3];
        let subsampled = to.chroma != ChromaFormat::Yuv444;
        let mut out_frac = [0; 3];
        let mut hi = [Some(i32::from(to.depth.max())); 3];
        for k in 0..3 {
            if k > 0 && subsampled {
                out_frac[k] = FINE_OUT;
                hi[k] = None;
            }
            for j in 0..3 {
                a[k][j] = q[k].0 * m[k][j] * deq[j].0 / fine;
            }
            c[k] = q[k].0 * t[k] + q[k].1;
        }
        let in_max = [i64::from(from.depth.max()) << FINE; 3];
        Ok(Self {
            from: *from,
            to: *to,
            upsample: options.upsample,
            affine: Affine::new(&a, c, in_max, out_frac, hi)?,
        })
    }

    /// `image`, coded as the pipeline's source, coded as its target.
    ///
    /// # Errors
    ///
    /// As [`convert::convert`](crate::convert::convert).
    pub fn run(&self, image: &Image) -> Result<Image, Error> {
        self.run_with(image, &Scalar)
    }

    /// Row by row: each source row is widened, its chroma upsampled from
    /// a cache of two horizontally upsampled chroma rows, put through the
    /// matrix, and written; colour differences for a subsampled target
    /// wait in a ring of four rows until a chroma row has all its taps.
    /// Only a few rows live at a time, so they stay in cache.
    #[allow(
        clippy::too_many_lines,
        reason = "one row loop with its caches, read top to bottom"
    )]
    pub(crate) fn run_with(&self, image: &Image, k: &dyn Kernels) -> Result<Image, Error> {
        image.check()?;
        let (from, to) = (&self.from, &self.to);
        if image.chroma != from.chroma || image.depth != from.depth {
            return Err(Error::Format);
        }
        let (w, h) = (image.width, image.height);
        let max = to.depth.max();
        let up = if from.chroma == ChromaFormat::Yuv444 {
            None
        } else {
            let (cw, ch) = from.chroma.plane(w, h);
            Some(UpPlan::new(
                cw * ch,
                w,
                h,
                from.chroma,
                from.location,
                self.upsample,
            )?)
        };
        let down = if to.chroma == ChromaFormat::Yuv444 {
            None
        } else {
            Some(DownPlan::new(w * h, w, h, to.chroma, to.location)?)
        };
        let mut result = Image::new(w, h, to.chroma, to.depth)?;
        let mut input = [vec![0i32; w], vec![0i32; w], vec![0i32; w]];
        let mut output = [vec![0i32; w], vec![0i32; w], vec![0i32; w]];
        // Horizontally upsampled chroma rows, tagged by chroma row: the
        // vertical taps of consecutive rows only move forward.
        let mut cache: [[(usize, Vec<i32>); 2]; 2] =
            core::array::from_fn(|_| [(usize::MAX, vec![0; w]), (usize::MAX, vec![0; w])]);
        // The last four rows of colour differences (a chroma row's taps
        // span at most three).
        let mut ring: [[(usize, Vec<i32>); 4]; 2] =
            core::array::from_fn(|_| core::array::from_fn(|_| (usize::MAX, vec![0; w])));
        let mut vertical = vec![0i32; w];
        let mut next = 0;
        for y in 0..h {
            k.widen(&image.planes[0][y * w..][..w], FINE, &mut input[0]);
            for p in 1..3 {
                match &up {
                    None => k.widen(&image.planes[p][y * w..][..w], FINE, &mut input[p]),
                    Some(plan) => {
                        let [(a, wa), (b, wb)] = plan.y[y];
                        let slots = &mut cache[p - 1];
                        for r in [a, b] {
                            if slots[0].0 != r && slots[1].0 != r {
                                // Into the slot that doesn't hold `a`.
                                let s = usize::from(slots[0].0 == a);
                                slots[s].0 = r;
                                k.up_h(
                                    &image.planes[p][r * plan.cw..][..plan.cw],
                                    plan,
                                    &mut slots[s].1,
                                );
                            }
                        }
                        let slots = &cache[p - 1];
                        let row = |r: usize| {
                            if slots[0].0 == r {
                                &slots[0].1
                            } else {
                                &slots[1].1
                            }
                        };
                        k.up_v(row(a), row(b), wa, wb, &mut input[p]);
                    }
                }
            }
            {
                let [o0, o1, o2] = &mut output;
                k.affine(
                    &self.affine,
                    [&input[0], &input[1], &input[2]],
                    [o0, o1, o2],
                );
            }
            k.narrow(&output[0], max, &mut result.planes[0][y * w..][..w]);
            match &down {
                None => {
                    for p in 1..3 {
                        k.narrow(&output[p], max, &mut result.planes[p][y * w..][..w]);
                    }
                }
                Some(plan) => {
                    for p in 1..3 {
                        let slot = &mut ring[p - 1][y % 4];
                        slot.0 = y;
                        slot.1.copy_from_slice(&output[p]);
                    }
                    // Every chroma row whose taps have all arrived.
                    while next < plan.ch && plan.y[next].iter().all(|&(r, wt)| wt == 0 || r <= y) {
                        let [(a, wa), (b, wb), (c, wc)] = plan.y[next];
                        // A tap of weight 0 reads a row that is there.
                        let (b, c) = (if wb == 0 { a } else { b }, if wc == 0 { a } else { c });
                        for p in 1..3 {
                            let rows = &ring[p - 1];
                            let at = |r: usize| {
                                let s = &rows[r % 4];
                                debug_assert_eq!(s.0, r, "a tap's row left the ring");
                                s.1.as_slice()
                            };
                            k.down_v([at(a), at(b), at(c)], [wa, wb, wc], &mut vertical);
                            k.down_h(
                                &vertical,
                                plan,
                                FINE_OUT,
                                max,
                                &mut result.planes[p][next * plan.cw..][..plan.cw],
                            );
                        }
                        next += 1;
                    }
                }
            }
        }
        result.alpha = image.alpha.as_ref().map(|a| {
            a.iter()
                .map(|&c| {
                    crate::quant::requantize(c, from.depth, to.depth, Range::Full, Signal::Luma)
                })
                .collect()
        });
        Ok(result)
    }
}

/// The row kernels a [`Pipeline`] runs on: scalar here, SIMD in
/// [`simd`](crate::simd).
pub(crate) trait Kernels {
    fn widen(&self, src: &[u16], shift: u32, out: &mut [i32]);
    fn narrow(&self, src: &[i32], max: u16, out: &mut [u16]);
    fn affine(&self, k: &Affine, input: [&[i32]; 3], out: [&mut [i32]; 3]);
    fn up_h(&self, src: &[u16], plan: &UpPlan, out: &mut [i32]);
    fn up_v(&self, a: &[i32], b: &[i32], wa: i32, wb: i32, out: &mut [i32]);
    fn down_v(&self, r: [&[i32]; 3], w: [i32; 3], out: &mut [i32]);
    fn down_h(&self, src: &[i32], plan: &DownPlan, frac: u32, max: u16, out: &mut [u16]);
}

pub(crate) struct Scalar;

impl Kernels for Scalar {
    fn widen(&self, src: &[u16], shift: u32, out: &mut [i32]) {
        widen_row(src, shift, out);
    }
    fn narrow(&self, src: &[i32], max: u16, out: &mut [u16]) {
        narrow_row(src, max, out);
    }
    fn affine(&self, k: &Affine, input: [&[i32]; 3], out: [&mut [i32]; 3]) {
        affine(k, input, out);
    }
    fn up_h(&self, src: &[u16], plan: &UpPlan, out: &mut [i32]) {
        up_h_row(src, plan, out);
    }
    fn up_v(&self, a: &[i32], b: &[i32], wa: i32, wb: i32, out: &mut [i32]) {
        up_v_row(a, b, wa, wb, out);
    }
    fn down_v(&self, r: [&[i32]; 3], w: [i32; 3], out: &mut [i32]) {
        down_v_row(r, w, out);
    }
    fn down_h(&self, src: &[i32], plan: &DownPlan, frac: u32, max: u16, out: &mut [u16]) {
        down_h_row(src, plan, frac, max, out);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::chroma::{self, Downsample};
    use crate::cicp::{Cicp, ColourPrimaries, MatrixCoefficients, TransferCharacteristics};
    use crate::convert;

    /// A pseudo-random image (fixed seed).
    pub(crate) fn noise(
        w: usize,
        h: usize,
        chroma: ChromaFormat,
        depth: BitDepth,
        seed: u32,
    ) -> Image {
        let mut img = Image::new(w, h, chroma, depth).unwrap();
        let mut s = seed;
        let m = u32::from(depth.max()) + 1;
        for p in &mut img.planes {
            for v in p.iter_mut() {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_possible_truncation, reason = "below max")]
                let c = ((s >> 8) % m) as u16;
                *v = c;
            }
        }
        img
    }

    fn fmt(cicp: Cicp, bits: u8, chroma: ChromaFormat, location: ChromaLocation) -> Format {
        Format {
            cicp,
            depth: BitDepth::new(bits).unwrap(),
            chroma,
            location,
        }
    }

    /// How many codes differ, and by at most how much.
    pub(crate) fn diff(a: &Image, b: &Image) -> (usize, u16, usize) {
        let mut count = 0;
        let mut worst = 0;
        let mut total = 0;
        for (p, q) in a.planes.iter().zip(&b.planes) {
            for (&x, &y) in p.iter().zip(q) {
                total += 1;
                if x != y {
                    count += 1;
                    worst = worst.max(x.abs_diff(y));
                }
            }
        }
        (count, worst, total)
    }

    #[test]
    fn chroma_kernels_are_exactly_the_reference() {
        for (w, h) in [
            (1, 1),
            (2, 3),
            (3, 2),
            (4, 4),
            (5, 5),
            (6, 1),
            (7, 3),
            (8, 2),
            (9, 9),
            (13, 9),
            (16, 4),
            (31, 7),
            (64, 3),
        ] {
            chroma_exact(w, h);
        }
    }

    fn chroma_exact(w: usize, h: usize) {
        for format in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv440,
        ] {
            for location in ChromaLocation::ALL {
                for filter in [Upsample::Linear, Upsample::Nearest] {
                    let (cw, ch) = format.plane(w, h);
                    let img = noise(cw, ch, ChromaFormat::Yuv444, BitDepth::B12, 7);
                    let sub = &img.planes[0];
                    let fast: Vec<u16> = upsample(sub, w, h, format, location, filter)
                        .unwrap()
                        .iter()
                        .map(|&v| narrow((v + 8) >> 4, 4095))
                        .collect();
                    let reference =
                        chroma::upsample_codes(sub, w, h, format, location, filter, 4095).unwrap();
                    assert_eq!(fast, reference, "up {format:?} {location:?} {filter:?}");
                }
                let full = noise(w, h, ChromaFormat::Yuv444, BitDepth::B12, 9);
                let fine: Vec<i32> = full.planes[0].iter().map(|&c| i32::from(c)).collect();
                let fast = downsample(&fine, w, h, format, location, 0, 4095).unwrap();
                let reference = chroma::downsample_codes(
                    &full.planes[0],
                    w,
                    h,
                    format,
                    location,
                    Downsample::Area,
                    4095,
                )
                .unwrap();
                assert_eq!(fast, reference, "down {format:?} {location:?}");
            }
        }
    }

    #[test]
    fn affine_conversions_are_within_one_code_of_the_reference() {
        use ChromaFormat::{Yuv420, Yuv422, Yuv444};
        let gbr = |full| Cicp {
            matrix: MatrixCoefficients::Identity,
            full_range: full,
            ..Cicp::BT709
        };
        let ycgco = Cicp {
            matrix: MatrixCoefficients::YCgCo,
            full_range: true,
            ..Cicp::BT709
        };
        let p3 = Cicp {
            primaries: ColourPrimaries::Smpte432,
            transfer: TransferCharacteristics::Srgb,
            matrix: MatrixCoefficients::ChromaDerivedNcl,
            full_range: false,
        };
        let loc = ChromaLocation::Left;
        let cases = [
            (
                fmt(gbr(true), 8, Yuv444, loc),
                fmt(Cicp::BT709, 8, Yuv420, loc),
            ),
            (
                fmt(Cicp::BT709, 8, Yuv420, loc),
                fmt(gbr(true), 8, Yuv444, loc),
            ),
            (
                fmt(gbr(true), 10, Yuv444, loc),
                fmt(
                    Cicp {
                        matrix: MatrixCoefficients::Bt2020Ncl,
                        ..Cicp::BT709
                    },
                    10,
                    Yuv422,
                    ChromaLocation::Center,
                ),
            ),
            (
                fmt(Cicp::BT709, 10, Yuv420, ChromaLocation::TopLeft),
                fmt(Cicp::BT709, 8, Yuv444, loc),
            ),
            (
                fmt(Cicp::BT709, 8, Yuv444, loc),
                fmt(
                    Cicp {
                        full_range: true,
                        ..Cicp::BT709
                    },
                    12,
                    Yuv444,
                    loc,
                ),
            ),
            (
                fmt(Cicp::BT601_525, 8, Yuv444, loc),
                fmt(
                    Cicp {
                        primaries: ColourPrimaries::Bt601,
                        ..Cicp::BT709
                    },
                    8,
                    Yuv444,
                    loc,
                ),
            ),
            (
                fmt(gbr(false), 12, Yuv444, loc),
                fmt(ycgco, 12, Yuv420, ChromaLocation::Bottom),
            ),
            (
                fmt(
                    Cicp {
                        matrix: MatrixCoefficients::Identity,
                        ..p3
                    },
                    8,
                    Yuv444,
                    loc,
                ),
                fmt(p3, 10, Yuv420, loc),
            ),
        ];
        let options = Options::default();
        for (from, to) in cases {
            let img = noise(64, 48, from.chroma, from.depth, 3);
            let fast = Pipeline::new(&from, &to, &options)
                .unwrap()
                .run(&img)
                .unwrap();
            let reference = convert::convert(&img, &from, &to, &options).unwrap();
            let (count, worst, total) = diff(&fast, &reference);
            assert!(worst <= 1, "{from:?} -> {to:?}: off by {worst}");
            // Rare: well under 1 % of the codes.
            assert!(
                count * 100 < total,
                "{from:?} -> {to:?}: {count} of {total}"
            );
        }
    }

    /// The pipeline's stages on whole planes, one after another.
    fn plane_by_plane(p: &Pipeline, img: &Image) -> Image {
        let (w, h) = (img.width, img.height);
        let widen = |c: &[u16]| -> Vec<i32> { c.iter().map(|&v| i32::from(v) << FINE).collect() };
        let up = |c: &[u16]| {
            if p.from.chroma == ChromaFormat::Yuv444 {
                widen(c)
            } else {
                upsample(c, w, h, p.from.chroma, p.from.location, p.upsample).unwrap()
            }
        };
        let x = [
            widen(&img.planes[0]),
            up(&img.planes[1]),
            up(&img.planes[2]),
        ];
        let mut o = [vec![0; w * h], vec![0; w * h], vec![0; w * h]];
        {
            let [o0, o1, o2] = &mut o;
            affine(&p.affine, [&x[0], &x[1], &x[2]], [o0, o1, o2]);
        }
        let max = p.to.depth.max();
        let mut out = Image::new(w, h, p.to.chroma, p.to.depth).unwrap();
        out.planes[0] = o[0].iter().map(|&v| narrow(v, max)).collect();
        for k in 1..3 {
            out.planes[k] = if p.to.chroma == ChromaFormat::Yuv444 {
                o[k].iter().map(|&v| narrow(v, max)).collect()
            } else {
                downsample(&o[k], w, h, p.to.chroma, p.to.location, FINE_OUT, max).unwrap()
            };
        }
        out
    }

    #[test]
    fn streaming_rows_is_the_plane_by_plane_pipeline() {
        use ChromaFormat::{Yuv420, Yuv422, Yuv440, Yuv444};
        let gbr = Cicp {
            matrix: MatrixCoefficients::Identity,
            full_range: true,
            ..Cicp::BT709
        };
        for (w, h) in [(1, 1), (2, 2), (3, 5), (5, 3), (7, 4), (17, 9), (40, 11)] {
            for from_chroma in [Yuv420, Yuv422, Yuv440, Yuv444] {
                for to_chroma in [Yuv420, Yuv422, Yuv440, Yuv444] {
                    for (lf, lt) in [
                        (ChromaLocation::Left, ChromaLocation::Bottom),
                        (ChromaLocation::TopLeft, ChromaLocation::Center),
                        (ChromaLocation::Bottom, ChromaLocation::Top),
                    ] {
                        let src = fmt(Cicp::BT709, 10, from_chroma, lf);
                        let dst = fmt(
                            if to_chroma == Yuv444 {
                                gbr
                            } else {
                                Cicp::BT709
                            },
                            8,
                            to_chroma,
                            lt,
                        );
                        let p = Pipeline::new(&src, &dst, &Options::default()).unwrap();
                        let img = noise(w, h, from_chroma, src.depth, 21);
                        assert_eq!(
                            p.run(&img).unwrap(),
                            plane_by_plane(&p, &img),
                            "{w}x{h} {from_chroma:?} {lf:?} -> {to_chroma:?} {lt:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn only_affine_conversions_are_taken() {
        let f = |cicp| fmt(cicp, 10, ChromaFormat::Yuv420, ChromaLocation::Left);
        let o = Options::default();
        assert!(Pipeline::new(&f(Cicp::BT709), &f(Cicp::HDR10), &o).is_err());
        assert!(
            Pipeline::new(
                &f(Cicp::BT709),
                &fmt(Cicp::BT709, 16, ChromaFormat::Yuv420, ChromaLocation::Left),
                &o
            )
            .is_err()
        );
        let cl = Cicp {
            matrix: MatrixCoefficients::Bt2020Cl,
            ..Cicp::BT2020
        };
        assert!(Pipeline::new(&f(Cicp::BT2020), &f(cl), &o).is_err());
        assert!(
            Pipeline::new(
                &f(Cicp::BT709),
                &f(Cicp {
                    full_range: true,
                    ..Cicp::BT709
                }),
                &o
            )
            .is_ok()
        );
    }
}
