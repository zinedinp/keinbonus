//! Conversions through light (another curve or other primaries) from RGB
//! to RGB, bit for bit the [reference](crate::convert), without a power
//! function per sample:
//!
//! - Code to light: a [`DecodeTable`] of the reference's own values, one
//!   per code (the input is discrete, so this is exact).
//! - Gamut, tone mapping, clipping: [`Converter::light_of_nits`], the
//!   function the reference calls.
//! - Light to code: an [`EncodeTable`]. The code rises with the light, so
//!   each code begins at a boundary, found by bisection over the `f64`
//!   values: kept as a window a billionth of a code wide, inside which the
//!   reference itself decides (it may wobble there by an ulp). A cubic per
//!   octave guesses the code, the windows correct it: exact whatever the
//!   guess, which only sets the speed.
//!
//! Tables per curve and depth: `8 * 2^n` bytes to decode, `16 * 2^n` to
//! encode (at 10 bits 8 and 16 KiB, with the cubics about 25 KiB, within
//! L1d; at 12 bits 96 KiB, in L2).
//!
//! Per channel only: RGB both sides (matrix GBR, 4:4:4), neither side HLG,
//! whose OOTF mixes the channels. YCbCr targets mix the encoded channels
//! in the matrix, which a threshold table can't follow: see `TODO.md`.
//!
//! Exact as long as the reference's own rounding error stays below
//! [`MARGIN`] (it is near 1e-13 of a code); the tests check every window
//! edge and random light against the reference.

use alloc::vec::Vec;

use crate::chroma::ChromaFormat;
use crate::cicp::MatrixCoefficients;
use crate::convert::{Converter, Format, Options};
use crate::hdr::{self, Luminance};
use crate::image::Image;
use crate::math::{self, round};
use crate::quant::{self, BitDepth, Range, Signal};
use crate::transfer::Curve;
use crate::{Colour, Error};

use super::MAX_BITS;

/// Code to display light (cd/m²) for one curve, range and depth: the
/// reference's [`hdr::to_nits`] of each dequantized code.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeTable {
    nits: Vec<f64>,
}

impl DecodeTable {
    /// # Errors
    ///
    /// [`Error::Unsupported`] for HLG (its OOTF mixes the channels) or a
    /// depth above [`MAX_BITS`].
    pub fn new(colour: &Colour, depth: BitDepth, lum: &Luminance) -> Result<Self, Error> {
        check(colour.curve, depth)?;
        let luma = colour.primaries.luminance();
        let nits = (0..=depth.max())
            .map(|c| {
                let e = quant::dequantize(c, depth, colour.range, Signal::Luma);
                hdr::to_nits(colour.curve, [e; 3], lum, luma)[0]
            })
            .collect();
        Ok(Self { nits })
    }

    #[must_use]
    #[inline]
    pub fn get(&self, code: u16) -> f64 {
        self.nits[usize::from(code)]
    }
}

fn check(curve: Curve, depth: BitDepth) -> Result<(), Error> {
    if curve == Curve::Hlg {
        return Err(Error::Unsupported("HLG mixes the channels"));
    }
    if depth.bits() > MAX_BITS {
        return Err(Error::Unsupported("tables above 12 bits"));
    }
    Ok(())
}

/// `f64` values in their order as `i64`s (an involution).
fn key(x: f64) -> i64 {
    #[allow(clippy::cast_possible_wrap, reason = "the bits, reinterpreted")]
    let b = x.to_bits() as i64;
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "a mask of the low 63 bits"
    )]
    let mask = (((b >> 63) as u64) >> 1) as i64;
    b ^ mask
}

fn unkey(k: i64) -> f64 {
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "a mask of the low 63 bits"
    )]
    let mask = (((k >> 63) as u64) >> 1) as i64;
    #[allow(clippy::cast_sign_loss, reason = "the bits, reinterpreted")]
    f64::from_bits((k ^ mask) as u64)
}

/// A cubic over one octave `[2^e, 2^(e+1))` in `u = L / 2^e - 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cubic([f64; 4]);

impl Cubic {
    /// Through `g` at four Chebyshev nodes of the octave.
    fn fit(e: i32, g: impl Fn(f64) -> f64) -> Self {
        let base = libm::exp2(f64::from(e));
        let nodes = [0, 1, 2, 3].map(|i| {
            let theta = core::f64::consts::PI * (2.0 * f64::from(i) + 1.0) / 8.0;
            0.5 - 0.5 * libm::cos(theta)
        });
        // Vandermonde system, Gaussian elimination with partial pivoting.
        let mut m = nodes.map(|u| [1.0, u, u * u, u * u * u, g(base * (1.0 + u))]);
        for col in 0..4 {
            let pivot = (col..4)
                .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
                .unwrap_or(col);
            m.swap(col, pivot);
            for row in 0..4 {
                if row != col && m[col][col] != 0.0 {
                    let f = m[row][col] / m[col][col];
                    for k in col..5 {
                        m[row][k] -= f * m[col][k];
                    }
                }
            }
        }
        Self([0, 1, 2, 3].map(|i| {
            if m[i][i] == 0.0 {
                0.0
            } else {
                m[i][4] / m[i][i]
            }
        }))
    }

    #[inline]
    fn eval(&self, u: f64) -> f64 {
        let [a, b, c, d] = self.0;
        ((d * u + c) * u + b) * u + a
    }
}

/// How far from a rounding boundary (in codes) the table decides: closer,
/// the reference itself does. Far above the reference's own rounding noise
/// (libm's `pow` is within an ulp, about 1e-13 of a code), so outside
/// these windows its result can't differ from the side the table sees.
pub const MARGIN: f64 = 1e-9;

/// Display light (cd/m²) to code for one curve, range and depth: exactly
/// the reference's `quantize(from_nits(L))`.
///
/// Code `k` begins where the code before rounding, `q(L)`, passes
/// `k - 1/2`. Each boundary gets a window: `lo[k]`, the least light with
/// `q >= k - 1/2 - MARGIN`, and `hi[k]`, the least with
/// `q >= k - 1/2 + MARGIN`. Below `lo[k]` the code is under `k`, from
/// `hi[k]` on at least `k`; in between the reference decides (its ulp-level
/// wobble can make it non-monotone there, which a single threshold would
/// get wrong).
#[derive(Debug, Clone, PartialEq)]
pub struct EncodeTable {
    /// Per code `k`; index 0 is minus infinity, unreachable codes plus
    /// infinity.
    lo: Vec<f64>,
    hi: Vec<f64>,
    max: u16,
    /// Octave of the first cubic, and the cubics: one per octave from the
    /// first positive boundary to the last finite one.
    first: i32,
    cubics: Vec<Cubic>,
    colour: (Curve, Range, [f64; 3]),
    depth: BitDepth,
    lum: Luminance,
}

/// The least `f64` for which the monotone `above` holds, or an infinity.
fn least(above: impl Fn(f64) -> bool, from: i64) -> (f64, i64) {
    let (lo_key, hi_key) = (from, key(f64::MAX));
    if above(unkey(lo_key)) {
        return (f64::NEG_INFINITY, lo_key);
    }
    if !above(f64::MAX) {
        return (f64::INFINITY, lo_key);
    }
    let (mut lo, mut hi) = (lo_key, hi_key);
    // The keys span all of i64: their midpoint in i128.
    while i128::from(hi) - i128::from(lo) > 1 {
        let mid = lo.midpoint(hi);
        if above(unkey(mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    (unkey(hi), lo)
}

impl EncodeTable {
    /// Two bisections per code: `128 * 2^n` evaluations of the reference
    /// (milliseconds), so build it once per format.
    ///
    /// # Errors
    ///
    /// As [`DecodeTable::new`].
    pub fn new(colour: &Colour, depth: BitDepth, lum: &Luminance) -> Result<Self, Error> {
        check(colour.curve, depth)?;
        let luma = colour.primaries.luminance();
        let (curve, range) = (colour.curve, colour.range);
        let (scale, offset) = super::quant_affine(depth, range, Signal::Luma);
        // The code before rounding.
        let q = |l: f64| scale * hdr::from_nits(curve, [l; 3], lum, luma)[0] + offset;
        let max = depth.max();
        let (mut lo, mut hi) = (
            Vec::with_capacity(usize::from(max) + 1),
            Vec::with_capacity(usize::from(max) + 1),
        );
        lo.push(f64::NEG_INFINITY);
        hi.push(f64::NEG_INFINITY);
        // Boundaries rise with k: each search starts below the last.
        let (mut from_lo, mut from_hi) = (key(-f64::MAX), key(-f64::MAX));
        for k in 1..=max {
            let edge = f64::from(k) - 0.5;
            let (l, f) = least(|x| q(x) >= edge - MARGIN, from_lo);
            let (h, g) = least(|x| q(x) >= edge + MARGIN, from_hi.max(f));
            (from_lo, from_hi) = (f, g);
            lo.push(l);
            hi.push(h);
        }
        let positive: Vec<f64> = hi
            .iter()
            .copied()
            .filter(|t| t.is_finite() && *t > 0.0)
            .collect();
        let (first, cubics) = match (positive.first(), positive.last()) {
            (Some(&a), Some(&b)) => {
                let (e0, e1) = (libm::ilogb(a), libm::ilogb(b));
                (e0, (e0..=e1).map(|e| Cubic::fit(e, q)).collect())
            }
            _ => (0, Vec::new()),
        };
        Ok(Self {
            lo,
            hi,
            max,
            first,
            cubics,
            colour: (curve, range, luma),
            depth,
            lum: *lum,
        })
    }

    /// The reference's code, computed.
    fn reference(&self, l: f64) -> u16 {
        let (curve, range, luma) = self.colour;
        quant::quantize(
            hdr::from_nits(curve, [l; 3], &self.lum, luma)[0],
            self.depth,
            range,
            Signal::Luma,
        )
    }

    /// The code of display light `l`.
    #[must_use]
    #[inline]
    pub fn code(&self, l: f64) -> u16 {
        // NaN and infinities are the reference's to answer (PQ makes NaN
        // of infinity): never from the pipeline, which clips to the peak.
        if !l.is_finite() {
            return self.reference(l);
        }
        let mut c = self.guess(l);
        loop {
            let i = usize::from(c);
            if c > 0 && l < self.hi[i] {
                if l >= self.lo[i] {
                    return self.reference(l);
                }
                c -= 1;
            } else if c < self.max && l >= self.lo[i + 1] {
                if l < self.hi[i + 1] {
                    return self.reference(l);
                }
                c += 1;
            } else {
                return c;
            }
        }
    }

    /// A code near `l`'s: the octave's cubic, else a binary search.
    #[inline]
    fn guess(&self, l: f64) -> u16 {
        if l >= f64::MIN_POSITIVE {
            // Exponent and mantissa from the bits: `l = 2^e (1 + u)` for a
            // normal `l`, exactly.
            let bits = l.to_bits();
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                reason = "11 bits"
            )]
            let e = ((bits >> 52) & 0x7ff) as i32 - 1023;
            if let Some(cubic) = usize::try_from(e - self.first)
                .ok()
                .and_then(|i| self.cubics.get(i))
            {
                let u = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52)) - 1.0;
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clipped to 0..=max"
                )]
                let c = math::clip3(0.0, f64::from(self.max), round(cubic.eval(u))) as u16;
                return c;
            }
        }
        // Outside the octaves (black, negative or above the last code):
        // count the boundaries below.
        let n = self.hi[1..].partition_point(|&t| t <= l);
        u16::try_from(n).unwrap_or(self.max)
    }
}

/// [`convert`](crate::convert::convert) from RGB to RGB through light,
/// with tables: the same codes.
#[derive(Debug, Clone, PartialEq)]
pub struct LightPipeline {
    from: Format,
    to: Format,
    converter: Converter,
    decode: DecodeTable,
    encode: EncodeTable,
}

impl LightPipeline {
    /// # Errors
    ///
    /// [`Error::Unsupported`] unless both sides are RGB (GBR matrix,
    /// 4:4:4), neither is HLG, both at most [`MAX_BITS`] deep, and the
    /// conversion goes through light (other curve or primaries; else
    /// [`Pipeline`](super::Pipeline) is the fast path); what
    /// [`Converter::new`] refuses.
    pub fn new(from: &Format, to: &Format, options: &Options) -> Result<Self, Error> {
        let rgb = |f: &Format| {
            f.cicp.matrix == MatrixCoefficients::Identity && f.chroma == ChromaFormat::Yuv444
        };
        if !rgb(from) || !rgb(to) {
            return Err(Error::Unsupported("RGB to RGB only"));
        }
        let converter = Converter::new(&from.cicp, &to.cicp, options)?;
        if converter.same_light() {
            return Err(Error::Unsupported("no change of light: use fast::Pipeline"));
        }
        Ok(Self {
            from: *from,
            to: *to,
            decode: DecodeTable::new(converter.from(), from.depth, &options.luminance)?,
            encode: EncodeTable::new(converter.to(), to.depth, &options.luminance)?,
            converter,
        })
    }

    /// `image`, coded as the source, coded as the target.
    ///
    /// # Errors
    ///
    /// As [`convert`](crate::convert::convert).
    pub fn run(&self, image: &Image) -> Result<Image, Error> {
        image.check()?;
        if image.chroma != self.from.chroma || image.depth != self.from.depth {
            return Err(Error::Format);
        }
        let mut out = Image::new(image.width, image.height, self.to.chroma, self.to.depth)?;
        let [g, b, r] = &image.planes;
        {
            let [og, ob, or] = &mut out.planes;
            for i in 0..g.len() {
                let d = |c: u16| self.decode.get(c);
                let [lr, lg, lb] = self.converter.light_of_nits([d(r[i]), d(g[i]), d(b[i])]);
                (or[i], og[i], ob[i]) = (
                    self.encode.code(lr),
                    self.encode.code(lg),
                    self.encode.code(lb),
                );
            }
        }
        out.alpha = image.alpha.as_ref().map(|a| {
            a.iter()
                .map(|&c| {
                    quant::requantize(c, self.from.depth, self.to.depth, Range::Full, Signal::Luma)
                })
                .collect()
        });
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chroma::ChromaLocation;
    use crate::cicp::{Cicp, ColourPrimaries, TransferCharacteristics};
    use crate::convert;
    use crate::fast::tests::noise;
    use crate::hdr::ToneMap;

    fn rgb(
        primaries: ColourPrimaries,
        transfer: TransferCharacteristics,
        full: bool,
        bits: u8,
    ) -> Format {
        Format {
            cicp: Cicp {
                primaries,
                transfer,
                matrix: MatrixCoefficients::Identity,
                full_range: full,
            },
            depth: BitDepth::new(bits).unwrap(),
            chroma: ChromaFormat::Yuv444,
            location: ChromaLocation::Left,
        }
    }

    #[test]
    fn encode_tables_give_the_reference_codes() {
        use TransferCharacteristics as T;
        let lum = Luminance::default();
        for t in [
            T::Pq,
            T::Srgb,
            T::Bt709,
            T::Gamma22,
            T::Linear,
            T::Log100,
            T::Iec61966_2_4,
            T::Smpte428,
        ] {
            for (bits, full) in [(8, true), (10, false), (12, true)] {
                let f = rgb(ColourPrimaries::Bt2020, t, full, bits);
                let colour = f.cicp.resolve().unwrap();
                let table = EncodeTable::new(&colour, f.depth, &lum).unwrap();
                let luma = colour.primaries.luminance();
                let reference = |l: f64| {
                    quant::quantize(
                        hdr::from_nits(colour.curve, [l; 3], &lum, luma)[0],
                        f.depth,
                        colour.range,
                        Signal::Luma,
                    )
                };
                // Every threshold and its neighbours: where a monotonicity
                // failure would show.
                for &th in table.lo.iter().chain(&table.hi).filter(|t| t.is_finite()) {
                    for l in [
                        th.next_down().next_down(),
                        th.next_down(),
                        th,
                        th.next_up(),
                        th.next_up().next_up(),
                    ] {
                        assert_eq!(table.code(l), reference(l), "{t:?} {bits} {l:e}");
                    }
                }
                // And light from black past HDR peaks, log-spaced, negative too.
                let mut s = 1u64;
                for _ in 0..20_000 {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    #[allow(clippy::cast_precision_loss, reason = "a random fraction")]
                    let u = (s >> 11) as f64 / (1u64 << 53) as f64;
                    let l = libm::exp10(-7.0 + 12.0 * u) * if s & 1 == 0 { 1.0 } else { -0.01 };
                    assert_eq!(table.code(l), reference(l), "{t:?} {bits} {l:e}");
                }
                for l in [
                    0.0,
                    -0.0,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NAN,
                    f64::MAX,
                    f64::MIN_POSITIVE,
                ] {
                    assert_eq!(table.code(l), reference(l), "{t:?} {bits} {l:e}");
                }
            }
        }
        let hlg = rgb(ColourPrimaries::Bt2020, T::Hlg, false, 10);
        assert!(EncodeTable::new(&hlg.cicp.resolve().unwrap(), hlg.depth, &lum).is_err());
    }

    #[test]
    fn rgb_through_light_is_the_reference_bit_for_bit() {
        use ColourPrimaries as P;
        use TransferCharacteristics as T;
        let pq_rgb = rgb(P::Bt2020, T::Pq, false, 10);
        let cases = [
            // SDR textures into an HDR10 swapchain (and its 8-bit form).
            (
                rgb(P::Bt709, T::Srgb, true, 8),
                rgb(P::Bt2020, T::Pq, true, 10),
                Options::default(),
            ),
            (rgb(P::Bt709, T::Srgb, true, 8), pq_rgb, Options::default()),
            // Wide gamut down to sRGB: clipped, white adapted.
            (
                rgb(P::Smpte432, T::Srgb, true, 10),
                rgb(P::Bt709, T::Srgb, true, 8),
                Options::default(),
            ),
            (
                rgb(P::Smpte431, T::Smpte428, true, 12),
                rgb(P::Bt709, T::Bt709, false, 10),
                Options::default(),
            ),
            // HDR10 to SDR, tone mapped.
            (
                pq_rgb,
                rgb(P::Bt709, T::Srgb, true, 8),
                Options {
                    tone_map: ToneMap::Bt2390 {
                        source_black: 0.0,
                        source_peak: 1000.0,
                    },
                    luminance: Luminance {
                        sdr_white: 100.0,
                        ..Luminance::default()
                    },
                    ..Options::default()
                },
            ),
            // Only the curve changes.
            (
                rgb(P::Bt709, T::Bt709, false, 10),
                rgb(P::Bt709, T::Srgb, true, 10),
                Options::default(),
            ),
        ];
        for (from, to, options) in cases {
            let p = LightPipeline::new(&from, &to, &options).unwrap();
            for (w, h, seed) in [(37, 23, 4), (64, 64, 9)] {
                let img = noise(w, h, from.chroma, from.depth, seed);
                assert_eq!(
                    p.run(&img).unwrap(),
                    convert::convert(&img, &from, &to, &options).unwrap(),
                    "{from:?} -> {to:?}"
                );
            }
            // Every code on every channel.
            let max = from.depth.max();
            let mut ramp =
                Image::new(usize::from(max) + 1, 1, ChromaFormat::Yuv444, from.depth).unwrap();
            let all: Vec<u16> = (0..=max).collect();
            ramp.planes = [all.clone(), all.iter().rev().copied().collect(), all];
            assert_eq!(
                p.run(&ramp).unwrap(),
                convert::convert(&ramp, &from, &to, &options).unwrap(),
                "ramp {from:?} -> {to:?}"
            );
        }
        let same = rgb(P::Bt709, T::Srgb, true, 8);
        assert!(LightPipeline::new(&same, &same, &Options::default()).is_err());
    }
}
