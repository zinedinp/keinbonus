//! Pixel layouts in memory: [`pack`] an [`Image`] into bytes, [`unpack`]
//! bytes into one. All multi-byte words little-endian; rows without padding
//! except where the layout defines it (v210).
//!
//! | layout | chroma | depths | samples |
//! |---|---|---|---|
//! | [`Planar`](Layout::Planar) | any | 8..=16 | Y, Cb, Cr planes; bytes at 8 bits, else `u16` with the code in the low bits (`yuv420p10le`) |
//! | [`SemiPlanar`](Layout::SemiPlanar) | 4:2:0, 4:2:2, 4:4:4 | 8..=16 | Y plane, then Cb Cr pairs (NV12, NV16, NV24; NV21 with `vu`); above 8 bits `u16` with the code in the high bits (P010, P012, P016, P210, P216, P410, P416) |
//! | [`Packed422`](Layout::Packed422) | 4:2:2 | 8..=16 | Y0 Cb Y1 Cr (YUY2; Y210, Y212, Y216 high-aligned) or Cb Y0 Cr Y1 (UYVY) |
//! | [`V210`](Layout::V210) | 4:2:2 | 10 | 6 pixels in 4 words of three 10-bit codes, rows padded to 128 bytes |
//! | [`Ayuv`](Layout::Ayuv) | 4:4:4 | 8 | bytes Cr, Cb, Y, A |
//! | [`Y410`](Layout::Y410) | 4:4:4 | 10 | one word: Cb bits 0-9, Y 10-19, Cr 20-29, A 30-31 |
//! | [`Y416`](Layout::Y416) | 4:4:4 | 9..=16 | `u16` Cb, Y, Cr, A, high-aligned (Y412 at 12 bits) |
//! | [`Rgb8`](Layout::Rgb8), [`Rgba8`](Layout::Rgba8), [`Bgra8`](Layout::Bgra8) | 4:4:4 GBR | 8 | bytes |
//! | [`Rgba16`](Layout::Rgba16) | 4:4:4 GBR | 16 | `u16` R, G, B, A |
//! | [`A2R10G10B10`](Layout::A2R10G10B10) | 4:4:4 GBR | 10 | one word: B bits 0-9, G 10-19, R 20-29, A 30-31 (Vulkan `A2R10G10B10_UNORM_PACK32`) |
//! | [`A2B10G10R10`](Layout::A2B10G10R10) | 4:4:4 GBR | 10 | R bits 0-9, G, B, A (DXGI `R10G10B10A2_UNORM`) |
//!
//! Layouts with alpha write opaque where the image has none, and unpack
//! into an image with alpha. A 2-bit alpha is the image's alpha rescaled.

use alloc::vec;
use alloc::vec::Vec;

use crate::Error;
use crate::chroma::ChromaFormat;
use crate::image::Image;
use crate::math::round;
use crate::quant::BitDepth;

/// A memory layout of an image's codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    Planar,
    SemiPlanar { vu: bool },
    Packed422 { uyvy: bool },
    V210,
    Ayuv,
    Y410,
    Y416,
    Rgb8,
    Rgba8,
    Bgra8,
    Rgba16,
    A2R10G10B10,
    A2B10G10R10,
}

impl Layout {
    /// Whether the layout can hold an image of `chroma` at `depth`.
    #[must_use]
    pub const fn fits(self, chroma: ChromaFormat, depth: BitDepth) -> bool {
        use ChromaFormat as C;
        let bits = depth.bits();
        match self {
            Self::Planar => true,
            Self::SemiPlanar { .. } => matches!(chroma, C::Yuv420 | C::Yuv422 | C::Yuv444),
            Self::Packed422 { .. } => matches!(chroma, C::Yuv422),
            Self::V210 => matches!(chroma, C::Yuv422) && bits == 10,
            Self::Ayuv | Self::Rgb8 | Self::Rgba8 | Self::Bgra8 => {
                matches!(chroma, C::Yuv444) && bits == 8
            }
            Self::Y410 | Self::A2R10G10B10 | Self::A2B10G10R10 => {
                matches!(chroma, C::Yuv444) && bits == 10
            }
            Self::Y416 => matches!(chroma, C::Yuv444) && bits > 8,
            Self::Rgba16 => matches!(chroma, C::Yuv444) && bits == 16,
        }
    }

    /// Bytes of a `width` x `height` image.
    #[must_use]
    pub const fn size(
        self,
        width: usize,
        height: usize,
        chroma: ChromaFormat,
        depth: BitDepth,
    ) -> usize {
        let (cw, ch) = chroma.plane(width, height);
        let word = if depth.bits() > 8 { 2 } else { 1 };
        let px = width * height;
        match self {
            Self::Planar | Self::SemiPlanar { .. } => (px + 2 * cw * ch) * word,
            Self::Packed422 { .. } => 2 * cw * 2 * height * word,
            Self::V210 => v210_stride(width) * height,
            Self::Ayuv
            | Self::Y410
            | Self::Rgba8
            | Self::Bgra8
            | Self::A2R10G10B10
            | Self::A2B10G10R10 => 4 * px,
            Self::Y416 | Self::Rgba16 => 8 * px,
            Self::Rgb8 => 3 * px,
        }
    }
}

/// Bytes per v210 row: groups of 48 pixels in 128 bytes.
#[must_use]
pub const fn v210_stride(width: usize) -> usize {
    width.div_ceil(48) * 128
}

fn rescale(code: u16, from_max: u16, to_max: u16) -> u16 {
    if from_max == to_max {
        return code;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "0..=to_max"
    )]
    let c = round(f64::from(code) * f64::from(to_max) / f64::from(from_max)) as u16;
    c
}

struct Out(Vec<u8>);

impl Out {
    fn u8(&mut self, v: u16) {
        #[allow(clippy::cast_possible_truncation, reason = "8-bit codes")]
        self.0.push(v as u8);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    /// A code as a byte (8 bits) or a word, low- or high-aligned.
    fn sample(&mut self, v: u16, depth: BitDepth, high: bool) {
        if depth.bits() == 8 {
            self.u8(v);
        } else if high {
            self.u16(v << (16 - depth.bits()));
        } else {
            self.u16(v);
        }
    }
}

struct In<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl In<'_> {
    fn u8(&mut self) -> u16 {
        let v = self.bytes[self.at];
        self.at += 1;
        u16::from(v)
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.bytes[self.at], self.bytes[self.at + 1]]);
        self.at += 2;
        v
    }
    fn u32(&mut self) -> u32 {
        let b = &self.bytes[self.at..self.at + 4];
        self.at += 4;
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    fn sample(&mut self, depth: BitDepth, high: bool) -> u16 {
        if depth.bits() == 8 {
            self.u8()
        } else if high {
            self.u16() >> (16 - depth.bits())
        } else {
            self.u16() & depth.max()
        }
    }
}

/// `image` laid out as `layout`.
///
/// # Errors
///
/// [`Error::Layout`] when the layout can't hold the image's chroma format
/// or bit depth; [`Error::Size`] / [`Error::Code`] for a malformed image.
#[allow(
    clippy::too_many_lines,
    reason = "one arm per layout, side by side with unpack"
)]
pub fn pack(image: &Image, layout: Layout) -> Result<Vec<u8>, Error> {
    image.check()?;
    if !layout.fits(image.chroma, image.depth) {
        return Err(Error::Layout);
    }
    let (w, h, depth) = (image.width, image.height, image.depth);
    let (cw, ch) = image.chroma_size();
    let [y, cb, cr] = &image.planes;
    let max = depth.max();
    let alpha = |i: usize| image.alpha.as_ref().map_or(max, |a| a[i]);
    let mut out = Out(Vec::with_capacity(layout.size(w, h, image.chroma, depth)));
    match layout {
        Layout::Planar => {
            for &v in y.iter().chain(cb).chain(cr) {
                out.sample(v, depth, false);
            }
        }
        Layout::SemiPlanar { vu } => {
            for &v in y {
                out.sample(v, depth, true);
            }
            for i in 0..cw * ch {
                let (a, b) = if vu { (cr[i], cb[i]) } else { (cb[i], cr[i]) };
                out.sample(a, depth, true);
                out.sample(b, depth, true);
            }
        }
        Layout::Packed422 { uyvy } => {
            for row in 0..h {
                for i in 0..cw {
                    let y0 = y[row * w + 2 * i];
                    // An odd width repeats the last luma sample.
                    let y1 = y[row * w + (2 * i + 1).min(w - 1)];
                    let (u, v) = (cb[row * cw + i], cr[row * cw + i]);
                    let order = if uyvy { [u, y0, v, y1] } else { [y0, u, y1, v] };
                    for s in order {
                        out.sample(s, depth, true);
                    }
                }
            }
        }
        Layout::V210 => {
            for row in 0..h {
                let start = out.0.len();
                let luma = |x: usize| y[row * w + x.min(w - 1)];
                let chroma = |p: &[u16], i: usize| p[row * cw + i.min(cw - 1)];
                for g in 0..w.div_ceil(6) {
                    let (x, c) = (6 * g, 3 * g);
                    let word = |a: u16, b: u16, d: u16| {
                        u32::from(a) | u32::from(b) << 10 | u32::from(d) << 20
                    };
                    out.u32(word(chroma(cb, c), luma(x), chroma(cr, c)));
                    out.u32(word(luma(x + 1), chroma(cb, c + 1), luma(x + 2)));
                    out.u32(word(chroma(cr, c + 1), luma(x + 3), chroma(cb, c + 2)));
                    out.u32(word(luma(x + 4), chroma(cr, c + 2), luma(x + 5)));
                }
                out.0.resize(start + v210_stride(w), 0);
            }
        }
        Layout::Ayuv => {
            for i in 0..w * h {
                for s in [cr[i], cb[i], y[i], alpha(i)] {
                    out.u8(s);
                }
            }
        }
        Layout::Y410 => {
            for i in 0..w * h {
                let a = u32::from(rescale(alpha(i), max, 3));
                out.u32(
                    u32::from(cb[i]) | u32::from(y[i]) << 10 | u32::from(cr[i]) << 20 | a << 30,
                );
            }
        }
        Layout::Y416 => {
            for i in 0..w * h {
                for s in [cb[i], y[i], cr[i], alpha(i)] {
                    out.sample(s, depth, true);
                }
            }
        }
        // GBR: planes are G, B, R.
        Layout::Rgb8 | Layout::Rgba8 | Layout::Bgra8 | Layout::Rgba16 => {
            for i in 0..w * h {
                let (r, g, b) = (cr[i], y[i], cb[i]);
                let px: &[u16] = match layout {
                    Layout::Rgb8 => &[r, g, b],
                    Layout::Bgra8 => &[b, g, r, alpha(i)],
                    _ => &[r, g, b, alpha(i)],
                };
                for &s in px {
                    out.sample(s, depth, false);
                }
            }
        }
        Layout::A2R10G10B10 | Layout::A2B10G10R10 => {
            for i in 0..w * h {
                let (r, g, b) = (u32::from(cr[i]), u32::from(y[i]), u32::from(cb[i]));
                let a = u32::from(rescale(alpha(i), max, 3));
                let (lo, hi) = if layout == Layout::A2R10G10B10 {
                    (b, r)
                } else {
                    (r, b)
                };
                out.u32(lo | g << 10 | hi << 20 | a << 30);
            }
        }
    }
    Ok(out.0)
}

/// An image of `width` x `height`, `chroma`, `depth` read from `bytes` laid
/// out as `layout`.
///
/// # Errors
///
/// [`Error::Layout`] when the layout can't hold that format,
/// [`Error::Size`] when `bytes` has another length than
/// [`Layout::size`].
#[allow(
    clippy::too_many_lines,
    reason = "one arm per layout, side by side with pack"
)]
pub fn unpack(
    bytes: &[u8],
    layout: Layout,
    width: usize,
    height: usize,
    chroma: ChromaFormat,
    depth: BitDepth,
) -> Result<Image, Error> {
    if !layout.fits(chroma, depth) {
        return Err(Error::Layout);
    }
    let mut image = Image::new(width, height, chroma, depth)?;
    if bytes.len() != layout.size(width, height, chroma, depth) {
        return Err(Error::Size);
    }
    let (w, h) = (width, height);
    let (cw, ch) = image.chroma_size();
    let max = depth.max();
    let mut input = In { bytes, at: 0 };
    let mut alpha = vec![0; w * h];
    let has_alpha = !matches!(
        layout,
        Layout::Planar
            | Layout::SemiPlanar { .. }
            | Layout::Packed422 { .. }
            | Layout::V210
            | Layout::Rgb8
    );
    let [y, cb, cr] = &mut image.planes;
    match layout {
        Layout::Planar => {
            for v in y.iter_mut().chain(cb.iter_mut()).chain(cr.iter_mut()) {
                *v = input.sample(depth, false);
            }
        }
        Layout::SemiPlanar { vu } => {
            for v in y.iter_mut() {
                *v = input.sample(depth, true);
            }
            for i in 0..cw * ch {
                let (a, b) = (input.sample(depth, true), input.sample(depth, true));
                (cb[i], cr[i]) = if vu { (b, a) } else { (a, b) };
            }
        }
        Layout::Packed422 { uyvy } => {
            for row in 0..h {
                for i in 0..cw {
                    let s = [0; 4].map(|_| input.sample(depth, true));
                    let [y0, u, y1, v] = if uyvy { [s[1], s[0], s[3], s[2]] } else { s };
                    y[row * w + 2 * i] = y0;
                    if 2 * i + 1 < w {
                        y[row * w + 2 * i + 1] = y1;
                    }
                    (cb[row * cw + i], cr[row * cw + i]) = (u, v);
                }
            }
        }
        Layout::V210 => {
            for row in 0..h {
                let start = input.at;
                for g in 0..w.div_ceil(6) {
                    let words = [0; 4].map(|_| input.u32());
                    #[allow(clippy::cast_possible_truncation, reason = "10 bits")]
                    let f = |word: u32, k: u32| ((word >> (10 * k)) & 0x3ff) as u16;
                    let (x, c) = (6 * g, 3 * g);
                    let lumas = [
                        f(words[0], 1),
                        f(words[1], 0),
                        f(words[1], 2),
                        f(words[2], 1),
                        f(words[3], 0),
                        f(words[3], 2),
                    ];
                    let cbs = [f(words[0], 0), f(words[1], 1), f(words[2], 2)];
                    let crs = [f(words[0], 2), f(words[2], 0), f(words[3], 1)];
                    for (k, &l) in lumas.iter().enumerate() {
                        if x + k < w {
                            y[row * w + x + k] = l;
                        }
                    }
                    for k in 0..3 {
                        if c + k < cw {
                            cb[row * cw + c + k] = cbs[k];
                            cr[row * cw + c + k] = crs[k];
                        }
                    }
                }
                input.at = start + v210_stride(w);
            }
        }
        Layout::Ayuv => {
            for i in 0..w * h {
                (cr[i], cb[i], y[i], alpha[i]) = (input.u8(), input.u8(), input.u8(), input.u8());
            }
        }
        Layout::Y410 => {
            for i in 0..w * h {
                let word = input.u32();
                #[allow(clippy::cast_possible_truncation, reason = "10 and 2 bits")]
                let f = |shift: u32, bits: u32| ((word >> shift) & ((1 << bits) - 1)) as u16;
                (cb[i], y[i], cr[i]) = (f(0, 10), f(10, 10), f(20, 10));
                alpha[i] = rescale(f(30, 2), 3, max);
            }
        }
        Layout::Y416 => {
            for i in 0..w * h {
                let s = [0; 4].map(|_| input.sample(depth, true));
                (cb[i], y[i], cr[i], alpha[i]) = (s[0], s[1], s[2], s[3]);
            }
        }
        Layout::Rgb8 | Layout::Rgba8 | Layout::Bgra8 | Layout::Rgba16 => {
            for i in 0..w * h {
                let (r, g, b, a) = match layout {
                    Layout::Rgb8 => (input.u8(), input.u8(), input.u8(), max),
                    Layout::Bgra8 => {
                        let (b, g, r, a) = (input.u8(), input.u8(), input.u8(), input.u8());
                        (r, g, b, a)
                    }
                    _ => {
                        let s = [0; 4].map(|_| input.sample(depth, false));
                        (s[0], s[1], s[2], s[3])
                    }
                };
                (y[i], cb[i], cr[i], alpha[i]) = (g, b, r, a);
            }
        }
        Layout::A2R10G10B10 | Layout::A2B10G10R10 => {
            for i in 0..w * h {
                let word = input.u32();
                #[allow(clippy::cast_possible_truncation, reason = "10 and 2 bits")]
                let f = |shift: u32, bits: u32| ((word >> shift) & ((1 << bits) - 1)) as u16;
                let (lo, g, hi) = (f(0, 10), f(10, 10), f(20, 10));
                let (r, b) = if layout == Layout::A2R10G10B10 {
                    (hi, lo)
                } else {
                    (lo, hi)
                };
                (y[i], cb[i], cr[i]) = (g, b, r);
                alpha[i] = rescale(f(30, 2), 3, max);
            }
        }
    }
    if has_alpha {
        image.alpha = Some(alpha);
    }
    Ok(image)
}

/// Linear or encoded RGBA as half floats (`R16G16B16A16_SFLOAT`, scRGB),
/// `R, G, B, A` per pixel.
#[must_use]
pub fn pack_rgba_f16(pixels: &[[f64; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 * pixels.len());
    for px in pixels {
        for &v in px {
            out.extend_from_slice(&crate::half::from_f64(v).to_le_bytes());
        }
    }
    out
}

/// [`pack_rgba_f16`] back.
///
/// # Errors
///
/// [`Error::Size`] when `bytes` isn't a whole number of pixels.
pub fn unpack_rgba_f16(bytes: &[u8]) -> Result<Vec<[f64; 4]>, Error> {
    if !bytes.len().is_multiple_of(8) {
        return Err(Error::Size);
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|p| {
            [0, 1, 2, 3].map(|k| crate::half::to_f64(u16::from_le_bytes([p[2 * k], p[2 * k + 1]])))
        })
        .collect())
}

/// RGBA as `f32` (`R32G32B32A32_SFLOAT`).
#[must_use]
pub fn pack_rgba_f32(pixels: &[[f64; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 * pixels.len());
    for px in pixels {
        for &v in px {
            #[allow(clippy::cast_possible_truncation, reason = "f32 is the format")]
            out.extend_from_slice(&(v as f32).to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize, chroma: ChromaFormat, bits: u8, alpha: bool) -> Image {
        let depth = BitDepth::new(bits).unwrap();
        let mut img = Image::new(w, h, chroma, depth).unwrap();
        let max = u32::from(depth.max());
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[allow(clippy::cast_possible_truncation, reason = "below max")]
            let v = ((seed >> 8) % (max + 1)) as u16;
            v
        };
        for p in &mut img.planes {
            for v in p.iter_mut() {
                *v = next();
            }
        }
        if alpha {
            img.alpha = Some((0..w * h).map(|_| next()).collect());
        }
        img
    }

    #[test]
    fn every_layout_round_trips() {
        use ChromaFormat::{Yuv420, Yuv422, Yuv444};
        let cases: &[(Layout, ChromaFormat, &[u8])] = &[
            (Layout::Planar, Yuv420, &[8, 10, 12, 14, 16]),
            (Layout::Planar, Yuv422, &[8, 10]),
            (Layout::Planar, Yuv444, &[8, 12]),
            (Layout::Planar, ChromaFormat::Monochrome, &[8, 16]),
            (Layout::Planar, ChromaFormat::Yuv440, &[10]),
            (Layout::SemiPlanar { vu: false }, Yuv420, &[8, 10, 12, 16]),
            (Layout::SemiPlanar { vu: true }, Yuv420, &[8]),
            (Layout::SemiPlanar { vu: false }, Yuv422, &[8, 10, 16]),
            (Layout::SemiPlanar { vu: false }, Yuv444, &[8, 10, 16]),
            (Layout::Packed422 { uyvy: false }, Yuv422, &[8, 10, 12, 16]),
            (Layout::Packed422 { uyvy: true }, Yuv422, &[8]),
            (Layout::V210, Yuv422, &[10]),
            (Layout::Y416, Yuv444, &[12, 16]),
        ];
        for &(layout, chroma, depths) in cases {
            for &bits in depths {
                // Even and odd sizes, and v210's partial groups.
                for (w, h) in [(4, 2), (7, 3), (50, 2)] {
                    let img = image(w, h, chroma, bits, false);
                    let bytes = pack(&img, layout).unwrap();
                    assert_eq!(bytes.len(), layout.size(w, h, chroma, img.depth));
                    let mut back = unpack(&bytes, layout, w, h, chroma, img.depth).unwrap();
                    back.alpha = None;
                    assert_eq!(back, img, "{layout:?} {chroma:?} {bits} {w}x{h}");
                }
            }
        }
        // Alpha layouts.
        for (layout, bits) in [
            (Layout::Ayuv, 8),
            (Layout::Y416, 16),
            (Layout::Rgba8, 8),
            (Layout::Bgra8, 8),
            (Layout::Rgba16, 16),
        ] {
            let img = image(5, 3, Yuv444, bits, true);
            let back = unpack(
                &pack(&img, layout).unwrap(),
                layout,
                5,
                3,
                Yuv444,
                img.depth,
            )
            .unwrap();
            assert_eq!(back, img, "{layout:?}");
        }
        let rgb = image(5, 3, Yuv444, 8, false);
        assert_eq!(
            unpack(
                &pack(&rgb, Layout::Rgb8).unwrap(),
                Layout::Rgb8,
                5,
                3,
                Yuv444,
                rgb.depth
            )
            .unwrap(),
            rgb
        );
    }

    #[test]
    fn two_bit_alpha_keeps_the_four_levels() {
        for layout in [Layout::Y410, Layout::A2R10G10B10, Layout::A2B10G10R10] {
            let mut img = image(4, 1, ChromaFormat::Yuv444, 10, false);
            img.alpha = Some(vec![0, 341, 682, 1023]);
            let back = unpack(
                &pack(&img, layout).unwrap(),
                layout,
                4,
                1,
                ChromaFormat::Yuv444,
                img.depth,
            )
            .unwrap();
            assert_eq!(back.planes, img.planes, "{layout:?}");
            assert_eq!(back.alpha, img.alpha, "{layout:?}");
        }
    }

    #[test]
    fn bits_sit_where_the_formats_put_them() {
        let d10 = BitDepth::B10;
        let mut img = Image::new(2, 2, ChromaFormat::Yuv420, d10).unwrap();
        img.planes = [vec![0x3ff, 1, 2, 3], vec![0x200], vec![0x155]];
        // P010: luma high-aligned, then Cb, Cr.
        let p010 = pack(&img, Layout::SemiPlanar { vu: false }).unwrap();
        assert_eq!(&p010[..2], &0xffc0u16.to_le_bytes());
        assert_eq!(&p010[8..12], &[0x00, 0x80, 0x40, 0x55]);
        // yuv420p10le: low-aligned.
        assert_eq!(
            &pack(&img, Layout::Planar).unwrap()[..2],
            &0x03ffu16.to_le_bytes()
        );
        // Y410: Cb | Y << 10 | Cr << 20 | A << 30.
        let mut px = Image::new(1, 1, ChromaFormat::Yuv444, d10).unwrap();
        px.planes = [vec![0x2aa], vec![0x111], vec![0x3ff]];
        let word = 0x111 | 0x2aa << 10 | 0x3ff << 20 | 0x3 << 30;
        assert_eq!(pack(&px, Layout::Y410).unwrap(), u32::to_le_bytes(word));
        // A2R10G10B10: B | G << 10 | R << 20 (planes are G, B, R).
        assert_eq!(
            pack(&px, Layout::A2R10G10B10).unwrap(),
            u32::to_le_bytes(0x111 | 0x2aa << 10 | 0x3ff << 20 | 0x3 << 30)
        );
        assert_eq!(
            pack(&px, Layout::A2B10G10R10).unwrap(),
            u32::to_le_bytes(0x3ff | 0x2aa << 10 | 0x111 << 20 | 0x3 << 30)
        );
        // AYUV: Cr, Cb, Y, A.
        let mut a = Image::new(1, 1, ChromaFormat::Yuv444, BitDepth::B8).unwrap();
        a.planes = [vec![1], vec![2], vec![3]];
        assert_eq!(pack(&a, Layout::Ayuv).unwrap(), [3, 2, 1, 255]);
        assert_eq!(pack(&a, Layout::Bgra8).unwrap(), [2, 1, 3, 255]);
        // YUY2 and UYVY.
        let mut p = Image::new(2, 1, ChromaFormat::Yuv422, BitDepth::B8).unwrap();
        p.planes = [vec![10, 11], vec![20], vec![30]];
        assert_eq!(
            pack(&p, Layout::Packed422 { uyvy: false }).unwrap(),
            [10, 20, 11, 30]
        );
        assert_eq!(
            pack(&p, Layout::Packed422 { uyvy: true }).unwrap(),
            [20, 10, 30, 11]
        );
        // v210: word 0 is Cb0 | Y0 << 10 | Cr0 << 20; a row of 6 takes 128 bytes.
        let mut v = Image::new(6, 1, ChromaFormat::Yuv422, d10).unwrap();
        v.planes = [
            vec![100, 101, 102, 103, 104, 105],
            vec![200, 201, 202],
            vec![300, 301, 302],
        ];
        let bytes = pack(&v, Layout::V210).unwrap();
        assert_eq!(bytes.len(), 128);
        let w = |k: usize| {
            u32::from_le_bytes([
                bytes[4 * k],
                bytes[4 * k + 1],
                bytes[4 * k + 2],
                bytes[4 * k + 3],
            ])
        };
        // Three 10-bit codes per word, lowest first.
        let codes = |a: u32, b: u32, c: u32| a | b << 10 | c << 20;
        assert_eq!(w(0), codes(200, 100, 300));
        assert_eq!(w(1), codes(101, 201, 102));
        assert_eq!(w(2), codes(301, 103, 202));
        assert_eq!(w(3), codes(104, 302, 105));
        assert!(bytes[16..].iter().all(|&b| b == 0));
    }

    #[test]
    fn layouts_refuse_what_they_cant_hold() {
        let img = image(4, 2, ChromaFormat::Yuv420, 10, false);
        assert_eq!(pack(&img, Layout::V210), Err(Error::Layout));
        assert_eq!(pack(&img, Layout::Rgba8), Err(Error::Layout));
        assert_eq!(
            unpack(
                &[0; 3],
                Layout::Planar,
                4,
                2,
                ChromaFormat::Yuv420,
                BitDepth::B8
            ),
            Err(Error::Size)
        );
        let mut bad = image(4, 2, ChromaFormat::Yuv420, 8, false);
        bad.planes[0][0] = 300;
        assert_eq!(pack(&bad, Layout::Planar), Err(Error::Code));
    }

    #[test]
    fn float_rgba_round_trips_through_half() {
        let px = [[0.0, 0.5, 1.0, 1.0], [-0.25, 12.5, 65504.0, 0.0]];
        let back = unpack_rgba_f16(&pack_rgba_f16(&px)).unwrap();
        assert_eq!(back, px);
        assert_eq!(pack_rgba_f32(&px).len(), 32);
    }
}
