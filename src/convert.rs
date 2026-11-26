//! Conversion between any two colour descriptions, bit depths, ranges and
//! chroma formats: the reference pipeline, in `f64`.
//!
//! 1. Codes to signals ([`quant::dequantize`]).
//! 2. Chroma to full resolution ([`chroma::upsample`], at the source's
//!    location).
//! 3. `Y'CbCr` to `R'G'B'` (the source matrix).
//! 4. Signals to display light in cd/m² ([`hdr::to_nits`]).
//! 5. Source to target primaries, white adapted ([`Primaries::conversion`](crate::primaries::Primaries::conversion)).
//! 6. Tone mapping to the target's peak ([`ToneMap`]).
//! 7. Gamut clipping: no negative light where the target can't carry it.
//! 8. Display light to target signals, `R'G'B'` to `Y'CbCr`, chroma
//!    subsampled ([`chroma::downsample`]), signals to codes.
//!
//! Where source and target share curve and primaries, steps 4 to 8's
//! light part is skipped: the signals pass as they are, so a change of
//! matrix, range, bit depth or chroma format alone touches nothing else
//! (head- and footroom included).

use alloc::vec::Vec;

use crate::chroma::{self, ChromaFormat, ChromaLocation, Downsample, Upsample};
use crate::cicp::Cicp;
use crate::hdr::{self, Eetf, Luminance, ToneMap};
use crate::image::Image;
use crate::math::{self, Mat3};
use crate::primaries::Adaptation;
use crate::quant::{self, BitDepth, Range, Signal};
use crate::{Colour, Error};

/// What a coded image is: its colour, bit depth and chroma format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Format {
    pub cicp: Cicp,
    pub depth: BitDepth,
    pub chroma: ChromaFormat,
    pub location: ChromaLocation,
}

/// Out-of-gamut colours (negative light after a change of primaries).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Gamut {
    /// Negative light is set to 0, unless the target curve carries it
    /// (xvYCC, BT.1361).
    #[default]
    Clip,
    /// Kept; quantization clips what the codes can't hold.
    Keep,
}

/// The choices the standards leave open.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Options {
    pub luminance: Luminance,
    pub tone_map: ToneMap,
    pub adaptation: Adaptation,
    pub gamut: Gamut,
    pub upsample: Upsample,
    pub downsample: Downsample,
}

/// One colour description to another, per pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Converter {
    from: Colour,
    to: Colour,
    /// The light part is skipped (same curve and primaries).
    same_light: bool,
    gamut: Option<Mat3>,
    eetf: Option<Eetf>,
    peak: f64,
    options: Options,
}

impl Converter {
    /// # Errors
    ///
    /// What [`Cicp::resolve`] refuses, on either side.
    pub fn new(from: &Cicp, to: &Cicp, options: &Options) -> Result<Self, Error> {
        let (from, to) = (from.resolve()?, to.resolve()?);
        let same_primaries = from.cicp.primaries == to.cicp.primaries
            || from.primaries.chromaticities == to.primaries.chromaticities;
        let same_light = from.curve == to.curve && same_primaries;
        let gamut =
            (!same_primaries).then(|| from.primaries.conversion(&to.primaries, options.adaptation));
        let peak = hdr::peak(to.curve, &options.luminance);
        let eetf = match options.tone_map {
            ToneMap::Clip => None,
            ToneMap::Bt2390 {
                source_black,
                source_peak,
            } => Some(Eetf::new(source_black, source_peak, 0.0, peak)),
        };
        Ok(Self {
            from,
            to,
            same_light,
            gamut,
            eetf,
            peak,
            options: *options,
        })
    }

    /// Whether source and target share curve and primaries: then the
    /// signals pass the light part unchanged, and the whole conversion is
    /// the two matrices and the quantizations.
    #[must_use]
    pub const fn same_light(&self) -> bool {
        self.same_light
    }

    #[must_use]
    pub const fn from(&self) -> &Colour {
        &self.from
    }

    #[must_use]
    pub const fn to(&self) -> &Colour {
        &self.to
    }

    /// Display light in cd/m² (target primaries) after gamut, tone mapping
    /// and clipping, from source `[R', G', B']`.
    #[must_use]
    pub fn light(&self, rgb: [f64; 3]) -> [f64; 3] {
        let lum = &self.options.luminance;
        self.light_of_nits(hdr::to_nits(
            self.from.curve,
            rgb,
            lum,
            self.from.primaries.luminance(),
        ))
    }

    /// [`light`](Self::light) from the source's display light: gamut, tone
    /// mapping, clipping. The fast paths call this very function, so their
    /// results in this step are the reference's.
    #[must_use]
    pub fn light_of_nits(&self, mut nits: [f64; 3]) -> [f64; 3] {
        if let Some(m) = &self.gamut {
            nits = math::apply(m, nits);
        }
        if let Some(eetf) = &self.eetf {
            nits = eetf.apply_nits(nits);
        }
        let floor = if self.options.gamut == Gamut::Clip && !self.to.curve.is_extended() {
            0.0
        } else {
            f64::NEG_INFINITY
        };
        nits.map(|l| l.min(self.peak).max(floor))
    }

    /// Source `[R', G', B']` to target `[R', G', B']`.
    #[must_use]
    pub fn rgb(&self, rgb: [f64; 3]) -> [f64; 3] {
        if self.same_light {
            return rgb;
        }
        let nits = self.light(rgb);
        hdr::from_nits(
            self.to.curve,
            nits,
            &self.options.luminance,
            self.to.primaries.luminance(),
        )
    }

    /// Source `[Y', Cb, Cr]` to target `[Y', Cb, Cr]` (each through its
    /// matrix).
    #[must_use]
    pub fn ycbcr(&self, ycc: [f64; 3]) -> [f64; 3] {
        if self.same_light && self.from.matrix == self.to.matrix {
            return ycc;
        }
        self.to
            .matrix
            .to_ycbcr(self.rgb(self.from.matrix.to_rgb(ycc)))
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

/// `image`, coded as `from`, coded as `to`.
///
/// # Errors
///
/// [`Error::Format`] when the image isn't of `from`'s chroma format and
/// bit depth, what [`Converter::new`] refuses, [`Error::Size`] /
/// [`Error::Code`] for a malformed image.
pub fn convert(
    image: &Image,
    from: &Format,
    to: &Format,
    options: &Options,
) -> Result<Image, Error> {
    image.check()?;
    if image.chroma != from.chroma || image.depth != from.depth {
        return Err(Error::Format);
    }
    let converter = Converter::new(&from.cicp, &to.cicp, options)?;
    let (w, h) = (image.width, image.height);
    let n = w * h;
    let src_range = converter.from.range;
    let dst_range = converter.to.range;
    let src_sig = signals(&converter.from);
    let dst_sig = signals(&converter.to);
    let deq = |plane: &[u16], s: Signal| -> Vec<f64> {
        plane
            .iter()
            .map(|&c| quant::dequantize(c, from.depth, src_range, s))
            .collect()
    };

    // 1, 2: signals at full resolution.
    let y = deq(&image.planes[0], src_sig[0]);
    let (cb, cr) = if from.chroma == ChromaFormat::Monochrome {
        // No colour: neutral differences, or for GBR, grey (B = R = G).
        if src_sig[1] == Signal::Chroma {
            (alloc::vec![0.0; n], alloc::vec![0.0; n])
        } else {
            (y.clone(), y.clone())
        }
    } else {
        let up =
            |p: Vec<f64>| chroma::upsample(&p, w, h, from.chroma, from.location, options.upsample);
        (
            up(deq(&image.planes[1], src_sig[1]))?,
            up(deq(&image.planes[2], src_sig[2]))?,
        )
    };

    // 3 to 8: per pixel.
    let mut out = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for i in 0..n {
        let v = converter.ycbcr([y[i], cb[i], cr[i]]);
        for (plane, value) in out.iter_mut().zip(v) {
            plane.push(value);
        }
    }

    // Chroma subsampling and codes.
    let mut result = Image::new(w, h, to.chroma, to.depth)?;
    let q = |plane: &[f64], s: Signal| -> Vec<u16> {
        plane
            .iter()
            .map(|&e| quant::quantize(e, to.depth, dst_range, s))
            .collect()
    };
    result.planes[0] = q(&out[0], dst_sig[0]);
    if to.chroma != ChromaFormat::Monochrome {
        for k in 1..3 {
            let sub =
                chroma::downsample(&out[k], w, h, to.chroma, to.location, options.downsample)?;
            result.planes[k] = q(&sub, dst_sig[k]);
        }
    }
    result.alpha = image.alpha.as_ref().map(|a| {
        a.iter()
            .map(|&c| quant::requantize(c, from.depth, to.depth, Range::Full, Signal::Luma))
            .collect()
    });
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cicp::{ColourPrimaries, MatrixCoefficients, TransferCharacteristics};
    use crate::math::close;
    use alloc::vec;

    fn fmt(cicp: Cicp, bits: u8, chroma: ChromaFormat) -> Format {
        Format {
            cicp,
            depth: BitDepth::new(bits).unwrap(),
            chroma,
            location: ChromaLocation::Left,
        }
    }

    fn pixel(f: Format, codes: [u16; 3]) -> Image {
        let mut img = Image::new(1, 1, ChromaFormat::Yuv444, f.depth).unwrap();
        img.planes = [vec![codes[0]], vec![codes[1]], vec![codes[2]]];
        img
    }

    fn codes(img: &Image) -> [u16; 3] {
        [img.planes[0][0], img.planes[1][0], img.planes[2][0]]
    }

    #[test]
    fn srgb_red_to_bt709_video_is_63_102_240() {
        // Same primaries; sRGB display light re-encoded through BT.1886.
        // The classic (63, 102, 240) is BT.709 red with the signal as is:
        // there R' = 1 either way, so it holds here too.
        let src = fmt(Cicp::SRGB, 8, ChromaFormat::Yuv444);
        let dst = fmt(Cicp::BT709, 8, ChromaFormat::Yuv444);
        // GBR planes: G, B, R.
        let red = pixel(src, [0, 0, 255]);
        let out = convert(&red, &src, &dst, &Options::default()).unwrap();
        assert_eq!(codes(&out), [63, 102, 240]);
        let bt601 = fmt(Cicp::BT601_525, 8, ChromaFormat::Yuv444);
        let red601 = fmt(
            Cicp {
                primaries: ColourPrimaries::Bt601,
                ..Cicp::SRGB
            },
            8,
            ChromaFormat::Yuv444,
        );
        let out = convert(
            &pixel(red601, [0, 0, 255]),
            &red601,
            &bt601,
            &Options::default(),
        )
        .unwrap();
        assert_eq!(codes(&out), [81, 90, 240]);
    }

    #[test]
    fn range_depth_and_matrix_changes_alone_are_exact() {
        let limited = fmt(Cicp::BT709, 8, ChromaFormat::Yuv444);
        let full = fmt(
            Cicp {
                full_range: true,
                ..Cicp::BT709
            },
            8,
            ChromaFormat::Yuv444,
        );
        let ten = fmt(Cicp::BT709, 10, ChromaFormat::Yuv444);
        for c in [[16, 128, 128], [235, 240, 16], [0, 0, 255], [126, 77, 201]] {
            let px = pixel(limited, c);
            let up = convert(&px, &limited, &ten, &Options::default()).unwrap();
            assert_eq!(codes(&up), c.map(|v| v << 2), "head- and footroom kept");
            let back = convert(&up, &ten, &limited, &Options::default()).unwrap();
            assert_eq!(codes(&back), c);
        }
        let black = convert(
            &pixel(limited, [16, 128, 128]),
            &limited,
            &full,
            &Options::default(),
        )
        .unwrap();
        assert_eq!(codes(&black), [0, 128, 128]);
        let white = convert(
            &pixel(limited, [235, 128, 128]),
            &limited,
            &full,
            &Options::default(),
        )
        .unwrap();
        assert_eq!(codes(&white), [255, 128, 128]);
        // Codes in the nominal span survive limited -> 16-bit full -> limited
        // (full range has no foot- or headroom to keep the rest in).
        let deep = fmt(
            Cicp {
                full_range: true,
                ..Cicp::BT709
            },
            16,
            ChromaFormat::Yuv444,
        );
        for yv in (16..=235).step_by(9) {
            for u in (16..=240).step_by(14) {
                let px = pixel(limited, [yv, u, 256 - u]);
                let there = convert(&px, &limited, &deep, &Options::default()).unwrap();
                assert_eq!(
                    codes(&convert(&there, &deep, &limited, &Options::default()).unwrap()),
                    codes(&px)
                );
            }
        }
    }

    #[test]
    fn sdr_white_lands_on_bt2408s_codes_in_hdr() {
        let sdr = fmt(Cicp::BT709, 10, ChromaFormat::Yuv444);
        let pq = fmt(Cicp::HDR10, 10, ChromaFormat::Yuv444);
        let hlg = fmt(Cicp::HLG, 10, ChromaFormat::Yuv444);
        let white = pixel(sdr, [940, 512, 512]);
        let in_pq = convert(&white, &sdr, &pq, &Options::default()).unwrap();
        let in_hlg = convert(&white, &sdr, &hlg, &Options::default()).unwrap();
        // BT.2408 Table 1: 203 cd/m² is 58 % PQ, 75 % HLG; neutral stays neutral.
        let pct = |code: u16| (f64::from(code) - 64.0) / 876.0;
        assert!(
            close(pct(codes(&in_pq)[0]), 0.58, 0.002),
            "{:?}",
            codes(&in_pq)
        );
        assert!(
            close(pct(codes(&in_hlg)[0]), 0.75, 0.002),
            "{:?}",
            codes(&in_hlg)
        );
        assert_eq!(&codes(&in_pq)[1..], [512, 512]);
        assert_eq!(&codes(&in_hlg)[1..], [512, 512]);
        // And back: SDR white again.
        assert_eq!(
            codes(&convert(&in_pq, &pq, &sdr, &Options::default()).unwrap()),
            [940, 512, 512]
        );
    }

    #[test]
    fn hdr_above_sdr_white_clips_or_tone_maps() {
        let pq = fmt(Cicp::HDR10, 10, ChromaFormat::Yuv444);
        let sdr = fmt(Cicp::BT709, 10, ChromaFormat::Yuv444);
        let lum = Luminance {
            sdr_white: 100.0,
            ..Luminance::default()
        };
        let px = |nits: f64| {
            let e = crate::transfer::Curve::Pq.encode(nits / 10_000.0);
            pixel(
                pq,
                [
                    quant::quantize(e, pq.depth, Range::Limited, Signal::Luma),
                    512,
                    512,
                ],
            )
        };
        let clip = Options {
            luminance: lum,
            ..Options::default()
        };
        let mapped = Options {
            luminance: lum,
            tone_map: ToneMap::Bt2390 {
                source_black: 0.0,
                source_peak: 1000.0,
            },
            ..Options::default()
        };
        let y = |img: &Image, o: &Options| codes(&convert(img, &pq, &sdr, o).unwrap())[0];
        // Clipped: 200 and 1000 cd/m² are both SDR white.
        assert_eq!(y(&px(200.0), &clip), 940);
        assert_eq!(y(&px(1000.0), &clip), 940);
        // Tone mapped: 1000 reaches white, 200 stays below it.
        assert_eq!(y(&px(1000.0), &mapped), 940);
        assert!(y(&px(200.0), &mapped) < 940);
        assert!(y(&px(50.0), &mapped) < y(&px(200.0), &mapped));
    }

    #[test]
    fn gamut_conversion_and_clipping() {
        let rgb2020 = Cicp {
            primaries: ColourPrimaries::Bt2020,
            transfer: TransferCharacteristics::Linear,
            matrix: MatrixCoefficients::Identity,
            full_range: true,
        };
        let rgb709 = Cicp {
            primaries: ColourPrimaries::Bt709,
            ..rgb2020
        };
        let lum = Luminance {
            sdr_white: 1.0,
            ..Luminance::default()
        };
        let options = Options {
            luminance: lum,
            ..Options::default()
        };
        let to2020 = Converter::new(&rgb709, &rgb2020, &options).unwrap();
        // BT.709 red in BT.2020 (BT.2087's first column).
        let red = to2020.rgb([1.0, 0.0, 0.0]);
        assert!(
            close(red[0], 0.6274, 1e-4)
                && close(red[1], 0.0691, 1e-4)
                && close(red[2], 0.0164, 1e-4)
        );
        // BT.2020 green is outside BT.709: clipped at 0, or kept.
        let to709 = Converter::new(&rgb2020, &rgb709, &options).unwrap();
        let g = to709.rgb([0.0, 1.0, 0.0]);
        assert!(g[0] == 0.0 && g[2] == 0.0 && g[1] > 1.0 - 1e-12);
        let keep = Converter::new(
            &rgb2020,
            &rgb709,
            &Options {
                gamut: Gamut::Keep,
                ..options
            },
        )
        .unwrap();
        assert!(keep.light([0.0, 1.0, 0.0])[0] < 0.0);
    }

    #[test]
    fn chroma_formats_convert_and_flat_colour_survives() {
        let f420 = fmt(Cicp::BT709, 8, ChromaFormat::Yuv420);
        let f444 = fmt(Cicp::BT709, 10, ChromaFormat::Yuv444);
        let mut img = Image::new(6, 4, ChromaFormat::Yuv420, f420.depth).unwrap();
        img.planes = [vec![100; 24], vec![90; 6], vec![200; 6]];
        let up = convert(&img, &f420, &f444, &Options::default()).unwrap();
        assert!(up.planes[1].iter().all(|&c| c == 360) && up.planes[2].iter().all(|&c| c == 800));
        let down = convert(&up, &f444, &f420, &Options::default()).unwrap();
        assert_eq!(down, img);
        // Monochrome to colour: neutral chroma.
        let mono = fmt(Cicp::BT709, 8, ChromaFormat::Monochrome);
        let mut m = Image::new(2, 2, ChromaFormat::Monochrome, mono.depth).unwrap();
        m.planes[0] = vec![50, 60, 70, 80];
        let c = convert(
            &m,
            &mono,
            &fmt(Cicp::BT709, 8, ChromaFormat::Yuv422),
            &Options::default(),
        )
        .unwrap();
        assert_eq!(c.planes[1], [128, 128]);
        assert_eq!(
            convert(&m, &f420, &f444, &Options::default()),
            Err(Error::Format)
        );
    }

    #[test]
    fn identity_formats_are_exact_for_every_code() {
        let f = fmt(Cicp::HDR10, 10, ChromaFormat::Yuv444);
        let mut img = Image::new(1024, 1, ChromaFormat::Yuv444, f.depth).unwrap();
        let ramp: Vec<u16> = (0..1024).collect();
        img.planes = [ramp.clone(), ramp.clone(), ramp];
        assert_eq!(convert(&img, &f, &f, &Options::default()).unwrap(), img);
    }
}
