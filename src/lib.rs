//! # keinbonus
//!
//! Colour ground truth: every colour description of H.273 (CICP), with the
//! reference arithmetic of the standards behind it, for CPU encoders and
//! decoders, GPU renderers and hardware video engines to share and to be
//! checked against.
//!
//! The reference is `f64`, scalar, deterministic (`libm`, the same bits on
//! every platform) and `no_std`. Rounding is H.273's `Round` (halves away
//! from 0) wherever a code comes out. Faster paths (fixed point, SIMD
//! through miraculix, GPU kernels in numerobis) are to match it bit for bit
//! or say by how much they don't.
//!
//! | layer | module | standards |
//! |---|---|---|
//! | code points | [`cicp`] | H.273 / ISO/IEC 23091-2 |
//! | primaries, XYZ, gamut, white adaptation | [`primaries`] | H.273 Table 2, BT.2087, Bradford |
//! | transfer functions | [`transfer`] | H.273 Table 3, BT.709, IEC 61966-2-1/-2-4, BT.1361, ST 2084, BT.2100, ST 428 |
//! | YCbCr, YCgCo, CL, ICtCp | [`matrix`] | H.273 matrix equations, BT.2020, BT.2100 |
//! | bit depth 8..=16, limited and full range | [`quant`] | H.273 equations 20 to 31 |
//! | 4:4:4, 4:2:2, 4:2:0, 4:4:0, chroma location | [`chroma`] | H.273 `ChromaSampleLocType`, AV1 |
//! | memory layouts | [`layout`] | P010 and friends, v210, Y410, AYUV, RGB10A2, FP16 |
//! | display light, HLG OOTF, tone mapping | [`hdr`] | BT.2100, BT.2390, BT.2408 |
//! | HDR10, HDR10+, HLG, Dolby Vision and their metadata | [`hdr::formats`] | ST 2086, CTA-861.3, ST 2094-40, Dolby Vision configuration record |
//! | the whole pipeline | [`convert`] | |
//! | fixed-point fast paths | [`fast`] | |
//! | curve and gamut changes through tables, bit for bit | [`fast::light`] | |
//! | their SIMD kernels (feature `simd`, miraculix) | `simd` | |
//!
//! ```
//! use keinbonus::chroma::{ChromaFormat, ChromaLocation};
//! use keinbonus::cicp::Cicp;
//! use keinbonus::convert::{self, Format, Options};
//! use keinbonus::image::Image;
//! use keinbonus::quant::BitDepth;
//!
//! // One SDR white 4:2:0 frame, BT.709 8-bit, into HDR10 10-bit 4:2:0.
//! let sdr = Format { cicp: Cicp::BT709, depth: BitDepth::B8, chroma: ChromaFormat::Yuv420, location: ChromaLocation::Left };
//! let hdr10 = Format { cicp: Cicp::HDR10, depth: BitDepth::B10, location: ChromaLocation::TopLeft, ..sdr };
//! let mut frame = Image::new(4, 2, ChromaFormat::Yuv420, BitDepth::B8)?;
//! frame.planes = [vec![235; 8], vec![128; 2], vec![128; 2]];
//! let out = convert::convert(&frame, &sdr, &hdr10, &Options::default())?;
//! // BT.2408: SDR white at 203 cd/m², 58 % PQ.
//! assert_eq!(out.planes[0][0], 573);
//! assert_eq!(out.planes[1], [512, 512]);
//! # Ok::<(), keinbonus::Error>(())
//! ```

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use core::fmt;

pub mod chroma;
pub mod cicp;
pub mod convert;
pub mod fast;
pub mod half;
pub mod hdr;
pub mod image;
pub mod layout;
pub mod math;
pub mod matrix;
pub mod primaries;
pub mod quant;
#[cfg(feature = "simd")]
pub mod simd;
pub mod transfer;

pub use cicp::Cicp;

/// What can go wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Error {
    /// A field is "unspecified": nothing to compute with.
    Unspecified(&'static str),
    /// Defined by H.273, without a reference here yet.
    Unsupported(&'static str),
    /// A combination the standards don't define.
    Invalid(&'static str),
    /// A bit depth outside 8..=16.
    BitDepth(u8),
    /// Plane or buffer sizes don't fit the image.
    Size,
    /// A code above the bit depth's maximum.
    Code,
    /// The layout can't hold that chroma format or bit depth.
    Layout,
    /// The image isn't of the format it is said to be.
    Format,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unspecified(what) => write!(f, "{what} unspecified"),
            Self::Unsupported(what) => write!(f, "{what}: no reference implementation yet"),
            Self::Invalid(what) => f.write_str(what),
            Self::BitDepth(bits) => write!(f, "bit depth {bits}: 8 to 16 only"),
            Self::Size => f.write_str("plane or buffer size doesn't fit the image"),
            Self::Code => f.write_str("a code above the bit depth's maximum"),
            Self::Layout => f.write_str("the layout can't hold that chroma format or bit depth"),
            Self::Format => f.write_str("the image isn't of the stated chroma format or bit depth"),
        }
    }
}

impl core::error::Error for Error {}

/// A [`Cicp`] resolved into its functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Colour {
    pub cicp: Cicp,
    pub primaries: primaries::Primaries,
    pub curve: transfer::Curve,
    pub matrix: matrix::Matrix,
    pub range: quant::Range,
}

impl Colour {
    /// # Errors
    ///
    /// As [`Cicp::resolve`].
    pub fn new(cicp: &Cicp) -> Result<Self, Error> {
        let curve = cicp.transfer.curve()?;
        Ok(Self {
            cicp: *cicp,
            primaries: cicp.primaries.primaries()?,
            curve,
            matrix: cicp.matrix.matrix(cicp.primaries, curve)?,
            range: quant::Range::from_flag(cicp.full_range),
        })
    }
}
