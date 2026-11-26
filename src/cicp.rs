//! H.273 (ISO/IEC 23091-2) code points: the colour description every video
//! bitstream (H.264/HEVC/VVC VUI, the AV1 sequence header), container (MP4
//! `colr` `nclx`), image format (AVIF, PNG `cICP`, JPEG XL) and graphics API
//! maps onto. [`Cicp::resolve`] turns one into the functions of
//! [`crate::Colour`].

use core::fmt;

use crate::Error;

macro_rules! code_points {
    (
        $(#[$m:meta])* $name:ident {
            $($(#[$vm:meta])* $variant:ident = $raw:literal, $label:literal;)*
        }
    ) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum $name {
            $($(#[$vm])* $variant = $raw,)*
        }

        impl $name {
            pub const ALL: &[Self] = &[$(Self::$variant,)*];

            /// The code point, `None` where H.273 reserves it.
            #[must_use]
            pub const fn from_raw(raw: u8) -> Option<Self> {
                match raw {
                    $($raw => Some(Self::$variant),)*
                    _ => None,
                }
            }

            #[must_use]
            pub const fn raw(self) -> u8 {
                self as u8
            }

            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $label,)*
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.name())
            }
        }
    };
}

code_points! {
    /// `ColourPrimaries`: the chromaticities of R, G, B and white.
    ColourPrimaries {
        /// BT.709, sRGB, BT.1361, IEC 61966-2-4.
        Bt709 = 1, "BT.709";
        Unspecified = 2, "unspecified";
        /// BT.470 System M (NTSC 1953), white C.
        Bt470M = 4, "BT.470 M";
        /// BT.470 System B, G (PAL, SECAM), BT.601 625.
        Bt470Bg = 5, "BT.470 B/G";
        /// BT.601 525, SMPTE 170M (NTSC).
        Bt601 = 6, "BT.601 525";
        /// SMPTE 240M: the same as [`Bt601`](Self::Bt601).
        Smpte240 = 7, "SMPTE 240M";
        /// Generic film (colour filters with illuminant C).
        Film = 8, "film";
        /// BT.2020, BT.2100.
        Bt2020 = 9, "BT.2020";
        /// SMPTE ST 428-1: CIE 1931 XYZ itself.
        Smpte428 = 10, "SMPTE 428 (XYZ)";
        /// SMPTE RP 431-2: DCI-P3, the theatre white.
        Smpte431 = 11, "DCI-P3";
        /// SMPTE EG 432-1: Display P3 (P3 primaries, D65).
        Smpte432 = 12, "Display P3";
        /// EBU Tech. 3213-E.
        Ebu3213 = 22, "EBU 3213";
    }
}

code_points! {
    /// `TransferCharacteristics`: the opto-electronic transfer (or the
    /// inverse of the electro-optical one) the signal was made with.
    TransferCharacteristics {
        /// BT.709, the BT.601 and BT.2020 curve.
        Bt709 = 1, "BT.709";
        Unspecified = 2, "unspecified";
        /// BT.470 System M: display gamma 2.2.
        Gamma22 = 4, "gamma 2.2";
        /// BT.470 System B, G: display gamma 2.8.
        Gamma28 = 5, "gamma 2.8";
        /// BT.601: the BT.709 curve.
        Bt601 = 6, "BT.601";
        Smpte240 = 7, "SMPTE 240M";
        Linear = 8, "linear";
        /// Logarithmic, 100:1.
        Log100 = 9, "log 100:1";
        /// Logarithmic, 100 * sqrt(10):1.
        Log316 = 10, "log 316:1";
        /// IEC 61966-2-4 (xvYCC): the BT.709 curve, mirrored below 0.
        Iec61966_2_4 = 11, "xvYCC";
        /// BT.1361 extended colour gamut.
        Bt1361 = 12, "BT.1361";
        /// IEC 61966-2-1: sRGB, sYCC.
        Srgb = 13, "sRGB";
        /// BT.2020 10-bit: the BT.709 curve.
        Bt2020_10 = 14, "BT.2020 10-bit";
        /// BT.2020 12-bit: the BT.709 curve.
        Bt2020_12 = 15, "BT.2020 12-bit";
        /// SMPTE ST 2084, BT.2100 PQ.
        Pq = 16, "PQ";
        /// SMPTE ST 428-1.
        Smpte428 = 17, "SMPTE 428";
        /// ARIB STD-B67, BT.2100 HLG.
        Hlg = 18, "HLG";
    }
}

code_points! {
    /// `MatrixCoefficients`: how R'G'B' becomes luma and two chroma
    /// signals.
    MatrixCoefficients {
        /// GBR: no matrix (RGB coded as is, 4:4:4 usually).
        Identity = 0, "identity (GBR)";
        Bt709 = 1, "BT.709";
        Unspecified = 2, "unspecified";
        /// US FCC 73.682.
        Fcc = 4, "FCC";
        /// BT.470 System B, G; BT.601 625: the BT.601 matrix.
        Bt470Bg = 5, "BT.470 B/G";
        /// BT.601 525, SMPTE 170M.
        Bt601 = 6, "BT.601";
        Smpte240 = 7, "SMPTE 240M";
        YCgCo = 8, "YCgCo";
        /// BT.2020 non-constant luminance.
        Bt2020Ncl = 9, "BT.2020 NCL";
        /// BT.2020 constant luminance.
        Bt2020Cl = 10, "BT.2020 CL";
        /// SMPTE ST 2085 (Y'D'zD'x).
        Smpte2085 = 11, "SMPTE 2085 (YDzDx)";
        /// Non-constant luminance from the primaries' luminance.
        ChromaDerivedNcl = 12, "chroma-derived NCL";
        /// Constant luminance from the primaries' luminance.
        ChromaDerivedCl = 13, "chroma-derived CL";
        /// BT.2100 ICtCp.
        Ictcp = 14, "ICtCp";
        /// IPT-PQ-C2 (SMPTE ST 2128).
        IptC2 = 15, "IPT-C2";
        /// YCgCo-R, even bit-depth increase.
        YCgCoRe = 16, "YCgCo-Re";
        /// YCgCo-R, odd bit-depth increase.
        YCgCoRo = 17, "YCgCo-Ro";
    }
}

/// A colour description: H.273's four fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cicp {
    pub primaries: ColourPrimaries,
    pub transfer: TransferCharacteristics,
    pub matrix: MatrixCoefficients,
    /// `VideoFullRangeFlag`: codes span `0..2^n` instead of the
    /// head- and footroom of 16..235 (luma) and 16..240 (chroma) at 8 bits.
    pub full_range: bool,
}

const fn cicp(p: u8, t: u8, m: u8, full_range: bool) -> Cicp {
    match (
        ColourPrimaries::from_raw(p),
        TransferCharacteristics::from_raw(t),
        MatrixCoefficients::from_raw(m),
    ) {
        (Some(primaries), Some(transfer), Some(matrix)) => Cicp {
            primaries,
            transfer,
            matrix,
            full_range,
        },
        _ => panic!("not a code point"),
    }
}

impl Cicp {
    /// sRGB images: RGB, full range.
    pub const SRGB: Self = cicp(1, 13, 0, true);
    /// HD video.
    pub const BT709: Self = cicp(1, 1, 1, false);
    /// SD video, 525 lines (NTSC).
    pub const BT601_525: Self = cicp(6, 6, 6, false);
    /// SD video, 625 lines (PAL).
    pub const BT601_625: Self = cicp(5, 6, 5, false);
    /// UHD SDR video.
    pub const BT2020: Self = cicp(9, 14, 9, false);
    /// HDR10: BT.2020, PQ, limited range.
    pub const HDR10: Self = cicp(9, 16, 9, false);
    /// BT.2100 HLG, limited range.
    pub const HLG: Self = cicp(9, 18, 9, false);
    /// Display P3 images (RGB, full range), as Apple writes them.
    pub const DISPLAY_P3: Self = cicp(12, 13, 0, true);

    /// From the four raw fields; `None` for a reserved code point.
    #[must_use]
    pub const fn from_raw(
        primaries: u8,
        transfer: u8,
        matrix: u8,
        full_range: bool,
    ) -> Option<Self> {
        match (
            ColourPrimaries::from_raw(primaries),
            TransferCharacteristics::from_raw(transfer),
            MatrixCoefficients::from_raw(matrix),
        ) {
            (Some(primaries), Some(transfer), Some(matrix)) => Some(Self {
                primaries,
                transfer,
                matrix,
                full_range,
            }),
            _ => None,
        }
    }

    /// The functions this description stands for.
    ///
    /// # Errors
    ///
    /// [`Error::Unspecified`] for an unspecified field,
    /// [`Error::Unsupported`] for a matrix without a reference here yet,
    /// [`Error::Invalid`] for a combination H.273 doesn't define (ICtCp
    /// without PQ or HLG, or outside BT.2020).
    pub fn resolve(&self) -> Result<crate::Colour, Error> {
        crate::Colour::new(self)
    }
}

impl fmt::Display for Cicp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}/{}/{} ({}, {}, {}, {} range)",
            self.primaries.raw(),
            self.transfer.raw(),
            self.matrix.raw(),
            u8::from(self.full_range),
            self.primaries,
            self.transfer,
            self.matrix,
            if self.full_range { "full" } else { "limited" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_points_round_trip_and_reserved_ones_are_none() {
        for raw in 0..=255u8 {
            if let Some(p) = ColourPrimaries::from_raw(raw) {
                assert_eq!(p.raw(), raw);
            }
            if let Some(t) = TransferCharacteristics::from_raw(raw) {
                assert_eq!(t.raw(), raw);
            }
            if let Some(m) = MatrixCoefficients::from_raw(raw) {
                assert_eq!(m.raw(), raw);
            }
        }
        assert_eq!(ColourPrimaries::from_raw(0), None);
        assert_eq!(ColourPrimaries::from_raw(3), None);
        assert_eq!(ColourPrimaries::from_raw(13), None);
        assert_eq!(TransferCharacteristics::from_raw(19), None);
        assert_eq!(MatrixCoefficients::from_raw(3), None);
        assert_eq!(MatrixCoefficients::from_raw(18), None);
        assert_eq!(ColourPrimaries::ALL.len(), 12);
        assert_eq!(TransferCharacteristics::ALL.len(), 17);
        assert_eq!(MatrixCoefficients::ALL.len(), 17);
    }

    #[test]
    fn presets_are_the_usual_quadruples() {
        assert_eq!(
            Cicp::HDR10.to_string(),
            "9/16/9/0 (BT.2020, PQ, BT.2020 NCL, limited range)"
        );
        assert_eq!(Cicp::from_raw(1, 13, 0, true), Some(Cicp::SRGB));
        assert_eq!(Cicp::from_raw(1, 13, 3, true), None);
    }
}
