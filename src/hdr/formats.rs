//! The HDR formats and their metadata: what each one is, what a decoder
//! without its metadata shows, and how much of it keinbonus covers.
//!
//! | format | signal | metadata | here |
//! |---|---|---|---|
//! | HDR10 | PQ, BT.2020, 10 bit | static: ST 2086 mastering display, CTA-861.3 MaxCLL/MaxFALL | all: [`Hdr10`], the HEVC and AV1 codings, BT.2390 tone mapping from it |
//! | HLG | HLG, BT.2020 | none needed (the display's peak sets the OOTF) | all: [`hdr`](super)'s OOTF |
//! | HDR10+ | as HDR10 | dynamic, per scene or frame: SMPTE ST 2094-40 (ITU-T T.35 payload) | the data model and its tone curve: [`Hdr10Plus`]; parsing the T.35 payload is still to do |
//! | Dolby Vision | per profile: an HDR10, SDR or HLG base layer, or the proprietary IPT-PQ-c2 | the RPU (per frame, proprietary), the configuration record | the profiles, their base layers and the configuration record: [`DolbyVision`]; RPU and reshaping are Dolby's (licensed), not here |
//!
//! What a player does: HDR10 tone-maps with the static metadata, HDR10+
//! with each scene's curve (and falls back to HDR10, which is the same
//! signal), Dolby Vision applies the RPU where licensed and otherwise
//! shows the base layer ([`DolbyVision::base_layer`]).

use crate::cicp::Cicp;
#[cfg(feature = "display")]
pub mod edid;
use crate::hdr::ToneMap;
use crate::math::{self, round};
use crate::primaries::Chromaticities;

/// Which HDR format a stream is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HdrFormat {
    Hdr10,
    Hdr10Plus,
    Hlg,
    DolbyVision(DolbyVisionProfile),
}

impl HdrFormat {
    /// The colour description of the signal a decoder without the
    /// format's metadata shows; `None` for Dolby Vision profile 5, whose
    /// IPT-PQ-c2 signal only a Dolby Vision decoder can show.
    #[must_use]
    pub const fn signal(self) -> Option<Cicp> {
        match self {
            Self::Hdr10 | Self::Hdr10Plus => Some(Cicp::HDR10),
            Self::Hlg => Some(Cicp::HLG),
            Self::DolbyVision(p) => p.base_layer(),
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hdr10 => "HDR10",
            Self::Hdr10Plus => "HDR10+",
            Self::Hlg => "HLG",
            Self::DolbyVision(_) => "Dolby Vision",
        }
    }
}

/// HDR10's static metadata.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Hdr10 {
    pub mastering: MasteringDisplay,
    pub content: ContentLight,
}

impl Hdr10 {
    /// BT.2390 tone mapping from this content: black at the mastering
    /// display's, peak at MaxCLL when the stream gives it (the brightest
    /// pixel there is), else at the mastering display's peak.
    #[must_use]
    pub fn tone_map(&self) -> ToneMap {
        let peak = if self.content.max_cll > 0 {
            f64::from(self.content.max_cll)
        } else {
            self.mastering.max_luminance
        };
        ToneMap::Bt2390 {
            source_black: self.mastering.min_luminance,
            source_peak: if peak > 0.0 { peak } else { 1000.0 },
        }
    }
}

/// SMPTE ST 2086: the display the content was mastered on.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MasteringDisplay {
    pub primaries: Chromaticities,
    /// cd/m².
    pub max_luminance: f64,
    /// cd/m².
    pub min_luminance: f64,
}

/// CTA-861.3 content light levels, cd/m² (0: unknown).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct ContentLight {
    /// MaxCLL: the brightest pixel.
    pub max_cll: u16,
    /// MaxFALL: the brightest frame average.
    pub max_fall: u16,
}

/// HEVC/VVC `mastering_display_colour_volume` SEI (and H.264's): primaries
/// in G, B, R order and white in 0.000 02 steps (0..=50 000), luminance in
/// 0.0001 cd/m² steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct HevcMastering {
    pub display_primaries_x: [u16; 3],
    pub display_primaries_y: [u16; 3],
    pub white_point_x: u16,
    pub white_point_y: u16,
    pub max_display_mastering_luminance: u32,
    pub min_display_mastering_luminance: u32,
}

/// AV1 `metadata_hdr_mdcv`: primaries in R, G, B order and white as 0.16
/// fixed point, `luminance_max` 24.8, `luminance_min` 18.14.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Av1Mastering {
    pub primary_chromaticity_x: [u16; 3],
    pub primary_chromaticity_y: [u16; 3],
    pub white_point_chromaticity_x: u16,
    pub white_point_chromaticity_y: u16,
    pub luminance_max: u32,
    pub luminance_min: u32,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clipped to the field"
)]
fn fixed16(v: f64, scale: f64, max: f64) -> u16 {
    math::clip3(0.0, max, round(v * scale)) as u16
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clipped to the field"
)]
fn fixed32(v: f64, scale: f64) -> u32 {
    math::clip3(0.0, f64::from(u32::MAX), round(v * scale)) as u32
}

impl MasteringDisplay {
    #[must_use]
    pub fn hevc(&self) -> HevcMastering {
        let c = |v: f64| fixed16(v, 50_000.0, 50_000.0);
        let p = &self.primaries;
        let gbr = [p.green, p.blue, p.red];
        HevcMastering {
            display_primaries_x: gbr.map(|xy| c(xy.x)),
            display_primaries_y: gbr.map(|xy| c(xy.y)),
            white_point_x: c(p.white.x),
            white_point_y: c(p.white.y),
            max_display_mastering_luminance: fixed32(self.max_luminance, 10_000.0),
            min_display_mastering_luminance: fixed32(self.min_luminance, 10_000.0),
        }
    }

    #[must_use]
    pub fn from_hevc(h: &HevcMastering) -> Self {
        let c = |v: u16| f64::from(v) / 50_000.0;
        let xy = |i: usize| {
            crate::primaries::xy(c(h.display_primaries_x[i]), c(h.display_primaries_y[i]))
        };
        Self {
            primaries: Chromaticities {
                red: xy(2),
                green: xy(0),
                blue: xy(1),
                white: crate::primaries::xy(c(h.white_point_x), c(h.white_point_y)),
            },
            max_luminance: f64::from(h.max_display_mastering_luminance) / 10_000.0,
            min_luminance: f64::from(h.min_display_mastering_luminance) / 10_000.0,
        }
    }

    #[must_use]
    pub fn av1(&self) -> Av1Mastering {
        let c = |v: f64| fixed16(v, 65_536.0, 65_535.0);
        let p = &self.primaries;
        let rgb = [p.red, p.green, p.blue];
        Av1Mastering {
            primary_chromaticity_x: rgb.map(|xy| c(xy.x)),
            primary_chromaticity_y: rgb.map(|xy| c(xy.y)),
            white_point_chromaticity_x: c(p.white.x),
            white_point_chromaticity_y: c(p.white.y),
            luminance_max: fixed32(self.max_luminance, 256.0),
            luminance_min: fixed32(self.min_luminance, 16_384.0),
        }
    }

    #[must_use]
    pub fn from_av1(a: &Av1Mastering) -> Self {
        let c = |v: u16| f64::from(v) / 65_536.0;
        let xy = |i: usize| {
            crate::primaries::xy(
                c(a.primary_chromaticity_x[i]),
                c(a.primary_chromaticity_y[i]),
            )
        };
        Self {
            primaries: Chromaticities {
                red: xy(0),
                green: xy(1),
                blue: xy(2),
                white: crate::primaries::xy(
                    c(a.white_point_chromaticity_x),
                    c(a.white_point_chromaticity_y),
                ),
            },
            max_luminance: f64::from(a.luminance_max) / 256.0,
            min_luminance: f64::from(a.luminance_min) / 16_384.0,
        }
    }
}

/// The ITU-T T.35 header of an HDR10+ payload (HEVC `user_data_registered
/// _itu_t_t35` SEI, AV1 metadata OBU of type ITU-T T.35).
pub mod t35 {
    /// `itu_t_t35_country_code`: United States.
    pub const COUNTRY_CODE: u8 = 0xB5;
    /// `itu_t_t35_terminal_provider_code`: Samsung.
    pub const PROVIDER_CODE: u16 = 0x003C;
    /// `itu_t_t35_terminal_provider_oriented_code`.
    pub const PROVIDER_ORIENTED_CODE: u16 = 0x0001;
    /// `application_identifier` of ST 2094-40.
    pub const APPLICATION_IDENTIFIER: u8 = 4;
}

/// HDR10+ dynamic metadata (SMPTE ST 2094-40), one processing window (the
/// whole picture), in the units the payload codes them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Hdr10Plus {
    pub application_version: u8,
    /// The display the curve below is made for, cd/m² (27 bits).
    pub targeted_system_display_maximum_luminance: u32,
    /// Per scene: the maximum of each of R, G, B, in 1/100 000 of
    /// 10 000 cd/m² (that is, 0.1 cd/m²; 17 bits).
    pub maxscl: [u32; 3],
    /// The average of `max(R, G, B)` over the picture, same units.
    pub average_maxrgb: u32,
    /// `(percentage, percentile)`: `percentage` % of the pixels have
    /// `max(R, G, B)` at or below `percentile` (same units).
    pub distribution_maxrgb: alloc::vec::Vec<(u8, u32)>,
    /// The share of pixels brighter than the percentiles, in 1/1000.
    pub fraction_bright_pixels: u16,
    /// The tone curve's knee, in 1/4095 of the normalized input and output.
    pub knee_point: (u16, u16),
    /// The Bezier anchors above the knee, in 1/1023 (up to 15).
    pub bezier_curve_anchors: alloc::vec::Vec<u16>,
}

impl Hdr10Plus {
    /// The scene's peak in cd/m²: the largest of `maxscl`.
    #[must_use]
    pub fn scene_peak(&self) -> f64 {
        f64::from(self.maxscl.iter().copied().max().unwrap_or(0)) / 10.0
    }

    /// ST 2094-40's basic tone mapping curve, `x` and the result
    /// normalized to 0..1 (input to the scene, output to the targeted
    /// display): linear up to the knee, then a Bezier curve of degree
    /// `N = anchors + 1` through `(knee, 1)` with control points
    /// `0, anchors..., 1` scaled into the rest of the range.
    #[must_use]
    pub fn tone_curve(&self, x: f64) -> f64 {
        let x = math::clip3(0.0, 1.0, x);
        let (kx, ky) = (
            f64::from(self.knee_point.0) / 4095.0,
            f64::from(self.knee_point.1) / 4095.0,
        );
        if x <= kx {
            return if kx > 0.0 { ky * x / kx } else { 0.0 };
        }
        let t = (x - kx) / (1.0 - kx);
        // Control points 0, P1 .. P(N-1), 1.
        let n = self.bezier_curve_anchors.len() + 1;
        let point = |k: usize| match k {
            0 => 0.0,
            k if k == n => 1.0,
            k => f64::from(self.bezier_curve_anchors[k - 1]) / 1023.0,
        };
        // De Casteljau: stable for any degree.
        let mut p: alloc::vec::Vec<f64> = (0..=n).map(point).collect();
        for r in 1..=n {
            for k in 0..=n - r {
                p[k] = (1.0 - t) * p[k] + t * p[k + 1];
            }
        }
        ky + (1.0 - ky) * p[0]
    }
}

/// Dolby Vision profiles (bitstream profiles of the Dolby Vision
/// specification).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DolbyVisionProfile {
    /// Single layer, IPT-PQ-c2, 10-bit HEVC: no base layer any other
    /// decoder can show (streaming).
    P5,
    /// Dual layer, HDR10 base layer plus enhancement layer (UHD Blu-ray).
    P7,
    /// Single layer with an RPU over a compatible base layer: 8.1 HDR10,
    /// 8.2 SDR, 8.4 HLG (by compatibility id).
    P8(BaseLayer),
    /// AVC, SDR base layer.
    P9,
    /// AV1, like profile 8 (10.0 without a compatible base layer).
    P10(Option<BaseLayer>),
    /// MV-HEVC (stereo).
    P20,
}

/// What a Dolby Vision base layer is to other decoders
/// (`dv_bl_signal_compatibility_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseLayer {
    /// 1: HDR10.
    Hdr10,
    /// 2: SDR, BT.709.
    Sdr,
    /// 4: HLG.
    Hlg,
    /// 6: HDR10 as UHD Blu-ray carries it.
    BluRayHdr10,
}

impl BaseLayer {
    #[must_use]
    pub const fn from_id(id: u8) -> Option<Self> {
        match id {
            1 => Some(Self::Hdr10),
            2 => Some(Self::Sdr),
            4 => Some(Self::Hlg),
            6 => Some(Self::BluRayHdr10),
            _ => None,
        }
    }

    #[must_use]
    pub const fn id(self) -> u8 {
        match self {
            Self::Hdr10 => 1,
            Self::Sdr => 2,
            Self::Hlg => 4,
            Self::BluRayHdr10 => 6,
        }
    }

    #[must_use]
    pub const fn cicp(self) -> Cicp {
        match self {
            Self::Hdr10 | Self::BluRayHdr10 => Cicp::HDR10,
            Self::Sdr => Cicp::BT709,
            Self::Hlg => Cicp::HLG,
        }
    }
}

impl DolbyVisionProfile {
    /// The profile and compatibility id of a configuration record; `None`
    /// for unknown or deprecated profiles (0 to 4, 6).
    #[must_use]
    pub const fn new(profile: u8, compatibility: u8) -> Option<Self> {
        Some(match profile {
            5 => Self::P5,
            7 => Self::P7,
            8 => match BaseLayer::from_id(compatibility) {
                Some(b) => Self::P8(b),
                None => return None,
            },
            9 => Self::P9,
            10 => Self::P10(BaseLayer::from_id(compatibility)),
            20 => Self::P20,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn number(self) -> u8 {
        match self {
            Self::P5 => 5,
            Self::P7 => 7,
            Self::P8(_) => 8,
            Self::P9 => 9,
            Self::P10(_) => 10,
            Self::P20 => 20,
        }
    }

    /// The base layer's colour description; `None` where there is no
    /// compatible base layer (profile 5, 10.0).
    #[must_use]
    pub const fn base_layer(self) -> Option<Cicp> {
        match self {
            Self::P5 | Self::P10(None) => None,
            Self::P7 | Self::P20 => Some(Cicp::HDR10),
            Self::P8(b) | Self::P10(Some(b)) => Some(b.cicp()),
            Self::P9 => Some(Cicp::BT709),
        }
    }
}

/// A Dolby Vision decoder configuration record (`DOVIDecoderConfigurationRecord`,
/// the payload of the ISO BMFF boxes `dvcC`, `dvvC`, `dvwC`, and of the
/// Matroska `BlockAdditionMapping`): 24 bytes, big-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DolbyVision {
    pub version: (u8, u8),
    pub profile: u8,
    pub level: u8,
    pub rpu: bool,
    pub enhancement_layer: bool,
    pub base_layer: bool,
    pub compatibility: u8,
}

impl DolbyVision {
    pub const RECORD_BYTES: usize = 24;

    /// From the record's bytes.
    #[must_use]
    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.len() < Self::RECORD_BYTES {
            return None;
        }
        Some(Self {
            version: (b[0], b[1]),
            profile: b[2] >> 1,
            level: ((b[2] & 1) << 5) | (b[3] >> 3),
            rpu: b[3] & 0x4 != 0,
            enhancement_layer: b[3] & 0x2 != 0,
            base_layer: b[3] & 0x1 != 0,
            compatibility: b[4] >> 4,
        })
    }

    /// The record's bytes (reserved bits 0).
    #[must_use]
    pub fn write(&self) -> [u8; Self::RECORD_BYTES] {
        let mut b = [0u8; Self::RECORD_BYTES];
        b[0] = self.version.0;
        b[1] = self.version.1;
        b[2] = (self.profile << 1) | ((self.level >> 5) & 1);
        b[3] = ((self.level & 0x1f) << 3)
            | (u8::from(self.rpu) << 2)
            | (u8::from(self.enhancement_layer) << 1)
            | u8::from(self.base_layer);
        b[4] = self.compatibility << 4;
        b
    }

    /// The ISO BMFF box the record goes in: `dvcC` up to profile 7, `dvvC`
    /// for 8 to 10, `dvwC` above.
    #[must_use]
    pub const fn box_type(&self) -> [u8; 4] {
        match self.profile {
            0..=7 => *b"dvcC",
            8..=10 => *b"dvvC",
            _ => *b"dvwC",
        }
    }

    #[must_use]
    pub const fn profile(&self) -> Option<DolbyVisionProfile> {
        DolbyVisionProfile::new(self.profile, self.compatibility)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cicp::ColourPrimaries;
    use crate::math::close;
    use alloc::vec;

    #[test]
    fn mastering_metadata_codes() {
        let p3 = ColourPrimaries::Smpte432.chromaticities().unwrap();
        let md = MasteringDisplay {
            primaries: p3,
            max_luminance: 1000.0,
            min_luminance: 0.0001,
        };
        let h = md.hevc();
        // G, B, R order; 0.000 02 steps.
        assert_eq!(h.display_primaries_x, [13250, 7500, 34000]);
        assert_eq!(h.display_primaries_y, [34500, 3000, 16000]);
        assert_eq!((h.white_point_x, h.white_point_y), (15635, 16450));
        assert_eq!(h.max_display_mastering_luminance, 10_000_000);
        assert_eq!(h.min_display_mastering_luminance, 1);
        let back = MasteringDisplay::from_hevc(&h);
        assert!(
            close(back.primaries.red.x, 0.68, 1e-12) && close(back.max_luminance, 1000.0, 1e-12)
        );
        let a = md.av1();
        assert_eq!(a.primary_chromaticity_x[0], 44564, "0.68 * 65536 rounded");
        assert_eq!(a.luminance_max, 256_000);
        assert_eq!(a.luminance_min, 2, "0.0001 * 16384 = 1.6");
        let back = MasteringDisplay::from_av1(&a);
        assert!(close(back.primaries.green.y, 0.69, 1e-4));
        assert!(close(back.max_luminance, 1000.0, 1e-12));
    }

    #[test]
    fn formats_show_their_signal() {
        assert_eq!(HdrFormat::Hdr10Plus.signal(), Some(Cicp::HDR10));
        assert_eq!(
            HdrFormat::DolbyVision(DolbyVisionProfile::P5).signal(),
            None
        );
        assert_eq!(
            HdrFormat::DolbyVision(DolbyVisionProfile::P8(BaseLayer::Hlg)).signal(),
            Some(Cicp::HLG)
        );
        assert_eq!(DolbyVisionProfile::P10(None).base_layer(), None);
        assert_eq!(
            DolbyVisionProfile::new(8, 2),
            Some(DolbyVisionProfile::P8(BaseLayer::Sdr))
        );
        assert_eq!(
            DolbyVisionProfile::new(8, 0),
            None,
            "8 needs a compatible base layer"
        );
        assert_eq!(DolbyVisionProfile::new(4, 2), None, "deprecated");
    }

    #[test]
    fn hdr10_tone_maps_from_max_cll_or_the_mastering_peak() {
        let mut m = Hdr10 {
            mastering: MasteringDisplay {
                max_luminance: 4000.0,
                min_luminance: 0.005,
                ..MasteringDisplay::default()
            },
            content: ContentLight::default(),
        };
        assert_eq!(
            m.tone_map(),
            ToneMap::Bt2390 {
                source_black: 0.005,
                source_peak: 4000.0
            }
        );
        m.content.max_cll = 1200;
        assert_eq!(
            m.tone_map(),
            ToneMap::Bt2390 {
                source_black: 0.005,
                source_peak: 1200.0
            }
        );
    }

    #[test]
    fn dolby_vision_records_round_trip() {
        // Profile 8.1, level 6, RPU and base layer, no enhancement layer.
        let dv = DolbyVision {
            version: (1, 0),
            profile: 8,
            level: 6,
            rpu: true,
            enhancement_layer: false,
            base_layer: true,
            compatibility: 1,
        };
        let b = dv.write();
        assert_eq!(&b[..5], &[1, 0, 0x10, 0x35, 0x10]);
        assert!(b[5..].iter().all(|&x| x == 0));
        assert_eq!(DolbyVision::parse(&b), Some(dv));
        assert_eq!(dv.box_type(), *b"dvvC");
        assert_eq!(dv.profile(), Some(DolbyVisionProfile::P8(BaseLayer::Hdr10)));
        // Profile 7 dual layer, level 13 (crosses the byte boundary).
        let p7 = DolbyVision {
            profile: 7,
            level: 13,
            enhancement_layer: true,
            compatibility: 6,
            ..dv
        };
        assert_eq!(DolbyVision::parse(&p7.write()), Some(p7));
        assert_eq!(p7.box_type(), *b"dvcC");
        assert_eq!(DolbyVision::parse(&b[..23]), None);
    }

    #[test]
    fn hdr10_plus_curve_runs_from_0_through_the_knee_to_1() {
        let m = Hdr10Plus {
            knee_point: (1024, 1536),
            bezier_curve_anchors: vec![500, 750, 900],
            maxscl: [8000, 12000, 6000],
            ..Hdr10Plus::default()
        };
        let (kx, ky) = (1024.0 / 4095.0, 1536.0 / 4095.0);
        assert!(close(m.tone_curve(0.0), 0.0, 1e-15));
        assert!(close(m.tone_curve(kx), ky, 1e-12));
        assert!(close(m.tone_curve(kx + 1e-12), ky, 1e-9));
        assert!(close(m.tone_curve(1.0), 1.0, 1e-12));
        assert!(close(m.tone_curve(2.0), 1.0, 1e-12));
        let mut last = 0.0;
        for i in 0..=1000 {
            let y = m.tone_curve(f64::from(i) / 1000.0);
            assert!(y >= last - 1e-12, "monotone for rising anchors");
            last = y;
        }
        assert!(close(m.scene_peak(), 1200.0, 1e-12));
        // Without anchors the part above the knee is a straight line.
        let line = Hdr10Plus {
            knee_point: (0, 0),
            ..Hdr10Plus::default()
        };
        assert!(close(line.tone_curve(0.3), 0.3, 1e-15));
        let _ = t35::APPLICATION_IDENTIFIER;
    }
}
