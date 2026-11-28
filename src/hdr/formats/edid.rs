//! What a display says about its colour in its EDID: the VESA E-EDID 1.4
//! base block (primaries, white, bit depth, name) and the CTA-861 extension's
//! HDR Static Metadata, Colorimetry and vendor-specific video data blocks
//! (HDR10+, Dolby Vision).

use alloc::string::String;

use crate::primaries::{Chromaticities, xy};

/// The base block's header.
const HEADER: [u8; 8] = [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00];
/// An EDID block's size.
pub const BLOCK: usize = 128;
/// The CTA-861 extension's tag.
const CTA_TAG: u8 = 0x02;
/// The data block tag that carries an extended tag.
const EXTENDED: u8 = 7;
/// Extended tags.
const VENDOR_VIDEO: u8 = 1;
const COLORIMETRY: u8 = 5;
const HDR_STATIC: u8 = 6;
/// IEEE OUIs of vendor-specific video data blocks.
const HDR10_PLUS_OUI: u32 = 0x90_848B;
const DOLBY_OUI: u32 = 0x00_D046;

/// A display's EDID, the parts about colour.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Edid {
    /// The PNP manufacturer ID, three letters.
    pub manufacturer: [u8; 3],
    pub product: u16,
    /// The product name descriptor (`0xFC`).
    pub name: Option<String>,
    /// Bits per primary colour of a digital input (6 to 16), if given.
    pub bits_per_color: Option<u8>,
    /// The display's primaries and white, 10-bit precision.
    pub chromaticities: Chromaticities,
    /// CTA-861's HDR Static Metadata data block.
    pub hdr: Option<HdrStatic>,
    pub colorimetry: Colorimetry,
    /// An HDR10+ vendor-specific video data block.
    pub hdr10_plus: bool,
    /// A Dolby Vision vendor-specific video data block.
    pub dolby_vision: bool,
}

/// CTA-861.3's HDR Static Metadata data block.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HdrStatic {
    pub eotfs: Eotfs,
    /// Desired content max luminance, cd/m².
    pub max_luminance: Option<f64>,
    /// Desired content max frame-average luminance, cd/m².
    pub max_frame_average: Option<f64>,
    /// Desired content min luminance, cd/m² (needs the max).
    pub min_luminance: Option<f64>,
}

/// The EOTFs a display takes (the block's `ET_n` bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Eotfs(pub u8);

impl Eotfs {
    /// Traditional gamma, SDR luminance range.
    #[must_use]
    pub const fn sdr(self) -> bool {
        self.0 & 1 != 0
    }

    /// Traditional gamma, HDR luminance range.
    #[must_use]
    pub const fn hdr_gamma(self) -> bool {
        self.0 & 2 != 0
    }

    /// SMPTE ST 2084 (PQ).
    #[must_use]
    pub const fn pq(self) -> bool {
        self.0 & 4 != 0
    }

    /// BT.2100 HLG.
    #[must_use]
    pub const fn hlg(self) -> bool {
        self.0 & 8 != 0
    }
}

/// CTA-861's Colorimetry data block, the wide-gamut parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Colorimetry {
    pub bt2020_rgb: bool,
    pub bt2020_ycc: bool,
    /// DCI-P3 (CTA-861-G).
    pub dci_p3: bool,
}

/// CTA-861.3: a luminance code value to cd/m², `50 * 2^(cv / 32)`.
#[must_use]
pub fn max_luminance(cv: u8) -> f64 {
    50.0 * crate::math::pow(2.0, f64::from(cv) / 32.0)
}

/// CTA-861.3: the min luminance code value against `max` cd/m²,
/// `max * (cv / 255)^2 / 100`.
#[must_use]
pub fn min_luminance(max: f64, cv: u8) -> f64 {
    let c = f64::from(cv) / 255.0;
    max * c * c / 100.0
}

impl Edid {
    /// Reads `bytes`: the base block and as many extension blocks as follow.
    /// `None` when the base block's header or checksum is wrong; an
    /// extension with a wrong checksum is passed over.
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let base = bytes.get(..BLOCK)?;
        if base[..8] != HEADER || !checksum_ok(base) {
            return None;
        }
        let id = u16::from_be_bytes([base[8], base[9]]);
        let letter = |shift: u16| {
            let code = ((id >> shift) & 0x1F) as u8;
            if (1..=26).contains(&code) { b'A' + code - 1 } else { b'?' }
        };
        let mut edid = Self {
            manufacturer: [letter(10), letter(5), letter(0)],
            product: u16::from_le_bytes([base[10], base[11]]),
            bits_per_color: bits_per_color(base[20]),
            chromaticities: chromaticities(base),
            ..Self::default()
        };
        for d in base[54..126].as_chunks::<18>().0 {
            if d[..3] == [0, 0, 0] && d[3] == 0xFC {
                let text: String = d[5..18]
                    .iter()
                    .take_while(|&&b| b != 0x0A)
                    .map(|&b| char::from(b))
                    .collect();
                edid.name = Some(String::from(text.trim_end()));
            }
        }
        for block in bytes[BLOCK..].as_chunks::<BLOCK>().0 {
            if block[0] == CTA_TAG && checksum_ok(block) {
                edid.read_cta(block);
            }
        }
        Some(edid)
    }

    /// The data block collection of one CTA-861 extension block.
    fn read_cta(&mut self, block: &[u8]) {
        let end = usize::from(block[2]).min(BLOCK - 1);
        let mut at = 4;
        while at < end {
            let header = block[at];
            let len = usize::from(header & 0x1F);
            let Some(payload) = block.get(at + 1..at + 1 + len) else {
                break;
            };
            if header >> 5 == EXTENDED
                && let Some((&tag, data)) = payload.split_first()
            {
                self.read_extended(tag, data);
            }
            at += 1 + len;
        }
    }

    fn read_extended(&mut self, tag: u8, data: &[u8]) {
        match tag {
            VENDOR_VIDEO if data.len() >= 3 => {
                let oui = u32::from(data[0]) | u32::from(data[1]) << 8 | u32::from(data[2]) << 16;
                self.hdr10_plus |= oui == HDR10_PLUS_OUI;
                self.dolby_vision |= oui == DOLBY_OUI;
            }
            COLORIMETRY if !data.is_empty() => {
                self.colorimetry = Colorimetry {
                    bt2020_ycc: data[0] & 0x40 != 0,
                    bt2020_rgb: data[0] & 0x80 != 0,
                    dci_p3: data.get(1).is_some_and(|b| b & 0x80 != 0),
                };
            }
            HDR_STATIC if data.len() >= 2 => {
                let max = data.get(2).filter(|&&cv| cv != 0).map(|&cv| max_luminance(cv));
                self.hdr = Some(HdrStatic {
                    eotfs: Eotfs(data[0] & 0x3F),
                    max_luminance: max,
                    max_frame_average: data
                        .get(3)
                        .filter(|&&cv| cv != 0)
                        .map(|&cv| max_luminance(cv)),
                    min_luminance: max.zip(data.get(4)).map(|(m, &cv)| min_luminance(m, cv)),
                });
            }
            _ => {}
        }
    }
}

fn checksum_ok(block: &[u8]) -> bool {
    block.iter().fold(0u8, |sum, &b| sum.wrapping_add(b)) == 0
}

/// Byte 20, the video input definition: bits per colour of a digital input.
fn bits_per_color(input: u8) -> Option<u8> {
    if input & 0x80 == 0 {
        return None;
    }
    match (input >> 4) & 7 {
        depth @ 1..=6 => Some(4 + 2 * depth),
        _ => None,
    }
}

/// Bytes 25 to 34: each coordinate's high 8 bits, its low 2 in byte 25 or 26.
fn chromaticities(b: &[u8]) -> Chromaticities {
    let coord = |high: usize, low: u8, shift: u8| {
        f64::from(u16::from(b[high]) << 2 | u16::from((low >> shift) & 3)) / 1024.0
    };
    let (rg, bw) = (b[25], b[26]);
    Chromaticities {
        red: xy(coord(27, rg, 6), coord(28, rg, 4)),
        green: xy(coord(29, rg, 2), coord(30, rg, 0)),
        blue: xy(coord(31, bw, 6), coord(32, bw, 4)),
        white: xy(coord(33, bw, 2), coord(34, bw, 0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sets a block's checksum byte.
    fn seal(block: &mut [u8]) {
        let sum = block[..BLOCK - 1].iter().fold(0u8, |s, &b| s.wrapping_add(b));
        block[BLOCK - 1] = 0u8.wrapping_sub(sum);
    }

    /// A 10-bit HDR monitor: "DEL", BT.2020-ish primaries, a CTA block with
    /// HDR static metadata, colorimetry and HDR10+.
    fn sample() -> alloc::vec::Vec<u8> {
        let mut edid = alloc::vec![0u8; 2 * BLOCK];
        edid[..8].copy_from_slice(&HEADER);
        // D=4, E=5, L=12: 00100 00101 01100.
        edid[8..10].copy_from_slice(&(0b00100_u16 << 10 | 0b00101 << 5 | 0b01100).to_be_bytes());
        edid[10..12].copy_from_slice(&0xA0B1u16.to_le_bytes());
        edid[18] = 1;
        edid[19] = 4;
        // Digital, 10 bits per colour, DisplayPort.
        edid[20] = 0x80 | 3 << 4 | 5;
        // Red (0.680, 0.320), green (0.265, 0.690), blue (0.150, 0.060),
        // white D65 (0.3127, 0.3290), in 1/1024.
        let coords: [u16; 8] = [6800, 3200, 2650, 6900, 1500, 600, 3127, 3290]
            .map(|v: u32| u16::try_from((v * 1024 + 5000) / 10_000).unwrap());
        let low = |i: usize| u8::try_from(coords[i] & 3).unwrap();
        edid[25] = low(0) << 6 | low(1) << 4 | low(2) << 2 | low(3);
        edid[26] = low(4) << 6 | low(5) << 4 | low(6) << 2 | low(7);
        for (i, c) in coords.iter().enumerate() {
            edid[27 + i] = u8::try_from(c >> 2).unwrap();
        }
        let name = b"\0\0\0\xFC\0HDR Monitor\n ";
        edid[54..72].copy_from_slice(name);
        edid[126] = 1;
        let cta = &mut edid[BLOCK..];
        cta[0] = CTA_TAG;
        cta[1] = 3;
        let blocks: &[&[u8]] = &[
            // HDR static metadata: SDR + PQ + HLG, type 1, max 1000-ish,
            // frame average 400-ish, min.
            &[EXTENDED << 5 | 6, HDR_STATIC, 0x0D, 0x01, 138, 96, 10],
            // Colorimetry: BT.2020 RGB and YCC, DCI-P3.
            &[EXTENDED << 5 | 3, COLORIMETRY, 0xC0, 0x80],
            // HDR10+: OUI 90-84-8B, application version 1.
            &[EXTENDED << 5 | 5, VENDOR_VIDEO, 0x8B, 0x84, 0x90, 0x01],
        ];
        let mut at = 4;
        for b in blocks {
            cta[at..at + b.len()].copy_from_slice(b);
            at += b.len();
        }
        cta[2] = u8::try_from(at).unwrap();
        seal(&mut edid[..BLOCK]);
        seal(&mut edid[BLOCK..]);
        edid
    }

    #[test]
    fn a_monitors_colour_is_read() {
        let edid = Edid::parse(&sample()).unwrap();
        assert_eq!(&edid.manufacturer, b"DEL");
        assert_eq!(edid.product, 0xA0B1);
        assert_eq!(edid.name.as_deref(), Some("HDR Monitor"));
        assert_eq!(edid.bits_per_color, Some(10));
        let c = edid.chromaticities;
        assert!((c.red.x - 0.680).abs() < 1.0 / 1024.0 && (c.white.y - 0.3290).abs() < 1.0 / 1024.0);
        let hdr = edid.hdr.unwrap();
        assert!(hdr.eotfs.sdr() && hdr.eotfs.pq() && hdr.eotfs.hlg() && !hdr.eotfs.hdr_gamma());
        let max = hdr.max_luminance.unwrap();
        assert!((max - max_luminance(138)).abs() < 1e-9 && (990.0..1010.0).contains(&max), "{max}");
        assert!((hdr.max_frame_average.unwrap() - max_luminance(96)).abs() < 1e-9);
        assert!((hdr.min_luminance.unwrap() - min_luminance(max, 10)).abs() < 1e-12);
        assert_eq!(
            edid.colorimetry,
            Colorimetry { bt2020_rgb: true, bt2020_ycc: true, dci_p3: true }
        );
        assert!(edid.hdr10_plus && !edid.dolby_vision);
    }

    #[test]
    fn cta_861_3_luminance_codes() {
        assert_eq!(max_luminance(0), 50.0);
        assert!((max_luminance(32) - 100.0).abs() < 1e-12);
        assert!((min_luminance(1000.0, 255) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn a_broken_block_is_refused_or_passed_over() {
        let mut edid = sample();
        edid[BLOCK + 10] ^= 0xFF;
        let read = Edid::parse(&edid).unwrap();
        assert!(read.hdr.is_none(), "the bad extension is skipped");
        edid[30] ^= 1;
        assert!(Edid::parse(&edid).is_none());
        assert!(Edid::parse(&edid[..64]).is_none());
    }
}
