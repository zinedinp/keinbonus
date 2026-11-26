//! Chroma subsampling: formats, where the chroma samples sit, and the
//! reference filters between a full-resolution colour-difference plane and
//! a subsampled one.
//!
//! The standards fix where a chroma sample sits (H.273's
//! `ChromaSampleLocType`, AV1's `chroma_sample_position`), not how it is
//! filtered. The filters here are a fixed choice, the common one:
//!
//! - [`Downsample::Area`]: each chroma sample is the mean of the luma-grid
//!   area it stands for, centred on its position: `[1, 1] / 2` between two
//!   samples, `[1, 2, 1] / 4` on one. It keeps linear ramps exact.
//! - [`Upsample::Linear`]: linear interpolation between chroma sample
//!   positions, phase-correct for the location; also exact on ramps, so a
//!   ramp survives down- and upsampling unchanged away from the edges.
//! - [`Upsample::Nearest`]: the nearest chroma sample.
//!
//! Edges repeat the outermost sample. All filters are separable:
//! horizontal first, then vertical.

use alloc::vec::Vec;

use crate::Error;
use crate::math::{floor, round};

/// How the colour-difference planes are subsampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaFormat {
    /// Luma only (4:0:0).
    Monochrome,
    /// Half width, half height.
    Yuv420,
    /// Half width.
    Yuv422,
    /// Half height (4:4:0).
    Yuv440,
    /// Full resolution.
    Yuv444,
}

impl ChromaFormat {
    /// Horizontal and vertical subsampling factors.
    #[must_use]
    pub const fn factors(self) -> (usize, usize) {
        match self {
            Self::Monochrome | Self::Yuv444 => (1, 1),
            Self::Yuv420 => (2, 2),
            Self::Yuv422 => (2, 1),
            Self::Yuv440 => (1, 2),
        }
    }

    /// The size of a chroma plane of a `width` x `height` image (rounded
    /// up: odd sizes get a last sample for the last luma column or row).
    /// `(0, 0)` for monochrome.
    #[must_use]
    pub const fn plane(self, width: usize, height: usize) -> (usize, usize) {
        if matches!(self, Self::Monochrome) {
            return (0, 0);
        }
        let (sx, sy) = self.factors();
        (width.div_ceil(sx), height.div_ceil(sy))
    }

    /// `chroma_format_idc` of H.264, HEVC and VVC (4:4:0 has none).
    #[must_use]
    pub const fn idc(self) -> Option<u8> {
        match self {
            Self::Monochrome => Some(0),
            Self::Yuv420 => Some(1),
            Self::Yuv422 => Some(2),
            Self::Yuv444 => Some(3),
            Self::Yuv440 => None,
        }
    }
}

/// H.273 `ChromaSampleLocType` (H.264/HEVC `chroma_sample_loc_type`): where
/// a chroma sample sits among the luma samples it covers, for 4:2:0. For
/// 4:2:2 only the horizontal part counts, for 4:4:0 the vertical one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ChromaLocation {
    /// With the left luma column, between two rows (MPEG-2, H.264's
    /// default, AV1 "vertical").
    #[default]
    Left = 0,
    /// Between two columns and two rows (MPEG-1, JPEG, H.261).
    Center = 1,
    /// With the top-left luma sample (BT.2020 and BT.2100 4:2:0, AV1
    /// "colocated").
    TopLeft = 2,
    /// With the top row, between two columns.
    Top = 3,
    /// With the left column and the bottom row.
    BottomLeft = 4,
    /// With the bottom row, between two columns.
    Bottom = 5,
}

impl ChromaLocation {
    pub const ALL: [Self; 6] = [
        Self::Left,
        Self::Center,
        Self::TopLeft,
        Self::Top,
        Self::BottomLeft,
        Self::Bottom,
    ];

    #[must_use]
    pub const fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Left),
            1 => Some(Self::Center),
            2 => Some(Self::TopLeft),
            3 => Some(Self::Top),
            4 => Some(Self::BottomLeft),
            5 => Some(Self::Bottom),
            _ => None,
        }
    }

    /// AV1 `chroma_sample_position`: 1 vertical (left), 2 colocated
    /// (top-left); 0 (unknown) is `None`.
    #[must_use]
    pub const fn from_av1(position: u8) -> Option<Self> {
        match position {
            1 => Some(Self::Left),
            2 => Some(Self::TopLeft),
            _ => None,
        }
    }

    /// Where chroma sample `(0, 0)` sits, in luma samples from luma sample
    /// `(0, 0)`, for `format`. Sample `(i, j)` sits at `(sx i + x, sy j + y)`.
    #[must_use]
    pub const fn offset(self, format: ChromaFormat) -> (f64, f64) {
        let (sx, sy) = format.factors();
        let x = if sx == 2 {
            match self {
                Self::Left | Self::TopLeft | Self::BottomLeft => 0.0,
                Self::Center | Self::Top | Self::Bottom => 0.5,
            }
        } else {
            0.0
        };
        let y = if sy == 2 {
            match self {
                Self::TopLeft | Self::Top => 0.0,
                Self::Left | Self::Center => 0.5,
                Self::BottomLeft | Self::Bottom => 1.0,
            }
        } else {
            0.0
        };
        (x, y)
    }
}

/// Full resolution to subsampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Downsample {
    /// The mean over the area each chroma sample stands for.
    #[default]
    Area,
}

/// Subsampled to full resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Upsample {
    /// The nearest chroma sample (ties to the later one).
    Nearest,
    /// Linear interpolation between chroma positions.
    #[default]
    Linear,
}

/// A sample index or count as a real number (exact: images are far
/// smaller than 2^52 samples).
#[allow(clippy::cast_precision_loss, reason = "exact below 2^52")]
fn real(n: usize) -> f64 {
    n as f64
}

/// `floor(x)` as an index that may lie outside the plane.
#[allow(
    clippy::cast_possible_truncation,
    reason = "positions are small and finite"
)]
fn index(x: f64) -> i64 {
    floor(x) as i64
}

/// The `(index, weight)` taps of one output sample: [`Downsample::Area`]
/// over `n` inputs, factor `s`, output `i` centred at `s i + o`.
fn area_taps(i: usize, s: usize, o: f64, n: usize, taps: &mut Vec<(usize, f64)>) {
    taps.clear();
    let centre = real(s * i) + o;
    let half = real(s) / 2.0;
    let (lo, hi) = (centre - half, centre + half);
    // Input k covers k - 0.5 .. k + 0.5.
    for k in index(lo + 0.5)..=index(hi + 0.5) {
        #[allow(clippy::cast_precision_loss, reason = "small")]
        let kf = k as f64;
        let overlap = (hi.min(kf + 0.5) - lo.max(kf - 0.5)).max(0.0);
        if overlap > 0.0 {
            taps.push((clamp_index(k, n), overlap / real(s)));
        }
    }
}

/// The `(index, weight)` taps of full-resolution sample `x` from `n`
/// subsampled ones (factor `s`, offset `o`).
fn up_taps(x: usize, s: usize, o: f64, n: usize, filter: Upsample) -> [(usize, f64); 2] {
    let u = (real(x) - o) / real(s);
    match filter {
        Upsample::Nearest => [(clamp_index(index(u + 0.5), n), 1.0), (0, 0.0)],
        Upsample::Linear => {
            let i0 = floor(u);
            let t = u - i0;
            let i0 = index(i0);
            [(clamp_index(i0, n), 1.0 - t), (clamp_index(i0 + 1, n), t)]
        }
    }
}

/// `k` clamped into `0..n` (the edge repeated).
fn clamp_index(k: i64, n: usize) -> usize {
    let last = n.saturating_sub(1);
    usize::try_from(k).map_or(0, |k| k.min(last))
}

fn check(len: usize, width: usize, height: usize) -> Result<(), Error> {
    if width == 0 || height == 0 || len != width * height {
        Err(Error::Size)
    } else {
        Ok(())
    }
}

/// A full-resolution colour-difference plane (`width` x `height`, row by
/// row) subsampled for `format`.
///
/// # Errors
///
/// [`Error::Size`] when `plane` isn't `width * height` long or is empty.
pub fn downsample(
    plane: &[f64],
    width: usize,
    height: usize,
    format: ChromaFormat,
    location: ChromaLocation,
    _filter: Downsample,
) -> Result<Vec<f64>, Error> {
    check(plane.len(), width, height)?;
    let (sx, sy) = format.factors();
    let (cw, ch) = format.plane(width, height);
    let (ox, oy) = location.offset(format);
    let mut taps = Vec::new();
    // Horizontal.
    let mut rows = alloc::vec![0.0; cw * height];
    for i in 0..cw {
        area_taps(i, sx, ox, width, &mut taps);
        for y in 0..height {
            rows[y * cw + i] = taps.iter().map(|&(k, w)| w * plane[y * width + k]).sum();
        }
    }
    // Vertical.
    let mut out = alloc::vec![0.0; cw * ch];
    for j in 0..ch {
        area_taps(j, sy, oy, height, &mut taps);
        for i in 0..cw {
            out[j * cw + i] = taps.iter().map(|&(k, w)| w * rows[k * cw + i]).sum();
        }
    }
    Ok(out)
}

/// A subsampled plane of `format` (its size from
/// [`ChromaFormat::plane`] of `width` x `height`) to full resolution.
///
/// # Errors
///
/// [`Error::Size`] when `plane` doesn't have the subsampled size.
pub fn upsample(
    plane: &[f64],
    width: usize,
    height: usize,
    format: ChromaFormat,
    location: ChromaLocation,
    filter: Upsample,
) -> Result<Vec<f64>, Error> {
    let (sx, sy) = format.factors();
    let (cw, ch) = format.plane(width, height);
    check(plane.len(), cw, ch)?;
    let (ox, oy) = location.offset(format);
    let mut rows = alloc::vec![0.0; width * ch];
    for x in 0..width {
        let taps = up_taps(x, sx, ox, cw, filter);
        for j in 0..ch {
            rows[j * width + x] = taps.iter().map(|&(k, w)| w * plane[j * cw + k]).sum();
        }
    }
    let mut out = alloc::vec![0.0; width * height];
    for y in 0..height {
        let taps = up_taps(y, sy, oy, ch, filter);
        for x in 0..width {
            out[y * width + x] = taps.iter().map(|&(k, w)| w * rows[k * width + x]).sum();
        }
    }
    Ok(out)
}

/// [`downsample`] on integer codes, rounded (H.273 `Round`) and clipped to
/// `0..=max`.
///
/// # Errors
///
/// As [`downsample`].
pub fn downsample_codes(
    plane: &[u16],
    width: usize,
    height: usize,
    format: ChromaFormat,
    location: ChromaLocation,
    filter: Downsample,
    max: u16,
) -> Result<Vec<u16>, Error> {
    let f: Vec<f64> = plane.iter().map(|&c| f64::from(c)).collect();
    Ok(to_codes(
        &downsample(&f, width, height, format, location, filter)?,
        max,
    ))
}

/// [`upsample`] on integer codes, rounded and clipped.
///
/// # Errors
///
/// As [`upsample`].
pub fn upsample_codes(
    plane: &[u16],
    width: usize,
    height: usize,
    format: ChromaFormat,
    location: ChromaLocation,
    filter: Upsample,
    max: u16,
) -> Result<Vec<u16>, Error> {
    let f: Vec<f64> = plane.iter().map(|&c| f64::from(c)).collect();
    Ok(to_codes(
        &upsample(&f, width, height, format, location, filter)?,
        max,
    ))
}

fn to_codes(plane: &[f64], max: u16) -> Vec<u16> {
    plane
        .iter()
        .map(|&v| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clipped"
            )]
            let c = round(v).clamp(0.0, f64::from(max)) as u16;
            c
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::close;
    use alloc::vec;

    const FORMATS: [ChromaFormat; 4] = [
        ChromaFormat::Yuv420,
        ChromaFormat::Yuv422,
        ChromaFormat::Yuv440,
        ChromaFormat::Yuv444,
    ];

    #[test]
    fn plane_sizes_round_up() {
        assert_eq!(ChromaFormat::Yuv420.plane(1920, 1080), (960, 540));
        assert_eq!(ChromaFormat::Yuv420.plane(1921, 1081), (961, 541));
        assert_eq!(ChromaFormat::Yuv422.plane(5, 3), (3, 3));
        assert_eq!(ChromaFormat::Yuv440.plane(5, 3), (5, 2));
        assert_eq!(ChromaFormat::Monochrome.plane(5, 3), (0, 0));
        assert_eq!(ChromaFormat::Yuv444.idc(), Some(3));
    }

    #[test]
    fn the_classic_filters_come_out_of_the_area_mean() {
        let row = [0.0, 4.0, 8.0, 0.0, 16.0, 0.0];
        let down =
            |loc| downsample(&row, 6, 1, ChromaFormat::Yuv422, loc, Downsample::Area).unwrap();
        // Co-sited (left): [1, 2, 1] / 4 around 0, 2, 4, edge repeated.
        assert_eq!(
            down(ChromaLocation::Left),
            [
                (0.0 + 0.0 + 4.0) / 4.0,
                (4.0 + 16.0 + 0.0) / 4.0,
                (0.0 + 32.0 + 0.0) / 4.0
            ]
        );
        // Centred: [1, 1] / 2.
        assert_eq!(down(ChromaLocation::Center), [2.0, 4.0, 8.0]);
        // 4:2:0, left: [1, 2, 1] across, [1, 1] down.
        let plane = [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0];
        let d = downsample(
            &plane,
            4,
            2,
            ChromaFormat::Yuv420,
            ChromaLocation::Left,
            Downsample::Area,
        )
        .unwrap();
        assert_eq!(d, [2.0, 2.0]);
        // Top-left: on the first row, [1, 2, 1] / 4 vertically, the edge repeated.
        let d = downsample(
            &plane,
            4,
            2,
            ChromaFormat::Yuv420,
            ChromaLocation::TopLeft,
            Downsample::Area,
        )
        .unwrap();
        assert_eq!(d, [1.5, 1.5]);
    }

    #[test]
    fn ramps_survive_down_and_up_away_from_the_edges() {
        let (w, h) = (16, 12);
        for format in FORMATS {
            for loc in ChromaLocation::ALL {
                let ramp: Vec<f64> = (0..w * h)
                    .map(|p| 3.0 * real(p % w) + 2.0 * real(p / w))
                    .collect();
                let down = downsample(&ramp, w, h, format, loc, Downsample::Area).unwrap();
                // Each chroma sample is the ramp at its position.
                let (cw, ch) = format.plane(w, h);
                let (sx, sy) = format.factors();
                let (ox, oy) = loc.offset(format);
                for j in 1..ch - 1 {
                    for i in 1..cw - 1 {
                        let want = 3.0 * (real(sx * i) + ox) + 2.0 * (real(sy * j) + oy);
                        assert!(close(down[j * cw + i], want, 1e-12), "{format:?} {loc:?}");
                    }
                }
                let up = upsample(&down, w, h, format, loc, Upsample::Linear).unwrap();
                for y in 2..h - 2 {
                    for x in 2..w - 2 {
                        assert!(
                            close(up[y * w + x], ramp[y * w + x], 1e-12),
                            "{format:?} {loc:?} {x} {y}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn flat_planes_stay_flat_and_444_is_the_identity() {
        for format in FORMATS {
            for loc in ChromaLocation::ALL {
                let flat = vec![0.25; 7 * 5];
                let d = downsample(&flat, 7, 5, format, loc, Downsample::Area).unwrap();
                assert!(d.iter().all(|&v| close(v, 0.25, 1e-15)));
                for filter in [Upsample::Nearest, Upsample::Linear] {
                    let u = upsample(&d, 7, 5, format, loc, filter).unwrap();
                    assert!(u.iter().all(|&v| close(v, 0.25, 1e-15)));
                }
            }
        }
        let p: Vec<f64> = (0..12).map(f64::from).collect();
        assert_eq!(
            downsample(
                &p,
                4,
                3,
                ChromaFormat::Yuv444,
                ChromaLocation::Center,
                Downsample::Area
            )
            .unwrap(),
            p
        );
        assert_eq!(
            upsample(
                &p,
                4,
                3,
                ChromaFormat::Yuv444,
                ChromaLocation::Center,
                Upsample::Linear
            )
            .unwrap(),
            p
        );
    }

    #[test]
    fn nearest_picks_the_covering_sample_and_codes_round() {
        let c = [10.0, 20.0];
        let u = upsample(
            &c,
            4,
            1,
            ChromaFormat::Yuv422,
            ChromaLocation::Left,
            Upsample::Nearest,
        )
        .unwrap();
        assert_eq!(u, [10.0, 20.0, 20.0, 20.0]);
        let u = upsample(
            &c,
            4,
            1,
            ChromaFormat::Yuv422,
            ChromaLocation::Left,
            Upsample::Linear,
        )
        .unwrap();
        assert_eq!(u, [10.0, 15.0, 20.0, 20.0]);
        let codes = downsample_codes(
            &[0, 1, 1, 0],
            4,
            1,
            ChromaFormat::Yuv422,
            ChromaLocation::Center,
            Downsample::Area,
            255,
        )
        .unwrap();
        assert_eq!(codes, [1, 1], "0.5 rounds away from 0");
        assert_eq!(
            downsample(
                &[0.0; 3],
                2,
                2,
                ChromaFormat::Yuv420,
                ChromaLocation::Left,
                Downsample::Area
            ),
            Err(Error::Size)
        );
        assert_eq!(ChromaLocation::from_av1(2), Some(ChromaLocation::TopLeft));
        assert_eq!(ChromaLocation::from_raw(6), None);
    }
}
