//! SIMD kernels for the [`fast`] paths (feature `simd`, through
//! miraculix): the same results bit for bit, picked at run time. One file
//! per architecture, each with the same kernels:
//!
//! | file | target | kernels |
//! |---|---|---|
//! | `x86.rs` | x86, x86-64 | AVX2: matrix 8 pixels, vertical chroma passes 8 columns at a time; the rest compiled for AVX2 |
//! | `aarch64.rs` | AArch64 | NEON: matrix 4 pixels, vertical chroma passes 4 columns at a time |
//! | `riscv.rs` | riscv32, riscv64 | RVV 1.0 through inline assembly (as miraculix does): matrix and vertical chroma passes for any VLEN |
//! | `generic.rs` | everything else | none yet: the scalar [`fast`] kernels |
//!
//! AVX-512 is planned: see `TODO.md`.
//!
//! Without the CPU feature (an x86 CPU without AVX2) the scalar kernels
//! run as well. [`warm_up`] detects once and caches the choice.
//!
//! ```
//! use keinbonus::chroma::{ChromaFormat, ChromaLocation};
//! use keinbonus::cicp::{Cicp, MatrixCoefficients};
//! use keinbonus::convert::{Format, Options};
//! use keinbonus::fast::Pipeline;
//! use keinbonus::image::Image;
//! use keinbonus::quant::BitDepth;
//!
//! keinbonus::simd::warm_up();
//! let rgb = Format {
//!     cicp: Cicp { matrix: MatrixCoefficients::Identity, full_range: true, ..Cicp::BT709 },
//!     depth: BitDepth::B8,
//!     chroma: ChromaFormat::Yuv444,
//!     location: ChromaLocation::Left,
//! };
//! let yuv = Format { cicp: Cicp::BT709, chroma: ChromaFormat::Yuv420, ..rgb };
//! let pipeline = Pipeline::new(&rgb, &yuv, &Options::default())?;
//! let frame = Image::new(64, 32, ChromaFormat::Yuv444, BitDepth::B8)?;
//! let fast = keinbonus::simd::run(&pipeline, &frame)?;
//! assert_eq!(fast, pipeline.run(&frame)?);
//! println!("{:?}", keinbonus::simd::level());
//! # Ok::<(), keinbonus::Error>(())
//! ```

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, Ordering};

use crate::Error;
use crate::chroma::{ChromaFormat, ChromaLocation, Upsample};
use crate::fast::{self, Affine, DownPlan, Kernels, Pipeline, UpPlan};
use crate::image::Image;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv32",
    target_arch = "riscv64"
)))]
mod generic;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod riscv;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

#[cfg(target_arch = "aarch64")]
use aarch64 as arch;
#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv32",
    target_arch = "riscv64"
)))]
use generic as arch;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
use riscv as arch;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use x86 as arch;

/// Which kernels run on this CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Level {
    Scalar,
    Avx2,
    Neon,
    Rvv,
}

/// The chosen level, once known: 0 not yet, else `Level as u8 + 1`.
static LEVEL: AtomicU8 = AtomicU8::new(0);

/// Detects the CPU once (miraculix's feature cache included) and keeps the
/// kernels to use for the process. Optional: the first conversion does it
/// otherwise; call it at startup to keep that out of the first frame.
pub fn warm_up() -> Level {
    arch::warm_up();
    let level = if arch::detect().is_some() {
        arch::LEVEL
    } else {
        Level::Scalar
    };
    LEVEL.store(level as u8 + 1, Ordering::Relaxed);
    level
}

/// The kernels the functions here use: the cached choice, detected on the
/// first call ([`warm_up`]).
#[must_use]
pub fn level() -> Level {
    match LEVEL.load(Ordering::Relaxed) {
        0 => warm_up(),
        1 => Level::Scalar,
        2 => Level::Avx2,
        3 => Level::Neon,
        _ => Level::Rvv,
    }
}

/// [`Pipeline::run`] on the fastest kernels.
///
/// # Errors
///
/// As [`Pipeline::run`].
pub fn run(pipeline: &Pipeline, image: &Image) -> Result<Image, Error> {
    pipeline.run_with(image, &Simd::detect())
}

/// [`fast::affine`], vectorized.
///
/// # Panics
///
/// When the planes differ in length.
pub fn affine(k: &Affine, input: [&[i32]; 3], out: [&mut [i32]; 3]) {
    let n = input[0].len();
    assert!(input.iter().all(|p| p.len() == n) && out.iter().all(|p| p.len() == n));
    Simd::detect().affine(k, input, out);
}

/// [`fast::upsample`], vectorized.
///
/// # Errors
///
/// As [`fast::upsample`].
pub fn upsample(
    plane: &[u16],
    width: usize,
    height: usize,
    format: ChromaFormat,
    location: ChromaLocation,
    filter: Upsample,
) -> Result<Vec<i32>, Error> {
    fast::upsample_with(
        &Simd::detect(),
        plane,
        width,
        height,
        format,
        location,
        filter,
    )
}

/// [`fast::downsample`], vectorized.
///
/// # Errors
///
/// As [`fast::downsample`].
pub fn downsample(
    plane: &[i32],
    width: usize,
    height: usize,
    format: ChromaFormat,
    location: ChromaLocation,
    frac: u32,
    max: u16,
) -> Result<Vec<u16>, Error> {
    fast::downsample_with(
        &Simd::detect(),
        plane,
        width,
        height,
        format,
        location,
        frac,
        max,
    )
}

/// The kernels of the cached level: the architecture's token when it was
/// chosen, else none (scalar). The token itself still comes from
/// miraculix's detection, one cached read: a token can't be made without.
struct Simd {
    token: Option<arch::Token>,
}

impl Simd {
    fn detect() -> Self {
        Self {
            token: if level() == arch::LEVEL {
                arch::detect()
            } else {
                None
            },
        }
    }
}

impl Kernels for Simd {
    fn widen(&self, src: &[u16], shift: u32, out: &mut [i32]) {
        match self.token {
            Some(t) => arch::widen(t, src, shift, out),
            None => fast::widen_row(src, shift, out),
        }
    }
    fn narrow(&self, src: &[i32], max: u16, out: &mut [u16]) {
        match self.token {
            Some(t) => arch::narrow(t, src, max, out),
            None => fast::narrow_row(src, max, out),
        }
    }
    fn affine(&self, k: &Affine, input: [&[i32]; 3], out: [&mut [i32]; 3]) {
        match self.token {
            Some(t) => {
                let [o0, o1, o2] = out;
                arch::affine(t, k, input[0], input[1], input[2], o0, o1, o2);
            }
            None => fast::affine(k, input, out),
        }
    }
    fn up_h(&self, src: &[u16], plan: &UpPlan, out: &mut [i32]) {
        match self.token {
            Some(t) => arch::up_h(t, src, plan, out),
            None => fast::up_h_row(src, plan, out),
        }
    }
    fn up_v(&self, a: &[i32], b: &[i32], wa: i32, wb: i32, out: &mut [i32]) {
        match self.token {
            Some(t) => arch::up_v(t, a, b, wa, wb, out),
            None => fast::up_v_row(a, b, wa, wb, out),
        }
    }
    fn down_v(&self, r: [&[i32]; 3], w: [i32; 3], out: &mut [i32]) {
        match self.token {
            Some(t) => arch::down_v(t, r, w, out),
            None => fast::down_v_row(r, w, out),
        }
    }
    fn down_h(&self, src: &[i32], plan: &DownPlan, frac: u32, max: u16, out: &mut [u16]) {
        match self.token {
            Some(t) => arch::down_h(t, src, plan, frac, max, out),
            None => fast::down_h_row(src, plan, frac, max, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chroma::ChromaLocation;
    use crate::cicp::{Cicp, MatrixCoefficients};
    use crate::convert::{Format, Options};
    use crate::fast::tests::noise;
    use crate::quant::BitDepth;
    use alloc::vec;

    #[test]
    fn warm_up_caches_the_level() {
        let first = warm_up();
        assert_eq!(level(), first);
        assert_eq!(warm_up(), first);
    }

    #[test]
    fn simd_affine_is_the_scalar_one_for_every_tail() {
        let rgb = Format {
            cicp: Cicp {
                matrix: MatrixCoefficients::Identity,
                full_range: false,
                ..Cicp::BT709
            },
            depth: BitDepth::B12,
            chroma: ChromaFormat::Yuv444,
            location: ChromaLocation::Left,
        };
        let ncl2020 = Cicp {
            matrix: MatrixCoefficients::Bt2020Ncl,
            ..Cicp::BT709
        };
        for (to, chroma) in [
            (Cicp::BT709, ChromaFormat::Yuv444),
            (ncl2020, ChromaFormat::Yuv420),
        ] {
            let to = Format {
                cicp: to,
                chroma,
                ..rgb
            };
            let k = Pipeline::new(&rgb, &to, &Options::default())
                .unwrap()
                .affine;
            for n in 0..40 {
                let img = noise(n.max(1), 1, ChromaFormat::Yuv444, BitDepth::B16, 11);
                // Inputs across the whole allowed box (up to 2^16).
                let x: Vec<Vec<i32>> = img
                    .planes
                    .iter()
                    .map(|p| p.iter().take(n).map(|&c| i32::from(c)).collect())
                    .collect();
                let (mut a, mut b) = (
                    [vec![0; n], vec![0; n], vec![0; n]],
                    [vec![0; n], vec![0; n], vec![0; n]],
                );
                {
                    let [a0, a1, a2] = &mut a;
                    fast::affine(&k, [&x[0], &x[1], &x[2]], [a0, a1, a2]);
                    let [b0, b1, b2] = &mut b;
                    affine(&k, [&x[0], &x[1], &x[2]], [b0, b1, b2]);
                }
                assert_eq!(a, b, "n = {n}");
            }
        }
    }

    #[test]
    fn simd_chroma_is_the_scalar_one() {
        for (w, h) in [(1, 1), (7, 3), (16, 2), (33, 9), (64, 5)] {
            for format in [
                ChromaFormat::Yuv420,
                ChromaFormat::Yuv422,
                ChromaFormat::Yuv440,
                ChromaFormat::Yuv444,
            ] {
                for location in ChromaLocation::ALL {
                    let (cw, ch) = format.plane(w, h);
                    let sub = noise(cw, ch, ChromaFormat::Yuv444, BitDepth::B12, 5);
                    for filter in [Upsample::Linear, Upsample::Nearest] {
                        assert_eq!(
                            upsample(&sub.planes[0], w, h, format, location, filter),
                            fast::upsample(&sub.planes[0], w, h, format, location, filter),
                            "up {w}x{h} {format:?} {location:?}"
                        );
                    }
                    let full = noise(w, h, ChromaFormat::Yuv444, BitDepth::B16, 6);
                    // Signed values with FINE_OUT bits, as the matrix leaves them.
                    let fine: Vec<i32> = full.planes[0]
                        .iter()
                        .map(|&c| (i32::from(c) - 30_000) * 37)
                        .collect();
                    assert_eq!(
                        downsample(&fine, w, h, format, location, fast::FINE_OUT, 4095),
                        fast::downsample(&fine, w, h, format, location, fast::FINE_OUT, 4095),
                        "down {w}x{h} {format:?} {location:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn simd_pipeline_is_the_scalar_one() {
        let rgb = Format {
            cicp: Cicp {
                matrix: MatrixCoefficients::Identity,
                full_range: true,
                ..Cicp::BT709
            },
            depth: BitDepth::B8,
            chroma: ChromaFormat::Yuv444,
            location: ChromaLocation::Left,
        };
        let yuv = Format {
            cicp: Cicp::BT709,
            depth: BitDepth::B10,
            chroma: ChromaFormat::Yuv420,
            location: ChromaLocation::TopLeft,
        };
        for (from, to) in [(rgb, yuv), (yuv, rgb)] {
            let p = Pipeline::new(&from, &to, &Options::default()).unwrap();
            for (w, h) in [(1, 1), (9, 7), (37, 21)] {
                let img = noise(w, h, from.chroma, from.depth, 13);
                assert_eq!(run(&p, &img).unwrap(), p.run(&img).unwrap(), "{w}x{h}");
            }
        }
    }
}
