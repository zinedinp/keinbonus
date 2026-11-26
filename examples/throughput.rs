//! Reference, fixed point and SIMD on one 1080p frame each way, and how
//! far the fast paths are from the reference.
//!
//! cargo run --release --features simd --example throughput

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::semicolon_if_nothing_returned,
    reason = "a benchmark"
)]

use std::time::Instant;

use keinbonus::chroma::{ChromaFormat, ChromaLocation};
use keinbonus::cicp::{Cicp, MatrixCoefficients};
use keinbonus::cicp::{ColourPrimaries, TransferCharacteristics};
use keinbonus::convert::{self, Format, Options};
use keinbonus::fast::Pipeline;
use keinbonus::fast::light::LightPipeline;
use keinbonus::image::Image;
use keinbonus::quant::BitDepth;

fn noise(w: usize, h: usize, chroma: ChromaFormat, depth: BitDepth) -> Image {
    let mut img = Image::new(w, h, chroma, depth).unwrap();
    let mut s = 1u32;
    let m = u32::from(depth.max()) + 1;
    for p in &mut img.planes {
        for v in p.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = u16::try_from((s >> 8) % m).unwrap();
        }
    }
    img
}

/// The result and the best time of `runs` (s): the least disturbed run.
fn time<T>(runs: u32, f: impl Fn() -> T) -> (T, f64) {
    let mut out = f();
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let start = Instant::now();
        out = f();
        best = best.min(start.elapsed().as_secs_f64());
    }
    (out, best)
}

fn main() {
    println!("SIMD level: {:?}", keinbonus::simd::warm_up());
    let (w, h) = (1920, 1080);
    let rgb8 = Format {
        cicp: Cicp {
            matrix: MatrixCoefficients::Identity,
            full_range: true,
            ..Cicp::BT709
        },
        depth: BitDepth::B8,
        chroma: ChromaFormat::Yuv444,
        location: ChromaLocation::Left,
    };
    let yuv420 = Format {
        cicp: Cicp::BT709,
        chroma: ChromaFormat::Yuv420,
        ..rgb8
    };
    let p010 = Format {
        cicp: Cicp::BT709,
        depth: BitDepth::B10,
        chroma: ChromaFormat::Yuv420,
        location: ChromaLocation::TopLeft,
    };
    let rgb10 = Format {
        depth: BitDepth::B10,
        ..rgb8
    };
    let cases = [
        ("RGB 8-bit -> BT.709 4:2:0 8-bit", rgb8, yuv420),
        ("BT.709 4:2:0 8-bit -> RGB 8-bit", yuv420, rgb8),
        ("BT.709 4:2:0 10-bit -> RGB 10-bit", p010, rgb10),
        ("BT.709 4:2:0 8-bit limited -> 10-bit 4:2:0", yuv420, p010),
    ];
    let o = Options::default();
    let mpix = |s: f64| (w * h) as f64 / s / 1e6;
    for (name, from, to) in cases {
        let img = noise(w, h, from.chroma, from.depth);
        let p = Pipeline::new(&from, &to, &o).unwrap();
        let (reference, t_ref) = time(3, || convert::convert(&img, &from, &to, &o).unwrap());
        let (fast, t_fast) = time(15, || p.run(&img).unwrap());
        let (simd, t_simd) = time(30, || keinbonus::simd::run(&p, &img).unwrap());
        assert_eq!(fast, simd, "SIMD must equal the fixed-point scalar path");
        let (mut differ, mut worst, mut total) = (0usize, 0u16, 0usize);
        for (a, b) in fast.planes.iter().zip(&reference.planes) {
            for (&x, &y) in a.iter().zip(b) {
                total += 1;
                if x != y {
                    differ += 1;
                    worst = worst.max(x.abs_diff(y));
                }
            }
        }
        println!("{name}");
        println!(
            "  reference {:7.1} Mpx/s, fixed point {:7.1} Mpx/s, SIMD {:7.1} Mpx/s ({:.1}x scalar, {:.0}x reference)",
            mpix(t_ref),
            mpix(t_fast),
            mpix(t_simd),
            t_fast / t_simd,
            t_ref / t_simd
        );
        println!(
            "  vs reference: {differ} of {total} codes differ ({:.4} %), by at most {worst}",
            100.0 * differ as f64 / total as f64
        );
    }
    // Through light: sRGB textures into an HDR10 swapchain (RGB, PQ, BT.2020).
    let srgb = Format {
        cicp: Cicp {
            matrix: MatrixCoefficients::Identity,
            full_range: true,
            ..Cicp::SRGB
        },
        ..rgb8
    };
    let hdr10_rgb = Format {
        cicp: Cicp {
            primaries: ColourPrimaries::Bt2020,
            transfer: TransferCharacteristics::Pq,
            matrix: MatrixCoefficients::Identity,
            full_range: true,
        },
        depth: BitDepth::B10,
        ..rgb8
    };
    let img = noise(w, h, srgb.chroma, srgb.depth);
    let build = Instant::now();
    let p = LightPipeline::new(&srgb, &hdr10_rgb, &o).unwrap();
    let build = build.elapsed().as_secs_f64() * 1e3;
    let (reference, t_ref) = time(2, || convert::convert(&img, &srgb, &hdr10_rgb, &o).unwrap());
    let (fast, t_fast) = time(10, || p.run(&img).unwrap());
    println!("sRGB 8-bit -> HDR10 RGB 10-bit (tables built in {build:.1} ms)");
    println!(
        "  reference {:7.1} Mpx/s, tables {:7.1} Mpx/s ({:.0}x), identical: {}",
        mpix(t_ref),
        mpix(t_fast),
        t_ref / t_fast,
        fast == reference
    );
}
