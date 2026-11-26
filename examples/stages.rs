//! Time per stage of the fast pipeline, scalar against SIMD (1080p).
//!
//! cargo run --release --features simd --example stages

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::semicolon_if_nothing_returned,
    reason = "a benchmark"
)]

use std::time::Instant;

use keinbonus::chroma::{ChromaFormat, ChromaLocation, Upsample};
use keinbonus::cicp::{Cicp, MatrixCoefficients};
use keinbonus::convert::{Format, Options};
use keinbonus::fast::{self, Pipeline};
use keinbonus::quant::BitDepth;
use keinbonus::simd;

fn best(runs: u32, mut f: impl FnMut()) -> f64 {
    (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let (w, h) = (1920usize, 1080usize);
    let n = w * h;
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
        chroma: ChromaFormat::Yuv420,
        ..rgb
    };
    let k = Pipeline::new(&rgb, &yuv, &Options::default())
        .unwrap()
        .affine;
    let codes: Vec<u16> = (0..n).map(|i| (i * 7 % 256) as u16).collect();
    let fine: Vec<i32> = codes.iter().map(|&c| i32::from(c) << 4).collect();
    let chroma: Vec<u16> = codes[..n / 4].to_vec();
    let (mut o0, mut o1, mut o2) = (vec![0; n], vec![0; n], vec![0; n]);
    let f = ChromaFormat::Yuv420;
    let l = ChromaLocation::Left;
    println!("SIMD level: {:?}", simd::warm_up());
    println!("stage (ms, best of 15)       scalar     SIMD");
    let row = |name: &str, a: f64, b: f64| println!("{name:<28} {a:7.2}  {b:7.2}  ({:.1}x)", a / b);
    let a = best(15, || {
        fast::affine(&k, [&fine, &fine, &fine], [&mut o0, &mut o1, &mut o2])
    });
    let b = best(15, || {
        simd::affine(&k, [&fine, &fine, &fine], [&mut o0, &mut o1, &mut o2])
    });
    row("affine", a, b);
    row(
        "upsample 4:2:0",
        best(15, || {
            drop(fast::upsample(&chroma, w, h, f, l, Upsample::Linear))
        }),
        best(15, || {
            drop(simd::upsample(&chroma, w, h, f, l, Upsample::Linear))
        }),
    );
    row(
        "downsample 4:2:0",
        best(15, || drop(fast::downsample(&fine, w, h, f, l, 8, 255))),
        best(15, || drop(simd::downsample(&fine, w, h, f, l, 8, 255))),
    );
}
