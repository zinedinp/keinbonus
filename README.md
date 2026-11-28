# keinbonus

Colour ground truth for Rust: every colour description of H.273 (CICP)
with the reference arithmetic of the standards behind it. Colour spaces,
transfer functions, YCbCr matrices, bit depths 8 to 16, limited and full
range, chroma subsampling, pixel layouts and HDR, in one crate that CPU
encoders and decoders, GPU renderers and hardware video engines can share
and be checked against.

The reference is scalar `f64`, deterministic and `no_std`: transcendental
functions come from `libm`, so the results are the same bits on every
platform (std would use the platform's libm, which differs in the last
place between MSVC and glibc). Fast paths sit on top and either match the
reference bit for bit or state by how much they don't.

> **Status: 0.1.** The reference is complete for what is listed below; the
> fast paths cover the common conversions. APIs may still change.

## Install

```toml
[dependencies]
keinbonus = "0.1"
# SIMD kernels for the fast paths (x86 AVX2, AArch64 NEON, RISC-V V):
# keinbonus = { version = "0.1", features = ["simd"] }
```

MSRV **1.100.0**. The only dependency is `libm`; the `simd` feature adds
[miraculix](https://crates.io/crates/miraculix) on the targets that have
kernels.

## Example

```rust
use keinbonus::chroma::{ChromaFormat, ChromaLocation};
use keinbonus::cicp::Cicp;
use keinbonus::convert::{self, Format, Options};
use keinbonus::image::Image;
use keinbonus::quant::BitDepth;

// An SDR white 4:2:0 frame (BT.709, 8 bit) into HDR10 (10 bit, 4:2:0).
let sdr = Format { cicp: Cicp::BT709, depth: BitDepth::B8, chroma: ChromaFormat::Yuv420, location: ChromaLocation::Left };
let hdr10 = Format { cicp: Cicp::HDR10, depth: BitDepth::B10, location: ChromaLocation::TopLeft, ..sdr };
let mut frame = Image::new(4, 2, ChromaFormat::Yuv420, BitDepth::B8)?;
frame.planes = [vec![235; 8], vec![128; 2], vec![128; 2]];
let out = convert::convert(&frame, &sdr, &hdr10, &Options::default())?;
// BT.2408: SDR white at 203 cd/m², 58 % PQ.
assert_eq!(out.planes[0][0], 573);
# Ok::<(), keinbonus::Error>(())
```

## What is in it

| Layer | Module | Standards |
|---|---|---|
| Code points | `cicp` | H.273 / ISO/IEC 23091-2: all primaries, transfer and matrix code points, presets |
| Primaries | `primaries` | H.273 Table 2, RGB to XYZ, gamut matrices (BT.2087), Bradford white adaptation |
| Transfer functions | `transfer` | BT.709, gamma 2.2/2.8, SMPTE 240M, log, xvYCC, BT.1361, sRGB, PQ (ST 2084), ST 428, HLG |
| Matrices | `matrix` | NCL (every Kr/Kb, also chroma-derived), BT.2020 constant luminance, ICtCp (PQ and HLG), YCgCo, GBR |
| Quantization | `quant` | H.273 equations, any depth 8 to 16, limited and full range, depth changes |
| Chroma | `chroma` | 4:4:4, 4:2:2, 4:2:0, 4:4:0, monochrome; all six chroma locations (H.273, HEVC, AV1) |
| Layouts | `layout` | planar, NV12/NV21/NV16/NV24, P010/P012/P016/P210/P216/P410/P416, YUY2/UYVY/Y210/Y216, v210, AYUV/Y410/Y412/Y416, RGB8/RGBA8/BGRA8/RGBA16, 10:10:10:2 both ways, FP16/FP32 |
| HDR processing | `hdr` | display light in cd/m², HLG OOTF (BT.2100), BT.2390 EETF, BT.2408 reference white |
| HDR formats | `hdr::formats` | HDR10 (ST 2086, CTA-861.3, HEVC and AV1 codings), HDR10+ (ST 2094-40 data and tone curve), HLG, Dolby Vision (profiles, base layers, configuration record) |
| Display (feature `display`) | `hdr::itm`, `primaries::gamut`, `hdr::formats::edid`, `hdr::shader` | SDR to HDR and back (BT.2446-1 Method C), exact and expanded gamut, EDID colour (E-EDID 1.4, CTA-861.3 HDR static metadata, colorimetry, HDR10+ and Dolby Vision blocks), the constants a shader needs |
| Half floats | `half` | binary16, rounded once from `f64` |
| Pipeline | `convert` | any description, depth, range and chroma format to any other |

Not yet: YDzDx (SMPTE 2085), IPT-PQ-c2 and YCgCo-R return
`Error::Unsupported` until their definitions are checked against the
standards. Dolby Vision RPU and reshaping are proprietary and stay out.

## Choices the standards leave open

- **Chroma filters.** The standards fix where a chroma sample sits, not
  how it is filtered. Downsampling is the area mean (it gives the usual
  `[1, 2, 1] / 4` and `[1, 1] / 2` filters), upsampling linear or nearest.
  Both keep linear ramps exact.
- **Display-referred conversion.** Signals are compared as a reference
  display shows them: BT.1886 for the BT.709 family, SDR white at
  203 cd/m² by default (BT.2408), HLG through its OOTF.
- **Rounding** is H.273's `Round` (halves away from zero) wherever a code
  comes out.

## Fast paths

| Path | Covers | Against the reference |
|---|---|---|
| `fast::Pipeline` | conversions that are affine on the codes: RGB and YCbCr (GBR, NCL, YCgCo), range, depth up to 12 bit, any chroma format | chroma resampling exact; the matrix at most 1 code off near rounding boundaries (measured: 0 to 0.02 % of the codes) |
| `fast::light::LightPipeline` | RGB to RGB through light: other curve, other primaries, tone mapping | bit for bit (tables of the reference's values, exact boundary windows) |
| `simd` (feature) | the kernels of `fast::Pipeline` | bit for bit the scalar `fast` kernels |

SIMD kernels, one file per architecture in `src/simd/`:

| Target | Kernels | Tested here |
|---|---|---|
| x86, x86-64 | AVX2 | yes |
| AArch64 | NEON | built and linted |
| riscv32, riscv64 | RVV 1.0 through inline assembly | built, assembly checked |
| others | scalar | yes |

`simd::warm_up()` detects the CPU once and caches the choice.

Throughput on one 1080p frame, best of several runs, AVX2 desktop CPU
(`cargo run --release --features simd --example throughput`):

| Conversion | Reference | Fixed point | SIMD |
|---|---|---|---|
| RGB 8 bit to BT.709 4:2:0 | 12 Mpx/s | 134 Mpx/s | 316 Mpx/s |
| BT.709 4:2:0 to RGB 8 bit | 11 Mpx/s | 132 Mpx/s | 346 Mpx/s |
| BT.709 4:2:0 10 bit to RGB | 11 Mpx/s | 130 Mpx/s | 346 Mpx/s |
| sRGB 8 bit to HDR10 RGB 10 bit | 1.4 Mpx/s | 18.5 Mpx/s (tables) | |

`--example stages` times each kernel scalar against SIMD.

## Testing

```sh
cargo test                    # reference and fast paths
cargo test --features simd    # plus SIMD equal to scalar, on this CPU's kernels
```

The tests check the reference against values published in the standards
(BT.2087, BT.2408, BT.2020 Table 4, ST 2084, sRGB), exhaustively where
the domain allows (every code at every depth and range, every half
float), and every fast path against the reference.

Open work, AVX-512 among it, is in [TODO.md](TODO.md).

## License

MIT OR Apache-2.0.
