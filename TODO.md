# keinbonus TODO

Order: correctness first, then vectorizing, then the rest. Every fast
path stays bit for bit the scalar `fast` path (SIMD) or the reference
(tables), or says by how much it isn't.

## AVX-512 (`src/simd/x86_512.rs`)

A second x86 level above AVX2, chosen at `warm_up` when miraculix's
`Avx512FVl`/`V4` token is there (`Level::Avx512`), AVX2 otherwise. Same
kernels, same results; the equality tests in `simd/mod.rs` run against
it as they do against AVX2 (on a machine with AVX-512: Zen 4/5, Intel
server and Ice/Tiger/Rocket Lake clients).

- **Matrix** (`affine`): 16 pixels per step, `vpmulld` + `vpaddd` on zmm,
  the two-part coefficients as now; the variable shift with `vpsravd`.
  The clamp with `vpminsd`/`vpmaxsd`. Masked loads and stores
  (`k` registers) for the tail instead of the scalar remainder.
- **Vertical chroma passes** (`up_v`, `down_v`): 16 columns per step.
- **Horizontal passes** (`up_h`, `down_h`): explicit instead of
  compiled-for-AVX2 loops. Factor-2 interleave and deinterleave with
  `vpermt2d` (two-source permute): upsampling writes the even and odd
  outputs of 16 pairs with one permute each, downsampling reads the
  `2i - 1, 2i, 2i + 1` taps of 16 outputs from three permutes. AVX2 has
  no two-source dword permute across lanes, which is why it stays scalar
  there.
- **Widen and narrow**: `vpmovzxwd` / `vpmovusdw` (unsigned saturating
  narrow) with the clamp folded in.
- **Downclocking**: on Skylake-SP and Ice Lake heavy zmm use lowers the
  clock; measure 512-bit against 256-bit EVEX (`avx512vl` on ymm, which
  also gets the extra registers and masks) and pick per CPU family.
- **Light tables** (below): `vpgatherdq` for the decode table and the
  window lookups; `vpermi2pd`-based 8-entry lookups where a table is
  that small.
- Tests: the existing `simd_*_is_the_scalar_one` tests per level, plus a
  forced-level switch for tests so AVX2 and AVX-512 both run on one
  machine.

## Light paths (`fast::light`)

- **YCbCr targets** (SDR video to HDR10 4:2:0, the encoder case): the
  matrix mixes the encoded channels, so per-channel windows don't apply.
  Plan: encode with the cubics plus a certified error bound, compute
  Y'CbCr and the code before rounding; where it lies within the bound of
  a rounding boundary, run the reference for that pixel. Exact, and the
  fallback is as rare as the bound is tight. Chroma subsampling as in
  `fast::Pipeline` (the light part at full resolution first).
- **YCbCr sources**: decode needs R'G'B' from the matrix first, which is
  continuous: same guarded approach on the decode side.
- **HLG** either side: the OOTF mixes the channels through luminance;
  tabulate in the scene-light domain and apply the OOTF in `f64` (it is
  one `pow` per pixel, not per channel), or guard as above.
- **Speed**: finer segments (four per octave) to make the guess exact
  more often; SIMD versions of the lookups (AVX2 `vgatherdpd`, AVX-512
  above); 12-bit tables (96 KiB) as `f32` windows widened by the `f32`
  rounding, the reference deciding inside.

## HDR formats (`hdr::formats`)

- **HDR10+**: parse and write the ST 2094-40 T.35 payload (HEVC SEI, AV1
  metadata OBU) into `Hdr10Plus`, all windows; display adaptation (the
  curve between the targeted and the actual display).
- **Dolby Vision**: RPU parsing for what is public (frame-level L1
  min/max/average luminance, L2 trims) where it helps tone mapping
  without a licence; reshaping stays out.
- **HDR Vivid** (CUVA) and **SL-HDR** (ETSI TS 103 433): record what they
  are, like the others.

## SIMD on hardware

- **NEON** (`simd/aarch64.rs`) and **RVV** (`simd/riscv.rs`) are built
  and linted here, not run: run `cargo test --features simd` on an
  AArch64 machine and a RISC-V board with V (or qemu with `-cpu rv64,v=true`).
- Move the generic RVV i32 kernels (`vmacc.vx` rows, vertical passes) into
  miraculix as `Rvv` methods, so other crates get them too.

## Reference

- YDzDx (SMPTE 2085), IPT-PQ-c2 (matrix 15, Dolby Vision profile 5),
  YCgCo-R (16, 17): `Error::Unsupported` until their definitions are
  checked against the standards.
