//! Every other target (PowerPC, LoongArch, Wasm, ...): no SIMD kernels
//! yet, the scalar [`fast`](crate::fast) ones run.
//!
//! The token can't exist, so the kernels below are never called.

#![allow(
    clippy::too_many_arguments,
    reason = "the same signatures as the SIMD modules"
)]

use super::Level;
use crate::fast::{Affine, DownPlan, UpPlan};

/// No proof: no value of this type exists.
#[derive(Debug, Clone, Copy)]
pub(super) enum Token {}

pub(super) const LEVEL: Level = Level::Scalar;

pub(super) fn warm_up() {}

pub(super) fn detect() -> Option<Token> {
    None
}

pub(super) fn affine(
    t: Token,
    _: &Affine,
    _: &[i32],
    _: &[i32],
    _: &[i32],
    _: &mut [i32],
    _: &mut [i32],
    _: &mut [i32],
) {
    match t {}
}

pub(super) fn up_v(t: Token, _: &[i32], _: &[i32], _: i32, _: i32, _: &mut [i32]) {
    match t {}
}

pub(super) fn down_v(t: Token, _: [&[i32]; 3], _: [i32; 3], _: &mut [i32]) {
    match t {}
}

pub(super) fn widen(t: Token, _: &[u16], _: u32, _: &mut [i32]) {
    match t {}
}

pub(super) fn narrow(t: Token, _: &[i32], _: u16, _: &mut [u16]) {
    match t {}
}

pub(super) fn up_h(t: Token, _: &[u16], _: &UpPlan, _: &mut [i32]) {
    match t {}
}

pub(super) fn down_h(t: Token, _: &[i32], _: &DownPlan, _: u32, _: u16, _: &mut [u16]) {
    match t {}
}
