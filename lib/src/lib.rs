#![cfg_attr(not(test), no_std)]

//! Centaur explores a simple way to keep SIMD algorithms understandable.
//!
//! A register is one value of type `V`. For a scalar implementation, `V` can
//! be `f32`, which holds one number. For a SIMD implementation, `V` can be a
//! four-lane floating-point register, which holds four numbers. A
//! [`RegisterPack`] owns `M` independent registers of either kind.
//!
//! The public [`dot_f32`] function accepts ordinary slices and processes their
//! common prefix. The current implementation uses four independent scalar
//! accumulators. A later backend can replace those scalar registers with SIMD
//! registers without changing the slice-based API or the pack shape.
//!
//! The library is `no_std`: its production code uses `core` and does not
//! allocate. Tests and measurement tooling may use `std` under configuration.
//!
//! ## Invariant registry
//!
//! - `INV-CRATE-001`: The production library does not require `std`,
//!   allocation, or operating-system services. This keeps the numerical
//!   kernel usable from embedded and other `no_std` programs. The
//!   `#![cfg_attr(not(test), no_std)]` attribute and `cargo check --lib` are
//!   its build witnesses.

mod pack;
mod scalar;

pub use pack::RegisterPack;

/// Compute a dot product over the common prefix of two slices.
///
/// Returns positive zero when either slice is empty. Extra elements in the
/// longer slice are ignored. Summation order and use of fused multiply-add are
/// backend-specific, so bitwise agreement between implementations is not
/// promised.
///
/// NaNs, infinities, overflow, and signed zeros follow ordinary floating-point
/// operations in that order; no special-value normalization is performed.
///
/// ```
/// assert_eq!(centaur::dot_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0]), 14.0);
/// ```
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    scalar::dot_f32_packed(a, b)
}
