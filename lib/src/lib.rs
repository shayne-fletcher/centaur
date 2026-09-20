//! SIMD experiments in Rust.

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
