//! Scalar reference and pack-shaped fallback implementations.
//!
//! `INV-SCALAR-001`: The sequential reference adds products in input order;
//! the packed implementation keeps four independent sums and may therefore
//! round differently.
//!
//! The two functions have different roles:
//!
//! ```text
//! dot_f32_scalar
//!   sum ← p0 + p1 + p2 + p3 + p4 + p5 + ...
//!
//! dot_f32_packed
//!   sum0 ← p0 + p4 + p8  + ...
//!   sum1 ← p1 + p5 + p9  + ...
//!   sum2 ← p2 + p6 + p10 + ...
//!   sum3 ← p3 + p7 + p11 + ...
//! ```
//!
//! Here `pi` means `a[i] × b[i]`. [`dot_f32_scalar`] is the sequential
//! comparison point. [`dot_f32_packed`] runs the shared kernel with the
//! [`f32` implementation](crate::kernel::DotRegister) of `DotRegister`; it is
//! the fallback selected when an architecture-specific backend is unavailable
//! or the `scalar-only` feature is enabled.

// Retained as an internal comparison point for the later benchmark API.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    // Pair matching elements, stopping at the shorter slice. Starting from
    // zero, add each product to one running sum in input order.
    a.iter().zip(b).fold(0.0, |acc, (&x, &y)| acc + x * y)
}

/// Compute a dot product with four independent scalar accumulators.
///
/// The function processes the common prefix of the two slices in groups of
/// four, keeps one running sum for each position within a group, combines the
/// four sums at the end, and then adds the remaining one to three products.
/// The grouping changes the floating-point addition order compared with
/// [`dot_f32_scalar`].
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn dot_f32_packed(a: &[f32], b: &[f32]) -> f32 {
    crate::kernel::dot_f32::<f32>(a, b)
}
