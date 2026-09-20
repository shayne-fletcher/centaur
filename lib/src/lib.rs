#![cfg_attr(not(test), no_std)]

//! Centaur explores a simple way to keep SIMD algorithms understandable.
//!
//! A register is one value of type `V`. For a scalar implementation, `V` can
//! be `f32`, which holds one number. For a SIMD implementation, `V` can be a
//! four-lane floating-point register, which holds four numbers. A
//! [`RegisterPack`] owns `M` independent registers of either kind.
//!
//! The public [`dot_f32`] function accepts ordinary slices and processes their
//! common prefix. The scalar fallback uses four independent scalar
//! accumulators; the AArch64 backend uses four independent four-lane NEON
//! registers. Both use the same pack-shaped kernel without changing the
//! slice-based API.
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
//! - `INV-API-001`: `dot_f32` processes exactly the common prefix. Complete
//!   groups and the final partial group each contribute once; an empty prefix
//!   returns positive zero.
//! - `INV-API-002`: The public result stays within the documented,
//!   length-scaled bound of the widened `f64` oracle.
//! - `INV-API-003`: Elements after the common prefix cannot affect the result,
//!   including values that would produce NaN if read.
//! - `INV-API-004`: NaN, infinity, invalid infinity-times-zero, and signed
//!   zero follow ordinary `f32` arithmetic without normalization.

mod kernel;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
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
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon::dot_f32(a, b)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        scalar::dot_f32_packed(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::dot_f32_packed;
    use crate::scalar::dot_f32_scalar;

    /// Check the public result against the widened oracle and the scalar
    /// reference.
    fn check(a: &[f32], b: &[f32]) {
        let (oracle, sum_abs) = a
            .iter()
            .zip(b)
            .fold((0.0_f64, 0.0_f64), |(sum, abs), (&a, &b)| {
                let p = f64::from(a) * f64::from(b);
                (sum + p, abs + p.abs())
            });
        // Allow rounding to grow with the number of f32 operations and with
        // the total size of the products. In symbols:
        //
        //              n − 1
        // bound = n × ε ×  Σ |aᵢ × bᵢ| + 10⁻⁶
        //              i = 0
        //
        // Here ε is f32::EPSILON. sum_abs remains meaningful when large
        // products cancel and make the final oracle value small.
        let bound = a.len().min(b.len()) as f64 * f64::from(f32::EPSILON) * sum_abs + 1e-6;
        let sequential = f64::from(dot_f32_scalar(a, b));
        let packed = f64::from(dot_f32_packed(a, b));
        let public_result = f64::from(dot_f32(a, b));
        assert!((sequential - oracle).abs() <= bound);
        assert!((packed - oracle).abs() <= bound);
        assert!((public_result - oracle).abs() <= bound);
        // Each result may be one bound away from the oracle, so their
        // difference may be as large as two bounds.
        assert!((public_result - sequential).abs() <= 2.0 * bound);
    }

    #[test]
    // Witnesses: INV-API-001 and INV-API-002.
    fn boundaries_and_unequal_lengths() {
        for n in (0_usize..=33).chain([127, 1024, 4097, 65536]) {
            let a: Vec<_> = (0..n)
                .map(|i| ((i * 17 % 101) as f32 - 50.0) / 50.0)
                .collect();
            let b: Vec<_> = (0..n)
                .map(|i| ((i * 29 % 97) as f32 - 48.0) / 48.0)
                .collect();
            check(&a, &b);
            for short in [0, n / 2, n.saturating_sub(1)] {
                check(&a[..short], &b);
                check(&a, &b[..short]);
            }
        }
        assert_eq!(dot_f32(&[], &[1.0]).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    // Witnesses: INV-API-003.
    fn ignored_suffix_does_not_affect_result() {
        for n in 0..=33 {
            let mut longer = vec![1.0; n];
            longer.extend([f32::NAN; 4]);
            let shorter = vec![1.0; n];
            assert_eq!(dot_f32(&shorter, &longer), n as f32);
            assert_eq!(dot_f32(&longer, &shorter), n as f32);
        }
    }

    #[test]
    // Witnesses: INV-SCALAR-001.
    fn sequential_reference_retains_slice_order() {
        let a = [1e8, 1.0, -1e8, 1.0];
        let b = [1.0; 4];
        assert_eq!(dot_f32_scalar(&a, &b), 1.0);
        assert_eq!(dot_f32_packed(&a, &b), 0.0);
    }

    #[test]
    // Witnesses: INV-API-004.
    fn special_values_follow_arithmetic() {
        assert!(dot_f32(&[f32::NAN], &[1.0]).is_nan());
        assert_eq!(dot_f32(&[f32::INFINITY], &[1.0]), f32::INFINITY);
        assert!(dot_f32(&[f32::INFINITY], &[0.0]).is_nan());
        assert_eq!(dot_f32(&[-0.0], &[1.0]).to_bits(), 0.0_f32.to_bits());
    }
}
