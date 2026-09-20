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
//! registers; and the x86-64 backend uses four independent four-lane SSE
//! registers. All three use the same pack-shaped kernel without changing the
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
//! - `INV-X86-001`: On x86-64 with SSE enabled, the default `dot_f32` backend
//!   uses four `__m128` registers.
//! - `INV-X86-002`: One complete SSE iteration consumes sixteen input floats;
//!   incomplete input remains in the scalar tail.
//! - `INV-X86-003`: SSE loads and stores require four valid contiguous `f32`
//!   values but do not require 16-byte SIMD alignment.
//! - `INV-DISPATCH-001`: The `scalar-only` feature excludes the NEON and SSE
//!   modules and selects the packed `f32` backend on every target.
//! - `INV-NEON-001`: On AArch64 with NEON enabled, the default `dot_f32`
//!   backend uses four `float32x4_t` registers.
//! - `INV-BENCH-001`: Enabling `bench-api` does not introduce `std` into the
//!   library.
//! - `INV-BENCH-002`: Every benchmark variant processes the common prefix and
//!   remains within the shared oracle bound.
//! - `INV-BENCH-003`: M4 release assembly contains one scalar reference
//!   chain, one NEON register chain, and four production NEON register chains.
//! - `INV-BENCH-004`: The comparison surface and benchmark example are absent
//!   unless `bench-api` is enabled.

mod kernel;
#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "aarch64",
    target_feature = "neon"
))]
mod neon;
#[cfg(any(test, feature = "bench-api"))]
mod oracle;
mod pack;
mod scalar;
#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "x86_64",
    target_feature = "sse"
))]
mod sse;

pub use pack::RegisterPack;

/// Measurement-only functions used by the benchmark example.
///
/// This module is not supported application API. It exists only when the
/// `bench-api` feature is enabled.
#[cfg(feature = "bench-api")]
#[doc(hidden)]
pub mod comparison {
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "aarch64",
        target_feature = "neon"
    ))]
    pub use crate::neon::dot_f32 as dot_f32_neon_four;
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "aarch64",
        target_feature = "neon"
    ))]
    pub use crate::neon::dot_f32_one_register as dot_f32_neon_one;
    pub use crate::oracle::dot_f32 as oracle;
    pub use crate::scalar::dot_f32_scalar as dot_f32_sequential;
}

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
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "aarch64",
        target_feature = "neon"
    ))]
    {
        neon::dot_f32(a, b)
    }
    #[cfg(all(
        not(feature = "scalar-only"),
        target_arch = "x86_64",
        target_feature = "sse"
    ))]
    {
        sse::dot_f32(a, b)
    }
    #[cfg(any(
        feature = "scalar-only",
        not(any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "x86_64", target_feature = "sse")
        ))
    ))]
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
        let (oracle, bound) = crate::oracle::dot_f32(a, b);
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
    // Witnesses: INV-X86-001, INV-X86-002, INV-DISPATCH-001, and
    // INV-NEON-001.
    fn backend_block_width_is_observable() {
        let mut a = [0.0; 32];
        a[0] = 1e8;
        a[4] = 1.0;
        a[16] = -1e8;
        let b = [1.0; 32];

        #[cfg(feature = "scalar-only")]
        assert_eq!(dot_f32(&a, &b), 0.0);

        #[cfg(all(
            not(feature = "scalar-only"),
            any(
                target_arch = "x86_64",
                all(target_arch = "aarch64", target_feature = "neon")
            )
        ))]
        assert_eq!(dot_f32(&a, &b), 1.0);

        #[cfg(all(
            not(feature = "scalar-only"),
            not(any(
                target_arch = "x86_64",
                all(target_arch = "aarch64", target_feature = "neon")
            ))
        ))]
        assert_eq!(dot_f32(&a, &b), 0.0);
    }

    #[cfg(all(
        feature = "bench-api",
        not(feature = "scalar-only"),
        target_arch = "aarch64",
        target_feature = "neon"
    ))]
    #[test]
    // Witnesses: INV-BENCH-002.
    fn benchmark_variants_follow_the_public_contract() {
        fn check_variants(a: &[f32], b: &[f32]) {
            let (oracle, bound) = crate::oracle::dot_f32(a, b);
            let variants = [
                crate::comparison::dot_f32_sequential(a, b),
                crate::comparison::dot_f32_neon_one(a, b),
                crate::comparison::dot_f32_neon_four(a, b),
            ];
            for result in variants {
                assert!((f64::from(result) - oracle).abs() <= bound);
            }
        }

        for n in (0_usize..=33).chain([127, 1024]) {
            let lhs: Vec<_> = (0..n + 3)
                .map(|i| ((i * 17 % 101) as f32 - 50.0) / 50.0)
                .collect();
            let rhs: Vec<_> = (0..n + 5)
                .map(|i| ((i * 29 % 97) as f32 - 48.0) / 48.0)
                .collect();
            check_variants(&lhs[..n], &rhs[..n]);
            check_variants(&lhs[..n], &rhs);
            check_variants(&lhs, &rhs[..n]);
        }
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
