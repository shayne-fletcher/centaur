//! x86-64 SSE implementation of the shared dot-product register operations.
//!
//! The backend uses Rust's native `__m128` type directly. Four values of that
//! type form the same independent-register pack used by the scalar and NEON
//! backends:
//!
//! ```text
//! RegisterPack<__m128, 4>
//!
//!   [ f32 f32 f32 f32 ]   accumulator 0
//!   [ f32 f32 f32 f32 ]   accumulator 1
//!   [ f32 f32 f32 f32 ]   accumulator 2
//!   [ f32 f32 f32 f32 ]   accumulator 3
//! ```
//!
//! `_mm_loadu_ps` loads four contiguous floats, `_mm_setzero_ps` creates a
//! zero register, and `_mm_mul_ps` plus `_mm_add_ps` performs lane-wise
//! multiply-accumulate without requiring the optional FMA feature. SSE is
//! part of the x86-64 baseline.
//!
//! The `u` in `_mm_loadu_ps` means unaligned. Each address must still point to
//! four contiguous, initialized `f32` values, but it need not be aligned to a
//! 16-byte SIMD boundary.

use core::arch::x86_64::__m128;
use core::arch::x86_64::_mm_add_ps;
use core::arch::x86_64::_mm_loadu_ps;
use core::arch::x86_64::_mm_mul_ps;
use core::arch::x86_64::_mm_setzero_ps;
use core::arch::x86_64::_mm_storeu_ps;

use crate::RegisterPack;
use crate::kernel::DotRegister;

impl DotRegister for __m128 {
    const BLOCK_LEN: usize = 16;

    fn zero() -> Self {
        // SAFETY: this module is compiled only when x86-64 SSE is enabled.
        unsafe { _mm_setzero_ps() }
    }

    fn load_pack(input: &[f32]) -> RegisterPack<Self, 4> {
        let (chunks, tail) = input.as_chunks::<4>();
        debug_assert!(tail.is_empty());
        // SAFETY: the SSE gate is active, and each pointer addresses four
        // initialized contiguous f32 values. _mm_loadu_ps requires no
        // 16-byte alignment.
        unsafe {
            RegisterPack::new([
                _mm_loadu_ps(chunks[0].as_ptr()),
                _mm_loadu_ps(chunks[1].as_ptr()),
                _mm_loadu_ps(chunks[2].as_ptr()),
                _mm_loadu_ps(chunks[3].as_ptr()),
            ])
        }
    }

    fn mul_add(
        accumulators: RegisterPack<Self, 4>,
        lhs: RegisterPack<Self, 4>,
        rhs: RegisterPack<Self, 4>,
    ) -> RegisterPack<Self, 4> {
        accumulators.zip_map(
            lhs.zip_map(rhs, |lhs, rhs| (lhs, rhs)),
            |sum, (lhs, rhs)| {
                // SAFETY: this module is compiled only when x86-64 SSE is
                // enabled, and all values are native __m128 registers.
                unsafe { _mm_add_ps(sum, _mm_mul_ps(lhs, rhs)) }
            },
        )
    }

    fn horizontal_sum(accumulators: RegisterPack<Self, 4>) -> f32 {
        let [sum0, sum1, sum2, sum3] = accumulators.into_array().map(|value| {
            let mut lanes = [0.0; 4];
            // SAFETY: the SSE gate is active, value is a native __m128
            // register, and lanes has space for four contiguous f32s.
            unsafe { _mm_storeu_ps(lanes.as_mut_ptr(), value) };
            (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
        });
        (sum0 + sum1) + (sum2 + sum3)
    }
}

pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    crate::kernel::dot_f32::<__m128>(a, b)
}
