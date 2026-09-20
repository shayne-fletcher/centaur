//! AArch64 NEON implementation of the shared dot-product register operations.
//!
//! The backend uses Rust's native `float32x4_t` type directly. Four values of
//! that type form the independent-register pack used by the kernel:
//!
//! ```text
//! RegisterPack<float32x4_t, 4>
//!
//!   [ f32 f32 f32 f32 ]   accumulator 0
//!   [ f32 f32 f32 f32 ]   accumulator 1
//!   [ f32 f32 f32 f32 ]   accumulator 2
//!   [ f32 f32 f32 f32 ]   accumulator 3
//! ```
//!
//! NEON intrinsic names follow a regular pattern:
//!
//! ```text
//! v  operation  shape/modifier  lane type
//! │      │             │              │
//! │      │             │              └── f32
//! │      │             └── q: 128-bit vector
//! │      └── ld1, dup, fma, or add
//! └── vector operation
//! ```
//!
//! `vld1q_f32` loads four contiguous floats, `vdupq_n_f32` broadcasts one
//! float to all lanes, `vfmaq_f32` performs lane-wise fused multiply-add, and
//! `vaddvq_f32` adds the four lanes horizontally into one scalar.
//!
//! The loads are valid for any properly aligned `f32` slice; they do not
//! require the slice start to be aligned to a 16-byte SIMD boundary. The
//! `vld1q_f32` intrinsic is an AArch64 `LD1` load, which accepts an unaligned
//! address. `as_chunks::<4>()` supplies four contiguous, initialized `f32`
//! values for each load, while preserving the slice's ordinary `f32`
//! alignment requirement.

use core::arch::aarch64::float32x4_t;
use core::arch::aarch64::vaddvq_f32;
use core::arch::aarch64::vdupq_n_f32;
use core::arch::aarch64::vfmaq_f32;
use core::arch::aarch64::vld1q_f32;

use crate::RegisterPack;
use crate::kernel::DotRegister;

impl DotRegister for float32x4_t {
    const BLOCK_LEN: usize = 16;

    fn zero() -> Self {
        // vdupq_n_f32 broadcasts one scalar zero into all four lanes.
        // SAFETY: this module is compiled only when AArch64 NEON is enabled.
        unsafe { vdupq_n_f32(0.0) }
    }

    fn load_pack(input: &[f32]) -> RegisterPack<Self, 4> {
        let (chunks, tail) = input.as_chunks::<4>();
        debug_assert!(tail.is_empty());
        // Each typed chunk contains four valid f32 values, so every vld1q_f32
        // load reads exactly within the complete input block supplied by the
        // kernel. No 16-byte alignment is required by this LD1 load.
        // SAFETY: the NEON gate is active, and each pointer addresses four
        // initialized contiguous f32 values from the complete input block.
        unsafe {
            RegisterPack::new([
                vld1q_f32(chunks[0].as_ptr()),
                vld1q_f32(chunks[1].as_ptr()),
                vld1q_f32(chunks[2].as_ptr()),
                vld1q_f32(chunks[3].as_ptr()),
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
            // vfmaq_f32 computes sum + lhs × rhs independently in each lane.
            |sum, (lhs, rhs)| {
                // SAFETY: this module is compiled only when AArch64 NEON is
                // enabled, and all values are native float32x4_t registers.
                unsafe { vfmaq_f32(sum, lhs, rhs) }
            },
        )
    }

    fn horizontal_sum(accumulators: RegisterPack<Self, 4>) -> f32 {
        // vaddvq_f32 reduces each vector horizontally; the iterator then adds
        // the four independent register totals together.
        accumulators
            .into_array()
            .into_iter()
            .map(|value| {
                // SAFETY: this module is compiled only when AArch64 NEON is
                // enabled, and value is a native float32x4_t register.
                unsafe { vaddvq_f32(value) }
            })
            .sum()
    }
}

pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    crate::kernel::dot_f32::<float32x4_t>(a, b)
}
