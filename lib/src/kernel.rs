//! The dot-product algorithm shared by every register backend.
//!
//! # One algorithm, three register types
//!
//! [`DotRegister`] describes the operations the algorithm needs from one
//! register type. The generic [`dot_f32`] loop is compiled separately for the
//! selected type; there is no trait object or runtime method dispatch.
//!
//! ```text
//! backend   register type    lanes per register    registers    block length
//! ───────   ─────────────    ──────────────────    ─────────    ────────────
//! scalar    f32                       1                 4              4
//! NEON      float32x4_t               4                 4             16
//! SSE       __m128                    4                 4             16
//! ```
//!
//! A complete SIMD input block is divided into four registers:
//!
//! ```text
//! input block
//! ┌─────────────┬─────────────┬───────────────┬─────────────────┐
//! │ x0 x1 x2 x3 │ x4 x5 x6 x7 │ x8 x9 x10 x11 │ x12 x13 x14 x15 │
//! └──────┬──────┴──────┬──────┴───────┬───────┴────────┬────────┘
//!        │             │              │                │
//!        ▼             ▼              ▼                ▼
//!   register 0    register 1     register 2       register 3
//! ```
//!
//! The scalar backend uses the same four positions, with one `f32` in each
//! position. A pack position and, for SIMD, a lane identify one independent
//! running sum. The next block updates those same sums with the next input
//! values.
//!
//! # Division of responsibility
//!
//! This module owns common-prefix selection, complete-block traversal, four
//! independent accumulation chains, and the final scalar tail. A backend owns
//! the register representation, loads, multiply-add operation, and horizontal
//! reduction. The public entry point selects a backend at compile time.

use crate::RegisterPack;

/// The operations one register type supplies to the dot-product kernel.
///
/// `V` is one register value. It may be a scalar such as `f32`, which holds
/// one number, or a SIMD register such as `float32x4_t`, which holds four
/// numbers. The kernel keeps four independent values of `V` in a
/// `RegisterPack<V, 4>`:
///
/// ```text
/// RegisterPack<V, 4>
///   [ accumulator 0 ]
///   [ accumulator 1 ]
///   [ accumulator 2 ]
///   [ accumulator 3 ]
/// ```
///
/// The pack count is the number of independent registers in flight. It is
/// separate from the number of lanes inside one SIMD register. The kernel
/// uses this trait to keep its traversal and four independent accumulation
/// chains the same for scalar and SIMD backends. Each backend defines what one
/// register means and how its final horizontal reduction is performed.
pub(crate) trait DotRegister: Copy {
    /// Number of input floats consumed by one complete kernel iteration.
    ///
    /// ```text
    /// BLOCK_LEN = registers in the pack × f32 lanes in each register
    /// ```
    ///
    /// This kernel fixes the pack at four registers. Therefore the scalar
    /// backend consumes `4 × 1 = 4` floats, while the NEON and SSE backends
    /// consume `4 × 4 = 16` floats.
    const BLOCK_LEN: usize;

    /// Return the additive identity for one register value.
    ///
    /// The kernel uses four copies of this value to initialize its independent
    /// accumulators.
    fn zero() -> Self;

    /// Load four registers from one complete input block.
    ///
    /// The caller supplies exactly [`Self::BLOCK_LEN`] floats. The returned
    /// pack contains the four registers that correspond to the four running
    /// accumulators; a SIMD register may contain several lanes of those
    /// values.
    fn load_pack(input: &[f32]) -> RegisterPack<Self, 4>;

    /// Multiply corresponding registers and add the products to the sums.
    ///
    /// Each position in the three packs corresponds:
    ///
    /// ```text
    /// sums[i] ← sums[i] + lhs[i] × rhs[i]
    /// ```
    ///
    /// A SIMD implementation performs that equation lane by lane inside each
    /// register. A scalar implementation performs it on one `f32` at a time.
    fn mul_add(
        accumulators: RegisterPack<Self, 4>,
        lhs: RegisterPack<Self, 4>,
        rhs: RegisterPack<Self, 4>,
    ) -> RegisterPack<Self, 4>;

    /// Combine the four accumulated registers into one scalar result.
    ///
    /// This is the final horizontal reduction: first reduce any lanes inside
    /// each register, then add the four independent register totals together.
    fn horizontal_sum(accumulators: RegisterPack<Self, 4>) -> f32;
}

/// Run the common-prefix dot product over one [`DotRegister`] backend.
///
/// The two inputs are truncated to their common prefix. Complete blocks are
/// loaded through [`DotRegister::load_pack`], so a backend never receives a
/// partial block. The final one to `BLOCK_LEN - 1` values are multiplied and
/// added with scalar operations after the register loop.
///
/// The backend supplies the meaning of one register; this function supplies
/// the shared traversal, four-way accumulation, and tail behavior.
pub(crate) fn dot_f32<R: DotRegister>(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let block_len = R::BLOCK_LEN;
    let lhs_blocks = a[..n].chunks_exact(block_len);
    let rhs_blocks = b[..n].chunks_exact(block_len);
    let lhs_tail = lhs_blocks.remainder();
    let rhs_tail = rhs_blocks.remainder();
    let mut accumulators = RegisterPack::new([R::zero(); 4]);

    for (lhs_block, rhs_block) in lhs_blocks.zip(rhs_blocks) {
        accumulators = R::mul_add(
            accumulators,
            R::load_pack(lhs_block),
            R::load_pack(rhs_block),
        );
    }

    let mut total = R::horizontal_sum(accumulators);
    for (&lhs, &rhs) in lhs_tail.iter().zip(rhs_tail) {
        total += lhs * rhs;
    }
    total
}

/// Use one scalar `f32` as each register in the four-position pack.
///
/// One iteration consumes four inputs. Pack position `i` accumulates input
/// positions `i`, `i + 4`, `i + 8`, and so on.
impl DotRegister for f32 {
    const BLOCK_LEN: usize = 4;

    fn zero() -> Self {
        0.0
    }

    fn load_pack(input: &[f32]) -> RegisterPack<Self, 4> {
        RegisterPack::new([input[0], input[1], input[2], input[3]])
    }

    fn mul_add(
        accumulators: RegisterPack<Self, 4>,
        lhs: RegisterPack<Self, 4>,
        rhs: RegisterPack<Self, 4>,
    ) -> RegisterPack<Self, 4> {
        accumulators.zip_map(lhs.zip_map(rhs, |lhs, rhs| lhs * rhs), |sum, product| {
            sum + product
        })
    }

    fn horizontal_sum(accumulators: RegisterPack<Self, 4>) -> f32 {
        let [sum0, sum1, sum2, sum3] = accumulators.into_array();
        (sum0 + sum1) + (sum2 + sum3)
    }
}
