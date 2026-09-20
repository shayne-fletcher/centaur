//! Shared dot-product machinery for scalar and SIMD register values.
//!
//! The kernel owns the common-prefix and tail traversal. A `DotRegister`
//! supplies the meaning of one register-sized block, so the same loop can use
//! four scalar registers or four four-lane NEON registers.

use crate::RegisterPack;

/// The operations needed to use one register type in the dot-product kernel.
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
