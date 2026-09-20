//! Scalar dot-product implementations and their behavioral witnesses.
//!
//! The tests below are named witnesses for these invariants:
//!
//! - `INV-SCALAR-001`: `dot_f32` processes exactly the common prefix. Complete
//!   groups and the final partial group each contribute once; an empty prefix
//!   returns positive zero.
//! - `INV-SCALAR-002`: For the finite deterministic fixtures used here, the
//!   sequential and packed results stay within the documented, length-scaled
//!   bound of the widened `f64` oracle and within twice that bound of each
//!   other.
//! - `INV-SCALAR-003`: Elements after the common prefix cannot affect the
//!   result, including values that would produce NaN if read.
//! - `INV-SCALAR-004`: The sequential reference adds products in input order;
//!   the packed implementation keeps four independent sums and may therefore
//!   round differently.
//! - `INV-SCALAR-005`: NaN, infinity, invalid infinity-times-zero, and signed
//!   zero follow ordinary `f32` arithmetic without normalization.

use crate::RegisterPack;

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
pub(crate) fn dot_f32_packed(a: &[f32], b: &[f32]) -> f32 {
    // Only process the common prefix: every element needs a matching partner.
    let n = a.len().min(b.len());

    // Borrow the first n elements and split them into typed complete groups of
    // four plus a zero-to-three-element remainder. For n = 11, each iterator
    // yields two four-element arrays and a remainder of three elements. Both
    // tails have the same length because both prefixes have n.
    let (lhs_chunks, lhs_tail) = a[..n].as_chunks::<4>();
    let (rhs_chunks, rhs_tail) = b[..n].as_chunks::<4>();

    // Four scalar registers, each holding its own running sum. As in the
    // diagram in pack.rs, each row below is one register. There, V has four
    // float lanes; here V = f32, so each row has just one lane. M = 4 in both
    // cases: the pack still contains four registers.
    //
    //          accumulators: RegisterPack<f32, 4>
    //                 ┌─────────────────┐
    //                 │   lane index    │
    //                 │        0        │
    //                 │    ┌───────┐    │
    // [0]: f32        │    │  0.0  │    │
    //                 │    └───────┘    │
    //                 │    ┌───────┐    │
    // [1]: f32        │    │  0.0  │    │
    //                 │    └───────┘    │
    //                 │    ┌───────┐    │
    // [2]: f32        │    │  0.0  │    │
    //                 │    └───────┘    │
    //                 │    ┌───────┐    │
    // [3]: f32        │    │  0.0  │    │
    //                 │    └───────┘    │
    //                 └─────────────────┘
    //                    4 registers × 1 lane
    //
    // [0.0; 4] creates the array of four zeros; new wraps it in a pack.
    let mut accumulators = RegisterPack::new([0.0; 4]);

    // Pair corresponding groups from the two input slices.
    for (&lhs_chunk, &rhs_chunk) in lhs_chunks.iter().zip(rhs_chunks) {
        // Copy each group into a pack: V is f32 and M is 4.
        let lhs_registers = RegisterPack::new(lhs_chunk);
        let rhs_registers = RegisterPack::new(rhs_chunk);

        // The inner zip_map multiplies corresponding registers. The outer
        // zip_map adds each product to its corresponding running sum.
        //
        //     lhs:     [ L0 ] [ L1 ] [ L2 ] [ L3 ]
        //                 │      │      │      │
        //     rhs:     [ R0 ] [ R1 ] [ R2 ] [ R3 ]
        //                 │      │      │      │
        //     product: [L0×R0] [L1×R1] [L2×R2] [L3×R3]
        //
        // accumulators[0] collects input positions 0, 4, 8, ...;
        // accumulators[1] collects positions 1, 5, 9, ...; and so on.
        // Each running sum is independent.
        let products = lhs_registers.zip_map(rhs_registers, |lhs, rhs| lhs * rhs);
        accumulators = accumulators.zip_map(products, |sum, product| sum + product);
    }

    // Combine the four running sums in pairs. This addition order can round
    // differently from the sequential reference above.
    let [sum0, sum1, sum2, sum3] = accumulators.into_array();
    let mut total = (sum0 + sum1) + (sum2 + sum3);

    // Finish the leftover pairs one at a time. The & patterns copy the f32
    // values out of the references yielded by the slice iterators.
    for (&lhs, &rhs) in lhs_tail.iter().zip(rhs_tail) {
        total += lhs * rhs;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dot_f32;

    fn check(a: &[f32], b: &[f32]) {
        let (oracle, sum_abs) = a
            .iter()
            .zip(b)
            .fold((0.0_f64, 0.0_f64), |(sum, abs), (&a, &b)| {
                let p = f64::from(a) * f64::from(b);
                (sum + p, abs + p.abs())
            });
        let bound = a.len().min(b.len()) as f64 * f64::from(f32::EPSILON) * sum_abs + 1e-6;
        let sequential = f64::from(dot_f32_scalar(a, b));
        let packed = f64::from(dot_f32(a, b));
        assert!((sequential - oracle).abs() <= bound);
        assert!((packed - oracle).abs() <= bound);
        assert!((packed - sequential).abs() <= 2.0 * bound);
    }

    #[test]
    // Witnesses: INV-SCALAR-001 and INV-SCALAR-002.
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
    // Witnesses: INV-SCALAR-003.
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
    // Witnesses: INV-SCALAR-004.
    fn sequential_reference_retains_slice_order() {
        let a = [1e8, 1.0, -1e8, 1.0];
        let b = [1.0; 4];
        assert_eq!(dot_f32_scalar(&a, &b), 1.0);
        assert_eq!(dot_f32(&a, &b), 0.0);
    }

    #[test]
    // Witnesses: INV-SCALAR-005.
    fn special_values_follow_arithmetic() {
        assert!(dot_f32(&[f32::NAN], &[1.0]).is_nan());
        assert_eq!(dot_f32(&[f32::INFINITY], &[1.0]), f32::INFINITY);
        assert!(dot_f32(&[f32::INFINITY], &[0.0]).is_nan());
        assert_eq!(dot_f32(&[-0.0], &[1.0]).to_bits(), 0.0_f32.to_bits());
    }
}
