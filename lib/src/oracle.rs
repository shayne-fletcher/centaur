//! Widened reference result and error bound shared by tests and measurements.

/// Compute the common-prefix dot product in `f64` and its `f32` error bound.
///
/// The bound is:
///
/// ```text
///             n − 1
/// n × ε ×      Σ |aᵢ × bᵢ| + 10⁻⁶
///             i = 0
/// ```
///
/// Here `n` is the common-prefix length and `ε` is [`f32::EPSILON`].
pub fn dot_f32(a: &[f32], b: &[f32]) -> (f64, f64) {
    let n = a.len().min(b.len());
    let (value, sum_abs) =
        a.iter()
            .zip(b)
            .fold((0.0_f64, 0.0_f64), |(sum, sum_abs), (&lhs, &rhs)| {
                let product = f64::from(lhs) * f64::from(rhs);
                (sum + product, sum_abs + product.abs())
            });
    let bound = n as f64 * f64::from(f32::EPSILON) * sum_abs + 1e-6;
    (value, bound)
}
