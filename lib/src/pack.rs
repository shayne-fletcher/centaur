//! A pack is an array of registers.
//!
//! In `RegisterPack<V, M>`, `V` is the register type and `M` is the number of
//! registers. A lane is one value within a SIMD register. The register type
//! determines the lane type and lane count.
//!
//! For the scalar version, we use `f32` as a one-lane register.
//! `RegisterPack<f32, 4>` therefore holds four floats. If `V` is a SIMD
//! register with four float lanes, `RegisterPack<V, 4>` holds four registers
//! and sixteen floats. Call this pack `a`:
//!
//! ```text
//!                a: RegisterPack<V, 4>
//!              ┌─────────────────────────┐
//!              │       lane index        │
//!              │     0    1    2    3    │
//!              │   ┌────┬────┬────┬────┐ │
//! a[0]: V      │   │f32 │f32 │f32 │f32 │ │
//!              │   └────┴────┴────┴────┘ │
//!              │   ┌────┬────┬────┬────┐ │
//! a[1]: V      │   │f32 │f32 │f32 │f32 │ │
//!              │   └────┴────┴────┴────┘ │
//!              │   ┌────┬────┬────┬────┐ │
//! a[2]: V      │   │f32 │f32 │f32 │f32 │ │
//!              │   └────┴────┴────┴────┘ │
//!              │   ┌────┬────┬────┬────┐ │
//! a[3]: V      │   │f32 │f32 │f32 │f32 │ │
//!              │   └────┴────┴────┴────┘ │
//!              └─────────────────────────┘
//!                  4 registers × 4 lanes
//! ```
//!
//! The pack owns its registers. Indexing accesses one register. `map` applies
//! a function to each register. `zip_map` applies a function to each pair of
//! corresponding registers from two packs. Both visit registers in array
//! order; the function supplies any arithmetic on their lanes.
//!
//! Now take two packs, `a` and `b`, each shaped like the one above. Each
//! `a[i]` or `b[i]` below is a whole register containing four float lanes.
//! Let `f` take two such registers and return one register of the same type.
//! Then `c = a.zip_map(b, f)` produces a third pack with the same shape:
//! four registers, each with four float lanes.
//!
//! ```text
//!                  c = a.zip_map(b, f)
//!
//!      registers from packs a and b           result pack c
//!                                            ┌────────────┐
//!      a[0] ───┐                             │            │
//!              ├──▶ f(a[0], b[0]) ──────────▶│    c[0]    │
//!      b[0] ───┘                             │            │
//!                                            ├────────────┤
//!      a[1] ───┐                             │            │
//!              ├──▶ f(a[1], b[1]) ──────────▶│    c[1]    │
//!      b[1] ───┘                             │            │
//!                                            ├────────────┤
//!      a[2] ───┐                             │            │
//!              ├──▶ f(a[2], b[2]) ──────────▶│    c[2]    │
//!      b[2] ───┘                             │            │
//!                                            ├────────────┤
//!      a[3] ───┐                             │            │
//!              ├──▶ f(a[3], b[3]) ──────────▶│    c[3]    │
//!      b[3] ───┘                             │            │
//!                                            └────────────┘
//! ```
//!
//! A dot product uses several registers to keep separate running sums, then
//! combines those sums at the end. Here, "register" describes a value's role
//! in the algorithm. The compiler decides whether it lives in a hardware
//! register or in memory. The pack itself accepts any element type.
//!
//! The tests below are named witnesses for these invariants:
//!
//! - `INV-PACK-001`: A `RegisterPack<V, M>` owns exactly `M` values of type
//!   `V`, and indexing and borrowing expose those same values.
//! - `INV-PACK-002`: `map` and `zip_map` preserve pack length and apply their
//!   functions by corresponding register index.
//! - `INV-PACK-003`: Operations on an empty pack invoke no element callback.

use core::ops::Index;
use core::ops::IndexMut;

/// An array of `M` registers of type `V`.
///
/// The pack adds no lane, arithmetic, or architecture policy to its elements.
/// Empty packs are supported. Mapping consumes elements in index order and
/// does not require `Copy`, `Clone`, or `Default`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterPack<V, const M: usize>(pub [V; M]);

impl<V, const M: usize> RegisterPack<V, M> {
    /// Own an array of values.
    pub const fn new(values: [V; M]) -> Self {
        Self(values)
    }

    /// Borrow the underlying array.
    pub const fn as_array(&self) -> &[V; M] {
        &self.0
    }

    /// Mutably borrow the underlying array.
    pub fn as_mut_array(&mut self) -> &mut [V; M] {
        &mut self.0
    }

    /// Recover the owned array.
    pub fn into_array(self) -> [V; M] {
        self.0
    }

    /// Transform each element in index order.
    pub fn map<U>(self, f: impl FnMut(V) -> U) -> RegisterPack<U, M> {
        RegisterPack(self.0.map(f))
    }

    /// Transform pairs of elements from equally sized packs in index order.
    pub fn zip_map<W, U>(
        self,
        other: RegisterPack<W, M>,
        mut f: impl FnMut(V, W) -> U,
    ) -> RegisterPack<U, M> {
        let mut right = other.0.into_iter();
        self.map(|left| f(left, right.next().expect("equal pack lengths")))
    }
}

impl<V, const M: usize> Index<usize> for RegisterPack<V, M> {
    type Output = V;

    /// Index a value, panicking if the index is outside the pack.
    fn index(&self, index: usize) -> &V {
        &self.0[index]
    }
}

impl<V, const M: usize> IndexMut<usize> for RegisterPack<V, M> {
    fn index_mut(&mut self, index: usize) -> &mut V {
        &mut self.0[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Witnesses: INV-PACK-001 and INV-PACK-002.
    fn owned_mapping_and_access() {
        let mut values = RegisterPack::new([String::from("a"), String::from("bb")]);
        values[0].push('x');
        values.as_mut_array()[1].push('y');
        assert_eq!(values.as_array()[0], "ax");
        let result = values.zip_map(RegisterPack::new([1, 2]), |s, n| s.len() + n);
        assert_eq!(result.map(|n| n * 2).into_array(), [6, 10]);
    }

    #[test]
    // Witnesses: INV-PACK-003.
    fn empty_pack_invokes_no_callback() {
        let empty = RegisterPack::<String, 0>::new([]);
        let mapped: RegisterPack<usize, 0> = empty.map(|_| panic!("empty"));
        let result: RegisterPack<(), 0> =
            mapped.zip_map(RegisterPack::<u8, 0>::new([]), |_, _| panic!("empty"));
        assert_eq!(result.into_array(), []);
    }
}
