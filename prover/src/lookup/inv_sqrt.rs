//! Structured lookup tables for approximate inverse square root.
//!
//! Approximates f(x) = SCALE / sqrt(x) using two additive sub-tables of size 2^16:
//!   T_0[x & 0xFFFF] + T_1[x >> 16] ≈ SCALE / sqrt(x)
//!
//! The Lasso protocol uses chunk ordering k=0 → low bits, k=1 → high bits:
//!   chunk_k(idx) = (idx >> (k * m)) & mask
//! so T_0 must be the table indexed by low bits and T_1 by high bits.
//!
//! T_0 (low bits) = 0 — the low 16 bits contribute negligible correction.
//! T_1 (high bits) carries the dominant approximation:
//!   T_1[h] = round(SCALE / sqrt((h + 0.5) * 2^16))
//!
//! These tables plug directly into `LassoInstance` with `bits_per_chunk = 16`,
//! `c = 2` sub-tables, where Lasso verifies: output = T_0[lo] + T_1[hi].

use crate::field::F;
use crate::lookup::lasso::{precommit_lasso_tables, LassoProvingKey};
use crate::pcs::HyraxParams;
use ark_ff::Field;

/// Fixed-point scale for inv_sigma: the looked-up value represents
/// `round(INV_SQRT_SCALE / sqrt(var_x))`.
pub const INV_SQRT_BITS: usize = 16;
pub const INV_SQRT_SCALE: u64 = 1u64 << INV_SQRT_BITS; // 65536
pub const CHUNK_BITS: usize = 16;
pub const CHUNK_SIZE: usize = 1 << CHUNK_BITS; // 65536

/// Build the two additive sub-tables of size 2^16.
///
/// For query index x (32-bit variance):
///   lo = x & 0xFFFF,  hi = x >> 16
///   approx_inv_sqrt(x) = T_0[lo] + T_1[hi]
///
/// This matches the Lasso chunk ordering: chunk_0 = low bits (k=0), chunk_1 = high bits (k=1).
///
/// T_0[l] = 0  (low bits contribute negligible correction)
/// T_1[h] = round(SCALE / sqrt((h + 0.5) * 2^16))  using midpoint to reduce bias.
pub fn build_inv_sqrt_tables() -> (Vec<F>, Vec<F>) {
    let scale = INV_SQRT_SCALE as f64;
    let chunk_f = CHUNK_SIZE as f64; // 2^16

    let t0 = vec![F::ZERO; CHUNK_SIZE]; // low bits → zero
    let mut t1 = vec![F::ZERO; CHUNK_SIZE]; // high bits → approximation

    for h in 0..CHUNK_SIZE {
        let x_center = (h as f64 + 0.5) * chunk_f;
        let val: u64 = if x_center > 0.0 {
            (scale / x_center.sqrt()).round() as u64
        } else {
            // h == 0: var_x < 2^16 (near-constant row). Use the max table entry
            // from h=1 as a conservative approximation.
            (scale / (0.5 * chunk_f).sqrt()).round() as u64
        };
        t1[h] = F::from(val);
    }

    (t0, t1)
}

/// Evaluate the two-table approximation for a single var_x value (u64).
///
/// Matches Lasso chunk ordering: T_0 indexed by low bits (k=0), T_1 by high bits (k=1).
pub fn lookup_inv_sqrt(var_x: u64, t0: &[F], t1: &[F]) -> F {
    let hi = (var_x >> CHUNK_BITS) as usize;
    let lo = (var_x & ((1u64 << CHUNK_BITS) - 1)) as usize;
    t0[lo.min(CHUNK_SIZE - 1)] + t1[hi.min(CHUNK_SIZE - 1)]
}

/// Precommit inv-sqrt tables for LayerNorm setup.
/// Call once at setup; share the resulting key across all LayerNorm instances.
pub fn precommit_inv_sqrt_tables(params: &HyraxParams) -> (LassoProvingKey, Vec<F>, Vec<F>) {
    let (t0, t1) = build_inv_sqrt_tables();
    let pk = precommit_lasso_tables(&[t0.clone(), t1.clone()], CHUNK_BITS, params);
    (pk, t0, t1)
}
