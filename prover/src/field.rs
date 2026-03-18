//! Type alias and helpers for the BN254 scalar field.

use ark_bn254::Fr;

/// The scalar field used throughout PiFormer: BN254 Fr (~254-bit prime field).
pub type F = Fr;

/// Evaluate the multilinear equality polynomial:
///   eq(a, b) = Π_i ( a_i·b_i + (1-a_i)·(1-b_i) )
/// for two equal-length slices of field elements.
pub fn eq_eval(a: &[F], b: &[F]) -> F {
    use ark_ff::Field;
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).fold(F::ONE, |acc, (&ai, &bi)| {
        acc * (ai * bi + (F::ONE - ai) * (F::ONE - bi))
    })
}

/// Convert an integer index to its little-endian binary representation
/// as field elements (bit k = (idx >> k) & 1).
pub fn index_to_bits(idx: usize, num_bits: usize) -> Vec<F> {
    use ark_ff::Field;
    (0..num_bits)
        .map(|k| if (idx >> k) & 1 == 1 { F::ONE } else { F::ZERO })
        .collect()
}
