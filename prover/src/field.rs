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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Field;

    // -----------------------------------------------------------------------
    // eq_eval tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_eq_eval_equal_binary_vectors() {
        // eq(b, b) = 1 for any binary vector
        let a = vec![F::ONE, F::ZERO, F::ONE, F::ZERO];
        assert_eq!(eq_eval(&a, &a), F::ONE);
    }

    #[test]
    fn test_eq_eval_disjoint_binary_vectors() {
        // eq(a, b) = 0 when a and b differ on at least one coordinate
        let a = vec![F::ONE, F::ZERO];
        let b = vec![F::ZERO, F::ZERO];
        assert_eq!(eq_eval(&a, &b), F::ZERO);
    }

    #[test]
    fn test_eq_eval_partial_mismatch() {
        // Vectors agree on index 0 and 2, differ on index 1
        let a = vec![F::ONE, F::ONE, F::ZERO];
        let b = vec![F::ONE, F::ZERO, F::ZERO];
        assert_eq!(eq_eval(&a, &b), F::ZERO);
    }

    #[test]
    fn test_eq_eval_all_zeros() {
        let z = vec![F::ZERO; 4];
        assert_eq!(eq_eval(&z, &z), F::ONE);
    }

    #[test]
    fn test_eq_eval_empty_slices() {
        // Product over empty set = 1 (multiplicative identity)
        assert_eq!(eq_eval(&[], &[]), F::ONE);
    }

    #[test]
    fn test_eq_eval_non_binary_same_value() {
        // eq(r, r) for r = 1/2: factor = r*r + (1-r)*(1-r) = 1/4 + 1/4 = 1/2
        let half = F::from(1u64) * F::from(2u64).inverse().unwrap();
        let result = eq_eval(&[half], &[half]);
        assert_eq!(result, half);
    }

    #[test]
    fn test_eq_eval_symmetry() {
        // eq(a, b) == eq(b, a)
        let a = vec![F::from(3u64), F::from(7u64)];
        let b = vec![F::from(5u64), F::from(2u64)];
        assert_eq!(eq_eval(&a, &b), eq_eval(&b, &a));
    }

    // -----------------------------------------------------------------------
    // index_to_bits tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_index_to_bits_zero() {
        let bits = index_to_bits(0, 4);
        assert_eq!(bits, vec![F::ZERO; 4]);
    }

    #[test]
    fn test_index_to_bits_one() {
        // 1 = 0b0001 (LE): bits[0]=1, bits[1..3]=0
        let bits = index_to_bits(1, 4);
        assert_eq!(bits[0], F::ONE);
        assert_eq!(bits[1], F::ZERO);
        assert_eq!(bits[2], F::ZERO);
        assert_eq!(bits[3], F::ZERO);
    }

    #[test]
    fn test_index_to_bits_seven() {
        // 7 = 0b0111 (LE): bits[0]=1, bits[1]=1, bits[2]=1, bits[3]=0
        let bits = index_to_bits(7, 4);
        assert_eq!(bits[0], F::ONE);
        assert_eq!(bits[1], F::ONE);
        assert_eq!(bits[2], F::ONE);
        assert_eq!(bits[3], F::ZERO);
    }

    #[test]
    fn test_index_to_bits_twelve() {
        // 12 = 0b1100 (LE): bits[0]=0, bits[1]=0, bits[2]=1, bits[3]=1
        let bits = index_to_bits(12, 4);
        assert_eq!(bits[0], F::ZERO);
        assert_eq!(bits[1], F::ZERO);
        assert_eq!(bits[2], F::ONE);
        assert_eq!(bits[3], F::ONE);
    }

    #[test]
    fn test_index_to_bits_length() {
        for num_bits in [1, 4, 8, 16] {
            let bits = index_to_bits(0, num_bits);
            assert_eq!(bits.len(), num_bits);
        }
    }

    #[test]
    fn test_index_to_bits_roundtrip() {
        // Reconstruct the index from its bits: idx = Σ_k bits[k] * 2^k
        for idx in 0..32usize {
            let bits = index_to_bits(idx, 6);
            let reconstructed: usize = bits
                .iter()
                .enumerate()
                .filter(|(_, b)| **b == F::ONE)
                .map(|(k, _)| 1usize << k)
                .sum();
            assert_eq!(reconstructed, idx, "roundtrip failed for idx={idx}");
        }
    }

    #[test]
    fn test_eq_eval_and_index_to_bits_consistency() {
        // eq(bits(i), bits(j)) = 1 iff i == j, 0 otherwise (for binary inputs)
        let m = 4;
        for i in 0..(1usize << m) {
            let bits_i = index_to_bits(i, m);
            for j in 0..(1usize << m) {
                let bits_j = index_to_bits(j, m);
                let val = eq_eval(&bits_i, &bits_j);
                if i == j {
                    assert_eq!(val, F::ONE, "eq(bits({i}), bits({j})) should be 1");
                } else {
                    assert_eq!(val, F::ZERO, "eq(bits({i}), bits({j})) should be 0");
                }
            }
        }
    }
}
