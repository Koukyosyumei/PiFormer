//! Ternary weight constraint proof.
//!
//! Prove every element of a weight vector lies in {-1, 0, 1} (field elements
//! {p-1, 0, 1} on BN254) using a 4-element Lasso lookup table.
//!
//! **Table:** T = [0, 1, p-1, 0]  (index 3 is a dummy to pad to power-of-two)
//!   - index 0 → 0   (zero weight)
//!   - index 1 → 1   (positive weight)
//!   - index 2 → -1  (negative weight, stored as p-1 in the field)
//!
//! **Encoding:** each weight w is represented by its table index:
//!   enc(0) = 0,  enc(1) = 1,  enc(-1) = 2
//!
//! The Lasso sumcheck then proves T[enc(w)] == w for every weight.
//!
//! **Self-contained:** uses its own tiny HyraxParams (sigma=1, 2 generators).

use crate::field::F;
use crate::lookup::lasso::{
    precommit_lasso_tables, prove_lasso, verify_lasso,
    LassoInstance, LassoProof,
};
use crate::pcs::HyraxParams;
use crate::transcript::Transcript;
use ark_ff::{Field, PrimeField};

/// Public witness for the ternary weight check.
pub struct TernaryWeightInstance {
    /// Weights as field elements; each must be 0, 1, or p-1 (≡ -1 mod p).
    pub weights: Vec<F>,
}

/// Proof that all weights lie in {-1, 0, 1}.
pub struct TernaryWeightProof {
    pub lasso_proof: LassoProof,
}

pub fn prove_ternary_weights(
    inst: &TernaryWeightInstance,
    transcript: &mut Transcript,
) -> TernaryWeightProof {
    let (lasso_inst, query_indices, params) = build_lasso_instance(inst);
    let pk = precommit_lasso_tables(&lasso_inst.tables, lasso_inst.bits_per_chunk, &params);
    let lasso_proof = prove_lasso(&lasso_inst, &query_indices, &pk, transcript, &params);
    TernaryWeightProof { lasso_proof }
}

pub fn verify_ternary_weights(
    proof: &TernaryWeightProof,
    inst: &TernaryWeightInstance,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let (lasso_inst, _query_indices, params) = build_lasso_instance(inst);
    let pk = precommit_lasso_tables(&lasso_inst.tables, lasso_inst.bits_per_chunk, &params);
    let vk = pk.vk();
    verify_lasso(&proof.lasso_proof, &lasso_inst, &vk, transcript, &params)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn build_lasso_instance(inst: &TernaryWeightInstance) -> (LassoInstance, Vec<usize>, HyraxParams) {
    // 2-bit table (size 4); single chunk (c=1).
    let bits_per_chunk = 2usize;
    let neg_one = F::ZERO - F::ONE;
    // T[0] = 0, T[1] = 1, T[2] = -1, T[3] = 0 (dummy)
    let table = vec![F::ZERO, F::ONE, neg_one, F::ZERO];

    let mut query_indices = Vec::with_capacity(inst.weights.len());
    let mut outputs = Vec::with_capacity(inst.weights.len());

    for &w in &inst.weights {
        let raw = w.into_bigint().as_ref()[0];
        let enc = if raw == 0 {
            0usize
        } else if raw == 1 {
            1usize
        } else {
            // p-1 encodes -1
            2usize
        };
        query_indices.push(enc);
        outputs.push(w);
    }

    // Hyrax: m = bits_per_chunk = 2, nu = 1, sigma = 1.
    let sigma = 1usize;
    let params = HyraxParams::new(sigma);

    (
        LassoInstance {
            tables: vec![table],
            outputs,
            bits_per_chunk,
        },
        query_indices,
        params,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod ternary_tests {
    use super::*;
    use crate::transcript::Transcript;
    use ark_ff::Field;

    fn neg1() -> F {
        F::ZERO - F::ONE
    }

    #[test]
    fn test_ternary_all_values() {
        let inst = TernaryWeightInstance {
            weights: vec![F::ZERO, F::ONE, neg1(), F::ZERO, F::ONE, neg1()],
        };
        let mut pt = Transcript::new(b"ternary-test");
        let proof = prove_ternary_weights(&inst, &mut pt);

        let mut vt = Transcript::new(b"ternary-test");
        let result = verify_ternary_weights(&proof, &inst, &mut vt);
        assert!(result.is_ok(), "verify failed: {:?}", result.err());
    }

    #[test]
    fn test_ternary_all_zeros() {
        let inst = TernaryWeightInstance {
            weights: vec![F::ZERO; 8],
        };
        let mut pt = Transcript::new(b"ternary-zero");
        let proof = prove_ternary_weights(&inst, &mut pt);
        let mut vt = Transcript::new(b"ternary-zero");
        assert!(verify_ternary_weights(&proof, &inst, &mut vt).is_ok());
    }

    #[test]
    fn test_ternary_all_ones() {
        let inst = TernaryWeightInstance {
            weights: vec![F::ONE; 8],
        };
        let mut pt = Transcript::new(b"ternary-ones");
        let proof = prove_ternary_weights(&inst, &mut pt);
        let mut vt = Transcript::new(b"ternary-ones");
        assert!(verify_ternary_weights(&proof, &inst, &mut vt).is_ok());
    }

    #[test]
    fn test_ternary_all_neg_ones() {
        let inst = TernaryWeightInstance {
            weights: vec![neg1(); 8],
        };
        let mut pt = Transcript::new(b"ternary-neg");
        let proof = prove_ternary_weights(&inst, &mut pt);
        let mut vt = Transcript::new(b"ternary-neg");
        assert!(verify_ternary_weights(&proof, &inst, &mut vt).is_ok());
    }

    #[test]
    fn test_ternary_rejects_invalid_weight() {
        let inst = TernaryWeightInstance {
            weights: vec![F::ONE, F::ZERO],
        };
        let mut pt = Transcript::new(b"ternary-bad");
        let proof = prove_ternary_weights(&inst, &mut pt);

        // Tamper: pretend weight was 2 (not in {-1, 0, 1})
        let bad_inst = TernaryWeightInstance {
            weights: vec![F::ONE, F::from(2u64)],
        };
        let mut vt = Transcript::new(b"ternary-bad");
        let result = verify_ternary_weights(&proof, &bad_inst, &mut vt);
        assert!(result.is_err(), "should reject non-ternary weight");
    }
}
