//! Range proof: prove each value v lies in [0, 2^bits).
//!
//! **Protocol:**
//! Decompose v into `c` chunks of `bits_per_chunk = ceil(bits/c)` bits each:
//!   v = Σ_{k=0}^{c-1} chunk_k(v) * 2^(k * bits_per_chunk)
//!
//! Define sub-tables T_k[i] = i * 2^(k * bits_per_chunk).
//! Then Lasso proves: v = Σ_k T_k[chunk_k(v_int)]
//! Since each chunk_k(v) ∈ [0, 2^bits_per_chunk), this proves v ∈ [0, 2^bits).
//!
//! **Self-contained:** creates its own HyraxParams internally, so callers do not
//! need to size params around the chunk tables.

use crate::field::F;
use crate::lookup::lasso::{prove_lasso, verify_lasso, LassoInstance, LassoProof};
use crate::pcs::HyraxParams;
use crate::transcript::Transcript;
use ark_ff::PrimeField;

/// Number of sub-table chunks used for the range decomposition.
const RANGE_C: usize = 4;

/// Public instance for a batched range proof.
pub struct RangeProofInstance {
    /// Values to be range-checked; each must lie in [0, 2^bits).
    pub values: Vec<F>,
    /// Total number of bits (e.g., 32 → proof of [0, 2^32)).
    pub bits: usize,
}

/// A batched range proof (wraps a Lasso proof over decomposed sub-tables).
pub struct RangeProof {
    pub lasso_proof: LassoProof,
}

/// Prove that every value in `inst.values` lies in [0, 2^{inst.bits}).
pub fn prove_range(
    inst: &RangeProofInstance,
    transcript: &mut Transcript,
) -> Result<RangeProof, String> {
    let (lasso_inst, params) = build_lasso_instance(inst)?;
    let lasso_proof = prove_lasso(&lasso_inst, transcript, &params);
    Ok(RangeProof { lasso_proof })
}

/// Verify a range proof.
pub fn verify_range(
    proof: &RangeProof,
    inst: &RangeProofInstance,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let (lasso_inst, params) = build_lasso_instance(inst)?;
    verify_lasso(&proof.lasso_proof, &lasso_inst, transcript, &params)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_lasso_instance(inst: &RangeProofInstance) -> Result<(LassoInstance, HyraxParams), String> {
    let c = RANGE_C;
    let bits_per_chunk = (inst.bits + c - 1) / c; // ceiling division
    if bits_per_chunk > 16 {
        return Err(format!(
            "bits_per_chunk={bits_per_chunk} > 16 (bits={}, c={c}); use larger c",
            inst.bits
        ));
    }
    let chunk_size = 1usize << bits_per_chunk;

    // Sub-tables: T_k[i] = i * 2^(k * bits_per_chunk)
    let tables: Vec<Vec<F>> = (0..c)
        .map(|k| {
            let shift = k * bits_per_chunk;
            (0..chunk_size)
                .map(|i| F::from((i as u64) << shift))
                .collect()
        })
        .collect();

    // Convert field elements to integer query indices.
    // We read only the first 64 bits of the big-integer representation.
    let mut query_indices = Vec::with_capacity(inst.values.len());
    let mut outputs = Vec::with_capacity(inst.values.len());
    for &v in &inst.values {
        let v_int = v.into_bigint().as_ref()[0] as usize;
        query_indices.push(v_int);
        outputs.push(v);
    }

    // Hyrax params: nu + sigma = bits_per_chunk; choose sigma = ceil(m/2).
    let nu = bits_per_chunk / 2;
    let sigma = bits_per_chunk - nu;
    let params = HyraxParams::new(sigma);

    Ok((
        LassoInstance {
            tables,
            query_indices,
            outputs,
            bits_per_chunk,
        },
        params,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod range_tests {
    use super::*;
    use crate::transcript::Transcript;

    fn small_inst(vals: Vec<u64>, bits: usize) -> RangeProofInstance {
        RangeProofInstance {
            values: vals.into_iter().map(|v| F::from(v)).collect(),
            bits,
        }
    }

    #[test]
    fn test_range_proof_success_8bit() {
        let inst = small_inst(vec![0, 1, 127, 255], 8);
        let mut pt = Transcript::new(b"range-test");
        let proof = prove_range(&inst, &mut pt).expect("prove failed");

        let mut vt = Transcript::new(b"range-test");
        let result = verify_range(&proof, &inst, &mut vt);
        assert!(result.is_ok(), "verify failed: {:?}", result.err());
    }

    #[test]
    fn test_range_proof_success_16bit() {
        let inst = small_inst(vec![0, 1000, 65535], 16);
        let mut pt = Transcript::new(b"range-16");
        let proof = prove_range(&inst, &mut pt).expect("prove failed");

        let mut vt = Transcript::new(b"range-16");
        let result = verify_range(&proof, &inst, &mut vt);
        assert!(result.is_ok(), "verify failed: {:?}", result.err());
    }

    #[test]
    fn test_range_proof_single_zero() {
        let inst = small_inst(vec![0], 8);
        let mut pt = Transcript::new(b"range-zero");
        let proof = prove_range(&inst, &mut pt).expect("prove failed");

        let mut vt = Transcript::new(b"range-zero");
        assert!(verify_range(&proof, &inst, &mut vt).is_ok());
    }

    #[test]
    fn test_range_proof_tampered_output_rejected() {
        let inst = small_inst(vec![10, 20, 30], 8);
        let mut pt = Transcript::new(b"range-tamper");
        let proof = prove_range(&inst, &mut pt).expect("prove failed");

        // Change claimed output: verifier should reject
        let mut bad_inst = small_inst(vec![10, 20, 30], 8);
        bad_inst.values[1] = F::from(999u64);

        let mut vt = Transcript::new(b"range-tamper");
        let result = verify_range(&proof, &bad_inst, &mut vt);
        assert!(result.is_err(), "should have rejected tampered output");
    }

    #[test]
    fn test_range_proof_large_batch() {
        let vals: Vec<u64> = (0..32).map(|i| i * 7).collect();
        let inst = small_inst(vals, 8);
        let mut pt = Transcript::new(b"range-batch");
        let proof = prove_range(&inst, &mut pt).expect("prove failed");
        let mut vt = Transcript::new(b"range-batch");
        assert!(verify_range(&proof, &inst, &mut vt).is_ok());
    }
}
