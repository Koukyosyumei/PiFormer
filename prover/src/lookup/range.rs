//! Chunked Range Proof based on Lasso & Hyrax Polynomial Commitments
//!
//! Proves that every value v in a batch lies in [0, 2^total_bits).
//!
//! **Security pillars:**
//!
//! 1. **Chunk decomposition:** v = Σ_{i=0}^{c-1} C_i · 2^(i·8),  C_i ∈ [0, 256).
//!
//! 2. **Chunk polynomial commitments (Hyrax):** before the Lasso sub-protocol
//!    runs, the prover commits to each chunk polynomial C_i(·) and absorbs the
//!    commitments into the Fiat-Shamir transcript.  This binds the prover to
//!    the decomposition before any challenge is generated.
//!
//! 3. **Lasso range check:** a single identity table T[i] = i of size 2^8 = 256
//!    proves via sumcheck + Hyrax that every chunk evaluation lies in [0, 256).
//!
//! 4. **MLE recombination check:** the verifier draws a random point r and
//!    checks  V(r) = Σ_i C_i(r) · 2^(i·8)  over the committed chunk
//!    evaluations.  This algebraic identity (with soundness error ≤ deg/|F|)
//!    proves that the committed chunks correctly reconstruct the value polynomial,
//!    closing the gap that Lasso alone cannot close.
//!
//! 5. **Hyrax openings at r:** each Hyrax open proof certifies that the
//!    committed C_i evaluates to the claimed value at r.

use crate::field::F;
use crate::lookup::lasso::{prove_lasso, verify_lasso, LassoInstance, LassoProof};
use crate::pcs::{hyrax_commit, hyrax_open, hyrax_verify, HyraxCommitment, HyraxParams, HyraxProof};
use crate::poly::DenseMLPoly;
use crate::transcript::Transcript;
use ark_ff::{Field, PrimeField};
use ark_serialize::CanonicalSerialize;

/// Width of each chunk in bits; table size = 2^CHUNK_BITS = 256.
const CHUNK_BITS: usize = 8;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Public instance for a batched range proof.
pub struct RangeProofInstance {
    /// Values to prove; each must lie in [0, 2^bits).
    pub values: Vec<F>,
    /// Total range in bits (e.g. 32 → prove [0, 2^32)).
    /// Must be a multiple of CHUNK_BITS (= 8).
    pub bits: usize,
}

/// A complete chunked range proof.
pub struct RangeProof {
    /// Hyrax commitments to chunk polynomials C_0, …, C_{c-1}.
    pub chunk_coms: Vec<HyraxCommitment>,
    /// Evaluations C_i(r) at the recombination challenge r.
    pub chunk_evals: Vec<F>,
    /// Hyrax opening proofs for each C_i(r).
    pub chunk_opens: Vec<HyraxProof>,
    /// Lasso proof that every C_i(j) ∈ [0, 2^CHUNK_BITS).
    pub lasso_proof: LassoProof,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prove that every value in `inst.values` lies in [0, 2^{inst.bits}).
pub fn prove_range(
    inst: &RangeProofInstance,
    transcript: &mut Transcript,
) -> Result<RangeProof, String> {
    validate(inst)?;

    let (num_chunks, chunk_params) = setup(inst);
    let n      = inst.values.len();
    let n_p2   = n.next_power_of_two().max(2); // ≥ 2 so n_vars ≥ 1
    let n_vars = n_p2.trailing_zeros() as usize;
    let nu     = n_vars / 2;
    let sigma  = n_vars - nu;
    let hyrax  = HyraxParams::new(sigma);

    // ── 1. Decompose values into c chunk arrays ──────────────────────────────
    let mask = (1usize << CHUNK_BITS) - 1;
    let mut chunk_evals_all = vec![vec![F::ZERO; n_p2]; num_chunks];
    let mut lasso_queries   = Vec::with_capacity(n * num_chunks);
    let mut lasso_outputs   = Vec::with_capacity(n * num_chunks);

    for (j, &v) in inst.values.iter().enumerate() {
        let v_int = v.into_bigint().as_ref()[0] as usize;
        for i in 0..num_chunks {
            let ch = (v_int >> (i * CHUNK_BITS)) & mask;
            chunk_evals_all[i][j] = F::from(ch as u64);
            // Chunk-major ordering: all queries for chunk i before chunk i+1.
            // Appended below after the outer loop to keep the right order.
            let _ = ch; // populated below
        }
    }
    // Flatten in chunk-major order (chunk 0 all values, then chunk 1, …)
    for i in 0..num_chunks {
        for j in 0..n {
            let v_int = inst.values[j].into_bigint().as_ref()[0] as usize;
            let ch = (v_int >> (i * CHUNK_BITS)) & mask;
            lasso_queries.push(ch);
            lasso_outputs.push(F::from(ch as u64));
        }
    }

    // ── 2. Commit to each chunk polynomial; absorb into transcript ───────────
    let chunk_coms: Vec<HyraxCommitment> = chunk_evals_all
        .iter()
        .map(|evals| {
            let com = hyrax_commit(evals, nu, &hyrax);
            absorb_com(transcript, &com);
            com
        })
        .collect();

    // ── 3. Lasso: prove each chunk ∈ [0, 2^CHUNK_BITS) ──────────────────────
    let identity_table: Vec<F> = (0..1usize << CHUNK_BITS)
        .map(|i| F::from(i as u64))
        .collect();
    let lasso_proof = prove_lasso(
        &LassoInstance {
            tables: vec![identity_table],
            query_indices: lasso_queries,
            outputs: lasso_outputs,
            bits_per_chunk: CHUNK_BITS,
        },
        transcript,
        &chunk_params,
    );

    // ── 4. Recombination challenge r ─────────────────────────────────────────
    let r = squeeze_r(transcript, n_vars);

    // ── 5. Evaluate + open each chunk polynomial at r ────────────────────────
    let chunk_evals: Vec<F> = chunk_evals_all
        .iter()
        .map(|evals| DenseMLPoly::new(evals.clone()).evaluate(&r))
        .collect();
    let chunk_opens: Vec<HyraxProof> = chunk_evals_all
        .iter()
        .map(|evals| hyrax_open(evals, &r, nu, sigma))
        .collect();

    Ok(RangeProof { chunk_coms, chunk_evals, chunk_opens, lasso_proof })
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify a chunked range proof.
pub fn verify_range(
    proof: &RangeProof,
    inst: &RangeProofInstance,
    transcript: &mut Transcript,
) -> Result<(), String> {
    validate(inst)?;

    let (num_chunks, chunk_params) = setup(inst);
    let n      = inst.values.len();
    let n_p2   = n.next_power_of_two().max(2);
    let n_vars = n_p2.trailing_zeros() as usize;
    let nu     = n_vars / 2;
    let sigma  = n_vars - nu;
    let hyrax  = HyraxParams::new(sigma);

    if proof.chunk_coms.len() != num_chunks
        || proof.chunk_evals.len() != num_chunks
        || proof.chunk_opens.len() != num_chunks
    {
        return Err("Range proof has wrong number of chunks".to_string());
    }

    // ── 1. Recompute chunk arrays from public values ─────────────────────────
    let mask = (1usize << CHUNK_BITS) - 1;
    let mut chunk_evals_all = vec![vec![F::ZERO; n_p2]; num_chunks];
    let mut lasso_queries   = Vec::with_capacity(n * num_chunks);
    let mut lasso_outputs   = Vec::with_capacity(n * num_chunks);

    for i in 0..num_chunks {
        for j in 0..n {
            let v_int = inst.values[j].into_bigint().as_ref()[0] as usize;
            let ch = (v_int >> (i * CHUNK_BITS)) & mask;
            chunk_evals_all[i][j] = F::from(ch as u64);
            lasso_queries.push(ch);
            lasso_outputs.push(F::from(ch as u64));
        }
    }

    // ── 2. Absorb chunk commitments; sanity-check against public values ───────
    for (i, com) in proof.chunk_coms.iter().enumerate() {
        absorb_com(transcript, com);
        let expected = hyrax_commit(&chunk_evals_all[i], nu, &hyrax);
        if expected.row_coms != com.row_coms {
            return Err(format!("Chunk {i} commitment does not match public values"));
        }
    }

    // ── 3. Lasso verification ────────────────────────────────────────────────
    let identity_table: Vec<F> = (0..1usize << CHUNK_BITS)
        .map(|i| F::from(i as u64))
        .collect();
    verify_lasso(
        &proof.lasso_proof,
        &LassoInstance {
            tables: vec![identity_table],
            query_indices: lasso_queries,
            outputs: lasso_outputs,
            bits_per_chunk: CHUNK_BITS,
        },
        transcript,
        &chunk_params,
    )
    .map_err(|e| format!("Range Lasso: {e}"))?;

    // ── 4. Recombination challenge r ─────────────────────────────────────────
    let r = squeeze_r(transcript, n_vars);

    // ── 5. Hyrax opening proofs ───────────────────────────────────────────────
    for i in 0..num_chunks {
        hyrax_verify(&proof.chunk_coms[i], proof.chunk_evals[i], &r, &proof.chunk_opens[i], &hyrax)
            .map_err(|e| format!("Chunk {i} Hyrax open: {e}"))?;
    }

    // ── 6. Recombination check: V(r) = Σ C_i(r) · 2^(i·CHUNK_BITS) ──────────
    let v_poly = {
        let mut v_evals = vec![F::ZERO; n_p2];
        for (j, &v) in inst.values.iter().enumerate() {
            v_evals[j] = v;
        }
        DenseMLPoly::new(v_evals)
    };
    let v_at_r = v_poly.evaluate(&r);

    let mut recombined = F::ZERO;
    let mut shift = F::ONE;
    let base = F::from(1u64 << CHUNK_BITS);
    for i in 0..num_chunks {
        recombined += proof.chunk_evals[i] * shift;
        shift *= base;
    }

    if recombined != v_at_r {
        return Err(format!(
            "Recombination check failed: V(r)={v_at_r:?} ≠ Σ C_i(r)·2^(i·8)={recombined:?}"
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate(inst: &RangeProofInstance) -> Result<(), String> {
    if inst.bits == 0 || inst.bits % CHUNK_BITS != 0 {
        return Err(format!(
            "bits={} must be a positive multiple of CHUNK_BITS={}",
            inst.bits, CHUNK_BITS
        ));
    }
    if inst.values.is_empty() {
        return Err("RangeProofInstance: no values to prove".to_string());
    }
    Ok(())
}

/// Returns (num_chunks, HyraxParams sized for the identity table).
fn setup(inst: &RangeProofInstance) -> (usize, HyraxParams) {
    let num_chunks = inst.bits / CHUNK_BITS;
    // Lasso HyraxParams: nu + sigma = CHUNK_BITS; choose nu = CHUNK_BITS/2.
    let lasso_sigma = CHUNK_BITS - CHUNK_BITS / 2;
    (num_chunks, HyraxParams::new(lasso_sigma))
}

fn squeeze_r(transcript: &mut Transcript, n_vars: usize) -> Vec<F> {
    (0..n_vars)
        .map(|_| transcript.challenge_field::<F>(b"range_r"))
        .collect()
}

fn absorb_com(transcript: &mut Transcript, com: &HyraxCommitment) {
    for pt in &com.row_coms {
        let mut buf = Vec::new();
        pt.serialize_compressed(&mut buf).unwrap();
        transcript.append_bytes(b"chunk_com", &buf);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod range_tests {
    use super::*;
    use crate::transcript::Transcript;

    fn inst(vals: Vec<u64>, bits: usize) -> RangeProofInstance {
        RangeProofInstance {
            values: vals.into_iter().map(F::from).collect(),
            bits,
        }
    }

    fn roundtrip(label: &[u8], ri: RangeProofInstance) -> Result<(), String> {
        let mut pt = Transcript::new(label);
        let proof = prove_range(&ri, &mut pt)?;
        let mut vt = Transcript::new(label);
        verify_range(&proof, &ri, &mut vt)
    }

    #[test]
    fn test_range_8bit_success() {
        roundtrip(b"r8", inst(vec![0, 1, 127, 255], 8)).expect("8-bit range proof failed");
    }

    #[test]
    fn test_range_16bit_success() {
        roundtrip(b"r16", inst(vec![0, 1000, 65535], 16)).expect("16-bit range proof failed");
    }

    #[test]
    fn test_range_32bit_success() {
        roundtrip(b"r32", inst(vec![0, 1, 255, 256, 65535, 1 << 24], 32))
            .expect("32-bit range proof failed");
    }

    #[test]
    fn test_range_single_zero() {
        roundtrip(b"rz", inst(vec![0], 8)).expect("single-zero range proof failed");
    }

    #[test]
    fn test_range_large_batch() {
        let vals: Vec<u64> = (0u64..32).map(|i| i * 7).collect();
        roundtrip(b"rbatch", inst(vals, 8)).expect("large-batch range proof failed");
    }

    /// Out-of-range value: the recombination check must catch it.
    ///
    /// With bits=8 and v=999: the chunk is 999 & 0xFF = 231 (valid in [0,256)).
    /// Lasso accepts the chunk, but V(r) = 999*L_j(r) ≠ 231*L_j(r) = C_0(r),
    /// so the recombination check fails.
    #[test]
    fn test_range_out_of_range_rejected() {
        let good = inst(vec![10, 20, 30], 8);

        let mut pt = Transcript::new(b"rtamper");
        let proof = prove_range(&good, &mut pt).expect("prove failed");

        // Verifier uses a tampered instance where values[1] is out of [0, 256).
        let mut bad = inst(vec![10, 20, 30], 8);
        bad.values[1] = F::from(999u64);

        let mut vt = Transcript::new(b"rtamper");
        let result = verify_range(&proof, &bad, &mut vt);
        assert!(result.is_err(), "should reject out-of-range value");
    }

    #[test]
    fn test_range_invalid_bits_rejected() {
        // bits=10 is not a multiple of 8.
        let i = inst(vec![1, 2], 10);
        let mut t = Transcript::new(b"rbits");
        assert!(prove_range(&i, &mut t).is_err());
    }
}
