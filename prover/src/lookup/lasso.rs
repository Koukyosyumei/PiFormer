//! Lasso lookup argument for the structured activation φ.
//!
//! **What we prove:**
//!   For each query j ∈ [n]:
//!     output_j = Σ_{k=0}^{c-1} T_k[ chunk_k(idx_j) ]
//!   where chunk_k(x) = (x >> (k * m)) & ((1<<m)-1), m = bits_per_chunk.
//!
//! **How (batched MLE evaluation via sumcheck + Hyrax PCS):**
//!
//!   For each sub-table k:
//!   1. Commit to T_k via Hyrax: C_k = HyraxCommit(T_k).
//!   2. Build a "selector" polynomial
//!        L_k(x) = Σ_j ρ^j · eq(binary(chunk_k(idx_j)), x)
//!   3. Run sumcheck:
//!        Σ_{x ∈ {0,1}^m} T_k(x) · L_k(x) = Σ_j ρ^j · T_k[chunk_k(idx_j)]
//!   4. Open T_k at the random sumcheck point r_k via Hyrax, yielding a proof
//!      that the claimed T_k(r_k) is consistent with the committed polynomial.
//!
//! The Hyrax commitment is a transparent (no trusted setup) vector commitment
//! over BN254 G1, with O(√N) proof size and O(√N) verifier work.

use ark_ff::Field;
use crate::field::{F, eq_eval, index_to_bits};
use crate::pcs::{HyraxCommitment, HyraxParams, HyraxProof, hyrax_commit, hyrax_open, hyrax_verify};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{SumcheckProof, prove_sumcheck, verify_sumcheck};
use crate::transcript::Transcript;

/// Public description of a lookup instance.
pub struct LassoInstance {
    /// c sub-tables T_0, ..., T_{c-1}, each of size 2^bits_per_chunk.
    pub tables: Vec<Vec<F>>,
    /// Query indices in [0, 2^(c*bits_per_chunk)).
    pub query_indices: Vec<usize>,
    /// Claimed outputs: output_j = Σ_k T_k[chunk_k(idx_j)].
    pub outputs: Vec<F>,
    /// Bits per chunk (m); table size = 2^m.
    pub bits_per_chunk: usize,
}

/// Proof for a Lasso lookup (with Hyrax PCS).
pub struct LassoProof {
    /// Batched sum per sub-table: Σ_j ρ^j · T_k[chunk_k(idx_j)].
    pub sub_claims: Vec<F>,
    /// Sumcheck proof per sub-table.
    pub sumcheck_proofs: Vec<SumcheckProof>,
    /// Claimed T_k(r_k) at the sumcheck random point.
    pub table_openings: Vec<F>,
    /// Hyrax commitments to each sub-table (sent to verifier).
    pub hyrax_commitments: Vec<HyraxCommitment>,
    /// Hyrax opening proofs for T_k(r_k).
    pub hyrax_proofs: Vec<HyraxProof>,
}

pub fn prove_lasso(
    instance: &LassoInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> LassoProof {
    let c   = instance.tables.len();
    let m   = instance.bits_per_chunk;
    let n   = instance.query_indices.len();
    let mask = (1usize << m) - 1;

    // nu + sigma = m; choose nu = m/2 (square-ish layout)
    let nu    = m / 2;
    let sigma = m - nu;
    assert_eq!(params.sigma, sigma, "HyraxParams sigma mismatch");

    // Step 1: commit to each sub-table and absorb commitments into transcript
    let mut hyrax_commitments = Vec::with_capacity(c);
    for k in 0..c {
        let commitment = hyrax_commit(&instance.tables[k], nu, params);
        for pt in &commitment.row_coms {
            let bytes = {
                use ark_serialize::CanonicalSerialize;
                let mut buf = Vec::new();
                pt.serialize_compressed(&mut buf).unwrap();
                buf
            };
            transcript.append_bytes(b"hyrax_com", &bytes);
        }
        hyrax_commitments.push(commitment);
    }

    // Commit claimed outputs to transcript
    for &out in &instance.outputs {
        transcript.append_field(b"lasso_out", &out);
    }

    // Squeeze batching challenge ρ
    let rho = transcript.challenge_field::<F>(b"lasso_rho");
    let rho_pows = powers_of(rho, n);

    let mut sumcheck_proofs = Vec::with_capacity(c);
    let mut table_openings  = Vec::with_capacity(c);
    let mut hyrax_proofs    = Vec::with_capacity(c);
    let mut sub_claims      = Vec::with_capacity(c);

    for k in 0..c {
        let t_poly = DenseMLPoly::new(instance.tables[k].clone());

        // Build selector polynomial L_k as a dense MLE
        let size = 1usize << m;
        let mut l_evals = vec![F::ZERO; size];
        for j in 0..n {
            let ch = chunk(instance.query_indices[j], k, m, mask);
            for x in 0..size {
                let mut eq_val = F::ONE;
                for bit in 0..m {
                    let a = F::from(((ch >> bit) & 1) as u64);
                    let b = F::from(((x  >> bit) & 1) as u64);
                    eq_val *= a * b + (F::ONE - a) * (F::ONE - b);
                }
                l_evals[x] += rho_pows[j] * eq_val;
            }
        }
        let l_poly = DenseMLPoly::new(l_evals);

        // Claimed sum for this sub-table
        let claimed: F = (0..n)
            .map(|j| rho_pows[j] * instance.tables[k][chunk(instance.query_indices[j], k, m, mask)])
            .sum();
        sub_claims.push(claimed);

        let (sc_proof, r_vec) = prove_sumcheck(&t_poly, &l_poly, claimed, transcript);

        // Hyrax opening at r_vec
        let opening = t_poly.evaluate(&r_vec);
        transcript.append_field(b"lasso_opening", &opening);
        let hyrax_proof = hyrax_open(&instance.tables[k], &r_vec, nu, sigma);

        table_openings.push(opening);
        sumcheck_proofs.push(sc_proof);
        hyrax_proofs.push(hyrax_proof);
    }

    LassoProof { sub_claims, sumcheck_proofs, table_openings, hyrax_commitments, hyrax_proofs }
}

pub fn verify_lasso(
    proof: &LassoProof,
    instance: &LassoInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let c    = instance.tables.len();
    let m    = instance.bits_per_chunk;
    let n    = instance.query_indices.len();
    let mask = (1usize << m) - 1;

    let nu    = m / 2;
    let sigma = m - nu;
    assert_eq!(params.sigma, sigma, "HyraxParams sigma mismatch");

    // Replay commitment absorptions
    for k in 0..c {
        for pt in &proof.hyrax_commitments[k].row_coms {
            let bytes = {
                use ark_serialize::CanonicalSerialize;
                let mut buf = Vec::new();
                pt.serialize_compressed(&mut buf).unwrap();
                buf
            };
            transcript.append_bytes(b"hyrax_com", &bytes);
        }
    }

    for &out in &instance.outputs {
        transcript.append_field(b"lasso_out", &out);
    }
    let rho = transcript.challenge_field::<F>(b"lasso_rho");
    let rho_pows = powers_of(rho, n);

    for k in 0..c {
        // Check sub-claim matches public inputs
        let expected: F = (0..n)
            .map(|j| rho_pows[j] * instance.tables[k][chunk(instance.query_indices[j], k, m, mask)])
            .sum();
        if expected != proof.sub_claims[k] {
            return Err(format!("Lasso sub-claim mismatch for table {k}"));
        }

        let (r_vec, _) = verify_sumcheck(&proof.sumcheck_proofs[k], proof.sub_claims[k], m, transcript)
            .map_err(|e| format!("Table {k} sumcheck: {e}"))?;

        // Replay opening absorption
        let t_opening = proof.table_openings[k];
        transcript.append_field(b"lasso_opening", &t_opening);

        // Verify sumcheck final claim: T_k(r) · L_k(r) == final
        //
        // Note on bit ordering: DenseMLPoly fixes the MSB first (r_vec[0] = bit_{m-1}).
        // index_to_bits is LSB-first, and eq_eval pairs r_rev[i] with bits[i].
        // So we reverse r_vec before computing L_k(r).
        let r_rev: Vec<F> = r_vec.iter().copied().rev().collect();
        let l_at_r: F = (0..n)
            .map(|j| {
                let ch   = chunk(instance.query_indices[j], k, m, mask);
                let bits = index_to_bits(ch, m);
                rho_pows[j] * eq_eval(&bits, &r_rev)
            })
            .sum();

        let expected_final = t_opening * l_at_r;
        let actual_final   = proof.sumcheck_proofs[k].final_eval_f
            * proof.sumcheck_proofs[k].final_eval_g;
        if expected_final != actual_final {
            return Err(format!(
                "Table {k} sumcheck final check failed: T(r)*L(r)={expected_final:?} ≠ final={actual_final:?}"
            ));
        }

        // Verify Hyrax opening: T_k(r_vec) == t_opening
        hyrax_verify(
            &proof.hyrax_commitments[k],
            t_opening,
            &r_vec,
            &proof.hyrax_proofs[k],
            params,
        ).map_err(|e| format!("Table {k} Hyrax: {e}"))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the k-th chunk of `idx`: (idx >> (k*m)) & mask.
fn chunk(idx: usize, k: usize, m: usize, mask: usize) -> usize {
    (idx >> (k * m)) & mask
}

/// Compute [1, ρ, ρ², ..., ρ^{n-1}].
fn powers_of(rho: F, n: usize) -> Vec<F> {
    let mut v = Vec::with_capacity(n);
    let mut cur = F::ONE;
    for _ in 0..n {
        v.push(cur);
        cur *= rho;
    }
    v
}
