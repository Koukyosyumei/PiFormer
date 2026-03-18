//! Simplified Lasso lookup argument for the structured activation φ.
//!
//! **What we prove:**
//!   For each query j ∈ [n]:
//!     output_j = Σ_{k=0}^{c-1} T_k[ chunk_k(idx_j) ]
//!   where chunk_k(x) = (x >> (k * m)) & ((1<<m)-1), m = bits_per_chunk.
//!
//! **How (batched MLE evaluation via sumcheck):**
//!
//!   For each sub-table k:
//!   1. Represent T_k as a DenseMLPoly over m variables (size 2^m).
//!   2. Build a "selector" polynomial
//!        L_k(x) = Σ_j ρ^j · eq(binary(chunk_k(idx_j)), x)
//!      which has exactly one nonzero "spike" at each queried index.
//!   3. Run sumcheck:
//!        Σ_{x ∈ {0,1}^m} T_k(x) · L_k(x) = Σ_j ρ^j · T_k[chunk_k(idx_j)]
//!   4. Open T_k at the random sumcheck point r_k (trivial PCS: reveal T_k(r_k)).
//!      A production system replaces this with a Dory/IPA opening proof.
//!
//! **Note on security:** The trivial PCS is sound only in an honest-prover / testing
//! setting. The prover commits T_k before seeing queries; the opening at random r
//! is binding under the random-oracle assumption plus the PCS binding property.

use ark_ff::Field;
use crate::field::{F, eq_eval, index_to_bits};
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

/// Proof for a Lasso lookup.
pub struct LassoProof {
    pub sumcheck_proofs: Vec<SumcheckProof>,
    /// T_k(r_k): prover's evaluation claim at the sumcheck random point (trivial PCS).
    pub table_openings: Vec<F>,
    /// Claimed batched sum per sub-table: Σ_j ρ^j · T_k[chunk_k(idx_j)].
    pub sub_claims: Vec<F>,
}

pub fn prove_lasso(instance: &LassoInstance, transcript: &mut Transcript) -> LassoProof {
    let c   = instance.tables.len();
    let m   = instance.bits_per_chunk;
    let n   = instance.query_indices.len();
    let mask = (1usize << m) - 1;

    // Commit claimed outputs to transcript
    for &out in &instance.outputs {
        transcript.append_field(b"lasso_out", &out);
    }

    // Squeeze batching challenge ρ
    let rho = transcript.challenge_field::<F>(b"lasso_rho");
    let rho_pows = powers_of(rho, n);

    let mut sumcheck_proofs = Vec::with_capacity(c);
    let mut table_openings  = Vec::with_capacity(c);
    let mut sub_claims      = Vec::with_capacity(c);

    for k in 0..c {
        let t_poly = DenseMLPoly::new(instance.tables[k].clone());

        // Build L_k as a dense MLE
        let size = 1usize << m;
        let mut l_evals = vec![F::ZERO; size];
        for j in 0..n {
            let ch = chunk(instance.query_indices[j], k, m, mask);
            // Add ρ^j · eq(binary(ch), ·) point-wise
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
            .map(|j| {
                let ch = chunk(instance.query_indices[j], k, m, mask);
                rho_pows[j] * instance.tables[k][ch]
            })
            .sum();
        sub_claims.push(claimed);

        let (sc_proof, r_vec) = prove_sumcheck(&t_poly, &l_poly, claimed, transcript);

        // Trivial PCS: reveal T_k(r_vec) directly
        let opening = t_poly.evaluate(&r_vec);
        transcript.append_field(b"lasso_opening", &opening);
        table_openings.push(opening);
        sumcheck_proofs.push(sc_proof);
    }

    LassoProof { sumcheck_proofs, table_openings, sub_claims }
}

pub fn verify_lasso(
    proof: &LassoProof,
    instance: &LassoInstance,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let c    = instance.tables.len();
    let m    = instance.bits_per_chunk;
    let n    = instance.query_indices.len();
    let mask = (1usize << m) - 1;

    for &out in &instance.outputs {
        transcript.append_field(b"lasso_out", &out);
    }
    let rho = transcript.challenge_field::<F>(b"lasso_rho");
    let rho_pows = powers_of(rho, n);

    for k in 0..c {
        // Recompute expected sub-claim from public inputs
        let expected: F = (0..n)
            .map(|j| {
                let ch = chunk(instance.query_indices[j], k, m, mask);
                rho_pows[j] * instance.tables[k][ch]
            })
            .sum();
        if expected != proof.sub_claims[k] {
            return Err(format!("Lasso sub-claim mismatch for table {k}"));
        }

        let (r_vec, _) = verify_sumcheck(&proof.sumcheck_proofs[k], proof.sub_claims[k], m, transcript)
            .map_err(|e| format!("Table {k} sumcheck: {e}"))?;

        // Verify trivial PCS: the verifier recomputes L_k(r_vec) and checks
        // T_k(r_vec) · L_k(r_vec) equals the final sumcheck claim.
        //
        // Note on bit ordering: DenseMLPoly fixes the MSB (highest-bit variable) first
        // in each sumcheck round (because evals[i] for i < half has the top bit = 0).
        // So r_vec[0] corresponds to bit_{m-1}, r_vec[1] to bit_{m-2}, ..., r_vec[m-1]
        // to bit_0. We reverse r_vec so that index_to_bits (LSB-first) pairs correctly.
        let r_rev: Vec<F> = r_vec.iter().copied().rev().collect();
        let l_at_r: F = (0..n)
            .map(|j| {
                let ch   = chunk(instance.query_indices[j], k, m, mask);
                let bits = index_to_bits(ch, m);   // LSB-first: bits[i] = bit_i(ch)
                rho_pows[j] * eq_eval(&bits, &r_rev) // r_rev[i] = r_vec[m-1-i] = challenge for bit_i
            })
            .sum();

        let t_opening = proof.table_openings[k];
        transcript.append_field(b"lasso_opening", &t_opening);

        let expected_final = t_opening * l_at_r;
        let actual_final   = proof.sumcheck_proofs[k].final_eval_f
            * proof.sumcheck_proofs[k].final_eval_g;
        if expected_final != actual_final {
            return Err(format!(
                "Table {k} PCS check failed: T(r)*L(r)={expected_final:?} ≠ final={actual_final:?}"
            ));
        }
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
