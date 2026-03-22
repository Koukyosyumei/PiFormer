//! Arithmetic circuit for one head of linear attention (MLE/GKR based).
//!
//! **Computations proved:**
//!   1. phiQ = φ(Q)                   — via Lasso lookup (per token × feature)
//!   2. phiK = φ(K)                   — via Lasso lookup
//!   3. context[i][j] = Σ_t phiK[t][i]·V[t][j]   — matrix multiply via sumcheck
//!   4. out[t][j]     = Σ_i phiQ[t][i]·context[i][j]  — matrix-vector via sumcheck
//!
//! Steps 3 & 4 are proved using Thaler '13 matrix multiplication reduction.
//! Instead of picking a specific integer index, the verifier evaluates the
//! multilinear extension (MLE) of the matrices at a random point r ∈ F^n.

use crate::field::F;
use crate::lookup::{prove_lasso, verify_lasso, LassoInstance, LassoProof};
use crate::pcs::HyraxParams;
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;

/// Full witness for proving one attention head.
pub struct LinearAttentionInstance {
    pub seq_len: usize,
    pub d_head: usize,
    /// Q, K, V matrices as field elements: shape (seq_len, d_head).
    pub q: Vec<Vec<F>>,
    pub k: Vec<Vec<F>>,
    pub v: Vec<Vec<F>>,
    /// φ(Q) and φ(K): same shape.
    pub phi_q: Vec<Vec<F>>,
    pub phi_k: Vec<Vec<F>>,
    /// context = φ(K)^T · V: shape (d_head, d_head).
    pub context: Vec<Vec<F>>,
    /// out = φ(Q) · context: shape (seq_len, d_head).
    pub out: Vec<Vec<F>>,
    /// Lasso instances for the φ lookups.
    pub q_lasso: LassoInstance,
    pub k_lasso: LassoInstance,
}

#[derive(Clone, Debug)]
pub struct AttentionEvals {
    pub phi_q_at_r: F,
    pub phi_k_at_r: F,
    pub v_at_r: F,
    pub ctx_at_r: F,
}

/// Proof for one attention head.
pub struct LinearAttentionProof {
    pub phi_q_proof: LassoProof,
    pub phi_k_proof: LassoProof,
    /// Sumcheck over D (head-dim) for the output matrix MLE.
    pub out_sumcheck: SumcheckProof,
    /// Sumcheck over T (seq dimension) for the context matrix MLE.
    pub context_sumcheck: SumcheckProof,
    /// Final evaluations of the multilinear extensions at the sumcheck random points.
    pub final_evals: AttentionEvals,
}

pub fn prove_linear_attention(
    inst: &LinearAttentionInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> LinearAttentionProof {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;

    // Step 1 & 2: Lasso for φ(Q) and φ(K)
    let phi_q_proof = prove_lasso(&inst.q_lasso, transcript, params);
    let phi_k_proof = prove_lasso(&inst.k_lasso, transcript, params);

    // Verifier challenges for the output matrix MLE (rx ∈ F^t_bits, ry ∈ F^d_bits)
    let rx_out = generate_challenge_vector(transcript, t_bits, b"rx_out");
    let ry_out = generate_challenge_vector(transcript, d_bits, b"ry_out");

    // Prover claims the evaluation of OUT at (rx_out, ry_out)
    let claimed_out = eval_2d(&inst.out, &rx_out, &ry_out, t, d);
    transcript.append_field(b"claimed_out", &claimed_out);

    // Step 3: Sumcheck for OUT = φ(Q) · context
    // We prove: eval_2d(OUT, rx, ry) = Σ_i phi_q(rx, i) * context(i, ry)
    let f_out_vec = eval_cols(&inst.phi_q, &rx_out, t, d); // f_out[i] = phi_q(rx, i)
    let g_out_vec = eval_rows(&inst.context, &ry_out, d, d); // g_out[i] = context(i, ry)

    let f_out = DenseMLPoly::from_vec_padded(f_out_vec);
    let g_out = DenseMLPoly::from_vec_padded(g_out_vec);

    let (out_sumcheck, r_inner_i) = prove_sumcheck(&f_out, &g_out, claimed_out, transcript);

    let phi_q_at_r = out_sumcheck.final_eval_f;
    let ctx_at_r = out_sumcheck.final_eval_g;

    // Step 4: Sumcheck for CONTEXT = φ(K)^T · V
    // We must prove the evaluation of Context at (r_inner_i, ry_out)
    // ctx_at_r = eval_2d(context, r_inner_i, ry_out) = Σ_t phi_k(t, r_inner_i) * V(t, ry_out)
    let f_ctx_vec = eval_rows(&inst.phi_k, &r_inner_i, t, d); // f_ctx[t] = phi_k(t, r_inner_i)
    let g_ctx_vec = eval_rows(&inst.v, &ry_out, t, d); // g_ctx[t] = V(t, ry_out)

    let f_ctx = DenseMLPoly::from_vec_padded(f_ctx_vec);
    let g_ctx = DenseMLPoly::from_vec_padded(g_ctx_vec);

    transcript.append_field(b"claimed_ctx", &ctx_at_r);
    let (context_sumcheck, _r_inner_t) = prove_sumcheck(&f_ctx, &g_ctx, ctx_at_r, transcript);

    let phi_k_at_r = context_sumcheck.final_eval_f;
    let v_at_r = context_sumcheck.final_eval_g;

    LinearAttentionProof {
        phi_q_proof,
        phi_k_proof,
        out_sumcheck,
        context_sumcheck,
        final_evals: AttentionEvals {
            phi_q_at_r,
            phi_k_at_r,
            v_at_r,
            ctx_at_r,
        },
    }
}

pub fn verify_linear_attention(
    proof: &LinearAttentionProof,
    inst: &LinearAttentionInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;

    verify_lasso(&proof.phi_q_proof, &inst.q_lasso, transcript, params)
        .map_err(|e| format!("phi_q: {e}"))?;
    verify_lasso(&proof.phi_k_proof, &inst.k_lasso, transcript, params)
        .map_err(|e| format!("phi_k: {e}"))?;

    // Replicate verifier challenges for OUT
    let rx_out = generate_challenge_vector(transcript, t_bits, b"rx_out");
    let ry_out = generate_challenge_vector(transcript, d_bits, b"ry_out");

    // Retrieve the claimed output value. In a full system, this OUT matrix is
    // public or committed. We evaluate the public instance here.
    let claimed_out = eval_2d(&inst.out, &rx_out, &ry_out, t, d);
    transcript.append_field(b"claimed_out", &claimed_out);

    // Verify OUT sumcheck
    let (r_inner_i, final_claim_out) =
        verify_sumcheck(&proof.out_sumcheck, claimed_out, d_bits, transcript)
            .map_err(|e| format!("out sumcheck: {e}"))?;

    let expected_final_out = proof.final_evals.phi_q_at_r * proof.final_evals.ctx_at_r;
    if final_claim_out != expected_final_out {
        return Err("Out sumcheck final evaluations do not match".to_string());
    }

    // Verify CONTEXT sumcheck
    transcript.append_field(b"claimed_ctx", &proof.final_evals.ctx_at_r);
    let (r_inner_t, final_claim_ctx) = verify_sumcheck(
        &proof.context_sumcheck,
        proof.final_evals.ctx_at_r,
        t_bits,
        transcript,
    )
    .map_err(|e| format!("context sumcheck: {e}"))?;

    let expected_final_ctx = proof.final_evals.phi_k_at_r * proof.final_evals.v_at_r;
    if final_claim_ctx != expected_final_ctx {
        return Err("Context sumcheck final evaluations do not match".to_string());
    }

    // Evaluate base matrices at the final random points to complete the GKR proof.
    // In a fully succinct protocol, these would be PCS openings.
    let actual_phi_q = eval_2d(&inst.phi_q, &rx_out, &r_inner_i, t, d);
    if actual_phi_q != proof.final_evals.phi_q_at_r {
        return Err("phi_q eval mismatch".to_string());
    }

    let actual_phi_k = eval_2d(&inst.phi_k, &r_inner_t, &r_inner_i, t, d);
    if actual_phi_k != proof.final_evals.phi_k_at_r {
        return Err("phi_k eval mismatch".to_string());
    }

    let actual_v = eval_2d(&inst.v, &r_inner_t, &ry_out, t, d);
    if actual_v != proof.final_evals.v_at_r {
        return Err("v eval mismatch".to_string());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers for 2D Multilinear Polynomial Evaluation
// ---------------------------------------------------------------------------

fn generate_challenge_vector(transcript: &mut Transcript, len: usize, label: &[u8]) -> Vec<F> {
    (0..len)
        .map(|_| transcript.challenge_field::<F>(label))
        .collect()
}

/// Evaluates the columns of a 2D matrix at a fixed row challenge.
/// Returns a 1D vector of length `cols`.
fn eval_cols(matrix: &[Vec<F>], r_row: &[F], rows: usize, cols: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(cols);
    for c in 0..cols {
        let mut col_vec = Vec::with_capacity(rows);
        for r in 0..rows {
            col_vec.push(matrix[r][c]);
        }
        let poly = DenseMLPoly::from_vec_padded(col_vec);
        res.push(poly.evaluate(r_row));
    }
    res
}

/// Evaluates the rows of a 2D matrix at a fixed column challenge.
/// Returns a 1D vector of length `rows`.
fn eval_rows(matrix: &[Vec<F>], r_col: &[F], rows: usize, _cols: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(rows);
    for r in 0..rows {
        let poly = DenseMLPoly::from_vec_padded(matrix[r].clone());
        res.push(poly.evaluate(r_col));
    }
    res
}

/// Fully evaluates a 2D matrix MLE at the given point (r_row, r_col).
fn eval_2d(matrix: &[Vec<F>], r_row: &[F], r_col: &[F], rows: usize, cols: usize) -> F {
    let col_evals = eval_cols(matrix, r_row, rows, cols); // Collapses the rows
    let poly = DenseMLPoly::from_vec_padded(col_evals); // Creates polynomial over the columns
    poly.evaluate(r_col) // Collapses the columns
}

#[cfg(test)]
mod linear_attention_tests {
    use super::*;
    use crate::pcs::HyraxParams;
    use crate::transcript::Transcript;
    use ark_ff::PrimeField;
    use ark_ff::{One, Zero};

    /// Helper function to generate a small-scale test instance
    fn setup_test_instance(seq_len: usize, d_head: usize) -> LinearAttentionInstance {
        let m = 4; // 4-bit lookup table (size 16)
        let table_size = 1 << m;

        // 1. Definition of phi(x): For simplicity, let phi(x) = x + 1
        let table: Vec<F> = (0..table_size).map(|i| F::from((i + 1) as u64)).collect();

        // 2. Generate Q, K, V matrices (T x D)
        let q = vec![vec![F::from(1), F::from(2)], vec![F::from(3), F::from(4)]];
        let k = vec![vec![F::from(0), F::from(1)], vec![F::from(2), F::from(3)]];
        let v = vec![vec![F::from(5), F::from(6)], vec![F::from(7), F::from(8)]];

        // 3. Compute phi(Q) and phi(K)
        let phi_q: Vec<Vec<F>> = q
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| table[x.into_bigint().as_ref()[0] as usize])
                    .collect()
            })
            .collect();

        let phi_k: Vec<Vec<F>> = k
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| table[x.into_bigint().as_ref()[0] as usize])
                    .collect()
            })
            .collect();

        // 4. Compute context = phi(K)^T * V (D x D)
        let mut context = vec![vec![F::zero(); d_head]; d_head];
        for i in 0..d_head {
            for j in 0..d_head {
                for t in 0..seq_len {
                    context[i][j] += phi_k[t][i] * v[t][j];
                }
            }
        }

        // 5. Compute out = phi(Q) * context (T x D)
        let mut out = vec![vec![F::zero(); d_head]; seq_len];
        for t in 0..seq_len {
            for j in 0..d_head {
                for i in 0..d_head {
                    out[t][j] += phi_q[t][i] * context[i][j];
                }
            }
        }

        // 6. Setup Lasso Instances
        let q_flat: Vec<usize> = q
            .iter()
            .flatten()
            .map(|x| x.into_bigint().as_ref()[0] as usize)
            .collect();
        let q_out_flat: Vec<F> = phi_q.iter().flatten().copied().collect();

        let k_flat: Vec<usize> = k
            .iter()
            .flatten()
            .map(|x| x.into_bigint().as_ref()[0] as usize)
            .collect();
        let k_out_flat: Vec<F> = phi_k.iter().flatten().copied().collect();

        let q_lasso = LassoInstance {
            tables: vec![table.clone()],
            query_indices: q_flat,
            outputs: q_out_flat,
            bits_per_chunk: m,
        };

        let k_lasso = LassoInstance {
            tables: vec![table],
            query_indices: k_flat,
            outputs: k_out_flat,
            bits_per_chunk: m,
        };

        LinearAttentionInstance {
            seq_len,
            d_head,
            q,
            k,
            v,
            phi_q,
            phi_k,
            context,
            out,
            q_lasso,
            k_lasso,
        }
    }

    #[test]
    fn test_linear_attention_e2e_success() {
        let t = 2;
        let d = 2;
        let inst = setup_test_instance(t, d);
        let params = HyraxParams::new(2);

        let mut prover_transcript = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&inst, &mut prover_transcript, &params);

        let mut verifier_transcript = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &mut verifier_transcript, &params);

        assert!(
            result.is_ok(),
            "Linear Attention verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_rejects_tampered_context_matrix() {
        let mut inst = setup_test_instance(2, 2);
        let params = HyraxParams::new(2);

        // A malicious prover tampers with a single element of the context matrix
        inst.context[0][0] += F::one();

        let mut prover_transcript = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&inst, &mut prover_transcript, &params);

        let mut verifier_transcript = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &mut verifier_transcript, &params);

        assert!(result.is_err());
        let err_msg = result.unwrap_err();

        // Print it so you can see exactly where the verifier caught the lie!
        println!("Caught tampered context: {}", err_msg);

        // The corrupted context matrix will cause the OUT sumcheck to fail,
        // typically at Round 0 because the actual sum no longer matches the claim.
        assert!(
            err_msg.contains("out sumcheck") || err_msg.contains("context sumcheck"),
            "Expected a sumcheck error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_rejects_tampered_phi_q_lasso() {
        let mut inst = setup_test_instance(2, 2);
        let params = HyraxParams::new(2);

        // Tamper with the claimed outputs of phi(Q)
        inst.q_lasso.outputs[0] += F::one();

        let mut pt = Transcript::new(b"test");
        let proof = prove_linear_attention(&inst, &mut pt, &params);
        let mut vt = Transcript::new(b"test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &params);

        assert!(result.is_err());
        // Must specifically trigger the phi_q lasso failure
        assert!(result.unwrap_err().contains("phi_q:"));
    }

    #[test]
    fn test_rejects_tampered_phi_k_lasso() {
        let mut inst = setup_test_instance(2, 2);
        let params = HyraxParams::new(2);

        // Tamper with the claimed outputs of phi(K)
        inst.k_lasso.outputs[0] += F::one();

        let mut pt = Transcript::new(b"test");
        let proof = prove_linear_attention(&inst, &mut pt, &params);
        let mut vt = Transcript::new(b"test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &params);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("phi_k:"));
    }

    #[test]
    fn test_rejects_tampered_out_matrix() {
        let mut inst = setup_test_instance(2, 2);
        let params = HyraxParams::new(2);

        // Tamper with a single element in the final output matrix
        inst.out[1][1] += F::one();

        let mut prover_transcript = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&inst, &mut prover_transcript, &params);

        let mut verifier_transcript = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &mut verifier_transcript, &params);

        assert!(result.is_err());
        // The initial claim (claimed_out) for OUT will be incorrect, breaking the
        // sumcheck consistency.
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("Out sumcheck final evaluations do not match")
                || err_msg.contains("out sumcheck")
        );
    }

    #[test]
    fn test_rejects_tampered_final_evals() {
        let inst = setup_test_instance(2, 2);
        let params = HyraxParams::new(2);

        let mut prover_transcript = Transcript::new(b"linear_attn_test");
        let mut proof = prove_linear_attention(&inst, &mut prover_transcript, &params);

        // Suppose the prover performs the sumcheck correctly but lies about the
        // "evaluation of the original matrix" at the very end.
        proof.final_evals.phi_q_at_r += F::one();

        let mut verifier_transcript = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &mut verifier_transcript, &params);

        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        // The mismatch will be detected either because the product of the faked
        // final_evals doesn't match the sumcheck's internal result, or because
        // the MLE re-evaluation exposes the lie.
        assert!(
            err_msg.contains("Out sumcheck final evaluations do not match")
                || err_msg.contains("phi_q eval mismatch")
        );
    }

    #[test]
    fn test_rejects_tampered_ctx_at_r() {
        let inst = setup_test_instance(2, 2);
        let params = HyraxParams::new(2);
        let mut pt = Transcript::new(b"test");
        let mut proof = prove_linear_attention(&inst, &mut pt, &params);

        // Tamper with the context's evaluation point claimed by the prover
        proof.final_evals.ctx_at_r += F::one();

        let mut vt = Transcript::new(b"test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &params);

        assert!(result.is_err());
        let err = result.unwrap_err();
        // Changing ctx_at_r breaks the OUT expected final check, OR the context sumcheck initial claim
        assert!(
            err.contains("Out sumcheck final evaluations do not match")
                || err.contains("context sumcheck:")
        );
    }

    #[test]
    fn test_rejects_tampered_v_at_r() {
        let inst = setup_test_instance(2, 2);
        let params = HyraxParams::new(2);
        let mut pt = Transcript::new(b"test");
        let mut proof = prove_linear_attention(&inst, &mut pt, &params);

        // Prover lies about the evaluation of V
        proof.final_evals.v_at_r += F::one();

        let mut vt = Transcript::new(b"test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &params);

        assert!(result.is_err());
        let err = result.unwrap_err();
        // Will break the CONTEXT final check or the v eval mismatch check
        assert!(
            err.contains("Context sumcheck final evaluations do not match")
                || err.contains("v eval mismatch")
        );
    }

    #[test]
    fn test_rejects_mismatched_v_matrix() {
        let mut inst = setup_test_instance(2, 2);
        let params = HyraxParams::new(2);

        let mut pt = Transcript::new(b"test");
        let proof = prove_linear_attention(&inst, &mut pt, &params);

        // After proof is generated, alter the public V matrix the verifier holds
        inst.v[0][1] += F::one();

        let mut vt = Transcript::new(b"test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &params);

        assert!(result.is_err());
        // The proof was for the old V. The MLE evaluation of the new V will fail.
        assert!(result.unwrap_err().contains("v eval mismatch"));
    }

    #[test]
    fn test_rejects_mismatched_phi_q_matrix() {
        let mut inst = setup_test_instance(2, 2);
        let params = HyraxParams::new(2);

        let mut pt = Transcript::new(b"test");
        let proof = prove_linear_attention(&inst, &mut pt, &params);

        inst.phi_q[1][0] += F::one();

        let mut vt = Transcript::new(b"test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &params);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("phi_q eval mismatch"));
    }

    #[test]
    fn test_rejects_mismatched_phi_k_matrix() {
        let mut inst = setup_test_instance(2, 2);
        let params = HyraxParams::new(2);

        let mut pt = Transcript::new(b"test");
        let proof = prove_linear_attention(&inst, &mut pt, &params);

        inst.phi_k[0][0] += F::one();

        let mut vt = Transcript::new(b"test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &params);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("phi_k eval mismatch"));
    }
}
