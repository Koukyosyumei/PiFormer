//! Arithmetic circuit for one head of linear attention (MLE/GKR + PCS based).
//!
//! **Computation proved:**
//!   1. phiQ = φ(Q)                                 — Lasso lookup
//!   2. phiK = φ(K)                                 — Lasso lookup
//!   3. context[i][j] = Σ_t phiK[t][i]·V[t][j]    — GKR sumcheck over t
//!   4. out[t][j]     = Σ_i phiQ[t][i]·context[i][j] — GKR sumcheck over i
//!
//! **Succinctness:**
//!   * All intermediate matrices (phiQ, phiK, V, context, out) are committed
//!     with Hyrax *before* any challenge is drawn, binding the prover.
//!   * Two GKR sumchecks reduce the 2-D matrix-multiply claims to scalar
//!     MLE evaluations at random points.
//!   * Five Hyrax opening proofs certify those scalar claims in O(√N) work.
//!   * Verifier recomputes commitments from its copy of the instance and checks
//!     they match the proof, ensuring the right polynomials were committed.

use crate::field::F;
use crate::lookup::lasso::{prove_lasso, verify_lasso, LassoInstance, LassoProof};
use crate::pcs::{
    hyrax_commit, hyrax_open, hyrax_verify, HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::Field;
use ark_serialize::CanonicalSerialize;

// ---------------------------------------------------------------------------
// Public instance (raw matrices — polys are built internally)
// ---------------------------------------------------------------------------

/// Witness for one linear-attention head.
///
/// `context` has shape d_head × d_head;
/// all other 2-D matrices have shape seq_len × d_head.
pub struct LinearAttentionInstance {
    pub seq_len: usize,
    pub d_head: usize,
    pub q: Vec<Vec<F>>,
    pub k: Vec<Vec<F>>,
    pub v: Vec<Vec<F>>,
    pub phi_q: Vec<Vec<F>>,
    pub phi_k: Vec<Vec<F>>,
    pub context: Vec<Vec<F>>,
    pub out: Vec<Vec<F>>,
    pub q_lasso: LassoInstance,
    pub k_lasso: LassoInstance,
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

pub struct AttentionCommitments {
    pub phi_q_com: HyraxCommitment,
    pub phi_k_com: HyraxCommitment,
    pub v_com: HyraxCommitment,
    pub context_com: HyraxCommitment,
    pub out_com: HyraxCommitment,
}

pub struct AttentionOpenings {
    /// out(rx, ry) — used as the OUT sumcheck claim.
    pub out_eval: F,
    pub out_open: HyraxProof,
    /// phiQ(rx, r_i) — final eval_f of the OUT sumcheck.
    pub phi_q_eval: F,
    pub phi_q_open: HyraxProof,
    /// context(r_i, ry) — final eval_g of the OUT sumcheck / claim for context sumcheck.
    pub ctx_eval: F,
    pub ctx_open: HyraxProof,
    /// phiK(r_t, r_i) — final eval_f of the context sumcheck.
    pub phi_k_eval: F,
    pub phi_k_open: HyraxProof,
    /// V(r_t, ry) — final eval_g of the context sumcheck.
    pub v_eval: F,
    pub v_open: HyraxProof,
}

pub struct LinearAttentionProof {
    pub commitments: AttentionCommitments,
    pub phi_q_lasso: LassoProof,
    pub phi_k_lasso: LassoProof,
    pub out_sumcheck: SumcheckProof,
    pub context_sumcheck: SumcheckProof,
    pub openings: AttentionOpenings,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn prove_linear_attention(
    inst: &LinearAttentionInstance,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> LinearAttentionProof {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;

    // Build MLE polynomials from raw matrices.
    let phi_q_mle = mat_to_mle(&inst.phi_q, t, d);
    let phi_k_mle = mat_to_mle(&inst.phi_k, t, d);
    let v_mle = mat_to_mle(&inst.v, t, d);
    let ctx_mle = mat_to_mle(&inst.context, d, d); // d × d
    let out_mle = mat_to_mle(&inst.out, t, d);

    // Hyrax params: one set for T×D polys, one for D×D context.
    let (nu_td, sigma_td, params_td) = poly_hyrax(&phi_q_mle);
    let (nu_dd, sigma_dd, params_dd) = poly_hyrax(&ctx_mle);

    // ── 1. Commit to all intermediate matrices ────────────────────────────────
    let phi_q_com = hyrax_commit(&phi_q_mle.evaluations, nu_td, &params_td);
    let phi_k_com = hyrax_commit(&phi_k_mle.evaluations, nu_td, &params_td);
    let v_com = hyrax_commit(&v_mle.evaluations, nu_td, &params_td);
    let context_com = hyrax_commit(&ctx_mle.evaluations, nu_dd, &params_dd);
    let out_com = hyrax_commit(&out_mle.evaluations, nu_td, &params_td);

    absorb_com(transcript, b"phi_q_com", &phi_q_com);
    absorb_com(transcript, b"phi_k_com", &phi_k_com);
    absorb_com(transcript, b"v_com", &v_com);
    absorb_com(transcript, b"context_com", &context_com);
    absorb_com(transcript, b"out_com", &out_com);

    // ── 2. Lasso: prove phiQ = φ(Q) and phiK = φ(K) ─────────────────────────
    let phi_q_lasso = prove_lasso(&inst.q_lasso, transcript, lasso_params);
    let phi_k_lasso = prove_lasso(&inst.k_lasso, transcript, lasso_params);

    // ── 3. Sumcheck for OUT = phiQ · context ─────────────────────────────────
    let rx = challenge_vec(transcript, t_bits, b"rx_out");
    let ry = challenge_vec(transcript, d_bits, b"ry_out");

    // Claim: out(rx, ry).
    let out_eval = out_mle.evaluate(&combine(&rx, &ry));
    transcript.append_field(b"claimed_out", &out_eval);

    // f[i] = phiQ(rx, ·), fixed over the row dimension.
    let f_out = DenseMLPoly::from_vec_padded(eval_rows(&phi_q_mle, t_bits, &rx));
    // g[i] = context(·, ry), fixed over the column dimension.
    let g_out = DenseMLPoly::from_vec_padded(eval_cols(&ctx_mle, d_bits, &ry));

    let (out_sumcheck, r_i) = prove_sumcheck(&f_out, &g_out, out_eval, transcript);

    // ── 4. Sumcheck for context = phiK^T · V ─────────────────────────────────
    // Consistency bridge: context(r_i, ry) is what the OUT sumcheck reduced to.
    let ctx_eval = out_sumcheck.final_eval_g;
    transcript.append_field(b"claimed_ctx", &ctx_eval);

    // f[t] = phiK(·, r_i), fixed over the column dimension.
    let f_ctx = DenseMLPoly::from_vec_padded(eval_cols(&phi_k_mle, t_bits, &r_i));
    // g[t] = V(·, ry), fixed over the column dimension.
    let g_ctx = DenseMLPoly::from_vec_padded(eval_cols(&v_mle, t_bits, &ry));

    let (context_sumcheck, r_t) = prove_sumcheck(&f_ctx, &g_ctx, ctx_eval, transcript);

    // ── 5. PCS opening proofs ─────────────────────────────────────────────────
    // out at (rx, ry)
    let out_open = hyrax_open(&out_mle.evaluations, &combine(&rx, &ry), nu_td, sigma_td);

    // phiQ at (rx, r_i)
    let phi_q_eval = out_sumcheck.final_eval_f;
    let phi_q_open = hyrax_open(&phi_q_mle.evaluations, &combine(&rx, &r_i), nu_td, sigma_td);

    // context at (r_i, ry)
    let ctx_open = hyrax_open(&ctx_mle.evaluations, &combine(&r_i, &ry), nu_dd, sigma_dd);

    // phiK at (r_t, r_i)
    let phi_k_eval = context_sumcheck.final_eval_f;
    let phi_k_open = hyrax_open(&phi_k_mle.evaluations, &combine(&r_t, &r_i), nu_td, sigma_td);

    // V at (r_t, ry)
    let v_eval = context_sumcheck.final_eval_g;
    let v_open = hyrax_open(&v_mle.evaluations, &combine(&r_t, &ry), nu_td, sigma_td);

    LinearAttentionProof {
        commitments: AttentionCommitments {
            phi_q_com,
            phi_k_com,
            v_com,
            context_com,
            out_com,
        },
        phi_q_lasso,
        phi_k_lasso,
        out_sumcheck,
        context_sumcheck,
        openings: AttentionOpenings {
            out_eval,
            out_open,
            phi_q_eval,
            phi_q_open,
            ctx_eval,
            ctx_open,
            phi_k_eval,
            phi_k_open,
            v_eval,
            v_open,
        },
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn verify_linear_attention(
    proof: &LinearAttentionProof,
    inst: &LinearAttentionInstance,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> Result<(), String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;

    // Rebuild MLEs from the instance (verifier's copy) and derive params.
    let phi_q_mle = mat_to_mle(&inst.phi_q, t, d);
    let phi_k_mle = mat_to_mle(&inst.phi_k, t, d);
    let v_mle = mat_to_mle(&inst.v, t, d);
    let ctx_mle = mat_to_mle(&inst.context, d, d);
    let out_mle = mat_to_mle(&inst.out, t, d);

    let (nu_td, _sigma_td, params_td) = poly_hyrax(&phi_q_mle);
    let (nu_dd, _sigma_dd, params_dd) = poly_hyrax(&ctx_mle);

    // ── 1. Check commitments against instance data ────────────────────────────
    let expected_phi_q = hyrax_commit(&phi_q_mle.evaluations, nu_td, &params_td);
    let expected_phi_k = hyrax_commit(&phi_k_mle.evaluations, nu_td, &params_td);
    let expected_v = hyrax_commit(&v_mle.evaluations, nu_td, &params_td);
    let expected_ctx = hyrax_commit(&ctx_mle.evaluations, nu_dd, &params_dd);
    let expected_out = hyrax_commit(&out_mle.evaluations, nu_td, &params_td);

    if expected_phi_q.row_coms != proof.commitments.phi_q_com.row_coms {
        return Err("phi_q commitment mismatch".to_string());
    }
    if expected_phi_k.row_coms != proof.commitments.phi_k_com.row_coms {
        return Err("phi_k commitment mismatch".to_string());
    }
    if expected_v.row_coms != proof.commitments.v_com.row_coms {
        return Err("V commitment mismatch".to_string());
    }
    if expected_ctx.row_coms != proof.commitments.context_com.row_coms {
        return Err("context commitment mismatch".to_string());
    }
    if expected_out.row_coms != proof.commitments.out_com.row_coms {
        return Err("out commitment mismatch".to_string());
    }

    // Replay transcript absorptions (Fiat-Shamir binding).
    absorb_com(transcript, b"phi_q_com", &proof.commitments.phi_q_com);
    absorb_com(transcript, b"phi_k_com", &proof.commitments.phi_k_com);
    absorb_com(transcript, b"v_com", &proof.commitments.v_com);
    absorb_com(transcript, b"context_com", &proof.commitments.context_com);
    absorb_com(transcript, b"out_com", &proof.commitments.out_com);

    // ── 2. Lasso verification ─────────────────────────────────────────────────
    verify_lasso(&proof.phi_q_lasso, &inst.q_lasso, transcript, lasso_params)
        .map_err(|e| format!("phi_q: {e}"))?;
    verify_lasso(&proof.phi_k_lasso, &inst.k_lasso, transcript, lasso_params)
        .map_err(|e| format!("phi_k: {e}"))?;

    // ── 3. Replay challenges ──────────────────────────────────────────────────
    let rx = challenge_vec(transcript, t_bits, b"rx_out");
    let ry = challenge_vec(transcript, d_bits, b"ry_out");

    // ── 4. OUT sumcheck ───────────────────────────────────────────────────────
    let claimed_out = proof.openings.out_eval;
    transcript.append_field(b"claimed_out", &claimed_out);

    let (r_i, final_out) =
        verify_sumcheck(&proof.out_sumcheck, claimed_out, d_bits, transcript)
            .map_err(|e| format!("out sumcheck: {e}"))?;

    let expected_out_final = proof.openings.phi_q_eval * proof.openings.ctx_eval;
    if final_out != expected_out_final {
        return Err("Out sumcheck final evaluations do not match".to_string());
    }

    // ── 5. Context sumcheck ───────────────────────────────────────────────────
    let claimed_ctx = proof.openings.ctx_eval;
    transcript.append_field(b"claimed_ctx", &claimed_ctx);

    let (r_t, final_ctx) =
        verify_sumcheck(&proof.context_sumcheck, claimed_ctx, t_bits, transcript)
            .map_err(|e| format!("context sumcheck: {e}"))?;

    let expected_ctx_final = proof.openings.phi_k_eval * proof.openings.v_eval;
    if final_ctx != expected_ctx_final {
        return Err("Context sumcheck final evaluations do not match".to_string());
    }

    // ── 6. Hyrax opening proofs ───────────────────────────────────────────────
    // out at (rx, ry)
    hyrax_verify(
        &proof.commitments.out_com,
        proof.openings.out_eval,
        &combine(&rx, &ry),
        &proof.openings.out_open,
        &params_td,
    )
    .map_err(|e| format!("out opening: {e}"))?;

    // phiQ at (rx, r_i)
    hyrax_verify(
        &proof.commitments.phi_q_com,
        proof.openings.phi_q_eval,
        &combine(&rx, &r_i),
        &proof.openings.phi_q_open,
        &params_td,
    )
    .map_err(|e| format!("phi_q eval mismatch: {e}"))?;

    // context at (r_i, ry)
    hyrax_verify(
        &proof.commitments.context_com,
        proof.openings.ctx_eval,
        &combine(&r_i, &ry),
        &proof.openings.ctx_open,
        &params_dd,
    )
    .map_err(|e| format!("context opening: {e}"))?;

    // phiK at (r_t, r_i)
    hyrax_verify(
        &proof.commitments.phi_k_com,
        proof.openings.phi_k_eval,
        &combine(&r_t, &r_i),
        &proof.openings.phi_k_open,
        &params_td,
    )
    .map_err(|e| format!("phi_k eval mismatch: {e}"))?;

    // V at (r_t, ry)
    hyrax_verify(
        &proof.commitments.v_com,
        proof.openings.v_eval,
        &combine(&r_t, &ry),
        &proof.openings.v_open,
        &params_td,
    )
    .map_err(|e| format!("v eval mismatch: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a 2-D MLE from a raw matrix stored as `rows × cols` Vecs.
/// Padded to (row_p2 × col_p2) evaluations in row-major order (row bits = MSB).
fn mat_to_mle(mat: &[Vec<F>], rows: usize, cols: usize) -> DenseMLPoly {
    let r_p2 = rows.next_power_of_two().max(1);
    let c_p2 = cols.next_power_of_two().max(1);
    let mut evals = vec![F::ZERO; r_p2 * c_p2];
    for (i, row) in mat.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            evals[i * c_p2 + j] = v;
        }
    }
    DenseMLPoly::new(evals)
}

/// Determine Hyrax params for a poly: nu = num_vars/2, sigma = num_vars−nu (≥1).
fn poly_hyrax(poly: &DenseMLPoly) -> (usize, usize, HyraxParams) {
    let total = poly.num_vars;
    let nu = total / 2;
    let sigma = (total - nu).max(1);
    (nu, sigma, HyraxParams::new(sigma))
}

/// Fix the first `n_row_vars` (MSB) variables of `poly` at `r_row`.
/// Returns the remaining evaluations (a 1-D Vec over column variables).
fn eval_rows(poly: &DenseMLPoly, n_row_vars: usize, r_row: &[F]) -> Vec<F> {
    assert_eq!(r_row.len(), n_row_vars);
    let mut p = poly.clone();
    for &r in r_row {
        p = p.fix_first_variable(r);
    }
    p.evaluations
}

/// Fix the last `n_col_vars` (LSB) variables of `poly` at `r_col`.
/// Returns a Vec of length `2^n_row_vars` (one evaluation per row).
fn eval_cols(poly: &DenseMLPoly, n_row_vars: usize, r_col: &[F]) -> Vec<F> {
    let n_p2_rows = 1 << n_row_vars;
    let n_p2_cols = poly.evaluations.len() / n_p2_rows;
    (0..n_p2_rows)
        .map(|i| {
            let row = poly.evaluations[i * n_p2_cols..(i + 1) * n_p2_cols].to_vec();
            DenseMLPoly::new(row).evaluate(r_col)
        })
        .collect()
}

fn combine(a: &[F], b: &[F]) -> Vec<F> {
    let mut res = a.to_vec();
    res.extend_from_slice(b);
    res
}

fn challenge_vec(transcript: &mut Transcript, len: usize, label: &[u8]) -> Vec<F> {
    (0..len).map(|_| transcript.challenge_field::<F>(label)).collect()
}

fn absorb_com(transcript: &mut Transcript, label: &[u8], com: &HyraxCommitment) {
    for pt in &com.row_coms {
        let mut buf = Vec::new();
        pt.serialize_compressed(&mut buf).unwrap();
        transcript.append_bytes(label, &buf);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod linear_attention_tests {
    use super::*;
    use crate::pcs::HyraxParams;
    use crate::transcript::Transcript;
    use ark_ff::{One, PrimeField, Zero};

    fn setup_test_instance(seq_len: usize, d_head: usize) -> LinearAttentionInstance {
        let m = 4usize;
        let table_size = 1 << m;
        let table: Vec<F> = (0..table_size).map(|i| F::from((i + 1) as u64)).collect();

        let q = vec![vec![F::from(1u64), F::from(2u64)], vec![F::from(3u64), F::from(4u64)]];
        let k = vec![vec![F::from(0u64), F::from(1u64)], vec![F::from(2u64), F::from(3u64)]];
        let v = vec![vec![F::from(5u64), F::from(6u64)], vec![F::from(7u64), F::from(8u64)]];

        let apply_phi = |mat: &Vec<Vec<F>>| -> Vec<Vec<F>> {
            mat.iter()
                .map(|row| row.iter().map(|&x| table[x.into_bigint().as_ref()[0] as usize]).collect())
                .collect()
        };
        let phi_q = apply_phi(&q);
        let phi_k = apply_phi(&k);

        let mut context = vec![vec![F::zero(); d_head]; d_head];
        for i in 0..d_head {
            for j in 0..d_head {
                for t in 0..seq_len {
                    context[i][j] += phi_k[t][i] * v[t][j];
                }
            }
        }
        let mut out = vec![vec![F::zero(); d_head]; seq_len];
        for t in 0..seq_len {
            for j in 0..d_head {
                for i in 0..d_head {
                    out[t][j] += phi_q[t][i] * context[i][j];
                }
            }
        }

        let build_lasso = |mat: &Vec<Vec<F>>, phi: &Vec<Vec<F>>| -> LassoInstance {
            let indices: Vec<usize> = mat.iter().flatten()
                .map(|x| x.into_bigint().as_ref()[0] as usize).collect();
            let outputs: Vec<F> = phi.iter().flatten().copied().collect();
            LassoInstance { tables: vec![table.clone()], query_indices: indices, outputs, bits_per_chunk: m }
        };

        LinearAttentionInstance {
            seq_len, d_head,
            q, k, v, phi_q, phi_k, context, out,
            q_lasso: build_lasso(&vec![vec![F::from(1u64), F::from(2u64)], vec![F::from(3u64), F::from(4u64)]], &apply_phi(&vec![vec![F::from(1u64), F::from(2u64)], vec![F::from(3u64), F::from(4u64)]])),
            k_lasso: build_lasso(&vec![vec![F::from(0u64), F::from(1u64)], vec![F::from(2u64), F::from(3u64)]], &apply_phi(&vec![vec![F::from(0u64), F::from(1u64)], vec![F::from(2u64), F::from(3u64)]])),
        }
    }

    fn lasso_params() -> HyraxParams { HyraxParams::new(2) } // 4-bit table: sigma=4-2=2

    #[test]
    fn test_linear_attention_e2e_success() {
        let inst = setup_test_instance(2, 2);
        let lp = lasso_params();

        let mut pt = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&inst, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &lp);
        assert!(result.is_ok(), "verification failed: {:?}", result.err());
    }

    #[test]
    fn test_rejects_tampered_context_matrix() {
        let mut inst = setup_test_instance(2, 2);
        let lp = lasso_params();

        // Tamper context before proving (breaks sumcheck)
        inst.context[0][0] += F::one();

        let mut pt = Transcript::new(b"linear_attn_tamper_ctx");
        let proof = prove_linear_attention(&inst, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_tamper_ctx");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &lp);
        assert!(result.is_err(), "should reject tampered context");
    }

    #[test]
    fn test_rejects_tampered_phi_q_lasso() {
        let mut inst = setup_test_instance(2, 2);
        let lp = lasso_params();

        inst.q_lasso.outputs[0] += F::one();

        let mut pt = Transcript::new(b"tq");
        let proof = prove_linear_attention(&inst, &mut pt, &lp);
        let mut vt = Transcript::new(b"tq");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &lp);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("phi_q:"));
    }

    #[test]
    fn test_rejects_tampered_phi_k_lasso() {
        let mut inst = setup_test_instance(2, 2);
        let lp = lasso_params();

        inst.k_lasso.outputs[0] += F::one();

        let mut pt = Transcript::new(b"tk");
        let proof = prove_linear_attention(&inst, &mut pt, &lp);
        let mut vt = Transcript::new(b"tk");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &lp);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("phi_k:"));
    }

    #[test]
    fn test_rejects_tampered_out_matrix() {
        let mut inst = setup_test_instance(2, 2);
        let lp = lasso_params();

        let mut pt = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&inst, &mut pt, &lp);

        // Tamper out AFTER proof generation → commitment mismatch
        inst.out[1][1] += F::one();

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &lp);
        assert!(result.is_err(), "should reject tampered out");
        assert!(result.unwrap_err().contains("out commitment mismatch"));
    }

    #[test]
    fn test_rejects_tampered_phi_q_matrix() {
        let mut inst = setup_test_instance(2, 2);
        let lp = lasso_params();

        let mut pt = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&inst, &mut pt, &lp);

        inst.phi_q[1][0] += F::one();

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &lp);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("phi_q commitment mismatch"));
    }

    #[test]
    fn test_rejects_tampered_v_matrix() {
        let mut inst = setup_test_instance(2, 2);
        let lp = lasso_params();

        let mut pt = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&inst, &mut pt, &lp);

        inst.v[0][1] += F::one();

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &lp);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("V commitment mismatch"));
    }

    #[test]
    fn test_rejects_tampered_phi_k_matrix() {
        let mut inst = setup_test_instance(2, 2);
        let lp = lasso_params();

        let mut pt = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&inst, &mut pt, &lp);

        inst.phi_k[0][0] += F::one();

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &mut vt, &lp);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("phi_k commitment mismatch"));
    }
}
