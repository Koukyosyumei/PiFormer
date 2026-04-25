//! Global Verifier for a full Transformer Model.
//!
//! **Cross-block Batch Sumcheck Architecture:**
//! Phase 1: verify range proofs + LN1 + LN2 for all L blocks; absorb intermediate commitments.
//! Phase 2: four cross-block SumcheckProofMulti (QKV, O-proj, FFN-Y, FFN-M) share one r_k per type.
//!   Algebraic checks happen inline; batch Hyrax opens happen at the end (after mu challenges)
//!   to match the prover's transcript ordering.
//! Global: 5L intermediate matrices verified at shared r_td (inter_batch_open), then 13
//!   cross-block batch opens, then global Lasso.

use crate::field::F;
use crate::lookup::lasso::{verify_lasso, verify_lasso_multi, LassoMultiInstance, LassoMultiVerifyingKey};
use crate::lookup::range::verify_range_batched;
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_verify, hyrax_verify_batch, params_from_vars, HyraxBatchAccumulator,
    HyraxCommitment, HyraxParams,
};
use crate::poly::utils::{combine, mat_to_mle};
use crate::subprotocols::verify_sumcheck_multi_batched;
use crate::transcript::{challenge_vec, Transcript};

use crate::attention::attention::{
    AttentionProvingKey,
    LinearAttentionInstance,
};
use crate::attention::layernorm::{
    verify_layernorm, LayerNormIOCommitments, LayerNormVerifyingKey,
};
use crate::attention::projection::{
    verify_projection, verify_qkv_projections, BatchedQKVProjectionIOCommitments,
    ProjectionIOCommitments, ProjectionProvingKey, ProjectionVerifyingKey,
};
use crate::ffn::ffn::{FFNInstance, FFNProvingKey, FFNVerifyingKey};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::Field;
use std::ops::AddAssign;

use crate::prover::{TransformerModelProof, TransformerModelVerifyingKey};

// ---------------------------------------------------------------------------
// Verifying Key
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TransformerBlockVerifyingKey {
    pub seq_len: usize,
    pub d_model: usize,
    pub ln1_vk: LayerNormVerifyingKey,
    pub q_vk: ProjectionVerifyingKey,
    pub k_vk: ProjectionVerifyingKey,
    pub v_vk: ProjectionVerifyingKey,
    pub o_vk: ProjectionVerifyingKey,
    pub ln2_vk: LayerNormVerifyingKey,
    pub ffn_vk: FFNVerifyingKey,
    pub q_pk: ProjectionProvingKey,
    pub k_pk: ProjectionProvingKey,
    pub v_pk: ProjectionProvingKey,
    pub o_pk: ProjectionProvingKey,
    pub ffn_pk: FFNProvingKey,
    pub attn_pk: AttentionProvingKey,
}

// ---------------------------------------------------------------------------
// Cryptographic Helper: Homomorphic Addition
// ---------------------------------------------------------------------------

pub fn add_commitments(a: &HyraxCommitment, b: &HyraxCommitment) -> HyraxCommitment {
    assert_eq!(a.row_coms.len(), b.row_coms.len(), "Commitment dimensions must match");
    let mut result_coms = Vec::with_capacity(a.row_coms.len());
    for (pt_a, pt_b) in a.row_coms.iter().zip(b.row_coms.iter()) {
        let mut sum_proj = pt_a.into_group();
        sum_proj.add_assign(&pt_b.into_group());
        result_coms.push(sum_proj.into_affine());
    }
    HyraxCommitment { row_coms: result_coms, nu: a.nu, sigma: a.sigma }
}

fn powers_of(base: F, n: usize) -> Vec<F> {
    let mut v = Vec::with_capacity(n);
    let mut cur = F::from(1u64);
    for _ in 0..n {
        v.push(cur);
        cur *= base;
    }
    v
}

fn commit_public_mat(mat: &[Vec<F>], rows: usize, cols: usize) -> Result<HyraxCommitment, String> {
    if mat.len() != rows || mat.iter().any(|r| r.len() != cols) {
        return Err(format!(
            "public matrix dimension mismatch: got {}x{}, expected {}x{}",
            mat.len(),
            mat.first().map(|r| r.len()).unwrap_or(0),
            rows,
            cols
        ));
    }
    let mle = mat_to_mle(mat, rows, cols);
    let vars = rows.next_power_of_two().trailing_zeros() as usize
        + cols.next_power_of_two().trailing_zeros() as usize;
    let (nu, _, params) = params_from_vars(vars);
    Ok(hyrax_commit(&mle.evaluations, nu, &params))
}

fn commitments_equal(a: &HyraxCommitment, b: &HyraxCommitment) -> bool {
    a.nu == b.nu && a.sigma == b.sigma && a.row_coms == b.row_coms
}

// ---------------------------------------------------------------------------
// Model Verifier (E2E)
// ---------------------------------------------------------------------------

pub fn verify(
    proof: &TransformerModelProof,
    vk: &TransformerModelVerifyingKey,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    public_x_in: &[Vec<F>],
    public_logits: &[Vec<F>],
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> Result<(), String> {
    use std::time::Instant;

    // 1. Bind initial input
    absorb_com(transcript, b"x_in_com", &proof.x_in_com);

    let t = vk.seq_len;
    let d = vk.d_model;
    let v_vocab = vk.vocab_size;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;
    let (_, _, params_t) = params_from_vars(t_bits);
    let (_, _, params_td) = params_from_vars(td_num_vars);

    let q_vk = &vk.block_vks[0].q_vk;
    let qkvo_in_bits = q_vk.d_in.next_power_of_two().trailing_zeros() as usize;
    let qkvo_out_bits = q_vk.d_out.next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_qkvo_w) = params_from_vars(qkvo_in_bits + qkvo_out_bits);
    let (_, _, params_qkvo_b) = params_from_vars(qkvo_out_bits);

    let lmh_in_bits = vk.lm_head_vk.d_in.next_power_of_two().trailing_zeros() as usize;
    let lmh_out_bits = vk.lm_head_vk.d_out.next_power_of_two().trailing_zeros() as usize;
    let params_lmh_w = params_from_vars(lmh_in_bits + lmh_out_bits).2;
    let params_lmh_b = params_from_vars(lmh_out_bits).2;

    let ln_sig_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let ln_y_n_global = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_range_sig) = params_from_vars(ln_sig_n);
    let (_, _, params_range_y) = params_from_vars(ln_y_n_global);
    let (_, _, params_range_m) = params_from_vars(crate::lookup::range::CHUNK_BITS);

    let d_ff = vk.block_vks[0].ffn_vk.d_ff;
    let f_bits = d_ff.next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_wff) = params_from_vars(f_bits + d_bits);
    let (_, _, params_mff) = params_from_vars(t_bits + f_bits);

    let num_blocks = vk.num_blocks;

    let expected_x_in_com = commit_public_mat(public_x_in, t, d)?;
    if !commitments_equal(&proof.x_in_com, &expected_x_in_com) {
        return Err("public input does not match x_in commitment".into());
    }
    let expected_logits_com = commit_public_mat(public_logits, t, v_vocab)?;
    if !commitments_equal(&proof.logits_com, &expected_logits_com) {
        return Err("public output does not match logits commitment".into());
    }

    let mut ln_acc_t = HyraxBatchAccumulator::new();
    let mut ln_acc_td = HyraxBatchAccumulator::new();
    let mut proj_acc_w = HyraxBatchAccumulator::new();
    let mut proj_acc_b = HyraxBatchAccumulator::new();
    let mut lmh_acc_w = HyraxBatchAccumulator::new();
    let mut lmh_acc_b = HyraxBatchAccumulator::new();
    let mut acc_range_sig = HyraxBatchAccumulator::new();
    let mut acc_range_y = HyraxBatchAccumulator::new();
    let mut acc_range_m = HyraxBatchAccumulator::new();
    // inter_acc: per-block v_attn opens (different eval point per block)
    let mut inter_acc = HyraxBatchAccumulator::new();

    // =========================================================================
    // 2. Phase 1: verify range proofs + LN1 + LN2 for all blocks
    // =========================================================================
    let mut current_x_com = proof.x_in_com.clone();

    for i in 0..num_blocks {
        let bp = &proof.block_proofs[i];
        let bvk = &vk.block_vks[i];
        let ln_sigma_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
        let ln_y_n = (2 * t * d).next_power_of_two().trailing_zeros() as usize;

        let _t0 = Instant::now();
        let (block_r_vs, _) = verify_range_batched(
            &[
                &bp.ln1_proof.sigma_range_proof,
                &bp.ln1_proof.y_range_proof,
                &bp.ln2_proof.sigma_range_proof,
                &bp.ln2_proof.y_range_proof,
            ],
            &bp.block_range_m,
            &[ln_sigma_n, ln_y_n, ln_sigma_n, ln_y_n],
            32,
            transcript,
            &mut acc_range_sig,
            &mut acc_range_y,
            &mut acc_range_m,
        )?;
        eprintln!("[block {}] range_batch:{:>8.3}ms", i, _t0.elapsed().as_secs_f64()*1000.0);

        let ln1_sig_rv = &block_r_vs[0];
        let ln1_y_rv = &block_r_vs[1];
        let ln2_sig_rv = &block_r_vs[2];
        let ln2_y_rv = &block_r_vs[3];

        // LN1
        let ln1_io = LayerNormIOCommitments {
            x_com: current_x_com.clone(),
            y_com: Some(bp.x_norm1_com.clone()),
        };
        let _t0 = Instant::now();
        verify_layernorm(
            &bp.ln1_proof, &ln1_io, &bvk.ln1_vk,
            ln1_sig_rv, ln1_y_rv, transcript, &mut ln_acc_t, &mut ln_acc_td,
        )?;
        eprintln!("[block {}] ln1:{:>8.3}ms", i, _t0.elapsed().as_secs_f64()*1000.0);

        absorb_com(transcript, b"q_com", &bp.q_com);
        absorb_com(transcript, b"k_com", &bp.k_com);
        absorb_com(transcript, b"v_com", &bp.v_com);
        absorb_com(transcript, b"out_attn_com", &bp.out_attn_com);

        let x_mid_com = add_commitments(&current_x_com, &bp.out_attn_com);

        // LN2
        let ln2_io = LayerNormIOCommitments {
            x_com: x_mid_com.clone(),
            y_com: Some(bp.x_norm2_com.clone()),
        };
        let _t0 = Instant::now();
        verify_layernorm(
            &bp.ln2_proof, &ln2_io, &bvk.ln2_vk,
            ln2_sig_rv, ln2_y_rv, transcript, &mut ln_acc_t, &mut ln_acc_td,
        )?;
        eprintln!("[block {}] ln2:{:>8.3}ms", i, _t0.elapsed().as_secs_f64()*1000.0);

        absorb_com(transcript, b"y_com", &bp.out_ffn_com);

        let next_x_com = add_commitments(&x_mid_com, &bp.out_ffn_com);
        current_x_com = next_x_com;
    }

    // =========================================================================
    // 3. Derive global r_td after ALL blocks' Phase 1
    // =========================================================================
    let r_td = challenge_vec(transcript, td_num_vars, b"gkr_r_td");
    let r_t = r_td[..t_bits].to_vec();
    let r_out = r_td[t_bits..].to_vec();

    // =========================================================================
    // 4. Batch QKV (sumcheck only; batch opens deferred to step 13)
    // =========================================================================
    let _tqkv = Instant::now();

    let mut pb_lambda: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_mu: Vec<F> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let bvk = &vk.block_vks[i];
        let bp = &proof.block_proofs[i];

        absorb_com(transcript, b"qkv_w_q_com", &bvk.q_vk.w_com);
        absorb_com(transcript, b"qkv_w_k_com", &bvk.k_vk.w_com);
        absorb_com(transcript, b"qkv_w_v_com", &bvk.v_vk.w_com);
        transcript.append_field(b"qkv_alpha_q", &bvk.q_vk.alpha);
        transcript.append_field(b"qkv_alpha_k", &bvk.k_vk.alpha);
        transcript.append_field(b"qkv_alpha_v", &bvk.v_vk.alpha);
        absorb_com(transcript, b"qkv_bias_q_com", &bvk.q_vk.bias_com);
        absorb_com(transcript, b"qkv_bias_k_com", &bvk.k_vk.bias_com);
        absorb_com(transcript, b"qkv_bias_v_com", &bvk.v_vk.bias_com);

        let lambda: F = transcript.challenge_field(b"qkv_lambda");
        let mu: F = transcript.challenge_field(b"qkv_mu");

        transcript.append_field(b"qkv_q_eval", &bp.q_eval);
        transcript.append_field(b"qkv_k_eval", &bp.k_eval);
        transcript.append_field(b"qkv_v_eval", &bp.v_eval_rtd);

        pb_lambda.push(lambda);
        pb_mu.push(mu);
    }

    let eta_qkv: F = transcript.challenge_field(b"batch_eta_qkv");
    let weights_qkv = powers_of(eta_qkv, num_blocks);

    let claim_qkv: F = (0..num_blocks)
        .map(|i| {
            let bp = &proof.block_proofs[i];
            let target = pb_lambda[i] * (bp.q_eval - bp.qkv_bias_q_eval)
                + pb_mu[i] * (bp.k_eval - bp.qkv_bias_k_eval)
                + (bp.v_eval_rtd - bp.qkv_bias_v_eval);
            weights_qkv[i] * target
        })
        .sum();

    let (r_k_qkv, _) = verify_sumcheck_multi_batched(
        &proof.batch_qkv, &weights_qkv, claim_qkv, d_bits, transcript,
    )?;

    // Algebraic check: g_eval == lambda*alpha_q*wq + mu*alpha_k*wk + alpha_v*wv
    for i in 0..num_blocks {
        let bvk = &vk.block_vks[i];
        let bp = &proof.block_proofs[i];
        let g_reconstructed = pb_lambda[i] * bvk.q_vk.alpha * bp.qkv_w_q_eval
            + pb_mu[i] * bvk.k_vk.alpha * bp.qkv_w_k_eval
            + bvk.v_vk.alpha * bp.qkv_w_v_eval;
        if proof.batch_qkv.final_evals_g[i] != g_reconstructed {
            return Err(format!("Block {i}: QKV g_eval algebraic check failed"));
        }
    }

    eprintln!("[model] batch_qkv:{:>8.3}ms", _tqkv.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // 5. Batch O-proj (sumcheck only; batch opens deferred to step 13)
    // =========================================================================
    let _to = Instant::now();

    for i in 0..num_blocks {
        let bvk = &vk.block_vks[i];
        let bp = &proof.block_proofs[i];

        absorb_com(transcript, b"w_com", &bvk.o_vk.w_com);
        transcript.append_field(b"alpha", &bvk.o_vk.alpha);
        absorb_com(transcript, b"bias_com", &bvk.o_vk.bias_com);
        transcript.append_field(b"claimed_y", &bp.out_attn_eval);
    }

    let eta_oproj: F = transcript.challenge_field(b"batch_eta_oproj");
    let weights_oproj = powers_of(eta_oproj, num_blocks);

    let claim_oproj: F = (0..num_blocks)
        .map(|i| {
            let bp = &proof.block_proofs[i];
            weights_oproj[i] * (bp.out_attn_eval - bp.oproj_bias_o_eval)
        })
        .sum();

    let (r_k_o, _) = verify_sumcheck_multi_batched(
        &proof.batch_oproj, &weights_oproj, claim_oproj, d_bits, transcript,
    )?;

    // Algebraic check: final_evals_g[i] == bp.oproj_w_o_eval
    for i in 0..num_blocks {
        let bp = &proof.block_proofs[i];
        if proof.batch_oproj.final_evals_g[i] != bp.oproj_w_o_eval {
            return Err(format!("Block {i}: O-proj g_eval algebraic check failed"));
        }
    }

    eprintln!("[model] batch_oproj:{:>8.3}ms", _to.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // 6. Cross-block batch attention sumchecks
    // =========================================================================
    let _tattn = Instant::now();

    // 6a. Absorb phi_q_com, phi_k_com per block (mirrors prover step 6a)
    for i in 0..num_blocks {
        let bp = &proof.block_proofs[i];
        absorb_com(transcript, b"phi_q_com", &bp.attn_phi_q_com);
        absorb_com(transcript, b"phi_k_com", &bp.attn_phi_k_com);
    }

    // 6b. Verify batch out sumcheck
    for i in 0..num_blocks {
        transcript.append_field(b"attn_out_eval", &proof.block_proofs[i].attn_out_eval);
    }
    let eta_attn_out: F = transcript.challenge_field(b"batch_eta_attn_out");
    let weights_attn_out = powers_of(eta_attn_out, num_blocks);
    let claim_attn_out: F = (0..num_blocks)
        .map(|i| weights_attn_out[i] * proof.block_proofs[i].attn_out_eval)
        .sum();
    let (batch_r_attn_out, _) = verify_sumcheck_multi_batched(
        &proof.batch_attn_out, &weights_attn_out, claim_attn_out, d_bits, transcript,
    )?;

    // Algebraic checks: leaves of batch_attn_out match claimed scalars
    for i in 0..num_blocks {
        let bp = &proof.block_proofs[i];
        // Derive expected attn_out_eval from batch_oproj leaf / alpha_o
        let alpha_o = vk.block_vks[i].o_vk.alpha;
        let expected_out = if alpha_o == F::from(0u64) {
            F::from(0u64)
        } else {
            proof.batch_oproj.final_evals_f[i] * alpha_o.inverse().unwrap()
        };
        if bp.attn_out_eval != expected_out {
            return Err(format!("Block {i}: attn_out_eval mismatch with o_proj leaf"));
        }
        if proof.batch_attn_out.final_evals_f[i] != bp.attn_phi_q_eval {
            return Err(format!("Block {i}: batch_attn_out final_f != phi_q_eval"));
        }
        if proof.batch_attn_out.final_evals_g[i] != bp.attn_ctx_eval {
            return Err(format!("Block {i}: batch_attn_out final_g != ctx_eval"));
        }
    }

    // 6c. Verify batch ctx sumcheck
    for i in 0..num_blocks {
        transcript.append_field(b"attn_ctx_eval", &proof.block_proofs[i].attn_ctx_eval);
    }
    let eta_attn_ctx: F = transcript.challenge_field(b"batch_eta_attn_ctx");
    let weights_attn_ctx = powers_of(eta_attn_ctx, num_blocks);
    let claim_attn_ctx: F = (0..num_blocks)
        .map(|i| weights_attn_ctx[i] * proof.block_proofs[i].attn_ctx_eval)
        .sum();
    let (batch_r_attn_ctx, _) = verify_sumcheck_multi_batched(
        &proof.batch_attn_ctx, &weights_attn_ctx, claim_attn_ctx, t_bits, transcript,
    )?;

    // Algebraic checks: leaves of batch_attn_ctx match claimed scalars
    for i in 0..num_blocks {
        let bp = &proof.block_proofs[i];
        if proof.batch_attn_ctx.final_evals_f[i] != bp.attn_phi_k_eval {
            return Err(format!("Block {i}: batch_attn_ctx final_f != phi_k_eval"));
        }
        if proof.batch_attn_ctx.final_evals_g[i] != bp.attn_v_eval {
            return Err(format!("Block {i}: batch_attn_ctx final_g != v_eval"));
        }
    }

    eprintln!("[model] batch_attn:{:>8.3}ms", _tattn.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // 7. Per-block FFN: Lasso + M absorb
    // =========================================================================
    for i in 0..num_blocks {
        let bvk = &vk.block_vks[i];
        let bp = &proof.block_proofs[i];

        absorb_com(transcript, b"w1_com", &bvk.ffn_vk.w1_com);
        absorb_com(transcript, b"w2_com", &bvk.ffn_vk.w2_com);

        let _t0 = Instant::now();
        verify_lasso(
            &bp.ffn_lasso_proof,
            &inst_ffn.activation_lasso,
            &bvk.ffn_vk.activation_lasso_vk,
            transcript,
            lasso_params,
        )?;
        eprintln!("[block {}] ffn_lasso:{:>8.3}ms", i, _t0.elapsed().as_secs_f64()*1000.0);

        absorb_com(transcript, b"m_com", &bp.ffn_m_com);
    }

    // =========================================================================
    // 8. Batch FFN-Y (sumcheck only)
    // =========================================================================
    let _tffy = Instant::now();

    for i in 0..num_blocks {
        transcript.append_field(b"claim_y", &proof.block_proofs[i].out_ffn_eval);
    }

    let eta_ffn_y: F = transcript.challenge_field(b"batch_eta_ffn_y");
    let weights_ffn_y = powers_of(eta_ffn_y, num_blocks);

    let claim_ffn_y: F = (0..num_blocks)
        .map(|i| weights_ffn_y[i] * proof.block_proofs[i].out_ffn_eval)
        .sum();

    let (r_k_fy, _) = verify_sumcheck_multi_batched(
        &proof.batch_ffn_y, &weights_ffn_y, claim_ffn_y, f_bits, transcript,
    )?;

    eprintln!("[model] batch_ffn_y:{:>8.3}ms", _tffy.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // 9. Batch FFN-M (sumcheck only)
    // =========================================================================
    let _tffm = Instant::now();

    let rx_m = challenge_vec(transcript, t_bits, b"ffn_rx_m");
    let ry_m = challenge_vec(transcript, f_bits, b"ffn_ry_m");

    for i in 0..num_blocks {
        transcript.append_field(b"claim_m", &proof.block_proofs[i].ffn_m_eval);
    }

    let eta_ffn_m: F = transcript.challenge_field(b"batch_eta_ffn_m");
    let weights_ffn_m = powers_of(eta_ffn_m, num_blocks);

    let claim_ffn_m: F = (0..num_blocks)
        .map(|i| weights_ffn_m[i] * proof.block_proofs[i].ffn_m_eval)
        .sum();

    let (r_k_m, _) = verify_sumcheck_multi_batched(
        &proof.batch_ffn_m, &weights_ffn_m, claim_ffn_m, d_bits, transcript,
    )?;

    eprintln!("[model] batch_ffn_m:{:>8.3}ms", _tffm.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // 10. Final LayerNorm
    // =========================================================================
    let final_sigma_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let final_y_n = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
    let _t0 = Instant::now();
    let (final_r_vs, _) = verify_range_batched(
        &[&proof.final_ln_proof.sigma_range_proof, &proof.final_ln_proof.y_range_proof],
        &proof.final_range_m,
        &[final_sigma_n, final_y_n],
        32,
        transcript,
        &mut acc_range_sig,
        &mut acc_range_y,
        &mut acc_range_m,
    )
    .map_err(|e| format!("Final LN range: {e}"))?;
    eprintln!("[model] final_range:{:>8.3}ms", _t0.elapsed().as_secs_f64()*1000.0);

    let _t0 = Instant::now();
    let ln_io = LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: Some(proof.final_ln_out_com.clone()),
    };
    verify_layernorm(
        &proof.final_ln_proof, &ln_io, &vk.final_ln_vk,
        &final_r_vs[0], &final_r_vs[1], transcript, &mut ln_acc_t, &mut ln_acc_td,
    )
    .map_err(|e| format!("Final LN: {e}"))?;
    eprintln!("[model] final_ln:{:>8.3}ms", _t0.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // 11. LM Head
    // =========================================================================
    let _t0 = Instant::now();
    let lm_io = ProjectionIOCommitments { x_com: Some(proof.final_ln_out_com.clone()) };
    let (lm_y_claim, _) = verify_projection(
        &proof.lm_head_proof, &vk.lm_head_vk, &lm_io, transcript,
        &mut lmh_acc_w, &mut lmh_acc_b, None,
    )
    .map_err(|e| format!("LM Head: {e}"))?;
    eprintln!("[model] lm_head:{:>8.3}ms", _t0.elapsed().as_secs_f64()*1000.0);

    let v_bits = v_vocab.next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_logits) = params_from_vars(t_bits + v_bits);
    hyrax_verify(
        &proof.logits_com, lm_y_claim.value, &lm_y_claim.point,
        &proof.lm_head_logits_open, &params_logits,
    )
    .map_err(|e| format!("Logits commit: {e}"))?;

    // =========================================================================
    // 12. Finalize 10 accumulators (same order as prover's mu challenge loop)
    // =========================================================================
    let _tacc = Instant::now();
    let mu_inter  = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_ln_t   = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_ln_td  = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_proj_w = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_proj_b = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_lmh_w  = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_lmh_b  = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_rng_sig = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_rng_y  = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_rng_m  = transcript.challenge_field::<F>(b"hyrax_group_mu");

    let ((r0, r1), (r2, r3)) = rayon::join(
        || rayon::join(
            || inter_acc.finalize_with_mu(&params_td, mu_inter).map_err(|e| format!("inter_acc: {e}")),
            || ln_acc_t.finalize_with_mu(&params_t, mu_ln_t).map_err(|e| format!("ln_acc_t: {e}")),
        ),
        || rayon::join(
            || ln_acc_td.finalize_with_mu(&params_td, mu_ln_td).map_err(|e| format!("ln_acc_td: {e}")),
            || proj_acc_w.finalize_with_mu(&params_qkvo_w, mu_proj_w).map_err(|e| format!("proj_acc_w: {e}")),
        ),
    );
    let ((r4, r5), (r6, r7)) = rayon::join(
        || rayon::join(
            || proj_acc_b.finalize_with_mu(&params_qkvo_b, mu_proj_b).map_err(|e| format!("proj_acc_b: {e}")),
            || lmh_acc_w.finalize_with_mu(&params_lmh_w, mu_lmh_w).map_err(|e| format!("lmh_acc_w: {e}")),
        ),
        || rayon::join(
            || lmh_acc_b.finalize_with_mu(&params_lmh_b, mu_lmh_b).map_err(|e| format!("lmh_acc_b: {e}")),
            || acc_range_sig.finalize_with_mu(&params_range_sig, mu_rng_sig).map_err(|e| format!("acc_range_sig: {e}")),
        ),
    );
    let (r8, r9) = rayon::join(
        || acc_range_y.finalize_with_mu(&params_range_y, mu_rng_y).map_err(|e| format!("acc_range_y: {e}")),
        || acc_range_m.finalize_with_mu(&params_range_m, mu_rng_m).map_err(|e| format!("acc_range_m: {e}")),
    );
    eprintln!("[model] acc_finalize:{:>8.3}ms", _tacc.elapsed().as_secs_f64()*1000.0);
    r0?; r1?; r2?; r3?; r4?; r5?; r6?; r7?; r8?; r9?;

    // =========================================================================
    // 13. Global batch open for 5L intermediate matrices at r_td (inter_batch_open)
    // =========================================================================
    let _t0 = Instant::now();
    let mut all_coms: Vec<HyraxCommitment> = Vec::with_capacity(5 * num_blocks);
    let mut all_evals: Vec<F> = Vec::with_capacity(5 * num_blocks);
    for bp in &proof.block_proofs {
        all_coms.push(bp.q_com.clone());
        all_coms.push(bp.k_com.clone());
        all_coms.push(bp.v_com.clone());
        all_coms.push(bp.out_attn_com.clone());
        all_coms.push(bp.out_ffn_com.clone());
        all_evals.push(bp.q_eval);
        all_evals.push(bp.k_eval);
        all_evals.push(bp.v_eval_rtd);
        all_evals.push(bp.out_attn_eval);
        all_evals.push(bp.out_ffn_eval);
    }
    hyrax_verify_batch(
        &all_coms, &all_evals, &r_td, &proof.inter_batch_open, &params_td, transcript,
    )
    .map_err(|e| format!("Global inter_batch: {e}"))?;
    eprintln!("[model] inter_batch:{:>8.3}ms", _t0.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // 14. 13 cross-block weight/activation batch opens
    //     (must be in same order as prover's step 15)
    // =========================================================================
    let _tbatch = Instant::now();

    // x_norm1 at combine(r_t, r_k_qkv)
    let x_norm1_point = combine(&r_t, &r_k_qkv);
    let x_norm1_coms: Vec<HyraxCommitment> =
        proof.block_proofs.iter().map(|bp| bp.x_norm1_com.clone()).collect();
    let x_norm1_evals: Vec<F> = proof.batch_qkv.final_evals_f.clone();
    hyrax_verify_batch(
        &x_norm1_coms, &x_norm1_evals, &x_norm1_point,
        &proof.x_norm1_batch_open, &params_td, transcript,
    ).map_err(|e| format!("x_norm1_batch: {e}"))?;

    // wq/wk/wv at combine(r_k_qkv, r_out)
    let wq_point = combine(&r_k_qkv, &r_out);
    let wq_coms: Vec<HyraxCommitment> =
        vk.block_vks.iter().map(|bvk| bvk.q_vk.w_com.clone()).collect();
    let wq_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.qkv_w_q_eval).collect();
    hyrax_verify_batch(
        &wq_coms, &wq_evals, &wq_point,
        &proof.w_q_batch_open, &params_qkvo_w, transcript,
    ).map_err(|e| format!("w_q_batch: {e}"))?;

    let wk_coms: Vec<HyraxCommitment> =
        vk.block_vks.iter().map(|bvk| bvk.k_vk.w_com.clone()).collect();
    let wk_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.qkv_w_k_eval).collect();
    hyrax_verify_batch(
        &wk_coms, &wk_evals, &wq_point,
        &proof.w_k_batch_open, &params_qkvo_w, transcript,
    ).map_err(|e| format!("w_k_batch: {e}"))?;

    let wv_coms: Vec<HyraxCommitment> =
        vk.block_vks.iter().map(|bvk| bvk.v_vk.w_com.clone()).collect();
    let wv_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.qkv_w_v_eval).collect();
    hyrax_verify_batch(
        &wv_coms, &wv_evals, &wq_point,
        &proof.w_v_batch_open, &params_qkvo_w, transcript,
    ).map_err(|e| format!("w_v_batch: {e}"))?;

    // bias_q/k/v at r_out
    let bq_coms: Vec<HyraxCommitment> =
        vk.block_vks.iter().map(|bvk| bvk.q_vk.bias_com.clone()).collect();
    let bq_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.qkv_bias_q_eval).collect();
    hyrax_verify_batch(
        &bq_coms, &bq_evals, &r_out,
        &proof.bias_q_batch_open, &params_qkvo_b, transcript,
    ).map_err(|e| format!("bias_q_batch: {e}"))?;

    let bk_coms: Vec<HyraxCommitment> =
        vk.block_vks.iter().map(|bvk| bvk.k_vk.bias_com.clone()).collect();
    let bk_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.qkv_bias_k_eval).collect();
    hyrax_verify_batch(
        &bk_coms, &bk_evals, &r_out,
        &proof.bias_k_batch_open, &params_qkvo_b, transcript,
    ).map_err(|e| format!("bias_k_batch: {e}"))?;

    let bv_coms: Vec<HyraxCommitment> =
        vk.block_vks.iter().map(|bvk| bvk.v_vk.bias_com.clone()).collect();
    let bv_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.qkv_bias_v_eval).collect();
    hyrax_verify_batch(
        &bv_coms, &bv_evals, &r_out,
        &proof.bias_v_batch_open, &params_qkvo_b, transcript,
    ).map_err(|e| format!("bias_v_batch: {e}"))?;

    // wo at combine(r_k_o, r_out)
    let wo_point = combine(&r_k_o, &r_out);
    let wo_coms: Vec<HyraxCommitment> =
        vk.block_vks.iter().map(|bvk| bvk.o_vk.w_com.clone()).collect();
    let wo_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.oproj_w_o_eval).collect();
    hyrax_verify_batch(
        &wo_coms, &wo_evals, &wo_point,
        &proof.w_o_batch_open, &params_qkvo_w, transcript,
    ).map_err(|e| format!("w_o_batch: {e}"))?;

    // bias_o at r_out
    let bo_coms: Vec<HyraxCommitment> =
        vk.block_vks.iter().map(|bvk| bvk.o_vk.bias_com.clone()).collect();
    let bo_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.oproj_bias_o_eval).collect();
    hyrax_verify_batch(
        &bo_coms, &bo_evals, &r_out,
        &proof.bias_o_batch_open, &params_qkvo_b, transcript,
    ).map_err(|e| format!("bias_o_batch: {e}"))?;

    // w2 at combine(r_k_fy, r_out)
    let w2_point = combine(&r_k_fy, &r_out);
    let w2_coms: Vec<HyraxCommitment> =
        vk.block_vks.iter().map(|bvk| bvk.ffn_vk.w2_com.clone()).collect();
    let w2_evals: Vec<F> = proof.batch_ffn_y.final_evals_g.clone();
    hyrax_verify_batch(
        &w2_coms, &w2_evals, &w2_point,
        &proof.w2_batch_open, &params_wff, transcript,
    ).map_err(|e| format!("w2_batch: {e}"))?;

    // w1 at combine(r_k_m, ry_m) — uses same params as prover (params_qkvo_w)
    let w1_point = combine(&r_k_m, &ry_m);
    let w1_coms: Vec<HyraxCommitment> =
        vk.block_vks.iter().map(|bvk| bvk.ffn_vk.w1_com.clone()).collect();
    let w1_evals: Vec<F> = proof.batch_ffn_m.final_evals_g.clone();
    hyrax_verify_batch(
        &w1_coms, &w1_evals, &w1_point,
        &proof.w1_batch_open, &params_wff, transcript,
    ).map_err(|e| format!("w1_batch: {e}"))?;

    // x_norm2 at combine(rx_m, r_k_m)
    let x_norm2_point = combine(&rx_m, &r_k_m);
    let x_norm2_coms: Vec<HyraxCommitment> =
        proof.block_proofs.iter().map(|bp| bp.x_norm2_com.clone()).collect();
    let x_norm2_evals: Vec<F> = proof.batch_ffn_m.final_evals_f.clone();
    hyrax_verify_batch(
        &x_norm2_coms, &x_norm2_evals, &x_norm2_point,
        &proof.x_norm2_batch_open, &params_td, transcript,
    ).map_err(|e| format!("x_norm2_batch: {e}"))?;

    // ffn_m_com at combine(rx_m, ry_m)
    let ffn_m_point = combine(&rx_m, &ry_m);
    let ffn_m_coms: Vec<HyraxCommitment> =
        proof.block_proofs.iter().map(|bp| bp.ffn_m_com.clone()).collect();
    let ffn_m_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.ffn_m_eval).collect();
    hyrax_verify_batch(
        &ffn_m_coms, &ffn_m_evals, &ffn_m_point,
        &proof.ffn_m_com_batch_open, &params_mff, transcript,
    ).map_err(|e| format!("ffn_m_com_batch: {e}"))?;

    // phi_q at combine(r_t, batch_r_attn_out)
    let phi_q_attn_point = combine(&r_t, &batch_r_attn_out);
    let phi_q_coms: Vec<HyraxCommitment> =
        proof.block_proofs.iter().map(|bp| bp.attn_phi_q_com.clone()).collect();
    let phi_q_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.attn_phi_q_eval).collect();
    hyrax_verify_batch(
        &phi_q_coms, &phi_q_evals, &phi_q_attn_point,
        &proof.phi_q_batch_open, &params_td, transcript,
    ).map_err(|e| format!("phi_q_batch: {e}"))?;

    // phi_k at combine(batch_r_attn_ctx, batch_r_attn_out)
    let phi_k_attn_point = combine(&batch_r_attn_ctx, &batch_r_attn_out);
    let phi_k_coms: Vec<HyraxCommitment> =
        proof.block_proofs.iter().map(|bp| bp.attn_phi_k_com.clone()).collect();
    let phi_k_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.attn_phi_k_eval).collect();
    hyrax_verify_batch(
        &phi_k_coms, &phi_k_evals, &phi_k_attn_point,
        &proof.phi_k_batch_open, &params_td, transcript,
    ).map_err(|e| format!("phi_k_batch: {e}"))?;

    // v_attn at combine(batch_r_attn_ctx, r_k_o)
    let v_attn_batch_point = combine(&batch_r_attn_ctx, &r_k_o);
    let v_attn_coms: Vec<HyraxCommitment> =
        proof.block_proofs.iter().map(|bp| bp.v_com.clone()).collect();
    let v_attn_evals: Vec<F> =
        proof.block_proofs.iter().map(|bp| bp.attn_v_eval).collect();
    hyrax_verify_batch(
        &v_attn_coms, &v_attn_evals, &v_attn_batch_point,
        &proof.v_attn_batch_open, &params_td, transcript,
    ).map_err(|e| format!("v_attn_batch: {e}"))?;

    eprintln!("[model] cross_batch_opens:{:>8.3}ms", _tbatch.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // 15. Global batched Lasso (attention Q/K)
    // =========================================================================
    let _t0 = Instant::now();
    let mut all_lasso_instances = Vec::new();
    let mut all_instance_coms = Vec::new();
    let mut all_output_coms: Vec<(HyraxCommitment, usize)> = Vec::new();
    for i in 0..num_blocks {
        let bvk = &vk.block_vks[i];
        all_lasso_instances.push(inst_attn.q_lasso.clone());
        all_lasso_instances.push(inst_attn.k_lasso.clone());
        all_instance_coms.push(bvk.attn_pk.qk_lasso_pk.instance_table_coms[0].clone());
        all_instance_coms.push(bvk.attn_pk.qk_lasso_pk.instance_table_coms[1].clone());
        all_output_coms.push((proof.block_proofs[i].attn_phi_q_com.clone(), td_num_vars));
        all_output_coms.push((proof.block_proofs[i].attn_phi_k_com.clone(), td_num_vars));
    }
    let global_multi_inst = LassoMultiInstance { instances: all_lasso_instances };
    let global_lasso_vk = LassoMultiVerifyingKey { instance_table_coms: all_instance_coms };
    verify_lasso_multi(
        &proof.all_lasso_proof, &global_multi_inst, &global_lasso_vk,
        &all_output_coms, transcript, lasso_params,
    )
    .map_err(|e| format!("Global Lasso: {e}"))?;
    eprintln!("[model] lasso:{:>8.3}ms", _t0.elapsed().as_secs_f64()*1000.0);

    Ok(())
}
