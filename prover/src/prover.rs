//! Global Prover for a full Transformer Model.
//!
//! **Cross-block Batch Sumcheck Architecture:**
//! Phase 1 (all L blocks): commit 7 intermediate matrices per block, run LN proofs.
//! Phase 2 (model-level batch): four cross-block SumcheckProofMulti (QKV, O-proj, FFN-Y, FFN-M)
//!   share a single r_k per type across all L blocks → O(1) batch Hyrax opens per weight type.
//! Global: 5L intermediate matrices opened at shared r_td (inter_batch_open),
//!   plus 13 cross-block weight/activation batch opens.

use ark_ff::Field;
use crate::field::F;
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_open, hyrax_open_batch, params_from_vars,
    HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::poly::utils::{
    combine, convert_tm_to_fm, eval_cols, eval_cols_ternary, eval_rows, mat_to_mle, vec_to_mle,
};
use crate::poly::DenseMLPoly;
use crate::transcript::{challenge_vec, Transcript};

use crate::attention::attention::{
    LinearAttentionInstance, LinearAttentionWitness,
};
use crate::attention::layernorm::{
    compute_range_witnesses, prove_layernorm, LayerNormIOCommitments, LayerNormProof,
    LayerNormVerifyingKey, LayerNormWitness,
};
use crate::attention::projection::{
    prove_projection, ProjectionIOCommitments,
    ProjectionProof, ProjectionProvingKey, ProjectionVerifyingKey, ProjectionWitness,
};
use crate::ffn::ffn::{FFNInstance, FFNWitness};
use crate::lookup::lasso::{
    prove_lasso, prove_lasso_multi, LassoMultiInstance, LassoMultiProof, LassoMultiProvingKey,
    LassoOutputBinding, LassoProof,
};
use crate::lookup::range::{prove_range_batched, GlobalRangeM};
use crate::subprotocols::{prove_sumcheck_multi_batched, SumcheckProofMulti};
use crate::verifier::{add_commitments, TransformerBlockVerifyingKey};

// ---------------------------------------------------------------------------
// Proof Structures
// ---------------------------------------------------------------------------

/// ZK Proof for one Transformer Block (per-block data only).
///
/// The cross-block sumchecks and batch opens live in TransformerModelProof.
pub struct TransformerBlockProof {
    pub ln1_proof: LayerNormProof,
    pub ln2_proof: LayerNormProof,
    pub block_range_m: GlobalRangeM,

    // FFN per-block: Lasso for activation + M commitment
    pub ffn_lasso_proof: LassoProof,
    pub ffn_m_com: HyraxCommitment,

    // Committed intermediate matrices (7 per block)
    pub x_norm1_com: HyraxCommitment,
    pub q_com: HyraxCommitment,
    pub k_com: HyraxCommitment,
    pub v_com: HyraxCommitment,
    pub out_attn_com: HyraxCommitment,
    pub x_norm2_com: HyraxCommitment,
    pub out_ffn_com: HyraxCommitment,

    // Scalar evals at shared r_td (proven by global inter_batch_open)
    pub q_eval: F,
    pub k_eval: F,
    pub v_eval_rtd: F,
    pub out_attn_eval: F,
    pub out_ffn_eval: F,

    // Per-block scalars for batch QKV algebraic check
    pub qkv_lambda: F,
    pub qkv_mu: F,
    pub qkv_w_q_eval: F,
    pub qkv_w_k_eval: F,
    pub qkv_w_v_eval: F,
    pub qkv_bias_q_eval: F,
    pub qkv_bias_k_eval: F,
    pub qkv_bias_v_eval: F,

    // Per-block scalars for batch O-proj algebraic check
    pub oproj_w_o_eval: F,
    pub oproj_bias_o_eval: F,

    // Per-block scalars for batch FFN-M algebraic check
    pub ffn_m_eval: F,

    // Attention: phi_q/phi_k commitments (for Lasso binding + cross-block batch opens)
    pub attn_phi_q_com: HyraxCommitment,
    pub attn_phi_k_com: HyraxCommitment,

    // Per-block scalars for cross-block attention batch sumchecks
    pub attn_out_eval: F,    // x_inner_i(r_t, r_k_o) = claim for out sumcheck
    pub attn_phi_q_eval: F,  // phi_q_i(r_t, batch_r_attn_out) = leaf of batch_attn_out f
    pub attn_phi_k_eval: F,  // phi_k_i(batch_r_attn_ctx, batch_r_attn_out) = leaf of batch_attn_ctx f
    pub attn_ctx_eval: F,    // ctx_i(batch_r_attn_out, r_k_o) = leaf of batch_attn_out g
    pub attn_v_eval: F,      // v_i(batch_r_attn_ctx, r_k_o) = leaf of batch_attn_ctx g
}

// ---------------------------------------------------------------------------
// Witness Structures
// ---------------------------------------------------------------------------

pub struct TransformerBlockWitness {
    pub x_in: Vec<Vec<F>>,
    pub ln1_wit: LayerNormWitness,
    pub q_proj_wit: ProjectionWitness,
    pub k_proj_wit: ProjectionWitness,
    pub v_proj_wit: ProjectionWitness,
    pub attn_wit: LinearAttentionWitness,
    pub o_proj_wit: ProjectionWitness,
    pub x_mid: Vec<Vec<F>>,
    pub ln2_wit: LayerNormWitness,
    pub ffn_wit: FFNWitness,
    pub x_out: Vec<Vec<F>>,
}

// ---------------------------------------------------------------------------
// Internal Phase 1 data (not part of public proof)
// ---------------------------------------------------------------------------

struct BlockPhase1Data {
    ln1_proof: LayerNormProof,
    ln2_proof: LayerNormProof,
    block_range_m: GlobalRangeM,
    x_norm1_com: HyraxCommitment,
    q_com: HyraxCommitment,
    k_com: HyraxCommitment,
    v_com: HyraxCommitment,
    out_attn_com: HyraxCommitment,
    x_norm2_com: HyraxCommitment,
    out_ffn_com: HyraxCommitment,
    x_mid_com: HyraxCommitment,
}

// ---------------------------------------------------------------------------
// Phase 1: commit + LN proofs + absorb into transcript
// ---------------------------------------------------------------------------

fn commit_block_phase1(
    witness: &TransformerBlockWitness,
    x_in_com: &HyraxCommitment,
    pk: &TransformerBlockVerifyingKey,
    transcript: &mut Transcript,
) -> Result<BlockPhase1Data, String> {
    let t = pk.seq_len;
    let d = pk.d_model;

    let commit_mat = |mat: &[Vec<F>], rows: usize, cols: usize| -> HyraxCommitment {
        let mle = mat_to_mle(mat, rows, cols);
        let vars = rows.next_power_of_two().trailing_zeros()
            + cols.next_power_of_two().trailing_zeros();
        let (nu, _, params) = params_from_vars(vars as usize);
        hyrax_commit(&mle.evaluations, nu, &params)
    };

    // Commit all 7 intermediate matrices
    let x_norm1_com = commit_mat(&witness.ln1_wit.y, t, d);
    let q_com = commit_mat(&witness.attn_wit.q, t, d);
    let k_com = commit_mat(&witness.attn_wit.k, t, d);
    let v_com = commit_mat(&witness.attn_wit.v, t, d);
    let out_attn_com = commit_mat(&witness.o_proj_wit.y, t, d);
    let x_norm2_com = commit_mat(&witness.ln2_wit.y, t, d);
    let out_ffn_com = commit_mat(&witness.ffn_wit.y, t, d);

    // Range proofs for LN1 and LN2
    let ln1_rw = compute_range_witnesses(&witness.ln1_wit, &pk.ln1_vk);
    let ln2_rw = compute_range_witnesses(&witness.ln2_wit, &pk.ln2_vk);
    let (mut block_range_proofs, block_range_m, block_r_vs) = prove_range_batched(
        &[
            &ln1_rw.sigma_witness,
            &ln1_rw.y_witness,
            &ln2_rw.sigma_witness,
            &ln2_rw.y_witness,
        ],
        32,
        transcript,
    )?;
    let ln2_y_rp = block_range_proofs.remove(3);
    let ln2_sig_rp = block_range_proofs.remove(2);
    let ln1_y_rp = block_range_proofs.remove(1);
    let ln1_sig_rp = block_range_proofs.remove(0);
    let ln2_y_rv = block_r_vs[3].clone();
    let ln2_sig_rv = block_r_vs[2].clone();
    let ln1_y_rv = block_r_vs[1].clone();
    let ln1_sig_rv = block_r_vs[0].clone();

    // LN1 sub-prover: absorbs x_norm1_com as y_com
    let ln1_io = LayerNormIOCommitments {
        x_com: x_in_com.clone(),
        y_com: Some(x_norm1_com.clone()),
    };
    let ln1_proof = prove_layernorm(
        &witness.ln1_wit,
        &ln1_io,
        &pk.ln1_vk,
        (ln1_sig_rp, ln1_sig_rv),
        (ln1_y_rp, ln1_y_rv),
        transcript,
    )?;

    // Explicitly absorb q/k/v_com with the same labels attention uses.
    absorb_com(transcript, b"q_com", &q_com);
    absorb_com(transcript, b"k_com", &k_com);
    absorb_com(transcript, b"v_com", &v_com);
    // out_attn_com not absorbed by any sub-prover; absorb explicitly here.
    absorb_com(transcript, b"out_attn_com", &out_attn_com);

    // Residual 1 (homomorphic)
    let x_mid_com = add_commitments(x_in_com, &out_attn_com);

    // LN2 sub-prover: absorbs x_norm2_com as y_com
    let ln2_io = LayerNormIOCommitments {
        x_com: x_mid_com.clone(),
        y_com: Some(x_norm2_com.clone()),
    };
    let ln2_proof = prove_layernorm(
        &witness.ln2_wit,
        &ln2_io,
        &pk.ln2_vk,
        (ln2_sig_rp, ln2_sig_rv),
        (ln2_y_rp, ln2_y_rv),
        transcript,
    )?;

    // Absorb out_ffn_com so transcript stays consistent with Phase 1.
    absorb_com(transcript, b"y_com", &out_ffn_com);

    Ok(BlockPhase1Data {
        ln1_proof,
        ln2_proof,
        block_range_m,
        x_norm1_com,
        q_com,
        k_com,
        v_com,
        out_attn_com,
        x_norm2_com,
        out_ffn_com,
        x_mid_com,
    })
}

// ---------------------------------------------------------------------------
// Global Model Keys and Structures
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TransformerModelVerifyingKey {
    pub num_blocks: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub vocab_size: usize,
    pub block_vks: Vec<TransformerBlockVerifyingKey>,
    pub final_ln_vk: LayerNormVerifyingKey,
    pub lm_head_vk: ProjectionVerifyingKey,
}

pub struct TransformerModelProvingKey {
    pub vk: TransformerModelVerifyingKey,
    pub block_pks: Vec<crate::verifier::TransformerBlockVerifyingKey>,
    pub lm_head_pk: ProjectionProvingKey,
}

pub struct TransformerModelWitness {
    pub x_in: Vec<Vec<F>>,
    pub block_witnesses: Vec<TransformerBlockWitness>,
    pub final_ln_wit: LayerNormWitness,
    pub lm_head_wit: ProjectionWitness,
}

pub struct TransformerModelProof {
    pub x_in_com: HyraxCommitment,
    pub block_proofs: Vec<TransformerBlockProof>,
    pub final_ln_proof: LayerNormProof,
    pub lm_head_proof: ProjectionProof,
    pub final_ln_out_com: HyraxCommitment,
    pub logits_com: HyraxCommitment,
    pub lm_head_logits_open: HyraxProof,
    pub all_lasso_proof: LassoMultiProof,
    pub final_range_m: GlobalRangeM,

    // Cross-block batch sumchecks (one per projection type + attention)
    pub batch_qkv: SumcheckProofMulti,
    pub batch_oproj: SumcheckProofMulti,
    pub batch_ffn_y: SumcheckProofMulti,
    pub batch_ffn_m: SumcheckProofMulti,
    pub batch_attn_out: SumcheckProofMulti,
    pub batch_attn_ctx: SumcheckProofMulti,

    // Global batch open for 5L intermediate matrices at shared r_td
    pub inter_batch_open: HyraxProof,

    // Cross-block batch opens (13 total, one per weight/activation type)
    pub x_norm1_batch_open: HyraxProof,
    pub w_q_batch_open: HyraxProof,
    pub w_k_batch_open: HyraxProof,
    pub w_v_batch_open: HyraxProof,
    pub bias_q_batch_open: HyraxProof,
    pub bias_k_batch_open: HyraxProof,
    pub bias_v_batch_open: HyraxProof,
    pub w_o_batch_open: HyraxProof,
    pub bias_o_batch_open: HyraxProof,
    pub w2_batch_open: HyraxProof,
    pub w1_batch_open: HyraxProof,
    pub x_norm2_batch_open: HyraxProof,
    pub ffn_m_com_batch_open: HyraxProof,

    // Cross-block batch opens for attention phi_q, phi_k, v (shared eval points)
    pub phi_q_batch_open: HyraxProof,
    pub phi_k_batch_open: HyraxProof,
    pub v_attn_batch_open: HyraxProof,
}

// ---------------------------------------------------------------------------
// Helper: compute powers of a field element
// ---------------------------------------------------------------------------

fn powers_of(base: F, n: usize) -> Vec<F> {
    let mut v = Vec::with_capacity(n);
    let mut cur = F::from(1u64);
    for _ in 0..n {
        v.push(cur);
        cur *= base;
    }
    v
}

// ---------------------------------------------------------------------------
// Model Prover (E2E) — cross-block batch sumcheck Phase 2
// ---------------------------------------------------------------------------

pub fn prove(
    pk: &TransformerModelProvingKey,
    witness: &TransformerModelWitness,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> Result<TransformerModelProof, String> {
    let num_blocks = pk.vk.num_blocks;
    let t = pk.vk.seq_len;
    let d = pk.vk.d_model;
    let v = pk.vk.vocab_size;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;
    let d_ff = pk.block_pks[0].ffn_pk.vk.d_ff;
    let f_bits = d_ff.next_power_of_two().trailing_zeros() as usize;
    let (nu_td, sigma_td, _) = params_from_vars(td_num_vars);
    let (nu_w, sigma_w, _) = params_from_vars(d_bits + d_bits);
    let (nu_b, sigma_b, _) = params_from_vars(d_bits);
    let (nu_wff, sigma_wff, _) = params_from_vars(f_bits + d_bits);
    let (nu_mff, sigma_mff, _) = params_from_vars(t_bits + f_bits);

    let commit_mat = |mat: &[Vec<F>], rows: usize, cols: usize| -> HyraxCommitment {
        let mle = mat_to_mle(mat, rows, cols);
        let vars = rows.next_power_of_two().trailing_zeros()
            + cols.next_power_of_two().trailing_zeros();
        let (nu, _, params) = params_from_vars(vars as usize);
        hyrax_commit(&mle.evaluations, nu, &params)
    };

    // =========================================================================
    // 1. Initial input commitment
    // =========================================================================
    let x_in_com = commit_mat(&witness.x_in, t, d);
    absorb_com(transcript, b"x_in_com", &x_in_com);

    // =========================================================================
    // 2. Phase 1: commit all blocks' intermediates + run LN proofs
    // =========================================================================
    let mut phase1_data: Vec<BlockPhase1Data> = Vec::with_capacity(num_blocks);
    let mut current_x_com = x_in_com.clone();

    for i in 0..num_blocks {
        let p1 = commit_block_phase1(
            &witness.block_witnesses[i],
            &current_x_com,
            &pk.block_pks[i],
            transcript,
        )?;
        let x_mid_com = p1.x_mid_com.clone();
        let next_x_com = add_commitments(&x_mid_com, &p1.out_ffn_com);
        current_x_com = next_x_com;
        phase1_data.push(p1);
    }

    // =========================================================================
    // 3. Derive global r_td after ALL blocks' Phase 1 commitments
    // =========================================================================
    let r_td = challenge_vec(transcript, td_num_vars, b"gkr_r_td");
    let r_t = r_td[..t_bits].to_vec();
    let r_out = r_td[t_bits..].to_vec();

    // =========================================================================
    // 4. Batch QKV: absorb per-block coms, derive lambda/mu, build polys
    // =========================================================================
    let mut fs_qkv: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_qkv: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut qkv_targets: Vec<F> = Vec::with_capacity(num_blocks);
    // Per-block QKV data needed for proof construction
    let mut pb_qkv_lambda: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_qkv_mu: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_q_eval: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_k_eval: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_v_eval: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_bias_q_eval: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_bias_k_eval: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_bias_v_eval: Vec<F> = Vec::with_capacity(num_blocks);

    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let bw = &witness.block_witnesses[i];

        // Absorb QKV weight/bias commitments (mirrors verify_qkv_projections)
        absorb_com(transcript, b"qkv_w_q_com", &bpk.q_pk.vk.w_com);
        absorb_com(transcript, b"qkv_w_k_com", &bpk.k_pk.vk.w_com);
        absorb_com(transcript, b"qkv_w_v_com", &bpk.v_pk.vk.w_com);
        transcript.append_field(b"qkv_alpha_q", &bpk.q_pk.vk.alpha);
        transcript.append_field(b"qkv_alpha_k", &bpk.k_pk.vk.alpha);
        transcript.append_field(b"qkv_alpha_v", &bpk.v_pk.vk.alpha);
        absorb_com(transcript, b"qkv_bias_q_com", &bpk.q_pk.vk.bias_com);
        absorb_com(transcript, b"qkv_bias_k_com", &bpk.k_pk.vk.bias_com);
        absorb_com(transcript, b"qkv_bias_v_com", &bpk.v_pk.vk.bias_com);

        let lambda: F = transcript.challenge_field(b"qkv_lambda");
        let mu: F = transcript.challenge_field(b"qkv_mu");

        // Evaluate outputs and biases at challenge points
        let x_mle = mat_to_mle(&bw.ln1_wit.y, t, d);
        let q_mle = mat_to_mle(&bw.attn_wit.q, t, d);
        let k_mle = mat_to_mle(&bw.attn_wit.k, t, d);
        let v_mle = mat_to_mle(&bw.attn_wit.v, t, d);
        let bias_q_mle = vec_to_mle(&bpk.q_pk.bias, d);
        let bias_k_mle = vec_to_mle(&bpk.k_pk.bias, d);
        let bias_v_mle = vec_to_mle(&bpk.v_pk.bias, d);

        let q_eval = q_mle.evaluate(&combine(&r_t, &r_out));
        let k_eval = k_mle.evaluate(&combine(&r_t, &r_out));
        let v_eval = v_mle.evaluate(&combine(&r_t, &r_out));
        let bias_q_eval = bias_q_mle.evaluate(&r_out);
        let bias_k_eval = bias_k_mle.evaluate(&r_out);
        let bias_v_eval = bias_v_mle.evaluate(&r_out);

        // Build f_i(k) = X_norm1_i[r_t, k]
        let f_x = eval_rows(&x_mle, t_bits, &r_t);

        // Build g_i(k) = lambda*alpha_q*Wq[k,r_out] + mu*alpha_k*Wk[k,r_out] + alpha_v*Wv[k,r_out]
        let g_wq = eval_cols_ternary(&bpk.q_pk.w, &r_out, d, d);
        let g_wk = eval_cols_ternary(&bpk.k_pk.w, &r_out, d, d);
        let g_wv = eval_cols_ternary(&bpk.v_pk.w, &r_out, d, d);
        let n_pad = d.next_power_of_two();
        let g_combined: Vec<F> = (0..n_pad)
            .map(|k| {
                lambda * bpk.q_pk.vk.alpha * g_wq[k]
                    + mu * bpk.k_pk.vk.alpha * g_wk[k]
                    + bpk.v_pk.vk.alpha * g_wv[k]
            })
            .collect();

        let target =
            lambda * (q_eval - bias_q_eval) + mu * (k_eval - bias_k_eval) + (v_eval - bias_v_eval);

        // Bind QKV output evals to transcript (before batch eta challenge)
        transcript.append_field(b"qkv_q_eval", &q_eval);
        transcript.append_field(b"qkv_k_eval", &k_eval);
        transcript.append_field(b"qkv_v_eval", &v_eval);

        fs_qkv.push(DenseMLPoly::from_vec_padded(f_x));
        gs_qkv.push(DenseMLPoly::from_vec_padded(g_combined));
        qkv_targets.push(target);
        pb_qkv_lambda.push(lambda);
        pb_qkv_mu.push(mu);
        pb_q_eval.push(q_eval);
        pb_k_eval.push(k_eval);
        pb_v_eval.push(v_eval);
        pb_bias_q_eval.push(bias_q_eval);
        pb_bias_k_eval.push(bias_k_eval);
        pb_bias_v_eval.push(bias_v_eval);
    }

    // Cross-block QKV batch sumcheck
    let eta_qkv: F = transcript.challenge_field(b"batch_eta_qkv");
    let weights_qkv = powers_of(eta_qkv, num_blocks);
    let claim_qkv: F = weights_qkv
        .iter()
        .zip(qkv_targets.iter())
        .map(|(w, t)| *w * *t)
        .sum();
    let (batch_qkv, r_k_qkv) =
        prove_sumcheck_multi_batched(&fs_qkv, &gs_qkv, &weights_qkv, claim_qkv, transcript);

    // Per-block weight evals at shared (r_k_qkv, r_out)
    let mut pb_w_q_eval: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_w_k_eval: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_w_v_eval: Vec<F> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let wq_mle = mat_to_mle(&convert_tm_to_fm(&bpk.q_pk.w), d, d);
        let wk_mle = mat_to_mle(&convert_tm_to_fm(&bpk.k_pk.w), d, d);
        let wv_mle = mat_to_mle(&convert_tm_to_fm(&bpk.v_pk.w), d, d);
        pb_w_q_eval.push(wq_mle.evaluate(&combine(&r_k_qkv, &r_out)));
        pb_w_k_eval.push(wk_mle.evaluate(&combine(&r_k_qkv, &r_out)));
        pb_w_v_eval.push(wv_mle.evaluate(&combine(&r_k_qkv, &r_out)));
    }

    // =========================================================================
    // 5. Batch O-proj: absorb per-block coms, build polys
    // =========================================================================
    let mut fs_oproj: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_oproj: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut oproj_targets: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_oproj_bias_o_eval: Vec<F> = Vec::with_capacity(num_blocks);

    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let bw = &witness.block_witnesses[i];

        // Absorb O-proj commitments (mirrors prove_projection ordering)
        absorb_com(transcript, b"w_com", &bpk.o_pk.vk.w_com);
        transcript.append_field(b"alpha", &bpk.o_pk.vk.alpha);
        absorb_com(transcript, b"bias_com", &bpk.o_pk.vk.bias_com);
        // x_com = None (GKR backward)

        let x_inner_mle = mat_to_mle(&bw.o_proj_wit.x, t, d);
        let bias_o_mle = vec_to_mle(&bpk.o_pk.bias, d);

        let bias_o_eval = bias_o_mle.evaluate(&r_out);
        let out_attn_eval = mat_to_mle(&bw.o_proj_wit.y, t, d).evaluate(&combine(&r_t, &r_out));

        // f_i(k) = alpha_o * X_inner_i[r_t, k]
        let alpha_o = bpk.o_pk.vk.alpha;
        let f_x_raw = eval_rows(&x_inner_mle, t_bits, &r_t);
        let f_x: Vec<F> = f_x_raw.iter().map(|v| *v * alpha_o).collect();

        // g_i(k) = Wo_i[k, r_out]
        let g_wo = eval_cols_ternary(&bpk.o_pk.w, &r_out, d, d);

        let target = out_attn_eval - bias_o_eval;

        // Bind claimed_y to transcript before eta (mirrors prove_projection)
        transcript.append_field(b"claimed_y", &out_attn_eval);

        fs_oproj.push(DenseMLPoly::from_vec_padded(f_x));
        gs_oproj.push(DenseMLPoly::from_vec_padded(g_wo));
        oproj_targets.push(target);
        pb_oproj_bias_o_eval.push(bias_o_eval);
    }

    // Cross-block O-proj batch sumcheck
    let eta_oproj: F = transcript.challenge_field(b"batch_eta_oproj");
    let weights_oproj = powers_of(eta_oproj, num_blocks);
    let claim_oproj: F = weights_oproj
        .iter()
        .zip(oproj_targets.iter())
        .map(|(w, t)| *w * *t)
        .sum();
    let (batch_oproj, r_k_o) =
        prove_sumcheck_multi_batched(&fs_oproj, &gs_oproj, &weights_oproj, claim_oproj, transcript);

    // Per-block Wo evals at shared (r_k_o, r_out)
    let mut pb_w_o_eval: Vec<F> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let wo_mle = mat_to_mle(&convert_tm_to_fm(&bpk.o_pk.w), d, d);
        pb_w_o_eval.push(wo_mle.evaluate(&combine(&r_k_o, &r_out)));
    }

    // =========================================================================
    // 6. Cross-block Attention batch sumchecks
    //    6a: Commit phi_q/phi_k per block, absorb. Collect out_evals.
    //    6b: Cross-block batch out sumcheck → shared batch_r_attn_out.
    //    6c: Cross-block batch ctx sumcheck → shared batch_r_attn_ctx.
    // =========================================================================

    // 6a. Commit phi_q, phi_k per block and absorb into transcript.
    let mut phi_q_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut phi_k_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut v_mles_attn: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut ctx_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut attn_phi_q_coms: Vec<HyraxCommitment> = Vec::with_capacity(num_blocks);
    let mut attn_phi_k_coms: Vec<HyraxCommitment> = Vec::with_capacity(num_blocks);
    let mut attn_out_evals: Vec<F> = Vec::with_capacity(num_blocks);

    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let bw = &witness.block_witnesses[i];

        let phi_q_mle = mat_to_mle(&bw.attn_wit.phi_q, t, d);
        let phi_k_mle = mat_to_mle(&bw.attn_wit.phi_k, t, d);
        let v_mle = mat_to_mle(&bw.attn_wit.v, t, d);
        let ctx_mle = mat_to_mle(&bw.attn_wit.context, d, d);

        let phi_q_com = commit_mat(&bw.attn_wit.phi_q, t, d);
        let phi_k_com = commit_mat(&bw.attn_wit.phi_k, t, d);
        absorb_com(transcript, b"phi_q_com", &phi_q_com);
        absorb_com(transcript, b"phi_k_com", &phi_k_com);

        // out_i(r_t, r_k_o) = x_inner_i = batch_oproj.final_evals_f[i] / alpha_o
        let alpha_o = bpk.o_pk.vk.alpha;
        let out_eval_i = if alpha_o == F::from(0u64) {
            F::from(0u64)
        } else {
            batch_oproj.final_evals_f[i] * alpha_o.inverse().unwrap()
        };

        phi_q_mles.push(phi_q_mle);
        phi_k_mles.push(phi_k_mle);
        v_mles_attn.push(v_mle);
        ctx_mles.push(ctx_mle);
        attn_phi_q_coms.push(phi_q_com);
        attn_phi_k_coms.push(phi_k_com);
        attn_out_evals.push(out_eval_i);
    }

    // 6b. Batch out sumcheck: out_i(r_t, r_k_o) = Σ_k phi_q_i(r_t,k) · ctx_i(k, r_k_o)
    let mut fs_attn_out: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_attn_out: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        transcript.append_field(b"attn_out_eval", &attn_out_evals[i]);
        let f_out = DenseMLPoly::from_vec_padded(eval_rows(&phi_q_mles[i], t_bits, &r_t));
        let g_out = DenseMLPoly::from_vec_padded(eval_cols(&ctx_mles[i], d_bits, &r_k_o));
        fs_attn_out.push(f_out);
        gs_attn_out.push(g_out);
    }
    let eta_attn_out: F = transcript.challenge_field(b"batch_eta_attn_out");
    let weights_attn_out = powers_of(eta_attn_out, num_blocks);
    let claim_attn_out: F = weights_attn_out
        .iter()
        .zip(attn_out_evals.iter())
        .map(|(w, e)| *w * *e)
        .sum();
    let (batch_attn_out, batch_r_attn_out) = prove_sumcheck_multi_batched(
        &fs_attn_out, &gs_attn_out, &weights_attn_out, claim_attn_out, transcript,
    );
    // final_evals_f[i] = phi_q_i(r_t, batch_r_attn_out)
    // final_evals_g[i] = ctx_i(batch_r_attn_out, r_k_o) = claim for ctx sumcheck
    let attn_phi_q_evals: Vec<F> = batch_attn_out.final_evals_f.clone();
    let attn_ctx_evals: Vec<F> = batch_attn_out.final_evals_g.clone();

    // 6c. Batch ctx sumcheck: ctx_i(batch_r_attn_out, r_k_o) = Σ_t phi_k_i(t, batch_r_attn_out) · v_i(t, r_k_o)
    let mut fs_attn_ctx: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_attn_ctx: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        transcript.append_field(b"attn_ctx_eval", &attn_ctx_evals[i]);
        let f_ctx = DenseMLPoly::from_vec_padded(eval_cols(&phi_k_mles[i], t_bits, &batch_r_attn_out));
        let g_v = DenseMLPoly::from_vec_padded(eval_cols(&v_mles_attn[i], t_bits, &r_k_o));
        fs_attn_ctx.push(f_ctx);
        gs_attn_ctx.push(g_v);
    }
    let eta_attn_ctx: F = transcript.challenge_field(b"batch_eta_attn_ctx");
    let weights_attn_ctx = powers_of(eta_attn_ctx, num_blocks);
    let claim_attn_ctx: F = weights_attn_ctx
        .iter()
        .zip(attn_ctx_evals.iter())
        .map(|(w, e)| *w * *e)
        .sum();
    let (batch_attn_ctx, batch_r_attn_ctx) = prove_sumcheck_multi_batched(
        &fs_attn_ctx, &gs_attn_ctx, &weights_attn_ctx, claim_attn_ctx, transcript,
    );
    // final_evals_f[i] = phi_k_i(batch_r_attn_ctx, batch_r_attn_out)
    // final_evals_g[i] = v_i(batch_r_attn_ctx, r_k_o)
    let attn_phi_k_evals: Vec<F> = batch_attn_ctx.final_evals_f.clone();
    let attn_v_evals: Vec<F> = batch_attn_ctx.final_evals_g.clone();

    // =========================================================================
    // 7. Per-block FFN: Lasso + M commit + absorb coms
    // =========================================================================
    let mut ffn_lasso_proofs: Vec<LassoProof> = Vec::with_capacity(num_blocks);
    let mut ffn_m_coms: Vec<HyraxCommitment> = Vec::with_capacity(num_blocks);
    let mut ffn_m_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);

    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let bw = &witness.block_witnesses[i];

        // Absorb FFN weight commitments (mirrors prove_ffn ordering)
        absorb_com(transcript, b"w1_com", &bpk.ffn_pk.vk.w1_com);
        absorb_com(transcript, b"w2_com", &bpk.ffn_pk.vk.w2_com);
        // x_com = None (GKR mode), y_com = None (already in Phase 1)

        // GKR backward: run Lasso FIRST to commit A before rx_y is sampled
        let ffn_lasso_proof = prove_lasso(
            &inst_ffn.activation_lasso,
            &bw.ffn_wit.activation_query_indices,
            &bpk.ffn_pk.activation_lasso_pk,
            transcript,
            lasso_params,
        );

        // Commit M
        let m_mle = mat_to_mle(&bw.ffn_wit.m, t, d_ff);
        let (nu_m, _, params_m) = params_from_vars(t_bits + f_bits);
        let ffn_m_com = hyrax_commit(&m_mle.evaluations, nu_m, &params_m);
        absorb_com(transcript, b"m_com", &ffn_m_com);

        ffn_lasso_proofs.push(ffn_lasso_proof);
        ffn_m_coms.push(ffn_m_com);
        ffn_m_mles.push(m_mle);
    }

    // =========================================================================
    // 8. Batch FFN-Y: Y = A · W2 at shared (r_t, r_out) = r_td
    // =========================================================================
    let mut fs_ffn_y: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_ffn_y: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut ffn_y_targets: Vec<F> = Vec::with_capacity(num_blocks);

    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let bw = &witness.block_witnesses[i];

        let a_mle = mat_to_mle(&bw.ffn_wit.a, t, d_ff);
        let w2_mle = mat_to_mle(&convert_tm_to_fm(&bpk.ffn_pk.w2), d_ff, d);
        let y_mle = mat_to_mle(&bw.ffn_wit.y, t, d);

        let y_eval = y_mle.evaluate(&combine(&r_t, &r_out));

        // Bind y_eval claim to transcript (mirrors prove_ffn step 4)
        transcript.append_field(b"claim_y", &y_eval);

        // f_i(k) = A_i[r_t, k]
        let f_a = eval_rows(&a_mle, t_bits, &r_t);
        // g_i(k) = W2_i[k, r_out]
        let g_w2 = eval_cols(&w2_mle, f_bits, &r_out);

        fs_ffn_y.push(DenseMLPoly::from_vec_padded(f_a));
        gs_ffn_y.push(DenseMLPoly::from_vec_padded(g_w2));
        ffn_y_targets.push(y_eval);
    }

    // Cross-block FFN-Y batch sumcheck
    let eta_ffn_y: F = transcript.challenge_field(b"batch_eta_ffn_y");
    let weights_ffn_y = powers_of(eta_ffn_y, num_blocks);
    let claim_ffn_y: F = weights_ffn_y
        .iter()
        .zip(ffn_y_targets.iter())
        .map(|(w, t)| *w * *t)
        .sum();
    let (batch_ffn_y, r_k_fy) =
        prove_sumcheck_multi_batched(&fs_ffn_y, &gs_ffn_y, &weights_ffn_y, claim_ffn_y, transcript);

    // =========================================================================
    // 9. Batch FFN-M: M = X2 · W1 with shared rx_m, ry_m
    // =========================================================================
    let rx_m = challenge_vec(transcript, t_bits, b"ffn_rx_m");
    let ry_m = challenge_vec(transcript, f_bits, b"ffn_ry_m");

    let mut fs_ffn_m: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_ffn_m: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut ffn_m_targets: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_ffn_m_eval: Vec<F> = Vec::with_capacity(num_blocks);

    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let bw = &witness.block_witnesses[i];

        let x_norm2_mle = mat_to_mle(&bw.ln2_wit.y, t, d);
        let w1_mle = mat_to_mle(&convert_tm_to_fm(&bpk.ffn_pk.w1), d, d_ff);
        let m_eval = ffn_m_mles[i].evaluate(&combine(&rx_m, &ry_m));

        // Bind m_eval claim to transcript
        transcript.append_field(b"claim_m", &m_eval);

        // f_i(k) = X2_i[rx_m, k]
        let f_x = eval_rows(&x_norm2_mle, t_bits, &rx_m);
        // g_i(k) = W1_i[k, ry_m]
        let g_w1 = eval_cols(&w1_mle, d_bits, &ry_m);

        fs_ffn_m.push(DenseMLPoly::from_vec_padded(f_x));
        gs_ffn_m.push(DenseMLPoly::from_vec_padded(g_w1));
        ffn_m_targets.push(m_eval);
        pb_ffn_m_eval.push(m_eval);
    }

    // Cross-block FFN-M batch sumcheck
    let eta_ffn_m: F = transcript.challenge_field(b"batch_eta_ffn_m");
    let weights_ffn_m = powers_of(eta_ffn_m, num_blocks);
    let claim_ffn_m: F = weights_ffn_m
        .iter()
        .zip(ffn_m_targets.iter())
        .map(|(w, t)| *w * *t)
        .sum();
    let (batch_ffn_m, r_k_m) =
        prove_sumcheck_multi_batched(&fs_ffn_m, &gs_ffn_m, &weights_ffn_m, claim_ffn_m, transcript);

    // =========================================================================
    // 10. Build per-block proof structs
    // =========================================================================
    let mut block_proofs: Vec<TransformerBlockProof> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let p1 = &phase1_data[i];
        let bw = &witness.block_witnesses[i];

        block_proofs.push(TransformerBlockProof {
            ln1_proof: p1.ln1_proof.clone(),
            ln2_proof: p1.ln2_proof.clone(),
            block_range_m: p1.block_range_m.clone(),
            ffn_lasso_proof: ffn_lasso_proofs.remove(0),
            ffn_m_com: ffn_m_coms[i].clone(),
            x_norm1_com: p1.x_norm1_com.clone(),
            q_com: p1.q_com.clone(),
            k_com: p1.k_com.clone(),
            v_com: p1.v_com.clone(),
            out_attn_com: p1.out_attn_com.clone(),
            x_norm2_com: p1.x_norm2_com.clone(),
            out_ffn_com: p1.out_ffn_com.clone(),
            q_eval: pb_q_eval[i],
            k_eval: pb_k_eval[i],
            v_eval_rtd: pb_v_eval[i],
            out_attn_eval: mat_to_mle(&bw.o_proj_wit.y, t, d)
                .evaluate(&combine(&r_t, &r_out)),
            out_ffn_eval: ffn_y_targets[i],
            qkv_lambda: pb_qkv_lambda[i],
            qkv_mu: pb_qkv_mu[i],
            qkv_w_q_eval: pb_w_q_eval[i],
            qkv_w_k_eval: pb_w_k_eval[i],
            qkv_w_v_eval: pb_w_v_eval[i],
            qkv_bias_q_eval: pb_bias_q_eval[i],
            qkv_bias_k_eval: pb_bias_k_eval[i],
            qkv_bias_v_eval: pb_bias_v_eval[i],
            oproj_w_o_eval: pb_w_o_eval[i],
            oproj_bias_o_eval: pb_oproj_bias_o_eval[i],
            ffn_m_eval: pb_ffn_m_eval[i],
            attn_phi_q_com: attn_phi_q_coms[i].clone(),
            attn_phi_k_com: attn_phi_k_coms[i].clone(),
            attn_out_eval: attn_out_evals[i],
            attn_phi_q_eval: attn_phi_q_evals[i],
            attn_phi_k_eval: attn_phi_k_evals[i],
            attn_ctx_eval: attn_ctx_evals[i],
            attn_v_eval: attn_v_evals[i],
        });
    }

    // =========================================================================
    // 11. Final LayerNorm
    // =========================================================================
    let final_rw = compute_range_witnesses(&witness.final_ln_wit, &pk.vk.final_ln_vk);
    let (mut final_range_proofs, final_range_m, final_r_vs) = prove_range_batched(
        &[&final_rw.sigma_witness, &final_rw.y_witness],
        32,
        transcript,
    )?;
    let final_y_rp = final_range_proofs.remove(1);
    let final_sig_rp = final_range_proofs.remove(0);
    let final_ln_out_com = commit_mat(&witness.final_ln_wit.y, t, d);
    let ln_io = LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: Some(final_ln_out_com.clone()),
    };
    let final_ln_proof = prove_layernorm(
        &witness.final_ln_wit,
        &ln_io,
        &pk.vk.final_ln_vk,
        (final_sig_rp, final_r_vs[0].clone()),
        (final_y_rp, final_r_vs[1].clone()),
        transcript,
    )?;

    // =========================================================================
    // 12. LM Head
    // =========================================================================
    let logits_mle = mat_to_mle(&witness.lm_head_wit.y, t, v);
    let logits_com = commit_mat(&witness.lm_head_wit.y, t, v);
    let lm_io = ProjectionIOCommitments { x_com: Some(final_ln_out_com.clone()) };
    let (lm_head_proof, lm_y_claim, _) =
        prove_projection(&pk.lm_head_pk, &witness.lm_head_wit, &lm_io, transcript, None)?;
    let v_bits = v.next_power_of_two().trailing_zeros() as usize;
    let lm_logits_num_vars = t_bits + v_bits;
    let (lm_nu, lm_sigma, _) = params_from_vars(lm_logits_num_vars);
    let lm_head_logits_open =
        hyrax_open(&logits_mle.evaluations, &lm_y_claim.point, lm_nu, lm_sigma);

    // =========================================================================
    // 13. Advance transcript for accumulator mu challenges (10 accumulators)
    // =========================================================================
    for _ in 0..10 {
        let _ = transcript.challenge_field::<F>(b"hyrax_group_mu");
    }

    // =========================================================================
    // 14. Global hyrax_open_batch for 5L intermediate matrices at r_td
    // =========================================================================
    let mut all_evals_vecs: Vec<Vec<F>> = Vec::with_capacity(5 * num_blocks);
    for i in 0..num_blocks {
        let bw = &witness.block_witnesses[i];
        all_evals_vecs.push(mat_to_mle(&bw.attn_wit.q, t, d).evaluations);
        all_evals_vecs.push(mat_to_mle(&bw.attn_wit.k, t, d).evaluations);
        all_evals_vecs.push(mat_to_mle(&bw.attn_wit.v, t, d).evaluations);
        all_evals_vecs.push(mat_to_mle(&bw.o_proj_wit.y, t, d).evaluations);
        all_evals_vecs.push(mat_to_mle(&bw.ffn_wit.y, t, d).evaluations);
    }
    let evals_refs: Vec<&[F]> = all_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let inter_batch_open =
        hyrax_open_batch(&evals_refs, &r_td, nu_td, sigma_td, transcript);

    // =========================================================================
    // 15. Cross-block batch opens
    // =========================================================================

    // x_norm1_batch: L x_norm1_i at combine(r_t, r_k_qkv) [td_num_vars]
    let x_norm1_point = combine(&r_t, &r_k_qkv);
    let x_norm1_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| mat_to_mle(&witness.block_witnesses[i].ln1_wit.y, t, d).evaluations)
        .collect();
    let x_norm1_refs: Vec<&[F]> = x_norm1_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let x_norm1_batch_open =
        hyrax_open_batch(&x_norm1_refs, &x_norm1_point, nu_td, sigma_td, transcript);

    // w_q_batch: L Wq_i at combine(r_k_qkv, r_out) [d_bits + d_bits]
    let wq_point = combine(&r_k_qkv, &r_out);
    let wq_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| mat_to_mle(&convert_tm_to_fm(&pk.block_pks[i].q_pk.w), d, d).evaluations)
        .collect();
    let wq_refs: Vec<&[F]> = wq_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let w_q_batch_open = hyrax_open_batch(&wq_refs, &wq_point, nu_w, sigma_w, transcript);

    let wk_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| mat_to_mle(&convert_tm_to_fm(&pk.block_pks[i].k_pk.w), d, d).evaluations)
        .collect();
    let wk_refs: Vec<&[F]> = wk_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let w_k_batch_open = hyrax_open_batch(&wk_refs, &wq_point, nu_w, sigma_w, transcript);

    let wv_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| mat_to_mle(&convert_tm_to_fm(&pk.block_pks[i].v_pk.w), d, d).evaluations)
        .collect();
    let wv_refs: Vec<&[F]> = wv_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let w_v_batch_open = hyrax_open_batch(&wv_refs, &wq_point, nu_w, sigma_w, transcript);

    // bias_q/k/v batch at r_out [d_bits]
    let bq_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| vec_to_mle(&pk.block_pks[i].q_pk.bias, d).evaluations)
        .collect();
    let bq_refs: Vec<&[F]> = bq_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let bias_q_batch_open = hyrax_open_batch(&bq_refs, &r_out, nu_b, sigma_b, transcript);

    let bk_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| vec_to_mle(&pk.block_pks[i].k_pk.bias, d).evaluations)
        .collect();
    let bk_refs: Vec<&[F]> = bk_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let bias_k_batch_open = hyrax_open_batch(&bk_refs, &r_out, nu_b, sigma_b, transcript);

    let bv_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| vec_to_mle(&pk.block_pks[i].v_pk.bias, d).evaluations)
        .collect();
    let bv_refs: Vec<&[F]> = bv_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let bias_v_batch_open = hyrax_open_batch(&bv_refs, &r_out, nu_b, sigma_b, transcript);

    // w_o_batch: L Wo_i at combine(r_k_o, r_out) [d_bits + d_bits]
    let wo_point = combine(&r_k_o, &r_out);
    let wo_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| mat_to_mle(&convert_tm_to_fm(&pk.block_pks[i].o_pk.w), d, d).evaluations)
        .collect();
    let wo_refs: Vec<&[F]> = wo_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let w_o_batch_open = hyrax_open_batch(&wo_refs, &wo_point, nu_w, sigma_w, transcript);

    // bias_o_batch at r_out
    let bo_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| vec_to_mle(&pk.block_pks[i].o_pk.bias, d).evaluations)
        .collect();
    let bo_refs: Vec<&[F]> = bo_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let bias_o_batch_open = hyrax_open_batch(&bo_refs, &r_out, nu_b, sigma_b, transcript);

    // w2_batch: L W2_i at combine(r_k_fy, r_out) [f_bits + d_bits]
    let w2_point = combine(&r_k_fy, &r_out);
    let w2_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| mat_to_mle(&convert_tm_to_fm(&pk.block_pks[i].ffn_pk.w2), d_ff, d).evaluations)
        .collect();
    let w2_refs: Vec<&[F]> = w2_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let w2_batch_open = hyrax_open_batch(&w2_refs, &w2_point, nu_wff, sigma_wff, transcript);

    // w1_batch: L W1_i at combine(r_k_m, ry_m) [d_bits + f_bits]
    let w1_point = combine(&r_k_m, &ry_m);
    let w1_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| mat_to_mle(&convert_tm_to_fm(&pk.block_pks[i].ffn_pk.w1), d, d_ff).evaluations)
        .collect();
    let w1_refs: Vec<&[F]> = w1_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let w1_batch_open = hyrax_open_batch(&w1_refs, &w1_point, nu_wff, sigma_wff, transcript);

    // x_norm2_batch: L x_norm2_i at combine(rx_m, r_k_m) [td_num_vars]
    let x_norm2_point = combine(&rx_m, &r_k_m);
    let x_norm2_evals_vecs: Vec<Vec<F>> = (0..num_blocks)
        .map(|i| mat_to_mle(&witness.block_witnesses[i].ln2_wit.y, t, d).evaluations)
        .collect();
    let x_norm2_refs: Vec<&[F]> = x_norm2_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let x_norm2_batch_open =
        hyrax_open_batch(&x_norm2_refs, &x_norm2_point, nu_td, sigma_td, transcript);

    // ffn_m_com_batch: L M_i at combine(rx_m, ry_m) [t_bits + f_bits]
    let ffn_m_point = combine(&rx_m, &ry_m);
    let ffn_m_evals_vecs: Vec<Vec<F>> = ffn_m_mles.iter().map(|m| m.evaluations.clone()).collect();
    let ffn_m_refs: Vec<&[F]> = ffn_m_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let ffn_m_com_batch_open =
        hyrax_open_batch(&ffn_m_refs, &ffn_m_point, nu_mff, sigma_mff, transcript);

    // phi_q_batch: L phi_q_i at combine(r_t, batch_r_attn_out) [td_num_vars]
    let phi_q_attn_point = combine(&r_t, &batch_r_attn_out);
    let phi_q_evals_vecs: Vec<Vec<F>> = phi_q_mles.iter().map(|m| m.evaluations.clone()).collect();
    let phi_q_refs: Vec<&[F]> = phi_q_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let phi_q_batch_open =
        hyrax_open_batch(&phi_q_refs, &phi_q_attn_point, nu_td, sigma_td, transcript);

    // phi_k_batch: L phi_k_i at combine(batch_r_attn_ctx, batch_r_attn_out) [td_num_vars]
    let phi_k_attn_point = combine(&batch_r_attn_ctx, &batch_r_attn_out);
    let phi_k_evals_vecs: Vec<Vec<F>> = phi_k_mles.iter().map(|m| m.evaluations.clone()).collect();
    let phi_k_refs: Vec<&[F]> = phi_k_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let phi_k_batch_open =
        hyrax_open_batch(&phi_k_refs, &phi_k_attn_point, nu_td, sigma_td, transcript);

    // v_attn_batch: L v_i at combine(batch_r_attn_ctx, r_k_o) [td_num_vars]
    let v_attn_batch_point = combine(&batch_r_attn_ctx, &r_k_o);
    let v_attn_evals_vecs: Vec<Vec<F>> = v_mles_attn.iter().map(|m| m.evaluations.clone()).collect();
    let v_attn_refs: Vec<&[F]> = v_attn_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let v_attn_batch_open =
        hyrax_open_batch(&v_attn_refs, &v_attn_batch_point, nu_td, sigma_td, transcript);

    // =========================================================================
    // 16. Global batched Lasso (attention)
    // =========================================================================
    let mut all_lasso_instances = Vec::new();
    let mut all_instance_coms = Vec::new();
    let mut all_output_bindings: Vec<LassoOutputBinding> = Vec::new();
    let mut all_query_indices: Vec<Vec<usize>> = Vec::new();
    let mut global_nu = 0usize;
    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        all_lasso_instances.push(inst_attn.q_lasso.clone());
        all_lasso_instances.push(inst_attn.k_lasso.clone());
        all_query_indices.push(inst_attn.q_query_indices.clone());
        all_query_indices.push(inst_attn.k_query_indices.clone());
        all_instance_coms.push(bpk.attn_pk.qk_lasso_pk.instance_table_coms[0].clone());
        all_instance_coms.push(bpk.attn_pk.qk_lasso_pk.instance_table_coms[1].clone());
        global_nu = bpk.attn_pk.qk_lasso_pk.nu;

        all_output_bindings.push(LassoOutputBinding {
            com: block_proofs[i].attn_phi_q_com.clone(),
            num_vars: td_num_vars,
            mle_evals: phi_q_mles[i].evaluations.clone(),
        });
        all_output_bindings.push(LassoOutputBinding {
            com: block_proofs[i].attn_phi_k_com.clone(),
            num_vars: td_num_vars,
            mle_evals: phi_k_mles[i].evaluations.clone(),
        });
    }
    let global_multi_inst = LassoMultiInstance { instances: all_lasso_instances };
    let global_lasso_pk =
        LassoMultiProvingKey { instance_table_coms: all_instance_coms, nu: global_nu };
    let all_lasso_proof = prove_lasso_multi(
        &global_multi_inst,
        &all_query_indices,
        &global_lasso_pk,
        &all_output_bindings,
        transcript,
        lasso_params,
    );

    Ok(TransformerModelProof {
        x_in_com,
        block_proofs,
        final_ln_proof,
        lm_head_proof,
        final_ln_out_com,
        logits_com,
        lm_head_logits_open,
        all_lasso_proof,
        final_range_m,
        batch_qkv,
        batch_oproj,
        batch_ffn_y,
        batch_ffn_m,
        batch_attn_out,
        batch_attn_ctx,
        inter_batch_open,
        x_norm1_batch_open,
        w_q_batch_open,
        w_k_batch_open,
        w_v_batch_open,
        bias_q_batch_open,
        bias_k_batch_open,
        bias_v_batch_open,
        w_o_batch_open,
        bias_o_batch_open,
        w2_batch_open,
        w1_batch_open,
        x_norm2_batch_open,
        ffn_m_com_batch_open,
        phi_q_batch_open,
        phi_k_batch_open,
        v_attn_batch_open,
    })
}

// ---------------------------------------------------------------------------
// Cryptographic Helper: Homomorphic Addition
// ---------------------------------------------------------------------------

pub fn add_commitments_prover(a: &HyraxCommitment, b: &HyraxCommitment) -> HyraxCommitment {
    crate::verifier::add_commitments(a, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::attention::{LinearAttentionInstance, LinearAttentionWitness};
    use crate::attention::layernorm::LayerNormWitness;
    use crate::attention::projection::ProjectionWitness;
    use crate::ffn::ffn::{FFNInstance, FFNWitness};
    use crate::lookup::lasso::LassoInstance;
    use crate::pcs::{hyrax_commit, HyraxParams};
    use crate::poly::utils::{mat_to_mle, TernaryValue};
    use crate::setup::{
        preprocess_transformer_model, TransformerBlockWeights, TransformerModelWeights,
    };
    use crate::transcript::Transcript;
    use crate::verifier::{add_commitments, verify};
    use ark_ff::Field;

    const T: usize = 2;
    const D: usize = 2;
    const D_FF: usize = 4;
    const V: usize = 2;
    const M_BITS: usize = 4;

    fn zero_mat(rows: usize, cols: usize) -> Vec<Vec<F>> {
        vec![vec![F::ZERO; cols]; rows]
    }

    fn zero_ternary_mat(rows: usize, cols: usize) -> Vec<Vec<TernaryValue>> {
        vec![vec![TernaryValue::ZERO; cols]; rows]
    }

    fn build_test_weights() -> TransformerModelWeights {
        let mut q_w = zero_ternary_mat(D, D);
        let mut k_w = zero_ternary_mat(D, D);
        let mut o_w = zero_ternary_mat(D, D);
        let mut ffn_w1 = zero_ternary_mat(D, D_FF);
        let mut ffn_w2 = zero_ternary_mat(D_FF, D);
        let mut lm_head_w = zero_ternary_mat(D, V);

        q_w[0][0] = TernaryValue::ONE;
        k_w[0][0] = TernaryValue::ONE;
        o_w[0][0] = TernaryValue::ONE;
        ffn_w1[0][0] = TernaryValue::ONE;
        ffn_w2[0][0] = TernaryValue::ONE;
        for i in 0..D.min(V) {
            lm_head_w[i][i] = TernaryValue::ONE;
        }

        let identity_table: Vec<F> = (0u64..1 << M_BITS).map(F::from).collect();
        let block = TransformerBlockWeights {
            ln1_gamma: vec![F::from(2u64); D],
            ln1_beta: vec![F::from(5u64); D],
            q_w,
            q_alpha: F::ONE,
            q_bias: vec![F::ZERO; D],
            k_w,
            k_alpha: F::ONE,
            k_bias: vec![F::ZERO; D],
            v_w: zero_ternary_mat(D, D),
            v_alpha: F::ONE,
            v_bias: vec![F::ZERO; D],
            o_w,
            o_alpha: F::ONE,
            o_bias: vec![F::ZERO; D],
            ln2_gamma: vec![F::from(2u64); D],
            ln2_beta: vec![F::from(5u64); D],
            ffn_w1,
            ffn_w2,
            ffn_activation_tables: vec![identity_table.clone()],
            ffn_activation_bits_per_chunk: M_BITS,
            q_activation_tables: vec![identity_table.clone()],
            k_activation_tables: vec![identity_table.clone()],
            qk_activation_bits_per_chunk: M_BITS,
        };

        TransformerModelWeights {
            num_blocks: 1,
            d_model: D,
            d_ff: D_FF,
            vocab_size: V,
            blocks: vec![block],
            final_ln_gamma: vec![F::from(2u64); D],
            final_ln_beta: vec![F::from(5u64); D],
            lm_head_w,
            lm_head_alpha: F::ONE,
            lm_head_bias: vec![F::ZERO; V],
        }
    }

    fn commit_mat_test(mat: &[Vec<F>], rows: usize, cols: usize) -> HyraxCommitment {
        let mle = mat_to_mle(mat, rows, cols);
        let total_vars = rows.next_power_of_two().trailing_zeros() as usize
            + cols.next_power_of_two().trailing_zeros() as usize;
        let (nu, _, params) = crate::pcs::params_from_vars(total_vars);
        hyrax_commit(&mle.evaluations, nu, &params)
    }

    fn build_ln1_witness() -> LayerNormWitness {
        LayerNormWitness {
            x: vec![
                vec![F::from(14u64), F::from(10u64)],
                vec![F::from(20u64), F::from(16u64)],
            ],
            y: vec![
                vec![F::from(7u64), F::from(3u64)],
                vec![F::from(7u64), F::from(3u64)],
            ],
            sum_x: vec![F::from(24u64), F::from(36u64)],
            sigma: vec![F::from(2u64), F::from(2u64)],
            sq_sum_x: vec![F::from(296u64), F::from(656u64)],
            sum_x_sq: vec![F::from(576u64), F::from(1296u64)],
            sigma_sq_scaled: vec![F::from(16u64), F::from(16u64)],
        }
    }

    fn build_ln2_witness() -> LayerNormWitness {
        LayerNormWitness {
            x: vec![
                vec![F::from(14u64), F::from(10u64)],
                vec![F::from(20u64), F::from(16u64)],
            ],
            y: vec![
                vec![F::from(7u64), F::from(3u64)],
                vec![F::from(7u64), F::from(3u64)],
            ],
            sum_x: vec![F::from(24u64), F::from(36u64)],
            sigma: vec![F::from(2u64), F::from(2u64)],
            sq_sum_x: vec![F::from(296u64), F::from(656u64)],
            sum_x_sq: vec![F::from(576u64), F::from(1296u64)],
            sigma_sq_scaled: vec![F::from(16u64), F::from(16u64)],
        }
    }

    fn build_ln_final_witness() -> LayerNormWitness {
        LayerNormWitness {
            x: vec![
                vec![F::from(21u64), F::from(10u64)],
                vec![F::from(27u64), F::from(16u64)],
            ],
            y: vec![
                vec![F::from(7u64), F::from(3u64)],
                vec![F::from(7u64), F::from(3u64)],
            ],
            sum_x: vec![F::from(31u64), F::from(43u64)],
            sigma: vec![F::from(7u64), F::from(7u64)],
            sq_sum_x: vec![F::from(541u64), F::from(985u64)],
            sum_x_sq: vec![F::from(961u64), F::from(1849u64)],
            sigma_sq_scaled: vec![F::from(196u64), F::from(196u64)],
        }
    }

    fn build_lasso(indices: Vec<usize>, outputs: Vec<u64>) -> (LassoInstance, Vec<usize>) {
        let table: Vec<F> = (0u64..1 << M_BITS).map(F::from).collect();
        (
            LassoInstance {
                tables: vec![table],
                outputs: outputs.into_iter().map(F::from).collect(),
                bits_per_chunk: M_BITS,
            },
            indices,
        )
    }

    fn lasso_params() -> HyraxParams {
        HyraxParams::new(M_BITS / 2)
    }

    fn build_block_witness_and_instances() -> (
        TransformerBlockWitness,
        LinearAttentionInstance,
        FFNInstance,
    ) {
        let x_in = vec![
            vec![F::from(14u64), F::from(10u64)],
            vec![F::from(20u64), F::from(16u64)],
        ];
        let ln1_wit = build_ln1_witness();
        let y_norm1 = ln1_wit.y.clone();
        let qk_proj_out = vec![
            vec![F::from(7u64), F::from(0u64)],
            vec![F::from(7u64), F::from(0u64)],
        ];
        let zero_out = vec![
            vec![F::from(0u64), F::from(0u64)],
            vec![F::from(0u64), F::from(0u64)],
        ];
        let q_proj_wit = ProjectionWitness { x: y_norm1.clone(), y: qk_proj_out.clone() };
        let k_proj_wit = ProjectionWitness { x: y_norm1.clone(), y: qk_proj_out.clone() };
        let v_proj_wit = ProjectionWitness { x: y_norm1.clone(), y: zero_out.clone() };
        let attn_wit = LinearAttentionWitness {
            q: qk_proj_out.clone(),
            k: qk_proj_out.clone(),
            v: zero_out.clone(),
            phi_q: qk_proj_out.clone(),
            phi_k: qk_proj_out.clone(),
            context: zero_out.clone(),
            out: zero_out.clone(),
        };
        let o_proj_wit = ProjectionWitness { x: zero_out.clone(), y: zero_out.clone() };
        let x_mid = x_in.clone();
        let ln2_wit = build_ln2_witness();
        let y_norm2 = ln2_wit.y.clone();
        let m_ffn = vec![
            vec![F::from(7u64), F::from(0u64), F::from(0u64), F::from(0u64)],
            vec![F::from(7u64), F::from(0u64), F::from(0u64), F::from(0u64)],
        ];
        let ffn_out = vec![
            vec![F::from(7u64), F::from(0u64)],
            vec![F::from(7u64), F::from(0u64)],
        ];
        let ffn_wit = FFNWitness {
            x: y_norm2,
            m: m_ffn.clone(),
            a: m_ffn.clone(),
            y: ffn_out.clone(),
            activation_query_indices: vec![7, 0, 0, 0, 7, 0, 0, 0],
        };
        let x_out = vec![
            vec![F::from(21u64), F::from(10u64)],
            vec![F::from(27u64), F::from(16u64)],
        ];
        let witness = TransformerBlockWitness {
            x_in: x_in.clone(),
            ln1_wit,
            q_proj_wit,
            k_proj_wit,
            v_proj_wit,
            attn_wit,
            o_proj_wit,
            x_mid,
            ln2_wit,
            ffn_wit,
            x_out,
        };
        let (q_lasso, q_query_indices) = build_lasso(vec![7, 0, 7, 0], vec![7, 0, 7, 0]);
        let (k_lasso, k_query_indices) = build_lasso(vec![7, 0, 7, 0], vec![7, 0, 7, 0]);
        let inst_attn = LinearAttentionInstance {
            seq_len: T,
            d_head: D,
            q_lasso,
            k_lasso,
            q_query_indices,
            k_query_indices,
        };
        let (ffn_lasso, _) =
            build_lasso(vec![7, 0, 0, 0, 7, 0, 0, 0], vec![7, 0, 0, 0, 7, 0, 0, 0]);
        let inst_ffn = FFNInstance { activation_lasso: ffn_lasso };
        (witness, inst_attn, inst_ffn)
    }

    fn build_model_witness(block_wit: TransformerBlockWitness) -> TransformerModelWitness {
        let x_in = block_wit.x_in.clone();
        let final_ln_wit = build_ln_final_witness();
        let y_final = final_ln_wit.y.clone();
        let lm_head_wit = ProjectionWitness { x: y_final.clone(), y: y_final };
        TransformerModelWitness {
            x_in,
            block_witnesses: vec![block_wit],
            final_ln_wit,
            lm_head_wit,
        }
    }

    // -----------------------------------------------------------------------
    // Model-level tests (single block L=1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_prove_verify_full_model_e2e() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_e2e");
        let proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();

        let mut vt = Transcript::new(b"model_e2e");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &model_wit.x_in, &model_wit.lm_head_wit.y, &mut vt, &lp);
        assert!(result.is_ok(), "Model verification failed: {:?}", result.err());
    }

    #[test]
    fn test_model_rejects_tampered_block_ln1_proof() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_ln1");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.block_proofs[0].ln1_proof.openings.sum_x_at_rt += F::ONE;

        let mut vt = Transcript::new(b"model_tamper_ln1");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &model_wit.x_in, &model_wit.lm_head_wit.y, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered LN1 proof");
    }

    #[test]
    fn test_model_rejects_tampered_x_norm1_com() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_xnorm1");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        // Tamper x_norm1_batch_open — opens x_norm1_com at (r_t, r_k_qkv)
        proof.x_norm1_batch_open = proof.inter_batch_open.clone();

        let mut vt = Transcript::new(b"model_tamper_xnorm1");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &model_wit.x_in, &model_wit.lm_head_wit.y, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered x_norm1_batch_open");
    }

    #[test]
    fn test_model_rejects_tampered_x_norm2_com() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_xnorm2");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.x_norm2_batch_open = proof.inter_batch_open.clone();

        let mut vt = Transcript::new(b"model_tamper_xnorm2");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &model_wit.x_in, &model_wit.lm_head_wit.y, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered x_norm2_batch_open");
    }

    #[test]
    fn test_model_rejects_tampered_final_ln_proof() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_final_ln");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.final_ln_proof.openings.sum_x_at_rt += F::ONE;

        let mut vt = Transcript::new(b"model_tamper_final_ln");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &model_wit.x_in, &model_wit.lm_head_wit.y, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered final LN proof");
    }

    #[test]
    fn test_model_rejects_tampered_lm_head_proof() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_lm");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.lm_head_proof.openings.y_eval += F::ONE;

        let mut vt = Transcript::new(b"model_tamper_lm");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &model_wit.x_in, &model_wit.lm_head_wit.y, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered LM head proof");
    }

    #[test]
    fn test_model_rejects_tampered_x_in_com() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_xin");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.x_in_com = commit_mat_test(
            &vec![vec![F::from(1u64), F::from(1u64)], vec![F::from(1u64), F::from(1u64)]],
            T, D,
        );

        let mut vt = Transcript::new(b"model_tamper_xin");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &model_wit.x_in, &model_wit.lm_head_wit.y, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered x_in_com");
    }

    #[test]
    fn test_model_rejects_tampered_batch_qkv_eval() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_qkv_eval");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        // Tamper x_norm1_eval in the batch sumcheck
        proof.batch_qkv.final_evals_f[0] += F::ONE;

        let mut vt = Transcript::new(b"model_tamper_qkv_eval");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &model_wit.x_in, &model_wit.lm_head_wit.y, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered batch_qkv final_evals_f");
    }

    #[test]
    fn test_model_rejects_tampered_fraudulent_ln1_output() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_fraud_ln1");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.block_proofs[0].x_norm1_com =
            commit_mat_test(&vec![vec![F::from(1u64), F::from(2u64)]; T], T, D);

        let mut vt = Transcript::new(b"model_tamper_fraud_ln1");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &model_wit.x_in, &model_wit.lm_head_wit.y, &mut vt, &lp);
        assert!(result.is_err(), "Should reject fraudulent LN1 output commitment");
    }

    #[test]
    fn test_add_commitments_is_homomorphic() {
        use crate::pcs::params_from_vars;
        let a = vec![F::from(3u64), F::from(5u64), F::from(7u64), F::from(11u64)];
        let b = vec![F::from(1u64), F::from(2u64), F::from(3u64), F::from(4u64)];
        let apb: Vec<F> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        let (nu, _sigma, params) = params_from_vars(2);
        let com_a = hyrax_commit(&a, nu, &params);
        let com_b = hyrax_commit(&b, nu, &params);
        let com_apb = hyrax_commit(&apb, nu, &params);
        let com_sum = add_commitments(&com_a, &com_b);
        assert_eq!(com_sum.row_coms, com_apb.row_coms, "Com(a) + Com(b) must equal Com(a+b)");
    }

    #[test]
    fn test_add_commitments_with_zero_is_identity() {
        use crate::pcs::params_from_vars;
        let a = vec![F::from(5u64), F::from(9u64), F::from(2u64), F::from(14u64)];
        let zero = vec![F::ZERO; 4];
        let (nu, _sigma, params) = params_from_vars(2);
        let com_a = hyrax_commit(&a, nu, &params);
        let com_zero = hyrax_commit(&zero, nu, &params);
        let com_sum = add_commitments(&com_a, &com_zero);
        assert_eq!(com_sum.row_coms, com_a.row_coms, "Com(a) + Com(0) must equal Com(a)");
    }
}
