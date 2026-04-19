//! Global Verifier for a full Transformer Block.
//!
//! **GKR-style Two-Phase Architecture:**
//! Phase 1: verify range proofs + LN1 + LN2; absorb intermediate commitments into transcript.
//! Phase 2 (after r_td derived): verify QKV/O-proj/Attn/FFN sumchecks using shared r_td.
//! Global: ONE hyrax_verify_batch at r_td for all 5L intermediate matrices.

use crate::field::F;
use crate::lookup::lasso::{verify_lasso_multi, LassoMultiInstance, LassoMultiVerifyingKey};
use crate::lookup::range::verify_range_batched;
use crate::pcs::{
    absorb_com, hyrax_verify, hyrax_verify_batch, params_from_vars, HyraxBatchAccumulator,
    HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::transcript::{challenge_vec, Transcript};

use crate::attention::attention::{
    verify_linear_attention, AttentionIOCommitments, AttentionProvingKey,
    LinearAttentionInstance,
};
use crate::attention::layernorm::{
    verify_layernorm, LayerNormIOCommitments, LayerNormVerifyingKey,
};
use crate::attention::projection::{
    verify_projection, verify_qkv_projections, BatchedQKVProjectionIOCommitments,
    ProjectionIOCommitments, ProjectionProvingKey, ProjectionVerifyingKey,
};
use crate::ffn::ffn::{verify_ffn, FFNInstance, FFNIOCommitments, FFNProvingKey, FFNVerifyingKey};
use ark_ec::{AffineRepr, CurveGroup};
use std::ops::AddAssign;

use crate::prover::{TransformerBlockProof, TransformerModelProof, TransformerModelVerifyingKey};

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
// Block Verifier (self-contained, for testing)
// ---------------------------------------------------------------------------

/// Verify a single transformer block. Derives r_td locally (mirrors prove_transformer_block).
/// `inter_batch_open` is the batch Hyrax proof for the 5 intermediate matrices at r_td.
pub fn verify_transformer_block(
    proof: &TransformerBlockProof,
    inter_batch_open: &HyraxProof,
    x_in_com: &HyraxCommitment,
    x_out_com: &HyraxCommitment,
    vk: &TransformerBlockVerifyingKey,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
    ln_acc_t: &mut HyraxBatchAccumulator,
    ln_acc_td: &mut HyraxBatchAccumulator,
    proj_acc_w: &mut HyraxBatchAccumulator,
    proj_acc_b: &mut HyraxBatchAccumulator,
    acc_range_sig: &mut HyraxBatchAccumulator,
    acc_range_y: &mut HyraxBatchAccumulator,
    acc_range_m: &mut HyraxBatchAccumulator,
    inter_acc: &mut HyraxBatchAccumulator,
) -> Result<(), String> {
    let t = vk.seq_len;
    let d = vk.d_model;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;
    let (nu_td, sigma_td, params_td) = params_from_vars(td_num_vars);

    // =========================================================================
    // Phase 1: range proofs + LN1 + explicit absorptions + LN2 + absorb out_ffn
    // =========================================================================

    let ln_sigma_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let ln_y_n = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
    let (block_r_vs, _block_r_m) = verify_range_batched(
        &[
            &proof.ln1_proof.sigma_range_proof,
            &proof.ln1_proof.y_range_proof,
            &proof.ln2_proof.sigma_range_proof,
            &proof.ln2_proof.y_range_proof,
        ],
        &proof.block_range_m,
        &[ln_sigma_n, ln_y_n, ln_sigma_n, ln_y_n],
        32,
        transcript,
        acc_range_sig,
        acc_range_y,
        acc_range_m,
    )?;
    let ln1_sig_rv = &block_r_vs[0];
    let ln1_y_rv = &block_r_vs[1];
    let ln2_sig_rv = &block_r_vs[2];
    let ln2_y_rv = &block_r_vs[3];

    // LN1 (absorbs x_norm1_com as y_com)
    let ln1_io = LayerNormIOCommitments {
        x_com: x_in_com.clone(),
        y_com: Some(proof.x_norm1_com.clone()),
    };
    verify_layernorm(
        &proof.ln1_proof, &ln1_io, &vk.ln1_vk,
        ln1_sig_rv, ln1_y_rv, transcript, ln_acc_t, ln_acc_td,
    )?;

    // Absorb q/k/v_com (mirrors Phase 1 prover: same labels as attention uses)
    absorb_com(transcript, b"q_com", &proof.q_com);
    absorb_com(transcript, b"k_com", &proof.k_com);
    absorb_com(transcript, b"v_com", &proof.v_com);
    absorb_com(transcript, b"out_attn_com", &proof.out_attn_com);

    let x_mid_com = add_commitments(x_in_com, &proof.out_attn_com);

    // LN2 (absorbs x_norm2_com as y_com)
    let ln2_io = LayerNormIOCommitments {
        x_com: x_mid_com.clone(),
        y_com: Some(proof.x_norm2_com.clone()),
    };
    verify_layernorm(
        &proof.ln2_proof, &ln2_io, &vk.ln2_vk,
        ln2_sig_rv, ln2_y_rv, transcript, ln_acc_t, ln_acc_td,
    )?;

    // Absorb out_ffn_com (mirrors Phase 1 prover)
    absorb_com(transcript, b"y_com", &proof.out_ffn_com);

    // Derive local r_td (mirrors prove_transformer_block)
    let r_td = challenge_vec(transcript, td_num_vars, b"gkr_r_td");

    // =========================================================================
    // Phase 2: sumchecks using r_td
    // =========================================================================

    // QKV projections: x_com=None
    let qkv_io = BatchedQKVProjectionIOCommitments { x_com: None };
    let (q_y_claim, k_y_claim, v_y_claim, x_norm1_claim) = verify_qkv_projections(
        &proof.qkv_proj_proof, &vk.q_vk, &vk.k_vk, &vk.v_vk,
        &qkv_io, transcript, proj_acc_w, proj_acc_b, &r_td,
    )?;

    // O-proj: x_com=None
    let o_io = ProjectionIOCommitments { x_com: None };
    let (o_y_claim, o_x_claim) = verify_projection(
        &proof.o_proj_proof, &vk.o_vk, &o_io, transcript, proj_acc_w, proj_acc_b, Some(&r_td),
    )?;

    // Attention: skip_io_absorb=true
    let attn_io = AttentionIOCommitments {
        q_com: proof.q_com.clone(),
        k_com: proof.k_com.clone(),
        v_com: proof.v_com.clone(),
        skip_io_absorb: true,
    };
    let (_attn_out_claim, attn_v_claim) = verify_linear_attention(
        &proof.attn_proof, inst_attn, &attn_io, Some(o_x_claim.clone()), transcript,
    )?;

    // FFN: x_com=Some(x_norm2_com), y_com=None
    let ffn_io = FFNIOCommitments {
        x_com: Some(proof.x_norm2_com.clone()),
        y_com: None,
    };
    let (ffn_y_claim, ffn_x_claim) = verify_ffn(
        &proof.ffn_proof, inst_ffn, &vk.ffn_vk, &ffn_io, transcript, lasso_params, Some(&r_td),
    )?;

    // =========================================================================
    // Residual Connection 2 check
    // =========================================================================
    let expected_x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);
    if expected_x_out_com.row_coms != x_out_com.row_coms {
        return Err("Transformer Block Output Commitment (Residual) Mismatch!".into());
    }

    // =========================================================================
    // Per-block deferred opens (at block-specific eval points)
    // =========================================================================
    inter_acc
        .add_verify(&proof.x_norm1_com, x_norm1_claim.value, &x_norm1_claim.point, &proof.x_norm1_open)
        .map_err(|e| format!("x_norm1 inter: {e}"))?;
    inter_acc
        .add_verify(&proof.x_norm2_com, ffn_x_claim.value, &ffn_x_claim.point, &proof.x_norm2_open)
        .map_err(|e| format!("x_norm2 inter: {e}"))?;
    inter_acc
        .add_verify(&proof.v_com, proof.v_attn_eval, &attn_v_claim.point, &proof.v_attn_open)
        .map_err(|e| format!("v_attn inter: {e}"))?;

    // =========================================================================
    // Global batch open at r_td for 5 intermediate matrices (single-block mode)
    // =========================================================================
    hyrax_verify_batch(
        &[
            proof.q_com.clone(),
            proof.k_com.clone(),
            proof.v_com.clone(),
            proof.out_attn_com.clone(),
            proof.out_ffn_com.clone(),
        ],
        &[proof.q_eval, proof.k_eval, proof.v_eval_rtd, proof.out_attn_eval, proof.out_ffn_eval],
        &r_td,
        inter_batch_open,
        &params_td,
        transcript,
    )
    .map_err(|e| format!("inter_batch_open: {e}"))?;

    // Sanity: sumcheck output evals must match what's stored in proof
    if q_y_claim.value != proof.q_eval {
        return Err("q_eval mismatch between sumcheck and proof".into());
    }
    if k_y_claim.value != proof.k_eval {
        return Err("k_eval mismatch between sumcheck and proof".into());
    }
    if v_y_claim.value != proof.v_eval_rtd {
        return Err("v_eval_rtd mismatch between sumcheck and proof".into());
    }
    if o_y_claim.value != proof.out_attn_eval {
        return Err("out_attn_eval mismatch between sumcheck and proof".into());
    }
    if ffn_y_claim.value != proof.out_ffn_eval {
        return Err("out_ffn_eval mismatch between sumcheck and proof".into());
    }

    Ok(())
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

// ---------------------------------------------------------------------------
// Model Verifier (E2E)
// ---------------------------------------------------------------------------

pub fn verify(
    proof: &TransformerModelProof,
    vk: &TransformerModelVerifyingKey,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
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
    let params_qkvo_w = params_from_vars(qkvo_in_bits + qkvo_out_bits).2;
    let params_qkvo_b = params_from_vars(qkvo_out_bits).2;
    let lmh_in_bits = vk.lm_head_vk.d_in.next_power_of_two().trailing_zeros() as usize;
    let lmh_out_bits = vk.lm_head_vk.d_out.next_power_of_two().trailing_zeros() as usize;
    let params_lmh_w = params_from_vars(lmh_in_bits + lmh_out_bits).2;
    let params_lmh_b = params_from_vars(lmh_out_bits).2;

    let ln_sig_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let ln_y_n_global = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_range_sig) = params_from_vars(ln_sig_n);
    let (_, _, params_range_y) = params_from_vars(ln_y_n_global);
    let (_, _, params_range_m) =
        params_from_vars(crate::lookup::range::CHUNK_BITS);

    let mut ln_acc_t = HyraxBatchAccumulator::new();
    let mut ln_acc_td = HyraxBatchAccumulator::new();
    let mut proj_acc_w = HyraxBatchAccumulator::new();
    let mut proj_acc_b = HyraxBatchAccumulator::new();
    let mut lmh_acc_w = HyraxBatchAccumulator::new();
    let mut lmh_acc_b = HyraxBatchAccumulator::new();
    let mut acc_range_sig = HyraxBatchAccumulator::new();
    let mut acc_range_y = HyraxBatchAccumulator::new();
    let mut acc_range_m = HyraxBatchAccumulator::new();
    // inter_acc: per-block x_norm1/x_norm2/v_attn opens (3L entries, not 7L)
    let mut inter_acc = HyraxBatchAccumulator::new();

    // 2. Phase 1: verify range proofs + LN1 + LN2 for all blocks
    let mut current_x_com = proof.x_in_com.clone();

    for i in 0..vk.num_blocks {
        let bp = &proof.block_proofs[i];
        let bvk = &vk.block_vks[i];
        let ln_sigma_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
        let ln_y_n = (2 * t * d).next_power_of_two().trailing_zeros() as usize;

        let _t = Instant::now();
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
        eprintln!("[block {}] range_batch:{:>8.3}ms", i, _t.elapsed().as_secs_f64()*1000.0);

        let ln1_sig_rv = &block_r_vs[0];
        let ln1_y_rv = &block_r_vs[1];
        let ln2_sig_rv = &block_r_vs[2];
        let ln2_y_rv = &block_r_vs[3];

        // LN1
        let ln1_io = LayerNormIOCommitments {
            x_com: current_x_com.clone(),
            y_com: Some(bp.x_norm1_com.clone()),
        };
        let _t = Instant::now();
        verify_layernorm(
            &bp.ln1_proof, &ln1_io, &bvk.ln1_vk,
            ln1_sig_rv, ln1_y_rv, transcript, &mut ln_acc_t, &mut ln_acc_td,
        )?;
        eprintln!("[block {}] ln1:{:>8.3}ms", i, _t.elapsed().as_secs_f64()*1000.0);

        // Absorb q/k/v/out_attn_com (same order as prover Phase 1)
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
        let _t = Instant::now();
        verify_layernorm(
            &bp.ln2_proof, &ln2_io, &bvk.ln2_vk,
            ln2_sig_rv, ln2_y_rv, transcript, &mut ln_acc_t, &mut ln_acc_td,
        )?;
        eprintln!("[block {}] ln2:{:>8.3}ms", i, _t.elapsed().as_secs_f64()*1000.0);

        // Absorb out_ffn_com
        absorb_com(transcript, b"y_com", &bp.out_ffn_com);

        let next_x_com = add_commitments(&x_mid_com, &bp.out_ffn_com);
        current_x_com = next_x_com;
    }

    // 3. Derive global r_td after all Phase 1
    let r_td = challenge_vec(transcript, td_num_vars, b"gkr_r_td");

    // 4. Phase 2: verify sumchecks for all blocks using shared r_td
    for i in 0..vk.num_blocks {
        let bp = &proof.block_proofs[i];
        let bvk = &vk.block_vks[i];

        // QKV: x_com=None
        let qkv_io = BatchedQKVProjectionIOCommitments { x_com: None };
        let _t = Instant::now();
        let (q_y_claim, k_y_claim, v_y_claim, x_norm1_claim) = verify_qkv_projections(
            &bp.qkv_proj_proof, &bvk.q_vk, &bvk.k_vk, &bvk.v_vk,
            &qkv_io, transcript, &mut proj_acc_w, &mut proj_acc_b, &r_td,
        )?;
        eprintln!("[block {}] qkv:{:>8.3}ms", i, _t.elapsed().as_secs_f64()*1000.0);

        // O-proj: x_com=None, r_td
        let o_io = ProjectionIOCommitments { x_com: None };
        let _t = Instant::now();
        let (o_y_claim, o_x_claim) = verify_projection(
            &bp.o_proj_proof, &bvk.o_vk, &o_io, transcript,
            &mut proj_acc_w, &mut proj_acc_b, Some(&r_td),
        )?;
        eprintln!("[block {}] o_proj:{:>8.3}ms", i, _t.elapsed().as_secs_f64()*1000.0);

        // Attention: skip_io_absorb=true
        let attn_io = AttentionIOCommitments {
            q_com: bp.q_com.clone(),
            k_com: bp.k_com.clone(),
            v_com: bp.v_com.clone(),
            skip_io_absorb: true,
        };
        let _t = Instant::now();
        let (_attn_out_claim, attn_v_claim) = verify_linear_attention(
            &bp.attn_proof, inst_attn, &attn_io, Some(o_x_claim.clone()), transcript,
        )?;
        eprintln!("[block {}] attn:{:>8.3}ms", i, _t.elapsed().as_secs_f64()*1000.0);

        // FFN: x_com=Some(x_norm2_com), y_com=None
        let ffn_io = FFNIOCommitments {
            x_com: Some(bp.x_norm2_com.clone()),
            y_com: None,
        };
        let _t = Instant::now();
        let (ffn_y_claim, ffn_x_claim) = verify_ffn(
            &bp.ffn_proof, inst_ffn, &bvk.ffn_vk, &ffn_io,
            transcript, lasso_params, Some(&r_td),
        )?;
        eprintln!("[block {}] ffn:{:>8.3}ms", i, _t.elapsed().as_secs_f64()*1000.0);

        // Sanity checks: sumcheck outputs must match stored evals
        if q_y_claim.value != bp.q_eval {
            return Err(format!("Block {i}: q_eval mismatch"));
        }
        if k_y_claim.value != bp.k_eval {
            return Err(format!("Block {i}: k_eval mismatch"));
        }
        if v_y_claim.value != bp.v_eval_rtd {
            return Err(format!("Block {i}: v_eval_rtd mismatch"));
        }
        if o_y_claim.value != bp.out_attn_eval {
            return Err(format!("Block {i}: out_attn_eval mismatch"));
        }
        if ffn_y_claim.value != bp.out_ffn_eval {
            return Err(format!("Block {i}: out_ffn_eval mismatch"));
        }

        // Per-block opens at block-specific eval points → inter_acc
        inter_acc
            .add_verify(&bp.x_norm1_com, x_norm1_claim.value, &x_norm1_claim.point, &bp.x_norm1_open)
            .map_err(|e| format!("Block {i} x_norm1: {e}"))?;
        inter_acc
            .add_verify(&bp.x_norm2_com, ffn_x_claim.value, &ffn_x_claim.point, &bp.x_norm2_open)
            .map_err(|e| format!("Block {i} x_norm2: {e}"))?;
        inter_acc
            .add_verify(&bp.v_com, bp.v_attn_eval, &attn_v_claim.point, &bp.v_attn_open)
            .map_err(|e| format!("Block {i} v_attn: {e}"))?;
    }

    // 5. Final LayerNorm
    let final_sigma_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let final_y_n = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
    let _t = Instant::now();
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
    eprintln!("[model] final_range:{:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    let _t = Instant::now();
    let ln_io = LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: Some(proof.final_ln_out_com.clone()),
    };
    verify_layernorm(
        &proof.final_ln_proof, &ln_io, &vk.final_ln_vk,
        &final_r_vs[0], &final_r_vs[1], transcript, &mut ln_acc_t, &mut ln_acc_td,
    )
    .map_err(|e| format!("Final LN: {e}"))?;
    eprintln!("[model] final_ln:{:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // 6. LM Head
    let _t = Instant::now();
    let lm_io = ProjectionIOCommitments { x_com: Some(proof.final_ln_out_com.clone()) };
    let (lm_y_claim, _) = verify_projection(
        &proof.lm_head_proof, &vk.lm_head_vk, &lm_io, transcript,
        &mut lmh_acc_w, &mut lmh_acc_b, None,
    )
    .map_err(|e| format!("LM Head: {e}"))?;
    eprintln!("[model] lm_head:{:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // Verify logits_com opens to lm_y_claim
    let v_bits = v_vocab.next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_logits) = params_from_vars(t_bits + v_bits);
    hyrax_verify(
        &proof.logits_com, lm_y_claim.value, &lm_y_claim.point,
        &proof.lm_head_logits_open, &params_logits,
    )
    .map_err(|e| format!("Logits commit: {e}"))?;

    // 7. Finalize 10 accumulators in parallel (derive all mu challenges first)
    let _tacc = Instant::now();
    let mu_inter = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_ln_t = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_ln_td = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_proj_w = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_proj_b = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_lmh_w = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_lmh_b = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_rng_sig = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_rng_y = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_rng_m = transcript.challenge_field::<F>(b"hyrax_group_mu");

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

    // 8. Global batch verification for all 5L intermediate matrices at r_td
    let _t = Instant::now();
    let (nu_td2, _, params_td2) = params_from_vars(td_num_vars);
    let _ = nu_td2;
    let mut all_coms: Vec<HyraxCommitment> = Vec::with_capacity(5 * vk.num_blocks);
    let mut all_evals: Vec<F> = Vec::with_capacity(5 * vk.num_blocks);
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
        &all_coms, &all_evals, &r_td, &proof.inter_batch_open, &params_td2, transcript,
    )
    .map_err(|e| format!("Global inter_batch: {e}"))?;
    eprintln!("[model] inter_batch:{:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // 9. Global batched Lasso
    let _t = Instant::now();
    let mut all_lasso_instances = Vec::new();
    let mut all_instance_coms = Vec::new();
    let mut all_output_coms: Vec<(HyraxCommitment, usize)> = Vec::new();
    for i in 0..vk.num_blocks {
        let bvk = &vk.block_vks[i];
        all_lasso_instances.push(inst_attn.q_lasso.clone());
        all_lasso_instances.push(inst_attn.k_lasso.clone());
        all_instance_coms.push(bvk.attn_pk.qk_lasso_pk.instance_table_coms[0].clone());
        all_instance_coms.push(bvk.attn_pk.qk_lasso_pk.instance_table_coms[1].clone());
        all_output_coms.push((proof.block_proofs[i].attn_proof.phi_q_com.clone(), td_num_vars));
        all_output_coms.push((proof.block_proofs[i].attn_proof.phi_k_com.clone(), td_num_vars));
    }
    let global_multi_inst = LassoMultiInstance { instances: all_lasso_instances };
    let global_lasso_vk = LassoMultiVerifyingKey { instance_table_coms: all_instance_coms };
    verify_lasso_multi(
        &proof.all_lasso_proof, &global_multi_inst, &global_lasso_vk,
        &all_output_coms, transcript, lasso_params,
    )
    .map_err(|e| format!("Global Lasso: {e}"))?;
    eprintln!("[model] lasso:{:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    Ok(())
}
