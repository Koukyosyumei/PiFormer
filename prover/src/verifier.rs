//! Global Verifier for a full Transformer Block.
//!
//! **Production-Grade Succinctness:**
//! 1. Strict O(√N) runtime. No matrices are constructed. No O(N) operations exist.
//! 2. Homomorphic Residuals: X_mid and X_out are verified by directly adding
//!    their Pedersen (Hyrax) commitments. No additional proofs needed!
//! 3. Commitment Chaining: Intermediate IO commitments passed from the Prover
//!    are cryptographically bound across adjacent sub-verifiers.

use crate::field::F;
use crate::lookup::lasso::{verify_lasso_multi, LassoMultiInstance, LassoMultiVerifyingKey};
use crate::lookup::range::verify_range_batched;
use crate::pcs::{absorb_com, hyrax_verify, params_from_vars, HyraxBatchAccumulator, HyraxCommitment, HyraxParams};
use crate::subprotocols::verify_combine_deferred;
use crate::transcript::Transcript;

// Sub-module keys and verifiers
use crate::attention::attention::{
    verify_linear_attention, AttentionProvingKey, LinearAttentionInstance,
};
use crate::attention::layernorm::{
    verify_layernorm, LayerNormIOCommitments, LayerNormVerifyingKey,
};
use crate::attention::projection::{
    verify_projection, verify_qkv_projections, BatchedQKVProjectionIOCommitments,
    ProjectionIOCommitments, ProjectionProvingKey, ProjectionVerifyingKey,
};
use crate::ffn::ffn::{verify_ffn, FFNInstance, FFNProvingKey, FFNVerifyingKey};
use ark_ec::{AffineRepr, CurveGroup}; // Arkworks 0.4.0+ の正しいトレイト
use std::ops::AddAssign;

use crate::prover::{TransformerBlockProof, TransformerModelProof, TransformerModelVerifyingKey}; // Imported from prover.rs

// ---------------------------------------------------------------------------
// Global Verifying Key
// ---------------------------------------------------------------------------

/// Contains ALL static weight commitments for one block.
/// Loaded ONCE offline. O(1) size regardless of model depth.
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

    // Proving keys are attached here just for the Prover reference in real code,
    // but the Verifier only reads the `_vk` fields.
    pub q_pk: ProjectionProvingKey,
    pub k_pk: ProjectionProvingKey,
    pub v_pk: ProjectionProvingKey,
    pub o_pk: ProjectionProvingKey,
    pub ffn_pk: FFNProvingKey,
    pub attn_pk: AttentionProvingKey,
}

// ---------------------------------------------------------------------------
// Verifier Implementation
// ---------------------------------------------------------------------------

pub fn verify_transformer_block(
    proof: &TransformerBlockProof,
    x_in_com: &HyraxCommitment,  // Output from the previous block!
    x_out_com: &HyraxCommitment, // The expected output to pass to the next block!
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
    // Layer-folding accumulator: collects all 7 intermediate opens across all blocks.
    // Caller must finalize once after all blocks are verified (2 MSMs total vs 2L).
    inter_acc: &mut HyraxBatchAccumulator,
) -> Result<(), String> {
    // =========================================================================
    // Pipeline Stitching (The Binding of IO Commitments)
    // =========================================================================

    use std::time::Instant;
    let t_bits = vk.seq_len.next_power_of_two().trailing_zeros() as usize;
    let d_bits = vk.d_model.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;

    // --- 0. Global range batch for all 4 range proofs in this block ---
    //    Transcript ordering must match prove_transformer_block: range batch first.
    let t = vk.seq_len;
    let d = vk.d_model;
    let ln_sigma_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let ln_y_n = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
    let _t = Instant::now();
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
    eprintln!("[block] range_batch:{:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);
    let ln1_sig_rv = &block_r_vs[0];
    let ln1_y_rv   = &block_r_vs[1];
    let ln2_sig_rv = &block_r_vs[2];
    let ln2_y_rv   = &block_r_vs[3];

    // --- 1. LayerNorm 1 ---
    let ln1_io = LayerNormIOCommitments {
        x_com: x_in_com.clone(),
        y_com: proof.x_norm1_com.clone(),
    };
    let _t = Instant::now();
    verify_layernorm(&proof.ln1_proof, &ln1_io, &vk.ln1_vk, ln1_sig_rv, ln1_y_rv, transcript, ln_acc_t, ln_acc_td)?;
    eprintln!("[block] ln1:        {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // --- 2. Batched Q, K, V Projections — single sumcheck, single r_k ---
    let qkv_io = BatchedQKVProjectionIOCommitments { x_com: proof.x_norm1_com.clone() };
    let _t = Instant::now();
    let (q_y_claim, k_y_claim, v_y_claim, x_norm1_claim) = verify_qkv_projections(
        &proof.qkv_proj_proof, &vk.q_vk, &vk.k_vk, &vk.v_vk,
        &qkv_io, transcript, proj_acc_w, proj_acc_b,
    )?;
    eprintln!("[block] qkv_proj:   {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // --- 3. Output Projection (GKR backward: runs before attention) ---
    // out_inner is not committed; O_proj's x_claim serves as the shared eval point.
    let o_io = ProjectionIOCommitments { x_com: None };
    let _t = Instant::now(); let (o_y_claim, o_x_claim) = verify_projection(&proof.o_proj_proof, &vk.o_vk, &o_io, transcript, proj_acc_w, proj_acc_b)?;
    eprintln!("[block] o_proj:     {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // --- 4. Linear Attention — receives O_proj's x_claim as external out_inner claim ---
    // out_com removed from IO; both sumchecks reference the same out_inner eval point.
    let attn_io = crate::attention::attention::AttentionIOCommitments {
        q_com: proof.q_com.clone(),
        k_com: proof.k_com.clone(),
        v_com: proof.v_com.clone(),
    };
    let _t = Instant::now(); let (_attn_out_claim, attn_v_claim) =
        verify_linear_attention(&proof.attn_proof, inst_attn, &attn_io, Some(o_x_claim.clone()), transcript)?;
    eprintln!("[block] attn:       {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // Residual Connection 1: X_mid = X_in + Out_attn (homomorphic, no transcript)
    // =========================================================================
    let x_mid_com = add_commitments(x_in_com, &proof.out_attn_com);

    // --- 5. LayerNorm 2 ---
    let ln2_io = LayerNormIOCommitments {
        x_com: x_mid_com.clone(),
        y_com: proof.x_norm2_com.clone(),
    };
    let _t = Instant::now();
    verify_layernorm(&proof.ln2_proof, &ln2_io, &vk.ln2_vk, ln2_sig_rv, ln2_y_rv, transcript, ln_acc_t, ln_acc_td)?;
    eprintln!("[block] ln2:        {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // --- 6. FFN — returns (y_claim, x_claim) deferred to combine ---
    let ffn_io = crate::ffn::ffn::FFNIOCommitments {
        x_com: proof.x_norm2_com.clone(),
        y_com: proof.out_ffn_com.clone(),
    };
    let _t = Instant::now(); let (ffn_y_claim, ffn_x_claim) = verify_ffn(
        &proof.ffn_proof,
        inst_ffn,
        &vk.ffn_vk,
        &ffn_io,
        transcript,
        lasso_params,
    )?;
    eprintln!("[block] ffn:        {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // --- 7. Layer-folded intermediate opens ---
    // All 7 per-block openings (6 direct + 1 deferred combine) are accumulated into
    // `inter_acc` instead of being batch-verified immediately with hyrax_verify_multi_point.
    // The caller finalizes inter_acc once after all blocks, reducing 2L MSMs → 2 MSMs.
    let _t = Instant::now();
    let (v_r, v_f) = verify_combine_deferred(
        &proof.v_combine, &proof.v_com,
        &[v_y_claim.clone(), attn_v_claim.clone()], td_num_vars, transcript,
    ).map_err(|e| format!("v_combine: {e}"))?;
    inter_acc.add_verify(&proof.q_com,        q_y_claim.value,     &q_y_claim.point,     &proof.q_open)
        .map_err(|e| format!("q_open inter: {e}"))?;
    inter_acc.add_verify(&proof.k_com,        k_y_claim.value,     &k_y_claim.point,     &proof.k_open)
        .map_err(|e| format!("k_open inter: {e}"))?;
    inter_acc.add_verify(&proof.x_norm1_com,  x_norm1_claim.value, &x_norm1_claim.point, &proof.x_norm1_open)
        .map_err(|e| format!("x_norm1 inter: {e}"))?;
    inter_acc.add_verify(&proof.out_attn_com, o_y_claim.value,     &o_y_claim.point,     &proof.out_attn_open)
        .map_err(|e| format!("out_attn inter: {e}"))?;
    inter_acc.add_verify(&proof.x_norm2_com,  ffn_x_claim.value,   &ffn_x_claim.point,   &proof.x_norm2_open)
        .map_err(|e| format!("x_norm2 inter: {e}"))?;
    inter_acc.add_verify(&proof.out_ffn_com,  ffn_y_claim.value,   &ffn_y_claim.point,   &proof.out_ffn_open)
        .map_err(|e| format!("out_ffn inter: {e}"))?;
    inter_acc.add_verify(&proof.v_com,        v_f,                 &v_r,                 &proof.v_combine.hyrax_proof)
        .map_err(|e| format!("v_com inter: {e}"))?;
    eprintln!("[block] combines:   {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // Residual Connection 2: X_out = X_mid + Out_ffn
    // =========================================================================
    let expected_x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

    if expected_x_out_com.row_coms != x_out_com.row_coms {
        return Err("Transformer Block Output Commitment (Residual) Mismatch!".into());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Cryptographic Helper: Homomorphic Addition for Commitments
// ---------------------------------------------------------------------------

/// Exploit the homomorphic property of Pedersen (Hyrax) commitments:
/// Com(A) + Com(B) = Com(A + B)
/// This allows the Verifier to compute the commitment to the residual connection
/// in strictly O(√N) time without any prover assistance or sumcheck.
pub fn add_commitments(a: &HyraxCommitment, b: &HyraxCommitment) -> HyraxCommitment {
    assert_eq!(
        a.row_coms.len(),
        b.row_coms.len(),
        "Commitment dimensions must match"
    );
    let mut result_coms = Vec::with_capacity(a.row_coms.len());

    for (pt_a, pt_b) in a.row_coms.iter().zip(b.row_coms.iter()) {
        // Elliptic curve point addition
        let mut sum_proj = pt_a.into_group();
        let b_proj = pt_b.into_group();

        sum_proj.add_assign(&b_proj); // または sum_proj += b_proj;

        // 再び Affine 座標に戻して保存
        result_coms.push(sum_proj.into_affine());
    }

    HyraxCommitment {
        row_coms: result_coms,
        nu: a.nu,
        sigma: a.sigma,
    }
}

// ---------------------------------------------------------------------------
// Model Verifier (E2E)
// ---------------------------------------------------------------------------

/// Verifies the entire LLM forward pass.
/// The verifier guarantees that `logits_com` is the mathematically correct
/// output of the model for the given input `x_in_com`.
pub fn verify(
    proof: &TransformerModelProof,
    vk: &TransformerModelVerifyingKey,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> Result<(), String> {
    // 1. Bind Initial Input
    absorb_com(transcript, b"x_in_com", &proof.x_in_com);

    // Create cross-model batch accumulators
    use std::time::Instant;
    let t_bits = vk.seq_len.next_power_of_two().trailing_zeros() as usize;
    let d_bits = vk.d_model.next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_t) = params_from_vars(t_bits);
    let (_, _, params_td) = params_from_vars(t_bits + d_bits);

    let q_vk = &vk.block_vks[0].q_vk;
    let qkvo_in_bits = q_vk.d_in.next_power_of_two().trailing_zeros() as usize;
    let qkvo_out_bits = q_vk.d_out.next_power_of_two().trailing_zeros() as usize;
    let params_qkvo_w = params_from_vars(qkvo_in_bits + qkvo_out_bits).2;
    let params_qkvo_b = params_from_vars(qkvo_out_bits).2;

    let lmh_in_bits = vk.lm_head_vk.d_in.next_power_of_two().trailing_zeros() as usize;
    let lmh_out_bits = vk.lm_head_vk.d_out.next_power_of_two().trailing_zeros() as usize;
    let params_lmh_w = params_from_vars(lmh_in_bits + lmh_out_bits).2;
    let params_lmh_b = params_from_vars(lmh_out_bits).2;

    let mut ln_acc_t = HyraxBatchAccumulator::new();
    let mut ln_acc_td = HyraxBatchAccumulator::new();
    let mut proj_acc_w = HyraxBatchAccumulator::new();
    let mut proj_acc_b = HyraxBatchAccumulator::new();
    let mut lmh_acc_w = HyraxBatchAccumulator::new();
    let mut lmh_acc_b = HyraxBatchAccumulator::new();
    // Range proof chunk/m accumulators: sigma witnesses, y witnesses, m_com
    let ln_sig_n = (2 * vk.seq_len).next_power_of_two().trailing_zeros() as usize;
    let ln_y_n_global = (2 * vk.seq_len * vk.d_model).next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_range_sig) = params_from_vars(ln_sig_n);
    let (_, _, params_range_y)   = params_from_vars(ln_y_n_global);
    let (_, _, params_range_m)   = params_from_vars(crate::lookup::range::CHUNK_BITS);
    let mut acc_range_sig = HyraxBatchAccumulator::new();
    let mut acc_range_y   = HyraxBatchAccumulator::new();
    let mut acc_range_m   = HyraxBatchAccumulator::new();
    // Layer-folding accumulator: collects 7 intermediate opens per block.
    // Finalized once after all blocks (2 MSMs total vs 2L).
    let mut inter_acc = HyraxBatchAccumulator::new();

    // 2. Block Verification Chaining
    let mut current_x_com = proof.x_in_com.clone();

    for i in 0..vk.num_blocks {
        let bp = &proof.block_proofs[i];

        // Reconstruct the expected output commitment for this block
        let x_mid_com = add_commitments(&current_x_com, &bp.out_attn_com);
        let expected_x_out_com = add_commitments(&x_mid_com, &bp.out_ffn_com);

        // Verify the block
        verify_transformer_block(
            bp,
            &current_x_com,
            &expected_x_out_com,
            &vk.block_vks[i],
            inst_attn,
            inst_ffn,
            transcript,
            lasso_params,
            &mut ln_acc_t,
            &mut ln_acc_td,
            &mut proj_acc_w,
            &mut proj_acc_b,
            &mut acc_range_sig,
            &mut acc_range_y,
            &mut acc_range_m,
            &mut inter_acc,
        )
        .map_err(|e| format!("Block {} failed: {}", i, e))?;

        // Chain the output to the next block
        current_x_com = expected_x_out_com;
    }

    // 3. Final LayerNorm — global range batch for final_ln (sigma + y)
    let t = vk.seq_len;
    let d = vk.d_model;
    let final_sigma_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let final_y_n = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
    let _t = Instant::now();
    let (final_r_vs, _final_r_m) = verify_range_batched(
        &[
            &proof.final_ln_proof.sigma_range_proof,
            &proof.final_ln_proof.y_range_proof,
        ],
        &proof.final_range_m,
        &[final_sigma_n, final_y_n],
        32,
        transcript,
        &mut acc_range_sig,
        &mut acc_range_y,
        &mut acc_range_m,
    )
    .map_err(|e| format!("Final LN range batch failed: {}", e))?;
    eprintln!("[model] range_batch:{:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);
    let _t = Instant::now();
    let ln_io = LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: proof.final_ln_out_com.clone(),
    };
    verify_layernorm(
        &proof.final_ln_proof,
        &ln_io,
        &vk.final_ln_vk,
        &final_r_vs[0],
        &final_r_vs[1],
        transcript,
        &mut ln_acc_t,
        &mut ln_acc_td,
    )
    .map_err(|e| format!("Final LN failed: {}", e))?;
    eprintln!("[model] final_ln:   {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // 4. LM Head Verification
    let _t = Instant::now();
    let lm_io = ProjectionIOCommitments { x_com: Some(proof.final_ln_out_com.clone()) };
    let (lm_y_claim, _) = verify_projection(&proof.lm_head_proof, &vk.lm_head_vk, &lm_io, transcript, &mut lmh_acc_w, &mut lmh_acc_b)
        .map_err(|e| format!("LM Head failed: {}", e))?;
    eprintln!("[model] lm_head:    {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);
    // Verify logits_com opens to lm_y_claim — binds prover's committed logits to LM head output.
    let t_bits_lm = vk.seq_len.next_power_of_two().trailing_zeros() as usize;
    let v_bits = vk.vocab_size.next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_logits) = params_from_vars(t_bits_lm + v_bits);
    hyrax_verify(&proof.logits_com, lm_y_claim.value, &lm_y_claim.point, &proof.lm_head_logits_open, &params_logits)
        .map_err(|e| format!("Logits commitment verification failed: {e}"))?;

    // Finalize all 10 accumulators in two phases:
    //   Phase 1 (sequential): derive all 10 mu challenges in transcript order.
    //   Phase 2 (parallel):   run the 2 MSMs inside each finalize concurrently.
    // This preserves Fiat-Shamir transcript integrity while parallelising the
    // MSM work (the dominant cost in each finalize call).
    let _tacc = Instant::now();
    let mu_inter    = transcript.challenge_field::<F>(b"hyrax_group_mu"); // inter_acc
    let mu_ln_t     = transcript.challenge_field::<F>(b"hyrax_group_mu"); // ln_acc_t
    let mu_ln_td    = transcript.challenge_field::<F>(b"hyrax_group_mu"); // ln_acc_td
    let mu_proj_w   = transcript.challenge_field::<F>(b"hyrax_group_mu"); // proj_acc_w
    let mu_proj_b   = transcript.challenge_field::<F>(b"hyrax_group_mu"); // proj_acc_b
    let mu_lmh_w    = transcript.challenge_field::<F>(b"hyrax_group_mu"); // lmh_acc_w
    let mu_lmh_b    = transcript.challenge_field::<F>(b"hyrax_group_mu"); // lmh_acc_b
    let mu_rng_sig  = transcript.challenge_field::<F>(b"hyrax_group_mu"); // acc_range_sig
    let mu_rng_y    = transcript.challenge_field::<F>(b"hyrax_group_mu"); // acc_range_y
    let mu_rng_m    = transcript.challenge_field::<F>(b"hyrax_group_mu"); // acc_range_m

    let ((r0, r1), (r2, r3)) = rayon::join(
        || rayon::join(
            || inter_acc.finalize_with_mu(&params_td, mu_inter)
                        .map_err(|e| format!("inter_acc: {e}")),
            || ln_acc_t.finalize_with_mu(&params_t, mu_ln_t)
                       .map_err(|e| format!("ln_acc_t: {e}")),
        ),
        || rayon::join(
            || ln_acc_td.finalize_with_mu(&params_td, mu_ln_td)
                        .map_err(|e| format!("ln_acc_td: {e}")),
            || proj_acc_w.finalize_with_mu(&params_qkvo_w, mu_proj_w)
                         .map_err(|e| format!("proj_acc_w: {e}")),
        ),
    );
    let ((r4, r5), (r6, r7)) = rayon::join(
        || rayon::join(
            || proj_acc_b.finalize_with_mu(&params_qkvo_b, mu_proj_b)
                         .map_err(|e| format!("proj_acc_b: {e}")),
            || lmh_acc_w.finalize_with_mu(&params_lmh_w, mu_lmh_w)
                        .map_err(|e| format!("lmh_acc_w: {e}")),
        ),
        || rayon::join(
            || lmh_acc_b.finalize_with_mu(&params_lmh_b, mu_lmh_b)
                        .map_err(|e| format!("lmh_acc_b: {e}")),
            || acc_range_sig.finalize_with_mu(&params_range_sig, mu_rng_sig)
                            .map_err(|e| format!("acc_range_sig: {e}")),
        ),
    );
    let (r8, r9) = rayon::join(
        || acc_range_y.finalize_with_mu(&params_range_y, mu_rng_y)
                      .map_err(|e| format!("acc_range_y: {e}")),
        || acc_range_m.finalize_with_mu(&params_range_m, mu_rng_m)
                      .map_err(|e| format!("acc_range_m: {e}")),
    );
    eprintln!("[model] acc_finalize:{:>8.3}ms", _tacc.elapsed().as_secs_f64()*1000.0);
    r0?; r1?; r2?; r3?; r4?; r5?; r6?; r7?; r8?; r9?;

    // 5. Global batched Lasso
    let _t = Instant::now();
    let t_bits = vk.seq_len.next_power_of_two().trailing_zeros() as usize;
    let d_bits = vk.d_model.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;
    let mut all_lasso_instances = Vec::new();
    let mut all_instance_coms = Vec::new();
    let mut all_output_coms: Vec<(HyraxCommitment, usize)> = Vec::new();
    for i in 0..vk.num_blocks {
        let bvk = &vk.block_vks[i];
        all_lasso_instances.push(inst_attn.q_lasso.clone());
        all_lasso_instances.push(inst_attn.k_lasso.clone());
        all_instance_coms.push(bvk.attn_pk.qk_lasso_pk.instance_table_coms[0].clone());
        all_instance_coms.push(bvk.attn_pk.qk_lasso_pk.instance_table_coms[1].clone());
        // phi_q_com/phi_k_com from block proofs bind Lasso outputs to committed activations.
        all_output_coms.push((proof.block_proofs[i].attn_proof.phi_q_com.clone(), td_num_vars));
        all_output_coms.push((proof.block_proofs[i].attn_proof.phi_k_com.clone(), td_num_vars));
    }
    let global_multi_inst = LassoMultiInstance {
        instances: all_lasso_instances,
    };
    let global_lasso_vk = LassoMultiVerifyingKey {
        instance_table_coms: all_instance_coms,
    };
    verify_lasso_multi(
        &proof.all_lasso_proof,
        &global_multi_inst,
        &global_lasso_vk,
        &all_output_coms,
        transcript,
        lasso_params,
    )
    .map_err(|e| format!("Global batched Lasso failed: {}", e))?;
    eprintln!("[model] lasso:      {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    Ok(())
}
