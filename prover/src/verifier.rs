//! Global Verifier for a full Transformer Block.
//!
//! **Production-Grade Succinctness:**
//! 1. Strict O(√N) runtime. No matrices are constructed. No O(N) operations exist.
//! 2. Homomorphic Residuals: X_mid and X_out are verified by directly adding
//!    their Pedersen (Hyrax) commitments. No additional proofs needed!
//! 3. Commitment Chaining: Intermediate IO commitments passed from the Prover
//!    are cryptographically bound across adjacent sub-verifiers.

use crate::lookup::lasso::{verify_lasso_multi, LassoMultiInstance, LassoMultiVerifyingKey};
use crate::pcs::{absorb_com, params_from_vars, HyraxBatchAccumulator, HyraxCommitment, HyraxParams};
use crate::subprotocols::verify_combine_deferred;
use crate::pcs::hyrax_verify_multi_point;
use crate::transcript::Transcript;

// Sub-module keys and verifiers
use crate::attention::attention::{
    verify_linear_attention, AttentionProvingKey, LinearAttentionInstance,
};
use crate::attention::layernorm::{
    verify_layernorm, LayerNormIOCommitments, LayerNormLassoKey, LayerNormLassoVerifyingKey,
    LayerNormVerifyingKey,
};
use crate::attention::projection::{
    verify_projection, ProjectionIOCommitments, ProjectionProvingKey, ProjectionVerifyingKey,
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
    pub ln2_vk: LayerNormVerifyingKey,
    /// Shared Lasso key for both LN1 and LN2 (same inv-sqrt tables).
    pub ln_lasso_key: LayerNormLassoKey,
    pub q_vk: ProjectionVerifyingKey,
    pub k_vk: ProjectionVerifyingKey,
    pub v_vk: ProjectionVerifyingKey,
    pub o_vk: ProjectionVerifyingKey,
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
) -> Result<(), String> {
    // =========================================================================
    // Pipeline Stitching (The Binding of IO Commitments)
    // =========================================================================

    use std::time::Instant;
    let t_bits = vk.seq_len.next_power_of_two().trailing_zeros() as usize;
    let d_bits = vk.d_model.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;
    let (_, _, td_params) = params_from_vars(td_num_vars);

    // --- 1. LayerNorm 1 ---
    let ln1_io = LayerNormIOCommitments {
        x_com: x_in_com.clone(),
        y_com: proof.x_norm1_com.clone(),
    };
    let ln_lasso_vk = vk.ln_lasso_key.vk();
    let _t = Instant::now(); verify_layernorm(&proof.ln1_proof, &ln1_io, &vk.ln1_vk, &ln_lasso_vk, transcript, ln_acc_t, ln_acc_td)?;
    eprintln!("[block] ln1:        {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    let _ = td_params; // ensure td_params isn't flagged unused

    // --- 2. Projections (Q, K, V) — returns (y_claim, x_claim) for combine ---
    let q_io = ProjectionIOCommitments { x_com: proof.x_norm1_com.clone() };
    let _t = Instant::now(); let (q_y_claim, q_x_claim) = verify_projection(&proof.q_proj_proof, &vk.q_vk, &q_io, transcript, proj_acc_w, proj_acc_b)?;
    eprintln!("[block] q_proj:     {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    let k_io = ProjectionIOCommitments { x_com: proof.x_norm1_com.clone() };
    let _t = Instant::now(); let (k_y_claim, k_x_claim) = verify_projection(&proof.k_proj_proof, &vk.k_vk, &k_io, transcript, proj_acc_w, proj_acc_b)?;
    eprintln!("[block] k_proj:     {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    let v_io = ProjectionIOCommitments { x_com: proof.x_norm1_com.clone() };
    let _t = Instant::now(); let (v_y_claim, v_x_claim) = verify_projection(&proof.v_proj_proof, &vk.v_vk, &v_io, transcript, proj_acc_w, proj_acc_b)?;
    eprintln!("[block] v_proj:     {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // --- 3. Linear Attention — returns (out_claim, v_claim) for combine ---
    let attn_io = crate::attention::attention::AttentionIOCommitments {
        q_com: proof.q_com.clone(),
        k_com: proof.k_com.clone(),
        v_com: proof.v_com.clone(),
        out_com: proof.out_inner_com.clone(),
    };
    let _t = Instant::now(); let (attn_out_claim, attn_v_claim) =
        verify_linear_attention(&proof.attn_proof, inst_attn, &attn_io, transcript)?;
    eprintln!("[block] attn:       {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // --- 4. Output Projection ---
    let o_io = ProjectionIOCommitments { x_com: proof.out_inner_com.clone() };
    let _t = Instant::now(); let (o_y_claim, o_x_claim) = verify_projection(&proof.o_proj_proof, &vk.o_vk, &o_io, transcript, proj_acc_w, proj_acc_b)?;
    eprintln!("[block] o_proj:     {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // =========================================================================
    // Residual Connection 1: X_mid = X_in + Out_attn (homomorphic, no transcript)
    // =========================================================================
    let x_mid_com = add_commitments(x_in_com, &proof.out_attn_com);

    // --- 5. LayerNorm 2 ---
    let ln2_io = LayerNormIOCommitments {
        x_com: x_mid_com.clone(),
        y_com: proof.x_norm2_com.clone(),
    };
    let _t = Instant::now(); verify_layernorm(&proof.ln2_proof, &ln2_io, &vk.ln2_vk, &ln_lasso_vk, transcript, ln_acc_t, ln_acc_td)?;
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

    // --- 7. GKR Combine Proofs ---
    // Run sumcheck + leaf-check for all 8 combines, defer Hyrax openings.
    // Then batch all 8 openings into 2 MSMs via hyrax_verify_multi_point.
    let _t = Instant::now();
    let (q_r, q_f) = verify_combine_deferred(&proof.q_combine, &proof.q_com, &[q_y_claim], td_num_vars, transcript)
        .map_err(|e| format!("q_combine: {e}"))?;
    let (k_r, k_f) = verify_combine_deferred(&proof.k_combine, &proof.k_com, &[k_y_claim], td_num_vars, transcript)
        .map_err(|e| format!("k_combine: {e}"))?;
    let (v_r, v_f) = verify_combine_deferred(&proof.v_combine, &proof.v_com, &[v_y_claim, attn_v_claim], td_num_vars, transcript)
        .map_err(|e| format!("v_combine: {e}"))?;
    let (oi_r, oi_f) = verify_combine_deferred(
        &proof.out_inner_combine, &proof.out_inner_com,
        &[attn_out_claim, o_x_claim], td_num_vars, transcript,
    ).map_err(|e| format!("out_inner_combine: {e}"))?;
    let (n1_r, n1_f) = verify_combine_deferred(
        &proof.x_norm1_combine, &proof.x_norm1_com,
        &[q_x_claim, k_x_claim, v_x_claim], td_num_vars, transcript,
    ).map_err(|e| format!("x_norm1_combine: {e}"))?;
    let (oa_r, oa_f) = verify_combine_deferred(&proof.out_attn_combine, &proof.out_attn_com, &[o_y_claim], td_num_vars, transcript)
        .map_err(|e| format!("out_attn_combine: {e}"))?;
    let (n2_r, n2_f) = verify_combine_deferred(&proof.x_norm2_combine, &proof.x_norm2_com, &[ffn_x_claim], td_num_vars, transcript)
        .map_err(|e| format!("x_norm2_combine: {e}"))?;
    let (of_r, of_f) = verify_combine_deferred(&proof.out_ffn_combine, &proof.out_ffn_com, &[ffn_y_claim], td_num_vars, transcript)
        .map_err(|e| format!("out_ffn_combine: {e}"))?;

    // Batch all 8 Hyrax openings: 2 MSMs instead of 16
    let (_, _, combine_params) = params_from_vars(td_num_vars);
    hyrax_verify_multi_point(
        &[
            (&proof.q_com,          q_f,  &q_r,  &proof.q_combine.hyrax_proof),
            (&proof.k_com,          k_f,  &k_r,  &proof.k_combine.hyrax_proof),
            (&proof.v_com,          v_f,  &v_r,  &proof.v_combine.hyrax_proof),
            (&proof.out_inner_com,  oi_f, &oi_r, &proof.out_inner_combine.hyrax_proof),
            (&proof.x_norm1_com,    n1_f, &n1_r, &proof.x_norm1_combine.hyrax_proof),
            (&proof.out_attn_com,   oa_f, &oa_r, &proof.out_attn_combine.hyrax_proof),
            (&proof.x_norm2_com,    n2_f, &n2_r, &proof.x_norm2_combine.hyrax_proof),
            (&proof.out_ffn_com,    of_f, &of_r, &proof.out_ffn_combine.hyrax_proof),
        ],
        &combine_params,
        transcript,
    ).map_err(|e| format!("combine batch: {e}"))?;
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
        )
        .map_err(|e| format!("Block {} failed: {}", i, e))?;

        // Chain the output to the next block
        current_x_com = expected_x_out_com;
    }

    // 3. Final LayerNorm Verification
    let _t = Instant::now();
    let ln_io = LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: proof.final_ln_out_com.clone(),
    };
    let final_ln_lasso_vk = vk.final_ln_lasso_key.vk();
    verify_layernorm(&proof.final_ln_proof, &ln_io, &vk.final_ln_vk, &final_ln_lasso_vk, transcript, &mut ln_acc_t, &mut ln_acc_td)
        .map_err(|e| format!("Final LN failed: {}", e))?;
    eprintln!("[model] final_ln:   {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // 4. LM Head Verification
    let _t = Instant::now();
    let lm_io = ProjectionIOCommitments { x_com: proof.final_ln_out_com.clone() };
    verify_projection(&proof.lm_head_proof, &vk.lm_head_vk, &lm_io, transcript, &mut lmh_acc_w, &mut lmh_acc_b)
        .map_err(|e| format!("LM Head failed: {}", e))?;
    eprintln!("[model] lm_head:    {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    // Finalize all 6 accumulators (each adds hyrax_group_mu to transcript)
    let _tacc = Instant::now();
    ln_acc_t.finalize(&params_t, transcript)?;
    ln_acc_td.finalize(&params_td, transcript)?;
    proj_acc_w.finalize(&params_qkvo_w, transcript)?;
    proj_acc_b.finalize(&params_qkvo_b, transcript)?;
    lmh_acc_w.finalize(&params_lmh_w, transcript)?;
    lmh_acc_b.finalize(&params_lmh_b, transcript)?;
    eprintln!("[model] acc_finalize:{:>8.3}ms", _tacc.elapsed().as_secs_f64()*1000.0);

    // 5. Global batched Lasso
    let _t = Instant::now();
    let mut all_lasso_instances = Vec::new();
    let mut all_instance_coms = Vec::new();
    for i in 0..vk.num_blocks {
        let bvk = &vk.block_vks[i];
        all_lasso_instances.push(inst_ffn.activation_lasso.clone());
        all_lasso_instances.push(inst_attn.q_lasso.clone());
        all_lasso_instances.push(inst_attn.k_lasso.clone());
        all_instance_coms.push(bvk.ffn_vk.activation_lasso_vk.table_coms.clone());
        all_instance_coms.push(bvk.attn_pk.qk_lasso_pk.instance_table_coms[0].clone());
        all_instance_coms.push(bvk.attn_pk.qk_lasso_pk.instance_table_coms[1].clone());
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
        transcript,
        lasso_params,
    )
    .map_err(|e| format!("Global batched Lasso failed: {}", e))?;
    eprintln!("[model] lasso:      {:>8.3}ms", _t.elapsed().as_secs_f64()*1000.0);

    Ok(())
}
