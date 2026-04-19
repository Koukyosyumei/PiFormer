//! Global Prover for a full Transformer Block.
//!
//! **Production-Grade Architecture:**
//! 1. Executes the forward pass to generate all intermediate dynamic activations (Witness).
//! 2. Computes the IO Commitments for all layer boundaries exactly ONCE to prevent O(N) bloat.
//! 3. Delegates the proof generation to the sub-provers, passing the strongly bound IO commitments.
//!
//! **Block Architecture:**
//!   X_norm1 = LayerNorm(X_in)
//!   Q, K, V = Projection(X_norm1, W_Q/K/V)
//!   Out_inner = LinearAttention(Q, K, V)
//!   Out_attn = Projection(Out_inner, W_O)
//!   X_mid = X_in + Out_attn   <-- Residual 1
//!   X_norm2 = LayerNorm(X_mid)
//!   Out_ffn = FFN(X_norm2)
//!   X_out = X_mid + Out_ffn   <-- Residual 2

use crate::field::F;
use crate::pcs::{absorb_com, hyrax_commit, hyrax_open, params_from_vars, HyraxCommitment, HyraxParams, HyraxProof};
use crate::poly::utils::mat_to_mle;
use crate::subprotocols::{prove_combine, CombineProof};
use crate::transcript::Transcript;

// Sub-module imports (Assuming the interfaces we built previously)
use crate::attention::attention::{
    prove_linear_attention, AttentionIOCommitments,
    LinearAttentionInstance, LinearAttentionProof, LinearAttentionWitness,
};
use crate::attention::layernorm::{
    compute_range_witnesses, prove_layernorm, LayerNormIOCommitments, LayerNormProof,
    LayerNormVerifyingKey,
    LayerNormWitness,
};
use crate::attention::projection::{
    prove_projection, prove_qkv_projections, BatchedQKVProjectionIOCommitments,
    BatchedQKVProjectionProof, BatchedQKVProjectionWitness, ProjectionIOCommitments,
    ProjectionProof, ProjectionProvingKey, ProjectionVerifyingKey, ProjectionWitness,
};
use crate::ffn::ffn::{prove_ffn, FFNIOCommitments, FFNInstance, FFNProof, FFNWitness};
use crate::lookup::lasso::{
    prove_lasso_multi, LassoMultiInstance, LassoMultiProof, LassoMultiProvingKey,
    LassoOutputBinding,
};
use crate::lookup::range::{prove_range_batched, GlobalRangeM};
use crate::verifier::{add_commitments, TransformerBlockVerifyingKey}; // Imported from verifier.rs

// ---------------------------------------------------------------------------
// Global Proof Structure
// ---------------------------------------------------------------------------

/// The complete ZK Proof for one Transformer Block.
///
/// GKR backward fusion (out_inner elimination):
/// O_proj runs before attention in the transcript. O_proj's x_claim is passed as
/// the external out_inner evaluation point to attention. Both sub-provers reference
/// the same out_inner evaluation, eliminating the need for out_inner_com or
/// out_inner_combine. Soundness holds because the two sumchecks together pin down
/// out_inner at the shared evaluation point.
pub struct TransformerBlockProof {
    // Sub-proofs (O_proj runs before attention in transcript order)
    pub ln1_proof: LayerNormProof,
    pub qkv_proj_proof: BatchedQKVProjectionProof,
    pub o_proj_proof: ProjectionProof,  // runs before attn (GKR backward)
    pub attn_proof: LinearAttentionProof,
    pub ln2_proof: LayerNormProof,
    pub ffn_proof: FFNProof,

    // Intermediate IO Commitments (out_inner_com eliminated via GKR backward fusion)
    pub x_norm1_com: HyraxCommitment,
    pub q_com: HyraxCommitment,
    pub k_com: HyraxCommitment,
    pub v_com: HyraxCommitment,
    // out_inner_com: eliminated — out_inner bound by shared sumcheck claim
    pub out_attn_com: HyraxCommitment,
    pub x_norm2_com: HyraxCommitment,
    pub out_ffn_com: HyraxCommitment,

    /// Direct Hyrax opening of q_com at the Q-projection y_claim point.
    pub q_open: HyraxProof,
    /// Direct Hyrax opening of k_com at the K-projection y_claim point.
    pub k_open: HyraxProof,
    /// Verifies v_com: 2 claims — V-projection y_claim + attention v_claim.
    pub v_combine: CombineProof,
    // out_inner_combine: eliminated — no out_inner_com to combine
    /// Direct Hyrax opening of x_norm1_com at the shared batched-QKV x_claim point.
    pub x_norm1_open: HyraxProof,
    /// Direct Hyrax opening of out_attn_com at the O-projection y_claim point.
    pub out_attn_open: HyraxProof,
    /// Direct Hyrax opening of x_norm2_com at the FFN x_claim point.
    pub x_norm2_open: HyraxProof,
    /// Direct Hyrax opening of out_ffn_com at the FFN y_claim point.
    pub out_ffn_open: HyraxProof,

    /// Shared multiplicity commitment for all 4 range proofs in this block
    /// (ln1_sigma, ln1_y, ln2_sigma, ln2_y).  One m_com instead of 4.
    pub block_range_m: GlobalRangeM,
}

// ---------------------------------------------------------------------------
// Global Witness Structure
// ---------------------------------------------------------------------------

pub struct TransformerBlockWitness {
    pub x_in: Vec<Vec<F>>,
    pub ln1_wit: LayerNormWitness,
    pub q_proj_wit: ProjectionWitness,
    pub k_proj_wit: ProjectionWitness,
    pub v_proj_wit: ProjectionWitness,
    pub attn_wit: LinearAttentionWitness,
    pub o_proj_wit: ProjectionWitness,
    pub x_mid: Vec<Vec<F>>, // Residual 1 output
    pub ln2_wit: LayerNormWitness,
    pub ffn_wit: FFNWitness,
    pub x_out: Vec<Vec<F>>, // Residual 2 output
}

// ---------------------------------------------------------------------------
// Prover Implementation
// ---------------------------------------------------------------------------

pub fn prove_transformer_block(
    witness: &TransformerBlockWitness,
    x_in_com: &HyraxCommitment,        // Provided by previous block
    pk: &TransformerBlockVerifyingKey, // The VK contains all static weights
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> Result<TransformerBlockProof, String> {
    let t = pk.seq_len;
    let d = pk.d_model;

    // Helper to commit to a matrix
    let commit_mat = |mat: &[Vec<F>], rows: usize, cols: usize| -> HyraxCommitment {
        let mle = mat_to_mle(mat, rows, cols); // Implement mat_to_mle as usual
        let total_vars =
            rows.next_power_of_two().trailing_zeros() + cols.next_power_of_two().trailing_zeros();
        let nu = total_vars as usize / 2;
        let sigma = (total_vars as usize - nu).max(1);
        hyrax_commit(&mle.evaluations, nu, &HyraxParams::new(sigma))
    };

    // 1. Generate Intermediate IO Commitments (out_inner_com eliminated via GKR backward)
    let x_norm1_com = commit_mat(&witness.ln1_wit.y, t, d);
    let q_com = commit_mat(&witness.attn_wit.q, t, d);
    let k_com = commit_mat(&witness.attn_wit.k, t, d);
    let v_com = commit_mat(&witness.attn_wit.v, t, d);
    // out_inner_com: NOT committed — bound by shared sumcheck claim (GKR backward)
    let out_attn_com = commit_mat(&witness.o_proj_wit.y, t, d);
    let x_norm2_com = commit_mat(&witness.ln2_wit.y, t, d);
    let out_ffn_com = commit_mat(&witness.ffn_wit.y, t, d);

    // Helper: variables for t×d matrices
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;

    // 2. Global range batch for all LayerNorm range proofs in this block.
    //    Phase 1+2: commit chunk_coms for all 4 witnesses, one shared m_com, then sumchecks.
    //    Transcript ordering: all range material comes BEFORE any layernorm sub-prover.
    let ln1_rw = compute_range_witnesses(&witness.ln1_wit, &pk.ln1_vk);
    let ln2_rw = compute_range_witnesses(&witness.ln2_wit, &pk.ln2_vk);
    let ln1_sigma_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let ln1_y_n = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
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
    // Destructure in reverse order to avoid index shifting
    let ln2_y_rp    = block_range_proofs.remove(3);
    let ln2_sig_rp  = block_range_proofs.remove(2);
    let ln1_y_rp    = block_range_proofs.remove(1);
    let ln1_sig_rp  = block_range_proofs.remove(0);
    let ln2_y_rv    = block_r_vs[3].clone();
    let ln2_sig_rv  = block_r_vs[2].clone();
    let ln1_y_rv    = block_r_vs[1].clone();
    let ln1_sig_rv  = block_r_vs[0].clone();
    let _ = (ln1_sigma_n, ln1_y_n); // used only for verifier num_vars

    // 3. Execute Sub-Provers with strictly bound IO Commitments
    // --- LayerNorm 1 ---
    let ln1_io = LayerNormIOCommitments {
        x_com: x_in_com.clone(),
        y_com: x_norm1_com.clone(),
    };
    let ln1_proof = prove_layernorm(
        &witness.ln1_wit,
        &ln1_io,
        &pk.ln1_vk,
        (ln1_sig_rp, ln1_sig_rv),
        (ln1_y_rp, ln1_y_rv),
        transcript,
    )?;

    // --- Batched Q, K, V Projections — one sumcheck, single r_k ---
    let qkv_wit = BatchedQKVProjectionWitness {
        x: witness.ln1_wit.y.clone(),
        q: witness.q_proj_wit.y.clone(),
        k: witness.k_proj_wit.y.clone(),
        v: witness.v_proj_wit.y.clone(),
    };
    let qkv_io = BatchedQKVProjectionIOCommitments { x_com: x_norm1_com.clone() };
    let (qkv_proj_proof, q_y_claim, k_y_claim, v_y_claim, x_norm1_claim) =
        prove_qkv_projections(&pk.q_pk, &pk.k_pk, &pk.v_pk, &qkv_wit, &qkv_io, transcript)?;

    // --- GKR backward fusion: O_proj runs BEFORE attention ---
    // O_proj's x_claim gives an evaluation of out_inner at a transcript-derived point.
    // out_inner is NOT committed; x_com is None (GKR backward, no out_inner_com).
    let o_io = ProjectionIOCommitments { x_com: None };
    let (o_proj_proof, o_y_claim, o_x_claim) =
        prove_projection(&pk.o_pk, &witness.o_proj_wit, &o_io, transcript)?;

    // --- Linear Attention — receives O_proj's x_claim as the external out_inner claim ---
    // out_com removed from IO; both sub-provers reference the same out_inner eval point.
    let attn_io = AttentionIOCommitments {
        q_com: q_com.clone(),
        k_com: k_com.clone(),
        v_com: v_com.clone(),
    };
    let (attn_proof, _attn_out_claim, attn_v_claim) = prove_linear_attention(
        &witness.attn_wit,
        inst_attn,
        &pk.attn_pk,
        &attn_io,
        Some(o_x_claim.clone()),  // GKR backward: out_inner eval from O_proj
        transcript,
        lasso_params,
    );

    // --- LayerNorm 2 ---
    let x_mid_com = add_commitments(x_in_com, &out_attn_com);
    let ln2_io = LayerNormIOCommitments {
        x_com: x_mid_com.clone(),
        y_com: x_norm2_com.clone(),
    };
    let ln2_proof = prove_layernorm(
        &witness.ln2_wit,
        &ln2_io,
        &pk.ln2_vk,
        (ln2_sig_rp, ln2_sig_rv),
        (ln2_y_rp, ln2_y_rv),
        transcript,
    )?;

    // --- FFN — returns (proof, y_claim, x_claim) ---
    let ffn_io = FFNIOCommitments {
        x_com: x_norm2_com.clone(),
        y_com: out_ffn_com.clone(),
    };
    let (ffn_proof, ffn_y_claim, ffn_x_claim) = prove_ffn(
        &pk.ffn_pk,
        &witness.ffn_wit,
        inst_ffn,
        &ffn_io,
        transcript,
        lasso_params,
    )?;

    // --- GKR Combine Proofs ---
    // Build flat MLE evaluations for each intermediate commitment.
    let mat_evals = |mat: &[Vec<F>], rows: usize, cols: usize| -> Vec<F> {
        mat_to_mle(mat, rows, cols).evaluations
    };

    let (nu_td, sigma_td, _) = params_from_vars(td_num_vars);

    // q_com: direct open at Q-proj y_claim point
    let q_evals = mat_evals(&witness.attn_wit.q, t, d);
    let q_open = hyrax_open(&q_evals, &q_y_claim.point, nu_td, sigma_td);

    // k_com: direct open at K-proj y_claim point
    let k_evals = mat_evals(&witness.attn_wit.k, t, d);
    let k_open = hyrax_open(&k_evals, &k_y_claim.point, nu_td, sigma_td);

    // v_com: 2 claims — V-proj y_claim + attention v_claim
    let v_evals = mat_evals(&witness.attn_wit.v, t, d);
    let (v_combine, _) =
        prove_combine(&v_evals, &v_com, &[v_y_claim, attn_v_claim], td_num_vars, transcript);

    // out_inner_com: ELIMINATED via GKR backward fusion.
    // O_proj's x_claim and attention's external_out_claim both reference the same
    // out_inner evaluation point, so no combine proof or commitment is needed.

    // x_norm1_com: direct open at the single batched-QKV x_norm1_claim point
    let x_norm1_evals = mat_evals(&witness.ln1_wit.y, t, d);
    let x_norm1_open = hyrax_open(&x_norm1_evals, &x_norm1_claim.point, nu_td, sigma_td);

    // out_attn_com: direct open at O-proj y_claim point
    let out_attn_evals = mat_evals(&witness.o_proj_wit.y, t, d);
    let out_attn_open = hyrax_open(&out_attn_evals, &o_y_claim.point, nu_td, sigma_td);

    // x_norm2_com: direct open at FFN x_claim point
    let x_norm2_evals = mat_evals(&witness.ln2_wit.y, t, d);
    let x_norm2_open = hyrax_open(&x_norm2_evals, &ffn_x_claim.point, nu_td, sigma_td);

    // out_ffn_com: direct open at FFN y_claim point
    let out_ffn_evals = mat_evals(&witness.ffn_wit.y, t, d);
    let out_ffn_open = hyrax_open(&out_ffn_evals, &ffn_y_claim.point, nu_td, sigma_td);

    Ok(TransformerBlockProof {
        ln1_proof,
        qkv_proj_proof,
        o_proj_proof,  // O_proj before attn (GKR backward)
        attn_proof,
        ln2_proof,
        ffn_proof,
        x_norm1_com,
        q_com,
        k_com,
        v_com,
        // out_inner_com: eliminated via GKR backward fusion
        out_attn_com,
        x_norm2_com,
        out_ffn_com,
        q_open,
        k_open,
        v_combine,
        // out_inner_combine: eliminated via GKR backward fusion
        x_norm1_open,
        out_attn_open,
        x_norm2_open,
        out_ffn_open,
        block_range_m,
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
    pub block_pks: Vec<crate::verifier::TransformerBlockVerifyingKey>, // 簡略化のためVKと同じものをPKとして扱う
    pub lm_head_pk: ProjectionProvingKey,
}

pub struct TransformerModelWitness {
    pub x_in: Vec<Vec<F>>, // Initial embeddings
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
    /// Hyrax opening of logits_com at the LM head sumcheck y_claim point.
    pub lm_head_logits_open: HyraxProof,

    pub all_lasso_proof: LassoMultiProof,

    pub final_range_m: GlobalRangeM,
}

// ---------------------------------------------------------------------------
// Model Prover (E2E)
// ---------------------------------------------------------------------------

pub fn prove(
    pk: &TransformerModelProvingKey,
    witness: &TransformerModelWitness,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> Result<TransformerModelProof, String> {
    let t = pk.vk.seq_len;
    let d = pk.vk.d_model;
    let v = pk.vk.vocab_size;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;

    let commit_mat = |mat: &[Vec<F>], rows: usize, cols: usize| -> HyraxCommitment {
        let mle = mat_to_mle(mat, rows, cols);
        let vars =
            rows.next_power_of_two().trailing_zeros() + cols.next_power_of_two().trailing_zeros();
        let nu = vars as usize / 2;
        let sigma = (vars as usize - nu).max(1);
        hyrax_commit(&mle.evaluations, nu, &HyraxParams::new(sigma))
    };

    // 1. Initial Input Commitment
    let x_in_com = commit_mat(&witness.x_in, t, d);
    absorb_com(transcript, b"x_in_com", &x_in_com);

    // 2. Iterate through Transformer Blocks
    let mut block_proofs = Vec::with_capacity(pk.vk.num_blocks);
    let mut current_x_com = x_in_com.clone();

    for i in 0..pk.vk.num_blocks {
        let block_proof = prove_transformer_block(
            &witness.block_witnesses[i],
            &current_x_com,
            &pk.block_pks[i],
            inst_attn,
            inst_ffn,
            transcript,
            lasso_params,
        )?;

        // 次のブロックへの入力となる残差接続出力のコミットメントを計算
        let x_mid_com = add_commitments(&current_x_com, &block_proof.out_attn_com);
        let next_x_com = add_commitments(&x_mid_com, &block_proof.out_ffn_com);

        current_x_com = next_x_com;
        block_proofs.push(block_proof);
    }

    // 3. Final LayerNorm — global range batch for final_ln (sigma + y)
    let final_rw = compute_range_witnesses(&witness.final_ln_wit, &pk.vk.final_ln_vk);
    let (mut final_range_proofs, final_range_m, final_r_vs) = prove_range_batched(
        &[&final_rw.sigma_witness, &final_rw.y_witness],
        32,
        transcript,
    )?;
    let final_y_rp   = final_range_proofs.remove(1);
    let final_sig_rp = final_range_proofs.remove(0);
    let final_ln_out_com = commit_mat(&witness.final_ln_wit.y, t, d);
    let ln_io = LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: final_ln_out_com.clone(),
    };
    let final_ln_proof = prove_layernorm(
        &witness.final_ln_wit,
        &ln_io,
        &pk.vk.final_ln_vk,
        (final_sig_rp, final_r_vs[0].clone()),
        (final_y_rp, final_r_vs[1].clone()),
        transcript,
    )?;

    // 4. LM Head (Final Projection to Vocab Size)
    let logits_mle = mat_to_mle(&witness.lm_head_wit.y, t, v);
    let logits_com = commit_mat(&witness.lm_head_wit.y, t, v);
    let lm_io = ProjectionIOCommitments { x_com: Some(final_ln_out_com.clone()) };
    let (lm_head_proof, lm_y_claim, _) =
        prove_projection(&pk.lm_head_pk, &witness.lm_head_wit, &lm_io, transcript)?;
    // Open logits_com at the LM head y_claim point to bind the output commitment.
    let v_bits = v.next_power_of_two().trailing_zeros() as usize;
    let lm_logits_num_vars = t_bits + v_bits;
    let (lm_nu, lm_sigma, _) = params_from_vars(lm_logits_num_vars);
    let lm_head_logits_open = hyrax_open(&logits_mle.evaluations, &lm_y_claim.point, lm_nu, lm_sigma);

    // Advance transcript to match verifier's 10 accumulator finalizations.
    // inter_acc (layer-folded intermediate opens) is finalized BEFORE the weight accs.
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu"); // inter_acc (layer folding)
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu"); // ln_acc_t
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu"); // ln_acc_td
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu"); // proj_acc_w (QKVO)
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu"); // proj_acc_b (QKVO)
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu"); // lmh_acc_w
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu"); // lmh_acc_b
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu"); // acc_range_sig
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu"); // acc_range_y
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu"); // acc_range_m

    // 5. Global batched Lasso: one proof for all activation lookups across all layers.
    // Instance order per block: [Q_i, K_i].
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;
    let mut all_lasso_instances: Vec<_> = Vec::new();
    let mut all_instance_coms = Vec::new();
    let mut all_output_bindings: Vec<LassoOutputBinding> = Vec::new();
    let mut global_nu = 0usize;
    for i in 0..pk.vk.num_blocks {
        let bpk = &pk.block_pks[i];
        let attn_wit = &witness.block_witnesses[i].attn_wit;
        let phi_q_mle = mat_to_mle(&attn_wit.phi_q, t, d);
        let phi_k_mle = mat_to_mle(&attn_wit.phi_k, t, d);

        all_lasso_instances.push(inst_attn.q_lasso.clone());
        all_lasso_instances.push(inst_attn.k_lasso.clone());
        all_instance_coms.push(bpk.attn_pk.qk_lasso_pk.instance_table_coms[0].clone());
        all_instance_coms.push(bpk.attn_pk.qk_lasso_pk.instance_table_coms[1].clone());
        global_nu = bpk.attn_pk.qk_lasso_pk.nu;

        // Output bindings: link phi_q_com/phi_k_com to Lasso outputs.
        all_output_bindings.push(LassoOutputBinding {
            com: block_proofs[i].attn_proof.phi_q_com.clone(),
            num_vars: td_num_vars,
            mle_evals: phi_q_mle.evaluations,
        });
        all_output_bindings.push(LassoOutputBinding {
            com: block_proofs[i].attn_proof.phi_k_com.clone(),
            num_vars: td_num_vars,
            mle_evals: phi_k_mle.evaluations,
        });
    }
    let global_multi_inst = LassoMultiInstance { instances: all_lasso_instances };
    let global_lasso_pk = LassoMultiProvingKey { instance_table_coms: all_instance_coms, nu: global_nu };
    let all_lasso_proof = prove_lasso_multi(&global_multi_inst, &global_lasso_pk, &all_output_bindings, transcript, lasso_params);

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
    })
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
    use crate::verifier::{add_commitments, verify, verify_transformer_block};
    use ark_ff::Field;

    // -----------------------------------------------------------------------
    // Fixed small dimensions for all tests
    // -----------------------------------------------------------------------

    const T: usize = 2; // seq_len
    const D: usize = 2; // d_model
    const D_FF: usize = 4; // d_ff
    const V: usize = 2; // vocab_size
    const M_BITS: usize = 4; // bits per chunk for Lasso (table size 16)

    // -----------------------------------------------------------------------
    // Weight helpers
    // -----------------------------------------------------------------------

    fn zero_mat(rows: usize, cols: usize) -> Vec<Vec<F>> {
        vec![vec![F::ZERO; cols]; rows]
    }

    fn zero_ternary_mat(rows: usize, cols: usize) -> Vec<Vec<TernaryValue>> {
        vec![vec![TernaryValue::ZERO; cols]; rows]
    }

    /// Build a single-block model with simple non-zero weights.
    /// Each D×D attention matrix has only W[0][0]=+1; D×D_FF and D_FF×D FFN
    /// matrices likewise have only W[0][0]=+1.  The lm_head is the D×V identity.
    /// This keeps all intermediate activations non-negative and within the 4-bit
    /// Lasso table range [0, 15], making the witness easy to verify by hand.
    fn build_test_weights() -> TransformerModelWeights {
        let mut q_w = zero_ternary_mat(D, D);
        let mut k_w = zero_ternary_mat(D, D);
        let mut v_w = zero_ternary_mat(D, D);
        let mut o_w = zero_ternary_mat(D, D);
        let mut ffn_w1 = zero_ternary_mat(D, D_FF);
        let mut ffn_w2 = zero_ternary_mat(D_FF, D);
        let mut lm_head_w = zero_ternary_mat(D, V);

        q_w[0][0] = TernaryValue::ONE;
        k_w[0][0] = TernaryValue::ONE;
        v_w[0][0] = TernaryValue::ONE;
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
            v_w,
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

    // -----------------------------------------------------------------------
    // Witness helpers
    // -----------------------------------------------------------------------

    /// Build a Hyrax commitment to a matrix using the same formula as the prover.
    fn commit_mat_test(mat: &[Vec<F>], rows: usize, cols: usize) -> HyraxCommitment {
        let mle = mat_to_mle(mat, rows, cols);
        let total_vars = rows.next_power_of_two().trailing_zeros() as usize
            + cols.next_power_of_two().trailing_zeros() as usize;
        let nu = total_vars / 2;
        let sigma = (total_vars - nu).max(1);
        hyrax_commit(&mle.evaluations, nu, &HyraxParams::new(sigma))
    }

    /// LN witness for x_in = [[10,20],[30,40]], gamma=[2,2], beta=[5,5].
    /// var=200, d·sigma=14, sigma=7, y=[[4,6],[4,6]].
    fn build_ln1_witness() -> LayerNormWitness {
        LayerNormWitness {
            x: vec![
                vec![F::from(10u64), F::from(20u64)],
                vec![F::from(30u64), F::from(40u64)],
            ],
            y: vec![
                vec![F::from(4u64), F::from(6u64)],
                vec![F::from(4u64), F::from(6u64)],
            ],
            sum_x: vec![F::from(30u64), F::from(70u64)],
            sigma: vec![F::from(7u64), F::from(7u64)],
            // sq_sum_x[i] = sum_j x[i][j]^2: row0=10^2+20^2=500, row1=30^2+40^2=2500
            sq_sum_x: vec![F::from(500u64), F::from(2500u64)],
            // sum_x_sq[i] = sum_x[i]^2: 30^2=900, 70^2=4900
            sum_x_sq: vec![F::from(900u64), F::from(4900u64)],
            // sigma_sq_scaled[i] = (d*sigma[i])^2 = (2*7)^2=196
            sigma_sq_scaled: vec![F::from(196u64), F::from(196u64)],
        }
    }

    /// LN witness for x_mid = [[138,20],[158,40]], gamma=[2,2], beta=[5,5].
    /// var=27848, d·sigma=166, sigma=83, y=[[6,4],[6,4]].
    fn build_ln2_witness() -> LayerNormWitness {
        LayerNormWitness {
            x: vec![
                vec![F::from(138u64), F::from(20u64)],
                vec![F::from(158u64), F::from(40u64)],
            ],
            y: vec![
                vec![F::from(6u64), F::from(4u64)],
                vec![F::from(6u64), F::from(4u64)],
            ],
            sum_x: vec![F::from(158u64), F::from(198u64)],
            sigma: vec![F::from(83u64), F::from(83u64)],
            // sq_sum_x[i] = sum_j x[i][j]^2: row0=138^2+20^2=19444, row1=158^2+40^2=26564
            sq_sum_x: vec![F::from(19444u64), F::from(26564u64)],
            // sum_x_sq[i] = sum_x[i]^2: 158^2=24964, 198^2=39204
            sum_x_sq: vec![F::from(24964u64), F::from(39204u64)],
            // sigma_sq_scaled[i] = (d*sigma[i])^2 = (2*83)^2=27556
            sigma_sq_scaled: vec![F::from(27556u64), F::from(27556u64)],
        }
    }

    /// LN witness for x_out = [[144,20],[164,40]], gamma=[2,2], beta=[5,5].
    /// var=30752, d·sigma=174, sigma=87, y=[[6,4],[6,4]].
    fn build_ln_final_witness() -> LayerNormWitness {
        LayerNormWitness {
            x: vec![
                vec![F::from(144u64), F::from(20u64)],
                vec![F::from(164u64), F::from(40u64)],
            ],
            y: vec![
                vec![F::from(6u64), F::from(4u64)],
                vec![F::from(6u64), F::from(4u64)],
            ],
            sum_x: vec![F::from(164u64), F::from(204u64)],
            sigma: vec![F::from(87u64), F::from(87u64)],
            // sq_sum_x[i] = sum_j x[i][j]^2: row0=144^2+20^2=21136, row1=164^2+40^2=28496
            sq_sum_x: vec![F::from(21136u64), F::from(28496u64)],
            // sum_x_sq[i] = sum_x[i]^2: 164^2=26896, 204^2=41616
            sum_x_sq: vec![F::from(26896u64), F::from(41616u64)],
            // sigma_sq_scaled[i] = (d*sigma[i])^2 = (2*87)^2=30276
            sigma_sq_scaled: vec![F::from(30276u64), F::from(30276u64)],
        }
    }

    /// Build a Lasso instance from explicit query indices and outputs.
    fn build_lasso(indices: Vec<usize>, outputs: Vec<u64>) -> LassoInstance {
        let table: Vec<F> = (0u64..1 << M_BITS).map(F::from).collect();
        LassoInstance {
            tables: vec![table],
            query_indices: indices,
            outputs: outputs.into_iter().map(F::from).collect(),
            bits_per_chunk: M_BITS,
        }
    }

    /// Build a Lasso instance whose every query indexes slot 0 (output = 0).
    fn build_zero_lasso(num_queries: usize) -> LassoInstance {
        build_lasso(vec![0usize; num_queries], vec![0u64; num_queries])
    }

    /// Shared HyraxParams for the 4-bit Lasso table (sigma = m - m/2 = 2).
    fn lasso_params() -> HyraxParams {
        HyraxParams::new(M_BITS / 2)
    }

    // -----------------------------------------------------------------------
    // Block-level fixture
    // -----------------------------------------------------------------------

    fn build_block_witness_and_instances() -> (
        TransformerBlockWitness,
        LinearAttentionInstance,
        FFNInstance,
    ) {
        // x_in = [[10,20],[30,40]]
        let x_in = vec![
            vec![F::from(10u64), F::from(20u64)],
            vec![F::from(30u64), F::from(40u64)],
        ];

        // LN1: y_norm1 = [[4,6],[4,6]]
        let ln1_wit = build_ln1_witness();
        let y_norm1 = ln1_wit.y.clone();

        // Q/K/V projections: y_norm1 @ W where W[0][0]=1 → [[4,0],[4,0]]
        let proj_out = vec![
            vec![F::from(4u64), F::from(0u64)],
            vec![F::from(4u64), F::from(0u64)],
        ];
        let q_proj_wit = ProjectionWitness {
            x: y_norm1.clone(),
            y: proj_out.clone(),
        };
        let k_proj_wit = ProjectionWitness {
            x: y_norm1.clone(),
            y: proj_out.clone(),
        };
        let v_proj_wit = ProjectionWitness {
            x: y_norm1.clone(),
            y: proj_out.clone(),
        };

        // phi is identity on [0,15]: phi_q = phi_k = [[4,0],[4,0]]
        // context = phi_k^T @ v = [[4,4],[0,0]] @ [[4,0],[4,0]] = [[32,0],[0,0]]
        let context = vec![
            vec![F::from(32u64), F::from(0u64)],
            vec![F::from(0u64), F::from(0u64)],
        ];
        // attn inner = phi_q @ context = [[4,0],[4,0]] @ [[32,0],[0,0]] = [[128,0],[128,0]]
        let attn_inner = vec![
            vec![F::from(128u64), F::from(0u64)],
            vec![F::from(128u64), F::from(0u64)],
        ];

        let attn_wit = LinearAttentionWitness {
            q: proj_out.clone(),
            k: proj_out.clone(),
            v: proj_out.clone(),
            phi_q: proj_out.clone(),
            phi_k: proj_out.clone(),
            context: context.clone(),
            out: attn_inner.clone(),
        };

        // O projection: attn_inner @ W where W[0][0]=1 → [[128,0],[128,0]]
        let out_attn = attn_inner.clone();
        let o_proj_wit = ProjectionWitness {
            x: attn_inner.clone(),
            y: out_attn.clone(),
        };

        // Residual 1: x_mid = x_in + out_attn = [[138,20],[158,40]]
        let x_mid = vec![
            vec![F::from(138u64), F::from(20u64)],
            vec![F::from(158u64), F::from(40u64)],
        ];

        // LN2: y_norm2 = [[6,4],[6,4]]
        let ln2_wit = build_ln2_witness();
        let y_norm2 = ln2_wit.y.clone();

        // FFN: W1[0][0]=1 → m=[[6,0,0,0],[6,0,0,0]], phi identity → a=m
        let m_ffn = vec![
            vec![F::from(6u64), F::from(0u64), F::from(0u64), F::from(0u64)],
            vec![F::from(6u64), F::from(0u64), F::from(0u64), F::from(0u64)],
        ];
        // W2[0][0]=1 → out_ffn=[[6,0],[6,0]]
        let out_ffn = vec![
            vec![F::from(6u64), F::from(0u64)],
            vec![F::from(6u64), F::from(0u64)],
        ];
        let ffn_wit = FFNWitness {
            x: y_norm2,
            m: m_ffn.clone(),
            a: m_ffn.clone(),
            y: out_ffn.clone(),
        };

        // Residual 2: x_out = x_mid + out_ffn = [[144,20],[164,40]]
        let x_out = vec![
            vec![F::from(144u64), F::from(20u64)],
            vec![F::from(164u64), F::from(40u64)],
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

        // Lasso for phi(q) and phi(k): indices [4,0,4,0], outputs [4,0,4,0]
        let inst_attn = LinearAttentionInstance {
            seq_len: T,
            d_head: D,
            q_lasso: build_lasso(vec![4, 0, 4, 0], vec![4, 0, 4, 0]),
            k_lasso: build_lasso(vec![4, 0, 4, 0], vec![4, 0, 4, 0]),
        };

        // Lasso for FFN phi(m): indices [6,0,0,0,6,0,0,0], outputs same
        let inst_ffn = FFNInstance {
            activation_lasso: build_lasso(
                vec![6, 0, 0, 0, 6, 0, 0, 0],
                vec![6, 0, 0, 0, 6, 0, 0, 0],
            ),
        };

        (witness, inst_attn, inst_ffn)
    }

    // -----------------------------------------------------------------------
    // Model-level fixture
    // -----------------------------------------------------------------------

    fn build_model_witness(block_wit: TransformerBlockWitness) -> TransformerModelWitness {
        let x_in = block_wit.x_in.clone();

        // Final LN: x_out = [[144,20],[164,40]] → y_final = [[6,4],[6,4]]
        let final_ln_wit = build_ln_final_witness();
        let y_final = final_ln_wit.y.clone();

        // LM head diagonal: y_final @ I = [[6,4],[6,4]]
        let lm_head_wit = ProjectionWitness {
            x: y_final.clone(),
            y: y_final,
        };

        TransformerModelWitness {
            x_in,
            block_witnesses: vec![block_wit],
            final_ln_wit,
            lm_head_wit,
        }
    }

    // -----------------------------------------------------------------------
    // prove_transformer_block / verify_transformer_block tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prove_verify_transformer_block_e2e() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_e2e");
        let proof = prove_transformer_block(
            &witness,
            &x_in_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut pt,
            &lp,
        )
        .unwrap();

        // Derive the expected output commitment the same way the verifier does.
        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_e2e");
        let mut ln_acc_t = crate::pcs::HyraxBatchAccumulator::new();
        let mut ln_acc_td = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_w = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_b = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_sig = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_y = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_m = crate::pcs::HyraxBatchAccumulator::new();
        let mut inter_acc = crate::pcs::HyraxBatchAccumulator::new();
        let result = verify_transformer_block(
            &proof,
            &x_in_com,
            &x_out_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut vt,
            &lp,
            &mut ln_acc_t,
            &mut ln_acc_td,
            &mut proj_acc_w,
            &mut proj_acc_b,
            &mut acc_range_sig,
            &mut acc_range_y,
            &mut acc_range_m,
            &mut inter_acc,
        );
        assert!(result.is_ok(), "Block verification failed: {:?}", result.err());
        // Finalize layer-folding accumulator to run the deferred MSM check.
        let t_bits = T.next_power_of_two().trailing_zeros() as usize;
        let d_bits = D.next_power_of_two().trailing_zeros() as usize;
        let (_, _, params_td) = crate::pcs::params_from_vars(t_bits + d_bits);
        inter_acc.finalize(&params_td, &mut vt)
            .expect("inter_acc finalize failed");
    }

    /// Passing the wrong x_out_com must trigger the residual-connection binding check.
    #[test]
    fn test_block_rejects_wrong_x_out_com() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_wrong_out");
        let proof = prove_transformer_block(
            &witness,
            &x_in_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut pt,
            &lp,
        )
        .unwrap();

        // Provide a commitment to a completely different matrix as x_out.
        let wrong_x_out_com = commit_mat_test(
            &vec![
                vec![F::from(1u64), F::from(2u64)],
                vec![F::from(3u64), F::from(4u64)],
            ],
            T,
            D,
        );

        let mut vt = Transcript::new(b"block_wrong_out");
        let mut ln_acc_t = crate::pcs::HyraxBatchAccumulator::new();
        let mut ln_acc_td = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_w = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_b = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_sig = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_y = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_m = crate::pcs::HyraxBatchAccumulator::new();
        let mut inter_acc = crate::pcs::HyraxBatchAccumulator::new();
        let result = verify_transformer_block(
            &proof,
            &x_in_com,
            &wrong_x_out_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut vt,
            &lp,
            &mut ln_acc_t,
            &mut ln_acc_td,
            &mut proj_acc_w,
            &mut proj_acc_b,
            &mut acc_range_sig,
            &mut acc_range_y,
            &mut acc_range_m,
            &mut inter_acc,
        );
        // Error comes from the residual commitment check, before inter_acc is finalized.
        assert!(result.is_err(), "Should reject wrong x_out_com");
    }

    /// Tampering with the LN1 proof must be detected by the sub-verifier.
    #[test]
    fn test_block_rejects_tampered_ln1_opening() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_tamper_ln1");
        let mut proof = prove_transformer_block(
            &witness,
            &x_in_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut pt,
            &lp,
        )
        .unwrap();

        // Perturb the sum_x opening inside the LN1 sub-proof.
        proof.ln1_proof.openings.sum_x_at_rt += F::ONE;

        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_tamper_ln1");
        let mut ln_acc_t = crate::pcs::HyraxBatchAccumulator::new();
        let mut ln_acc_td = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_w = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_b = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_sig = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_y = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_m = crate::pcs::HyraxBatchAccumulator::new();
        let mut inter_acc = crate::pcs::HyraxBatchAccumulator::new();
        let result = verify_transformer_block(
            &proof,
            &x_in_com,
            &x_out_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut vt,
            &lp,
            &mut ln_acc_t,
            &mut ln_acc_td,
            &mut proj_acc_w,
            &mut proj_acc_b,
            &mut acc_range_sig,
            &mut acc_range_y,
            &mut acc_range_m,
            &mut inter_acc,
        );
        // Error comes from LN1 sub-verifier (step 1), before any inter_acc entries.
        assert!(result.is_err(), "Should reject tampered LN1 proof");
    }

    /// Tampering with an intermediate commitment breaks the pipeline binding.
    #[test]
    fn test_block_rejects_tampered_x_norm1_com() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_tamper_xnorm1");
        let mut proof = prove_transformer_block(
            &witness,
            &x_in_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut pt,
            &lp,
        )
        .unwrap();

        // Swap x_norm1_com for a commitment to a different matrix.
        proof.x_norm1_com = commit_mat_test(
            &vec![
                vec![F::from(99u64), F::from(99u64)],
                vec![F::from(99u64), F::from(99u64)],
            ],
            T,
            D,
        );

        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_tamper_xnorm1");
        let mut ln_acc_t = crate::pcs::HyraxBatchAccumulator::new();
        let mut ln_acc_td = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_w = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_b = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_sig = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_y = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_m = crate::pcs::HyraxBatchAccumulator::new();
        let mut inter_acc = crate::pcs::HyraxBatchAccumulator::new();
        let result = verify_transformer_block(
            &proof,
            &x_in_com,
            &x_out_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut vt,
            &lp,
            &mut ln_acc_t,
            &mut ln_acc_td,
            &mut proj_acc_w,
            &mut proj_acc_b,
            &mut acc_range_sig,
            &mut acc_range_y,
            &mut acc_range_m,
            &mut inter_acc,
        );
        // The inner-product check in add_verify passes (w' is from the original data),
        // but the MSM check is deferred to finalize — that is where the tampered
        // commitment is caught.
        let t_bits = T.next_power_of_two().trailing_zeros() as usize;
        let d_bits = D.next_power_of_two().trailing_zeros() as usize;
        let (_, _, params_td) = crate::pcs::params_from_vars(t_bits + d_bits);
        let final_result = result.and_then(|_| inter_acc.finalize(&params_td, &mut vt));
        assert!(final_result.is_err(), "Should reject tampered x_norm1_com");
    }

    // -----------------------------------------------------------------------
    // prove (model-level) / verify (model-level) tests
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
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &mut vt, &lp);
        assert!(
            result.is_ok(),
            "Model verification failed: {:?}",
            result.err()
        );
    }

    /// Tampering with a block sub-proof must propagate up to the model verifier.
    #[test]
    fn test_model_rejects_tampered_block_proof() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_block");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();

        proof.block_proofs[0].ln1_proof.openings.sum_x_at_rt += F::ONE;

        let mut vt = Transcript::new(b"model_tamper_block");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered block LN1 proof");
    }

    /// Tampering with the final-LN proof must be detected.
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
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered final LN proof");
    }

    /// Tampering with the LM head proof must be detected.
    #[test]
    fn test_model_rejects_tampered_lm_head_proof() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_lm");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();

        // Shift the claimed y_eval in the LM-head Sumcheck
        proof.lm_head_proof.openings.y_eval += F::ONE;

        let mut vt = Transcript::new(b"model_tamper_lm");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered LM head proof");
    }

    /// Shifting x_in_com in the proof makes every downstream check fail.
    #[test]
    fn test_model_rejects_tampered_x_in_com() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_xin");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();

        // Replace x_in_com with a commitment to a different matrix.
        proof.x_in_com = commit_mat_test(
            &vec![
                vec![F::from(1u64), F::from(1u64)],
                vec![F::from(1u64), F::from(1u64)],
            ],
            T,
            D,
        );

        let mut vt = Transcript::new(b"model_tamper_xin");
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered x_in_com");
    }

    // -----------------------------------------------------------------------
    // Homomorphic commitment helper
    // -----------------------------------------------------------------------

    /// Verify that add_commitments correctly implements Com(a) + Com(b) = Com(a+b).
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

        assert_eq!(
            com_sum.row_coms, com_apb.row_coms,
            "Com(a) + Com(b) must equal Com(a+b)"
        );
    }

    /// add_commitments(C, Com(0)) must equal C (zero is the identity).
    #[test]
    fn test_add_commitments_with_zero_is_identity() {
        use crate::pcs::params_from_vars;

        let a = vec![F::from(5u64), F::from(9u64), F::from(2u64), F::from(14u64)];
        let zero = vec![F::ZERO; 4];

        let (nu, _sigma, params) = params_from_vars(2);
        let com_a = hyrax_commit(&a, nu, &params);
        let com_zero = hyrax_commit(&zero, nu, &params);
        let com_sum = add_commitments(&com_a, &com_zero);

        assert_eq!(
            com_sum.row_coms, com_a.row_coms,
            "Com(a) + Com(0) must equal Com(a)"
        );
    }
}
