//! Global Prover for a full Transformer Block.
//!
//! **GKR-style Two-Phase Architecture:**
//! Phase 1 (per block): commit all 7 intermediate matrices, absorb into transcript,
//!   run range proofs + LN1 + LN2 sub-provers.
//! Phase 2 (per block, after global r_td is derived): run QKV/O-proj/Attn/FFN
//!   sumchecks, all using r_td as the shared output eval point.
//! Global: ONE hyrax_open_batch at r_td for all 5L intermediate matrices
//!   (Q, K, V, out_attn, out_ffn across all L blocks) — reduces 5L MSMs to 1.

use crate::field::F;
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_open, hyrax_open_batch, params_from_vars,
    HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::poly::utils::mat_to_mle;
use crate::transcript::{challenge_vec, Transcript};

use crate::attention::attention::{
    prove_linear_attention, AttentionIOCommitments,
    LinearAttentionInstance, LinearAttentionProof, LinearAttentionWitness,
};
use crate::attention::layernorm::{
    compute_range_witnesses, prove_layernorm, LayerNormIOCommitments, LayerNormProof,
    LayerNormVerifyingKey, LayerNormWitness,
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
use crate::verifier::{add_commitments, TransformerBlockVerifyingKey};

// ---------------------------------------------------------------------------
// Proof Structures
// ---------------------------------------------------------------------------

/// ZK Proof for one Transformer Block.
///
/// Intermediate openings strategy (GKR-style):
///   - q_eval, k_eval, v_eval_rtd, out_attn_eval, out_ffn_eval: evals at shared r_td,
///     proven by the model-level global hyrax_open_batch (TransformerModelProof.inter_batch_open).
///   - x_norm1_open, x_norm2_open: per-block opens at block-specific sumcheck reduction points.
///   - v_attn_open: per-block open of v_com at attention's internal eval point.
pub struct TransformerBlockProof {
    pub ln1_proof: LayerNormProof,
    pub qkv_proj_proof: BatchedQKVProjectionProof,
    pub o_proj_proof: ProjectionProof,
    pub attn_proof: LinearAttentionProof,
    pub ln2_proof: LayerNormProof,
    pub ffn_proof: FFNProof,

    // Committed intermediate matrices
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

    // Per-block opens at points that differ per block
    pub x_norm1_open: HyraxProof,
    pub x_norm2_open: HyraxProof,
    pub v_attn_open: HyraxProof,
    pub v_attn_eval: F,

    pub block_range_m: GlobalRangeM,
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
    // This ensures skip_io_absorb=true in Phase 2 results in consistent transcript.
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

    // Absorb out_ffn_com with FFN's y_com label so FFN Phase 2 can skip it.
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
// Phase 2: sumchecks with shared r_td
// ---------------------------------------------------------------------------

struct BlockPhase2Result {
    qkv_proj_proof: BatchedQKVProjectionProof,
    o_proj_proof: ProjectionProof,
    attn_proof: LinearAttentionProof,
    ffn_proof: FFNProof,
    q_eval: F,
    k_eval: F,
    v_eval_rtd: F,
    out_attn_eval: F,
    out_ffn_eval: F,
    x_norm1_open: HyraxProof,
    x_norm2_open: HyraxProof,
    v_attn_open: HyraxProof,
    v_attn_eval: F,
}

fn prove_block_phase2(
    witness: &TransformerBlockWitness,
    phase1: &BlockPhase1Data,
    pk: &TransformerBlockVerifyingKey,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
    r_td: &[F],
) -> Result<BlockPhase2Result, String> {
    let t = pk.seq_len;
    let d = pk.d_model;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;
    let (nu_td, sigma_td, _) = params_from_vars(td_num_vars);

    // QKV projections: x_com=None (x_norm1_com already in transcript from LN1 Phase 1)
    let qkv_wit = BatchedQKVProjectionWitness {
        x: witness.ln1_wit.y.clone(),
        q: witness.q_proj_wit.y.clone(),
        k: witness.k_proj_wit.y.clone(),
        v: witness.v_proj_wit.y.clone(),
    };
    let qkv_io = BatchedQKVProjectionIOCommitments { x_com: None };
    let (qkv_proj_proof, q_y_claim, k_y_claim, v_y_claim, x_norm1_claim) =
        prove_qkv_projections(&pk.q_pk, &pk.k_pk, &pk.v_pk, &qkv_wit, &qkv_io, transcript, r_td)?;

    // O-proj: x_com=None (GKR backward, out_inner not committed)
    let o_io = ProjectionIOCommitments { x_com: None };
    let (o_proj_proof, o_y_claim, o_x_claim) =
        prove_projection(&pk.o_pk, &witness.o_proj_wit, &o_io, transcript, Some(r_td))?;

    // Attention: skip_io_absorb=true (q/k/v_com absorbed in Phase 1)
    let attn_io = AttentionIOCommitments {
        q_com: phase1.q_com.clone(),
        k_com: phase1.k_com.clone(),
        v_com: phase1.v_com.clone(),
        skip_io_absorb: true,
    };
    let (attn_proof, _attn_out_claim, attn_v_claim) = prove_linear_attention(
        &witness.attn_wit,
        inst_attn,
        &pk.attn_pk,
        &attn_io,
        Some(o_x_claim.clone()),
        transcript,
        lasso_params,
    );

    // FFN: x_com=Some(x_norm2_com), y_com=None (out_ffn_com absorbed in Phase 1)
    let ffn_io = FFNIOCommitments {
        x_com: Some(phase1.x_norm2_com.clone()),
        y_com: None,
    };
    let (ffn_proof, ffn_y_claim, ffn_x_claim) = prove_ffn(
        &pk.ffn_pk,
        &witness.ffn_wit,
        inst_ffn,
        &ffn_io,
        transcript,
        lasso_params,
        Some(r_td),
    )?;

    // Evals at r_td (from respective sumchecks)
    let q_eval = q_y_claim.value;
    let k_eval = k_y_claim.value;
    let v_eval_rtd = v_y_claim.value;
    let out_attn_eval = o_y_claim.value;
    let out_ffn_eval = ffn_y_claim.value;

    // Per-block opens at block-specific eval points
    let x_norm1_evals = mat_to_mle(&witness.ln1_wit.y, t, d).evaluations;
    let x_norm1_open = hyrax_open(&x_norm1_evals, &x_norm1_claim.point, nu_td, sigma_td);

    let x_norm2_evals = mat_to_mle(&witness.ln2_wit.y, t, d).evaluations;
    let x_norm2_open = hyrax_open(&x_norm2_evals, &ffn_x_claim.point, nu_td, sigma_td);

    let v_evals = mat_to_mle(&witness.attn_wit.v, t, d).evaluations;
    let v_attn_open = hyrax_open(&v_evals, &attn_v_claim.point, nu_td, sigma_td);
    let v_attn_eval = attn_v_claim.value;

    Ok(BlockPhase2Result {
        qkv_proj_proof,
        o_proj_proof,
        attn_proof,
        ffn_proof,
        q_eval,
        k_eval,
        v_eval_rtd,
        out_attn_eval,
        out_ffn_eval,
        x_norm1_open,
        x_norm2_open,
        v_attn_open,
        v_attn_eval,
    })
}

// ---------------------------------------------------------------------------
// Block-level prover (self-contained, for testing)
// ---------------------------------------------------------------------------

/// Prove a single transformer block. Derives r_td locally (from single-block Phase 1).
/// Returns (block_proof, inter_batch_open) where inter_batch_open is the batch
/// Hyrax proof for the 5 intermediate matrices at r_td.
pub fn prove_transformer_block(
    witness: &TransformerBlockWitness,
    x_in_com: &HyraxCommitment,
    pk: &TransformerBlockVerifyingKey,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> Result<(TransformerBlockProof, HyraxProof), String> {
    let t = pk.seq_len;
    let d = pk.d_model;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;

    // Phase 1
    let phase1 = commit_block_phase1(witness, x_in_com, pk, transcript)?;

    // Derive local r_td after single-block Phase 1
    let r_td = challenge_vec(transcript, td_num_vars, b"gkr_r_td");

    // Phase 2
    let p2 = prove_block_phase2(
        witness, &phase1, pk, inst_attn, inst_ffn, transcript, lasso_params, &r_td,
    )?;

    // Per-block batch open at r_td for 5 intermediate matrices
    let (nu_td, sigma_td, _) = params_from_vars(td_num_vars);
    let q_evals = mat_to_mle(&witness.attn_wit.q, t, d).evaluations;
    let k_evals = mat_to_mle(&witness.attn_wit.k, t, d).evaluations;
    let v_evals = mat_to_mle(&witness.attn_wit.v, t, d).evaluations;
    let out_attn_evals = mat_to_mle(&witness.o_proj_wit.y, t, d).evaluations;
    let out_ffn_evals = mat_to_mle(&witness.ffn_wit.y, t, d).evaluations;
    let inter_batch_open = hyrax_open_batch(
        &[
            q_evals.as_slice(),
            k_evals.as_slice(),
            v_evals.as_slice(),
            out_attn_evals.as_slice(),
            out_ffn_evals.as_slice(),
        ],
        &r_td,
        nu_td,
        sigma_td,
        transcript,
    );

    let block_proof = TransformerBlockProof {
        ln1_proof: phase1.ln1_proof,
        qkv_proj_proof: p2.qkv_proj_proof,
        o_proj_proof: p2.o_proj_proof,
        attn_proof: p2.attn_proof,
        ln2_proof: phase1.ln2_proof,
        ffn_proof: p2.ffn_proof,
        x_norm1_com: phase1.x_norm1_com,
        q_com: phase1.q_com,
        k_com: phase1.k_com,
        v_com: phase1.v_com,
        out_attn_com: phase1.out_attn_com,
        x_norm2_com: phase1.x_norm2_com,
        out_ffn_com: phase1.out_ffn_com,
        q_eval: p2.q_eval,
        k_eval: p2.k_eval,
        v_eval_rtd: p2.v_eval_rtd,
        out_attn_eval: p2.out_attn_eval,
        out_ffn_eval: p2.out_ffn_eval,
        x_norm1_open: p2.x_norm1_open,
        x_norm2_open: p2.x_norm2_open,
        v_attn_open: p2.v_attn_open,
        v_attn_eval: p2.v_attn_eval,
        block_range_m: phase1.block_range_m,
    };

    Ok((block_proof, inter_batch_open))
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
    /// Single Hyrax batch proof for all 5L intermediate matrices at shared r_td.
    pub inter_batch_open: HyraxProof,
}

// ---------------------------------------------------------------------------
// Model Prover (E2E) — two-phase loop for 7L → L MSM reduction
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
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;

    let commit_mat = |mat: &[Vec<F>], rows: usize, cols: usize| -> HyraxCommitment {
        let mle = mat_to_mle(mat, rows, cols);
        let vars = rows.next_power_of_two().trailing_zeros()
            + cols.next_power_of_two().trailing_zeros();
        let (nu, _, params) = params_from_vars(vars as usize);
        hyrax_commit(&mle.evaluations, nu, &params)
    };

    // 1. Initial input commitment
    let x_in_com = commit_mat(&witness.x_in, t, d);
    absorb_com(transcript, b"x_in_com", &x_in_com);

    // 2. Phase 1: commit all blocks' intermediates + run LN proofs
    let mut phase1_data: Vec<BlockPhase1Data> = Vec::with_capacity(pk.vk.num_blocks);
    let mut current_x_com = x_in_com.clone();

    for i in 0..pk.vk.num_blocks {
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

    // 3. Derive global r_td after ALL blocks' Phase 1 commitments
    let r_td = challenge_vec(transcript, td_num_vars, b"gkr_r_td");

    // 4. Phase 2: sumchecks for all blocks using shared r_td
    let mut block_proofs: Vec<TransformerBlockProof> = Vec::with_capacity(pk.vk.num_blocks);

    for i in 0..pk.vk.num_blocks {
        let p2 = prove_block_phase2(
            &witness.block_witnesses[i],
            &phase1_data[i],
            &pk.block_pks[i],
            inst_attn,
            inst_ffn,
            transcript,
            lasso_params,
            &r_td,
        )?;

        let p1 = &phase1_data[i];
        block_proofs.push(TransformerBlockProof {
            ln1_proof: p1.ln1_proof.clone(),
            qkv_proj_proof: p2.qkv_proj_proof,
            o_proj_proof: p2.o_proj_proof,
            attn_proof: p2.attn_proof,
            ln2_proof: p1.ln2_proof.clone(),
            ffn_proof: p2.ffn_proof,
            x_norm1_com: p1.x_norm1_com.clone(),
            q_com: p1.q_com.clone(),
            k_com: p1.k_com.clone(),
            v_com: p1.v_com.clone(),
            out_attn_com: p1.out_attn_com.clone(),
            x_norm2_com: p1.x_norm2_com.clone(),
            out_ffn_com: p1.out_ffn_com.clone(),
            q_eval: p2.q_eval,
            k_eval: p2.k_eval,
            v_eval_rtd: p2.v_eval_rtd,
            out_attn_eval: p2.out_attn_eval,
            out_ffn_eval: p2.out_ffn_eval,
            x_norm1_open: p2.x_norm1_open,
            x_norm2_open: p2.x_norm2_open,
            v_attn_open: p2.v_attn_open,
            v_attn_eval: p2.v_attn_eval,
            block_range_m: p1.block_range_m.clone(),
        });
    }

    // 5. Final LayerNorm
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

    // 6. LM Head
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

    // 7. Advance transcript to match verifier's 10 accumulator finalizations
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
    let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");

    // 8. Global hyrax_open_batch for 5L intermediate matrices at shared r_td
    let (nu_td, sigma_td, _) = params_from_vars(td_num_vars);
    let mut all_evals_vecs: Vec<Vec<F>> = Vec::with_capacity(5 * pk.vk.num_blocks);
    for i in 0..pk.vk.num_blocks {
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

    // 9. Global batched Lasso
    let mut all_lasso_instances = Vec::new();
    let mut all_instance_coms = Vec::new();
    let mut all_output_bindings: Vec<LassoOutputBinding> = Vec::new();
    let mut all_query_indices: Vec<Vec<usize>> = Vec::new();
    let mut global_nu = 0usize;
    for i in 0..pk.vk.num_blocks {
        let bpk = &pk.block_pks[i];
        let attn_wit = &witness.block_witnesses[i].attn_wit;
        let phi_q_mle = mat_to_mle(&attn_wit.phi_q, t, d);
        let phi_k_mle = mat_to_mle(&attn_wit.phi_k, t, d);

        all_lasso_instances.push(inst_attn.q_lasso.clone());
        all_lasso_instances.push(inst_attn.k_lasso.clone());
        all_query_indices.push(inst_attn.q_query_indices.clone());
        all_query_indices.push(inst_attn.k_query_indices.clone());
        all_instance_coms.push(bpk.attn_pk.qk_lasso_pk.instance_table_coms[0].clone());
        all_instance_coms.push(bpk.attn_pk.qk_lasso_pk.instance_table_coms[1].clone());
        global_nu = bpk.attn_pk.qk_lasso_pk.nu;

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
        inter_batch_open,
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
    // Block-level tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prove_verify_transformer_block_e2e() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();
        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_e2e");
        let (proof, inter_batch_open) = prove_transformer_block(
            &witness, &x_in_com, &pk.block_pks[0], &inst_attn, &inst_ffn, &mut pt, &lp,
        ).unwrap();

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
            &proof, &inter_batch_open, &x_in_com, &x_out_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut vt, &lp,
            &mut ln_acc_t, &mut ln_acc_td, &mut proj_acc_w, &mut proj_acc_b,
            &mut acc_range_sig, &mut acc_range_y, &mut acc_range_m, &mut inter_acc,
        );
        assert!(result.is_ok(), "Block verification failed: {:?}", result.err());
        let t_bits = T.next_power_of_two().trailing_zeros() as usize;
        let d_bits = D.next_power_of_two().trailing_zeros() as usize;
        let (_, _, params_td) = crate::pcs::params_from_vars(t_bits + d_bits);
        inter_acc.finalize(&params_td, &mut vt).expect("inter_acc finalize failed");
    }

    #[test]
    fn test_block_rejects_wrong_x_out_com() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();
        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_wrong_out");
        let (proof, inter_batch_open) = prove_transformer_block(
            &witness, &x_in_com, &pk.block_pks[0], &inst_attn, &inst_ffn, &mut pt, &lp,
        ).unwrap();

        let wrong_x_out_com = commit_mat_test(
            &vec![vec![F::from(1u64), F::from(2u64)], vec![F::from(3u64), F::from(4u64)]],
            T, D,
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
            &proof, &inter_batch_open, &x_in_com, &wrong_x_out_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut vt, &lp,
            &mut ln_acc_t, &mut ln_acc_td, &mut proj_acc_w, &mut proj_acc_b,
            &mut acc_range_sig, &mut acc_range_y, &mut acc_range_m, &mut inter_acc,
        );
        assert!(result.is_err(), "Should reject wrong x_out_com");
    }

    #[test]
    fn test_block_rejects_tampered_ln1_opening() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();
        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_tamper_ln1");
        let (mut proof, inter_batch_open) = prove_transformer_block(
            &witness, &x_in_com, &pk.block_pks[0], &inst_attn, &inst_ffn, &mut pt, &lp,
        ).unwrap();
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
            &proof, &inter_batch_open, &x_in_com, &x_out_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut vt, &lp,
            &mut ln_acc_t, &mut ln_acc_td, &mut proj_acc_w, &mut proj_acc_b,
            &mut acc_range_sig, &mut acc_range_y, &mut acc_range_m, &mut inter_acc,
        );
        assert!(result.is_err(), "Should reject tampered LN1 proof");
    }

    #[test]
    fn test_block_rejects_tampered_x_norm1_com() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();
        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_tamper_xnorm1");
        let (mut proof, inter_batch_open) = prove_transformer_block(
            &witness, &x_in_com, &pk.block_pks[0], &inst_attn, &inst_ffn, &mut pt, &lp,
        ).unwrap();
        // Swap x_norm1_open with v_attn_open (wrong proof for x_norm1_com)
        proof.x_norm1_open = proof.v_attn_open.clone();

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
            &proof, &inter_batch_open, &x_in_com, &x_out_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut vt, &lp,
            &mut ln_acc_t, &mut ln_acc_td, &mut proj_acc_w, &mut proj_acc_b,
            &mut acc_range_sig, &mut acc_range_y, &mut acc_range_m, &mut inter_acc,
        );
        let t_bits = T.next_power_of_two().trailing_zeros() as usize;
        let d_bits = D.next_power_of_two().trailing_zeros() as usize;
        let (_, _, params_td) = crate::pcs::params_from_vars(t_bits + d_bits);
        let final_result = result.and_then(|_| inter_acc.finalize(&params_td, &mut vt));
        assert!(final_result.is_err(), "Should reject tampered x_norm1_com");
    }

    #[test]
    fn test_block_rejects_tampered_x_norm2_com() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();
        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_tamper_xnorm2");
        let (mut proof, inter_batch_open) = prove_transformer_block(
            &witness, &x_in_com, &pk.block_pks[0], &inst_attn, &inst_ffn, &mut pt, &lp,
        ).unwrap();
        proof.x_norm2_open = proof.v_attn_open.clone();

        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_tamper_xnorm2");
        let mut ln_acc_t = crate::pcs::HyraxBatchAccumulator::new();
        let mut ln_acc_td = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_w = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_b = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_sig = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_y = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_m = crate::pcs::HyraxBatchAccumulator::new();
        let mut inter_acc = crate::pcs::HyraxBatchAccumulator::new();
        let result = verify_transformer_block(
            &proof, &inter_batch_open, &x_in_com, &x_out_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut vt, &lp,
            &mut ln_acc_t, &mut ln_acc_td, &mut proj_acc_w, &mut proj_acc_b,
            &mut acc_range_sig, &mut acc_range_y, &mut acc_range_m, &mut inter_acc,
        );
        let t_bits = T.next_power_of_two().trailing_zeros() as usize;
        let d_bits = D.next_power_of_two().trailing_zeros() as usize;
        let (_, _, params_td) = crate::pcs::params_from_vars(t_bits + d_bits);
        let final_result = result.and_then(|_| inter_acc.finalize(&params_td, &mut vt));
        assert!(final_result.is_err(), "Should reject tampered x_norm2_open");
    }

    #[test]
    fn test_block_rejects_fraudulent_ln1_output() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();
        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_fraud_ln1");
        let (mut proof, inter_batch_open) = prove_transformer_block(
            &witness, &x_in_com, &pk.block_pks[0], &inst_attn, &inst_ffn, &mut pt, &lp,
        ).unwrap();
        proof.x_norm1_com =
            commit_mat_test(&vec![vec![F::from(1u64), F::from(2u64)]; T], T, D);

        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_fraud_ln1");
        let mut ln_acc_t = crate::pcs::HyraxBatchAccumulator::new();
        let mut ln_acc_td = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_w = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_b = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_sig = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_y = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_m = crate::pcs::HyraxBatchAccumulator::new();
        let mut inter_acc = crate::pcs::HyraxBatchAccumulator::new();
        let result = verify_transformer_block(
            &proof, &inter_batch_open, &x_in_com, &x_out_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut vt, &lp,
            &mut ln_acc_t, &mut ln_acc_td, &mut proj_acc_w, &mut proj_acc_b,
            &mut acc_range_sig, &mut acc_range_y, &mut acc_range_m, &mut inter_acc,
        );
        assert!(result.is_err(), "Should reject fraudulent LN1 output commitment");
    }

    #[test]
    fn test_block_rejects_fraudulent_ln2_output() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();
        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_fraud_ln2");
        let (mut proof, inter_batch_open) = prove_transformer_block(
            &witness, &x_in_com, &pk.block_pks[0], &inst_attn, &inst_ffn, &mut pt, &lp,
        ).unwrap();
        proof.x_norm2_com =
            commit_mat_test(&vec![vec![F::from(3u64), F::from(4u64)]; T], T, D);

        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_fraud_ln2");
        let mut ln_acc_t = crate::pcs::HyraxBatchAccumulator::new();
        let mut ln_acc_td = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_w = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_b = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_sig = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_y = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_m = crate::pcs::HyraxBatchAccumulator::new();
        let mut inter_acc = crate::pcs::HyraxBatchAccumulator::new();
        let result = verify_transformer_block(
            &proof, &inter_batch_open, &x_in_com, &x_out_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut vt, &lp,
            &mut ln_acc_t, &mut ln_acc_td, &mut proj_acc_w, &mut proj_acc_b,
            &mut acc_range_sig, &mut acc_range_y, &mut acc_range_m, &mut inter_acc,
        );
        assert!(result.is_err(), "Should reject fraudulent LN2 output commitment");
    }

    #[test]
    fn test_block_rejects_tampered_ln1_sigma_y_binding() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();
        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_tamper_sy_ln1");
        let (mut proof, inter_batch_open) = prove_transformer_block(
            &witness, &x_in_com, &pk.block_pks[0], &inst_attn, &inst_ffn, &mut pt, &lp,
        ).unwrap();
        proof.ln1_proof.openings.y_at_rf_sy =
            proof.ln1_proof.openings.y_at_rf_sy.map(|v| v + F::ONE);

        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_tamper_sy_ln1");
        let mut ln_acc_t = crate::pcs::HyraxBatchAccumulator::new();
        let mut ln_acc_td = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_w = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_b = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_sig = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_y = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_m = crate::pcs::HyraxBatchAccumulator::new();
        let mut inter_acc = crate::pcs::HyraxBatchAccumulator::new();
        let result = verify_transformer_block(
            &proof, &inter_batch_open, &x_in_com, &x_out_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut vt, &lp,
            &mut ln_acc_t, &mut ln_acc_td, &mut proj_acc_w, &mut proj_acc_b,
            &mut acc_range_sig, &mut acc_range_y, &mut acc_range_m, &mut inter_acc,
        );
        assert!(result.is_err(), "Should reject tampered y_at_rf_sy (sigma_y binding)");
    }

    #[test]
    fn test_block_rejects_tampered_qkv_x_eval() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();
        let x_in_com = commit_mat_test(&witness.x_in, T, D);

        let mut pt = Transcript::new(b"block_tamper_qkv_x");
        let (mut proof, inter_batch_open) = prove_transformer_block(
            &witness, &x_in_com, &pk.block_pks[0], &inst_attn, &inst_ffn, &mut pt, &lp,
        ).unwrap();
        proof.qkv_proj_proof.openings.x_eval += F::ONE;

        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_tamper_qkv_x");
        let mut ln_acc_t = crate::pcs::HyraxBatchAccumulator::new();
        let mut ln_acc_td = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_w = crate::pcs::HyraxBatchAccumulator::new();
        let mut proj_acc_b = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_sig = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_y = crate::pcs::HyraxBatchAccumulator::new();
        let mut acc_range_m = crate::pcs::HyraxBatchAccumulator::new();
        let mut inter_acc = crate::pcs::HyraxBatchAccumulator::new();
        let result = verify_transformer_block(
            &proof, &inter_batch_open, &x_in_com, &x_out_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut vt, &lp,
            &mut ln_acc_t, &mut ln_acc_td, &mut proj_acc_w, &mut proj_acc_b,
            &mut acc_range_sig, &mut acc_range_y, &mut acc_range_m, &mut inter_acc,
        );
        assert!(result.is_err(), "Should reject tampered QKV x_eval");
    }

    // -----------------------------------------------------------------------
    // Model-level tests
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
        assert!(result.is_ok(), "Model verification failed: {:?}", result.err());
    }

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
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &mut vt, &lp);
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
        let result = verify(&proof, &pk.vk, &inst_attn, &inst_ffn, &mut vt, &lp);
        assert!(result.is_err(), "Should reject tampered x_in_com");
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
