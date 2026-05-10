//! Global Prover for a full Transformer Model.
//!
//! **Cross-block Batch Sumcheck Architecture:**
//! Phase 1 (all L blocks): commit 7 intermediate matrices per block, run LN proofs.
//! Phase 2 (model-level batch): four cross-block SumcheckProofMulti (QKV, O-proj, FFN-Y, FFN-M)
//!   share a single r_k per type across all L blocks → O(1) batch Hyrax opens per weight type.
//! Global: 5L intermediate matrices opened at shared r_td (inter_batch_open),
//!   plus 13 cross-block weight/activation batch opens.

use crate::field::F;
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_open, hyrax_open_batch, params_from_vars, HyraxCommitment,
    HyraxParams, HyraxProof,
};
use crate::poly::utils::{
    combine, convert_tm_to_fm, eval_cols, eval_cols_ternary, eval_rows, mat_to_mle, vec_to_mle,
};
use crate::poly::DenseMLPoly;
use crate::transcript::{challenge_vec, Transcript};
use ark_ff::{Field, PrimeField};

use crate::attention::attention::{LinearAttentionInstance, LinearAttentionWitness};
use crate::attention::layernorm::{
    compute_range_witnesses, prove_layernorms_batched, LayerNormIOCommitments,
    LayerNormRangeWitnesses, LayerNormVerifyingKey, LayerNormWitness, LayerNormsBatchedInput,
    LayerNormsBatchedProof,
};
use crate::attention::projection::{
    prove_projection, ProjectionIOCommitments, ProjectionProof, ProjectionProvingKey,
    ProjectionVerifyingKey, ProjectionWitness,
};
use crate::ffn::ffn::{FFNInstance, FFNWitness};
use crate::lookup::lasso::{
    prove_lasso_multi, LassoMultiInstance, LassoMultiProof, LassoMultiProvingKey,
    LassoOutputBinding, LassoProof,
};
use crate::lookup::quantization::{prove_quantization_batch, QuantizationProof};
use crate::lookup::range::{
    prove_range_batched, GlobalRangeM, RangeProofWitness, RangeWitnessProof,
};
use crate::subprotocols::{
    prove_sumcheck_cubic_multi_batched_owned, prove_sumcheck_multi_batched_owned,
    SumcheckCubicProofMulti, SumcheckProofMulti,
};
use crate::verifier::{add_commitments, TransformerBlockVerifyingKey};
use rayon::prelude::*;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Proof Structures
// ---------------------------------------------------------------------------

/// ZK Proof for one Transformer Block (per-block data only).
///
/// The cross-block sumchecks and batch opens live in TransformerModelProof.
/// As of PROOF_VERSION 13, all per-LayerNorm proofs are bundled into the
/// model-level `ln_batched_proof` (one cross-LN GKR-style protocol covering
/// every LN in the model); the per-block container only holds commitments.
pub struct TransformerBlockProof {
    // Committed outputs:
    //   q_norm_y_com / k_norm_y_com — post-norm Q/K (input to φ Lasso)
    //   attn_out_norm_y_com — post-norm o_proj output (input to residual)
    pub q_norm_y_com: HyraxCommitment,
    pub k_norm_y_com: HyraxCommitment,
    pub attn_out_norm_y_com: HyraxCommitment,

    // FFN per-block: Lasso for activation + A/M commitments
    pub ffn_lasso_proof: LassoProof,
    pub ffn_a_com: HyraxCommitment,
    pub ffn_m_com: HyraxCommitment,

    // Committed intermediate matrices (7 per block).
    // q_com / k_com / v_com commit to the **post-projection (pre-norm)** matrices
    // (q_proj_wit.y, k_proj_wit.y, v_proj_wit.y); the QKV sumcheck binds them to
    // W_q · ln1_y / W_k · ln1_y / W_v · ln1_y. Lasso φ binds against q_norm_y_com /
    // k_norm_y_com (the post-norm Q/K, which is the φ input).
    pub x_norm1_com: HyraxCommitment,
    pub q_com: HyraxCommitment,
    pub k_com: HyraxCommitment,
    pub v_com: HyraxCommitment,
    pub attn_norm_com: Option<HyraxCommitment>,
    pub attn_num_com: Option<HyraxCommitment>,
    pub attn_z_com: Option<HyraxCommitment>,
    pub attn_rem_com: Option<HyraxCommitment>,
    pub attn_diff_com: Option<HyraxCommitment>,
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
    pub causal_context_com: Option<HyraxCommitment>,

    // Per-block scalars for cross-block attention batch sumchecks
    pub attn_out_eval: F, // x_inner_i(r_t, r_k_o) = claim for out sumcheck
}

pub const ATTN_NORM_SCALE: u64 = 64;
pub const ATTN_NORM_RANGE_BITS: usize = 64;
const FAST_RANGE_BITS: usize = 32;
const WIDE_RANGE_BITS: usize = 64;

fn field_fits_bits(x: &F, bits: usize) -> bool {
    let limbs = x.into_bigint();
    let words = limbs.as_ref();
    match bits {
        32 => words[0] <= u32::MAX as u64 && words[1..].iter().all(|&w| w == 0),
        64 => words[1..].iter().all(|&w| w == 0),
        _ => false,
    }
}

fn choose_range_bits(witness: &RangeProofWitness) -> Result<usize, String> {
    if witness
        .values
        .iter()
        .all(|v| field_fits_bits(v, FAST_RANGE_BITS))
    {
        Ok(FAST_RANGE_BITS)
    } else if witness
        .values
        .iter()
        .all(|v| field_fits_bits(v, WIDE_RANGE_BITS))
    {
        Ok(WIDE_RANGE_BITS)
    } else {
        Err("range witness contains a value >= 2^64; no configured bucket can prove it".into())
    }
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
    /// LayerNorm on the post-q_proj output (q_raw -> q_n).
    pub q_norm_wit: LayerNormWitness,
    /// LayerNorm on the post-k_proj output (k_raw -> k_n).
    pub k_norm_wit: LayerNormWitness,
    /// attn_wit.q / attn_wit.k now hold q_n / k_n (post-norm, what φ reads).
    pub attn_wit: LinearAttentionWitness,
    pub o_proj_wit: ProjectionWitness,
    /// LayerNorm on the post-o_proj output (out_attn -> normalized).  The
    /// residual `x_mid = x_in + attn_out_norm_wit.y` flows through this LN.
    pub attn_out_norm_wit: LayerNormWitness,
    pub x_mid: Vec<Vec<F>>,
    pub ln2_wit: LayerNormWitness,
    pub ffn_wit: FFNWitness,
    pub x_out: Vec<Vec<F>>,
}

// ---------------------------------------------------------------------------
// Internal Phase 1 data (not part of public proof)
// ---------------------------------------------------------------------------

struct BlockPhase1Data {
    q_norm_y_com: HyraxCommitment,
    k_norm_y_com: HyraxCommitment,
    attn_out_norm_y_com: HyraxCommitment,
    x_norm1_com: HyraxCommitment,
    q_com: HyraxCommitment,
    k_com: HyraxCommitment,
    v_com: HyraxCommitment,
    attn_norm_com: Option<HyraxCommitment>,
    attn_num_com: Option<HyraxCommitment>,
    attn_z_com: Option<HyraxCommitment>,
    attn_rem_com: Option<HyraxCommitment>,
    attn_diff_com: Option<HyraxCommitment>,
    out_attn_com: HyraxCommitment,
    x_norm2_com: HyraxCommitment,
    out_ffn_com: HyraxCommitment,
}

// ---------------------------------------------------------------------------
// Phase 1: commit block intermediates, then prove LNs after global range batch
// ---------------------------------------------------------------------------

struct BlockCommitData {
    x_norm1_com: HyraxCommitment,
    q_com: HyraxCommitment,
    k_com: HyraxCommitment,
    v_com: HyraxCommitment,
    attn_norm_com: Option<HyraxCommitment>,
    attn_num_com: Option<HyraxCommitment>,
    attn_z_com: Option<HyraxCommitment>,
    attn_rem_com: Option<HyraxCommitment>,
    attn_diff_com: Option<HyraxCommitment>,
    out_attn_com: HyraxCommitment,
    q_norm_y_com: HyraxCommitment,
    k_norm_y_com: HyraxCommitment,
    attn_out_norm_y_com: HyraxCommitment,
    x_norm2_com: HyraxCommitment,
    out_ffn_com: HyraxCommitment,
    x_mid_com: HyraxCommitment,
}

/// Per-block commits that depend only on this block's witness — no residual
/// chain dependency. Computed in parallel across blocks; the residual
/// `x_mid_com` is finalized later in a sequential pass.
struct BlockCommitDataNoResidual {
    x_norm1_com: HyraxCommitment,
    q_com: HyraxCommitment,
    k_com: HyraxCommitment,
    v_com: HyraxCommitment,
    attn_norm_com: Option<HyraxCommitment>,
    attn_num_com: Option<HyraxCommitment>,
    attn_z_com: Option<HyraxCommitment>,
    attn_rem_com: Option<HyraxCommitment>,
    attn_diff_com: Option<HyraxCommitment>,
    out_attn_com: HyraxCommitment,
    q_norm_y_com: HyraxCommitment,
    k_norm_y_com: HyraxCommitment,
    attn_out_norm_y_com: HyraxCommitment,
    x_norm2_com: HyraxCommitment,
    out_ffn_com: HyraxCommitment,
}

fn commit_block_intermediates_no_residual(
    witness: &TransformerBlockWitness,
    pk: &TransformerBlockVerifyingKey,
) -> BlockCommitDataNoResidual {
    let t = pk.seq_len;
    let d = pk.d_model;

    let commit_mat = |mat: &[Vec<F>], rows: usize, cols: usize| -> HyraxCommitment {
        let mle = mat_to_mle(mat, rows, cols);
        let vars =
            rows.next_power_of_two().trailing_zeros() + cols.next_power_of_two().trailing_zeros();
        let (nu, _, params) = params_from_vars(vars as usize);
        hyrax_commit(&mle.evaluations, nu, &params)
    };

    // Commit all intermediate matrices.
    // q_com / k_com / v_com commit to the post-projection (pre-norm) values so
    // the QKV cross-block sumcheck (Q = W_q · ln1_y) keeps working.
    // q_norm_y_com / k_norm_y_com commit to the post-norm values (input to φ).
    let x_norm1_com = commit_mat(&witness.ln1_wit.y, t, d);
    let q_com = commit_mat(&witness.q_proj_wit.y, t, d);
    let k_com = commit_mat(&witness.k_proj_wit.y, t, d);
    let v_com = commit_mat(&witness.v_proj_wit.y, t, d);
    let attn_norm_com = witness
        .attn_wit
        .normalized_out
        .as_ref()
        .map(|m| commit_mat(m, t, d));
    let attn_num_com = witness
        .attn_wit
        .normalized_out
        .as_ref()
        .map(|_| commit_mat(&witness.attn_wit.out, t, d));
    let attn_z_com = witness.attn_wit.norm_z.as_ref().map(|v| {
        let mle = vec_to_mle(v, t);
        let vars = t.next_power_of_two().trailing_zeros() as usize;
        let (nu, _, params) = params_from_vars(vars);
        hyrax_commit(&mle.evaluations, nu, &params)
    });
    let attn_rem_com = witness
        .attn_wit
        .norm_rem
        .as_ref()
        .map(|m| commit_mat(m, t, d));
    let attn_diff_com = witness
        .attn_wit
        .norm_diff
        .as_ref()
        .map(|m| commit_mat(m, t, d));
    let out_attn_com = commit_mat(&witness.o_proj_wit.y, t, d);
    // Post-norm outputs (kernel inputs to φ and residual chain input).
    let q_norm_y_com = commit_mat(&witness.q_norm_wit.y, t, d);
    let k_norm_y_com = commit_mat(&witness.k_norm_wit.y, t, d);
    let attn_out_norm_y_com = commit_mat(&witness.attn_out_norm_wit.y, t, d);
    let x_norm2_com = commit_mat(&witness.ln2_wit.y, t, d);
    let out_ffn_com = commit_mat(&witness.ffn_wit.y, t, d);

    BlockCommitDataNoResidual {
        x_norm1_com,
        q_com,
        k_com,
        v_com,
        attn_norm_com,
        attn_num_com,
        attn_z_com,
        attn_rem_com,
        attn_diff_com,
        out_attn_com,
        q_norm_y_com,
        k_norm_y_com,
        attn_out_norm_y_com,
        x_norm2_com,
        out_ffn_com,
    }
}

/// Finalize the residual chain for one block: produce x_mid_com from the
/// block's normalized attention output (post-attn_out_norm) and the running
/// input commitment.  This makes the residual sum match the PyTorch model's
/// `x + attn_out_norm(attn(x))` rather than the un-normalized `x + attn(x)`.
fn finalize_block_commits(
    partial: BlockCommitDataNoResidual,
    x_in_com: &HyraxCommitment,
) -> BlockCommitData {
    let x_mid_com = add_commitments(x_in_com, &partial.attn_out_norm_y_com);
    BlockCommitData {
        x_norm1_com: partial.x_norm1_com,
        q_com: partial.q_com,
        k_com: partial.k_com,
        v_com: partial.v_com,
        attn_norm_com: partial.attn_norm_com,
        attn_num_com: partial.attn_num_com,
        attn_z_com: partial.attn_z_com,
        attn_rem_com: partial.attn_rem_com,
        attn_diff_com: partial.attn_diff_com,
        out_attn_com: partial.out_attn_com,
        q_norm_y_com: partial.q_norm_y_com,
        k_norm_y_com: partial.k_norm_y_com,
        attn_out_norm_y_com: partial.attn_out_norm_y_com,
        x_norm2_com: partial.x_norm2_com,
        out_ffn_com: partial.out_ffn_com,
        x_mid_com,
    }
}

/// Per-block LayerNorm input bundle collected for the model-level batched proof.
#[derive(Clone)]
pub(crate) struct BlockLnInputs {
    pub ln1_io: LayerNormIOCommitments,
    pub q_norm_io: LayerNormIOCommitments,
    pub k_norm_io: LayerNormIOCommitments,
    pub aon_io: LayerNormIOCommitments,
    pub ln2_io: LayerNormIOCommitments,
    /// Per-LN range artefacts in fixed order:
    /// [ln1_sig, ln1_y, ln2_sig, ln2_y, q_norm_sig, q_norm_y,
    ///  k_norm_sig, k_norm_y, aon_sig, aon_y].
    pub ranges: [(RangeWitnessProof, Vec<F>); 10],
}

fn prove_block_layernorms(
    _witness: &TransformerBlockWitness,
    x_in_com: &HyraxCommitment,
    _pk: &TransformerBlockVerifyingKey,
    commits: &BlockCommitData,
    range_proofs: [RangeWitnessProof; 10],
    range_rvs: [Vec<F>; 10],
    transcript: &mut Transcript,
) -> Result<(BlockPhase1Data, BlockLnInputs), String> {
    // PROOF_VERSION 13: LN sumchecks/openings are deferred to a single
    // `prove_layernorms_batched` call at model end. Here we (a) preserve the
    // transcript absorption pattern so subsequent sub-proofs still bind to
    // each LN's IO commitments, and (b) collect per-LN inputs for the batched
    // call.

    // LN1: x = x_in_com, y = x_norm1_com.
    absorb_com(transcript, b"x_com", x_in_com);
    absorb_com(transcript, b"y_com", &commits.x_norm1_com);
    let ln1_io = LayerNormIOCommitments {
        x_com: x_in_com.clone(),
        y_com: Some(commits.x_norm1_com.clone()),
    };

    // Explicitly absorb q/k/v_com with the same labels attention uses.
    absorb_com(transcript, b"q_com", &commits.q_com);
    absorb_com(transcript, b"k_com", &commits.k_com);
    absorb_com(transcript, b"v_com", &commits.v_com);
    if let Some(ref c) = commits.attn_norm_com {
        absorb_com(transcript, b"attn_norm_com", c);
    }
    if let Some(ref c) = commits.attn_num_com {
        absorb_com(transcript, b"attn_num_com", c);
    }
    if let Some(ref c) = commits.attn_z_com {
        absorb_com(transcript, b"attn_z_com", c);
    }
    if let Some(ref c) = commits.attn_rem_com {
        absorb_com(transcript, b"attn_rem_com", c);
    }
    if let Some(ref c) = commits.attn_diff_com {
        absorb_com(transcript, b"attn_diff_com", c);
    }
    absorb_com(transcript, b"out_attn_com", &commits.out_attn_com);

    // q_norm: x = q_raw (q_com), y = q_n (q_norm_y_com).
    absorb_com(transcript, b"x_com", &commits.q_com);
    absorb_com(transcript, b"y_com", &commits.q_norm_y_com);
    let q_norm_io = LayerNormIOCommitments {
        x_com: commits.q_com.clone(),
        y_com: Some(commits.q_norm_y_com.clone()),
    };
    absorb_com(transcript, b"q_norm_y_com", &commits.q_norm_y_com);

    // k_norm: x = k_raw (k_com), y = k_n (k_norm_y_com).
    absorb_com(transcript, b"x_com", &commits.k_com);
    absorb_com(transcript, b"y_com", &commits.k_norm_y_com);
    let k_norm_io = LayerNormIOCommitments {
        x_com: commits.k_com.clone(),
        y_com: Some(commits.k_norm_y_com.clone()),
    };
    absorb_com(transcript, b"k_norm_y_com", &commits.k_norm_y_com);

    // attn_out_norm: x = out_attn, y = post-norm output that feeds the residual.
    absorb_com(transcript, b"x_com", &commits.out_attn_com);
    absorb_com(transcript, b"y_com", &commits.attn_out_norm_y_com);
    let aon_io = LayerNormIOCommitments {
        x_com: commits.out_attn_com.clone(),
        y_com: Some(commits.attn_out_norm_y_com.clone()),
    };
    absorb_com(
        transcript,
        b"attn_out_norm_y_com",
        &commits.attn_out_norm_y_com,
    );

    // LN2: x = x_mid_com, y = x_norm2_com.
    absorb_com(transcript, b"x_com", &commits.x_mid_com);
    absorb_com(transcript, b"y_com", &commits.x_norm2_com);
    let ln2_io = LayerNormIOCommitments {
        x_com: commits.x_mid_com.clone(),
        y_com: Some(commits.x_norm2_com.clone()),
    };

    // Absorb out_ffn_com so transcript stays consistent with Phase 1.
    absorb_com(transcript, b"y_com", &commits.out_ffn_com);

    let [ln1_sig_rp, ln1_y_rp, ln2_sig_rp, ln2_y_rp, q_norm_sig_rp, q_norm_y_rp, k_norm_sig_rp, k_norm_y_rp, aon_sig_rp, aon_y_rp] =
        range_proofs;
    let [ln1_sig_rv, ln1_y_rv, ln2_sig_rv, ln2_y_rv, q_norm_sig_rv, q_norm_y_rv, k_norm_sig_rv, k_norm_y_rv, aon_sig_rv, aon_y_rv] =
        range_rvs;
    let ln_inputs = BlockLnInputs {
        ln1_io: ln1_io.clone(),
        q_norm_io: q_norm_io.clone(),
        k_norm_io: k_norm_io.clone(),
        aon_io: aon_io.clone(),
        ln2_io: ln2_io.clone(),
        ranges: [
            (ln1_sig_rp, ln1_sig_rv),
            (ln1_y_rp, ln1_y_rv),
            (ln2_sig_rp, ln2_sig_rv),
            (ln2_y_rp, ln2_y_rv),
            (q_norm_sig_rp, q_norm_sig_rv),
            (q_norm_y_rp, q_norm_y_rv),
            (k_norm_sig_rp, k_norm_sig_rv),
            (k_norm_y_rp, k_norm_y_rv),
            (aon_sig_rp, aon_sig_rv),
            (aon_y_rp, aon_y_rv),
        ],
    };

    let phase1 = BlockPhase1Data {
        q_norm_y_com: commits.q_norm_y_com.clone(),
        k_norm_y_com: commits.k_norm_y_com.clone(),
        attn_out_norm_y_com: commits.attn_out_norm_y_com.clone(),
        x_norm1_com: commits.x_norm1_com.clone(),
        q_com: commits.q_com.clone(),
        k_com: commits.k_com.clone(),
        v_com: commits.v_com.clone(),
        attn_norm_com: commits.attn_norm_com.clone(),
        attn_num_com: commits.attn_num_com.clone(),
        attn_z_com: commits.attn_z_com.clone(),
        attn_rem_com: commits.attn_rem_com.clone(),
        attn_diff_com: commits.attn_diff_com.clone(),
        out_attn_com: commits.out_attn_com.clone(),
        x_norm2_com: commits.x_norm2_com.clone(),
        out_ffn_com: commits.out_ffn_com.clone(),
    };
    Ok((phase1, ln_inputs))
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
    pub causal: bool,
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
    /// Cross-LN GKR-style batched proof for every LayerNorm in the model
    /// (5 per block: ln1, q_norm, k_norm, attn_out_norm, ln2; plus final_ln).
    /// Replaces the per-block + per-final LN proofs of PROOF_VERSION 12.
    pub ln_batched_proof: LayerNormsBatchedProof,
    pub lm_head_proof: ProjectionProof,
    pub final_ln_out_com: HyraxCommitment,
    pub logits_com: HyraxCommitment,
    pub lm_head_logits_open: HyraxProof,
    pub ffn_lasso_proof: LassoMultiProof,
    pub all_lasso_proof: LassoMultiProof,
    pub ffn_quant_proof: QuantizationProof,
    pub qk_quant_proof: QuantizationProof,
    /// Global range multiplicity proofs, one per bit-width bucket used by the
    /// model-level LayerNorm/attention-normalization range proofs.
    pub ln_range_ms: Vec<RangeBatchM>,

    // Cross-block batch sumchecks (one per projection type + attention)
    pub batch_qkv: SumcheckProofMulti,
    pub batch_oproj: SumcheckProofMulti,
    pub batch_ffn_y: SumcheckProofMulti,
    pub batch_ffn_m: SumcheckProofMulti,
    /// Non-causal: degree-2 sumcheck over k axis.
    /// Causal: None (replaced by batch_attn_out_causal).
    pub batch_attn_out: Option<SumcheckProofMulti>,
    /// Causal: degree-3 sumcheck over (t, k) with eq(r_t, ·) folded in as the
    /// third multiplicand so the claim equals the actual attn_out MLE eval.
    pub batch_attn_out_causal: Option<SumcheckCubicProofMulti>,
    /// Non-causal: degree-2 sumcheck over t axis.
    /// Causal: None (replaced by batch_attn_ctx_causal).
    pub batch_attn_ctx: Option<SumcheckProofMulti>,
    /// Causal: degree-3 sumcheck over (s, a, b) with the suffix·eq_a·eq_b
    /// factor folded into the third multiplicand so f and g remain plain phi_k
    /// and v MLEs respectively.
    pub batch_attn_ctx_causal: Option<SumcheckCubicProofMulti>,
    pub attn_norm_sumcheck: Option<SumcheckCubicProofMulti>,
    pub attn_z_sumcheck: Option<SumcheckProofMulti>,
    pub attn_z_ksum_sumcheck: Option<SumcheckProofMulti>,
    pub attn_z_causal_sumcheck: Option<SumcheckCubicProofMulti>,
    /// Attention normalization rem/diff range proofs.  Their bit-width metadata
    /// is stored separately so the verifier can replay the same range buckets.
    pub attn_norm_rem_range_proofs: Vec<RangeWitnessProof>,
    pub attn_norm_diff_range_proofs: Vec<RangeWitnessProof>,
    pub attn_norm_rem_range_bits: Vec<usize>,
    pub attn_norm_diff_range_bits: Vec<usize>,

    // Global batch open for 5L intermediate matrices at shared r_td
    pub inter_batch_open: HyraxProof,

    // Cross-block batch opens.  PROOF_VERSION 16 (P2): wq/wk/wv share point
    // `combine(r_k_qkv, r_out)` and bq/bk/bv/bo share point `r_out`, so they
    // are folded into one merged `hyrax_open_batch` per shared point.
    pub x_norm1_batch_open: HyraxProof,
    pub qkv_w_batch_open: HyraxProof,
    pub w_o_batch_open: HyraxProof,
    pub qkvo_bias_batch_open: HyraxProof,
    pub w2_batch_open: HyraxProof,
    pub ffn_a_batch_open: HyraxProof,
    pub w1_batch_open: HyraxProof,
    pub x_norm2_batch_open: HyraxProof,
    pub ffn_m_com_batch_open: HyraxProof,
    pub ffn_lasso_bind_open: HyraxProof,

    // Cross-block batch opens for attention phi_q, phi_k, v (shared eval points)
    pub phi_q_batch_open: HyraxProof,
    pub phi_k_batch_open: HyraxProof,
    pub v_attn_batch_open: HyraxProof,
    /// Merged open at r_norm covering, in order: num_coms, norm_coms, rem_coms, diff_coms (4L commits total).
    pub attn_norm_r_batch_open: Option<HyraxProof>,
    /// Merged open at attn_point=combine(r_t,r_k_o) covering num_coms then norm_coms (2L commits).
    pub attn_norm_attn_point_open: Option<HyraxProof>,
    pub attn_z_open: Option<HyraxProof>,
    pub attn_z_phi_q_open: Option<HyraxProof>,
    pub attn_z_phi_k_open: Option<HyraxProof>,
    pub causal_ctx_prefix_evals: Vec<F>,
    pub causal_phi_k_prefix_evals: Vec<F>,
    pub causal_v_prefix_evals: Vec<F>,
    pub causal_ctx_out_batch_open: Option<HyraxProof>,
    pub causal_ctx_prefix_batch_open: Option<HyraxProof>,
    pub qk_lasso_bind_open: HyraxProof,
}

#[derive(Clone)]
pub struct RangeBatchM {
    pub bits: usize,
    pub m: GlobalRangeM,
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

fn eq_evals_msb(r: &[F], n: usize) -> Vec<F> {
    let r_rev: Vec<F> = r.iter().rev().copied().collect();
    crate::poly::utils::compute_eq_evals(&r_rev, n)
}

fn suffix_eq_evals_msb(r: &[F], len: usize) -> Vec<F> {
    let padded = len.next_power_of_two();
    let eq = eq_evals_msb(r, len);
    let mut suffix = vec![F::ZERO; padded];
    let mut running = F::ZERO;
    for i in (0..len).rev() {
        running += eq[i];
        suffix[i] = running;
    }
    suffix
}

/// Build (f, g, h) polynomials for the causal prefix-context cubic sumcheck:
///   ctx_eval = Σ_{s,a,b} suffix(s) · eq_a(a) · eq_b(b) · phi_k[s][a] · v[s][b]
/// with
///   f[s,a,b] = phi_k[s][a]  (constant in b)
///   g[s,a,b] = v[s][b]      (constant in a)
///   h[s,a,b] = suffix(s) · eq_a(a) · eq_b(b)  (verifier-computable factor)
/// so that f_mle, g_mle, h_mle factor cleanly without MLE-of-pointwise-product
/// pitfalls. (When d is a power of two, the constant-axis MLE summand is 1.)
fn causal_prefix_polys(
    phi_k: &[Vec<F>],
    v: &[Vec<F>],
    suffix_evals: &[F],
    eq_a_evals: &[F],
    eq_b_evals: &[F],
    t: usize,
    d: usize,
) -> (DenseMLPoly, DenseMLPoly, DenseMLPoly) {
    let t_p2 = t.next_power_of_two();
    let d_p2 = d.next_power_of_two();
    let total = t_p2 * d_p2 * d_p2;
    let mut f = vec![F::ZERO; total];
    let mut g = vec![F::ZERO; total];
    let mut h = vec![F::ZERO; total];
    // h is multilinear over (s, a, b) and verifier-reproducible at any point;
    // populate every cell, including indices outside [0, t)×[0, d)×[0, d), so
    // that the MLE factors as suffix_mle(X_s)·eq_a_mle(X_a)·eq_b_mle(X_b).
    for s in 0..t_p2 {
        for a in 0..d_p2 {
            for b in 0..d_p2 {
                let idx = (s * d_p2 + a) * d_p2 + b;
                h[idx] = suffix_evals[s] * eq_a_evals[a] * eq_b_evals[b];
            }
        }
    }
    for s in 0..t {
        for a in 0..d {
            let f_val = phi_k[s][a];
            for b in 0..d {
                let idx = (s * d_p2 + a) * d_p2 + b;
                f[idx] = f_val;
                g[idx] = v[s][b];
            }
        }
    }
    (
        DenseMLPoly::new(f),
        DenseMLPoly::new(g),
        DenseMLPoly::new(h),
    )
}

fn causal_denominator_polys(
    phi_q: &[Vec<F>],
    phi_k: &[Vec<F>],
    eq_t_evals: &[F],
    t: usize,
    d: usize,
) -> (DenseMLPoly, DenseMLPoly, DenseMLPoly) {
    let t_p2 = t.next_power_of_two();
    let d_p2 = d.next_power_of_two();
    let total = t_p2 * t_p2 * d_p2;
    let mut f = vec![F::ZERO; total];
    let mut g = vec![F::ZERO; total];
    let mut h = vec![F::ZERO; total];
    for i in 0..t {
        for s in 0..=i {
            for a in 0..d {
                let idx = (i * t_p2 + s) * d_p2 + a;
                f[idx] = eq_t_evals[i];
                g[idx] = phi_q[i][a];
                h[idx] = phi_k[s][a];
            }
        }
    }
    (
        DenseMLPoly::new(f),
        DenseMLPoly::new(g),
        DenseMLPoly::new(h),
    )
}

fn build_causal_context(phi_k: &[Vec<F>], v: &[Vec<F>], t: usize, d: usize) -> Vec<Vec<F>> {
    let mut prefix = vec![vec![F::ZERO; d]; d];
    let mut context = vec![vec![F::ZERO; d]; t * d];
    for s in 0..t {
        for a in 0..d {
            let k_sa = phi_k[s][a];
            for b in 0..d {
                prefix[a][b] += k_sa * v[s][b];
                context[s * d + a][b] = prefix[a][b];
            }
        }
    }
    context
}

fn validate_causal_context_shape(context: &[Vec<F>], t: usize, d: usize) -> Result<(), String> {
    if context.len() != t * d {
        return Err(format!(
            "causal_context row count mismatch: got {}, expected {}",
            context.len(),
            t * d
        ));
    }
    if context.iter().any(|row| row.len() != d) {
        return Err(format!(
            "causal_context column count mismatch: expected {} columns",
            d
        ));
    }
    Ok(())
}

fn empty_lasso_proof() -> LassoProof {
    LassoProof {
        outputs: Vec::new(),
        query_indices: Vec::new(),
        sub_claims: Vec::new(),
        sumcheck_proofs: Vec::new(),
        table_openings: Vec::new(),
        hyrax_proofs: Vec::new(),
        output_sumcheck: None,
        output_open: None,
        l_k_evals: Vec::new(),
        index_proof: None,
    }
}

fn absorb_index_vectors(transcript: &mut Transcript, label: &[u8], vectors: &[&[usize]]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(vectors.len() as u64).to_le_bytes());
    for v in vectors {
        bytes.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for &idx in *v {
            bytes.extend_from_slice(&(idx as u64).to_le_bytes());
        }
    }
    transcript.append_bytes(label, &bytes);
}

fn flatten_mat_values(mat: &[Vec<F>]) -> Vec<F> {
    mat.iter().flat_map(|row| row.iter().copied()).collect()
}

fn flatten_mat_indices(mat: &[Vec<F>]) -> Vec<usize> {
    mat.iter()
        .flat_map(|row| row.iter().map(|x| x.into_bigint().as_ref()[0] as usize))
        .collect()
}

fn indices_to_mle_evals(indices: &[usize]) -> Vec<F> {
    DenseMLPoly::from_vec_padded(indices.iter().map(|&idx| F::from(idx as u64)).collect())
        .evaluations
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
    let prove_t0 = Instant::now();
    let mut phase_t0 = Instant::now();
    let num_blocks = pk.vk.num_blocks;
    let t = pk.vk.seq_len;
    let d = pk.vk.d_model;
    let v = pk.vk.vocab_size;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;
    let d_ff = pk.block_pks[0].ffn_pk.vk.d_ff;
    if pk.vk.causal != inst_attn.causal {
        return Err(format!(
            "attention mode mismatch: key causal={}, instance causal={}",
            pk.vk.causal, inst_attn.causal
        ));
    }
    let f_bits = d_ff.next_power_of_two().trailing_zeros() as usize;
    let (nu_td, sigma_td, _) = params_from_vars(td_num_vars);
    let (nu_w, sigma_w, _) = params_from_vars(d_bits + d_bits);
    let (nu_b, sigma_b, _) = params_from_vars(d_bits);
    let (nu_wff, sigma_wff, _) = params_from_vars(f_bits + d_bits);
    let (nu_mff, sigma_mff, _) = params_from_vars(t_bits + f_bits);

    let commit_mat = |mat: &[Vec<F>], rows: usize, cols: usize| -> HyraxCommitment {
        let mle = mat_to_mle(mat, rows, cols);
        let vars =
            rows.next_power_of_two().trailing_zeros() + cols.next_power_of_two().trailing_zeros();
        let (nu, _, params) = params_from_vars(vars as usize);
        hyrax_commit(&mle.evaluations, nu, &params)
    };

    struct StaticBlockMles {
        w_q: DenseMLPoly,
        w_k: DenseMLPoly,
        w_v: DenseMLPoly,
        w_o: DenseMLPoly,
        w1: DenseMLPoly,
        w2: DenseMLPoly,
        b_q: DenseMLPoly,
        b_k: DenseMLPoly,
        b_v: DenseMLPoly,
        b_o: DenseMLPoly,
    }

    let static_mles: Vec<StaticBlockMles> = (0..num_blocks)
        .into_par_iter()
        .map(|i| {
            let bpk = &pk.block_pks[i];
            StaticBlockMles {
                w_q: mat_to_mle(&convert_tm_to_fm(&bpk.q_pk.w), d, d),
                w_k: mat_to_mle(&convert_tm_to_fm(&bpk.k_pk.w), d, d),
                w_v: mat_to_mle(&convert_tm_to_fm(&bpk.v_pk.w), d, d),
                w_o: mat_to_mle(&convert_tm_to_fm(&bpk.o_pk.w), d, d),
                w1: mat_to_mle(&convert_tm_to_fm(&bpk.ffn_pk.w1), d, d_ff),
                w2: mat_to_mle(&convert_tm_to_fm(&bpk.ffn_pk.w2), d_ff, d),
                b_q: vec_to_mle(&bpk.q_pk.bias, d),
                b_k: vec_to_mle(&bpk.k_pk.bias, d),
                b_v: vec_to_mle(&bpk.v_pk.bias, d),
                b_o: vec_to_mle(&bpk.o_pk.bias, d),
            }
        })
        .collect();
    eprintln!(
        "[prove] static_mles: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();

    // =========================================================================
    // 1. Initial input commitment
    // =========================================================================
    let x_in_com = commit_mat(&witness.x_in, t, d);
    absorb_com(transcript, b"x_in_com", &x_in_com);

    // =========================================================================
    // 2. Phase 1: commit all block intermediates, prove one global LN range batch,
    // then run the per-LN algebraic proofs against those range challenges.
    // =========================================================================
    let mut block_commits: Vec<BlockCommitData> = Vec::with_capacity(num_blocks);
    let mut block_input_coms: Vec<HyraxCommitment> = Vec::with_capacity(num_blocks);
    let mut ln_range_witnesses: Vec<RangeProofWitness> = Vec::with_capacity(10 * num_blocks + 2);
    let mut phase1_data: Vec<BlockPhase1Data> = Vec::with_capacity(num_blocks);
    let mut block_ln_inputs: Vec<BlockLnInputs> = Vec::with_capacity(num_blocks);
    let mut current_x_com = x_in_com.clone();

    // Phase 1a (PARALLEL): Per-block witness commits and LN range witnesses are
    // independent across blocks — neither depends on the residual chain or the
    // transcript. Compute them concurrently across L blocks.
    let parallel_per_block: Vec<(
        BlockCommitDataNoResidual,
        LayerNormRangeWitnesses,
        LayerNormRangeWitnesses,
        LayerNormRangeWitnesses,
        LayerNormRangeWitnesses,
        LayerNormRangeWitnesses,
    )> = (0..num_blocks)
        .into_par_iter()
        .map(|i| {
            let bw = &witness.block_witnesses[i];
            let bpk = &pk.block_pks[i];
            let partial = commit_block_intermediates_no_residual(bw, bpk);
            let ln1_rw = compute_range_witnesses(&bw.ln1_wit, &bpk.ln1_vk);
            let ln2_rw = compute_range_witnesses(&bw.ln2_wit, &bpk.ln2_vk);
            let q_norm_rw = compute_range_witnesses(&bw.q_norm_wit, &bpk.q_norm_vk);
            let k_norm_rw = compute_range_witnesses(&bw.k_norm_wit, &bpk.k_norm_vk);
            let aon_rw = compute_range_witnesses(&bw.attn_out_norm_wit, &bpk.attn_out_norm_vk);
            (partial, ln1_rw, ln2_rw, q_norm_rw, k_norm_rw, aon_rw)
        })
        .collect();

    // Phase 1b (SEQUENTIAL): finalize the residual-chain x_mid_com per block
    // (homomorphic adds, cheap) and accumulate ln_range_witnesses in deterministic order.
    // Per block: 10 sub-witnesses [ln1_sigma, ln1_y, ln2_sigma, ln2_y,
    //                              q_norm_sigma, q_norm_y, k_norm_sigma, k_norm_y,
    //                              attn_out_norm_sigma, attn_out_norm_y].
    for (partial, ln1_rw, ln2_rw, q_norm_rw, k_norm_rw, aon_rw) in parallel_per_block {
        block_input_coms.push(current_x_com.clone());
        let commits = finalize_block_commits(partial, &current_x_com);
        ln_range_witnesses.push(ln1_rw.sigma_witness);
        ln_range_witnesses.push(ln1_rw.y_witness);
        ln_range_witnesses.push(ln2_rw.sigma_witness);
        ln_range_witnesses.push(ln2_rw.y_witness);
        ln_range_witnesses.push(q_norm_rw.sigma_witness);
        ln_range_witnesses.push(q_norm_rw.y_witness);
        ln_range_witnesses.push(k_norm_rw.sigma_witness);
        ln_range_witnesses.push(k_norm_rw.y_witness);
        ln_range_witnesses.push(aon_rw.sigma_witness);
        ln_range_witnesses.push(aon_rw.y_witness);
        let next_x_com = add_commitments(&commits.x_mid_com, &commits.out_ffn_com);
        current_x_com = next_x_com;
        block_commits.push(commits);
    }

    let final_rw = compute_range_witnesses(&witness.final_ln_wit, &pk.vk.final_ln_vk);
    ln_range_witnesses.push(final_rw.sigma_witness);
    ln_range_witnesses.push(final_rw.y_witness);

    // Range witnesses are routed to the narrowest configured bit-width bucket.
    // The verifier replays this from serialized metadata and verifies one
    // GlobalRangeM per used bucket.
    let _: usize = ATTN_NORM_RANGE_BITS;

    let mut norm_range_witnesses: Vec<RangeProofWitness> = Vec::new();
    let has_attn_norm = witness
        .block_witnesses
        .iter()
        .any(|b| b.attn_wit.normalized_out.is_some());
    if has_attn_norm {
        for (i, b) in witness.block_witnesses.iter().enumerate() {
            let rem =
                b.attn_wit.norm_rem.as_ref().ok_or_else(|| {
                    format!("block {i}: missing attention normalization remainder")
                })?;
            let diff = b
                .attn_wit
                .norm_diff
                .as_ref()
                .ok_or_else(|| format!("block {i}: missing attention normalization diff"))?;
            norm_range_witnesses.push(RangeProofWitness {
                values: flatten_mat_values(rem),
            });
            norm_range_witnesses.push(RangeProofWitness {
                values: flatten_mat_values(diff),
            });
        }
    }

    // Single unified witness list: 10·L LN sub-witnesses + 2 final_ln + (when
    // has_attn_norm) 2·L attn-norm rem/diff witnesses.
    let mut all_range_refs: Vec<&RangeProofWitness> =
        Vec::with_capacity(ln_range_witnesses.len() + norm_range_witnesses.len());
    for w in &ln_range_witnesses {
        all_range_refs.push(w);
    }
    for w in &norm_range_witnesses {
        all_range_refs.push(w);
    }

    let all_range_bits: Vec<usize> = all_range_refs
        .iter()
        .map(|w| choose_range_bits(w))
        .collect::<Result<_, _>>()?;

    let mut all_range_proofs_by_idx: Vec<Option<RangeWitnessProof>> =
        vec![None; all_range_refs.len()];
    let mut all_range_rvs_by_idx: Vec<Option<Vec<F>>> = vec![None; all_range_refs.len()];
    let mut ln_range_ms = Vec::new();

    for bits in [FAST_RANGE_BITS, WIDE_RANGE_BITS] {
        let bucket: Vec<(usize, &RangeProofWitness)> = all_range_refs
            .iter()
            .enumerate()
            .filter_map(|(idx, &w)| (all_range_bits[idx] == bits).then_some((idx, w)))
            .collect();
        if bucket.is_empty() {
            continue;
        }
        let bucket_refs: Vec<&RangeProofWitness> = bucket.iter().map(|(_, w)| *w).collect();
        let (bucket_proofs, bucket_m, bucket_rvs) =
            prove_range_batched(&bucket_refs, bits, transcript)?;
        ln_range_ms.push(RangeBatchM { bits, m: bucket_m });
        for (((idx, _), proof), rv) in bucket
            .into_iter()
            .zip(bucket_proofs.into_iter())
            .zip(bucket_rvs.into_iter())
        {
            all_range_proofs_by_idx[idx] = Some(proof);
            all_range_rvs_by_idx[idx] = Some(rv);
        }
    }

    let all_range_proofs: Vec<RangeWitnessProof> = all_range_proofs_by_idx
        .into_iter()
        .map(|p| p.expect("missing bucketed range proof"))
        .collect();
    let all_range_rvs: Vec<Vec<F>> = all_range_rvs_by_idx
        .into_iter()
        .map(|rv| rv.expect("missing bucketed range rv"))
        .collect();

    // Split back into LN proofs (first 10·L+2) and attn-norm proofs (next 2·L).
    let ln_count = ln_range_witnesses.len();
    let mut all_proofs_iter = all_range_proofs.into_iter();
    let mut all_rvs_iter = all_range_rvs.into_iter();

    let mut ln_range_proofs = (0..ln_count)
        .map(|_| all_proofs_iter.next().expect("missing LN range proof"))
        .collect::<Vec<_>>()
        .into_iter();
    let mut ln_range_rvs = (0..ln_count)
        .map(|_| all_rvs_iter.next().expect("missing LN range rv"))
        .collect::<Vec<_>>()
        .into_iter();

    let ln_range_bits_vec = all_range_bits[..ln_count].to_vec();
    let mut all_bits_iter = all_range_bits.into_iter().skip(ln_count);
    let (
        attn_norm_rem_range_proofs,
        attn_norm_diff_range_proofs,
        attn_norm_rem_range_bits,
        attn_norm_diff_range_bits,
    ) = if has_attn_norm {
        let mut rem = Vec::with_capacity(num_blocks);
        let mut diff = Vec::with_capacity(num_blocks);
        let mut rem_bits = Vec::with_capacity(num_blocks);
        let mut diff_bits = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            rem.push(
                all_proofs_iter
                    .next()
                    .expect("missing attn rem range proof"),
            );
            rem_bits.push(all_bits_iter.next().expect("missing attn rem range bits"));
            // discard attn-norm rvs — they aren't consumed by downstream sub-proofs
            let _ = all_rvs_iter.next().expect("missing attn rem range rv");
            diff.push(
                all_proofs_iter
                    .next()
                    .expect("missing attn diff range proof"),
            );
            diff_bits.push(all_bits_iter.next().expect("missing attn diff range bits"));
            let _ = all_rvs_iter.next().expect("missing attn diff range rv");
        }
        (rem, diff, rem_bits, diff_bits)
    } else {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };

    for i in 0..num_blocks {
        let range_proofs = [
            ln_range_proofs
                .next()
                .expect("missing ln1 sigma range proof"),
            ln_range_proofs.next().expect("missing ln1 y range proof"),
            ln_range_proofs
                .next()
                .expect("missing ln2 sigma range proof"),
            ln_range_proofs.next().expect("missing ln2 y range proof"),
            ln_range_proofs
                .next()
                .expect("missing q_norm sigma range proof"),
            ln_range_proofs
                .next()
                .expect("missing q_norm y range proof"),
            ln_range_proofs
                .next()
                .expect("missing k_norm sigma range proof"),
            ln_range_proofs
                .next()
                .expect("missing k_norm y range proof"),
            ln_range_proofs
                .next()
                .expect("missing attn_out_norm sigma range proof"),
            ln_range_proofs
                .next()
                .expect("missing attn_out_norm y range proof"),
        ];
        let range_rvs = [
            ln_range_rvs.next().expect("missing ln1 sigma range point"),
            ln_range_rvs.next().expect("missing ln1 y range point"),
            ln_range_rvs.next().expect("missing ln2 sigma range point"),
            ln_range_rvs.next().expect("missing ln2 y range point"),
            ln_range_rvs
                .next()
                .expect("missing q_norm sigma range point"),
            ln_range_rvs.next().expect("missing q_norm y range point"),
            ln_range_rvs
                .next()
                .expect("missing k_norm sigma range point"),
            ln_range_rvs.next().expect("missing k_norm y range point"),
            ln_range_rvs
                .next()
                .expect("missing attn_out_norm sigma range point"),
            ln_range_rvs
                .next()
                .expect("missing attn_out_norm y range point"),
        ];
        let (p1, ln_inputs) = prove_block_layernorms(
            &witness.block_witnesses[i],
            &block_input_coms[i],
            &pk.block_pks[i],
            &block_commits[i],
            range_proofs,
            range_rvs,
            transcript,
        )?;
        phase1_data.push(p1);
        block_ln_inputs.push(ln_inputs);
    }
    let final_sig_rp = ln_range_proofs
        .next()
        .expect("missing final LN sigma range proof");
    let final_y_rp = ln_range_proofs
        .next()
        .expect("missing final LN y range proof");
    let final_sig_rv = ln_range_rvs
        .next()
        .expect("missing final LN sigma range point");
    let final_y_rv = ln_range_rvs.next().expect("missing final LN y range point");
    eprintln!(
        "[prove] phase1_ln_range: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();

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
    let mut q_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut k_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut v_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut x_norm1_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    // Per-block QKV data needed for proof construction
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

        // Evaluate outputs and biases at challenge points.
        // q/k/v are the post-projection (pre-norm) matrices; the QKV sumcheck
        // binds them to W_q · ln1_y / W_k · ln1_y / W_v · ln1_y.
        let x_mle = mat_to_mle(&bw.ln1_wit.y, t, d);
        let q_mle = mat_to_mle(&bw.q_proj_wit.y, t, d);
        let k_mle = mat_to_mle(&bw.k_proj_wit.y, t, d);
        let v_mle = mat_to_mle(&bw.v_proj_wit.y, t, d);
        let static_mle = &static_mles[i];

        let q_eval = q_mle.evaluate(&combine(&r_t, &r_out));
        let k_eval = k_mle.evaluate(&combine(&r_t, &r_out));
        let v_eval = v_mle.evaluate(&combine(&r_t, &r_out));
        let bias_q_eval = static_mle.b_q.evaluate(&r_out);
        let bias_k_eval = static_mle.b_k.evaluate(&r_out);
        let bias_v_eval = static_mle.b_v.evaluate(&r_out);

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
        pb_q_eval.push(q_eval);
        pb_k_eval.push(k_eval);
        pb_v_eval.push(v_eval);
        pb_bias_q_eval.push(bias_q_eval);
        pb_bias_k_eval.push(bias_k_eval);
        pb_bias_v_eval.push(bias_v_eval);
        q_mles.push(q_mle);
        k_mles.push(k_mle);
        v_mles.push(v_mle);
        x_norm1_mles.push(x_mle);
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
        prove_sumcheck_multi_batched_owned(fs_qkv, gs_qkv, &weights_qkv, claim_qkv, transcript);

    // Per-block weight evals at shared (r_k_qkv, r_out)
    let mut pb_w_q_eval: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_w_k_eval: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_w_v_eval: Vec<F> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let point = combine(&r_k_qkv, &r_out);
        pb_w_q_eval.push(static_mles[i].w_q.evaluate(&point));
        pb_w_k_eval.push(static_mles[i].w_k.evaluate(&point));
        pb_w_v_eval.push(static_mles[i].w_v.evaluate(&point));
    }
    eprintln!(
        "[prove] batch_qkv: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();

    // =========================================================================
    // 5. Batch O-proj: absorb per-block coms, build polys
    // =========================================================================
    let mut fs_oproj: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_oproj: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut oproj_targets: Vec<F> = Vec::with_capacity(num_blocks);
    let mut out_attn_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
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
        let out_attn_mle = mat_to_mle(&bw.o_proj_wit.y, t, d);
        let static_mle = &static_mles[i];

        let bias_o_eval = static_mle.b_o.evaluate(&r_out);
        let out_attn_eval = out_attn_mle.evaluate(&combine(&r_t, &r_out));

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
        out_attn_mles.push(out_attn_mle);
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
    let (batch_oproj, r_k_o) = prove_sumcheck_multi_batched_owned(
        fs_oproj,
        gs_oproj,
        &weights_oproj,
        claim_oproj,
        transcript,
    );

    // Per-block Wo evals at shared (r_k_o, r_out)
    let mut pb_w_o_eval: Vec<F> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        pb_w_o_eval.push(static_mles[i].w_o.evaluate(&combine(&r_k_o, &r_out)));
    }
    eprintln!(
        "[prove] batch_oproj: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();

    // =========================================================================
    // 6. Cross-block Attention batch sumchecks
    //    6a: Commit phi_q/phi_k per block, absorb. Collect out_evals.
    //    6b: Cross-block batch out sumcheck → shared batch_r_attn_out.
    //    6c: Cross-block batch ctx sumcheck → shared batch_r_attn_ctx.
    // =========================================================================

    // 6a. Commit phi_q, phi_k per block and absorb into transcript.
    let mut phi_q_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut phi_k_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut ctx_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut attn_phi_q_coms: Vec<HyraxCommitment> = Vec::with_capacity(num_blocks);
    let mut attn_phi_k_coms: Vec<HyraxCommitment> = Vec::with_capacity(num_blocks);
    let mut causal_context_coms: Vec<Option<HyraxCommitment>> = Vec::with_capacity(num_blocks);
    let mut attn_out_evals: Vec<F> = Vec::with_capacity(num_blocks);

    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let bw = &witness.block_witnesses[i];

        let phi_q_mle = mat_to_mle(&bw.attn_wit.phi_q, t, d);
        let phi_k_mle = mat_to_mle(&bw.attn_wit.phi_k, t, d);
        // GKR chain: in causal mode, causal_context is no longer committed; the
        // CTX cubic sumcheck binds it algebraically via phi_k and v. In
        // non-causal mode we still need the per-block ctx_mle for the OUT
        // sumcheck's degree-2 dot product over k.
        let ctx_mle = if inst_attn.causal {
            // Validate the witness shape if provided, but otherwise we don't
            // need a committed MLE — the CTX cubic sumcheck rebuilds the
            // (s, a, b) tensor from phi_k and v directly.
            if let Some(context) = bw.attn_wit.causal_context.as_ref() {
                validate_causal_context_shape(context, t, d)?;
            }
            DenseMLPoly::new(vec![F::ZERO; 1])
        } else {
            mat_to_mle(&bw.attn_wit.context, d, d)
        };
        let causal_context_com: Option<HyraxCommitment> = None;

        let phi_q_com = commit_mat(&bw.attn_wit.phi_q, t, d);
        let phi_k_com = commit_mat(&bw.attn_wit.phi_k, t, d);
        absorb_com(transcript, b"phi_q_com", &phi_q_com);
        absorb_com(transcript, b"phi_k_com", &phi_k_com);

        let out_eval_i = if bw.attn_wit.normalized_out.is_some() {
            mat_to_mle(&bw.attn_wit.out, t, d).evaluate(&combine(&r_t, &r_k_o))
        } else {
            // Legacy unnormalized mode: attention output is the O-proj input.
            let alpha_o = bpk.o_pk.vk.alpha;
            if alpha_o == F::from(0u64) {
                F::from(0u64)
            } else {
                batch_oproj.final_evals_f[i] * alpha_o.inverse().unwrap()
            }
        };

        phi_q_mles.push(phi_q_mle);
        phi_k_mles.push(phi_k_mle);
        ctx_mles.push(ctx_mle);
        attn_phi_q_coms.push(phi_q_com);
        attn_phi_k_coms.push(phi_k_com);
        causal_context_coms.push(causal_context_com);
        attn_out_evals.push(out_eval_i);
    }

    // 6b. Batch out sumcheck.
    //   Non-causal: prove out_i(r_t, r_k_o) = Σ_k phi_q_i(r_t, k) · ctx_i(k, r_k_o)
    //               via a degree-2 sumcheck over k (d_bits rounds).
    //   Causal: handled separately below as a degree-3 sumcheck over (t, k)
    //               with three multiplicands (eq(r_t, ·), phi_q, c_partial).
    let mut fs_attn_out: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_attn_out: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    if !inst_attn.causal {
        for i in 0..num_blocks {
            let f_out_vals = eval_rows(&phi_q_mles[i], t_bits, &r_t);
            let g_out = eval_cols(&ctx_mles[i], d_bits, &r_k_o);
            transcript.append_field(b"attn_out_eval", &attn_out_evals[i]);
            fs_attn_out.push(DenseMLPoly::from_vec_padded(f_out_vals));
            gs_attn_out.push(DenseMLPoly::from_vec_padded(g_out));
        }
    } else {
        for i in 0..num_blocks {
            transcript.append_field(b"attn_out_eval", &attn_out_evals[i]);
        }
    }
    let eta_attn_out: F = transcript.challenge_field(b"batch_eta_attn_out");
    let weights_attn_out = powers_of(eta_attn_out, num_blocks);
    let claim_attn_out: F = weights_attn_out
        .iter()
        .zip(attn_out_evals.iter())
        .map(|(w, e)| *w * *e)
        .sum();
    let (batch_attn_out, batch_attn_out_causal, batch_r_attn_out) = if !inst_attn.causal {
        let (sc, r) = prove_sumcheck_multi_batched_owned(
            fs_attn_out,
            gs_attn_out,
            &weights_attn_out,
            claim_attn_out,
            transcript,
        );
        (Some(sc), None, r)
    } else {
        // Causal: degree-3 sumcheck Σ_{t,k} eq(r_t, t)·phi_q[t][k]·c_partial[t][k]
        // where c_partial[t][k] = Σ_b eq(r_k_o, b)·causal_context[t][k][b].
        // The (t_bits + d_bits)-variable polynomial layout uses MSB-first bits:
        // index t·d_p2 + k for t in [0, t_p2), k in [0, d_p2). h is constant in
        // the k axis, so Σ_k eq(X_k, k) = 1 makes h_mle(X_t, X_k) = eq(r_t, X_t).
        let eq_t_evals = eq_evals_msb(&r_t, t);
        let eq_k_o_evals = eq_evals_msb(&r_k_o, d);
        let t_p2 = t.next_power_of_two();
        let d_p2 = d.next_power_of_two();
        let mut fs_cubic: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
        let mut gs_cubic: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
        let mut hs_cubic: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let bw = &witness.block_witnesses[i];
            let computed;
            let causal_context = match bw.attn_wit.causal_context.as_ref() {
                Some(c) => c,
                None => {
                    computed = build_causal_context(&bw.attn_wit.phi_k, &bw.attn_wit.v, t, d);
                    &computed
                }
            };
            let mut f_evals = vec![F::ZERO; t_p2 * d_p2];
            let mut g_evals = vec![F::ZERO; t_p2 * d_p2];
            let mut h_evals = vec![F::ZERO; t_p2 * d_p2];
            // h is the constant-in-k extension of eq(r_t, ·); fill all
            // (t_p2 × d_p2) entries with eq_t_evals[t] regardless of k.
            for tt in 0..t_p2 {
                let eq_t = eq_t_evals[tt];
                for k in 0..d_p2 {
                    h_evals[tt * d_p2 + k] = eq_t;
                }
            }
            for tt in 0..t {
                for k in 0..d {
                    f_evals[tt * d_p2 + k] = bw.attn_wit.phi_q[tt][k];
                    let mut c_partial = F::ZERO;
                    for b in 0..d {
                        c_partial += eq_k_o_evals[b] * causal_context[tt * d + k][b];
                    }
                    g_evals[tt * d_p2 + k] = c_partial;
                }
            }
            fs_cubic.push(DenseMLPoly::new(f_evals));
            gs_cubic.push(DenseMLPoly::new(g_evals));
            hs_cubic.push(DenseMLPoly::new(h_evals));
        }
        let (sc, r) = prove_sumcheck_cubic_multi_batched_owned(
            fs_cubic,
            gs_cubic,
            hs_cubic,
            &weights_attn_out,
            claim_attn_out,
            transcript,
        );
        (None, Some(sc), r)
    };
    // 6c. Non-causal: ctx(batch_r_attn_out, r_k_o) = Σ_t phi_k(t, batch_r_attn_out) · v(t, r_k_o).
    //     Causal: prove a random linear combination of prefix-context equations
    //     C_i[a,b] = Σ_{s<=i} phi_k[s,a] · v[s,b].
    let mut fs_attn_ctx: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_attn_ctx: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut hs_attn_ctx: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    // GKR chain: in causal mode the CTX-relation is proved at the prefix point
    // implied by the OUT sumcheck terminal — (r_final_t, r_final_k) for the
    // (t, a) axes from batch_r_attn_out, plus r_k_o for the b-axis. The CTX
    // claim per block is OUT's final_evals_g[i] (= causal_context_mle at that
    // point), so we don't sample fresh prefix challenges and don't need to
    // separately bind the prefix-point opening of causal_context.
    let causal_ctx_prefix_evals: Vec<F> = Vec::new();
    let attn_ctx_evals: Vec<F> = if inst_attn.causal {
        let prefix_t = batch_r_attn_out[..t_bits].to_vec();
        let prefix_a = batch_r_attn_out[t_bits..t_bits + d_bits].to_vec();
        let prefix_b = r_k_o.clone();
        let suffix_evals = suffix_eq_evals_msb(&prefix_t, t);
        let eq_a_evals = eq_evals_msb(&prefix_a, d);
        let eq_b_evals = eq_evals_msb(&prefix_b, d);
        let claims = batch_attn_out_causal
            .as_ref()
            .expect("causal mode always produces batch_attn_out_causal")
            .final_evals_g
            .clone();
        for i in 0..num_blocks {
            let bw = &witness.block_witnesses[i];
            let (f_ctx, g_v, h_ctx) = causal_prefix_polys(
                &bw.attn_wit.phi_k,
                &bw.attn_wit.v,
                &suffix_evals,
                &eq_a_evals,
                &eq_b_evals,
                t,
                d,
            );
            fs_attn_ctx.push(f_ctx);
            gs_attn_ctx.push(g_v);
            hs_attn_ctx.push(h_ctx);
        }
        claims
    } else {
        let claims = batch_attn_out
            .as_ref()
            .expect("non-causal mode always produces batch_attn_out")
            .final_evals_g
            .clone();
        for i in 0..num_blocks {
            transcript.append_field(b"attn_ctx_eval", &claims[i]);
            let f_ctx =
                DenseMLPoly::from_vec_padded(eval_cols(&phi_k_mles[i], t_bits, &batch_r_attn_out));
            let g_v = DenseMLPoly::from_vec_padded(eval_cols(&v_mles[i], t_bits, &r_k_o));
            fs_attn_ctx.push(f_ctx);
            gs_attn_ctx.push(g_v);
        }
        claims
    };
    let eta_attn_ctx: F = transcript.challenge_field(b"batch_eta_attn_ctx");
    let weights_attn_ctx = powers_of(eta_attn_ctx, num_blocks);
    let claim_attn_ctx: F = weights_attn_ctx
        .iter()
        .zip(attn_ctx_evals.iter())
        .map(|(w, e)| *w * *e)
        .sum();
    let (batch_attn_ctx, batch_attn_ctx_causal, batch_r_attn_ctx) = if !inst_attn.causal {
        let (sc, r) = prove_sumcheck_multi_batched_owned(
            fs_attn_ctx,
            gs_attn_ctx,
            &weights_attn_ctx,
            claim_attn_ctx,
            transcript,
        );
        (Some(sc), None, r)
    } else {
        let (sc, r) = prove_sumcheck_cubic_multi_batched_owned(
            fs_attn_ctx,
            gs_attn_ctx,
            hs_attn_ctx,
            &weights_attn_ctx,
            claim_attn_ctx,
            transcript,
        );
        (None, Some(sc), r)
    };

    let (attn_norm_sumcheck, attn_norm_r) = if has_attn_norm {
        let r = challenge_vec(transcript, td_num_vars, b"attn_norm_r");
        let lambda = transcript.challenge_field::<F>(b"attn_norm_lambda");
        let eq = DenseMLPoly::new(eq_evals_msb(&r, 1usize << td_num_vars));
        let one = DenseMLPoly::new(vec![F::ONE; 1 << td_num_vars]);
        let scale_f = F::from(ATTN_NORM_SCALE);
        let mut fs = Vec::with_capacity(7 * num_blocks);
        let mut gs = Vec::with_capacity(7 * num_blocks);
        let mut hs = Vec::with_capacity(7 * num_blocks);
        let mut ws = Vec::with_capacity(7 * num_blocks);
        for (i, bw) in witness.block_witnesses.iter().enumerate() {
            let y = bw
                .attn_wit
                .normalized_out
                .as_ref()
                .ok_or_else(|| format!("block {i}: missing normalized attention output"))?;
            let z = bw
                .attn_wit
                .norm_z
                .as_ref()
                .ok_or_else(|| format!("block {i}: missing attention z"))?;
            let rem = bw
                .attn_wit
                .norm_rem
                .as_ref()
                .ok_or_else(|| format!("block {i}: missing attention rem"))?;
            let diff = bw
                .attn_wit
                .norm_diff
                .as_ref()
                .ok_or_else(|| format!("block {i}: missing attention diff"))?;
            let n_mle = mat_to_mle(&bw.attn_wit.out, t, d);
            let y_mle = mat_to_mle(y, t, d);
            let r_mle = mat_to_mle(rem, t, d);
            let d_mle = mat_to_mle(diff, t, d);
            let z_ext: Vec<Vec<F>> = (0..t).map(|row| vec![z[row]; d]).collect();
            let z_mle = mat_to_mle(&z_ext, t, d);

            fs.extend(vec![eq.clone(); 7]);
            gs.extend(vec![
                n_mle.clone(),
                r_mle.clone(),
                z_mle.clone(),
                z_mle.clone(),
                one.clone(),
                r_mle.clone(),
                d_mle.clone(),
            ]);
            hs.extend(vec![
                one.clone(),
                one.clone(),
                y_mle,
                one.clone(),
                one.clone(),
                one.clone(),
                one.clone(),
            ]);
            ws.extend(vec![
                scale_f,
                -F::ONE,
                -F::ONE,
                lambda,
                -lambda,
                -lambda,
                -lambda,
            ]);
        }
        let (proof, r_sc) =
            prove_sumcheck_cubic_multi_batched_owned(fs, gs, hs, &ws, F::ZERO, transcript);
        (Some(proof), Some(r_sc))
    } else {
        (None, None)
    };

    let (
        attn_z_sumcheck,
        attn_z_ksum_sumcheck,
        attn_z_causal_sumcheck,
        attn_z_phi_q_point,
        attn_z_phi_k_point,
        _attn_z_phi_q_evals,
        _attn_z_phi_k_evals,
    ) = if let (true, Some(ref r_norm), Some(ref norm_sc)) = (
        has_attn_norm,
        attn_norm_r.as_ref(),
        attn_norm_sumcheck.as_ref(),
    ) {
        let r_norm_t = r_norm[..t_bits].to_vec();
        let z_claims: Vec<F> = (0..num_blocks)
            .map(|i| norm_sc.final_evals_g[7 * i + 2])
            .collect();
        let eta_z = transcript.challenge_field::<F>(b"attn_z_eta");
        let weights_z = powers_of(eta_z, num_blocks);
        let claim_z: F = weights_z
            .iter()
            .zip(z_claims.iter())
            .map(|(w, z)| *w * *z)
            .sum();
        if inst_attn.causal {
            let eq_t = eq_evals_msb(&r_norm_t, t);
            let mut fs = Vec::with_capacity(num_blocks);
            let mut gs = Vec::with_capacity(num_blocks);
            let mut hs = Vec::with_capacity(num_blocks);
            for i in 0..num_blocks {
                let bw = &witness.block_witnesses[i];
                let (f, g, h) =
                    causal_denominator_polys(&bw.attn_wit.phi_q, &bw.attn_wit.phi_k, &eq_t, t, d);
                fs.push(f);
                gs.push(g);
                hs.push(h);
            }
            let (sc, r_z) = prove_sumcheck_cubic_multi_batched_owned(
                fs, gs, hs, &weights_z, claim_z, transcript,
            );
            let r_i = r_z[..t_bits].to_vec();
            let r_s = r_z[t_bits..2 * t_bits].to_vec();
            let r_a = r_z[2 * t_bits..].to_vec();
            let phi_q_point = combine(&r_i, &r_a);
            let phi_k_point = combine(&r_s, &r_a);
            let phi_q_evals = sc.final_evals_g.clone();
            let phi_k_evals = sc.final_evals_h.clone();
            (
                None,
                None,
                Some(sc),
                Some(phi_q_point),
                Some(phi_k_point),
                phi_q_evals,
                phi_k_evals,
            )
        } else {
            let mut fs = Vec::with_capacity(num_blocks);
            let mut gs = Vec::with_capacity(num_blocks);
            let mut ksum_mles = Vec::with_capacity(num_blocks);
            for i in 0..num_blocks {
                let bw = &witness.block_witnesses[i];
                fs.push(DenseMLPoly::from_vec_padded(eval_rows(
                    &phi_q_mles[i],
                    t_bits,
                    &r_norm_t,
                )));
                let ksum: Vec<F> = (0..d)
                    .map(|a| (0..t).map(|s| bw.attn_wit.phi_k[s][a]).sum())
                    .collect();
                let ksum_mle = vec_to_mle(&ksum, d);
                gs.push(ksum_mle.clone());
                ksum_mles.push(ksum_mle);
            }
            let (sc, r_a) =
                prove_sumcheck_multi_batched_owned(fs, gs, &weights_z, claim_z, transcript);
            let ksum_claims = sc.final_evals_g.clone();
            let claim_ksum: F = weights_z
                .iter()
                .zip(ksum_claims.iter())
                .map(|(w, v)| *w * *v)
                .sum();
            let mut fs_k = Vec::with_capacity(num_blocks);
            let mut gs_k = Vec::with_capacity(num_blocks);
            for i in 0..num_blocks {
                fs_k.push(DenseMLPoly::new(vec![F::ONE; 1 << t_bits]));
                gs_k.push(DenseMLPoly::from_vec_padded(eval_cols(
                    &phi_k_mles[i],
                    t_bits,
                    &r_a,
                )));
            }
            let (sc_k, r_s) =
                prove_sumcheck_multi_batched_owned(fs_k, gs_k, &weights_z, claim_ksum, transcript);
            let phi_q_point = combine(&r_norm_t, &r_a);
            let phi_k_point = combine(&r_s, &r_a);
            let phi_q_evals = sc.final_evals_f.clone();
            let phi_k_evals = sc_k.final_evals_g.clone();
            (
                Some(sc),
                Some(sc_k),
                None,
                Some(phi_q_point),
                Some(phi_k_point),
                phi_q_evals,
                phi_k_evals,
            )
        }
    } else {
        (None, None, None, None, None, Vec::new(), Vec::new())
    };
    eprintln!(
        "[prove] batch_attn: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();
    // =========================================================================
    // 7. Per-block FFN: Lasso + M commit + absorb coms
    // =========================================================================
    let mut ffn_a_coms: Vec<HyraxCommitment> = Vec::with_capacity(num_blocks);
    let mut ffn_a_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut ffn_m_coms: Vec<HyraxCommitment> = Vec::with_capacity(num_blocks);
    let mut ffn_m_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut ffn_lasso_instances = Vec::with_capacity(num_blocks);
    let mut ffn_query_indices_all: Vec<Vec<usize>> = Vec::with_capacity(num_blocks);
    let mut ffn_output_bindings = Vec::with_capacity(num_blocks);
    let mut ffn_instance_coms = Vec::with_capacity(num_blocks);
    let mut ffn_lasso_nu = 0usize;

    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let bw = &witness.block_witnesses[i];

        // Absorb FFN weight commitments (mirrors prove_ffn ordering)
        absorb_com(transcript, b"w1_com", &bpk.ffn_pk.vk.w1_com);
        absorb_com(transcript, b"w2_com", &bpk.ffn_pk.vk.w2_com);
        // x_com = None (GKR mode), y_com = None (already in Phase 1)

        // GKR backward: run Lasso FIRST to commit A before rx_y is sampled
        let ffn_query_indices = if bw.ffn_wit.activation_query_indices.is_empty() {
            flatten_mat_indices(&bw.ffn_wit.m)
        } else {
            bw.ffn_wit.activation_query_indices.clone()
        };
        let ffn_lasso_instance = crate::lookup::lasso::LassoInstance {
            tables: inst_ffn.activation_lasso.tables.clone(),
            outputs: flatten_mat_values(&bw.ffn_wit.a),
            bits_per_chunk: inst_ffn.activation_lasso.bits_per_chunk,
        };
        let a_mle = mat_to_mle(&bw.ffn_wit.a, t, d_ff);
        let (nu_a, _, params_a) = params_from_vars(t_bits + f_bits);
        let ffn_a_com = hyrax_commit(&a_mle.evaluations, nu_a, &params_a);
        let m_mle = mat_to_mle(&bw.ffn_wit.m, t, d_ff);
        let (nu_m, _, params_m) = params_from_vars(t_bits + f_bits);
        let ffn_m_com = hyrax_commit(&m_mle.evaluations, nu_m, &params_m);
        absorb_com(transcript, b"m_com", &ffn_m_com);

        ffn_instance_coms.push(bpk.ffn_pk.activation_lasso_pk.table_coms.clone());
        ffn_lasso_nu = bpk.ffn_pk.activation_lasso_pk.nu;
        ffn_lasso_instances.push(ffn_lasso_instance);
        ffn_query_indices_all.push(ffn_query_indices);
        ffn_output_bindings.push(LassoOutputBinding {
            com: ffn_a_com.clone(),
            num_vars: t_bits + f_bits,
            mle_evals: a_mle.evaluations.clone(),
        });
        ffn_a_coms.push(ffn_a_com);
        ffn_a_mles.push(a_mle);
        ffn_m_coms.push(ffn_m_com);
        ffn_m_mles.push(m_mle);
    }

    let global_ffn_lasso_inst = LassoMultiInstance {
        instances: ffn_lasso_instances,
    };
    let ffn_instance_to_group = crate::lookup::lasso::derive_instance_groups(&ffn_instance_coms);
    let global_ffn_lasso_pk = LassoMultiProvingKey {
        instance_table_coms: ffn_instance_coms,
        instance_to_group: ffn_instance_to_group,
        nu: ffn_lasso_nu,
    };
    let ffn_lasso_proof = prove_lasso_multi(
        &global_ffn_lasso_inst,
        &ffn_query_indices_all,
        &global_ffn_lasso_pk,
        &ffn_output_bindings,
        transcript,
        lasso_params,
    );

    let ffn_index_refs: Vec<&[usize]> = ffn_lasso_proof
        .all_query_indices
        .iter()
        .map(|v| v.as_slice())
        .collect();
    absorb_index_vectors(transcript, b"ffn_lasso_indices", &ffn_index_refs);
    let ffn_quant_proof = prove_quantization_batch(
        b"ffn_quant_rem_com",
        &ffn_m_mles,
        &ffn_m_coms,
        &ffn_lasso_proof.all_query_indices,
        inst_ffn.activation_lasso.tables.len(),
        inst_ffn.activation_lasso.bits_per_chunk,
        t,
        d_ff,
        &pk.vk.block_vks[0].ffn_activation_quant,
        transcript,
    )?;
    let ffn_lasso_bind_point = challenge_vec(transcript, t_bits + f_bits, b"ffn_lasso_bind_r");
    let ffn_bind_evals_vecs: Vec<Vec<F>> = ffn_lasso_proof
        .all_query_indices
        .iter()
        .map(|indices| indices_to_mle_evals(indices))
        .collect();
    let ffn_bind_refs: Vec<&[F]> = ffn_bind_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let (nu_mff_bind, sigma_mff_bind, _) = params_from_vars(t_bits + f_bits);
    let ffn_lasso_bind_open = hyrax_open_batch(
        &ffn_bind_refs,
        &ffn_lasso_bind_point,
        nu_mff_bind,
        sigma_mff_bind,
        transcript,
    );
    eprintln!(
        "[prove] ffn_lasso: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();

    // =========================================================================
    // 8. Batch FFN-Y: Y = A · W2 at shared (r_t, r_out) = r_td
    // =========================================================================
    let mut fs_ffn_y: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_ffn_y: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut ffn_y_targets: Vec<F> = Vec::with_capacity(num_blocks);
    let mut out_ffn_mles: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);

    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];
        let bw = &witness.block_witnesses[i];

        let a_mle = &ffn_a_mles[i];
        let y_mle = mat_to_mle(&bw.ffn_wit.y, t, d);

        let y_eval = y_mle.evaluate(&combine(&r_t, &r_out));

        // Bind y_eval claim to transcript (mirrors prove_ffn step 4)
        transcript.append_field(b"claim_y", &y_eval);

        // f_i(k) = A_i[r_t, k]
        let f_a = eval_rows(a_mle, t_bits, &r_t);
        // g_i(k) = W2_i[k, r_out]
        let g_w2 = eval_cols_ternary(&bpk.ffn_pk.w2, &r_out, d_ff, d);

        fs_ffn_y.push(DenseMLPoly::from_vec_padded(f_a));
        gs_ffn_y.push(DenseMLPoly::from_vec_padded(g_w2));
        ffn_y_targets.push(y_eval);
        out_ffn_mles.push(y_mle);
    }

    // Cross-block FFN-Y batch sumcheck
    let eta_ffn_y: F = transcript.challenge_field(b"batch_eta_ffn_y");
    let weights_ffn_y = powers_of(eta_ffn_y, num_blocks);
    let claim_ffn_y: F = weights_ffn_y
        .iter()
        .zip(ffn_y_targets.iter())
        .map(|(w, t)| *w * *t)
        .sum();
    let (batch_ffn_y, r_k_fy) = prove_sumcheck_multi_batched_owned(
        fs_ffn_y,
        gs_ffn_y,
        &weights_ffn_y,
        claim_ffn_y,
        transcript,
    );
    eprintln!(
        "[prove] batch_ffn_y: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();

    // =========================================================================
    // 9. Batch FFN-M: M = X2 · W1 with shared rx_m, ry_m
    // =========================================================================
    let rx_m = challenge_vec(transcript, t_bits, b"ffn_rx_m");
    let ry_m = challenge_vec(transcript, f_bits, b"ffn_ry_m");

    let mut fs_ffn_m: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut gs_ffn_m: Vec<DenseMLPoly> = Vec::with_capacity(num_blocks);
    let mut ffn_m_targets: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_ffn_m_eval: Vec<F> = Vec::with_capacity(num_blocks);
    let x_norm2_mles: Vec<DenseMLPoly> = witness
        .block_witnesses
        .iter()
        .map(|bw| mat_to_mle(&bw.ln2_wit.y, t, d))
        .collect();

    for i in 0..num_blocks {
        let bpk = &pk.block_pks[i];

        let m_eval = ffn_m_mles[i].evaluate(&combine(&rx_m, &ry_m));

        // Bind m_eval claim to transcript
        transcript.append_field(b"claim_m", &m_eval);

        // f_i(k) = X2_i[rx_m, k]
        let f_x = eval_rows(&x_norm2_mles[i], t_bits, &rx_m);
        // g_i(k) = W1_i[k, ry_m]
        let g_w1 = eval_cols_ternary(&bpk.ffn_pk.w1, &ry_m, d, d_ff);

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
    let (batch_ffn_m, r_k_m) = prove_sumcheck_multi_batched_owned(
        fs_ffn_m,
        gs_ffn_m,
        &weights_ffn_m,
        claim_ffn_m,
        transcript,
    );
    eprintln!(
        "[prove] batch_ffn_m: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();

    // =========================================================================
    // 10. Build per-block proof structs
    // =========================================================================
    let mut block_proofs: Vec<TransformerBlockProof> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let p1 = &phase1_data[i];

        block_proofs.push(TransformerBlockProof {
            q_norm_y_com: p1.q_norm_y_com.clone(),
            k_norm_y_com: p1.k_norm_y_com.clone(),
            attn_out_norm_y_com: p1.attn_out_norm_y_com.clone(),
            ffn_lasso_proof: empty_lasso_proof(),
            ffn_a_com: ffn_a_coms[i].clone(),
            ffn_m_com: ffn_m_coms[i].clone(),
            x_norm1_com: p1.x_norm1_com.clone(),
            q_com: p1.q_com.clone(),
            k_com: p1.k_com.clone(),
            v_com: p1.v_com.clone(),
            attn_norm_com: p1.attn_norm_com.clone(),
            attn_num_com: p1.attn_num_com.clone(),
            attn_z_com: p1.attn_z_com.clone(),
            attn_rem_com: p1.attn_rem_com.clone(),
            attn_diff_com: p1.attn_diff_com.clone(),
            out_attn_com: p1.out_attn_com.clone(),
            x_norm2_com: p1.x_norm2_com.clone(),
            out_ffn_com: p1.out_ffn_com.clone(),
            q_eval: pb_q_eval[i],
            k_eval: pb_k_eval[i],
            v_eval_rtd: pb_v_eval[i],
            out_attn_eval: out_attn_mles[i].evaluate(&combine(&r_t, &r_out)),
            out_ffn_eval: ffn_y_targets[i],
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
            causal_context_com: causal_context_coms[i].clone(),
            attn_out_eval: attn_out_evals[i],
        });
    }

    // =========================================================================
    // 11. Final LayerNorm — absorb IO commitments at the original transcript
    //     position so the LM-head proof binds to final_ln_out_com just like
    //     under PROOF_VERSION 12. The actual LN sumchecks/openings are emitted
    //     by the cross-LN batched proof at the end of the model.
    // =========================================================================
    let final_ln_out_com = commit_mat(&witness.final_ln_wit.y, t, d);
    absorb_com(transcript, b"x_com", &current_x_com);
    absorb_com(transcript, b"y_com", &final_ln_out_com);
    let final_ln_io = LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: Some(final_ln_out_com.clone()),
    };
    let final_ln_ranges = [(final_sig_rp, final_sig_rv), (final_y_rp, final_y_rv)];
    eprintln!(
        "[prove] final_ln_absorb: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();

    // =========================================================================
    // 11b. Cross-LN batched proof — covers every LN in the model (5L + 1).
    //      Order matches the verifier's mirrored collection in verifier.rs.
    // =========================================================================
    let mut ln_witnesses: Vec<&LayerNormWitness> = Vec::with_capacity(5 * num_blocks + 1);
    let mut ln_io_coms_local: Vec<&LayerNormIOCommitments> = Vec::with_capacity(5 * num_blocks + 1);
    let mut ln_vks: Vec<&LayerNormVerifyingKey> = Vec::with_capacity(5 * num_blocks + 1);
    let mut ln_sigma_ranges: Vec<(RangeWitnessProof, Vec<F>)> =
        Vec::with_capacity(5 * num_blocks + 1);
    let mut ln_y_ranges: Vec<(RangeWitnessProof, Vec<F>)> = Vec::with_capacity(5 * num_blocks + 1);
    let mut ln_sigma_range_bits: Vec<usize> = Vec::with_capacity(5 * num_blocks + 1);
    let mut ln_y_range_bits: Vec<usize> = Vec::with_capacity(5 * num_blocks + 1);
    for i in 0..num_blocks {
        let bw = &witness.block_witnesses[i];
        let bpk = &pk.block_pks[i];
        let li = &block_ln_inputs[i];
        let rb = 10 * i;
        // Order: ln1, q_norm, k_norm, attn_out_norm, ln2.
        ln_witnesses.push(&bw.ln1_wit);
        ln_io_coms_local.push(&li.ln1_io);
        ln_vks.push(&bpk.ln1_vk);
        ln_sigma_ranges.push(li.ranges[0].clone());
        ln_y_ranges.push(li.ranges[1].clone());
        ln_sigma_range_bits.push(ln_range_bits_vec[rb]);
        ln_y_range_bits.push(ln_range_bits_vec[rb + 1]);

        ln_witnesses.push(&bw.q_norm_wit);
        ln_io_coms_local.push(&li.q_norm_io);
        ln_vks.push(&bpk.q_norm_vk);
        ln_sigma_ranges.push(li.ranges[4].clone());
        ln_y_ranges.push(li.ranges[5].clone());
        ln_sigma_range_bits.push(ln_range_bits_vec[rb + 4]);
        ln_y_range_bits.push(ln_range_bits_vec[rb + 5]);

        ln_witnesses.push(&bw.k_norm_wit);
        ln_io_coms_local.push(&li.k_norm_io);
        ln_vks.push(&bpk.k_norm_vk);
        ln_sigma_ranges.push(li.ranges[6].clone());
        ln_y_ranges.push(li.ranges[7].clone());
        ln_sigma_range_bits.push(ln_range_bits_vec[rb + 6]);
        ln_y_range_bits.push(ln_range_bits_vec[rb + 7]);

        ln_witnesses.push(&bw.attn_out_norm_wit);
        ln_io_coms_local.push(&li.aon_io);
        ln_vks.push(&bpk.attn_out_norm_vk);
        ln_sigma_ranges.push(li.ranges[8].clone());
        ln_y_ranges.push(li.ranges[9].clone());
        ln_sigma_range_bits.push(ln_range_bits_vec[rb + 8]);
        ln_y_range_bits.push(ln_range_bits_vec[rb + 9]);

        ln_witnesses.push(&bw.ln2_wit);
        ln_io_coms_local.push(&li.ln2_io);
        ln_vks.push(&bpk.ln2_vk);
        ln_sigma_ranges.push(li.ranges[2].clone());
        ln_y_ranges.push(li.ranges[3].clone());
        ln_sigma_range_bits.push(ln_range_bits_vec[rb + 2]);
        ln_y_range_bits.push(ln_range_bits_vec[rb + 3]);
    }
    ln_witnesses.push(&witness.final_ln_wit);
    ln_io_coms_local.push(&final_ln_io);
    ln_vks.push(&pk.vk.final_ln_vk);
    let [final_sig_pair, final_y_pair] = final_ln_ranges;
    ln_sigma_ranges.push(final_sig_pair);
    ln_y_ranges.push(final_y_pair);
    ln_sigma_range_bits.push(ln_range_bits_vec[10 * num_blocks]);
    ln_y_range_bits.push(ln_range_bits_vec[10 * num_blocks + 1]);

    let ln_batched_input = LayerNormsBatchedInput {
        witnesses: ln_witnesses,
        io_coms: ln_io_coms_local,
        vks: ln_vks,
        sigma_ranges: ln_sigma_ranges,
        y_ranges: ln_y_ranges,
        sigma_range_bits: ln_sigma_range_bits,
        y_range_bits: ln_y_range_bits,
    };
    let ln_batched_proof = prove_layernorms_batched(&ln_batched_input, transcript)?;
    eprintln!(
        "[prove] ln_batched: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();

    // =========================================================================
    // 12. LM Head
    // =========================================================================
    let logits_mle = mat_to_mle(&witness.lm_head_wit.y, t, v);
    let logits_com = commit_mat(&witness.lm_head_wit.y, t, v);
    let lm_io = ProjectionIOCommitments {
        x_com: Some(final_ln_out_com.clone()),
    };
    let (lm_head_proof, lm_y_claim, _) = prove_projection(
        &pk.lm_head_pk,
        &witness.lm_head_wit,
        &lm_io,
        transcript,
        None,
    )?;
    let v_bits = v.next_power_of_two().trailing_zeros() as usize;
    let lm_logits_num_vars = t_bits + v_bits;
    let (lm_nu, lm_sigma, _) = params_from_vars(lm_logits_num_vars);
    let lm_head_logits_open =
        hyrax_open(&logits_mle.evaluations, &lm_y_claim.point, lm_nu, lm_sigma);
    eprintln!(
        "[prove] lm_head: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    phase_t0 = Instant::now();

    // =========================================================================
    // 13. Advance transcript for accumulator mu challenges
    // =========================================================================
    // Verifier draws 13 mus (PROOF_VERSION 15): inter, ln_t, ln_td, proj_w,
    // proj_b, lmh_w, lmh_b, rng_sig, rng_y, rng_m, quant_ffn, quant_m,
    // attn_norm.  Plus 2 read-only rho fuse challenges (consumed below).
    for _ in 0..13 {
        let _ = transcript.challenge_field::<F>(b"hyrax_group_mu");
    }

    // =========================================================================
    // 14. Global hyrax_open_batch for 5L intermediate matrices at r_td
    // =========================================================================
    let mut evals_refs: Vec<&[F]> = Vec::with_capacity(5 * num_blocks);
    for i in 0..num_blocks {
        evals_refs.push(q_mles[i].evaluations.as_slice());
        evals_refs.push(k_mles[i].evaluations.as_slice());
        evals_refs.push(v_mles[i].evaluations.as_slice());
        evals_refs.push(out_attn_mles[i].evaluations.as_slice());
        evals_refs.push(out_ffn_mles[i].evaluations.as_slice());
    }
    let inter_batch_open = hyrax_open_batch(&evals_refs, &r_td, nu_td, sigma_td, transcript);

    // =========================================================================
    // 15. Cross-block batch opens
    // =========================================================================

    // x_norm1_batch: L x_norm1_i at combine(r_t, r_k_qkv) [td_num_vars]
    let x_norm1_point = combine(&r_t, &r_k_qkv);
    let x_norm1_refs: Vec<&[F]> = x_norm1_mles
        .iter()
        .map(|m| m.evaluations.as_slice())
        .collect();
    let x_norm1_batch_open =
        hyrax_open_batch(&x_norm1_refs, &x_norm1_point, nu_td, sigma_td, transcript);

    // P2: wq/wk/wv all open at the same point combine(r_k_qkv, r_out) with
    // params_qkvo_w — fold into ONE hyrax_open_batch over 3·L commitments.
    let wq_point = combine(&r_k_qkv, &r_out);
    let mut qkv_w_refs: Vec<&[F]> = Vec::with_capacity(3 * num_blocks);
    for sm in &static_mles {
        qkv_w_refs.push(sm.w_q.evaluations.as_slice());
        qkv_w_refs.push(sm.w_k.evaluations.as_slice());
        qkv_w_refs.push(sm.w_v.evaluations.as_slice());
    }
    let qkv_w_batch_open = hyrax_open_batch(&qkv_w_refs, &wq_point, nu_w, sigma_w, transcript);

    // w_o_batch: L Wo_i at combine(r_k_o, r_out) [d_bits + d_bits].  Different
    // point from the qkv_w group; stays as its own (small) batch.
    let wo_point = combine(&r_k_o, &r_out);
    let wo_refs: Vec<&[F]> = static_mles
        .iter()
        .map(|m| m.w_o.evaluations.as_slice())
        .collect();
    let w_o_batch_open = hyrax_open_batch(&wo_refs, &wo_point, nu_w, sigma_w, transcript);

    // P2: bq/bk/bv/bo all open at r_out with params_qkvo_b — fold all four
    // bias groups into ONE hyrax_open_batch over 4·L commitments.
    let mut qkvo_bias_refs: Vec<&[F]> = Vec::with_capacity(4 * num_blocks);
    for sm in &static_mles {
        qkvo_bias_refs.push(sm.b_q.evaluations.as_slice());
        qkvo_bias_refs.push(sm.b_k.evaluations.as_slice());
        qkvo_bias_refs.push(sm.b_v.evaluations.as_slice());
        qkvo_bias_refs.push(sm.b_o.evaluations.as_slice());
    }
    let qkvo_bias_batch_open = hyrax_open_batch(&qkvo_bias_refs, &r_out, nu_b, sigma_b, transcript);

    // w2_batch: L W2_i at combine(r_k_fy, r_out) [f_bits + d_bits]
    let w2_point = combine(&r_k_fy, &r_out);
    let w2_refs: Vec<&[F]> = static_mles
        .iter()
        .map(|m| m.w2.evaluations.as_slice())
        .collect();
    let w2_batch_open = hyrax_open_batch(&w2_refs, &w2_point, nu_wff, sigma_wff, transcript);

    // ffn_a_batch: L A_i at combine(r_t, r_k_fy) [t_bits + f_bits]
    let ffn_a_point = combine(&r_t, &r_k_fy);
    let ffn_a_refs: Vec<&[F]> = ffn_a_mles
        .iter()
        .map(|m| m.evaluations.as_slice())
        .collect();
    let ffn_a_batch_open =
        hyrax_open_batch(&ffn_a_refs, &ffn_a_point, nu_mff, sigma_mff, transcript);

    // w1_batch: L W1_i at combine(r_k_m, ry_m) [d_bits + f_bits]
    let w1_point = combine(&r_k_m, &ry_m);
    let w1_refs: Vec<&[F]> = static_mles
        .iter()
        .map(|m| m.w1.evaluations.as_slice())
        .collect();
    let w1_batch_open = hyrax_open_batch(&w1_refs, &w1_point, nu_wff, sigma_wff, transcript);

    // x_norm2_batch: L x_norm2_i at combine(rx_m, r_k_m) [td_num_vars]
    let x_norm2_point = combine(&rx_m, &r_k_m);
    let x_norm2_refs: Vec<&[F]> = x_norm2_mles
        .iter()
        .map(|m| m.evaluations.as_slice())
        .collect();
    let x_norm2_batch_open =
        hyrax_open_batch(&x_norm2_refs, &x_norm2_point, nu_td, sigma_td, transcript);

    // ffn_m_com_batch: L M_i at combine(rx_m, ry_m) [t_bits + f_bits]
    let ffn_m_point = combine(&rx_m, &ry_m);
    let ffn_m_refs: Vec<&[F]> = ffn_m_mles
        .iter()
        .map(|m| m.evaluations.as_slice())
        .collect();
    let ffn_m_com_batch_open =
        hyrax_open_batch(&ffn_m_refs, &ffn_m_point, nu_mff, sigma_mff, transcript);

    // phi_q_batch: open phi_q at the (t, k) point implied by the attention-out
    // sumcheck. In causal mode the sumcheck folds in r_t internally, so
    // batch_r_attn_out is already (t_bits + d_bits) long; in non-causal mode
    // batch_r_attn_out has only d_bits and r_t prefixes the t-axis.
    let phi_q_attn_point = if inst_attn.causal {
        batch_r_attn_out.clone()
    } else {
        combine(&r_t, &batch_r_attn_out)
    };
    let phi_q_refs: Vec<&[F]> = phi_q_mles
        .iter()
        .map(|m| m.evaluations.as_slice())
        .collect();
    let phi_q_batch_open =
        hyrax_open_batch(&phi_q_refs, &phi_q_attn_point, nu_td, sigma_td, transcript);

    let mut causal_phi_k_prefix_evals = Vec::new();
    let mut causal_v_prefix_evals = Vec::new();
    let (phi_k_attn_point, v_attn_batch_point, causal_ctx_prefix_batch_open) = if inst_attn.causal {
        let r_s = batch_r_attn_ctx[..t_bits].to_vec();
        let r_a = batch_r_attn_ctx[t_bits..t_bits + d_bits].to_vec();
        let r_b = batch_r_attn_ctx[t_bits + d_bits..].to_vec();
        let phi_k_point = combine(&r_s, &r_a);
        let v_point = combine(&r_s, &r_b);
        causal_phi_k_prefix_evals = phi_k_mles
            .iter()
            .map(|m| m.evaluate(&phi_k_point))
            .collect();
        causal_v_prefix_evals = v_mles.iter().map(|m| m.evaluate(&v_point)).collect();
        // GKR chain: the CTX-relation prefix point is implied by the OUT
        // sumcheck terminal, so no separate prefix-point open of
        // causal_context_com is needed.
        (phi_k_point, v_point, None)
    } else {
        (
            combine(&batch_r_attn_ctx, &batch_r_attn_out),
            combine(&batch_r_attn_ctx, &r_k_o),
            None,
        )
    };

    // phi_k_batch: non-causal opens phi_k at (batch_r_attn_ctx, batch_r_attn_out);
    // causal opens the unscaled phi_k leaf from the prefix-context sumcheck.
    let phi_k_refs: Vec<&[F]> = phi_k_mles
        .iter()
        .map(|m| m.evaluations.as_slice())
        .collect();
    let phi_k_batch_open =
        hyrax_open_batch(&phi_k_refs, &phi_k_attn_point, nu_td, sigma_td, transcript);

    // v_attn_batch: non-causal opens v at (batch_r_attn_ctx, r_k_o);
    // causal opens the unscaled v leaf from the prefix-context sumcheck.
    let v_attn_refs: Vec<&[F]> = v_mles.iter().map(|m| m.evaluations.as_slice()).collect();
    let v_attn_batch_open = hyrax_open_batch(
        &v_attn_refs,
        &v_attn_batch_point,
        nu_td,
        sigma_td,
        transcript,
    );

    let attn_z_phi_q_open = if let Some(ref p) = attn_z_phi_q_point {
        let refs: Vec<&[F]> = phi_q_mles
            .iter()
            .map(|m| m.evaluations.as_slice())
            .collect();
        Some(hyrax_open_batch(&refs, p, nu_td, sigma_td, transcript))
    } else {
        None
    };
    let attn_z_phi_k_open = if let Some(ref p) = attn_z_phi_k_point {
        let refs: Vec<&[F]> = phi_k_mles
            .iter()
            .map(|m| m.evaluations.as_slice())
            .collect();
        Some(hyrax_open_batch(&refs, p, nu_td, sigma_td, transcript))
    } else {
        None
    };

    // GKR chain (step 2): causal_context is not committed, so no Hyrax open
    // is required at the OUT-sumcheck terminal point. The CTX cubic sumcheck
    // binds the OUT final_evals_g[i] to phi_k and v algebraically.
    let causal_ctx_out_batch_open: Option<HyraxProof> = None;

    let (attn_norm_r_batch_open, attn_z_open, attn_norm_attn_point_open) = if let Some(ref r_norm) =
        attn_norm_r
    {
        let r_norm_t = r_norm[..t_bits].to_vec();
        let num_evals: Vec<Vec<F>> = witness
            .block_witnesses
            .iter()
            .map(|b| mat_to_mle(&b.attn_wit.out, t, d).evaluations)
            .collect();
        let norm_evals: Vec<Vec<F>> = witness
            .block_witnesses
            .iter()
            .map(|b| {
                mat_to_mle(
                    b.attn_wit
                        .normalized_out
                        .as_ref()
                        .expect("missing normalized attention"),
                    t,
                    d,
                )
                .evaluations
            })
            .collect();
        let z_evals: Vec<Vec<F>> = witness
            .block_witnesses
            .iter()
            .map(|b| {
                vec_to_mle(b.attn_wit.norm_z.as_ref().expect("missing attention z"), t).evaluations
            })
            .collect();
        let rem_evals: Vec<Vec<F>> = witness
            .block_witnesses
            .iter()
            .map(|b| {
                mat_to_mle(b.attn_wit.norm_rem.as_ref().expect("missing rem"), t, d).evaluations
            })
            .collect();
        let diff_evals: Vec<Vec<F>> = witness
            .block_witnesses
            .iter()
            .map(|b| {
                mat_to_mle(b.attn_wit.norm_diff.as_ref().expect("missing diff"), t, d).evaluations
            })
            .collect();
        let z_refs: Vec<&[F]> = z_evals.iter().map(|v| v.as_slice()).collect();
        let (nu_t, sigma_t, _) = params_from_vars(t_bits);

        // Merged open at r_norm: [num | norm | rem | diff], same size & params.
        let r_norm_refs: Vec<&[F]> = num_evals
            .iter()
            .chain(norm_evals.iter())
            .chain(rem_evals.iter())
            .chain(diff_evals.iter())
            .map(|v| v.as_slice())
            .collect();
        let attn_norm_r_batch_open =
            hyrax_open_batch(&r_norm_refs, r_norm, nu_td, sigma_td, transcript);

        // attn_z is at a different point (r_norm_t) and different params.
        let attn_z_open = hyrax_open_batch(&z_refs, &r_norm_t, nu_t, sigma_t, transcript);

        // Merged open at attn_point: [num | norm].
        let attn_point = combine(&r_t, &r_k_o);
        let attn_point_refs: Vec<&[F]> = num_evals
            .iter()
            .chain(norm_evals.iter())
            .map(|v| v.as_slice())
            .collect();
        let attn_norm_attn_point_open =
            hyrax_open_batch(&attn_point_refs, &attn_point, nu_td, sigma_td, transcript);

        (
            Some(attn_norm_r_batch_open),
            Some(attn_z_open),
            Some(attn_norm_attn_point_open),
        )
    } else {
        (None, None, None)
    };

    // V1: drain mus for the 5 cross_batch_opens accumulators (params_td,
    // params_qkvo_w, params_qkvo_b, params_wff, params_mff).  Mirrors verifier
    // ordering at end of cross_batch_opens.
    for _ in 0..5 {
        let _ = transcript.challenge_field::<F>(b"hyrax_group_mu");
    }

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
        let bw = &witness.block_witnesses[i];
        all_lasso_instances.push(crate::lookup::lasso::LassoInstance {
            tables: inst_attn.q_lasso.tables.clone(),
            outputs: flatten_mat_values(&bw.attn_wit.phi_q),
            bits_per_chunk: inst_attn.q_lasso.bits_per_chunk,
        });
        all_lasso_instances.push(crate::lookup::lasso::LassoInstance {
            tables: inst_attn.k_lasso.tables.clone(),
            outputs: flatten_mat_values(&bw.attn_wit.phi_k),
            bits_per_chunk: inst_attn.k_lasso.bits_per_chunk,
        });
        all_query_indices.push(if bw.attn_wit.q_query_indices.is_empty() {
            flatten_mat_indices(&bw.attn_wit.q)
        } else {
            bw.attn_wit.q_query_indices.clone()
        });
        all_query_indices.push(if bw.attn_wit.k_query_indices.is_empty() {
            flatten_mat_indices(&bw.attn_wit.k)
        } else {
            bw.attn_wit.k_query_indices.clone()
        });
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
    let global_multi_inst = LassoMultiInstance {
        instances: all_lasso_instances,
    };
    let global_instance_to_group = crate::lookup::lasso::derive_instance_groups(&all_instance_coms);
    let global_lasso_pk = LassoMultiProvingKey {
        instance_table_coms: all_instance_coms,
        instance_to_group: global_instance_to_group,
        nu: global_nu,
    };
    let qk_index_refs: Vec<&[usize]> = all_query_indices.iter().map(|v| v.as_slice()).collect();
    absorb_index_vectors(transcript, b"qk_lasso_indices", &qk_index_refs);
    if inst_attn.q_lasso.tables.len() != inst_attn.k_lasso.tables.len()
        || inst_attn.q_lasso.bits_per_chunk != inst_attn.k_lasso.bits_per_chunk
    {
        return Err("Q/K quantization lookup domains must match".to_string());
    }
    // The Lasso φ_q / φ_k are computed from q_n / k_n (post-q_norm / post-k_norm).
    // Indices come from q_norm_wit.y / k_norm_wit.y, so the quant proof must bind
    // q_norm_y_com / k_norm_y_com — NOT q_com / k_com (which are q_raw / k_raw).
    let mut qk_raw_mles = Vec::with_capacity(2 * num_blocks);
    let mut qk_raw_coms = Vec::with_capacity(2 * num_blocks);
    for i in 0..num_blocks {
        let bw = &witness.block_witnesses[i];
        qk_raw_mles.push(mat_to_mle(&bw.q_norm_wit.y, t, d));
        qk_raw_mles.push(mat_to_mle(&bw.k_norm_wit.y, t, d));
        qk_raw_coms.push(block_proofs[i].q_norm_y_com.clone());
        qk_raw_coms.push(block_proofs[i].k_norm_y_com.clone());
    }
    let qk_quant_proof = prove_quantization_batch(
        b"qk_quant_rem_com",
        &qk_raw_mles,
        &qk_raw_coms,
        &all_query_indices,
        inst_attn.q_lasso.tables.len(),
        inst_attn.q_lasso.bits_per_chunk,
        t,
        d,
        &pk.vk.block_vks[0].qk_activation_quant,
        transcript,
    )?;
    for _ in 0..2 {
        let _ = transcript.challenge_field::<F>(b"hyrax_group_mu");
    }
    let qk_lasso_bind_point = challenge_vec(transcript, td_num_vars, b"qk_lasso_bind_r");
    let qk_bind_evals_vecs: Vec<Vec<F>> = all_query_indices
        .iter()
        .map(|indices| indices_to_mle_evals(indices))
        .collect();
    let qk_bind_refs: Vec<&[F]> = qk_bind_evals_vecs.iter().map(|v| v.as_slice()).collect();
    let qk_lasso_bind_open = hyrax_open_batch(
        &qk_bind_refs,
        &qk_lasso_bind_point,
        nu_td,
        sigma_td,
        transcript,
    );
    let all_lasso_proof = prove_lasso_multi(
        &global_multi_inst,
        &all_query_indices,
        &global_lasso_pk,
        &all_output_bindings,
        transcript,
        lasso_params,
    );
    eprintln!(
        "[prove] batch_opens_qk_lasso: {:7.3}ms",
        phase_t0.elapsed().as_secs_f64() * 1000.0
    );
    eprintln!(
        "[prove] total: {:7.3}ms",
        prove_t0.elapsed().as_secs_f64() * 1000.0
    );

    Ok(TransformerModelProof {
        x_in_com,
        block_proofs,
        ln_batched_proof,
        lm_head_proof,
        final_ln_out_com,
        logits_com,
        lm_head_logits_open,
        ffn_lasso_proof,
        all_lasso_proof,
        ffn_quant_proof,
        qk_quant_proof,
        ln_range_ms,
        batch_qkv,
        batch_oproj,
        batch_ffn_y,
        batch_ffn_m,
        batch_attn_out,
        batch_attn_out_causal,
        batch_attn_ctx,
        batch_attn_ctx_causal,
        attn_norm_sumcheck,
        attn_z_sumcheck,
        attn_z_ksum_sumcheck,
        attn_z_causal_sumcheck,
        attn_norm_rem_range_proofs,
        attn_norm_diff_range_proofs,
        attn_norm_rem_range_bits,
        attn_norm_diff_range_bits,
        inter_batch_open,
        x_norm1_batch_open,
        qkv_w_batch_open,
        w_o_batch_open,
        qkvo_bias_batch_open,
        w2_batch_open,
        ffn_a_batch_open,
        w1_batch_open,
        x_norm2_batch_open,
        ffn_m_com_batch_open,
        ffn_lasso_bind_open,
        phi_q_batch_open,
        phi_k_batch_open,
        v_attn_batch_open,
        attn_norm_r_batch_open,
        attn_norm_attn_point_open,
        attn_z_open,
        attn_z_phi_q_open,
        attn_z_phi_k_open,
        causal_ctx_prefix_evals,
        causal_phi_k_prefix_evals,
        causal_v_prefix_evals,
        causal_ctx_out_batch_open,
        causal_ctx_prefix_batch_open,
        qk_lasso_bind_open,
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
    use crate::lookup::quantization::QuantizationParams;

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
            // o_bias non-zero so out_attn = W_o·attn_proj_in + bias has non-zero
            // variance, which attn_out_norm requires for its range proofs.
            o_bias: vec![F::from(2u64), F::ZERO],
            // Sandwich-norm LayerNorms (q_norm, k_norm, attn_out_norm).
            q_norm_gamma: vec![F::from(2u64); D],
            q_norm_beta: vec![F::from(5u64); D],
            k_norm_gamma: vec![F::from(2u64); D],
            k_norm_beta: vec![F::from(5u64); D],
            attn_out_norm_gamma: vec![F::from(2u64); D],
            attn_out_norm_beta: vec![F::from(5u64); D],
            ln2_gamma: vec![F::from(2u64); D],
            ln2_beta: vec![F::from(5u64); D],
            ffn_w1,
            ffn_w2,
            ffn_activation_tables: vec![identity_table.clone()],
            ffn_activation_bits_per_chunk: M_BITS,
            ffn_activation_quant: QuantizationParams {
                scale_num: 2,
                scale_den: 2,
            },
            q_activation_tables: vec![identity_table.clone()],
            k_activation_tables: vec![identity_table.clone()],
            qk_activation_bits_per_chunk: M_BITS,
            qk_activation_quant: QuantizationParams {
                scale_num: 2,
                scale_den: 2,
            },
        };

        TransformerModelWeights {
            num_blocks: 1,
            d_model: D,
            d_ff: D_FF,
            vocab_size: V,
            causal: false,
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

    /// LN2 takes x_mid = x_in + aon.y = [[14+7, 10+3], [20+7, 16+3]]
    /// = [[21, 13], [27, 19]].  With γ=2, β=5: y = [[7, 3], [7, 3]], σ = [5, 5].
    fn build_ln2_witness() -> LayerNormWitness {
        LayerNormWitness {
            x: vec![
                vec![F::from(21u64), F::from(13u64)],
                vec![F::from(27u64), F::from(19u64)],
            ],
            y: vec![
                vec![F::from(7u64), F::from(3u64)],
                vec![F::from(7u64), F::from(3u64)],
            ],
            // sum_x: [21+13, 27+19] = [34, 46]
            sum_x: vec![F::from(34u64), F::from(46u64)],
            sigma: vec![F::from(5u64), F::from(5u64)],
            // sq_sum_x: [21²+13², 27²+19²] = [610, 1090]
            sq_sum_x: vec![F::from(610u64), F::from(1090u64)],
            // sum_x_sq: [34², 46²] = [1156, 2116]
            sum_x_sq: vec![F::from(1156u64), F::from(2116u64)],
            // (d·σ)² = 10² = 100
            sigma_sq_scaled: vec![F::from(100u64), F::from(100u64)],
        }
    }

    /// final_ln takes x_out = x_mid + ffn.y = [[21+7, 13+0], [27+7, 19+0]]
    /// = [[28, 13], [34, 19]].  With γ=2, β=5: y = [[7, 4], [7, 4]], σ = [10, 10].
    fn build_ln_final_witness() -> LayerNormWitness {
        LayerNormWitness {
            x: vec![
                vec![F::from(28u64), F::from(13u64)],
                vec![F::from(34u64), F::from(19u64)],
            ],
            y: vec![
                vec![F::from(7u64), F::from(4u64)],
                vec![F::from(7u64), F::from(4u64)],
            ],
            // sum_x: [28+13, 34+19] = [41, 53]
            sum_x: vec![F::from(41u64), F::from(53u64)],
            sigma: vec![F::from(10u64), F::from(10u64)],
            // sq_sum_x: [28²+13², 34²+19²] = [953, 1517]
            sq_sum_x: vec![F::from(953u64), F::from(1517u64)],
            // sum_x_sq: [41², 53²] = [1681, 2809]
            sum_x_sq: vec![F::from(1681u64), F::from(2809u64)],
            // (d·σ)² = 20² = 400
            sigma_sq_scaled: vec![F::from(400u64), F::from(400u64)],
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
        // q_proj_wit.y = q_raw (post-W_q · ln1_y, pre-norm).
        let qk_proj_out = vec![
            vec![F::from(7u64), F::from(0u64)],
            vec![F::from(7u64), F::from(0u64)],
        ];
        // q_norm_wit.y = q_n (post-norm; the input to φ).  Computed with
        // gamma=2, beta=5, sigma=4, scale_gamma=scale_beta=1, d=2 over x=qk_proj_out.
        let qk_norm_out = vec![
            vec![F::from(7u64), F::from(3u64)],
            vec![F::from(7u64), F::from(3u64)],
        ];
        let zero_out = vec![
            vec![F::from(0u64), F::from(0u64)],
            vec![F::from(0u64), F::from(0u64)],
        ];
        let q_proj_wit = ProjectionWitness {
            x: y_norm1.clone(),
            y: qk_proj_out.clone(),
        };
        let k_proj_wit = ProjectionWitness {
            x: y_norm1.clone(),
            y: qk_proj_out.clone(),
        };
        let v_proj_wit = ProjectionWitness {
            x: y_norm1.clone(),
            y: zero_out.clone(),
        };
        let attn_wit = LinearAttentionWitness {
            q: qk_norm_out.clone(),
            k: qk_norm_out.clone(),
            v: zero_out.clone(),
            phi_q: qk_norm_out.clone(),
            phi_k: qk_norm_out.clone(),
            q_query_indices: vec![7, 3, 7, 3],
            k_query_indices: vec![7, 3, 7, 3],
            context: zero_out.clone(),
            causal_context: None,
            normalized_out: None,
            norm_z: None,
            norm_rem: None,
            norm_diff: None,
            out: zero_out.clone(),
        };
        // o_bias = [2, 0], W_o = identity-like with attn_proj_in zero, so
        // o_proj.y = bias broadcast across rows = [[2, 0], [2, 0]].
        let aon_raw = vec![
            vec![F::from(2u64), F::from(0u64)],
            vec![F::from(2u64), F::from(0u64)],
        ];
        let o_proj_wit = ProjectionWitness {
            x: zero_out.clone(),
            y: aon_raw.clone(),
        };
        // x_mid = x_in + attn_out_norm.y = [[14+7, 10+3], [20+7, 16+3]]
        //                                = [[21, 13], [27, 19]].
        let x_mid = vec![
            vec![F::from(21u64), F::from(13u64)],
            vec![F::from(27u64), F::from(19u64)],
        ];
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
        // x_out = x_mid + ffn.y = [[21+7, 13+0], [27+7, 19+0]] = [[28, 13], [34, 19]]
        let x_out = vec![
            vec![F::from(28u64), F::from(13u64)],
            vec![F::from(34u64), F::from(19u64)],
        ];
        // Sandwich-norm LN witnesses: x = [[2, 0], [2, 0]] for attn_out_norm,
        // x = [[7, 0], [7, 0]] for q/k_norm.  Both produce y = [[7, 3], [7, 3]]
        // but with different (sigma, sum_x, sq_sum_x, sum_x_sq) statistics.
        let make_qk_norm_wit = || LayerNormWitness {
            x: qk_proj_out.clone(),
            y: qk_norm_out.clone(),
            sum_x: vec![F::from(7u64), F::from(7u64)],
            sigma: vec![F::from(4u64), F::from(4u64)],
            sq_sum_x: vec![F::from(49u64), F::from(49u64)],
            sum_x_sq: vec![F::from(49u64), F::from(49u64)],
            // sigma_sq_scaled[i] = (d * sigma[i])^2 = (2 * 4)^2 = 64.
            sigma_sq_scaled: vec![F::from(64u64), F::from(64u64)],
        };
        // attn_out_norm: same (γ, β) and same y as q/k_norm but x = [[2, 0], [2, 0]],
        // so smaller σ.  vi = d·(d·sq_sum - sum²) = 2·(8 - 4) = 8 = (2·1)² ≤ 8 ≤ (2·1+2)²-1 = 15.
        let attn_out_norm_wit = LayerNormWitness {
            x: aon_raw.clone(),
            y: qk_norm_out.clone(),
            sum_x: vec![F::from(2u64), F::from(2u64)],
            sigma: vec![F::from(1u64), F::from(1u64)],
            sq_sum_x: vec![F::from(4u64), F::from(4u64)],
            sum_x_sq: vec![F::from(4u64), F::from(4u64)],
            sigma_sq_scaled: vec![F::from(4u64), F::from(4u64)],
        };
        let q_norm_wit = make_qk_norm_wit();
        let k_norm_wit = make_qk_norm_wit();
        let witness = TransformerBlockWitness {
            x_in: x_in.clone(),
            ln1_wit,
            q_proj_wit,
            k_proj_wit,
            v_proj_wit,
            q_norm_wit,
            k_norm_wit,
            attn_wit,
            o_proj_wit,
            attn_out_norm_wit,
            x_mid,
            ln2_wit,
            ffn_wit,
            x_out,
        };
        let (q_lasso, q_query_indices) = build_lasso(vec![7, 3, 7, 3], vec![7, 3, 7, 3]);
        let (k_lasso, k_query_indices) = build_lasso(vec![7, 3, 7, 3], vec![7, 3, 7, 3]);
        let inst_attn = LinearAttentionInstance {
            seq_len: T,
            d_head: D,
            causal: false,
            q_lasso,
            k_lasso,
            q_query_indices,
            k_query_indices,
        };
        let (ffn_lasso, _) =
            build_lasso(vec![7, 0, 0, 0, 7, 0, 0, 0], vec![7, 0, 0, 0, 7, 0, 0, 0]);
        let inst_ffn = FFNInstance {
            activation_lasso: ffn_lasso,
        };
        (witness, inst_attn, inst_ffn)
    }

    fn build_model_witness(block_wit: TransformerBlockWitness) -> TransformerModelWitness {
        let x_in = block_wit.x_in.clone();
        let final_ln_wit = build_ln_final_witness();
        let y_final = final_ln_wit.y.clone();
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

    fn assert_tampered_model_rejected(
        label: &'static [u8],
        tamper: impl FnOnce(&mut TransformerModelProof, &mut TransformerModelVerifyingKey),
    ) {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(label);
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        let mut vk = pk.vk.clone();
        tamper(&mut proof, &mut vk);

        let mut vt = Transcript::new(label);
        let result = verify(
            &proof,
            &vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(result.is_err(), "tampered proof should be rejected");
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
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(
            result.is_ok(),
            "Model verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_prove_verify_full_model_causal_e2e() {
        let (mut block_wit, mut inst_attn, inst_ffn) = build_block_witness_and_instances();
        inst_attn.causal = true;
        block_wit.attn_wit.causal_context = None;
        let model_wit = build_model_witness(block_wit);
        let mut weights = build_test_weights();
        weights.causal = true;
        let pk = preprocess_transformer_model(weights, T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_causal_e2e");
        let proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        // GKR chain (step 2): causal_context_com and both causal_ctx opens are
        // dropped — the CTX cubic sumcheck binds the OUT terminal claim
        // algebraically through phi_k and v.
        assert!(proof.block_proofs[0].causal_context_com.is_none());
        assert!(proof.causal_ctx_out_batch_open.is_none());
        assert!(proof.causal_ctx_prefix_batch_open.is_none());

        let mut vt = Transcript::new(b"model_causal_e2e");
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(
            result.is_ok(),
            "Causal model verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_model_rejects_tampered_block_ln1_proof() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_ln1");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        // PROOF_VERSION 13: per-LN proofs are bundled. Tamper the first LN's
        // sum_x_at_rt opening within the batched proof's first group.
        proof.ln_batched_proof.groups[0].openings.sum_x_at_rt[0] += F::ONE;

        let mut vt = Transcript::new(b"model_tamper_ln1");
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
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
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
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
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
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
        // PROOF_VERSION 13: tamper the FINAL LN inside the batched proof.
        // The final LN sits at index `5*num_blocks` in input order (single block
        // here -> 5 LNs of d_model + 1 final). All d_model-shaped LNs land in
        // the same group; the final-LN slot within that group is the last one.
        let g0 = &mut proof.ln_batched_proof.groups[0];
        let last = g0.openings.sum_x_at_rt.len() - 1;
        g0.openings.sum_x_at_rt[last] += F::ONE;

        let mut vt = Transcript::new(b"model_tamper_final_ln");
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
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
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
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
            &vec![
                vec![F::from(1u64), F::from(1u64)],
                vec![F::from(1u64), F::from(1u64)],
            ],
            T,
            D,
        );

        let mut vt = Transcript::new(b"model_tamper_xin");
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(result.is_err(), "Should reject tampered x_in_com");
    }

    #[test]
    fn test_model_rejects_tampered_lasso_indices() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_lasso_indices");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.ffn_lasso_proof.all_query_indices[0][0] += 1;

        let mut vt = Transcript::new(b"model_tamper_lasso_indices");
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(
            result.is_err(),
            "Should reject tampered FFN Lasso committed indices"
        );

        let mut pt = Transcript::new(b"model_tamper_global_lasso_indices");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.all_lasso_proof.all_query_indices[0][0] += 1;

        let mut vt = Transcript::new(b"model_tamper_global_lasso_indices");
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(
            result.is_err(),
            "Should reject tampered attention Lasso query indices"
        );
    }

    #[test]
    fn test_model_rejects_tampered_ffn_quant_remainder_eval() {
        assert_tampered_model_rejected(b"model_tamper_ffn_quant_rem_eval", |proof, _| {
            proof.ffn_quant_proof.rem_evals[0] += F::ONE;
        });
    }

    #[test]
    fn test_model_rejects_tampered_ffn_quant_raw_eval() {
        assert_tampered_model_rejected(b"model_tamper_ffn_quant_raw_eval", |proof, _| {
            proof.ffn_quant_proof.raw_evals[0] += F::ONE;
        });
    }

    #[test]
    fn test_model_rejects_tampered_ffn_quant_range_claim() {
        // PROOF_VERSION 14: claim_v / Phase-2 sumcheck were removed (they reduced
        // a vacuous claim about an uncommitted v).  The equivalent surface to
        // tamper with is the chunk evaluation at r_v, which is bound to chunk_coms
        // via the deferred Hyrax batch open and must therefore be rejected.
        assert_tampered_model_rejected(b"model_tamper_ffn_quant_range_claim", |proof, _| {
            proof.ffn_quant_proof.rem_range_proofs[0].chunk_evals[0] += F::ONE;
        });
    }

    #[test]
    fn test_model_rejects_tampered_ffn_quant_remainder_commitment() {
        assert_tampered_model_rejected(b"model_tamper_ffn_quant_rem_com", |proof, _| {
            proof.ffn_quant_proof.rem_coms[0] = proof.block_proofs[0].ffn_m_com.clone();
        });
    }

    #[test]
    fn test_model_rejects_tampered_qk_quant_remainder_eval() {
        assert_tampered_model_rejected(b"model_tamper_qk_quant_rem_eval", |proof, _| {
            proof.qk_quant_proof.rem_evals[0] += F::ONE;
        });
    }

    #[test]
    fn test_model_rejects_tampered_qk_quant_raw_eval() {
        assert_tampered_model_rejected(b"model_tamper_qk_quant_raw_eval", |proof, _| {
            proof.qk_quant_proof.raw_evals[0] += F::ONE;
        });
    }

    #[test]
    fn test_model_rejects_tampered_qk_quant_range_claim() {
        // See note on test_model_rejects_tampered_ffn_quant_range_claim.
        assert_tampered_model_rejected(b"model_tamper_qk_quant_range_claim", |proof, _| {
            proof.qk_quant_proof.rem_range_proofs[0].chunk_evals[0] += F::ONE;
        });
    }

    #[test]
    fn test_model_rejects_tampered_qk_quant_remainder_commitment() {
        assert_tampered_model_rejected(b"model_tamper_qk_quant_rem_com", |proof, _| {
            proof.qk_quant_proof.rem_coms[0] = proof.block_proofs[0].q_com.clone();
        });
    }

    #[test]
    fn test_model_rejects_ffn_quant_scale_mismatch() {
        assert_tampered_model_rejected(b"model_tamper_ffn_quant_scale", |_, vk| {
            vk.block_vks[0].ffn_activation_quant.scale_den += 1;
        });
    }

    #[test]
    fn test_model_rejects_qk_quant_scale_mismatch() {
        assert_tampered_model_rejected(b"model_tamper_qk_quant_scale", |_, vk| {
            vk.block_vks[0].qk_activation_quant.scale_den += 1;
        });
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
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(
            result.is_err(),
            "Should reject tampered batch_qkv final_evals_f"
        );
    }

    #[test]
    fn test_model_rejects_tampered_ffn_a_com() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_ffn_a_com");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.block_proofs[0].ffn_a_com =
            commit_mat_test(&vec![vec![F::from(3u64); D_FF]; T], T, D_FF);

        let mut vt = Transcript::new(b"model_tamper_ffn_a_com");
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(result.is_err(), "Should reject tampered FFN A commitment");
    }

    #[test]
    fn test_model_rejects_tampered_ffn_lasso_output_sumcheck() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_ffn_lasso_output_sc");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof
            .ffn_lasso_proof
            .output_sumcheck
            .as_mut()
            .expect("FFN Lasso must use committed outputs")
            .final_evals_f[0] += F::ONE;

        let mut vt = Transcript::new(b"model_tamper_ffn_lasso_output_sc");
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(
            result.is_err(),
            "Should reject tampered FFN Lasso committed-output sumcheck"
        );
    }

    #[test]
    fn test_model_rejects_missing_ffn_lasso_output_binding() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_missing_ffn_lasso_output_binding");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.ffn_lasso_proof.output_sumcheck = None;
        proof.ffn_lasso_proof.output_batch_open = None;

        let mut vt = Transcript::new(b"model_missing_ffn_lasso_output_binding");
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(
            result.is_err(),
            "Should reject missing FFN Lasso committed-output binding"
        );
    }

    #[test]
    fn test_model_rejects_tampered_batch_ffn_y_a_eval() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T, &lasso_params());
        let lp = lasso_params();

        let mut pt = Transcript::new(b"model_tamper_ffn_y_a_eval");
        let mut proof = prove(&pk, &model_wit, &inst_attn, &inst_ffn, &mut pt, &lp).unwrap();
        proof.batch_ffn_y.final_evals_f[0] += F::ONE;

        let mut vt = Transcript::new(b"model_tamper_ffn_y_a_eval");
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(result.is_err(), "Should reject tampered FFN-Y A evaluation");
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
        let result = verify(
            &proof,
            &pk.vk,
            &inst_attn,
            &inst_ffn,
            &model_wit.x_in,
            &model_wit.lm_head_wit.y,
            &mut vt,
            &lp,
        );
        assert!(
            result.is_err(),
            "Should reject fraudulent LN1 output commitment"
        );
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
        assert_eq!(
            com_sum.row_coms, com_apb.row_coms,
            "Com(a) + Com(b) must equal Com(a+b)"
        );
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
        assert_eq!(
            com_sum.row_coms, com_a.row_coms,
            "Com(a) + Com(0) must equal Com(a)"
        );
    }
}
