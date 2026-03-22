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
use crate::pcs::{hyrax_commit, HyraxCommitment, HyraxParams};
use crate::poly::utils::mat_to_mle;
use crate::poly::DenseMLPoly;
use crate::transcript::Transcript;

// Sub-module imports (Assuming the interfaces we built previously)
use crate::attention::layernorm::{
    prove_layernorm, LayerNormIOCommitments, LayerNormProof, LayerNormWitness,
};
use crate::attention::linear::{
    prove_linear_attention, AttentionIOCommitments, LinearAttentionInstance, LinearAttentionProof,
    LinearAttentionWitness,
};
use crate::attention::projection::{
    prove_projection, ProjectionIOCommitments, ProjectionProof, ProjectionWitness,
};
use crate::ffn::ffn::{prove_ffn, FFNIOCommitments, FFNInstance, FFNProof, FFNWitness};
use crate::verifier::{add_commitments, TransformerBlockVerifyingKey}; // Imported from verifier.rs

// ---------------------------------------------------------------------------
// Global Proof Structure
// ---------------------------------------------------------------------------

/// The complete ZK Proof for one Transformer Block.
pub struct TransformerBlockProof {
    // Sub-proofs
    pub ln1_proof: LayerNormProof,
    pub q_proj_proof: ProjectionProof,
    pub k_proj_proof: ProjectionProof,
    pub v_proj_proof: ProjectionProof,
    pub attn_proof: LinearAttentionProof,
    pub o_proj_proof: ProjectionProof,
    pub ln2_proof: LayerNormProof,
    pub ffn_proof: FFNProof,

    // Intermediate IO Commitments (Passed to Verifier to stitch the pipeline)
    pub x_norm1_com: HyraxCommitment,
    pub q_com: HyraxCommitment,
    pub k_com: HyraxCommitment,
    pub v_com: HyraxCommitment,
    pub out_inner_com: HyraxCommitment,
    pub out_attn_com: HyraxCommitment,
    pub x_norm2_com: HyraxCommitment,
    pub out_ffn_com: HyraxCommitment,
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

    // 1. Generate all Intermediate IO Commitments (O(N) operations done ONLY ONCE)
    let x_norm1_com = commit_mat(&witness.ln1_wit.y, t, d);
    let q_com = commit_mat(&witness.attn_wit.q, t, d);
    let k_com = commit_mat(&witness.attn_wit.k, t, d);
    let v_com = commit_mat(&witness.attn_wit.v, t, d);
    let out_inner_com = commit_mat(&witness.attn_wit.out, t, d);
    let out_attn_com = commit_mat(&witness.o_proj_wit.y, t, d);
    let x_norm2_com = commit_mat(&witness.ln2_wit.y, t, d);
    let out_ffn_com = commit_mat(&witness.ffn_wit.y, t, d);

    // 2. Execute Sub-Provers with strictly bound IO Commitments
    // --- LayerNorm 1 ---
    let ln1_io = LayerNormIOCommitments {
        x_com: x_in_com.clone(),
        y_com: x_norm1_com.clone(),
    };
    let ln1_proof = prove_layernorm(&witness.ln1_wit, &ln1_io, &pk.ln1_vk, transcript)?;

    // --- Q, K, V Projections ---
    let q_io = ProjectionIOCommitments {
        x_com: x_norm1_com.clone(),
        y_com: q_com.clone(),
    };
    let q_proj_proof = prove_projection(&pk.q_pk, &witness.q_proj_wit, &q_io, transcript)?;

    let k_io = ProjectionIOCommitments {
        x_com: x_norm1_com.clone(),
        y_com: k_com.clone(),
    };
    let k_proj_proof = prove_projection(&pk.k_pk, &witness.k_proj_wit, &k_io, transcript)?;

    let v_io = ProjectionIOCommitments {
        x_com: x_norm1_com.clone(),
        y_com: v_com.clone(),
    };
    let v_proj_proof = prove_projection(&pk.v_pk, &witness.v_proj_wit, &v_io, transcript)?;

    // --- Linear Attention ---
    let attn_io = AttentionIOCommitments {
        q_com: q_com.clone(),
        k_com: k_com.clone(),
        v_com: v_com.clone(),
        out_com: out_inner_com.clone(),
    };
    let attn_proof = prove_linear_attention(
        &witness.attn_wit,
        inst_attn,
        &attn_io,
        transcript,
        lasso_params,
    );

    // --- Output Projection ---
    let o_io = ProjectionIOCommitments {
        x_com: out_inner_com.clone(),
        y_com: out_attn_com.clone(),
    };
    let o_proj_proof = prove_projection(&pk.o_pk, &witness.o_proj_wit, &o_io, transcript)?;

    // --- LayerNorm 2 ---
    // Note: The input to LN2 is X_mid = X_in + Out_attn.
    // We compute the commitment homomorphically!
    let x_mid_com = add_commitments(x_in_com, &out_attn_com);
    let ln2_io = LayerNormIOCommitments {
        x_com: x_mid_com.clone(),
        y_com: x_norm2_com.clone(),
    };
    let ln2_proof = prove_layernorm(&witness.ln2_wit, &ln2_io, &pk.ln2_vk, transcript)?;

    // --- FFN ---
    let ffn_io = FFNIOCommitments {
        x_com: x_norm2_com.clone(),
        y_com: out_ffn_com.clone(),
    };
    let ffn_proof = prove_ffn(
        &pk.ffn_pk,
        &witness.ffn_wit,
        inst_ffn,
        &ffn_io,
        transcript,
        lasso_params,
    )?;

    // Return the bundled proof and intermediate commitments
    Ok(TransformerBlockProof {
        ln1_proof,
        q_proj_proof,
        k_proj_proof,
        v_proj_proof,
        attn_proof,
        o_proj_proof,
        ln2_proof,
        ffn_proof,
        x_norm1_com,
        q_com,
        k_com,
        v_com,
        out_inner_com,
        out_attn_com,
        x_norm2_com,
        out_ffn_com,
    })
}
