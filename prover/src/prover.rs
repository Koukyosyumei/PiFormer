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
use crate::pcs::{absorb_com, hyrax_commit, HyraxCommitment, HyraxParams};
use crate::poly::utils::mat_to_mle;
use crate::transcript::Transcript;

// Sub-module imports (Assuming the interfaces we built previously)
use crate::attention::attention::{
    prove_linear_attention, AttentionIOCommitments, LinearAttentionInstance, LinearAttentionProof,
    LinearAttentionWitness,
};
use crate::attention::layernorm::{
    prove_layernorm, LayerNormIOCommitments, LayerNormProof, LayerNormVerifyingKey,
    LayerNormWitness,
};
use crate::attention::projection::{
    prove_projection, ProjectionIOCommitments, ProjectionProof, ProjectionProvingKey,
    ProjectionVerifyingKey, ProjectionWitness,
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
    pub x_in_com: HyraxCommitment, // コミットされた入力
    pub block_proofs: Vec<TransformerBlockProof>,
    pub final_ln_proof: LayerNormProof,
    pub lm_head_proof: ProjectionProof,

    // パイプライン接続用の中間コミットメント
    pub final_ln_out_com: HyraxCommitment,
    pub logits_com: HyraxCommitment, // 最終出力（Logits）のコミットメント
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

    // 3. Final LayerNorm
    let final_ln_out_com = commit_mat(&witness.final_ln_wit.y, t, d);
    let ln_io = LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: final_ln_out_com.clone(),
    };
    let final_ln_proof = prove_layernorm(
        &witness.final_ln_wit,
        &ln_io,
        &pk.vk.final_ln_vk,
        transcript,
    )?;

    // 4. LM Head (Final Projection to Vocab Size)
    let logits_com = commit_mat(&witness.lm_head_wit.y, t, v);
    let lm_io = ProjectionIOCommitments {
        x_com: final_ln_out_com.clone(),
        y_com: logits_com.clone(),
    };
    let lm_head_proof = prove_projection(&pk.lm_head_pk, &witness.lm_head_wit, &lm_io, transcript)?;

    Ok(TransformerModelProof {
        x_in_com,
        block_proofs,
        final_ln_proof,
        lm_head_proof,
        final_ln_out_com,
        logits_com,
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
        let pk = preprocess_transformer_model(build_test_weights(), T);
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
        let result = verify_transformer_block(
            &proof,
            &x_in_com,
            &x_out_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut vt,
            &lp,
        );
        assert!(
            result.is_ok(),
            "Block verification failed: {:?}",
            result.err()
        );
    }

    /// Passing the wrong x_out_com must trigger the residual-connection binding check.
    #[test]
    fn test_block_rejects_wrong_x_out_com() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T);
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
        let result = verify_transformer_block(
            &proof,
            &x_in_com,
            &wrong_x_out_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut vt,
            &lp,
        );
        assert!(result.is_err(), "Should reject wrong x_out_com");
    }

    /// Tampering with the LN1 proof must be detected by the sub-verifier.
    #[test]
    fn test_block_rejects_tampered_ln1_opening() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T);
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
        let result = verify_transformer_block(
            &proof,
            &x_in_com,
            &x_out_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut vt,
            &lp,
        );
        assert!(result.is_err(), "Should reject tampered LN1 proof");
    }

    /// Tampering with an intermediate commitment breaks the pipeline binding.
    #[test]
    fn test_block_rejects_tampered_x_norm1_com() {
        let (witness, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let pk = preprocess_transformer_model(build_test_weights(), T);
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
        let result = verify_transformer_block(
            &proof,
            &x_in_com,
            &x_out_com,
            &pk.block_pks[0],
            &inst_attn,
            &inst_ffn,
            &mut vt,
            &lp,
        );
        assert!(result.is_err(), "Should reject tampered x_norm1_com");
    }

    // -----------------------------------------------------------------------
    // prove (model-level) / verify (model-level) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prove_verify_full_model_e2e() {
        let (block_wit, inst_attn, inst_ffn) = build_block_witness_and_instances();
        let model_wit = build_model_witness(block_wit);
        let pk = preprocess_transformer_model(build_test_weights(), T);
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
        let pk = preprocess_transformer_model(build_test_weights(), T);
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
        let pk = preprocess_transformer_model(build_test_weights(), T);
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
        let pk = preprocess_transformer_model(build_test_weights(), T);
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
        let pk = preprocess_transformer_model(build_test_weights(), T);
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
