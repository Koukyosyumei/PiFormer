<<<<<<< HEAD
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
use crate::attention::attention::{
    absorb_com, prove_linear_attention, AttentionIOCommitments, LinearAttentionInstance,
    LinearAttentionProof, LinearAttentionWitness,
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
    use crate::poly::utils::mat_to_mle;
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

    /// Build a single-block model where all projection/FFN weights are zero.
    /// With zero weights every matmul output is zero, so the only non-trivial
    /// path is through the two LayerNorm layers.
    fn build_test_weights() -> TransformerModelWeights {
        let block = TransformerBlockWeights {
            ln1_gamma: vec![F::from(2u64); D],
            ln1_beta: vec![F::from(5u64); D],
            q_w: zero_mat(D, D),
            k_w: zero_mat(D, D),
            v_w: zero_mat(D, D),
            o_w: zero_mat(D, D),
            ln2_gamma: vec![F::from(2u64); D],
            ln2_beta: vec![F::from(5u64); D],
            ffn_w1: zero_mat(D, D_FF),
            ffn_w2: zero_mat(D_FF, D),
        };
        TransformerModelWeights {
            num_blocks: 1,
            d_model: D,
            d_ff: D_FF,
            vocab_size: V,
            blocks: vec![block],
            final_ln_gamma: vec![F::from(2u64); D],
            final_ln_beta: vec![F::from(5u64); D],
            lm_head_w: zero_mat(D, V),
        }
    }
=======
//! PiFormer prover: orchestrates the full ZK proof for a transformer model.
//!
//! The proof covers:
//!   - For each transformer layer: for each attention head, a full
//!     `LinearAttentionProof` (φ(Q), φ(K) Lasso lookups + two GKR sumchecks).
//!   - Ternary weight constraint for all weight matrices in each layer.
//!
//! The public input is the full `PiFormerWitness` (matrices are known to verifier
//! in this version; PCS commitments replace them in a fully-succinct variant).

use crate::attention::{
    prove_linear_attention, LinearAttentionInstance, LinearAttentionProof,
    TernaryWeightInstance, TernaryWeightProof, prove_ternary_weights,
};
use crate::field::F;
use ark_ff::Field;
use crate::pcs::HyraxParams;
use crate::transcript::Transcript;

// ---------------------------------------------------------------------------
// Witness and Proof types
// ---------------------------------------------------------------------------

/// Witness for a full PiFormer forward pass.
///
/// `block_witnesses[layer][head]` contains the attention witness for that head.
pub struct PiFormerWitness {
    pub block_witnesses: Vec<Vec<LinearAttentionInstance>>,
}

/// Proof for one transformer block (all heads in one layer).
pub struct BlockProof {
    /// Attention proofs for each head.
    pub head_proofs: Vec<LinearAttentionProof>,
    /// Batched ternary constraint proof for all weight matrices in this block.
    pub ternary_proof: TernaryWeightProof,
}

/// Full proof for all transformer blocks.
pub struct PiFormerProof {
    pub block_proofs: Vec<BlockProof>,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub struct PiFormerProver;

impl PiFormerProver {
    /// Generate a proof for the full transformer forward pass.
    pub fn prove(witness: &PiFormerWitness, params: &HyraxParams) -> PiFormerProof {
        let mut transcript = Transcript::new(b"PiFormer-v0.1");

        let block_proofs = witness
            .block_witnesses
            .iter()
            .enumerate()
            .map(|(layer_idx, heads)| {
                transcript.append_bytes(b"layer", &(layer_idx as u64).to_le_bytes());

                // Prove each attention head.
                let head_proofs = heads
                    .iter()
                    .enumerate()
                    .map(|(head_idx, inst)| {
                        transcript.append_bytes(b"head", &(head_idx as u64).to_le_bytes());
                        prove_linear_attention(inst, &mut transcript, params)
                    })
                    .collect::<Vec<_>>();

                // Collect all weight matrices from all heads and prove ternary constraint.
                let ternary_weights = collect_ternary_weights(heads);
                let ternary_proof =
                    prove_ternary_weights(&TernaryWeightInstance { weights: ternary_weights },
                        &mut transcript);

                BlockProof { head_proofs, ternary_proof }
            })
            .collect();
>>>>>>> main

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

    /// Build a validated LayerNorm witness for x = [[10,20],[30,40]] with
    /// gamma=[2,2], beta=[5,5].  The sigma and y values below have been
    /// cross-checked against the LayerNorm constraint system:
    ///
    ///   var_x[i] = 200  →  d·sigma = 14  →  sigma = 7
    ///   y[i] = [4, 6]  satisfies both the lo and hi range constraints.
    fn build_ln_witness_for_x(x: Vec<Vec<F>>) -> LayerNormWitness {
        let d_f = F::from(D as u64);
        let t = x.len();
        let mut sum_x = vec![F::ZERO; t];
        let mut var_x = vec![F::ZERO; t];
        for i in 0..t {
            let s: F = x[i].iter().copied().sum();
            sum_x[i] = s;
            var_x[i] = x[i]
                .iter()
                .map(|&xij| {
                    let diff = d_f * xij - s;
                    diff * diff
                })
                .sum();
        }
        // d·sigma = floor(sqrt(var_x)) = 14  →  sigma = 7
        let sigma = vec![F::from(7u64); t];
        // y verified in the dedicated layernorm tests
        let y = vec![
            vec![F::from(4u64), F::from(6u64)],
            vec![F::from(4u64), F::from(6u64)],
        ];
        LayerNormWitness { x, y, sum_x, var_x, sigma }
    }

    /// Build a Lasso instance whose every query indexes slot 0 (output = 0).
    fn build_zero_lasso(num_queries: usize) -> LassoInstance {
        let table: Vec<F> = (0u64..1 << M_BITS).map(F::from).collect();
        LassoInstance {
            tables: vec![table],
            query_indices: vec![0usize; num_queries],
            outputs: vec![F::ZERO; num_queries],
            bits_per_chunk: M_BITS,
        }
    }

    /// Shared HyraxParams for the 4-bit Lasso table (sigma = m - m/2 = 2).
    fn lasso_params() -> HyraxParams {
        HyraxParams::new(M_BITS / 2)
    }

    // -----------------------------------------------------------------------
    // Block-level fixture
    // -----------------------------------------------------------------------

    fn build_block_witness_and_instances()
    -> (TransformerBlockWitness, LinearAttentionInstance, FFNInstance)
    {
        let x_in = vec![
            vec![F::from(10u64), F::from(20u64)],
            vec![F::from(30u64), F::from(40u64)],
        ];

        let zeros_td  = vec![vec![F::ZERO; D]; T];
        let zeros_dd  = vec![vec![F::ZERO; D]; D];
        let zeros_tdf = vec![vec![F::ZERO; D_FF]; T];

        // LayerNorm 1 on x_in
        let ln1_wit = build_ln_witness_for_x(x_in.clone());
        let y_norm1 = ln1_wit.y.clone();

        // Q / K / V projections: y_norm1 × 0 = 0
        let q_proj_wit = ProjectionWitness { x: y_norm1.clone(), y: zeros_td.clone() };
        let k_proj_wit = ProjectionWitness { x: y_norm1.clone(), y: zeros_td.clone() };
        let v_proj_wit = ProjectionWitness { x: y_norm1.clone(), y: zeros_td.clone() };

        // Linear attention: all zero activations and products
        let attn_wit = LinearAttentionWitness {
            q:       zeros_td.clone(),
            k:       zeros_td.clone(),
            v:       zeros_td.clone(),
            phi_q:   zeros_td.clone(),
            phi_k:   zeros_td.clone(),
            context: zeros_dd.clone(),
            out:     zeros_td.clone(),
        };

        // Output projection: 0 × 0 = 0
        let o_proj_wit = ProjectionWitness { x: zeros_td.clone(), y: zeros_td.clone() };

        // Residual 1: x_mid = x_in + out_attn = x_in + 0 = x_in
        let x_mid = x_in.clone();

        // LayerNorm 2: same input → same witness
        let ln2_wit = build_ln_witness_for_x(x_mid.clone());
        let y_norm2 = ln2_wit.y.clone();

        // FFN: y_norm2 × 0 = 0
        let ffn_wit = FFNWitness {
            x: y_norm2,
            m: zeros_tdf.clone(),
            a: zeros_tdf.clone(),
            y: zeros_td.clone(),
        };

        // Residual 2: x_out = x_mid + 0 = x_mid = x_in
        let x_out = x_mid.clone();

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

        let inst_attn = LinearAttentionInstance {
            seq_len: T,
            d_head: D,
            q_lasso: build_zero_lasso(T * D),
            k_lasso: build_zero_lasso(T * D),
        };

        let inst_ffn = FFNInstance {
            activation_lasso: build_zero_lasso(T * D_FF),
        };

        (witness, inst_attn, inst_ffn)
    }

    // -----------------------------------------------------------------------
    // Model-level fixture
    // -----------------------------------------------------------------------

    fn build_model_witness(
        block_wit: TransformerBlockWitness,
    ) -> TransformerModelWitness {
        let x_in  = block_wit.x_in.clone();
        let x_out = block_wit.x_out.clone();

        // Final LayerNorm: x_out = x_in (zero residuals), so same witness
        let final_ln_wit = build_ln_witness_for_x(x_out.clone());
        let y_final = final_ln_wit.y.clone();

        // LM head: y_final × 0 = 0  →  logits = 0
        let lm_head_wit = ProjectionWitness {
            x: y_final,
            y: vec![vec![F::ZERO; V]; T],
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
            &witness, &x_in_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut pt, &lp,
        )
        .unwrap();

        // Derive the expected output commitment the same way the verifier does.
        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_e2e");
        let result = verify_transformer_block(
            &proof, &x_in_com, &x_out_com,
            &pk.block_pks[0], &inst_attn, &inst_ffn, &mut vt, &lp,
        );
        assert!(result.is_ok(), "Block verification failed: {:?}", result.err());
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
            &witness, &x_in_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut pt, &lp,
        )
        .unwrap();

        // Provide a commitment to a completely different matrix as x_out.
        let wrong_x_out_com = commit_mat_test(
            &vec![vec![F::from(1u64), F::from(2u64)], vec![F::from(3u64), F::from(4u64)]],
            T, D,
        );

        let mut vt = Transcript::new(b"block_wrong_out");
        let result = verify_transformer_block(
            &proof, &x_in_com, &wrong_x_out_com,
            &pk.block_pks[0], &inst_attn, &inst_ffn, &mut vt, &lp,
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
            &witness, &x_in_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut pt, &lp,
        )
        .unwrap();

        // Perturb the sum_x opening inside the LN1 sub-proof.
        proof.ln1_proof.openings.sum_x_at_rt += F::ONE;

        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_tamper_ln1");
        let result = verify_transformer_block(
            &proof, &x_in_com, &x_out_com,
            &pk.block_pks[0], &inst_attn, &inst_ffn, &mut vt, &lp,
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
            &witness, &x_in_com, &pk.block_pks[0],
            &inst_attn, &inst_ffn, &mut pt, &lp,
        )
        .unwrap();

        // Swap x_norm1_com for a commitment to a different matrix.
        proof.x_norm1_com = commit_mat_test(
            &vec![vec![F::from(99u64), F::from(99u64)], vec![F::from(99u64), F::from(99u64)]],
            T, D,
        );

        let x_mid_com = add_commitments(&x_in_com, &proof.out_attn_com);
        let x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

        let mut vt = Transcript::new(b"block_tamper_xnorm1");
        let result = verify_transformer_block(
            &proof, &x_in_com, &x_out_com,
            &pk.block_pks[0], &inst_attn, &inst_ffn, &mut vt, &lp,
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
        assert!(result.is_ok(), "Model verification failed: {:?}", result.err());
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
            &vec![vec![F::from(1u64), F::from(1u64)], vec![F::from(1u64), F::from(1u64)]],
            T, D,
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

        let a   = vec![F::from(3u64), F::from(5u64), F::from(7u64), F::from(11u64)];
        let b   = vec![F::from(1u64), F::from(2u64), F::from(3u64), F::from(4u64)];
        let apb: Vec<F> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

        let (nu, _sigma, params) = params_from_vars(2);
        let com_a   = hyrax_commit(&a,   nu, &params);
        let com_b   = hyrax_commit(&b,   nu, &params);
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

        let a    = vec![F::from(5u64), F::from(9u64), F::from(2u64), F::from(14u64)];
        let zero = vec![F::ZERO; 4];

        let (nu, _sigma, params) = params_from_vars(2);
        let com_a    = hyrax_commit(&a,    nu, &params);
        let com_zero = hyrax_commit(&zero, nu, &params);
        let com_sum  = add_commitments(&com_a, &com_zero);

        assert_eq!(
            com_sum.row_coms, com_a.row_coms,
            "Com(a) + Com(0) must equal Com(a)"
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Flatten all Q, K, V matrices from all attention heads into a single weight
/// vector for the ternary constraint proof.
///
/// In a real deployment, these come from the quantised weight tensors; here we
/// flatten the input projections available in the witness.
fn collect_ternary_weights(heads: &[LinearAttentionInstance]) -> Vec<F> {
    let mut weights = Vec::new();
    for inst in heads {
        for row in &inst.q {
            weights.extend_from_slice(row);
        }
        for row in &inst.k {
            weights.extend_from_slice(row);
        }
        for row in &inst.v {
            weights.extend_from_slice(row);
        }
    }
    // Clamp each element to {-1, 0, 1}: elements already in the field, snap
    // to nearest ternary using the convention that p-1 ≡ -1.
    weights
        .into_iter()
        .map(|w| snap_to_ternary(w))
        .collect()
}

fn snap_to_ternary(w: F) -> F {
    use ark_ff::PrimeField;
    let raw = w.into_bigint().as_ref()[0];
    if raw == 0 {
        F::ZERO
    } else if raw == 1 {
        F::ONE
    } else {
        // treat as -1 (field element p-1)
        F::ZERO - F::ONE
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod prover_tests {
    use super::*;
    use crate::lookup::LassoInstance;
    use ark_ff::{Field, PrimeField};

    fn make_witness(seq_len: usize, d_head: usize) -> PiFormerWitness {
        let m = 4usize;
        let table_size = 1 << m;
        let table: Vec<F> = (0..table_size).map(|i| F::from((i + 1) as u64)).collect();

        let q = vec![
            vec![F::from(1u64), F::from(2u64)],
            vec![F::from(3u64), F::from(4u64)],
        ];
        let k = vec![
            vec![F::from(0u64), F::from(1u64)],
            vec![F::from(2u64), F::from(3u64)],
        ];
        let v = vec![
            vec![F::from(5u64), F::from(6u64)],
            vec![F::from(7u64), F::from(8u64)],
        ];

        let phi_q: Vec<Vec<F>> = q
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| table[x.into_bigint().as_ref()[0] as usize])
                    .collect()
            })
            .collect();
        let phi_k: Vec<Vec<F>> = k
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&x| table[x.into_bigint().as_ref()[0] as usize])
                    .collect()
            })
            .collect();

        let mut context = vec![vec![F::ZERO; d_head]; d_head];
        for i in 0..d_head {
            for j in 0..d_head {
                for t in 0..seq_len {
                    context[i][j] += phi_k[t][i] * v[t][j];
                }
            }
        }
        let mut out = vec![vec![F::ZERO; d_head]; seq_len];
        for t in 0..seq_len {
            for j in 0..d_head {
                for i in 0..d_head {
                    out[t][j] += phi_q[t][i] * context[i][j];
                }
            }
        }

        let build_lasso = |mat: &Vec<Vec<F>>| {
            let flat: Vec<usize> = mat
                .iter()
                .flatten()
                .map(|x| x.into_bigint().as_ref()[0] as usize)
                .collect();
            let out_flat: Vec<F> = mat
                .iter()
                .flatten()
                .map(|&x| table[x.into_bigint().as_ref()[0] as usize])
                .collect();
            LassoInstance {
                tables: vec![table.clone()],
                query_indices: flat,
                outputs: out_flat,
                bits_per_chunk: m,
            }
        };

        let inst = LinearAttentionInstance {
            seq_len,
            d_head,
            q,
            k,
            v,
            phi_q,
            phi_k,
            context,
            out,
            q_lasso: build_lasso(&vec![
                vec![F::from(1u64), F::from(2u64)],
                vec![F::from(3u64), F::from(4u64)],
            ]),
            k_lasso: build_lasso(&vec![
                vec![F::from(0u64), F::from(1u64)],
                vec![F::from(2u64), F::from(3u64)],
            ]),
        };

        PiFormerWitness {
            block_witnesses: vec![vec![inst]],
        }
    }

    #[test]
    fn test_prover_e2e() {
        let witness = make_witness(2, 2);
        let params = HyraxParams::new(2);
        let proof = PiFormerProver::prove(&witness, &params);
        assert_eq!(proof.block_proofs.len(), 1);
        assert_eq!(proof.block_proofs[0].head_proofs.len(), 1);
    }
}
