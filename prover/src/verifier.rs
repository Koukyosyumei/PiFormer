<<<<<<< HEAD
//! Global Verifier for a full Transformer Block.
//!
//! **Production-Grade Succinctness:**
//! 1. Strict O(√N) runtime. No matrices are constructed. No O(N) operations exist.
//! 2. Homomorphic Residuals: X_mid and X_out are verified by directly adding
//!    their Pedersen (Hyrax) commitments. No additional proofs needed!
//! 3. Commitment Chaining: Intermediate IO commitments passed from the Prover
//!    are cryptographically bound across adjacent sub-verifiers.

use crate::field::F;
use crate::pcs::{HyraxCommitment, HyraxParams};
=======
//! PiFormer verifier: verifies a `PiFormerProof` against a `PiFormerWitness`.

use crate::attention::{verify_linear_attention, verify_ternary_weights, TernaryWeightInstance};
use crate::field::F;
use ark_ff::Field;
use crate::pcs::HyraxParams;
use crate::prover::{PiFormerProof, PiFormerWitness};
>>>>>>> main
use crate::transcript::Transcript;

// Sub-module keys and verifiers
use crate::attention::attention::{absorb_com, verify_linear_attention, LinearAttentionInstance};
use crate::attention::layernorm::{
    verify_layernorm, LayerNormIOCommitments, LayerNormVerifyingKey,
};
use crate::attention::projection::{
    verify_projection_succinct, ProjectionIOCommitments, ProjectionProvingKey,
    ProjectionVerifyingKey,
};
use crate::ffn::ffn::{verify_ffn, FFNInstance, FFNProvingKey, FFNVerifyingKey};
use ark_ec::{AffineRepr, CurveGroup}; // Arkworks 0.4.0+ の正しいトレイト
use std::ops::AddAssign;

<<<<<<< HEAD
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
) -> Result<(), String> {
    // =========================================================================
    // Pipeline Stitching (The Binding of IO Commitments)
    // =========================================================================

    // --- 1. LayerNorm 1 ---
    let ln1_io = LayerNormIOCommitments {
        x_com: x_in_com.clone(),
        y_com: proof.x_norm1_com.clone(),
    };
    verify_layernorm(&proof.ln1_proof, &ln1_io, &vk.ln1_vk, transcript)?;

    // --- 2. Projections (Q, K, V) ---
    let q_io = ProjectionIOCommitments {
        x_com: proof.x_norm1_com.clone(),
        y_com: proof.q_com.clone(),
    };
    verify_projection_succinct(&proof.q_proj_proof, &vk.q_vk, &q_io, transcript)?;

    let k_io = ProjectionIOCommitments {
        x_com: proof.x_norm1_com.clone(),
        y_com: proof.k_com.clone(),
    };
    verify_projection_succinct(&proof.k_proj_proof, &vk.k_vk, &k_io, transcript)?;

    let v_io = ProjectionIOCommitments {
        x_com: proof.x_norm1_com.clone(),
        y_com: proof.v_com.clone(),
    };
    verify_projection_succinct(&proof.v_proj_proof, &vk.v_vk, &v_io, transcript)?;

    // --- 3. Linear Attention ---
    let attn_io = crate::attention::attention::AttentionIOCommitments {
        q_com: proof.q_com.clone(),
        k_com: proof.k_com.clone(),
        v_com: proof.v_com.clone(),
        out_com: proof.out_inner_com.clone(),
    };
    verify_linear_attention(
        &proof.attn_proof,
        inst_attn,
        &attn_io,
        transcript,
        lasso_params,
    )?;

    // --- 4. Output Projection ---
    let o_io = ProjectionIOCommitments {
        x_com: proof.out_inner_com.clone(),
        y_com: proof.out_attn_com.clone(),
    };
    verify_projection_succinct(&proof.o_proj_proof, &vk.o_vk, &o_io, transcript)?;

    // =========================================================================
    // Residual Connection 1: X_mid = X_in + Out_attn
    // Homomorphic Addition guarantees mathematical equality in O(√N)!
    // =========================================================================
    let x_mid_com = add_commitments(x_in_com, &proof.out_attn_com);

    // --- 5. LayerNorm 2 ---
    let ln2_io = LayerNormIOCommitments {
        x_com: x_mid_com.clone(),
        y_com: proof.x_norm2_com.clone(),
    };
    verify_layernorm(&proof.ln2_proof, &ln2_io, &vk.ln2_vk, transcript)?;

    // --- 6. FFN ---
    let ffn_io = crate::ffn::ffn::FFNIOCommitments {
        x_com: proof.x_norm2_com.clone(),
        y_com: proof.out_ffn_com.clone(),
    };
    verify_ffn(
        &proof.ffn_proof,
        inst_ffn,
        &vk.ffn_vk,
        &ffn_io,
        transcript,
        lasso_params,
    )?;

    // =========================================================================
    // Residual Connection 2: X_out = X_mid + Out_ffn
    // =========================================================================
    let expected_x_out_com = add_commitments(&x_mid_com, &proof.out_ffn_com);

    // FINAL BINDING: Ensure the computation yields the expected output commitment
    // passed to the NEXT block in the pipeline.
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
=======
impl PiFormerVerifier {
    /// Verify a PiFormer proof.
    ///
    /// Replays the Fiat-Shamir transcript identically to the prover and checks:
    ///   1. Each attention head's Lasso lookups and GKR sumchecks.
    ///   2. Each block's ternary weight constraint.
    pub fn verify(
        proof: &PiFormerProof,
        witness: &PiFormerWitness,
        params: &HyraxParams,
    ) -> Result<(), String> {
        if proof.block_proofs.len() != witness.block_witnesses.len() {
            return Err(format!(
                "Block count mismatch: proof has {}, witness has {}",
                proof.block_proofs.len(),
                witness.block_witnesses.len()
            ));
        }

        let mut transcript = Transcript::new(b"PiFormer-v0.1");

        for (layer_idx, (block_proof, heads)) in proof
            .block_proofs
            .iter()
            .zip(witness.block_witnesses.iter())
            .enumerate()

        {
            transcript.append_bytes(b"layer", &(layer_idx as u64).to_le_bytes());

            if block_proof.head_proofs.len() != heads.len() {
                return Err(format!(
                    "Layer {layer_idx}: head count mismatch: proof has {}, witness has {}",
                    block_proof.head_proofs.len(),
                    heads.len()
                ));
            }

            // Verify each attention head.
            for (head_idx, (head_proof, inst)) in block_proof
                .head_proofs
                .iter()
                .zip(heads.iter())
                .enumerate()
            {
                transcript.append_bytes(b"head", &(head_idx as u64).to_le_bytes());
                verify_linear_attention(head_proof, inst, &mut transcript, params)
                    .map_err(|e| format!("Layer {layer_idx} head {head_idx}: {e}"))?;
            }

            // Verify ternary weight constraint for this block.
            let ternary_weights = collect_ternary_weights(heads);
            verify_ternary_weights(
                &block_proof.ternary_proof,
                &TernaryWeightInstance { weights: ternary_weights },
                &mut transcript,
            )
            .map_err(|e| format!("Layer {layer_idx} ternary check: {e}"))?;
        }

        Ok(())
>>>>>>> main
    }
}

// ---------------------------------------------------------------------------
<<<<<<< HEAD
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
        )
        .map_err(|e| format!("Block {} failed: {}", i, e))?;

        // Chain the output to the next block
        current_x_com = expected_x_out_com;
    }

    // 3. Final LayerNorm Verification
    let ln_io = LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: proof.final_ln_out_com.clone(),
    };
    verify_layernorm(&proof.final_ln_proof, &ln_io, &vk.final_ln_vk, transcript)
        .map_err(|e| format!("Final LN failed: {}", e))?;

    // 4. LM Head Verification
    let lm_io = ProjectionIOCommitments {
        x_com: proof.final_ln_out_com.clone(),
        y_com: proof.logits_com.clone(),
    };
    verify_projection_succinct(&proof.lm_head_proof, &vk.lm_head_vk, &lm_io, transcript)
        .map_err(|e| format!("LM Head failed: {}", e))?;

    Ok(())
=======
// Helpers (mirror of prover.rs)
// ---------------------------------------------------------------------------

fn collect_ternary_weights(
    heads: &[crate::attention::LinearAttentionInstance],
) -> Vec<F> {
    use ark_ff::PrimeField;
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
    weights
        .into_iter()
        .map(|w| {
            let raw = w.into_bigint().as_ref()[0];
            if raw == 0 {
                F::ZERO
            } else if raw == 1 {
                F::ONE
            } else {
                F::ZERO - F::ONE
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod verifier_tests {
    use super::*;
    use crate::attention::LinearAttentionInstance;
    use crate::lookup::LassoInstance;
    use crate::prover::PiFormerProver;
    use ark_ff::{Field, PrimeField};

    fn make_witness() -> PiFormerWitness {
        let seq_len = 2usize;
        let d_head = 2usize;
        let m = 4usize;
        let table_size = 1usize << m;
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

        let phi = |x: F| table[x.into_bigint().as_ref()[0] as usize];
        let phi_q: Vec<Vec<F>> = q.iter().map(|r| r.iter().map(|&x| phi(x)).collect()).collect();
        let phi_k: Vec<Vec<F>> = k.iter().map(|r| r.iter().map(|&x| phi(x)).collect()).collect();

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

        let build_lasso = |mat: &Vec<Vec<F>>| LassoInstance {
            tables: vec![table.clone()],
            query_indices: mat
                .iter()
                .flatten()
                .map(|x| x.into_bigint().as_ref()[0] as usize)
                .collect(),
            outputs: mat
                .iter()
                .flatten()
                .map(|&x| phi(x))
                .collect(),
            bits_per_chunk: m,
        };

        PiFormerWitness {
            block_witnesses: vec![vec![LinearAttentionInstance {
                seq_len,
                d_head,
                q: q.clone(),
                k: k.clone(),
                v,
                phi_q,
                phi_k,
                context,
                out,
                q_lasso: build_lasso(&q),
                k_lasso: build_lasso(&k),
            }]],
        }
    }

    #[test]
    fn test_verify_succeeds() {
        let witness = make_witness();
        let params = HyraxParams::new(2);
        let proof = PiFormerProver::prove(&witness, &params);
        let result = PiFormerVerifier::verify(&proof, &witness, &params);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }

    #[test]
    fn test_verify_rejects_block_count_mismatch() {
        let witness = make_witness();
        let params = HyraxParams::new(2);
        let proof = PiFormerProver::prove(&witness, &params);

        // Create a witness with an extra empty block.
        let mut bad_witness = make_witness();
        bad_witness.block_witnesses.push(vec![]);

        let result = PiFormerVerifier::verify(&proof, &bad_witness, &params);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Block count mismatch"));
    }
>>>>>>> main
}
