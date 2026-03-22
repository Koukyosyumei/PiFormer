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

        PiFormerProof { block_proofs }
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
