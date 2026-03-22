//! PiFormer verifier: verifies a `PiFormerProof` against a `PiFormerWitness`.

use crate::attention::{verify_linear_attention, verify_ternary_weights, TernaryWeightInstance};
use crate::field::F;
use ark_ff::Field;
use crate::pcs::HyraxParams;
use crate::prover::{PiFormerProof, PiFormerWitness};
use crate::transcript::Transcript;

pub struct PiFormerVerifier;

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
    }
}

// ---------------------------------------------------------------------------
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
}
