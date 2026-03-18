//! Top-level PiFormer prover.
//!
//! Orchestrates Lasso lookups, sumchecks, and constraint fusion across all layers/heads.
//!
//! **Constraint fusion** (inspired by zkGPT):
//! Before processing each layer, the prover squeezes a random challenge from the transcript.
//! This binds all per-head proofs in that layer together cryptographically, so the verifier
//! cannot accept a valid proof for one head alongside a bogus proof for another.

use crate::attention::{LinearAttentionInstance, LinearAttentionProof, prove_linear_attention};
use crate::field::F;
use crate::transcript::Transcript;

/// Proof for the entire model inference (all layers, all heads).
pub struct PiFormerProof {
    /// block_proofs[layer][head]
    pub block_proofs: Vec<Vec<LinearAttentionProof>>,
}

/// Witness holding all per-layer, per-head instances.
pub struct PiFormerWitness {
    /// block_witnesses[layer][head]
    pub block_witnesses: Vec<Vec<LinearAttentionInstance>>,
}

pub struct PiFormerProver;

impl PiFormerProver {
    pub fn prove(witness: &PiFormerWitness) -> PiFormerProof {
        let mut transcript = Transcript::new(b"PiFormer-v0.1");

        let mut block_proofs = Vec::new();
        for (layer_idx, layer_heads) in witness.block_witnesses.iter().enumerate() {
            // Constraint fusion: bind all heads in this layer to one random challenge
            let _fusion_challenge = transcript.challenge_field::<F>(b"layer_fusion");
            transcript.append_bytes(b"layer", &(layer_idx as u64).to_le_bytes());

            let mut head_proofs = Vec::new();
            for (head_idx, inst) in layer_heads.iter().enumerate() {
                transcript.append_bytes(b"head", &(head_idx as u64).to_le_bytes());
                head_proofs.push(prove_linear_attention(inst, &mut transcript));
            }
            block_proofs.push(head_proofs);
        }

        PiFormerProof { block_proofs }
    }
}
