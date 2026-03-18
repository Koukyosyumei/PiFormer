//! Top-level PiFormer verifier.

use crate::attention::verify_linear_attention;
use crate::field::F;
use crate::pcs::HyraxParams;
use crate::prover::{PiFormerProof, PiFormerWitness};
use crate::transcript::Transcript;

pub struct PiFormerVerifier;

impl PiFormerVerifier {
    /// Verify a PiFormer proof.
    ///
    /// The `witness` here carries the public inputs (Q, K, V, context, out)
    /// that both prover and verifier know. In a real deployment the verifier
    /// would only receive commitments, not the raw matrices; the sumcheck / PCS
    /// opening arguments replace the matrix reveals.
    pub fn verify(
        proof: &PiFormerProof,
        witness: &PiFormerWitness,
        params: &HyraxParams,
    ) -> Result<(), String> {
        let mut transcript = Transcript::new(b"PiFormer-v0.1");

        for (layer_idx, (layer_proofs, layer_heads)) in proof
            .block_proofs.iter()
            .zip(witness.block_witnesses.iter())
            .enumerate()
        {
            let _fusion_challenge = transcript.challenge_field::<F>(b"layer_fusion");
            transcript.append_bytes(b"layer", &(layer_idx as u64).to_le_bytes());

            for (head_idx, (head_proof, inst)) in layer_proofs
                .iter()
                .zip(layer_heads.iter())
                .enumerate()
            {
                transcript.append_bytes(b"head", &(head_idx as u64).to_le_bytes());
                verify_linear_attention(head_proof, inst, &mut transcript, params)
                    .map_err(|e| format!("Layer {layer_idx} head {head_idx}: {e}"))?;
            }
        }
        Ok(())
    }
}
