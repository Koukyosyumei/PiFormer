//! PiFormer Overall Prover Architecture (GKR-based Pipeline)
//!
//! Transformer全体の検証パイプライン。最終出力から入力に向かってSumcheckを連鎖させ、
//! 最後にRange ProofやTernary Checkなどのサイド制約を一括で証明します。

use crate::attention::layernorm::{prove_layernorm, LayerNormInstance, LayerNormProof};
use crate::attention::ternary_check::{
    prove_ternary_weights, TernaryWeightInstance, TernaryWeightProof,
};
use crate::attention::{prove_linear_attention, LinearAttentionInstance, LinearAttentionProof};
use crate::field::F;
use crate::lookup::range::{prove_range, RangeProof, RangeProofInstance};
use crate::pcs::HyraxParams;
use crate::transcript::Transcript;

/// Transformer 1ブロック分の証明
pub struct PiFormerBlockProof {
    // 1. GKRプロトコルによる演算の連鎖証明 (Backward)
    pub ffn_sumcheck: FFNProof,                // FFNの行列積
    pub ln2_proof: LayerNormProof,             // 後段のLayerNorm
    pub attention_proof: LinearAttentionProof, // Linear Attention (QKV)
    pub ln1_proof: LayerNormProof,             // 前段のLayerNorm

    // 2. サイド制約 (ネットワーク全体でバッチ化して一括証明)
    pub batched_range_proof: RangeProof, // 全LayerNorm・量子化のRange Proof
    pub batched_activation_proof: LassoProof, // phi(Q), phi(K), GeLU 等のLookup
    pub batched_ternary_proof: TernaryWeightProof, // 全重み行列の {-1, 0, 1} 証明

    // 3. 最終的な入力と重みの評価値 (PCSでOpenするための値)
    pub initial_input_eval: F,
    pub weight_evals: Vec<F>,
}

pub struct PiFormerProver {
    params: HyraxParams,
}

impl PiFormerProver {
    pub fn new(params: HyraxParams) -> Self {
        Self { params }
    }

    /// Transformerの順伝播を行い、ZKPを生成する
    pub fn prove_transformer_block(
        &self,
        input_x: &[Vec<F>],
        weights: &TransformerWeights,
        transcript: &mut Transcript,
    ) -> Result<PiFormerBlockProof, String> {
        // ====================================================================
        // Phase 1: Forward Pass (Witness Generation)
        // ====================================================================
        // 実際のLLM推論を実行し、すべての中間アクティベーション、量子化スケール、
        // LayerNormの分散などをメモリに保持します。
        let witness = self.generate_witness(input_x, weights);

        // ====================================================================
        // Phase 2: The GKR Pipeline (Backward Verification)
        // ====================================================================
        // Verifierは最終出力行列 Y のランダムな点 r_out を要求する
        let r_out = generate_challenge_vector(transcript, t_bits, b"r_out_t");
        let c_out = generate_challenge_vector(transcript, d_bits, b"r_out_d");

        // 1. FFN Layer (Y = GeLU(X * W1) * W2)
        // r_out での評価値を主張し、Sumcheckを行う。
        // 結果として、FFNへの入力（= LN2の出力）のランダムな点 r_ln2 での評価値が求まる。
        let (ffn_proof, r_ln2_t, r_ln2_d, claim_ln2_out) =
            prove_ffn(&witness.ffn_inst, r_out, c_out, transcript);

        // 2. LayerNorm 2
        // FFNから渡された r_ln2 と claim_ln2_out をそのままLNの主張として使用
        // 結果として、Attentionの出力の点 r_attn での評価値が求まる。
        let (ln2_proof, r_attn_t, r_attn_d, claim_attn_out) = prove_layernorm_chained(
            &witness.ln2_inst,
            r_ln2_t,
            r_ln2_d,
            claim_ln2_out,
            transcript,
        )?;

        // 3. Linear Attention
        // LN2から渡された r_attn と claim_attn_out を使用
        let (attention_proof, r_ln1_t, r_ln1_d, claim_ln1_out) = prove_linear_attention_chained(
            &witness.attn_inst,
            r_attn_t,
            r_attn_d,
            claim_attn_out,
            transcript,
        );

        // 4. LayerNorm 1
        let (ln1_proof, r_in_t, r_in_d, claim_in) = prove_layernorm_chained(
            &witness.ln1_inst,
            r_ln1_t,
            r_ln1_d,
            claim_ln1_out,
            transcript,
        )?;

        // Phase 2 終了: `claim_in` は Transformer ブロックの初期入力 `input_x` の
        // 点 (r_in_t, r_in_d) での評価値と一致しなければならない。

        // ====================================================================
        // Phase 3: Constraint Batching (サイド制約の収集と一括証明)
        // ====================================================================

        // A. すべての Range Proof 残差を収集
        let mut all_residuals = Vec::new();
        all_residuals.extend(witness.ln1_inst.get_range_residuals());
        all_residuals.extend(witness.ln2_inst.get_range_residuals());
        all_residuals.extend(witness.attn_inst.get_quantization_residuals());
        // 一括 Lasso Range Proof
        let batched_range_proof = prove_range(
            &RangeProofInstance {
                values: all_residuals,
                bits: 32,
            },
            transcript,
            &self.params,
        )?;

        // B. すべての Activation (Lasso Lookup) を収集
        let mut all_lookups = Vec::new();
        all_lookups.extend(witness.attn_inst.get_phi_queries());
        all_lookups.extend(witness.ffn_inst.get_gelu_queries());
        let batched_activation_proof = prove_batched_lasso(&all_lookups, transcript, &self.params);

        // C. すべての重みを結合して Ternary Check
        let mut all_weights = Vec::new();
        all_weights.extend(weights.w_q.clone());
        all_weights.extend(weights.w_k.clone());
        // ... (W_V, W_O, W_1, W_2)
        let batched_ternary_proof = prove_ternary_weights(
            &TernaryWeightInstance {
                weights: all_weights,
            },
            transcript,
        );

        Ok(PiFormerBlockProof {
            ffn_sumcheck: ffn_proof,
            ln2_proof,
            attention_proof,
            ln1_proof,
            batched_range_proof,
            batched_activation_proof,
            batched_ternary_proof,
            initial_input_eval: claim_in,
            weight_evals: witness.get_weight_evaluations(),
        })
    }
}
