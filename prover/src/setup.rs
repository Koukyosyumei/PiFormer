//! Model Setup & Preprocessing Phase
//!
//! **役割:** //! 学習済みのモデルの重み（Weights）を受け取り、システム全体で1回だけ
//! 高コストなO(N)のコミットメント計算（hyrax_commit）を行います。
//! 生成された `VerifyingKey` は非常に小さく、スマートコントラクトやスマホに配布されます。

use crate::attention::layernorm::LayerNormVerifyingKey;
use crate::attention::projection::preprocess_projection;
use crate::ffn::ffn::preprocess_ffn;
use crate::field::F;
use crate::prover::{TransformerModelProvingKey, TransformerModelVerifyingKey};
use crate::verifier::TransformerBlockVerifyingKey;

// ---------------------------------------------------------------------------
// 1. 生の重みデータを格納する構造体 (Pythonからのインポート用)
// ---------------------------------------------------------------------------

/// 1つのTransformerブロックの生の重み
pub struct TransformerBlockWeights {
    // LayerNorm 1
    pub ln1_gamma: Vec<F>,
    pub ln1_beta: Vec<F>,
    // Linear Projections (Attention)
    pub q_w: Vec<Vec<F>>, // d_model × d_model
    pub k_w: Vec<Vec<F>>, // d_model × d_model
    pub v_w: Vec<Vec<F>>, // d_model × d_model
    pub o_w: Vec<Vec<F>>, // d_model × d_model
    // LayerNorm 2
    pub ln2_gamma: Vec<F>,
    pub ln2_beta: Vec<F>,
    // FFN
    pub ffn_w1: Vec<Vec<F>>, // d_model × d_ff
    pub ffn_w2: Vec<Vec<F>>, // d_ff × d_model
}

/// モデル全体の生の重み
pub struct TransformerModelWeights {
    pub num_blocks: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub vocab_size: usize,

    pub blocks: Vec<TransformerBlockWeights>,

    // 最終LayerNorm と 言語モデルヘッド
    pub final_ln_gamma: Vec<F>,
    pub final_ln_beta: Vec<F>,
    pub lm_head_w: Vec<Vec<F>>, // d_model × vocab_size
}

// ---------------------------------------------------------------------------
// 2. モデル全体の事前計算 (Offline Preprocessing) 関数
// ---------------------------------------------------------------------------

/// 学習済みモデルの重みを取り込み、証明者用・検証者用の鍵を生成します。
/// ※ この関数は推論のたびに走らせるのではなく、サーバー起動時に1回だけ走らせます。
pub fn preprocess_transformer_model(
    weights: TransformerModelWeights,
    seq_len: usize,
) -> TransformerModelProvingKey {
    let t = seq_len;
    let d = weights.d_model;
    let f_dim = weights.d_ff;
    let v = weights.vocab_size;

    let mut block_pks = Vec::with_capacity(weights.num_blocks);
    let mut block_vks = Vec::with_capacity(weights.num_blocks);

    // 各ブロックの重みを順番にコミットしていく
    for block_idx in 0..weights.num_blocks {
        let bw = &weights.blocks[block_idx];

        // --- LayerNormの事前計算 ---
        // LayerNormの重み(gamma, beta)は1次元なのでコミット不要。VKに直接持たせる。
        let ln1_vk = LayerNormVerifyingKey {
            seq_len: t,
            d_head: d,
            gamma: bw.ln1_gamma.clone(),
            beta: bw.ln1_beta.clone(),
            scale_gamma: F::from(1u64),
            scale_beta: F::from(1u64), // 量子化スケール等
        };
        let ln2_vk = LayerNormVerifyingKey {
            seq_len: t,
            d_head: d,
            gamma: bw.ln2_gamma.clone(),
            beta: bw.ln2_beta.clone(),
            scale_gamma: F::from(1u64),
            scale_beta: F::from(1u64),
        };

        // --- Projectionの事前計算 (ここで重い O(N) の hyrax_commit が走る) ---
        let q_pk = preprocess_projection(t, d, d, bw.q_w.clone());
        let k_pk = preprocess_projection(t, d, d, bw.k_w.clone());
        let v_pk = preprocess_projection(t, d, d, bw.v_w.clone());
        let o_pk = preprocess_projection(t, d, d, bw.o_w.clone());

        // --- FFNの事前計算 ---
        let ffn_pk = preprocess_ffn(t, d, f_dim, bw.ffn_w1.clone(), bw.ffn_w2.clone());

        // ブロックごとのVKを組み立て
        let block_vk = TransformerBlockVerifyingKey {
            seq_len: t,
            d_model: d,
            ln1_vk,
            ln2_vk,
            q_vk: q_pk.vk.clone(),
            k_vk: k_pk.vk.clone(),
            v_vk: v_pk.vk.clone(),
            o_vk: o_pk.vk.clone(),
            ffn_vk: ffn_pk.vk.clone(),
            // Prover用にPKも保持
            q_pk,
            k_pk,
            v_pk,
            o_pk,
            ffn_pk,
        };

        block_vks.push(block_vk.clone());
        block_pks.push(block_vk);
    }

    // --- Final LayerNorm ---
    let final_ln_vk = LayerNormVerifyingKey {
        seq_len: t,
        d_head: d,
        gamma: weights.final_ln_gamma.clone(),
        beta: weights.final_ln_beta.clone(),
        scale_gamma: F::from(1u64),
        scale_beta: F::from(1u64),
    };

    // --- LM Head (Vocabulary Projection) ---
    let lm_head_pk = preprocess_projection(t, d, v, weights.lm_head_w.clone());

    // --- 全体のVerifying Key ---
    let model_vk = TransformerModelVerifyingKey {
        num_blocks: weights.num_blocks,
        seq_len: t,
        d_model: d,
        vocab_size: v,
        block_vks,
        final_ln_vk,
        lm_head_vk: lm_head_pk.vk.clone(),
    };

    // --- 全体のProving Key ---
    TransformerModelProvingKey {
        vk: model_vk,
        block_pks,
        lm_head_pk,
    }
}
