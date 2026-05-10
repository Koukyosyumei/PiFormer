//! Sample model and witness generator for demos and integration testing.
//!
//! Uses an all-zero-weight model so all matrix multiply outputs are zero,
//! leaving only the LayerNorm layers active and making witness construction
//! deterministic without running a real transformer forward pass.

use ark_ff::Field;

use piformer_prover::{
    attention::{
        attention::{LinearAttentionInstance, LinearAttentionWitness},
        layernorm::LayerNormWitness,
        projection::ProjectionWitness,
    },
    ffn::ffn::{FFNInstance, FFNWitness},
    lookup::{lasso::LassoInstance, quantization::QuantizationParams},
    poly::utils::TernaryValue,
    prover::{TransformerBlockWitness, TransformerModelWitness},
    setup::{TransformerBlockWeights, TransformerModelWeights},
    F,
};

fn zero_mat(rows: usize, cols: usize) -> Vec<Vec<F>> {
    vec![vec![F::ZERO; cols]; rows]
}

fn zero_ternary_mat(rows: usize, cols: usize) -> Vec<Vec<TernaryValue>> {
    vec![vec![TernaryValue::ZERO; cols]; rows]
}

fn make_block(d_model: usize, d_ff: usize, m_bits: usize) -> TransformerBlockWeights {
    let table_size = 1usize << m_bits;
    let dummy_table = vec![vec![F::ZERO; table_size]];
    TransformerBlockWeights {
        ln1_gamma: vec![F::from(2u64); d_model],
        ln1_beta: vec![F::from(5u64); d_model],
        q_w: zero_ternary_mat(d_model, d_model),
        q_alpha: F::ONE,
        // q_bias chosen so q_proj.y = bias has non-zero variance per row,
        // which the QK-norm LayerNorm requires to satisfy its range proofs.
        q_bias: {
            let mut b = vec![F::ZERO; d_model];
            if d_model >= 1 {
                b[0] = F::from(2u64);
            }
            b
        },
        k_w: zero_ternary_mat(d_model, d_model),
        k_alpha: F::ONE,
        k_bias: {
            let mut b = vec![F::ZERO; d_model];
            if d_model >= 1 {
                b[0] = F::from(2u64);
            }
            b
        },
        v_w: zero_ternary_mat(d_model, d_model),
        v_alpha: F::ONE,
        v_bias: vec![F::ZERO; d_model],
        o_w: zero_ternary_mat(d_model, d_model),
        o_alpha: F::ONE,
        // o_bias chosen so out_attn = α · 0 + bias has non-zero variance per
        // row, which the attn_out_norm LayerNorm requires for its range proofs.
        o_bias: {
            let mut b = vec![F::ZERO; d_model];
            if d_model >= 1 {
                b[0] = F::from(2u64);
            }
            b
        },
        // Sandwich-norm LayerNorms (q_norm, k_norm, attn_out_norm).
        q_norm_gamma: vec![F::from(2u64); d_model],
        q_norm_beta: vec![F::from(5u64); d_model],
        k_norm_gamma: vec![F::from(2u64); d_model],
        k_norm_beta: vec![F::from(5u64); d_model],
        attn_out_norm_gamma: vec![F::from(2u64); d_model],
        attn_out_norm_beta: vec![F::from(5u64); d_model],
        ln2_gamma: vec![F::from(2u64); d_model],
        ln2_beta: vec![F::from(5u64); d_model],
        ffn_w1: zero_ternary_mat(d_model, d_ff),
        ffn_w2: zero_ternary_mat(d_ff, d_model),
        ffn_activation_tables: dummy_table.clone(),
        ffn_activation_bits_per_chunk: m_bits,
        ffn_activation_quant: QuantizationParams {
            scale_num: 2,
            scale_den: 2,
        },
        q_activation_tables: dummy_table.clone(),
        k_activation_tables: dummy_table.clone(),
        qk_activation_bits_per_chunk: m_bits,
        qk_activation_quant: QuantizationParams {
            scale_num: 2,
            scale_den: 2,
        },
    }
}

/// Build a model with `num_blocks` blocks, all weights zero.
pub fn build_zero_weights(
    num_blocks: usize,
    d_model: usize,
    d_ff: usize,
    vocab_size: usize,
) -> TransformerModelWeights {
    TransformerModelWeights {
        num_blocks,
        d_model,
        d_ff,
        vocab_size,
        causal: false,
        blocks: (0..num_blocks)
            .map(|_| make_block(d_model, d_ff, 4))
            .collect(),
        final_ln_gamma: vec![F::from(2u64); d_model],
        final_ln_beta: vec![F::from(5u64); d_model],
        lm_head_w: zero_ternary_mat(d_model, vocab_size),
        lm_head_alpha: F::ONE,
        lm_head_bias: vec![F::ZERO; vocab_size],
    }
}

/// Build a trivial witness for the zero-weight model.
///
/// Input x_in = [[10, 20], [30, 40]] for d_model=2, seq_len=2.
/// LayerNorm witness values are pre-validated for this specific input with
/// gamma=[2,2], beta=[5,5]: y=[[4,6],[4,6]], var_x=[200,200], sigma=[7,7].
///
pub fn build_zero_witness(
    seq_len: usize,
    d_model: usize,
    d_ff: usize,
    vocab_size: usize,
    m_bits: usize,
) -> (
    TransformerModelWitness,
    LinearAttentionInstance,
    FFNInstance,
) {
    assert_eq!(seq_len, 2, "sample witness is defined only for seq_len=2");
    assert_eq!(d_model, 2, "sample witness is defined only for d_model=2");
    let x_in: Vec<Vec<F>> = vec![
        vec![F::from(10u64), F::from(20u64)],
        vec![F::from(30u64), F::from(40u64)],
    ];

    let zero_td = zero_mat(seq_len, d_model);
    let zero_tff = zero_mat(seq_len, d_ff);
    let zero_tv = zero_mat(seq_len, vocab_size);
    let zero_dd = zero_mat(d_model, d_model);

    let make_ln_wit = || build_ln_witness(&x_in, d_model);
    let ln_y = make_ln_wit().y.clone();
    let ffn_zp = 0usize;

    // attn_out_norm_wit: x = o_proj.y = [[2, 0], [2, 0]] (W_o is zero, bias_o[0] = 2),
    // y = LN(x) = [[7, 3], [7, 3]] with gamma=[2,2], beta=[5,5], sigma=[1,1].
    // (Same shape as q/k_norm; reuse the same constants below.)
    let aon_raw: Vec<Vec<F>> = vec![
        vec![F::from(2u64), F::from(0u64)],
        vec![F::from(2u64), F::from(0u64)],
    ];
    let aon_y_mat: Vec<Vec<F>> = vec![
        vec![F::from(7u64), F::from(3u64)],
        vec![F::from(7u64), F::from(3u64)],
    ];
    let attn_out_norm_wit = LayerNormWitness {
        x: aon_raw.clone(),
        y: aon_y_mat.clone(),
        sum_x: vec![F::from(2u64); seq_len],
        sigma: vec![F::from(1u64); seq_len],
        sq_sum_x: vec![F::from(4u64); seq_len],
        sum_x_sq: vec![F::from(4u64); seq_len],
        sigma_sq_scaled: vec![F::from(4u64); seq_len],
    };

    // x_mid = x_in + attn_out_norm_y = [[10+7, 20+3], [30+7, 40+3]]
    //                                = [[17, 23], [37, 43]].
    let x_mid: Vec<Vec<F>> = vec![
        vec![F::from(17u64), F::from(23u64)],
        vec![F::from(37u64), F::from(43u64)],
    ];
    // ln2 of x_mid with gamma=[2,2], beta=[5,5]: y = [[4, 7], [4, 7]], sigma = 4.
    let ln2_y: Vec<Vec<F>> = vec![
        vec![F::from(4u64), F::from(7u64)],
        vec![F::from(4u64), F::from(7u64)],
    ];
    let make_ln_post_resid = || LayerNormWitness {
        x: x_mid.clone(),
        y: ln2_y.clone(),
        // sum_x = [40, 80]; sq_sum_x = [17²+23²=818, 37²+43²=3218];
        // sum_x_sq = [40²=1600, 80²=6400]; sigma = [4, 4];
        // sigma_sq_scaled = (d·σ)² = (2·4)² = 64.
        sum_x: vec![F::from(40u64), F::from(80u64)],
        sigma: vec![F::from(4u64); seq_len],
        sq_sum_x: vec![F::from(818u64), F::from(3218u64)],
        sum_x_sq: vec![F::from(1600u64), F::from(6400u64)],
        sigma_sq_scaled: vec![F::from(64u64); seq_len],
    };

    // q_proj.y = q_raw = q_bias broadcast across rows: each row = [2, 0].
    // (W_q is zero, so q_proj.y = α * 0 + bias = bias.)  The q_norm_wit below
    // is the LN of this with gamma=[2,2], beta=[5,5], yielding y=[[7,3],[7,3]],
    // which then becomes attn_wit.q (the input to φ).
    let qk_raw: Vec<Vec<F>> = vec![
        vec![F::from(2u64), F::from(0u64)],
        vec![F::from(2u64), F::from(0u64)],
    ];
    let qk_norm: Vec<Vec<F>> = vec![
        vec![F::from(7u64), F::from(3u64)],
        vec![F::from(7u64), F::from(3u64)],
    ];
    let make_qk_norm_wit = || LayerNormWitness {
        x: qk_raw.clone(),
        y: qk_norm.clone(),
        sum_x: vec![F::from(2u64); seq_len],
        sigma: vec![F::from(1u64); seq_len],
        sq_sum_x: vec![F::from(4u64); seq_len],
        sum_x_sq: vec![F::from(4u64); seq_len],
        // sigma_sq_scaled[i] = (d * sigma[i])^2 = (2 * 1)^2 = 4.
        sigma_sq_scaled: vec![F::from(4u64); seq_len],
    };
    // Lookup indices for the φ stage: chunk index = q_n value (M_BITS=4 → table 0..16).
    let qk_query_indices: Vec<usize> = vec![7, 3, 7, 3];

    let block_wit = TransformerBlockWitness {
        x_in: x_in.clone(),
        ln1_wit: make_ln_wit(),
        q_proj_wit: ProjectionWitness {
            x: ln_y.clone(),
            y: qk_raw.clone(),
        },
        k_proj_wit: ProjectionWitness {
            x: ln_y.clone(),
            y: qk_raw.clone(),
        },
        v_proj_wit: ProjectionWitness {
            x: ln_y.clone(),
            y: zero_td.clone(),
        },
        q_norm_wit: make_qk_norm_wit(),
        k_norm_wit: make_qk_norm_wit(),
        attn_wit: LinearAttentionWitness {
            q: qk_norm.clone(),
            k: qk_norm.clone(),
            v: zero_td.clone(),
            phi_q: zero_td.clone(),
            phi_k: zero_td.clone(),
            q_query_indices: qk_query_indices.clone(),
            k_query_indices: qk_query_indices.clone(),
            context: zero_dd,
            causal_context: None,
            normalized_out: None,
            norm_z: None,
            norm_rem: None,
            norm_diff: None,
            out: zero_td.clone(),
        },
        // o_proj.x = attn_proj_in (zero); o_proj.y = bias_o broadcast = aon_raw.
        o_proj_wit: ProjectionWitness {
            x: zero_td.clone(),
            y: aon_raw.clone(),
        },
        attn_out_norm_wit,
        x_mid: x_mid.clone(),
        // ln2 takes x_mid (post-residual, post-attn_out_norm).
        // Note: ffn.x is ln2.y, not ln1.y, so update accordingly.
        ln2_wit: make_ln_post_resid(),
        ffn_wit: FFNWitness {
            x: ln2_y.clone(),
            m: zero_tff.clone(),
            a: zero_tff.clone(),
            y: zero_td.clone(),
            activation_query_indices: vec![ffn_zp; seq_len * d_ff],
        },
        // x_out = x_mid + ffn.y = x_mid (ffn output is zero).
        x_out: x_mid.clone(),
    };

    let num_queries_td = seq_len * d_model;
    let num_queries_tff = seq_len * d_ff;
    let table_size = 1usize << m_bits;
    let _lasso_sigma = m_bits / 2;

    let make_lasso = |num_queries: usize| LassoInstance {
        tables: vec![vec![F::ZERO; table_size]],
        outputs: vec![F::ZERO; num_queries],
        bits_per_chunk: m_bits,
    };

    let inst_attn = LinearAttentionInstance {
        seq_len,
        d_head: d_model,
        causal: false,
        q_lasso: make_lasso(num_queries_td),
        k_lasso: make_lasso(num_queries_td),
        q_query_indices: qk_query_indices.clone(),
        k_query_indices: qk_query_indices,
    };
    let inst_ffn = FFNInstance {
        activation_lasso: make_lasso(num_queries_tff),
    };

    let witness = TransformerModelWitness {
        x_in: x_in.clone(),
        block_witnesses: vec![block_wit],
        // final_ln takes x_out (= x_mid in this fixture), so it's the
        // post-residual LN witness, not the pre-block one.
        final_ln_wit: make_ln_post_resid(),
        lm_head_wit: ProjectionWitness {
            x: ln2_y.clone(),
            y: zero_tv,
        },
    };

    (witness, inst_attn, inst_ffn)
}

/// Construct a LayerNorm witness for the given x matrix and unit weights.
fn build_ln_witness(x: &[Vec<F>], d_model: usize) -> LayerNormWitness {
    let t = x.len();
    if t == 2 && d_model == 2 {
        let expected = [
            [F::from(10u64), F::from(20u64)],
            [F::from(30u64), F::from(40u64)],
        ];
        let matches = x
            .iter()
            .zip(expected.iter())
            .all(|(row, exp): (&Vec<F>, &[F; 2])| row.iter().zip(exp.iter()).all(|(a, b)| a == b));
        if matches {
            return LayerNormWitness {
                x: x.to_vec(),
                y: vec![
                    vec![F::from(4u64), F::from(6u64)],
                    vec![F::from(4u64), F::from(6u64)],
                ],
                sum_x: vec![F::from(30u64), F::from(70u64)],
                sigma: vec![F::from(7u64), F::from(7u64)],
                // sq_sum_x[i] = sum_j x[i][j]^2: [10^2+20^2=500, 30^2+40^2=2500]
                sq_sum_x: vec![F::from(500u64), F::from(2500u64)],
                // sum_x_sq[i] = sum_x[i]^2: [30^2=900, 70^2=4900]
                sum_x_sq: vec![F::from(900u64), F::from(4900u64)],
                // sigma_sq_scaled[i] = (d*sigma[i])^2 = (2*7)^2=196
                sigma_sq_scaled: vec![F::from(196u64), F::from(196u64)],
            };
        }
    }
    panic!("sample LayerNorm witness is defined only for the built-in 2x2 input")
}
