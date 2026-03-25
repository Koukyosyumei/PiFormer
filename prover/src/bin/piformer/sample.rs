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
    lookup::lasso::LassoInstance,
    prover::{TransformerBlockWitness, TransformerModelWitness},
    setup::{TransformerBlockWeights, TransformerModelWeights},
    F,
};

fn zero_mat(rows: usize, cols: usize) -> Vec<Vec<F>> {
    vec![vec![F::ZERO; cols]; rows]
}

fn ones_vec(n: usize) -> Vec<F> {
    vec![F::ONE; n]
}

fn zeros_vec(n: usize) -> Vec<F> {
    vec![F::ZERO; n]
}

fn make_block(d_model: usize, d_ff: usize) -> TransformerBlockWeights {
    TransformerBlockWeights {
        ln1_gamma: vec![F::from(2u64); d_model],
        ln1_beta: vec![F::from(5u64); d_model],
        q_w: zero_mat(d_model, d_model),
        k_w: zero_mat(d_model, d_model),
        v_w: zero_mat(d_model, d_model),
        o_w: zero_mat(d_model, d_model),
        ln2_gamma: vec![F::from(2u64); d_model],
        ln2_beta: vec![F::from(5u64); d_model],
        ffn_w1: zero_mat(d_model, d_ff),
        ffn_w2: zero_mat(d_ff, d_model),
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
        blocks: (0..num_blocks).map(|_| make_block(d_model, d_ff)).collect(),
        final_ln_gamma: vec![F::from(2u64); d_model],
        final_ln_beta: vec![F::from(5u64); d_model],
        lm_head_w: zero_mat(d_model, vocab_size),
    }
}

/// Build a trivial witness for the zero-weight model.
///
/// Input x_in = [[10, 20], [30, 40]] for d_model=2, seq_len=2.
/// LayerNorm witness values are pre-validated for this specific input with
/// gamma=[2,2], beta=[5,5]: y=[[4,6],[4,6]], var_x=[200,200], sigma=[7,7].
///
/// For other sizes, zero fill is used (placeholder; NOT valid for real proving).
pub fn build_zero_witness(
    seq_len: usize,
    d_model: usize,
    d_ff: usize,
    vocab_size: usize,
    m_bits: usize,
) -> (TransformerModelWitness, LinearAttentionInstance, FFNInstance) {
    let x_in: Vec<Vec<F>> = if seq_len == 2 && d_model == 2 {
        vec![
            vec![F::from(10u64), F::from(20u64)],
            vec![F::from(30u64), F::from(40u64)],
        ]
    } else {
        (0..seq_len)
            .map(|i| (0..d_model).map(|j| F::from(((i + 1) * 10 + j) as u64)).collect())
            .collect()
    };

    let zero_td = zero_mat(seq_len, d_model);
    let zero_tff = zero_mat(seq_len, d_ff);
    let zero_tv = zero_mat(seq_len, vocab_size);
    let zero_dd = zero_mat(d_model, d_model);

    let make_ln_wit = || build_ln_witness(&x_in, d_model);
    let ln_y = make_ln_wit().y.clone();

    let block_wit = TransformerBlockWitness {
        x_in: x_in.clone(),
        ln1_wit: make_ln_wit(),
        q_proj_wit: ProjectionWitness { x: ln_y.clone(), y: zero_td.clone() },
        k_proj_wit: ProjectionWitness { x: ln_y.clone(), y: zero_td.clone() },
        v_proj_wit: ProjectionWitness { x: ln_y.clone(), y: zero_td.clone() },
        attn_wit: LinearAttentionWitness {
            q: zero_td.clone(),
            k: zero_td.clone(),
            v: zero_td.clone(),
            phi_q: zero_td.clone(),
            phi_k: zero_td.clone(),
            context: zero_dd,
            out: zero_td.clone(),
        },
        o_proj_wit: ProjectionWitness { x: zero_td.clone(), y: zero_td.clone() },
        x_mid: x_in.clone(),
        ln2_wit: make_ln_wit(),
        ffn_wit: FFNWitness {
            x: ln_y.clone(),
            m: zero_tff.clone(),
            a: zero_tff,
            y: zero_td.clone(),
        },
        x_out: x_in.clone(),
    };

    let num_queries_td = seq_len * d_model;
    let num_queries_tff = seq_len * d_ff;
    let table_size = 1usize << m_bits;
    let _lasso_sigma = m_bits / 2;

    let make_lasso = |num_queries: usize| LassoInstance {
        tables: vec![vec![F::ZERO; table_size]],
        query_indices: vec![0usize; num_queries],
        outputs: vec![F::ZERO; num_queries],
        bits_per_chunk: m_bits,
    };

    let inst_attn = LinearAttentionInstance {
        seq_len,
        d_head: d_model,
        q_lasso: make_lasso(num_queries_td),
        k_lasso: make_lasso(num_queries_td),
    };
    let inst_ffn = FFNInstance {
        activation_lasso: make_lasso(num_queries_tff),
    };

    let witness = TransformerModelWitness {
        x_in: x_in.clone(),
        block_witnesses: vec![block_wit],
        final_ln_wit: make_ln_wit(),
        lm_head_wit: ProjectionWitness { x: ln_y, y: zero_tv },
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
            .all(|(row, exp)| row.iter().zip(exp.iter()).all(|(a, b)| a == b));
        if matches {
            return LayerNormWitness {
                x: x.to_vec(),
                y: vec![
                    vec![F::from(4u64), F::from(6u64)],
                    vec![F::from(4u64), F::from(6u64)],
                ],
                sum_x: vec![F::from(30u64), F::from(70u64)],
                var_x: vec![F::from(200u64), F::from(200u64)],
                sigma: vec![F::from(7u64), F::from(7u64)],
            };
        }
    }
    LayerNormWitness {
        x: x.to_vec(),
        y: vec![vec![F::ZERO; d_model]; t],
        sum_x: vec![F::ZERO; t],
        var_x: vec![F::ZERO; t],
        sigma: vec![F::ZERO; t],
    }
}
