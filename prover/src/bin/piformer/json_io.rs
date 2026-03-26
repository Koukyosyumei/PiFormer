//! JSON serialization for model weights and witness data.
//!
//! Field elements (F) are encoded as lowercase hex strings prefixed with "0x".
//! This format is compatible with Python's `hex(int_value)` output and web3 tooling.

use std::io;

use ark_ff::{BigInteger, Field, PrimeField};
use serde::{Deserialize, Serialize};

use piformer_prover::{
    attention::{
        attention::{LinearAttentionInstance, LinearAttentionWitness},
        layernorm::LayerNormWitness,
        projection::ProjectionWitness,
    },
    ffn::ffn::{FFNInstance, FFNWitness},
    lookup::lasso::LassoInstance,
    poly::utils::TernaryValue,
    prover::{TransformerBlockWitness, TransformerModelWitness},
    setup::{TransformerBlockWeights, TransformerModelWeights},
    F,
};

// ---------------------------------------------------------------------------
// Field element helpers
// ---------------------------------------------------------------------------

fn f_to_hex(f: &F) -> String {
    let bytes = f.into_bigint().to_bytes_be();
    format!("0x{}", hex_encode(&bytes))
}

fn f_from_hex(s: &str) -> Result<F, String> {
    let hex = s.trim_start_matches("0x").trim_start_matches("0X");
    let bytes = hex_decode(hex).map_err(|e| format!("hex decode '{s}': {e}"))?;
    Ok(F::from_be_bytes_mod_order(&bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err("odd-length hex string".into());
    }
    (0..hex.len() / 2)
        .map(|i| {
            u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|e| format!("invalid hex byte at {i}: {e}"))
        })
        .collect()
}

fn mat_to_json(mat: &[Vec<F>]) -> Vec<Vec<String>> {
    mat.iter()
        .map(|row| row.iter().map(f_to_hex).collect())
        .collect()
}

fn mat_from_json(json: Vec<Vec<String>>) -> Result<Vec<Vec<F>>, String> {
    json.into_iter()
        .enumerate()
        .map(|(i, row)| {
            row.into_iter()
                .enumerate()
                .map(|(j, s)| f_from_hex(&s).map_err(|e| format!("mat[{i}][{j}]: {e}")))
                .collect()
        })
        .collect()
}

fn ternary_mat_to_json(mat: &[Vec<TernaryValue>]) -> Vec<Vec<i8>> {
    mat.iter()
        .map(|row| {
            row.iter()
                .map(|v| match v {
                    TernaryValue::ONE => 1i8,
                    TernaryValue::ZERO => 0i8,
                    TernaryValue::MINUSONE => -1i8,
                })
                .collect()
        })
        .collect()
}

fn ternary_mat_from_json(json: Vec<Vec<i8>>) -> Result<Vec<Vec<TernaryValue>>, String> {
    json.into_iter()
        .enumerate()
        .map(|(i, row)| {
            row.into_iter()
                .enumerate()
                .map(|(j, v)| match v {
                    1 => Ok(TernaryValue::ONE),
                    0 => Ok(TernaryValue::ZERO),
                    -1 => Ok(TernaryValue::MINUSONE),
                    other => Err(format!("mat[{i}][{j}]: invalid ternary value {other}")),
                })
                .collect()
        })
        .collect()
}

fn vec_to_json(v: &[F]) -> Vec<String> {
    v.iter().map(f_to_hex).collect()
}

fn vec_from_json(json: Vec<String>) -> Result<Vec<F>, String> {
    json.into_iter()
        .enumerate()
        .map(|(i, s)| f_from_hex(&s).map_err(|e| format!("vec[{i}]: {e}")))
        .collect()
}

fn io_err(msg: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

// ---------------------------------------------------------------------------
// JSON types for model weights
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct JsonBlockWeights {
    pub ln1_gamma: Vec<String>,
    pub ln1_beta: Vec<String>,
    pub q_w: Vec<Vec<i8>>,
    #[serde(default)]
    pub q_alpha: String,
    #[serde(default)]
    pub q_bias: Vec<String>,
    pub k_w: Vec<Vec<i8>>,
    #[serde(default)]
    pub k_alpha: String,
    #[serde(default)]
    pub k_bias: Vec<String>,
    pub v_w: Vec<Vec<i8>>,
    #[serde(default)]
    pub v_alpha: String,
    #[serde(default)]
    pub v_bias: Vec<String>,
    pub o_w: Vec<Vec<i8>>,
    #[serde(default)]
    pub o_alpha: String,
    #[serde(default)]
    pub o_bias: Vec<String>,
    pub ln2_gamma: Vec<String>,
    pub ln2_beta: Vec<String>,
    pub ffn_w1: Vec<Vec<i8>>,
    pub ffn_w2: Vec<Vec<i8>>,
}

#[derive(Serialize, Deserialize)]
pub struct JsonWeights {
    pub num_blocks: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub blocks: Vec<JsonBlockWeights>,
    pub final_ln_gamma: Vec<String>,
    pub final_ln_beta: Vec<String>,
    pub lm_head_w: Vec<Vec<i8>>,
    #[serde(default)]
    pub lm_head_alpha: String,
    #[serde(default)]
    pub lm_head_bias: Vec<String>,
}

fn f_from_hex_or_one(s: &str) -> Result<F, String> {
    if s.is_empty() {
        Ok(F::ONE)
    } else {
        f_from_hex(s)
    }
}

fn vec_from_json_or_empty(json: Vec<String>) -> Result<Vec<F>, String> {
    if json.is_empty() {
        Ok(vec![])
    } else {
        vec_from_json(json)
    }
}

pub fn weights_to_json(w: &TransformerModelWeights) -> JsonWeights {
    let blocks = w
        .blocks
        .iter()
        .map(|b| JsonBlockWeights {
            ln1_gamma: vec_to_json(&b.ln1_gamma),
            ln1_beta: vec_to_json(&b.ln1_beta),
            q_w: ternary_mat_to_json(&b.q_w),
            q_alpha: f_to_hex(&b.q_alpha),
            q_bias: vec_to_json(&b.q_bias),
            k_w: ternary_mat_to_json(&b.k_w),
            k_alpha: f_to_hex(&b.k_alpha),
            k_bias: vec_to_json(&b.k_bias),
            v_w: ternary_mat_to_json(&b.v_w),
            v_alpha: f_to_hex(&b.v_alpha),
            v_bias: vec_to_json(&b.v_bias),
            o_w: ternary_mat_to_json(&b.o_w),
            o_alpha: f_to_hex(&b.o_alpha),
            o_bias: vec_to_json(&b.o_bias),
            ln2_gamma: vec_to_json(&b.ln2_gamma),
            ln2_beta: vec_to_json(&b.ln2_beta),
            ffn_w1: ternary_mat_to_json(&b.ffn_w1),
            ffn_w2: ternary_mat_to_json(&b.ffn_w2),
        })
        .collect();
    JsonWeights {
        num_blocks: w.num_blocks,
        d_model: w.d_model,
        d_ff: w.d_ff,
        vocab_size: w.vocab_size,
        blocks,
        final_ln_gamma: vec_to_json(&w.final_ln_gamma),
        final_ln_beta: vec_to_json(&w.final_ln_beta),
        lm_head_w: ternary_mat_to_json(&w.lm_head_w),
        lm_head_alpha: f_to_hex(&w.lm_head_alpha),
        lm_head_bias: vec_to_json(&w.lm_head_bias),
    }
}

pub fn weights_from_json(j: JsonWeights) -> Result<TransformerModelWeights, String> {
    let blocks: Result<Vec<_>, _> = j
        .blocks
        .into_iter()
        .map(|b| -> Result<TransformerBlockWeights, String> {
            Ok(TransformerBlockWeights {
                ln1_gamma: vec_from_json(b.ln1_gamma)?,
                ln1_beta: vec_from_json(b.ln1_beta)?,
                q_w: ternary_mat_from_json(b.q_w)?,
                q_alpha: f_from_hex_or_one(&b.q_alpha)?,
                q_bias: vec_from_json_or_empty(b.q_bias)?,
                k_w: ternary_mat_from_json(b.k_w)?,
                k_alpha: f_from_hex_or_one(&b.k_alpha)?,
                k_bias: vec_from_json_or_empty(b.k_bias)?,
                v_w: ternary_mat_from_json(b.v_w)?,
                v_alpha: f_from_hex_or_one(&b.v_alpha)?,
                v_bias: vec_from_json_or_empty(b.v_bias)?,
                o_w: ternary_mat_from_json(b.o_w)?,
                o_alpha: f_from_hex_or_one(&b.o_alpha)?,
                o_bias: vec_from_json_or_empty(b.o_bias)?,
                ln2_gamma: vec_from_json(b.ln2_gamma)?,
                ln2_beta: vec_from_json(b.ln2_beta)?,
                ffn_w1: ternary_mat_from_json(b.ffn_w1)?,
                ffn_w2: ternary_mat_from_json(b.ffn_w2)?,
            })
        })
        .collect();
    Ok(TransformerModelWeights {
        num_blocks: j.num_blocks,
        d_model: j.d_model,
        d_ff: j.d_ff,
        vocab_size: j.vocab_size,
        blocks: blocks?,
        final_ln_gamma: vec_from_json(j.final_ln_gamma)?,
        final_ln_beta: vec_from_json(j.final_ln_beta)?,
        lm_head_w: ternary_mat_from_json(j.lm_head_w)?,
        lm_head_alpha: f_from_hex_or_one(&j.lm_head_alpha)?,
        lm_head_bias: vec_from_json_or_empty(j.lm_head_bias)?,
    })
}

// ---------------------------------------------------------------------------
// JSON types for witness
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct JsonLassoInstance {
    pub tables: Vec<Vec<String>>,
    pub query_indices: Vec<usize>,
    pub outputs: Vec<String>,
    pub bits_per_chunk: usize,
}

#[derive(Serialize, Deserialize)]
pub struct JsonAttnInstance {
    pub seq_len: usize,
    pub d_head: usize,
    pub q_lasso: JsonLassoInstance,
    pub k_lasso: JsonLassoInstance,
}

#[derive(Serialize, Deserialize)]
pub struct JsonFFNInstance {
    pub activation_lasso: JsonLassoInstance,
}

#[derive(Serialize, Deserialize)]
pub struct JsonLayerNormWitness {
    pub x: Vec<Vec<String>>,
    pub y: Vec<Vec<String>>,
    pub sum_x: Vec<String>,
    pub var_x: Vec<String>,
    pub sigma: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct JsonProjectionWitness {
    pub x: Vec<Vec<String>>,
    pub y: Vec<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
pub struct JsonAttnWitness {
    pub q: Vec<Vec<String>>,
    pub k: Vec<Vec<String>>,
    pub v: Vec<Vec<String>>,
    pub phi_q: Vec<Vec<String>>,
    pub phi_k: Vec<Vec<String>>,
    pub context: Vec<Vec<String>>,
    pub out: Vec<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
pub struct JsonFFNWitness {
    pub x: Vec<Vec<String>>,
    pub m: Vec<Vec<String>>,
    pub a: Vec<Vec<String>>,
    pub y: Vec<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
pub struct JsonBlockWitness {
    pub x_in: Vec<Vec<String>>,
    pub ln1: JsonLayerNormWitness,
    pub q_proj: JsonProjectionWitness,
    pub k_proj: JsonProjectionWitness,
    pub v_proj: JsonProjectionWitness,
    pub attn: JsonAttnWitness,
    pub o_proj: JsonProjectionWitness,
    pub x_mid: Vec<Vec<String>>,
    pub ln2: JsonLayerNormWitness,
    pub ffn: JsonFFNWitness,
    pub x_out: Vec<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
pub struct JsonWitness {
    /// Sigma for lasso HyraxParams (lasso_params = HyraxParams::new(lasso_sigma))
    pub lasso_sigma: usize,
    pub x_in: Vec<Vec<String>>,
    pub inst_attn: JsonAttnInstance,
    pub inst_ffn: JsonFFNInstance,
    pub blocks: Vec<JsonBlockWitness>,
    pub final_ln: JsonLayerNormWitness,
    pub lm_head: JsonProjectionWitness,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn lasso_to_json(inst: &LassoInstance) -> JsonLassoInstance {
    JsonLassoInstance {
        tables: inst.tables.iter().map(|t| vec_to_json(t)).collect(),
        query_indices: inst.query_indices.clone(),
        outputs: vec_to_json(&inst.outputs),
        bits_per_chunk: inst.bits_per_chunk,
    }
}

fn lasso_from_json(j: JsonLassoInstance) -> Result<LassoInstance, String> {
    Ok(LassoInstance {
        tables: j
            .tables
            .into_iter()
            .enumerate()
            .map(|(i, t)| vec_from_json(t).map_err(|e| format!("tables[{i}]: {e}")))
            .collect::<Result<_, _>>()?,
        query_indices: j.query_indices,
        outputs: vec_from_json(j.outputs)?,
        bits_per_chunk: j.bits_per_chunk,
    })
}

fn ln_wit_to_json(w: &LayerNormWitness) -> JsonLayerNormWitness {
    JsonLayerNormWitness {
        x: mat_to_json(&w.x),
        y: mat_to_json(&w.y),
        sum_x: vec_to_json(&w.sum_x),
        sigma: vec_to_json(&w.sigma),
        // var_x stores sq_sum_x (sum of squares per row)
        var_x: vec_to_json(&w.sq_sum_x),
    }
}

fn ln_wit_from_json(j: JsonLayerNormWitness) -> Result<LayerNormWitness, String> {
    let x = mat_from_json(j.x)?;
    let y = mat_from_json(j.y)?;
    let sum_x = vec_from_json(j.sum_x)?;
    let sigma = vec_from_json(j.sigma)?;
    // var_x in JSON encodes sq_sum_x (sum of squares per row)
    let sq_sum_x = vec_from_json(j.var_x)?;
    // sum_x_sq and sigma_sq_scaled are derived quantities
    let d = x.first().map(|r| r.len()).unwrap_or(0);
    let d_f = F::from(d as u64);
    let sum_x_sq: Vec<F> = sum_x.iter().map(|&s| s * s).collect();
    let sigma_sq_scaled: Vec<F> = sigma.iter().map(|&s| (d_f * s) * (d_f * s)).collect();
    Ok(LayerNormWitness {
        x,
        y,
        sum_x,
        sigma,
        sq_sum_x,
        sum_x_sq,
        sigma_sq_scaled,
    })
}

fn proj_wit_to_json(w: &ProjectionWitness) -> JsonProjectionWitness {
    JsonProjectionWitness {
        x: mat_to_json(&w.x),
        y: mat_to_json(&w.y),
    }
}

fn proj_wit_from_json(j: JsonProjectionWitness) -> Result<ProjectionWitness, String> {
    Ok(ProjectionWitness {
        x: mat_from_json(j.x)?,
        y: mat_from_json(j.y)?,
    })
}

fn attn_wit_to_json(w: &LinearAttentionWitness) -> JsonAttnWitness {
    JsonAttnWitness {
        q: mat_to_json(&w.q),
        k: mat_to_json(&w.k),
        v: mat_to_json(&w.v),
        phi_q: mat_to_json(&w.phi_q),
        phi_k: mat_to_json(&w.phi_k),
        context: mat_to_json(&w.context),
        out: mat_to_json(&w.out),
    }
}

fn attn_wit_from_json(j: JsonAttnWitness) -> Result<LinearAttentionWitness, String> {
    Ok(LinearAttentionWitness {
        q: mat_from_json(j.q)?,
        k: mat_from_json(j.k)?,
        v: mat_from_json(j.v)?,
        phi_q: mat_from_json(j.phi_q)?,
        phi_k: mat_from_json(j.phi_k)?,
        context: mat_from_json(j.context)?,
        out: mat_from_json(j.out)?,
    })
}

fn ffn_wit_to_json(w: &FFNWitness) -> JsonFFNWitness {
    JsonFFNWitness {
        x: mat_to_json(&w.x),
        m: mat_to_json(&w.m),
        a: mat_to_json(&w.a),
        y: mat_to_json(&w.y),
    }
}

fn ffn_wit_from_json(j: JsonFFNWitness) -> Result<FFNWitness, String> {
    Ok(FFNWitness {
        x: mat_from_json(j.x)?,
        m: mat_from_json(j.m)?,
        a: mat_from_json(j.a)?,
        y: mat_from_json(j.y)?,
    })
}

pub fn witness_to_json(
    w: &TransformerModelWitness,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    lasso_sigma: usize,
) -> JsonWitness {
    let blocks = w
        .block_witnesses
        .iter()
        .map(|b| JsonBlockWitness {
            x_in: mat_to_json(&b.x_in),
            ln1: ln_wit_to_json(&b.ln1_wit),
            q_proj: proj_wit_to_json(&b.q_proj_wit),
            k_proj: proj_wit_to_json(&b.k_proj_wit),
            v_proj: proj_wit_to_json(&b.v_proj_wit),
            attn: attn_wit_to_json(&b.attn_wit),
            o_proj: proj_wit_to_json(&b.o_proj_wit),
            x_mid: mat_to_json(&b.x_mid),
            ln2: ln_wit_to_json(&b.ln2_wit),
            ffn: ffn_wit_to_json(&b.ffn_wit),
            x_out: mat_to_json(&b.x_out),
        })
        .collect();
    JsonWitness {
        lasso_sigma,
        x_in: mat_to_json(&w.x_in),
        inst_attn: JsonAttnInstance {
            seq_len: inst_attn.seq_len,
            d_head: inst_attn.d_head,
            q_lasso: lasso_to_json(&inst_attn.q_lasso),
            k_lasso: lasso_to_json(&inst_attn.k_lasso),
        },
        inst_ffn: JsonFFNInstance {
            activation_lasso: lasso_to_json(&inst_ffn.activation_lasso),
        },
        blocks,
        final_ln: ln_wit_to_json(&w.final_ln_wit),
        lm_head: proj_wit_to_json(&w.lm_head_wit),
    }
}

pub fn witness_from_json(
    j: JsonWitness,
) -> Result<
    (
        TransformerModelWitness,
        LinearAttentionInstance,
        FFNInstance,
        usize,
    ),
    String,
> {
    let lasso_sigma = j.lasso_sigma;
    let x_in = mat_from_json(j.x_in)?;

    let inst_attn = LinearAttentionInstance {
        seq_len: j.inst_attn.seq_len,
        d_head: j.inst_attn.d_head,
        q_lasso: lasso_from_json(j.inst_attn.q_lasso)?,
        k_lasso: lasso_from_json(j.inst_attn.k_lasso)?,
    };
    let inst_ffn = FFNInstance {
        activation_lasso: lasso_from_json(j.inst_ffn.activation_lasso)?,
    };

    let block_witnesses: Result<Vec<_>, _> = j
        .blocks
        .into_iter()
        .map(|b| -> Result<TransformerBlockWitness, String> {
            Ok(TransformerBlockWitness {
                x_in: mat_from_json(b.x_in)?,
                ln1_wit: ln_wit_from_json(b.ln1)?,
                q_proj_wit: proj_wit_from_json(b.q_proj)?,
                k_proj_wit: proj_wit_from_json(b.k_proj)?,
                v_proj_wit: proj_wit_from_json(b.v_proj)?,
                attn_wit: attn_wit_from_json(b.attn)?,
                o_proj_wit: proj_wit_from_json(b.o_proj)?,
                x_mid: mat_from_json(b.x_mid)?,
                ln2_wit: ln_wit_from_json(b.ln2)?,
                ffn_wit: ffn_wit_from_json(b.ffn)?,
                x_out: mat_from_json(b.x_out)?,
            })
        })
        .collect();

    let witness = TransformerModelWitness {
        x_in,
        block_witnesses: block_witnesses?,
        final_ln_wit: ln_wit_from_json(j.final_ln)?,
        lm_head_wit: proj_wit_from_json(j.lm_head)?,
    };

    Ok((witness, inst_attn, inst_ffn, lasso_sigma))
}
