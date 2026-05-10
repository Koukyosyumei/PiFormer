//! Global Verifier for a full Transformer Model.
//!
//! **Cross-block Batch Sumcheck Architecture:**
//! Phase 1: verify range proofs + LN1 + LN2 for all L blocks; absorb intermediate commitments.
//! Phase 2: four cross-block SumcheckProofMulti (QKV, O-proj, FFN-Y, FFN-M) share one r_k per type.
//!   Algebraic checks happen inline; batch Hyrax opens happen at the end (after mu challenges)
//!   to match the prover's transcript ordering.
//! Global: 5L intermediate matrices verified at shared r_td (inter_batch_open), then 13
//!   cross-block batch opens, then global Lasso.

use crate::field::F;
use crate::lookup::lasso::{
    verify_lasso_multi_committed_outputs, LassoMultiInstance, LassoMultiVerifyingKey,
};
use crate::lookup::quantization::{verify_quantization_batch, QuantizationParams};
use crate::lookup::range::verify_range_batched;
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_verify, hyrax_verify_batch, hyrax_verify_batch_with_eta,
    lagrange_basis, params_from_vars, HyraxBatchAccumulator, HyraxCommitment, HyraxParams,
};
use crate::poly::utils::{combine, compute_eq_evals, mat_to_mle};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{
    eq_poly_eval, verify_sumcheck_cubic_multi_batched, verify_sumcheck_multi_batched,
};
use crate::transcript::{challenge_vec, Transcript};

use crate::attention::attention::{AttentionProvingKey, LinearAttentionInstance};
use crate::attention::layernorm::{
    verify_layernorm, LayerNormIOCommitments, LayerNormVerifyingKey, LAYERNORM_RANGE_BITS,
};
use crate::attention::projection::{
    verify_projection, ProjectionIOCommitments, ProjectionProvingKey, ProjectionVerifyingKey,
};
use crate::ffn::ffn::{FFNInstance, FFNProvingKey, FFNVerifyingKey};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::Field;
use std::ops::AddAssign;

use crate::prover::{TransformerModelProof, TransformerModelVerifyingKey};

// ---------------------------------------------------------------------------
// Verifying Key
// ---------------------------------------------------------------------------

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
    pub ffn_activation_quant: QuantizationParams,
    pub qk_activation_quant: QuantizationParams,
    pub q_pk: ProjectionProvingKey,
    pub k_pk: ProjectionProvingKey,
    pub v_pk: ProjectionProvingKey,
    pub o_pk: ProjectionProvingKey,
    pub ffn_pk: FFNProvingKey,
    pub attn_pk: AttentionProvingKey,
    // Sandwich-norm LayerNorms.  attn_out_norm is in the proof pipeline as of
    // the soundness fix (production range proofs are 64-bit, wide enough for
    // typical model dimensions).
    pub q_norm_vk: LayerNormVerifyingKey,
    pub k_norm_vk: LayerNormVerifyingKey,
    pub attn_out_norm_vk: LayerNormVerifyingKey,
}

// ---------------------------------------------------------------------------
// Cryptographic Helper: Homomorphic Addition
// ---------------------------------------------------------------------------

pub fn add_commitments(a: &HyraxCommitment, b: &HyraxCommitment) -> HyraxCommitment {
    assert_eq!(
        a.row_coms.len(),
        b.row_coms.len(),
        "Commitment dimensions must match"
    );
    let mut result_coms = Vec::with_capacity(a.row_coms.len());
    for (pt_a, pt_b) in a.row_coms.iter().zip(b.row_coms.iter()) {
        let mut sum_proj = pt_a.into_group();
        sum_proj.add_assign(&pt_b.into_group());
        result_coms.push(sum_proj.into_affine());
    }
    HyraxCommitment {
        row_coms: result_coms,
        nu: a.nu,
        sigma: a.sigma,
    }
}

fn powers_of(base: F, n: usize) -> Vec<F> {
    let mut v = Vec::with_capacity(n);
    let mut cur = F::from(1u64);
    for _ in 0..n {
        v.push(cur);
        cur *= base;
    }
    v
}

fn eq_evals_msb(r: &[F], n: usize) -> Vec<F> {
    let r_rev: Vec<F> = r.iter().rev().copied().collect();
    compute_eq_evals(&r_rev, n)
}

fn suffix_eq_evals_msb(r: &[F], len: usize) -> Vec<F> {
    let padded = len.next_power_of_two();
    let eq = eq_evals_msb(r, len);
    let mut suffix = vec![F::ZERO; padded];
    let mut running = F::ZERO;
    for i in (0..len).rev() {
        running += eq[i];
        suffix[i] = running;
    }
    suffix
}

fn commit_public_mat(mat: &[Vec<F>], rows: usize, cols: usize) -> Result<HyraxCommitment, String> {
    if mat.len() != rows || mat.iter().any(|r| r.len() != cols) {
        return Err(format!(
            "public matrix dimension mismatch: got {}x{}, expected {}x{}",
            mat.len(),
            mat.first().map(|r| r.len()).unwrap_or(0),
            rows,
            cols
        ));
    }
    let mle = mat_to_mle(mat, rows, cols);
    let vars = rows.next_power_of_two().trailing_zeros() as usize
        + cols.next_power_of_two().trailing_zeros() as usize;
    let (nu, _, params) = params_from_vars(vars);
    Ok(hyrax_commit(&mle.evaluations, nu, &params))
}

fn commitments_equal(a: &HyraxCommitment, b: &HyraxCommitment) -> bool {
    a.nu == b.nu && a.sigma == b.sigma && a.row_coms == b.row_coms
}

fn absorb_index_vectors(transcript: &mut Transcript, label: &[u8], vectors: &[&[usize]]) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(vectors.len() as u64).to_le_bytes());
    for v in vectors {
        bytes.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for &idx in *v {
            bytes.extend_from_slice(&(idx as u64).to_le_bytes());
        }
    }
    transcript.append_bytes(label, &bytes);
}

fn eval_query_indices(
    indices: &[usize],
    rows: usize,
    cols: usize,
    point: &[F],
) -> Result<F, String> {
    if indices.len() != rows * cols {
        return Err(format!(
            "lookup index length mismatch: got {}, expected {}",
            indices.len(),
            rows * cols
        ));
    }
    let num_vars = rows.next_power_of_two().trailing_zeros() as usize
        + cols.next_power_of_two().trailing_zeros() as usize;
    if point.len() != num_vars {
        return Err(format!(
            "lookup index evaluation point length mismatch: got {}, expected {}",
            point.len(),
            num_vars
        ));
    }
    let evals: Vec<F> = indices.iter().map(|&idx| F::from(idx as u64)).collect();
    Ok(DenseMLPoly::from_vec_padded(evals).evaluate(point))
}

fn verify_query_indices_bound_batch(
    label: &str,
    indices: &[&[usize]],
    rows: usize,
    cols: usize,
    point: &[F],
    proof: &crate::pcs::HyraxProof,
    params: &HyraxParams,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let count = indices.len();
    if count == 0 {
        return Ok(());
    }

    let sigma = params.sigma;
    let nu = point
        .len()
        .checked_sub(sigma)
        .ok_or_else(|| format!("{label} lookup binding point shorter than sigma"))?;
    let num_cols = 1usize << sigma;
    let num_rows = 1usize << nu;
    let padded_len = num_rows * num_cols;
    if proof.w_prime.len() != num_cols {
        return Err(format!(
            "{label} lookup binding proof length mismatch: got {}, expected {}",
            proof.w_prime.len(),
            num_cols
        ));
    }
    let expected_unpadded_len = rows * cols;
    for idx in indices {
        if idx.len() != expected_unpadded_len {
            return Err(format!(
                "{label} lookup index length mismatch: got {}, expected {}",
                idx.len(),
                expected_unpadded_len
            ));
        }
    }

    let eta = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_pows = powers_of(eta, count);
    let r_l_rev: Vec<F> = point[..nu].iter().rev().copied().collect();
    let r_r_rev: Vec<F> = point[nu..].iter().rev().copied().collect();
    let l_vec = lagrange_basis(&r_l_rev);
    let r_vec = lagrange_basis(&r_r_rev);

    let mut expected_w_prime = vec![F::ZERO; num_cols];
    for (k, idx) in indices.iter().enumerate() {
        let eta_k = eta_pows[k];
        for flat in 0..padded_len {
            let value = if flat < idx.len() {
                F::from(idx[flat] as u64)
            } else {
                F::ZERO
            };
            if value == F::ZERO {
                continue;
            }
            expected_w_prime[flat % num_cols] += eta_k * l_vec[flat / num_cols] * value;
        }
    }
    if proof.w_prime != expected_w_prime {
        return Err(format!(
            "{label} lookup index binding opening vector mismatch"
        ));
    }

    let inner: F = r_vec
        .iter()
        .zip(proof.w_prime.iter())
        .map(|(&r, &w)| r * w)
        .sum();
    let expected_inner = indices
        .iter()
        .zip(eta_pows.iter())
        .map(|(idx, eta_k)| eval_query_indices(idx, rows, cols, point).map(|v| *eta_k * v))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum();
    if inner != expected_inner {
        return Err(format!(
            "{label} lookup index binding inner product mismatch"
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Model Verifier (E2E)
// ---------------------------------------------------------------------------

pub fn verify(
    proof: &TransformerModelProof,
    vk: &TransformerModelVerifyingKey,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    public_x_in: &[Vec<F>],
    public_logits: &[Vec<F>],
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> Result<(), String> {
    use std::time::Instant;

    // 1. Bind initial input
    absorb_com(transcript, b"x_in_com", &proof.x_in_com);

    let t = vk.seq_len;
    let d = vk.d_model;
    let v_vocab = vk.vocab_size;
    if vk.causal != inst_attn.causal {
        return Err(format!(
            "attention mode mismatch: key causal={}, instance causal={}",
            vk.causal, inst_attn.causal
        ));
    }
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let td_num_vars = t_bits + d_bits;
    let (_, _, params_t) = params_from_vars(t_bits);
    let (_, _, params_td) = params_from_vars(td_num_vars);

    let q_vk = &vk.block_vks[0].q_vk;
    let qkvo_in_bits = q_vk.d_in.next_power_of_two().trailing_zeros() as usize;
    let qkvo_out_bits = q_vk.d_out.next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_qkvo_w) = params_from_vars(qkvo_in_bits + qkvo_out_bits);
    let (_, _, params_qkvo_b) = params_from_vars(qkvo_out_bits);

    let lmh_in_bits = vk.lm_head_vk.d_in.next_power_of_two().trailing_zeros() as usize;
    let lmh_out_bits = vk.lm_head_vk.d_out.next_power_of_two().trailing_zeros() as usize;
    let params_lmh_w = params_from_vars(lmh_in_bits + lmh_out_bits).2;
    let params_lmh_b = params_from_vars(lmh_out_bits).2;

    let ln_sig_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let ln_y_n_global = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_range_sig) = params_from_vars(ln_sig_n);
    let (_, _, params_range_y) = params_from_vars(ln_y_n_global);
    let (_, _, params_range_m) = params_from_vars(crate::lookup::range::CHUNK_BITS);

    let d_ff = vk.block_vks[0].ffn_vk.d_ff;
    let f_bits = d_ff.next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_wff) = params_from_vars(f_bits + d_bits);
    let (_, _, params_mff) = params_from_vars(t_bits + f_bits);

    let num_blocks = vk.num_blocks;

    let expected_x_in_com = commit_public_mat(public_x_in, t, d)?;
    if !commitments_equal(&proof.x_in_com, &expected_x_in_com) {
        return Err("public input does not match x_in commitment".into());
    }
    let expected_logits_com = commit_public_mat(public_logits, t, v_vocab)?;
    if !commitments_equal(&proof.logits_com, &expected_logits_com) {
        return Err("public output does not match logits commitment".into());
    }

    let mut ln_acc_t = HyraxBatchAccumulator::new();
    let mut ln_acc_td = HyraxBatchAccumulator::new();
    let proj_acc_w = HyraxBatchAccumulator::new();
    let proj_acc_b = HyraxBatchAccumulator::new();
    let mut lmh_acc_w = HyraxBatchAccumulator::new();
    let mut lmh_acc_b = HyraxBatchAccumulator::new();
    let mut acc_range_sig = HyraxBatchAccumulator::new();
    let mut acc_range_y = HyraxBatchAccumulator::new();
    let mut acc_range_m = HyraxBatchAccumulator::new();
    let mut acc_attn_norm_rem = HyraxBatchAccumulator::new();
    let mut acc_attn_norm_diff = HyraxBatchAccumulator::new();
    let mut acc_attn_norm_m = HyraxBatchAccumulator::new();
    let mut acc_quant_ffn = HyraxBatchAccumulator::new();
    let mut acc_quant_unused = HyraxBatchAccumulator::new();
    let mut acc_quant_m = HyraxBatchAccumulator::new();
    // inter_acc: per-block v_attn opens (different eval point per block)
    let inter_acc = HyraxBatchAccumulator::new();

    // =========================================================================
    // 2. Phase 1: verify range proofs + LN1 + LN2 for all blocks
    // =========================================================================
    let mut current_x_com = proof.x_in_com.clone();
    let ln_sigma_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let ln_y_n = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
    let mut ln_range_proofs = Vec::with_capacity(10 * num_blocks + 2);
    let mut ln_range_num_vars = Vec::with_capacity(10 * num_blocks + 2);
    for bp in &proof.block_proofs {
        ln_range_proofs.push(&bp.ln1_proof.sigma_range_proof);
        ln_range_num_vars.push(ln_sigma_n);
        ln_range_proofs.push(&bp.ln1_proof.y_range_proof);
        ln_range_num_vars.push(ln_y_n);
        ln_range_proofs.push(&bp.ln2_proof.sigma_range_proof);
        ln_range_num_vars.push(ln_sigma_n);
        ln_range_proofs.push(&bp.ln2_proof.y_range_proof);
        ln_range_num_vars.push(ln_y_n);
        // Sandwich-norm range witnesses: order matches prover's push order
        // (q_norm, k_norm, attn_out_norm).
        ln_range_proofs.push(&bp.q_norm_proof.sigma_range_proof);
        ln_range_num_vars.push(ln_sigma_n);
        ln_range_proofs.push(&bp.q_norm_proof.y_range_proof);
        ln_range_num_vars.push(ln_y_n);
        ln_range_proofs.push(&bp.k_norm_proof.sigma_range_proof);
        ln_range_num_vars.push(ln_sigma_n);
        ln_range_proofs.push(&bp.k_norm_proof.y_range_proof);
        ln_range_num_vars.push(ln_y_n);
        ln_range_proofs.push(&bp.attn_out_norm_proof.sigma_range_proof);
        ln_range_num_vars.push(ln_sigma_n);
        ln_range_proofs.push(&bp.attn_out_norm_proof.y_range_proof);
        ln_range_num_vars.push(ln_y_n);
    }
    ln_range_proofs.push(&proof.final_ln_proof.sigma_range_proof);
    ln_range_num_vars.push(ln_sigma_n);
    ln_range_proofs.push(&proof.final_ln_proof.y_range_proof);
    ln_range_num_vars.push(ln_y_n);

    let _t0 = Instant::now();
    let (ln_range_r_vs, _) = verify_range_batched(
        &ln_range_proofs,
        &proof.ln_range_m,
        &ln_range_num_vars,
        LAYERNORM_RANGE_BITS,
        transcript,
        &mut acc_range_sig,
        &mut acc_range_y,
        &mut acc_range_m,
    )?;
    eprintln!(
        "[model] ln_range_batch:{:>8.3}ms",
        _t0.elapsed().as_secs_f64() * 1000.0
    );

    let has_attn_norm = proof
        .block_proofs
        .iter()
        .any(|bp| bp.attn_norm_com.is_some());
    if has_attn_norm {
        let m = proof
            .attn_norm_range_m
            .as_ref()
            .ok_or_else(|| "missing attention normalization range proof".to_string())?;
        if proof.attn_norm_rem_range_proofs.len() != num_blocks
            || proof.attn_norm_diff_range_proofs.len() != num_blocks
        {
            return Err("attention normalization range proof count mismatch".into());
        }
        let mut proofs = Vec::with_capacity(2 * num_blocks);
        let mut num_vars = Vec::with_capacity(2 * num_blocks);
        let n = (t * d).next_power_of_two().trailing_zeros() as usize;
        for i in 0..num_blocks {
            proofs.push(&proof.attn_norm_rem_range_proofs[i]);
            num_vars.push(n);
            proofs.push(&proof.attn_norm_diff_range_proofs[i]);
            num_vars.push(n);
        }
        verify_range_batched(
            &proofs,
            m,
            &num_vars,
            crate::prover::ATTN_NORM_RANGE_BITS,
            transcript,
            &mut acc_attn_norm_rem,
            &mut acc_attn_norm_diff,
            &mut acc_attn_norm_m,
        )?;
    } else if proof.attn_norm_range_m.is_some()
        || !proof.attn_norm_rem_range_proofs.is_empty()
        || !proof.attn_norm_diff_range_proofs.is_empty()
    {
        return Err("unexpected attention normalization range proof".into());
    }

    for i in 0..num_blocks {
        let bp = &proof.block_proofs[i];
        let bvk = &vk.block_vks[i];
        // 10 sub-witnesses per block in the model-level batched range proof:
        //   [ln1_sigma, ln1_y, ln2_sigma, ln2_y,
        //    q_norm_sigma, q_norm_y, k_norm_sigma, k_norm_y,
        //    attn_out_norm_sigma, attn_out_norm_y]
        let rv_base = 10 * i;
        let ln1_sig_rv = &ln_range_r_vs[rv_base];
        let ln1_y_rv = &ln_range_r_vs[rv_base + 1];
        let ln2_sig_rv = &ln_range_r_vs[rv_base + 2];
        let ln2_y_rv = &ln_range_r_vs[rv_base + 3];
        let q_norm_sig_rv = &ln_range_r_vs[rv_base + 4];
        let q_norm_y_rv = &ln_range_r_vs[rv_base + 5];
        let k_norm_sig_rv = &ln_range_r_vs[rv_base + 6];
        let k_norm_y_rv = &ln_range_r_vs[rv_base + 7];
        let aon_sig_rv = &ln_range_r_vs[rv_base + 8];
        let aon_y_rv = &ln_range_r_vs[rv_base + 9];

        // LN1
        let ln1_io = LayerNormIOCommitments {
            x_com: current_x_com.clone(),
            y_com: Some(bp.x_norm1_com.clone()),
        };
        let _t0 = Instant::now();
        verify_layernorm(
            &bp.ln1_proof,
            &ln1_io,
            &bvk.ln1_vk,
            ln1_sig_rv,
            ln1_y_rv,
            transcript,
            &mut ln_acc_t,
            &mut ln_acc_td,
        )?;
        eprintln!(
            "[block {}] ln1:{:>8.3}ms",
            i,
            _t0.elapsed().as_secs_f64() * 1000.0
        );

        absorb_com(transcript, b"q_com", &bp.q_com);
        absorb_com(transcript, b"k_com", &bp.k_com);
        absorb_com(transcript, b"v_com", &bp.v_com);
        if let Some(ref c) = bp.attn_norm_com {
            absorb_com(transcript, b"attn_norm_com", c);
        }
        if let Some(ref c) = bp.attn_num_com {
            absorb_com(transcript, b"attn_num_com", c);
        }
        if let Some(ref c) = bp.attn_z_com {
            absorb_com(transcript, b"attn_z_com", c);
        }
        if let Some(ref c) = bp.attn_rem_com {
            absorb_com(transcript, b"attn_rem_com", c);
        }
        if let Some(ref c) = bp.attn_diff_com {
            absorb_com(transcript, b"attn_diff_com", c);
        }
        absorb_com(transcript, b"out_attn_com", &bp.out_attn_com);

        // q_norm: x = q_raw (q_com), y = q_n (q_norm_y_com).
        let q_norm_io = LayerNormIOCommitments {
            x_com: bp.q_com.clone(),
            y_com: Some(bp.q_norm_y_com.clone()),
        };
        verify_layernorm(
            &bp.q_norm_proof,
            &q_norm_io,
            &bvk.q_norm_vk,
            q_norm_sig_rv,
            q_norm_y_rv,
            transcript,
            &mut ln_acc_t,
            &mut ln_acc_td,
        )?;
        absorb_com(transcript, b"q_norm_y_com", &bp.q_norm_y_com);

        // k_norm: x = k_raw (k_com), y = k_n (k_norm_y_com).
        let k_norm_io = LayerNormIOCommitments {
            x_com: bp.k_com.clone(),
            y_com: Some(bp.k_norm_y_com.clone()),
        };
        verify_layernorm(
            &bp.k_norm_proof,
            &k_norm_io,
            &bvk.k_norm_vk,
            k_norm_sig_rv,
            k_norm_y_rv,
            transcript,
            &mut ln_acc_t,
            &mut ln_acc_td,
        )?;
        absorb_com(transcript, b"k_norm_y_com", &bp.k_norm_y_com);

        // attn_out_norm: x = out_attn (out_attn_com), y = post-norm output.
        let attn_out_norm_io = LayerNormIOCommitments {
            x_com: bp.out_attn_com.clone(),
            y_com: Some(bp.attn_out_norm_y_com.clone()),
        };
        verify_layernorm(
            &bp.attn_out_norm_proof,
            &attn_out_norm_io,
            &bvk.attn_out_norm_vk,
            aon_sig_rv,
            aon_y_rv,
            transcript,
            &mut ln_acc_t,
            &mut ln_acc_td,
        )?;
        absorb_com(
            transcript,
            b"attn_out_norm_y_com",
            &bp.attn_out_norm_y_com,
        );

        // Residual flows through the normalized attention output.
        let x_mid_com = add_commitments(&current_x_com, &bp.attn_out_norm_y_com);

        // LN2
        let ln2_io = LayerNormIOCommitments {
            x_com: x_mid_com.clone(),
            y_com: Some(bp.x_norm2_com.clone()),
        };
        let _t0 = Instant::now();
        verify_layernorm(
            &bp.ln2_proof,
            &ln2_io,
            &bvk.ln2_vk,
            ln2_sig_rv,
            ln2_y_rv,
            transcript,
            &mut ln_acc_t,
            &mut ln_acc_td,
        )?;
        eprintln!(
            "[block {}] ln2:{:>8.3}ms",
            i,
            _t0.elapsed().as_secs_f64() * 1000.0
        );

        absorb_com(transcript, b"y_com", &bp.out_ffn_com);

        let next_x_com = add_commitments(&x_mid_com, &bp.out_ffn_com);
        current_x_com = next_x_com;
    }

    // =========================================================================
    // 3. Derive global r_td after ALL blocks' Phase 1
    // =========================================================================
    let r_td = challenge_vec(transcript, td_num_vars, b"gkr_r_td");
    let r_t = r_td[..t_bits].to_vec();
    let r_out = r_td[t_bits..].to_vec();

    // =========================================================================
    // 4. Batch QKV (sumcheck only; batch opens deferred to step 13)
    // =========================================================================
    let _tqkv = Instant::now();

    let mut pb_lambda: Vec<F> = Vec::with_capacity(num_blocks);
    let mut pb_mu: Vec<F> = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let bvk = &vk.block_vks[i];
        let bp = &proof.block_proofs[i];

        absorb_com(transcript, b"qkv_w_q_com", &bvk.q_vk.w_com);
        absorb_com(transcript, b"qkv_w_k_com", &bvk.k_vk.w_com);
        absorb_com(transcript, b"qkv_w_v_com", &bvk.v_vk.w_com);
        transcript.append_field(b"qkv_alpha_q", &bvk.q_vk.alpha);
        transcript.append_field(b"qkv_alpha_k", &bvk.k_vk.alpha);
        transcript.append_field(b"qkv_alpha_v", &bvk.v_vk.alpha);
        absorb_com(transcript, b"qkv_bias_q_com", &bvk.q_vk.bias_com);
        absorb_com(transcript, b"qkv_bias_k_com", &bvk.k_vk.bias_com);
        absorb_com(transcript, b"qkv_bias_v_com", &bvk.v_vk.bias_com);

        let lambda: F = transcript.challenge_field(b"qkv_lambda");
        let mu: F = transcript.challenge_field(b"qkv_mu");

        transcript.append_field(b"qkv_q_eval", &bp.q_eval);
        transcript.append_field(b"qkv_k_eval", &bp.k_eval);
        transcript.append_field(b"qkv_v_eval", &bp.v_eval_rtd);

        pb_lambda.push(lambda);
        pb_mu.push(mu);
    }

    let eta_qkv: F = transcript.challenge_field(b"batch_eta_qkv");
    let weights_qkv = powers_of(eta_qkv, num_blocks);

    let claim_qkv: F = (0..num_blocks)
        .map(|i| {
            let bp = &proof.block_proofs[i];
            let target = pb_lambda[i] * (bp.q_eval - bp.qkv_bias_q_eval)
                + pb_mu[i] * (bp.k_eval - bp.qkv_bias_k_eval)
                + (bp.v_eval_rtd - bp.qkv_bias_v_eval);
            weights_qkv[i] * target
        })
        .sum();

    let (r_k_qkv, _) = verify_sumcheck_multi_batched(
        &proof.batch_qkv,
        &weights_qkv,
        claim_qkv,
        d_bits,
        transcript,
    )?;

    // Algebraic check: g_eval == lambda*alpha_q*wq + mu*alpha_k*wk + alpha_v*wv
    for i in 0..num_blocks {
        let bvk = &vk.block_vks[i];
        let bp = &proof.block_proofs[i];
        let g_reconstructed = pb_lambda[i] * bvk.q_vk.alpha * bp.qkv_w_q_eval
            + pb_mu[i] * bvk.k_vk.alpha * bp.qkv_w_k_eval
            + bvk.v_vk.alpha * bp.qkv_w_v_eval;
        if proof.batch_qkv.final_evals_g[i] != g_reconstructed {
            return Err(format!("Block {i}: QKV g_eval algebraic check failed"));
        }
    }

    eprintln!(
        "[model] batch_qkv:{:>8.3}ms",
        _tqkv.elapsed().as_secs_f64() * 1000.0
    );

    // =========================================================================
    // 5. Batch O-proj (sumcheck only; batch opens deferred to step 13)
    // =========================================================================
    let _to = Instant::now();

    for i in 0..num_blocks {
        let bvk = &vk.block_vks[i];
        let bp = &proof.block_proofs[i];

        absorb_com(transcript, b"w_com", &bvk.o_vk.w_com);
        transcript.append_field(b"alpha", &bvk.o_vk.alpha);
        absorb_com(transcript, b"bias_com", &bvk.o_vk.bias_com);
        transcript.append_field(b"claimed_y", &bp.out_attn_eval);
    }

    let eta_oproj: F = transcript.challenge_field(b"batch_eta_oproj");
    let weights_oproj = powers_of(eta_oproj, num_blocks);

    let claim_oproj: F = (0..num_blocks)
        .map(|i| {
            let bp = &proof.block_proofs[i];
            weights_oproj[i] * (bp.out_attn_eval - bp.oproj_bias_o_eval)
        })
        .sum();

    let (r_k_o, _) = verify_sumcheck_multi_batched(
        &proof.batch_oproj,
        &weights_oproj,
        claim_oproj,
        d_bits,
        transcript,
    )?;

    // Algebraic check: final_evals_g[i] == bp.oproj_w_o_eval
    for i in 0..num_blocks {
        let bp = &proof.block_proofs[i];
        if proof.batch_oproj.final_evals_g[i] != bp.oproj_w_o_eval {
            return Err(format!("Block {i}: O-proj g_eval algebraic check failed"));
        }
    }

    eprintln!(
        "[model] batch_oproj:{:>8.3}ms",
        _to.elapsed().as_secs_f64() * 1000.0
    );

    // =========================================================================
    // 6. Cross-block batch attention sumchecks
    // =========================================================================
    let _tattn = Instant::now();

    // 6a. Absorb phi_q_com, phi_k_com per block (mirrors prover step 6a)
    for i in 0..num_blocks {
        let bp = &proof.block_proofs[i];
        absorb_com(transcript, b"phi_q_com", &bp.attn_phi_q_com);
        absorb_com(transcript, b"phi_k_com", &bp.attn_phi_k_com);
        if bp.causal_context_com.is_some() {
            return Err(format!(
                "Block {i}: causal_context_com is no longer used (GKR chain step 2)"
            ));
        }
    }

    // 6b. Verify batch out sumcheck
    for i in 0..num_blocks {
        transcript.append_field(b"attn_out_eval", &proof.block_proofs[i].attn_out_eval);
    }
    let eta_attn_out: F = transcript.challenge_field(b"batch_eta_attn_out");
    let weights_attn_out = powers_of(eta_attn_out, num_blocks);
    let claim_attn_out: F = (0..num_blocks)
        .map(|i| weights_attn_out[i] * proof.block_proofs[i].attn_out_eval)
        .sum();
    // Causal mode runs a degree-3 sumcheck (eq(r_t, ·) folded as the third
    // multiplicand) over (t, k); non-causal runs a degree-2 sumcheck over k only.
    let (batch_r_attn_out, attn_out_final_evals_f, attn_out_final_evals_g) = if inst_attn.causal {
        if proof.batch_attn_out.is_some() {
            return Err("unexpected batch_attn_out in causal proof".to_string());
        }
        let sc = proof
            .batch_attn_out_causal
            .as_ref()
            .ok_or_else(|| "missing batch_attn_out_causal".to_string())?;
        let (r, _) = verify_sumcheck_cubic_multi_batched(
            sc,
            &weights_attn_out,
            claim_attn_out,
            t_bits + d_bits,
            transcript,
        )
        .map_err(|e| format!("batch_attn_out_causal: {e}"))?;
        // Verifier-computable check: h(r_final) must equal the MLE of
        // eq(r_t, ·) padded constantly along k, evaluated at r_final, which is
        // simply eq(r_t, r_final[..t_bits]).
        let r_final_t = &r[..t_bits];
        let expected_eq = eq_poly_eval(&r_t, r_final_t);
        for (i, &h_val) in sc.final_evals_h.iter().enumerate() {
            if h_val != expected_eq {
                return Err(format!(
                    "Block {i}: causal batch_attn_out final_evals_h does not match eq(r_t, r_final_t)"
                ));
            }
        }
        (r, sc.final_evals_f.clone(), sc.final_evals_g.clone())
    } else {
        if proof.batch_attn_out_causal.is_some() {
            return Err("unexpected batch_attn_out_causal in non-causal proof".to_string());
        }
        let sc = proof
            .batch_attn_out
            .as_ref()
            .ok_or_else(|| "missing batch_attn_out".to_string())?;
        let (r, _) = verify_sumcheck_multi_batched(
            sc,
            &weights_attn_out,
            claim_attn_out,
            d_bits,
            transcript,
        )
        .map_err(|e| format!("batch_attn_out: {e}"))?;
        (r, sc.final_evals_f.clone(), sc.final_evals_g.clone())
    };

    // Algebraic check: in legacy unnormalized mode the attention claim is
    // derived from the O-proj leaf. Normalized mode is checked by the
    // attention normalization proof below.
    for i in 0..num_blocks {
        let bp = &proof.block_proofs[i];
        if bp.attn_norm_com.is_some() {
            continue;
        }
        let alpha_o = vk.block_vks[i].o_vk.alpha;
        let expected_out = if alpha_o == F::from(0u64) {
            F::from(0u64)
        } else {
            proof.batch_oproj.final_evals_f[i] * alpha_o.inverse().unwrap()
        };
        if bp.attn_out_eval != expected_out {
            return Err(format!(
                "Block {i}: attn_out_eval mismatch with o_proj leaf"
            ));
        }
    }

    // 6c. Verify context relation. With the GKR chain, the causal CTX-relation
    // is bound at the OUT sumcheck terminal point, so no fresh prefix
    // challenges or prefix-point Hyrax open are needed: the per-block claim
    // for CTX is OUT's final_evals_g[i].
    let attn_ctx_claims: Vec<F> = if inst_attn.causal {
        if !proof.causal_ctx_prefix_evals.is_empty() {
            return Err(
                "unexpected causal_ctx_prefix_evals in causal proof (GKR-chained)".to_string(),
            );
        }
        if proof.causal_ctx_prefix_batch_open.is_some() {
            return Err(
                "unexpected causal_ctx_prefix_batch_open in causal proof (GKR-chained)".to_string(),
            );
        }
        attn_out_final_evals_g.clone()
    } else {
        if !proof.causal_ctx_prefix_evals.is_empty() {
            return Err("unexpected causal_ctx_prefix_evals in non-causal proof".to_string());
        }
        attn_out_final_evals_g.clone()
    };
    if !inst_attn.causal {
        for claim in &attn_ctx_claims {
            transcript.append_field(b"attn_ctx_eval", claim);
        }
    }
    let eta_attn_ctx: F = transcript.challenge_field(b"batch_eta_attn_ctx");
    let weights_attn_ctx = powers_of(eta_attn_ctx, num_blocks);
    let claim_attn_ctx: F = (0..num_blocks)
        .map(|i| weights_attn_ctx[i] * attn_ctx_claims[i])
        .sum();
    let (batch_r_attn_ctx, attn_ctx_final_evals_f, attn_ctx_final_evals_g) = if inst_attn.causal {
        if proof.batch_attn_ctx.is_some() {
            return Err("unexpected batch_attn_ctx in causal proof".to_string());
        }
        let sc = proof
            .batch_attn_ctx_causal
            .as_ref()
            .ok_or_else(|| "missing batch_attn_ctx_causal".to_string())?;
        let (r, _) = verify_sumcheck_cubic_multi_batched(
            sc,
            &weights_attn_ctx,
            claim_attn_ctx,
            t_bits + 2 * d_bits,
            transcript,
        )
        .map_err(|e| format!("batch_attn_ctx_causal: {e}"))?;
        // GKR chain: the prefix point is implied by the OUT-sumcheck terminal
        // — (r_final_t, r_final_k) for the (t, a) axes and r_k_o for the b
        // axis, so suffix·eq_a·eq_b factors are computed off those.
        let prefix_t: &[F] = &batch_r_attn_out[..t_bits];
        let prefix_a: &[F] = &batch_r_attn_out[t_bits..t_bits + d_bits];
        let prefix_b: &[F] = &r_k_o[..];
        let r_s = &r[..t_bits];
        let r_a = &r[t_bits..t_bits + d_bits];
        let r_b = &r[t_bits + d_bits..];
        let suffix_at_rs =
            DenseMLPoly::from_vec_padded(suffix_eq_evals_msb(prefix_t, t)).evaluate(r_s);
        let eq_a_at_ra = DenseMLPoly::from_vec_padded(eq_evals_msb(prefix_a, d)).evaluate(r_a);
        let eq_b_at_rb = DenseMLPoly::from_vec_padded(eq_evals_msb(prefix_b, d)).evaluate(r_b);
        let expected_h = suffix_at_rs * eq_a_at_ra * eq_b_at_rb;
        for (i, &h_val) in sc.final_evals_h.iter().enumerate() {
            if h_val != expected_h {
                return Err(format!(
                    "Block {i}: causal batch_attn_ctx h leaf does not match suffix·eq_a·eq_b"
                ));
            }
        }
        (r, sc.final_evals_f.clone(), sc.final_evals_g.clone())
    } else {
        if proof.batch_attn_ctx_causal.is_some() {
            return Err("unexpected batch_attn_ctx_causal in non-causal proof".to_string());
        }
        let sc = proof
            .batch_attn_ctx
            .as_ref()
            .ok_or_else(|| "missing batch_attn_ctx".to_string())?;
        let (r, _) = verify_sumcheck_multi_batched(
            sc,
            &weights_attn_ctx,
            claim_attn_ctx,
            t_bits,
            transcript,
        )
        .map_err(|e| format!("batch_attn_ctx: {e}"))?;
        (r, sc.final_evals_f.clone(), sc.final_evals_g.clone())
    };

    eprintln!(
        "[model] batch_attn:{:>8.3}ms",
        _tattn.elapsed().as_secs_f64() * 1000.0
    );

    let attn_norm_r = if has_attn_norm {
        let r_eq = challenge_vec(transcript, td_num_vars, b"attn_norm_r");
        let lambda = transcript.challenge_field::<F>(b"attn_norm_lambda");
        let mut weights = Vec::with_capacity(7 * num_blocks);
        for _ in 0..num_blocks {
            weights.extend(vec![
                F::from(crate::prover::ATTN_NORM_SCALE),
                -F::ONE,
                -F::ONE,
                lambda,
                -lambda,
                -lambda,
                -lambda,
            ]);
        }
        let sc = proof
            .attn_norm_sumcheck
            .as_ref()
            .ok_or_else(|| "missing attention normalization sumcheck".to_string())?;
        let (r_sc, _) =
            verify_sumcheck_cubic_multi_batched(sc, &weights, F::ZERO, td_num_vars, transcript)?;
        let eq_eval = eq_poly_eval(&r_sc, &r_eq);
        for f in &sc.final_evals_f {
            if *f != eq_eval {
                return Err("attention normalization eq leaf mismatch".into());
            }
        }
        Some(r_sc)
    } else {
        if proof.attn_norm_sumcheck.is_some() {
            return Err("unexpected attention normalization sumcheck".into());
        }
        None
    };

    let (attn_z_phi_q_point, attn_z_phi_k_point, attn_z_phi_q_evals, attn_z_phi_k_evals) =
        if let (true, Some(ref r_norm), Some(ref norm_sc)) = (
            has_attn_norm,
            attn_norm_r.as_ref(),
            proof.attn_norm_sumcheck.as_ref(),
        ) {
            let r_norm_t = r_norm[..t_bits].to_vec();
            let z_claims: Vec<F> = (0..num_blocks)
                .map(|i| norm_sc.final_evals_g[7 * i + 2])
                .collect();
            let eta_z = transcript.challenge_field::<F>(b"attn_z_eta");
            let weights_z = powers_of(eta_z, num_blocks);
            let claim_z: F = weights_z
                .iter()
                .zip(z_claims.iter())
                .map(|(w, z)| *w * *z)
                .sum();
            if inst_attn.causal {
                if proof.attn_z_sumcheck.is_some() || proof.attn_z_ksum_sumcheck.is_some() {
                    return Err("unexpected non-causal denominator proof in causal mode".into());
                }
                let sc = proof
                    .attn_z_causal_sumcheck
                    .as_ref()
                    .ok_or_else(|| "missing causal attention denominator proof".to_string())?;
                let (r_z, _) = verify_sumcheck_cubic_multi_batched(
                    sc,
                    &weights_z,
                    claim_z,
                    2 * t_bits + d_bits,
                    transcript,
                )?;
                let r_i = r_z[..t_bits].to_vec();
                let r_s = r_z[t_bits..2 * t_bits].to_vec();
                let r_a = r_z[2 * t_bits..].to_vec();
                (
                    Some(combine(&r_i, &r_a)),
                    Some(combine(&r_s, &r_a)),
                    sc.final_evals_g.clone(),
                    sc.final_evals_h.clone(),
                )
            } else {
                if proof.attn_z_causal_sumcheck.is_some() {
                    return Err("unexpected causal denominator proof in non-causal mode".into());
                }
                let sc = proof
                    .attn_z_sumcheck
                    .as_ref()
                    .ok_or_else(|| "missing attention denominator proof".to_string())?;
                let (r_a, _) =
                    verify_sumcheck_multi_batched(sc, &weights_z, claim_z, d_bits, transcript)?;
                let ksum_claims = sc.final_evals_g.clone();
                let claim_ksum: F = weights_z
                    .iter()
                    .zip(ksum_claims.iter())
                    .map(|(w, v)| *w * *v)
                    .sum();
                let sc_k = proof
                    .attn_z_ksum_sumcheck
                    .as_ref()
                    .ok_or_else(|| "missing attention k-sum denominator proof".to_string())?;
                let (r_s, _) = verify_sumcheck_multi_batched(
                    sc_k, &weights_z, claim_ksum, t_bits, transcript,
                )?;
                (
                    Some(combine(&r_norm_t, &r_a)),
                    Some(combine(&r_s, &r_a)),
                    sc.final_evals_f.clone(),
                    sc_k.final_evals_g.clone(),
                )
            }
        } else {
            if proof.attn_z_sumcheck.is_some()
                || proof.attn_z_ksum_sumcheck.is_some()
                || proof.attn_z_causal_sumcheck.is_some()
            {
                return Err("unexpected attention denominator proof".into());
            }
            (None, None, Vec::new(), Vec::new())
        };

    // =========================================================================
    // 7. Per-block FFN: Lasso + M absorb
    // =========================================================================
    for i in 0..num_blocks {
        let bvk = &vk.block_vks[i];
        let bp = &proof.block_proofs[i];

        absorb_com(transcript, b"w1_com", &bvk.ffn_vk.w1_com);
        absorb_com(transcript, b"w2_com", &bvk.ffn_vk.w2_com);

        absorb_com(transcript, b"m_com", &bp.ffn_m_com);
    }
    let _tffn_lasso = Instant::now();
    if proof.ffn_lasso_proof.all_query_indices.len() != num_blocks {
        return Err(format!(
            "FFN Lasso query index count mismatch: got {}, expected {}",
            proof.ffn_lasso_proof.all_query_indices.len(),
            num_blocks
        ));
    }
    let ffn_lasso_instances = LassoMultiInstance {
        instances: (0..num_blocks)
            .map(|_| inst_ffn.activation_lasso.clone())
            .collect(),
    };
    let ffn_instance_table_coms: Vec<Vec<HyraxCommitment>> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.ffn_vk.activation_lasso_vk.table_coms.clone())
        .collect();
    let ffn_instance_to_group =
        crate::lookup::lasso::derive_instance_groups(&ffn_instance_table_coms);
    let ffn_lasso_vk = LassoMultiVerifyingKey {
        instance_table_coms: ffn_instance_table_coms,
        instance_to_group: ffn_instance_to_group,
    };
    let ffn_output_coms: Vec<(HyraxCommitment, usize)> = proof
        .block_proofs
        .iter()
        .map(|bp| (bp.ffn_a_com.clone(), t_bits + f_bits))
        .collect();
    verify_lasso_multi_committed_outputs(
        &proof.ffn_lasso_proof,
        &ffn_lasso_instances,
        &ffn_lasso_vk,
        &ffn_output_coms,
        transcript,
        lasso_params,
    )
    .map_err(|e| format!("FFN global Lasso: {e}"))?;
    let ffn_index_refs: Vec<&[usize]> = proof
        .ffn_lasso_proof
        .all_query_indices
        .iter()
        .map(|v| v.as_slice())
        .collect();
    absorb_index_vectors(transcript, b"ffn_lasso_indices", &ffn_index_refs);
    let ffn_raw_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.ffn_m_com.clone())
        .collect();
    verify_quantization_batch(
        b"ffn_quant_rem_com",
        &proof.ffn_quant_proof,
        &ffn_raw_coms,
        &ffn_index_refs,
        inst_ffn.activation_lasso.tables.len(),
        inst_ffn.activation_lasso.bits_per_chunk,
        t,
        d_ff,
        &vk.block_vks[0].ffn_activation_quant,
        transcript,
        &mut acc_quant_ffn,
        &mut acc_quant_unused,
        &mut acc_quant_m,
    )?;
    let ffn_lasso_bind_point = challenge_vec(transcript, t_bits + f_bits, b"ffn_lasso_bind_r");
    let (_, _, params_mff_bind) = params_from_vars(t_bits + f_bits);
    verify_query_indices_bound_batch(
        "FFN",
        &ffn_index_refs,
        t,
        d_ff,
        &ffn_lasso_bind_point,
        &proof.ffn_lasso_bind_open,
        &params_mff_bind,
        transcript,
    )?;
    eprintln!(
        "[model] ffn_lasso:{:>8.3}ms",
        _tffn_lasso.elapsed().as_secs_f64() * 1000.0
    );
    // =========================================================================
    // 8. Batch FFN-Y (sumcheck only)
    // =========================================================================
    let _tffy = Instant::now();

    for i in 0..num_blocks {
        transcript.append_field(b"claim_y", &proof.block_proofs[i].out_ffn_eval);
    }

    let eta_ffn_y: F = transcript.challenge_field(b"batch_eta_ffn_y");
    let weights_ffn_y = powers_of(eta_ffn_y, num_blocks);

    let claim_ffn_y: F = (0..num_blocks)
        .map(|i| weights_ffn_y[i] * proof.block_proofs[i].out_ffn_eval)
        .sum();

    let (r_k_fy, _) = verify_sumcheck_multi_batched(
        &proof.batch_ffn_y,
        &weights_ffn_y,
        claim_ffn_y,
        f_bits,
        transcript,
    )?;

    eprintln!(
        "[model] batch_ffn_y:{:>8.3}ms",
        _tffy.elapsed().as_secs_f64() * 1000.0
    );

    // =========================================================================
    // 9. Batch FFN-M (sumcheck only)
    // =========================================================================
    let _tffm = Instant::now();

    let rx_m = challenge_vec(transcript, t_bits, b"ffn_rx_m");
    let ry_m = challenge_vec(transcript, f_bits, b"ffn_ry_m");

    for i in 0..num_blocks {
        transcript.append_field(b"claim_m", &proof.block_proofs[i].ffn_m_eval);
    }

    let eta_ffn_m: F = transcript.challenge_field(b"batch_eta_ffn_m");
    let weights_ffn_m = powers_of(eta_ffn_m, num_blocks);

    let claim_ffn_m: F = (0..num_blocks)
        .map(|i| weights_ffn_m[i] * proof.block_proofs[i].ffn_m_eval)
        .sum();

    let (r_k_m, _) = verify_sumcheck_multi_batched(
        &proof.batch_ffn_m,
        &weights_ffn_m,
        claim_ffn_m,
        d_bits,
        transcript,
    )?;

    eprintln!(
        "[model] batch_ffn_m:{:>8.3}ms",
        _tffm.elapsed().as_secs_f64() * 1000.0
    );

    // =========================================================================
    // 10. Final LayerNorm
    // =========================================================================
    let _t0 = Instant::now();
    let ln_io = LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: Some(proof.final_ln_out_com.clone()),
    };
    verify_layernorm(
        &proof.final_ln_proof,
        &ln_io,
        &vk.final_ln_vk,
        &ln_range_r_vs[10 * num_blocks],
        &ln_range_r_vs[10 * num_blocks + 1],
        transcript,
        &mut ln_acc_t,
        &mut ln_acc_td,
    )
    .map_err(|e| format!("Final LN: {e}"))?;
    eprintln!(
        "[model] final_ln:{:>8.3}ms",
        _t0.elapsed().as_secs_f64() * 1000.0
    );

    // =========================================================================
    // 11. LM Head
    // =========================================================================
    let _t0 = Instant::now();
    let lm_io = ProjectionIOCommitments {
        x_com: Some(proof.final_ln_out_com.clone()),
    };
    let (lm_y_claim, _) = verify_projection(
        &proof.lm_head_proof,
        &vk.lm_head_vk,
        &lm_io,
        transcript,
        &mut lmh_acc_w,
        &mut lmh_acc_b,
        None,
    )
    .map_err(|e| format!("LM Head: {e}"))?;
    eprintln!(
        "[model] lm_head:{:>8.3}ms",
        _t0.elapsed().as_secs_f64() * 1000.0
    );

    let v_bits = v_vocab.next_power_of_two().trailing_zeros() as usize;
    let (_, _, params_logits) = params_from_vars(t_bits + v_bits);
    hyrax_verify(
        &proof.logits_com,
        lm_y_claim.value,
        &lm_y_claim.point,
        &proof.lm_head_logits_open,
        &params_logits,
    )
    .map_err(|e| format!("Logits commit: {e}"))?;

    // =========================================================================
    // 12. Finalize accumulators (same order as prover's mu challenge loop)
    // =========================================================================
    let _tacc = Instant::now();
    let mu_inter = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_ln_t = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_ln_td = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_proj_w = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_proj_b = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_lmh_w = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_lmh_b = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_rng_sig = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_rng_y = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_rng_m = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_quant_ffn = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_quant_m = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_attn_norm_rem = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_attn_norm_diff = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_attn_norm_m = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let rho_td = transcript.challenge_field_readonly::<F>(b"hyrax_fuse_td");
    let rho_range_m = transcript.challenge_field_readonly::<F>(b"hyrax_fuse_range_m");

    let ((r0, r1), (r2, r3)) = rayon::join(
        || {
            rayon::join(
                || {
                    HyraxBatchAccumulator::finalize_many_with_mus(
                        &params_td,
                        vec![(inter_acc, mu_inter), (ln_acc_td, mu_ln_td)],
                        rho_td,
                    )
                    .map_err(|e| format!("td fused acc: {e}"))
                },
                || {
                    ln_acc_t
                        .finalize_with_mu(&params_t, mu_ln_t)
                        .map_err(|e| format!("ln_acc_t: {e}"))
                },
            )
        },
        || {
            rayon::join(
                || {
                    proj_acc_w
                        .finalize_with_mu(&params_qkvo_w, mu_proj_w)
                        .map_err(|e| format!("proj_acc_w: {e}"))
                },
                || {
                    acc_quant_ffn
                        .finalize_with_mu(&params_mff, mu_quant_ffn)
                        .map_err(|e| format!("acc_quant_ffn: {e}"))
                },
            )
        },
    );
    let ((r4, r5), (r6, r7)) = rayon::join(
        || {
            rayon::join(
                || {
                    proj_acc_b
                        .finalize_with_mu(&params_qkvo_b, mu_proj_b)
                        .map_err(|e| format!("proj_acc_b: {e}"))
                },
                || {
                    lmh_acc_w
                        .finalize_with_mu(&params_lmh_w, mu_lmh_w)
                        .map_err(|e| format!("lmh_acc_w: {e}"))
                },
            )
        },
        || {
            rayon::join(
                || {
                    lmh_acc_b
                        .finalize_with_mu(&params_lmh_b, mu_lmh_b)
                        .map_err(|e| format!("lmh_acc_b: {e}"))
                },
                || {
                    acc_range_sig
                        .finalize_with_mu(&params_range_sig, mu_rng_sig)
                        .map_err(|e| format!("acc_range_sig: {e}"))
                },
            )
        },
    );
    let (r8, r9) = rayon::join(
        || {
            acc_range_y
                .finalize_with_mu(&params_range_y, mu_rng_y)
                .map_err(|e| format!("acc_range_y: {e}"))
        },
        || {
            HyraxBatchAccumulator::finalize_many_with_mus(
                &params_range_m,
                vec![(acc_range_m, mu_rng_m), (acc_quant_m, mu_quant_m)],
                rho_range_m,
            )
            .map_err(|e| format!("range_m fused acc: {e}"))
        },
    );
    let ((r10, r11), r12) = rayon::join(
        || {
            rayon::join(
                || {
                    acc_attn_norm_rem
                        .finalize_with_mu(&params_td, mu_attn_norm_rem)
                        .map_err(|e| format!("acc_attn_norm_rem: {e}"))
                },
                || {
                    acc_attn_norm_diff
                        .finalize_with_mu(&params_td, mu_attn_norm_diff)
                        .map_err(|e| format!("acc_attn_norm_diff: {e}"))
                },
            )
        },
        || {
            acc_attn_norm_m
                .finalize_with_mu(&params_range_m, mu_attn_norm_m)
                .map_err(|e| format!("acc_attn_norm_m: {e}"))
        },
    );
    eprintln!(
        "[model] acc_finalize:{:>8.3}ms",
        _tacc.elapsed().as_secs_f64() * 1000.0
    );
    r0?;
    r1?;
    r2?;
    r3?;
    r4?;
    r5?;
    r6?;
    r7?;
    r8?;
    r9?;
    r10?;
    r11?;
    r12?;

    // =========================================================================
    // 13. Global batch open for 5L intermediate matrices at r_td (inter_batch_open)
    // =========================================================================
    let _t0 = Instant::now();
    let mut all_coms: Vec<HyraxCommitment> = Vec::with_capacity(5 * num_blocks);
    let mut all_evals: Vec<F> = Vec::with_capacity(5 * num_blocks);
    for bp in &proof.block_proofs {
        all_coms.push(bp.q_com.clone());
        all_coms.push(bp.k_com.clone());
        all_coms.push(bp.v_com.clone());
        all_coms.push(bp.out_attn_com.clone());
        all_coms.push(bp.out_ffn_com.clone());
        all_evals.push(bp.q_eval);
        all_evals.push(bp.k_eval);
        all_evals.push(bp.v_eval_rtd);
        all_evals.push(bp.out_attn_eval);
        all_evals.push(bp.out_ffn_eval);
    }
    hyrax_verify_batch(
        &all_coms,
        &all_evals,
        &r_td,
        &proof.inter_batch_open,
        &params_td,
        transcript,
    )
    .map_err(|e| format!("Global inter_batch: {e}"))?;
    eprintln!(
        "[model] inter_batch:{:>8.3}ms",
        _t0.elapsed().as_secs_f64() * 1000.0
    );

    // =========================================================================
    // 14. 13 cross-block weight/activation batch opens
    //     (must be in same order as prover's step 15)
    // =========================================================================
    let _tbatch = Instant::now();

    // x_norm1 at combine(r_t, r_k_qkv)
    let x_norm1_point = combine(&r_t, &r_k_qkv);
    let x_norm1_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.x_norm1_com.clone())
        .collect();
    hyrax_verify_batch(
        &x_norm1_coms,
        &proof.batch_qkv.final_evals_f,
        &x_norm1_point,
        &proof.x_norm1_batch_open,
        &params_td,
        transcript,
    )
    .map_err(|e| format!("x_norm1_batch: {e}"))?;

    // wq/wk/wv at combine(r_k_qkv, r_out)
    let wq_point = combine(&r_k_qkv, &r_out);
    let wq_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.q_vk.w_com.clone())
        .collect();
    let wq_evals: Vec<F> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.qkv_w_q_eval)
        .collect();

    let wk_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.k_vk.w_com.clone())
        .collect();
    let wk_evals: Vec<F> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.qkv_w_k_eval)
        .collect();

    let wv_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.v_vk.w_com.clone())
        .collect();
    let wv_evals: Vec<F> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.qkv_w_v_eval)
        .collect();
    let eta_wq = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_wk = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_wv = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let ((rwq, rwk), rwv) = rayon::join(
        || {
            rayon::join(
                || {
                    hyrax_verify_batch_with_eta(
                        &wq_coms,
                        &wq_evals,
                        &wq_point,
                        &proof.w_q_batch_open,
                        &params_qkvo_w,
                        eta_wq,
                    )
                    .map_err(|e| format!("w_q_batch: {e}"))
                },
                || {
                    hyrax_verify_batch_with_eta(
                        &wk_coms,
                        &wk_evals,
                        &wq_point,
                        &proof.w_k_batch_open,
                        &params_qkvo_w,
                        eta_wk,
                    )
                    .map_err(|e| format!("w_k_batch: {e}"))
                },
            )
        },
        || {
            hyrax_verify_batch_with_eta(
                &wv_coms,
                &wv_evals,
                &wq_point,
                &proof.w_v_batch_open,
                &params_qkvo_w,
                eta_wv,
            )
            .map_err(|e| format!("w_v_batch: {e}"))
        },
    );
    rwq?;
    rwk?;
    rwv?;

    // bias_q/k/v at r_out
    let bq_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.q_vk.bias_com.clone())
        .collect();
    let bq_evals: Vec<F> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.qkv_bias_q_eval)
        .collect();

    let bk_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.k_vk.bias_com.clone())
        .collect();
    let bk_evals: Vec<F> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.qkv_bias_k_eval)
        .collect();

    let bv_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.v_vk.bias_com.clone())
        .collect();
    let bv_evals: Vec<F> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.qkv_bias_v_eval)
        .collect();
    let eta_bq = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_bk = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_bv = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let ((rbq, rbk), rbv) = rayon::join(
        || {
            rayon::join(
                || {
                    hyrax_verify_batch_with_eta(
                        &bq_coms,
                        &bq_evals,
                        &r_out,
                        &proof.bias_q_batch_open,
                        &params_qkvo_b,
                        eta_bq,
                    )
                    .map_err(|e| format!("bias_q_batch: {e}"))
                },
                || {
                    hyrax_verify_batch_with_eta(
                        &bk_coms,
                        &bk_evals,
                        &r_out,
                        &proof.bias_k_batch_open,
                        &params_qkvo_b,
                        eta_bk,
                    )
                    .map_err(|e| format!("bias_k_batch: {e}"))
                },
            )
        },
        || {
            hyrax_verify_batch_with_eta(
                &bv_coms,
                &bv_evals,
                &r_out,
                &proof.bias_v_batch_open,
                &params_qkvo_b,
                eta_bv,
            )
            .map_err(|e| format!("bias_v_batch: {e}"))
        },
    );
    rbq?;
    rbk?;
    rbv?;

    // wo at combine(r_k_o, r_out)
    let wo_point = combine(&r_k_o, &r_out);
    let wo_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.o_vk.w_com.clone())
        .collect();
    let wo_evals: Vec<F> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.oproj_w_o_eval)
        .collect();

    // bias_o at r_out
    let bo_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.o_vk.bias_com.clone())
        .collect();
    let bo_evals: Vec<F> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.oproj_bias_o_eval)
        .collect();
    let eta_wo = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_bo = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let (rwo, rbo) = rayon::join(
        || {
            hyrax_verify_batch_with_eta(
                &wo_coms,
                &wo_evals,
                &wo_point,
                &proof.w_o_batch_open,
                &params_qkvo_w,
                eta_wo,
            )
            .map_err(|e| format!("w_o_batch: {e}"))
        },
        || {
            hyrax_verify_batch_with_eta(
                &bo_coms,
                &bo_evals,
                &r_out,
                &proof.bias_o_batch_open,
                &params_qkvo_b,
                eta_bo,
            )
            .map_err(|e| format!("bias_o_batch: {e}"))
        },
    );
    rwo?;
    rbo?;

    // w2 at combine(r_k_fy, r_out)
    let w2_point = combine(&r_k_fy, &r_out);
    let w2_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.ffn_vk.w2_com.clone())
        .collect();

    // A at combine(r_t, r_k_fy)
    let ffn_a_point = combine(&r_t, &r_k_fy);
    let ffn_a_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.ffn_a_com.clone())
        .collect();

    // w1 at combine(r_k_m, ry_m) — uses same params as prover (params_qkvo_w)
    let w1_point = combine(&r_k_m, &ry_m);
    let w1_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.ffn_vk.w1_com.clone())
        .collect();

    // x_norm2 at combine(rx_m, r_k_m)
    let x_norm2_point = combine(&rx_m, &r_k_m);
    let x_norm2_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.x_norm2_com.clone())
        .collect();

    // ffn_m_com at combine(rx_m, ry_m)
    let ffn_m_point = combine(&rx_m, &ry_m);
    let ffn_m_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.ffn_m_com.clone())
        .collect();
    let ffn_m_evals: Vec<F> = proof.block_proofs.iter().map(|bp| bp.ffn_m_eval).collect();
    let eta_w2 = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_ffn_a = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_w1 = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_x_norm2 = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_ffn_m = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let ((rw2, rffn_a), (rw1, (rxn2, rffn_m))) = rayon::join(
        || {
            rayon::join(
                || {
                    hyrax_verify_batch_with_eta(
                        &w2_coms,
                        &proof.batch_ffn_y.final_evals_g,
                        &w2_point,
                        &proof.w2_batch_open,
                        &params_wff,
                        eta_w2,
                    )
                    .map_err(|e| format!("w2_batch: {e}"))
                },
                || {
                    hyrax_verify_batch_with_eta(
                        &ffn_a_coms,
                        &proof.batch_ffn_y.final_evals_f,
                        &ffn_a_point,
                        &proof.ffn_a_batch_open,
                        &params_mff,
                        eta_ffn_a,
                    )
                    .map_err(|e| format!("ffn_a_batch: {e}"))
                },
            )
        },
        || {
            rayon::join(
                || {
                    hyrax_verify_batch_with_eta(
                        &w1_coms,
                        &proof.batch_ffn_m.final_evals_g,
                        &w1_point,
                        &proof.w1_batch_open,
                        &params_wff,
                        eta_w1,
                    )
                    .map_err(|e| format!("w1_batch: {e}"))
                },
                || {
                    rayon::join(
                        || {
                            hyrax_verify_batch_with_eta(
                                &x_norm2_coms,
                                &proof.batch_ffn_m.final_evals_f,
                                &x_norm2_point,
                                &proof.x_norm2_batch_open,
                                &params_td,
                                eta_x_norm2,
                            )
                            .map_err(|e| format!("x_norm2_batch: {e}"))
                        },
                        || {
                            hyrax_verify_batch_with_eta(
                                &ffn_m_coms,
                                &ffn_m_evals,
                                &ffn_m_point,
                                &proof.ffn_m_com_batch_open,
                                &params_mff,
                                eta_ffn_m,
                            )
                            .map_err(|e| format!("ffn_m_com_batch: {e}"))
                        },
                    )
                },
            )
        },
    );
    rw2?;
    rffn_a?;
    rw1?;
    rxn2?;
    rffn_m?;

    // phi_q at the (t, k) point implied by the attention-out sumcheck.
    // Causal mode folds r_t into the sumcheck so batch_r_attn_out covers both
    // axes; non-causal mode prefixes r_t externally.
    let phi_q_attn_point = if inst_attn.causal {
        batch_r_attn_out.clone()
    } else {
        combine(&r_t, &batch_r_attn_out)
    };
    let phi_q_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.attn_phi_q_com.clone())
        .collect();
    hyrax_verify_batch(
        &phi_q_coms,
        &attn_out_final_evals_f,
        &phi_q_attn_point,
        &proof.phi_q_batch_open,
        &params_td,
        transcript,
    )
    .map_err(|e| format!("phi_q_batch: {e}"))?;

    // GKR chain (step 2): causal_context_com is dropped — neither prefix-point
    // nor terminal-point Hyrax open is needed; the CTX cubic sumcheck binds
    // attn_out_final_evals_g algebraically via phi_k and v.
    let causal_ctx_coms: Option<Vec<HyraxCommitment>> = None;

    let (phi_k_attn_point, phi_k_evals, v_attn_batch_point, v_evals) = if inst_attn.causal {
        if proof.causal_phi_k_prefix_evals.len() != num_blocks
            || proof.causal_v_prefix_evals.len() != num_blocks
        {
            return Err("causal prefix leaf eval count mismatch".to_string());
        }
        let r_s = &batch_r_attn_ctx[..t_bits];
        let r_a = &batch_r_attn_ctx[t_bits..t_bits + d_bits];
        let r_b = &batch_r_attn_ctx[t_bits + d_bits..];
        // The cubic batch_attn_ctx_causal sumcheck factors suffix·eq_a·eq_b
        // into the h multiplicand (already checked above), so f and g leaves
        // are plain phi_k_mle(r_s, r_a) and v_mle(r_s, r_b) — no v_denom
        // scaling needed.
        for i in 0..num_blocks {
            if attn_ctx_final_evals_f[i] != proof.causal_phi_k_prefix_evals[i] {
                return Err(format!("Block {i}: causal phi_k leaf mismatch"));
            }
            if attn_ctx_final_evals_g[i] != proof.causal_v_prefix_evals[i] {
                return Err(format!("Block {i}: causal v leaf mismatch"));
            }
        }
        (
            combine(r_s, r_a),
            proof.causal_phi_k_prefix_evals.clone(),
            combine(r_s, r_b),
            proof.causal_v_prefix_evals.clone(),
        )
    } else {
        if !proof.causal_phi_k_prefix_evals.is_empty() || !proof.causal_v_prefix_evals.is_empty() {
            return Err("unexpected causal prefix leaf evals in non-causal proof".to_string());
        }
        (
            combine(&batch_r_attn_ctx, &batch_r_attn_out),
            attn_ctx_final_evals_f.clone(),
            combine(&batch_r_attn_ctx, &r_k_o),
            attn_ctx_final_evals_g.clone(),
        )
    };

    // phi_k opening for the context/prefix-context sumcheck leaf.
    let phi_k_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.attn_phi_k_com.clone())
        .collect();
    hyrax_verify_batch(
        &phi_k_coms,
        &phi_k_evals,
        &phi_k_attn_point,
        &proof.phi_k_batch_open,
        &params_td,
        transcript,
    )
    .map_err(|e| format!("phi_k_batch: {e}"))?;

    // v_attn opening for the context/prefix-context sumcheck leaf.
    let v_attn_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.v_com.clone())
        .collect();
    hyrax_verify_batch(
        &v_attn_coms,
        &v_evals,
        &v_attn_batch_point,
        &proof.v_attn_batch_open,
        &params_td,
        transcript,
    )
    .map_err(|e| format!("v_attn_batch: {e}"))?;

    if let Some(ref p) = attn_z_phi_q_point {
        hyrax_verify_batch(
            &phi_q_coms,
            &attn_z_phi_q_evals,
            p,
            proof
                .attn_z_phi_q_open
                .as_ref()
                .ok_or_else(|| "missing attention denominator phi_q opening".to_string())?,
            &params_td,
            transcript,
        )
        .map_err(|e| format!("attn_z_phi_q_batch: {e}"))?;
    } else if proof.attn_z_phi_q_open.is_some() {
        return Err("unexpected attention denominator phi_q opening".into());
    }
    if let Some(ref p) = attn_z_phi_k_point {
        hyrax_verify_batch(
            &phi_k_coms,
            &attn_z_phi_k_evals,
            p,
            proof
                .attn_z_phi_k_open
                .as_ref()
                .ok_or_else(|| "missing attention denominator phi_k opening".to_string())?,
            &params_td,
            transcript,
        )
        .map_err(|e| format!("attn_z_phi_k_batch: {e}"))?;
    } else if proof.attn_z_phi_k_open.is_some() {
        return Err("unexpected attention denominator phi_k opening".into());
    }

    // GKR chain (step 2): causal_context_com is dropped; the CTX cubic sumcheck
    // binds attn_out_final_evals_g algebraically via phi_k and v opens. So no
    // Hyrax open of causal_context is required.
    if proof.causal_ctx_out_batch_open.is_some()
        || proof.causal_ctx_prefix_batch_open.is_some()
    {
        return Err("unexpected causal context openings (cc_com is dropped)".to_string());
    }
    let _ = causal_ctx_coms;

    if let Some(ref r_norm) = attn_norm_r {
        let sc = proof.attn_norm_sumcheck.as_ref().unwrap();
        let mut eval_num = Vec::with_capacity(num_blocks);
        let mut eval_norm = Vec::with_capacity(num_blocks);
        let mut eval_z = Vec::with_capacity(num_blocks);
        let mut eval_rem = Vec::with_capacity(num_blocks);
        let mut eval_diff = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let base = 7 * i;
            eval_num.push(sc.final_evals_g[base]);
            eval_rem.push(sc.final_evals_g[base + 1]);
            if sc.final_evals_g[base + 2] != sc.final_evals_g[base + 3] {
                return Err(format!("Block {i}: attention z leaf mismatch"));
            }
            if sc.final_evals_g[base + 1] != sc.final_evals_g[base + 5] {
                return Err(format!("Block {i}: attention rem leaf mismatch"));
            }
            eval_z.push(sc.final_evals_g[base + 2]);
            eval_norm.push(sc.final_evals_h[base + 2]);
            eval_diff.push(sc.final_evals_g[base + 6]);
        }
        let num_coms: Vec<HyraxCommitment> = proof
            .block_proofs
            .iter()
            .map(|bp| {
                bp.attn_num_com
                    .clone()
                    .ok_or_else(|| "missing attn_num_com".to_string())
            })
            .collect::<Result<_, _>>()?;
        let norm_coms: Vec<HyraxCommitment> = proof
            .block_proofs
            .iter()
            .map(|bp| {
                bp.attn_norm_com
                    .clone()
                    .ok_or_else(|| "missing attn_norm_com".to_string())
            })
            .collect::<Result<_, _>>()?;
        let z_coms: Vec<HyraxCommitment> = proof
            .block_proofs
            .iter()
            .map(|bp| {
                bp.attn_z_com
                    .clone()
                    .ok_or_else(|| "missing attn_z_com".to_string())
            })
            .collect::<Result<_, _>>()?;
        let rem_coms: Vec<HyraxCommitment> = proof
            .block_proofs
            .iter()
            .map(|bp| {
                bp.attn_rem_com
                    .clone()
                    .ok_or_else(|| "missing attn_rem_com".to_string())
            })
            .collect::<Result<_, _>>()?;
        let diff_coms: Vec<HyraxCommitment> = proof
            .block_proofs
            .iter()
            .map(|bp| {
                bp.attn_diff_com
                    .clone()
                    .ok_or_else(|| "missing attn_diff_com".to_string())
            })
            .collect::<Result<_, _>>()?;
        // Merged open at r_norm: num_coms | norm_coms | rem_coms | diff_coms.
        let r_norm_coms: Vec<HyraxCommitment> = num_coms
            .iter()
            .chain(norm_coms.iter())
            .chain(rem_coms.iter())
            .chain(diff_coms.iter())
            .cloned()
            .collect();
        let r_norm_evals: Vec<F> = eval_num
            .iter()
            .chain(eval_norm.iter())
            .chain(eval_rem.iter())
            .chain(eval_diff.iter())
            .copied()
            .collect();
        hyrax_verify_batch(
            &r_norm_coms,
            &r_norm_evals,
            r_norm,
            proof
                .attn_norm_r_batch_open
                .as_ref()
                .ok_or_else(|| "missing attn_norm_r batch open".to_string())?,
            &params_td,
            transcript,
        )
        .map_err(|e| format!("attn_norm_r_batch: {e}"))?;

        let r_norm_t = r_norm[..t_bits].to_vec();
        let (_, _, params_t) = params_from_vars(t_bits);
        hyrax_verify_batch(
            &z_coms,
            &eval_z,
            &r_norm_t,
            proof
                .attn_z_open
                .as_ref()
                .ok_or_else(|| "missing attn_z open".to_string())?,
            &params_t,
            transcript,
        )
        .map_err(|e| format!("attn_z_batch: {e}"))?;

        let attn_point = combine(&r_t, &r_k_o);
        let attn_num_evals: Vec<F> = proof
            .block_proofs
            .iter()
            .map(|bp| bp.attn_out_eval)
            .collect();
        let norm_oproj_evals: Vec<F> = (0..num_blocks)
            .map(|i| {
                let alpha_o = vk.block_vks[i].o_vk.alpha;
                if alpha_o == F::ZERO {
                    F::ZERO
                } else {
                    proof.batch_oproj.final_evals_f[i] * alpha_o.inverse().unwrap()
                }
            })
            .collect();
        // Merged open at attn_point: num_coms | norm_coms.
        let attn_point_coms: Vec<HyraxCommitment> =
            num_coms.iter().chain(norm_coms.iter()).cloned().collect();
        let attn_point_evals: Vec<F> = attn_num_evals
            .iter()
            .chain(norm_oproj_evals.iter())
            .copied()
            .collect();
        hyrax_verify_batch(
            &attn_point_coms,
            &attn_point_evals,
            &attn_point,
            proof
                .attn_norm_attn_point_open
                .as_ref()
                .ok_or_else(|| "missing attn_norm_attn_point open".to_string())?,
            &params_td,
            transcript,
        )
        .map_err(|e| format!("attn_norm_attn_point_batch: {e}"))?;
    } else if proof.attn_norm_r_batch_open.is_some()
        || proof.attn_norm_attn_point_open.is_some()
        || proof.attn_z_open.is_some()
    {
        return Err("unexpected attention normalization openings".to_string());
    }

    eprintln!(
        "[model] cross_batch_opens:{:>8.3}ms",
        _tbatch.elapsed().as_secs_f64() * 1000.0
    );

    // =========================================================================
    // 15. Global batched Lasso (attention Q/K)
    // =========================================================================
    let _t0 = Instant::now();
    let mut all_lasso_instances = Vec::new();
    let mut all_instance_coms = Vec::new();
    let mut all_output_coms: Vec<(HyraxCommitment, usize)> = Vec::new();
    if proof.all_lasso_proof.all_query_indices.len() != 2 * num_blocks {
        return Err(format!(
            "Global Lasso query index count mismatch: got {}, expected {}",
            proof.all_lasso_proof.all_query_indices.len(),
            2 * num_blocks
        ));
    }
    for i in 0..num_blocks {
        let bvk = &vk.block_vks[i];
        all_lasso_instances.push(inst_attn.q_lasso.clone());
        all_lasso_instances.push(inst_attn.k_lasso.clone());
        all_instance_coms.push(bvk.attn_pk.qk_lasso_pk.instance_table_coms[0].clone());
        all_instance_coms.push(bvk.attn_pk.qk_lasso_pk.instance_table_coms[1].clone());
        all_output_coms.push((proof.block_proofs[i].attn_phi_q_com.clone(), td_num_vars));
        all_output_coms.push((proof.block_proofs[i].attn_phi_k_com.clone(), td_num_vars));
    }
    let global_multi_inst = LassoMultiInstance {
        instances: all_lasso_instances,
    };
    let global_instance_to_group =
        crate::lookup::lasso::derive_instance_groups(&all_instance_coms);
    let global_lasso_vk = LassoMultiVerifyingKey {
        instance_table_coms: all_instance_coms,
        instance_to_group: global_instance_to_group,
    };
    let qk_index_refs: Vec<&[usize]> = proof
        .all_lasso_proof
        .all_query_indices
        .iter()
        .map(|v| v.as_slice())
        .collect();
    absorb_index_vectors(transcript, b"qk_lasso_indices", &qk_index_refs);
    if inst_attn.q_lasso.tables.len() != inst_attn.k_lasso.tables.len()
        || inst_attn.q_lasso.bits_per_chunk != inst_attn.k_lasso.bits_per_chunk
    {
        return Err("Q/K quantization lookup domains must match".to_string());
    }
    // The Lasso φ_q / φ_k are computed from q_n / k_n (post-q_norm / post-k_norm).
    // So query_indices come from q_norm_y / k_norm_y and must be bound against
    // q_norm_y_com / k_norm_y_com — NOT q_com / k_com (which are q_raw / k_raw).
    let mut qk_raw_coms = Vec::with_capacity(2 * num_blocks);
    for bp in &proof.block_proofs {
        qk_raw_coms.push(bp.q_norm_y_com.clone());
        qk_raw_coms.push(bp.k_norm_y_com.clone());
    }
    let mut acc_quant_qk = HyraxBatchAccumulator::new();
    let mut acc_quant_qk_unused = HyraxBatchAccumulator::new();
    let mut acc_quant_qk_m = HyraxBatchAccumulator::new();
    verify_quantization_batch(
        b"qk_quant_rem_com",
        &proof.qk_quant_proof,
        &qk_raw_coms,
        &qk_index_refs,
        inst_attn.q_lasso.tables.len(),
        inst_attn.q_lasso.bits_per_chunk,
        t,
        d,
        &vk.block_vks[0].qk_activation_quant,
        transcript,
        &mut acc_quant_qk,
        &mut acc_quant_qk_unused,
        &mut acc_quant_qk_m,
    )?;
    let mu_quant_qk = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_quant_qk_m = transcript.challenge_field::<F>(b"hyrax_group_mu");
    acc_quant_qk
        .finalize_with_mu(&params_td, mu_quant_qk)
        .map_err(|e| format!("acc_quant_qk: {e}"))?;
    acc_quant_qk_m
        .finalize_with_mu(&params_range_m, mu_quant_qk_m)
        .map_err(|e| format!("acc_quant_qk_m: {e}"))?;
    let qk_lasso_bind_point = challenge_vec(transcript, td_num_vars, b"qk_lasso_bind_r");
    verify_query_indices_bound_batch(
        "attention Q/K",
        &qk_index_refs,
        t,
        d,
        &qk_lasso_bind_point,
        &proof.qk_lasso_bind_open,
        &params_td,
        transcript,
    )?;
    verify_lasso_multi_committed_outputs(
        &proof.all_lasso_proof,
        &global_multi_inst,
        &global_lasso_vk,
        &all_output_coms,
        transcript,
        lasso_params,
    )
    .map_err(|e| format!("Global Lasso: {e}"))?;
    eprintln!(
        "[model] lasso:{:>8.3}ms",
        _t0.elapsed().as_secs_f64() * 1000.0
    );

    Ok(())
}
