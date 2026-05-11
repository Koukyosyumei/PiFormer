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
    verify_lasso_multi_committed_outputs, verify_lasso_terminal_eval, LassoMultiInstance,
    LassoMultiVerifyingKey,
};
use crate::lookup::quantization::{verify_quantization_batch, QuantizationParams};
use crate::lookup::range::{verify_range_batched, RangeWitnessProof};
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_verify_batch, lagrange_basis, params_from_vars,
    HyraxBatchAccumulator, HyraxCommitment, HyraxParams,
};
use crate::poly::utils::{combine, compute_eq_evals, mat_to_mle};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{
    eq_poly_eval, verify_sumcheck_cubic_multi_batched, verify_sumcheck_multi_batched, EvalClaim,
};
use crate::transcript::{challenge_vec, Transcript};

use crate::attention::attention::{AttentionProvingKey, LinearAttentionInstance};
use crate::attention::layernorm::{
    verify_layernorms_batched, LayerNormIOCommitments, LayerNormVerifyingKey,
};
use crate::attention::projection::{
    verify_projection_gkr, ProjectionProvingKey, ProjectionVerifyingKey,
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
    let v_bits = v_vocab.next_power_of_two().trailing_zeros() as usize;
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
    if public_logits.len() != t || public_logits.iter().any(|row| row.len() != v_vocab) {
        return Err(format!(
            "public output shape mismatch: expected {t}x{v_vocab}"
        ));
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
    // PROOF_VERSION 15: attn-norm rem/diff witnesses share the global LN range
    // batch (same 64-bit table, same m_com).  One accumulator per distinct
    // num_vars value; rem and diff have identical params (T·d) so they share.
    let mut acc_attn_norm = HyraxBatchAccumulator::new();
    let mut acc_quant_ffn = HyraxBatchAccumulator::new();
    let mut acc_quant_m = HyraxBatchAccumulator::new();
    // inter_acc: global intermediate opening at r_td; mu is drawn before the
    // opening eta to match the prover's transcript schedule.
    let mut inter_acc = HyraxBatchAccumulator::new();

    // =========================================================================
    // 2. Phase 1: verify range proofs (LN sumchecks/openings deferred to the
    //    cross-LN batched proof at end of model verification).
    // =========================================================================
    let mut current_x_com = proof.x_in_com.clone();
    let ln_sigma_n = (2 * t).next_power_of_two().trailing_zeros() as usize;
    let ln_y_n = (2 * t * d).next_power_of_two().trailing_zeros() as usize;

    // Range proofs live in `ln_batched_proof.{sigma,y}_range_proofs` in the
    // SAME order as the prover collected them (per block: ln1, ln2, q_norm,
    // k_norm, attn_out_norm — interleaved sigma/y; then final_ln sigma + y).
    if proof.ln_batched_proof.sigma_range_proofs.len() != 5 * num_blocks + 1
        || proof.ln_batched_proof.y_range_proofs.len() != 5 * num_blocks + 1
        || proof.ln_batched_proof.sigma_range_bits.len() != 5 * num_blocks + 1
        || proof.ln_batched_proof.y_range_bits.len() != 5 * num_blocks + 1
    {
        return Err("LN range proof count mismatch (expected 5L+1 each)".into());
    }
    let mut ln_range_proofs: Vec<&RangeWitnessProof> = Vec::with_capacity(10 * num_blocks + 2);
    let mut ln_range_num_vars: Vec<usize> = Vec::with_capacity(10 * num_blocks + 2);
    let mut ln_range_bits: Vec<usize> = Vec::with_capacity(10 * num_blocks + 2);
    // Build the global range batch input in the SAME order the prover used:
    //   per block i: ln1_sig, ln1_y, ln2_sig, ln2_y, q_norm_sig, q_norm_y,
    //                k_norm_sig, k_norm_y, aon_sig, aon_y
    //   then: final_sig, final_y.
    // This must match the `ln_witnesses.push` ordering in the cross-LN batched
    // proof for the rv mapping below.
    let sig_rps = &proof.ln_batched_proof.sigma_range_proofs;
    let y_rps = &proof.ln_batched_proof.y_range_proofs;
    let sig_bits = &proof.ln_batched_proof.sigma_range_bits;
    let y_bits = &proof.ln_batched_proof.y_range_bits;
    // Index mapping: for block i, the LN order in the *batched proof* is
    // (ln1, q_norm, k_norm, aon, ln2) == sig_rps positions (5i+0..5i+4).
    // We want global range input in (ln1, ln2, q_norm, k_norm, aon) order.
    let ln_idx_in_batched = |block_i: usize, slot: usize| -> usize {
        // slot: 0=ln1, 1=ln2, 2=q_norm, 3=k_norm, 4=aon
        let in_block = match slot {
            0 => 0, // ln1
            1 => 4, // ln2
            2 => 1, // q_norm
            3 => 2, // k_norm
            4 => 3, // aon
            _ => unreachable!(),
        };
        5 * block_i + in_block
    };
    for i in 0..num_blocks {
        for slot in 0..5 {
            let k = ln_idx_in_batched(i, slot);
            ln_range_proofs.push(&sig_rps[k]);
            ln_range_num_vars.push(ln_sigma_n);
            ln_range_bits.push(sig_bits[k]);
            ln_range_proofs.push(&y_rps[k]);
            ln_range_num_vars.push(ln_y_n);
            ln_range_bits.push(y_bits[k]);
        }
    }
    let final_idx = 5 * num_blocks;
    ln_range_proofs.push(&sig_rps[final_idx]);
    ln_range_num_vars.push(ln_sigma_n);
    ln_range_bits.push(sig_bits[final_idx]);
    ln_range_proofs.push(&y_rps[final_idx]);
    ln_range_num_vars.push(ln_y_n);
    ln_range_bits.push(y_bits[final_idx]);

    // PROOF_VERSION 15 (P1): attn-norm rem/diff are appended to the LN range
    // batch and verified in a single call (one shared m_com / α / β / RHS
    // sumcheck).  Order must mirror prover.rs Phase 1: all 10·L+2 LN witnesses
    // first, then per-block (rem, diff) pairs when has_attn_norm.
    let has_attn_norm = proof
        .block_proofs
        .iter()
        .any(|bp| bp.attn_norm_com.is_some());
    let n_attn_norm = (t * d).next_power_of_two().trailing_zeros() as usize;

    if has_attn_norm {
        if proof.attn_norm_rem_range_proofs.len() != num_blocks
            || proof.attn_norm_diff_range_proofs.len() != num_blocks
            || proof.attn_norm_rem_range_bits.len() != num_blocks
            || proof.attn_norm_diff_range_bits.len() != num_blocks
        {
            return Err("attention normalization range proof count mismatch".into());
        }
        for i in 0..num_blocks {
            ln_range_proofs.push(&proof.attn_norm_rem_range_proofs[i]);
            ln_range_num_vars.push(n_attn_norm);
            ln_range_bits.push(proof.attn_norm_rem_range_bits[i]);
            ln_range_proofs.push(&proof.attn_norm_diff_range_proofs[i]);
            ln_range_num_vars.push(n_attn_norm);
            ln_range_bits.push(proof.attn_norm_diff_range_bits[i]);
        }
    } else if !proof.attn_norm_rem_range_proofs.is_empty()
        || !proof.attn_norm_diff_range_proofs.is_empty()
        || !proof.attn_norm_rem_range_bits.is_empty()
        || !proof.attn_norm_diff_range_bits.is_empty()
    {
        return Err("unexpected attention normalization range proof".into());
    }

    let _t0 = Instant::now();
    for &bits in &ln_range_bits {
        if bits != 32 && bits != 64 {
            return Err(format!("unsupported LN range bit width: {bits}"));
        }
    }
    let expected_bucket_bits: Vec<usize> = [32usize, 64usize]
        .into_iter()
        .filter(|bits| ln_range_bits.iter().any(|b| b == bits))
        .collect();
    if proof.ln_range_ms.len() != expected_bucket_bits.len() {
        return Err("LN range batch count mismatch".into());
    }
    for (batch, expected_bits) in proof.ln_range_ms.iter().zip(expected_bucket_bits.iter()) {
        if batch.bits != *expected_bits {
            return Err("LN range batch bit-width order mismatch".into());
        }
    }

    let mut ln_range_r_vs_by_idx: Vec<Option<Vec<F>>> = vec![None; ln_range_proofs.len()];
    for batch in &proof.ln_range_ms {
        let bucket_indices: Vec<usize> = ln_range_bits
            .iter()
            .enumerate()
            .filter_map(|(idx, &bits)| (bits == batch.bits).then_some(idx))
            .collect();
        let bucket_proofs: Vec<&RangeWitnessProof> = bucket_indices
            .iter()
            .map(|&idx| ln_range_proofs[idx])
            .collect();
        let bucket_num_vars: Vec<usize> = bucket_indices
            .iter()
            .map(|&idx| ln_range_num_vars[idx])
            .collect();

        // Build chunk_accs slice — one accumulator per distinct num_vars value.
        let mut chunk_accs: Vec<(usize, &mut HyraxBatchAccumulator)> = Vec::with_capacity(3);
        chunk_accs.push((ln_sigma_n, &mut acc_range_sig));
        if ln_y_n != ln_sigma_n {
            chunk_accs.push((ln_y_n, &mut acc_range_y));
        }
        if has_attn_norm && n_attn_norm != ln_sigma_n && n_attn_norm != ln_y_n {
            chunk_accs.push((n_attn_norm, &mut acc_attn_norm));
        }

        let (bucket_r_vs, _) = verify_range_batched(
            &bucket_proofs,
            &batch.m,
            &bucket_num_vars,
            batch.bits,
            transcript,
            &mut chunk_accs,
            &mut acc_range_m,
        )?;
        drop(chunk_accs);
        for (idx, rv) in bucket_indices.into_iter().zip(bucket_r_vs.into_iter()) {
            ln_range_r_vs_by_idx[idx] = Some(rv);
        }
    }
    let ln_range_r_vs: Vec<Vec<F>> = ln_range_r_vs_by_idx
        .into_iter()
        .map(|rv| rv.expect("missing LN range rv"))
        .collect();
    eprintln!(
        "[model] ln_range_batch:{:>8.3}ms",
        _t0.elapsed().as_secs_f64() * 1000.0
    );

    // Collect per-LN IO commitments for the cross-LN batched verify call at
    // the end. For each block we push 5 LNs in the same order the prover used:
    //   ln1, q_norm, k_norm, attn_out_norm, ln2.
    let mut ln_io_coms_owned: Vec<LayerNormIOCommitments> = Vec::with_capacity(5 * num_blocks + 1);
    let mut ln_vks_refs: Vec<&LayerNormVerifyingKey> = Vec::with_capacity(5 * num_blocks + 1);
    let mut ln_sigma_r_vs_idx: Vec<usize> = Vec::with_capacity(5 * num_blocks + 1);
    let mut ln_y_r_vs_idx: Vec<usize> = Vec::with_capacity(5 * num_blocks + 1);

    for i in 0..num_blocks {
        let bp = &proof.block_proofs[i];
        let bvk = &vk.block_vks[i];
        // 10 sub-witnesses per block in the model-level batched range proof:
        //   [ln1_sigma, ln1_y, ln2_sigma, ln2_y,
        //    q_norm_sigma, q_norm_y, k_norm_sigma, k_norm_y,
        //    attn_out_norm_sigma, attn_out_norm_y]
        let rv_base = 10 * i;

        // LN1: absorb x_in_com (current_x_com) + x_norm1_com.
        absorb_com(transcript, b"x_com", &current_x_com);
        absorb_com(transcript, b"y_com", &bp.x_norm1_com);
        ln_io_coms_owned.push(LayerNormIOCommitments {
            x_com: current_x_com.clone(),
            y_com: Some(bp.x_norm1_com.clone()),
        });
        ln_vks_refs.push(&bvk.ln1_vk);
        ln_sigma_r_vs_idx.push(rv_base);
        ln_y_r_vs_idx.push(rv_base + 1);

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

        // q_norm: x = q_raw, y = q_n.
        absorb_com(transcript, b"x_com", &bp.q_com);
        absorb_com(transcript, b"y_com", &bp.q_norm_y_com);
        ln_io_coms_owned.push(LayerNormIOCommitments {
            x_com: bp.q_com.clone(),
            y_com: Some(bp.q_norm_y_com.clone()),
        });
        ln_vks_refs.push(&bvk.q_norm_vk);
        ln_sigma_r_vs_idx.push(rv_base + 4);
        ln_y_r_vs_idx.push(rv_base + 5);
        absorb_com(transcript, b"q_norm_y_com", &bp.q_norm_y_com);

        // k_norm: x = k_raw, y = k_n.
        absorb_com(transcript, b"x_com", &bp.k_com);
        absorb_com(transcript, b"y_com", &bp.k_norm_y_com);
        ln_io_coms_owned.push(LayerNormIOCommitments {
            x_com: bp.k_com.clone(),
            y_com: Some(bp.k_norm_y_com.clone()),
        });
        ln_vks_refs.push(&bvk.k_norm_vk);
        ln_sigma_r_vs_idx.push(rv_base + 6);
        ln_y_r_vs_idx.push(rv_base + 7);
        absorb_com(transcript, b"k_norm_y_com", &bp.k_norm_y_com);

        // attn_out_norm: x = out_attn, y = post-norm output.
        absorb_com(transcript, b"x_com", &bp.out_attn_com);
        absorb_com(transcript, b"y_com", &bp.attn_out_norm_y_com);
        ln_io_coms_owned.push(LayerNormIOCommitments {
            x_com: bp.out_attn_com.clone(),
            y_com: Some(bp.attn_out_norm_y_com.clone()),
        });
        ln_vks_refs.push(&bvk.attn_out_norm_vk);
        ln_sigma_r_vs_idx.push(rv_base + 8);
        ln_y_r_vs_idx.push(rv_base + 9);
        absorb_com(transcript, b"attn_out_norm_y_com", &bp.attn_out_norm_y_com);

        // Residual flows through the normalized attention output.
        let x_mid_com = add_commitments(&current_x_com, &bp.attn_out_norm_y_com);

        // LN2
        absorb_com(transcript, b"x_com", &x_mid_com);
        absorb_com(transcript, b"y_com", &bp.x_norm2_com);
        ln_io_coms_owned.push(LayerNormIOCommitments {
            x_com: x_mid_com.clone(),
            y_com: Some(bp.x_norm2_com.clone()),
        });
        ln_vks_refs.push(&bvk.ln2_vk);
        ln_sigma_r_vs_idx.push(rv_base + 2);
        ln_y_r_vs_idx.push(rv_base + 3);

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
    if proof.ffn_lasso_query_indices.len() != num_blocks {
        return Err(format!(
            "FFN Lasso query index count mismatch: got {}, expected {}",
            proof.ffn_lasso_query_indices.len(),
            num_blocks
        ));
    }
    let ffn_index_refs: Vec<&[usize]> = proof
        .ffn_lasso_query_indices
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
    let ffn_a_terminal_weights: Vec<F> = weights_ffn_y
        .iter()
        .zip(proof.batch_ffn_y.final_evals_g.iter())
        .map(|(w, g)| *w * *g)
        .collect();
    let ffn_a_terminal_claim: F = ffn_a_terminal_weights
        .iter()
        .zip(proof.batch_ffn_y.final_evals_f.iter())
        .map(|(w, f)| *w * *f)
        .sum();
    let ffn_a_terminal_point = combine(&r_t, &r_k_fy);
    verify_lasso_terminal_eval(
        &proof.ffn_a_terminal_proof,
        &ffn_lasso_instances,
        &ffn_lasso_vk,
        &proof.ffn_lasso_query_indices,
        &ffn_a_terminal_weights,
        &ffn_a_terminal_point,
        ffn_a_terminal_claim,
        transcript,
        lasso_params,
    )
    .map_err(|e| format!("FFN A terminal Lasso: {e}"))?;

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
    // 10. Final LayerNorm — absorb IO commitments only; the cross-LN batched
    //     verify call below covers the actual sumchecks/openings.
    // =========================================================================
    let _t0 = Instant::now();
    absorb_com(transcript, b"x_com", &current_x_com);
    absorb_com(transcript, b"y_com", &proof.final_ln_out_com);
    ln_io_coms_owned.push(LayerNormIOCommitments {
        x_com: current_x_com.clone(),
        y_com: Some(proof.final_ln_out_com.clone()),
    });
    ln_vks_refs.push(&vk.final_ln_vk);
    ln_sigma_r_vs_idx.push(10 * num_blocks);
    ln_y_r_vs_idx.push(10 * num_blocks + 1);
    eprintln!(
        "[model] final_ln_absorb:{:>8.3}ms",
        _t0.elapsed().as_secs_f64() * 1000.0
    );

    // =========================================================================
    // 10b. Cross-LN batched verify — covers every LN in the model (5L + 1).
    // =========================================================================
    let _t0 = Instant::now();
    let ln_io_coms_refs: Vec<&LayerNormIOCommitments> = ln_io_coms_owned.iter().collect();
    let ln_sigma_r_vs: Vec<&[F]> = ln_sigma_r_vs_idx
        .iter()
        .map(|&i| ln_range_r_vs[i].as_slice())
        .collect();
    let ln_y_r_vs: Vec<&[F]> = ln_y_r_vs_idx
        .iter()
        .map(|&i| ln_range_r_vs[i].as_slice())
        .collect();
    verify_layernorms_batched(
        &proof.ln_batched_proof,
        &ln_io_coms_refs,
        &ln_vks_refs,
        &ln_sigma_r_vs,
        &ln_y_r_vs,
        transcript,
        &mut ln_acc_t,
        &mut ln_acc_td,
    )
    .map_err(|e| format!("LN batched: {e}"))?;
    eprintln!(
        "[model] ln_batched:{:>8.3}ms",
        _t0.elapsed().as_secs_f64() * 1000.0
    );

    // =========================================================================
    // 11. LM Head
    // =========================================================================
    let _t0 = Instant::now();
    let public_logits_mle = mat_to_mle(public_logits, t, v_vocab);
    let lm_y_point = challenge_vec(transcript, t_bits + v_bits, b"lm_gkr_y");
    let lm_y_value = public_logits_mle.evaluate(&lm_y_point);
    let lm_y_claim = EvalClaim {
        point: lm_y_point,
        value: lm_y_value,
    };
    let lm_x_claim = verify_projection_gkr(
        &proof.lm_head_proof,
        &vk.lm_head_vk,
        &lm_y_claim,
        transcript,
        &mut lmh_acc_w,
        &mut lmh_acc_b,
    )
    .map_err(|e| format!("LM Head: {e}"))?;
    ln_acc_td
        .add_verify(
            &proof.final_ln_out_com,
            lm_x_claim.value,
            &lm_x_claim.point,
            &proof.lm_head_input_open,
        )
        .map_err(|e| format!("LM head input opening: {e}"))?;
    eprintln!(
        "[model] lm_head:{:>8.3}ms",
        _t0.elapsed().as_secs_f64() * 1000.0
    );

    // =========================================================================
    // 12. Draw accumulator challenges (same order as prover's mu challenge loop)
    // =========================================================================
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
    // PROOF_VERSION 15: attn-norm rem/diff/m collapsed into one accumulator
    // sharing the global LN range batch's m_com.  Single mu draw (no-op when
    // !has_attn_norm — finalize_with_mu short-circuits on empty slots).
    let mu_attn_norm = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let rho_td = transcript.challenge_field_readonly::<F>(b"hyrax_fuse_td");
    let rho_range_m = transcript.challenge_field_readonly::<F>(b"hyrax_fuse_range_m");

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
    inter_acc
        .add_verify_batch(
            &all_coms,
            &all_evals,
            &r_td,
            &proof.inter_batch_open,
            transcript,
        )
        .map_err(|e| format!("Global inter_batch (deferred): {e}"))?;
    eprintln!(
        "[model] inter_batch:{:>8.3}ms",
        _t0.elapsed().as_secs_f64() * 1000.0
    );

    let _tacc = Instant::now();
    let params_td_ref = &params_td;
    let params_t_ref = &params_t;
    let params_qkvo_w_ref = &params_qkvo_w;
    let params_qkvo_b_ref = &params_qkvo_b;
    let params_lmh_w_ref = &params_lmh_w;
    let params_lmh_b_ref = &params_lmh_b;
    let params_range_sig_ref = &params_range_sig;
    let params_range_y_ref = &params_range_y;
    let params_range_m_ref = &params_range_m;
    let params_mff_ref = &params_mff;
    let (finalize_tx, finalize_rx) = std::sync::mpsc::channel();
    rayon::scope(|s| {
        let tx = finalize_tx.clone();
        s.spawn(move |_| {
            let result = HyraxBatchAccumulator::finalize_many_with_mus(
                params_td_ref,
                vec![
                    (inter_acc, mu_inter),
                    (ln_acc_td, mu_ln_td),
                    (acc_attn_norm, mu_attn_norm),
                ],
                rho_td,
            )
            .map_err(|e| format!("td fused acc: {e}"));
            let _ = tx.send((0usize, result));
        });

        let tx = finalize_tx.clone();
        s.spawn(move |_| {
            let result = ln_acc_t
                .finalize_with_mu(params_t_ref, mu_ln_t)
                .map_err(|e| format!("ln_acc_t: {e}"));
            let _ = tx.send((1usize, result));
        });

        let tx = finalize_tx.clone();
        s.spawn(move |_| {
            let result = proj_acc_w
                .finalize_with_mu(params_qkvo_w_ref, mu_proj_w)
                .map_err(|e| format!("proj_acc_w: {e}"));
            let _ = tx.send((2usize, result));
        });

        let tx = finalize_tx.clone();
        s.spawn(move |_| {
            let result = acc_quant_ffn
                .finalize_with_mu(params_mff_ref, mu_quant_ffn)
                .map_err(|e| format!("acc_quant_ffn: {e}"));
            let _ = tx.send((3usize, result));
        });

        let tx = finalize_tx.clone();
        s.spawn(move |_| {
            let result = proj_acc_b
                .finalize_with_mu(params_qkvo_b_ref, mu_proj_b)
                .map_err(|e| format!("proj_acc_b: {e}"));
            let _ = tx.send((4usize, result));
        });

        let tx = finalize_tx.clone();
        s.spawn(move |_| {
            let result = lmh_acc_w
                .finalize_with_mu(params_lmh_w_ref, mu_lmh_w)
                .map_err(|e| format!("lmh_acc_w: {e}"));
            let _ = tx.send((5usize, result));
        });

        let tx = finalize_tx.clone();
        s.spawn(move |_| {
            let result = lmh_acc_b
                .finalize_with_mu(params_lmh_b_ref, mu_lmh_b)
                .map_err(|e| format!("lmh_acc_b: {e}"));
            let _ = tx.send((6usize, result));
        });

        let tx = finalize_tx.clone();
        s.spawn(move |_| {
            let result = acc_range_sig
                .finalize_with_mu(params_range_sig_ref, mu_rng_sig)
                .map_err(|e| format!("acc_range_sig: {e}"));
            let _ = tx.send((7usize, result));
        });

        let tx = finalize_tx.clone();
        s.spawn(move |_| {
            let result = acc_range_y
                .finalize_with_mu(params_range_y_ref, mu_rng_y)
                .map_err(|e| format!("acc_range_y: {e}"));
            let _ = tx.send((8usize, result));
        });

        let tx = finalize_tx.clone();
        s.spawn(move |_| {
            let result = HyraxBatchAccumulator::finalize_many_with_mus(
                params_range_m_ref,
                vec![(acc_range_m, mu_rng_m), (acc_quant_m, mu_quant_m)],
                rho_range_m,
            )
            .map_err(|e| format!("range_m fused acc: {e}"));
            let _ = tx.send((9usize, result));
        });
    });
    drop(finalize_tx);
    let mut finalize_results: Vec<Option<Result<(), String>>> = vec![None; 10];
    for (idx, result) in finalize_rx {
        finalize_results[idx] = Some(result);
    }
    eprintln!(
        "[model] acc_finalize:{:>8.3}ms",
        _tacc.elapsed().as_secs_f64() * 1000.0
    );
    for result in finalize_results {
        result.expect("missing accumulator finalize result")?;
    }

    // =========================================================================
    // 14. 13 cross-block weight/activation batch opens
    //     (must be in same order as prover's step 15)
    // =========================================================================
    let _tbatch = Instant::now();

    // V1: instead of running ~13 independent `hyrax_verify_batch` calls (each
    // doing its own MSM), accumulate every cross-block batch open into one
    // HyraxBatchAccumulator per Hyrax-params group.  Inner-product checks happen
    // immediately inside `add_verify_batch` (field ops only); the heavy MSMs
    // are deferred to a single per-params `finalize_with_mu` at the end of
    // this phase, run in parallel via rayon::join.
    //
    // The order of `add_verify_batch` calls is preserved relative to the
    // prover's `hyrax_open_batch` sequence so that the auto-drawn `eta`
    // challenges (label `b"hyrax_batch_eta"`) line up.
    let mut acc_cb_td = HyraxBatchAccumulator::new();
    let mut acc_cb_qkvo_w = HyraxBatchAccumulator::new();
    let mut acc_cb_qkvo_b = HyraxBatchAccumulator::new();
    let mut acc_cb_wff = HyraxBatchAccumulator::new();
    let mut acc_cb_mff = HyraxBatchAccumulator::new();

    // x_norm1 at combine(r_t, r_k_qkv)
    let x_norm1_point = combine(&r_t, &r_k_qkv);
    let x_norm1_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.x_norm1_com.clone())
        .collect();
    acc_cb_td
        .add_verify_batch(
            &x_norm1_coms,
            &proof.batch_qkv.final_evals_f,
            &x_norm1_point,
            &proof.x_norm1_batch_open,
            transcript,
        )
        .map_err(|e| format!("x_norm1_batch: {e}"))?;

    // P2: wq/wk/wv merged into one open at combine(r_k_qkv, r_out).
    // Interleave per block: [wq_0, wk_0, wv_0, wq_1, wk_1, wv_1, ...] to match
    // the prover's static_mles loop ordering.
    let wq_point = combine(&r_k_qkv, &r_out);
    let mut qkv_w_coms: Vec<HyraxCommitment> = Vec::with_capacity(3 * num_blocks);
    let mut qkv_w_evals: Vec<F> = Vec::with_capacity(3 * num_blocks);
    for (bvk, bp) in vk.block_vks.iter().zip(proof.block_proofs.iter()) {
        qkv_w_coms.push(bvk.q_vk.w_com.clone());
        qkv_w_evals.push(bp.qkv_w_q_eval);
        qkv_w_coms.push(bvk.k_vk.w_com.clone());
        qkv_w_evals.push(bp.qkv_w_k_eval);
        qkv_w_coms.push(bvk.v_vk.w_com.clone());
        qkv_w_evals.push(bp.qkv_w_v_eval);
    }
    acc_cb_qkvo_w
        .add_verify_batch(
            &qkv_w_coms,
            &qkv_w_evals,
            &wq_point,
            &proof.qkv_w_batch_open,
            transcript,
        )
        .map_err(|e| format!("qkv_w_batch: {e}"))?;

    // wo at combine(r_k_o, r_out) — different point from qkv_w group, so kept
    // as its own batch.
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
    acc_cb_qkvo_w
        .add_verify_batch(
            &wo_coms,
            &wo_evals,
            &wo_point,
            &proof.w_o_batch_open,
            transcript,
        )
        .map_err(|e| format!("w_o_batch: {e}"))?;

    // P2: bq/bk/bv/bo merged into one open at r_out.  Interleave per block:
    // [bq_0, bk_0, bv_0, bo_0, bq_1, ...].
    let mut qkvo_bias_coms: Vec<HyraxCommitment> = Vec::with_capacity(4 * num_blocks);
    let mut qkvo_bias_evals: Vec<F> = Vec::with_capacity(4 * num_blocks);
    for (bvk, bp) in vk.block_vks.iter().zip(proof.block_proofs.iter()) {
        qkvo_bias_coms.push(bvk.q_vk.bias_com.clone());
        qkvo_bias_evals.push(bp.qkv_bias_q_eval);
        qkvo_bias_coms.push(bvk.k_vk.bias_com.clone());
        qkvo_bias_evals.push(bp.qkv_bias_k_eval);
        qkvo_bias_coms.push(bvk.v_vk.bias_com.clone());
        qkvo_bias_evals.push(bp.qkv_bias_v_eval);
        qkvo_bias_coms.push(bvk.o_vk.bias_com.clone());
        qkvo_bias_evals.push(bp.oproj_bias_o_eval);
    }
    acc_cb_qkvo_b
        .add_verify_batch(
            &qkvo_bias_coms,
            &qkvo_bias_evals,
            &r_out,
            &proof.qkvo_bias_batch_open,
            transcript,
        )
        .map_err(|e| format!("qkvo_bias_batch: {e}"))?;

    // w2 at combine(r_k_fy, r_out)
    let w2_point = combine(&r_k_fy, &r_out);
    let w2_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.ffn_vk.w2_com.clone())
        .collect();
    acc_cb_wff
        .add_verify_batch(
            &w2_coms,
            &proof.batch_ffn_y.final_evals_g,
            &w2_point,
            &proof.w2_batch_open,
            transcript,
        )
        .map_err(|e| format!("w2_batch: {e}"))?;

    // w1 at combine(r_k_m, ry_m)
    let w1_point = combine(&r_k_m, &ry_m);
    let w1_coms: Vec<HyraxCommitment> = vk
        .block_vks
        .iter()
        .map(|bvk| bvk.ffn_vk.w1_com.clone())
        .collect();
    acc_cb_wff
        .add_verify_batch(
            &w1_coms,
            &proof.batch_ffn_m.final_evals_g,
            &w1_point,
            &proof.w1_batch_open,
            transcript,
        )
        .map_err(|e| format!("w1_batch: {e}"))?;

    // x_norm2 at combine(rx_m, r_k_m)
    let x_norm2_point = combine(&rx_m, &r_k_m);
    let x_norm2_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.x_norm2_com.clone())
        .collect();
    acc_cb_td
        .add_verify_batch(
            &x_norm2_coms,
            &proof.batch_ffn_m.final_evals_f,
            &x_norm2_point,
            &proof.x_norm2_batch_open,
            transcript,
        )
        .map_err(|e| format!("x_norm2_batch: {e}"))?;

    // ffn_m_com at combine(rx_m, ry_m)
    let ffn_m_point = combine(&rx_m, &ry_m);
    let ffn_m_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.ffn_m_com.clone())
        .collect();
    let ffn_m_evals: Vec<F> = proof.block_proofs.iter().map(|bp| bp.ffn_m_eval).collect();
    acc_cb_mff
        .add_verify_batch(
            &ffn_m_coms,
            &ffn_m_evals,
            &ffn_m_point,
            &proof.ffn_m_com_batch_open,
            transcript,
        )
        .map_err(|e| format!("ffn_m_com_batch: {e}"))?;

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
    acc_cb_td
        .add_verify_batch(
            &phi_q_coms,
            &attn_out_final_evals_f,
            &phi_q_attn_point,
            &proof.phi_q_batch_open,
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
    acc_cb_td
        .add_verify_batch(
            &phi_k_coms,
            &phi_k_evals,
            &phi_k_attn_point,
            &proof.phi_k_batch_open,
            transcript,
        )
        .map_err(|e| format!("phi_k_batch: {e}"))?;

    // v_attn opening for the context/prefix-context sumcheck leaf.
    let v_attn_coms: Vec<HyraxCommitment> = proof
        .block_proofs
        .iter()
        .map(|bp| bp.v_com.clone())
        .collect();
    acc_cb_td
        .add_verify_batch(
            &v_attn_coms,
            &v_evals,
            &v_attn_batch_point,
            &proof.v_attn_batch_open,
            transcript,
        )
        .map_err(|e| format!("v_attn_batch: {e}"))?;

    if let Some(ref p) = attn_z_phi_q_point {
        acc_cb_td
            .add_verify_batch(
                &phi_q_coms,
                &attn_z_phi_q_evals,
                p,
                proof
                    .attn_z_phi_q_open
                    .as_ref()
                    .ok_or_else(|| "missing attention denominator phi_q opening".to_string())?,
                transcript,
            )
            .map_err(|e| format!("attn_z_phi_q_batch: {e}"))?;
    } else if proof.attn_z_phi_q_open.is_some() {
        return Err("unexpected attention denominator phi_q opening".into());
    }
    if let Some(ref p) = attn_z_phi_k_point {
        acc_cb_td
            .add_verify_batch(
                &phi_k_coms,
                &attn_z_phi_k_evals,
                p,
                proof
                    .attn_z_phi_k_open
                    .as_ref()
                    .ok_or_else(|| "missing attention denominator phi_k opening".to_string())?,
                transcript,
            )
            .map_err(|e| format!("attn_z_phi_k_batch: {e}"))?;
    } else if proof.attn_z_phi_k_open.is_some() {
        return Err("unexpected attention denominator phi_k opening".into());
    }

    // GKR chain (step 2): causal_context_com is dropped; the CTX cubic sumcheck
    // binds attn_out_final_evals_g algebraically via phi_k and v opens. So no
    // Hyrax open of causal_context is required.
    if proof.causal_ctx_out_batch_open.is_some() || proof.causal_ctx_prefix_batch_open.is_some() {
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
        acc_cb_td
            .add_verify_batch(
                &r_norm_coms,
                &r_norm_evals,
                r_norm,
                proof
                    .attn_norm_r_batch_open
                    .as_ref()
                    .ok_or_else(|| "missing attn_norm_r batch open".to_string())?,
                transcript,
            )
            .map_err(|e| format!("attn_norm_r_batch: {e}"))?;

        let r_norm_t = r_norm[..t_bits].to_vec();
        let (_, _, params_t) = params_from_vars(t_bits);
        // attn_z_batch is the only opening at params_t (size 2^t_bits) — single
        // call, no fusing benefit, so verify it inline as a one-shot MSM.
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
        acc_cb_td
            .add_verify_batch(
                &attn_point_coms,
                &attn_point_evals,
                &attn_point,
                proof
                    .attn_norm_attn_point_open
                    .as_ref()
                    .ok_or_else(|| "missing attn_norm_attn_point open".to_string())?,
                transcript,
            )
            .map_err(|e| format!("attn_norm_attn_point_batch: {e}"))?;
    } else if proof.attn_norm_r_batch_open.is_some()
        || proof.attn_norm_attn_point_open.is_some()
        || proof.attn_z_open.is_some()
    {
        return Err("unexpected attention normalization openings".to_string());
    }

    // V1 finalize: single MSM per Hyrax-params group, all in parallel.
    let mu_cb_td = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_cb_qkvo_w = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_cb_qkvo_b = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_cb_wff = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_cb_mff = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let ((rcb_td, rcb_qw), (rcb_qb, (rcb_w, rcb_m))) = rayon::join(
        || {
            rayon::join(
                || {
                    acc_cb_td
                        .finalize_with_mu(&params_td, mu_cb_td)
                        .map_err(|e| format!("acc_cb_td: {e}"))
                },
                || {
                    acc_cb_qkvo_w
                        .finalize_with_mu(&params_qkvo_w, mu_cb_qkvo_w)
                        .map_err(|e| format!("acc_cb_qkvo_w: {e}"))
                },
            )
        },
        || {
            rayon::join(
                || {
                    acc_cb_qkvo_b
                        .finalize_with_mu(&params_qkvo_b, mu_cb_qkvo_b)
                        .map_err(|e| format!("acc_cb_qkvo_b: {e}"))
                },
                || {
                    rayon::join(
                        || {
                            acc_cb_wff
                                .finalize_with_mu(&params_wff, mu_cb_wff)
                                .map_err(|e| format!("acc_cb_wff: {e}"))
                        },
                        || {
                            acc_cb_mff
                                .finalize_with_mu(&params_mff, mu_cb_mff)
                                .map_err(|e| format!("acc_cb_mff: {e}"))
                        },
                    )
                },
            )
        },
    );
    rcb_td?;
    rcb_qw?;
    rcb_qb?;
    rcb_w?;
    rcb_m?;

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
    let global_instance_to_group = crate::lookup::lasso::derive_instance_groups(&all_instance_coms);
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
        &mut acc_quant_qk_m,
    )?;
    let mu_quant_qk = transcript.challenge_field::<F>(b"hyrax_group_mu");
    let mu_quant_qk_m = transcript.challenge_field::<F>(b"hyrax_group_mu");
    // V3: the two QK quant accumulators have different Hyrax params (params_td
    // vs params_range_m) so they cannot be MSM-fused via `finalize_many_with_mus`
    // (which requires a single shared params).  Best we can do here is run the
    // two finalize MSMs in parallel — typically a ~2x wall-clock win for this
    // pair on multi-core.
    let (rq, rqm) = rayon::join(
        || {
            acc_quant_qk
                .finalize_with_mu(&params_td, mu_quant_qk)
                .map_err(|e| format!("acc_quant_qk: {e}"))
        },
        || {
            acc_quant_qk_m
                .finalize_with_mu(&params_range_m, mu_quant_qk_m)
                .map_err(|e| format!("acc_quant_qk_m: {e}"))
        },
    );
    rq?;
    rqm?;
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
