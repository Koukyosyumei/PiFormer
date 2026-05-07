//! Lasso lookup argument for the structured activation φ.
//!
//! **What we prove:**
//!   For each query j ∈ [n]:
//!     output_j = Σ_{k=0}^{c-1} T_k[ chunk_k(idx_j) ]
//!   where chunk_k(x) = (x >> (k * m)) & ((1<<m)-1), m = bits_per_chunk.
//!
//! **How (batched MLE evaluation via sumcheck + Hyrax PCS):**
//!
//!   For each sub-table k:
//!   1. Commit to T_k via Hyrax: C_k = HyraxCommit(T_k).
//!   2. Build a "selector" polynomial
//!        L_k(x) = Σ_j ρ^j · eq(binary(chunk_k(idx_j)), x)
//!   3. Run sumcheck:
//!        Σ_{x ∈ {0,1}^m} T_k(x) · L_k(x) = Σ_j ρ^j · T_k[chunk_k(idx_j)]
//!   4. Open T_k at the random sumcheck point r_k via Hyrax, yielding a proof
//!      that the claimed T_k(r_k) is consistent with the committed polynomial.
//!
//! The Hyrax commitment is a transparent (no trusted setup) vector commitment
//! over BN254 G1, with O(√N) proof size and O(√N) verifier work.

use crate::field::F;
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_open, hyrax_open_batch, hyrax_verify, hyrax_verify_batch,
    params_from_vars, HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::sumcheck::{
    prove_sumcheck_multi_batched, verify_sumcheck_multi_batched, SumcheckProofMulti,
};
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::{challenge_vec, Transcript};
use ark_ff::Field;
use rayon::prelude::*;

/// Precomputed table commitments — computed once at setup, not per proof.
#[derive(Clone)]
pub struct LassoProvingKey {
    pub table_coms: Vec<HyraxCommitment>,
    pub nu: usize,
}

impl LassoProvingKey {
    pub fn vk(&self) -> LassoVerifyingKey {
        LassoVerifyingKey {
            table_coms: self.table_coms.clone(),
        }
    }
}

/// Verifier-side key: same precommitted table commitments.
#[derive(Clone)]
pub struct LassoVerifyingKey {
    pub table_coms: Vec<HyraxCommitment>,
}

/// Precommit activation tables during setup phase (call once, not per inference).
pub fn precommit_lasso_tables(
    tables: &[Vec<F>],
    bits_per_chunk: usize,
    params: &HyraxParams,
) -> LassoProvingKey {
    let nu = bits_per_chunk / 2;
    let table_coms = tables.iter().map(|t| hyrax_commit(t, nu, params)).collect();
    LassoProvingKey { table_coms, nu }
}

/// Public description of a lookup instance.
/// Lookup indices are carried by the proof so the verifier can bind selector
/// polynomials to the committed lookup input tensor at the model call site.
#[derive(Clone)]
pub struct LassoInstance {
    /// c sub-tables T_0, ..., T_{c-1}, each of size 2^bits_per_chunk.
    pub tables: Vec<Vec<F>>,
    /// Claimed outputs: output_j = Σ_k T_k[chunk_k(idx_j)].
    pub outputs: Vec<F>,
    /// Bits per chunk (m); table size = 2^m.
    pub bits_per_chunk: usize,
}

/// Proof for a Lasso lookup (with Hyrax PCS).
pub struct LassoProof {
    /// Lookup outputs used in the grand-sum check.
    pub outputs: Vec<F>,
    /// Lookup indices used to build the selector polynomials.
    /// Verifiers bind these to the committed lookup input tensor at the call site.
    pub query_indices: Vec<usize>,
    /// Batched sum per sub-table: Σ_j ρ^j · T_k[chunk_k(idx_j)].
    pub sub_claims: Vec<F>,
    /// Sumcheck proof per sub-table.
    pub sumcheck_proofs: Vec<SumcheckProof>,
    /// Claimed T_k(r_k) at the sumcheck random point.
    pub table_openings: Vec<F>,
    /// Hyrax opening proofs for T_k(r_k).
    pub hyrax_proofs: Vec<HyraxProof>,
    /// Optional committed-output binding for model mode:
    /// proves Σ_j ρ^j O[j] = Σ_k sub_claim_k, where O is committed by the
    /// surrounding protocol.
    pub output_sumcheck: Option<SumcheckProof>,
    pub output_open: Option<HyraxProof>,
    /// L_k evaluated at the sumcheck output point r (one per sub-table).
    /// The verifier recomputes the same value from `query_indices`, so no PCS
    /// opening is needed for this deterministic selector polynomial.
    pub l_k_evals: Vec<F>,
    /// Optional committed-index binding. In model mode this replaces the raw
    /// proof-carried query indices with commitments to the chunk polynomials
    /// of the lookup input tensor.
    pub index_proof: Option<LassoIndexProof>,
}

#[derive(Clone, Debug)]
pub struct HighDegreeRoundPoly {
    pub evals: Vec<F>,
}

impl HighDegreeRoundPoly {
    pub fn evaluate(&self, x: F) -> F {
        let mut acc = F::ZERO;
        for (i, &yi) in self.evals.iter().enumerate() {
            let xi = F::from(i as u64);
            let mut num = F::ONE;
            let mut den = F::ONE;
            for j in 0..self.evals.len() {
                if i == j {
                    continue;
                }
                let xj = F::from(j as u64);
                num *= x - xj;
                den *= xi - xj;
            }
            acc += yi * num * den.inverse().expect("distinct interpolation points");
        }
        acc
    }
}

#[derive(Clone, Debug)]
pub struct SelectorSumcheckProof {
    pub round_polys: Vec<HighDegreeRoundPoly>,
    pub final_eval_chunk: F,
}

#[derive(Clone)]
pub struct SelectorBindingProof {
    pub sumcheck: SelectorSumcheckProof,
    pub chunk_open: HyraxProof,
}

#[derive(Clone)]
pub struct LassoIndexProof {
    pub query_len: usize,
    pub rho: F,
    pub chunk_coms: Vec<HyraxCommitment>,
    pub selector_points: Vec<Vec<F>>,
    pub bind_chunk_evals: Vec<F>,
    pub bind_open: HyraxProof,
    pub selector_proofs: Vec<SelectorBindingProof>,
}

pub struct LassoIndexBinding {
    pub com: HyraxCommitment,
    pub num_vars: usize,
    pub mle_evals: Vec<F>,
}

pub fn prove_lasso(
    instance: &LassoInstance,
    query_indices: &[usize],
    pk: &LassoProvingKey,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> LassoProof {
    let c = instance.tables.len();
    let m = instance.bits_per_chunk;
    let n = query_indices.len();
    let mask = (1usize << m) - 1;

    let nu = pk.nu;
    let sigma = params.sigma;
    assert_eq!(
        nu + sigma,
        instance.bits_per_chunk,
        "LassoProvingKey nu/sigma mismatch with bits_per_chunk"
    );
    for k in 0..c {
        absorb_com(transcript, b"hyrax_com", &pk.table_coms[k]);
    }

    for &out in &instance.outputs {
        transcript.append_field(b"lasso_out", &out);
    }

    let rho = transcript.challenge_field::<F>(b"lasso_rho");
    let rho_pows = powers_of(rho, n);

    let mut sumcheck_proofs = Vec::with_capacity(c);
    let mut table_openings = Vec::with_capacity(c);
    let mut hyrax_proofs = Vec::with_capacity(c);
    let mut sub_claims = Vec::with_capacity(c);
    let mut l_k_evals = Vec::with_capacity(c);

    for k in 0..c {
        let t_poly = DenseMLPoly::new(instance.tables[k].clone());

        // Build counting polynomial L_k as a histogram weighted by ρ^j.
        // L_k[ch] = Σ_{j: chunk_k(idx_j)==ch} ρ^j.
        // Building via histogram (O(n + 2^m)) is faster than the full eq-expansion.
        let size = 1usize << m;
        let mut l_hist = vec![F::ZERO; size];
        for j in 0..n {
            let ch = chunk(query_indices[j], k, m, mask);
            l_hist[ch] += rho_pows[j];
        }
        let l_poly = DenseMLPoly::new(l_hist.clone());

        // Claimed sum for this sub-table: Σ_j ρ^j * T_k[chunk_k(idx_j)]
        let claimed: F = (0..n)
            .map(|j| rho_pows[j] * instance.tables[k][chunk(query_indices[j], k, m, mask)])
            .sum();
        sub_claims.push(claimed);

        let (sc_proof, r_vec) = prove_sumcheck(&t_poly, &l_poly, claimed, transcript);

        // Hyrax opening for T_k at sumcheck output r_vec
        let t_opening = t_poly.evaluate(&r_vec);
        transcript.append_field(b"lasso_opening", &t_opening);
        let hyrax_proof = hyrax_open(&instance.tables[k], &r_vec, nu, sigma);

        let l_eval = l_poly.evaluate(&r_vec);

        table_openings.push(t_opening);
        sumcheck_proofs.push(sc_proof);
        hyrax_proofs.push(hyrax_proof);
        l_k_evals.push(l_eval);
    }

    LassoProof {
        outputs: instance.outputs.clone(),
        query_indices: query_indices.to_vec(),
        sub_claims,
        sumcheck_proofs,
        table_openings,
        hyrax_proofs,
        output_sumcheck: None,
        output_open: None,
        l_k_evals,
        index_proof: None,
    }
}

fn prove_lasso_committed_output_inner(
    instance: &LassoInstance,
    query_indices: &[usize],
    pk: &LassoProvingKey,
    output_binding: &LassoOutputBinding,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> (LassoProof, Vec<Vec<F>>, F) {
    let c = instance.tables.len();
    let m = instance.bits_per_chunk;
    let n = query_indices.len();
    let mask = (1usize << m) - 1;

    let nu = pk.nu;
    let sigma = params.sigma;
    assert_eq!(nu + sigma, instance.bits_per_chunk);
    for k in 0..c {
        absorb_com(transcript, b"hyrax_com", &pk.table_coms[k]);
    }
    absorb_com(transcript, b"lasso_output_com", &output_binding.com);

    let rho = transcript.challenge_field::<F>(b"lasso_rho");
    let rho_pows = powers_of(rho, n);

    let mut sumcheck_proofs = Vec::with_capacity(c);
    let mut table_openings = Vec::with_capacity(c);
    let mut hyrax_proofs = Vec::with_capacity(c);
    let mut sub_claims = Vec::with_capacity(c);
    let mut l_k_evals = Vec::with_capacity(c);
    let mut selector_points = Vec::with_capacity(c);

    for k in 0..c {
        let t_poly = DenseMLPoly::new(instance.tables[k].clone());
        let size = 1usize << m;
        let mut l_hist = vec![F::ZERO; size];
        for j in 0..n {
            let ch = chunk(query_indices[j], k, m, mask);
            l_hist[ch] += rho_pows[j];
        }
        let l_poly = DenseMLPoly::new(l_hist);
        let claimed: F = (0..n)
            .map(|j| rho_pows[j] * instance.tables[k][chunk(query_indices[j], k, m, mask)])
            .sum();
        sub_claims.push(claimed);
        let (sc_proof, r_vec) = prove_sumcheck(&t_poly, &l_poly, claimed, transcript);
        let t_opening = t_poly.evaluate(&r_vec);
        transcript.append_field(b"lasso_opening", &t_opening);
        let hyrax_proof = hyrax_open(&instance.tables[k], &r_vec, nu, sigma);
        table_openings.push(t_opening);
        sumcheck_proofs.push(sc_proof);
        hyrax_proofs.push(hyrax_proof);
        l_k_evals.push(l_poly.evaluate(&r_vec));
        selector_points.push(r_vec);
    }

    let combined_output_claim: F = sub_claims.iter().sum();
    let output_poly = DenseMLPoly::new(output_binding.mle_evals.clone());
    let mut weight_evals = vec![F::ZERO; 1usize << output_binding.num_vars];
    for j in 0..n {
        weight_evals[j] = rho_pows[j];
    }
    let weight_poly = DenseMLPoly::new(weight_evals);
    let (output_sumcheck, r_out) =
        prove_sumcheck(&output_poly, &weight_poly, combined_output_claim, transcript);
    let (nu_out, sigma_out, _) = params_from_vars(output_binding.num_vars);
    let output_open = hyrax_open(&output_binding.mle_evals, &r_out, nu_out, sigma_out);

    let proof = LassoProof {
        outputs: Vec::new(),
        query_indices: query_indices.to_vec(),
        sub_claims,
        sumcheck_proofs,
        table_openings,
        hyrax_proofs,
        output_sumcheck: Some(output_sumcheck),
        output_open: Some(output_open),
        l_k_evals,
        index_proof: None,
    };
    (proof, selector_points, rho)
}

pub fn prove_lasso_committed_output(
    instance: &LassoInstance,
    query_indices: &[usize],
    pk: &LassoProvingKey,
    output_binding: &LassoOutputBinding,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> LassoProof {
    prove_lasso_committed_output_inner(
        instance,
        query_indices,
        pk,
        output_binding,
        transcript,
        params,
    )
    .0
}

pub fn prove_lasso_committed_output_and_indices(
    instance: &LassoInstance,
    query_indices: &[usize],
    pk: &LassoProvingKey,
    output_binding: &LassoOutputBinding,
    input_binding: &LassoIndexBinding,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> LassoProof {
    let (mut proof, selector_points, rho) = prove_lasso_committed_output_inner(
        instance,
        query_indices,
        pk,
        output_binding,
        transcript,
        params,
    );
    let c = instance.tables.len();
    let index_proof = prove_index_binding(
        query_indices,
        c,
        instance.bits_per_chunk,
        input_binding,
        &proof.l_k_evals,
        &selector_points,
        rho,
        transcript,
    );
    proof.query_indices.clear();
    proof.index_proof = Some(index_proof);
    proof
}

pub fn verify_lasso(
    proof: &LassoProof,
    instance: &LassoInstance,
    vk: &LassoVerifyingKey,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    verify_lasso_with_outputs(
        proof,
        instance,
        vk,
        &instance.outputs,
        None,
        transcript,
        params,
    )
}

pub fn verify_lasso_from_proof_outputs(
    proof: &LassoProof,
    instance: &LassoInstance,
    vk: &LassoVerifyingKey,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    verify_lasso_with_outputs(
        proof,
        instance,
        vk,
        &proof.outputs,
        None,
        transcript,
        params,
    )
}

pub fn verify_lasso_committed_output(
    proof: &LassoProof,
    instance: &LassoInstance,
    vk: &LassoVerifyingKey,
    output_com: &HyraxCommitment,
    output_num_vars: usize,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    verify_lasso_with_outputs(
        proof,
        instance,
        vk,
        &[],
        Some((output_com, output_num_vars)),
        transcript,
        params,
    )
}

pub fn verify_lasso_committed_output_and_indices(
    proof: &LassoProof,
    instance: &LassoInstance,
    vk: &LassoVerifyingKey,
    output_com: &HyraxCommitment,
    output_num_vars: usize,
    input_com: &HyraxCommitment,
    input_num_vars: usize,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    verify_lasso_with_outputs(
        proof,
        instance,
        vk,
        &[],
        Some((output_com, output_num_vars)),
        transcript,
        params,
    )?;
    verify_index_binding(
        proof,
        instance.tables.len(),
        instance.bits_per_chunk,
        input_com,
        input_num_vars,
        transcript,
    )
}

fn verify_lasso_with_outputs(
    proof: &LassoProof,
    instance: &LassoInstance,
    vk: &LassoVerifyingKey,
    transcript_outputs: &[F],
    committed_output: Option<(&HyraxCommitment, usize)>,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let c = instance.tables.len();
    let m = instance.bits_per_chunk;
    let committed_index_mode = proof.index_proof.is_some();
    let n = if committed_index_mode {
        proof
            .index_proof
            .as_ref()
            .map(|p| p.query_len)
            .unwrap_or(0)
    } else {
        proof.query_indices.len()
    };

    let nu = m / 2;
    let sigma = m - nu;
    assert_eq!(params.sigma, sigma, "HyraxParams sigma mismatch");

    if proof.l_k_evals.len() != c {
        return Err("LassoProof l_k fields length mismatch".into());
    }

    // Replay precommitted table absorptions using verifying key
    for k in 0..c {
        absorb_com(transcript, b"hyrax_com", &vk.table_coms[k]);
    }
    if let Some((output_com, _)) = committed_output {
        absorb_com(transcript, b"lasso_output_com", output_com);
    }

    if committed_output.is_none() && (transcript_outputs.len() != n || proof.outputs.len() != n) {
        return Err(format!(
            "Lasso output length mismatch: transcript has {}, proof has {}, expected {n}",
            transcript_outputs.len(),
            proof.outputs.len()
        ));
    }
    for &out in transcript_outputs {
        transcript.append_field(b"lasso_out", &out);
    }
    let rho = transcript.challenge_field::<F>(b"lasso_rho");
    let rho_pows = powers_of(rho, n);
    if !committed_index_mode && proof.query_indices.len() != n {
        return Err(format!(
            "Lasso query index length mismatch: got {}, expected {}",
            proof.query_indices.len(),
            n
        ));
    }

    // Grand Sum Identity: Σ_j ρ^j · output_j = Σ_k sub_claim_k.
    // This is still O(n) but unavoidable given public outputs.
    let sub_claims_combined_sum: F = proof.sub_claims.iter().sum();
    if committed_output.is_none() {
        let output_batched_sum: F = (0..n).map(|j| rho_pows[j] * proof.outputs[j]).sum();
        if output_batched_sum != sub_claims_combined_sum {
            return Err("Lasso Grand Sum Identity failed".to_string());
        }
    }

    for k in 0..c {
        let (r_vec, _) = verify_sumcheck(
            &proof.sumcheck_proofs[k],
            proof.sub_claims[k],
            m,
            transcript,
        )
        .map_err(|e| format!("Table {k} sumcheck: {e}"))?;

        let t_opening = proof.table_openings[k];
        transcript.append_field(b"lasso_opening", &t_opening);

        // Verify sumcheck final and bind L_k(r) to the proof-carried indices.
        // The Hyrax opening still binds L_k(r) to the committed L_k polynomial.
        let l_at_r = proof.l_k_evals[k];
        if !committed_index_mode {
            let size = 1usize << m;
            let mask = size - 1;
            let mut expected_l_hist = vec![F::ZERO; size];
            for j in 0..n {
                let ch = chunk(proof.query_indices[j], k, m, mask);
                expected_l_hist[ch] += rho_pows[j];
            }
            let expected_l_at_r = DenseMLPoly::new(expected_l_hist).evaluate(&r_vec);
            if l_at_r != expected_l_at_r {
                return Err(format!(
                    "Table {k} selector is not derived from query indices"
                ));
            }
        } else {
            let index_proof = proof.index_proof.as_ref().unwrap();
            if index_proof.rho != rho {
                return Err("committed Lasso index challenge mismatch".into());
            }
            if index_proof.selector_points.get(k) != Some(&r_vec) {
                return Err(format!(
                    "Table {k} committed selector point does not match Lasso sumcheck point"
                ));
            }
        }
        let expected_final = t_opening * l_at_r;
        let actual_final =
            proof.sumcheck_proofs[k].final_eval_f * proof.sumcheck_proofs[k].final_eval_g;
        if expected_final != actual_final {
            return Err(format!(
                "Table {k} sumcheck final check failed: T(r)*L(r)={expected_final:?} ≠ final={actual_final:?}"
            ));
        }

        // Verify T_k(r) via Hyrax of committed table.
        hyrax_verify(
            &vk.table_coms[k],
            t_opening,
            &r_vec,
            &proof.hyrax_proofs[k],
            params,
        )
        .map_err(|e| format!("Table {k} Hyrax T: {e}"))?;

    }

    if let Some((output_com, output_num_vars)) = committed_output {
        let output_sumcheck = proof
            .output_sumcheck
            .as_ref()
            .ok_or_else(|| "missing committed Lasso output sumcheck".to_string())?;
        let output_open = proof
            .output_open
            .as_ref()
            .ok_or_else(|| "missing committed Lasso output opening".to_string())?;
        let (r_out, _) =
            verify_sumcheck(output_sumcheck, sub_claims_combined_sum, output_num_vars, transcript)
                .map_err(|e| format!("Lasso committed output sumcheck: {e}"))?;
        let mut weight_evals = vec![F::ZERO; 1usize << output_num_vars];
        for j in 0..n {
            weight_evals[j] = rho_pows[j];
        }
        let expected_weight_eval = DenseMLPoly::new(weight_evals).evaluate(&r_out);
        if output_sumcheck.final_eval_g != expected_weight_eval {
            return Err("Lasso committed output weight eval mismatch".into());
        }
        let (_, _, params_out) = params_from_vars(output_num_vars);
        hyrax_verify(
            output_com,
            output_sumcheck.final_eval_f,
            &r_out,
            output_open,
            &params_out,
        )
        .map_err(|e| format!("Lasso committed output opening: {e}"))?;
    }

    Ok(())
}

/// Binding between a Lasso instance's outputs and the polynomial commitment
/// used in the sub-module proof (a_com for FFN, phi_q_com / phi_k_com for attention).
///
/// Without this binding, a cheating prover could commit to a different `a` in
/// `a_com` while providing the correct activation values in `LassoInstance::outputs`,
/// bypassing the activation constraint entirely.
pub struct LassoOutputBinding {
    /// The Hyrax commitment that was created from this instance's output polynomial
    /// (e.g. a_com, phi_q_com, phi_k_com).
    pub com: HyraxCommitment,
    /// Number of MLE variables: t_bits + f_bits (FFN) or t_bits + d_bits (attention).
    pub num_vars: usize,
    /// Padded MLE evaluations — same layout used when `com` was created via hyrax_commit.
    pub mle_evals: Vec<F>,
}

/// Precomputed table commitments for multi-instance Lasso (setup phase).
#[derive(Clone)]
pub struct LassoMultiProvingKey {
    /// instance_table_coms[t][k] = commitment to sub-table k of instance t
    pub instance_table_coms: Vec<Vec<HyraxCommitment>>,
    pub nu: usize,
}

impl LassoMultiProvingKey {
    pub fn vk(&self) -> LassoMultiVerifyingKey {
        LassoMultiVerifyingKey {
            instance_table_coms: self.instance_table_coms.clone(),
        }
    }
}

/// Verifier-side key for multi-instance Lasso.
#[derive(Clone)]
pub struct LassoMultiVerifyingKey {
    pub instance_table_coms: Vec<Vec<HyraxCommitment>>,
}

/// Precommit tables for all instances in a multi-Lasso (call at setup, once).
pub fn precommit_lasso_multi_tables(
    multi_instance: &LassoMultiInstance,
    bits_per_chunk: usize,
    params: &HyraxParams,
) -> LassoMultiProvingKey {
    let nu = bits_per_chunk / 2;
    let instance_table_coms = multi_instance
        .instances
        .iter()
        .map(|inst| {
            inst.tables
                .iter()
                .map(|t| hyrax_commit(t, nu, params))
                .collect()
        })
        .collect();
    LassoMultiProvingKey {
        instance_table_coms,
        nu,
    }
}

/// 複数のLassoルックアップ要求をまとめたもの
pub struct LassoMultiInstance {
    pub instances: Vec<LassoInstance>,
}

/// 集約されたLasso証明
pub struct LassoMultiProof {
    /// Lookup outputs for every instance.
    pub all_outputs: Vec<Vec<F>>,
    /// Private lookup indices for every instance. Verifiers bind these to the
    /// corresponding committed lookup input tensors at the call site.
    pub all_query_indices: Vec<Vec<usize>>,
    /// 全インスタンスの重み付き合計: Σ_t α^t · (Σ_j ρ^j · output_{t,j})
    pub combined_grand_sum: F,
    /// 全インスタンス・全サブテーブルを統合した単一のSumcheck証明
    pub combined_sumcheck_proof: SumcheckProofMulti,
    /// 各サブテーブル T_{t,k}(r) の評価値
    pub table_openings: Vec<Vec<F>>,
    /// 全評価値を一括で証明するバッチHyrax証明 (for T_k tables)
    pub hyrax_proof: HyraxProof,
    /// Output binding: one Hyrax opening per instance binding output_com to
    /// the MLE of instance.outputs at a Fiat-Shamir challenge point.
    /// Order matches `output_bindings` passed to `prove_lasso_multi`.
    pub output_opening_proofs: Vec<HyraxProof>,
    /// Optional committed-output binding for model mode:
    /// proves combined_grand_sum = Σ_t α^t Σ_j ρ^j O_t[j], where O_t is
    /// already committed by the surrounding protocol.
    pub output_sumcheck: Option<SumcheckProofMulti>,
    /// Batched Hyrax opening for all committed output polynomials at the
    /// output_sumcheck challenge point.
    pub output_batch_open: Option<HyraxProof>,
    /// L_{t,k}(r) at the combined sumcheck output point r.
    pub l_k_evals_multi: Vec<Vec<F>>,
}

pub fn prove_lasso_multi(
    multi_instance: &LassoMultiInstance,
    // One slice of query_indices per instance (private witness, same order as multi_instance.instances).
    all_query_indices: &[Vec<usize>],
    pk: &LassoMultiProvingKey,
    // One binding per instance (same order as multi_instance.instances).
    output_bindings: &[LassoOutputBinding],
    transcript: &mut Transcript,
    _params: &HyraxParams,
) -> LassoMultiProof {
    let t_count = multi_instance.instances.len();
    let m = multi_instance.instances[0].bits_per_chunk;
    let mask = (1usize << m) - 1;
    assert_eq!(
        all_query_indices.len(),
        t_count,
        "all_query_indices length must match instance count"
    );

    // インスタンスリストが空でないことを確認
    assert!(
        !multi_instance.instances.is_empty(),
        "Multi-instance must contain at least one LassoInstance"
    );
    // すべてのインスタンスの bits_per_chunk が一致するかチェック
    for (i, inst) in multi_instance.instances.iter().enumerate() {
        assert_eq!(
            inst.bits_per_chunk, m,
            "LassoInstance at index {} has inconsistent bits_per_chunk: expected {}, found {}",
            i, m, inst.bits_per_chunk
        );
    }

    // --- STEP 1: absorb precommitted table commitments (from setup phase) ---
    for inst_coms in &pk.instance_table_coms {
        for com in inst_coms {
            absorb_com(transcript, b"hyrax_com", com);
        }
    }

    // --- STEP 2: Output absorption ---
    // In committed-output model mode, the outputs are bound by a later
    // sumcheck against existing commitments, so we do not stream them into the
    // transcript or outer proof.
    if output_bindings.is_empty() {
        for instance in &multi_instance.instances {
            for out in &instance.outputs {
                transcript.append_field(b"lasso_out", out);
            }
        }
    } else {
        for binding in output_bindings {
            absorb_com(transcript, b"lasso_output_com", &binding.com);
        }
    }

    // --- STEP 3: チャレンジの生成 ---
    let alpha = transcript.challenge_field::<F>(b"instance_batch_alpha");
    let rho = transcript.challenge_field::<F>(b"lookup_batch_rho");
    // 注: gammaはサブテーブルを個別に扱う場合に必要ですが、
    // 合計出力(output = ΣT_k)と照合する場合は不要です。

    let nu = m / 2;
    let sigma = m - nu;
    let alpha_pows = powers_of(alpha, t_count);

    // All instances share the same query length n; compute rho_pows once.
    let n = all_query_indices[0].len();
    debug_assert!(
        all_query_indices.iter().all(|qi| qi.len() == n),
        "All query index slices must have the same length"
    );
    let rho_pows = powers_of(rho, n);

    // Parallel grand sum: Σ_t α^t Σ_j ρ^j output_{t,j}.
    // Computed independently from the l_hist construction so the inner dot product
    // (O(n) per instance) runs in parallel across all L instances.
    let combined_grand_sum: F = multi_instance
        .instances
        .par_iter()
        .enumerate()
        .map(|(t, instance)| {
            let inst_sum: F = instance
                .outputs
                .iter()
                .zip(rho_pows.iter())
                .map(|(&out, &rp)| rp * out)
                .sum();
            alpha_pows[t] * inst_sum
        })
        .sum();

    // Sequential l_hist construction: ordering must be deterministic for downstream
    // poly indices, so we keep this loop sequential.
    let mut all_t_polys = Vec::new();
    let mut all_l_polys = Vec::new();
    let mut flatten_tables = Vec::new();
    for (t, instance) in multi_instance.instances.iter().enumerate() {
        let query_indices = &all_query_indices[t];

        for (k, table_evals) in instance.tables.iter().enumerate() {
            let t_poly = DenseMLPoly::new(table_evals.clone());
            let mut l_hist = vec![F::ZERO; 1 << m];
            for j in 0..n {
                let ch = chunk(query_indices[j], k, m, mask);
                l_hist[ch] += alpha_pows[t] * rho_pows[j];
            }
            all_t_polys.push(t_poly);
            all_l_polys.push(DenseMLPoly::new(l_hist.clone()));
            flatten_tables.push(table_evals);
        }
    }

    // --- STEP 4: Combined Sumcheck ---
    let (combined_sumcheck_proof, r_vec) = prove_sumcheck_multi_batched(
        &all_t_polys,
        &all_l_polys,
        &vec![F::ONE; all_t_polys.len()],
        combined_grand_sum,
        transcript,
    );

    let mut table_openings = Vec::new();
    for (t, instance) in multi_instance.instances.iter().enumerate() {
        let mut inst_openings = Vec::new();
        for k in 0..instance.tables.len() {
            let eval = all_t_polys[t * instance.tables.len() + k].evaluate(&r_vec);
            inst_openings.push(eval);
        }
        table_openings.push(inst_openings);
    }

    let flatten_tables_slices: Vec<&[F]> = flatten_tables.iter().map(|v| v.as_slice()).collect();
    let hyrax_proof = hyrax_open_batch(&flatten_tables_slices, &r_vec, nu, sigma, transcript);

    let mut l_k_evals_multi: Vec<Vec<F>> = Vec::new();
    for (t, instance) in multi_instance.instances.iter().enumerate() {
        let mut t_evals = Vec::new();
        for k in 0..instance.tables.len() {
            let l_eval = all_l_polys[t * instance.tables.len() + k].evaluate(&r_vec);
            t_evals.push(l_eval);
        }
        l_k_evals_multi.push(t_evals);
    }

    // --- OUTPUT BINDING ---
    // Model mode: bind the combined output grand sum directly to committed
    // output polynomials. This removes the need for the verifier to receive
    // every output value for the multi-Lasso instances.
    let output_opening_proofs = Vec::new();
    let mut output_sumcheck = None;
    let mut output_batch_open = None;
    if !output_bindings.is_empty() {
        assert_eq!(
            output_bindings.len(),
            multi_instance.instances.len(),
            "output_bindings length must equal number of Lasso instances"
        );
        let output_num_vars = output_bindings[0].num_vars;
        assert!(
            output_bindings
                .iter()
                .all(|binding| binding.num_vars == output_num_vars),
            "committed multi-Lasso outputs must share a domain"
        );
        let padded_len = 1usize << output_num_vars;
        let output_polys: Vec<DenseMLPoly> = output_bindings
            .iter()
            .map(|binding| DenseMLPoly::new(binding.mle_evals.clone()))
            .collect();
        let weight_polys: Vec<DenseMLPoly> = (0..t_count)
            .map(|t| {
                let mut evals = vec![F::ZERO; padded_len];
                for j in 0..n {
                    evals[j] = alpha_pows[t] * rho_pows[j];
                }
                DenseMLPoly::new(evals)
            })
            .collect();
        let weights = vec![F::ONE; t_count];
        let (sc, r_out) = prove_sumcheck_multi_batched(
            &output_polys,
            &weight_polys,
            &weights,
            combined_grand_sum,
            transcript,
        );
        let (nu, sigma, _) = params_from_vars(output_num_vars);
        let output_refs: Vec<&[F]> = output_bindings
            .iter()
            .map(|binding| binding.mle_evals.as_slice())
            .collect();
        output_batch_open = Some(hyrax_open_batch(
            &output_refs,
            &r_out,
            nu,
            sigma,
            transcript,
        ));
        output_sumcheck = Some(sc);
    }

    LassoMultiProof {
        all_outputs: if output_bindings.is_empty() {
            multi_instance
                .instances
                .iter()
                .map(|inst| inst.outputs.clone())
                .collect()
        } else {
            Vec::new()
        },
        all_query_indices: all_query_indices.to_vec(),
        combined_grand_sum,
        combined_sumcheck_proof,
        table_openings,
        hyrax_proof,
        output_opening_proofs,
        output_sumcheck,
        output_batch_open,
        l_k_evals_multi,
    }
}

pub fn verify_lasso_multi(
    proof: &LassoMultiProof,
    multi_instance: &LassoMultiInstance,
    vk: &LassoMultiVerifyingKey,
    // One (commitment, num_vars) pair per instance, same order as instances.
    // Pass &[] to skip output binding (e.g. in unit tests).
    output_coms: &[(HyraxCommitment, usize)],
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let transcript_outputs: Vec<&[F]> = multi_instance
        .instances
        .iter()
        .map(|instance| instance.outputs.as_slice())
        .collect();
    verify_lasso_multi_with_outputs(
        proof,
        multi_instance,
        vk,
        output_coms,
        &transcript_outputs,
        transcript,
        params,
    )
}

pub fn verify_lasso_multi_from_proof_outputs(
    proof: &LassoMultiProof,
    multi_instance: &LassoMultiInstance,
    vk: &LassoMultiVerifyingKey,
    output_coms: &[(HyraxCommitment, usize)],
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let transcript_outputs: Vec<&[F]> = proof
        .all_outputs
        .iter()
        .map(|outputs| outputs.as_slice())
        .collect();
    verify_lasso_multi_with_outputs(
        proof,
        multi_instance,
        vk,
        output_coms,
        &transcript_outputs,
        transcript,
        params,
    )
}

pub fn verify_lasso_multi_committed_outputs(
    proof: &LassoMultiProof,
    multi_instance: &LassoMultiInstance,
    vk: &LassoMultiVerifyingKey,
    output_coms: &[(HyraxCommitment, usize)],
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    if proof.output_sumcheck.is_none() || proof.output_batch_open.is_none() {
        return Err("committed multi-Lasso proof is missing output binding".into());
    }
    let empty_outputs: Vec<&[F]> = vec![&[]; multi_instance.instances.len()];
    verify_lasso_multi_with_outputs(
        proof,
        multi_instance,
        vk,
        output_coms,
        &empty_outputs,
        transcript,
        params,
    )
}

fn verify_lasso_multi_with_outputs(
    proof: &LassoMultiProof,
    multi_instance: &LassoMultiInstance,
    vk: &LassoMultiVerifyingKey,
    output_coms: &[(HyraxCommitment, usize)],
    transcript_outputs: &[&[F]],
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let t_count = multi_instance.instances.len();
    let m = multi_instance.instances[0].bits_per_chunk;

    // --- STEP 1: replay precommitted table absorptions (from verifying key) ---
    for inst_coms in &vk.instance_table_coms {
        for com in inst_coms {
            absorb_com(transcript, b"hyrax_com", com);
        }
    }
    if transcript_outputs.len() != t_count {
        return Err(format!(
            "Multi-Lasso transcript output count mismatch: got {}, expected {t_count}",
            transcript_outputs.len()
        ));
    }
    let committed_output_mode = proof.output_sumcheck.is_some();
    if !committed_output_mode {
        for outputs in transcript_outputs {
            for out in *outputs {
                transcript.append_field(b"lasso_out", out);
            }
        }
    } else {
        if output_coms.len() != t_count {
            return Err(format!(
                "output_coms length {} does not match instance count {t_count}",
                output_coms.len()
            ));
        }
        for (com, _) in output_coms {
            absorb_com(transcript, b"lasso_output_com", com);
        }
    }

    let alpha = transcript.challenge_field::<F>(b"instance_batch_alpha");
    let rho = transcript.challenge_field::<F>(b"lookup_batch_rho");
    let alpha_pows = powers_of(alpha, t_count);
    if proof.all_query_indices.len() != t_count {
        return Err(format!(
            "Multi-Lasso query index instance count mismatch: got {}, expected {}",
            proof.all_query_indices.len(),
            t_count
        ));
    }

    if !committed_output_mode && proof.all_outputs.len() != t_count {
        return Err(format!(
            "Multi-Lasso output instance count mismatch: got {}, expected {}",
            proof.all_outputs.len(),
            t_count
        ));
    }

    let n = proof
        .all_query_indices
        .first()
        .map(|qi| qi.len())
        .ok_or_else(|| "Multi-Lasso has no query-index instances".to_string())?;
    for (t, qi) in proof.all_query_indices.iter().enumerate() {
        if qi.len() != n {
            return Err(format!(
                "Multi-Lasso query index length mismatch at instance {t}: got {}, expected {n}",
                qi.len()
            ));
        }
    }
    let rho_pows = powers_of(rho, n);
    if !committed_output_mode {
        for (t, outputs) in proof.all_outputs.iter().enumerate() {
            if outputs.len() != n {
                return Err(format!(
                    "Multi-Lasso output length mismatch at instance {t}: got {}, expected {n}",
                    outputs.len()
                ));
            }
        }
        let expected_grand_sum: F = proof
            .all_outputs
            .par_iter()
            .enumerate()
            .map(|(t, outputs)| {
                let inst_sum: F = outputs
                    .iter()
                    .zip(rho_pows.iter())
                    .map(|(&out, &rp)| rp * out)
                    .sum();
                alpha_pows[t] * inst_sum
            })
            .sum();
        if expected_grand_sum != proof.combined_grand_sum {
            return Err("Multi-Lasso Grand Sum mismatch".into());
        }
    }

    // --- STEP 2: Sumcheck verification ---
    let total_tables: usize = multi_instance
        .instances
        .iter()
        .map(|i| i.tables.len())
        .sum();
    let (r_vec, sumcheck_final_val) = verify_sumcheck_multi_batched(
        &proof.combined_sumcheck_proof,
        &vec![F::ONE; total_tables],
        proof.combined_grand_sum,
        m,
        transcript,
    )?;

    // Verify sumcheck final via committed L_k evaluations and proof-carried indices.
    // l_k_evals_multi[t][k] = (alpha^t * L_{t,k})(r): the prover builds l_hist with
    // alpha-scaling already included, so the eval is alpha-scaled — no extra multiply needed.
    let mut expected_final_eval = F::ZERO;
    let mut table_idx = 0;

    for (t, instance) in multi_instance.instances.iter().enumerate() {
        for k in 0..instance.tables.len() {
            // l_tk_at_r already includes alpha_pows[t] (baked into l_hist during prove).
            let l_tk_at_r = proof.l_k_evals_multi[t][k];
            let size = 1usize << m;
            let mask = size - 1;
            let mut expected_l_hist = vec![F::ZERO; size];
            for j in 0..n {
                let ch = chunk(proof.all_query_indices[t][j], k, m, mask);
                expected_l_hist[ch] += alpha_pows[t] * rho_pows[j];
            }
            let expected_l_at_r = DenseMLPoly::new(expected_l_hist).evaluate(&r_vec);
            if l_tk_at_r != expected_l_at_r {
                return Err(format!(
                    "Weighted selector is not derived from query indices at instance {t}, table {k}"
                ));
            }

            if proof.table_openings[t][k] != proof.combined_sumcheck_proof.final_evals_f[table_idx]
            {
                return Err(format!("Table opening mismatch at index {table_idx}"));
            }
            if l_tk_at_r != proof.combined_sumcheck_proof.final_evals_g[table_idx] {
                return Err(format!(
                    "Weighted selector evaluation mismatch at index {table_idx}"
                ));
            }

            expected_final_eval += proof.table_openings[t][k] * l_tk_at_r;
            table_idx += 1;
        }
    }

    // 2. Sumcheckの最終評価値との一貫性チェック
    if expected_final_eval != sumcheck_final_val {
        return Err(
            "Multi-Lasso Sumcheck final check failed: expected sum does not match sumcheck claim"
                .into(),
        );
    }

    let flatten_commitments: Vec<_> = vk.instance_table_coms.iter().flatten().cloned().collect();
    let flatten_openings: Vec<_> = proof.table_openings.iter().flatten().copied().collect();

    // 検証側も transcript を含めた引数構成に修正
    hyrax_verify_batch(
        &flatten_commitments,
        &flatten_openings,
        &r_vec,
        &proof.hyrax_proof,
        params,
        transcript,
    )?;

    if committed_output_mode {
        if output_coms.len() != t_count {
            return Err(format!(
                "output_coms length {} does not match instance count {t_count}",
                output_coms.len()
            ));
        }
        let output_num_vars = output_coms[0].1;
        if !output_coms
            .iter()
            .all(|(_, num_vars)| *num_vars == output_num_vars)
        {
            return Err("committed multi-Lasso outputs must share a domain".into());
        }
        let sc = proof
            .output_sumcheck
            .as_ref()
            .ok_or_else(|| "missing committed output sumcheck".to_string())?;
        let output_batch_open = proof
            .output_batch_open
            .as_ref()
            .ok_or_else(|| "missing committed output batch opening".to_string())?;
        let weights = vec![F::ONE; t_count];
        let (r_out, _) = verify_sumcheck_multi_batched(
            sc,
            &weights,
            proof.combined_grand_sum,
            output_num_vars,
            transcript,
        )
        .map_err(|e| format!("Committed output sumcheck: {e}"))?;

        let padded_len = 1usize << output_num_vars;
        for t in 0..t_count {
            let mut evals = vec![F::ZERO; padded_len];
            for j in 0..n {
                evals[j] = alpha_pows[t] * rho_pows[j];
            }
            let expected_weight_eval = DenseMLPoly::new(evals).evaluate(&r_out);
            if sc.final_evals_g[t] != expected_weight_eval {
                return Err(format!(
                    "Committed output weight eval mismatch at instance {t}"
                ));
            }
        }

        let commitments: Vec<HyraxCommitment> =
            output_coms.iter().map(|(com, _)| com.clone()).collect();
        let evals = sc.final_evals_f.clone();
        let (_, _, params_out) = params_from_vars(output_num_vars);
        hyrax_verify_batch(
            &commitments,
            &evals,
            &r_out,
            output_batch_open,
            &params_out,
            transcript,
        )
        .map_err(|e| format!("Committed output batch opening: {e}"))?;
    } else if !output_coms.is_empty() {
        if output_coms.len() != multi_instance.instances.len() {
            return Err(format!(
                "output_coms length {} does not match instance count {}",
                output_coms.len(),
                multi_instance.instances.len()
            ));
        }
        if proof.output_opening_proofs.len() != output_coms.len() {
            return Err("output_opening_proofs count mismatch in proof".into());
        }
        let count = output_coms.len();

        // Phase 1: derive all r_out challenge points sequentially (transcript must be sequential).
        let r_outs: Vec<Vec<F>> = output_coms
            .iter()
            .map(|(_, num_vars)| challenge_vec(transcript, *num_vars, b"lasso_output_r"))
            .collect();

        // Phase 2: compute expected MLE evaluations in parallel.
        let expected_evals: Vec<F> = (0..count)
            .into_par_iter()
            .map(|i| {
                let (_, num_vars) = output_coms[i];
                let r_out = &r_outs[i];
                let padded_len = 1usize << num_vars;
                let mut padded = vec![F::ZERO; padded_len];
                for (j, &out) in proof.all_outputs[i].iter().enumerate() {
                    padded[j] = out;
                }
                DenseMLPoly::new(padded).evaluate(r_out)
            })
            .collect();

        // Phase 3: verify Hyrax proofs sequentially (uses only inner-product field ops and deferred MSM).
        for i in 0..count {
            let (com, num_vars) = &output_coms[i];
            let hp = &proof.output_opening_proofs[i];
            let (_, _, params_out) = params_from_vars(*num_vars);
            hyrax_verify(com, expected_evals[i], &r_outs[i], hp, &params_out)
                .map_err(|e| format!("Output binding Hyrax failed: {e}"))?;
        }
    }

    Ok(())
}

fn prove_index_binding(
    query_indices: &[usize],
    chunks: usize,
    bits_per_chunk: usize,
    input_binding: &LassoIndexBinding,
    l_k_evals: &[F],
    selector_points: &[Vec<F>],
    rho: F,
    transcript: &mut Transcript,
) -> LassoIndexProof {
    let n = query_indices.len();
    let padded_len = 1usize << input_binding.num_vars;
    assert!(n <= padded_len, "lookup index domain exceeds input binding");
    let (nu, sigma, params) = params_from_vars(input_binding.num_vars);
    let mask = (1usize << bits_per_chunk) - 1;

    let mut chunk_mles = Vec::with_capacity(chunks);
    let mut chunk_coms = Vec::with_capacity(chunks);
    for k in 0..chunks {
        let mut evals = vec![F::ZERO; padded_len];
        for j in 0..n {
            evals[j] = F::from(chunk(query_indices[j], k, bits_per_chunk, mask) as u64);
        }
        let com = hyrax_commit(&evals, nu, &params);
        absorb_com(transcript, b"lasso_idx_chunk_com", &com);
        chunk_coms.push(com);
        chunk_mles.push(DenseMLPoly::new(evals));
    }

    let bind_r = challenge_vec(transcript, input_binding.num_vars, b"lasso_idx_bind_r");
    let bind_chunk_evals: Vec<F> = chunk_mles.iter().map(|m| m.evaluate(&bind_r)).collect();
    transcript.append_field_vec(b"lasso_idx_bind_chunk_evals", &bind_chunk_evals);
    let bind_refs: Vec<&[F]> = std::iter::once(input_binding.mle_evals.as_slice())
        .chain(chunk_mles.iter().map(|m| m.evaluations.as_slice()))
        .collect();
    let bind_open = hyrax_open_batch(&bind_refs, &bind_r, nu, sigma, transcript);

    let mut weight_evals = vec![F::ZERO; padded_len];
    let mut cur = F::ONE;
    for j in 0..n {
        weight_evals[j] = cur;
        cur *= rho;
    }
    let weight_poly = DenseMLPoly::new(weight_evals);

    let mut selector_proofs = Vec::with_capacity(chunks);
    for k in 0..chunks {
        transcript.append_field(b"lasso_idx_selector_claim", &l_k_evals[k]);
        let (sumcheck, r_sel) = prove_selector_sumcheck(
            &chunk_mles[k],
            &weight_poly,
            l_k_evals[k],
            bits_per_chunk,
            &selector_points[k],
            transcript,
        );
        let chunk_open = hyrax_open(&chunk_mles[k].evaluations, &r_sel, nu, sigma);
        selector_proofs.push(SelectorBindingProof {
            sumcheck,
            chunk_open,
        });
    }

    LassoIndexProof {
        query_len: n,
        rho,
        chunk_coms,
        selector_points: selector_points.to_vec(),
        bind_chunk_evals,
        bind_open,
        selector_proofs,
    }
}

fn verify_index_binding(
    proof: &LassoProof,
    chunks: usize,
    bits_per_chunk: usize,
    input_com: &HyraxCommitment,
    input_num_vars: usize,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let index_proof = proof
        .index_proof
        .as_ref()
        .ok_or_else(|| "missing committed Lasso index proof".to_string())?;
    if index_proof.chunk_coms.len() != chunks
        || index_proof.bind_chunk_evals.len() != chunks
        || index_proof.selector_proofs.len() != chunks
        || index_proof.selector_points.len() != chunks
    {
        return Err("committed Lasso index proof chunk count mismatch".into());
    }
    let (_, _, params) = params_from_vars(input_num_vars);
    for com in &index_proof.chunk_coms {
        absorb_com(transcript, b"lasso_idx_chunk_com", com);
    }

    let bind_r = challenge_vec(transcript, input_num_vars, b"lasso_idx_bind_r");
    transcript.append_field_vec(
        b"lasso_idx_bind_chunk_evals",
        &index_proof.bind_chunk_evals,
    );
    let shift_mult = F::from(1u64 << bits_per_chunk);
    let mut shift = F::ONE;
    let mut input_eval = F::ZERO;
    for &chunk_eval in &index_proof.bind_chunk_evals {
        input_eval += shift * chunk_eval;
        shift *= shift_mult;
    }
    let commitments: Vec<HyraxCommitment> = std::iter::once(input_com.clone())
        .chain(index_proof.chunk_coms.iter().cloned())
        .collect();
    let evals: Vec<F> = std::iter::once(input_eval)
        .chain(index_proof.bind_chunk_evals.iter().copied())
        .collect();
    hyrax_verify_batch(
        &commitments,
        &evals,
        &bind_r,
        &index_proof.bind_open,
        &params,
        transcript,
    )
    .map_err(|e| format!("committed Lasso index fusion opening: {e}"))?;

    let rho = index_proof.rho;
    let padded_len = 1usize << input_num_vars;
    let mut weight_evals = vec![F::ZERO; padded_len];
    let mut cur = F::ONE;
    for j in 0..index_proof.query_len {
        weight_evals[j] = cur;
        cur *= rho;
    }
    let weight_poly = DenseMLPoly::new(weight_evals);

    for k in 0..chunks {
        let claim = proof.l_k_evals[k];
        transcript.append_field(b"lasso_idx_selector_claim", &claim);
        let selector = &index_proof.selector_proofs[k];
        let r_sel = verify_selector_sumcheck(
            &selector.sumcheck,
            &weight_poly,
            claim,
            bits_per_chunk,
            &index_proof.selector_points[k],
            transcript,
        )
        .map_err(|e| format!("committed Lasso selector {k}: {e}"))?;
        hyrax_verify(
            &index_proof.chunk_coms[k],
            selector.sumcheck.final_eval_chunk,
            &r_sel,
            &selector.chunk_open,
            &params,
        )
        .map_err(|e| format!("committed Lasso selector chunk opening {k}: {e}"))?;
    }
    Ok(())
}

fn prove_selector_sumcheck(
    chunk_poly: &DenseMLPoly,
    weight_poly: &DenseMLPoly,
    claim: F,
    bits_per_chunk: usize,
    table_point: &[F],
    transcript: &mut Transcript,
) -> (SelectorSumcheckProof, Vec<F>) {
    assert_eq!(chunk_poly.num_vars, weight_poly.num_vars);
    let mut chunk_cur = chunk_poly.clone();
    let mut weight_cur = weight_poly.clone();
    let degree = 1usize << bits_per_chunk;
    let mut running_claim = claim;
    let mut round_polys = Vec::with_capacity(chunk_poly.num_vars);
    let mut challenges = Vec::with_capacity(chunk_poly.num_vars);

    for _ in 0..chunk_poly.num_vars {
        let half = chunk_cur.evaluations.len() >> 1;
        let mut evals = Vec::with_capacity(degree + 1);
        for z_i in 0..=degree {
            let z = F::from(z_i as u64);
            let mut sum = F::ZERO;
            for i in 0..half {
                let c0 = chunk_cur.evaluations[i];
                let c1 = chunk_cur.evaluations[i + half];
                let w0 = weight_cur.evaluations[i];
                let w1 = weight_cur.evaluations[i + half];
                let c_z = c0 + z * (c1 - c0);
                let w_z = w0 + z * (w1 - w0);
                sum += w_z * selector_eq_eval(bits_per_chunk, c_z, table_point);
            }
            evals.push(sum);
        }
        let round = HighDegreeRoundPoly { evals };
        if round.evals[0] + round.evals[1] != running_claim {
            debug_assert_eq!(round.evals[0] + round.evals[1], running_claim);
        }
        transcript.append_field_vec(b"lasso_idx_sc_round", &round.evals);
        let r = transcript.challenge_field::<F>(b"lasso_idx_sc_r");
        running_claim = round.evaluate(r);
        round_polys.push(round);
        challenges.push(r);
        chunk_cur = fold_poly(&chunk_cur, r);
        weight_cur = fold_poly(&weight_cur, r);
    }

    (
        SelectorSumcheckProof {
            round_polys,
            final_eval_chunk: chunk_cur.evaluations[0],
        },
        challenges,
    )
}

fn verify_selector_sumcheck(
    proof: &SelectorSumcheckProof,
    weight_poly: &DenseMLPoly,
    claim: F,
    bits_per_chunk: usize,
    table_point: &[F],
    transcript: &mut Transcript,
) -> Result<Vec<F>, String> {
    if proof.round_polys.len() != weight_poly.num_vars {
        return Err("selector sumcheck round count mismatch".into());
    }
    let mut running_claim = claim;
    let mut r = Vec::with_capacity(weight_poly.num_vars);
    for round in &proof.round_polys {
        if round.evals.len() != (1usize << bits_per_chunk) + 1 {
            return Err("selector sumcheck degree mismatch".into());
        }
        if round.evals[0] + round.evals[1] != running_claim {
            return Err("selector sumcheck claim mismatch".into());
        }
        transcript.append_field_vec(b"lasso_idx_sc_round", &round.evals);
        let ch = transcript.challenge_field::<F>(b"lasso_idx_sc_r");
        running_claim = round.evaluate(ch);
        r.push(ch);
    }
    let final_weight = weight_poly.evaluate(&r);
    let expected = final_weight * selector_eq_eval(bits_per_chunk, proof.final_eval_chunk, table_point);
    if running_claim != expected {
        return Err("selector sumcheck final check failed".into());
    }
    Ok(r)
}

fn fold_poly(poly: &DenseMLPoly, r: F) -> DenseMLPoly {
    let half = poly.evaluations.len() >> 1;
    let evals = (0..half)
        .map(|i| poly.evaluations[i] + r * (poly.evaluations[i + half] - poly.evaluations[i]))
        .collect();
    DenseMLPoly::new(evals)
}

fn selector_eq_eval(bits_per_chunk: usize, z: F, _prefix: &[F]) -> F {
    let size = 1usize << bits_per_chunk;
    let mut acc = F::ZERO;
    for a in 0..size {
        let y = if z == F::from(a as u64) {
            F::ONE
        } else {
            let xa = F::from(a as u64);
            let mut num = F::ONE;
            let mut den = F::ONE;
            for b in 0..size {
                if a == b {
                    continue;
                }
                let xb = F::from(b as u64);
                num *= z - xb;
                den *= xa - xb;
            }
            num * den.inverse().expect("distinct interpolation points")
        };
        acc += eq_table_point(bits_per_chunk, a, _prefix) * y;
    }
    acc
}

fn eq_table_point(bits_per_chunk: usize, value: usize, point: &[F]) -> F {
    let mut acc = F::ONE;
    for i in 0..bits_per_chunk {
        let bit = (value >> (bits_per_chunk - 1 - i)) & 1;
        let r = point.get(i).copied().unwrap_or(F::ZERO);
        acc *= if bit == 1 { r } else { F::ONE - r };
    }
    acc
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the k-th chunk of `idx`: (idx >> (k*m)) & mask.
fn chunk(idx: usize, k: usize, m: usize, mask: usize) -> usize {
    (idx >> (k * m)) & mask
}

/// Compute [1, ρ, ρ², ..., ρ^{n-1}].
fn powers_of(rho: F, n: usize) -> Vec<F> {
    let mut v = Vec::with_capacity(n);
    let mut cur = F::ONE;
    for _ in 0..n {
        v.push(cur);
        cur *= rho;
    }
    v
}

#[cfg(test)]
mod lasso_tests {
    use super::*;
    use crate::field::F;
    use crate::transcript::Transcript;
    use ark_ff::One;

    /// Helper to create a dummy 2-chunk Lasso instance.
    /// Returns (instance, query_indices) — query_indices are private witness.
    fn setup_test_instance() -> (LassoInstance, Vec<usize>) {
        let m = 4;
        let table_size = 1 << m;

        // Table 0: T[i] = i
        let t0: Vec<F> = (0..table_size).map(|i| F::from(i as u64)).collect();
        // Table 1: T[i] = i * 10
        let t1: Vec<F> = (0..table_size).map(|i| F::from((i * 10) as u64)).collect();

        // Queries: [Index 0x12, Index 0x34]
        // 0x12 -> chunk0=2, chunk1=1. Output = T0[2] + T1[1] = 2 + 10 = 12
        // 0x34 -> chunk0=4, chunk1=3. Output = T0[4] + T1[3] = 4 + 30 = 34
        let query_indices = vec![0x12, 0x34];
        let outputs = vec![F::from(12), F::from(34)];

        (
            LassoInstance {
                tables: vec![t0, t1],
                outputs,
                bits_per_chunk: m,
            },
            query_indices,
        )
    }

    #[test]
    fn test_lasso_e2e_success() {
        let (instance, query_indices) = setup_test_instance();
        let m = instance.bits_per_chunk;

        // For Hyrax, nu + sigma = m. Let's pick sigma = m/2 = 2.
        let sigma = m / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        // --- Prover Side ---
        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let proof = prove_lasso(
            &instance,
            &query_indices,
            &pk,
            &mut prover_transcript,
            &params,
        );

        // --- Verifier Side ---
        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &vk, &mut verifier_transcript, &params);

        assert!(
            result.is_ok(),
            "Lasso verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_lasso_rejects_wrong_output() {
        let (mut instance, query_indices) = setup_test_instance();
        let sigma = instance.bits_per_chunk / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        // Maliciously change the claimed output
        instance.outputs[0] = F::from(999);

        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let proof = prove_lasso(
            &instance,
            &query_indices,
            &pk,
            &mut prover_transcript,
            &params,
        );

        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &vk, &mut verifier_transcript, &params);

        // Verifier should catch the grand sum mismatch
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Grand Sum Identity failed"));
    }

    #[test]
    fn test_lasso_tampered_sumcheck() {
        let (instance, query_indices) = setup_test_instance();
        let sigma = instance.bits_per_chunk / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let mut proof = prove_lasso(
            &instance,
            &query_indices,
            &pk,
            &mut prover_transcript,
            &params,
        );

        // Tamper with a sumcheck round polynomial
        proof.sumcheck_proofs[0].round_polys[0].evals[0] += F::one();

        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &vk, &mut verifier_transcript, &params);

        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_rejects_tampered_sub_claim_specifically() {
        let (instance, query_indices) = setup_test_instance();
        let sigma = instance.bits_per_chunk / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let mut proof = prove_lasso(
            &instance,
            &query_indices,
            &pk,
            &mut prover_transcript,
            &params,
        );

        // --- CLEVER TAMPERING ---
        // Add 1 to sub_claim[0] and subtract 1 from sub_claim[1].
        // The total sum (Grand Sum) remains unchanged, but sumcheck will reject.
        proof.sub_claims[0] += F::one();
        proof.sub_claims[1] -= F::one();

        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &vk, &mut verifier_transcript, &params);

        // Grand sum passes (sum unchanged), but per-table sumcheck fails.
        assert!(result.is_err());
    }

    /// Single-table Lasso: c=1, queries index directly into the table.
    #[test]
    fn test_lasso_single_table() {
        let m = 4;
        let table_size = 1 << m;
        // T[i] = i^2 mod table_size (structured table)
        let t0: Vec<F> = (0..table_size)
            .map(|i| F::from((i * i % table_size) as u64))
            .collect();

        let query_indices = vec![0, 1, 3, 7, 15];
        let outputs: Vec<F> = query_indices.iter().map(|&idx| t0[idx]).collect();

        let instance = LassoInstance {
            tables: vec![t0],
            outputs,
            bits_per_chunk: m,
        };
        let sigma = m / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"single-table");
        let proof = prove_lasso(&instance, &query_indices, &pk, &mut pt, &params);

        let mut vt = Transcript::new(b"single-table");
        let result = verify_lasso(&proof, &instance, &vk, &mut vt, &params);
        assert!(
            result.is_ok(),
            "single-table lasso failed: {:?}",
            result.err()
        );
    }

    /// Single-query Lasso: only one lookup entry to prove.
    #[test]
    fn test_lasso_single_query() {
        let m = 4;
        let table_size = 1 << m;
        let t0: Vec<F> = (0..table_size).map(|i| F::from(i as u64)).collect();
        let t1: Vec<F> = (0..table_size).map(|i| F::from((i * 5) as u64)).collect();

        // Single query: index 0x23 → chunk0=3, chunk1=2
        // Output = T0[3] + T1[2] = 3 + 10 = 13
        let query_indices = vec![0x23usize];
        let instance = LassoInstance {
            tables: vec![t0, t1],
            outputs: vec![F::from(13u64)],
            bits_per_chunk: m,
        };
        let sigma = m / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"single-query");
        let proof = prove_lasso(&instance, &query_indices, &pk, &mut pt, &params);

        let mut vt = Transcript::new(b"single-query");
        let result = verify_lasso(&proof, &instance, &vk, &mut vt, &params);
        assert!(
            result.is_ok(),
            "single-query lasso failed: {:?}",
            result.err()
        );
    }

    /// Verify the internal `chunk` helper function correctness.
    #[test]
    fn test_chunk_helper() {
        let m = 4;
        let mask = (1usize << m) - 1;

        // Index 0xAB: chunk0 = 0xB = 11, chunk1 = 0xA = 10
        assert_eq!(chunk(0xAB, 0, m, mask), 0xB);
        assert_eq!(chunk(0xAB, 1, m, mask), 0xA);

        // Index 0x12: chunk0 = 0x2 = 2, chunk1 = 0x1 = 1
        assert_eq!(chunk(0x12, 0, m, mask), 0x2);
        assert_eq!(chunk(0x12, 1, m, mask), 0x1);
    }

    /// Verify the `powers_of` helper function.
    #[test]
    fn test_powers_of_helper() {
        let rho = F::from(3u64);
        let pows = powers_of(rho, 5);
        assert_eq!(pows.len(), 5);
        assert_eq!(pows[0], F::ONE);
        assert_eq!(pows[1], F::from(3u64));
        assert_eq!(pows[2], F::from(9u64));
        assert_eq!(pows[3], F::from(27u64));
        assert_eq!(pows[4], F::from(81u64));
    }
}

#[cfg(test)]
mod lasso_multi_tests {
    use super::*;
    use crate::transcript::Transcript;
    use crate::{field::F, pcs::setup_hyrax_params};
    use ark_ff::One;

    // --- テスト用ヘルパー関数 ---

    /// インスタンス A: 2つのサブテーブルを持つルックアップ。Returns (instance, query_indices).
    fn setup_instance_a() -> (LassoInstance, Vec<usize>) {
        let m = 4;
        let table_size = 1 << m;
        // Table 0: T[i] = i, Table 1: T[i] = i * 10
        let t0: Vec<F> = (0..table_size).map(|i| F::from(i as u64)).collect();
        let t1: Vec<F> = (0..table_size).map(|i| F::from((i * 10) as u64)).collect();

        // Index 0x12 -> T0[2] + T1[1] = 2 + 10 = 12
        // Index 0x34 -> T0[4] + T1[3] = 4 + 30 = 34
        (
            LassoInstance {
                tables: vec![t0, t1],
                outputs: vec![F::from(12), F::from(34)],
                bits_per_chunk: m,
            },
            vec![0x12, 0x34],
        )
    }

    /// インスタンス B: 2つのサブテーブルを持つルックアップ。Returns (instance, query_indices).
    fn setup_instance_b() -> (LassoInstance, Vec<usize>) {
        let m = 4;
        let table_size = 1 << m;
        // T[i] = i * i
        let t0: Vec<F> = (0..table_size).map(|i| F::from((i * i) as u64)).collect();
        let t1: Vec<F> = (0..table_size)
            .map(|i| F::from((i * i * i) as u64))
            .collect();

        // Index 5 (=0x05): chunk0=5, chunk1=0. T0[5]+T1[0] = 25+0 = 25.
        // Index 1 (=0x01): chunk0=1, chunk1=0. T0[1]+T1[0] = 1+0 = 1.
        (
            LassoInstance {
                tables: vec![t0, t1],
                outputs: vec![F::from(25), F::from(1)],
                bits_per_chunk: m,
            },
            vec![5, 1],
        )
    }

    /// 複数のインスタンスを統合した MultiInstance を作成。Returns (multi_instance, all_query_indices).
    fn setup_multi_instance() -> (LassoMultiInstance, Vec<Vec<usize>>) {
        let (inst_a, qi_a) = setup_instance_a();
        let (inst_b, qi_b) = setup_instance_b();
        (
            LassoMultiInstance {
                instances: vec![inst_a, inst_b],
            },
            vec![qi_a, qi_b],
        )
    }

    // --- テストケース ---

    /// 正常系: 複数のインスタンスを単一の証明で一括検証
    #[test]
    fn test_lasso_multi_e2e_success() {
        let (multi_instance, all_qi) = setup_multi_instance();
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        // Prover
        let mut prover_transcript = Transcript::new(b"multi-lasso-test");
        let proof = prove_lasso_multi(
            &multi_instance,
            &all_qi,
            &pk,
            &[],
            &mut prover_transcript,
            &params,
        );

        // Verifier
        let mut verifier_transcript = Transcript::new(b"multi-lasso-test");
        let result = verify_lasso_multi(
            &proof,
            &multi_instance,
            &vk,
            &[],
            &mut verifier_transcript,
            &params,
        );

        assert!(
            result.is_ok(),
            "Multi-Lasso verification failed: {:?}",
            result.err()
        );
    }

    /// 異常系: 特定のインスタンスの公開出力値が改ざんされた場合
    #[test]
    fn test_lasso_multi_rejects_wrong_output() {
        let (mut multi_instance, all_qi) = setup_multi_instance();
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        // インスタンスBの出力を不正な値に変更
        multi_instance.instances[1].outputs[0] = F::from(999u64);

        let mut pt = Transcript::new(b"multi-lasso-test");
        let proof = prove_lasso_multi(&multi_instance, &all_qi, &pk, &[], &mut pt, &params);

        let mut vt = Transcript::new(b"multi-lasso-test");
        let result = verify_lasso_multi(&proof, &multi_instance, &vk, &[], &mut vt, &params);

        // 検証側で expected_final_eval と sumcheck の主張が一致しなくなるため失敗する
        assert!(result.is_err(), "Should reject due to wrong public output");
    }

    /// 異常系: 統合されたSumcheck証明の一部が改ざんされた場合
    #[test]
    fn test_lasso_multi_tampered_sumcheck() {
        let (multi_instance, all_qi) = setup_multi_instance();
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"multi-lasso-test");
        let mut proof = prove_lasso_multi(&multi_instance, &all_qi, &pk, &[], &mut pt, &params);

        // 統合Sumcheckのラウンド多項式を改ざん
        proof.combined_sumcheck_proof.round_polys[0].evals[0] += F::one();

        let mut vt = Transcript::new(b"multi-lasso-test");
        let result = verify_lasso_multi(&proof, &multi_instance, &vk, &[], &mut vt, &params);

        assert!(result.is_err(), "Should reject tampered sumcheck proof");
    }

    /// 異常系: トランスクリプト（コンテキスト）が不一致の場合
    #[test]
    fn test_lasso_multi_transcript_mismatch() {
        let (multi_instance, all_qi) = setup_multi_instance();
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"domain-A");
        let proof = prove_lasso_multi(&multi_instance, &all_qi, &pk, &[], &mut pt, &params);

        let mut vt = Transcript::new(b"domain-B"); // ラベルが異なる
        let result = verify_lasso_multi(&proof, &multi_instance, &vk, &[], &mut vt, &params);

        assert!(
            result.is_err(),
            "Should reject due to transcript domain mismatch"
        );
    }

    /// 異常系: プロverから送られた個別のテーブル評価値が改ざんされた場合
    #[test]
    fn test_lasso_multi_tampered_opening() {
        let (multi_instance, all_qi) = setup_multi_instance();
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"multi-lasso-test");
        let mut proof = prove_lasso_multi(&multi_instance, &all_qi, &pk, &[], &mut pt, &params);

        // 個別の評価値 table_openings を改ざん
        proof.table_openings[0][0] += F::one();

        let mut vt = Transcript::new(b"multi-lasso-test");
        let result = verify_lasso_multi(&proof, &multi_instance, &vk, &[], &mut vt, &params);

        // expected_final_eval の計算に不整合が生じるか、Hyraxのバッチ検証で失敗する
        assert!(result.is_err(), "Should reject tampered table opening");
    }

    /// 境界値: 1つのインスタンスのみが含まれる場合の動作確認
    #[test]
    fn test_lasso_multi_with_single_instance() {
        let (inst_a, qi_a) = setup_instance_a();
        let multi_instance = LassoMultiInstance {
            instances: vec![inst_a],
        };
        let all_qi = vec![qi_a];
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"single-in-multi");
        let proof = prove_lasso_multi(&multi_instance, &all_qi, &pk, &[], &mut pt, &params);

        let mut vt = Transcript::new(b"single-in-multi");
        let result = verify_lasso_multi(&proof, &multi_instance, &vk, &[], &mut vt, &params);

        assert!(
            result.is_ok(),
            "Single instance within multi-lasso should work"
        );
    }
}
