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
use crate::poly::utils::compute_eq_evals;
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
/// query_indices are now private witness — they are passed separately to prove_lasso
/// and are no longer needed by verify_lasso (L_k is committed in the proof instead).
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
    /// Batched sum per sub-table: Σ_j ρ^j · T_k[chunk_k(idx_j)].
    pub sub_claims: Vec<F>,
    /// Sumcheck proof per sub-table.
    pub sumcheck_proofs: Vec<SumcheckProof>,
    /// Claimed T_k(r_k) at the sumcheck random point.
    pub table_openings: Vec<F>,
    /// Hyrax opening proofs for T_k(r_k).
    pub hyrax_proofs: Vec<HyraxProof>,
    /// Committed counting polynomial L_k (size 2^m, one per sub-table).
    /// L_k[ch] = Σ_{j: chunk_k(idx_j)==ch} ρ^j.
    /// Committing L_k lets the verifier check L_k(r) via Hyrax instead of O(n) histogram work.
    pub l_k_coms: Vec<HyraxCommitment>,
    /// L_k evaluated at the sumcheck output point r (one per sub-table).
    pub l_k_evals: Vec<F>,
    /// Hyrax opening proofs for L_k(r).
    pub l_k_opens: Vec<HyraxProof>,
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
    let mut l_k_coms = Vec::with_capacity(c);
    let mut l_k_evals = Vec::with_capacity(c);
    let mut l_k_opens = Vec::with_capacity(c);

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

        // Commit L_k so the verifier can check L_k(r) via Hyrax instead of O(n) recomputation.
        let l_com = hyrax_commit(&l_hist, nu, params);
        absorb_com(transcript, b"lasso_l_com", &l_com);

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

        // Hyrax opening for L_k at the same r_vec (replaces O(n) hist recomputation in verifier)
        let l_eval = l_poly.evaluate(&r_vec);
        let l_open = hyrax_open(&l_hist, &r_vec, nu, sigma);

        table_openings.push(t_opening);
        sumcheck_proofs.push(sc_proof);
        hyrax_proofs.push(hyrax_proof);
        l_k_coms.push(l_com);
        l_k_evals.push(l_eval);
        l_k_opens.push(l_open);
    }

    LassoProof {
        sub_claims,
        sumcheck_proofs,
        table_openings,
        hyrax_proofs,
        l_k_coms,
        l_k_evals,
        l_k_opens,
    }
}

pub fn verify_lasso(
    proof: &LassoProof,
    instance: &LassoInstance,
    vk: &LassoVerifyingKey,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let c = instance.tables.len();
    let m = instance.bits_per_chunk;
    let n = instance.outputs.len();

    let nu = m / 2;
    let sigma = m - nu;
    assert_eq!(params.sigma, sigma, "HyraxParams sigma mismatch");

    if proof.l_k_coms.len() != c || proof.l_k_evals.len() != c || proof.l_k_opens.len() != c {
        return Err("LassoProof l_k fields length mismatch".into());
    }

    // Replay precommitted table absorptions using verifying key
    for k in 0..c {
        absorb_com(transcript, b"hyrax_com", &vk.table_coms[k]);
    }

    for &out in &instance.outputs {
        transcript.append_field(b"lasso_out", &out);
    }
    let rho = transcript.challenge_field::<F>(b"lasso_rho");
    let rho_pows = powers_of(rho, n);

    // Grand Sum Identity: Σ_j ρ^j · output_j = Σ_k sub_claim_k.
    // This is still O(n) but unavoidable given public outputs.
    let output_batched_sum: F = (0..n).map(|j| rho_pows[j] * instance.outputs[j]).sum();
    let sub_claims_combined_sum: F = proof.sub_claims.iter().sum();
    if output_batched_sum != sub_claims_combined_sum {
        return Err("Lasso Grand Sum Identity failed".to_string());
    }

    for k in 0..c {
        // Absorb committed L_k into transcript (matches prover's absorb_com call).
        absorb_com(transcript, b"lasso_l_com", &proof.l_k_coms[k]);

        let (r_vec, _) = verify_sumcheck(
            &proof.sumcheck_proofs[k],
            proof.sub_claims[k],
            m,
            transcript,
        )
        .map_err(|e| format!("Table {k} sumcheck: {e}"))?;

        let t_opening = proof.table_openings[k];
        transcript.append_field(b"lasso_opening", &t_opening);

        // Verify sumcheck final: T_k(r) * L_k(r) == sumcheck_final.
        // L_k(r) is verified via Hyrax of committed L_k — O(sqrt(2^m)) instead of O(n).
        let l_at_r = proof.l_k_evals[k];
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

        // Verify L_k(r) via Hyrax of committed counting polynomial (replaces O(n) histogram).
        hyrax_verify(
            &proof.l_k_coms[k],
            l_at_r,
            &r_vec,
            &proof.l_k_opens[k],
            params,
        )
        .map_err(|e| format!("Table {k} Hyrax L: {e}"))?;
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
    /// Committed counting polynomials L_{t,k}: l_k_coms_multi[t][k] = Hyrax commit of L_{t,k}.
    pub l_k_coms_multi: Vec<Vec<HyraxCommitment>>,
    /// L_{t,k}(r) at the combined sumcheck output point r.
    pub l_k_evals_multi: Vec<Vec<F>>,
    /// Hyrax opening proofs for L_{t,k}(r).
    pub l_k_opens_multi: Vec<Vec<HyraxProof>>,
}

pub fn prove_lasso_multi(
    multi_instance: &LassoMultiInstance,
    // One slice of query_indices per instance (private witness, same order as multi_instance.instances).
    all_query_indices: &[Vec<usize>],
    pk: &LassoMultiProvingKey,
    // One binding per instance (same order as multi_instance.instances).
    output_bindings: &[LassoOutputBinding],
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> LassoMultiProof {
    let t_count = multi_instance.instances.len();
    let m = multi_instance.instances[0].bits_per_chunk;
    let mask = (1usize << m) - 1;
    assert_eq!(all_query_indices.len(), t_count, "all_query_indices length must match instance count");

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

    // --- STEP 2: 公開出力の吸収 ---
    for instance in &multi_instance.instances {
        for out in &instance.outputs {
            transcript.append_field(b"lasso_out", out);
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
    let mut combined_grand_sum = F::ZERO;
    let mut all_t_polys = Vec::new();
    let mut all_l_polys = Vec::new();
    let mut flatten_tables = Vec::new();
    // Per-instance-per-table counting polynomial histograms (for committing after sumcheck).
    let mut all_l_hists: Vec<Vec<Vec<F>>> = Vec::new();

    for (t, instance) in multi_instance.instances.iter().enumerate() {
        let query_indices = &all_query_indices[t];
        let n = query_indices.len();
        let rho_pows = powers_of(rho, n);
        let mut inst_l_hists = Vec::new();

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
            inst_l_hists.push(l_hist);
        }
        all_l_hists.push(inst_l_hists);

        let instance_out_sum: F = (0..n).map(|j| rho_pows[j] * instance.outputs[j]).sum();
        combined_grand_sum += alpha_pows[t] * instance_out_sum;
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
    let hyrax_proof = hyrax_open_batch(
        &flatten_tables_slices,
        &r_vec,
        nu,
        sigma,
        transcript,
    );

    // Commit L_{t,k} for each instance and sub-table so the verifier can use Hyrax
    // instead of O(n) histogram recomputation.
    let mut l_k_coms_multi: Vec<Vec<HyraxCommitment>> = Vec::new();
    let mut l_k_evals_multi: Vec<Vec<F>> = Vec::new();
    let mut l_k_opens_multi: Vec<Vec<HyraxProof>> = Vec::new();
    for (t, instance) in multi_instance.instances.iter().enumerate() {
        let mut t_coms = Vec::new();
        let mut t_evals = Vec::new();
        let mut t_opens = Vec::new();
        for k in 0..instance.tables.len() {
            let l_hist = &all_l_hists[t][k];
            let l_com = hyrax_commit(l_hist, nu, params);
            absorb_com(transcript, b"lasso_multi_l_com", &l_com);
            let l_eval = all_l_polys[t * instance.tables.len() + k].evaluate(&r_vec);
            let l_open = hyrax_open(l_hist, &r_vec, nu, sigma);
            t_coms.push(l_com);
            t_evals.push(l_eval);
            t_opens.push(l_open);
        }
        l_k_coms_multi.push(t_coms);
        l_k_evals_multi.push(t_evals);
        l_k_opens_multi.push(t_opens);
    }

    // --- OUTPUT BINDING ---
    // For each instance, open the module-level output commitment at a fresh
    // Fiat-Shamir challenge point and verify it matches the MLE of instance.outputs.
    // This binds a_com / phi_q_com / phi_k_com to the outputs used above.
    // Empty slice means no binding (used in unit tests that don't have real commitments).
    let mut output_opening_proofs = Vec::new();
    if !output_bindings.is_empty() {
        assert_eq!(
            output_bindings.len(),
            multi_instance.instances.len(),
            "output_bindings length must equal number of Lasso instances"
        );
        for binding in output_bindings {
            let r_out = challenge_vec(transcript, binding.num_vars, b"lasso_output_r");
            let (nu, sigma, _) = params_from_vars(binding.num_vars);
            let hp = hyrax_open(&binding.mle_evals, &r_out, nu, sigma);
            output_opening_proofs.push(hp);
        }
    }

    LassoMultiProof {
        combined_grand_sum,
        combined_sumcheck_proof,
        table_openings,
        hyrax_proof,
        output_opening_proofs,
        l_k_coms_multi,
        l_k_evals_multi,
        l_k_opens_multi,
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
    let t_count = multi_instance.instances.len();
    let m = multi_instance.instances[0].bits_per_chunk;

    // --- STEP 1: replay precommitted table absorptions (from verifying key) ---
    for inst_coms in &vk.instance_table_coms {
        for com in inst_coms {
            absorb_com(transcript, b"hyrax_com", com);
        }
    }
    for instance in &multi_instance.instances {
        for out in &instance.outputs {
            transcript.append_field(b"lasso_out", out);
        }
    }

    let alpha = transcript.challenge_field::<F>(b"instance_batch_alpha");
    let rho = transcript.challenge_field::<F>(b"lookup_batch_rho");
    let alpha_pows = powers_of(alpha, t_count);

    // Grand Sum: Σ_t α^t * Σ_j ρ^j * output_{t,j}
    // Still O(Σ_t n_t) but unavoidable given public outputs.
    let mut expected_grand_sum = F::ZERO;
    for (t, instance) in multi_instance.instances.iter().enumerate() {
        let n = instance.outputs.len();
        let rho_pows = powers_of(rho, n);
        let inst_sum: F = (0..n).map(|j| rho_pows[j] * instance.outputs[j]).sum();
        expected_grand_sum += alpha_pows[t] * inst_sum;
    }
    if expected_grand_sum != proof.combined_grand_sum {
        return Err("Multi-Lasso Grand Sum mismatch".into());
    }

    // --- STEP 2: Sumcheck verification ---
    let total_tables: usize = multi_instance.instances.iter().map(|i| i.tables.len()).sum();
    let (r_vec, sumcheck_final_val) = verify_sumcheck_multi_batched(
        &proof.combined_sumcheck_proof,
        &vec![F::ONE; total_tables],
        proof.combined_grand_sum,
        m,
        transcript,
    )?;

    // Verify sumcheck final via committed L_k evaluations (no O(n) histogram needed).
    // l_k_evals_multi[t][k] = (alpha^t * L_{t,k})(r): the prover builds l_hist with
    // alpha-scaling already included, so the eval is alpha-scaled — no extra multiply needed.
    let mut expected_final_eval = F::ZERO;
    let mut table_idx = 0;

    for (t, instance) in multi_instance.instances.iter().enumerate() {
        for k in 0..instance.tables.len() {
            // l_tk_at_r already includes alpha_pows[t] (baked into l_hist during prove).
            let l_tk_at_r = proof.l_k_evals_multi[t][k];

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

    // --- OUTPUT BINDING verification ---
    // If output_coms is non-empty, verify that each instance's output commitment
    // opens correctly to the MLE of instance.outputs at a fresh random point.
    if !output_coms.is_empty() {
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
        for (((com, num_vars), hp), instance) in output_coms
            .iter()
            .zip(proof.output_opening_proofs.iter())
            .zip(multi_instance.instances.iter())
        {
            let r_out = challenge_vec(transcript, *num_vars, b"lasso_output_r");
            let (_, _, params_out) = params_from_vars(*num_vars);

            // Compute expected eval: MLE of instance.outputs (zero-padded) at r_out
            let padded_len = 1usize << num_vars;
            let mut padded = vec![F::ZERO; padded_len];
            for (j, &out) in instance.outputs.iter().enumerate() {
                padded[j] = out;
            }
            let expected_eval = DenseMLPoly::new(padded).evaluate(&r_out);

            hyrax_verify(com, expected_eval, &r_out, hp, &params_out)
                .map_err(|e| format!("Output binding Hyrax failed: {e}"))?;
        }
    }

    Ok(())
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

        (LassoInstance {
            tables: vec![t0, t1],
            outputs,
            bits_per_chunk: m,
        }, query_indices)
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
        let proof = prove_lasso(&instance, &query_indices, &pk, &mut prover_transcript, &params);

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
        let proof = prove_lasso(&instance, &query_indices, &pk, &mut prover_transcript, &params);

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
        let mut proof = prove_lasso(&instance, &query_indices, &pk, &mut prover_transcript, &params);

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
        let mut proof = prove_lasso(&instance, &query_indices, &pk, &mut prover_transcript, &params);

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
    use ark_ff::{One, Zero};

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
        (LassoInstance {
            tables: vec![t0, t1],
            outputs: vec![F::from(12), F::from(34)],
            bits_per_chunk: m,
        }, vec![0x12, 0x34])
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
        (LassoInstance {
            tables: vec![t0, t1],
            outputs: vec![F::from(25), F::from(1)],
            bits_per_chunk: m,
        }, vec![5, 1])
    }

    /// 複数のインスタンスを統合した MultiInstance を作成。Returns (multi_instance, all_query_indices).
    fn setup_multi_instance() -> (LassoMultiInstance, Vec<Vec<usize>>) {
        let (inst_a, qi_a) = setup_instance_a();
        let (inst_b, qi_b) = setup_instance_b();
        (LassoMultiInstance {
            instances: vec![inst_a, inst_b],
        }, vec![qi_a, qi_b])
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
        let proof = prove_lasso_multi(&multi_instance, &all_qi, &pk, &[], &mut prover_transcript, &params);

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
