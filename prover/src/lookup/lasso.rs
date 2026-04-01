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
    HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::sumcheck::{
    prove_sumcheck_multi_batched, verify_sumcheck_multi_batched, SumcheckProofMulti,
};
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
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
#[derive(Clone)]
pub struct LassoInstance {
    /// c sub-tables T_0, ..., T_{c-1}, each of size 2^bits_per_chunk.
    pub tables: Vec<Vec<F>>,
    /// Query indices in [0, 2^(c*bits_per_chunk)).
    pub query_indices: Vec<usize>,
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
}

pub fn prove_lasso(
    instance: &LassoInstance,
    pk: &LassoProvingKey,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> LassoProof {
    let c = instance.tables.len();
    let m = instance.bits_per_chunk;
    let n = instance.query_indices.len();
    let mask = (1usize << m) - 1;

    // Step 1: absorb precommitted table commitments into transcript
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

    // Commit claimed outputs to transcript
    for &out in &instance.outputs {
        transcript.append_field(b"lasso_out", &out);
    }

    // Squeeze batching challenge ρ
    let rho = transcript.challenge_field::<F>(b"lasso_rho");
    let rho_pows = powers_of(rho, n);

    let mut sumcheck_proofs = Vec::with_capacity(c);
    let mut table_openings = Vec::with_capacity(c);
    let mut hyrax_proofs = Vec::with_capacity(c);
    let mut sub_claims = Vec::with_capacity(c);

    for k in 0..c {
        let t_poly = DenseMLPoly::new(instance.tables[k].clone());

        // Build selector polynomial L_k as a dense MLE.
        // Each x ∈ [size] is independent — compute in parallel over x.
        let size = 1usize << m;
        let l_evals: Vec<F> = (0..size)
            .into_iter()
            .map(|x| {
                (0..n)
                    .map(|j| {
                        let ch = chunk(instance.query_indices[j], k, m, mask);
                        let eq_val: F = (0..m)
                            .map(|bit| {
                                let a = F::from(((ch >> bit) & 1) as u64);
                                let b = F::from(((x >> bit) & 1) as u64);
                                a * b + (F::ONE - a) * (F::ONE - b)
                            })
                            .product();
                        rho_pows[j] * eq_val
                    })
                    .sum()
            })
            .collect();
        let l_poly = DenseMLPoly::new(l_evals);

        // Claimed sum for this sub-table
        let claimed: F = (0..n)
            .map(|j| rho_pows[j] * instance.tables[k][chunk(instance.query_indices[j], k, m, mask)])
            .sum();
        sub_claims.push(claimed);

        let (sc_proof, r_vec) = prove_sumcheck(&t_poly, &l_poly, claimed, transcript);

        // Hyrax opening at r_vec
        let opening = t_poly.evaluate(&r_vec);
        transcript.append_field(b"lasso_opening", &opening);
        let hyrax_proof = hyrax_open(&instance.tables[k], &r_vec, nu, sigma);

        table_openings.push(opening);
        sumcheck_proofs.push(sc_proof);
        hyrax_proofs.push(hyrax_proof);
    }

    LassoProof {
        sub_claims,
        sumcheck_proofs,
        table_openings,
        hyrax_proofs,
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
    let n = instance.query_indices.len();
    let mask = (1usize << m) - 1;

    let nu = m / 2;
    let sigma = m - nu;
    assert_eq!(params.sigma, sigma, "HyraxParams sigma mismatch");

    // Replay precommitted table absorptions using verifying key
    for k in 0..c {
        absorb_com(transcript, b"hyrax_com", &vk.table_coms[k]);
    }

    for &out in &instance.outputs {
        transcript.append_field(b"lasso_out", &out);
    }
    let rho = transcript.challenge_field::<F>(b"lasso_rho");
    let rho_pows = powers_of(rho, n);

    // 1. Calculate the batched sum of claimed outputs: Σ_j ρ^j · output_j
    let output_batched_sum: F = (0..n).map(|j| rho_pows[j] * instance.outputs[j]).sum();
    // 2. Calculate the sum of all sub-claims: Σ_k (Σ_j ρ^j · T_k[chunk_k(idx_j)])
    let sub_claims_combined_sum: F = proof.sub_claims.iter().sum();
    // 3. The Grand Sum Identity: The sum of outputs must match the sum of table lookups
    if output_batched_sum != sub_claims_combined_sum {
        return Err(
            "Lasso Grand Sum Identity failed: outputs do not match table lookups".to_string(),
        );
    }

    for k in 0..c {
        // Check sub-claim matches public inputs
        let expected: F = (0..n)
            .map(|j| rho_pows[j] * instance.tables[k][chunk(instance.query_indices[j], k, m, mask)])
            .sum();
        if expected != proof.sub_claims[k] {
            return Err(format!("Lasso sub-claim mismatch for table {k}"));
        }

        let (r_vec, _) = verify_sumcheck(
            &proof.sumcheck_proofs[k],
            proof.sub_claims[k],
            m,
            transcript,
        )
        .map_err(|e| format!("Table {k} sumcheck: {e}"))?;

        // Replay opening absorption
        let t_opening = proof.table_openings[k];
        transcript.append_field(b"lasso_opening", &t_opening);

        // Verify sumcheck final claim: T_k(r) · L_k(r) == final
        //
        // Note on bit ordering: DenseMLPoly fixes the MSB first (r_vec[0] = bit_{m-1}).
        // index_to_bits is LSB-first, and eq_eval pairs r_rev[i] with bits[i].
        // So we reverse r_vec before computing L_k(r).
        let r_rev: Vec<F> = r_vec.iter().copied().rev().collect();
        // Precompute eq_table for this table's r_rev (r_rev differs per table k).
        let eq_table_k = compute_eq_evals(&r_rev, 1 << m);
        let mut hist_k = vec![F::ZERO; 1 << m];
        for j in 0..n {
            let ch = chunk(instance.query_indices[j], k, m, mask);
            hist_k[ch] += rho_pows[j];
        }
        let l_at_r: F = hist_k
            .iter()
            .zip(eq_table_k.iter())
            .map(|(&h, &e)| h * e)
            .sum();

        let expected_final = t_opening * l_at_r;
        let actual_final =
            proof.sumcheck_proofs[k].final_eval_f * proof.sumcheck_proofs[k].final_eval_g;
        if expected_final != actual_final {
            return Err(format!(
                "Table {k} sumcheck final check failed: T(r)*L(r)={expected_final:?} ≠ final={actual_final:?}"
            ));
        }

        // Verify Hyrax opening: T_k(r_vec) == t_opening
        hyrax_verify(
            &vk.table_coms[k],
            t_opening,
            &r_vec,
            &proof.hyrax_proofs[k],
            params,
        )
        .map_err(|e| format!("Table {k} Hyrax: {e}"))?;
    }
    Ok(())
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
    /// 全評価値を一括で証明するバッチHyrax証明
    pub hyrax_proof: HyraxProof,
}

pub fn prove_lasso_multi(
    multi_instance: &LassoMultiInstance,
    pk: &LassoMultiProvingKey,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> LassoMultiProof {
    let t_count = multi_instance.instances.len();
    let m = multi_instance.instances[0].bits_per_chunk;
    let mask = (1usize << m) - 1;

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

    let alpha_pows = powers_of(alpha, t_count);
    let mut combined_grand_sum = F::ZERO;
    let mut all_t_polys = Vec::new();
    let mut all_l_polys = Vec::new();
    let mut flatten_tables = Vec::new();

    for (t, instance) in multi_instance.instances.iter().enumerate() {
        let n = instance.query_indices.len();
        let rho_pows = powers_of(rho, n);

        for (k, table_evals) in instance.tables.iter().enumerate() {
            let t_poly = DenseMLPoly::new(table_evals.clone());
            let mut l_evals = vec![F::ZERO; 1 << m];
            for j in 0..n {
                let ch = chunk(instance.query_indices[j], k, m, mask);
                // 重みは α^t * ρ^j のみ (Σ_k T_k = output に対応させるため)
                l_evals[ch] += alpha_pows[t] * rho_pows[j];
            }
            all_t_polys.push(t_poly);
            all_l_polys.push(DenseMLPoly::new(l_evals));
            flatten_tables.push(table_evals);
        }

        let instance_out_sum: F = (0..n).map(|j| rho_pows[j] * instance.outputs[j]).sum();
        combined_grand_sum += alpha_pows[t] * instance_out_sum;
    }

    // --- STEP 4: Sumcheckの実行 ---
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

    // 型の不一致を修正: Vec<&Vec<F>> から Vec<&[F]> への変換
    let flatten_tables_slices: Vec<&[F]> = flatten_tables.iter().map(|v| v.as_slice()).collect();

    // 引数を5つに修正
    let hyrax_proof = hyrax_open_batch(
        &flatten_tables_slices,
        &r_vec,
        m / 2,      // nu
        m - m / 2,  // sigma
        transcript, // 5th argument
    );

    LassoMultiProof {
        combined_grand_sum,
        combined_sumcheck_proof,
        table_openings,
        hyrax_proof,
    }
}

pub fn verify_lasso_multi(
    proof: &LassoMultiProof,
    multi_instance: &LassoMultiInstance,
    vk: &LassoMultiVerifyingKey,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let t_count = multi_instance.instances.len();
    let m = multi_instance.instances[0].bits_per_chunk;
    let mask = (1usize << m) - 1;

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

    // チャレンジ生成
    let alpha = transcript.challenge_field::<F>(b"instance_batch_alpha");
    let rho = transcript.challenge_field::<F>(b"lookup_batch_rho");
    let alpha_pows = powers_of(alpha, t_count);

    // --- STEP 2: Sumcheckの検証 ---
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

    let r_rev: Vec<F> = r_vec.iter().copied().rev().collect();
    // Precompute eq_table[ch] = eq(index_to_bits(ch,m), r_rev) for ch in 0..2^m.
    // compute_eq_evals builds the same table in O(2^m) via iterative doubling,
    // avoiding n*m individual eq_eval calls inside the inner loop.
    let eq_table = compute_eq_evals(&r_rev, 1 << m);
    let mut expected_final_eval = F::ZERO;
    let mut table_idx = 0;

    for (t, instance) in multi_instance.instances.iter().enumerate() {
        let n = instance.query_indices.len();
        let rho_pows = powers_of(rho, n);

        for k in 0..instance.tables.len() {
            // Build histogram: hist[ch] = Σ_{j: chunk_k(query_j)==ch} rho_pows[j]
            // Then l_tk_at_r = <hist, eq_table>, reducing O(n*m) → O(n + 2^m).
            let mut hist = vec![F::ZERO; 1 << m];
            for j in 0..n {
                let ch = chunk(instance.query_indices[j], k, m, mask);
                hist[ch] += rho_pows[j];
            }
            let l_tk_at_r: F = hist
                .iter()
                .zip(eq_table.iter())
                .map(|(&h, &e)| h * e)
                .sum();

            // 重み付けされたセレクタ評価値
            let weighted_l = alpha_pows[t] * l_tk_at_r;

            // 重要: Sumcheck内の個別の評価値が、検証者の計算およびOpeningと一致するか確認
            if proof.table_openings[t][k] != proof.combined_sumcheck_proof.final_evals_f[table_idx]
            {
                return Err(format!("Table opening mismatch at index {table_idx}"));
            }
            if weighted_l != proof.combined_sumcheck_proof.final_evals_g[table_idx] {
                return Err(format!(
                    "Weighted selector evaluation mismatch at index {table_idx}"
                ));
            }

            expected_final_eval += proof.table_openings[t][k] * weighted_l;
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
    )
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
    /// Table size = 2^4 = 16. Total address space = 16 * 16 = 256.
    fn setup_test_instance() -> LassoInstance {
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

        LassoInstance {
            tables: vec![t0, t1],
            query_indices,
            outputs,
            bits_per_chunk: m,
        }
    }

    #[test]
    fn test_lasso_e2e_success() {
        let instance = setup_test_instance();
        let m = instance.bits_per_chunk;

        // For Hyrax, nu + sigma = m. Let's pick sigma = m/2 = 2.
        let sigma = m / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        // --- Prover Side ---
        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let proof = prove_lasso(&instance, &pk, &mut prover_transcript, &params);

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
        let mut instance = setup_test_instance();
        let sigma = instance.bits_per_chunk / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        // Maliciously change the claimed output
        instance.outputs[0] = F::from(999);

        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let proof = prove_lasso(&instance, &pk, &mut prover_transcript, &params);

        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &vk, &mut verifier_transcript, &params);

        // Verifier should catch the sub-claim mismatch immediately
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Grand Sum Identity failed"));
    }

    #[test]
    fn test_lasso_tampered_sumcheck() {
        let instance = setup_test_instance();
        let sigma = instance.bits_per_chunk / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let mut proof = prove_lasso(&instance, &pk, &mut prover_transcript, &params);

        // Tamper with a sumcheck round polynomial
        proof.sumcheck_proofs[0].round_polys[0].evals[0] += F::one();

        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &vk, &mut verifier_transcript, &params);

        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_rejects_tampered_sub_claim_specifically() {
        let instance = setup_test_instance();
        let sigma = instance.bits_per_chunk / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let mut proof = prove_lasso(&instance, &pk, &mut prover_transcript, &params);

        // --- CLEVER TAMPERING ---
        // Add 1 to sub_claim[0] and subtract 1 from sub_claim[1].
        // The total sum (Grand Sum) remains unchanged!
        proof.sub_claims[0] += F::one();
        proof.sub_claims[1] -= F::one();

        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &vk, &mut verifier_transcript, &params);

        assert!(result.is_err());
        let err_msg = result.unwrap_err();

        // Now this check will definitely fail!
        assert!(err_msg.contains("Lasso sub-claim mismatch for table 0"));
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
            query_indices,
            outputs,
            bits_per_chunk: m,
        };
        let sigma = m / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"single-table");
        let proof = prove_lasso(&instance, &pk, &mut pt, &params);

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
        let instance = LassoInstance {
            tables: vec![t0, t1],
            query_indices: vec![0x23],
            outputs: vec![F::from(13u64)],
            bits_per_chunk: m,
        };
        let sigma = m / 2;
        let params = HyraxParams::new(sigma);

        let pk = precommit_lasso_tables(&instance.tables, instance.bits_per_chunk, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"single-query");
        let proof = prove_lasso(&instance, &pk, &mut pt, &params);

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

    /// インスタンス A: 2つのサブテーブルを持つルックアップ
    fn setup_instance_a() -> LassoInstance {
        let m = 4;
        let table_size = 1 << m;
        // Table 0: T[i] = i, Table 1: T[i] = i * 10
        let t0: Vec<F> = (0..table_size).map(|i| F::from(i as u64)).collect();
        let t1: Vec<F> = (0..table_size).map(|i| F::from((i * 10) as u64)).collect();

        // Index 0x12 -> T0[2] + T1[1] = 2 + 10 = 12
        // Index 0x34 -> T0[4] + T1[3] = 4 + 30 = 34
        LassoInstance {
            tables: vec![t0, t1],
            query_indices: vec![0x12, 0x34],
            outputs: vec![F::from(12), F::from(34)],
            bits_per_chunk: m,
        }
    }

    /// インスタンス B: 1つのサブテーブル（二乗テーブル）を持つルックアップ
    fn setup_instance_b() -> LassoInstance {
        let m = 4;
        let table_size = 1 << m;
        // T[i] = i * i
        let t0: Vec<F> = (0..table_size).map(|i| F::from((i * i) as u64)).collect();
        let t1: Vec<F> = (0..table_size)
            .map(|i| F::from((i * i * i) as u64))
            .collect();

        // Index 5 -> 25, Index 1 -> 1
        LassoInstance {
            tables: vec![t0, t1],
            query_indices: vec![5, 1],
            outputs: vec![F::from(25), F::from(1)],
            bits_per_chunk: m,
        }
    }

    /// 複数のインスタンスを統合した MultiInstance を作成
    fn setup_multi_instance() -> LassoMultiInstance {
        LassoMultiInstance {
            instances: vec![setup_instance_a(), setup_instance_b()],
        }
    }

    // --- テストケース ---

    /// 正常系: 複数のインスタンスを単一の証明で一括検証
    #[test]
    fn test_lasso_multi_e2e_success() {
        let multi_instance = setup_multi_instance();
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        // Prover
        let mut prover_transcript = Transcript::new(b"multi-lasso-test");
        let proof = prove_lasso_multi(&multi_instance, &pk, &mut prover_transcript, &params);

        // Verifier
        let mut verifier_transcript = Transcript::new(b"multi-lasso-test");
        let result = verify_lasso_multi(
            &proof,
            &multi_instance,
            &vk,
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
        let mut multi_instance = setup_multi_instance();
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        // インスタンスBの出力を不正な値に変更
        multi_instance.instances[1].outputs[0] = F::from(999u64);

        let mut pt = Transcript::new(b"multi-lasso-test");
        let proof = prove_lasso_multi(&multi_instance, &pk, &mut pt, &params);

        let mut vt = Transcript::new(b"multi-lasso-test");
        let result = verify_lasso_multi(&proof, &multi_instance, &vk, &mut vt, &params);

        // 検証側で expected_final_eval と sumcheck の主張が一致しなくなるため失敗する
        assert!(result.is_err(), "Should reject due to wrong public output");
    }

    /// 異常系: 統合されたSumcheck証明の一部が改ざんされた場合
    #[test]
    fn test_lasso_multi_tampered_sumcheck() {
        let multi_instance = setup_multi_instance();
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"multi-lasso-test");
        let mut proof = prove_lasso_multi(&multi_instance, &pk, &mut pt, &params);

        // 統合Sumcheckのラウンド多項式を改ざん
        proof.combined_sumcheck_proof.round_polys[0].evals[0] += F::one();

        let mut vt = Transcript::new(b"multi-lasso-test");
        let result = verify_lasso_multi(&proof, &multi_instance, &vk, &mut vt, &params);

        assert!(result.is_err(), "Should reject tampered sumcheck proof");
    }

    /// 異常系: トランスクリプト（コンテキスト）が不一致の場合
    #[test]
    fn test_lasso_multi_transcript_mismatch() {
        let multi_instance = setup_multi_instance();
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"domain-A");
        let proof = prove_lasso_multi(&multi_instance, &pk, &mut pt, &params);

        let mut vt = Transcript::new(b"domain-B"); // ラベルが異なる
        let result = verify_lasso_multi(&proof, &multi_instance, &vk, &mut vt, &params);

        assert!(
            result.is_err(),
            "Should reject due to transcript domain mismatch"
        );
    }

    /// 異常系: プロverから送られた個別のテーブル評価値が改ざんされた場合
    #[test]
    fn test_lasso_multi_tampered_opening() {
        let multi_instance = setup_multi_instance();
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"multi-lasso-test");
        let mut proof = prove_lasso_multi(&multi_instance, &pk, &mut pt, &params);

        // 個別の評価値 table_openings を改ざん
        proof.table_openings[0][0] += F::one();

        let mut vt = Transcript::new(b"multi-lasso-test");
        let result = verify_lasso_multi(&proof, &multi_instance, &vk, &mut vt, &params);

        // expected_final_eval の計算に不整合が生じるか、Hyraxのバッチ検証で失敗する
        assert!(result.is_err(), "Should reject tampered table opening");
    }

    /// 境界値: 1つのインスタンスのみが含まれる場合の動作確認
    #[test]
    fn test_lasso_multi_with_single_instance() {
        let multi_instance = LassoMultiInstance {
            instances: vec![setup_instance_a()],
        };
        let m = multi_instance.instances[0].bits_per_chunk;
        let params = setup_hyrax_params(m);

        let pk = precommit_lasso_multi_tables(&multi_instance, m, &params);
        let vk = pk.vk();

        let mut pt = Transcript::new(b"single-in-multi");
        let proof = prove_lasso_multi(&multi_instance, &pk, &mut pt, &params);

        let mut vt = Transcript::new(b"single-in-multi");
        let result = verify_lasso_multi(&proof, &multi_instance, &vk, &mut vt, &params);

        assert!(
            result.is_ok(),
            "Single instance within multi-lasso should work"
        );
    }
}
