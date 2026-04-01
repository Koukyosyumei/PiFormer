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

use crate::field::{eq_eval, index_to_bits, F};
use crate::pcs::{
    hyrax_commit, hyrax_open, hyrax_open_batch, hyrax_verify, HyraxCommitment, HyraxParams,
    HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::sumcheck::{prove_sumcheck_multi_batched, SumcheckProofMulti};
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::Field;

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
    /// Hyrax commitments to each sub-table (sent to verifier).
    pub hyrax_commitments: Vec<HyraxCommitment>,
    /// Hyrax opening proofs for T_k(r_k).
    pub hyrax_proofs: Vec<HyraxProof>,
}

pub fn prove_lasso(
    instance: &LassoInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> LassoProof {
    let c = instance.tables.len();
    let m = instance.bits_per_chunk;
    let n = instance.query_indices.len();
    let mask = (1usize << m) - 1;

    // nu + sigma = m; choose nu = m/2 (square-ish layout)
    let nu = m / 2;
    let sigma = m - nu;
    assert_eq!(params.sigma, sigma, "HyraxParams sigma mismatch");

    // Step 1: commit to each sub-table and absorb commitments into transcript
    let mut hyrax_commitments = Vec::with_capacity(c);
    for k in 0..c {
        let commitment = hyrax_commit(&instance.tables[k], nu, params);
        for pt in &commitment.row_coms {
            let bytes = {
                use ark_serialize::CanonicalSerialize;
                let mut buf = Vec::new();
                pt.serialize_compressed(&mut buf).unwrap();
                buf
            };
            transcript.append_bytes(b"hyrax_com", &bytes);
        }
        hyrax_commitments.push(commitment);
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

        // Build selector polynomial L_k as a dense MLE
        let size = 1usize << m;
        let mut l_evals = vec![F::ZERO; size];
        for j in 0..n {
            let ch = chunk(instance.query_indices[j], k, m, mask);
            for x in 0..size {
                let mut eq_val = F::ONE;
                for bit in 0..m {
                    let a = F::from(((ch >> bit) & 1) as u64);
                    let b = F::from(((x >> bit) & 1) as u64);
                    eq_val *= a * b + (F::ONE - a) * (F::ONE - b);
                }
                l_evals[x] += rho_pows[j] * eq_val;
            }
        }
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
        hyrax_commitments,
        hyrax_proofs,
    }
}

pub fn verify_lasso(
    proof: &LassoProof,
    instance: &LassoInstance,
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

    // Replay commitment absorptions
    for k in 0..c {
        for pt in &proof.hyrax_commitments[k].row_coms {
            let bytes = {
                use ark_serialize::CanonicalSerialize;
                let mut buf = Vec::new();
                pt.serialize_compressed(&mut buf).unwrap();
                buf
            };
            transcript.append_bytes(b"hyrax_com", &bytes);
        }
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
        let l_at_r: F = (0..n)
            .map(|j| {
                let ch = chunk(instance.query_indices[j], k, m, mask);
                let bits = index_to_bits(ch, m);
                rho_pows[j] * eq_eval(&bits, &r_rev)
            })
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
            &proof.hyrax_commitments[k],
            t_opening,
            &r_vec,
            &proof.hyrax_proofs[k],
            params,
        )
        .map_err(|e| format!("Table {k} Hyrax: {e}"))?;
    }
    Ok(())
}

/*
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
    /// インスタンスごとのHyraxコミットメント
    pub hyrax_commitments: Vec<Vec<HyraxCommitment>>,
    /// 全評価値を一括で証明するバッチHyrax証明
    pub hyrax_proof: HyraxProof,
}

pub fn prove_lasso_multi(
    multi_instance: &LassoMultiInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> LassoMultiProof {
    let t_count = multi_instance.instances.len();
    let m = multi_instance.instances[0].bits_per_chunk; // 全インスタンスで共通と仮定

    // 1. すべてのサブテーブルに対してHyraxコミットメントを作成 [cite: 1104]
    let mut hyrax_commitments = Vec::new();
    for instance in &multi_instance.instances {
        let coms: Vec<_> = instance
            .tables
            .iter()
            .map(|table| hyrax_commit(table, m / 2, params))
            .collect();
        hyrax_commitments.push(coms);
    }

    // 2. チャレンジの生成 (Lassoのバッチ化 )
    let alpha = transcript.challenge_field::<F>(b"instance_batch_alpha"); // インスタンス間
    let rho = transcript.challenge_field::<F>(b"lookup_batch_rho"); // 同一インスタンス内
    let gamma = transcript.challenge_field::<F>(b"table_batch_gamma"); // サブテーブル間

    let alpha_pows = powers_of(alpha, t_count);
    let mut combined_grand_sum = F::ZERO;
    let mut all_t_polys = Vec::new(); // 全ての T_{t,k}
    let mut all_l_polys = Vec::new(); // 全ての α^t * γ^k * L_{t,k}

    // 3. インスタンスごとに多項式を構築
    for (t, instance) in multi_instance.instances.iter().enumerate() {
        let n = instance.query_indices.len();
        let rho_pows = powers_of(rho, n);
        let gamma_pows = powers_of(gamma, instance.tables.len());

        for (k, table_evals) in instance.tables.iter().enumerate() {
            let t_poly = DenseMLPoly::new(table_evals.clone());

            // 重み付けされたセレクタ L_{t,k} の構築
            let mut l_evals = vec![F::ZERO; 1 << m];
            for j in 0..n {
                let ch = chunk(instance.query_indices[j], k, m, instance.mask());
                // α^t * γ^k * ρ^j を重みとして eq(ch, x) に加算
                l_evals[ch] += alpha_pows[t] * gamma_pows[k] * rho_pows[j];
            }

            all_t_polys.push(t_poly);
            all_l_polys.push(DenseMLPoly::new(l_evals));
        }

        // インスタンスの出力を加算
        let instance_out_sum: F = (0..n).map(|j| rho_pows[j] * instance.outputs[j]).sum();
        combined_grand_sum += alpha_pows[t] * instance_out_sum;
    }

    // 4. 単一の集約Sumcheckプロトコルを実行
    // P(r) = Σ_t Σ_k (α^t * γ^k) * T_{t,k}(r) * L_{t,k}(r)
    let (combined_sumcheck_proof, r_vec) = prove_sumcheck_multi_batched(
        &all_t_polys,
        &all_l_polys,
        &alpha_pows,
        combined_grand_sum,
        transcript,
    );

    // 5. Hyraxオープニングの生成 [cite: 1210]
    let mut table_openings = Vec::new();
    let mut flatten_tables = Vec::new();
    let mut flatten_openings = Vec::new();

    for (t, instance) in multi_instance.instances.iter().enumerate() {
        let mut inst_openings = Vec::new();
        for k in 0..instance.tables.len() {
            let eval = all_t_polys[t * instance.tables.len() + k].evaluate(&r_vec);
            inst_openings.push(eval);
            flatten_tables.push(&instance.tables[k]);
            flatten_openings.push(eval);
        }
        table_openings.push(inst_openings);
    }

    let hyrax_proof = hyrax_open_batch(&flatten_tables, &r_vec, flatten_openings, params);

    LassoMultiProof {
        combined_grand_sum,
        combined_sumcheck_proof,
        table_openings,
        hyrax_commitments,
        hyrax_proof,
    }
}*/

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

        // --- Prover Side ---
        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let proof = prove_lasso(&instance, &mut prover_transcript, &params);

        // --- Verifier Side ---
        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &mut verifier_transcript, &params);

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

        // Maliciously change the claimed output
        instance.outputs[0] = F::from(999);

        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let proof = prove_lasso(&instance, &mut prover_transcript, &params);

        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &mut verifier_transcript, &params);

        // Verifier should catch the sub-claim mismatch immediately
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Grand Sum Identity failed"));
    }

    #[test]
    fn test_lasso_tampered_sumcheck() {
        let instance = setup_test_instance();
        let sigma = instance.bits_per_chunk / 2;
        let params = HyraxParams::new(sigma);

        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let mut proof = prove_lasso(&instance, &mut prover_transcript, &params);

        // Tamper with a sumcheck round polynomial
        proof.sumcheck_proofs[0].round_polys[0].evals[0] += F::one();

        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &mut verifier_transcript, &params);

        assert!(result.is_err());
    }

    #[test]
    fn test_lasso_rejects_tampered_sub_claim_specifically() {
        let instance = setup_test_instance();
        let sigma = instance.bits_per_chunk / 2;
        let params = HyraxParams::new(sigma);
        let mut prover_transcript = Transcript::new(b"lasso-protocol");
        let mut proof = prove_lasso(&instance, &mut prover_transcript, &params);

        // --- CLEVER TAMPERING ---
        // Add 1 to sub_claim[0] and subtract 1 from sub_claim[1].
        // The total sum (Grand Sum) remains unchanged!
        proof.sub_claims[0] += F::one();
        proof.sub_claims[1] -= F::one();

        let mut verifier_transcript = Transcript::new(b"lasso-protocol");
        let result = verify_lasso(&proof, &instance, &mut verifier_transcript, &params);

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

        let mut pt = Transcript::new(b"single-table");
        let proof = prove_lasso(&instance, &mut pt, &params);

        let mut vt = Transcript::new(b"single-table");
        let result = verify_lasso(&proof, &instance, &mut vt, &params);
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

        let mut pt = Transcript::new(b"single-query");
        let proof = prove_lasso(&instance, &mut pt, &params);

        let mut vt = Transcript::new(b"single-query");
        let result = verify_lasso(&proof, &instance, &mut vt, &params);
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
