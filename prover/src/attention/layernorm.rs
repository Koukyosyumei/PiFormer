//! LayerNorm Protocol with Constraint Fusion & MLE Evaluation
//!
//! 平均と分散の集計（Sumcheck）において、特定のインデックスをサンプリングするのではなく、
//! 行列の行次元(T)の多線形延長(MLE)をランダムな点 r_t で評価し、その評価値に対して
//! 列次元(D)のSumcheckを実行します（GKRプロトコルに準拠）。

use crate::field::F;
use crate::lookup::range::{prove_range, verify_range, RangeProof, RangeProofInstance};
use crate::pcs::HyraxParams;
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::{One, PrimeField, Zero};

/// LayerNormの公開入力と証明者のアドバイス（コミットメント）
#[derive(Clone)]
pub struct LayerNormInstance {
    pub seq_len: usize,
    pub d_head: usize,
    pub x: Vec<Vec<F>>,
    pub gamma: Vec<F>,
    pub beta: Vec<F>,
    pub y: Vec<Vec<F>>,

    // Proverがコミットする中間値
    pub sum_x: Vec<F>,
    pub var_x: Vec<F>,
    pub sigma: Vec<F>,
    pub scale_gamma: F,
    pub scale_beta: F,
}

pub struct LayerNormProof {
    pub mean_sumcheck: SumcheckProof,
    pub variance_sumcheck: SumcheckProof,
    pub sigma_range_proof: RangeProof,
    pub y_range_proof: RangeProof,
    // Verifierの最終チェック用（実際のプロトコルではPCSのOpen値）
    pub x_eval_at_r: F,
}

pub fn prove_layernorm(
    inst: &LayerNormInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<LayerNormProof, String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);

    // -----------------------------------------------------------------------
    // 1. MLE空間上のランダムな点 r_t による監査 (GKRアプローチ)
    // -----------------------------------------------------------------------
    // Verifierは整数インデックスではなく、F^t_bits のベクトルをチャレンジとして送る
    let r_t = generate_challenge_vector(transcript, t_bits, b"layernorm_audit_rt");

    // Proverは、1次元配列 sum_x と var_x の MLE を r_t で評価した値を計算
    let claim_mean = eval_1d(&inst.sum_x, &r_t);
    let claim_var = eval_1d(&inst.var_x, &r_t);

    transcript.append_field(b"claimed_mean", &claim_mean);
    transcript.append_field(b"claimed_var", &claim_var);

    // 行列 X の行を r_t で潰す（評価する） -> 長さ D の1次元ベクトルになる
    let x_collapsed = eval_cols(&inst.x, &r_t, t, d);

    // --- Mean Sumcheck ---
    // \sum_j X(r_t, j) * 1 = claim_mean
    let f_mean = DenseMLPoly::from_vec_padded(x_collapsed.clone());
    let g_mean = DenseMLPoly::from_vec_padded(vec![F::ONE; d]);
    let (mean_sumcheck, r_d_mean) = prove_sumcheck(&f_mean, &g_mean, claim_mean, transcript);

    // --- Variance Sumcheck ---
    // \sum_j (d * X(r_t, j) - claim_mean)^2 = claim_var
    let mut h_var_vec = Vec::with_capacity(d);
    for j in 0..d {
        h_var_vec.push((d_f * x_collapsed[j]) - claim_mean);
    }
    let f_var = DenseMLPoly::from_vec_padded(h_var_vec.clone());
    let g_var = DenseMLPoly::from_vec_padded(h_var_vec); // (d*X - mean) * (d*X - mean)
    let (variance_sumcheck, r_d_var) = prove_sumcheck(&f_var, &g_var, claim_var, transcript);

    // -----------------------------------------------------------------------
    // 2. Sigma Square Root Constraint (Range Proof)
    // -----------------------------------------------------------------------
    // (2*sigma_i - 1)^2 * d^2 <= 4 * V_i <= (2*sigma_i + 1)^2 * d^2 - 1
    // Range Proof (Lasso) はベクトル全体に対して一括で行われるためMLE評価は不要
    let mut sigma_residuals = Vec::with_capacity(t * 2);
    let two = F::from(2u64);
    let four = F::from(4u64);
    let d_sq = d_f * d_f;

    for i in 0..t {
        let sig = inst.sigma[i];
        let v = inst.var_x[i];
        let sig_minus = (two * sig) - F::ONE;
        let sig_plus = (two * sig) + F::ONE;

        sigma_residuals.push((four * v) - (sig_minus * sig_minus * d_sq));
        sigma_residuals.push((sig_plus * sig_plus * d_sq) - F::ONE - (four * v));
    }
    let sigma_rp_inst = RangeProofInstance {
        values: sigma_residuals,
        bits: 32,
    };
    let sigma_range_proof = prove_range(&sigma_rp_inst, transcript, params)?;

    // -----------------------------------------------------------------------
    // 3. Division & Affine Constraint (Range Proof)
    // -----------------------------------------------------------------------
    // d * sigma_i * (2 * y_ij - 1) <= 2 * expr <= d * sigma_i * (2 * y_ij + 1) - 1
    let mut y_residuals = Vec::with_capacity(t * d * 2);
    for i in 0..t {
        let sig_d = inst.sigma[i] * d_f;
        let sum_i = inst.sum_x[i];
        for j in 0..d {
            let expr = (inst.scale_gamma * inst.gamma[j] * ((d_f * inst.x[i][j]) - sum_i))
                + (inst.scale_beta * inst.beta[j] * sig_d);
            let expr_2 = two * expr;
            let y_ij = inst.y[i][j];

            y_residuals.push(expr_2 - (sig_d * ((two * y_ij) - F::ONE)));
            y_residuals.push((sig_d * ((two * y_ij) + F::ONE)) - F::ONE - expr_2);
        }
    }
    let y_rp_inst = RangeProofInstance {
        values: y_residuals,
        bits: 32,
    };
    let y_range_proof = prove_range(&y_rp_inst, transcript, params)?;

    // 最終的な X の評価値 (Sumcheckの検証に必要)
    let x_eval_at_r = f_mean.evaluate(&r_d_mean);

    Ok(LayerNormProof {
        mean_sumcheck,
        variance_sumcheck,
        sigma_range_proof,
        y_range_proof,
        x_eval_at_r,
    })
}

pub fn verify_layernorm(
    proof: &LayerNormProof,
    inst: &LayerNormInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);

    // 1. MLE空間上のランダムベクトル r_t を再現
    let r_t = generate_challenge_vector(transcript, t_bits, b"layernorm_audit_rt");

    // （実際のプロトコルでは、ここでPCSを使って sum_x(r_t) と var_x(r_t) の開示を受けます）
    let claim_mean = eval_1d(&inst.sum_x, &r_t);
    let claim_var = eval_1d(&inst.var_x, &r_t);

    // --- Mean Sumcheck の検証 ---
    transcript.append_field(b"claimed_mean", &claim_mean);
    let (r_d_mean, final_claim_mean) =
        verify_sumcheck(&proof.mean_sumcheck, claim_mean, d_bits, transcript)
            .map_err(|e| format!("Mean sumcheck failed: {}", e))?;

    // final_claim_mean == X(r_t, r_d_mean) * 1
    if final_claim_mean != proof.x_eval_at_r {
        return Err("Mean sumcheck final evaluation mismatch".to_string());
    }

    // --- Variance Sumcheck の検証 ---
    transcript.append_field(b"claimed_var", &claim_var);
    let (r_d_var, final_claim_var) =
        verify_sumcheck(&proof.variance_sumcheck, claim_var, d_bits, transcript)
            .map_err(|e| format!("Variance sumcheck failed: {}", e))?;

    // final_claim_var == (d * X(r_t, r_d_var) - claim_mean)^2
    // 実際にはPCSでX(r_t, r_d_var)をOpenして確認しますが、ここではシミュレート
    let x_eval_var = eval_2d(&inst.x, &r_t, &r_d_var, t, d);
    let expected_var_claim = (d_f * x_eval_var - claim_mean) * (d_f * x_eval_var - claim_mean);
    if final_claim_var != expected_var_claim {
        return Err("Variance sumcheck final evaluation mismatch".to_string());
    }

    // 2. Sigma Range Proof の検証 (Lasso)
    let mut sigma_residuals = Vec::with_capacity(t * 2);
    let two = F::from(2u64);
    let four = F::from(4u64);
    let d_sq = d_f * d_f;
    for i in 0..t {
        let sig = inst.sigma[i];
        let v = inst.var_x[i];
        sigma_residuals.push((four * v) - (((two * sig) - F::ONE) * ((two * sig) - F::ONE) * d_sq));
        sigma_residuals
            .push((((two * sig) + F::ONE) * ((two * sig) + F::ONE) * d_sq) - F::ONE - (four * v));
    }
    let sigma_rp_inst = RangeProofInstance {
        values: sigma_residuals,
        bits: 32,
    };
    verify_range(&proof.sigma_range_proof, &sigma_rp_inst, transcript, params)
        .map_err(|e| format!("Sigma Range Proof Error: {}", e))?;

    // 3. Y Range Proof の検証 (Lasso)
    let mut y_residuals = Vec::with_capacity(t * d * 2);
    for i in 0..t {
        let sig_d = inst.sigma[i] * d_f;
        let sum_i = inst.sum_x[i];
        for j in 0..d {
            let expr = (inst.scale_gamma * inst.gamma[j] * ((d_f * inst.x[i][j]) - sum_i))
                + (inst.scale_beta * inst.beta[j] * sig_d);
            let expr_2 = two * expr;
            let y_ij = inst.y[i][j];
            y_residuals.push(expr_2 - (sig_d * ((two * y_ij) - F::ONE)));
            y_residuals.push((sig_d * ((two * y_ij) + F::ONE)) - F::ONE - expr_2);
        }
    }
    let y_rp_inst = RangeProofInstance {
        values: y_residuals,
        bits: 32,
    };
    verify_range(&proof.y_range_proof, &y_rp_inst, transcript, params)
        .map_err(|e| format!("Y Range Proof Error: {}", e))?;

    Ok(())
}

// --- Helpers for MLE ---
fn generate_challenge_vector(transcript: &mut Transcript, len: usize, label: &[u8]) -> Vec<F> {
    (0..len)
        .map(|_| transcript.challenge_field::<F>(label))
        .collect()
}
fn eval_1d(vec: &[F], r: &[F]) -> F {
    DenseMLPoly::from_vec_padded(vec.to_vec()).evaluate(r)
}
fn eval_cols(matrix: &[Vec<F>], r_row: &[F], rows: usize, cols: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(cols);
    for c in 0..cols {
        let mut col_vec = Vec::with_capacity(rows);
        for r in 0..rows {
            col_vec.push(matrix[r][c]);
        }
        res.push(eval_1d(&col_vec, r_row));
    }
    res
}
fn eval_2d(matrix: &[Vec<F>], r_row: &[F], r_col: &[F], rows: usize, cols: usize) -> F {
    let col_evals = eval_cols(matrix, r_row, rows, cols);
    eval_1d(&col_evals, r_col)
}

// ---------------------------------------------------------------------------
// 網羅的テストスイート
// ---------------------------------------------------------------------------

#[cfg(test)]
mod layernorm_tests {
    use super::*;
    use crate::pcs::setup_hyrax_params;
    use ark_ff::One;

    fn setup_layernorm_instance() -> LayerNormInstance {
        let t = 2;
        let d = 2;
        let d_f = F::from(d as u64);
        let x = vec![
            vec![F::from(10), F::from(20)],
            vec![F::from(30), F::from(40)],
        ];
        let gamma = vec![F::from(2), F::from(2)];
        let beta = vec![F::from(5), F::from(5)];

        let mut sum_x = vec![F::ZERO; t];
        let mut var_x = vec![F::ZERO; t];
        let mut sigma = vec![F::ZERO; t];
        let mut y = vec![vec![F::ZERO; d]; t];

        for i in 0..t {
            let mut sum = F::ZERO;
            for j in 0..d {
                sum += x[i][j];
            }
            sum_x[i] = sum;

            let mut var = F::ZERO;
            for j in 0..d {
                let diff = (d_f * x[i][j]) - sum;
                var += diff * diff;
            }
            var_x[i] = var;

            // ダミーの平方根シミュレーション
            let var_int = var.into_bigint().as_ref()[0];
            let sig_val = (f64::sqrt((var_int as f64) / 4.0)).round() as u64;
            sigma[i] = F::from(if sig_val == 0 { 1 } else { sig_val });

            for j in 0..d {
                let diff_int = (d_f * x[i][j] - sum).into_bigint().as_ref()[0] as i64;
                let diff_val = if diff_int > 100000 {
                    diff_int - 21888242871839275222246405745257275088548364400416034343698204186575808495617
                } else {
                    diff_int
                };

                let expr = 2 * diff_val + 5 * 2 * (sig_val as i64);
                let denom = 2 * (sig_val as i64);
                let y_val = (expr as f64 / denom as f64).round() as i64;
                y[i][j] = if y_val < 0 {
                    F::from(0) - F::from((-y_val) as u64)
                } else {
                    F::from(y_val as u64)
                };
            }
        }

        LayerNormInstance {
            seq_len: t,
            d_head: d,
            x,
            gamma,
            beta,
            y,
            sum_x,
            var_x,
            sigma,
            scale_gamma: F::ONE,
            scale_beta: F::ONE,
        }
    }

    #[test]
    fn test_layernorm_mle_success() {
        let inst = setup_layernorm_instance();
        let params = setup_hyrax_params(16);

        let mut pt = Transcript::new(b"layernorm_test");
        let proof = prove_layernorm(&inst, &mut pt, &params).unwrap();

        let mut vt = Transcript::new(b"layernorm_test");
        let result = verify_layernorm(&proof, &inst, &mut vt, &params);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }

    #[test]
    fn test_layernorm_tampered_x_matrix() {
        let mut inst = setup_layernorm_instance();
        let params = setup_hyrax_params(16);

        // Proverが入力Xの1要素だけをごまかした場合
        inst.x[0][0] += F::ONE;

        let mut pt = Transcript::new(b"layernorm_test");
        let proof_res = prove_layernorm(&inst, &mut pt, &params);

        // xを変更すると sum_x や var_x との整合性がMLE全体で崩れるため、
        // 最初のSumcheckか、RangeProofに渡る前の制約で確定でエラーになる
        if let Ok(proof) = proof_res {
            let mut vt = Transcript::new(b"layernorm_test");
            let result = verify_layernorm(&proof, &inst, &mut vt, &params);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.contains("sumcheck failed") || err.contains("mismatch"));
        }
    }

    #[test]
    fn test_layernorm_tampered_variance_claim() {
        let mut inst = setup_layernorm_instance();
        let params = setup_hyrax_params(16);

        // 分散ベクトル var_x を1要素だけ改竄
        inst.var_x[1] += F::ONE;

        let mut pt = Transcript::new(b"layernorm_test");
        // Sumcheckは嘘の var_x に基づいて構築される
        let proof = prove_layernorm(&inst, &mut pt, &params).unwrap();

        let mut vt = Transcript::new(b"layernorm_test");
        let result = verify_layernorm(&proof, &inst, &mut vt, &params);

        // Verifier側でMLE空間のランダムな点 r_t で評価した期待値と合わなくなる
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Variance sumcheck failed"));
    }
}
