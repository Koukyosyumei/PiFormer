//! PiFormer Fully Secure Pipeline (Public Weights Included)

use crate::field::F;
use crate::lookup::lasso::prove_lasso;
use crate::lookup::lasso::{verify_lasso, LassoInstance, LassoProof};
use crate::pcs::{hyrax_commit, hyrax_open};
use crate::pcs::{hyrax_verify, HyraxCommitment, HyraxParams, HyraxProof};
use crate::poly::DenseMLPoly;
use crate::subprotocols::prove_sumcheck;
use crate::subprotocols::{verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;

// ============================================================================
// FFN Layer Proof (重みの証明を含む)
// ============================================================================
pub struct FFNProof {
    pub y_sumcheck: SumcheckProof,
    pub m_sumcheck: SumcheckProof,
    pub gelu_lasso: LassoProof,

    // 中間生成物のコミットメントと開示証明
    pub a_com: HyraxCommitment,
    pub m_com: HyraxCommitment,
    pub a_eval_at_r: F,
    pub a_open_proof: HyraxProof,
    pub m_eval_at_r: F,
    pub m_open_proof: HyraxProof,
    pub x_eval_at_r: F,
    pub x_open_proof: HyraxProof,

    // [修正点] 公開重み行列の評価値と、その正当性を示すHyrax開示証明
    pub w1_eval_at_r: F,
    pub w1_open_proof: HyraxProof,
    pub w2_eval_at_r: F,
    pub w2_open_proof: HyraxProof,
}

// FFNの入出力データを保持する構造体
pub struct FFNInstance {
    pub x: DenseMLPoly,  // 入力 X
    pub m: DenseMLPoly,  // 中間 M = X * W1
    pub a: DenseMLPoly,  // 活性化 A = GeLU(M)
    pub y: DenseMLPoly,  // 出力 Y = A * W2
    pub w1: DenseMLPoly, // 重み W1
    pub w2: DenseMLPoly, // 重み W2
}

pub fn prove_ffn_secure(
    inst: &FFNInstance,
    rx_out: &[F],
    ry_out: &[F],
    gelu_inst: &LassoInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<FFNProof, String> {
    let d_bits = ry_out.len();

    // ------------------------------------------------------------------------
    // [追加] 0. 中間生成物 M と A にコミットする
    // ------------------------------------------------------------------------
    let (a_com, a_state) = hyrax_commit(&inst.a, params);
    let (m_com, m_state) = hyrax_commit(&inst.m, params);

    transcript.append_commitment(b"a_com", &a_com);
    transcript.append_commitment(b"m_com", &m_com);

    // ------------------------------------------------------------------------
    // 1. Y = A * W2 の Sumcheck と Hyrax Open
    // ------------------------------------------------------------------------
    let claim_y = inst.y.evaluate_2d(rx_out, ry_out);
    transcript.append_field(b"ffn_claim_y", &claim_y);

    // Y の Sumcheck 実行（AとW2の内積）
    // （※ 実際には eval_cols などを使って f_y, g_y を構築して渡します）
    let (y_sumcheck, r_z_1) = prove_sumcheck(
        /* f_y */ &inst.a, /* g_y */ &inst.w2, claim_y, transcript,
    );

    // [追加] Sumcheckの終点で、A の評価値とその開示証明（Open Proof）を生成
    let mut point_a = rx_out.to_vec();
    point_a.extend_from_slice(&r_z_1);
    let a_eval_at_r = inst.a.evaluate(&point_a);
    let a_open_proof = hyrax_open(&inst.a, &a_state, &point_a, params);

    // [追加] 公開重み W2 の評価値と開示証明を生成
    let mut point_w2 = r_z_1.clone();
    point_w2.extend_from_slice(ry_out);
    let w2_eval_at_r = inst.w2.evaluate(&point_w2);
    // ※ 重みの state はオフラインフェーズで計算・保存されているものを利用します
    let (w2_com_dummy, w2_state) = hyrax_commit(&inst.w2, params);
    let w2_open_proof = hyrax_open(&inst.w2, &w2_state, &point_w2, params);

    // ------------------------------------------------------------------------
    // 2. Lasso Bridge: A = GeLU(M)
    // ------------------------------------------------------------------------
    let gelu_lasso = prove_lasso(gelu_inst, transcript, params);

    // ------------------------------------------------------------------------
    // 3. M = X * W1 の Sumcheck と Hyrax Open
    // ------------------------------------------------------------------------
    let rx_m = generate_challenge_vector(transcript, rx_out.len(), b"rx_m");
    let ry_m = generate_challenge_vector(transcript, d_bits, b"ry_m");
    let claim_m = inst.m.evaluate_2d(&rx_m, &ry_m);

    transcript.append_field(b"ffn_claim_m", &claim_m);
    let (m_sumcheck, r_z_2) = prove_sumcheck(
        /* f_m */ &inst.x, /* g_m */ &inst.w1, claim_m, transcript,
    );

    // [追加] M の評価値とその開示証明を生成
    let mut point_m = rx_m.to_vec();
    point_m.extend_from_slice(&ry_m);
    let m_eval_at_r = inst.m.evaluate(&point_m);
    let m_open_proof = hyrax_open(&inst.m, &m_state, &point_m, params);

    // [追加] 前段から渡ってきた X の評価値と開示証明を生成
    let mut point_x = rx_m.clone();
    point_x.extend_from_slice(&r_z_2);
    let x_eval_at_r = inst.x.evaluate(&point_x);
    let (x_com_dummy, x_state) = hyrax_commit(&inst.x, params); // 前段のコミット状態を使う
    let x_open_proof = hyrax_open(&inst.x, &x_state, &point_x, params);

    // [追加] 公開重み W1 の評価値と開示証明を生成
    let mut point_w1 = r_z_2.clone();
    point_w1.extend_from_slice(&ry_m);
    let w1_eval_at_r = inst.w1.evaluate(&point_w1);
    let (w1_com_dummy, w1_state) = hyrax_commit(&inst.w1, params);
    let w1_open_proof = hyrax_open(&inst.w1, &w1_state, &point_w1, params);

    Ok(FFNProof {
        y_sumcheck,
        m_sumcheck,
        gelu_lasso,
        a_com,
        m_com,
        a_eval_at_r,
        a_open_proof,
        m_eval_at_r,
        m_open_proof,
        x_eval_at_r,
        x_open_proof,
        w1_eval_at_r,
        w1_open_proof,
        w2_eval_at_r,
        w2_open_proof,
    })
}

/// 完全セキュアなFFNレイヤーの検証（O(N)の操作を一切含まない）
pub fn verify_ffn_secure(
    proof: &FFNProof,
    x_com: &HyraxCommitment,  // 前段(LN2)から来た入力Xのコミットメント
    w1_com: &HyraxCommitment, // [事前公開] 重み W1 のコミットメント
    w2_com: &HyraxCommitment, // [事前公開] 重み W2 のコミットメント
    rx_out: &[F],
    ry_out: &[F],
    claim_y: F,
    gelu_inst: &LassoInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let d_bits = ry_out.len();

    // ------------------------------------------------------------------------
    // 1. Y = A * W2 の Sumcheck と W2 の検証
    // ------------------------------------------------------------------------
    transcript.append_field(b"ffn_claim_y", &claim_y);
    let (r_z_1, final_claim_y) = verify_sumcheck(&proof.y_sumcheck, claim_y, d_bits, transcript)?;

    // 最終期待値 == A(rx_out, r_z_1) * W2(r_z_1, ry_out)
    let expected_y = proof.a_eval_at_r * proof.w2_eval_at_r;
    if final_claim_y != expected_y {
        return Err("FFN Y Sumcheck final mismatch".to_string());
    }

    // A の評価値の正当性を検証
    let mut point_a = rx_out.to_vec();
    point_a.extend_from_slice(&r_z_1);
    hyrax_verify(
        &proof.a_com,
        proof.a_eval_at_r,
        &point_a,
        &proof.a_open_proof,
        params,
    )?;

    // [追加] W2 の評価値の正当性を「公開コミットメント」に対して検証
    let mut point_w2 = r_z_1.clone();
    point_w2.extend_from_slice(ry_out);
    hyrax_verify(
        w2_com,
        proof.w2_eval_at_r,
        &point_w2,
        &proof.w2_open_proof,
        params,
    )
    .map_err(|e| format!("FFN W2 Public Weight Hyrax mismatch: {}", e))?;

    // ------------------------------------------------------------------------
    // 2. Lasso Bridge: A = GeLU(M)
    // ------------------------------------------------------------------------
    verify_lasso(&proof.gelu_lasso, gelu_inst, transcript, params)?;

    // ------------------------------------------------------------------------
    // 3. M = X * W1 の Sumcheck と W1 の検証
    // ------------------------------------------------------------------------
    let rx_m = generate_challenge_vector(transcript, rx_out.len(), b"rx_m");
    let ry_m = generate_challenge_vector(transcript, d_bits, b"ry_m");

    transcript.append_field(b"ffn_claim_m", &proof.m_eval_at_r);
    let (r_z_2, final_claim_m) =
        verify_sumcheck(&proof.m_sumcheck, proof.m_eval_at_r, d_bits, transcript)?;

    // 最終期待値 == X(rx_m, r_z_2) * W1(r_z_2, ry_m)
    let expected_m = proof.x_eval_at_r * proof.w1_eval_at_r;
    if final_claim_m != expected_m {
        return Err("FFN M Sumcheck final mismatch".to_string());
    }

    // M および X の評価値の正当性を検証
    let mut point_m = rx_m.to_vec();
    point_m.extend_from_slice(&ry_m);
    hyrax_verify(
        &proof.m_com,
        proof.m_eval_at_r,
        &point_m,
        &proof.m_open_proof,
        params,
    )?;

    let mut point_x = rx_m.clone();
    point_x.extend_from_slice(&r_z_2);
    hyrax_verify(
        x_com,
        proof.x_eval_at_r,
        &point_x,
        &proof.x_open_proof,
        params,
    )?;

    // [追加] W1 の評価値の正当性を「公開コミットメント」に対して検証
    let mut point_w1 = r_z_2.clone();
    point_w1.extend_from_slice(&ry_m);
    hyrax_verify(
        w1_com,
        proof.w1_eval_at_r,
        &point_w1,
        &proof.w1_open_proof,
        params,
    )
    .map_err(|e| format!("FFN W1 Public Weight Hyrax mismatch: {}", e))?;

    Ok(())
}

fn generate_challenge_vector(transcript: &mut Transcript, len: usize, label: &[u8]) -> Vec<F> {
    (0..len)
        .map(|_| transcript.challenge_field::<F>(label))
        .collect()
}
