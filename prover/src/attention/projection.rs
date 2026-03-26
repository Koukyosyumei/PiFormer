//! Linear Projection (Dense) Layer Protocol
//!
//! **Production-Grade Architecture:**
//!  1. OFFLINE WEIGHT BINDING: The static weight matrix W is committed ONCE during
//!     a preprocessing phase. The Verifier only holds the `w_com` inside the VK.
//!  2. STRICT IO BOUNDARIES: The Verifier does NOT trust the Prover for the
//!     commitments of X and Y. These MUST be passed via `ProjectionIOCommitments`
//!     from the global pipeline (e.g., from LayerNorm).
//!  3. SUCCINCT GKR CHAINING: The O(N^3) matrix multiplication is reduced to a
//!     single Sumcheck protocol. The Verifier runs in strictly sub-linear time.
//!
//! **Computation proved:**
//!   Y[i][j] = Σ_k X[i][k] · W[k][j]

use crate::field::F;
use crate::pcs::absorb_com;
use crate::pcs::{
    hyrax_commit, hyrax_open, hyrax_verify, params_from_vars, HyraxCommitment, HyraxProof,
};
use crate::poly::utils::TernaryValue;
use crate::poly::utils::{combine, eval_cols_ternary, eval_rows, mat_to_mle};
use crate::poly::utils::{convert_tm_to_fm, vec_to_mle};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::{challenge_vec, Transcript};

// ---------------------------------------------------------------------------
// Pipeline Interfaces & Keys
// ---------------------------------------------------------------------------

/// Trusted IO Commitments provided by the Global Pipeline Verifier.
pub struct ProjectionIOCommitments {
    pub x_com: HyraxCommitment,
    pub y_com: HyraxCommitment,
}

/// Preprocessing Key for the Verifier.
/// Contains the cryptographic commitment to the static model weights.
#[derive(Clone)]
pub struct ProjectionVerifyingKey {
    pub seq_len: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub w_com: HyraxCommitment,
    pub alpha: F,
    pub bias_com: HyraxCommitment,
}

/// Preprocessing Key for the Prover.
/// Contains the raw static weights and the Verifying Key.
#[derive(Clone)]
pub struct ProjectionProvingKey {
    pub vk: ProjectionVerifyingKey,
    pub w: Vec<Vec<TernaryValue>>,
    pub bias: Vec<F>,
}

/// Private witness data (dynamic activations). ONLY the Prover holds this.
pub struct ProjectionWitness {
    pub x: Vec<Vec<F>>,
    pub y: Vec<Vec<F>>,
}

// ---------------------------------------------------------------------------
// Preprocessing (Offline Phase)
// ---------------------------------------------------------------------------

/// Run ONCE when the model is loaded. Commits to the static weights W.
pub fn preprocess_projection(
    seq_len: usize,
    d_in: usize,
    d_out: usize,
    w: Vec<Vec<TernaryValue>>,
    alpha: F,     // 【追加】スケール因子
    bias: Vec<F>, // 【追加】バイアスベクトル
) -> ProjectionProvingKey {
    let w_mle = mat_to_mle(&convert_tm_to_fm(&w), d_in, d_out);
    let (nu_w, _sigma_w, params_w) = params_from_vars(
        d_in.next_power_of_two().trailing_zeros() as usize
            + d_out.next_power_of_two().trailing_zeros() as usize,
    );
    let w_com = hyrax_commit(&w_mle.evaluations, nu_w, &params_w);

    let bias_mle = vec_to_mle(&bias, d_out);
    let (nu_b, _sigma_b, params_b) =
        params_from_vars(d_out.next_power_of_two().trailing_zeros() as usize);
    let bias_com = hyrax_commit(&bias_mle.evaluations, nu_b, &params_b);

    let vk = ProjectionVerifyingKey {
        seq_len,
        d_in,
        d_out,
        w_com,
        alpha,    // 【追加】
        bias_com, // 【追加】
    };

    ProjectionProvingKey {
        vk,
        w,
        bias, // 【追加】証明には生データが必要
    }
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

pub struct ProjectionOpenings {
    pub y_eval: F,
    pub y_open: HyraxProof,
    pub x_eval: F,
    pub x_open: HyraxProof,
    pub w_eval: F,
    pub w_open: HyraxProof,
    pub bias_at_rj: F,
    pub bias_opening_proof: HyraxProof,
}

pub struct ProjectionProof {
    pub sumcheck: SumcheckProof,
    pub openings: ProjectionOpenings,
}

// ---------------------------------------------------------------------------
// Prover (Online Phase)
// ---------------------------------------------------------------------------

pub fn prove_projection(
    pk: &ProjectionProvingKey,
    witness: &ProjectionWitness,
    io_coms: &ProjectionIOCommitments,
    transcript: &mut Transcript,
) -> Result<ProjectionProof, String> {
    let t = pk.vk.seq_len;
    let d_in = pk.vk.d_in;
    let d_out = pk.vk.d_out;

    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let in_bits = d_in.next_power_of_two().trailing_zeros() as usize;
    let out_bits = d_out.next_power_of_two().trailing_zeros() as usize;

    let x_mle = mat_to_mle(&witness.x, t, d_in);
    let y_mle = mat_to_mle(&witness.y, t, d_out);
    let w_mle = mat_to_mle(&convert_tm_to_fm(&pk.w), d_in, d_out);
    let bias_mle = vec_to_mle(&pk.bias, d_out);

    let (nu_x, sigma_x, _) = params_from_vars(t_bits + in_bits);
    let (nu_y, sigma_y, _) = params_from_vars(t_bits + out_bits);
    let (nu_w, sigma_w, _) = params_from_vars(in_bits + out_bits);
    let (nu_b, sigma_b, _) = params_from_vars(out_bits);

    // 1. Absorb
    absorb_com(transcript, b"w_com", &pk.vk.w_com);
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);
    transcript.append_field(b"alpha", &pk.vk.alpha);
    absorb_com(transcript, b"bias_com", &pk.vk.bias_com);

    // 2. Challenges
    let r_t = challenge_vec(transcript, t_bits, b"proj_rt");
    let r_out = challenge_vec(transcript, out_bits, b"proj_rout");

    // 3. Sumcheck: Z = alpha * Σ X * W
    let f_x_evals = eval_rows(&x_mle, t_bits, &r_t);
    // 【変更】多項式 f にあらかじめ alpha を掛けておく (alpha * X)
    let f_x_scaled =
        DenseMLPoly::from_vec_padded(f_x_evals.iter().map(|val| *val * pk.vk.alpha).collect());

    let g_w_evals = eval_cols_ternary(&pk.w, &r_out, d_in, d_out);
    let g_w = DenseMLPoly::from_vec_padded(g_w_evals);

    // Sumcheck の和の期待値は Y(r_t, r_out) - bias(r_out) と一致するはず
    // mat_to_mle layout: evals[i*c_p2+j], rows=MSB → evaluate([r_row, r_col])
    // Y is (t × d_out): evaluate([r_t, r_out]) = combine(&r_t, &r_out)
    let y_eval = y_mle.evaluate(&combine(&r_t, &r_out));
    let bias_eval = bias_mle.evaluate(&r_out);
    let target_z = y_eval - bias_eval;

    let (sumcheck, r_k) = prove_sumcheck(&f_x_scaled, &g_w, target_z, transcript);
    transcript.append_field(b"claimed_y", &y_eval);

    Ok(ProjectionProof {
        sumcheck,
        openings: ProjectionOpenings {
            y_eval,
            // Y is (t × d_out): combine(&r_t, &r_out)
            y_open: hyrax_open(&y_mle.evaluations, &combine(&r_t, &r_out), nu_y, sigma_y),
            // X is (t × d_in): evaluate([r_t, r_k]) = combine(&r_t, &r_k)
            x_eval: x_mle.evaluate(&combine(&r_t, &r_k)),
            x_open: hyrax_open(&x_mle.evaluations, &combine(&r_t, &r_k), nu_x, sigma_x),
            // W is (d_in × d_out): evaluate([r_k, r_out]) = combine(&r_k, &r_out)
            w_eval: w_mle.evaluate(&combine(&r_k, &r_out)),
            w_open: hyrax_open(&w_mle.evaluations, &combine(&r_k, &r_out), nu_w, sigma_w),
            bias_at_rj: bias_eval,
            bias_opening_proof: hyrax_open(&bias_mle.evaluations, &r_out, nu_b, sigma_b),
        },
    })
}

// ---------------------------------------------------------------------------
// Verifier (Online Phase)
// ---------------------------------------------------------------------------

/// **Production-Grade Succinct Verifier**
///
/// Ensures strict O(log N) or O(√N) execution. Binds the computation entirely
/// to the offline static `vk` and the online dynamic `io_coms`.
pub fn verify_projection(
    proof: &ProjectionProof,
    vk: &ProjectionVerifyingKey,
    io_coms: &ProjectionIOCommitments,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let t_bits = vk.seq_len.next_power_of_two().trailing_zeros() as usize;
    let in_bits = vk.d_in.next_power_of_two().trailing_zeros() as usize;
    let out_bits = vk.d_out.next_power_of_two().trailing_zeros() as usize;

    // 1. Absorb (Proverと完全一致)
    absorb_com(transcript, b"w_com", &vk.w_com);
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);
    transcript.append_field(b"alpha", &vk.alpha);
    absorb_com(transcript, b"bias_com", &vk.bias_com);

    let r_t = challenge_vec(transcript, t_bits, b"proj_rt");
    let r_out = challenge_vec(transcript, out_bits, b"proj_rout");

    // 2. Sumcheck 検証
    // ターゲットは Y - bias (alpha.inv を使わない)
    let target_z = proof.openings.y_eval - proof.openings.bias_at_rj;
    let (r_k, final_sumcheck_val) =
        verify_sumcheck(&proof.sumcheck, target_z, in_bits, transcript)?;
    transcript.append_field(b"claimed_y", &proof.openings.y_eval);

    // 3. 代数関係のチェック
    // final_sumcheck_val は alpha * X(r_t, r_k) * W(r_k, r_out) であるべき
    if final_sumcheck_val != vk.alpha * proof.openings.x_eval * proof.openings.w_eval {
        return Err("Algebraic relation: alpha * X * W != sumcheck_final".into());
    }

    // 4. PCS Binding Verification
    // X is (t × d_in): combine(&r_t, &r_k)
    let p_x = params_from_vars(t_bits + in_bits).2;
    hyrax_verify(
        &io_coms.x_com,
        proof.openings.x_eval,
        &combine(&r_t, &r_k),
        &proof.openings.x_open,
        &p_x,
    )?;

    // W is (d_in × d_out): combine(&r_k, &r_out)
    let p_w = params_from_vars(in_bits + out_bits).2;
    hyrax_verify(
        &vk.w_com,
        proof.openings.w_eval,
        &combine(&r_k, &r_out),
        &proof.openings.w_open,
        &p_w,
    )?;

    // Y is (t × d_out): combine(&r_t, &r_out)
    let p_y = params_from_vars(t_bits + out_bits).2;
    hyrax_verify(
        &io_coms.y_com,
        proof.openings.y_eval,
        &combine(&r_t, &r_out),
        &proof.openings.y_open,
        &p_y,
    )?;

    let p_b = params_from_vars(out_bits).2;
    hyrax_verify(
        &vk.bias_com,
        proof.openings.bias_at_rj,
        &r_out,
        &proof.openings.bias_opening_proof,
        &p_b,
    )?;

    Ok(())
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod projection_full_tests {
    use super::*;
    use ark_ff::{One, Zero};

    fn setup_mock_projection(
        t: usize,
        d_in: usize,
        d_out: usize,
    ) -> (
        ProjectionProvingKey,
        ProjectionWitness,
        ProjectionIOCommitments,
    ) {
        let alpha = F::from(3u64);
        let bias = vec![F::from(5u64); d_out];
        let x = vec![vec![F::from(2u64); d_in]; t];
        let w = vec![vec![TernaryValue::ONE; d_out]; d_in];

        // Y = alpha * (X @ W) + bias
        //   = 3 * (2 * d_in) + 5
        let y_val = alpha * F::from(2 * d_in as u64) + F::from(5u64);
        let y = vec![vec![y_val; d_out]; t];

        let x_mle = mat_to_mle(&x, t, d_in);
        let y_mle = mat_to_mle(&y, t, d_out);
        let b_mle = vec_to_mle(&bias, d_out);
        let w_mle = mat_to_mle(&convert_tm_to_fm(&w), d_in, d_out);

        let vk = ProjectionVerifyingKey {
            seq_len: t,
            d_in,
            d_out,
            alpha,
            w_com: hyrax_commit(
                &w_mle.evaluations,
                params_from_vars(w_mle.num_vars).0,
                &params_from_vars(w_mle.num_vars).2,
            ),
            bias_com: hyrax_commit(
                &b_mle.evaluations,
                params_from_vars(b_mle.num_vars).0,
                &params_from_vars(b_mle.num_vars).2,
            ),
        };

        (
            ProjectionProvingKey { vk, w, bias },
            ProjectionWitness { x, y },
            ProjectionIOCommitments {
                x_com: hyrax_commit(
                    &x_mle.evaluations,
                    params_from_vars(x_mle.num_vars).0,
                    &params_from_vars(x_mle.num_vars).2,
                ),
                y_com: hyrax_commit(
                    &y_mle.evaluations,
                    params_from_vars(y_mle.num_vars).0,
                    &params_from_vars(y_mle.num_vars).2,
                ),
            },
        )
    }

    #[test]
    fn test_projection_scaled_no_inv_success() {
        let (pk, witness, io) = setup_mock_projection(2, 4, 2);
        let mut transcript = Transcript::new(b"test");
        let proof = prove_projection(&pk, &witness, &io, &mut transcript).unwrap();

        let mut v_transcript = Transcript::new(b"test");
        assert!(verify_projection(&proof, &pk.vk, &io, &mut v_transcript).is_ok());
    }

    #[test]
    fn test_rejects_incorrect_alpha() {
        let (pk, witness, io) = setup_mock_projection(2, 4, 2);
        let mut transcript = Transcript::new(b"test");
        let proof = prove_projection(&pk, &witness, &io, &mut transcript).unwrap();

        let mut vk_bad = pk.vk;
        vk_bad.alpha = F::one(); // 正解は3
        let mut v_transcript = Transcript::new(b"test");
        assert!(verify_projection(&proof, &vk_bad, &io, &mut v_transcript).is_err());
    }

    #[test]
    fn test_rejects_incorrect_y_eval() {
        let (pk, witness, io) = setup_mock_projection(2, 4, 2);
        let mut transcript = Transcript::new(b"test");
        let mut proof = prove_projection(&pk, &witness, &io, &mut transcript).unwrap();

        proof.openings.y_eval += F::one(); // Yの値を改ざん

        let mut v_transcript = Transcript::new(b"test");
        assert!(verify_projection(&proof, &pk.vk, &io, &mut v_transcript).is_err());
    }
}
