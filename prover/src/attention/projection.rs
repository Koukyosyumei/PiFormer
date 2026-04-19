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
    hyrax_commit, hyrax_open, params_from_vars, HyraxBatchAccumulator, HyraxCommitment, HyraxProof,
};
use crate::poly::utils::TernaryValue;
use crate::poly::utils::{combine, eval_cols_ternary, eval_rows, mat_to_mle};
use crate::poly::utils::{convert_tm_to_fm, vec_to_mle};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, EvalClaim, SumcheckProof};
use crate::transcript::{challenge_vec, Transcript};

// ---------------------------------------------------------------------------
// Pipeline Interfaces & Keys
// ---------------------------------------------------------------------------

/// IO Commitments for the projection layer.
/// x_com is optional: when None (GKR backward mode), the input is not committed
/// independently — the binding comes from the sumcheck argument itself.
pub struct ProjectionIOCommitments {
    pub x_com: Option<HyraxCommitment>,
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
    /// Prover's claimed value of y at (r_t, r_out). Verified via combine proof.
    pub y_eval: F,
    /// Prover's claimed value of x at (r_t, r_k). Returned as EvalClaim for combine.
    pub x_eval: F,
    /// Static weight opening — verified directly.
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

/// Returns (proof, y_claim, x_claim).
///
/// `y_claim` = EvalClaim on the output y at (r_t, r_out) — deferred to combine proof.
/// `x_claim` = EvalClaim on the input x at (r_t, r_k) — propagated backward for combine.
pub fn prove_projection(
    pk: &ProjectionProvingKey,
    witness: &ProjectionWitness,
    io_coms: &ProjectionIOCommitments,
    transcript: &mut Transcript,
) -> Result<(ProjectionProof, EvalClaim, EvalClaim), String> {
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

    let (nu_w, sigma_w, _) = params_from_vars(in_bits + out_bits);
    let (nu_b, sigma_b, _) = params_from_vars(out_bits);

    // 1. Absorb static weight commitments. x_com is only absorbed if present
    //    (GKR backward: when x_com is None, x is bound by the sumcheck itself).
    absorb_com(transcript, b"w_com", &pk.vk.w_com);
    if let Some(ref xc) = io_coms.x_com {
        absorb_com(transcript, b"x_com", xc);
    }
    transcript.append_field(b"alpha", &pk.vk.alpha);
    absorb_com(transcript, b"bias_com", &pk.vk.bias_com);

    // 2. Challenges
    let r_t = challenge_vec(transcript, t_bits, b"proj_rt");
    let r_out = challenge_vec(transcript, out_bits, b"proj_rout");

    // 3. Sumcheck: Z = alpha * Σ X * W
    let f_x_evals = eval_rows(&x_mle, t_bits, &r_t);
    let f_x_scaled =
        DenseMLPoly::from_vec_padded(f_x_evals.iter().map(|val| *val * pk.vk.alpha).collect());

    let g_w_evals = eval_cols_ternary(&pk.w, &r_out, d_in, d_out);
    let g_w = DenseMLPoly::from_vec_padded(g_w_evals);

    let y_eval = y_mle.evaluate(&combine(&r_t, &r_out));
    let bias_eval = bias_mle.evaluate(&r_out);
    let target_z = y_eval - bias_eval;

    let (sumcheck, r_k) = prove_sumcheck(&f_x_scaled, &g_w, target_z, transcript);
    transcript.append_field(b"claimed_y", &y_eval);

    let x_eval = x_mle.evaluate(&combine(&r_t, &r_k));

    let proof = ProjectionProof {
        sumcheck,
        openings: ProjectionOpenings {
            y_eval,
            x_eval,
            // W is (d_in × d_out): evaluate([r_k, r_out]) = combine(&r_k, &r_out)
            w_eval: w_mle.evaluate(&combine(&r_k, &r_out)),
            w_open: hyrax_open(&w_mle.evaluations, &combine(&r_k, &r_out), nu_w, sigma_w),
            bias_at_rj: bias_eval,
            bias_opening_proof: hyrax_open(&bias_mle.evaluations, &r_out, nu_b, sigma_b),
        },
    };

    let y_claim = EvalClaim { point: combine(&r_t, &r_out), value: y_eval };
    let x_claim = EvalClaim { point: combine(&r_t, &r_k), value: x_eval };

    Ok((proof, y_claim, x_claim))
}

// ---------------------------------------------------------------------------
// Verifier (Online Phase)
// ---------------------------------------------------------------------------

/// GKR-style succinct verifier.
///
/// Does NOT open y_com or x_com — deferred to block-level combine proof.
/// Returns (y_claim, x_claim) so the caller can feed them into verify_combine.
pub fn verify_projection(
    proof: &ProjectionProof,
    vk: &ProjectionVerifyingKey,
    io_coms: &ProjectionIOCommitments,
    transcript: &mut Transcript,
    acc_w: &mut HyraxBatchAccumulator,
    acc_b: &mut HyraxBatchAccumulator,
) -> Result<(EvalClaim, EvalClaim), String> {
    let t_bits = vk.seq_len.next_power_of_two().trailing_zeros() as usize;
    let in_bits = vk.d_in.next_power_of_two().trailing_zeros() as usize;
    let out_bits = vk.d_out.next_power_of_two().trailing_zeros() as usize;

    // 1. Absorb (mirrors prover — y_com removed; x_com optional in GKR backward mode)
    absorb_com(transcript, b"w_com", &vk.w_com);
    if let Some(ref xc) = io_coms.x_com {
        absorb_com(transcript, b"x_com", xc);
    }
    transcript.append_field(b"alpha", &vk.alpha);
    absorb_com(transcript, b"bias_com", &vk.bias_com);

    let r_t = challenge_vec(transcript, t_bits, b"proj_rt");
    let r_out = challenge_vec(transcript, out_bits, b"proj_rout");

    // 2. Sumcheck — target is y_eval - bias (y_eval trusted; verified by combine proof)
    let target_z = proof.openings.y_eval - proof.openings.bias_at_rj;
    let (r_k, final_sumcheck_val) = verify_sumcheck(&proof.sumcheck, target_z, in_bits, transcript)
        .map_err(|e| format!("Projection Sumcheck: {e}"))?;
    transcript.append_field(b"claimed_y", &proof.openings.y_eval);

    // 3. Algebraic relation: sumcheck final = alpha * X(r_t,r_k) * W(r_k,r_out)
    if final_sumcheck_val != vk.alpha * proof.openings.x_eval * proof.openings.w_eval {
        return Err("Algebraic relation: alpha * X * W != sumcheck_final".into());
    }

    // 4. Open static weight commitments via deferred accumulators
    acc_w.add_verify(&vk.w_com, proof.openings.w_eval, &combine(&r_k, &r_out), &proof.openings.w_open)?;
    acc_b.add_verify(&vk.bias_com, proof.openings.bias_at_rj, &r_out, &proof.openings.bias_opening_proof)?;

    // Return both claims for block-level combine verification
    let y_claim = EvalClaim { point: combine(&r_t, &r_out), value: proof.openings.y_eval };
    let x_claim = EvalClaim { point: combine(&r_t, &r_k), value: proof.openings.x_eval };
    Ok((y_claim, x_claim))
}
// ---------------------------------------------------------------------------
// Batched QKV Projection (Level 3 Optimization)
//
// Instead of three independent sumchecks for Q, K, V (each finding a separate
// r_k point), a single batched sumcheck is run:
//
//   Σ_k X[r_t, k] · (λ·α_Q·W_Q[k,r_out_q] + μ·α_K·W_K[k,r_out_k] + α_V·W_V[k,r_out_v])
//   = λ·(q_eval - bias_q_eval) + μ·(k_eval - bias_k_eval) + (v_eval - bias_v_eval)
//
// This yields a single r_k, so x_norm1_com can be directly opened at (r_t, r_k)
// instead of requiring a 3-claim CombineProof. Savings: 2 sumchecks + 1 combine.
// ---------------------------------------------------------------------------

pub struct BatchedQKVProjectionWitness {
    pub x: Vec<Vec<F>>,
    pub q: Vec<Vec<F>>,
    pub k: Vec<Vec<F>>,
    pub v: Vec<Vec<F>>,
}

/// IO commitments for QKV projection.
/// x_com is None in GKR mode: x (= LN1.y) is never committed; binding comes
/// from the block-level GKR check using LN1's internal commitments.
pub struct BatchedQKVProjectionIOCommitments {
    pub x_com: Option<HyraxCommitment>,
}

pub struct BatchedQKVProjectionOpenings {
    pub q_eval: F,
    pub k_eval: F,
    pub v_eval: F,
    pub x_eval: F,
    pub w_q_eval: F,
    pub w_k_eval: F,
    pub w_v_eval: F,
    pub bias_q_eval: F,
    pub bias_k_eval: F,
    pub bias_v_eval: F,
    pub w_q_open: HyraxProof,
    pub w_k_open: HyraxProof,
    pub w_v_open: HyraxProof,
    pub bias_q_open: HyraxProof,
    pub bias_k_open: HyraxProof,
    pub bias_v_open: HyraxProof,
}

pub struct BatchedQKVProjectionProof {
    pub sumcheck: SumcheckProof,
    pub openings: BatchedQKVProjectionOpenings,
}

/// Returns (proof, q_claim, k_claim, v_claim, x_norm1_claim).
///
/// All three projections share the same input x_norm1 and the same r_t / r_k
/// after the batched sumcheck, eliminating the 3-claim CombineProof for x_norm1_com.
pub fn prove_qkv_projections(
    pk_q: &ProjectionProvingKey,
    pk_k: &ProjectionProvingKey,
    pk_v: &ProjectionProvingKey,
    witness: &BatchedQKVProjectionWitness,
    io_coms: &BatchedQKVProjectionIOCommitments,
    transcript: &mut Transcript,
) -> Result<(BatchedQKVProjectionProof, EvalClaim, EvalClaim, EvalClaim, EvalClaim), String> {
    let t = pk_q.vk.seq_len;
    let d_in = pk_q.vk.d_in;
    let d_out = pk_q.vk.d_out;

    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let in_bits = d_in.next_power_of_two().trailing_zeros() as usize;
    let out_bits = d_out.next_power_of_two().trailing_zeros() as usize;

    let x_mle = mat_to_mle(&witness.x, t, d_in);
    let q_mle = mat_to_mle(&witness.q, t, d_out);
    let k_mle = mat_to_mle(&witness.k, t, d_out);
    let v_mle = mat_to_mle(&witness.v, t, d_out);
    let w_q_mle = mat_to_mle(&convert_tm_to_fm(&pk_q.w), d_in, d_out);
    let w_k_mle = mat_to_mle(&convert_tm_to_fm(&pk_k.w), d_in, d_out);
    let w_v_mle = mat_to_mle(&convert_tm_to_fm(&pk_v.w), d_in, d_out);
    let bias_q_mle = vec_to_mle(&pk_q.bias, d_out);
    let bias_k_mle = vec_to_mle(&pk_k.bias, d_out);
    let bias_v_mle = vec_to_mle(&pk_v.bias, d_out);

    let (nu_w, sigma_w, _) = params_from_vars(in_bits + out_bits);
    let (nu_b, sigma_b, _) = params_from_vars(out_bits);

    // 1. Absorb all static commitments
    absorb_com(transcript, b"qkv_w_q_com", &pk_q.vk.w_com);
    absorb_com(transcript, b"qkv_w_k_com", &pk_k.vk.w_com);
    absorb_com(transcript, b"qkv_w_v_com", &pk_v.vk.w_com);
    if let Some(ref xc) = io_coms.x_com {
        absorb_com(transcript, b"qkv_x_com", xc);
    }
    transcript.append_field(b"qkv_alpha_q", &pk_q.vk.alpha);
    transcript.append_field(b"qkv_alpha_k", &pk_k.vk.alpha);
    transcript.append_field(b"qkv_alpha_v", &pk_v.vk.alpha);
    absorb_com(transcript, b"qkv_bias_q_com", &pk_q.vk.bias_com);
    absorb_com(transcript, b"qkv_bias_k_com", &pk_k.vk.bias_com);
    absorb_com(transcript, b"qkv_bias_v_com", &pk_v.vk.bias_com);

    // 2. Challenges: shared r_t, independent r_out per projection, batch scalars λ, μ
    let r_t = challenge_vec(transcript, t_bits, b"qkv_rt");
    let r_out_q = challenge_vec(transcript, out_bits, b"qkv_rout_q");
    let r_out_k = challenge_vec(transcript, out_bits, b"qkv_rout_k");
    let r_out_v = challenge_vec(transcript, out_bits, b"qkv_rout_v");
    let lambda: F = transcript.challenge_field(b"qkv_lambda");
    let mu: F = transcript.challenge_field(b"qkv_mu");

    // 3. Evaluate outputs and biases at the challenge points
    let q_eval = q_mle.evaluate(&combine(&r_t, &r_out_q));
    let k_eval = k_mle.evaluate(&combine(&r_t, &r_out_k));
    let v_eval = v_mle.evaluate(&combine(&r_t, &r_out_v));
    let bias_q_eval = bias_q_mle.evaluate(&r_out_q);
    let bias_k_eval = bias_k_mle.evaluate(&r_out_k);
    let bias_v_eval = bias_v_mle.evaluate(&r_out_v);

    // 4. Build batched sumcheck polynomials
    // f(k) = X[r_t, k]
    // g(k) = λ·α_Q·W_Q[k,r_out_q] + μ·α_K·W_K[k,r_out_k] + α_V·W_V[k,r_out_v]
    let f_x_evals = eval_rows(&x_mle, t_bits, &r_t);
    let g_w_q_evals = eval_cols_ternary(&pk_q.w, &r_out_q, d_in, d_out);
    let g_w_k_evals = eval_cols_ternary(&pk_k.w, &r_out_k, d_in, d_out);
    let g_w_v_evals = eval_cols_ternary(&pk_v.w, &r_out_v, d_in, d_out);
    let n_padded = d_in.next_power_of_two();
    let g_w_combined: Vec<F> = (0..n_padded)
        .map(|i| {
            lambda * pk_q.vk.alpha * g_w_q_evals[i]
                + mu * pk_k.vk.alpha * g_w_k_evals[i]
                + pk_v.vk.alpha * g_w_v_evals[i]
        })
        .collect();

    let target =
        lambda * (q_eval - bias_q_eval) + mu * (k_eval - bias_k_eval) + (v_eval - bias_v_eval);

    let f_poly = DenseMLPoly::from_vec_padded(f_x_evals);
    let g_poly = DenseMLPoly::from_vec_padded(g_w_combined);
    let (sumcheck, r_k) = prove_sumcheck(&f_poly, &g_poly, target, transcript);

    transcript.append_field(b"qkv_q_eval", &q_eval);
    transcript.append_field(b"qkv_k_eval", &k_eval);
    transcript.append_field(b"qkv_v_eval", &v_eval);

    // 5. Evaluate prover witnesses at r_k
    let x_eval = x_mle.evaluate(&combine(&r_t, &r_k));
    let w_q_eval = w_q_mle.evaluate(&combine(&r_k, &r_out_q));
    let w_k_eval = w_k_mle.evaluate(&combine(&r_k, &r_out_k));
    let w_v_eval = w_v_mle.evaluate(&combine(&r_k, &r_out_v));

    let proof = BatchedQKVProjectionProof {
        sumcheck,
        openings: BatchedQKVProjectionOpenings {
            q_eval,
            k_eval,
            v_eval,
            x_eval,
            w_q_eval,
            w_k_eval,
            w_v_eval,
            bias_q_eval,
            bias_k_eval,
            bias_v_eval,
            w_q_open: hyrax_open(&w_q_mle.evaluations, &combine(&r_k, &r_out_q), nu_w, sigma_w),
            w_k_open: hyrax_open(&w_k_mle.evaluations, &combine(&r_k, &r_out_k), nu_w, sigma_w),
            w_v_open: hyrax_open(&w_v_mle.evaluations, &combine(&r_k, &r_out_v), nu_w, sigma_w),
            bias_q_open: hyrax_open(&bias_q_mle.evaluations, &r_out_q, nu_b, sigma_b),
            bias_k_open: hyrax_open(&bias_k_mle.evaluations, &r_out_k, nu_b, sigma_b),
            bias_v_open: hyrax_open(&bias_v_mle.evaluations, &r_out_v, nu_b, sigma_b),
        },
    };

    let q_claim = EvalClaim { point: combine(&r_t, &r_out_q), value: q_eval };
    let k_claim = EvalClaim { point: combine(&r_t, &r_out_k), value: k_eval };
    let v_claim = EvalClaim { point: combine(&r_t, &r_out_v), value: v_eval };
    let x_norm1_claim = EvalClaim { point: combine(&r_t, &r_k), value: x_eval };

    Ok((proof, q_claim, k_claim, v_claim, x_norm1_claim))
}

/// GKR-style succinct verifier for batched QKV projections.
///
/// Returns (q_claim, k_claim, v_claim, x_norm1_claim).
/// x_norm1_claim is at a single point (r_t, r_k) — the caller opens x_norm1_com directly.
pub fn verify_qkv_projections(
    proof: &BatchedQKVProjectionProof,
    vk_q: &ProjectionVerifyingKey,
    vk_k: &ProjectionVerifyingKey,
    vk_v: &ProjectionVerifyingKey,
    io_coms: &BatchedQKVProjectionIOCommitments,
    transcript: &mut Transcript,
    acc_w: &mut HyraxBatchAccumulator,
    acc_b: &mut HyraxBatchAccumulator,
) -> Result<(EvalClaim, EvalClaim, EvalClaim, EvalClaim), String> {
    let t_bits = vk_q.seq_len.next_power_of_two().trailing_zeros() as usize;
    let in_bits = vk_q.d_in.next_power_of_two().trailing_zeros() as usize;
    let out_bits = vk_q.d_out.next_power_of_two().trailing_zeros() as usize;

    // 1. Absorb (mirrors prover)
    absorb_com(transcript, b"qkv_w_q_com", &vk_q.w_com);
    absorb_com(transcript, b"qkv_w_k_com", &vk_k.w_com);
    absorb_com(transcript, b"qkv_w_v_com", &vk_v.w_com);
    if let Some(ref xc) = io_coms.x_com {
        absorb_com(transcript, b"qkv_x_com", xc);
    }
    transcript.append_field(b"qkv_alpha_q", &vk_q.alpha);
    transcript.append_field(b"qkv_alpha_k", &vk_k.alpha);
    transcript.append_field(b"qkv_alpha_v", &vk_v.alpha);
    absorb_com(transcript, b"qkv_bias_q_com", &vk_q.bias_com);
    absorb_com(transcript, b"qkv_bias_k_com", &vk_k.bias_com);
    absorb_com(transcript, b"qkv_bias_v_com", &vk_v.bias_com);

    let r_t = challenge_vec(transcript, t_bits, b"qkv_rt");
    let r_out_q = challenge_vec(transcript, out_bits, b"qkv_rout_q");
    let r_out_k = challenge_vec(transcript, out_bits, b"qkv_rout_k");
    let r_out_v = challenge_vec(transcript, out_bits, b"qkv_rout_v");
    let lambda: F = transcript.challenge_field(b"qkv_lambda");
    let mu: F = transcript.challenge_field(b"qkv_mu");

    // 2. Verify batched sumcheck
    let target = lambda * (proof.openings.q_eval - proof.openings.bias_q_eval)
        + mu * (proof.openings.k_eval - proof.openings.bias_k_eval)
        + (proof.openings.v_eval - proof.openings.bias_v_eval);
    let (r_k, final_val) =
        verify_sumcheck(&proof.sumcheck, target, in_bits, transcript)
            .map_err(|e| format!("BatchedQKV Sumcheck: {e}"))?;

    transcript.append_field(b"qkv_q_eval", &proof.openings.q_eval);
    transcript.append_field(b"qkv_k_eval", &proof.openings.k_eval);
    transcript.append_field(b"qkv_v_eval", &proof.openings.v_eval);

    // 3. Algebraic relation: final = x_eval * (λ·α_Q·w_q_eval + μ·α_K·w_k_eval + α_V·w_v_eval)
    let combined_w = lambda * vk_q.alpha * proof.openings.w_q_eval
        + mu * vk_k.alpha * proof.openings.w_k_eval
        + vk_v.alpha * proof.openings.w_v_eval;
    if final_val != proof.openings.x_eval * combined_w {
        return Err("BatchedQKV algebraic relation failed".into());
    }

    // 4. Defer weight and bias openings to block-level accumulators
    acc_w.add_verify(
        &vk_q.w_com, proof.openings.w_q_eval, &combine(&r_k, &r_out_q), &proof.openings.w_q_open,
    )?;
    acc_w.add_verify(
        &vk_k.w_com, proof.openings.w_k_eval, &combine(&r_k, &r_out_k), &proof.openings.w_k_open,
    )?;
    acc_w.add_verify(
        &vk_v.w_com, proof.openings.w_v_eval, &combine(&r_k, &r_out_v), &proof.openings.w_v_open,
    )?;
    acc_b.add_verify(
        &vk_q.bias_com, proof.openings.bias_q_eval, &r_out_q, &proof.openings.bias_q_open,
    )?;
    acc_b.add_verify(
        &vk_k.bias_com, proof.openings.bias_k_eval, &r_out_k, &proof.openings.bias_k_open,
    )?;
    acc_b.add_verify(
        &vk_v.bias_com, proof.openings.bias_v_eval, &r_out_v, &proof.openings.bias_v_open,
    )?;

    let q_claim = EvalClaim { point: combine(&r_t, &r_out_q), value: proof.openings.q_eval };
    let k_claim = EvalClaim { point: combine(&r_t, &r_out_k), value: proof.openings.k_eval };
    let v_claim = EvalClaim { point: combine(&r_t, &r_out_v), value: proof.openings.v_eval };
    let x_norm1_claim = EvalClaim { point: combine(&r_t, &r_k), value: proof.openings.x_eval };

    Ok((q_claim, k_claim, v_claim, x_norm1_claim))
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
                x_com: Some(hyrax_commit(
                    &x_mle.evaluations,
                    params_from_vars(x_mle.num_vars).0,
                    &params_from_vars(x_mle.num_vars).2,
                )),
            },
        )
    }

    #[test]
    fn test_projection_scaled_no_inv_success() {
        let (pk, witness, io) = setup_mock_projection(2, 4, 2);
        let mut transcript = Transcript::new(b"test");
        let (proof, _y_claim, _x_claim) =
            prove_projection(&pk, &witness, &io, &mut transcript).unwrap();
        // Advance transcript to match verifier's 2 finalize calls.
        let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
        let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");

        let mut v_transcript = Transcript::new(b"test");
        let mut acc_w = HyraxBatchAccumulator::new();
        let mut acc_b = HyraxBatchAccumulator::new();
        let result = verify_projection(&proof, &pk.vk, &io, &mut v_transcript, &mut acc_w, &mut acc_b);
        if result.is_ok() {
            let in_bits = pk.vk.d_in.next_power_of_two().trailing_zeros() as usize;
            let out_bits = pk.vk.d_out.next_power_of_two().trailing_zeros() as usize;
            let params_w = params_from_vars(in_bits + out_bits).2;
            let params_b = params_from_vars(out_bits).2;
            acc_w.finalize(&params_w, &mut v_transcript).unwrap();
            acc_b.finalize(&params_b, &mut v_transcript).unwrap();
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_rejects_incorrect_alpha() {
        let (pk, witness, io) = setup_mock_projection(2, 4, 2);
        let mut transcript = Transcript::new(b"test");
        let (proof, _y_claim, _x_claim) =
            prove_projection(&pk, &witness, &io, &mut transcript).unwrap();
        // Advance transcript to match verifier's 2 finalize calls.
        let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
        let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");

        let mut vk_bad = pk.vk;
        vk_bad.alpha = F::one(); // 正解は3
        let mut v_transcript = Transcript::new(b"test");
        let mut acc_w = HyraxBatchAccumulator::new();
        let mut acc_b = HyraxBatchAccumulator::new();
        let result = verify_projection(&proof, &vk_bad, &io, &mut v_transcript, &mut acc_w, &mut acc_b);
        assert!(result.is_err());
    }

    #[test]
    fn test_rejects_incorrect_y_eval() {
        let (pk, witness, io) = setup_mock_projection(2, 4, 2);
        let mut transcript = Transcript::new(b"test");
        let (mut proof, _y_claim, _x_claim) =
            prove_projection(&pk, &witness, &io, &mut transcript).unwrap();
        // Advance transcript to match verifier's 2 finalize calls.
        let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");
        let _ = transcript.challenge_field::<crate::field::F>(b"hyrax_group_mu");

        proof.openings.y_eval += F::one(); // Yの値を改ざん

        let mut v_transcript = Transcript::new(b"test");
        let mut acc_w = HyraxBatchAccumulator::new();
        let mut acc_b = HyraxBatchAccumulator::new();
        let result = verify_projection(&proof, &pk.vk, &io, &mut v_transcript, &mut acc_w, &mut acc_b);
        assert!(result.is_err());
    }
}
