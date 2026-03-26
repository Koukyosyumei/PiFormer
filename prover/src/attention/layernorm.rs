//! LayerNorm Protocol with Constraint Fusion & MLE Evaluation
//!
//! **Production-Grade Security Architecture:**
//!  1. STRICT IO BOUNDARIES: The Verifier does NOT trust the Prover for the
//!     commitments of X and Y. These MUST be passed via `LayerNormIOCommitments`
//!     from the global pipeline.
//!  2. VERIFIER KEY BINDING: Public weights (gamma, beta) are strictly bound
//!     via `LayerNormVerifyingKey`.
//!  3. O(1) CONSTRAINT FUSION: The Verifier NEVER computes O(N) residual arrays.
//!     Instead, it evaluates the residual equations in the polynomial extension
//!     space at a single random point (r_t, r_d, r_b) in O(1) time.

use crate::field::F;
use crate::lookup::range::{prove_range, verify_range, RangeProof, RangeProofWitness};
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_open, hyrax_verify, params_from_n, poly_hyrax, HyraxCommitment,
    HyraxProof,
};
use crate::poly::utils::{combine, eval_rows, mat_to_mle, vec_to_mle};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::{challenge_vec, Transcript};
use ark_ff::Field;

// ---------------------------------------------------------------------------
// Pipeline Interfaces
// ---------------------------------------------------------------------------

/// Trusted IO Commitments provided by the Global Pipeline Verifier.
pub struct LayerNormIOCommitments {
    pub x_com: HyraxCommitment,
    pub y_com: HyraxCommitment,
}

/// Preprocessing key. Contains public weights.
#[derive(Clone)]
pub struct LayerNormVerifyingKey {
    pub seq_len: usize,
    pub d_head: usize,
    pub gamma: Vec<F>,
    pub beta: Vec<F>,
    pub scale_gamma: F,
    pub scale_beta: F,
}

/// Private witness data. ONLY the Prover holds this.
pub struct LayerNormWitness {
    pub x: Vec<Vec<F>>,
    pub y: Vec<Vec<F>>,
    pub sum_x: Vec<F>,
    pub sq_sum_x: Vec<F>,
    pub sum_x_sq: Vec<F>,
    pub sigma: Vec<F>,
    pub sigma_sq_scaled: Vec<F>,
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

pub struct LayerNormInternalCommitments {
    pub sum_x_com: HyraxCommitment,
    pub sq_sum_x_com: HyraxCommitment,
    pub sum_x_sq_com: HyraxCommitment,
    pub sigma_com: HyraxCommitment,
    pub sigma_sq_com: HyraxCommitment,
    pub sigma_y_com: HyraxCommitment,
    pub gamma_x_com: HyraxCommitment,
}

pub struct LayerNormOpenings {
    // Batch Sumcheck point at r_t
    pub sum_x_at_rt: F,
    pub sq_sum_x_at_rt: F,
    pub sum_x_rt_proof: HyraxProof,
    pub sq_sum_x_rt_proof: HyraxProof,
    pub sum_x_sq_at_rsig: F, // 【追加】
    pub sum_x_sq_rsig_proof: HyraxProof,
    pub x_at_rt_rmean: F,
    pub x_rt_rmean_proof: HyraxProof,

    // Constraint Fusion points for Sigma: r_sig_t
    pub sum_x_at_rsig: F,
    pub sum_x_rsig_proof: HyraxProof,
    pub sq_sum_x_at_rsig: F,
    pub sq_sum_x_rsig_proof: HyraxProof,
    pub sigma_at_rsig: F,
    pub sigma_rsig_proof: HyraxProof,
    pub sigma_sq_at_rsig: F,             // 【追加】
    pub sigma_sq_rsig_proof: HyraxProof, // 【追加】

    // Constraint Fusion points for Y: (r_y_t, r_y_d)
    pub x_at_ry: F,
    pub x_ry_proof: HyraxProof,
    pub y_at_ry: F,
    pub y_ry_proof: HyraxProof,
    pub sum_x_at_ryt: F,
    pub sum_x_ryt_proof: HyraxProof,
    pub sigma_at_ryt: F,
    pub sigma_ryt_proof: HyraxProof,
    pub sigma_y_at_ry: F,
    pub sigma_y_ry_proof: HyraxProof,
    pub gamma_x_at_ry: F,
    pub gamma_x_ry_proof: HyraxProof,
}

pub struct LayerNormProof {
    pub internal_coms: LayerNormInternalCommitments,
    pub mean_sumcheck: SumcheckProof,
    pub sigma_range_proof: RangeProof,
    pub y_range_proof: RangeProof,
    pub openings: LayerNormOpenings,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn prove_layernorm(
    witness: &LayerNormWitness,
    io_coms: &LayerNormIOCommitments,
    vk: &LayerNormVerifyingKey,
    transcript: &mut Transcript,
) -> Result<LayerNormProof, String> {
    let t = vk.seq_len;
    let d = vk.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);

    let sum_x_sq: Vec<F> = witness.sum_x.iter().map(|&s| s * s).collect();
    let sigma_sq_scaled: Vec<F> = witness
        .sigma
        .iter()
        .map(|&s| (d_f * s) * (d_f * s))
        .collect();
    let mut sigma_y_evals = vec![F::ZERO; 1 << (t_bits + d_bits)];
    for i in 0..t {
        for j in 0..d {
            sigma_y_evals[i << d_bits | j] = witness.sigma[i] * witness.y[i][j];
        }
    }
    let mut gamma_x_evals = vec![F::ZERO; 1 << (t_bits + d_bits)];
    for i in 0..t {
        for j in 0..d {
            gamma_x_evals[i << d_bits | j] = vk.gamma[j] * witness.x[i][j];
        }
    }

    let x_mle = mat_to_mle(&witness.x, t, d); // x
    let y_mle = mat_to_mle(&witness.y, t, d); // y
    let sum_x_mle = vec_to_mle(&witness.sum_x, t); // \sum_{d} x_{t,d}
    let sq_sum_x_mle = vec_to_mle(&witness.sq_sum_x, t); // \sum_{d} x^2_{t,d}
    let sum_x_sq_mle = vec_to_mle(&sum_x_sq, t); // // (\sum_{d} x_{t,d})^2
    let sigma_mle = vec_to_mle(&witness.sigma, t);
    let sigma_sq_mle = vec_to_mle(&sigma_sq_scaled, t);
    let sigma_y_mle = DenseMLPoly::new(sigma_y_evals);
    let gamma_x_mle = DenseMLPoly::new(gamma_x_evals);

    let (nu_td, sigma_td, params_td) = poly_hyrax(&x_mle);
    let (nu_t, sigma_t, params_t) = poly_hyrax(&sum_x_mle);

    // 1. Absorb IO commitments
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);

    // 2. Commit to internal variables
    let sum_x_com = hyrax_commit(&sum_x_mle.evaluations, nu_t, &params_t);
    let sq_sum_x_com = hyrax_commit(&sq_sum_x_mle.evaluations, nu_t, &params_t);
    let sum_x_sq_com = hyrax_commit(&sum_x_sq_mle.evaluations, nu_t, &params_t);
    let sigma_com = hyrax_commit(&sigma_mle.evaluations, nu_t, &params_t);
    let sigma_sq_com = hyrax_commit(&sigma_sq_mle.evaluations, nu_t, &params_t);
    let sigma_y_com = hyrax_commit(&sigma_y_mle.evaluations, nu_td, &params_td);
    let gamma_x_com = hyrax_commit(&gamma_x_mle.evaluations, nu_td, &params_td);

    absorb_com(transcript, b"sum_x_com", &sum_x_com);
    absorb_com(transcript, b"sq_sum_x_com", &sq_sum_x_com);
    absorb_com(transcript, b"sum_x_sq_com", &sum_x_sq_com);
    absorb_com(transcript, b"sigma_com", &sigma_com);
    absorb_com(transcript, b"sigma_sq_com", &sigma_sq_com);
    absorb_com(transcript, b"sigma_y_com", &sigma_y_com);
    absorb_com(transcript, b"gamma_x_com", &gamma_x_com);

    // 3. Row audit challenge
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");

    let claim_s = sum_x_mle.evaluate(&r_t);
    let claim_q = sq_sum_x_mle.evaluate(&r_t);

    transcript.append_field(b"claimed_s", &claim_s);
    transcript.append_field(b"claimed_q", &claim_q);

    // 4. Mean sumcheck
    let x_collapsed = eval_rows(&x_mle, t_bits, &r_t);
    let f_mean = DenseMLPoly::from_vec_padded(x_collapsed.clone());
    let g_mean = DenseMLPoly::from_vec_padded(vec![F::ONE; d]);
    let (mean_sumcheck, r_d_mean) = prove_sumcheck(&f_mean, &g_mean, claim_s, transcript);

    // 6. Range Proofs & Constraint Fusion Challenges
    // Instead of building arrays for Verifier, Prover does it locally.
    //let r_sig_t = challenge_vec(transcript, t_bits, b"rsig_t");
    //let r_sig_b = transcript.challenge_field::<F>(b"rsig_b"); // 1 bit for lo/hi toggle

    let mut sigma_res = Vec::with_capacity(2 * t);
    for i in 0..t {
        let vi = d_f * (d_f * witness.sq_sum_x[i] - witness.sum_x[i] * witness.sum_x[i]);
        let dsi = d_f * witness.sigma[i];
        sigma_res.push(vi - dsi * dsi);
        sigma_res.push((dsi + d_f) * (dsi + d_f) - F::ONE - vi);
    }
    let (sigma_range_proof, r_sig) =
        prove_range(&RangeProofWitness { values: sigma_res }, 32, transcript)?;
    let r_sig_t = r_sig[0..t_bits].to_vec();

    /*
        let r_y_t = challenge_vec(transcript, t_bits, b"ry_t");
        let r_y_d = challenge_vec(transcript, d_bits, b"ry_d");
        let r_y_b = transcript.challenge_field::<F>(b"ry_b"); // 1 bit for lo/hi toggle
    */

    let mut y_res = Vec::with_capacity(2 * t * d);
    let two = F::from(2u64);
    for i in 0..t {
        let sig_d = witness.sigma[i] * d_f;
        let sum_i = witness.sum_x[i];
        for j in 0..d {
            let expr = vk.scale_gamma * vk.gamma[j] * (d_f * witness.x[i][j] - sum_i)
                + vk.scale_beta * vk.beta[j] * sig_d;
            let expr2 = two * expr;
            let y_ij = witness.y[i][j];
            y_res.push(expr2 - sig_d * (two * y_ij - F::ONE)); // lo
            y_res.push(sig_d * (two * y_ij + F::ONE) - F::ONE - expr2); // hi
        }
    }
    // 【重要】prove_rangeから返された r_y を、以降の計算で使用する！
    let (y_range_proof, r_y) = prove_range(&RangeProofWitness { values: y_res }, 32, transcript)?;
    let r_y_t = r_y[0..t_bits].to_vec();
    let r_y_d = r_y[t_bits..t_bits + d_bits].to_vec();
    let r_y_b = r_y[t_bits + d_bits];
    println!("{:?} {:?} {}", r_y_t, r_y_d, r_y_b);

    // 7. Openings

    Ok(LayerNormProof {
        internal_coms: LayerNormInternalCommitments {
            sum_x_com,
            sq_sum_x_com,
            sigma_com,
            sum_x_sq_com,
            sigma_sq_com,
            sigma_y_com,
            gamma_x_com,
        },
        mean_sumcheck,
        sigma_range_proof,
        y_range_proof,
        openings: LayerNormOpenings {
            sum_x_at_rt: claim_s,
            sq_sum_x_at_rt: claim_q,
            sum_x_rt_proof: hyrax_open(&sum_x_mle.evaluations, &r_t, nu_t, sigma_t),
            sq_sum_x_rt_proof: hyrax_open(&sq_sum_x_mle.evaluations, &r_t, nu_t, sigma_t),
            x_at_rt_rmean: x_mle.evaluate(&combine(&r_t, &r_d_mean)),
            x_rt_rmean_proof: hyrax_open(
                &x_mle.evaluations,
                &combine(&r_t, &r_d_mean),
                nu_td,
                sigma_td,
            ),
            sum_x_at_rsig: sum_x_mle.evaluate(&r_sig_t),
            sum_x_rsig_proof: hyrax_open(&sum_x_mle.evaluations, &r_sig_t, nu_t, sigma_t),
            sq_sum_x_at_rsig: sq_sum_x_mle.evaluate(&r_sig_t),
            sq_sum_x_rsig_proof: hyrax_open(&sq_sum_x_mle.evaluations, &r_sig_t, nu_t, sigma_t),
            sigma_at_rsig: sigma_mle.evaluate(&r_sig_t),
            sigma_rsig_proof: hyrax_open(&sigma_mle.evaluations, &r_sig_t, nu_t, sigma_t),
            sigma_sq_at_rsig: sigma_sq_mle.evaluate(&r_sig_t),
            sigma_sq_rsig_proof: hyrax_open(&sigma_sq_mle.evaluations, &r_sig_t, nu_t, sigma_t),
            x_at_ry: x_mle.evaluate(&combine(&r_y_t, &r_y_d)),
            x_ry_proof: hyrax_open(
                &x_mle.evaluations,
                &combine(&r_y_t, &r_y_d),
                nu_td,
                sigma_td,
            ),
            y_at_ry: y_mle.evaluate(&combine(&r_y_t, &r_y_d)),
            y_ry_proof: hyrax_open(
                &mat_to_mle(&witness.y, t, d).evaluations,
                &combine(&r_y_t, &r_y_d),
                nu_td,
                sigma_td,
            ),
            sum_x_at_ryt: sum_x_mle.evaluate(&r_y_t),
            sum_x_ryt_proof: hyrax_open(&sum_x_mle.evaluations, &r_y_t, nu_t, sigma_t),
            sigma_at_ryt: sigma_mle.evaluate(&r_y_t),
            sigma_ryt_proof: hyrax_open(&sigma_mle.evaluations, &r_y_t, nu_t, sigma_t),
            sum_x_sq_at_rsig: sum_x_sq_mle.evaluate(&r_sig_t),
            sum_x_sq_rsig_proof: hyrax_open(
                &sum_x_sq_mle.evaluations,
                &r_sig_t,
                nu_t,
                params_t.sigma,
            ),
            gamma_x_at_ry: gamma_x_mle.evaluate(&combine(&r_y_t, &r_y_d)),
            gamma_x_ry_proof: hyrax_open(
                &gamma_x_mle.evaluations,
                &combine(&r_y_t, &r_y_d),
                nu_td,
                sigma_td,
            ),
            sigma_y_at_ry: sigma_y_mle.evaluate(&combine(&r_y_t, &r_y_d)),
            sigma_y_ry_proof: hyrax_open(
                &sigma_y_mle.evaluations,
                &combine(&r_y_t, &r_y_d),
                nu_td,
                sigma_td,
            ),
        },
    })
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// **Production-Grade Succinct Verifier**
///
/// Ensures $O(D)$ or $O(\log N)$ computation. NO `hyrax_commit` is executed.
/// Constraints are mathematically fused into a single polynomial evaluation.
pub fn verify_layernorm(
    proof: &LayerNormProof,
    io_coms: &LayerNormIOCommitments,
    vk: &LayerNormVerifyingKey,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let t = vk.seq_len;
    let d = vk.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);

    let n_td = t.next_power_of_two().max(1) * d.next_power_of_two().max(1);
    let n_t = t.next_power_of_two().max(1);
    let (_nu_td, _sigma_td, params_td) = params_from_n(n_td);
    let (_nu_t, _sigma_t, params_t) = params_from_n(n_t);

    // 1. Absorb IO & Internal Commitments
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);
    absorb_com(transcript, b"sum_x_com", &proof.internal_coms.sum_x_com);
    absorb_com(
        transcript,
        b"sq_sum_x_com",
        &proof.internal_coms.sq_sum_x_com,
    );
    absorb_com(
        transcript,
        b"sum_x_sq_com",
        &proof.internal_coms.sum_x_sq_com,
    );
    absorb_com(transcript, b"sigma_com", &proof.internal_coms.sigma_com);
    absorb_com(
        transcript,
        b"sigma_sq_com",
        &proof.internal_coms.sigma_sq_com,
    );
    absorb_com(transcript, b"sigma_y_com", &proof.internal_coms.sigma_y_com);
    absorb_com(transcript, b"gamma_x_com", &proof.internal_coms.gamma_x_com);

    // 2. Sumchecks
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");

    transcript.append_field(b"claimed_s", &proof.openings.sum_x_at_rt);
    transcript.append_field(b"claimed_q", &proof.openings.sq_sum_x_at_rt);

    let (r_d_mean, final_mean) = verify_sumcheck(
        &proof.mean_sumcheck,
        proof.openings.sum_x_at_rt,
        d_bits,
        transcript,
    )
    .map_err(|e| format!("Mean Mean Sumcheck: {e}"))?;
    if final_mean != proof.openings.x_at_rt_rmean {
        return Err("Mean sumcheck mismatch".into());
    }

    // 3. Sigma Constraint Fusion (O(1))
    // 【重要】Verifierも、verify_range_succinct から返された評価点を受け取る
    let (r_sig, sig_eval) = verify_range(&proof.sigma_range_proof, t_bits + 1, 32, transcript)
        .map_err(|e| format!("Sigam Range: {e}"))?;
    let r_sig_b = r_sig[t_bits];

    let v_ev = d_f * (d_f * proof.openings.sq_sum_x_at_rsig - proof.openings.sum_x_sq_at_rsig);
    let z_ev = proof.openings.sigma_sq_at_rsig;
    let s_ev = proof.openings.sigma_at_rsig;
    let lo_sig = v_ev - z_ev;
    let hi_sig = z_ev + F::from(2u64) * d_f * d_f * s_ev + d_f * d_f - F::ONE - v_ev;
    let expected = (F::ONE - r_sig_b) * lo_sig + r_sig_b * hi_sig;
    if sig_eval != expected {
        return Err("Chunk fusion mismatch: Sigma logic".into());
    }

    // 4. Y Constraint Fusion (O(D) to eval public weights, O(1) for residual)
    let (r_y, y_eval) = verify_range(&proof.y_range_proof, t_bits + d_bits + 1, 32, transcript)
        .map_err(|e| format!("Y Range: {e}"))?;
    let r_y_t = r_y[0..t_bits].to_vec();
    let r_y_d = r_y[t_bits..t_bits + d_bits].to_vec();
    let r_y_b = r_y[t_bits + d_bits];
    println!("{:?} {:?} {}", r_y_t, r_y_d, r_y_b);

    let gamma_r = vec_to_mle(&vk.gamma, d).evaluate(&r_y_d);
    let beta_r = vec_to_mle(&vk.beta, d).evaluate(&r_y_d);

    let sig_d = proof.openings.sigma_at_ryt * d_f;
    let sigma_y = proof.openings.sigma_y_at_ry;
    let gamma_x = proof.openings.gamma_x_at_ry;

    // expr = scale_gamma * (D * gamma_x - gamma * S) + scale_beta * beta * sig_d
    // 外積 (gamma * S) と (beta * sigma) は MLE の積と一致するのでそのままでOK
    let two = F::from(2u64);
    let expr = vk.scale_gamma * (d_f * gamma_x - gamma_r * proof.openings.sum_x_at_ryt)
        + vk.scale_beta * beta_r * sig_d;
    let expr2 = two * expr;
    let lo_y = expr2 - (two * d_f * sigma_y - sig_d);
    let hi_y = (two * d_f * sigma_y + sig_d) - F::ONE - expr2;

    let expected_y_res = (F::ONE - r_y_b) * lo_y + r_y_b * hi_y;

    println!("aaaaaaaaa {} {}", y_eval, expected_y_res);
    if y_eval != expected_y_res {
        return Err("Y constraint fusion mismatch".into());
    }

    // 5. Openings (Binding check against IO and Internal commitments)
    // 1. sum_x_at_rt
    hyrax_verify(
        &proof.internal_coms.sum_x_com,
        proof.openings.sum_x_at_rt,
        &r_t,
        &proof.openings.sum_x_rt_proof,
        &params_t,
    )?;
    hyrax_verify(
        &proof.internal_coms.sum_x_com,
        proof.openings.sum_x_at_rt,
        &r_t,
        &proof.openings.sum_x_rt_proof,
        &params_t,
    )?;
    hyrax_verify(
        &proof.internal_coms.sq_sum_x_com,
        proof.openings.sq_sum_x_at_rt,
        &r_t,
        &proof.openings.sq_sum_x_rt_proof,
        &params_t,
    )?;
    hyrax_verify(
        &io_coms.x_com,
        proof.openings.x_at_rt_rmean,
        &combine(&r_t, &r_d_mean),
        &proof.openings.x_rt_rmean_proof,
        &params_td,
    )?;

    // 7. x_at_ry
    hyrax_verify(
        &io_coms.x_com,
        proof.openings.x_at_ry,
        &combine(&r_y_t, &r_y_d),
        &proof.openings.x_ry_proof,
        &params_td,
    )?;
    // 8. y_at_ry
    hyrax_verify(
        &io_coms.y_com,
        proof.openings.y_at_ry,
        &combine(&r_y_t, &r_y_d),
        &proof.openings.y_ry_proof,
        &params_td,
    )?;

    // 9. sum_x_at_ryt
    hyrax_verify(
        &proof.internal_coms.sum_x_com,
        proof.openings.sum_x_at_ryt,
        &r_y_t,
        &proof.openings.sum_x_ryt_proof,
        &params_t,
    )?;
    // 10. sigma_at_ryt
    hyrax_verify(
        &proof.internal_coms.sigma_com,
        proof.openings.sigma_at_ryt,
        &r_y_t,
        &proof.openings.sigma_ryt_proof,
        &params_t,
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod layernorm_tests {
    use super::*;
    use ark_ff::{One, Zero};

    fn setup_test_pipeline() -> (
        LayerNormWitness,
        LayerNormIOCommitments,
        LayerNormVerifyingKey,
    ) {
        let t = 2usize;
        let d = 2usize;
        let d_f = F::from(d as u64);
        let x = vec![
            vec![F::from(10u64), F::from(20u64)],
            vec![F::from(25u64), F::from(40u64)],
        ];
        let gamma = vec![F::from(2u64); d];
        let beta = vec![F::from(5u64); d];

        let mut sum_x = vec![F::zero(); t];
        let mut sq_sum_x = vec![F::zero(); t];
        let mut sum_x_sq = vec![F::zero(); t];
        let mut var_x = vec![F::zero(); t];
        for i in 0..t {
            let s: F = x[i].iter().copied().sum();
            let sq_s: F = x[i].iter().map(|v| v * v).sum();
            sum_x[i] = s;
            sq_sum_x[i] = sq_s;
            sum_x_sq[i] = s * s;
            var_x[i] = x[i]
                .iter()
                .map(|&xij| {
                    let diff = d_f * xij - s;
                    diff * diff
                })
                .sum();
        }
        let sigma = vec![F::from(7u64), F::from(10u64)];
        let y = vec![
            vec![F::from(4u64), F::from(6u64)],
            vec![F::from(4u64), F::from(7u64)],
        ];
        let sigma_sq_scaled = sigma
            .iter()
            .map(|&s| (d_f * s) * (d_f * s))
            .collect::<Vec<F>>();

        let witness = LayerNormWitness {
            x: x.clone(),
            y: y.clone(),
            sum_x,
            sq_sum_x,
            sigma,
            sum_x_sq,
            sigma_sq_scaled,
        };
        let vk = LayerNormVerifyingKey {
            seq_len: t,
            d_head: d,
            gamma,
            beta,
            scale_gamma: F::one(),
            scale_beta: F::one(),
        };

        let x_mle = mat_to_mle(&x, t, d);
        let y_mle = mat_to_mle(&y, t, d);
        let (nu_td, _, params_td) = poly_hyrax(&x_mle);
        let io_coms = LayerNormIOCommitments {
            x_com: hyrax_commit(&x_mle.evaluations, nu_td, &params_td),
            y_com: hyrax_commit(&y_mle.evaluations, nu_td, &params_td),
        };
        (witness, io_coms, vk)
    }

    #[test]
    fn test_layernorm_succinct_e2e() {
        let (witness, io_coms, vk) = setup_test_pipeline();
        let mut pt = Transcript::new(b"layernorm_test");
        let proof = prove_layernorm(&witness, &io_coms, &vk, &mut pt).unwrap();

        let mut vt = Transcript::new(b"layernorm_test");
        let result = verify_layernorm(&proof, &io_coms, &vk, &mut vt);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }

    #[test]
    fn test_rejects_tampered_io_x() {
        let (mut witness, io_coms, vk) = setup_test_pipeline();
        witness.x[0][0] += F::one(); // Tamper locally

        let mut pt = Transcript::new(b"layernorm_test");
        if let Ok(proof) = prove_layernorm(&witness, &io_coms, &vk, &mut pt) {
            let mut vt = Transcript::new(b"layernorm_test");
            let result = verify_layernorm(&proof, &io_coms, &vk, &mut vt);
            assert!(
                result.is_err(),
                "Should reject forged proof against trusted IO"
            );
        }
    }
}
