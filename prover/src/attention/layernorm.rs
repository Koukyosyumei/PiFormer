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
    hyrax_commit, hyrax_open, hyrax_verify, params_from_n, poly_hyrax, HyraxCommitment,
    HyraxParams, HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
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
    pub var_x: Vec<F>,
    pub sigma: Vec<F>,
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

pub struct LayerNormInternalCommitments {
    pub sum_x_com: HyraxCommitment,
    pub var_x_com: HyraxCommitment,
    pub sigma_com: HyraxCommitment,
}

pub struct LayerNormOpenings {
    // Audit point r_t for sumchecks
    pub sum_x_at_rt: F,
    pub sum_x_rt_proof: HyraxProof,
    pub var_x_at_rt: F,
    pub var_x_rt_proof: HyraxProof,
    pub x_at_rt_rmean: F,
    pub x_rt_rmean_proof: HyraxProof,
    pub x_at_rt_rvar: F,
    pub x_rt_rvar_proof: HyraxProof,

    // Constraint Fusion points for Sigma: r_sig_t
    pub var_x_at_rsig: F,
    pub var_x_rsig_proof: HyraxProof,
    pub sigma_at_rsig: F,
    pub sigma_rsig_proof: HyraxProof,

    // Constraint Fusion points for Y: (r_y_t, r_y_d)
    pub x_at_ry: F,
    pub x_ry_proof: HyraxProof,
    pub y_at_ry: F,
    pub y_ry_proof: HyraxProof,
    pub sum_x_at_ryt: F,
    pub sum_x_ryt_proof: HyraxProof,
    pub sigma_at_ryt: F,
    pub sigma_ryt_proof: HyraxProof,
}

pub struct LayerNormProof {
    pub internal_coms: LayerNormInternalCommitments,
    pub mean_sumcheck: SumcheckProof,
    pub variance_sumcheck: SumcheckProof,
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

    let x_mle = mat_to_mle(&witness.x, t, d);
    let y_mle = mat_to_mle(&witness.y, t, d);
    let sum_x_mle = vec_to_mle(&witness.sum_x, t);
    let var_x_mle = vec_to_mle(&witness.var_x, t);
    let sigma_mle = vec_to_mle(&witness.sigma, t);

    let (nu_td, sigma_td, params_td) = poly_hyrax(&x_mle);
    let (nu_t, sigma_t, params_t) = poly_hyrax(&sum_x_mle);

    // 1. Absorb IO commitments
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);

    // 2. Commit to internal variables
    let sum_x_com = hyrax_commit(&sum_x_mle.evaluations, nu_t, &params_t);
    let var_x_com = hyrax_commit(&var_x_mle.evaluations, nu_t, &params_t);
    let sigma_com = hyrax_commit(&sigma_mle.evaluations, nu_t, &params_t);

    absorb_com(transcript, b"sum_x_com", &sum_x_com);
    absorb_com(transcript, b"var_x_com", &var_x_com);
    absorb_com(transcript, b"sigma_com", &sigma_com);

    // 3. Row audit challenge
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");
    let claim_mean = sum_x_mle.evaluate(&r_t);
    let claim_var = var_x_mle.evaluate(&r_t);
    transcript.append_field(b"claimed_mean", &claim_mean);
    transcript.append_field(b"claimed_var", &claim_var);

    // 4. Mean sumcheck
    let x_collapsed = eval_rows(&x_mle, t_bits, &r_t);
    let f_mean = DenseMLPoly::from_vec_padded(x_collapsed.clone());
    let g_mean = DenseMLPoly::from_vec_padded(vec![F::ONE; d]);
    let (mean_sumcheck, r_d_mean) = prove_sumcheck(&f_mean, &g_mean, claim_mean, transcript);

    // 5. Variance sumcheck
    let h: Vec<F> = x_collapsed
        .iter()
        .map(|&xj| d_f * xj - claim_mean)
        .collect();
    let f_var = DenseMLPoly::from_vec_padded(h);
    let (variance_sumcheck, r_d_var) = prove_sumcheck(&f_var, &f_var, claim_var, transcript);

    // 6. Range Proofs & Constraint Fusion Challenges
    // Instead of building arrays for Verifier, Prover does it locally.
    //let r_sig_t = challenge_vec(transcript, t_bits, b"rsig_t");
    //let r_sig_b = transcript.challenge_field::<F>(b"rsig_b"); // 1 bit for lo/hi toggle

    let mut sigma_res = Vec::with_capacity(2 * t);
    for i in 0..t {
        let dsi = d_f * witness.sigma[i];
        sigma_res.push(witness.var_x[i] - dsi * dsi); // lo
        sigma_res.push((dsi + d_f) * (dsi + d_f) - F::ONE - witness.var_x[i]); // hi
    }
    // 【重要】prove_rangeから返された r_sig を、以降の計算で使用する！
    let (sigma_range_proof, r_sig) =
        prove_range(&RangeProofWitness { values: sigma_res }, 16, transcript)?;
    let r_sig_t = r_sig[0..t_bits].to_vec();
    let r_sig_b = r_sig[t_bits];

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
    let (y_range_proof, r_y) = prove_range(&RangeProofWitness { values: y_res }, 16, transcript)?;
    let r_y_t = r_y[0..t_bits].to_vec();
    let r_y_d = r_y[t_bits..t_bits + d_bits].to_vec();
    let r_y_b = r_y[t_bits + d_bits];

    // 7. Openings
    let sum_x_at_rt = sum_x_mle.evaluate(&r_t);
    let sum_x_rt_proof = hyrax_open(&sum_x_mle.evaluations, &r_t, nu_t, sigma_t);
    let var_x_at_rt = var_x_mle.evaluate(&r_t);
    let var_x_rt_proof = hyrax_open(&var_x_mle.evaluations, &r_t, nu_t, sigma_t);

    let x_at_rt_rmean = x_mle.evaluate(&combine(&r_t, &r_d_mean));
    let x_rt_rmean_proof = hyrax_open(
        &x_mle.evaluations,
        &combine(&r_t, &r_d_mean),
        nu_td,
        sigma_td,
    );
    let x_at_rt_rvar = x_mle.evaluate(&combine(&r_t, &r_d_var));
    let x_rt_rvar_proof = hyrax_open(
        &x_mle.evaluations,
        &combine(&r_t, &r_d_var),
        nu_td,
        sigma_td,
    );

    let var_x_at_rsig = var_x_mle.evaluate(&r_sig_t);
    let var_x_rsig_proof = hyrax_open(&var_x_mle.evaluations, &r_sig_t, nu_t, sigma_t);
    let sigma_at_rsig = sigma_mle.evaluate(&r_sig_t);
    let sigma_rsig_proof = hyrax_open(&sigma_mle.evaluations, &r_sig_t, nu_t, sigma_t);

    let x_at_ry = x_mle.evaluate(&combine(&r_y_t, &r_y_d));
    let x_ry_proof = hyrax_open(
        &x_mle.evaluations,
        &combine(&r_y_t, &r_y_d),
        nu_td,
        sigma_td,
    );
    let y_at_ry = y_mle.evaluate(&combine(&r_y_t, &r_y_d));
    let y_ry_proof = hyrax_open(
        &y_mle.evaluations,
        &combine(&r_y_t, &r_y_d),
        nu_td,
        sigma_td,
    );
    let sum_x_at_ryt = sum_x_mle.evaluate(&r_y_t);
    let sum_x_ryt_proof = hyrax_open(&sum_x_mle.evaluations, &r_y_t, nu_t, sigma_t);
    let sigma_at_ryt = sigma_mle.evaluate(&r_y_t);
    let sigma_ryt_proof = hyrax_open(&sigma_mle.evaluations, &r_y_t, nu_t, sigma_t);

    Ok(LayerNormProof {
        internal_coms: LayerNormInternalCommitments {
            sum_x_com,
            var_x_com,
            sigma_com,
        },
        mean_sumcheck,
        variance_sumcheck,
        sigma_range_proof,
        y_range_proof,
        openings: LayerNormOpenings {
            sum_x_at_rt,
            sum_x_rt_proof,
            var_x_at_rt,
            var_x_rt_proof,
            x_at_rt_rmean,
            x_rt_rmean_proof,
            x_at_rt_rvar,
            x_rt_rvar_proof,
            var_x_at_rsig,
            var_x_rsig_proof,
            sigma_at_rsig,
            sigma_rsig_proof,
            x_at_ry,
            x_ry_proof,
            y_at_ry,
            y_ry_proof,
            sum_x_at_ryt,
            sum_x_ryt_proof,
            sigma_at_ryt,
            sigma_ryt_proof,
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
    absorb_com(transcript, b"var_x_com", &proof.internal_coms.var_x_com);
    absorb_com(transcript, b"sigma_com", &proof.internal_coms.sigma_com);

    // 2. Sumchecks
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");
    transcript.append_field(b"claimed_mean", &proof.openings.sum_x_at_rt);
    transcript.append_field(b"claimed_var", &proof.openings.var_x_at_rt);

    let (r_d_mean, final_mean) = verify_sumcheck(
        &proof.mean_sumcheck,
        proof.openings.sum_x_at_rt,
        d_bits,
        transcript,
    )?;
    if final_mean != proof.openings.x_at_rt_rmean {
        return Err("Mean sumcheck mismatch".into());
    }

    let (r_d_var, final_var) = verify_sumcheck(
        &proof.variance_sumcheck,
        proof.openings.var_x_at_rt,
        d_bits,
        transcript,
    )?;
    let h_eval = d_f * proof.openings.x_at_rt_rvar - proof.openings.sum_x_at_rt;
    if final_var != h_eval * h_eval {
        return Err("Variance sumcheck mismatch".into());
    }

    // 3. Sigma Constraint Fusion (O(1))
    // 【重要】Verifierも、verify_range_succinct から返された評価点を受け取る
    let (r_sig, sig_eval) = verify_range(&proof.sigma_range_proof, t_bits + 1, 16, transcript)?;
    let r_sig_t = r_sig[0..t_bits].to_vec();
    let r_sig_b = r_sig[t_bits];

    let dsi = d_f * proof.openings.sigma_at_rsig;
    let lo_sig = proof.openings.var_x_at_rsig - dsi * dsi;
    let hi_sig = (dsi + d_f) * (dsi + d_f) - F::ONE - proof.openings.var_x_at_rsig;
    let expected_sig_res = (F::ONE - r_sig_b) * lo_sig + r_sig_b * hi_sig;

    if sig_eval != expected_sig_res {
        return Err("Sigma constraint fusion mismatch".into());
    }

    // 4. Y Constraint Fusion (O(D) to eval public weights, O(1) for residual)
    let (r_y, y_eval) = verify_range(&proof.y_range_proof, t_bits + d_bits + 1, 16, transcript)?;
    let r_y_t = r_y[0..t_bits].to_vec();
    let r_y_d = r_y[t_bits..t_bits + d_bits].to_vec();
    let r_y_b = r_y[t_bits + d_bits];

    let gamma_r = vec_to_mle(&vk.gamma, d).evaluate(&r_y_d);
    let beta_r = vec_to_mle(&vk.beta, d).evaluate(&r_y_d);

    let sig_d = proof.openings.sigma_at_ryt * d_f;
    let two = F::from(2u64);
    let expr =
        vk.scale_gamma * gamma_r * (d_f * proof.openings.x_at_ry - proof.openings.sum_x_at_ryt)
            + vk.scale_beta * beta_r * sig_d;
    let expr2 = two * expr;

    let lo_y = expr2 - sig_d * (two * proof.openings.y_at_ry - F::ONE);
    let hi_y = sig_d * (two * proof.openings.y_at_ry + F::ONE) - F::ONE - expr2;
    let expected_y_res = (F::ONE - r_y_b) * lo_y + r_y_b * hi_y;

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
    // 2. var_x_at_rt
    hyrax_verify(
        &proof.internal_coms.var_x_com,
        proof.openings.var_x_at_rt,
        &r_t,
        &proof.openings.var_x_rt_proof,
        &params_t,
    )?;

    // 3. x_at_rt_rmean
    hyrax_verify(
        &io_coms.x_com,
        proof.openings.x_at_rt_rmean,
        &combine(&r_t, &r_d_mean),
        &proof.openings.x_rt_rmean_proof,
        &params_td,
    )?;
    // 4. x_at_rt_rvar
    hyrax_verify(
        &io_coms.x_com,
        proof.openings.x_at_rt_rvar,
        &combine(&r_t, &r_d_var),
        &proof.openings.x_rt_rvar_proof,
        &params_td,
    )?;

    // 5. var_x_at_rsig
    hyrax_verify(
        &proof.internal_coms.var_x_com,
        proof.openings.var_x_at_rsig,
        &r_sig_t,
        &proof.openings.var_x_rsig_proof,
        &params_t,
    )?;
    // 6. sigma_at_rsig
    hyrax_verify(
        &proof.internal_coms.sigma_com,
        proof.openings.sigma_at_rsig,
        &r_sig_t,
        &proof.openings.sigma_rsig_proof,
        &params_t,
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
// Helpers
// ---------------------------------------------------------------------------
fn mat_to_mle(mat: &[Vec<F>], rows: usize, cols: usize) -> DenseMLPoly {
    let r_p2 = rows.next_power_of_two().max(1);
    let c_p2 = cols.next_power_of_two().max(1);
    let mut evals = vec![F::ZERO; r_p2 * c_p2];
    for (i, row) in mat.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            evals[i * c_p2 + j] = v;
        }
    }
    DenseMLPoly::new(evals)
}
fn vec_to_mle(v: &[F], len: usize) -> DenseMLPoly {
    let padded = len.next_power_of_two().max(2);
    let mut evals = vec![F::ZERO; padded];
    for (i, &x) in v.iter().enumerate() {
        evals[i] = x;
    }
    DenseMLPoly::new(evals)
}

fn eval_rows(poly: &DenseMLPoly, n_row_vars: usize, r_row: &[F]) -> Vec<F> {
    let mut p = poly.clone();
    for &r in r_row {
        p = p.fix_first_variable(r);
    }
    p.evaluations
}
fn combine(a: &[F], b: &[F]) -> Vec<F> {
    let mut res = a.to_vec();
    res.extend_from_slice(b);
    res
}
fn challenge_vec(transcript: &mut Transcript, len: usize, label: &[u8]) -> Vec<F> {
    (0..len)
        .map(|_| transcript.challenge_field::<F>(label))
        .collect()
}
fn absorb_com(transcript: &mut Transcript, label: &[u8], com: &HyraxCommitment) {
    use ark_serialize::CanonicalSerialize;
    for pt in &com.row_coms {
        let mut buf = Vec::new();
        pt.serialize_compressed(&mut buf).unwrap();
        transcript.append_bytes(label, &buf);
    }
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
            vec![F::from(30u64), F::from(40u64)],
        ];
        let gamma = vec![F::from(2u64); d];
        let beta = vec![F::from(5u64); d];

        let mut sum_x = vec![F::zero(); t];
        let mut var_x = vec![F::zero(); t];
        for i in 0..t {
            let s: F = x[i].iter().copied().sum();
            sum_x[i] = s;
            var_x[i] = x[i]
                .iter()
                .map(|&xij| {
                    let diff = d_f * xij - s;
                    diff * diff
                })
                .sum();
        }
        let sigma = vec![F::from(7u64); t];
        let y = vec![
            vec![F::from(4u64), F::from(6u64)],
            vec![F::from(4u64), F::from(6u64)],
        ];

        let witness = LayerNormWitness {
            x: x.clone(),
            y: y.clone(),
            sum_x,
            var_x,
            sigma,
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
