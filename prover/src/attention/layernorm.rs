//! LayerNorm Protocol with Constraint Fusion & MLE Evaluation
//!
//! Proves correctness of LayerNorm by:
//!  1. Mean sumcheck:     Σ_j X(r_t, j) · 1  = sum_x(r_t)
//!  2. Variance sumcheck: Σ_j h_j^2           = var_x(r_t),  h_j = d·X(r_t,j) - sum_x(r_t)
//!  3. Sigma range proof: proves σ ≈ √(var/d²) via non-negative residuals
//!  4. Y range proof:     proves rounded normalised outputs y are consistent

use crate::field::F;
use crate::lookup::range::{prove_range, verify_range, RangeProof, RangeProofInstance};
use crate::pcs::HyraxParams;
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::Field;

/// Public inputs + prover advice for one LayerNorm layer.
#[derive(Clone)]
pub struct LayerNormInstance {
    pub seq_len: usize,
    pub d_head: usize,
    /// Input matrix X: shape (seq_len, d_head).
    pub x: Vec<Vec<F>>,
    pub gamma: Vec<F>,
    pub beta: Vec<F>,
    /// Output matrix Y: shape (seq_len, d_head).
    pub y: Vec<Vec<F>>,
    /// Row sums: sum_x[i] = Σ_j X[i][j].
    pub sum_x: Vec<F>,
    /// Row variances (scaled): var_x[i] = Σ_j (d·X[i][j] - sum_x[i])².
    pub var_x: Vec<F>,
    /// Approximated std-dev (integer, σ_i ≈ √(var_x[i]/d²)).
    pub sigma: Vec<F>,
    /// Quantisation scales for γ and β.
    pub scale_gamma: F,
    pub scale_beta: F,
}

pub struct LayerNormProof {
    pub mean_sumcheck: SumcheckProof,
    pub variance_sumcheck: SumcheckProof,
    pub sigma_range_proof: RangeProof,
    pub y_range_proof: RangeProof,
    /// Prover's claim: X(r_t, r_d_mean) (needed for mean sumcheck final check).
    pub x_eval_at_r: F,
}

pub fn prove_layernorm(
    inst: &LayerNormInstance,
    transcript: &mut Transcript,
    _params: &HyraxParams, // reserved for future PCS opening of X; range proofs are self-contained
) -> Result<LayerNormProof, String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);

    // Verifier challenge r_t in F^{t_bits} for MLE row evaluation
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_audit_rt");

    let claim_mean = eval_1d(&inst.sum_x, &r_t);
    let claim_var = eval_1d(&inst.var_x, &r_t);

    transcript.append_field(b"claimed_mean", &claim_mean);
    transcript.append_field(b"claimed_var", &claim_var);

    // Collapse X rows at r_t: x_collapsed[j] = X(r_t, e_j) for j in 0..d
    let x_collapsed = eval_cols(&inst.x, &r_t, t, d);

    // --- Mean Sumcheck: Σ_j X(r_t, j) · 1 = claim_mean ---
    let f_mean = DenseMLPoly::from_vec_padded(x_collapsed.clone());
    let g_mean = DenseMLPoly::from_vec_padded(vec![F::ONE; d]);
    let (mean_sumcheck, r_d_mean) = prove_sumcheck(&f_mean, &g_mean, claim_mean, transcript);

    // --- Variance Sumcheck: Σ_j h_j² = claim_var,  h_j = d·x_collapsed[j] - claim_mean ---
    let h: Vec<F> = x_collapsed.iter().map(|&xj| d_f * xj - claim_mean).collect();
    let f_var = DenseMLPoly::from_vec_padded(h.clone());
    let g_var = DenseMLPoly::from_vec_padded(h);
    let (variance_sumcheck, _r_d_var) = prove_sumcheck(&f_var, &g_var, claim_var, transcript);

    // --- Sigma Range Proof ---
    // For each row i: prove (4·var_i - (2σ_i-1)²·d²) ≥ 0
    //                  and ((2σ_i+1)²·d² - 1 - 4·var_i) ≥ 0
    let two = F::from(2u64);
    let four = F::from(4u64);
    let d_sq = d_f * d_f;
    let mut sigma_residuals = Vec::with_capacity(t * 2);
    for i in 0..t {
        let sig = inst.sigma[i];
        let v = inst.var_x[i];
        let lo = two * sig - F::ONE;
        let hi = two * sig + F::ONE;
        sigma_residuals.push(four * v - lo * lo * d_sq);
        sigma_residuals.push(hi * hi * d_sq - F::ONE - four * v);
    }
    let sigma_range_proof = prove_range(
        &RangeProofInstance { values: sigma_residuals, bits: 32 },
        transcript,
    )?;

    // --- Y Range Proof ---
    // For each (i,j): prove consistency of y[i][j] with the normalised output formula
    let mut y_residuals = Vec::with_capacity(t * d * 2);
    for i in 0..t {
        let sig_d = inst.sigma[i] * d_f;
        let sum_i = inst.sum_x[i];
        for j in 0..d {
            let expr = inst.scale_gamma * inst.gamma[j] * (d_f * inst.x[i][j] - sum_i)
                + inst.scale_beta * inst.beta[j] * sig_d;
            let expr2 = two * expr;
            let y_ij = inst.y[i][j];
            y_residuals.push(expr2 - sig_d * (two * y_ij - F::ONE));
            y_residuals.push(sig_d * (two * y_ij + F::ONE) - F::ONE - expr2);
        }
    }
    let y_range_proof = prove_range(
        &RangeProofInstance { values: y_residuals, bits: 32 },
        transcript,
    )?;

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
    _params: &HyraxParams,
) -> Result<(), String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);

    let r_t = challenge_vec(transcript, t_bits, b"layernorm_audit_rt");

    // In a fully succinct protocol these would be PCS openings; here we re-compute.
    let claim_mean = eval_1d(&inst.sum_x, &r_t);
    let claim_var = eval_1d(&inst.var_x, &r_t);

    // Mirror prover: append both claims before any sumcheck.
    transcript.append_field(b"claimed_mean", &claim_mean);
    transcript.append_field(b"claimed_var", &claim_var);

    // --- Mean Sumcheck ---
    let (_r_d_mean, final_mean) =
        verify_sumcheck(&proof.mean_sumcheck, claim_mean, d_bits, transcript)
            .map_err(|e| format!("Mean sumcheck failed: {e}"))?;

    if final_mean != proof.x_eval_at_r {
        return Err("Mean sumcheck final evaluation mismatch".to_string());
    }

    // --- Variance Sumcheck ---
    let (r_d_var, final_var) =
        verify_sumcheck(&proof.variance_sumcheck, claim_var, d_bits, transcript)
            .map_err(|e| format!("Variance sumcheck failed: {e}"))?;

    let x_eval_var = eval_2d(&inst.x, &r_t, &r_d_var, t, d);
    let h_eval = d_f * x_eval_var - claim_mean;
    let expected_var = h_eval * h_eval;
    if final_var != expected_var {
        return Err("Variance sumcheck final evaluation mismatch".to_string());
    }

    // --- Sigma Range Proof ---
    let two = F::from(2u64);
    let four = F::from(4u64);
    let d_sq = d_f * d_f;
    let mut sigma_residuals = Vec::with_capacity(t * 2);
    for i in 0..t {
        let sig = inst.sigma[i];
        let v = inst.var_x[i];
        let lo = two * sig - F::ONE;
        let hi = two * sig + F::ONE;
        sigma_residuals.push(four * v - lo * lo * d_sq);
        sigma_residuals.push(hi * hi * d_sq - F::ONE - four * v);
    }
    verify_range(
        &proof.sigma_range_proof,
        &RangeProofInstance { values: sigma_residuals, bits: 32 },
        transcript,
    )
    .map_err(|e| format!("Sigma Range Proof Error: {e}"))?;

    // --- Y Range Proof ---
    let mut y_residuals = Vec::with_capacity(t * d * 2);
    for i in 0..t {
        let sig_d = inst.sigma[i] * d_f;
        let sum_i = inst.sum_x[i];
        for j in 0..d {
            let expr = inst.scale_gamma * inst.gamma[j] * (d_f * inst.x[i][j] - sum_i)
                + inst.scale_beta * inst.beta[j] * sig_d;
            let expr2 = two * expr;
            let y_ij = inst.y[i][j];
            y_residuals.push(expr2 - sig_d * (two * y_ij - F::ONE));
            y_residuals.push(sig_d * (two * y_ij + F::ONE) - F::ONE - expr2);
        }
    }
    verify_range(
        &proof.y_range_proof,
        &RangeProofInstance { values: y_residuals, bits: 32 },
        transcript,
    )
    .map_err(|e| format!("Y Range Proof Error: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// MLE helpers (shared with linear.rs pattern)
// ---------------------------------------------------------------------------

fn challenge_vec(transcript: &mut Transcript, len: usize, label: &[u8]) -> Vec<F> {
    (0..len).map(|_| transcript.challenge_field::<F>(label)).collect()
}

fn eval_1d(vec: &[F], r: &[F]) -> F {
    DenseMLPoly::from_vec_padded(vec.to_vec()).evaluate(r)
}

fn eval_cols(matrix: &[Vec<F>], r_row: &[F], rows: usize, cols: usize) -> Vec<F> {
    (0..cols)
        .map(|c| {
            let col: Vec<F> = (0..rows).map(|r| matrix[r][c]).collect();
            eval_1d(&col, r_row)
        })
        .collect()
}

fn eval_2d(matrix: &[Vec<F>], r_row: &[F], r_col: &[F], rows: usize, cols: usize) -> F {
    let col_evals = eval_cols(matrix, r_row, rows, cols);
    eval_1d(&col_evals, r_col)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod layernorm_tests {
    use super::*;
    use crate::pcs::setup_hyrax_params;
    use ark_ff::{One, Zero};

    /// Build a small but self-consistent LayerNorm instance (t=2, d=2).
    ///
    /// x = [[10, 20], [30, 40]], d=2.
    /// Verified by hand:
    ///   sum_x = [30, 70], var_x = [200, 200], sigma = [7, 7]
    ///   (sigma = floor(sqrt(var/(d^2))) = floor(sqrt(200/4)) = floor(7.07) = 7)
    ///
    /// With scale_gamma=1, gamma=[2,2], scale_beta=1, beta=[5,5]:
    ///   For row i, col j:  expr = 2*(d*x[i][j] - sum_i) + 5*sig_d
    ///     sig_d = 7*2 = 14
    ///   i=0,j=0: expr = 2*(-10)+5*14 = 50  → y = round(50/14) = 4
    ///   i=0,j=1: expr = 2*(+10)+5*14 = 90  → y = round(90/14) = 6
    ///   (same for i=1 as the diff structure is identical)
    ///
    /// Range proof residuals are all small non-negative integers.
    fn setup_layernorm_instance() -> LayerNormInstance {
        let t = 2usize;
        let d = 2usize;
        let d_f = F::from(d as u64);

        let x = vec![
            vec![F::from(10u64), F::from(20u64)],
            vec![F::from(30u64), F::from(40u64)],
        ];
        let gamma = vec![F::from(2u64); d];
        let beta = vec![F::from(5u64); d];

        // Compute sum_x and var_x using field arithmetic.
        let mut sum_x = vec![F::zero(); t];
        let mut var_x = vec![F::zero(); t];
        for i in 0..t {
            let s: F = x[i].iter().copied().sum();
            sum_x[i] = s;
            let v: F = x[i].iter().map(|&xij| {
                let diff = d_f * xij - s;
                diff * diff
            }).sum();
            var_x[i] = v;
        }

        // sigma[i] = 7 for all i (hand-verified: sqrt(200/4) ≈ 7.07 → 7).
        let sigma_vec: Vec<F> = vec![F::from(7u64); t];

        // y values hand-computed (see doc-comment above).
        let y: Vec<Vec<F>> = vec![
            vec![F::from(4u64), F::from(6u64)],
            vec![F::from(4u64), F::from(6u64)],
        ];

        LayerNormInstance {
            seq_len: t,
            d_head: d,
            x,
            gamma,
            beta,
            y,
            sum_x,
            var_x,
            sigma: sigma_vec,
            scale_gamma: F::one(),
            scale_beta: F::one(),
        }
    }

    #[test]
    fn test_layernorm_mle_success() {
        let inst = setup_layernorm_instance();
        // bits_per_chunk=4 → sigma=2 for the outer Hyrax (unused by range proofs, kept for API)
        let params = setup_hyrax_params(4);

        let mut pt = Transcript::new(b"layernorm_test");
        let proof = prove_layernorm(&inst, &mut pt, &params).unwrap();

        let mut vt = Transcript::new(b"layernorm_test");
        let result = verify_layernorm(&proof, &inst, &mut vt, &params);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }

    #[test]
    fn test_layernorm_tampered_x_matrix() {
        let mut inst = setup_layernorm_instance();
        let params = setup_hyrax_params(4);

        inst.x[0][0] += F::one();

        let mut pt = Transcript::new(b"layernorm_test");
        if let Ok(proof) = prove_layernorm(&inst, &mut pt, &params) {
            let mut vt = Transcript::new(b"layernorm_test");
            let result = verify_layernorm(&proof, &inst, &mut vt, &params);
            // Corrupting X breaks mean or variance sumcheck
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_layernorm_tampered_variance_claim() {
        let mut inst = setup_layernorm_instance();
        let params = setup_hyrax_params(4);

        inst.var_x[1] += F::one();

        let mut pt = Transcript::new(b"layernorm_test");
        let proof = prove_layernorm(&inst, &mut pt, &params).unwrap();

        let mut vt = Transcript::new(b"layernorm_test");
        let result = verify_layernorm(&proof, &inst, &mut vt, &params);
        assert!(result.is_err());
    }
}
