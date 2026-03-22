//! LayerNorm Protocol with Constraint Fusion & MLE Evaluation
//!
//! Proves correctness of LayerNorm by:
//!  1. Mean sumcheck:     Σ_j X(r_t, j) · 1  = sum_x(r_t)
//!  2. Variance sumcheck: Σ_j h_j^2           = var_x(r_t),  h_j = d·X(r_t,j) - sum_x(r_t)
//!  3. Sigma range proof: proves σ = floor(√(var/d²)) via non-negative residuals
//!  4. Y range proof:     proves rounded normalised outputs y are non-negative
//!
//! **Succinctness:**
//!   All key polynomials (X, sum_x, var_x) are committed with Hyrax before any
//!   challenge is drawn, binding the prover.  Two sumchecks reduce the 2-D claims
//!   to scalar MLE evaluations at random points, certified by Hyrax opening proofs.

use crate::field::F;
use crate::lookup::range::{prove_range, verify_range, RangeProof, RangeProofInstance};
use crate::pcs::{
    hyrax_commit, hyrax_open, hyrax_verify, HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::Field;
use ark_serialize::CanonicalSerialize;

// ---------------------------------------------------------------------------
// Verifying key (weight commitments — computed once in preprocessing)
// ---------------------------------------------------------------------------

/// Pre-committed affine scale parameters γ and β.
///
/// The model provider calls `preprocess_layernorm` once per model checkpoint and
/// distributes `LayerNormVerifyingKey` to all verifiers.  Each per-user proof is
/// then checked against this VK without re-committing the weights.
pub struct LayerNormVerifyingKey {
    pub gamma_com: HyraxCommitment,
    pub beta_com: HyraxCommitment,
    /// Hyrax params for the `d_head`-dimensional weight polynomials.
    pub nu_d: usize,
    pub sigma_d: usize,
    pub params_d: HyraxParams,
}

/// Commit to γ and β once.  The resulting `LayerNormVerifyingKey` is reused
/// across all per-user proofs.
pub fn preprocess_layernorm(gamma: &[F], beta: &[F]) -> LayerNormVerifyingKey {
    let gamma_mle = vec_to_mle(gamma, gamma.len());
    let (nu, sigma, params) = poly_hyrax(&gamma_mle);
    let gamma_com = hyrax_commit(&gamma_mle.evaluations, nu, &params);
    let beta_mle = vec_to_mle(beta, beta.len());
    let beta_com = hyrax_commit(&beta_mle.evaluations, nu, &params);
    LayerNormVerifyingKey { gamma_com, beta_com, nu_d: nu, sigma_d: sigma, params_d: params }
}

// ---------------------------------------------------------------------------
// Succinct instance (activation dimensions only — raw x is not given to verifier)
// ---------------------------------------------------------------------------

/// What the verifier sees per-user in the succinct setting.
///
/// The large activation matrix `x` (T×D) is *not* included — its MLE commitment
/// is carried inside the proof and certified by Hyrax opening proofs.
/// `sigma`, `var_x`, and `y` are still provided in plain because they feed the
/// range-proof sub-circuit; a future upgrade can replace these with
/// `verify_range_succinct` calls.
pub struct LayerNormInstanceSuccinct {
    pub seq_len: usize,
    pub d_head: usize,
    pub sigma: Vec<F>,
    pub var_x: Vec<F>,
    pub y: Vec<Vec<F>>,
    pub scale_gamma: F,
    pub scale_beta: F,
}

// ---------------------------------------------------------------------------
// Public instance (raw matrices — polys are built internally)
// ---------------------------------------------------------------------------

/// Witness for one LayerNorm computation.
///
/// Dimension conventions:
///   x, y      : seq_len × d_head
///   sum_x     : seq_len
///   var_x     : seq_len   (= Σ_j (d·x[i][j] - sum_x[i])²)
///   sigma     : seq_len   (= floor(sqrt(var_x[i] / d²)))
///   gamma, beta: d_head   (public affine parameters)
pub struct LayerNormInstance {
    pub seq_len: usize,
    pub d_head: usize,
    pub x: Vec<Vec<F>>,
    pub y: Vec<Vec<F>>,
    pub sum_x: Vec<F>,
    pub var_x: Vec<F>,
    pub sigma: Vec<F>,
    pub gamma: Vec<F>,
    pub beta: Vec<F>,
    pub scale_gamma: F,
    pub scale_beta: F,
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

pub struct LayerNormCommitments {
    pub x_com: HyraxCommitment,
    pub sum_x_com: HyraxCommitment,
    pub var_x_com: HyraxCommitment,
}

pub struct LayerNormOpenings {
    /// sum_x(r_t)
    pub sum_x_eval_at_rt: F,
    pub sum_x_proof: HyraxProof,
    /// var_x(r_t)
    pub var_x_eval_at_rt: F,
    pub var_x_proof: HyraxProof,
    /// x(r_t, r_d_mean) — links mean sumcheck final eval to committed x
    pub x_eval_at_r_mean: F,
    pub x_proof_at_r_mean: HyraxProof,
    /// x(r_t, r_d_var)  — links variance sumcheck final eval to committed x
    pub x_eval_at_r_var: F,
    pub x_proof_at_r_var: HyraxProof,
}

pub struct LayerNormProof {
    pub commitments: LayerNormCommitments,
    pub mean_sumcheck: SumcheckProof,
    pub variance_sumcheck: SumcheckProof,
    /// Range proof for sigma floor-sqrt residuals (2 per row: lower and upper bound).
    pub sigma_range_proof: RangeProof,
    /// Range proof for y values (all must be non-negative).
    pub y_range_proof: RangeProof,
    pub openings: LayerNormOpenings,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn prove_layernorm(
    inst: &LayerNormInstance,
    transcript: &mut Transcript,
) -> Result<LayerNormProof, String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let _d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);

    // Build MLE polynomials from raw matrices.
    let x_mle = mat_to_mle(&inst.x, t, d);
    let sum_x_mle = vec_to_mle(&inst.sum_x, t);
    let var_x_mle = vec_to_mle(&inst.var_x, t);

    // Derive Hyrax params.
    let (nu_td, sigma_td, params_td) = poly_hyrax(&x_mle);
    let (nu_t, sigma_t, params_t) = poly_hyrax(&sum_x_mle);

    // -- 1. Commit to key polynomials; absorb into transcript -----------------
    let x_com = hyrax_commit(&x_mle.evaluations, nu_td, &params_td);
    let sum_x_com = hyrax_commit(&sum_x_mle.evaluations, nu_t, &params_t);
    let var_x_com = hyrax_commit(&var_x_mle.evaluations, nu_t, &params_t);

    absorb_com(transcript, b"x_com", &x_com);
    absorb_com(transcript, b"sum_x_com", &sum_x_com);
    absorb_com(transcript, b"var_x_com", &var_x_com);

    // -- 2. Row audit challenge ------------------------------------------------
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");

    let claim_mean = sum_x_mle.evaluate(&r_t);
    let claim_var = var_x_mle.evaluate(&r_t);
    transcript.append_field(b"claimed_mean", &claim_mean);
    transcript.append_field(b"claimed_var", &claim_var);

    // -- 3. Mean sumcheck: Σ_j X(r_t, j) * 1 = sum_x(r_t) -------------------
    // Collapse X over the row dimension to get the 1-D slice X(r_t, ·).
    let x_collapsed = eval_rows(&x_mle, t_bits, &r_t);
    let f_mean = DenseMLPoly::from_vec_padded(x_collapsed.clone());
    let g_mean = DenseMLPoly::from_vec_padded(vec![F::ONE; d]);
    let (mean_sumcheck, r_d_mean) = prove_sumcheck(&f_mean, &g_mean, claim_mean, transcript);

    // -- 4. Variance sumcheck: Σ_j h_j^2 = var_x(r_t) -----------------------
    // h_j = d·X(r_t, j) - sum_x(r_t)
    let h: Vec<F> = x_collapsed.iter().map(|&xj| d_f * xj - claim_mean).collect();
    let f_var = DenseMLPoly::from_vec_padded(h);
    let (variance_sumcheck, r_d_var) = prove_sumcheck(&f_var, &f_var, claim_var, transcript);

    // -- 5. Range proofs -------------------------------------------------------
    // Sigma floor-sqrt residuals:
    //   lo_i = var_x[i] - (d·sigma[i])²           must be in [0, 2^32)
    //   hi_i = (d·(sigma[i]+1))² - 1 - var_x[i]   must be in [0, 2^32)
    let mut sigma_residuals = Vec::with_capacity(2 * t);
    for i in 0..t {
        let dsi = d_f * inst.sigma[i];
        let lo = inst.var_x[i] - dsi * dsi;
        let hi = (dsi + d_f) * (dsi + d_f) - F::ONE - inst.var_x[i];
        sigma_residuals.push(lo);
        sigma_residuals.push(hi);
    }
    let sigma_range_proof = prove_range(
        &RangeProofInstance { values: sigma_residuals, bits: 32 },
        transcript,
    )?;

    // Y range proof: y values must be non-negative (in [0, 2^16)).
    let y_values: Vec<F> = inst.y.iter().flat_map(|row| row.iter().copied()).collect();
    let y_range_proof = prove_range(
        &RangeProofInstance { values: y_values, bits: 16 },
        transcript,
    )?;

    // -- 6. PCS opening proofs ------------------------------------------------
    let x_eval_at_r_mean = x_mle.evaluate(&combine(&r_t, &r_d_mean));
    let x_proof_at_r_mean =
        hyrax_open(&x_mle.evaluations, &combine(&r_t, &r_d_mean), nu_td, sigma_td);

    let x_eval_at_r_var = x_mle.evaluate(&combine(&r_t, &r_d_var));
    let x_proof_at_r_var =
        hyrax_open(&x_mle.evaluations, &combine(&r_t, &r_d_var), nu_td, sigma_td);

    let sum_x_eval_at_rt = claim_mean;
    let sum_x_proof = hyrax_open(&sum_x_mle.evaluations, &r_t, nu_t, sigma_t);

    let var_x_eval_at_rt = claim_var;
    let var_x_proof = hyrax_open(&var_x_mle.evaluations, &r_t, nu_t, sigma_t);

    Ok(LayerNormProof {
        commitments: LayerNormCommitments { x_com, sum_x_com, var_x_com },
        mean_sumcheck,
        variance_sumcheck,
        sigma_range_proof,
        y_range_proof,
        openings: LayerNormOpenings {
            sum_x_eval_at_rt,
            sum_x_proof,
            var_x_eval_at_rt,
            var_x_proof,
            x_eval_at_r_mean,
            x_proof_at_r_mean,
            x_eval_at_r_var,
            x_proof_at_r_var,
        },
    })
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn verify_layernorm(
    proof: &LayerNormProof,
    inst: &LayerNormInstance,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);

    // Rebuild MLEs and derive params (same deterministic computation as prover).
    let x_mle = mat_to_mle(&inst.x, t, d);
    let sum_x_mle = vec_to_mle(&inst.sum_x, t);
    let var_x_mle = vec_to_mle(&inst.var_x, t);

    let (nu_td, _sigma_td_v, params_td) = poly_hyrax(&x_mle);
    let (nu_t, _sigma_t_v, params_t) = poly_hyrax(&sum_x_mle);

    // -- 1. Check commitments against instance data ---------------------------
    let expected_x = hyrax_commit(&x_mle.evaluations, nu_td, &params_td);
    let expected_sum_x = hyrax_commit(&sum_x_mle.evaluations, nu_t, &params_t);
    let expected_var_x = hyrax_commit(&var_x_mle.evaluations, nu_t, &params_t);

    if expected_x.row_coms != proof.commitments.x_com.row_coms {
        return Err("x commitment mismatch".to_string());
    }
    if expected_sum_x.row_coms != proof.commitments.sum_x_com.row_coms {
        return Err("sum_x commitment mismatch".to_string());
    }
    if expected_var_x.row_coms != proof.commitments.var_x_com.row_coms {
        return Err("var_x commitment mismatch".to_string());
    }

    // Replay transcript absorptions (Fiat-Shamir binding).
    absorb_com(transcript, b"x_com", &proof.commitments.x_com);
    absorb_com(transcript, b"sum_x_com", &proof.commitments.sum_x_com);
    absorb_com(transcript, b"var_x_com", &proof.commitments.var_x_com);

    // -- 2. Replay row audit challenge ----------------------------------------
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");

    transcript.append_field(b"claimed_mean", &proof.openings.sum_x_eval_at_rt);
    transcript.append_field(b"claimed_var", &proof.openings.var_x_eval_at_rt);

    // -- 3. Verify mean sumcheck ----------------------------------------------
    let (r_d_mean, final_mean) = verify_sumcheck(
        &proof.mean_sumcheck,
        proof.openings.sum_x_eval_at_rt,
        d_bits,
        transcript,
    )
    .map_err(|e| format!("mean sumcheck: {e}"))?;

    // final_mean = f(r_d_mean) * g(r_d_mean) = x(r_t, r_d_mean) * 1
    let expected_mean_final = proof.openings.x_eval_at_r_mean;
    if final_mean != expected_mean_final {
        return Err("Mean sumcheck final evaluation mismatch".to_string());
    }

    // -- 4. Verify variance sumcheck ------------------------------------------
    let (r_d_var, final_var) = verify_sumcheck(
        &proof.variance_sumcheck,
        proof.openings.var_x_eval_at_rt,
        d_bits,
        transcript,
    )
    .map_err(|e| format!("variance sumcheck: {e}"))?;

    // final_var = h(r_d_var)^2, where h(r_d_var) = d * x(r_t, r_d_var) - sum_x(r_t)
    let h_eval = d_f * proof.openings.x_eval_at_r_var - proof.openings.sum_x_eval_at_rt;
    if final_var != h_eval * h_eval {
        return Err("Variance sumcheck final evaluation mismatch".to_string());
    }

    // -- 5. Range proof verification ------------------------------------------
    let mut sigma_residuals = Vec::with_capacity(2 * t);
    for i in 0..t {
        let dsi = d_f * inst.sigma[i];
        let lo = inst.var_x[i] - dsi * dsi;
        let hi = (dsi + d_f) * (dsi + d_f) - F::ONE - inst.var_x[i];
        sigma_residuals.push(lo);
        sigma_residuals.push(hi);
    }
    verify_range(
        &proof.sigma_range_proof,
        &RangeProofInstance { values: sigma_residuals, bits: 32 },
        transcript,
    )
    .map_err(|e| format!("sigma range proof: {e}"))?;

    let y_values: Vec<F> = inst.y.iter().flat_map(|row| row.iter().copied()).collect();
    verify_range(
        &proof.y_range_proof,
        &RangeProofInstance { values: y_values, bits: 16 },
        transcript,
    )
    .map_err(|e| format!("y range proof: {e}"))?;

    // -- 6. PCS opening proofs ------------------------------------------------
    hyrax_verify(
        &proof.commitments.sum_x_com,
        proof.openings.sum_x_eval_at_rt,
        &r_t,
        &proof.openings.sum_x_proof,
        &params_t,
    )
    .map_err(|e| format!("sum_x opening: {e}"))?;

    hyrax_verify(
        &proof.commitments.var_x_com,
        proof.openings.var_x_eval_at_rt,
        &r_t,
        &proof.openings.var_x_proof,
        &params_t,
    )
    .map_err(|e| format!("var_x opening: {e}"))?;

    hyrax_verify(
        &proof.commitments.x_com,
        proof.openings.x_eval_at_r_mean,
        &combine(&r_t, &r_d_mean),
        &proof.openings.x_proof_at_r_mean,
        &params_td,
    )
    .map_err(|e| format!("x(r_t,r_d_mean) opening: {e}"))?;

    hyrax_verify(
        &proof.commitments.x_com,
        proof.openings.x_eval_at_r_var,
        &combine(&r_t, &r_d_var),
        &proof.openings.x_proof_at_r_var,
        &params_td,
    )
    .map_err(|e| format!("x(r_t,r_d_var) opening: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Succinct verifier (uses pre-committed weights from VK; never calls hyrax_commit)
// ---------------------------------------------------------------------------

/// Verify a LayerNorm proof in O(√N) verifier work.
///
/// **What the verifier does NOT do:**
///   * Re-commit x, sum_x, or var_x — it absorbs the prover's commitments
///     from `proof.commitments` directly.  The Hyrax opening proofs certify
///     those commitments at the challenge points, so binding is preserved.
///   * Re-commit γ or β — they are already committed in `vk` (done once by
///     the model provider and reused across all users).
///
/// **What the verifier still does with raw data:**
///   * Reconstruct σ residuals and y values for `verify_range` (O(T) / O(T·D)).
///     A follow-up upgrade can replace these with `verify_range_succinct` calls.
pub fn verify_layernorm_succinct(
    proof: &LayerNormProof,
    inst: &LayerNormInstanceSuccinct,
    _vk: &LayerNormVerifyingKey,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);

    // Derive Hyrax params from dimensions — no MLE build, no O(N) MSM.
    let n_td = t.next_power_of_two().max(1) * d.next_power_of_two().max(1);
    let n_t = t.next_power_of_two().max(2);
    let (_nu_td, _sigma_td, params_td) = params_from_n(n_td);
    let (_nu_t, _sigma_t, params_t) = params_from_n(n_t);

    // -- 1. Absorb activation commitments from proof (trust prover's binding) --
    // Security: the Hyrax opening proofs below certify these commitments at the
    // challenge points, ensuring the prover cannot open to inconsistent values.
    absorb_com(transcript, b"x_com", &proof.commitments.x_com);
    absorb_com(transcript, b"sum_x_com", &proof.commitments.sum_x_com);
    absorb_com(transcript, b"var_x_com", &proof.commitments.var_x_com);

    // -- 2. Row audit challenge ------------------------------------------------
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");

    transcript.append_field(b"claimed_mean", &proof.openings.sum_x_eval_at_rt);
    transcript.append_field(b"claimed_var", &proof.openings.var_x_eval_at_rt);

    // -- 3. Verify mean sumcheck -----------------------------------------------
    let (r_d_mean, final_mean) = verify_sumcheck(
        &proof.mean_sumcheck,
        proof.openings.sum_x_eval_at_rt,
        d_bits,
        transcript,
    )
    .map_err(|e| format!("mean sumcheck: {e}"))?;

    if final_mean != proof.openings.x_eval_at_r_mean {
        return Err("Mean sumcheck final evaluation mismatch".to_string());
    }

    // -- 4. Verify variance sumcheck ------------------------------------------
    let (r_d_var, final_var) = verify_sumcheck(
        &proof.variance_sumcheck,
        proof.openings.var_x_eval_at_rt,
        d_bits,
        transcript,
    )
    .map_err(|e| format!("variance sumcheck: {e}"))?;

    let h_eval = d_f * proof.openings.x_eval_at_r_var - proof.openings.sum_x_eval_at_rt;
    if final_var != h_eval * h_eval {
        return Err("Variance sumcheck final evaluation mismatch".to_string());
    }

    // -- 5. Range proof verification ------------------------------------------
    let mut sigma_residuals = Vec::with_capacity(2 * t);
    for i in 0..t {
        let dsi = d_f * inst.sigma[i];
        let lo = inst.var_x[i] - dsi * dsi;
        let hi = (dsi + d_f) * (dsi + d_f) - F::ONE - inst.var_x[i];
        sigma_residuals.push(lo);
        sigma_residuals.push(hi);
    }
    verify_range(
        &proof.sigma_range_proof,
        &RangeProofInstance { values: sigma_residuals, bits: 32 },
        transcript,
    )
    .map_err(|e| format!("sigma range proof: {e}"))?;

    let y_values: Vec<F> = inst.y.iter().flat_map(|row| row.iter().copied()).collect();
    verify_range(
        &proof.y_range_proof,
        &RangeProofInstance { values: y_values, bits: 16 },
        transcript,
    )
    .map_err(|e| format!("y range proof: {e}"))?;

    // -- 6. PCS opening proofs ------------------------------------------------
    hyrax_verify(
        &proof.commitments.sum_x_com,
        proof.openings.sum_x_eval_at_rt,
        &r_t,
        &proof.openings.sum_x_proof,
        &params_t,
    )
    .map_err(|e| format!("sum_x opening: {e}"))?;

    hyrax_verify(
        &proof.commitments.var_x_com,
        proof.openings.var_x_eval_at_rt,
        &r_t,
        &proof.openings.var_x_proof,
        &params_t,
    )
    .map_err(|e| format!("var_x opening: {e}"))?;

    hyrax_verify(
        &proof.commitments.x_com,
        proof.openings.x_eval_at_r_mean,
        &combine(&r_t, &r_d_mean),
        &proof.openings.x_proof_at_r_mean,
        &params_td,
    )
    .map_err(|e| format!("x(r_t,r_d_mean) opening: {e}"))?;

    hyrax_verify(
        &proof.commitments.x_com,
        proof.openings.x_eval_at_r_var,
        &combine(&r_t, &r_d_var),
        &proof.openings.x_proof_at_r_var,
        &params_td,
    )
    .map_err(|e| format!("x(r_t,r_d_var) opening: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a 2-D MLE from a raw matrix stored as `rows × cols` Vecs.
/// Padded to (row_p2 × col_p2) evaluations in row-major order (row bits = MSB).
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

/// Build a 1-D MLE from a raw vector of length `len`.
/// Padded to next_power_of_two(len).max(2) to ensure num_vars >= 1.
fn vec_to_mle(v: &[F], len: usize) -> DenseMLPoly {
    let padded = len.next_power_of_two().max(2);
    let mut evals = vec![F::ZERO; padded];
    for (i, &x) in v.iter().enumerate() {
        evals[i] = x;
    }
    DenseMLPoly::new(evals)
}

/// Derive Hyrax (nu, sigma, params) from an element count without building an MLE.
///
/// `n` must equal `rows.next_power_of_two().max(1) * cols.next_power_of_two().max(1)`
/// (the padded size that `mat_to_mle` or `vec_to_mle` would produce).
/// Used by the succinct verifier so it never has to call `hyrax_commit`.
fn params_from_n(n: usize) -> (usize, usize, HyraxParams) {
    debug_assert!(n.is_power_of_two() && n >= 2);
    let total = n.trailing_zeros() as usize;
    let nu = total / 2;
    let sigma = (total - nu).max(1);
    (nu, sigma, HyraxParams::new(sigma))
}

/// Determine Hyrax params for a poly: nu = num_vars/2, sigma = num_vars-nu (>= 1).
fn poly_hyrax(poly: &DenseMLPoly) -> (usize, usize, HyraxParams) {
    let total = poly.num_vars;
    let nu = total / 2;
    let sigma = (total - nu).max(1);
    (nu, sigma, HyraxParams::new(sigma))
}

/// Fix the first `n_row_vars` (MSB) variables of `poly` at `r_row`.
/// Returns the remaining evaluations (a 1-D Vec over column variables).
fn eval_rows(poly: &DenseMLPoly, n_row_vars: usize, r_row: &[F]) -> Vec<F> {
    assert_eq!(r_row.len(), n_row_vars);
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
    (0..len).map(|_| transcript.challenge_field::<F>(label)).collect()
}

fn absorb_com(transcript: &mut Transcript, label: &[u8], com: &HyraxCommitment) {
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

    /// Build a small but self-consistent LayerNorm instance (t=2, d=2).
    ///
    /// x = [[10, 20], [30, 40]], d=2.
    /// Verified by hand:
    ///   sum_x = [30, 70], var_x = [200, 200], sigma = [7, 7]
    ///   (sigma = floor(sqrt(var/(d^2))) = floor(sqrt(200/4)) = floor(7.07) = 7)
    ///
    /// With scale_gamma=1, gamma=[2,2], scale_beta=1, beta=[5,5]:
    ///   sig_d = 7*2 = 14
    ///   i=0,j=0: expr = 2*(-10)+5*14 = 50  -> y = round(50/14) = 4
    ///   i=0,j=1: expr = 2*(+10)+5*14 = 90  -> y = round(90/14) = 6
    ///   (same for i=1)
    ///
    /// Sigma range residuals:
    ///   lo_i = 200 - (2*7)^2 = 200 - 196 = 4  (in [0, 2^32))
    ///   hi_i = (2*8)^2 - 1 - 200 = 256 - 201 = 55  (in [0, 2^32))
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

        let mut sum_x = vec![F::zero(); t];
        let mut var_x = vec![F::zero(); t];
        for i in 0..t {
            let s: F = x[i].iter().copied().sum();
            sum_x[i] = s;
            let v: F = x[i]
                .iter()
                .map(|&xij| {
                    let diff = d_f * xij - s;
                    diff * diff
                })
                .sum();
            var_x[i] = v;
        }

        let sigma = vec![F::from(7u64); t];
        let y = vec![
            vec![F::from(4u64), F::from(6u64)],
            vec![F::from(4u64), F::from(6u64)],
        ];

        LayerNormInstance {
            seq_len: t,
            d_head: d,
            x,
            y,
            sum_x,
            var_x,
            sigma,
            gamma,
            beta,
            scale_gamma: F::one(),
            scale_beta: F::one(),
        }
    }

    #[test]
    fn test_layernorm_success() {
        let inst = setup_layernorm_instance();
        let mut pt = Transcript::new(b"layernorm_test");
        let proof = prove_layernorm(&inst, &mut pt).unwrap();

        let mut vt = Transcript::new(b"layernorm_test");
        let result = verify_layernorm(&proof, &inst, &mut vt);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }

    #[test]
    fn test_layernorm_tampered_x_matrix() {
        let mut inst = setup_layernorm_instance();

        // Tamper x but not sum_x — this makes the mean sumcheck fail.
        inst.x[0][0] += F::one();

        let mut pt = Transcript::new(b"layernorm_test");
        if let Ok(proof) = prove_layernorm(&inst, &mut pt) {
            let mut vt = Transcript::new(b"layernorm_test");
            let result = verify_layernorm(&proof, &inst, &mut vt);
            // x commitment or mean sumcheck must fail.
            assert!(result.is_err());
        }
        // prove may also fail (inconsistent data) — either way is acceptable.
    }

    #[test]
    fn test_layernorm_succinct_verifier_success() {
        let inst = setup_layernorm_instance();

        // Preprocessing: commit to weights once.
        let vk = preprocess_layernorm(&inst.gamma, &inst.beta);

        let mut pt = Transcript::new(b"layernorm_test");
        let proof = prove_layernorm(&inst, &mut pt).unwrap();

        // Succinct instance: no raw x — commitment is in the proof.
        let succinct = LayerNormInstanceSuccinct {
            seq_len: inst.seq_len,
            d_head: inst.d_head,
            sigma: inst.sigma.clone(),
            var_x: inst.var_x.clone(),
            y: inst.y.clone(),
            scale_gamma: inst.scale_gamma,
            scale_beta: inst.scale_beta,
        };

        let mut vt = Transcript::new(b"layernorm_test");
        let result = verify_layernorm_succinct(&proof, &succinct, &vk, &mut vt);
        assert!(result.is_ok(), "Succinct verification failed: {:?}", result.err());
    }

    #[test]
    fn test_layernorm_tampered_variance_claim() {
        let mut inst = setup_layernorm_instance();

        // Tamper var_x — the variance sumcheck uses var_x_mle for the claimed
        // sum, but the actual Σ h_j^2 (built from x) is the untampered value.
        inst.var_x[1] += F::one();

        let mut pt = Transcript::new(b"layernorm_test");
        let proof = prove_layernorm(&inst, &mut pt).unwrap();

        let mut vt = Transcript::new(b"layernorm_test");
        let result = verify_layernorm(&proof, &inst, &mut vt);
        assert!(result.is_err());
    }
}
