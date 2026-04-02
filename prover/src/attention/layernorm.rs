//! LayerNorm with Lasso-based inv_sigma lookup.
//!
//! **Protocol overview:**
//!
//!  Old approach: commit sigma, sigma_sq, sigma_y; prove sigma = floor_sqrt(var_x) via range proof.
//!  New approach: define inv_sigma via a precommitted lookup table (approximating 1/sqrt(var_x)),
//!                prove correct table application via Lasso, and express Y directly with norm_x.
//!
//!  Removed commitments: sigma_com, sigma_sq_com, sigma_y_com, gamma_x_com.
//!  Removed proofs:      sigma_range_proof.
//!  Added:               inv_sigma_com, norm_x_com, inv_sigma_lasso (one global Lasso).
//!
//! **Constraint summary (per row i, column j):**
//!
//!   1. Mean:    sum_x[i] = Σ_j x[i][j]                           (sumcheck)
//!   2. Lookup:  inv_sigma[i] = T_0[var_x_hi[i]] + T_1[var_x_lo[i]] (Lasso)
//!   3. Binding: var_x[i] = d*(d*sq_sum_x[i] - sum_x_sq[i])        (poly opening)
//!   4. Product: norm_x[i][j] = (d*x[i][j] - sum_x[i])*inv_sigma[i] (poly identity)
//!   5. Output:  y[i][j] ≈ (γ[j]*norm_x[i][j] + β[j]*S) / S        (range proof)
//!               where S = INV_SQRT_SCALE = 2^16.

use crate::field::F;
use crate::lookup::inv_sqrt::{
    lookup_inv_sqrt, precommit_inv_sqrt_tables, CHUNK_BITS, CHUNK_SIZE, INV_SQRT_SCALE,
};
use crate::lookup::lasso::{
    prove_lasso, verify_lasso, LassoInstance, LassoProof, LassoProvingKey, LassoVerifyingKey,
};
use crate::lookup::range::{
    prove_range, verify_range_deferred, verify_range_m_batch, RangeProof, RangeProofWitness,
};
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_open, hyrax_open_batch, params_from_n, poly_hyrax,
    HyraxBatchAccumulator, HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::poly::utils::{combine, compute_eq_evals, eval_rows, mat_to_mle, vec_to_mle};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::{challenge_vec, Transcript};
use ark_ff::{Field, PrimeField};

// ---------------------------------------------------------------------------
// Setup Keys
// ---------------------------------------------------------------------------

/// Precomputed Lasso key for the inv_sigma lookup tables T_0, T_1.
/// Compute once at model setup; share across all LayerNorm sub-provers.
#[derive(Clone)]
pub struct LayerNormLassoKey {
    pub pk: LassoProvingKey,
    pub t0: Vec<F>,
    pub t1: Vec<F>,
    pub table_params: HyraxParams,
}

impl LayerNormLassoKey {
    /// One-time setup: build and commit the two inv-sqrt sub-tables.
    pub fn setup() -> Self {
        let (_, _, table_params) = params_from_n(CHUNK_SIZE);
        let (pk, t0, t1) = precommit_inv_sqrt_tables(&table_params);
        LayerNormLassoKey { pk, t0, t1, table_params }
    }

    pub fn vk(&self) -> LayerNormLassoVerifyingKey {
        LayerNormLassoVerifyingKey {
            lasso_vk: self.pk.vk(),
            t0: self.t0.clone(),
            t1: self.t1.clone(),
            table_params: self.table_params.clone(),
        }
    }
}

/// Verifier-side Lasso key for inv_sigma.
#[derive(Clone)]
pub struct LayerNormLassoVerifyingKey {
    pub lasso_vk: LassoVerifyingKey,
    pub t0: Vec<F>,
    pub t1: Vec<F>,
    pub table_params: HyraxParams,
}

// ---------------------------------------------------------------------------
// Pipeline Interfaces
// ---------------------------------------------------------------------------

/// Trusted IO Commitments provided by the Global Pipeline Verifier.
pub struct LayerNormIOCommitments {
    pub x_com: HyraxCommitment,
    pub y_com: HyraxCommitment,
}

/// Public model weights.
#[derive(Clone)]
pub struct LayerNormVerifyingKey {
    pub seq_len: usize,
    pub d_head: usize,
    pub gamma: Vec<F>,
    pub beta: Vec<F>,
    pub scale_gamma: F,
    pub scale_beta: F,
}

// ---------------------------------------------------------------------------
// Witness
// ---------------------------------------------------------------------------

/// Private witness. ONLY the Prover holds this.
pub struct LayerNormWitness {
    pub x: Vec<Vec<F>>,
    pub y: Vec<Vec<F>>,
    pub sum_x: Vec<F>,
    pub sq_sum_x: Vec<F>,
    pub sum_x_sq: Vec<F>,  // sum_x[i]^2 per row
    pub var_x: Vec<u64>,   // d*(d*sq_sum_x[i]-sum_x_sq[i]) as u64, Lasso query index
    pub inv_sigma: Vec<F>, // T_0[var_x >> 16] + T_1[var_x & 0xFFFF]
    pub norm_x: Vec<Vec<F>>, // (d*x[i][j] - sum_x[i]) * inv_sigma[i]
}

impl LayerNormWitness {
    /// Compute witness from raw inputs using the given lookup tables.
    pub fn from_inputs(
        x: Vec<Vec<F>>,
        y: Vec<Vec<F>>,
        d: usize,
        t0: &[F],
        t1: &[F],
    ) -> Self {
        let t = x.len();
        let d_f = F::from(d as u64);

        let mut sum_x = vec![F::ZERO; t];
        let mut sq_sum_x = vec![F::ZERO; t];
        let mut sum_x_sq = vec![F::ZERO; t];
        let mut var_x = vec![0u64; t];
        let mut inv_sigma = vec![F::ZERO; t];
        let mut norm_x = vec![vec![F::ZERO; d]; t];

        for i in 0..t {
            let s: F = x[i].iter().copied().sum();
            let q: F = x[i].iter().map(|v| *v * *v).sum();
            sum_x[i] = s;
            sq_sum_x[i] = q;
            sum_x_sq[i] = s * s;

            // Compute var_x as integer. Note: d*(d*sq_sum_x - sum_x^2) must fit in u64.
            // We extract this via the canonical integer representation of the field element.
            // For correctly bounded inputs, this value is non-negative and small.
            let v_f = d_f * (d_f * q - s * s);
            // Convert to u64 by interpreting the field element as a small integer.
            var_x[i] = field_to_u64(v_f);

            inv_sigma[i] = lookup_inv_sqrt(var_x[i], t0, t1);

            for j in 0..d {
                norm_x[i][j] = (d_f * x[i][j] - s) * inv_sigma[i];
            }
        }

        LayerNormWitness { x, y, sum_x, sq_sum_x, sum_x_sq, var_x, inv_sigma, norm_x }
    }

    /// Compute the full witness from x, including correct y via the LayerNorm formula.
    /// Equivalent to `from_inputs` but derives y = round((γ·norm_x + β·S)/S) instead of
    /// requiring the caller to pre-compute it.
    pub fn from_forward(
        x: Vec<Vec<F>>,
        gamma: &[F],
        beta: &[F],
        scale_gamma: F,
        scale_beta: F,
        t0: &[F],
        t1: &[F],
    ) -> Self {
        let d = x.first().map(|r| r.len()).unwrap_or(0);
        let t = x.len();
        let inv_scale_f = F::from(INV_SQRT_SCALE);
        let scale = INV_SQRT_SCALE;

        // First compute norm_x using placeholder y
        let partial = Self::from_inputs(x.clone(), vec![vec![F::ZERO; d]; t], d, t0, t1);

        // Derive y from norm_x
        let mut y = vec![vec![F::ZERO; d]; t];
        for i in 0..t {
            for j in 0..d {
                let expr = scale_gamma * gamma[j] * partial.norm_x[i][j]
                    + scale_beta * beta[j] * inv_scale_f;
                let expr_int = field_to_u64(expr);
                y[i][j] = F::from((expr_int + scale / 2) / scale);
            }
        }

        // Re-build with correct y (norm_x, inv_sigma etc. are the same)
        Self { y, ..partial }
    }
}

/// Convert a field element representing a small non-negative integer to u64.
/// Panics if the value doesn't fit (upper 3 limbs of BN254 Fr are non-zero).
pub fn field_to_u64(v: F) -> u64 {
    let bi = v.into_bigint(); // BigInt<4> for BN254 Fr
    assert!(
        bi.0[1] == 0 && bi.0[2] == 0 && bi.0[3] == 0,
        "field_to_u64: value too large for u64"
    );
    bi.0[0]
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

pub struct LayerNormInternalCommitments {
    pub sum_x_com: HyraxCommitment,
    pub sq_sum_x_com: HyraxCommitment,
    pub sum_x_sq_com: HyraxCommitment,
    pub inv_sigma_com: HyraxCommitment,
    pub norm_x_com: HyraxCommitment,
}

pub struct LayerNormOpenings {
    // Group A: at r_t — [sum_x, sq_sum_x]
    pub sum_x_at_rt: F,
    pub sq_sum_x_at_rt: F,
    pub rt_batch_proof: HyraxProof,

    // Individual: x at combine(r_t, r_d_mean)
    pub x_at_rt_rmean: F,
    pub x_rt_rmean_proof: HyraxProof,

    // Group B: at r_bind_t — [sq_sum_x, sum_x_sq, inv_sigma]
    // Used to bind var_x (Lasso query indices) and inv_sigma to committed polynomials.
    pub sq_sum_x_at_rbind: F,
    pub sum_x_sq_at_rbind: F,
    pub inv_sigma_at_rbind: F,
    pub rbind_batch_proof: HyraxProof,

    // Group C: at combine(r_y_t, r_y_d) — [x, y, norm_x]
    pub x_at_ry: F,
    pub y_at_ry: F,
    pub norm_x_at_ry: F,
    pub ry_td_batch_proof: HyraxProof,

    // Group D: at r_y_t — [sum_x, inv_sigma]
    // Used for norm_x product check: norm_x_at_ry = (d*x_at_ry - sum_x_at_ryt)*inv_sigma_at_ryt
    pub sum_x_at_ryt: F,
    pub inv_sigma_at_ryt: F,
    pub ryt_batch_proof: HyraxProof,
}

pub struct LayerNormProof {
    pub internal_coms: LayerNormInternalCommitments,
    pub mean_sumcheck: SumcheckProof,
    // Per-proof Lasso data: query indices (var_x as usize) and outputs (inv_sigma).
    // The verifier reconstructs the full LassoInstance from these + vk tables.
    pub lasso_query_indices: Vec<usize>,
    pub lasso_outputs: Vec<F>,
    pub inv_sigma_lasso: LassoProof,
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
    lk: &LayerNormLassoKey,
    transcript: &mut Transcript,
) -> Result<LayerNormProof, String> {
    let t = vk.seq_len;
    let d = vk.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);
    let inv_scale_f = F::from(INV_SQRT_SCALE);
    let two = F::from(2u64);

    // -- Build MLEs --
    let x_mle = mat_to_mle(&witness.x, t, d);
    let y_mle = mat_to_mle(&witness.y, t, d);
    let sum_x_mle = vec_to_mle(&witness.sum_x, t);
    let sq_sum_x_mle = vec_to_mle(&witness.sq_sum_x, t);
    let sum_x_sq_mle = vec_to_mle(&witness.sum_x_sq, t);
    let inv_sigma_mle = vec_to_mle(&witness.inv_sigma, t);
    let norm_x_mle = mat_to_mle(&witness.norm_x, t, d);

    let (nu_td, sigma_td, params_td) = poly_hyrax(&x_mle);
    let (nu_t, sigma_t, params_t) = poly_hyrax(&sum_x_mle);

    // 1. Absorb IO commitments
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);

    // 2. Commit to internal variables
    let sum_x_com = hyrax_commit(&sum_x_mle.evaluations, nu_t, &params_t);
    let sq_sum_x_com = hyrax_commit(&sq_sum_x_mle.evaluations, nu_t, &params_t);
    let sum_x_sq_com = hyrax_commit(&sum_x_sq_mle.evaluations, nu_t, &params_t);
    let inv_sigma_com = hyrax_commit(&inv_sigma_mle.evaluations, nu_t, &params_t);
    let norm_x_com = hyrax_commit(&norm_x_mle.evaluations, nu_td, &params_td);

    absorb_com(transcript, b"sum_x_com", &sum_x_com);
    absorb_com(transcript, b"sq_sum_x_com", &sq_sum_x_com);
    absorb_com(transcript, b"sum_x_sq_com", &sum_x_sq_com);
    absorb_com(transcript, b"inv_sigma_com", &inv_sigma_com);
    absorb_com(transcript, b"norm_x_com", &norm_x_com);

    // 3. Row challenge + mean sumcheck
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");
    let claim_s = sum_x_mle.evaluate(&r_t);
    let claim_q = sq_sum_x_mle.evaluate(&r_t);
    transcript.append_field(b"claimed_s", &claim_s);
    transcript.append_field(b"claimed_q", &claim_q);

    let x_collapsed = eval_rows(&x_mle, t_bits, &r_t);
    let f_mean = DenseMLPoly::from_vec_padded(x_collapsed);
    let g_mean = DenseMLPoly::from_vec_padded(vec![F::ONE; d]);
    let (mean_sumcheck, r_d_mean) = prove_sumcheck(&f_mean, &g_mean, claim_s, transcript);

    // 4. Lasso proof for inv_sigma lookup
    let lasso_query_indices: Vec<usize> = witness.var_x.iter().map(|&v| v as usize).collect();
    let lasso_outputs: Vec<F> = witness.inv_sigma.clone();
    let lasso_instance = LassoInstance {
        tables: vec![lk.t0.clone(), lk.t1.clone()],
        query_indices: lasso_query_indices.clone(),
        outputs: lasso_outputs.clone(),
        bits_per_chunk: CHUNK_BITS,
    };
    let inv_sigma_lasso = prove_lasso(&lasso_instance, &lk.pk, transcript, &lk.table_params);

    // 5. Binding challenge — ties var_x and inv_sigma to committed polynomials
    let r_bind_t = challenge_vec(transcript, t_bits, b"ln_rbind");

    // 6. Y range proof (new constraint: y = round((γ·norm_x + β·S) / S))
    // Rounding condition: 2*expr - S*(2*y-1) ≥ 0 and S*(2*y+1) - 1 - 2*expr ≥ 0
    let mut y_res = Vec::with_capacity(2 * t * d);
    for i in 0..t {
        for j in 0..d {
            let expr = vk.scale_gamma * vk.gamma[j] * witness.norm_x[i][j]
                + vk.scale_beta * vk.beta[j] * inv_scale_f;
            let expr2 = two * expr;
            let y_ij = witness.y[i][j];
            // lo: 2*expr - S*(2*y - 1) >= 0
            y_res.push(expr2 - inv_scale_f * (two * y_ij - F::ONE));
            // hi: S*(2*y + 1) - 1 - 2*expr >= 0
            y_res.push(inv_scale_f * (two * y_ij + F::ONE) - F::ONE - expr2);
        }
    }
    let (y_range_proof, r_y) =
        prove_range(&RangeProofWitness { values: y_res }, 32, transcript)?;
    let r_y_t = r_y[0..t_bits].to_vec();
    let r_y_d = r_y[t_bits..t_bits + d_bits].to_vec();

    // Advance transcript to match verifier's verify_range_m_batch lambda challenge.
    let _ = transcript.challenge_field::<F>(b"hyrax_mp_lambda");

    // 7. Batched Hyrax openings

    // Group A: [sum_x, sq_sum_x] at r_t
    let rt_batch_proof = hyrax_open_batch(
        &[&sum_x_mle.evaluations, &sq_sum_x_mle.evaluations],
        &r_t,
        nu_t,
        sigma_t,
        transcript,
    );

    // Individual: x at combine(r_t, r_d_mean)
    let x_rt_rmean_proof =
        hyrax_open(&x_mle.evaluations, &combine(&r_t, &r_d_mean), nu_td, sigma_td);

    // Group B: [sq_sum_x, sum_x_sq, inv_sigma] at r_bind_t
    let rbind_batch_proof = hyrax_open_batch(
        &[
            &sq_sum_x_mle.evaluations,
            &sum_x_sq_mle.evaluations,
            &inv_sigma_mle.evaluations,
        ],
        &r_bind_t,
        nu_t,
        sigma_t,
        transcript,
    );

    // Group C: [x, y, norm_x] at combine(r_y_t, r_y_d)
    let ry_td = combine(&r_y_t, &r_y_d);
    let ry_td_batch_proof = hyrax_open_batch(
        &[
            &x_mle.evaluations,
            &y_mle.evaluations,
            &norm_x_mle.evaluations,
        ],
        &ry_td,
        nu_td,
        sigma_td,
        transcript,
    );

    // Group D: [sum_x, inv_sigma] at r_y_t
    let ryt_batch_proof = hyrax_open_batch(
        &[&sum_x_mle.evaluations, &inv_sigma_mle.evaluations],
        &r_y_t,
        nu_t,
        sigma_t,
        transcript,
    );

    Ok(LayerNormProof {
        internal_coms: LayerNormInternalCommitments {
            sum_x_com,
            sq_sum_x_com,
            sum_x_sq_com,
            inv_sigma_com,
            norm_x_com,
        },
        mean_sumcheck,
        lasso_query_indices,
        lasso_outputs,
        inv_sigma_lasso,
        y_range_proof,
        openings: LayerNormOpenings {
            sum_x_at_rt: claim_s,
            sq_sum_x_at_rt: claim_q,
            rt_batch_proof,
            x_at_rt_rmean: x_mle.evaluate(&combine(&r_t, &r_d_mean)),
            x_rt_rmean_proof,
            sq_sum_x_at_rbind: sq_sum_x_mle.evaluate(&r_bind_t),
            sum_x_sq_at_rbind: sum_x_sq_mle.evaluate(&r_bind_t),
            inv_sigma_at_rbind: inv_sigma_mle.evaluate(&r_bind_t),
            rbind_batch_proof,
            x_at_ry: x_mle.evaluate(&ry_td),
            y_at_ry: y_mle.evaluate(&ry_td),
            norm_x_at_ry: norm_x_mle.evaluate(&ry_td),
            ry_td_batch_proof,
            sum_x_at_ryt: sum_x_mle.evaluate(&r_y_t),
            inv_sigma_at_ryt: inv_sigma_mle.evaluate(&r_y_t),
            ryt_batch_proof,
        },
    })
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn verify_layernorm(
    proof: &LayerNormProof,
    io_coms: &LayerNormIOCommitments,
    vk: &LayerNormVerifyingKey,
    lasso_vk: &LayerNormLassoVerifyingKey,
    transcript: &mut Transcript,
    acc_t: &mut HyraxBatchAccumulator,
    acc_td: &mut HyraxBatchAccumulator,
) -> Result<(), String> {
    let t = vk.seq_len;
    let d = vk.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);
    let inv_scale_f = F::from(INV_SQRT_SCALE);
    let two = F::from(2u64);

    // 1. Absorb commitments
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);
    absorb_com(transcript, b"sum_x_com", &proof.internal_coms.sum_x_com);
    absorb_com(transcript, b"sq_sum_x_com", &proof.internal_coms.sq_sum_x_com);
    absorb_com(transcript, b"sum_x_sq_com", &proof.internal_coms.sum_x_sq_com);
    absorb_com(transcript, b"inv_sigma_com", &proof.internal_coms.inv_sigma_com);
    absorb_com(transcript, b"norm_x_com", &proof.internal_coms.norm_x_com);

    // 2. Mean sumcheck
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");
    transcript.append_field(b"claimed_s", &proof.openings.sum_x_at_rt);
    transcript.append_field(b"claimed_q", &proof.openings.sq_sum_x_at_rt);

    let (r_d_mean, final_mean) =
        verify_sumcheck(&proof.mean_sumcheck, proof.openings.sum_x_at_rt, d_bits, transcript)
            .map_err(|e| format!("Mean sumcheck: {e}"))?;
    if final_mean != proof.openings.x_at_rt_rmean {
        return Err("Mean sumcheck final mismatch".into());
    }

    // 3. Lasso verification for inv_sigma
    let lasso_instance = LassoInstance {
        tables: vec![lasso_vk.t0.clone(), lasso_vk.t1.clone()],
        query_indices: proof.lasso_query_indices.clone(),
        outputs: proof.lasso_outputs.clone(),
        bits_per_chunk: CHUNK_BITS,
    };
    verify_lasso(
        &proof.inv_sigma_lasso,
        &lasso_instance,
        &lasso_vk.lasso_vk,
        transcript,
        &lasso_vk.table_params,
    )
    .map_err(|e| format!("inv_sigma Lasso: {e}"))?;

    // 4. Binding challenge
    let r_bind_t = challenge_vec(transcript, t_bits, b"ln_rbind");

    // 4a. Verify var_x binding: var_x_mle(r_bind) == d*(d*sq_sum_x - sum_x_sq) at r_bind.
    // The verifier computes var_x_mle from the prover-supplied query indices.
    let var_x_at_rbind = {
        let eq_evals = compute_eq_evals_for_t(t, &r_bind_t);
        proof
            .lasso_query_indices
            .iter()
            .zip(eq_evals.iter())
            .map(|(&idx, &eq)| F::from(idx as u64) * eq)
            .sum::<F>()
    };
    let expected_var_x = d_f * (d_f * proof.openings.sq_sum_x_at_rbind
        - proof.openings.sum_x_sq_at_rbind);
    if var_x_at_rbind != expected_var_x {
        return Err("var_x binding mismatch: query indices inconsistent with committed sq_sum_x/sum_x_sq".into());
    }

    // 4b. Verify inv_sigma binding: inv_sigma_com(r_bind) == MLE of Lasso outputs at r_bind.
    let inv_sigma_lasso_at_rbind = {
        let eq_evals = compute_eq_evals_for_t(t, &r_bind_t);
        proof
            .lasso_outputs
            .iter()
            .zip(eq_evals.iter())
            .map(|(&out, &eq)| out * eq)
            .sum::<F>()
    };
    if proof.openings.inv_sigma_at_rbind != inv_sigma_lasso_at_rbind {
        return Err("inv_sigma binding mismatch: inv_sigma_com inconsistent with Lasso outputs".into());
    }

    // 5. Y range proof verification
    let (r_y, y_eval, r_m_y) =
        verify_range_deferred(&proof.y_range_proof, t_bits + d_bits + 1, 32, transcript)
            .map_err(|e| format!("Y Range: {e}"))?;
    let r_y_t = r_y[0..t_bits].to_vec();
    let r_y_d = r_y[t_bits..t_bits + d_bits].to_vec();
    let r_y_b = r_y[t_bits + d_bits];

    let gamma_r = vec_to_mle(&vk.gamma, d).evaluate(&r_y_d);
    let beta_r = vec_to_mle(&vk.beta, d).evaluate(&r_y_d);
    // New Y constraint fusion:
    // expr = scale_gamma * γ(r_d) * norm_x(r_td) + scale_beta * β(r_d) * S
    // lo = 2*expr - S*(2*y - 1) >= 0  (S = INV_SQRT_SCALE)
    // hi = S*(2*y + 1) - 1 - 2*expr >= 0
    let expr = vk.scale_gamma * gamma_r * proof.openings.norm_x_at_ry
        + vk.scale_beta * beta_r * inv_scale_f;
    let expr2 = two * expr;
    let y_ev = proof.openings.y_at_ry;
    let lo_y = expr2 - inv_scale_f * (two * y_ev - F::ONE);
    let hi_y = inv_scale_f * (two * y_ev + F::ONE) - F::ONE - expr2;
    let expected_y_res = (F::ONE - r_y_b) * lo_y + r_y_b * hi_y;
    if y_eval != expected_y_res {
        return Err("Y constraint fusion mismatch".into());
    }

    // 5a. norm_x product check: norm_x_at_ry == (d*x_at_ry - sum_x_at_ryt) * inv_sigma_at_ryt
    let expected_norm_x = (d_f * proof.openings.x_at_ry - proof.openings.sum_x_at_ryt)
        * proof.openings.inv_sigma_at_ryt;
    if proof.openings.norm_x_at_ry != expected_norm_x {
        return Err("norm_x product check failed".into());
    }

    // 6. Batch-verify m_com opening for y_range_proof
    verify_range_m_batch(&[(&proof.y_range_proof, &r_m_y)], transcript)
        .map_err(|e| format!("LN range m_batch: {e}"))?;

    // 7. Accumulate all batched Hyrax openings

    // Group A: [sum_x, sq_sum_x] at r_t → acc_t
    acc_t.add_verify_batch(
        &[
            proof.internal_coms.sum_x_com.clone(),
            proof.internal_coms.sq_sum_x_com.clone(),
        ],
        &[proof.openings.sum_x_at_rt, proof.openings.sq_sum_x_at_rt],
        &r_t,
        &proof.openings.rt_batch_proof,
        transcript,
    )?;

    // Individual: x at combine(r_t, r_d_mean) → acc_td
    acc_td.add_verify(
        &io_coms.x_com,
        proof.openings.x_at_rt_rmean,
        &combine(&r_t, &r_d_mean),
        &proof.openings.x_rt_rmean_proof,
    )?;

    // Group B: [sq_sum_x, sum_x_sq, inv_sigma] at r_bind_t → acc_t
    acc_t.add_verify_batch(
        &[
            proof.internal_coms.sq_sum_x_com.clone(),
            proof.internal_coms.sum_x_sq_com.clone(),
            proof.internal_coms.inv_sigma_com.clone(),
        ],
        &[
            proof.openings.sq_sum_x_at_rbind,
            proof.openings.sum_x_sq_at_rbind,
            proof.openings.inv_sigma_at_rbind,
        ],
        &r_bind_t,
        &proof.openings.rbind_batch_proof,
        transcript,
    )?;

    // Group C: [x, y, norm_x] at combine(r_y_t, r_y_d) → acc_td
    let ry_td = combine(&r_y_t, &r_y_d);
    acc_td.add_verify_batch(
        &[
            io_coms.x_com.clone(),
            io_coms.y_com.clone(),
            proof.internal_coms.norm_x_com.clone(),
        ],
        &[proof.openings.x_at_ry, proof.openings.y_at_ry, proof.openings.norm_x_at_ry],
        &ry_td,
        &proof.openings.ry_td_batch_proof,
        transcript,
    )?;

    // Group D: [sum_x, inv_sigma] at r_y_t → acc_t
    acc_t.add_verify_batch(
        &[
            proof.internal_coms.sum_x_com.clone(),
            proof.internal_coms.inv_sigma_com.clone(),
        ],
        &[proof.openings.sum_x_at_ryt, proof.openings.inv_sigma_at_ryt],
        &r_y_t,
        &proof.openings.ryt_batch_proof,
        transcript,
    )?;

    Ok(())
}

/// Compute eq polynomial evaluations for the first `t` positions at challenge `r`.
/// Returns a vector of length `t` where entry i = eq(binary(i), r).
fn compute_eq_evals_for_t(t: usize, r: &[F]) -> Vec<F> {
    let t_pow = t.next_power_of_two();
    let all_evals = compute_eq_evals(r, t_pow);
    all_evals[..t].to_vec()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod layernorm_tests {
    use super::*;
    use crate::pcs::params_from_n;

    fn setup_test_pipeline() -> (
        LayerNormWitness,
        LayerNormIOCommitments,
        LayerNormVerifyingKey,
        LayerNormLassoKey,
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
        let scale_gamma = F::ONE;
        let scale_beta = F::ONE;

        let lk = LayerNormLassoKey::setup();

        // Compute witness using the lookup tables
        let mut y = vec![vec![F::ZERO; d]; t];
        let witness_partial =
            LayerNormWitness::from_inputs(x.clone(), y.clone(), d, &lk.t0, &lk.t1);

        // Compute correct y from the lookup-based formula
        let inv_scale_f = F::from(INV_SQRT_SCALE);
        for i in 0..t {
            for j in 0..d {
                let expr = scale_gamma * gamma[j] * witness_partial.norm_x[i][j]
                    + scale_beta * beta[j] * inv_scale_f;
                // y = round(expr / inv_scale)
                let expr_int = field_to_u64(expr);
                let scale = INV_SQRT_SCALE;
                y[i][j] = F::from((expr_int + scale / 2) / scale);
            }
        }

        let witness = LayerNormWitness::from_inputs(x.clone(), y.clone(), d, &lk.t0, &lk.t1);

        let vk = LayerNormVerifyingKey { seq_len: t, d_head: d, gamma, beta, scale_gamma, scale_beta };

        let x_mle = mat_to_mle(&x, t, d);
        let y_mle = mat_to_mle(&y, t, d);
        let (nu_td, _, params_td) = poly_hyrax(&x_mle);
        let io_coms = LayerNormIOCommitments {
            x_com: hyrax_commit(&x_mle.evaluations, nu_td, &params_td),
            y_com: hyrax_commit(&y_mle.evaluations, nu_td, &params_td),
        };

        (witness, io_coms, vk, lk)
    }

    #[test]
    fn test_layernorm_lasso_e2e() {
        let (witness, io_coms, vk, lk) = setup_test_pipeline();
        let lasso_vk = lk.vk();

        let mut pt = Transcript::new(b"layernorm_lasso_test");
        let proof = prove_layernorm(&witness, &io_coms, &vk, &lk, &mut pt).unwrap();

        // Advance transcript to match verifier's 2 finalize calls.
        let _ = pt.challenge_field::<F>(b"hyrax_group_mu");
        let _ = pt.challenge_field::<F>(b"hyrax_group_mu");

        let mut vt = Transcript::new(b"layernorm_lasso_test");
        let mut acc_t = HyraxBatchAccumulator::new();
        let mut acc_td = HyraxBatchAccumulator::new();
        let result =
            verify_layernorm(&proof, &io_coms, &vk, &lasso_vk, &mut vt, &mut acc_t, &mut acc_td);

        if result.is_ok() {
            let n_t = vk.seq_len.next_power_of_two().max(1);
            let n_td = n_t * vk.d_head.next_power_of_two().max(1);
            let (_, _, params_t) = params_from_n(n_t);
            let (_, _, params_td) = params_from_n(n_td);
            acc_t.finalize(&params_t, &mut vt).expect("acc_t finalize");
            acc_td.finalize(&params_td, &mut vt).expect("acc_td finalize");
        }
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }
}
