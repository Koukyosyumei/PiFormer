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
use crate::lookup::range::{RangeProofWitness, RangeWitnessProof};
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_open, hyrax_open_batch,
    poly_hyrax, HyraxBatchAccumulator, HyraxCommitment, HyraxProof,
};
use crate::poly::utils::{combine, eval_rows, mat_to_mle, vec_to_mle};
use crate::poly::DenseMLPoly;
use crate::subprotocols::sumcheck::{
    prove_sumcheck_cubic, prove_sumcheck_cubic_multi_batched, verify_sumcheck_cubic,
    verify_sumcheck_cubic_multi_batched, SumcheckCubicProof, SumcheckCubicProofMulti,
};
use crate::subprotocols::{eq_poly_eval, prove_sumcheck, verify_sumcheck, EvalClaim, SumcheckProof};
use crate::transcript::{challenge_vec, Transcript};
use ark_ff::Field;
use ark_ff::One;
use ark_ff::Zero;

// ---------------------------------------------------------------------------
// Pipeline Interfaces
// ---------------------------------------------------------------------------

pub struct LayerNormIOCommitments {
    pub x_com: HyraxCommitment,
    /// None = GKR backward mode: LN commits y internally and proves external_y_claim.
    pub y_com: Option<HyraxCommitment>,
}

#[derive(Clone)]
pub struct LayerNormVerifyingKey {
    pub seq_len: usize,
    pub d_head: usize,
    pub gamma: Vec<F>,
    pub beta: Vec<F>,
    pub scale_gamma: F,
    pub scale_beta: F,
}

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
    pub sigma_com: HyraxCommitment,
    pub sq_sum_x_com: HyraxCommitment,
    /// Populated when io_coms.y_com is None (GKR backward): LN commits y internally.
    pub y_com: Option<HyraxCommitment>,
}

pub struct LayerNormOpenings {
    pub sum_x_at_rt: F,
    pub sq_sum_x_at_rt: F,
    pub rt_batch_proof: HyraxProof,

    pub x_at_rt_rmean: F,
    pub x_rt_rmean_proof: HyraxProof,

    pub x_at_r_final_q: F,
    pub x_at_r_final_q_proof: HyraxProof,

    pub sq_sum_x_at_rsig: F,
    pub sigma_at_rsig: F,
    pub sigma_sq_at_rsig: F, // bound by sigma_sq_sumcheck
    pub sum_x_sq_at_rsig: F,
    pub rsig_batch_proof: HyraxProof, // opens [sigma_com, sq_sum_x_com] at r_sig_t

    // sigma_sq binding: sigma_at_rf_sigma_sq opened against sigma_com
    pub sigma_at_rf_sigma_sq: F,
    pub sigma_at_rf_sigma_sq_proof: HyraxProof,

    pub x_at_ry: F,
    pub y_at_ry: F,
    pub gamma_x_at_ry: F, // 評価値のみ（コミットなし）
    pub sigma_y_at_ry: F, // 評価値のみ（コミットなし）
    pub ry_td_batch_proof: HyraxProof,

    pub sum_x_at_ryt: F,
    pub sigma_at_ryt: F,
    pub ryt_batch_proof: HyraxProof,

    // 2. gamma_x_sumcheck 用の Opening (点 r_f_gx)
    pub x_at_rf_gx: F,
    pub x_at_rf_gx_proof: HyraxProof,

    // 3. sigma_y_sumcheck 用の Opening (点 r_f_sy)
    pub y_at_rf_sy: F,
    pub y_at_rf_sy_proof: HyraxProof,
    pub sigma_at_rf_sy_t: F,
    pub sigma_at_rf_sy_t_proof: HyraxProof,

    pub sum_x_at_rf_sig: F,
    pub sum_x_at_rf_sig_proof: HyraxProof,

    /// GKR backward: opening of y_com at the external claim point from downstream.
    pub external_y_eval: Option<F>,
    pub external_y_proof: Option<HyraxProof>,
}

pub struct LayerNormProof {
    pub internal_coms: LayerNormInternalCommitments,
    pub mean_sumcheck: SumcheckProof,
    pub sq_sum_sumcheck: SumcheckCubicProof,
    pub sum_x_sq_sumcheck: SumcheckCubicProof,
    /// Proves Σ_i eq(r_sig_t, i) * (d*sigma[i])^2 = sigma_sq_at_rsig, binding sigma_sq to sigma_com.
    pub sigma_sq_sumcheck: SumcheckCubicProof,
    /// Batched cubic sumcheck for gamma*X and sigma*Y (shared eq_y, same challenge vector)
    pub gamma_sigma_sumcheck: SumcheckCubicProofMulti,
    pub sigma_range_proof: RangeWitnessProof,
    pub y_range_proof: RangeWitnessProof,
    pub openings: LayerNormOpenings,
}

// ---------------------------------------------------------------------------
// Helper (eq_poly ベクトル生成)
// ---------------------------------------------------------------------------

fn gen_eq_poly(r: &[F]) -> DenseMLPoly {
    let n = r.len();
    let mut evals = vec![F::one(); 1 << n];
    for j in 0..n {
        let _bit_size = 1 << j;
        for i in 0..(1 << n) {
            if (i >> j) & 1 == 1 {
                evals[i] *= r[n - 1 - j];
            } else {
                evals[i] *= F::one() - r[n - 1 - j];
            }
        }
    }
    DenseMLPoly::new(evals)
}

pub fn vec_to_bits(n: usize, num_bits: usize) -> Vec<F> {
    let mut bits = Vec::with_capacity(num_bits);
    for i in (0..num_bits).rev() {
        if (n >> i) & 1 == 1 {
            bits.push(F::one());
        } else {
            bits.push(F::zero());
        }
    }
    bits
}

// ---------------------------------------------------------------------------
// Range-witness extraction (call before the global range batch)
// ---------------------------------------------------------------------------

/// Intermediate values that need 32-bit range proofs in a LayerNorm.
pub struct LayerNormRangeWitnesses {
    /// Residual pairs for σ (size 2*T): each pair enforces σ ≥ 0.
    pub sigma_witness: RangeProofWitness,
    /// Residual pairs for y (size 2*T*D): each pair enforces y ∈ [0,1].
    pub y_witness: RangeProofWitness,
}

/// Extract sigma_res and y_res from a LayerNorm witness WITHOUT touching
/// the transcript.  Call this before `prove_range_batched` to get the
/// witnesses for the global multiplicity batch.
pub fn compute_range_witnesses(
    witness: &LayerNormWitness,
    vk: &LayerNormVerifyingKey,
) -> LayerNormRangeWitnesses {
    let t = vk.seq_len;
    let d = vk.d_head;
    let d_f = F::from(d as u64);
    let two = F::from(2u64);

    let mut sigma_res = Vec::with_capacity(2 * t);
    for i in 0..t {
        let vi = d_f * (d_f * witness.sq_sum_x[i] - witness.sum_x_sq[i]);
        let dsi = d_f * witness.sigma[i];
        sigma_res.push(vi - dsi * dsi);
        sigma_res.push((dsi + d_f) * (dsi + d_f) - F::ONE - vi);
    }

    let mut y_res = Vec::with_capacity(2 * t * d);
    for i in 0..t {
        let sig_d = witness.sigma[i] * d_f;
        let sum_i = witness.sum_x[i];
        for j in 0..d {
            let expr = vk.scale_gamma * vk.gamma[j] * (d_f * witness.x[i][j] - sum_i)
                + vk.scale_beta * vk.beta[j] * sig_d;
            let y_ij = witness.y[i][j];
            y_res.push(two * expr - sig_d * (two * y_ij - F::ONE));
            y_res.push(sig_d * (two * y_ij + F::ONE) - F::ONE - two * expr);
        }
    }

    LayerNormRangeWitnesses {
        sigma_witness: RangeProofWitness { values: sigma_res },
        y_witness: RangeProofWitness { values: y_res },
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prove a LayerNorm step using pre-supplied range proofs from the global batch.
///
/// `sigma_range = (proof, r_v)` where `r_v` came from `prove_range_batched`.
/// `y_range     = (proof, r_v)` likewise.
/// Prove a LayerNorm step using pre-supplied range proofs from the global batch.
///
/// `external_y_claim`: GKR backward claim from the downstream sub-protocol (e.g. QKV
/// or FFN).  When Some, io_coms.y_com must be None — LN will commit y internally and
/// prove y_MLE(claim.point) == claim.value.  When None, io_coms.y_com must be Some.
pub fn prove_layernorm(
    witness: &LayerNormWitness,
    io_coms: &LayerNormIOCommitments,
    vk: &LayerNormVerifyingKey,
    sigma_range: (RangeWitnessProof, Vec<F>),
    y_range: (RangeWitnessProof, Vec<F>),
    external_y_claim: Option<EvalClaim>,
    transcript: &mut Transcript,
) -> Result<LayerNormProof, String> {
    let t = vk.seq_len;
    let d = vk.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(d as u64);

    let (sigma_range_proof, r_sig) = sigma_range;
    let (y_range_proof, r_y) = y_range;

    let x_mle = mat_to_mle(&witness.x, t, d);
    let y_mle = mat_to_mle(&witness.y, t, d);
    let sum_x_mle = vec_to_mle(&witness.sum_x, t);
    let sq_sum_x_mle = vec_to_mle(&witness.sq_sum_x, t);
    let sum_x_sq_mle = vec_to_mle(&witness.sum_x_sq, t);
    let sigma_mle = vec_to_mle(&witness.sigma, t);
    // d_sigma_mle[i] = d * sigma[i]; used for sigma_sq_sumcheck
    let d_sigma_evals: Vec<F> = witness.sigma.iter().map(|&s| d_f * s).collect();
    let d_sigma_mle = vec_to_mle(&d_sigma_evals, t);
    // sigma_sq_mle[i] = (d * sigma[i])^2; MLE for correct sumcheck claim
    let sigma_sq_mle = vec_to_mle(&witness.sigma_sq_scaled, t);

    let (nu_td, sigma_td, params_td) = poly_hyrax(&x_mle);
    let (nu_t, sigma_t, params_t) = poly_hyrax(&sum_x_mle);

    // Resolve y_com: either trusted external or committed internally (GKR backward).
    let (y_com, y_com_is_internal) = match &io_coms.y_com {
        Some(yc) => (yc.clone(), false),
        None => (hyrax_commit(&y_mle.evaluations, nu_td, &params_td), true),
    };

    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &y_com);

    let sum_x_com = hyrax_commit(&sum_x_mle.evaluations, nu_t, &params_t);
    let sigma_com = hyrax_commit(&sigma_mle.evaluations, nu_t, &params_t);
    let sq_sum_x_com = hyrax_commit(&sq_sum_x_mle.evaluations, nu_t, &params_t);

    absorb_com(transcript, b"sum_x_com", &sum_x_com);
    absorb_com(transcript, b"sigma_com", &sigma_com);
    absorb_com(transcript, b"sq_sum_x_com", &sq_sum_x_com);

    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");
    let claim_s = sum_x_mle.evaluate(&r_t);
    let claim_q = sq_sum_x_mle.evaluate(&r_t);
    transcript.append_field(b"claimed_s", &claim_s);
    transcript.append_field(b"claimed_q", &claim_q);

    let x_collapsed = eval_rows(&x_mle, t_bits, &r_t);
    let f_rows = DenseMLPoly::from_vec_padded(x_collapsed.clone());
    let g_ones = DenseMLPoly::from_vec_padded(vec![F::one(); d]);

    let (mean_sumcheck, r_d_mean) = prove_sumcheck(&f_rows, &g_ones, claim_s, transcript);

    let mut eq_t_evals = vec![F::zero(); 1 << (t_bits + d_bits)];
    for i in 0..t {
        let eq_i = eq_poly_eval(&vec_to_bits(i, t_bits), &r_t); // zをビットに変換して評価
        for j in 0..d {
            eq_t_evals[i << d_bits | j] = eq_i;
        }
    }
    let eq_t_ext = DenseMLPoly::new(eq_t_evals);
    let (sq_sum_sumcheck, r_final_q) =
        prove_sumcheck_cubic(&eq_t_ext, &x_mle, &x_mle, claim_q, transcript);

    // r_sig and r_y come from the pre-computed global range batch (already in transcript)
    let r_sig_t = r_sig[0..t_bits].to_vec();

    // Binding: sum_x_sq(r_sig_t) -> (sum_x)^2
    let eq_sig = gen_eq_poly(&r_sig_t);
    let claim_x_sq = sum_x_sq_mle.evaluate(&r_sig_t);
    let (sum_x_sq_sumcheck, r_f_sig) =
        prove_sumcheck_cubic(&eq_sig, &sum_x_mle, &sum_x_mle, claim_x_sq, transcript);

    // Binding: sigma_sq(r_sig_t) = Σ_i eq(r_sig_t,i)*(d*sigma[i])^2  [binds to sigma_com]
    let claim_sigma_sq = sigma_sq_mle.evaluate(&r_sig_t);
    let (sigma_sq_sumcheck, r_f_sigma_sq) =
        prove_sumcheck_cubic(&eq_sig, &d_sigma_mle, &d_sigma_mle, claim_sigma_sq, transcript);

    let r_y_t = r_y[0..t_bits].to_vec();
    let r_y_d = r_y[t_bits..t_bits + d_bits].to_vec();
    let ry_td = combine(&r_y_t, &r_y_d);

    // [Binding] gamma_x = gamma * X  and  sigma_y = sigma * Y  (batched)
    let eq_y = gen_eq_poly(&ry_td);
    let mut gamma_x_evals = vec![F::zero(); 1 << (t_bits + d_bits)];
    for i in 0..t {
        for j in 0..d {
            gamma_x_evals[i << d_bits | j] = vk.gamma[j] * witness.x[i][j];
        }
    }
    let gamma_x_mle = DenseMLPoly::new(gamma_x_evals);
    let g_x_eval = gamma_x_mle.evaluate(&ry_td);
    let gamma_ext = DenseMLPoly::new(
        (0..(1 << (t_bits + d_bits)))
            .map(|i| vk.gamma[i & ((1 << d_bits) - 1)])
            .collect(),
    );

    let mut sigma_y_evals = vec![F::zero(); 1 << (t_bits + d_bits)];
    for i in 0..t {
        for j in 0..d {
            sigma_y_evals[i << d_bits | j] = witness.sigma[i] * witness.y[i][j];
        }
    }
    let sigma_y_mle = DenseMLPoly::new(sigma_y_evals);
    let s_y_eval = sigma_y_mle.evaluate(&ry_td);
    let sigma_ext = DenseMLPoly::new(
        (0..(1 << (t_bits + d_bits)))
            .map(|i| witness.sigma[i >> d_bits])
            .collect(),
    );

    let lambda = transcript.challenge_field::<F>(b"gamma_sigma_batch_lambda");
    let claim_gamma_sigma = g_x_eval + lambda * s_y_eval;
    let (gamma_sigma_sumcheck, r_f) = prove_sumcheck_cubic_multi_batched(
        &[eq_y.clone(), eq_y],
        &[gamma_ext, sigma_ext],
        &[x_mle.clone(), y_mle.clone()],
        &[F::one(), lambda],
        claim_gamma_sigma,
        transcript,
    );
    // Both gamma_x and sigma_y now share the same challenge vector r_f
    let r_f_gx = r_f.clone();
    let r_f_sy = r_f;

    let x_at_rf_gx = x_mle.evaluate(&r_f_gx);
    let x_at_rf_gx_proof = hyrax_open(&x_mle.evaluations, &r_f_gx, nu_td, sigma_td);

    let y_at_rf_sy = y_mle.evaluate(&r_f_sy);
    let y_at_rf_sy_proof = hyrax_open(&y_mle.evaluations, &r_f_sy, nu_td, sigma_td);

    // sigma は t_bits しかないので r_f_sy の前半部分で Opening
    let r_f_sy_t = &r_f_sy[0..t_bits];
    let sigma_at_rf_sy_t = sigma_mle.evaluate(r_f_sy_t);
    let sigma_at_rf_sy_t_proof = hyrax_open(&sigma_mle.evaluations, r_f_sy_t, nu_t, sigma_t);

    // GKR backward: open y at the external claim point from the downstream sub-protocol.
    let (external_y_eval, external_y_proof) = match external_y_claim {
        Some(ref claim) => {
            let ev = y_mle.evaluate(&claim.point);
            let pf = hyrax_open(&y_mle.evaluations, &claim.point, nu_td, sigma_td);
            (Some(ev), Some(pf))
        }
        None => (None, None),
    };

    Ok(LayerNormProof {
        internal_coms: LayerNormInternalCommitments {
            sum_x_com,
            sigma_com,
            sq_sum_x_com,
            y_com: if y_com_is_internal { Some(y_com.clone()) } else { None },
        },
        mean_sumcheck,
        sq_sum_sumcheck,
        sum_x_sq_sumcheck,
        sigma_sq_sumcheck,
        gamma_sigma_sumcheck,
        sigma_range_proof,
        y_range_proof,
        openings: LayerNormOpenings {
            sum_x_at_rt: claim_s,
            sq_sum_x_at_rt: claim_q,
            rt_batch_proof: hyrax_open_batch(
                &[&sum_x_mle.evaluations],
                &r_t,
                nu_t,
                sigma_t,
                transcript,
            ),
            x_at_rt_rmean: x_mle.evaluate(&combine(&r_t, &r_d_mean)),
            x_rt_rmean_proof: hyrax_open(
                &x_mle.evaluations,
                &combine(&r_t, &r_d_mean),
                nu_td,
                sigma_td,
            ),
            sq_sum_x_at_rsig: sq_sum_x_mle.evaluate(&r_sig_t),
            sigma_at_rsig: sigma_mle.evaluate(&r_sig_t),
            sigma_sq_at_rsig: claim_sigma_sq,
            sum_x_sq_at_rsig: claim_x_sq,
            rsig_batch_proof: hyrax_open_batch(
                &[&sigma_mle.evaluations, &sq_sum_x_mle.evaluations],
                &r_sig_t,
                nu_t,
                sigma_t,
                transcript,
            ),
            x_at_ry: x_mle.evaluate(&ry_td),
            y_at_ry: y_mle.evaluate(&ry_td),
            ry_td_batch_proof: hyrax_open_batch(
                &[&x_mle.evaluations, &y_mle.evaluations],
                &ry_td,
                nu_td,
                sigma_td,
                transcript,
            ),
            gamma_x_at_ry: g_x_eval,
            sigma_y_at_ry: s_y_eval,
            sum_x_at_ryt: sum_x_mle.evaluate(&r_y_t),
            sigma_at_ryt: sigma_mle.evaluate(&r_y_t),
            ryt_batch_proof: hyrax_open_batch(
                &[&sum_x_mle.evaluations, &sigma_mle.evaluations],
                &r_y_t,
                nu_t,
                sigma_t,
                transcript,
            ),
            x_at_r_final_q: x_mle.evaluate(&r_final_q),
            x_at_r_final_q_proof: hyrax_open(&x_mle.evaluations, &r_final_q, nu_td, sigma_td),
            x_at_rf_gx,
            x_at_rf_gx_proof,
            y_at_rf_sy,
            y_at_rf_sy_proof,
            sigma_at_rf_sy_t,
            sigma_at_rf_sy_t_proof,
            sum_x_at_rf_sig: sum_x_mle.evaluate(&r_f_sig),
            sum_x_at_rf_sig_proof: hyrax_open(&sum_x_mle.evaluations, &r_f_sig, nu_t, sigma_t),
            // sigma_sq binding: open sigma_com at r_f_sigma_sq
            sigma_at_rf_sigma_sq: sigma_mle.evaluate(&r_f_sigma_sq),
            sigma_at_rf_sigma_sq_proof: hyrax_open(
                &sigma_mle.evaluations,
                &r_f_sigma_sq,
                nu_t,
                sigma_t,
            ),
            external_y_eval,
            external_y_proof,
        },
    })
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify a LayerNorm step using pre-supplied range-proof evaluation points.
///
/// `sigma_r_v` and `y_r_v` must be the `r_v` values returned by the
/// matching `prove_range_batched` call (verified separately via
/// `verify_range_batched`).
///
/// `external_y_claim`: when Some, io_coms.y_com must be None.  The verifier
/// checks that the internally-committed y_com (from the proof) opens to
/// external_y_claim.value at external_y_claim.point.
pub fn verify_layernorm(
    proof: &LayerNormProof,
    io_coms: &LayerNormIOCommitments,
    vk: &LayerNormVerifyingKey,
    sigma_r_v: &[F],
    y_r_v: &[F],
    external_y_claim: Option<&EvalClaim>,
    transcript: &mut Transcript,
    acc_t: &mut HyraxBatchAccumulator,
    acc_td: &mut HyraxBatchAccumulator,
) -> Result<(), String> {
    let t_bits = vk.seq_len.next_power_of_two().trailing_zeros() as usize;
    let d_bits = vk.d_head.next_power_of_two().trailing_zeros() as usize;
    let d_f = F::from(vk.d_head as u64);

    // Resolve y_com: either trusted external IO or internally committed (GKR backward).
    let y_com = match &io_coms.y_com {
        Some(yc) => yc.clone(),
        None => proof.internal_coms.y_com.as_ref()
            .ok_or_else(|| "GKR backward: internal y_com missing from proof".to_string())?
            .clone(),
    };

    // 1. Absorb IO & Internal Commitments
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &y_com);
    absorb_com(transcript, b"sum_x_com", &proof.internal_coms.sum_x_com);
    absorb_com(transcript, b"sigma_com", &proof.internal_coms.sigma_com);
    absorb_com(transcript, b"sq_sum_x_com", &proof.internal_coms.sq_sum_x_com);

    // 2. Row Audit Challenge
    let r_t = challenge_vec(transcript, t_bits, b"layernorm_rt");
    transcript.append_field(b"claimed_s", &proof.openings.sum_x_at_rt);
    transcript.append_field(b"claimed_q", &proof.openings.sq_sum_x_at_rt);

    // -----------------------------------------------------------------------
    // 3. Sumchecks & Bindings
    // -----------------------------------------------------------------------

    // A. Mean Sumcheck Binding
    let (r_d_mean, f_m) = verify_sumcheck(
        &proof.mean_sumcheck,
        proof.openings.sum_x_at_rt,
        d_bits,
        transcript,
    )?;
    if f_m != proof.openings.x_at_rt_rmean {
        return Err("Binding of mean to X failed".into());
    }

    // B. sq_sum_x (Σx^2) Binding (Cubic)
    let (r_final_q, f_q) = verify_sumcheck_cubic(
        &proof.sq_sum_sumcheck,
        proof.openings.sq_sum_x_at_rt,
        t_bits + d_bits,
        transcript,
    )?;
    let eq_val_q = eq_poly_eval(&r_final_q[0..t_bits], &r_t);
    if f_q != eq_val_q * proof.openings.x_at_r_final_q * proof.openings.x_at_r_final_q {
        return Err("sq_sum_sumcheck binding failed".into());
    }

    // C. Sigma Range: use pre-supplied evaluation point from global range batch.
    //    The range proof itself is verified via verify_range_batched before this call.
    //    V(r_sig) = Σ_c chunk_eval[c] * 2^(16c) (verified by the fusion check in verify_range_batched).
    let r_sig = sigma_r_v;
    let sig_eval: F = {
        let mut ev = F::ZERO;
        let mut shift = F::ONE;
        let shift_mult = F::from(1u64 << crate::lookup::range::CHUNK_BITS);
        for &ce in &proof.sigma_range_proof.chunk_evals {
            ev += ce * shift;
            shift *= shift_mult;
        }
        ev
    };
    let r_sig_t = r_sig[0..t_bits].to_vec();

    let (r_f_sig, f_sq) = verify_sumcheck_cubic(
        &proof.sum_x_sq_sumcheck,
        proof.openings.sum_x_sq_at_rsig,
        t_bits,
        transcript,
    )?;
    let eq_sig_eval = eq_poly_eval(&r_f_sig, &r_sig_t);
    if f_sq != eq_sig_eval * proof.openings.sum_x_at_rf_sig * proof.openings.sum_x_at_rf_sig {
        return Err("sum_x_sq binding failed".into());
    }

    // sigma_sq_sumcheck: proves Σ eq(r_sig_t, i) * (d*sigma[i])^2 = sigma_sq_at_rsig
    let (r_f_sigma_sq, f_sigma_sq) = verify_sumcheck_cubic(
        &proof.sigma_sq_sumcheck,
        proof.openings.sigma_sq_at_rsig,
        t_bits,
        transcript,
    )?;
    let eq_sigma_sq_eval = eq_poly_eval(&r_f_sigma_sq, &r_sig_t);
    let d_sigma_rf = d_f * proof.openings.sigma_at_rf_sigma_sq;
    if f_sigma_sq != eq_sigma_sq_eval * d_sigma_rf * d_sigma_rf {
        return Err("sigma_sq sumcheck binding failed".into());
    }

    // D. Y Range: use pre-supplied evaluation point from global range batch.
    let r_y = y_r_v;
    let y_eval: F = {
        let mut ev = F::ZERO;
        let mut shift = F::ONE;
        let shift_mult = F::from(1u64 << crate::lookup::range::CHUNK_BITS);
        for &ce in &proof.y_range_proof.chunk_evals {
            ev += ce * shift;
            shift *= shift_mult;
        }
        ev
    };
    let r_y_t = r_y[0..t_bits].to_vec();
    let r_y_d = r_y[t_bits..t_bits + d_bits].to_vec();
    let ry_td = combine(&r_y_t, &r_y_d);

    // gamma * X  and  sigma * Y  Binding (batched)
    let lambda = transcript.challenge_field::<F>(b"gamma_sigma_batch_lambda");
    let claim_gamma_sigma =
        proof.openings.gamma_x_at_ry + lambda * proof.openings.sigma_y_at_ry;
    let (r_f, _) = verify_sumcheck_cubic_multi_batched(
        &proof.gamma_sigma_sumcheck,
        &[F::one(), lambda],
        claim_gamma_sigma,
        t_bits + d_bits,
        transcript,
    )?;
    let eq_val = eq_poly_eval(&r_f, &ry_td);
    let gamma_eval_at_f = vec_to_mle(&vk.gamma, vk.d_head).evaluate(&r_f[t_bits..]);
    // gamma_x final check
    if proof.gamma_sigma_sumcheck.final_evals_f[0]
        * proof.gamma_sigma_sumcheck.final_evals_g[0]
        * proof.gamma_sigma_sumcheck.final_evals_h[0]
        != eq_val * gamma_eval_at_f * proof.openings.x_at_rf_gx
    {
        return Err("gamma_x binding failed".into());
    }
    // sigma_y final check
    if proof.gamma_sigma_sumcheck.final_evals_f[1]
        * proof.gamma_sigma_sumcheck.final_evals_g[1]
        * proof.gamma_sigma_sumcheck.final_evals_h[1]
        != eq_val * proof.openings.sigma_at_rf_sy_t * proof.openings.y_at_rf_sy
    {
        return Err("sigma_y binding failed".into());
    }
    let r_f_gx = r_f.clone();
    let r_f_sy = r_f;

    // -----------------------------------------------------------------------
    // 4. Final Fusion Checks (Algebraic Consistency)
    // -----------------------------------------------------------------------

    // --- Sigma Fusion Check ---
    let v_ev = d_f * (d_f * proof.openings.sq_sum_x_at_rsig - proof.openings.sum_x_sq_at_rsig);
    // sigma_sq_at_rsig is bound by sigma_sq_sumcheck above
    let z_ev = proof.openings.sigma_sq_at_rsig;
    let dsi = d_f * proof.openings.sigma_at_rsig;

    let lo_sig = v_ev - z_ev;
    let hi_sig = z_ev + F::from(2u64) * d_f * dsi + d_f * d_f - F::one() - v_ev;

    let r_sig_b = r_sig[t_bits];
    if sig_eval != (F::one() - r_sig_b) * lo_sig + r_sig_b * hi_sig {
        return Err("Sigma fusion check failed".into());
    }

    // --- Y Fusion Check ---
    let gamma_r = vec_to_mle(&vk.gamma, vk.d_head).evaluate(&r_y_d);
    let beta_r = vec_to_mle(&vk.beta, vk.d_head).evaluate(&r_y_d);
    let sig_d = proof.openings.sigma_at_ryt * d_f;

    let expr = vk.scale_gamma
        * (d_f * proof.openings.gamma_x_at_ry - gamma_r * proof.openings.sum_x_at_ryt)
        + vk.scale_beta * beta_r * sig_d;
    let two_expr = F::from(2u64) * expr;
    let sigma_d_y = d_f * proof.openings.sigma_y_at_ry;

    let lo_y = two_expr - (F::from(2u64) * sigma_d_y - sig_d);
    let hi_y = (F::from(2u64) * sigma_d_y + sig_d) - F::one() - two_expr;

    let r_y_b = r_y[t_bits + d_bits];
    if y_eval != (F::one() - r_y_b) * lo_y + r_y_b * hi_y {
        return Err("Y fusion check failed".into());
    }

    // -----------------------------------------------------------------------
    // 5. Batched Opening Verifications
    // -----------------------------------------------------------------------

    // m_com verified globally via verify_range_batched before this call.

    // 1. rt_batch_proof (Group 1)
    acc_t.add_verify_batch(
        &[proof.internal_coms.sum_x_com.clone()],
        &[proof.openings.sum_x_at_rt],
        &r_t,
        &proof.openings.rt_batch_proof,
        transcript,
    )?;

    // 2. x_rt_rmean_proof (Individual - No eta consumed)
    acc_td.add_verify(
        &io_coms.x_com,
        proof.openings.x_at_rt_rmean,
        &combine(&r_t, &r_d_mean),
        &proof.openings.x_rt_rmean_proof,
    )?;

    // 3. rsig_batch_proof (Group 2): opens sigma_com and sq_sum_x_com at r_sig_t
    acc_t.add_verify_batch(
        &[
            proof.internal_coms.sigma_com.clone(),
            proof.internal_coms.sq_sum_x_com.clone(),
        ],
        &[proof.openings.sigma_at_rsig, proof.openings.sq_sum_x_at_rsig],
        &r_sig_t,
        &proof.openings.rsig_batch_proof,
        transcript,
    )?;

    // 4. ry_td_batch_proof (Group 3)
    acc_td.add_verify_batch(
        &[io_coms.x_com.clone(), y_com.clone()],
        &[proof.openings.x_at_ry, proof.openings.y_at_ry],
        &ry_td,
        &proof.openings.ry_td_batch_proof,
        transcript,
    )?;

    // 5. ryt_batch_proof (Group 4)
    acc_t.add_verify_batch(
        &[
            proof.internal_coms.sum_x_com.clone(),
            proof.internal_coms.sigma_com.clone(),
        ],
        &[proof.openings.sum_x_at_ryt, proof.openings.sigma_at_ryt],
        &r_y_t,
        &proof.openings.ryt_batch_proof,
        transcript,
    )?;

    // --- 以降、Individual Openings (Prover の 275行目以降と同期) ---

    acc_td.add_verify(
        &io_coms.x_com,
        proof.openings.x_at_r_final_q,
        &r_final_q, // verify_sumcheck_cubic で得られた点
        &proof.openings.x_at_r_final_q_proof,
    )?;

    acc_td.add_verify(
        &io_coms.x_com,
        proof.openings.x_at_rf_gx,
        &r_f_gx,
        &proof.openings.x_at_rf_gx_proof,
    )?;

    acc_td.add_verify(
        &y_com,
        proof.openings.y_at_rf_sy,
        &r_f_sy,
        &proof.openings.y_at_rf_sy_proof,
    )?;

    acc_t.add_verify(
        &proof.internal_coms.sigma_com,
        proof.openings.sigma_at_rf_sy_t,
        &r_f_sy[0..t_bits],
        &proof.openings.sigma_at_rf_sy_t_proof,
    )?;

    acc_t.add_verify(
        &proof.internal_coms.sum_x_com,
        proof.openings.sum_x_at_rf_sig,
        &r_f_sig,
        &proof.openings.sum_x_at_rf_sig_proof,
    )?;

    // sigma_sq_sumcheck final binding: open sigma_com at r_f_sigma_sq
    acc_t.add_verify(
        &proof.internal_coms.sigma_com,
        proof.openings.sigma_at_rf_sigma_sq,
        &r_f_sigma_sq,
        &proof.openings.sigma_at_rf_sigma_sq_proof,
    )?;

    // GKR backward: verify opening of y_com at the external claim point.
    if let Some(claim) = external_y_claim {
        let ext_eval = proof.openings.external_y_eval
            .ok_or("GKR backward: external_y_eval missing from proof")?;
        let ext_proof = proof.openings.external_y_proof.as_ref()
            .ok_or("GKR backward: external_y_proof missing from proof")?;
        if ext_eval != claim.value {
            return Err(format!(
                "GKR backward: external y eval mismatch: prover says {}, claim says {}",
                ext_eval, claim.value
            ));
        }
        acc_td.add_verify(&y_com, ext_eval, &claim.point, ext_proof)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod layernorm_tests {
    use super::*;
    use ark_ff::{One, Zero};
    use crate::lookup::range::{prove_range_batched, verify_range_batched, GlobalRangeM};
    use crate::pcs::{params_from_n, params_from_vars};

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
            y_com: Some(hyrax_commit(&y_mle.evaluations, nu_td, &params_td)),
        };
        (witness, io_coms, vk)
    }

    /// Helper: run the full prove + verify cycle for a single LayerNorm call,
    /// including the global range batch phase.
    fn prove_verify_ln(
        witness: &LayerNormWitness,
        io_coms: &LayerNormIOCommitments,
        vk: &LayerNormVerifyingKey,
    ) -> (LayerNormProof, GlobalRangeM, Vec<Vec<F>>) {
        let rw = compute_range_witnesses(witness, vk);
        let t = vk.seq_len;
        let d = vk.d_head;
        let sigma_n_vars = (2 * t).next_power_of_two().trailing_zeros() as usize;
        let y_n_vars = (2 * t * d).next_power_of_two().trailing_zeros() as usize;

        let mut pt = Transcript::new(b"layernorm_test");
        let (mut range_proofs, global_m, r_vs) = prove_range_batched(
            &[&rw.sigma_witness, &rw.y_witness],
            32,
            &mut pt,
        )
        .unwrap();
        let y_rp = range_proofs.remove(1);
        let sigma_rp = range_proofs.remove(0);
        let proof = prove_layernorm(
            witness,
            io_coms,
            vk,
            (sigma_rp, r_vs[0].clone()),
            (y_rp, r_vs[1].clone()),
            None,
            &mut pt,
        )
        .unwrap();
        let _ = pt.challenge_field::<F>(b"hyrax_group_mu"); // ln_acc_t
        let _ = pt.challenge_field::<F>(b"hyrax_group_mu"); // ln_acc_td
        let _ = pt.challenge_field::<F>(b"hyrax_group_mu"); // acc_range_sig
        let _ = pt.challenge_field::<F>(b"hyrax_group_mu"); // acc_range_y
        let _ = pt.challenge_field::<F>(b"hyrax_group_mu"); // acc_range_m

        // Verifier side
        let mut vt = Transcript::new(b"layernorm_test");
        let sigma_rp_ref = &proof.sigma_range_proof;
        let y_rp_ref = &proof.y_range_proof;
        let mut acc_range_sig = HyraxBatchAccumulator::new();
        let mut acc_range_y   = HyraxBatchAccumulator::new();
        let mut acc_range_m   = HyraxBatchAccumulator::new();
        let (rv_v, _r_m_v) = verify_range_batched(
            &[sigma_rp_ref, y_rp_ref],
            &global_m,
            &[sigma_n_vars, y_n_vars],
            32,
            &mut vt,
            &mut acc_range_sig,
            &mut acc_range_y,
            &mut acc_range_m,
        )
        .unwrap();

        let mut ln_acc_t = HyraxBatchAccumulator::new();
        let mut ln_acc_td = HyraxBatchAccumulator::new();
        let result = verify_layernorm(
            &proof,
            io_coms,
            vk,
            &rv_v[0],
            &rv_v[1],
            None,
            &mut vt,
            &mut ln_acc_t,
            &mut ln_acc_td,
        );
        if result.is_ok() {
            let n_t = vk.seq_len.next_power_of_two().max(1);
            let n_td = n_t * vk.d_head.next_power_of_two().max(1);
            let (_, _, params_t) = params_from_n(n_t);
            let (_, _, params_td) = params_from_n(n_td);
            let (_, _, params_rsig) = params_from_vars(sigma_n_vars);
            let (_, _, params_ry)   = params_from_vars(y_n_vars);
            let (_, _, params_rm)   = params_from_vars(crate::lookup::range::CHUNK_BITS);
            ln_acc_t.finalize(&params_t, &mut vt).unwrap();
            ln_acc_td.finalize(&params_td, &mut vt).unwrap();
            acc_range_sig.finalize(&params_rsig, &mut vt).unwrap();
            acc_range_y.finalize(&params_ry, &mut vt).unwrap();
            acc_range_m.finalize(&params_rm, &mut vt).unwrap();
        }
        result.expect("verify_layernorm failed");
        (proof, global_m, r_vs)
    }

    #[test]
    fn test_layernorm_succinct_e2e() {
        let (witness, io_coms, vk) = setup_test_pipeline();
        prove_verify_ln(&witness, &io_coms, &vk);
    }

    #[test]
    fn test_rejects_tampered_io_x() {
        let (mut witness, io_coms, vk) = setup_test_pipeline();
        witness.x[0][0] += F::one(); // Tamper locally
        // prove_layernorm with corrupted witness produces a proof; verifier must reject
        let rw = compute_range_witnesses(&witness, &vk);
        let t = vk.seq_len;
        let d = vk.d_head;
        let sigma_n_vars = (2 * t).next_power_of_two().trailing_zeros() as usize;
        let y_n_vars = (2 * t * d).next_power_of_two().trailing_zeros() as usize;
        let mut pt = Transcript::new(b"layernorm_test");
        let (mut range_proofs, global_m, r_vs) =
            prove_range_batched(&[&rw.sigma_witness, &rw.y_witness], 32, &mut pt).unwrap();
        let y_rp = range_proofs.remove(1);
        let sigma_rp = range_proofs.remove(0);
        if let Ok(proof) = prove_layernorm(
            &witness,
            &io_coms,
            &vk,
            (sigma_rp, r_vs[0].clone()),
            (y_rp, r_vs[1].clone()),
            None,
            &mut pt,
        ) {
            let _ = pt.challenge_field::<F>(b"hyrax_group_mu"); // ln_acc_t
            let _ = pt.challenge_field::<F>(b"hyrax_group_mu"); // ln_acc_td
            let _ = pt.challenge_field::<F>(b"hyrax_group_mu"); // acc_range_sig
            let _ = pt.challenge_field::<F>(b"hyrax_group_mu"); // acc_range_y
            let _ = pt.challenge_field::<F>(b"hyrax_group_mu"); // acc_range_m
            let mut vt = Transcript::new(b"layernorm_test");
            let mut acc_range_sig = HyraxBatchAccumulator::new();
            let mut acc_range_y   = HyraxBatchAccumulator::new();
            let mut acc_range_m   = HyraxBatchAccumulator::new();
            let (rv_v, _) = verify_range_batched(
                &[&proof.sigma_range_proof, &proof.y_range_proof],
                &global_m,
                &[sigma_n_vars, y_n_vars],
                32,
                &mut vt,
                &mut acc_range_sig,
                &mut acc_range_y,
                &mut acc_range_m,
            )
            .unwrap();
            let mut ln_acc_t = HyraxBatchAccumulator::new();
            let mut ln_acc_td = HyraxBatchAccumulator::new();
            let result = verify_layernorm(
                &proof,
                &io_coms,
                &vk,
                &rv_v[0],
                &rv_v[1],
                None,
                &mut vt,
                &mut ln_acc_t,
                &mut ln_acc_td,
            );
            if result.is_ok() {
                let n_t = vk.seq_len.next_power_of_two().max(1);
                let n_td = n_t * vk.d_head.next_power_of_two().max(1);
                let (_, _, params_t) = params_from_n(n_t);
                let (_, _, params_td) = params_from_n(n_td);
                let (_, _, params_rsig) = params_from_vars(sigma_n_vars);
                let (_, _, params_ry)   = params_from_vars(y_n_vars);
                let (_, _, params_rm)   = params_from_vars(crate::lookup::range::CHUNK_BITS);
                let _ = ln_acc_t.finalize(&params_t, &mut vt);
                let _ = ln_acc_td.finalize(&params_td, &mut vt);
                let _ = acc_range_sig.finalize(&params_rsig, &mut vt);
                let _ = acc_range_y.finalize(&params_ry, &mut vt);
                let _ = acc_range_m.finalize(&params_rm, &mut vt);
            }
            assert!(
                result.is_err(),
                "Should reject forged proof against trusted IO"
            );
        }
    }
}
