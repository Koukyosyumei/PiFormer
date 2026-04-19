//! Cross-Layer Sumcheck Batching for Projection Layers
//!
//! Instead of L independent projection sumchecks (each log(D_in) rounds),
//! this module proves all L layers with ONE cubic sumcheck over (l, k) variables
//! (log L + log D_in rounds total).
//!
//! **Protocol:**
//!   1. Commit X_all[l,t,k] and Y_all[l,t,j] (dynamic, per-inference)
//!   2. Verifier derives r_l, r_t, r_out from transcript
//!   3. Build cubic sumcheck polynomials over (b, c) ∈ {0,1}^{l_bits+k_bits}:
//!        f(b,c) = eq(r_l, b)          — equality selector (verifier-computable)
//!        g(b,c) = X_all(b, r_t, c)   — activations at fixed r_t
//!        h(b,c) = W_all(b, c, r_out) — weights at fixed r_out (alpha-scaled)
//!   4. Claim: Σ_{b,c} f·g·h = Y_all(r_l,r_t,r_out) - bias_all(r_l,r_out)
//!   5. After sumcheck → challenges (r_b, r_c); open X_all, W_all via PCS

use crate::field::F;
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_open, params_from_vars, HyraxBatchAccumulator,
    HyraxCommitment, HyraxProof,
};
use crate::poly::dense::DenseMLPoly;
use crate::poly::utils::{eval_rows, mat_to_mle};
use crate::subprotocols::sumcheck::{prove_sumcheck_cubic, verify_sumcheck_cubic, SumcheckCubicProof};
use crate::subprotocols::EvalClaim;
use crate::transcript::{challenge_vec, Transcript};
use crate::poly::utils::TernaryValue;
use ark_ff::Field;

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

/// Static verifying key for cross-layer projection (preprocessed from weights).
pub struct CrossLayerProjectionVK {
    pub num_layers: usize,
    pub seq_len: usize,
    pub d_in: usize,
    pub d_out: usize,
    /// Committed cross-layer weight MLE: L × D_in × D_out (alpha-scaled).
    pub w_all_com: HyraxCommitment,
    /// Committed cross-layer bias MLE: L × D_out.
    pub bias_all_com: HyraxCommitment,
}

/// Full proving key (includes raw weight evaluations for building h_poly).
pub struct CrossLayerProjectionPK {
    pub vk: CrossLayerProjectionVK,
    /// Flat MLE evaluations for W_all (l_p2 × k_p2 × j_p2).
    pub w_all_evals: Vec<F>,
    /// Flat MLE evaluations for bias_all (l_p2 × j_p2).
    pub bias_all_evals: Vec<F>,
}

/// Preprocess: commit the cross-layer weight tensor W_all[l, k, j] = alpha_l * W_l[k, j].
pub fn preprocess_cross_layer_projection(
    num_layers: usize,
    seq_len: usize,
    d_in: usize,
    d_out: usize,
    ws: &[Vec<Vec<TernaryValue>>],
    alphas: &[F],
    biases: &[Vec<F>],
) -> CrossLayerProjectionPK {
    let l_p2 = num_layers.next_power_of_two().max(1);
    let k_p2 = d_in.next_power_of_two().max(1);
    let j_p2 = d_out.next_power_of_two().max(1);

    let l_bits = l_p2.trailing_zeros() as usize;
    let k_bits = k_p2.trailing_zeros() as usize;
    let j_bits = j_p2.trailing_zeros() as usize;

    // W_all layout: [l, k, j] = l*k_p2*j_p2 + k*j_p2 + j
    let mut w_all_evals = vec![F::ZERO; l_p2 * k_p2 * j_p2];
    for li in 0..num_layers {
        let alpha = alphas[li];
        for ki in 0..d_in {
            for ji in 0..d_out {
                let w_val = match ws[li][ki][ji] {
                    TernaryValue::ONE => F::ONE,
                    TernaryValue::MINUSONE => -F::ONE,
                    TernaryValue::ZERO => F::ZERO,
                };
                w_all_evals[li * k_p2 * j_p2 + ki * j_p2 + ji] = alpha * w_val;
            }
        }
    }

    // bias_all layout: [l, j] = l*j_p2 + j
    let mut bias_all_evals = vec![F::ZERO; l_p2 * j_p2];
    for li in 0..num_layers {
        for ji in 0..d_out {
            bias_all_evals[li * j_p2 + ji] = biases[li][ji];
        }
    }

    let w_num_vars = l_bits + k_bits + j_bits;
    let (w_nu, _, w_params) = params_from_vars(w_num_vars);
    let w_all_com = hyrax_commit(&w_all_evals, w_nu, &w_params);

    let bias_num_vars = (l_bits + j_bits).max(1);
    let (b_nu, _, b_params) = params_from_vars(bias_num_vars);
    let bias_all_com = hyrax_commit(&bias_all_evals, b_nu, &b_params);

    CrossLayerProjectionPK {
        vk: CrossLayerProjectionVK {
            num_layers,
            seq_len,
            d_in,
            d_out,
            w_all_com,
            bias_all_com,
        },
        w_all_evals,
        bias_all_evals,
    }
}

// ---------------------------------------------------------------------------
// Proof struct
// ---------------------------------------------------------------------------

pub struct CrossLayerProjectionProof {
    /// Commitment to X_all: L × T × D_in activation MLE.
    pub x_all_com: HyraxCommitment,
    /// Commitment to Y_all: L × T × D_out output MLE.
    pub y_all_com: HyraxCommitment,
    /// Cubic sumcheck over (l_bits + k_bits) variables.
    pub sumcheck: SumcheckCubicProof,
    /// Prover's claimed PCS evaluations.
    pub x_all_eval: F,    // X_all(r_b, r_t, r_c)
    pub w_all_eval: F,    // W_all(r_b, r_c, r_out)
    pub y_all_eval: F,    // Y_all(r_l, r_t, r_out) — for the sumcheck claim
    pub bias_all_eval: F, // bias_all(r_l, r_out)
    /// PCS opening proofs.
    pub x_all_open: HyraxProof,
    pub w_all_open: HyraxProof,
    pub y_all_open: HyraxProof,
    pub bias_all_open: HyraxProof,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prove Y_l = alpha_l * X_l @ W_l + bias_l for all l ∈ [L] simultaneously.
///
/// Returns the proof and an `EvalClaim` for Y_all at (r_l ‖ r_t ‖ r_out).
pub fn prove_cross_layer_projection(
    xs: &[Vec<Vec<F>>],  // xs[l]: T × D_in activation matrix
    ys: &[Vec<Vec<F>>],  // ys[l]: T × D_out output matrix
    pk: &CrossLayerProjectionPK,
    transcript: &mut Transcript,
) -> Result<(CrossLayerProjectionProof, EvalClaim), String> {
    let num_l = pk.vk.num_layers;
    let t = pk.vk.seq_len;
    let d_in = pk.vk.d_in;
    let d_out = pk.vk.d_out;

    let l_p2 = num_l.next_power_of_two().max(1);
    let t_p2 = t.next_power_of_two().max(1);
    let k_p2 = d_in.next_power_of_two().max(1);
    let j_p2 = d_out.next_power_of_two().max(1);

    let l_bits = l_p2.trailing_zeros() as usize;
    let t_bits = t_p2.trailing_zeros() as usize;
    let k_bits = k_p2.trailing_zeros() as usize;
    let j_bits = j_p2.trailing_zeros() as usize;

    // --- Build X_all and Y_all MLE evaluations ---
    let mut x_all_evals = vec![F::ZERO; l_p2 * t_p2 * k_p2];
    for li in 0..num_l {
        for ti in 0..t {
            for ki in 0..d_in {
                x_all_evals[li * t_p2 * k_p2 + ti * k_p2 + ki] = xs[li][ti][ki];
            }
        }
    }

    let mut y_all_evals = vec![F::ZERO; l_p2 * t_p2 * j_p2];
    for li in 0..num_l {
        for ti in 0..t {
            for ji in 0..d_out {
                y_all_evals[li * t_p2 * j_p2 + ti * j_p2 + ji] = ys[li][ti][ji];
            }
        }
    }

    // --- Commit X_all and Y_all ---
    let x_num_vars = l_bits + t_bits + k_bits;
    let (x_nu, x_sigma, x_params) = params_from_vars(x_num_vars);
    let x_all_com = hyrax_commit(&x_all_evals, x_nu, &x_params);

    let y_num_vars = l_bits + t_bits + j_bits;
    let (y_nu, y_sigma, y_params) = params_from_vars(y_num_vars);
    let y_all_com = hyrax_commit(&y_all_evals, y_nu, &y_params);

    // --- Absorb into transcript ---
    absorb_com(transcript, b"cl_x_all_com", &x_all_com);
    absorb_com(transcript, b"cl_y_all_com", &y_all_com);
    absorb_com(transcript, b"cl_w_all_com", &pk.vk.w_all_com);
    absorb_com(transcript, b"cl_bias_all_com", &pk.vk.bias_all_com);

    // --- Derive challenges ---
    let r_l = challenge_vec(transcript, l_bits, b"cl_rl");
    let r_t = challenge_vec(transcript, t_bits, b"cl_rt");
    let r_out = challenge_vec(transcript, j_bits, b"cl_rout");

    // --- Build f(b,c) = eq(r_l, b) replicated over c ---
    // f has l_bits+k_bits variables; index = b*k_p2 + c
    let eq_l_evals = DenseMLPoly::eq_poly(&r_l).evaluations;
    let f_evals: Vec<F> = (0..l_p2 * k_p2)
        .map(|i| eq_l_evals[i / k_p2])
        .collect();
    let f_poly = DenseMLPoly::new(f_evals);

    // --- Build g(b,c) = X_all(b, r_t, c) ---
    // Fix r_t in each layer's activation matrix
    let mut g_evals = vec![F::ZERO; l_p2 * k_p2];
    for b in 0..num_l {
        let x_mle = mat_to_mle(&xs[b], t, d_in);
        let x_at_rt = eval_rows(&x_mle, t_bits, &r_t); // length k_p2
        for c in 0..k_p2 {
            g_evals[b * k_p2 + c] = x_at_rt[c];
        }
    }
    let g_poly = DenseMLPoly::new(g_evals);

    // --- Build h(b,c) = W_all(b, c, r_out) ---
    // For each (b, c), evaluate the j_p2-length slice at r_out
    let mut h_evals = vec![F::ZERO; l_p2 * k_p2];
    for b in 0..l_p2 {
        for c in 0..k_p2 {
            let start = b * k_p2 * j_p2 + c * j_p2;
            let slice: Vec<F> = pk.w_all_evals[start..start + j_p2].to_vec();
            h_evals[b * k_p2 + c] = DenseMLPoly::new(slice).evaluate(&r_out);
        }
    }
    let h_poly = DenseMLPoly::new(h_evals);

    // --- Compute claim = Y_all(r_l, r_t, r_out) - bias_all(r_l, r_out) ---
    let y_all_mle = DenseMLPoly::new(y_all_evals.clone());
    let bias_all_mle = DenseMLPoly::new(pk.bias_all_evals.clone());
    let rl_rt_rout: Vec<F> = r_l.iter().chain(r_t.iter()).chain(r_out.iter()).cloned().collect();
    let rl_rout: Vec<F> = r_l.iter().chain(r_out.iter()).cloned().collect();
    let y_all_eval = y_all_mle.evaluate(&rl_rt_rout);
    let bias_all_eval = bias_all_mle.evaluate(&rl_rout);
    let claim = y_all_eval - bias_all_eval;

    // --- Cubic sumcheck over l_bits + k_bits variables ---
    let (sumcheck, r_sc) =
        prove_sumcheck_cubic(&f_poly, &g_poly, &h_poly, claim, transcript);

    // Split sumcheck challenges: first l_bits → layer, last k_bits → inner dim
    let r_b: Vec<F> = r_sc[..l_bits].to_vec();
    let r_c: Vec<F> = r_sc[l_bits..].to_vec();

    // --- Compute opening evaluations ---
    let x_all_mle = DenseMLPoly::new(x_all_evals.clone());
    let rb_rt_rc: Vec<F> = r_b.iter().chain(r_t.iter()).chain(r_c.iter()).cloned().collect();
    let rb_rc_rout: Vec<F> = r_b.iter().chain(r_c.iter()).chain(r_out.iter()).cloned().collect();

    let x_all_eval = x_all_mle.evaluate(&rb_rt_rc);
    let w_all_eval = DenseMLPoly::new(pk.w_all_evals.clone()).evaluate(&rb_rc_rout);

    // --- PCS openings ---
    let x_all_open = hyrax_open(&x_all_evals, &rb_rt_rc, x_nu, x_sigma);
    let w_num_vars = l_bits + k_bits + j_bits;
    let (w_nu, w_sigma, _) = params_from_vars(w_num_vars);
    let w_all_open = hyrax_open(&pk.w_all_evals, &rb_rc_rout, w_nu, w_sigma);
    let y_all_open = hyrax_open(&y_all_evals, &rl_rt_rout, y_nu, y_sigma);
    let bias_num_vars = (l_bits + j_bits).max(1);
    let (b_nu, b_sigma, _) = params_from_vars(bias_num_vars);
    let bias_all_open = hyrax_open(&pk.bias_all_evals, &rl_rout, b_nu, b_sigma);

    let y_claim = EvalClaim { point: rl_rt_rout, value: y_all_eval };

    Ok((
        CrossLayerProjectionProof {
            x_all_com,
            y_all_com,
            sumcheck,
            x_all_eval,
            w_all_eval,
            y_all_eval,
            bias_all_eval,
            x_all_open,
            w_all_open,
            y_all_open,
            bias_all_open,
        },
        y_claim,
    ))
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify the cross-layer projection proof.
///
/// Returns an `EvalClaim` for Y_all at (r_l ‖ r_t ‖ r_out) for downstream chaining.
/// Defers all MSM checks to the provided accumulators.
pub fn verify_cross_layer_projection(
    proof: &CrossLayerProjectionProof,
    vk: &CrossLayerProjectionVK,
    transcript: &mut Transcript,
    acc_x: &mut HyraxBatchAccumulator,
    acc_w: &mut HyraxBatchAccumulator,
    acc_y: &mut HyraxBatchAccumulator,
    acc_b: &mut HyraxBatchAccumulator,
) -> Result<EvalClaim, String> {
    let num_l = vk.num_layers;
    let t = vk.seq_len;
    let d_in = vk.d_in;
    let d_out = vk.d_out;

    let l_p2 = num_l.next_power_of_two().max(1);
    let t_p2 = t.next_power_of_two().max(1);
    let k_p2 = d_in.next_power_of_two().max(1);
    let j_p2 = d_out.next_power_of_two().max(1);

    let l_bits = l_p2.trailing_zeros() as usize;
    let t_bits = t_p2.trailing_zeros() as usize;
    let k_bits = k_p2.trailing_zeros() as usize;
    let j_bits = j_p2.trailing_zeros() as usize;

    // Mirror prover transcript
    absorb_com(transcript, b"cl_x_all_com", &proof.x_all_com);
    absorb_com(transcript, b"cl_y_all_com", &proof.y_all_com);
    absorb_com(transcript, b"cl_w_all_com", &vk.w_all_com);
    absorb_com(transcript, b"cl_bias_all_com", &vk.bias_all_com);

    let r_l = challenge_vec(transcript, l_bits, b"cl_rl");
    let r_t = challenge_vec(transcript, t_bits, b"cl_rt");
    let r_out = challenge_vec(transcript, j_bits, b"cl_rout");

    // Reconstruct claim
    let claim = proof.y_all_eval - proof.bias_all_eval;

    // Verify cubic sumcheck
    let sc_num_vars = l_bits + k_bits;
    let (r_sc, _) = verify_sumcheck_cubic(&proof.sumcheck, claim, sc_num_vars, transcript)
        .map_err(|e| format!("CrossLayer Sumcheck: {e}"))?;

    let r_b: Vec<F> = r_sc[..l_bits].to_vec();
    let r_c: Vec<F> = r_sc[l_bits..].to_vec();

    // f_final = eq(r_l, r_b) — verifier computes directly, no PCS needed
    let f_final = DenseMLPoly::eq_poly(&r_l).evaluate(&r_b);
    if f_final != proof.sumcheck.final_eval_f {
        return Err("CrossLayer: eq(r_l, r_b) mismatch".into());
    }

    // g_final and h_final must match the proof's explicit eval claims
    if proof.x_all_eval != proof.sumcheck.final_eval_g {
        return Err("CrossLayer: x_all_eval != sumcheck final_eval_g".into());
    }
    if proof.w_all_eval != proof.sumcheck.final_eval_h {
        return Err("CrossLayer: w_all_eval != sumcheck final_eval_h".into());
    }

    // Deferred PCS openings
    let rb_rt_rc: Vec<F> = r_b.iter().chain(r_t.iter()).chain(r_c.iter()).cloned().collect();
    let rb_rc_rout: Vec<F> = r_b.iter().chain(r_c.iter()).chain(r_out.iter()).cloned().collect();
    let rl_rt_rout: Vec<F> = r_l.iter().chain(r_t.iter()).chain(r_out.iter()).cloned().collect();
    let rl_rout: Vec<F> = r_l.iter().chain(r_out.iter()).cloned().collect();

    acc_x.add_verify(&proof.x_all_com, proof.x_all_eval, &rb_rt_rc, &proof.x_all_open)
        .map_err(|e| format!("x_all_open: {e}"))?;
    acc_w.add_verify(&vk.w_all_com, proof.w_all_eval, &rb_rc_rout, &proof.w_all_open)
        .map_err(|e| format!("w_all_open: {e}"))?;
    acc_y.add_verify(&proof.y_all_com, proof.y_all_eval, &rl_rt_rout, &proof.y_all_open)
        .map_err(|e| format!("y_all_open: {e}"))?;
    acc_b.add_verify(&vk.bias_all_com, proof.bias_all_eval, &rl_rout, &proof.bias_all_open)
        .map_err(|e| format!("bias_all_open: {e}"))?;

    Ok(EvalClaim { point: rl_rt_rout, value: proof.y_all_eval })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::{params_from_vars, HyraxBatchAccumulator};
    use crate::transcript::Transcript;

    const T: usize = 2;
    const D_IN: usize = 4;
    const D_OUT: usize = 2;
    const L: usize = 2; // 2 layers

    fn build_test_pk() -> CrossLayerProjectionPK {
        // W_l[k][j] = if k==0 then ONE else ZERO → output col 0 = alpha * x[:,0]
        let ws: Vec<Vec<Vec<TernaryValue>>> = (0..L)
            .map(|_| {
                let mut w = vec![vec![TernaryValue::ZERO; D_OUT]; D_IN];
                w[0][0] = TernaryValue::ONE;
                w
            })
            .collect();
        let alphas: Vec<F> = vec![F::from(2u64); L];
        let biases: Vec<Vec<F>> = vec![vec![F::from(1u64); D_OUT]; L];
        preprocess_cross_layer_projection(L, T, D_IN, D_OUT, &ws, &alphas, &biases)
    }

    fn build_test_witness() -> (Vec<Vec<Vec<F>>>, Vec<Vec<Vec<F>>>) {
        // x[l][t] = [3, 5, 0, 0] for all l, t
        let x_row = vec![F::from(3u64), F::from(5u64), F::ZERO, F::ZERO];
        let xs: Vec<Vec<Vec<F>>> = (0..L)
            .map(|_| vec![x_row.clone(); T])
            .collect();

        // Y = alpha * X @ W + bias
        // Y[l][t][0] = 2 * 3 * 1 + 1 = 7, Y[l][t][1] = 2 * 3 * 0 + 1 = 1
        let y_row = vec![F::from(7u64), F::from(1u64)];
        let ys: Vec<Vec<Vec<F>>> = (0..L)
            .map(|_| vec![y_row.clone(); T])
            .collect();

        (xs, ys)
    }

    #[test]
    fn test_cross_layer_projection_prove_verify() {
        let pk = build_test_pk();
        let (xs, ys) = build_test_witness();

        let mut pt = Transcript::new(b"cl_proj_test");
        let (proof, _y_claim) =
            prove_cross_layer_projection(&xs, &ys, &pk, &mut pt).unwrap();

        // Advance transcript to match the 4 finalizations below
        let _ = pt.challenge_field::<crate::field::F>(b"hyrax_group_mu");
        let _ = pt.challenge_field::<crate::field::F>(b"hyrax_group_mu");
        let _ = pt.challenge_field::<crate::field::F>(b"hyrax_group_mu");
        let _ = pt.challenge_field::<crate::field::F>(b"hyrax_group_mu");

        let mut vt = Transcript::new(b"cl_proj_test");
        let mut acc_x = HyraxBatchAccumulator::new();
        let mut acc_w = HyraxBatchAccumulator::new();
        let mut acc_y = HyraxBatchAccumulator::new();
        let mut acc_b = HyraxBatchAccumulator::new();

        let result = verify_cross_layer_projection(
            &proof, &pk.vk, &mut vt, &mut acc_x, &mut acc_w, &mut acc_y, &mut acc_b,
        );
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());

        // Finalize all accumulators
        let x_num_vars = L.next_power_of_two().trailing_zeros() as usize
            + T.next_power_of_two().trailing_zeros() as usize
            + D_IN.next_power_of_two().trailing_zeros() as usize;
        let (_, _, x_params) = params_from_vars(x_num_vars);
        acc_x.finalize(&x_params, &mut vt).expect("acc_x finalize failed");

        let w_num_vars = L.next_power_of_two().trailing_zeros() as usize
            + D_IN.next_power_of_two().trailing_zeros() as usize
            + D_OUT.next_power_of_two().trailing_zeros() as usize;
        let (_, _, w_params) = params_from_vars(w_num_vars);
        acc_w.finalize(&w_params, &mut vt).expect("acc_w finalize failed");

        let y_num_vars = L.next_power_of_two().trailing_zeros() as usize
            + T.next_power_of_two().trailing_zeros() as usize
            + D_OUT.next_power_of_two().trailing_zeros() as usize;
        let (_, _, y_params) = params_from_vars(y_num_vars);
        acc_y.finalize(&y_params, &mut vt).expect("acc_y finalize failed");

        let b_num_vars = (L.next_power_of_two().trailing_zeros() as usize
            + D_OUT.next_power_of_two().trailing_zeros() as usize).max(1);
        let (_, _, b_params) = params_from_vars(b_num_vars);
        acc_b.finalize(&b_params, &mut vt).expect("acc_b finalize failed");
    }

    /// Tampering the output evaluation must trigger a sumcheck failure.
    #[test]
    fn test_cross_layer_rejects_tampered_y_eval() {
        let pk = build_test_pk();
        let (xs, ys) = build_test_witness();

        let mut pt = Transcript::new(b"cl_tamper_y");
        let (mut proof, _) =
            prove_cross_layer_projection(&xs, &ys, &pk, &mut pt).unwrap();

        proof.y_all_eval += F::ONE; // shift claim

        let mut vt = Transcript::new(b"cl_tamper_y");
        let mut acc_x = HyraxBatchAccumulator::new();
        let mut acc_w = HyraxBatchAccumulator::new();
        let mut acc_y = HyraxBatchAccumulator::new();
        let mut acc_b = HyraxBatchAccumulator::new();

        let result = verify_cross_layer_projection(
            &proof, &pk.vk, &mut vt, &mut acc_x, &mut acc_w, &mut acc_y, &mut acc_b,
        );
        assert!(result.is_err(), "Should reject tampered y_all_eval");
    }

    /// Tampering x_all_eval (the activation opening) must be caught.
    #[test]
    fn test_cross_layer_rejects_tampered_x_eval() {
        let pk = build_test_pk();
        let (xs, ys) = build_test_witness();

        let mut pt = Transcript::new(b"cl_tamper_x");
        let (mut proof, _) =
            prove_cross_layer_projection(&xs, &ys, &pk, &mut pt).unwrap();

        proof.x_all_eval += F::ONE;

        let mut vt = Transcript::new(b"cl_tamper_x");
        let mut acc_x = HyraxBatchAccumulator::new();
        let mut acc_w = HyraxBatchAccumulator::new();
        let mut acc_y = HyraxBatchAccumulator::new();
        let mut acc_b = HyraxBatchAccumulator::new();

        let result = verify_cross_layer_projection(
            &proof, &pk.vk, &mut vt, &mut acc_x, &mut acc_w, &mut acc_y, &mut acc_b,
        );
        assert!(result.is_err(), "Should reject tampered x_all_eval");
    }
}
