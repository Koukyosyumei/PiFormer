//! GKR-style combine sumcheck.
//!
//! Reduces k eval claims on the same committed polynomial f —
//!   f(z_1) = v_1, ..., f(z_k) = v_k
//! — to a **single** Hyrax opening by running one sumcheck over
//!   G(x) = Σ_i r_i · eq(z_i, x),
//! where r_i are Fiat-Shamir weights.  The verifier computes G(r_final)
//! locally in O(k · n) field operations; no PCS opening for G is needed.

use crate::field::F;
use crate::pcs::{hyrax_open, hyrax_verify, params_from_vars, HyraxCommitment, HyraxProof};
use crate::poly::utils::compute_eq_evals;
use crate::poly::DenseMLPoly;
use crate::subprotocols::sumcheck::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::{One, Zero};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// An assertion that polynomial f evaluates to `value` at `point`.
#[derive(Clone, Debug)]
pub struct EvalClaim {
    pub point: Vec<F>,
    pub value: F,
}

/// Proves k eval claims on one polynomial via a single Hyrax opening.
#[derive(Clone, Debug)]
pub struct CombineProof {
    /// Sumcheck proof for Σ_x G(x)·f(x) = combined_claim.
    /// `sumcheck.final_eval_f` = G(r_final) (verifier re-derives independently).
    /// `sumcheck.final_eval_g` = f(r_final) (bound by the Hyrax opening below).
    pub sumcheck: SumcheckProof,
    /// Hyrax opening of f at r_final.
    pub hyrax_proof: HyraxProof,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prove all claims in `claims` on polynomial `f_evals` via a single opening.
///
/// Returns `(proof, r_final)`.  The caller should ensure that
/// `f_evals[x] = f(x)` for x ∈ {0,1}^num_vars in the same layout used when
/// `f_com` was generated.
pub fn prove_combine(
    f_evals: &[F],
    _f_com: &HyraxCommitment,
    claims: &[EvalClaim],
    num_vars: usize,
    transcript: &mut Transcript,
) -> (CombineProof, Vec<F>) {
    let n = 1usize << num_vars;

    // Absorb claims + derive random weights via Fiat-Shamir
    let weights = derive_weights(claims, transcript);

    // Combined claim C = Σ_i r_i · v_i
    let combined_claim: F = claims
        .iter()
        .zip(weights.iter())
        .map(|(c, &w)| w * c.value)
        .sum();

    // Build G polynomial: G(x) = Σ_i weights[i] · eq(z_i, x)
    // Pre-compute each claim's eq table sequentially (few claims), then accumulate in parallel.
    let eq_arrays: Vec<Vec<F>> = claims
        .iter()
        .map(|claim| {
            let rev_point: Vec<F> = claim.point.iter().cloned().rev().collect();
            compute_eq_evals(&rev_point, n)
        })
        .collect();
    let g_evals: Vec<F> = (0..n)
        .into_par_iter()
        .map(|j| {
            eq_arrays
                .iter()
                .zip(weights.iter())
                .map(|(eq_arr, &w)| w * eq_arr[j])
                .sum()
        })
        .collect();
    let g_poly = DenseMLPoly::from_vec_padded(g_evals);
    let f_poly = DenseMLPoly::new(f_evals.to_vec());

    // prove_sumcheck(G, f, C, transcript)
    let (sumcheck, r_final) = prove_sumcheck(&g_poly, &f_poly, combined_claim, transcript);

    let (nu, sigma, _) = params_from_vars(num_vars);
    let hyrax_proof = hyrax_open(f_evals, &r_final, nu, sigma);

    (CombineProof { sumcheck, hyrax_proof }, r_final)
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify all `claims` on `f_com` using `proof`.  Returns `r_final`.
pub fn verify_combine(
    proof: &CombineProof,
    f_com: &HyraxCommitment,
    claims: &[EvalClaim],
    num_vars: usize,
    transcript: &mut Transcript,
) -> Result<Vec<F>, String> {
    // Re-derive weights (identical transcript path as prover)
    let weights = derive_weights(claims, transcript);

    // Combined claim
    let combined_claim: F = claims
        .iter()
        .zip(weights.iter())
        .map(|(c, &w)| w * c.value)
        .sum();

    // Verify sumcheck
    let (r_final, leaf) =
        verify_sumcheck(&proof.sumcheck, combined_claim, num_vars, transcript)?;

    // Verifier computes G(r_final) locally — no PCS opening needed for G
    let g_final: F = claims
        .iter()
        .zip(weights.iter())
        .map(|(claim, &wi)| wi * eq_poly_eval(&claim.point, &r_final))
        .sum();

    // Leaf check: leaf == G(r_final) · f(r_final)
    if leaf != g_final * proof.sumcheck.final_eval_g {
        return Err(format!(
            "Combine sumcheck leaf mismatch: leaf={:?}, G*f={:?}",
            leaf,
            g_final * proof.sumcheck.final_eval_g
        ));
    }

    // Single Hyrax opening of f at r_final
    let (_, _, params) = params_from_vars(num_vars);
    hyrax_verify(
        f_com,
        proof.sumcheck.final_eval_g,
        &r_final,
        &proof.hyrax_proof,
        &params,
    )?;

    Ok(r_final)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn derive_weights(claims: &[EvalClaim], transcript: &mut Transcript) -> Vec<F> {
    let mut weights = Vec::with_capacity(claims.len());
    for claim in claims {
        for &p in &claim.point {
            transcript.append_field(b"combine_point", &p);
        }
        transcript.append_field(b"combine_eval", &claim.value);
        let r_i = transcript.challenge_field::<F>(b"combine_weight");
        weights.push(r_i);
    }
    weights
}

/// Evaluates eq(z, r) = Π_j [ z_j · r_j + (1 − z_j)·(1 − r_j) ]
pub fn eq_poly_eval(z: &[F], r: &[F]) -> F {
    z.iter()
        .zip(r.iter())
        .map(|(&zi, &ri)| zi * ri + (F::one() - zi) * (F::one() - ri))
        .product()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod combine_tests {
    use super::*;
    use crate::pcs::{hyrax_commit, params_from_vars};
    use crate::transcript::Transcript;
    use ark_ff::Zero;



    #[test]
    fn test_combine_single_claim() {
        let num_vars = 4usize;
        let n = 1 << num_vars;
        let f_evals: Vec<F> = (0..n).map(|i| F::from(i as u64 + 1)).collect();

        let (nu, _sigma, params) = params_from_vars(num_vars);
        let f_com = hyrax_commit(&f_evals, nu, &params);

        let z = vec![F::from(1u64), F::from(0u64), F::from(1u64), F::from(0u64)];
        // eval at binary point (1,0,1,0): index = 0b1010 = 10 (bit ordering: z[0]=LSB? Let's use f_evals[z binary])
        // compute_eq_evals gives eq(j, z) — so to find f(z) we need the MLE eval
        let f_poly = DenseMLPoly::new(f_evals.clone());
        let v = f_poly.evaluate(&z);

        let claims = vec![EvalClaim { point: z, value: v }];

        let mut pt = Transcript::new(b"combine_test");
        let (proof, _) = prove_combine(&f_evals, &f_com, &claims, num_vars, &mut pt);

        let mut vt = Transcript::new(b"combine_test");
        let result = verify_combine(&proof, &f_com, &claims, num_vars, &mut vt);
        assert!(result.is_ok(), "single-claim combine failed: {:?}", result.err());
    }

    #[test]
    fn test_combine_multiple_claims() {
        let num_vars = 4usize;
        let n = 1 << num_vars;
        let f_evals: Vec<F> = (0..n).map(|i| F::from(i as u64 * 3 + 7)).collect();

        let (nu, _sigma, params) = params_from_vars(num_vars);
        let f_com = hyrax_commit(&f_evals, nu, &params);

        let f_poly = DenseMLPoly::new(f_evals.clone());
        let claims: Vec<EvalClaim> = vec![
            vec![F::from(1u64), F::from(0u64), F::from(1u64), F::from(0u64)],
            vec![F::from(0u64), F::from(1u64), F::from(0u64), F::from(1u64)],
            vec![F::from(1u64), F::from(1u64), F::from(0u64), F::from(0u64)],
        ]
        .into_iter()
        .map(|z| {
            let v = f_poly.evaluate(&z);
            EvalClaim { point: z, value: v }
        })
        .collect();

        let mut pt = Transcript::new(b"combine_multi_test");
        let (proof, _) = prove_combine(&f_evals, &f_com, &claims, num_vars, &mut pt);

        let mut vt = Transcript::new(b"combine_multi_test");
        let result = verify_combine(&proof, &f_com, &claims, num_vars, &mut vt);
        assert!(
            result.is_ok(),
            "multi-claim combine failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_combine_rejects_wrong_eval() {
        let num_vars = 4usize;
        let n = 1 << num_vars;
        let f_evals: Vec<F> = (0..n).map(|i| F::from(i as u64 + 1)).collect();

        let (nu, _sigma, params) = params_from_vars(num_vars);
        let f_com = hyrax_commit(&f_evals, nu, &params);

        let z = vec![F::from(1u64), F::from(0u64), F::from(1u64), F::from(0u64)];
        let f_poly = DenseMLPoly::new(f_evals.clone());
        let correct_v = f_poly.evaluate(&z);
        let wrong_v = correct_v + F::from(1u64); // tampered

        let claims = vec![EvalClaim { point: z, value: wrong_v }];

        let mut pt = Transcript::new(b"combine_bad_test");
        let (proof, _) = prove_combine(&f_evals, &f_com, &claims, num_vars, &mut pt);

        // Verifier uses wrong claim value — should fail
        let mut vt = Transcript::new(b"combine_bad_test");
        let result = verify_combine(&proof, &f_com, &claims, num_vars, &mut vt);
        assert!(result.is_err(), "should reject wrong eval");
    }
}
