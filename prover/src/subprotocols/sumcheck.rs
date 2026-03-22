//! Sumcheck protocol for the product of two multilinear polynomials.
//!
//! **Statement:** H = Σ_{x ∈ {0,1}^n} f(x) · g(x)
//!
//! Each round polynomial g_i(X) has degree ≤ 2 (product of two degree-1 MLEs).
//! We represent it by its evaluations at X ∈ {0, 1, 2} and use quadratic
//! Lagrange interpolation to evaluate at arbitrary points.
//!
//! **Prover cost:** O(n · 2^n) field multiplications.
//! **Verifier cost:** O(n) field operations (plus one external oracle query for f(r)·g(r)).

use crate::field::F;
use crate::poly::DenseMLPoly;
use crate::transcript::Transcript;
use ark_ff::Field;

/// A round polynomial g_i(X) given by its values at X = 0, 1, 2.
#[derive(Clone, Debug)]
pub struct RoundPoly {
    pub evals: [F; 3], // [g(0), g(1), g(2)]
}

impl RoundPoly {
    /// Evaluate at arbitrary x via quadratic Lagrange interpolation through (0,g0),(1,g1),(2,g2).
    pub fn evaluate(&self, x: F) -> F {
        let [g0, g1, g2] = self.evals;
        let two = F::from(2u64);
        let inv2 = two.inverse().unwrap();
        // L_0(x) = (x-1)(x-2)/2
        // L_1(x) = -x(x-2) = x(2-x)
        // L_2(x) = x(x-1)/2
        let l0 = (x - F::ONE) * (x - two) * inv2;
        let l1 = x * (two - x);
        let l2 = x * (x - F::ONE) * inv2;
        g0 * l0 + g1 * l1 + g2 * l2
    }
}

/// A complete sumcheck proof: one `RoundPoly` per variable, plus the prover's
/// final opening claims f(r) and g(r) which the verifier must check externally
/// (e.g. via a polynomial commitment opening).
#[derive(Clone, Debug)]
pub struct SumcheckProof {
    pub round_polys: Vec<RoundPoly>,
    /// Prover's claim: f evaluated at the final random point r.
    pub final_eval_f: F,
    /// Prover's claim: g evaluated at the final random point r.
    pub final_eval_g: F,
}

/// Prove Σ_{x ∈ {0,1}^n} f(x)·g(x) = `claim`.
///
/// Returns `(proof, r)` where `r` is the Fiat-Shamir random evaluation point.
pub fn prove_sumcheck(
    f: &DenseMLPoly,
    g: &DenseMLPoly,
    claim: F,
    transcript: &mut Transcript,
) -> (SumcheckProof, Vec<F>) {
    assert_eq!(
        f.num_vars, g.num_vars,
        "f and g must have the same number of variables"
    );
    let n = f.num_vars;

    transcript.append_field(b"sc_claim", &claim);

    let mut f_cur = f.clone();
    let mut g_cur = g.clone();
    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for _ in 0..n {
        let half = f_cur.evaluations.len() >> 1;

        // g_i(0) = Σ_{x2,...} f(0, x2,...) · g(0, x2,...)
        let e0: F = (0..half)
            .map(|i| f_cur.evaluations[i] * g_cur.evaluations[i])
            .sum();

        // g_i(1) = Σ_{x2,...} f(1, x2,...) · g(1, x2,...)
        let e1: F = (0..half)
            .map(|i| f_cur.evaluations[i + half] * g_cur.evaluations[i + half])
            .sum();

        // g_i(2): extrapolate each to x=2 then multiply
        // f(2, x2,...) = 2·f(1,x2,...) - f(0,x2,...) (linear extrapolation)
        let two = F::from(2u64);
        let e2: F = (0..half)
            .map(|i| {
                let f2 = two * f_cur.evaluations[i + half] - f_cur.evaluations[i];
                let g2 = two * g_cur.evaluations[i + half] - g_cur.evaluations[i];
                f2 * g2
            })
            .sum();

        let rp = RoundPoly {
            evals: [e0, e1, e2],
        };

        // Absorb round polynomial into transcript
        for e in &rp.evals {
            transcript.append_field(b"sc_round", e);
        }
        let r_i = transcript.challenge_field::<F>(b"sc_challenge");
        challenges.push(r_i);

        // Fix first variable to r_i for the next round
        f_cur = f_cur.fix_first_variable(r_i);
        g_cur = g_cur.fix_first_variable(r_i);
        round_polys.push(rp);
    }

    let final_eval_f = f_cur.evaluations[0];
    let final_eval_g = g_cur.evaluations[0];

    (
        SumcheckProof {
            round_polys,
            final_eval_f,
            final_eval_g,
        },
        challenges,
    )
}

/// Verify a sumcheck proof.
///
/// On success returns `(r, final_claim)` where `final_claim = final_eval_f * final_eval_g`.
/// The caller is responsible for verifying `final_eval_f` and `final_eval_g` independently
/// (e.g. via PCS openings).
pub fn verify_sumcheck(
    proof: &SumcheckProof,
    claim: F,
    num_vars: usize,
    transcript: &mut Transcript,
) -> Result<(Vec<F>, F), String> {
    if proof.round_polys.len() != num_vars {
        return Err(format!(
            "Wrong number of round polys: got {}, expected {}",
            proof.round_polys.len(),
            num_vars
        ));
    }

    transcript.append_field(b"sc_claim", &claim);

    let mut current = claim;
    let mut challenges = Vec::with_capacity(num_vars);

    for (i, rp) in proof.round_polys.iter().enumerate() {
        // Check consistency: g_i(0) + g_i(1) must equal the current claim
        if rp.evals[0] + rp.evals[1] != current {
            return Err(format!(
                "Round {}: g(0)+g(1) = {:?} ≠ claim {:?}",
                i,
                rp.evals[0] + rp.evals[1],
                current
            ));
        }
        for e in &rp.evals {
            transcript.append_field(b"sc_round", e);
        }
        let r_i = transcript.challenge_field::<F>(b"sc_challenge");
        challenges.push(r_i);
        current = rp.evaluate(r_i);
    }

    // Final consistency check
    let final_claim = proof.final_eval_f * proof.final_eval_g;
    if final_claim != current {
        return Err(format!(
            "Final check: f(r)*g(r) = {:?} ≠ claim {:?}",
            final_claim, current
        ));
    }

    Ok((challenges, final_claim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::DenseMLPoly;
    use crate::transcript::Transcript;
    use ark_ff::{One, Zero};

    /// Helper to compute the brute-force sum of f(x) * g(x) over {0,1}^n
    fn compute_brute_force_sum(f: &DenseMLPoly, g: &DenseMLPoly) -> F {
        f.evaluations
            .iter()
            .zip(g.evaluations.iter())
            .map(|(&fa, &ga)| fa * ga)
            .sum()
    }

    #[test]
    fn test_sumcheck_happy_path_2var() {
        let mut transcript = Transcript::new(b"test");

        // f(x0, x1) = [1, 2, 3, 4]
        // g(x0, x1) = [5, 6, 7, 8]
        // Sum = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let f = DenseMLPoly::new(vec![F::from(1), F::from(2), F::from(3), F::from(4)]);
        let g = DenseMLPoly::new(vec![F::from(5), F::from(6), F::from(7), F::from(8)]);
        let claim = compute_brute_force_sum(&f, &g);
        assert_eq!(claim, F::from(70));

        // Prover
        let (proof, challenges) = prove_sumcheck(&f, &g, claim, &mut transcript);

        // Verifier
        let mut verifier_transcript = Transcript::new(b"test");
        let result = verify_sumcheck(&proof, claim, f.num_vars, &mut verifier_transcript);

        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
        let (v_challenges, _) = result.unwrap();
        assert_eq!(challenges, v_challenges);

        // Final consistency: Check if final_eval_f and final_eval_g are actually f(r) and g(r)
        assert_eq!(proof.final_eval_f, f.evaluate(&challenges));
        assert_eq!(proof.final_eval_g, g.evaluate(&challenges));
    }

    #[test]
    fn test_sumcheck_invalid_claim() {
        let mut transcript = Transcript::new(b"test");
        let f = DenseMLPoly::new(vec![F::from(1), F::from(2)]);
        let g = DenseMLPoly::new(vec![F::from(3), F::from(4)]);

        // Correct sum is 1*3 + 2*4 = 11. Let's claim 12.
        let false_claim = F::from(12);

        let (proof, _) = prove_sumcheck(&f, &g, false_claim, &mut transcript);

        let mut verifier_transcript = Transcript::new(b"test");
        let result = verify_sumcheck(&proof, false_claim, f.num_vars, &mut verifier_transcript);

        // Verifier should catch g_0(0) + g_0(1) != false_claim
        assert!(result.is_err());
    }

    #[test]
    fn test_sumcheck_tampered_proof() {
        let mut transcript = Transcript::new(b"test");
        let f = DenseMLPoly::new(vec![F::from(1), F::from(2), F::from(3), F::from(4)]);
        let g = DenseMLPoly::new(vec![F::from(1), F::from(1), F::from(1), F::from(1)]);
        let claim = compute_brute_force_sum(&f, &g);

        let (mut proof, _) = prove_sumcheck(&f, &g, claim, &mut transcript);

        // Tamper with one of the round evaluations
        proof.round_polys[0].evals[0] += F::one();

        let mut verifier_transcript = Transcript::new(b"test");
        let result = verify_sumcheck(&proof, claim, f.num_vars, &mut verifier_transcript);
        assert!(result.is_err());
    }

    #[test]
    fn test_sumcheck_zero_polynomials() {
        let n = 3;
        let f = DenseMLPoly::zero(n);
        let g = DenseMLPoly::zero(n);
        let claim = F::zero();

        let mut transcript = Transcript::new(b"test");
        let (proof, _) = prove_sumcheck(&f, &g, claim, &mut transcript);

        let mut verifier_transcript = Transcript::new(b"test");
        let result = verify_sumcheck(&proof, claim, n, &mut verifier_transcript);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sumcheck_one_variable() {
        // f = [3, 7], g = [2, 5]; sum = 3*2 + 7*5 = 41
        let f = DenseMLPoly::new(vec![F::from(3u64), F::from(7u64)]);
        let g = DenseMLPoly::new(vec![F::from(2u64), F::from(5u64)]);
        let claim = compute_brute_force_sum(&f, &g);
        assert_eq!(claim, F::from(41u64));

        let mut pt = Transcript::new(b"test1");
        let (proof, challenges) = prove_sumcheck(&f, &g, claim, &mut pt);
        assert_eq!(proof.round_polys.len(), 1);

        let mut vt = Transcript::new(b"test1");
        let result = verify_sumcheck(&proof, claim, 1, &mut vt);
        assert!(result.is_ok(), "{:?}", result.err());

        let (v_challenges, _) = result.unwrap();
        assert_eq!(challenges, v_challenges);
    }

    #[test]
    fn test_sumcheck_four_variables() {
        let n = 4;
        let f = DenseMLPoly::new((1u64..=16).map(F::from).collect());
        let g = DenseMLPoly::new((1u64..=16).map(|i| F::from(i * 2)).collect());
        let claim = compute_brute_force_sum(&f, &g);

        let mut pt = Transcript::new(b"test4");
        let (proof, challenges) = prove_sumcheck(&f, &g, claim, &mut pt);

        let mut vt = Transcript::new(b"test4");
        let result = verify_sumcheck(&proof, claim, n, &mut vt);
        assert!(result.is_ok(), "{:?}", result.err());

        // Cross-check prover's final claims against direct polynomial evaluation
        let (r, _) = result.unwrap();
        assert_eq!(proof.final_eval_f, f.evaluate(&r));
        assert_eq!(proof.final_eval_g, g.evaluate(&r));
        assert_eq!(challenges, r);
    }

    #[test]
    fn test_sumcheck_g_all_ones() {
        // When g = all-ones, the sum equals Σ f(x), and the sumcheck
        // reduces to a claim about the sum of f.
        let n = 3;
        let f = DenseMLPoly::new(vec![
            F::from(1u64),
            F::from(2u64),
            F::from(3u64),
            F::from(4u64),
            F::from(5u64),
            F::from(6u64),
            F::from(7u64),
            F::from(8u64),
        ]);
        let g = DenseMLPoly::new(vec![F::ONE; 8]);
        let claim = f.sum_over_hypercube(); // = 36

        let mut pt = Transcript::new(b"ones");
        let (proof, _) = prove_sumcheck(&f, &g, claim, &mut pt);

        let mut vt = Transcript::new(b"ones");
        let result = verify_sumcheck(&proof, claim, n, &mut vt);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn test_sumcheck_round_poly_evaluate_at_0_and_1() {
        // Verify that the RoundPoly evaluates correctly at the boolean points.
        let rp = RoundPoly {
            evals: [F::from(3u64), F::from(7u64), F::from(15u64)],
        };
        assert_eq!(rp.evaluate(F::ZERO), F::from(3u64));
        assert_eq!(rp.evaluate(F::ONE), F::from(7u64));
        assert_eq!(rp.evaluate(F::from(2u64)), F::from(15u64));
    }

    #[test]
    fn test_sumcheck_wrong_num_vars_returns_err() {
        let f = DenseMLPoly::new(vec![F::ONE; 4]);
        let g = DenseMLPoly::new(vec![F::ONE; 4]);
        let claim = compute_brute_force_sum(&f, &g);

        let mut pt = Transcript::new(b"mismatch");
        let (proof, _) = prove_sumcheck(&f, &g, claim, &mut pt);

        // Pass num_vars=3 but proof has only 2 round polys
        let mut vt = Transcript::new(b"mismatch");
        let result = verify_sumcheck(&proof, claim, 3, &mut vt);
        assert!(result.is_err());
    }
}
