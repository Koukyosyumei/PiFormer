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

use ark_ff::Field;
use crate::field::F;
use crate::poly::DenseMLPoly;
use crate::transcript::Transcript;

/// A round polynomial g_i(X) given by its values at X = 0, 1, 2.
#[derive(Clone, Debug)]
pub struct RoundPoly {
    pub evals: [F; 3],  // [g(0), g(1), g(2)]
}

impl RoundPoly {
    /// Evaluate at arbitrary x via quadratic Lagrange interpolation through (0,g0),(1,g1),(2,g2).
    pub fn evaluate(&self, x: F) -> F {
        let [g0, g1, g2] = self.evals;
        let two  = F::from(2u64);
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
    assert_eq!(f.num_vars, g.num_vars, "f and g must have the same number of variables");
    let n = f.num_vars;

    transcript.append_field(b"sc_claim", &claim);

    let mut f_cur = f.clone();
    let mut g_cur = g.clone();
    let mut round_polys = Vec::with_capacity(n);
    let mut challenges  = Vec::with_capacity(n);

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

        let rp = RoundPoly { evals: [e0, e1, e2] };

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

    (SumcheckProof { round_polys, final_eval_f, final_eval_g }, challenges)
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
            proof.round_polys.len(), num_vars
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
                i, rp.evals[0] + rp.evals[1], current
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
