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
use rayon::prelude::*;

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

#[derive(Clone, Debug)]
pub struct CubicRoundPoly {
    /// 3次多項式を特定するための 4 つの評価点 [S(0), S(1), S(2), S(3)]
    pub evals: [F; 4],
}

impl CubicRoundPoly {
    /// ラグランジュ補間を用いて、任意の点 r における 3 次多項式の値を計算する
    pub fn evaluate(&self, r: F) -> F {
        let [y0, y1, y2, y3] = self.evals;

        // ラグランジュ基底多項式の計算
        // L0(r) = (r-1)(r-2)(r-3) / (-6)
        // L1(r) = r(r-2)(r-3) / 2
        // L2(r) = r(r-1)(r-3) / (-2)
        // L3(r) = r(r-1)(r-2) / 6

        let r_minus_1 = r - F::ONE;
        let r_minus_2 = r - F::from(2u64);
        let r_minus_3 = r - F::from(3u64);

        let inv2 = F::from(2u64).inverse().unwrap();
        let inv6 = F::from(6u64).inverse().unwrap();

        let l0 = r_minus_1 * r_minus_2 * r_minus_3 * (-inv6);
        let l1 = r * r_minus_2 * r_minus_3 * inv2;
        let l2 = r * r_minus_1 * r_minus_3 * (-inv2);
        let l3 = r * r_minus_1 * r_minus_2 * inv6;

        y0 * l0 + y1 * l1 + y2 * l2 + y3 * l3
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

#[derive(Clone, Debug)]
pub struct SumcheckCubicProof {
    pub round_polys: Vec<CubicRoundPoly>,
    pub final_eval_f: F,
    pub final_eval_g: F,
    pub final_eval_h: F,
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

    let two = F::from(2u64);
    for _ in 0..n {
        let half = f_cur.evaluations.len() >> 1;

        // Single pass: compute g_i(0), g_i(1), g_i(2) together,
        // loading each f/g pair once per index.
        // Use parallel reduction only for large halfs where threading pays off.
        const PAR_THRESHOLD: usize = 512;
        let (e0, e1, e2) = if half >= PAR_THRESHOLD {
            (0..half)
                .into_par_iter()
                .map(|i| {
                    let f0 = f_cur.evaluations[i];
                    let f1 = f_cur.evaluations[i + half];
                    let g0 = g_cur.evaluations[i];
                    let g1 = g_cur.evaluations[i + half];
                    let f2 = two * f1 - f0;
                    let g2 = two * g1 - g0;
                    (f0 * g0, f1 * g1, f2 * g2)
                })
                .reduce(
                    || (F::ZERO, F::ZERO, F::ZERO),
                    |(a0, a1, a2), (b0, b1, b2)| (a0 + b0, a1 + b1, a2 + b2),
                )
        } else {
            (0..half).fold((F::ZERO, F::ZERO, F::ZERO), |(a0, a1, a2), i| {
                let f0 = f_cur.evaluations[i];
                let f1 = f_cur.evaluations[i + half];
                let g0 = g_cur.evaluations[i];
                let g1 = g_cur.evaluations[i + half];
                let f2 = two * f1 - f0;
                let g2 = two * g1 - g0;
                (a0 + f0 * g0, a1 + f1 * g1, a2 + f2 * g2)
            })
        };

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

pub fn prove_sumcheck_cubic(
    f: &DenseMLPoly,
    g: &DenseMLPoly,
    h: &DenseMLPoly,
    claim: F,
    transcript: &mut Transcript,
) -> (SumcheckCubicProof, Vec<F>) {
    let n = f.num_vars;
    transcript.append_field(b"sc_claim", &claim);

    let mut f_cur = f.clone();
    let mut g_cur = g.clone();
    let mut h_cur = h.clone();
    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for _ in 0..n {
        let half = f_cur.evaluations.len() >> 1;
        let mut e = [F::ZERO; 4];

        for i in 0..half {
            let f0 = f_cur.evaluations[i];
            let f1 = f_cur.evaluations[i + half];
            let g0 = g_cur.evaluations[i];
            let g1 = g_cur.evaluations[i + half];
            let h0 = h_cur.evaluations[i];
            let h1 = h_cur.evaluations[i + half];

            // 線形延長を用いて x=2, x=3 の値を外挿
            let f2 = f1 + f1 - f0;
            let f3 = f2 + f1 - f0;
            let g2 = g1 + g1 - g0;
            let g3 = g2 + g1 - g0;
            let h2 = h1 + h1 - h0;
            let h3 = h2 + h1 - h0;

            e[0] += f0 * g0 * h0;
            e[1] += f1 * g1 * h1;
            e[2] += f2 * g2 * h2;
            e[3] += f3 * g3 * h3;
        }

        let rp = CubicRoundPoly { evals: e };
        for val in &rp.evals {
            transcript.append_field(b"sc_round", val);
        }

        let r_i = transcript.challenge_field::<F>(b"sc_challenge");
        challenges.push(r_i);

        f_cur = f_cur.fix_first_variable(r_i);
        g_cur = g_cur.fix_first_variable(r_i);
        h_cur = h_cur.fix_first_variable(r_i);
        round_polys.push(rp);
    }

    (
        SumcheckCubicProof {
            round_polys,
            final_eval_f: f_cur.evaluations[0],
            final_eval_g: g_cur.evaluations[0],
            final_eval_h: h_cur.evaluations[0],
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

pub fn verify_sumcheck_cubic(
    proof: &SumcheckCubicProof,
    claim: F,
    num_vars: usize,
    transcript: &mut Transcript,
) -> Result<(Vec<F>, F), String> {
    if proof.round_polys.len() != num_vars {
        return Err("Invalid number of rounds".into());
    }

    transcript.append_field(b"sc_claim", &claim);

    let mut current_claim = claim;
    let mut challenges = Vec::with_capacity(num_vars);

    for (i, rp) in proof.round_polys.iter().enumerate() {
        // 1. 和のチェック: S(0) + S(1) == current_claim
        if rp.evals[0] + rp.evals[1] != current_claim {
            return Err(format!("Sumcheck round {} failed", i));
        }

        for val in &rp.evals {
            transcript.append_field(b"sc_round", val);
        }

        // 2. チャレンジ取得と評価値の更新
        let r_i = transcript.challenge_field::<F>(b"sc_challenge");
        challenges.push(r_i);
        current_claim = rp.evaluate(r_i);
    }

    // 3. 最終的な積のチェック
    let final_eval = proof.final_eval_f * proof.final_eval_g * proof.final_eval_h;
    if final_eval != current_claim {
        return Err("Sumcheck final consistency check failed".into());
    }

    Ok((challenges, final_eval))
}

/// バッチ化されたSumcheck証明
#[derive(Clone, Debug)]
pub struct SumcheckProofMulti {
    pub round_polys: Vec<RoundPoly>,
    /// 点 r における各 f_i の評価値
    pub final_evals_f: Vec<F>,
    /// 点 r における各 g_i の評価値
    pub final_evals_g: Vec<F>,
}

/// Σ_k weights[k] · (Σ_{x ∈ {0,1}^n} f_k(x)·g_k(x)) = `claim` を証明する。
pub fn prove_sumcheck_multi_batched(
    fs: &[DenseMLPoly],
    gs: &[DenseMLPoly],
    weights: &[F],
    claim: F,
    transcript: &mut Transcript,
) -> (SumcheckProofMulti, Vec<F>) {
    let num_pairs = fs.len();
    assert_eq!(gs.len(), num_pairs);
    assert_eq!(weights.len(), num_pairs);

    let n = fs[0].num_vars;
    for k in 1..num_pairs {
        assert_eq!(
            fs[k].num_vars, n,
            "All polynomials must have the same number of variables"
        );
        assert_eq!(gs[k].num_vars, n);
    }

    transcript.append_field(b"sc_claim", &claim);

    let mut fs_cur: Vec<DenseMLPoly> = fs.to_vec();
    let mut gs_cur: Vec<DenseMLPoly> = gs.to_vec();
    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for _ in 0..n {
        let half = fs_cur[0].evaluations.len() >> 1;
        let two = F::from(2u64);

        // Single pass over half-size index range.
        // For each index i, accumulate weighted contributions from all k pairs,
        // computing e0/e1/e2 simultaneously to avoid redundant loads.
        // Parallelise only for large halfs where threading overhead is justified.
        const PAR_THRESHOLD: usize = 512;
        let (e0, e1, e2) = if half >= PAR_THRESHOLD {
            (0..half)
                .into_par_iter()
                .map(|i| {
                    let mut r0 = F::ZERO;
                    let mut r1 = F::ZERO;
                    let mut r2 = F::ZERO;
                    for k in 0..num_pairs {
                        let f0 = fs_cur[k].evaluations[i];
                        let f1 = fs_cur[k].evaluations[i + half];
                        let g0 = gs_cur[k].evaluations[i];
                        let g1 = gs_cur[k].evaluations[i + half];
                        let f2 = two * f1 - f0;
                        let g2 = two * g1 - g0;
                        r0 += weights[k] * (f0 * g0);
                        r1 += weights[k] * (f1 * g1);
                        r2 += weights[k] * (f2 * g2);
                    }
                    (r0, r1, r2)
                })
                .reduce(
                    || (F::ZERO, F::ZERO, F::ZERO),
                    |(a0, a1, a2), (b0, b1, b2)| (a0 + b0, a1 + b1, a2 + b2),
                )
        } else {
            (0..half).fold((F::ZERO, F::ZERO, F::ZERO), |(a0, a1, a2), i| {
                let mut r0 = F::ZERO;
                let mut r1 = F::ZERO;
                let mut r2 = F::ZERO;
                for k in 0..num_pairs {
                    let f0 = fs_cur[k].evaluations[i];
                    let f1 = fs_cur[k].evaluations[i + half];
                    let g0 = gs_cur[k].evaluations[i];
                    let g1 = gs_cur[k].evaluations[i + half];
                    let f2 = two * f1 - f0;
                    let g2 = two * g1 - g0;
                    r0 += weights[k] * (f0 * g0);
                    r1 += weights[k] * (f1 * g1);
                    r2 += weights[k] * (f2 * g2);
                }
                (a0 + r0, a1 + r1, a2 + r2)
            })
        };

        let rp = RoundPoly {
            evals: [e0, e1, e2],
        };

        // トランスクリプトへの吸収とチャレンジ生成
        for e in &rp.evals {
            transcript.append_field(b"sc_round", e);
        }
        let r_i = transcript.challenge_field::<F>(b"sc_challenge");
        challenges.push(r_i);

        // 次のラウンドに向けて全多項式の最初の変数を r_i に固定
        for k in 0..num_pairs {
            fs_cur[k] = fs_cur[k].fix_first_variable(r_i);
            gs_cur[k] = gs_cur[k].fix_first_variable(r_i);
        }
        round_polys.push(rp);
    }

    // 各多項式の最終的な評価値を取得
    let final_evals_f = fs_cur.iter().map(|p| p.evaluations[0]).collect();
    let final_evals_g = gs_cur.iter().map(|p| p.evaluations[0]).collect();

    (
        SumcheckProofMulti {
            round_polys,
            final_evals_f,
            final_evals_g,
        },
        challenges,
    )
}

/// バッチ版 Sumcheck プロトコルの検証
pub fn verify_sumcheck_multi_batched(
    proof: &SumcheckProofMulti,
    weights: &[F],
    claim: F,
    num_vars: usize,
    transcript: &mut Transcript,
) -> Result<(Vec<F>, F), String> {
    if proof.round_polys.len() != num_vars {
        return Err("Wrong number of round polys".into());
    }

    transcript.append_field(b"sc_claim", &claim);
    let mut current = claim;
    let mut challenges = Vec::with_capacity(num_vars);

    for (i, rp) in proof.round_polys.iter().enumerate() {
        // 各ラウンドの整合性チェック: g_i(0) + g_i(1) == current_claim
        if rp.evals[0] + rp.evals[1] != current {
            return Err(format!("Round {i} consistency failed"));
        }
        for e in &rp.evals {
            transcript.append_field(b"sc_round", e);
        }
        let r_i = transcript.challenge_field::<F>(b"sc_challenge");
        challenges.push(r_i);
        current = rp.evaluate(r_i);
    }

    // 最終チェック: Σ weights[k] * f_k(r) * g_k(r) == current_claim
    let mut final_combination = F::ZERO;
    for k in 0..proof.final_evals_f.len() {
        final_combination += weights[k] * proof.final_evals_f[k] * proof.final_evals_g[k];
    }

    if final_combination != current {
        return Err("Final batched check failed".into());
    }

    Ok((challenges, current))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::DenseMLPoly;
    use crate::transcript::Transcript;
    use ark_ff::UniformRand;
    use ark_ff::{One, Zero};
    use ark_std::test_rng;

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

    /// ランダムな多項式ペアを生成する
    fn random_polys(num_vars: usize) -> (DenseMLPoly, DenseMLPoly) {
        let mut rng = test_rng();
        let size = 1 << num_vars;
        let f_evals = (0..size).map(|_| F::rand(&mut rng)).collect();
        let g_evals = (0..size).map(|_| F::rand(&mut rng)).collect();
        (DenseMLPoly::new(f_evals), DenseMLPoly::new(g_evals))
    }

    // --- テストケース ---

    #[test]
    fn test_sumcheck_random_trials() {
        // 異なる変数個数で複数回のランダム試行を行い、一貫性を検証する
        let mut rng = test_rng();
        for num_vars in 1..8 {
            for _ in 0..10 {
                let (f, g) = random_polys(num_vars);
                let claim = compute_brute_force_sum(&f, &g);

                let mut prover_transcript = Transcript::new(b"comp_test");
                let (proof, challenges) = prove_sumcheck(&f, &g, claim, &mut prover_transcript);

                let mut verifier_transcript = Transcript::new(b"comp_test");
                let result = verify_sumcheck(&proof, claim, num_vars, &mut verifier_transcript);

                assert!(result.is_ok(), "Random trial failed for {} vars", num_vars);
                let (v_challenges, _) = result.unwrap();
                assert_eq!(challenges, v_challenges);
            }
        }
    }

    #[test]
    fn test_sumcheck_tamper_final_evaluations() {
        // ラウンド多項式は正しいが、最終的な f(r) や g(r) を偽った場合に検知できるか
        let num_vars = 3;
        let (f, g) = random_polys(num_vars);
        let claim = compute_brute_force_sum(&f, &g);

        let mut pt = Transcript::new(b"tamper_final");
        let (mut proof, _) = prove_sumcheck(&f, &g, claim, &mut pt);

        // f(r) を改ざん
        proof.final_eval_f += F::from(1u64);

        let mut vt = Transcript::new(b"tamper_final");
        let result = verify_sumcheck(&proof, claim, num_vars, &mut vt);

        // 最終チェック: f(r)*g(r) == current_claim で失敗するはず
        assert!(result.is_err(), "Should have caught tampered final_eval_f");
    }

    #[test]
    fn test_sumcheck_transcript_mismatch() {
        // プロverと検証者のラベル（コンテキスト）が異なる場合に失敗するか（Fiat-Shamirの安全性）
        let num_vars = 2;
        let (f, g) = random_polys(num_vars);
        let claim = compute_brute_force_sum(&f, &g);

        let mut pt = Transcript::new(b"context_A");
        let (proof, _) = prove_sumcheck(&f, &g, claim, &mut pt);

        let mut vt = Transcript::new(b"context_B"); // ラベルが異なる
        let result = verify_sumcheck(&proof, claim, num_vars, &mut vt);

        // チャレンジの値が変わるため、整合性チェックのどこかで失敗する
        assert!(
            result.is_err(),
            "Should fail due to transcript label mismatch"
        );
    }

    #[test]
    fn test_sumcheck_high_variable_count() {
        // 変数が多い（例: 12変数 = 4096エントリ）場合のパフォーマンスと再帰の深さを確認
        let num_vars = 12;
        let (f, g) = random_polys(num_vars);
        let claim = compute_brute_force_sum(&f, &g);

        let mut pt = Transcript::new(b"high_vars");
        let (proof, _) = prove_sumcheck(&f, &g, claim, &mut pt);

        let mut vt = Transcript::new(b"high_vars");
        let result = verify_sumcheck(&proof, claim, num_vars, &mut vt);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sumcheck_tamper_middle_round() {
        // 最初ではなく、中間（例: 第2ラウンド）の多項式評価を改ざんした場合
        let num_vars = 4;
        let (f, g) = random_polys(num_vars);
        let claim = compute_brute_force_sum(&f, &g);

        let mut pt = Transcript::new(b"tamper_middle");
        let (mut proof, _) = prove_sumcheck(&f, &g, claim, &mut pt);

        // 第2ラウンドの g_1(2) を改ざん
        proof.round_polys[1].evals[2] *= F::from(2u64);

        let mut vt = Transcript::new(b"tamper_middle");
        let result = verify_sumcheck(&proof, claim, num_vars, &mut vt);

        // 改ざんされたラウンドの次のラウンドの整合性チェックで失敗するはず
        assert!(
            result.is_err(),
            "Should fail at the round following the tampered one"
        );
    }

    #[test]
    fn test_sumcheck_constant_polynomials() {
        // 多項式が定数の場合（次数が低いケースの境界値テスト）
        let num_vars = 3;
        let f = DenseMLPoly::new(vec![F::from(5u64); 1 << num_vars]);
        let g = DenseMLPoly::new(vec![F::from(10u64); 1 << num_vars]);
        let claim = F::from((5 * 10 * (1 << num_vars)) as u64);

        let mut pt = Transcript::new(b"const");
        let (proof, _) = prove_sumcheck(&f, &g, claim, &mut pt);

        let mut vt = Transcript::new(b"const");
        let result = verify_sumcheck(&proof, claim, num_vars, &mut vt);
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod multi_sumcheck_tests {
    use super::*;
    use crate::poly::DenseMLPoly;
    use crate::transcript::Transcript;
    use ark_ff::{One, UniformRand, Zero};
    use ark_std::test_rng;

    /// 複数の多項式ペアの積の総和を計算するヘルパー
    fn compute_brute_force_sum_multi(fs: &[DenseMLPoly], gs: &[DenseMLPoly], weights: &[F]) -> F {
        let mut total = F::ZERO;
        for k in 0..fs.len() {
            let pair_sum: F = fs[k]
                .evaluations
                .iter()
                .zip(gs[k].evaluations.iter())
                .map(|(&fa, &ga)| fa * ga)
                .sum();
            total += weights[k] * pair_sum;
        }
        total
    }

    /// 正常系: 2つの多項式ペア、3変数のケース
    #[test]
    fn test_multi_sumcheck_happy_path() {
        let mut rng = test_rng();
        let n = 3;
        let num_pairs = 2;

        let fs = vec![
            DenseMLPoly::new((0..1 << n).map(|_| F::rand(&mut rng)).collect()),
            DenseMLPoly::new((0..1 << n).map(|_| F::rand(&mut rng)).collect()),
        ];
        let gs = vec![
            DenseMLPoly::new((0..1 << n).map(|_| F::rand(&mut rng)).collect()),
            DenseMLPoly::new((0..1 << n).map(|_| F::rand(&mut rng)).collect()),
        ];
        let weights = vec![F::rand(&mut rng), F::rand(&mut rng)];

        let claim = compute_brute_force_sum_multi(&fs, &gs, &weights);

        // Prover
        let mut p_transcript = Transcript::new(b"multi_test");
        let (proof, challenges) =
            prove_sumcheck_multi_batched(&fs, &gs, &weights, claim, &mut p_transcript);

        // Verifier
        let mut v_transcript = Transcript::new(b"multi_test");
        let result = verify_sumcheck_multi_batched(&proof, &weights, claim, n, &mut v_transcript);

        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
        let (v_challenges, _) = result.unwrap();
        assert_eq!(challenges, v_challenges);

        // 各多項式の最終評価値の正当性確認
        for k in 0..num_pairs {
            assert_eq!(proof.final_evals_f[k], fs[k].evaluate(&challenges));
            assert_eq!(proof.final_evals_g[k], gs[k].evaluate(&challenges));
        }
    }

    /// 異常系: プロverが偽った最終評価値を送った場合
    #[test]
    fn test_multi_sumcheck_tampered_final_eval() {
        let mut rng = test_rng();
        let n = 2;
        let (fs, gs) = (
            vec![DenseMLPoly::new(vec![F::rand(&mut rng); 4])],
            vec![DenseMLPoly::new(vec![F::rand(&mut rng); 4])],
        );
        let weights = vec![F::one()];
        let claim = compute_brute_force_sum_multi(&fs, &gs, &weights);

        let mut p_pt = Transcript::new(b"tamper");
        let (mut proof, _) = prove_sumcheck_multi_batched(&fs, &gs, &weights, claim, &mut p_pt);

        // 最終評価値の一つを改ざん
        proof.final_evals_f[0] += F::one();

        let mut v_vt = Transcript::new(b"tamper");
        let result = verify_sumcheck_multi_batched(&proof, &weights, claim, n, &mut v_vt);
        assert!(result.is_err(), "Should catch tampered final evaluation");
    }

    /// 異常系: 重み(weights)がプロverと検証者で異なる場合
    #[test]
    fn test_multi_sumcheck_weight_mismatch() {
        let mut rng = test_rng();
        let n = 2;
        let fs = vec![
            DenseMLPoly::new(vec![F::one(); 4]),
            DenseMLPoly::new(vec![F::one(); 4]),
        ];
        let gs = vec![
            DenseMLPoly::new(vec![F::one(); 4]),
            DenseMLPoly::new(vec![F::one(); 4]),
        ];

        let p_weights = vec![F::from(2), F::from(3)];
        let v_weights = vec![F::from(2), F::from(4)]; // 重みが異なる

        let claim = compute_brute_force_sum_multi(&fs, &gs, &p_weights);

        let mut pt = Transcript::new(b"weight_test");
        let (proof, _) = prove_sumcheck_multi_batched(&fs, &gs, &p_weights, claim, &mut pt);

        let mut vt = Transcript::new(b"weight_test");
        let result = verify_sumcheck_multi_batched(&proof, &v_weights, claim, n, &mut vt);
        assert!(result.is_err(), "Should fail due to weight mismatch");
    }

    /// 境界値: 多項式が1組のみの場合 (単一Sumcheckと等価な動作を確認)
    #[test]
    fn test_multi_sumcheck_single_pair_edge_case() {
        let n = 2;
        let f = DenseMLPoly::new(vec![F::from(1), F::from(2), F::from(3), F::from(4)]);
        let g = DenseMLPoly::new(vec![F::from(1), F::from(1), F::from(1), F::from(1)]);
        let weights = vec![F::one()];
        let claim = F::from(10); // 1+2+3+4

        let mut pt = Transcript::new(b"edge");
        let (proof, _) = prove_sumcheck_multi_batched(&[f], &[g], &weights, claim, &mut pt);

        let mut vt = Transcript::new(b"edge");
        let result = verify_sumcheck_multi_batched(&proof, &weights, claim, n, &mut vt);
        assert!(result.is_ok());
    }

    /// 境界値: 全てがゼロ多項式のケース
    #[test]
    fn test_multi_sumcheck_zero_polynomials() {
        let n = 2;
        let fs = vec![DenseMLPoly::zero(n), DenseMLPoly::zero(n)];
        let gs = vec![DenseMLPoly::zero(n), DenseMLPoly::zero(n)];
        let weights = vec![F::rand(&mut test_rng()), F::rand(&mut test_rng())];
        let claim = F::zero();

        let mut pt = Transcript::new(b"zero");
        let (proof, _) = prove_sumcheck_multi_batched(&fs, &gs, &weights, claim, &mut pt);

        let mut vt = Transcript::new(b"zero");
        let result = verify_sumcheck_multi_batched(&proof, &weights, claim, n, &mut vt);
        assert!(result.is_ok());
    }
}
