//! Hyrax-style polynomial commitment scheme (transparent, no trusted setup).
//!
//! For a multilinear polynomial with 2^n = 2^(nu+sigma) evaluations, arranged
//! as a 2^nu × 2^sigma matrix M:
//!
//!   **Commit:** C_i = MSM(gens, M[i])  for each row i ∈ [2^nu]
//!
//!   **Open** at point r = (r_L || r_R)  (r_L ∈ F^nu, r_R ∈ F^sigma):
//!       w'_j = Σ_i L_i · M[i][j]   where L = lagrange_basis(r_L)
//!
//!   **Verify:**
//!       1. Σ_i L_i · C_i  ==  MSM(gens, w')   (homomorphic commitment check)
//!       2. inner(R, w')    ==  claimed_eval     where R = lagrange_basis(r_R)
//!
//! Soundness relies on the discrete-log hardness of BN254 G1 and the binding
//! property of the vector commitments (generators have unknown discrete logs
//! relative to each other, derived via hash-and-multiply).

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{Group, VariableBaseMSM};
use ark_ff::{Field, PrimeField, Zero};
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::{field::F, poly::DenseMLPoly, transcript::Transcript};

// ---------------------------------------------------------------------------
// Global HyraxParams cache: HyraxParams::new(sigma) is expensive (2^sigma G1
// scalar muls). Cache by sigma so setup runs at most once per distinct sigma.
// ---------------------------------------------------------------------------
static HYRAX_PARAMS_CACHE: OnceLock<Mutex<HashMap<usize, HyraxParams>>> = OnceLock::new();

fn cached_hyrax_params(sigma: usize) -> HyraxParams {
    let cache = HYRAX_PARAMS_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    map.entry(sigma)
        .or_insert_with(|| HyraxParams::new(sigma))
        .clone()
}

// ---------------------------------------------------------------------------
// Public parameters
// ---------------------------------------------------------------------------

/// Hyrax public parameters: 2^sigma independent G1 generators.
///
/// Each generator g_j is derived as h_j · G where h_j = SHA3("piformer-hyrax" || j)
/// reduced mod the group order, giving "nothing-up-my-sleeve" independence.
#[derive(Clone)]
pub struct HyraxParams {
    pub gens: Vec<G1Affine>,
    pub sigma: usize,
}

impl HyraxParams {
    /// Generate parameters for polynomials with `sigma` column-variables.
    /// The number of generators is 2^sigma.
    pub fn new(sigma: usize) -> Self {
        let num_cols = 1usize << sigma;
        let base = G1Projective::generator();
        let gens = (0..num_cols)
            .map(|i| {
                let mut h = Sha3_256::new();
                h.update(b"piformer-hyrax-gen");
                h.update(&(i as u64).to_le_bytes());
                let bytes = h.finalize();
                let s = Fr::from_le_bytes_mod_order(&bytes);
                (base * s).into()
            })
            .collect();
        HyraxParams { gens, sigma }
    }
}

// ---------------------------------------------------------------------------
// Commitment
// ---------------------------------------------------------------------------

/// A Hyrax commitment: one G1 point per row of the evaluation matrix.
#[derive(Clone)]
pub struct HyraxCommitment {
    pub row_coms: Vec<G1Affine>,
    pub nu: usize,
    pub sigma: usize,
}

/// Commit to a multilinear polynomial given as a flat evaluation vector.
///
/// `evals` must have length 2^(nu+sigma).
pub fn hyrax_commit(evals: &[F], nu: usize, params: &HyraxParams) -> HyraxCommitment {
    let sigma = params.sigma;
    let num_rows = 1 << nu;
    let num_cols = 1 << sigma;
    assert_eq!(evals.len(), num_rows * num_cols, "eval length mismatch");

    let row_coms = (0..num_rows)
        .into_par_iter()
        .map(|i| msm(&params.gens, &evals[i * num_cols..(i + 1) * num_cols]))
        .collect();

    HyraxCommitment {
        row_coms,
        nu,
        sigma,
    }
}

// ---------------------------------------------------------------------------
// Opening
// ---------------------------------------------------------------------------

/// An opening proof: the intermediate vector w' of length 2^sigma.
#[derive(Clone, Debug)]
pub struct HyraxProof {
    pub w_prime: Vec<F>,
}

/// Generate an opening proof for evaluation at `point` (length nu + sigma).
///
/// `evals` must have length 2^(nu+sigma) and match what was committed.
pub fn hyrax_open(evals: &[F], point: &[F], nu: usize, sigma: usize) -> HyraxProof {
    let num_cols = 1 << sigma;
    assert_eq!(point.len(), nu + sigma, "point dimension mismatch");
    assert_eq!(
        evals.len(),
        (1 << nu) * num_cols,
        "eval length mismatch in open"
    );

    // DenseMLPoly::fix_first_variable processes challenges MSB-first (r[0] → bit_{n-1}).
    // Our lagrange_basis assigns r[k] to bit_k(i) (LSB-first).
    // To reconcile: reverse the sub-challenges so r[0] lands on bit_0.
    let r_l_rev: Vec<F> = point[..nu].iter().rev().copied().collect();
    let l_vec = lagrange_basis(&r_l_rev);

    let w_prime: Vec<F> = (0..num_cols)
        .into_par_iter()
        .map(|j| {
            l_vec
                .iter()
                .enumerate()
                .map(|(i, &l_i)| l_i * evals[i * num_cols + j])
                .sum()
        })
        .collect();

    HyraxProof { w_prime }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Verify a Hyrax opening proof.
///
/// Checks:
///   1. Σ_i L_i · C_i  ==  MSM(gens, w')
///   2. inner(R, w')    ==  eval
pub fn hyrax_verify(
    commitment: &HyraxCommitment,
    eval: F,
    point: &[F],
    proof: &HyraxProof,
    params: &HyraxParams,
) -> Result<(), String> {
    let nu = commitment.nu;
    let sigma = commitment.sigma;
    assert_eq!(point.len(), nu + sigma, "point dimension mismatch");

    // Same reversal as in hyrax_open: DenseMLPoly is MSB-first, lagrange_basis is LSB-first.
    let r_l_rev: Vec<F> = point[..nu].iter().rev().copied().collect();
    let r_r_rev: Vec<F> = point[nu..].iter().rev().copied().collect();
    let l_vec = lagrange_basis(&r_l_rev);
    let r_vec = lagrange_basis(&r_r_rev);

    // Check 1: homomorphic commitment combination
    let lhs = msm_g1(&commitment.row_coms, &l_vec);
    let rhs = msm(&params.gens, &proof.w_prime);
    if lhs != rhs {
        return Err("Hyrax: commitment check failed".to_string());
    }

    // Check 2: inner product
    let inner: F = r_vec
        .iter()
        .zip(proof.w_prime.iter())
        .map(|(&r, &w)| r * w)
        .sum();
    if inner != eval {
        return Err(format!(
            "Hyrax: inner product check failed: got {inner:?}, expected {eval:?}"
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Batching (Multiple polynomials at the same point)
// ---------------------------------------------------------------------------

/// 複数のコミットメントを同一の点 `point` で一括して証明する。
///
/// `evals_list`: 各多項式の評価ベクトル（長さ 2^(nu+sigma)）のリスト
pub fn hyrax_open_batch(
    evals_list: &[&[F]],
    point: &[F],
    nu: usize,
    sigma: usize,
    transcript: &mut Transcript,
) -> HyraxProof {
    let count = evals_list.len();
    let num_cols = 1 << sigma;

    // 1. バッチ化用のチャレンジ η を生成
    let eta = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_pows = powers_of(eta, count);

    // 2. ラグランジュ基底の計算 (row部分)
    let r_l_rev: Vec<F> = point[..nu].iter().rev().copied().collect();
    let l_vec = lagrange_basis(&r_l_rev);

    // 3. 統合された w' ベクトルを計算: w'_batch = Σ_k η^k * (Σ_i L_i * Row_{k,i})
    let mut w_prime_batched = vec![F::ZERO; num_cols];
    for (k, evals) in evals_list.iter().enumerate() {
        let eta_k = eta_pows[k];
        for (i, &l_i) in l_vec.iter().enumerate() {
            let row = &evals[i * num_cols..(i + 1) * num_cols];
            let coeff = eta_k * l_i;
            for (j, &m_ij) in row.iter().enumerate() {
                w_prime_batched[j] += coeff * m_ij;
            }
        }
    }

    HyraxProof {
        w_prime: w_prime_batched,
    }
}

/// 複数の Hyrax 証明を一括で検証する。
pub fn hyrax_verify_batch(
    commitments: &[HyraxCommitment],
    evals: &[F],
    point: &[F],
    proof: &HyraxProof,
    params: &HyraxParams,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let count = commitments.len();
    assert_eq!(evals.len(), count);
    let nu = commitments[0].nu;
    let sigma = commitments[0].sigma;

    // 1. チャレンジ η の再現
    let eta = transcript.challenge_field::<F>(b"hyrax_batch_eta");
    let eta_pows = powers_of(eta, count);

    // 2. ラグランジュ基底の計算
    let r_l_rev: Vec<F> = point[..nu].iter().rev().copied().collect();
    let r_r_rev: Vec<F> = point[nu..].iter().rev().copied().collect();
    let l_vec = lagrange_basis(&r_l_rev);
    let r_vec = lagrange_basis(&r_r_rev);

    // 3. Check 1: Σ_k Σ_i (η^k · L_i) · C_{k,i} == MSM(gens, w'_batch)
    //
    // Old approach: K×num_rows full G1 scalar muls (254-bit each, ~60 µs each).
    // New approach: K×num_rows cheap *field* multiplications, then ONE big MSM.
    //   scalar(k,i) = η^k · L_i  (field mul, ~50 ns)
    //   lhs = MSM over all K×num_rows row_coms with these scalars
    let num_rows = 1usize << nu;
    let mut all_row_points: Vec<G1Affine> = Vec::with_capacity(count * num_rows);
    let mut all_row_scalars: Vec<F> = Vec::with_capacity(count * num_rows);
    for (k, com) in commitments.iter().enumerate() {
        for (i, &row_com) in com.row_coms.iter().enumerate() {
            all_row_points.push(row_com);
            all_row_scalars.push(eta_pows[k] * l_vec[i]);
        }
    }
    let lhs = msm(&all_row_points, &all_row_scalars);
    let rhs = msm(&params.gens, &proof.w_prime);
    if lhs != rhs {
        return Err("Hyrax Batch: commitment check failed".to_string());
    }

    // 4. 内積の検証 (統合された評価値との比較)
    // Check 2: <R, w'_batch> == Σ_k η^k * eval_k
    let inner: F = r_vec
        .iter()
        .zip(proof.w_prime.iter())
        .map(|(&r, &w)| r * w)
        .sum();
    let expected_inner: F = evals
        .iter()
        .zip(eta_pows.iter())
        .map(|(&v, &e)| v * e)
        .sum();

    if inner != expected_inner {
        return Err(format!(
            "Hyrax Batch: inner product check failed: got {inner:?}, expected {expected_inner:?}"
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Multi-point batched verification (K independent openings → 2 MSMs total)
// ---------------------------------------------------------------------------

/// Batch-verify K independent Hyrax openings at potentially different points
/// using a single Fiat-Shamir random linear combination.
///
/// Instead of K×2 MSMs (one lhs + one rhs each), we do:
///   - 1 combined lhs MSM  over all K×2^nu row_coms
///   - 1 combined rhs MSM  over 2^sigma generators
///   - K inner-product checks (field ops only, O(2^sigma) each)
///
/// `entries`: slice of (commitment, claimed_eval, point, proof) tuples.
/// All commitments must share the same `params` (same sigma/generators).
pub fn hyrax_verify_multi_point(
    entries: &[(&HyraxCommitment, F, &[F], &HyraxProof)],
    params: &HyraxParams,
    transcript: &mut Transcript,
) -> Result<(), String> {
    if entries.is_empty() {
        return Ok(());
    }
    let count = entries.len();
    let nu = entries[0].0.nu;
    let sigma = entries[0].0.sigma;
    let num_rows = 1usize << nu;
    let num_cols = 1usize << sigma;

    // 1. Random linear combination challenge
    let lambda = transcript.challenge_field::<F>(b"hyrax_mp_lambda");
    let lambda_pows = powers_of(lambda, count);

    // 2. Combined rhs scalar vector: w'_combined[j] = Σ_i λ^i * w'_i[j]
    let mut w_prime_combined = vec![F::ZERO; num_cols];
    for (i, &(_, _, _, proof)) in entries.iter().enumerate() {
        let lam_i = lambda_pows[i];
        for (j, &w) in proof.w_prime.iter().enumerate() {
            w_prime_combined[j] += lam_i * w;
        }
    }
    let rhs = msm(&params.gens, &w_prime_combined);

    // 3. Combined lhs: MSM over all K*num_rows row_coms
    //    scalar(i,j) = λ^i * l_vec_i[j]
    let mut all_points: Vec<G1Affine> = Vec::with_capacity(count * num_rows);
    let mut all_scalars: Vec<F> = Vec::with_capacity(count * num_rows);
    for (i, &(com, _, point, _)) in entries.iter().enumerate() {
        let r_l_rev: Vec<F> = point[..nu].iter().rev().copied().collect();
        let l_vec = lagrange_basis(&r_l_rev);
        let lam_i = lambda_pows[i];
        for (j, &row_com) in com.row_coms.iter().enumerate() {
            all_points.push(row_com);
            all_scalars.push(lam_i * l_vec[j]);
        }
    }
    let lhs = msm(&all_points, &all_scalars);

    if lhs != rhs {
        return Err("Hyrax multi-point: commitment check failed".to_string());
    }

    // 4. Inner product checks (cheap field ops, no MSM)
    for (i, &(_, eval, point, proof)) in entries.iter().enumerate() {
        let r_r_rev: Vec<F> = point[nu..].iter().rev().copied().collect();
        let r_vec = lagrange_basis(&r_r_rev);
        let inner: F = r_vec
            .iter()
            .zip(proof.w_prime.iter())
            .map(|(&r, &w)| r * w)
            .sum();
        if inner != eval {
            return Err(format!(
                "Hyrax multi-point: inner product check failed at entry {i}: got {inner:?}, expected {eval:?}"
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Additional Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// 指定された要素の累乗 [1, x, x^2, ..., x^{n-1}] を計算する。
pub fn powers_of(x: F, n: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(n);
    let mut cur = F::ONE;
    for _ in 0..n {
        res.push(cur);
        cur *= x;
    }
    res
}

/// Compute the 2^n multilinear Lagrange basis evaluations for a given point.
///
/// `basis[i] = Π_k ( bit_k(i) · r[k] + (1 − bit_k(i)) · (1 − r[k]) )`
///
/// This is consistent with `DenseMLPoly::evaluate`: both compute the same MLE.
pub fn lagrange_basis(point: &[F]) -> Vec<F> {
    let mut table = vec![F::ONE];
    for &r in point {
        let half = table.len();
        let mut new = vec![F::ZERO; half * 2];
        for (i, &v) in table.iter().enumerate() {
            new[i] = v * (F::ONE - r);
            new[i + half] = v * r;
        }
        table = new;
    }
    table
}

/// MSM with G1Affine bases and F scalars.
fn msm(bases: &[G1Affine], scalars: &[F]) -> G1Affine {
    assert_eq!(bases.len(), scalars.len());
    if bases.is_empty() {
        return G1Affine::identity();
    }
    let bigints: Vec<_> = scalars.iter().map(|s| s.into_bigint()).collect();
    <G1Projective as VariableBaseMSM>::msm_bigint(bases, &bigints).into()
}

/// MSM where scalars are already F field elements (same as `msm` but avoids confusion).
fn msm_g1(bases: &[G1Affine], scalars: &[F]) -> G1Affine {
    msm(bases, scalars)
}

/// Convenience constructor: create HyraxParams sized for a table with
/// `bits_per_chunk` variables (table size = 2^bits_per_chunk).
///
/// Uses the standard Hyrax split: nu = bits_per_chunk/2, sigma = bits_per_chunk - nu.
pub fn setup_hyrax_params(bits_per_chunk: usize) -> HyraxParams {
    let nu = bits_per_chunk / 2;
    let sigma = bits_per_chunk - nu;
    cached_hyrax_params(sigma)
}

pub fn params_from_vars(total_vars: usize) -> (usize, usize, HyraxParams) {
    let nu = total_vars / 2;
    let sigma = (total_vars - nu).max(1);
    (nu, sigma, cached_hyrax_params(sigma))
}

pub fn poly_hyrax(poly: &DenseMLPoly) -> (usize, usize, HyraxParams) {
    // 修正: next_power_of_two() などを介さず、変数の数をそのまま渡す
    params_from_vars(poly.num_vars)
}

// ※layernorm.rs と linear.rs にある params_from_n は以下のようにします
pub fn params_from_n(n: usize) -> (usize, usize, HyraxParams) {
    let total_vars = n.trailing_zeros() as usize;
    params_from_vars(total_vars)
}

pub fn absorb_com(transcript: &mut Transcript, label: &[u8], com: &HyraxCommitment) {
    use ark_serialize::CanonicalSerialize;
    for pt in &com.row_coms {
        let mut buf = Vec::new();
        pt.serialize_compressed(&mut buf).unwrap();
        transcript.append_bytes(label, &buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::{test_rng, UniformRand};

    /// Tests a complete workflow of the Hyrax commitment scheme:
    /// 1. Parameters generation
    /// 2. Polynomial commitment
    /// 3. Evaluation and proof generation
    /// 4. Successful verification
    /// 5. Verification failure on malicious input
    #[test]
    fn test_hyrax_full_flow() {
        let mut rng = test_rng();

        // Let nu = 2, sigma = 3 (Total variables = 5, Table size = 32)
        let nu = 2;
        let sigma = 3;
        let params = HyraxParams::new(sigma);

        // Generate a random multilinear polynomial (evaluations on the Boolean hypercube)
        let num_evals = 1 << (nu + sigma);
        let evals: Vec<Fr> = (0..num_evals).map(|_| Fr::rand(&mut rng)).collect();

        // 1. Commit
        let commitment = hyrax_commit(&evals, nu, &params);

        // 2. Choose a random evaluation point
        let point: Vec<Fr> = (0..(nu + sigma)).map(|_| Fr::rand(&mut rng)).collect();

        // Calculate the ground truth evaluation using a naive MLE evaluation
        // for verification of the test itself.
        let full_lagrange = lagrange_basis(&point.iter().rev().copied().collect::<Vec<_>>());
        let expected_eval: Fr = evals
            .iter()
            .zip(full_lagrange.iter())
            .map(|(e, l)| *e * l)
            .sum();

        // 3. Open
        let proof = hyrax_open(&evals, &point, nu, sigma);

        // 4. Verify (Success)
        let result = hyrax_verify(&commitment, expected_eval, &point, &proof, &params);
        assert!(
            result.is_ok(),
            "Verification should pass: {:?}",
            result.err()
        );

        // 5. Verify (Failure - Wrong Evaluation)
        let fake_eval = expected_eval + Fr::from(1u64);
        let result_fail = hyrax_verify(&commitment, fake_eval, &point, &proof, &params);
        assert!(
            result_fail.is_err(),
            "Verification should fail with wrong evaluation"
        );

        // 6. Verify (Failure - Wrong Proof)
        let mut malicious_proof = proof.clone();
        malicious_proof.w_prime[0] += Fr::from(1u64);
        let result_fail_proof = hyrax_verify(
            &commitment,
            expected_eval,
            &point,
            &malicious_proof,
            &params,
        );
        assert!(
            result_fail_proof.is_err(),
            "Verification should fail with corrupted proof"
        );
    }

    #[test]
    fn test_lagrange_basis_sum_to_one() {
        let mut rng = test_rng();
        let point: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
        let basis = lagrange_basis(&point);
        let sum: Fr = basis.iter().sum();
        assert_eq!(sum, Fr::ONE, "Lagrange basis evaluations must sum to 1");
    }

    #[test]
    fn test_lagrange_basis_on_hypercube_is_indicator() {
        // At a binary point, exactly one basis element is 1 and all others 0.
        let n = 3;
        for idx in 0..(1usize << n) {
            let point: Vec<Fr> = (0..n)
                .map(|k| {
                    if (idx >> k) & 1 == 1 {
                        Fr::ONE
                    } else {
                        Fr::ZERO
                    }
                })
                .collect();
            let basis = lagrange_basis(&point);
            for (i, &b) in basis.iter().enumerate() {
                if i == idx {
                    assert_eq!(b, Fr::ONE, "basis[{i}] should be 1 at binary point {idx}");
                } else {
                    assert_eq!(b, Fr::ZERO, "basis[{i}] should be 0 at binary point {idx}");
                }
            }
        }
    }

    #[test]
    fn test_hyrax_commit_open_verify_zero_poly() {
        // Committing to the all-zero polynomial should still verify correctly.
        let nu = 2;
        let sigma = 2;
        let params = HyraxParams::new(sigma);
        let evals = vec![Fr::ZERO; 1 << (nu + sigma)];

        let commitment = hyrax_commit(&evals, nu, &params);

        let mut rng = test_rng();
        let point: Vec<Fr> = (0..(nu + sigma)).map(|_| Fr::rand(&mut rng)).collect();
        let proof = hyrax_open(&evals, &point, nu, sigma);
        let result = hyrax_verify(&commitment, Fr::ZERO, &point, &proof, &params);
        assert!(
            result.is_ok(),
            "Zero polynomial should verify: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_hyrax_nu_zero_single_row() {
        // nu=0 means the polynomial is a single row (no row variables).
        let nu = 0;
        let sigma = 3;
        let params = HyraxParams::new(sigma);
        let mut rng = test_rng();
        let evals: Vec<Fr> = (0..(1 << sigma)).map(|_| Fr::rand(&mut rng)).collect();

        let commitment = hyrax_commit(&evals, nu, &params);
        assert_eq!(
            commitment.row_coms.len(),
            1,
            "nu=0 should give a single row commitment"
        );

        let point: Vec<Fr> = (0..sigma).map(|_| Fr::rand(&mut rng)).collect();
        let proof = hyrax_open(&evals, &point, nu, sigma);

        // Compute expected eval manually
        let basis = lagrange_basis(&point.iter().rev().copied().collect::<Vec<_>>());
        let expected: Fr = evals.iter().zip(basis.iter()).map(|(e, l)| *e * l).sum();

        let result = hyrax_verify(&commitment, expected, &point, &proof, &params);
        assert!(
            result.is_ok(),
            "nu=0 verify should pass: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_hyrax_eval_matches_dense_ml_poly() {
        // Cross-check: Hyrax opening eval must match DenseMLPoly::evaluate.
        use crate::poly::DenseMLPoly;
        let nu = 2;
        let sigma = 2;
        let params = HyraxParams::new(sigma);
        let mut rng = test_rng();
        let evals: Vec<Fr> = (0..(1 << (nu + sigma)))
            .map(|_| Fr::rand(&mut rng))
            .collect();

        let poly = DenseMLPoly::new(evals.clone());
        let point: Vec<Fr> = (0..(nu + sigma)).map(|_| Fr::rand(&mut rng)).collect();

        let expected = poly.evaluate(&point);
        let proof = hyrax_open(&evals, &point, nu, sigma);
        let commitment = hyrax_commit(&evals, nu, &params);

        let result = hyrax_verify(&commitment, expected, &point, &proof, &params);
        assert!(
            result.is_ok(),
            "Hyrax eval should match DenseMLPoly eval: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_params_from_vars_even() {
        let (nu, sigma, params) = params_from_vars(6);
        assert_eq!(nu, 3);
        assert_eq!(sigma, 3);
        assert_eq!(params.sigma, 3);
        assert_eq!(params.gens.len(), 1 << 3);
    }

    #[test]
    fn test_params_from_vars_odd() {
        let (nu, sigma, params) = params_from_vars(5);
        assert_eq!(nu, 2);
        assert_eq!(sigma, 3);
        assert_eq!(params.sigma, 3);
    }

    #[test]
    fn test_setup_hyrax_params_produces_correct_sigma() {
        for bpc in [2, 4, 6, 8] {
            let params = setup_hyrax_params(bpc);
            let expected_sigma = bpc - bpc / 2;
            assert_eq!(params.sigma, expected_sigma, "bits_per_chunk={bpc}");
            assert_eq!(params.gens.len(), 1 << expected_sigma);
        }
    }
}

#[cfg(test)]
mod batched_hyrax_tests {
    use super::*;
    use crate::transcript::Transcript;
    use ark_std::{test_rng, UniformRand};

    /// 正常系: 複数のランダムな多項式を同一の点で一括検証する
    #[test]
    fn test_hyrax_batch_verify_success() {
        let mut rng = test_rng();
        let nu = 2;
        let sigma = 3;
        let num_polys = 5; // 5つの多項式を同時に証明
        let params = HyraxParams::new(sigma);
        let point: Vec<Fr> = (0..(nu + sigma)).map(|_| Fr::rand(&mut rng)).collect();

        let mut evals_list = Vec::new();
        let mut commitments = Vec::new();
        let mut claimed_evals = Vec::new();

        // 1. 各多項式の生成、コミット、および個別評価値の計算
        for _ in 0..num_polys {
            let num_evals = 1 << (nu + sigma);
            let evals: Vec<Fr> = (0..num_evals).map(|_| Fr::rand(&mut rng)).collect();
            let commitment = hyrax_commit(&evals, nu, &params);

            // 地点 point における真の評価値を計算 (MLE)
            let poly = DenseMLPoly::new(evals.clone());
            let eval = poly.evaluate(&point);

            evals_list.push(evals);
            commitments.push(commitment);
            claimed_evals.push(eval);
        }

        // 借用のための変換
        let evals_refs: Vec<&[Fr]> = evals_list.iter().map(|v| v.as_slice()).collect();

        // 2. プロver: バッチ証明の生成
        let mut prover_transcript = Transcript::new(b"test_hyrax_batch");
        let proof = hyrax_open_batch(&evals_refs, &point, nu, sigma, &mut prover_transcript);

        // 3. 検証者: バッチ証明の検証
        let mut verifier_transcript = Transcript::new(b"test_hyrax_batch");
        let result = hyrax_verify_batch(
            &commitments,
            &claimed_evals,
            &point,
            &proof,
            &params,
            &mut verifier_transcript,
        );

        assert!(
            result.is_ok(),
            "Batch verification should pass: {:?}",
            result.err()
        );
    }

    /// 異常系: 1つの評価値が間違っている場合に拒否されるか
    #[test]
    fn test_hyrax_batch_verify_tampered_eval() {
        let mut rng = test_rng();
        let (nu, sigma) = (2, 2);
        let params = HyraxParams::new(sigma);
        let num_polys = 3;
        let point: Vec<Fr> = (0..(nu + sigma)).map(|_| Fr::rand(&mut rng)).collect();

        let mut evals_list = Vec::new();
        let mut commitments = Vec::new();
        let mut claimed_evals = Vec::new();

        for _ in 0..num_polys {
            let evals: Vec<Fr> = (0..(1 << (nu + sigma)))
                .map(|_| Fr::rand(&mut rng))
                .collect();
            commitments.push(hyrax_commit(&evals, nu, &params));
            claimed_evals.push(DenseMLPoly::new(evals.clone()).evaluate(&point));
            evals_list.push(evals);
        }

        let evals_refs: Vec<&[Fr]> = evals_list.iter().map(|v| v.as_slice()).collect();
        let mut pt = Transcript::new(b"tamper_eval");
        let proof = hyrax_open_batch(&evals_refs, &point, nu, sigma, &mut pt);

        // 2番目の評価値を改ざん
        claimed_evals[1] += Fr::from(1u64);

        let mut vt = Transcript::new(b"tamper_eval");
        let result = hyrax_verify_batch(
            &commitments,
            &claimed_evals,
            &point,
            &proof,
            &params,
            &mut vt,
        );

        assert!(
            result.is_err(),
            "Should fail because one evaluation is incorrect"
        );
    }

    /// 異常系: フィアット・シャミールのチャレンジ (eta) が不一致の場合
    #[test]
    fn test_hyrax_batch_transcript_mismatch() {
        let mut rng = test_rng();
        let (nu, sigma) = (2, 2);
        let params = HyraxParams::new(sigma);

        // 2つの多項式を用意する
        let evals_0 = vec![Fr::rand(&mut rng); 16];
        let evals_1 = vec![Fr::rand(&mut rng); 16];
        let point = vec![Fr::rand(&mut rng); 4];

        let com_0 = hyrax_commit(&evals_0, nu, &params);
        let com_1 = hyrax_commit(&evals_1, nu, &params);

        let eval_0 = DenseMLPoly::new(evals_0.clone()).evaluate(&point);
        let eval_1 = DenseMLPoly::new(evals_1.clone()).evaluate(&point);

        // Proverはラベル "label_A" で証明生成
        let mut pt = Transcript::new(b"label_A");
        let proof = hyrax_open_batch(&[&evals_0, &evals_1], &point, nu, sigma, &mut pt);

        // Verifierは異なるラベル "label_B" で検証
        let mut vt = Transcript::new(b"label_B");
        let result = hyrax_verify_batch(
            &[com_0, com_1],
            &[eval_0, eval_1],
            &point,
            &proof,
            &params,
            &mut vt,
        );

        // チャレンジ η の値が異なるため、検証は失敗するはず
        assert!(
            result.is_err(),
            "Should fail due to transcript label mismatch"
        );
    }

    /// 境界値: 1つの多項式のみをバッチで扱う場合 (通常のHyraxと等価)
    #[test]
    fn test_hyrax_batch_single_poly() {
        let mut rng = test_rng();
        let (nu, sigma) = (2, 2);
        let params = HyraxParams::new(sigma);
        let evals = vec![Fr::rand(&mut rng); 16];
        let point = vec![Fr::rand(&mut rng); 4];
        let commitment = hyrax_commit(&evals, nu, &params);
        let eval = DenseMLPoly::new(evals.clone()).evaluate(&point);

        let mut pt = Transcript::new(b"single");
        let proof = hyrax_open_batch(&[&evals], &point, nu, sigma, &mut pt);

        let mut vt = Transcript::new(b"single");
        let result = hyrax_verify_batch(&[commitment], &[eval], &point, &proof, &params, &mut vt);

        assert!(result.is_ok(), "Single polynomial batch should work");
    }

    /// 異常系: コミットメントのリストが改ざん（順序入れ替え）された場合
    #[test]
    fn test_hyrax_batch_shuffled_commitments() {
        let mut rng = test_rng();
        let (nu, sigma) = (2, 2);
        let params = HyraxParams::new(sigma);
        let mut evals_list = vec![vec![Fr::rand(&mut rng); 16], vec![Fr::rand(&mut rng); 16]];
        let point = vec![Fr::rand(&mut rng); 4];

        let mut commitments: Vec<_> = evals_list
            .iter()
            .map(|e| hyrax_commit(e, nu, &params))
            .collect();
        let mut claimed_evals: Vec<_> = evals_list
            .iter()
            .map(|e| DenseMLPoly::new(e.clone()).evaluate(&point))
            .collect();

        let mut pt = Transcript::new(b"shuffle");
        let proof = hyrax_open_batch(
            &[&evals_list[0], &evals_list[1]],
            &point,
            nu,
            sigma,
            &mut pt,
        );

        // 検証時にコミットメントと評価値の順序を入れ替える
        commitments.reverse();
        claimed_evals.reverse();

        let mut vt = Transcript::new(b"shuffle");
        let result = hyrax_verify_batch(
            &commitments,
            &claimed_evals,
            &point,
            &proof,
            &params,
            &mut vt,
        );

        assert!(
            result.is_err(),
            "Should fail when order of commitments is inconsistent"
        );
    }
}
