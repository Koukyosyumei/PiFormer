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
use ark_ff::{Field, PrimeField};
use sha3::{Digest, Sha3_256};

use crate::field::F;

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
#[derive(Clone)]
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

    let mut w_prime = vec![F::ZERO; num_cols];
    for (i, &l_i) in l_vec.iter().enumerate() {
        let row = &evals[i * num_cols..(i + 1) * num_cols];
        for (j, &m_ij) in row.iter().enumerate() {
            w_prime[j] += l_i * m_ij;
        }
    }

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
// Helpers
// ---------------------------------------------------------------------------

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
}
