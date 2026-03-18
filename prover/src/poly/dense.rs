//! Dense multilinear polynomial over the Boolean hypercube {0,1}^n.
//!
//! Stored as a flat Vec of 2^n evaluations in little-endian lexicographic order:
//!   evaluations[i] = f( bit_0(i), bit_1(i), ..., bit_{n-1}(i) )
//! where bit_k(i) = (i >> k) & 1.
//!
//! The key operations needed for the sumcheck and Lasso protocols are:
//!   - evaluate(r): multilinear extension at an arbitrary point  (O(2^n))
//!   - fix_first_variable(r): reduce n-var poly to (n-1)-var poly (O(2^n))
//!   - eq_poly(r): construct the equality polynomial eq(r, ·)    (O(n · 2^n))

use ark_ff::Field;
use crate::field::F;

#[derive(Clone, Debug)]
pub struct DenseMLPoly {
    pub num_vars: usize,
    pub evaluations: Vec<F>,
}

impl DenseMLPoly {
    /// Construct from a vector of 2^n evaluations.
    pub fn new(evaluations: Vec<F>) -> Self {
        let n = evaluations.len();
        assert!(n.is_power_of_two(), "evaluations length must be a power of two");
        let num_vars = n.trailing_zeros() as usize;
        Self { num_vars, evaluations }
    }

    /// Construct the zero polynomial over `num_vars` variables.
    pub fn zero(num_vars: usize) -> Self {
        Self::new(vec![F::ZERO; 1 << num_vars])
    }

    /// Evaluate the multilinear extension at an arbitrary point r ∈ F^n
    /// via the "sumcheck streaming" (repeated halving) algorithm.
    /// Cost: O(2^n) field multiplications.
    pub fn evaluate(&self, r: &[F]) -> F {
        assert_eq!(r.len(), self.num_vars);
        let mut evals = self.evaluations.clone();
        let mut half = evals.len() >> 1;
        for &ri in r {
            for i in 0..half {
                evals[i] = evals[i] * (F::ONE - ri) + evals[i + half] * ri;
            }
            if half > 0 { half >>= 1; }
        }
        evals[0]
    }

    /// Fix the *first* variable to `r`, returning a poly over the remaining n-1 variables.
    /// The new evaluation layout is: new[i] = old[i]*(1-r) + old[i + half]*r.
    pub fn fix_first_variable(&self, r: F) -> Self {
        let half = self.evaluations.len() >> 1;
        let new_evals: Vec<F> = (0..half)
            .map(|i| self.evaluations[i] * (F::ONE - r) + self.evaluations[i + half] * r)
            .collect();
        Self::new(new_evals)
    }

    /// Sum all evaluations (= H = Σ_{x ∈ {0,1}^n} f(x), the claimed sumcheck sum).
    pub fn sum_over_hypercube(&self) -> F {
        self.evaluations.iter().copied().sum()
    }

    /// Construct the equality polynomial eq(r, ·) as a dense MLE over n variables:
    ///   eq(r, x) = Π_i ( r_i·x_i + (1-r_i)·(1-x_i) )
    /// Used to build the selector polynomial L for Lasso.
    pub fn eq_poly(r: &[F]) -> Self {
        let n = r.len();
        let size = 1usize << n;
        let mut evals = vec![F::ONE; size];
        for (j, &rj) in r.iter().enumerate() {
            for i in 0..size {
                let bit = F::from(((i >> j) & 1) as u64);
                evals[i] *= rj * bit + (F::ONE - rj) * (F::ONE - bit);
            }
        }
        Self::new(evals)
    }

    /// Point-wise product (Hadamard) — both polys must have the same num_vars.
    pub fn hadamard(&self, other: &DenseMLPoly) -> Self {
        assert_eq!(self.num_vars, other.num_vars);
        let evals = self.evaluations.iter()
            .zip(&other.evaluations)
            .map(|(&a, &b)| a * b)
            .collect();
        Self::new(evals)
    }

    /// Pad `v` to the next power of two by appending F::ZERO.
    pub fn from_vec_padded(mut v: Vec<F>) -> Self {
        let target = v.len().next_power_of_two().max(1);
        v.resize(target, F::ZERO);
        Self::new(v)
    }
}
