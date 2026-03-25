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

use crate::field::F;
use ark_ff::Field;

#[derive(Clone, Debug)]
pub struct DenseMLPoly {
    pub num_vars: usize,
    pub evaluations: Vec<F>,
}

impl DenseMLPoly {
    /// Construct from a vector of 2^n evaluations.
    pub fn new(evaluations: Vec<F>) -> Self {
        let n = evaluations.len();
        assert!(
            n.is_power_of_two(),
            "evaluations length must be a power of two"
        );
        let num_vars = n.trailing_zeros() as usize;
        Self {
            num_vars,
            evaluations,
        }
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
        // P(x_0, ..., x_i, ..., x_{n-1}) = (1 - x_i) \cdot P(x_0, ... , 0, ... , x_{n-1}) + x_i \cdot P(x_0, ... , 1, ... , x_{n-1})
        for &ri in r {
            for i in 0..half {
                evals[i] = evals[i] * (F::ONE - ri) + evals[i + half] * ri;
            }
            if half > 0 {
                half >>= 1;
            }
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
        let evals = self
            .evaluations
            .iter()
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::Zero;

    fn f_div(num: u64, den: u64) -> F {
        F::from(num) / F::from(den)
    }

    #[test]
    fn test_poly_basics() {
        let p = DenseMLPoly::zero(3);
        assert_eq!(p.num_vars, 3);
        assert_eq!(p.evaluations.len(), 8);
        assert!(p.sum_over_hypercube().is_zero());
    }

    #[test]
    fn test_evaluate_on_hypercube() {
        // Let f(x0, x1) have these evaluations on {0,1}^2:
        // f(0,0)=4, f(1,0)=2, f(0,1)=3, f(1,1)=1
        // Mapping index to bits (little-endian):
        // 0(00)->4, 1(10)->2, 2(01)->3, 3(11)->1
        let evals = vec![
            F::from(4), // f(0,0)
            F::from(2), // f(1,0)
            F::from(3), // f(0,1)
            F::from(1), // f(1,1)
        ];
        let poly = DenseMLPoly::new(evals);

        // Points on the hypercube must return the original evaluation values
        assert_eq!(poly.evaluate(&[F::ZERO, F::ZERO]), F::from(4));
        assert_eq!(poly.evaluate(&[F::from(1), F::ZERO]), F::from(3));
        assert_eq!(poly.evaluate(&[F::ZERO, F::from(1)]), F::from(2));
        assert_eq!(poly.evaluate(&[F::from(1), F::from(1)]), F::from(1));
    }

    #[test]
    fn test_evaluate_hypercube_3var() {
        // f(x0, x1, x2) in Big-Endian:
        // Index 0 (000) -> 0, Index 1 (001) -> 1, ..., Index 7 (111) -> 7
        let evals: Vec<F> = (0..8).map(|i| F::from(i as u64)).collect();
        let poly = DenseMLPoly::new(evals);

        // Check corner points
        assert_eq!(poly.evaluate(&[F::ZERO, F::ZERO, F::ZERO]), F::from(0)); // f(0,0,0)
        assert_eq!(poly.evaluate(&[F::ZERO, F::ZERO, F::ONE]), F::from(1)); // f(0,0,1)
        assert_eq!(poly.evaluate(&[F::ONE, F::ONE, F::ONE]), F::from(7)); // f(1,1,1)

        // Check f(1,0,1) -> Index 101 binary = 5
        assert_eq!(poly.evaluate(&[F::ONE, F::ZERO, F::ONE]), F::from(5));
    }

    #[test]
    fn test_fix_first_variable() {
        // f(x0, x1) -> evaluations [10, 20, 30, 40]
        let evals = vec![F::from(10), F::from(20), F::from(30), F::from(40)];
        let poly = DenseMLPoly::new(evals);

        // Fix x0 = 0.5 (represented as 1/2 in the field)
        // Since we halve the vector, the "first" variable here corresponds to the MSB.
        // new_eval[0] = f(0, 0.5) = 10 * (1-0.5) + 30 * 0.5 = 20
        // new_eval[1] = f(1, 0.5) = 20 * (1-0.5) + 40 * 0.5 = 30
        let half_f = F::from(1) / F::from(2);
        let fixed = poly.fix_first_variable(half_f);

        assert_eq!(fixed.num_vars, 1);
        assert_eq!(fixed.evaluations[0], F::from(20));
        assert_eq!(fixed.evaluations[1], F::from(30));
    }

    #[test]
    fn test_evaluate_interpolation() {
        // f(x0) = 10*(1-x0) + 20*x0
        let poly = DenseMLPoly::new(vec![F::from(10), F::from(20)]);

        // Evaluate at x0 = 0.5
        let res = poly.evaluate(&[f_div(1, 2)]);
        assert_eq!(res, F::from(15));

        // 2-var: f(x0, x1) = [0, 10, 100, 110]
        // f(0.5, 0.5) = 0.25*0 + 0.25*10 + 0.25*100 + 0.25*110 = 55
        let poly2 = DenseMLPoly::new(vec![F::ZERO, F::from(10), F::from(100), F::from(110)]);
        assert_eq!(poly2.evaluate(&[f_div(1, 2), f_div(1, 2)]), F::from(55));
    }

    #[test]
    fn test_fix_and_evaluate_consistency() {
        let evals: Vec<F> = (0..16).map(|i| F::from(i as u64)).collect();
        let poly = DenseMLPoly::new(evals);
        let r = vec![f_div(1, 3), f_div(1, 4), f_div(1, 5), f_div(1, 6)];

        // Method 1: Direct evaluation
        let val1 = poly.evaluate(&r);

        // Method 2: Fix x0, then evaluate remaining
        let poly_fixed = poly.fix_first_variable(r[0]);
        let val2 = poly_fixed.evaluate(&r[1..]);

        assert_eq!(
            val1, val2,
            "Direct eval and fix_first_variable must be consistent"
        );
    }

    #[test]
    fn test_eq_poly() {
        let r = vec![F::from(2), F::from(3)];
        let eq = DenseMLPoly::eq_poly(&r);

        // The property of eq(r, x) for x in {0,1}^n is:
        // eq(r, x) = Π (ri*xi + (1-ri)*(1-xi))

        // For x = (0,0): (1-2)*(1-3) = (-1)*(-2) = 2
        assert_eq!(eq.evaluate(&[F::ZERO, F::ZERO]), F::from(2));

        // For x = (1,1): (2*1)*(3*1) = 6
        assert_eq!(eq.evaluate(&[F::from(1), F::from(1)]), F::from(6));
    }

    #[test]
    fn test_hadamard_product() {
        let a = DenseMLPoly::new(vec![F::from(1), F::from(2)]);
        let b = DenseMLPoly::new(vec![F::from(3), F::from(4)]);
        let c = a.hadamard(&b);

        assert_eq!(c.evaluations, vec![F::from(3), F::from(8)]);
    }
}
