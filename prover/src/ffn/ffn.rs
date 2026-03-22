//! FFN layer prover/verifier for PiFormer.
//!
<<<<<<< HEAD
//! **Production-Grade Architecture:**
//!  1. OFFLINE WEIGHT BINDING: W1 and W2 are committed ONCE during preprocessing.
//!  2. STRICT IO BOUNDARIES: X and Y commitments are passed from the global pipeline.
//!  3. SUCCINCTNESS: The Verifier executes in strictly sub-linear time O(√N),
//!     relying entirely on Hyrax openings and Sumcheck reductions.
//!
//! The FFN computation is:
//!   M = X · W1        (Sumcheck 1)
//!   A = φ(M)          (Lasso)
//!   Y = A · W2        (Sumcheck 2)

use crate::field::F;
use crate::lookup::lasso::{prove_lasso, verify_lasso, LassoInstance, LassoProof};
use crate::pcs::{
    hyrax_commit, hyrax_open, hyrax_verify, params_from_vars, HyraxCommitment, HyraxParams,
    HyraxProof,
=======
//! The FFN computation is:
//!   M = X · W1        (ternary matrix multiply)
//!   A = φ(M)          (structured lookup activation via Lasso)
//!   Y = A · W2        (ternary matrix multiply)
//!
//! **Proof strategy (GKR-style):**
//!
//!   Let shapes be:  X: T×D,  W1: D×F,  M,A: T×F,  W2: F×D,  Y: T×D.
//!   (T = seq_len, D = d_model, F = d_ff)
//!
//!   Given verifier challenges (rx_t, ry_d) for Y:
//!   1. Claim: Y(rx_t, ry_d) = Σ_k A(rx_t, k) · W2(k, ry_d)
//!      → sumcheck over k ∈ {0,1}^log(F)
//!      → Hyrax open A at (rx_t, r_k) and W2 at (r_k, ry_d).
//!   2. Lasso proves: A[query_indices] = φ(M[query_indices]).
//!   3. Claim: M(rx_t, r_k) = Σ_j X(rx_t, j) · W1(j, r_k)
//!      → sumcheck over j ∈ {0,1}^log(D)
//!      → Hyrax open X at (rx_t, r_j) and W1 at (r_j, r_k).

use crate::field::F;
use crate::lookup::{prove_lasso, verify_lasso, LassoInstance, LassoProof};
use crate::pcs::{
    hyrax_commit, hyrax_open, hyrax_verify, HyraxCommitment, HyraxParams, HyraxProof,
>>>>>>> main
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::Field;
<<<<<<< HEAD
use ark_ff::PrimeField;

// ---------------------------------------------------------------------------
// Pipeline Interfaces & Keys
// ---------------------------------------------------------------------------

/// Trusted IO Commitments provided by the Global Pipeline Verifier.
pub struct FFNIOCommitments {
    pub x_com: HyraxCommitment,
    pub y_com: HyraxCommitment,
}

/// Preprocessing Key for the Verifier. (Static model weights)
#[derive(Clone)]
pub struct FFNVerifyingKey {
    pub seq_len: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub w1_com: HyraxCommitment,
    pub w2_com: HyraxCommitment,
}

/// Preprocessing Key for the Prover.
#[derive(Clone)]
pub struct FFNProvingKey {
    pub vk: FFNVerifyingKey,
    pub w1: Vec<Vec<F>>,
    pub w2: Vec<Vec<F>>,
}

/// Private witness data. ONLY the Prover holds this.
pub struct FFNWitness {
    pub x: Vec<Vec<F>>,
    pub m: Vec<Vec<F>>,
    pub a: Vec<Vec<F>>,
    pub y: Vec<Vec<F>>,
}

pub struct FFNInstance {
    pub activation_lasso: LassoInstance,
}

// ---------------------------------------------------------------------------
// Preprocessing (Offline Phase)
// ---------------------------------------------------------------------------

pub fn preprocess_ffn(
    seq_len: usize,
    d_model: usize,
    d_ff: usize,
    w1: Vec<Vec<F>>,
    w2: Vec<Vec<F>>,
) -> FFNProvingKey {
    let w1_mle = mat_to_mle(&w1, d_model, d_ff);
    let w2_mle = mat_to_mle(&w2, d_ff, d_model);

    let in_bits = d_model.next_power_of_two().trailing_zeros() as usize;
    let ff_bits = d_ff.next_power_of_two().trailing_zeros() as usize;

    let (nu_w1, _, params_w1) = params_from_vars(in_bits + ff_bits);
    let (nu_w2, _, params_w2) = params_from_vars(ff_bits + in_bits);

    let w1_com = hyrax_commit(&w1_mle.evaluations, nu_w1, &params_w1);
    let w2_com = hyrax_commit(&w2_mle.evaluations, nu_w2, &params_w2);

    let vk = FFNVerifyingKey {
        seq_len,
        d_model,
        d_ff,
        w1_com,
        w2_com,
    };
    FFNProvingKey { vk, w1, w2 }
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

pub struct FFNInternalCommitments {
    pub m_com: HyraxCommitment,
    pub a_com: HyraxCommitment,
}

pub struct FFNOpenings {
    pub y_eval: F,
    pub y_open: HyraxProof,
    pub a_eval: F,
    pub a_open: HyraxProof,
    pub w2_eval: F,
    pub w2_open: HyraxProof,

    pub m_eval: F,
    pub m_open: HyraxProof,
    pub x_eval: F,
    pub x_open: HyraxProof,
    pub w1_eval: F,
    pub w1_open: HyraxProof,
}

pub struct FFNProof {
    pub internal_coms: FFNInternalCommitments,
    pub activation_proof: LassoProof,
    pub y_sumcheck: SumcheckProof,
    pub m_sumcheck: SumcheckProof,
    pub openings: FFNOpenings,
=======

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// 2D matrix stored row-major as a flat evaluation vector of length T×F
/// (or D×F etc.), padded to a power-of-two total size.
///
/// num_vars_row = log₂(T_pow2), num_vars_col = log₂(F_pow2).
struct Mat2D {
    evals: Vec<F>,
    num_rows: usize,
    num_cols: usize,
    num_vars_row: usize,
    num_vars_col: usize,
}

impl Mat2D {
    fn from_2d(mat: &[Vec<F>]) -> Self {
        let rows = mat.len();
        let cols = if rows == 0 { 0 } else { mat[0].len() };
        let row_p2 = rows.next_power_of_two().max(1);
        let col_p2 = cols.next_power_of_two().max(1);
        let mut evals = vec![F::ZERO; row_p2 * col_p2];
        for (i, row) in mat.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                evals[i * col_p2 + j] = v;
            }
        }
        Mat2D {
            evals,
            num_rows: row_p2,
            num_cols: col_p2,
            num_vars_row: row_p2.trailing_zeros() as usize,
            num_vars_col: col_p2.trailing_zeros() as usize,
        }
    }

    fn total_vars(&self) -> usize {
        self.num_vars_row + self.num_vars_col
    }

    /// Evaluate MLE at (r_row || r_col).
    fn eval_at(&self, r_row: &[F], r_col: &[F]) -> F {
        assert_eq!(r_row.len(), self.num_vars_row);
        assert_eq!(r_col.len(), self.num_vars_col);
        let mut r = r_row.to_vec();
        r.extend_from_slice(r_col);
        DenseMLPoly::new(self.evals.clone()).evaluate(&r)
    }

    /// Fix row variables at `r_row`, return a 1-D vector over columns.
    fn collapse_rows(&self, r_row: &[F]) -> Vec<F> {
        assert_eq!(r_row.len(), self.num_vars_row);
        (0..self.num_cols)
            .map(|c| {
                let col: Vec<F> = (0..self.num_rows)
                    .map(|r| self.evals[r * self.num_cols + c])
                    .collect();
                DenseMLPoly::from_vec_padded(col).evaluate(r_row)
            })
            .collect()
    }

    /// Fix col variables at `r_col`, return a 1-D vector over rows.
    fn collapse_cols(&self, r_col: &[F]) -> Vec<F> {
        (0..self.num_rows)
            .map(|r| {
                let row = &self.evals[r * self.num_cols..(r + 1) * self.num_cols];
                DenseMLPoly::from_vec_padded(row.to_vec()).evaluate(r_col)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Public structs
// ---------------------------------------------------------------------------

/// Full witness for the FFN layer.
pub struct FFNInstance {
    /// Input X: shape (seq_len, d_model).
    pub x: Vec<Vec<F>>,
    /// Intermediate M = X·W1: shape (seq_len, d_ff).
    pub m: Vec<Vec<F>>,
    /// Activated A = φ(M): shape (seq_len, d_ff).
    pub a: Vec<Vec<F>>,
    /// Output Y = A·W2: shape (seq_len, d_model).
    pub y: Vec<Vec<F>>,
    /// Weight W1: shape (d_model, d_ff).
    pub w1: Vec<Vec<F>>,
    /// Weight W2: shape (d_ff, d_model).
    pub w2: Vec<Vec<F>>,
    /// Lasso instance proving A = φ(M).
    pub activation_lasso: LassoInstance,
}

/// Proof for the FFN layer.
pub struct FFNProof {
    // Commitments to intermediate values
    pub a_com: HyraxCommitment,
    pub m_com: HyraxCommitment,

    // Step 1: Y = A · W2 sumcheck
    pub y_sumcheck: SumcheckProof,
    pub a_eval: F,
    pub a_open: HyraxProof,
    pub w2_eval: F,
    pub w2_com: HyraxCommitment,
    pub w2_open: HyraxProof,

    // Step 2: Lasso proof for A = φ(M)
    pub activation_proof: LassoProof,

    // Step 3: M = X · W1 sumcheck
    pub m_sumcheck: SumcheckProof,
    pub m_eval: F,
    pub m_open: HyraxProof,
    pub x_eval: F,
    pub x_com: HyraxCommitment,
    pub x_open: HyraxProof,
    pub w1_eval: F,
    pub w1_com: HyraxCommitment,
    pub w1_open: HyraxProof,

    // Challenge points (needed by verifier to reconstruct transcript)
    pub rx_t: Vec<F>,
    pub ry_d: Vec<F>,
    pub r_k: Vec<F>,
    pub r_j: Vec<F>,
>>>>>>> main
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn prove_ffn(
<<<<<<< HEAD
    pk: &FFNProvingKey,
    witness: &FFNWitness,
    inst: &FFNInstance,
    io_coms: &FFNIOCommitments,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> Result<FFNProof, String> {
    let t = pk.vk.seq_len;
    let d = pk.vk.d_model;
    let f = pk.vk.d_ff;

    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let f_bits = f.next_power_of_two().trailing_zeros() as usize;

    let x_mle = mat_to_mle(&witness.x, t, d);
    let m_mle = mat_to_mle(&witness.m, t, f);
    let a_mle = mat_to_mle(&witness.a, t, f);
    let y_mle = mat_to_mle(&witness.y, t, d);
    let w1_mle = mat_to_mle(&pk.w1, d, f);
    let w2_mle = mat_to_mle(&pk.w2, f, d);

    let (nu_x, sigma_x, _params_x) = params_from_vars(t_bits + d_bits);
    let (nu_m, sigma_m, params_m) = params_from_vars(t_bits + f_bits);
    let (nu_w1, sigma_w1, _params_w1) = params_from_vars(d_bits + f_bits);
    let (nu_w2, sigma_w2, _params_w2) = params_from_vars(f_bits + d_bits);

    // 1. Absorb Trusted IO & VK Commitments
    absorb_com(transcript, b"w1_com", &pk.vk.w1_com);
    absorb_com(transcript, b"w2_com", &pk.vk.w2_com);
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);

    // 2. Commit to Internal Variables (M and A)
    let m_com = hyrax_commit(&m_mle.evaluations, nu_m, &params_m);
    let a_com = hyrax_commit(&a_mle.evaluations, nu_m, &params_m);
    absorb_com(transcript, b"m_com", &m_com);
    absorb_com(transcript, b"a_com", &a_com);

    // 3. Lasso for A = φ(M)
    let activation_proof = prove_lasso(&inst.activation_lasso, transcript, lasso_params);

    // 4. Sumcheck 1: Y = A · W2
    let rx_y = challenge_vec(transcript, t_bits, b"ffn_rx_y");
    let ry_y = challenge_vec(transcript, d_bits, b"ffn_ry_y");
    let y_eval = y_mle.evaluate(&combine(&rx_y, &ry_y));
    transcript.append_field(b"claim_y", &y_eval);

    let f_a = DenseMLPoly::from_vec_padded(eval_rows(&a_mle, t_bits, &rx_y));
    let g_w2 = DenseMLPoly::from_vec_padded(eval_cols(&w2_mle, f_bits, &ry_y));
    let (y_sumcheck, r_k) = prove_sumcheck(&f_a, &g_w2, y_eval, transcript);

    let a_eval = y_sumcheck.final_eval_f;
    let w2_eval = y_sumcheck.final_eval_g;

    // 5. Sumcheck 2: M = X · W1
    let rx_m = challenge_vec(transcript, t_bits, b"ffn_rx_m");
    let ry_m = challenge_vec(transcript, f_bits, b"ffn_ry_m");
    let m_eval = m_mle.evaluate(&combine(&rx_m, &ry_m));
    transcript.append_field(b"claim_m", &m_eval);

    let f_x = DenseMLPoly::from_vec_padded(eval_rows(&x_mle, t_bits, &rx_m));
    let g_w1 = DenseMLPoly::from_vec_padded(eval_cols(&w1_mle, d_bits, &ry_m));
    let (m_sumcheck, r_j) = prove_sumcheck(&f_x, &g_w1, m_eval, transcript);

    let x_eval = m_sumcheck.final_eval_f;
    let w1_eval = m_sumcheck.final_eval_g;

    // 6. Openings
    let y_open = hyrax_open(&y_mle.evaluations, &combine(&rx_y, &ry_y), nu_x, sigma_x);
    let a_open = hyrax_open(&a_mle.evaluations, &combine(&rx_y, &r_k), nu_m, sigma_m);
    let w2_open = hyrax_open(&w2_mle.evaluations, &combine(&r_k, &ry_y), nu_w2, sigma_w2);

    let m_open = hyrax_open(&m_mle.evaluations, &combine(&rx_m, &ry_m), nu_m, sigma_m);
    let x_open = hyrax_open(&x_mle.evaluations, &combine(&rx_m, &r_j), nu_x, sigma_x);
    let w1_open = hyrax_open(&w1_mle.evaluations, &combine(&r_j, &ry_m), nu_w1, sigma_w1);

    Ok(FFNProof {
        internal_coms: FFNInternalCommitments { m_com, a_com },
        activation_proof,
        y_sumcheck,
        m_sumcheck,
        openings: FFNOpenings {
            y_eval,
            y_open,
            a_eval,
            a_open,
            w2_eval,
            w2_open,
            m_eval,
            m_open,
            x_eval,
            x_open,
            w1_eval,
            w1_open,
        },
=======
    inst: &FFNInstance,
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<FFNProof, String> {
    let y_mat = Mat2D::from_2d(&inst.y);
    let a_mat = Mat2D::from_2d(&inst.a);
    let m_mat = Mat2D::from_2d(&inst.m);
    let x_mat = Mat2D::from_2d(&inst.x);
    let w2_mat = Mat2D::from_2d(&inst.w2);
    let w1_mat = Mat2D::from_2d(&inst.w1);

    let t_vars = y_mat.num_vars_row; // log(seq_len)
    let d_vars = y_mat.num_vars_col; // log(d_model)
    let k_vars = a_mat.num_vars_col; // log(d_ff) — inner sumcheck dimension for Y

    // --- Commit to A and M ---
    let nu_a = a_mat.total_vars() / 2;
    let sigma_a = a_mat.total_vars() - nu_a;
    let a_params = HyraxParams::new(sigma_a);
    let a_com = hyrax_commit(&a_mat.evals, nu_a, &a_params);

    let nu_m = m_mat.total_vars() / 2;
    let sigma_m = m_mat.total_vars() - nu_m;
    let m_params = HyraxParams::new(sigma_m);
    let m_com = hyrax_commit(&m_mat.evals, nu_m, &m_params);

    absorb_commitment(transcript, b"a_com", &a_com);
    absorb_commitment(transcript, b"m_com", &m_com);

    // --- Step 1: Y = A · W2 sumcheck ---
    // rx_t, ry_d: verifier challenges for Y's MLE
    let rx_t = challenge_vec(transcript, t_vars, b"ffn_rx_t");
    let ry_d = challenge_vec(transcript, d_vars, b"ffn_ry_d");

    let claim_y = y_mat.eval_at(&rx_t, &ry_d);
    transcript.append_field(b"ffn_claim_y", &claim_y);

    // f[k] = A(rx_t, k),  g[k] = W2(k, ry_d)
    let f_a_vec = a_mat.collapse_rows(&rx_t);
    let g_w2_vec = w2_mat.collapse_cols(&ry_d);
    let f_a = DenseMLPoly::from_vec_padded(f_a_vec);
    let g_w2 = DenseMLPoly::from_vec_padded(g_w2_vec);

    let (y_sumcheck, r_k) = prove_sumcheck(&f_a, &g_w2, claim_y, transcript);
    let a_eval = y_sumcheck.final_eval_f;
    let w2_eval = y_sumcheck.final_eval_g;

    // Open A at (rx_t || r_k)
    let mut point_a = rx_t.clone();
    point_a.extend_from_slice(&r_k);
    let a_open = hyrax_open(&a_mat.evals, &point_a, nu_a, sigma_a);

    // Commit + open W2 at (r_k || ry_d)
    let nu_w2 = w2_mat.total_vars() / 2;
    let sigma_w2 = w2_mat.total_vars() - nu_w2;
    let w2_params = HyraxParams::new(sigma_w2);
    let w2_com = hyrax_commit(&w2_mat.evals, nu_w2, &w2_params);
    let mut point_w2 = r_k.clone();
    point_w2.extend_from_slice(&ry_d);
    let w2_open = hyrax_open(&w2_mat.evals, &point_w2, nu_w2, sigma_w2);

    // --- Step 2: Lasso for A = φ(M) ---
    let activation_proof = prove_lasso(&inst.activation_lasso, transcript, params);

    // --- Step 3: M = X · W1 sumcheck ---
    // rx_m, ry_ff: fresh challenges for M's MLE
    let rx_m = challenge_vec(transcript, t_vars, b"ffn_rx_m");
    let ry_k = challenge_vec(transcript, k_vars, b"ffn_ry_k");

    let claim_m = m_mat.eval_at(&rx_m, &ry_k);
    transcript.append_field(b"ffn_claim_m", &claim_m);

    // f[j] = X(rx_m, j),  g[j] = W1(j, ry_k)
    let f_x_vec = x_mat.collapse_rows(&rx_m);
    let g_w1_vec = w1_mat.collapse_cols(&ry_k);
    let f_x = DenseMLPoly::from_vec_padded(f_x_vec);
    let g_w1 = DenseMLPoly::from_vec_padded(g_w1_vec);

    let (m_sumcheck, r_j) = prove_sumcheck(&f_x, &g_w1, claim_m, transcript);
    let x_eval = m_sumcheck.final_eval_f;
    let w1_eval = m_sumcheck.final_eval_g;

    // Open M at (rx_m || ry_k)
    let mut point_m = rx_m.clone();
    point_m.extend_from_slice(&ry_k);
    let m_open = hyrax_open(&m_mat.evals, &point_m, nu_m, sigma_m);

    // Commit + open X at (rx_m || r_j)
    let nu_x = x_mat.total_vars() / 2;
    let sigma_x = x_mat.total_vars() - nu_x;
    let x_params = HyraxParams::new(sigma_x);
    let x_com = hyrax_commit(&x_mat.evals, nu_x, &x_params);
    let mut point_x = rx_m.clone();
    point_x.extend_from_slice(&r_j);
    let x_open = hyrax_open(&x_mat.evals, &point_x, nu_x, sigma_x);

    // Commit + open W1 at (r_j || ry_k)
    let nu_w1 = w1_mat.total_vars() / 2;
    let sigma_w1 = w1_mat.total_vars() - nu_w1;
    let w1_params = HyraxParams::new(sigma_w1);
    let w1_com = hyrax_commit(&w1_mat.evals, nu_w1, &w1_params);
    let mut point_w1 = r_j.clone();
    point_w1.extend_from_slice(&ry_k);
    let w1_open = hyrax_open(&w1_mat.evals, &point_w1, nu_w1, sigma_w1);

    Ok(FFNProof {
        a_com,
        m_com,
        y_sumcheck,
        a_eval,
        a_open,
        w2_eval,
        w2_com,
        w2_open,
        activation_proof,
        m_sumcheck,
        m_eval: claim_m,
        m_open,
        x_eval,
        x_com,
        x_open,
        w1_eval,
        w1_com,
        w1_open,
        rx_t,
        ry_d,
        r_k,
        r_j,
>>>>>>> main
    })
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn verify_ffn(
    proof: &FFNProof,
    inst: &FFNInstance,
<<<<<<< HEAD
    vk: &FFNVerifyingKey,
    io_coms: &FFNIOCommitments,
    transcript: &mut Transcript,
    lasso_params: &HyraxParams,
) -> Result<(), String> {
    let t_bits = vk.seq_len.next_power_of_two().trailing_zeros() as usize;
    let d_bits = vk.d_model.next_power_of_two().trailing_zeros() as usize;
    let f_bits = vk.d_ff.next_power_of_two().trailing_zeros() as usize;

    let (_, _, params_x) = params_from_vars(t_bits + d_bits);
    let (_, _, params_m) = params_from_vars(t_bits + f_bits);
    let (_, _, params_w1) = params_from_vars(d_bits + f_bits);
    let (_, _, params_w2) = params_from_vars(f_bits + d_bits);

    // 1. Absorb Cryptographic Commitments (Zero O(N) operations!)
    absorb_com(transcript, b"w1_com", &vk.w1_com);
    absorb_com(transcript, b"w2_com", &vk.w2_com);
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);
    absorb_com(transcript, b"m_com", &proof.internal_coms.m_com);
    absorb_com(transcript, b"a_com", &proof.internal_coms.a_com);

    // 2. Verify Lasso (Binds A to M)
=======
    transcript: &mut Transcript,
    params: &HyraxParams,
) -> Result<(), String> {
    let y_mat = Mat2D::from_2d(&inst.y);
    let a_mat = Mat2D::from_2d(&inst.a);
    let m_mat = Mat2D::from_2d(&inst.m);
    let x_mat = Mat2D::from_2d(&inst.x);
    let w2_mat = Mat2D::from_2d(&inst.w2);
    let w1_mat = Mat2D::from_2d(&inst.w1);

    let t_vars = y_mat.num_vars_row;
    let d_vars = y_mat.num_vars_col;
    let k_vars = a_mat.num_vars_col;
    let j_vars = x_mat.num_vars_col; // log(d_model) — inner dimension for M sumcheck

    let nu_a = a_mat.total_vars() / 2;
    let sigma_a = a_mat.total_vars() - nu_a;
    let a_params = HyraxParams::new(sigma_a);

    let nu_m = m_mat.total_vars() / 2;
    let sigma_m = m_mat.total_vars() - nu_m;
    let m_params = HyraxParams::new(sigma_m);

    absorb_commitment(transcript, b"a_com", &proof.a_com);
    absorb_commitment(transcript, b"m_com", &proof.m_com);

    // --- Step 1: Y sumcheck ---
    let rx_t = challenge_vec(transcript, t_vars, b"ffn_rx_t");
    let ry_d = challenge_vec(transcript, d_vars, b"ffn_ry_d");

    let claim_y = y_mat.eval_at(&rx_t, &ry_d);
    transcript.append_field(b"ffn_claim_y", &claim_y);

    let (r_k, final_y) = verify_sumcheck(&proof.y_sumcheck, claim_y, k_vars, transcript)
        .map_err(|e| format!("FFN Y sumcheck: {e}"))?;

    let expected_y = proof.a_eval * proof.w2_eval;
    if final_y != expected_y {
        return Err("FFN Y sumcheck final mismatch".to_string());
    }

    // Verify A opening
    let mut point_a = rx_t.clone();
    point_a.extend_from_slice(&r_k);
    // Sanity: A commitment matches public A
    let a_com_check = hyrax_commit(&a_mat.evals, nu_a, &a_params);
    if a_com_check.row_coms != proof.a_com.row_coms {
        return Err("FFN A commitment mismatch".to_string());
    }
    hyrax_verify(
        &proof.a_com,
        proof.a_eval,
        &point_a,
        &proof.a_open,
        &a_params,
    )
    .map_err(|e| format!("FFN A open: {e}"))?;

    // Verify W2 opening
    let nu_w2 = w2_mat.total_vars() / 2;
    let sigma_w2 = w2_mat.total_vars() - nu_w2;
    let w2_params = HyraxParams::new(sigma_w2);
    let mut point_w2 = r_k.clone();
    point_w2.extend_from_slice(&ry_d);
    hyrax_verify(
        &proof.w2_com,
        proof.w2_eval,
        &point_w2,
        &proof.w2_open,
        &w2_params,
    )
    .map_err(|e| format!("FFN W2 open: {e}"))?;
    // Sanity: W2 commitment matches public W2
    let w2_com_check = hyrax_commit(&w2_mat.evals, nu_w2, &w2_params);
    if w2_com_check.row_coms != proof.w2_com.row_coms {
        return Err("FFN W2 commitment mismatch".to_string());
    }

    // --- Step 2: Lasso ---
>>>>>>> main
    verify_lasso(
        &proof.activation_proof,
        &inst.activation_lasso,
        transcript,
<<<<<<< HEAD
        lasso_params,
    )
    .map_err(|e| format!("FFN Lasso failed: {}", e))?;

    // 3. Verify Y Sumcheck
    let rx_y = challenge_vec(transcript, t_bits, b"ffn_rx_y");
    let ry_y = challenge_vec(transcript, d_bits, b"ffn_ry_y");
    transcript.append_field(b"claim_y", &proof.openings.y_eval);

    let (r_k, final_y) =
        verify_sumcheck(&proof.y_sumcheck, proof.openings.y_eval, f_bits, transcript)?;
    if final_y != proof.openings.a_eval * proof.openings.w2_eval {
        return Err("Y Sumcheck mismatch".into());
    }

    // 4. Verify M Sumcheck
    let rx_m = challenge_vec(transcript, t_bits, b"ffn_rx_m");
    let ry_m = challenge_vec(transcript, f_bits, b"ffn_ry_m");
    transcript.append_field(b"claim_m", &proof.openings.m_eval);

    let (r_j, final_m) =
        verify_sumcheck(&proof.m_sumcheck, proof.openings.m_eval, d_bits, transcript)?;
    if final_m != proof.openings.x_eval * proof.openings.w1_eval {
        return Err("M Sumcheck mismatch".into());
    }

    // 5. Verify ALL Openings (Cryptographically binds evaluations to VK and IO)
    hyrax_verify(
        &io_coms.y_com,
        proof.openings.y_eval,
        &combine(&rx_y, &ry_y),
        &proof.openings.y_open,
        &params_x,
    )?;
    hyrax_verify(
        &proof.internal_coms.a_com,
        proof.openings.a_eval,
        &combine(&rx_y, &r_k),
        &proof.openings.a_open,
        &params_m,
    )?;
    hyrax_verify(
        &vk.w2_com,
        proof.openings.w2_eval,
        &combine(&r_k, &ry_y),
        &proof.openings.w2_open,
        &params_w2,
    )?;

    hyrax_verify(
        &proof.internal_coms.m_com,
        proof.openings.m_eval,
        &combine(&rx_m, &ry_m),
        &proof.openings.m_open,
        &params_m,
    )?;
    hyrax_verify(
        &io_coms.x_com,
        proof.openings.x_eval,
        &combine(&rx_m, &r_j),
        &proof.openings.x_open,
        &params_x,
    )?;
    hyrax_verify(
        &vk.w1_com,
        proof.openings.w1_eval,
        &combine(&r_j, &ry_m),
        &proof.openings.w1_open,
        &params_w1,
    )?;
=======
        params,
    )
    .map_err(|e| format!("FFN activation Lasso: {e}"))?;

    // --- Step 3: M sumcheck ---
    let rx_m = challenge_vec(transcript, t_vars, b"ffn_rx_m");
    let ry_k = challenge_vec(transcript, k_vars, b"ffn_ry_k");

    transcript.append_field(b"ffn_claim_m", &proof.m_eval);
    let (r_j, final_m) = verify_sumcheck(&proof.m_sumcheck, proof.m_eval, j_vars, transcript)
        .map_err(|e| format!("FFN M sumcheck: {e}"))?;

    let expected_m = proof.x_eval * proof.w1_eval;
    if final_m != expected_m {
        return Err("FFN M sumcheck final mismatch".to_string());
    }

    // Verify M opening
    let mut point_m = rx_m.clone();
    point_m.extend_from_slice(&ry_k);
    hyrax_verify(
        &proof.m_com,
        proof.m_eval,
        &point_m,
        &proof.m_open,
        &m_params,
    )
    .map_err(|e| format!("FFN M open: {e}"))?;

    // Verify X opening
    let nu_x = x_mat.total_vars() / 2;
    let sigma_x = x_mat.total_vars() - nu_x;
    let x_params = HyraxParams::new(sigma_x);
    let mut point_x = rx_m.clone();
    point_x.extend_from_slice(&r_j);
    hyrax_verify(
        &proof.x_com,
        proof.x_eval,
        &point_x,
        &proof.x_open,
        &x_params,
    )
    .map_err(|e| format!("FFN X open: {e}"))?;
    let x_com_check = hyrax_commit(&x_mat.evals, nu_x, &x_params);
    if x_com_check.row_coms != proof.x_com.row_coms {
        return Err("FFN X commitment mismatch".to_string());
    }

    // Verify W1 opening
    let nu_w1 = w1_mat.total_vars() / 2;
    let sigma_w1 = w1_mat.total_vars() - nu_w1;
    let w1_params = HyraxParams::new(sigma_w1);
    let mut point_w1 = r_j.clone();
    point_w1.extend_from_slice(&ry_k);
    hyrax_verify(
        &proof.w1_com,
        proof.w1_eval,
        &point_w1,
        &proof.w1_open,
        &w1_params,
    )
    .map_err(|e| format!("FFN W1 open: {e}"))?;
    let w1_com_check = hyrax_commit(&w1_mat.evals, nu_w1, &w1_params);
    if w1_com_check.row_coms != proof.w1_com.row_coms {
        return Err("FFN W1 commitment mismatch".to_string());
    }
>>>>>>> main

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
<<<<<<< HEAD
fn mat_to_mle(mat: &[Vec<F>], rows: usize, cols: usize) -> DenseMLPoly {
    let r_p2 = rows.next_power_of_two().max(1);
    let c_p2 = cols.next_power_of_two().max(1);
    let mut evals = vec![F::ZERO; r_p2 * c_p2];
    for (i, row) in mat.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            evals[i * c_p2 + j] = v;
        }
    }
    DenseMLPoly::new(evals)
}

fn eval_rows(poly: &DenseMLPoly, n_row_vars: usize, r_row: &[F]) -> Vec<F> {
    let mut p = poly.clone();
    for &r in r_row {
        p = p.fix_first_variable(r);
    }
    p.evaluations
}
fn eval_cols(poly: &DenseMLPoly, n_row_vars: usize, r_col: &[F]) -> Vec<F> {
    let n_p2_rows = 1 << n_row_vars;
    let n_p2_cols = poly.evaluations.len() / n_p2_rows;
    (0..n_p2_rows)
        .map(|i| {
            DenseMLPoly::new(poly.evaluations[i * n_p2_cols..(i + 1) * n_p2_cols].to_vec())
                .evaluate(r_col)
        })
        .collect()
}
fn combine(a: &[F], b: &[F]) -> Vec<F> {
    let mut res = a.to_vec();
    res.extend_from_slice(b);
    res
}
=======

>>>>>>> main
fn challenge_vec(transcript: &mut Transcript, len: usize, label: &[u8]) -> Vec<F> {
    (0..len)
        .map(|_| transcript.challenge_field::<F>(label))
        .collect()
}
<<<<<<< HEAD
fn absorb_com(transcript: &mut Transcript, label: &[u8], com: &HyraxCommitment) {
=======

fn absorb_commitment(transcript: &mut Transcript, label: &[u8], com: &HyraxCommitment) {
>>>>>>> main
    use ark_serialize::CanonicalSerialize;
    for pt in &com.row_coms {
        let mut buf = Vec::new();
        pt.serialize_compressed(&mut buf).unwrap();
        transcript.append_bytes(label, &buf);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
<<<<<<< HEAD
#[cfg(test)]
mod ffn_tests {
    use super::*;
    use ark_ff::{One, Zero};

    fn setup_test_pipeline() -> (FFNProvingKey, FFNWitness, FFNInstance, FFNIOCommitments) {
        let t = 2usize;
        let d = 2usize;
        let f_dim = 4usize;
        let w1 = vec![
            vec![F::ONE, F::ZERO, F::ZERO - F::ONE, F::ONE],
            vec![F::ZERO, F::ONE, F::ONE, F::ZERO],
        ];
        let w2 = vec![
=======

#[cfg(test)]
mod ffn_tests {
    use super::*;
    use crate::pcs::HyraxParams;
    use crate::transcript::Transcript;
    use ark_ff::{Field, One};

    /// Build a self-consistent tiny FFN instance.
    ///
    /// seq_len=2, d_model=2, d_ff=4.
    /// W1 ∈ {-1,0,1}^{2×4},  W2 ∈ {-1,0,1}^{4×2}.
    /// φ(x) = x mod 16 + 1  (single-table Lasso, 4-bit, c=1).
    fn build_ffn_instance() -> FFNInstance {
        let seq_len = 2usize;
        let d_model = 2usize;
        let d_ff = 4usize;

        // W1: (d_model=2) × (d_ff=4)
        let w1: Vec<Vec<F>> = vec![
            vec![F::ONE, F::ZERO, F::ZERO - F::ONE, F::ONE],
            vec![F::ZERO, F::ONE, F::ONE, F::ZERO],
        ];
        // W2: (d_ff=4) × (d_model=2)
        let w2: Vec<Vec<F>> = vec![
>>>>>>> main
            vec![F::ONE, F::ZERO],
            vec![F::ZERO - F::ONE, F::ONE],
            vec![F::ONE, F::ZERO - F::ONE],
            vec![F::ZERO, F::ONE],
        ];
<<<<<<< HEAD
        let x = vec![
=======

        // X: seq_len × d_model
        let x: Vec<Vec<F>> = vec![
>>>>>>> main
            vec![F::from(3u64), F::from(1u64)],
            vec![F::from(2u64), F::from(4u64)],
        ];

<<<<<<< HEAD
=======
        // M = X · W1
>>>>>>> main
        let matmul = |a: &Vec<Vec<F>>, b: &Vec<Vec<F>>, r, k, c| -> Vec<Vec<F>> {
            (0..r)
                .map(|i| {
                    (0..c)
                        .map(|j| (0..k).map(|l| a[i][l] * b[l][j]).sum())
                        .collect()
                })
                .collect()
        };
<<<<<<< HEAD
        let m = matmul(&x, &w1, t, d, f_dim);

        let table_size = 16usize;
        let m_bits = 4usize;
        let table: Vec<F> = (0..table_size).map(|i| F::from(i as u64)).collect();
        let a = m.clone(); // Identity for test

        let mut query_indices = Vec::new();
        let mut outputs = Vec::new();
        for row in &m {
            for &v in row {
                let idx = (v.into_bigint().as_ref()[0] as usize) & (table_size - 1);
=======
        let m = matmul(&x, &w1, seq_len, d_model, d_ff);

        // φ is identity for this test (table T[i] = i, 4-bit lookup)
        let bits_pc = 4usize;
        let table_size = 1usize << bits_pc;
        let table: Vec<F> = (0..table_size).map(|i| F::from(i as u64)).collect();

        let a = m.clone(); // φ(m) = m (identity lookup for zero-valued φ test)

        // Build Lasso: for each element of M, look up in identity table
        let mask = table_size - 1;
        let mut query_indices: Vec<usize> = Vec::new();
        let mut outputs: Vec<F> = Vec::new();
        for row in &m {
            for &v in row {
                use ark_ff::PrimeField;
                let idx = (v.into_bigint().as_ref()[0] as usize) & mask;
>>>>>>> main
                query_indices.push(idx);
                outputs.push(F::from(idx as u64));
            }
        }
        let activation_lasso = LassoInstance {
            tables: vec![table],
            query_indices,
            outputs,
<<<<<<< HEAD
            bits_per_chunk: m_bits,
        };
        let y = matmul(&a, &w2, t, f_dim, d);

        // 1. Offline Preprocessing
        let pk = preprocess_ffn(t, d, f_dim, w1, w2);

        // 2. IO Commitments from Pipeline
        let x_mle = mat_to_mle(&x, t, d);
        let y_mle = mat_to_mle(&y, t, d);
        let (nu_x, _, params_x) = params_from_vars(
            t.next_power_of_two().trailing_zeros() as usize
                + d.next_power_of_two().trailing_zeros() as usize,
        );
        let io_coms = FFNIOCommitments {
            x_com: hyrax_commit(&x_mle.evaluations, nu_x, &params_x),
            y_com: hyrax_commit(&y_mle.evaluations, nu_x, &params_x),
        };

        let witness = FFNWitness { x, m, a, y };
        let inst = FFNInstance { activation_lasso };

        (pk, witness, inst, io_coms)
    }

    #[test]
    fn test_ffn_succinct_e2e() {
        let (pk, witness, inst, io_coms) = setup_test_pipeline();
        let lasso_params = HyraxParams::new(2);

        let mut pt = Transcript::new(b"ffn-test");
        let proof = prove_ffn(&pk, &witness, &inst, &io_coms, &mut pt, &lasso_params).unwrap();

        let mut vt = Transcript::new(b"ffn-test");
        let result = verify_ffn(&proof, &inst, &pk.vk, &io_coms, &mut vt, &lasso_params);
=======
            bits_per_chunk: bits_pc,
        };

        // Recompute Y = A · W2
        let y = matmul(&a, &w2, seq_len, d_ff, d_model);

        FFNInstance {
            x,
            m,
            a,
            y,
            w1,
            w2,
            activation_lasso,
        }
    }

    #[test]
    fn test_ffn_e2e_success() {
        let inst = build_ffn_instance();
        // Use sigma=2 for the outer Lasso params (4-bit table, nu=2, sigma=2)
        let params = HyraxParams::new(2);

        let mut pt = Transcript::new(b"ffn-test");
        let proof = prove_ffn(&inst, &mut pt, &params).expect("prove failed");

        let mut vt = Transcript::new(b"ffn-test");
        let result = verify_ffn(&proof, &inst, &mut vt, &params);
>>>>>>> main
        assert!(result.is_ok(), "verify failed: {:?}", result.err());
    }

    #[test]
<<<<<<< HEAD
    fn test_rejects_invalid_weights() {
        let (mut pk, witness, inst, io_coms) = setup_test_pipeline();
        let lasso_params = HyraxParams::new(2);

        pk.w2[0][0] += F::one(); // Tamper weight internally

        let mut pt = Transcript::new(b"ffn-test");
        let proof = prove_ffn(&pk, &witness, &inst, &io_coms, &mut pt, &lasso_params).unwrap();

        let mut vt = Transcript::new(b"ffn-test");
        let result = verify_ffn(&proof, &inst, &pk.vk, &io_coms, &mut vt, &lasso_params);
        assert!(result.is_err(), "Should reject tampered weights");
    }

    /// Tampering with the input X while keeping io_coms fixed should be detected
    /// via the Hyrax opening check on X.
    #[test]
    fn test_rejects_tampered_x_input() {
        let (pk, mut witness, inst, io_coms) = setup_test_pipeline();
        let lasso_params = HyraxParams::new(2);

        // Tamper with X (but io_coms.x_com is still bound to the original X)
        witness.x[0][0] += F::one();

        let mut pt = Transcript::new(b"ffn-test");
        let proof = prove_ffn(&pk, &witness, &inst, &io_coms, &mut pt, &lasso_params).unwrap();

        let mut vt = Transcript::new(b"ffn-test");
        let result = verify_ffn(&proof, &inst, &pk.vk, &io_coms, &mut vt, &lasso_params);
        assert!(result.is_err(), "Should reject tampered X input");
    }

    /// Tampering with the output Y while keeping io_coms fixed should be detected.
    #[test]
    fn test_rejects_tampered_y_output() {
        let (pk, mut witness, inst, io_coms) = setup_test_pipeline();
        let lasso_params = HyraxParams::new(2);

        witness.y[0][0] += F::one();

        let mut pt = Transcript::new(b"ffn-test");
        let proof = prove_ffn(&pk, &witness, &inst, &io_coms, &mut pt, &lasso_params).unwrap();

        let mut vt = Transcript::new(b"ffn-test");
        let result = verify_ffn(&proof, &inst, &pk.vk, &io_coms, &mut vt, &lasso_params);
        assert!(result.is_err(), "Should reject tampered Y output");
    }

    /// Tampering with the intermediate activation A should be detected.
    #[test]
    fn test_rejects_tampered_activation_a() {
        let (pk, mut witness, inst, io_coms) = setup_test_pipeline();
        let lasso_params = HyraxParams::new(2);

        witness.a[0][0] += F::one();

        let mut pt = Transcript::new(b"ffn-test");
        let proof = prove_ffn(&pk, &witness, &inst, &io_coms, &mut pt, &lasso_params).unwrap();

        let mut vt = Transcript::new(b"ffn-test");
        let result = verify_ffn(&proof, &inst, &pk.vk, &io_coms, &mut vt, &lasso_params);
        assert!(result.is_err(), "Should reject tampered activation A");
    }

    /// Tampering with W1 internally (but not recomputing vk.w1_com) should fail.
    #[test]
    fn test_rejects_tampered_w1() {
        let (mut pk, witness, inst, io_coms) = setup_test_pipeline();
        let lasso_params = HyraxParams::new(2);

        pk.w1[0][0] += F::one();

        let mut pt = Transcript::new(b"ffn-test");
        let proof = prove_ffn(&pk, &witness, &inst, &io_coms, &mut pt, &lasso_params).unwrap();

        let mut vt = Transcript::new(b"ffn-test");
        let result = verify_ffn(&proof, &inst, &pk.vk, &io_coms, &mut vt, &lasso_params);
        assert!(result.is_err(), "Should reject tampered W1");
=======
    fn test_ffn_rejects_tampered_y() {
        let mut inst = build_ffn_instance();
        let params = HyraxParams::new(2);

        let mut pt = Transcript::new(b"ffn-tamper-y");
        let proof = prove_ffn(&inst, &mut pt, &params).expect("prove failed");

        // After proof generation, corrupt Y
        inst.y[0][0] += F::one();

        let mut vt = Transcript::new(b"ffn-tamper-y");
        let result = verify_ffn(&proof, &inst, &mut vt, &params);
        assert!(result.is_err(), "should reject tampered Y");
    }

    #[test]
    fn test_ffn_rejects_tampered_a_matrix() {
        let mut inst = build_ffn_instance();
        let params = HyraxParams::new(2);

        let mut pt = Transcript::new(b"ffn-tamper-a");
        let proof = prove_ffn(&inst, &mut pt, &params).expect("prove failed");

        inst.a[0][0] += F::one();

        let mut vt = Transcript::new(b"ffn-tamper-a");
        let result = verify_ffn(&proof, &inst, &mut vt, &params);
        assert!(result.is_err());
>>>>>>> main
    }
}
