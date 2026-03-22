//! FFN layer prover/verifier for PiFormer.
//!
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
use crate::pcs::{hyrax_commit, hyrax_open, hyrax_verify, HyraxCommitment, HyraxParams, HyraxProof};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::Field;

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
                let col: Vec<F> = (0..self.num_rows).map(|r| self.evals[r * self.num_cols + c]).collect();
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
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn prove_ffn(
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
    })
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn verify_ffn(
    proof: &FFNProof,
    inst: &FFNInstance,
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
    hyrax_verify(&proof.a_com, proof.a_eval, &point_a, &proof.a_open, &a_params)
        .map_err(|e| format!("FFN A open: {e}"))?;

    // Verify W2 opening
    let nu_w2 = w2_mat.total_vars() / 2;
    let sigma_w2 = w2_mat.total_vars() - nu_w2;
    let w2_params = HyraxParams::new(sigma_w2);
    let mut point_w2 = r_k.clone();
    point_w2.extend_from_slice(&ry_d);
    hyrax_verify(&proof.w2_com, proof.w2_eval, &point_w2, &proof.w2_open, &w2_params)
        .map_err(|e| format!("FFN W2 open: {e}"))?;
    // Sanity: W2 commitment matches public W2
    let w2_com_check = hyrax_commit(&w2_mat.evals, nu_w2, &w2_params);
    if w2_com_check.row_coms != proof.w2_com.row_coms {
        return Err("FFN W2 commitment mismatch".to_string());
    }

    // --- Step 2: Lasso ---
    verify_lasso(&proof.activation_proof, &inst.activation_lasso, transcript, params)
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
    hyrax_verify(&proof.m_com, proof.m_eval, &point_m, &proof.m_open, &m_params)
        .map_err(|e| format!("FFN M open: {e}"))?;

    // Verify X opening
    let nu_x = x_mat.total_vars() / 2;
    let sigma_x = x_mat.total_vars() - nu_x;
    let x_params = HyraxParams::new(sigma_x);
    let mut point_x = rx_m.clone();
    point_x.extend_from_slice(&r_j);
    hyrax_verify(&proof.x_com, proof.x_eval, &point_x, &proof.x_open, &x_params)
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
    hyrax_verify(&proof.w1_com, proof.w1_eval, &point_w1, &proof.w1_open, &w1_params)
        .map_err(|e| format!("FFN W1 open: {e}"))?;
    let w1_com_check = hyrax_commit(&w1_mat.evals, nu_w1, &w1_params);
    if w1_com_check.row_coms != proof.w1_com.row_coms {
        return Err("FFN W1 commitment mismatch".to_string());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn challenge_vec(transcript: &mut Transcript, len: usize, label: &[u8]) -> Vec<F> {
    (0..len).map(|_| transcript.challenge_field::<F>(label)).collect()
}

fn absorb_commitment(transcript: &mut Transcript, label: &[u8], com: &HyraxCommitment) {
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
            vec![F::ONE, F::ZERO],
            vec![F::ZERO - F::ONE, F::ONE],
            vec![F::ONE, F::ZERO - F::ONE],
            vec![F::ZERO, F::ONE],
        ];

        // X: seq_len × d_model
        let x: Vec<Vec<F>> = vec![
            vec![F::from(3u64), F::from(1u64)],
            vec![F::from(2u64), F::from(4u64)],
        ];

        // M = X · W1
        let matmul = |a: &Vec<Vec<F>>, b: &Vec<Vec<F>>, r, k, c| -> Vec<Vec<F>> {
            (0..r)
                .map(|i| {
                    (0..c)
                        .map(|j| (0..k).map(|l| a[i][l] * b[l][j]).sum())
                        .collect()
                })
                .collect()
        };
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
                query_indices.push(idx);
                outputs.push(F::from(idx as u64));
            }
        }
        let activation_lasso = LassoInstance {
            tables: vec![table],
            query_indices,
            outputs,
            bits_per_chunk: bits_pc,
        };

        // Recompute Y = A · W2
        let y = matmul(&a, &w2, seq_len, d_ff, d_model);

        FFNInstance { x, m, a, y, w1, w2, activation_lasso }
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
        assert!(result.is_ok(), "verify failed: {:?}", result.err());
    }

    #[test]
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
    }
}
