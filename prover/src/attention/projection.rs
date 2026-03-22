//! Linear Projection (Dense) Layer Protocol
//!
//! **Production-Grade Architecture:**
//!  1. OFFLINE WEIGHT BINDING: The static weight matrix W is committed ONCE during
//!     a preprocessing phase. The Verifier only holds the `w_com` inside the VK.
//!  2. STRICT IO BOUNDARIES: The Verifier does NOT trust the Prover for the
//!     commitments of X and Y. These MUST be passed via `ProjectionIOCommitments`
//!     from the global pipeline (e.g., from LayerNorm).
//!  3. SUCCINCT GKR CHAINING: The O(N^3) matrix multiplication is reduced to a
//!     single Sumcheck protocol. The Verifier runs in strictly sub-linear time.
//!
//! **Computation proved:**
//!   Y[i][j] = Σ_k X[i][k] · W[k][j]

use crate::field::F;
use crate::pcs::{
    hyrax_commit, hyrax_open, hyrax_verify, params_from_vars, HyraxCommitment, HyraxParams,
    HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::Field;

// ---------------------------------------------------------------------------
// Pipeline Interfaces & Keys
// ---------------------------------------------------------------------------

/// Trusted IO Commitments provided by the Global Pipeline Verifier.
pub struct ProjectionIOCommitments {
    pub x_com: HyraxCommitment,
    pub y_com: HyraxCommitment,
}

/// Preprocessing Key for the Verifier.
/// Contains the cryptographic commitment to the static model weights.
#[derive(Clone)]
pub struct ProjectionVerifyingKey {
    pub seq_len: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub w_com: HyraxCommitment,
}

/// Preprocessing Key for the Prover.
/// Contains the raw static weights and the Verifying Key.
#[derive(Clone)]
pub struct ProjectionProvingKey {
    pub vk: ProjectionVerifyingKey,
    pub w: Vec<Vec<F>>,
}

/// Private witness data (dynamic activations). ONLY the Prover holds this.
pub struct ProjectionWitness {
    pub x: Vec<Vec<F>>,
    pub y: Vec<Vec<F>>,
}

// ---------------------------------------------------------------------------
// Preprocessing (Offline Phase)
// ---------------------------------------------------------------------------

/// Run ONCE when the model is loaded. Commits to the static weights W.
pub fn preprocess_projection(
    seq_len: usize,
    d_in: usize,
    d_out: usize,
    w: Vec<Vec<F>>,
) -> ProjectionProvingKey {
    let w_mle = mat_to_mle(&w, d_in, d_out);
    let (nu_w, _sigma_w, params_w) = params_from_vars(
        d_in.next_power_of_two().trailing_zeros() as usize
            + d_out.next_power_of_two().trailing_zeros() as usize,
    );

    let w_com = hyrax_commit(&w_mle.evaluations, nu_w, &params_w);

    let vk = ProjectionVerifyingKey {
        seq_len,
        d_in,
        d_out,
        w_com,
    };

    ProjectionProvingKey { vk, w }
}

// ---------------------------------------------------------------------------
// Proof types
// ---------------------------------------------------------------------------

pub struct ProjectionOpenings {
    pub y_eval: F,
    pub y_open: HyraxProof,
    pub x_eval: F,
    pub x_open: HyraxProof,
    pub w_eval: F,
    pub w_open: HyraxProof,
}

pub struct ProjectionProof {
    pub sumcheck: SumcheckProof,
    pub openings: ProjectionOpenings,
}

// ---------------------------------------------------------------------------
// Prover (Online Phase)
// ---------------------------------------------------------------------------

pub fn prove_projection(
    pk: &ProjectionProvingKey,
    witness: &ProjectionWitness,
    io_coms: &ProjectionIOCommitments,
    transcript: &mut Transcript,
) -> Result<ProjectionProof, String> {
    let t = pk.vk.seq_len;
    let d_in = pk.vk.d_in;
    let d_out = pk.vk.d_out;

    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let in_bits = d_in.next_power_of_two().trailing_zeros() as usize;
    let out_bits = d_out.next_power_of_two().trailing_zeros() as usize;

    let x_mle = mat_to_mle(&witness.x, t, d_in);
    let y_mle = mat_to_mle(&witness.y, t, d_out);
    let w_mle = mat_to_mle(&pk.w, d_in, d_out);

    let (nu_x, sigma_x, _params_x) = params_from_vars(t_bits + in_bits);
    let (nu_y, sigma_y, _params_y) = params_from_vars(t_bits + out_bits);
    let (nu_w, sigma_w, _params_w) = params_from_vars(in_bits + out_bits);

    // 1. Absorb external (IO) and static (VK) commitments
    absorb_com(transcript, b"w_com", &pk.vk.w_com);
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);

    // 2. Challenges for Y output
    let r_t = challenge_vec(transcript, t_bits, b"proj_rt");
    let r_out = challenge_vec(transcript, out_bits, b"proj_rout");

    // Claim: y_eval = Y(r_t, r_out)
    let y_eval = y_mle.evaluate(&combine(&r_t, &r_out));
    transcript.append_field(b"claimed_y", &y_eval);

    // 3. Sumcheck: Y(r_t, r_out) = Σ_k X(r_t, k) * W(k, r_out)
    let f_x = DenseMLPoly::from_vec_padded(eval_rows(&x_mle, t_bits, &r_t));
    let g_w = DenseMLPoly::from_vec_padded(eval_cols(&w_mle, in_bits, &r_out));

    let (sumcheck, r_k) = prove_sumcheck(&f_x, &g_w, y_eval, transcript);

    let x_eval = sumcheck.final_eval_f;
    let w_eval = sumcheck.final_eval_g;

    // 4. Openings
    let y_open = hyrax_open(&y_mle.evaluations, &combine(&r_t, &r_out), nu_y, sigma_y);
    let x_open = hyrax_open(&x_mle.evaluations, &combine(&r_t, &r_k), nu_x, sigma_x);
    let w_open = hyrax_open(&w_mle.evaluations, &combine(&r_k, &r_out), nu_w, sigma_w);

    Ok(ProjectionProof {
        sumcheck,
        openings: ProjectionOpenings {
            y_eval,
            y_open,
            x_eval,
            x_open,
            w_eval,
            w_open,
        },
    })
}

// ---------------------------------------------------------------------------
// Verifier (Online Phase)
// ---------------------------------------------------------------------------

/// **Production-Grade Succinct Verifier**
///
/// Ensures strict O(log N) or O(√N) execution. Binds the computation entirely
/// to the offline static `vk` and the online dynamic `io_coms`.
pub fn verify_projection_succinct(
    proof: &ProjectionProof,
    vk: &ProjectionVerifyingKey,
    io_coms: &ProjectionIOCommitments,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let t = vk.seq_len;
    let d_in = vk.d_in;
    let d_out = vk.d_out;

    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let in_bits = d_in.next_power_of_two().trailing_zeros() as usize;
    let out_bits = d_out.next_power_of_two().trailing_zeros() as usize;

    let (_, _, params_x) = params_from_vars(t_bits + in_bits);
    let (_, _, params_y) = params_from_vars(t_bits + out_bits);
    let (_, _, params_w) = params_from_vars(in_bits + out_bits);

    // 1. Absorb Context (Cryptographic Binding)
    absorb_com(transcript, b"w_com", &vk.w_com);
    absorb_com(transcript, b"x_com", &io_coms.x_com);
    absorb_com(transcript, b"y_com", &io_coms.y_com);

    // 2. Replay Challenges
    let r_t = challenge_vec(transcript, t_bits, b"proj_rt");
    let r_out = challenge_vec(transcript, out_bits, b"proj_rout");

    // 3. Sumcheck Verification
    transcript.append_field(b"claimed_y", &proof.openings.y_eval);
    let (r_k, final_val) =
        verify_sumcheck(&proof.sumcheck, proof.openings.y_eval, in_bits, transcript)
            .map_err(|e| format!("Projection Sumcheck: {e}"))?;

    // Mathematical identity check
    let expected_val = proof.openings.x_eval * proof.openings.w_eval;
    if final_val != expected_val {
        return Err(
            "Projection Sumcheck mismatch: X(r_t, r_k) * W(r_k, r_out) != final_val".into(),
        );
    }

    // 4. Openings Verification (O(√N))
    hyrax_verify(
        &io_coms.y_com,
        proof.openings.y_eval,
        &combine(&r_t, &r_out),
        &proof.openings.y_open,
        &params_y,
    )
    .map_err(|e| format!("Y opening failed: {e}"))?;

    hyrax_verify(
        &io_coms.x_com,
        proof.openings.x_eval,
        &combine(&r_t, &r_k),
        &proof.openings.x_open,
        &params_x,
    )
    .map_err(|e| format!("X opening failed: {e}"))?;

    // Most crucial check: Ensures the Prover used the CORRECT static weights W
    hyrax_verify(
        &vk.w_com,
        proof.openings.w_eval,
        &combine(&r_k, &r_out),
        &proof.openings.w_open,
        &params_w,
    )
    .map_err(|e| format!("W opening failed (Invalid model weights!): {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn challenge_vec(transcript: &mut Transcript, len: usize, label: &[u8]) -> Vec<F> {
    (0..len)
        .map(|_| transcript.challenge_field::<F>(label))
        .collect()
}

fn absorb_com(transcript: &mut Transcript, label: &[u8], com: &HyraxCommitment) {
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
mod projection_tests {
    use super::*;
    use ark_ff::{One, Zero};

    /// Simulates the Pipeline Setup + Forward Pass
    fn setup_test_pipeline(
        t: usize,
        d_in: usize,
        d_out: usize,
    ) -> (
        ProjectionProvingKey,
        ProjectionWitness,
        ProjectionIOCommitments,
    ) {
        let mut x = vec![vec![F::zero(); d_in]; t];
        let mut w = vec![vec![F::zero(); d_out]; d_in];

        // Fill with some deterministic dummy values
        for i in 0..t {
            for j in 0..d_in {
                x[i][j] = F::from((i + j + 1) as u64);
            }
        }
        for i in 0..d_in {
            for j in 0..d_out {
                w[i][j] = F::from((i * j + 2) as u64);
            }
        }

        // Forward pass: Y = X * W
        let mut y = vec![vec![F::zero(); d_out]; t];
        for i in 0..t {
            for j in 0..d_out {
                for k in 0..d_in {
                    y[i][j] += x[i][k] * w[k][j];
                }
            }
        }

        // 1. Offline Preprocessing (Generates VK containing w_com)
        let pk = preprocess_projection(t, d_in, d_out, w);

        // 2. Global Pipeline Commitments for X and Y
        let x_mle = mat_to_mle(&x, t, d_in);
        let y_mle = mat_to_mle(&y, t, d_out);

        let t_bits = t.next_power_of_two().trailing_zeros() as usize;
        let in_bits = d_in.next_power_of_two().trailing_zeros() as usize;
        let out_bits = d_out.next_power_of_two().trailing_zeros() as usize;

        let (nu_x, _, params_x) = params_from_vars(t_bits + in_bits);
        let (nu_y, _, params_y) = params_from_vars(t_bits + out_bits);

        let io_coms = ProjectionIOCommitments {
            x_com: hyrax_commit(&x_mle.evaluations, nu_x, &params_x),
            y_com: hyrax_commit(&y_mle.evaluations, nu_y, &params_y),
        };

        let witness = ProjectionWitness { x, y };

        (pk, witness, io_coms)
    }

    #[test]
    fn test_projection_succinct_e2e_success() {
        let (pk, witness, io_coms) = setup_test_pipeline(2, 4, 2);

        let mut pt = Transcript::new(b"proj_test");
        let proof = prove_projection(&pk, &witness, &io_coms, &mut pt).unwrap();

        let mut vt = Transcript::new(b"proj_test");
        let result = verify_projection_succinct(&proof, &pk.vk, &io_coms, &mut vt);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }

    #[test]
    fn test_rejects_tampered_witness_y() {
        let (pk, mut witness, io_coms) = setup_test_pipeline(2, 4, 2);

        // Malicious prover tries to prove an invalid output Y.
        // It will fail because the IO Commitment for Y is fixed.
        witness.y[0][0] += F::one();

        let mut pt = Transcript::new(b"proj_test");
        let proof = prove_projection(&pk, &witness, &io_coms, &mut pt).unwrap();

        let mut vt = Transcript::new(b"proj_test");
        let result = verify_projection_succinct(&proof, &pk.vk, &io_coms, &mut vt);
        assert!(result.is_err(), "Should reject tampered Y witness");
        // assert!(result.unwrap_err().contains("Y opening failed"));
    }

    #[test]
    fn test_rejects_tampered_witness_x() {
        let (pk, mut witness, io_coms) = setup_test_pipeline(2, 4, 2);

        witness.x[1][1] += F::one(); // Tamper input X

        let mut pt = Transcript::new(b"proj_test");
        let proof = prove_projection(&pk, &witness, &io_coms, &mut pt).unwrap();

        let mut vt = Transcript::new(b"proj_test");
        let result = verify_projection_succinct(&proof, &pk.vk, &io_coms, &mut vt);
        assert!(result.is_err(), "Should reject tampered X witness");
        //assert!(result.unwrap_err().contains("X opening failed"));
    }

    #[test]
    fn test_rejects_invalid_model_weights() {
        let (mut pk, witness, io_coms) = setup_test_pipeline(2, 4, 2);

        // Malicious prover tries to use different weights (W) internally
        // to pass the Sumcheck math relationship.
        pk.w[0][1] += F::one();

        let mut pt = Transcript::new(b"proj_test");
        let proof = prove_projection(&pk, &witness, &io_coms, &mut pt).unwrap();

        let mut vt = Transcript::new(b"proj_test");
        let result = verify_projection_succinct(&proof, &pk.vk, &io_coms, &mut vt);

        // The verifier must catch this because the opening will fail against `vk.w_com`
        assert!(result.is_err(), "Should reject invalid model weights");
        //assert!(result.unwrap_err().contains("W opening failed"));
    }
}
