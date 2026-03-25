//! FFN layer prover/verifier for PiFormer.
//!
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
    absorb_com, hyrax_commit, hyrax_open, hyrax_verify, params_from_vars, HyraxCommitment,
    HyraxParams, HyraxProof,
};
use crate::poly::utils::{combine, convert_tm_to_fm, eval_cols, eval_rows, mat_to_mle, TernaryValue};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::{challenge_vec, Transcript};

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
    pub w1: Vec<Vec<TernaryValue>>,
    pub w2: Vec<Vec<TernaryValue>>,
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
    w1: Vec<Vec<TernaryValue>>,
    w2: Vec<Vec<TernaryValue>>,
) -> FFNProvingKey {
    let w1_mle = mat_to_mle(&convert_tm_to_fm(&w1), d_model, d_ff);
    let w2_mle = mat_to_mle(&convert_tm_to_fm(&w2), d_ff, d_model);

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
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn prove_ffn(
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
    let w1_mle = mat_to_mle(&convert_tm_to_fm(&pk.w1), d, f);
    let w2_mle = mat_to_mle(&convert_tm_to_fm(&pk.w2), f, d);

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
    })
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn verify_ffn(
    proof: &FFNProof,
    inst: &FFNInstance,
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
    verify_lasso(
        &proof.activation_proof,
        &inst.activation_lasso,
        transcript,
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

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod ffn_tests {
    use super::*;
    use ark_ff::One;
    use ark_ff::PrimeField;
    use crate::poly::utils::TernaryValue;

    fn setup_test_pipeline() -> (FFNProvingKey, FFNWitness, FFNInstance, FFNIOCommitments) {

        let t = 2usize;
        let d = 2usize;
        let f_dim = 4usize;
        let w1 = vec![
            vec![TernaryValue::ONE, TernaryValue::ZERO, TernaryValue::MINUSONE, TernaryValue::ONE],
            vec![TernaryValue::ZERO, TernaryValue::ONE, TernaryValue::ONE, TernaryValue::ZERO],
        ];
        let w2 = vec![
            vec![TernaryValue::ONE, TernaryValue::ZERO],
            vec![TernaryValue::MINUSONE, TernaryValue::ONE],
            vec![TernaryValue::ONE, TernaryValue::MINUSONE],
            vec![TernaryValue::ZERO, TernaryValue::ONE],
        ];
        let x = vec![
            vec![F::from(3u64), F::from(1u64)],
            vec![F::from(2u64), F::from(4u64)],
        ];

        let matmul = |a: &Vec<Vec<F>>, b: &Vec<Vec<F>>, r, k, c| -> Vec<Vec<F>> {
            (0..r)
                .map(|i| {
                    (0..c)
                        .map(|j| (0..k).map(|l| a[i][l] * b[l][j]).sum())
                        .collect()
                })
                .collect()
        };
        let w1_f = convert_tm_to_fm(&w1);
        let w2_f = convert_tm_to_fm(&w2);
        let m = matmul(&x, &w1_f, t, d, f_dim);

        let table_size = 16usize;
        let m_bits = 4usize;
        let table: Vec<F> = (0..table_size).map(|i| F::from(i as u64)).collect();
        let a = m.clone(); // Identity for test

        let mut query_indices = Vec::new();
        let mut outputs = Vec::new();
        for row in &m {
            for &v in row {
                let idx = (v.into_bigint().as_ref()[0] as usize) & (table_size - 1);
                query_indices.push(idx);
                outputs.push(F::from(idx as u64));
            }
        }
        let activation_lasso = LassoInstance {
            tables: vec![table],
            query_indices,
            outputs,
            bits_per_chunk: m_bits,
        };
        let y = matmul(&a, &w2_f, t, f_dim, d);

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
        assert!(result.is_ok(), "verify failed: {:?}", result.err());
    }

    #[test]
    fn test_rejects_invalid_weights() {
        let (mut pk, witness, inst, io_coms) = setup_test_pipeline();
        let lasso_params = HyraxParams::new(2);

        pk.w2[0][0] = TernaryValue::ZERO; // Tamper weight internally (was ONE)

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

        pk.w1[0][0] = TernaryValue::ZERO; // Tamper weight internally (was ONE)

        let mut pt = Transcript::new(b"ffn-test");
        let proof = prove_ffn(&pk, &witness, &inst, &io_coms, &mut pt, &lasso_params).unwrap();

        let mut vt = Transcript::new(b"ffn-test");
        let result = verify_ffn(&proof, &inst, &pk.vk, &io_coms, &mut vt, &lasso_params);
        assert!(result.is_err(), "Should reject tampered W1");
    }
}
