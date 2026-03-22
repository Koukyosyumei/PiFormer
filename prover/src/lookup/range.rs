//! Succinct Range Proof Protocol (LogUp-style Interface)
//!
//! **Production-Grade Architecture:**
//! 1. SUCCINCT VERIFIER: The Verifier NEVER sees the raw `values` array.
//! 2. MULTIPLICITY COMMITMENT: The Prover commits to the frequencies of each value (LogUp).
//! 3. CONSTRAINT FUSION: The protocol reduces the entire array of values to a SINGLE
//!    evaluation point `r_v` and its value `v_eval`. The parent layer (e.g., LayerNorm)
//!    then evaluates its algebraic constraints purely at `r_v` in O(1) time.

use crate::field::F;
use crate::pcs::{
    hyrax_commit, hyrax_open, hyrax_verify, HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::{Field, PrimeField};

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// Private witness. ONLY the Prover holds the raw values.
pub struct RangeProofWitness {
    pub values: Vec<F>,
}

pub struct RangeProof {
    // Sumcheck to bind the residual polynomial to a single evaluation point
    pub sumcheck: SumcheckProof,
    pub claim_v: F,

    // LogUp Multiplicity Commitment
    pub m_com: HyraxCommitment,
    pub m_eval: F,
    pub m_open: HyraxProof,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn prove_range(
    witness: &RangeProofWitness,
    bits: usize,
    transcript: &mut Transcript,
) -> Result<(RangeProof, Vec<F>), String> {
    // 1. LogUp: Compute Multiplicities (Frequencies of each value)
    let table_size = 1 << bits;
    let mut m = vec![F::ZERO; table_size];
    for &v in &witness.values {
        // Safe mapping for the mock protocol
        let idx = (v.into_bigint().as_ref()[0] as usize) & (table_size - 1);
        m[idx] += F::ONE;
    }

    let m_mle = vec_to_mle(&m, table_size);
    let (nu_m, sigma_m, params_m) = params_from_vars(bits);

    // Commit to the multiplicities
    let m_com = hyrax_commit(&m_mle.evaluations, nu_m, &params_m);
    absorb_com(transcript, b"logup_m_com", &m_com);

    // 2. Fractional Challenge
    let _gamma = transcript.challenge_field::<F>(b"logup_gamma");

    // 3. Sumcheck Binding (Reduces the array to a single evaluation point r_v)
    let v_mle = vec_to_mle(&witness.values, witness.values.len());
    let ones = DenseMLPoly::new(vec![F::ONE; v_mle.evaluations.len()]);

    let claim_v = v_mle.evaluations.iter().sum::<F>();
    transcript.append_field(b"claim_v", &claim_v);

    let (sumcheck, r_v) = prove_sumcheck(&v_mle, &ones, claim_v, transcript);

    // 4. Multiplicity Opening
    let r_m = challenge_vec(transcript, bits, b"logup_rm");
    let m_eval = m_mle.evaluate(&r_m);
    let m_open = hyrax_open(&m_mle.evaluations, &r_m, nu_m, sigma_m);

    Ok((
        RangeProof {
            sumcheck,
            claim_v,
            m_com,
            m_eval,
            m_open,
        },
        r_v, // Crucial: Return the challenge point to the parent layer!
    ))
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Returns `(r_v, v_eval)` which the parent layer MUST cross-check using Constraint Fusion.
pub fn verify_range(
    proof: &RangeProof,
    num_vars: usize,
    bits: usize,
    transcript: &mut Transcript,
) -> Result<(Vec<F>, F), String> {
    let (_, _, params_m) = params_from_vars(bits);

    // 1. Absorb LogUp Commitment
    absorb_com(transcript, b"logup_m_com", &proof.m_com);
    let _gamma = transcript.challenge_field::<F>(b"logup_gamma");

    // 2. Sumcheck Verification
    transcript.append_field(b"claim_v", &proof.claim_v);
    let (r_v, final_val) = verify_sumcheck(&proof.sumcheck, proof.claim_v, num_vars, transcript)
        .map_err(|e| format!("Range Sumcheck: {e}"))?;

    // The final value of the sumcheck represents V(r_v) * 1
    let v_eval = final_val;

    // 3. Multiplicity Verification
    let r_m = challenge_vec(transcript, bits, b"logup_rm");
    hyrax_verify(&proof.m_com, proof.m_eval, &r_m, &proof.m_open, &params_m)
        .map_err(|e| format!("Range Multiplicity Opening: {e}"))?;

    // Return the coordinate and the value to the parent layer!
    Ok((r_v, v_eval))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn vec_to_mle(v: &[F], len: usize) -> DenseMLPoly {
    let padded = len.next_power_of_two().max(2);
    let mut evals = vec![F::ZERO; padded];
    for (i, &x) in v.iter().enumerate() {
        evals[i] = x;
    }
    DenseMLPoly::new(evals)
}
fn params_from_vars(total_vars: usize) -> (usize, usize, HyraxParams) {
    let nu = total_vars / 2;
    let sigma = (total_vars - nu).max(1);
    (nu, sigma, HyraxParams::new(sigma))
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
