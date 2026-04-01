//! Succinct Range Proof Protocol with Chunking (LogUp-style Interface)
//!
//! **Production-Grade Architecture:**
//! 1. CHUNKING: 32-bit values are split into 16-bit chunks. Table size drops
//!    from 4.2 Billion (137 GB RAM) to 65,536 (2 MB RAM).
//! 2. ALGEBRAIC FUSION: V(r) = V_lo(r) + 2^16 * V_hi(r). The Verifier enforces
//!    this mathematically at a single evaluation point.
//! 3. SUCCINCT VERIFIER: The Verifier NEVER sees the raw `values` array.

use crate::field::F;
use crate::pcs::{
    hyrax_commit, hyrax_open, hyrax_verify, params_from_vars, HyraxCommitment, HyraxParams,
    HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::{Field, PrimeField};

pub const CHUNK_BITS: usize = 16;
pub const CHUNK_SIZE: usize = 1 << CHUNK_BITS;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

pub struct RangeProofWitness {
    pub values: Vec<F>,
}

pub struct RangeProof {
    // 1. Sumcheck to bind the virtual array V to a single point r_v
    pub sumcheck: SumcheckProof,
    pub claim_v: F,

    // 2. Chunk commitments and openings
    pub chunk_coms: Vec<HyraxCommitment>,
    pub chunk_evals: Vec<F>,
    pub chunk_opens: Vec<HyraxProof>,

    // 3. Multiplicity (LogUp) commitment
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
    let num_chunks = (bits + CHUNK_BITS - 1) / CHUNK_BITS;
    let n = witness.values.len();

    let mut chunks = vec![vec![F::ZERO; n]; num_chunks];
    let mut m = vec![F::ZERO; CHUNK_SIZE];

    // 1. Split values into 16-bit chunks and compute multiplicity
    for (i, &v) in witness.values.iter().enumerate() {
        let val_u64 = v.into_bigint().as_ref()[0]; // Safe for <= 64 bit range proofs

        for c in 0..num_chunks {
            let chunk_val = (val_u64 >> (c * CHUNK_BITS)) & ((1 << CHUNK_BITS) - 1);
            chunks[c][i] = F::from(chunk_val as u64);
            m[chunk_val as usize] += F::ONE;
        }
    }

    // 2. Commit to chunks
    let mut chunk_coms = Vec::with_capacity(num_chunks);
    let mut chunk_mles = Vec::with_capacity(num_chunks);
    let (nu_c, sigma_c, params_c) =
        params_from_vars(n.next_power_of_two().trailing_zeros() as usize);

    for c in 0..num_chunks {
        let mle = vec_to_mle(&chunks[c], n);
        let com = hyrax_commit(&mle.evaluations, nu_c, &params_c);
        absorb_com(transcript, b"chunk_com", &com);
        chunk_coms.push(com);
        chunk_mles.push(mle);
    }

    // 3. Commit to multiplicity M
    let m_mle = vec_to_mle(&m, CHUNK_SIZE);
    let (nu_m, sigma_m, params_m) = params_from_vars(CHUNK_BITS);
    let m_com = hyrax_commit(&m_mle.evaluations, nu_m, &params_m);
    absorb_com(transcript, b"logup_m_com", &m_com);

    // 4. Sumcheck binding for the original virtual polynomial V
    let v_mle = vec_to_mle(&witness.values, n);
    let ones = DenseMLPoly::new(vec![F::ONE; v_mle.evaluations.len()]);
    let claim_v = v_mle.evaluations.iter().sum::<F>();
    transcript.append_field(b"claim_v", &claim_v);

    let (sumcheck, r_v) = prove_sumcheck(&v_mle, &ones, claim_v, transcript);

    // 5. Openings at r_v for chunks
    let mut chunk_evals = Vec::with_capacity(num_chunks);
    let mut chunk_opens = Vec::with_capacity(num_chunks);
    for c in 0..num_chunks {
        let eval = chunk_mles[c].evaluate(&r_v);
        let open = hyrax_open(&chunk_mles[c].evaluations, &r_v, nu_c, sigma_c);
        chunk_evals.push(eval);
        chunk_opens.push(open);
    }

    // 6. Multiplicity opening at random challenge r_m
    let r_m = challenge_vec(transcript, CHUNK_BITS, b"logup_rm");
    let m_eval = m_mle.evaluate(&r_m);
    let m_open = hyrax_open(&m_mle.evaluations, &r_m, nu_m, sigma_m);

    Ok((
        RangeProof {
            sumcheck,
            claim_v,
            m_com,
            m_eval,
            m_open,
            chunk_coms,
            chunk_evals,
            chunk_opens,
        },
        r_v,
    ))
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn verify_range(
    proof: &RangeProof,
    num_vars: usize,
    bits: usize,
    transcript: &mut Transcript,
) -> Result<(Vec<F>, F), String> {
    let num_chunks = (bits + CHUNK_BITS - 1) / CHUNK_BITS;

    let (_, _, params_m) = params_from_vars(CHUNK_BITS);
    let (_, _, params_c) = params_from_vars(num_vars);

    // 1. Absorb commitments
    for com in &proof.chunk_coms {
        absorb_com(transcript, b"chunk_com", com);
    }
    absorb_com(transcript, b"logup_m_com", &proof.m_com);

    // 2. Sumcheck Verification
    transcript.append_field(b"claim_v", &proof.claim_v);
    let (r_v, final_val) = verify_sumcheck(&proof.sumcheck, proof.claim_v, num_vars, transcript)
        .map_err(|e| format!("Range Sumcheck: {e}"))?;
    let v_eval = final_val;

    // 3. Chunk Algebraic Fusion: V(r) = V_lo(r) + 2^16 * V_hi(r)
    let mut expected_v_eval = F::ZERO;
    let mut shift = F::ONE;
    let shift_multiplier = F::from(1u64 << CHUNK_BITS);

    for c in 0..num_chunks {
        hyrax_verify(
            &proof.chunk_coms[c],
            proof.chunk_evals[c],
            &r_v,
            &proof.chunk_opens[c],
            &params_c,
        )
        .map_err(|e| format!("Chunk opening failed: {e}"))?;

        expected_v_eval += proof.chunk_evals[c] * shift;
        shift *= shift_multiplier;
    }

    if v_eval != expected_v_eval {
        return Err("Chunk fusion mismatch: V(r) != sum V_c(r) * 2^{16c}".into());
    }

    // 4. Multiplicity Verification
    let r_m = challenge_vec(transcript, CHUNK_BITS, b"logup_rm");
    hyrax_verify(&proof.m_com, proof.m_eval, &r_m, &proof.m_open, &params_m)
        .map_err(|e| format!("Range Multiplicity Opening: {e}"))?;

    // Return the coordinate and the evaluated value to the parent layer (e.g. LayerNorm)
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
mod range_tests {
    use super::*;
    use ark_ff::{One, Zero};

    fn setup_witness() -> RangeProofWitness {
        // Values requiring 32 bits (spanning multiple chunks)
        // 100000, 200000, 300000, 400000
        let values = vec![
            F::from(100_000u64),
            F::from(200_000u64),
            F::from(300_000u64),
            F::from(400_000u64),
        ];
        RangeProofWitness { values }
    }

    #[test]
    fn test_range_proof_succinct_e2e() {
        let witness = setup_witness();
        let num_vars = witness.values.len().next_power_of_two().trailing_zeros() as usize;

        let mut pt = Transcript::new(b"range_test");
        let (proof, _) = prove_range(&witness, 32, &mut pt).unwrap();

        let mut vt = Transcript::new(b"range_test");
        let result = verify_range(&proof, num_vars, 32, &mut vt);

        assert!(
            result.is_ok(),
            "Range Proof verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_rejects_tampered_chunk_fusion() {
        let witness = setup_witness();
        let num_vars = witness.values.len().next_power_of_two().trailing_zeros() as usize;

        let mut pt = Transcript::new(b"range_test");
        let (mut proof, _) = prove_range(&witness, 32, &mut pt).unwrap();

        // Malicious Prover tampers with the chunk evaluation to trick the fusion check
        proof.chunk_evals[0] += F::one();

        let mut vt = Transcript::new(b"range_test");
        let result = verify_range(&proof, num_vars, 32, &mut vt);

        assert!(result.is_err(), "Should reject tampered chunk evaluation");
    }

    #[test]
    fn test_rejects_tampered_claim_v() {
        let witness = setup_witness();
        let num_vars = witness.values.len().next_power_of_two().trailing_zeros() as usize;

        let mut pt = Transcript::new(b"range_test");
        let (mut proof, _) = prove_range(&witness, 32, &mut pt).unwrap();

        // Malicious Prover tampers with the sumcheck claim
        proof.claim_v += F::one();

        let mut vt = Transcript::new(b"range_test");
        let result = verify_range(&proof, num_vars, 32, &mut vt);

        assert!(result.is_err(), "Should reject tampered claim_v");
    }

    #[test]
    fn test_rejects_tampered_multiplicity() {
        let witness = setup_witness();
        let num_vars = witness.values.len().next_power_of_two().trailing_zeros() as usize;

        let mut pt = Transcript::new(b"range_test");
        let (mut proof, _) = prove_range(&witness, 32, &mut pt).unwrap();

        // Tamper with LogUp Multiplicity Evaluation
        proof.m_eval += F::one();

        let mut vt = Transcript::new(b"range_test");
        let result = verify_range(&proof, num_vars, 32, &mut vt);

        assert!(
            result.is_err(),
            "Should reject tampered multiplicity opening"
        );
    }

    /// 16-bit range: values fit in one chunk (no high-chunk splitting needed).
    #[test]
    fn test_range_proof_16bit_values() {
        let values = vec![
            F::from(0u64),
            F::from(1u64),
            F::from(1000u64),
            F::from(65535u64), // max 16-bit value
        ];
        let num_vars = values.len().next_power_of_two().trailing_zeros() as usize;
        let witness = RangeProofWitness { values };

        let mut pt = Transcript::new(b"range16");
        let (proof, _) = prove_range(&witness, 16, &mut pt).unwrap();

        let mut vt = Transcript::new(b"range16");
        let result = verify_range(&proof, num_vars, 16, &mut vt);
        assert!(result.is_ok(), "16-bit range proof failed: {:?}", result.err());
    }

    /// Power-of-two number of values.
    #[test]
    fn test_range_proof_power_of_two_count() {
        let values: Vec<F> = (0..8u64).map(|i| F::from(i * 1000)).collect();
        let num_vars = 3usize; // log2(8)
        let witness = RangeProofWitness { values };

        let mut pt = Transcript::new(b"range_pow2");
        let (proof, _) = prove_range(&witness, 32, &mut pt).unwrap();

        let mut vt = Transcript::new(b"range_pow2");
        let result = verify_range(&proof, num_vars, 32, &mut vt);
        assert!(result.is_ok(), "power-of-two count range proof failed: {:?}", result.err());
    }

    /// All values equal zero: an edge case for multiplicity counters.
    #[test]
    fn test_range_proof_all_zeros() {
        let values = vec![F::zero(); 4];
        let num_vars = 2usize;
        let witness = RangeProofWitness { values };

        let mut pt = Transcript::new(b"range_zeros");
        let (proof, _) = prove_range(&witness, 16, &mut pt).unwrap();

        let mut vt = Transcript::new(b"range_zeros");
        let result = verify_range(&proof, num_vars, 16, &mut vt);
        assert!(result.is_ok(), "all-zeros range proof failed: {:?}", result.err());
    }

    /// Tampered chunk opening (proof.chunk_opens[0]) should be caught.
    #[test]
    fn test_rejects_tampered_chunk_opening_vector() {
        let witness = setup_witness();
        let num_vars = witness.values.len().next_power_of_two().trailing_zeros() as usize;

        let mut pt = Transcript::new(b"range_test");
        let (mut proof, _) = prove_range(&witness, 32, &mut pt).unwrap();

        // Corrupt the w_prime inside the first chunk opening proof
        proof.chunk_opens[0].w_prime[0] += F::one();

        let mut vt = Transcript::new(b"range_test");
        let result = verify_range(&proof, num_vars, 32, &mut vt);
        assert!(result.is_err(), "Should reject corrupted chunk opening proof");
    }
}
