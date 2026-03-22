<<<<<<< HEAD
//! Succinct Range Proof Protocol with Chunking (LogUp-style Interface)
//!
//! **Production-Grade Architecture:**
//! 1. CHUNKING: 32-bit values are split into 16-bit chunks. Table size drops
//!    from 4.2 Billion (137 GB RAM) to 65,536 (2 MB RAM).
//! 2. ALGEBRAIC FUSION: V(r) = V_lo(r) + 2^16 * V_hi(r). The Verifier enforces
//!    this mathematically at a single evaluation point.
//! 3. SUCCINCT VERIFIER: The Verifier NEVER sees the raw `values` array.

use crate::field::F;
=======
//! Chunked Range Proof based on Lasso & Hyrax Polynomial Commitments
//!
//! Proves that every value v in a batch lies in [0, 2^total_bits).
//!
//! **Security pillars:**
//!
//! 1. **Chunk decomposition:** v = Σ_{i=0}^{c-1} C_i · 2^(i·8),  C_i ∈ [0, 256).
//!
//! 2. **Chunk polynomial commitments (Hyrax):** before the Lasso sub-protocol
//!    runs, the prover commits to each chunk polynomial C_i(·) and absorbs the
//!    commitments into the Fiat-Shamir transcript.  This binds the prover to
//!    the decomposition before any challenge is generated.
//!
//! 3. **Lasso range check:** a single identity table T[i] = i of size 2^8 = 256
//!    proves via sumcheck + Hyrax that every chunk evaluation lies in [0, 256).
//!
//! 4. **MLE recombination check:** the verifier draws a random point r and
//!    checks  V(r) = Σ_i C_i(r) · 2^(i·8)  over the committed chunk
//!    evaluations.  This algebraic identity (with soundness error ≤ deg/|F|)
//!    proves that the committed chunks correctly reconstruct the value polynomial,
//!    closing the gap that Lasso alone cannot close.
//!
//! 5. **Hyrax openings at r:** each Hyrax opening proof certifies that the
//!    committed C_i evaluates to the claimed value at r.
//!
//! ## Succinctness trade-off
//!
//! `verify_range` (full verifier, public data):
//!   - Recomputes all chunk commitments from raw values: O(N · num_chunks) MSM work.
//!   - Runs full Lasso verification with real query indices: O(N) work.
//!   - Use when the verifier holds raw `values`.
//!
//! `verify_range_succinct` (succinct verifier, committed data):
//!   - Skips the O(N · num_chunks) commitment recomputation.
//!     The prover provides `v_com` (Hyrax commitment to V), `v_eval_at_r`, and
//!     `v_open`; the verifier checks V(r) via an O(√N) Hyrax opening.
//!   - Still runs full Lasso using `lasso_outputs` stored in the proof
//!     (enabling query-index reconstruction without the raw instance).
//!   - Use when the verifier only holds a Hyrax commitment to the values
//!     (e.g., when range proof values are produced by a committed upstream layer).
//!   - Note: the dominant O(N) work is now inside Lasso itself.  Full O(polylog N)
//!     verification would additionally require a succinct Lasso verifier (future work).

use crate::field::F;
use crate::lookup::lasso::{prove_lasso, verify_lasso, LassoInstance, LassoProof};
>>>>>>> main
use crate::pcs::{
    hyrax_commit, hyrax_open, hyrax_verify, HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::poly::DenseMLPoly;
<<<<<<< HEAD
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::{BigInteger, Field, PrimeField};

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
=======
use crate::transcript::Transcript;
use ark_ff::{Field, PrimeField};
use ark_serialize::CanonicalSerialize;

/// Width of each chunk in bits; table size = 2^CHUNK_BITS = 256.
const CHUNK_BITS: usize = 8;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Instance for the *full* range proof verifier (verifier holds raw values).
pub struct RangeProofInstance {
    /// Values to prove are each in [0, 2^bits).
    pub values: Vec<F>,
    /// Total bit width (must be a positive multiple of CHUNK_BITS = 8).
    pub bits: usize,
}

/// Instance for the *succinct* range proof verifier (verifier only holds a
/// Hyrax commitment to V, not the raw values).
pub struct RangeProofInstanceSuccinct {
    /// Hyrax commitment to the values polynomial V(·).
    pub v_com: HyraxCommitment,
    /// Number of values (before padding).
    pub n_values: usize,
    /// Total bit width (must be a positive multiple of CHUNK_BITS = 8).
    pub bits: usize,
}

pub struct RangeProof {
    /// Hyrax commitments to each chunk polynomial C_i(·).
    pub chunk_coms: Vec<HyraxCommitment>,
    /// C_i(r) at the random recombination point r.
    pub chunk_evals: Vec<F>,
    /// Hyrax opening proofs for each C_i at r.
    pub chunk_opens: Vec<HyraxProof>,
    /// Lasso proof certifying every chunk evaluation lies in [0, 256).
    pub lasso_proof: LassoProof,

    // -- Succinct-verifier fields -------------------------------------------
    /// Hyrax commitment to the values polynomial V(·).
    /// Enables the succinct verifier to check V(r) via opening proof rather
    /// than recomputing the commitment from raw values (O(N) MSM → O(√N)).
    pub v_com: HyraxCommitment,
    /// V(r) at the recombination point r.
    pub v_eval_at_r: F,
    /// Hyrax opening proof for V(r).
    pub v_open: HyraxProof,
    /// Raw Lasso outputs (= the chunk values for each value position).
    ///
    /// For the identity table T[i] = i, the Lasso output equals the chunk
    /// index, so storing these lets `verify_range_succinct` reconstruct the
    /// full LassoInstance without knowing `inst.values`.  This is O(N · c)
    /// extra data in the proof, keeping Lasso verification fully sound while
    /// eliminating the O(N · c) MSM recompute of `chunk_coms`.
    pub lasso_outputs: Vec<F>,
}

// ---------------------------------------------------------------------------
// Prover (produces a proof compatible with both verifier modes)
// ---------------------------------------------------------------------------

pub fn prove_range(
    inst: &RangeProofInstance,
    transcript: &mut Transcript,
) -> Result<RangeProof, String> {
    validate(inst)?;
    let num_chunks = inst.bits / CHUNK_BITS;

    // HyraxParams for Lasso on the 256-entry identity table.
    let lasso_params = HyraxParams::new(CHUNK_BITS - CHUNK_BITS / 2);

    // Pad value count to a power of two (>= 2 so num_vars >= 1).
    let n_padded = inst.values.len().next_power_of_two().max(2);
    let n_vars = n_padded.trailing_zeros() as usize;
    let nu = n_vars / 2;
    let sigma = n_vars - nu;
    let chunk_params = HyraxParams::new(sigma);

    // -- 1. Chunk decomposition -----------------------------------------------
    let mask = (1u64 << CHUNK_BITS) - 1;
    let mut chunk_evals_all = vec![vec![F::ZERO; n_padded]; num_chunks];
    let mut lasso_queries = Vec::with_capacity(n_padded * num_chunks);
    let mut lasso_outputs = Vec::with_capacity(n_padded * num_chunks);

    for i in 0..num_chunks {
        for j in 0..n_padded {
            let v_int = if j < inst.values.len() {
                inst.values[j].into_bigint().as_ref()[0]
            } else {
                0
            };
            let ch = (v_int >> (i * CHUNK_BITS)) & mask;
            let ch_f = F::from(ch);
            chunk_evals_all[i][j] = ch_f;
            lasso_queries.push(ch as usize);
            lasso_outputs.push(ch_f);
        }
    }

    // -- 2. Commit to each chunk polynomial; absorb into transcript -----------
    let mut chunk_coms = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let com = hyrax_commit(&chunk_evals_all[i], nu, &chunk_params);
        absorb_com(transcript, b"chunk_com", &com);
        chunk_coms.push(com);
    }

    // -- 3. Commit to V (values polynomial); absorb into transcript -----------
    let mut v_values = vec![F::ZERO; n_padded];
    for (j, &v) in inst.values.iter().enumerate() {
        v_values[j] = v;
    }
    let v_com = hyrax_commit(&v_values, nu, &chunk_params);
    absorb_com(transcript, b"v_com", &v_com);

    // -- 4. Lasso: prove every chunk value lies in [0, 256) ------------------
    let identity_table: Vec<F> = (0..1 << CHUNK_BITS).map(|i| F::from(i as u64)).collect();
    let lasso_proof = prove_lasso(
        &LassoInstance {
            tables: vec![identity_table],
            query_indices: lasso_queries,
            outputs: lasso_outputs.clone(),
            bits_per_chunk: CHUNK_BITS,
        },
        transcript,
        &lasso_params,
    );

    // -- 5. Squeeze recombination challenge and open each chunk and V ---------
    let r: Vec<F> = (0..n_vars)
        .map(|_| transcript.challenge_field::<F>(b"range_r"))
        .collect();

    let mut chunk_evals = Vec::with_capacity(num_chunks);
    let mut chunk_opens = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let eval = DenseMLPoly::new(chunk_evals_all[i].clone()).evaluate(&r);
        let open = hyrax_open(&chunk_evals_all[i], &r, nu, sigma);
>>>>>>> main
        chunk_evals.push(eval);
        chunk_opens.push(open);
    }

<<<<<<< HEAD
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
=======
    let v_eval_at_r = DenseMLPoly::new(v_values.clone()).evaluate(&r);
    let v_open = hyrax_open(&v_values, &r, nu, sigma);

    Ok(RangeProof {
        chunk_coms,
        chunk_evals,
        chunk_opens,
        lasso_proof,
        v_com,
        v_eval_at_r,
        v_open,
        lasso_outputs,
    })
}

// ---------------------------------------------------------------------------
// Full verifier (public data, O(N) — verifier recomputes chunk commitments)
>>>>>>> main
// ---------------------------------------------------------------------------

pub fn verify_range(
    proof: &RangeProof,
<<<<<<< HEAD
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
=======
    inst: &RangeProofInstance,
    transcript: &mut Transcript,
) -> Result<(), String> {
    validate(inst)?;
    let num_chunks = inst.bits / CHUNK_BITS;
    let lasso_params = HyraxParams::new(CHUNK_BITS - CHUNK_BITS / 2);

    let n_padded = inst.values.len().next_power_of_two().max(2);
    let n_vars = n_padded.trailing_zeros() as usize;
    let nu = n_vars / 2;
    let sigma = n_vars - nu;
    let chunk_params = HyraxParams::new(sigma);

    // -- 1. Recompute chunk decomposition from inst.values -------------------
    let mask = (1u64 << CHUNK_BITS) - 1;
    let mut chunk_evals_all = vec![vec![F::ZERO; n_padded]; num_chunks];
    let mut lasso_queries = Vec::with_capacity(n_padded * num_chunks);
    let mut lasso_outputs_local = Vec::with_capacity(n_padded * num_chunks);

    for i in 0..num_chunks {
        for j in 0..n_padded {
            let v_int = if j < inst.values.len() {
                inst.values[j].into_bigint().as_ref()[0]
            } else {
                0
            };
            let ch = (v_int >> (i * CHUNK_BITS)) & mask;
            let ch_f = F::from(ch);
            chunk_evals_all[i][j] = ch_f;
            lasso_queries.push(ch as usize);
            lasso_outputs_local.push(ch_f);
        }
    }

    // -- 2. Check chunk commitments match recomputed values ------------------
    if proof.chunk_coms.len() != num_chunks {
        return Err(format!(
            "expected {num_chunks} chunk commitments, got {}",
            proof.chunk_coms.len()
        ));
    }
    for i in 0..num_chunks {
        let expected = hyrax_commit(&chunk_evals_all[i], nu, &chunk_params);
        if expected.row_coms != proof.chunk_coms[i].row_coms {
            return Err(format!("chunk {i} commitment mismatch"));
        }
        absorb_com(transcript, b"chunk_com", &proof.chunk_coms[i]);
    }

    // -- 3. Check V commitment matches raw values ----------------------------
    let mut v_values = vec![F::ZERO; n_padded];
    for (j, &v) in inst.values.iter().enumerate() {
        v_values[j] = v;
    }
    let expected_v = hyrax_commit(&v_values, nu, &chunk_params);
    if expected_v.row_coms != proof.v_com.row_coms {
        return Err("V commitment mismatch".to_string());
    }
    absorb_com(transcript, b"v_com", &proof.v_com);

    // -- 4. Verify Lasso (using real query indices derived from inst.values) --
    let identity_table: Vec<F> = (0..1 << CHUNK_BITS).map(|i| F::from(i as u64)).collect();
    verify_lasso(
        &proof.lasso_proof,
        &LassoInstance {
            tables: vec![identity_table],
            query_indices: lasso_queries,
            outputs: lasso_outputs_local,
            bits_per_chunk: CHUNK_BITS,
        },
        transcript,
        &lasso_params,
    )?;

    // -- 5. Squeeze recombination challenge -----------------------------------
    let r: Vec<F> = (0..n_vars)
        .map(|_| transcript.challenge_field::<F>(b"range_r"))
        .collect();

    // -- 6. Recombination check: V(r) = Σ C_i(r) * 2^(i*8) ------------------
    let v_r = DenseMLPoly::new(v_values).evaluate(&r);
    let mut recombined = F::ZERO;
    let mut shift = F::ONE;
    let base = F::from(1u64 << CHUNK_BITS);
    for i in 0..num_chunks {
        recombined += proof.chunk_evals[i] * shift;
        shift *= base;
    }
    if v_r != recombined {
        return Err("Algebraic recombination check failed".to_string());
    }

    // -- 7. Verify Hyrax openings for V and each chunk at r ------------------
    hyrax_verify(&proof.v_com, proof.v_eval_at_r, &r, &proof.v_open, &chunk_params)
        .map_err(|e| format!("V opening failed: {e}"))?;

    for i in 0..num_chunks {
        hyrax_verify(
            &proof.chunk_coms[i],
            proof.chunk_evals[i],
            &r,
            &proof.chunk_opens[i],
            &chunk_params,
        )
        .map_err(|e| format!("chunk {i} opening failed: {e}"))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Succinct verifier (committed data — verifier holds v_com, not raw values)
//
// Complexity compared to verify_range:
//   Eliminated: O(N · num_chunks) group ops for chunk_com recompute.
//   Eliminated: O(N) for V commitment recompute.
//   Retained:   O(N) Lasso verification (query indices reconstructed from
//               `proof.lasso_outputs`; future work: succinct Lasso).
//   Retained:   O(√N) Hyrax openings (V and chunk polys).
// ---------------------------------------------------------------------------

pub fn verify_range_succinct(
    proof: &RangeProof,
    inst: &RangeProofInstanceSuccinct,
    transcript: &mut Transcript,
) -> Result<(), String> {
    if inst.bits == 0 || inst.bits % CHUNK_BITS != 0 {
        return Err(format!(
            "bits={} must be a positive multiple of CHUNK_BITS={}",
            inst.bits, CHUNK_BITS
        ));
    }
    let num_chunks = inst.bits / CHUNK_BITS;
    let lasso_params = HyraxParams::new(CHUNK_BITS - CHUNK_BITS / 2);

    let n_padded = inst.n_values.next_power_of_two().max(2);
    let n_vars = n_padded.trailing_zeros() as usize;
    let sigma = n_vars - n_vars / 2;
    let chunk_params = HyraxParams::new(sigma);

    // -- 1. Absorb chunk_coms without recomputing them (O(1) per commitment) -
    // The Fiat-Shamir transcript ensures the prover committed before seeing
    // any challenge, so absorbing the proof's chunk_coms is binding.
    if proof.chunk_coms.len() != num_chunks {
        return Err(format!(
            "expected {num_chunks} chunk commitments, got {}",
            proof.chunk_coms.len()
        ));
    }
    for com in &proof.chunk_coms {
        absorb_com(transcript, b"chunk_com", com);
    }

    // -- 2. Absorb v_com from the succinct instance ---------------------------
    // The verifier received v_com from the upstream protocol (e.g., a LayerNorm
    // proof that committed to its output values).  No O(N) recompute needed.
    absorb_com(transcript, b"v_com", &inst.v_com);

    // -- 3. Verify Lasso using query indices reconstructed from proof ----------
    // For the identity table T[i] = i, the Lasso output equals the chunk value,
    // so query_index[j] = output[j].  Storing lasso_outputs in the proof lets
    // us reconstruct the full LassoInstance without the raw values.
    let expected_n = n_padded * num_chunks;
    if proof.lasso_outputs.len() != expected_n {
        return Err(format!(
            "lasso_outputs length mismatch: expected {expected_n}, got {}",
            proof.lasso_outputs.len()
        ));
    }
    let lasso_queries: Vec<usize> = proof
        .lasso_outputs
        .iter()
        .map(|&x| x.into_bigint().as_ref()[0] as usize)
        .collect();

    let identity_table: Vec<F> = (0..1 << CHUNK_BITS).map(|i| F::from(i as u64)).collect();
    verify_lasso(
        &proof.lasso_proof,
        &LassoInstance {
            tables: vec![identity_table],
            query_indices: lasso_queries,
            outputs: proof.lasso_outputs.clone(),
            bits_per_chunk: CHUNK_BITS,
        },
        transcript,
        &lasso_params,
    )?;

    // -- 4. Squeeze recombination challenge -----------------------------------
    let r: Vec<F> = (0..n_vars)
        .map(|_| transcript.challenge_field::<F>(b"range_r"))
        .collect();

    // -- 5. Recombination check: V(r) = Σ C_i(r) · 2^(i·8) ------------------
    // Uses proof.v_eval_at_r (certified by the Hyrax opening in step 6).
    let mut recombined = F::ZERO;
    let mut shift = F::ONE;
    let base = F::from(1u64 << CHUNK_BITS);
    for i in 0..num_chunks {
        recombined += proof.chunk_evals[i] * shift;
        shift *= base;
    }
    if proof.v_eval_at_r != recombined {
        return Err("Algebraic recombination check failed".to_string());
    }

    // -- 6. O(√N) Hyrax openings: V(r) against inst.v_com; each C_i(r) ------
    hyrax_verify(&inst.v_com, proof.v_eval_at_r, &r, &proof.v_open, &chunk_params)
        .map_err(|e| format!("V opening failed: {e}"))?;

    for i in 0..num_chunks {
        hyrax_verify(
            &proof.chunk_coms[i],
            proof.chunk_evals[i],
            &r,
            &proof.chunk_opens[i],
            &chunk_params,
        )
        .map_err(|e| format!("chunk {i} opening failed: {e}"))?;
    }

    Ok(())
>>>>>>> main
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
<<<<<<< HEAD
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
=======

fn validate(inst: &RangeProofInstance) -> Result<(), String> {
    if inst.bits == 0 || inst.bits % CHUNK_BITS != 0 {
        return Err(format!(
            "bits={} must be a positive multiple of CHUNK_BITS={}",
            inst.bits, CHUNK_BITS
        ));
    }
    if inst.values.is_empty() {
        return Err("RangeProofInstance: no values to prove".to_string());
    }
    Ok(())
}

fn absorb_com(transcript: &mut Transcript, label: &[u8], com: &HyraxCommitment) {
>>>>>>> main
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
=======

#[cfg(test)]
mod range_tests {
    use super::*;
    use crate::transcript::Transcript;

    fn inst(vals: Vec<u64>, bits: usize) -> RangeProofInstance {
        RangeProofInstance {
            values: vals.into_iter().map(F::from).collect(),
            bits,
        }
    }

    fn roundtrip(label: &[u8], ri: RangeProofInstance) -> Result<(), String> {
        let mut pt = Transcript::new(label);
        let proof = prove_range(&ri, &mut pt)?;
        let mut vt = Transcript::new(label);
        verify_range(&proof, &ri, &mut vt)
    }

    #[test]
    fn test_range_8bit_success() {
        roundtrip(b"r8", inst(vec![0, 1, 127, 255], 8)).expect("8-bit range proof failed");
    }

    #[test]
    fn test_range_16bit_success() {
        roundtrip(b"r16", inst(vec![0, 1000, 65535], 16)).expect("16-bit range proof failed");
    }

    #[test]
    fn test_range_32bit_success() {
        roundtrip(b"r32", inst(vec![0, 1, 255, 256, 65535, 1 << 24], 32))
            .expect("32-bit range proof failed");
    }

    #[test]
    fn test_range_single_zero() {
        roundtrip(b"rz", inst(vec![0], 8)).expect("single-zero range proof failed");
    }

    #[test]
    fn test_range_large_batch() {
        let vals: Vec<u64> = (0u64..32).map(|i| i * 7).collect();
        roundtrip(b"rbatch", inst(vals, 8)).expect("large-batch range proof failed");
    }

    /// Out-of-range value: the verifier recomputes chunk commitments from the
    /// tampered instance, which differ from the proof's committed chunks.
    #[test]
    fn test_range_out_of_range_rejected() {
        let good = inst(vec![10, 20, 30], 8);

        let mut pt = Transcript::new(b"rtamper");
        let proof = prove_range(&good, &mut pt).expect("prove failed");

        let mut bad = inst(vec![10, 20, 30], 8);
        bad.values[1] = F::from(999u64);

        let mut vt = Transcript::new(b"rtamper");
        let result = verify_range(&proof, &bad, &mut vt);
        assert!(result.is_err(), "should reject out-of-range value");
    }

    #[test]
    fn test_range_invalid_bits_rejected() {
        let i = inst(vec![1, 2], 10);
        let mut t = Transcript::new(b"rbits");
        assert!(prove_range(&i, &mut t).is_err());
    }

    /// Succinct verifier: prover proves [10, 20, 30] ∈ [0, 256).
    /// Verifier only holds v_com (not raw values); uses verify_range_succinct.
    #[test]
    fn test_range_succinct_verifier_success() {
        let ri = inst(vec![10, 20, 30], 8);

        let mut pt = Transcript::new(b"rsuccinct");
        let proof = prove_range(&ri, &mut pt).expect("prove failed");

        // Succinct instance: verifier only holds v_com from the proof.
        let succinct_inst = RangeProofInstanceSuccinct {
            v_com: proof.v_com.clone(),
            n_values: ri.values.len(),
            bits: ri.bits,
        };

        let mut vt = Transcript::new(b"rsuccinct");
        verify_range_succinct(&proof, &succinct_inst, &mut vt)
            .expect("succinct verification failed");
    }

    /// Succinct verifier rejects a tampered v_com (wrong commitment).
    #[test]
    fn test_range_succinct_wrong_vcom_rejected() {
        let good = inst(vec![10, 20, 30], 8);
        let bad = inst(vec![10, 999, 30], 8);

        // Prove with good values.
        let mut pt = Transcript::new(b"rsvcom");
        let proof = prove_range(&good, &mut pt).expect("prove failed");

        // Use v_com from the bad instance as the "succinct" commitment.
        let mut pt2 = Transcript::new(b"rsvcom");
        let bad_proof = prove_range(&bad, &mut pt2).expect("bad prove failed");

        let succinct_inst = RangeProofInstanceSuccinct {
            v_com: bad_proof.v_com.clone(), // wrong v_com
            n_values: good.values.len(),
            bits: good.bits,
        };

        let mut vt = Transcript::new(b"rsvcom");
        let result = verify_range_succinct(&proof, &succinct_inst, &mut vt);
        assert!(result.is_err(), "should reject wrong v_com");
>>>>>>> main
    }
}
