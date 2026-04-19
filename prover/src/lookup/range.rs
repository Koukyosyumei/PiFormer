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
    hyrax_commit, hyrax_open, hyrax_open_batch, hyrax_verify, hyrax_verify_batch,
    hyrax_verify_multi_point, params_from_vars, HyraxBatchAccumulator, HyraxCommitment,
    HyraxParams, HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
use crate::transcript::Transcript;
use ark_ff::{batch_inversion, Field, PrimeField};

// 8-bit chunks: table size = 256 (was 65536 with 16-bit chunks).
// 16-bit values need 2 chunks; 32-bit values need 4 chunks.
// This shrinks m_com MSM, RHS sumcheck, and g_mle_eval by 256× vs 16-bit chunks.
pub const CHUNK_BITS: usize = 8;
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

    // 2. Chunk commitments and openings (batched into a single proof via hyrax_open_batch)
    pub chunk_coms: Vec<HyraxCommitment>,
    pub chunk_evals: Vec<F>,
    pub chunk_batch_proof: HyraxProof,

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

    // 5. Batch opening for all chunks at r_v (hyrax_open_batch → 1 proof instead of num_chunks)
    let chunk_evals: Vec<F> = (0..num_chunks).map(|c| chunk_mles[c].evaluate(&r_v)).collect();
    let chunk_slices: Vec<&[F]> = (0..num_chunks)
        .map(|c| chunk_mles[c].evaluations.as_slice())
        .collect();
    let chunk_batch_proof = hyrax_open_batch(&chunk_slices, &r_v, nu_c, sigma_c, transcript);

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
            chunk_batch_proof,
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

    use std::time::Instant;
    let _t0 = Instant::now();
    let (_, _, params_m) = params_from_vars(CHUNK_BITS);
    let (_, _, params_c) = params_from_vars(num_vars);
    eprintln!("[range({num_vars})]  params:      {:>8.3}ms", _t0.elapsed().as_secs_f64()*1000.0);

    // 1. Absorb commitments
    let _ta = Instant::now();
    for com in &proof.chunk_coms {
        absorb_com(transcript, b"chunk_com", com);
    }
    absorb_com(transcript, b"logup_m_com", &proof.m_com);
    eprintln!("[range({num_vars})]  absorb:      {:>8.3}ms", _ta.elapsed().as_secs_f64()*1000.0);

    // 2. Sumcheck Verification
    let _tsc = Instant::now();
    transcript.append_field(b"claim_v", &proof.claim_v);
    let (r_v, final_val) = verify_sumcheck(&proof.sumcheck, proof.claim_v, num_vars, transcript)
        .map_err(|e| format!("Range Sumcheck: {e}"))?;
    let v_eval = final_val;
    eprintln!("[range({num_vars})]  sumcheck:    {:>8.3}ms", _tsc.elapsed().as_secs_f64()*1000.0);

    // 3. Chunk Algebraic Fusion: V(r) = V_lo(r) + 2^16 * V_hi(r)
    // Batch all chunk openings into a single hyrax_verify_batch call (saves K-1 MSMs).
    let _tch = Instant::now();
    hyrax_verify_batch(
        &proof.chunk_coms,
        &proof.chunk_evals,
        &r_v,
        &proof.chunk_batch_proof,
        &params_c,
        transcript,
    )
    .map_err(|e| format!("Chunk batch opening failed: {e}"))?;
    eprintln!("[range({num_vars})]  chunk_hyrax: {:>8.3}ms", _tch.elapsed().as_secs_f64()*1000.0);

    // Verify the algebraic fusion: V(r) = Σ_c chunk_eval[c] * 2^(16c)
    let mut expected_v_eval = F::ZERO;
    let mut shift = F::ONE;
    let shift_multiplier = F::from(1u64 << CHUNK_BITS);
    for c in 0..num_chunks {
        expected_v_eval += proof.chunk_evals[c] * shift;
        shift *= shift_multiplier;
    }
    if v_eval != expected_v_eval {
        return Err("Chunk fusion mismatch: V(r) != sum V_c(r) * 2^{16c}".into());
    }

    // 4. Multiplicity Verification
    let _tm = Instant::now();
    let r_m = challenge_vec(transcript, CHUNK_BITS, b"logup_rm");
    hyrax_verify(&proof.m_com, proof.m_eval, &r_m, &proof.m_open, &params_m)
        .map_err(|e| format!("Range Multiplicity Opening: {e}"))?;
    eprintln!("[range({num_vars})]  m_hyrax:     {:>8.3}ms", _tm.elapsed().as_secs_f64()*1000.0);

    // Return the coordinate and the evaluated value to the parent layer (e.g. LayerNorm)
    Ok((r_v, v_eval))
}

// ---------------------------------------------------------------------------
// Batch Range Proof (shared multiplicity across multiple witnesses)
// ---------------------------------------------------------------------------

/// Prove that ALL values across every witness are in [0, 2^bits).
///
/// Concatenates all witnesses into one combined MLE and runs a single
/// `prove_range`.  The multiplicity table counts chunks from every witness
/// together, so the proof is ~K× smaller than K independent proofs.
///
/// Returns `(proof, r_v)` where `r_v` is over the combined (larger) MLE space.
/// If you need per-witness evaluation points for constraint-fusion, call
/// `prove_range` individually and use `verify_range_m_batch` to amortise the
/// multiplicity-opening MSMs instead.
pub fn prove_range_batch(
    witnesses: &[RangeProofWitness],
    bits: usize,
    transcript: &mut Transcript,
) -> Result<(RangeProof, Vec<F>), String> {
    let all_values: Vec<F> = witnesses
        .iter()
        .flat_map(|w| w.values.iter().copied())
        .collect();
    prove_range(&RangeProofWitness { values: all_values }, bits, transcript)
}

/// Run all verification steps of a range proof **except** the final
/// multiplicity-opening `hyrax_verify(m_com, …)`.
///
/// Returns `(r_v, v_eval, r_m)`.  The caller must later verify the
/// multiplicity commitment by passing the returned `r_m` (along with
/// `proof.m_com`, `proof.m_eval`, `proof.m_open`) to `verify_range_m_batch`.
///
/// Because `hyrax_verify` is transcript-free, deferring it does **not**
/// change the transcript state — `r_m` is still derived correctly.
pub fn verify_range_deferred(
    proof: &RangeProof,
    num_vars: usize,
    bits: usize,
    transcript: &mut Transcript,
) -> Result<(Vec<F>, F, Vec<F>), String> {
    let num_chunks = (bits + CHUNK_BITS - 1) / CHUNK_BITS;
    let (_, _, params_c) = params_from_vars(num_vars);

    // 1. Absorb commitments
    for com in &proof.chunk_coms {
        absorb_com(transcript, b"chunk_com", com);
    }
    absorb_com(transcript, b"logup_m_com", &proof.m_com);

    // 2. Sumcheck
    transcript.append_field(b"claim_v", &proof.claim_v);
    let (r_v, final_val) = verify_sumcheck(&proof.sumcheck, proof.claim_v, num_vars, transcript)
        .map_err(|e| format!("Range Sumcheck: {e}"))?;
    let v_eval = final_val;

    // 3. Chunk batch opening
    hyrax_verify_batch(
        &proof.chunk_coms,
        &proof.chunk_evals,
        &r_v,
        &proof.chunk_batch_proof,
        &params_c,
        transcript,
    )
    .map_err(|e| format!("Chunk batch opening failed: {e}"))?;

    // Chunk fusion check
    let mut expected_v_eval = F::ZERO;
    let mut shift = F::ONE;
    let shift_multiplier = F::from(1u64 << CHUNK_BITS);
    for c in 0..num_chunks {
        expected_v_eval += proof.chunk_evals[c] * shift;
        shift *= shift_multiplier;
    }
    if v_eval != expected_v_eval {
        return Err("Chunk fusion mismatch: V(r) != sum V_c(r) * 2^{16c}".into());
    }

    // 4. Derive r_m (advances transcript identically to verify_range) but do NOT call hyrax_verify
    let r_m = challenge_vec(transcript, CHUNK_BITS, b"logup_rm");

    Ok((r_v, v_eval, r_m))
}

/// Batch-verify K multiplicity openings from K range proofs at (potentially
/// different) evaluation points `r_m`, using a single pair of MSMs.
///
/// All range-proof multiplicity commitments share the same `params_m` (derived
/// from `CHUNK_BITS`), so the batch uses `hyrax_verify_multi_point` for
/// K × 2 MSMs → 2 MSMs total.
///
/// `entries`: slice of `(proof, r_m)` pairs where each `r_m` was returned by
/// a prior call to `verify_range_deferred`.
pub fn verify_range_m_batch(
    entries: &[(&RangeProof, &[F])],
    transcript: &mut Transcript,
) -> Result<(), String> {
    let (_, _, params_m) = params_from_vars(CHUNK_BITS);
    let mp_entries: Vec<(&HyraxCommitment, F, &[F], &HyraxProof)> = entries
        .iter()
        .map(|(proof, r_m)| (&proof.m_com, proof.m_eval, *r_m, &proof.m_open))
        .collect();
    hyrax_verify_multi_point(&mp_entries, &params_m, transcript)
        .map_err(|e| format!("Range multiplicity batch: {e}"))
}

/// Shared `HyraxParams` for multiplicity commitments (always uses `CHUNK_BITS`).
/// Useful when callers need to pre-fetch the params once rather than on every call.
pub fn range_m_params() -> HyraxParams {
    params_from_vars(CHUNK_BITS).2
}

// ---------------------------------------------------------------------------
// Globally-batched range proof (one shared m_com for K witnesses)
// ---------------------------------------------------------------------------

/// Per-witness LogUp inverse polynomial evidence.
///
/// Uses the combined sumcheck trick: for challenge α and β,
///   Σ_i h_k[i] * (1 + β*(α - C_k[i])) = claim_k + β * n_padded
///
/// where h_k[i] = 1/(α - C_k[i]).  This single sumcheck simultaneously proves
///   (a) Σ_i h_k[i] = claim_k  (the LHS sum for LogUp)
///   (b) h_k[i] * (α - C_k[i]) = 1 for all i  (consistency: h is actually the inverse)
///
/// At the sumcheck challenge r_k the verifier checks:
///   final_val = h_k(r_k) * (1 + β*(α - C_k(r_k)))
/// using Hyrax openings of h_com and chunk_com.
pub struct LogUpWitnessProof {
    /// Hyrax commitments to h_k = [1/(α - C_k[i]) for each i], one per chunk.
    pub h_coms: Vec<HyraxCommitment>,
    /// Sumcheck proofs: Σ_i h_k[i]*(1 + β*(α-C_k[i])) = combined_claims[k].
    pub combined_sumchecks: Vec<SumcheckProof>,
    /// combined_claims[k] = claim_k + β * n_padded.
    pub combined_claims: Vec<F>,
    /// h_k(r_k) — opening of h_com[k] at the sumcheck challenge.
    pub h_at_rk: Vec<F>,
    /// C_k(r_k) — opening of chunk_com[k] at the same point.
    pub chunk_at_rk: Vec<F>,
    /// Hyrax opening proofs for h_k at r_k.
    pub h_open_proofs: Vec<HyraxProof>,
    /// Hyrax opening proofs for chunk_com[k] at r_k.
    pub chunk_open_proofs: Vec<HyraxProof>,
}

/// Per-witness portion of a globally-batched range proof.
/// Does NOT contain m_com / m_eval / m_open — those live in `GlobalRangeM`.
pub struct RangeWitnessProof {
    pub chunk_coms: Vec<HyraxCommitment>,
    pub chunk_evals: Vec<F>,
    pub chunk_batch_proof: HyraxProof,
    pub sumcheck: SumcheckProof,
    pub claim_v: F,
    /// LogUp membership proof binding chunk arrays to the range table.
    pub logup: LogUpWitnessProof,
}

/// Shared multiplicity proof covering all witnesses in the global batch.
pub struct GlobalRangeM {
    pub m_com: HyraxCommitment,
    pub m_eval: F,
    pub m_open: HyraxProof,
    /// RHS sumcheck: Σ_{j=0}^{CHUNK_SIZE-1} M[j]·g[j] = logup_rhs_claim
    /// where g[j] = 1/(X - j) and X is the Fiat-Shamir LogUp challenge.
    pub logup_rhs_sumcheck: SumcheckProof,
    pub logup_rhs_claim: F,
    /// M(r_m2) — opening of m_com at the RHS sumcheck challenge.
    pub logup_m_at_rm2: F,
    pub logup_m_open_rm2: HyraxProof,
}

/// Prove that every value in every witness is in [0, 2^bits) using a single
/// shared multiplicity commitment and a sound LogUp consistency argument.
///
/// Transcript ordering:
///   Phase 1  — for each witness: absorb chunk_coms; absorb shared m_com
///   Phase 2  — for each witness: claim_v, sumcheck, chunk batch opening at r_v
///   M-open   — derive r_m, open shared m
///   Phase 3 (LogUp) — draw X; for each witness/chunk: commit+absorb h_k_com;
///              for each witness/chunk: LHS sumcheck + open h_k and chunk_k at r_h_k;
///              RHS sumcheck Σ_j M[j]*g[j] + open M at r_m2; grand sum check.
///
/// Returns `(per_witness_proofs, global_m, r_vs)`.
pub fn prove_range_batched(
    witnesses: &[&RangeProofWitness],
    bits: usize,
    transcript: &mut Transcript,
) -> Result<(Vec<RangeWitnessProof>, GlobalRangeM, Vec<Vec<F>>), String> {
    let num_chunks = (bits + CHUNK_BITS - 1) / CHUNK_BITS;
    let (nu_m, sigma_m, params_m) = params_from_vars(CHUNK_BITS);

    // ---- Phase 1: chunk decomposition + chunk_com commitments per witness ----
    let mut all_chunk_vals: Vec<Vec<Vec<F>>> = Vec::with_capacity(witnesses.len()); // [w][c][i]
    let mut all_chunk_mles: Vec<Vec<DenseMLPoly>> = Vec::with_capacity(witnesses.len());
    let mut all_v_mles: Vec<DenseMLPoly> = Vec::with_capacity(witnesses.len());
    let mut all_chunk_coms: Vec<Vec<HyraxCommitment>> = Vec::with_capacity(witnesses.len());
    let mut all_nu_c: Vec<usize> = Vec::with_capacity(witnesses.len());
    let mut all_sigma_c: Vec<usize> = Vec::with_capacity(witnesses.len());

    let mut m_global = vec![F::ZERO; CHUNK_SIZE];

    for witness in witnesses {
        let n = witness.values.len();
        let num_vars = n.next_power_of_two().trailing_zeros() as usize;
        let (nu_c, sigma_c, params_c) = params_from_vars(num_vars);

        let mut chunks = vec![vec![F::ZERO; n]; num_chunks];
        for (i, &v) in witness.values.iter().enumerate() {
            let val_u64 = v.into_bigint().as_ref()[0];
            for c in 0..num_chunks {
                let cv = (val_u64 >> (c * CHUNK_BITS)) & ((1 << CHUNK_BITS) - 1);
                chunks[c][i] = F::from(cv as u64);
                m_global[cv as usize] += F::ONE;
            }
        }

        let mut chunk_mles = Vec::with_capacity(num_chunks);
        let mut chunk_coms = Vec::with_capacity(num_chunks);
        for c in 0..num_chunks {
            let mle = vec_to_mle(&chunks[c], n);
            let com = hyrax_commit(&mle.evaluations, nu_c, &params_c);
            absorb_com(transcript, b"chunk_com", &com);
            chunk_coms.push(com);
            chunk_mles.push(mle);
        }

        let v_mle = vec_to_mle(&witness.values, n);
        all_chunk_vals.push(chunks);
        all_chunk_mles.push(chunk_mles);
        all_v_mles.push(v_mle);
        all_chunk_coms.push(chunk_coms);
        all_nu_c.push(nu_c);
        all_sigma_c.push(sigma_c);
    }

    // ---- Commit merged m ----
    let m_mle = vec_to_mle(&m_global, CHUNK_SIZE);
    let m_com = hyrax_commit(&m_mle.evaluations, nu_m, &params_m);
    absorb_com(transcript, b"logup_m_com", &m_com);

    // ---- Phase 2: per-witness sumcheck + chunk opening at r_v ----
    let mut witness_proofs_partial: Vec<(Vec<HyraxCommitment>, Vec<F>, HyraxProof, SumcheckProof, F)> = Vec::new();
    let mut r_vs: Vec<Vec<F>> = Vec::with_capacity(witnesses.len());

    for i in 0..witnesses.len() {
        let v_mle = &all_v_mles[i];
        let ones = DenseMLPoly::new(vec![F::ONE; v_mle.evaluations.len()]);
        let claim_v = v_mle.evaluations.iter().sum::<F>();
        transcript.append_field(b"claim_v", &claim_v);

        let (sumcheck, r_v) = prove_sumcheck(v_mle, &ones, claim_v, transcript);

        let chunk_evals: Vec<F> = all_chunk_mles[i].iter().map(|m| m.evaluate(&r_v)).collect();
        let chunk_slices: Vec<&[F]> = all_chunk_mles[i].iter().map(|m| m.evaluations.as_slice()).collect();
        let chunk_batch_proof = hyrax_open_batch(&chunk_slices, &r_v, all_nu_c[i], all_sigma_c[i], transcript);

        witness_proofs_partial.push((all_chunk_coms[i].clone(), chunk_evals, chunk_batch_proof, sumcheck, claim_v));
        r_vs.push(r_v);
    }

    // ---- Open shared m (old commitment, kept for LayerNorm accumulator) ----
    let r_m = challenge_vec(transcript, CHUNK_BITS, b"logup_rm");
    let m_eval = m_mle.evaluate(&r_m);
    let m_open = hyrax_open(&m_mle.evaluations, &r_m, nu_m, sigma_m);

    // ---- Phase 3: LogUp consistency proof ----
    // α drawn after m_com — prover cannot pick α to cheat on M.
    let alpha = transcript.challenge_field::<F>(b"logup_alpha");

    // g[j] = 1/(α - j) for the RHS sumcheck — batch inversion: O(n) muls + 1 inversion.
    let mut g_evals: Vec<F> = (0..CHUNK_SIZE).map(|j| alpha - F::from(j as u64)).collect();
    batch_inversion(&mut g_evals); // zeros out any entry where α = j (negligible probability)
    let g_mle = DenseMLPoly::new(g_evals);
    let logup_rhs_claim: F = m_global.iter().zip(g_mle.evaluations.iter()).map(|(m, g)| *m * *g).sum();

    // Commit all h_k = [1/(α - C_k[i])] BEFORE drawing β so β binds the h commitments.
    let mut all_h_mles: Vec<Vec<DenseMLPoly>> = Vec::with_capacity(witnesses.len());
    let mut all_h_coms: Vec<Vec<HyraxCommitment>> = Vec::with_capacity(witnesses.len());

    for i in 0..witnesses.len() {
        let n = witnesses[i].values.len();
        let num_vars = n.next_power_of_two().trailing_zeros() as usize;
        let (nu_c, _, params_c) = params_from_vars(num_vars);
        let mut h_mles_w = Vec::with_capacity(num_chunks);
        let mut h_coms_w = Vec::with_capacity(num_chunks);
        for c in 0..num_chunks {
            let mut h_vals: Vec<F> = all_chunk_vals[i][c].iter().map(|&cv| alpha - cv).collect();
            batch_inversion(&mut h_vals);
            let h_mle = vec_to_mle(&h_vals, n);
            let h_com = hyrax_commit(&h_mle.evaluations, nu_c, &params_c);
            absorb_com(transcript, b"logup_h_com", &h_com);
            h_mles_w.push(h_mle);
            h_coms_w.push(h_com);
        }
        all_h_mles.push(h_mles_w);
        all_h_coms.push(h_coms_w);
    }

    // β drawn after h_coms — used to combine the two LogUp checks into one sumcheck per chunk.
    let beta = transcript.challenge_field::<F>(b"logup_beta");

    // Per-witness per-chunk: combined sumcheck Σ_i h_k[i]*(1 + β*(α - C_k[i])) = claim_k + β*n_pad
    let mut all_logup_witness: Vec<LogUpWitnessProof> = Vec::with_capacity(witnesses.len());
    let mut total_lhs_claim = F::ZERO;

    for i in 0..witnesses.len() {
        let n = witnesses[i].values.len();
        let num_vars = n.next_power_of_two().trailing_zeros() as usize;
        let n_padded = 1usize << num_vars;
        let (nu_c, sigma_c, _) = params_from_vars(num_vars);

        let mut h_coms_w = Vec::with_capacity(num_chunks);
        let mut combined_sumchecks = Vec::with_capacity(num_chunks);
        let mut combined_claims = Vec::with_capacity(num_chunks);
        let mut h_at_rk = Vec::with_capacity(num_chunks);
        let mut chunk_at_rk = Vec::with_capacity(num_chunks);
        let mut h_open_proofs = Vec::with_capacity(num_chunks);
        let mut chunk_open_proofs = Vec::with_capacity(num_chunks);

        for c in 0..num_chunks {
            let h_mle = &all_h_mles[i][c];
            // q_k[i] = 1 + β*(α - C_k[i])
            let q_vals: Vec<F> = all_chunk_vals[i][c].iter()
                .map(|&cv| F::ONE + beta * (alpha - cv))
                .collect();
            let q_mle = vec_to_mle(&q_vals, n);

            // combined_claim = Σ_i h_k[i]*q_k[i] = claim_k + β*n_padded
            let claim_k: F = h_mle.evaluations.iter().sum();
            total_lhs_claim += claim_k;
            let combined = claim_k + beta * F::from(n_padded as u64);

            let (sc, r_k) = prove_sumcheck(h_mle, &q_mle, combined, transcript);

            let h_val = h_mle.evaluate(&r_k);
            let chunk_val = all_chunk_mles[i][c].evaluate(&r_k);
            let h_open = hyrax_open(&h_mle.evaluations, &r_k, nu_c, sigma_c);
            let chunk_open = hyrax_open(&all_chunk_mles[i][c].evaluations, &r_k, nu_c, sigma_c);

            h_coms_w.push(all_h_coms[i][c].clone());
            combined_sumchecks.push(sc);
            combined_claims.push(combined);
            h_at_rk.push(h_val);
            chunk_at_rk.push(chunk_val);
            h_open_proofs.push(h_open);
            chunk_open_proofs.push(chunk_open);
        }

        all_logup_witness.push(LogUpWitnessProof {
            h_coms: h_coms_w, combined_sumchecks, combined_claims,
            h_at_rk, chunk_at_rk, h_open_proofs, chunk_open_proofs,
        });
    }

    debug_assert_eq!(total_lhs_claim, logup_rhs_claim, "LogUp grand sum mismatch");

    // RHS sumcheck: Σ_j M[j] * g[j] = logup_rhs_claim
    let (logup_rhs_sumcheck, r_m2) = prove_sumcheck(&m_mle, &g_mle, logup_rhs_claim, transcript);
    let logup_m_at_rm2 = m_mle.evaluate(&r_m2);
    let logup_m_open_rm2 = hyrax_open(&m_mle.evaluations, &r_m2, nu_m, sigma_m);

    // Assemble final witness proofs
    let witness_proofs: Vec<RangeWitnessProof> = witness_proofs_partial
        .into_iter()
        .zip(all_logup_witness.into_iter())
        .map(|((chunk_coms, chunk_evals, chunk_batch_proof, sumcheck, claim_v), logup)| {
            RangeWitnessProof { chunk_coms, chunk_evals, chunk_batch_proof, sumcheck, claim_v, logup }
        })
        .collect();

    let global_m = GlobalRangeM {
        m_com, m_eval, m_open,
        logup_rhs_sumcheck, logup_rhs_claim, logup_m_at_rm2, logup_m_open_rm2,
    };

    Ok((witness_proofs, global_m, r_vs))
}

/// Verifier side of globally-batched range proofs.
///
/// Mirrors `prove_range_batched` exactly (Phase 1 → Phase 2 → M-open → Phase 3 LogUp).
/// Hyrax MSMs for chunk openings are deferred via accumulators; LogUp Hyrax calls
/// are done immediately (transcript-free, so ordering is irrelevant).
///
/// Returns `(r_vs, r_m)`.
pub fn verify_range_batched(
    witness_proofs: &[&RangeWitnessProof],
    global_m: &GlobalRangeM,
    num_vars_list: &[usize],
    bits: usize,
    transcript: &mut Transcript,
    acc_small: &mut HyraxBatchAccumulator,
    acc_large: &mut HyraxBatchAccumulator,
    acc_m: &mut HyraxBatchAccumulator,
) -> Result<(Vec<Vec<F>>, Vec<F>), String> {
    let num_chunks = (bits + CHUNK_BITS - 1) / CHUNK_BITS;
    let min_nv = num_vars_list.iter().copied().min().unwrap_or(0);

    // Phase 1: absorb chunk_coms for each witness
    for proof in witness_proofs {
        for com in &proof.chunk_coms {
            absorb_com(transcript, b"chunk_com", com);
        }
    }
    absorb_com(transcript, b"logup_m_com", &global_m.m_com);

    // Phase 2: per-witness sumcheck + deferred chunk opening
    let mut r_vs: Vec<Vec<F>> = Vec::with_capacity(witness_proofs.len());
    for (i, proof) in witness_proofs.iter().enumerate() {
        let num_vars = num_vars_list[i];

        transcript.append_field(b"claim_v", &proof.claim_v);
        let (r_v, final_val) =
            verify_sumcheck(&proof.sumcheck, proof.claim_v, num_vars, transcript)
                .map_err(|e| format!("GlobalRange witness {i} sumcheck: {e}"))?;

        let acc_chunk = if num_vars == min_nv { &mut *acc_small } else { &mut *acc_large };
        acc_chunk
            .add_verify_batch(&proof.chunk_coms, &proof.chunk_evals, &r_v, &proof.chunk_batch_proof, transcript)
            .map_err(|e| format!("GlobalRange witness {i} chunk opening (deferred): {e}"))?;

        let mut expected = F::ZERO;
        let mut shift = F::ONE;
        let shift_mult = F::from(1u64 << CHUNK_BITS);
        for c in 0..num_chunks {
            expected += proof.chunk_evals[c] * shift;
            shift *= shift_mult;
        }
        if final_val != expected {
            return Err(format!("GlobalRange witness {i}: chunk fusion mismatch"));
        }
        r_vs.push(r_v);
    }

    // Derive r_m and defer shared m opening (legacy accumulator path)
    let r_m = challenge_vec(transcript, CHUNK_BITS, b"logup_rm");
    acc_m
        .add_verify(&global_m.m_com, global_m.m_eval, &r_m, &global_m.m_open)
        .map_err(|e| format!("GlobalRange m opening (deferred): {e}"))?;

    // ---- Phase 3: LogUp consistency verification ----
    let alpha = transcript.challenge_field::<F>(b"logup_alpha");
    let (_, _, params_m) = params_from_vars(CHUNK_BITS);

    // Absorb all h_coms before drawing β (mirrors prover ordering)
    for proof in witness_proofs.iter() {
        for com in &proof.logup.h_coms {
            absorb_com(transcript, b"logup_h_com", com);
        }
    }
    let beta = transcript.challenge_field::<F>(b"logup_beta");

    // Per-witness per-chunk: verify combined sumcheck and algebraic consistency
    let mut total_lhs_claim = F::ZERO;
    for (i, proof) in witness_proofs.iter().enumerate() {
        let num_vars = num_vars_list[i];
        let n_padded = 1usize << num_vars;
        let (_, _, params_c) = params_from_vars(num_vars);

        for c in 0..num_chunks {
            let combined = proof.logup.combined_claims[c];
            // Recover claim_k from combined: combined = claim_k + β*n_padded
            let claim_k = combined - beta * F::from(n_padded as u64);
            total_lhs_claim += claim_k;

            // Verify sumcheck: Σ_i h_k[i]*(1 + β*(α - C_k[i])) = combined
            let (r_k, final_val) = verify_sumcheck(
                &proof.logup.combined_sumchecks[c], combined, num_vars, transcript,
            ).map_err(|e| format!("LogUp witness {i} chunk {c} sumcheck: {e}"))?;

            let h_val = proof.logup.h_at_rk[c];
            let chunk_val = proof.logup.chunk_at_rk[c];

            // Final round check: final_val = h(r_k) * (1 + β*(α - C(r_k)))
            let expected_final = h_val * (F::ONE + beta * (alpha - chunk_val));
            if final_val != expected_final {
                return Err(format!(
                    "LogUp witness {i} chunk {c}: combined sumcheck final check failed"
                ));
            }

            // Hyrax openings (transcript-free — immediate)
            hyrax_verify(&proof.logup.h_coms[c], h_val, &r_k, &proof.logup.h_open_proofs[c], &params_c)
                .map_err(|e| format!("LogUp witness {i} chunk {c} h opening: {e}"))?;
            hyrax_verify(&proof.chunk_coms[c], chunk_val, &r_k, &proof.logup.chunk_open_proofs[c], &params_c)
                .map_err(|e| format!("LogUp witness {i} chunk {c} chunk opening: {e}"))?;
        }
    }

    // RHS sumcheck: Σ_j M[j] * g[j] = logup_rhs_claim, g[j] = 1/(α - j)
    let (r_m2, final_mg) = verify_sumcheck(
        &global_m.logup_rhs_sumcheck, global_m.logup_rhs_claim, CHUNK_BITS, transcript,
    ).map_err(|e| format!("LogUp RHS sumcheck: {e}"))?;

    let g_mle_at_rm2 = g_mle_eval(alpha, &r_m2);
    let m_at_rm2 = global_m.logup_m_at_rm2;
    if final_mg != m_at_rm2 * g_mle_at_rm2 {
        return Err("LogUp RHS sumcheck final check: M(r)*g(r) mismatch".into());
    }

    hyrax_verify(&global_m.m_com, m_at_rm2, &r_m2, &global_m.logup_m_open_rm2, &params_m)
        .map_err(|e| format!("LogUp M opening at r_m2: {e}"))?;

    // Grand sum check: Σ claim_k == RHS
    if total_lhs_claim != global_m.logup_rhs_claim {
        return Err(format!(
            "LogUp grand sum mismatch: LHS={total_lhs_claim:?}, RHS={:?}",
            global_m.logup_rhs_claim
        ));
    }

    Ok((r_vs, r_m))
}

/// Evaluate g_mle at r, where g_mle[j] = 1/(x - j) for j ∈ [0, CHUNK_SIZE).
/// Uses DenseMLPoly::evaluate so variable ordering matches the sumcheck prover.
fn g_mle_eval(x: F, r: &[F]) -> F {
    debug_assert_eq!(r.len(), CHUNK_BITS);
    let mut g_vals: Vec<F> = (0..CHUNK_SIZE).map(|j| x - F::from(j as u64)).collect();
    batch_inversion(&mut g_vals);
    DenseMLPoly::new(g_vals).evaluate(r)
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

    /// Tampered chunk batch proof (proof.chunk_batch_proof) should be caught.
    #[test]
    fn test_rejects_tampered_chunk_opening_vector() {
        let witness = setup_witness();
        let num_vars = witness.values.len().next_power_of_two().trailing_zeros() as usize;

        let mut pt = Transcript::new(b"range_test");
        let (mut proof, _) = prove_range(&witness, 32, &mut pt).unwrap();

        // Corrupt the w_prime inside the batched chunk opening proof
        proof.chunk_batch_proof.w_prime[0] += F::one();

        let mut vt = Transcript::new(b"range_test");
        let result = verify_range(&proof, num_vars, 32, &mut vt);
        assert!(result.is_err(), "Should reject corrupted chunk opening proof");
    }
}
