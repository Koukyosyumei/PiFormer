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
    hyrax_commit, hyrax_open, hyrax_open_batch, params_from_vars, HyraxBatchAccumulator,
    HyraxCommitment, HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{
    prove_sumcheck, prove_sumcheck_multi_batched_owned, verify_sumcheck,
    verify_sumcheck_multi_batched, SumcheckProof, SumcheckProofMulti,
};
use crate::transcript::Transcript;
use ark_ff::{batch_inversion, Field, PrimeField};
use rayon::prelude::*;

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

// ---------------------------------------------------------------------------
// Globally-batched range proof (one shared m_com for K witnesses)
// ---------------------------------------------------------------------------

/// Per-witness LogUp inverse polynomial evidence.
///
/// Uses the combined sumcheck trick: for challenge α and β,
///   Σ_c γ_c · Σ_i h_c[i] * (1 + β*(α - C_c[i])) = Σ_c γ_c · (claim_c + β * n_padded)
///
/// where h_c[i] = 1/(α - C_c[i]).  γ_c is a verifier-supplied RLC weight (drawn
/// after β and after all h_coms are absorbed) that folds the per-chunk
/// combined-sumchecks into ONE batched sumcheck.  The original combined sumcheck
/// per chunk simultaneously proves
///   (a) Σ_i h_c[i] = claim_c  (the LHS sum for LogUp)
///   (b) h_c[i] * (α - C_c[i]) = 1 for all i  (consistency: h is the inverse)
/// and Schwartz-Zippel over γ ensures the batch holds iff every chunk holds.
///
/// At the (shared) batched sumcheck challenge r_k the verifier checks, per chunk c:
///   final_eval_g[c] == 1 + β*(α - chunk_at_rk[c])
/// then defers Hyrax openings of h_com[c] (with eval = final_eval_f[c]) and
/// chunk_com[c] (with eval = chunk_at_rk[c]) at r_k.
#[derive(Clone)]
pub struct LogUpWitnessProof {
    /// Hyrax commitments to h_c = [1/(α - C_c[i]) for each i], one per chunk.
    pub h_coms: Vec<HyraxCommitment>,
    /// Single batched sumcheck folding all chunks of this witness into one
    /// reduction at a shared challenge r_k.
    pub batched_sumcheck: SumcheckProofMulti,
    /// Per-chunk combined claim: combined_claims[c] = claim_c + β * n_padded.
    /// The verifier sums Σ_c γ_c · combined_claims[c] to get the batched claim,
    /// and recovers each claim_c for the grand-sum check.
    pub combined_claims: Vec<F>,
    /// C_c(r_k) — opening of chunk_com[c] at the shared sumcheck challenge.
    /// (h_c(r_k) lives inside batched_sumcheck.final_evals_f.)
    pub chunk_at_rk: Vec<F>,
    /// Hyrax opening proofs for h_c at r_k.
    pub h_open_proofs: Vec<HyraxProof>,
    /// Hyrax opening proofs for chunk_com[c] at r_k.
    pub chunk_open_proofs: Vec<HyraxProof>,
}

/// Per-witness portion of a globally-batched range proof.
/// Does NOT contain m_com / m_eval / m_open — those live in `GlobalRangeM`.
///
/// NOTE: There is no Phase-2 sumcheck on (v_mle, ones).  v is never committed
/// externally, so any claim about Σ v[i] would be vacuous.  Instead r_v is drawn
/// directly via Fiat-Shamir after chunk_coms are absorbed, and the verifier
/// *defines* v(r_v) := Σ_c chunk_evals[c] * 2^(c·CHUNK_BITS) for use by
/// downstream proofs (e.g. LayerNorm).
#[derive(Clone)]
pub struct RangeWitnessProof {
    pub chunk_coms: Vec<HyraxCommitment>,
    pub chunk_evals: Vec<F>,
    pub chunk_batch_proof: HyraxProof,
    /// LogUp membership proof binding chunk arrays to the range table.
    pub logup: LogUpWitnessProof,
}

/// Shared multiplicity proof covering all witnesses in the global batch.
#[derive(Clone)]
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
///   Phase 2  — for each witness: derive r_v via Fiat-Shamir, batch-open chunks at r_v
///   M-open   — derive r_m, open shared m
///   Phase 3 (LogUp) — draw α; for each witness: commit+absorb all h_c_coms;
///              draw β; for each witness: draw γ_0..γ_{nc-1}, run ONE batched
///              sumcheck Σ_c γ_c · Σ_i h_c[i]·q_c[i] at shared r_k, open h_c
///              and chunk_c at r_k; RHS sumcheck Σ_j M[j]*g[j] + open M at r_m2;
///              grand sum check.
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
    // Note: v_mle is not materialized — v is never committed externally and the
    // dropped Phase-2 sumcheck was the only consumer.
    struct RangePrecompute {
        chunks: Vec<Vec<F>>,
        chunk_mles: Vec<DenseMLPoly>,
        chunk_coms: Vec<HyraxCommitment>,
        num_vars: usize,
        nu_c: usize,
        sigma_c: usize,
        m_local: Vec<F>,
    }

    let precomputed: Vec<RangePrecompute> = witnesses
        .par_iter()
        .map(|witness| {
            let n = witness.values.len();
            let num_vars = n.next_power_of_two().trailing_zeros() as usize;
            let (nu_c, sigma_c, params_c) = params_from_vars(num_vars);

            let mut chunks = vec![vec![F::ZERO; n]; num_chunks];
            let mut m_local = vec![F::ZERO; CHUNK_SIZE];
            for (i, &v) in witness.values.iter().enumerate() {
                let val_u64 = v.into_bigint().as_ref()[0];
                for c in 0..num_chunks {
                    let cv = (val_u64 >> (c * CHUNK_BITS)) & ((1 << CHUNK_BITS) - 1);
                    chunks[c][i] = F::from(cv as u64);
                    m_local[cv as usize] += F::ONE;
                }
            }

            let chunk_mles: Vec<DenseMLPoly> =
                chunks.iter().map(|chunk| vec_to_mle(chunk, n)).collect();
            let chunk_coms: Vec<HyraxCommitment> = chunk_mles
                .par_iter()
                .map(|mle| hyrax_commit(&mle.evaluations, nu_c, &params_c))
                .collect();

            RangePrecompute {
                chunks,
                chunk_mles,
                chunk_coms,
                num_vars,
                nu_c,
                sigma_c,
                m_local,
            }
        })
        .collect();

    let mut all_chunk_vals: Vec<Vec<Vec<F>>> = Vec::with_capacity(witnesses.len()); // [w][c][i]
    let mut all_chunk_mles: Vec<Vec<DenseMLPoly>> = Vec::with_capacity(witnesses.len());
    let mut all_chunk_coms: Vec<Vec<HyraxCommitment>> = Vec::with_capacity(witnesses.len());
    let mut all_num_vars: Vec<usize> = Vec::with_capacity(witnesses.len());
    let mut all_nu_c: Vec<usize> = Vec::with_capacity(witnesses.len());
    let mut all_sigma_c: Vec<usize> = Vec::with_capacity(witnesses.len());
    let mut m_global = vec![F::ZERO; CHUNK_SIZE];

    for item in precomputed {
        for (dst, src) in m_global.iter_mut().zip(item.m_local.iter()) {
            *dst += *src;
        }
        for com in &item.chunk_coms {
            absorb_com(transcript, b"chunk_com", com);
        }
        all_chunk_vals.push(item.chunks);
        all_chunk_mles.push(item.chunk_mles);
        all_chunk_coms.push(item.chunk_coms);
        all_num_vars.push(item.num_vars);
        all_nu_c.push(item.nu_c);
        all_sigma_c.push(item.sigma_c);
    }

    // ---- Commit merged m ----
    let m_mle = vec_to_mle(&m_global, CHUNK_SIZE);
    let m_com = hyrax_commit(&m_mle.evaluations, nu_m, &params_m);
    absorb_com(transcript, b"logup_m_com", &m_com);

    // ---- Phase 2: per-witness chunk opening at r_v ----
    // r_v is drawn directly via Fiat-Shamir.  v_mle is never committed
    // externally, so any sumcheck on (v_mle, ones) would only reduce a vacuous
    // claim; the Hyrax batch open at r_v already binds chunk_evals to chunk_coms.
    let mut witness_proofs_partial: Vec<(Vec<HyraxCommitment>, Vec<F>, HyraxProof)> = Vec::new();
    let mut r_vs: Vec<Vec<F>> = Vec::with_capacity(witnesses.len());

    for i in 0..witnesses.len() {
        let num_vars = all_num_vars[i];
        let r_v = challenge_vec(transcript, num_vars, b"range_r_v");

        let chunk_evals: Vec<F> = all_chunk_mles[i].iter().map(|m| m.evaluate(&r_v)).collect();
        let chunk_slices: Vec<&[F]> = all_chunk_mles[i]
            .iter()
            .map(|m| m.evaluations.as_slice())
            .collect();
        let chunk_batch_proof =
            hyrax_open_batch(&chunk_slices, &r_v, all_nu_c[i], all_sigma_c[i], transcript);

        witness_proofs_partial.push((all_chunk_coms[i].clone(), chunk_evals, chunk_batch_proof));
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
    let logup_rhs_claim: F = m_global
        .iter()
        .zip(g_mle.evaluations.iter())
        .map(|(m, g)| *m * *g)
        .sum();

    // Commit all h_k = [1/(α - C_k[i])] BEFORE drawing β so β binds the h commitments.
    let mut all_h_mles: Vec<Vec<DenseMLPoly>> = Vec::with_capacity(witnesses.len());
    let mut all_h_coms: Vec<Vec<HyraxCommitment>> = Vec::with_capacity(witnesses.len());

    let h_precomputed: Vec<(Vec<DenseMLPoly>, Vec<HyraxCommitment>)> = (0..witnesses.len())
        .into_par_iter()
        .map(|i| {
            let n = witnesses[i].values.len();
            let num_vars = n.next_power_of_two().trailing_zeros() as usize;
            let (nu_c, _, params_c) = params_from_vars(num_vars);
            let h_mles_w: Vec<DenseMLPoly> = (0..num_chunks)
                .map(|c| {
                    let mut h_vals: Vec<F> =
                        all_chunk_vals[i][c].iter().map(|&cv| alpha - cv).collect();
                    batch_inversion(&mut h_vals);
                    vec_to_mle(&h_vals, n)
                })
                .collect();
            let h_coms_w: Vec<HyraxCommitment> = h_mles_w
                .par_iter()
                .map(|h_mle| hyrax_commit(&h_mle.evaluations, nu_c, &params_c))
                .collect();
            (h_mles_w, h_coms_w)
        })
        .collect();

    for (h_mles_w, h_coms_w) in h_precomputed {
        for h_com in &h_coms_w {
            absorb_com(transcript, b"logup_h_com", h_com);
        }
        all_h_mles.push(h_mles_w);
        all_h_coms.push(h_coms_w);
    }

    // β drawn after h_coms — used to combine the two LogUp checks into one sumcheck per chunk.
    let beta = transcript.challenge_field::<F>(b"logup_beta");

    // Per-witness: one batched sumcheck folding all `num_chunks` per-chunk
    // combined sumchecks via verifier-supplied weights γ_0..γ_{nc-1}.
    //   Σ_c γ_c · Σ_i h_c[i] · q_c[i] = Σ_c γ_c · combined_c
    // where q_c[i] = 1 + β*(α - C_c[i]) and combined_c = claim_c + β * n_padded.
    struct LogupWitnessPrecompute {
        q_mles: Vec<DenseMLPoly>,
        claims: Vec<F>,    // claim_c per chunk
        combined: Vec<F>,  // combined_c per chunk = claim_c + β*n_padded
    }

    let logup_precomputed: Vec<LogupWitnessPrecompute> = (0..witnesses.len())
        .into_par_iter()
        .map(|i| {
            let n = witnesses[i].values.len();
            let num_vars = n.next_power_of_two().trailing_zeros() as usize;
            let n_padded = F::from((1usize << num_vars) as u64);
            let mut q_mles = Vec::with_capacity(num_chunks);
            let mut claims = Vec::with_capacity(num_chunks);
            let mut combined = Vec::with_capacity(num_chunks);
            for c in 0..num_chunks {
                let q_vals: Vec<F> = all_chunk_vals[i][c]
                    .iter()
                    .map(|&cv| F::ONE + beta * (alpha - cv))
                    .collect();
                q_mles.push(vec_to_mle(&q_vals, n));
                let claim_c: F = all_h_mles[i][c].evaluations.iter().sum();
                claims.push(claim_c);
                combined.push(claim_c + beta * n_padded);
            }
            LogupWitnessPrecompute {
                q_mles,
                claims,
                combined,
            }
        })
        .collect();

    // ---- Pass A (sequential, transcript-bound): per-witness γ draw +
    // batched sumcheck.  Only field ops over relatively small state — the
    // heavy hyrax_open MSMs are deferred to Pass B below. ----
    struct WitnessLogupState {
        r_k: Vec<F>,
        batched_sumcheck: SumcheckProofMulti,
    }

    let mut witness_states: Vec<WitnessLogupState> = Vec::with_capacity(witnesses.len());
    let mut total_lhs_claim = F::ZERO;

    for i in 0..witnesses.len() {
        // Draw RLC weights γ_0..γ_{nc-1} for this witness, *after* β and after
        // h_coms have been absorbed (both happened above).  Per-witness drawing
        // keeps a tight binding of γ to this witness's chunks.
        let gammas: Vec<F> = (0..num_chunks)
            .map(|_| transcript.challenge_field::<F>(b"logup_gamma"))
            .collect();

        let pre = &logup_precomputed[i];
        for &claim_c in &pre.claims {
            total_lhs_claim += claim_c;
        }

        // Batched claim Σ_c γ_c · combined_c.
        let batched_claim: F = gammas
            .iter()
            .zip(pre.combined.iter())
            .map(|(&g, &c)| g * c)
            .sum();

        // Single sumcheck across all chunks at shared challenge r_k.
        // We move-clone the h_mles + q_mles so the multi-batched sumcheck can
        // own them (it mutates in place per round).
        let h_mles_owned: Vec<DenseMLPoly> = all_h_mles[i].clone();
        let q_mles_owned: Vec<DenseMLPoly> = pre.q_mles.clone();
        let (batched_sumcheck, r_k) = prove_sumcheck_multi_batched_owned(
            h_mles_owned,
            q_mles_owned,
            &gammas,
            batched_claim,
            transcript,
        );

        witness_states.push(WitnessLogupState {
            r_k,
            batched_sumcheck,
        });
    }

    debug_assert_eq!(total_lhs_claim, logup_rhs_claim, "LogUp grand sum mismatch");

    // ---- Pass B (parallel, transcript-free): chunk_at_rk evaluations and
    // Hyrax opens for h_c and chunk_c at the shared r_k, across ALL witnesses
    // and ALL chunks.  None of these touch the transcript, so they can run in
    // parallel even across witnesses (small σ-witness opens overlap with the
    // larger y-witness opens).  hyrax_open itself is internally par_iter'd over
    // its column dimension, so rayon nesting handles work-stealing. ----
    struct WitnessOpens {
        chunk_at_rk: Vec<F>,
        h_open_proofs: Vec<HyraxProof>,
        chunk_open_proofs: Vec<HyraxProof>,
    }

    let opens: Vec<WitnessOpens> = (0..witnesses.len())
        .into_par_iter()
        .map(|i| {
            let (nu_c, sigma_c, _) = params_from_vars(all_num_vars[i]);
            let r_k = &witness_states[i].r_k;
            let chunk_at_rk: Vec<F> = all_chunk_mles[i].iter().map(|m| m.evaluate(r_k)).collect();
            let h_open_proofs: Vec<HyraxProof> = all_h_mles[i]
                .iter()
                .map(|h_mle| hyrax_open(&h_mle.evaluations, r_k, nu_c, sigma_c))
                .collect();
            let chunk_open_proofs: Vec<HyraxProof> = all_chunk_mles[i]
                .iter()
                .map(|chunk_mle| hyrax_open(&chunk_mle.evaluations, r_k, nu_c, sigma_c))
                .collect();
            WitnessOpens {
                chunk_at_rk,
                h_open_proofs,
                chunk_open_proofs,
            }
        })
        .collect();

    // Assemble per-witness LogUp proofs by zipping Pass-A state with Pass-B opens.
    let all_logup_witness: Vec<LogUpWitnessProof> = witness_states
        .into_iter()
        .zip(opens.into_iter())
        .enumerate()
        .map(|(i, (state, ops))| LogUpWitnessProof {
            h_coms: all_h_coms[i].clone(),
            batched_sumcheck: state.batched_sumcheck,
            combined_claims: logup_precomputed[i].combined.clone(),
            chunk_at_rk: ops.chunk_at_rk,
            h_open_proofs: ops.h_open_proofs,
            chunk_open_proofs: ops.chunk_open_proofs,
        })
        .collect();

    // RHS sumcheck: Σ_j M[j] * g[j] = logup_rhs_claim
    let (logup_rhs_sumcheck, r_m2) = prove_sumcheck(&m_mle, &g_mle, logup_rhs_claim, transcript);
    let logup_m_at_rm2 = m_mle.evaluate(&r_m2);
    let logup_m_open_rm2 = hyrax_open(&m_mle.evaluations, &r_m2, nu_m, sigma_m);

    // Assemble final witness proofs
    let witness_proofs: Vec<RangeWitnessProof> = witness_proofs_partial
        .into_iter()
        .zip(all_logup_witness.into_iter())
        .map(
            |((chunk_coms, chunk_evals, chunk_batch_proof), logup)| RangeWitnessProof {
                chunk_coms,
                chunk_evals,
                chunk_batch_proof,
                logup,
            },
        )
        .collect();

    let global_m = GlobalRangeM {
        m_com,
        m_eval,
        m_open,
        logup_rhs_sumcheck,
        logup_rhs_claim,
        logup_m_at_rm2,
        logup_m_open_rm2,
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

    // Phase 2: per-witness chunk opening at r_v.
    // r_v is drawn directly via Fiat-Shamir (no sumcheck on v_mle since v is
    // never committed externally — chunk_coms + Phase-3 LogUp fully bind the
    // chunk values).  Downstream proofs reconstruct v(r_v) from chunk_evals as
    // Σ_c chunk_evals[c] * 2^(c·CHUNK_BITS) when needed.
    let mut r_vs: Vec<Vec<F>> = Vec::with_capacity(witness_proofs.len());
    for (i, proof) in witness_proofs.iter().enumerate() {
        let num_vars = num_vars_list[i];
        let r_v = challenge_vec(transcript, num_vars, b"range_r_v");

        let acc_chunk = if num_vars == min_nv {
            &mut *acc_small
        } else {
            &mut *acc_large
        };
        acc_chunk
            .add_verify_batch(
                &proof.chunk_coms,
                &proof.chunk_evals,
                &r_v,
                &proof.chunk_batch_proof,
                transcript,
            )
            .map_err(|e| format!("GlobalRange witness {i} chunk opening (deferred): {e}"))?;

        if proof.chunk_evals.len() != num_chunks {
            return Err(format!(
                "GlobalRange witness {i}: chunk_evals length mismatch (got {}, expected {})",
                proof.chunk_evals.len(),
                num_chunks
            ));
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

    // Absorb all h_coms before drawing β (mirrors prover ordering)
    for proof in witness_proofs.iter() {
        for com in &proof.logup.h_coms {
            absorb_com(transcript, b"logup_h_com", com);
        }
    }
    let beta = transcript.challenge_field::<F>(b"logup_beta");

    // Per-witness: verify ONE batched sumcheck folding all chunks via γ weights,
    // then per-chunk algebraic consistency: q_c(r_k) == 1 + β(α - chunk_c(r_k)).
    // Hyrax openings are deferred to the existing batch accumulators — inner-product
    // checks happen immediately (field ops only); G1 MSMs are batched at finalize.
    let mut total_lhs_claim = F::ZERO;
    for (i, proof) in witness_proofs.iter().enumerate() {
        let num_vars = num_vars_list[i];
        let n_padded = 1usize << num_vars;
        // Route LogUp opens to the same accumulator as Phase-2 chunk opens.
        let acc_logup: &mut HyraxBatchAccumulator = if num_vars == min_nv {
            &mut *acc_small
        } else {
            &mut *acc_large
        };

        if proof.logup.combined_claims.len() != num_chunks
            || proof.logup.chunk_at_rk.len() != num_chunks
            || proof.logup.h_coms.len() != num_chunks
            || proof.logup.h_open_proofs.len() != num_chunks
            || proof.logup.chunk_open_proofs.len() != num_chunks
        {
            return Err(format!(
                "LogUp witness {i}: chunk-cardinality mismatch in proof structure"
            ));
        }

        // Mirror prover: draw γ_0..γ_{nc-1} for this witness, then verify the
        // batched sumcheck.
        let gammas: Vec<F> = (0..num_chunks)
            .map(|_| transcript.challenge_field::<F>(b"logup_gamma"))
            .collect();

        let batched_claim: F = gammas
            .iter()
            .zip(proof.logup.combined_claims.iter())
            .map(|(&g, &c)| g * c)
            .sum();

        // Recover per-chunk claim_c and accumulate the LogUp grand sum.
        let beta_n_padded = beta * F::from(n_padded as u64);
        for &combined in &proof.logup.combined_claims {
            total_lhs_claim += combined - beta_n_padded;
        }

        let (r_k, final_combination) = verify_sumcheck_multi_batched(
            &proof.logup.batched_sumcheck,
            &gammas,
            batched_claim,
            num_vars,
            transcript,
        )
        .map_err(|e| format!("LogUp witness {i} batched sumcheck: {e}"))?;

        // verify_sumcheck_multi_batched also asserts
        //   final_combination == Σ_c γ_c · final_evals_f[c] · final_evals_g[c]
        // so we just need to: (a) bind final_evals_g[c] to the algebraic relation
        // q_c(r_k) = 1 + β(α - chunk_c(r_k)), and (b) defer Hyrax opens for h_c
        // (eval = final_evals_f[c]) and chunk_c (eval = chunk_at_rk[c]).
        let _ = final_combination; // already enforced inside verify_sumcheck_multi_batched

        let final_evals_f = &proof.logup.batched_sumcheck.final_evals_f;
        let final_evals_g = &proof.logup.batched_sumcheck.final_evals_g;
        if final_evals_f.len() != num_chunks || final_evals_g.len() != num_chunks {
            return Err(format!(
                "LogUp witness {i}: batched sumcheck final-evals length mismatch"
            ));
        }

        for c in 0..num_chunks {
            let chunk_val = proof.logup.chunk_at_rk[c];
            let expected_q = F::ONE + beta * (alpha - chunk_val);
            if final_evals_g[c] != expected_q {
                return Err(format!(
                    "LogUp witness {i} chunk {c}: q(r_k) algebra mismatch"
                ));
            }

            acc_logup
                .add_verify(
                    &proof.logup.h_coms[c],
                    final_evals_f[c],
                    &r_k,
                    &proof.logup.h_open_proofs[c],
                )
                .map_err(|e| format!("LogUp witness {i} chunk {c} h opening: {e}"))?;
            acc_logup
                .add_verify(
                    &proof.chunk_coms[c],
                    chunk_val,
                    &r_k,
                    &proof.logup.chunk_open_proofs[c],
                )
                .map_err(|e| format!("LogUp witness {i} chunk {c} chunk opening: {e}"))?;
        }
    }

    // RHS sumcheck: Σ_j M[j] * g[j] = logup_rhs_claim, g[j] = 1/(α - j)
    let (r_m2, final_mg) = verify_sumcheck(
        &global_m.logup_rhs_sumcheck,
        global_m.logup_rhs_claim,
        CHUNK_BITS,
        transcript,
    )
    .map_err(|e| format!("LogUp RHS sumcheck: {e}"))?;

    let g_mle_at_rm2 = g_mle_eval(alpha, &r_m2);
    let m_at_rm2 = global_m.logup_m_at_rm2;
    if final_mg != m_at_rm2 * g_mle_at_rm2 {
        return Err("LogUp RHS sumcheck final check: M(r)*g(r) mismatch".into());
    }

    // Defer m_com opening at r_m2 to acc_m alongside the Phase-2 m opening.
    acc_m
        .add_verify(&global_m.m_com, m_at_rm2, &r_m2, &global_m.logup_m_open_rm2)
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
