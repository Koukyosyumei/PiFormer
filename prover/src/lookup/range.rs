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
    hyrax_commit, hyrax_open, hyrax_open_batch_with_eta, params_from_vars, powers_of,
    HyraxBatchAccumulator, HyraxCommitment, HyraxProof,
};
use crate::poly::{compute_eq_evals, DenseMLPoly};
use crate::subprotocols::{
    prove_sumcheck, verify_sumcheck, SumcheckCubicProofMulti, SumcheckProof,
};
use crate::subprotocols::sumcheck::CubicRoundPoly;
use crate::transcript::Transcript;
use ark_ff::{batch_inversion, Field, PrimeField};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::Instant;

// 16-bit chunks: table size = 65536. 32-bit values need 2 chunks; 64-bit values
// need 4 chunks. The fixed m_com / RHS-sumcheck / g_mle overhead grows by 256×
// vs 8-bit chunks, but per-witness chunk commits + LogUp h commits + per-witness
// sumcheck rounds all halve. With many range witnesses (large LN y_witnesses
// at 2·T·D entries × tens of LNs), the per-witness savings dominate the fixed
// overhead by orders of magnitude.
pub const CHUNK_BITS: usize = 16;
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
/// The inverse polynomial h_c[i] = 1/(α - C_c[i]) is bound implicitly via a
/// zerocheck folded into the bucket sumcheck (see `prove_fused_logup_zerocheck`):
/// the prover never separately commits to h. The bucket sumcheck claim is
///   Σ_x [ w(x)·h(x)·(1 + β(α - C(x)))
///       + γ_fold · eq(r_zc, x) · (h(x)·(α - C(x)) - 1) ] = bucket_claim
/// where γ_fold and r_zc are fresh Fiat-Shamir challenges drawn after the
/// bucket sumcheck claim is fixed. Soundness: if h ≠ 1/(α - C) on the
/// hypercube, the zerocheck term is non-zero at random r_zc w.h.p., and the
/// combined sum then equals bucket_claim with prob ≤ 1/|F| over random γ_fold.
///
/// At the bucket's shared r_k, only the per-chunk chunk_c openings are needed;
/// the bucket's terminal h-claim is read directly from the sumcheck proof.
#[derive(Clone)]
pub struct LogUpWitnessProof {
    /// Per-chunk combined claim: combined_claims[c] = claim_c + β * n_padded.
    /// The verifier sums Σ_c γ_c · combined_claims[c] for the bucket-level
    /// batched claim, and recovers each claim_c for the grand-sum check.
    pub combined_claims: Vec<F>,
    /// C_c(r_k_bucket) — opening of chunk_com[c] at the bucket's shared sumcheck
    /// challenge.
    pub chunk_at_rk: Vec<F>,
    /// Hyrax opening proofs for chunk_com[c] at the bucket's r_k.
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
    /// Cubic LogUp Pass-A sumchecks, one per witness-size bucket. Each bucket
    /// fuses all inverse polynomials into H_bucket(pair, x) and proves
    ///   Σ_x [ w·h·(1+β(α-C)) + γ_fold · eq(r_zc, x) · (h·(α-C) - 1) ] = bucket_claim
    /// (see `prove_fused_logup_zerocheck` for the zerocheck fold).
    pub bucket_sumchecks: Vec<SumcheckCubicProofMulti>,
}

/// Fused LogUp + zerocheck sumcheck.
///
/// Proves
///   Σ_x [ w(x)·h(x)·(1 + β(α - C(x)))
///       + γ_fold · eq(r_zc, x) · (h(x)·(α - C(x)) - 1) ] = claim
/// in a single cubic sumcheck. The first term is the standard LogUp bucket
/// claim; the second is a zerocheck forcing h to be 1/(α - C) on the hypercube,
/// which replaces a separate Hyrax commitment to h.
///
/// The h-eval vector must be padded with α⁻¹ on positions where C is zero
/// (typically the pair-padding of a bucket): there the zerocheck term reads
/// (α⁻¹·α - 1) = 0, so padding contributes nothing.  The main term is
/// already zero on padding because w is zero there.
///
/// Returns (proof, challenges). `proof.final_evals_*` carry the
/// per-variable terminals as length-1 vectors: f = h(r), g = w(r),
/// h = C(r). The verifier reconstructs eq(r_zc, r) itself and checks the
/// combined identity.
fn prove_fused_logup_zerocheck(
    mut h_cur: Vec<F>,
    mut c_cur: Vec<F>,
    mut w_cur: Vec<F>,
    mut eq_cur: Vec<F>,
    alpha: F,
    beta: F,
    gamma_fold: F,
    transcript: &mut Transcript,
) -> (SumcheckCubicProofMulti, Vec<F>) {
    assert_eq!(h_cur.len(), c_cur.len());
    assert_eq!(h_cur.len(), w_cur.len());
    assert_eq!(h_cur.len(), eq_cur.len());
    let n = h_cur.len().trailing_zeros() as usize;
    // NOTE: bucket_claim is absorbed by the caller before γ_fold and r_zc are
    // drawn, so we do NOT re-absorb it here.

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    let two = F::from(2u64);

    // Per-round, eval the combined cubic integrand at 4 points P0..P3 = 0,1,2,3
    // (CubicRoundPoly stores those four evaluations).  Main and zerocheck
    // share h and C, so we extrapolate (h, c, w, eq) once and combine.
    #[inline(always)]
    fn point_contrib(h: F, c: F, w: F, eq: F, alpha: F, beta: F, gamma_fold: F) -> F {
        let alpha_minus_c = alpha - c;
        let main = w * h * (F::ONE + beta * alpha_minus_c);
        let zc = gamma_fold * eq * (h * alpha_minus_c - F::ONE);
        main + zc
    }

    for _ in 0..n {
        let half = h_cur.len() >> 1;
        const PAR_THRESHOLD: usize = 512;
        let e = if half >= PAR_THRESHOLD {
            (0..half)
                .into_par_iter()
                .map(|idx| {
                    let mut local = [F::ZERO; 4];
                    let h0 = h_cur[idx];
                    let h1 = h_cur[idx + half];
                    let c0 = c_cur[idx];
                    let c1 = c_cur[idx + half];
                    let w0 = w_cur[idx];
                    let w1 = w_cur[idx + half];
                    let eq0 = eq_cur[idx];
                    let eq1 = eq_cur[idx + half];
                    let h2 = two * h1 - h0;
                    let w2 = two * w1 - w0;
                    let c2 = two * c1 - c0;
                    let eq2 = two * eq1 - eq0;
                    let h3 = h2 + h1 - h0;
                    let w3 = w2 + w1 - w0;
                    let c3 = c2 + c1 - c0;
                    let eq3 = eq2 + eq1 - eq0;
                    local[0] = point_contrib(h0, c0, w0, eq0, alpha, beta, gamma_fold);
                    local[1] = point_contrib(h1, c1, w1, eq1, alpha, beta, gamma_fold);
                    local[2] = point_contrib(h2, c2, w2, eq2, alpha, beta, gamma_fold);
                    local[3] = point_contrib(h3, c3, w3, eq3, alpha, beta, gamma_fold);
                    local
                })
                .reduce(
                    || [F::ZERO; 4],
                    |mut a, b| {
                        for i in 0..4 {
                            a[i] += b[i];
                        }
                        a
                    },
                )
        } else {
            let mut e = [F::ZERO; 4];
            for idx in 0..half {
                let h0 = h_cur[idx];
                let h1 = h_cur[idx + half];
                let c0 = c_cur[idx];
                let c1 = c_cur[idx + half];
                let w0 = w_cur[idx];
                let w1 = w_cur[idx + half];
                let eq0 = eq_cur[idx];
                let eq1 = eq_cur[idx + half];
                let h2 = two * h1 - h0;
                let w2 = two * w1 - w0;
                let c2 = two * c1 - c0;
                let eq2 = two * eq1 - eq0;
                let h3 = h2 + h1 - h0;
                let w3 = w2 + w1 - w0;
                let c3 = c2 + c1 - c0;
                let eq3 = eq2 + eq1 - eq0;
                e[0] += point_contrib(h0, c0, w0, eq0, alpha, beta, gamma_fold);
                e[1] += point_contrib(h1, c1, w1, eq1, alpha, beta, gamma_fold);
                e[2] += point_contrib(h2, c2, w2, eq2, alpha, beta, gamma_fold);
                e[3] += point_contrib(h3, c3, w3, eq3, alpha, beta, gamma_fold);
            }
            e
        };

        let rp = CubicRoundPoly { evals: e };
        for e in &rp.evals {
            transcript.append_field(b"sc_round", e);
        }
        let r_i = transcript.challenge_field::<F>(b"sc_challenge");
        challenges.push(r_i);

        h_cur = (0..half)
            .into_par_iter()
            .map(|idx| {
                let lo = h_cur[idx];
                let hi = h_cur[idx + half];
                lo + r_i * (hi - lo)
            })
            .collect();
        c_cur = (0..half)
            .into_par_iter()
            .map(|idx| {
                let lo = c_cur[idx];
                let hi = c_cur[idx + half];
                lo + r_i * (hi - lo)
            })
            .collect();
        w_cur = (0..half)
            .into_par_iter()
            .map(|idx| {
                let lo = w_cur[idx];
                let hi = w_cur[idx + half];
                lo + r_i * (hi - lo)
            })
            .collect();
        eq_cur = (0..half)
            .into_par_iter()
            .map(|idx| {
                let lo = eq_cur[idx];
                let hi = eq_cur[idx + half];
                lo + r_i * (hi - lo)
            })
            .collect();

        round_polys.push(rp);
    }

    // Terminal evals: prover claims (h, w, C) at r_full. eq(r_zc, r_full) is
    // computed by the verifier from public r_zc and r_full and so is not sent.
    let final_evals_f = vec![h_cur[0]];
    let final_evals_g = vec![w_cur[0]];
    let final_evals_h = vec![c_cur[0]];

    (
        SumcheckCubicProofMulti {
            round_polys,
            final_evals_f,
            final_evals_g,
            final_evals_h,
        },
        challenges,
    )
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
    let n_w = witnesses.len();
    let mut t_section = Instant::now();
    eprintln!(
        "[range bits={bits}] start: {n_w} witnesses, num_chunks={num_chunks}, sizes: min={} max={}",
        witnesses.iter().map(|w| w.values.len()).min().unwrap_or(0),
        witnesses.iter().map(|w| w.values.len()).max().unwrap_or(0)
    );

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

    eprintln!(
        "[range bits={bits}]   p1_chunks_and_commits: {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );
    t_section = Instant::now();

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

    eprintln!(
        "[range bits={bits}]   p1_absorb_loop:        {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );
    t_section = Instant::now();

    // ---- Commit merged m ----
    let m_mle = vec_to_mle(&m_global, CHUNK_SIZE);
    let m_com = hyrax_commit(&m_mle.evaluations, nu_m, &params_m);
    absorb_com(transcript, b"logup_m_com", &m_com);

    eprintln!(
        "[range bits={bits}]   m_com_commit:          {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );
    t_section = Instant::now();

    // ---- Phase 2: per-witness chunk opening at r_v ----
    // Pre-draw all (r_v_i, eta_i) sequentially via FS so the heavy O(num_chunks·N)
    // chunk_evals + hyrax_open_batch_with_eta computation can run in parallel
    // across witnesses. The FS state advances exactly as before (same labels in
    // the same order), only the post-FS computation is parallelized.
    let mut r_vs: Vec<Vec<F>> = Vec::with_capacity(witnesses.len());
    let mut etas: Vec<F> = Vec::with_capacity(witnesses.len());
    for i in 0..witnesses.len() {
        let r_v = challenge_vec(transcript, all_num_vars[i], b"range_r_v");
        let eta = transcript.challenge_field::<F>(b"hyrax_batch_eta");
        r_vs.push(r_v);
        etas.push(eta);
    }

    let phase2_results: Vec<(Vec<F>, HyraxProof)> = (0..witnesses.len())
        .into_par_iter()
        .map(|i| {
            let r_v = &r_vs[i];
            let chunk_evals: Vec<F> =
                all_chunk_mles[i].iter().map(|m| m.evaluate(r_v)).collect();
            let chunk_slices: Vec<&[F]> = all_chunk_mles[i]
                .iter()
                .map(|m| m.evaluations.as_slice())
                .collect();
            let chunk_batch_proof = hyrax_open_batch_with_eta(
                &chunk_slices,
                r_v,
                all_nu_c[i],
                all_sigma_c[i],
                etas[i],
            );
            (chunk_evals, chunk_batch_proof)
        })
        .collect();

    let witness_proofs_partial: Vec<(Vec<HyraxCommitment>, Vec<F>, HyraxProof)> = phase2_results
        .into_iter()
        .enumerate()
        .map(|(i, (chunk_evals, chunk_batch_proof))| {
            (all_chunk_coms[i].clone(), chunk_evals, chunk_batch_proof)
        })
        .collect();

    eprintln!(
        "[range bits={bits}]   p2_chunk_opens:        {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );
    t_section = Instant::now();

    // ---- Open shared m (old commitment, kept for LayerNorm accumulator) ----
    let r_m = challenge_vec(transcript, CHUNK_BITS, b"logup_rm");
    let m_eval = m_mle.evaluate(&r_m);
    let m_open = hyrax_open(&m_mle.evaluations, &r_m, nu_m, sigma_m);

    eprintln!(
        "[range bits={bits}]   m_open:                {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );
    t_section = Instant::now();

    // ---- Phase 3: LogUp consistency proof ----
    // α drawn after m_com — prover cannot pick α to cheat on M.
    let alpha = transcript.challenge_field::<F>(b"logup_alpha");

    // g[j] = 1/(α - j) for the RHS sumcheck — batch inversion: O(n) muls + 1 inversion.
    let mut g_evals: Vec<F> = (0..CHUNK_SIZE).map(|j| alpha - F::from(j as u64)).collect();
    batch_inversion(&mut g_evals); // zeros out any entry where α = j (negligible probability)
    let g_table = g_evals.clone();
    let g_mle = DenseMLPoly::new(g_evals);
    let logup_rhs_claim: F = m_global
        .iter()
        .zip(g_mle.evaluations.iter())
        .map(|(m, g)| *m * *g)
        .sum();

    // Build all h_k = [1/(α - C_k[i])] BEFORE drawing β. The fused bucket
    // sumcheck later reduces its terminal H claim to these per-chunk openings.
    let mut all_h_mles: Vec<Vec<DenseMLPoly>> = Vec::with_capacity(witnesses.len());

    let h_precomputed: Vec<Vec<DenseMLPoly>> = (0..witnesses.len())
        .into_par_iter()
        .map(|i| {
            let n = witnesses[i].values.len();
            (0..num_chunks)
                .map(|c| {
                    let h_vals: Vec<F> = all_chunk_vals[i][c]
                        .iter()
                        .map(|cv| g_table[cv.into_bigint().as_ref()[0] as usize])
                        .collect();
                    vec_to_mle(&h_vals, n)
                })
                .collect()
        })
        .collect();

    for h_mles_w in h_precomputed {
        all_h_mles.push(h_mles_w);
    }

    let mut buckets_by_num_vars: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &nv) in all_num_vars.iter().enumerate() {
        buckets_by_num_vars.entry(nv).or_default().push(i);
    }

    // h_coms are no longer committed: the bucket sumcheck folds in a zerocheck
    // h(x)·(α - C(x)) - 1 ≡ 0 that forces h to be the inverse on the boolean
    // hypercube, so the prover cannot deviate from the unique inverse MLE.
    // Pair-padding rows are set to α⁻¹ instead of 0 so the zerocheck term
    // (α⁻¹·α - 1) = 0 on padding; the main term is still zero on padding
    // because w is zero there.
    let alpha_inv = alpha
        .inverse()
        .expect("α drawn from Fiat-Shamir is non-zero w.p. 1");

    let mut bucket_h_mles: Vec<DenseMLPoly> = Vec::with_capacity(buckets_by_num_vars.len());
    let mut bucket_c_mles: Vec<DenseMLPoly> = Vec::with_capacity(buckets_by_num_vars.len());
    for (&num_vars, indices) in buckets_by_num_vars.iter() {
        let pair_count = indices.len() * num_chunks;
        let pair_bits = pair_count.next_power_of_two().trailing_zeros() as usize;
        let pair_padded = 1usize << pair_bits;
        let n_elem = 1usize << num_vars;
        let mut h_evals = vec![alpha_inv; pair_padded * n_elem];
        let mut c_evals = vec![F::ZERO; pair_padded * n_elem];
        for (w, &i) in indices.iter().enumerate() {
            for c in 0..num_chunks {
                let pair_idx = w * num_chunks + c;
                let offset = pair_idx << num_vars;
                h_evals[offset..offset + n_elem].copy_from_slice(&all_h_mles[i][c].evaluations);
                c_evals[offset..offset + n_elem]
                    .copy_from_slice(&all_chunk_mles[i][c].evaluations);
            }
        }
        let h_mle = DenseMLPoly::new(h_evals);
        let c_mle = DenseMLPoly::new(c_evals);
        bucket_h_mles.push(h_mle);
        bucket_c_mles.push(c_mle);
    }

    eprintln!(
        "[range bits={bits}]   p3_bucket_build:       {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );
    t_section = Instant::now();

    // β drawn after h_coms — used to combine the two LogUp checks into one sumcheck per chunk.
    let beta = transcript.challenge_field::<F>(b"logup_beta");
    // Per-witness: one batched sumcheck folding all `num_chunks` per-chunk
    // combined sumchecks via verifier-supplied weights γ_0..γ_{nc-1}.
    //   Σ_c γ_c · Σ_i h_c[i] · q_c[i] = Σ_c γ_c · combined_c
    // where q_c[i] = 1 + β*(α - C_c[i]) and combined_c = claim_c + β * n_padded.
    struct LogupWitnessPrecompute {
        claims: Vec<F>,    // claim_c per chunk
        combined: Vec<F>,  // combined_c per chunk = claim_c + β*n_padded
    }

    let logup_precomputed: Vec<LogupWitnessPrecompute> = (0..witnesses.len())
        .into_par_iter()
        .map(|i| {
            let n = witnesses[i].values.len();
            let num_vars = n.next_power_of_two().trailing_zeros() as usize;
            let n_padded = F::from((1usize << num_vars) as u64);
            let mut claims = Vec::with_capacity(num_chunks);
            let mut combined = Vec::with_capacity(num_chunks);
            for c in 0..num_chunks {
                let claim_c: F = all_h_mles[i][c].evaluations.iter().sum();
                claims.push(claim_c);
                combined.push(claim_c + beta * n_padded);
            }
            LogupWitnessPrecompute { claims, combined }
        })
        .collect();

    eprintln!(
        "[range bits={bits}]   p3_logup_precompute:   {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );
    t_section = Instant::now();

    // ---- Pass A: ONE fused sumcheck per (num_vars) bucket. ----
    // All witnesses with the same num_vars share an FS challenge eta_b and
    // a single sumcheck that folds bucket_size · num_chunks triples via
    // weight[w·nc + c] = eta_b^w · γ_c. Buckets are processed in ascending
    // num_vars order (deterministic FS path; verifier mirrors).
    let mut total_lhs_claim = F::ZERO;
    let mut witness_r_k: Vec<Vec<F>> = vec![Vec::new(); witnesses.len()];
    let mut bucket_sumchecks: Vec<SumcheckCubicProofMulti> =
        Vec::with_capacity(buckets_by_num_vars.len());

    for (bucket_idx, (&num_vars, indices)) in buckets_by_num_vars.iter().enumerate() {
        let bucket_size = indices.len();
        let total_triples = bucket_size * num_chunks;
        let pair_bits = total_triples.next_power_of_two().trailing_zeros() as usize;

        // Bucket-shared FS challenges: one eta + num_chunks γ's per bucket.
        let eta = transcript.challenge_field::<F>(b"logup_bucket_eta");
        let eta_pows = powers_of(eta, bucket_size);
        let gammas: Vec<F> = (0..num_chunks)
            .map(|_| transcript.challenge_field::<F>(b"logup_gamma"))
            .collect();

        let mut weights = Vec::with_capacity(total_triples);
        let mut bucket_claim = F::ZERO;

        for (w, &i) in indices.iter().enumerate() {
            let pre = &logup_precomputed[i];
            for &claim_c in &pre.claims {
                total_lhs_claim += claim_c;
            }
            for c in 0..num_chunks {
                let weight = eta_pows[w] * gammas[c];
                weights.push(weight);
                bucket_claim += weight * pre.combined[c];
            }
        }

        let h_evals = bucket_h_mles[bucket_idx].evaluations.clone();
        let c_evals = bucket_c_mles[bucket_idx].evaluations.clone();
        let n_elem = 1usize << num_vars;
        let mut w_evals = vec![F::ZERO; h_evals.len()];
        for p in 0..total_triples {
            let offset = p * n_elem;
            for x in 0..n_elem {
                w_evals[offset + x] = weights[p];
            }
        }

        // FS order — absorb bucket_claim BEFORE drawing γ_fold and r_zc, so
        // the prover cannot adapt bucket_claim to those challenges.
        transcript.append_field(b"logup_bucket_claim", &bucket_claim);
        let n_total = pair_bits + num_vars;
        let gamma_fold = transcript.challenge_field::<F>(b"logup_zc_gamma");
        let r_zc: Vec<F> = (0..n_total)
            .map(|_| transcript.challenge_field::<F>(b"logup_zc_r"))
            .collect();
        // eq(r_zc, x) over the full bucket hypercube. compute_eq_evals treats
        // the i-th entry of its input as binding the i-th LSB of x; sumcheck
        // folds MSB-first, so reverse r_zc before building eq.
        let eq_input: Vec<F> = r_zc.iter().rev().copied().collect();
        let eq_evals = compute_eq_evals(&eq_input, 1usize << n_total);

        let (super_sumcheck, r_k_full) = prove_fused_logup_zerocheck(
            h_evals,
            c_evals,
            w_evals,
            eq_evals,
            alpha,
            beta,
            gamma_fold,
            transcript,
        );
        let r_k = r_k_full[pair_bits..].to_vec();

        // Every witness in this bucket shares r_k_bucket for Pass B.
        for &i in indices {
            witness_r_k[i] = r_k.clone();
        }
        bucket_sumchecks.push(super_sumcheck);
    }

    debug_assert_eq!(total_lhs_claim, logup_rhs_claim, "LogUp grand sum mismatch");

    eprintln!(
        "[range bits={bits}]   p3_pass_a_sumchecks:   {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );
    t_section = Instant::now();

    // ---- Pass B (parallel, transcript-free): chunk_at_rk evaluations and
    // Hyrax opens for chunk_c at the shared r_k, across ALL witnesses and ALL
    // chunks.  None of these touch the transcript, so they can run in
    // parallel even across witnesses (small σ-witness opens overlap with the
    // larger y-witness opens).  hyrax_open itself is internally par_iter'd over
    // its column dimension, so rayon nesting handles work-stealing. ----
    // Pass B: only chunk_c opens at the shared r_k. h is no longer
    // separately committed, so its per-chunk evaluations are absorbed only
    // implicitly through the bucket sumcheck's terminal h-claim.
    struct WitnessOpens {
        chunk_at_rk: Vec<F>,
        chunk_open_proofs: Vec<HyraxProof>,
    }

    let opens: Vec<WitnessOpens> = (0..witnesses.len())
        .into_par_iter()
        .map(|i| {
            let (nu_c, sigma_c, _) = params_from_vars(all_num_vars[i]);
            let r_k = &witness_r_k[i];
            let chunk_at_rk: Vec<F> = all_chunk_mles[i].iter().map(|m| m.evaluate(r_k)).collect();
            let chunk_open_proofs: Vec<HyraxProof> = all_chunk_mles[i]
                .iter()
                .map(|chunk_mle| hyrax_open(&chunk_mle.evaluations, r_k, nu_c, sigma_c))
                .collect();
            WitnessOpens {
                chunk_at_rk,
                chunk_open_proofs,
            }
        })
        .collect();

    eprintln!(
        "[range bits={bits}]   p3_pass_b_opens_par:   {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );
    t_section = Instant::now();

    // Assemble per-witness LogUp proofs (the bucket sumchecks live in GlobalRangeM).
    let all_logup_witness: Vec<LogUpWitnessProof> = opens
        .into_iter()
        .enumerate()
        .map(|(i, ops)| LogUpWitnessProof {
            combined_claims: logup_precomputed[i].combined.clone(),
            chunk_at_rk: ops.chunk_at_rk,
            chunk_open_proofs: ops.chunk_open_proofs,
        })
        .collect();

    eprintln!(
        "[range bits={bits}]   assemble_witnesses:    {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );
    t_section = Instant::now();

    // RHS sumcheck: Σ_j M[j] * g[j] = logup_rhs_claim
    let (logup_rhs_sumcheck, r_m2) = prove_sumcheck(&m_mle, &g_mle, logup_rhs_claim, transcript);
    let logup_m_at_rm2 = m_mle.evaluate(&r_m2);
    let logup_m_open_rm2 = hyrax_open(&m_mle.evaluations, &r_m2, nu_m, sigma_m);

    eprintln!(
        "[range bits={bits}]   rhs_sumcheck_m_open:   {:7.3}ms",
        t_section.elapsed().as_secs_f64() * 1000.0
    );

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
        bucket_sumchecks,
    };

    Ok((witness_proofs, global_m, r_vs))
}

/// Verifier side of globally-batched range proofs.
///
/// Mirrors `prove_range_batched` exactly (Phase 1 → Phase 2 → M-open → Phase 3 LogUp).
/// Hyrax MSMs for chunk openings are deferred via accumulators; LogUp Hyrax calls
/// are done immediately (transcript-free, so ordering is irrelevant).
///
/// `chunk_accs` is a slice of `(num_vars, &mut accumulator)` pairs.  The caller
/// must provide one accumulator per distinct `num_vars` value present in
/// `num_vars_list`; the function routes each witness's chunk-com and LogUp
/// h-com / chunk-com opens to the matching accumulator (since each accumulator
/// is finalized with `params_from_vars(num_vars)` and Hyrax params depend on
/// that size, mixing sizes within one accumulator would be unsound).
///
/// Returns `(r_vs, r_m)`.
pub fn verify_range_batched(
    witness_proofs: &[&RangeWitnessProof],
    global_m: &GlobalRangeM,
    num_vars_list: &[usize],
    bits: usize,
    transcript: &mut Transcript,
    chunk_accs: &mut [(usize, &mut HyraxBatchAccumulator)],
    acc_m: &mut HyraxBatchAccumulator,
) -> Result<(Vec<Vec<F>>, Vec<F>), String> {
    let num_chunks = (bits + CHUNK_BITS - 1) / CHUNK_BITS;

    // Pre-compute the per-witness accumulator index (linear search over the
    // small chunk_accs list).  Doing this in a separate pass keeps `chunk_accs`
    // free of any outstanding immutable borrow when we later index it mutably.
    let acc_indices: Vec<usize> = num_vars_list
        .iter()
        .map(|&nv| {
            chunk_accs
                .iter()
                .position(|(n, _)| *n == nv)
                .ok_or_else(|| {
                    format!(
                        "verify_range_batched: no accumulator registered for num_vars={nv} \
                         (available: {:?})",
                        chunk_accs.iter().map(|(n, _)| *n).collect::<Vec<_>>()
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

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

        let acc_chunk = &mut *chunk_accs[acc_indices[i]].1;
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
    let beta = transcript.challenge_field::<F>(b"logup_beta");

    // Bucket-level verification: mirror the prover's BTreeMap grouping by num_vars.
    // For each bucket: draw eta + gammas, absorb bucket_claim, draw γ_fold + r_zc,
    // verify the fused-zerocheck sumcheck round-by-round, then check the
    // combined terminal identity:
    //   w(r)·h(r)·(1+β(α-C(r))) + γ_fold·eq(r_zc,r)·(h(r)·(α-C(r)) - 1) == current
    // where w(r), C(r), eq(r_zc, r) are reconstructed by the verifier from
    // public data and h(r) is the prover-claimed terminal (bound by sumcheck).
    let mut buckets_by_num_vars: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &nv) in num_vars_list.iter().enumerate() {
        buckets_by_num_vars.entry(nv).or_default().push(i);
    }
    if buckets_by_num_vars.len() != global_m.bucket_sumchecks.len() {
        return Err(format!(
            "GlobalRange: bucket count mismatch (sumchecks={}, expected={})",
            global_m.bucket_sumchecks.len(),
            buckets_by_num_vars.len()
        ));
    }

    for (i, proof) in witness_proofs.iter().enumerate() {
        if proof.logup.combined_claims.len() != num_chunks
            || proof.logup.chunk_at_rk.len() != num_chunks
            || proof.logup.chunk_open_proofs.len() != num_chunks
        {
            return Err(format!(
                "LogUp witness {i}: chunk-cardinality mismatch in proof structure"
            ));
        }
    }

    let mut total_lhs_claim = F::ZERO;
    for ((num_vars, indices), super_sumcheck) in buckets_by_num_vars
        .iter()
        .zip(global_m.bucket_sumchecks.iter())
    {
        let bucket_size = indices.len();
        let total_triples = bucket_size * num_chunks;
        let pair_bits = total_triples.next_power_of_two().trailing_zeros() as usize;
        let n_padded = 1usize << num_vars;
        let beta_n_padded = beta * F::from(n_padded as u64);
        let n_total = pair_bits + *num_vars;

        // Bucket-shared FS challenges: eta then gammas (mirror prover order).
        let eta = transcript.challenge_field::<F>(b"logup_bucket_eta");
        let eta_pows = powers_of(eta, bucket_size);
        let gammas: Vec<F> = (0..num_chunks)
            .map(|_| transcript.challenge_field::<F>(b"logup_gamma"))
            .collect();

        // Build flat weights and bucket claim from per-witness combined_claims.
        let mut weights = Vec::with_capacity(total_triples);
        let mut bucket_claim = F::ZERO;
        for (w, &i) in indices.iter().enumerate() {
            let logup = &witness_proofs[i].logup;
            for c in 0..num_chunks {
                let weight = eta_pows[w] * gammas[c];
                weights.push(weight);
                bucket_claim += weight * logup.combined_claims[c];
                total_lhs_claim += logup.combined_claims[c] - beta_n_padded;
            }
        }

        // FS order — absorb bucket_claim BEFORE drawing γ_fold and r_zc.
        transcript.append_field(b"logup_bucket_claim", &bucket_claim);
        let gamma_fold = transcript.challenge_field::<F>(b"logup_zc_gamma");
        let r_zc: Vec<F> = (0..n_total)
            .map(|_| transcript.challenge_field::<F>(b"logup_zc_r"))
            .collect();

        // Inline cubic-sumcheck round verification (we don't use the canned
        // verify_sumcheck_cubic_multi_batched because its terminal check is
        // f·g·h, but our identity is f·g·(1+β(α-h)) + γ_fold·eq·(f·(α-h)-1)).
        if super_sumcheck.round_polys.len() != n_total {
            return Err(format!(
                "LogUp bucket nv={num_vars}: wrong number of round polys (got {}, expected {})",
                super_sumcheck.round_polys.len(),
                n_total
            ));
        }
        let mut current = bucket_claim;
        let mut r_full = Vec::with_capacity(n_total);
        for (idx, rp) in super_sumcheck.round_polys.iter().enumerate() {
            if rp.evals[0] + rp.evals[1] != current {
                return Err(format!(
                    "LogUp bucket nv={num_vars}: round {idx} consistency failed"
                ));
            }
            for val in &rp.evals {
                transcript.append_field(b"sc_round", val);
            }
            let r_i = transcript.challenge_field::<F>(b"sc_challenge");
            r_full.push(r_i);
            current = rp.evaluate(r_i);
        }
        let r_pair = &r_full[..pair_bits];
        let r_k = r_full[pair_bits..].to_vec();

        let final_evals_f = &super_sumcheck.final_evals_f;
        let final_evals_g = &super_sumcheck.final_evals_g;
        let final_evals_h = &super_sumcheck.final_evals_h;
        if final_evals_f.len() != 1 || final_evals_g.len() != 1 || final_evals_h.len() != 1 {
            return Err(format!(
                "LogUp bucket nv={num_vars}: fused sumcheck final-evals length mismatch"
            ));
        }
        let h_at_rfull = final_evals_f[0];
        let w_claim = final_evals_g[0];
        let c_at_rfull = final_evals_h[0];

        // Reconstruct W(r_full) and C(r_full) from per-pair weights and chunk
        // openings; verifier-computed eq(r_zc, r_full) by direct product.
        let mut expected_w = F::ZERO;
        let mut expected_c = F::ZERO;
        for (w, &i) in indices.iter().enumerate() {
            let logup = &witness_proofs[i].logup;
            let acc_logup: &mut HyraxBatchAccumulator = &mut *chunk_accs[acc_indices[i]].1;
            for c in 0..num_chunks {
                let idx = w * num_chunks + c;
                let chunk_val = logup.chunk_at_rk[c];
                let pair_weight = eq_basis_eval(idx, pair_bits, r_pair);
                expected_w += pair_weight * weights[idx];
                expected_c += pair_weight * chunk_val;

                acc_logup
                    .add_verify(
                        &witness_proofs[i].chunk_coms[c],
                        chunk_val,
                        &r_k,
                        &logup.chunk_open_proofs[c],
                    )
                    .map_err(|e| format!("LogUp witness {i} chunk {c} chunk opening: {e}"))?;
            }
        }
        if w_claim != expected_w {
            return Err(format!(
                "LogUp bucket nv={num_vars}: fused weight W(r) algebra mismatch"
            ));
        }
        if c_at_rfull != expected_c {
            return Err(format!(
                "LogUp bucket nv={num_vars}: fused C(r) algebra mismatch"
            ));
        }

        // Pair-padding rows in the bucket polynomial don't have corresponding
        // chunk_at_rk entries: those pairs contribute zero to expected_c (they
        // were skipped in the loop), which is the correct value because the
        // prover padded c_evals with 0. Similarly weights for those padded
        // pairs were zero, so expected_w already accounts for them.
        let eq_at_rfull: F = r_zc
            .iter()
            .zip(r_full.iter())
            .map(|(rzc_i, ri)| (*rzc_i) * (*ri) + (F::ONE - *rzc_i) * (F::ONE - *ri))
            .product();
        let alpha_minus_c = alpha - c_at_rfull;
        let main_term = w_claim * h_at_rfull * (F::ONE + beta * alpha_minus_c);
        let zc_term = gamma_fold * eq_at_rfull * (h_at_rfull * alpha_minus_c - F::ONE);
        if main_term + zc_term != current {
            return Err(format!(
                "LogUp bucket nv={num_vars}: fused zerocheck terminal mismatch"
            ));
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
fn eq_basis_eval(index: usize, bits: usize, r: &[F]) -> F {
    debug_assert_eq!(bits, r.len());
    let mut acc = F::ONE;
    for (j, &rj) in r.iter().enumerate() {
        let bit = (index >> (bits - 1 - j)) & 1;
        acc *= if bit == 1 { rj } else { F::ONE - rj };
    }
    acc
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
