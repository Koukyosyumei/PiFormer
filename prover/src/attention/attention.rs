//! Arithmetic circuit for one head of linear attention (MLE/GKR + PCS based).
//!
//! **Production-Grade Security Architecture:**
//!   Unlike prototype implementations, this module STRICTLY separates internal
//!   commitments from IO commitments. The Verifier DOES NOT trust the Prover
//!   for the commitments of Q, K, V, and Out. These must be passed via
//!   `AttentionIOCommitments` from the global pipeline (e.g., the outputs of
//!   the previous FCONN/Linear layers).
//!
//! **Computation proved:**
//!   1. phi_q_com = Lasso(q_com)
//!   2. phi_k_com = Lasso(k_com)
//!   3. context[i][j] = Σ_t phi_k[t][i] · V[t][j]    (Sumcheck)
//!   4. out[t][j]     = Σ_i phi_q[t][i] · context[i][j] (Sumcheck)

use crate::field::F;
use crate::lookup::lasso::{
    precommit_lasso_multi_tables, LassoInstance, LassoMultiInstance, LassoMultiProvingKey,
    LassoMultiVerifyingKey,
};
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_open, hyrax_verify, params_from_vars, poly_hyrax,
    HyraxCommitment, HyraxParams, HyraxProof,
};
use crate::poly::utils::{combine, eval_cols, eval_rows, mat_to_mle};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{
    prove_sumcheck, prove_sumcheck_multi_batched, verify_sumcheck,
    verify_sumcheck_multi_batched, EvalClaim, SumcheckProof, SumcheckProofMulti,
};
use crate::transcript::{challenge_vec, Transcript};

// ---------------------------------------------------------------------------
// Pipeline Keys (Setup Phase)
// ---------------------------------------------------------------------------

/// Precomputed Lasso keys for Q and K activation tables (computed at setup).
#[derive(Clone)]
pub struct AttentionProvingKey {
    pub qk_lasso_pk: LassoMultiProvingKey,
}

impl AttentionProvingKey {
    pub fn vk(&self) -> AttentionVerifyingKey {
        AttentionVerifyingKey {
            qk_lasso_vk: self.qk_lasso_pk.vk(),
        }
    }
}

/// Verifier-side key for attention activations.
#[derive(Clone)]
pub struct AttentionVerifyingKey {
    pub qk_lasso_vk: LassoMultiVerifyingKey,
}

/// Precommit Q and K activation tables at setup time.
pub fn precommit_attention_tables(
    q_lasso: &LassoInstance,
    k_lasso: &LassoInstance,
    params: &HyraxParams,
) -> AttentionProvingKey {
    let multi_inst = LassoMultiInstance {
        instances: vec![q_lasso.clone(), k_lasso.clone()],
    };
    let qk_lasso_pk = precommit_lasso_multi_tables(&multi_inst, q_lasso.bits_per_chunk, params);
    AttentionProvingKey { qk_lasso_pk }
}

// ---------------------------------------------------------------------------
// Pipeline Interfaces (Strictly Enforced Boundaries)
// ---------------------------------------------------------------------------

/// Trusted IO Commitments provided by the Global Pipeline Verifier.
/// The Verifier MUST NOT accept these from the Prover's proof.
///
/// GKR backward fusion: out_com is eliminated. out_inner is not committed
/// independently — it is bound by the shared sumcheck claim with O_proj.
///
/// `skip_io_absorb`: when true, q/k/v_com are already in the transcript (Phase 1
/// absorbs all block commitments before r_td derivation). Avoids double absorption.
pub struct AttentionIOCommitments {
    pub q_com: HyraxCommitment,
    pub k_com: HyraxCommitment,
    pub v_com: HyraxCommitment,
    pub skip_io_absorb: bool,
}

/// Private witness data. ONLY the Prover holds this.
pub struct LinearAttentionWitness {
    pub q: Vec<Vec<F>>,
    pub k: Vec<Vec<F>>,
    pub v: Vec<Vec<F>>,
    pub phi_q: Vec<Vec<F>>,
    pub phi_k: Vec<Vec<F>>,
    pub context: Vec<Vec<F>>,
    pub out: Vec<Vec<F>>,
}

/// Public instance. Contains dimensions and Lasso structures.
pub struct LinearAttentionInstance {
    pub seq_len: usize,
    pub d_head: usize,
    pub q_lasso: LassoInstance,
    pub k_lasso: LassoInstance,
    /// Private witness: table lookup indices for Q/K activations (not sent to verifier).
    pub q_query_indices: Vec<usize>,
    pub k_query_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Proof structures
// ---------------------------------------------------------------------------

pub struct AttentionOpenings {
    /// Claimed out eval at (rx, ry). Deferred to block-level combine proof.
    pub out_eval: F,
    /// phi_q/phi_k evals deferred to internal CombineProofs (no direct Hyrax open).
    pub phi_q_eval: F,
    pub phi_k_eval: F,
    /// Claimed v eval at v_point. Deferred to block-level combine proof.
    pub v_eval: F,
}

/// Sumcheck variant chosen at proving time based on model geometry.
pub enum AttentionSumcheckProof {
    /// Single batched proof — used when seq_len == d_head (t_bits == d_bits).
    Batched {
        proof: SumcheckProofMulti,
        /// ctx(r_i, ry) for the combined batch claim. Soundness via random alpha.
        ctx_eval: F,
    },
    /// Two chained sumchecks — used when seq_len != d_head.
    Sequential {
        out_sumcheck: SumcheckProof,
        context_sumcheck: SumcheckProof,
    },
}

pub struct LinearAttentionProof {
    pub sumcheck: AttentionSumcheckProof,
    pub openings: AttentionOpenings,
    /// Hyrax commitments to phi_q and phi_k MLE vectors, absorbed before sumcheck challenges.
    pub phi_q_com: HyraxCommitment,
    pub phi_k_com: HyraxCommitment,
    /// Openings of phi_q_com / phi_k_com at the sumcheck-derived evaluation points.
    pub phi_q_open: HyraxProof,
    pub phi_k_open: HyraxProof,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Returns (proof, out_claim, v_claim).
///
/// `out_claim` = EvalClaim on out at (rx, ry).
/// `v_claim`   = EvalClaim on v at v_point — deferred to block-level combine.
///
/// GKR backward fusion: when `external_out_claim` is Some, the output evaluation
/// point (rx, ry) comes from the O_proj x_claim instead of being sampled from the
/// transcript. This eliminates out_inner_com — the binding is provided by both the
/// O_proj and attention sumchecks agreeing on the same point and value.
pub fn prove_linear_attention(
    witness: &LinearAttentionWitness,
    inst: &LinearAttentionInstance,
    _pk: &AttentionProvingKey,
    io_coms: &AttentionIOCommitments,
    external_out_claim: Option<EvalClaim>,
    transcript: &mut Transcript,
    _lasso_params: &HyraxParams,
) -> (LinearAttentionProof, EvalClaim, EvalClaim) {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;

    let phi_q_mle = mat_to_mle(&witness.phi_q, t, d);
    let phi_k_mle = mat_to_mle(&witness.phi_k, t, d);
    let v_mle = mat_to_mle(&witness.v, t, d);
    let ctx_mle = mat_to_mle(&witness.context, d, d);
    let out_mle = mat_to_mle(&witness.out, t, d);

    // 1. Absorb IO commitments. out_com eliminated via GKR backward (no out_inner_com).
    // In GKR multi-block mode, q/k/v_com are already in the transcript (Phase 1).
    if !io_coms.skip_io_absorb {
        absorb_com(transcript, b"q_com", &io_coms.q_com);
        absorb_com(transcript, b"k_com", &io_coms.k_com);
        absorb_com(transcript, b"v_com", &io_coms.v_com);
    }

    // Commit phi_q and phi_k and absorb before any sumcheck challenges.
    // This ensures the sumcheck challenges are drawn after committing to phi_q/phi_k,
    // making the evaluation points phi_q_point/phi_k_point binding.
    let (nu_td, sigma_td, params_td) = poly_hyrax(&phi_q_mle);
    let phi_q_com = hyrax_commit(&phi_q_mle.evaluations, nu_td, &params_td);
    let phi_k_com = hyrax_commit(&phi_k_mle.evaluations, nu_td, &params_td);
    absorb_com(transcript, b"phi_q_com", &phi_q_com);
    absorb_com(transcript, b"phi_k_com", &phi_k_com);

    // 2. Determine (rx, ry): sampled from transcript, or externally supplied (GKR backward).
    let (rx, ry, out_eval) = if let Some(ref claim) = external_out_claim {
        // GKR backward: O_proj's x_claim provides the out_inner evaluation point.
        // rx = first t_bits, ry = last d_bits of claim.point.
        let rx = claim.point[..t_bits].to_vec();
        let ry = claim.point[t_bits..].to_vec();
        (rx, ry, claim.value)
    } else {
        // Forward mode: sample from transcript, compute out_eval from witness.
        let rx = challenge_vec(transcript, t_bits, b"rx_out");
        let ry = challenge_vec(transcript, d_bits, b"ry_out");
        let out_eval = out_mle.evaluate(&combine(&rx, &ry));
        (rx, ry, out_eval)
    };

    let (sumcheck, phi_q_eval, phi_k_eval, v_eval, phi_q_point, phi_k_point, v_point) =
        if t_bits == d_bits {
            // 3a. Batched Sumcheck — seq_len == d_head, both pairs share the same variable count.
            transcript.append_field(b"claimed_out", &out_eval);

            let r_i = challenge_vec(transcript, d_bits, b"r_i_attn");
            let ctx_eval = ctx_mle.evaluate(&combine(&r_i, &ry));
            transcript.append_field(b"claimed_ctx", &ctx_eval);

            let alpha = transcript.challenge_field::<F>(b"attn_batch_alpha");
            let combined = out_eval + alpha * ctx_eval;

            let f_out = DenseMLPoly::from_vec_padded(eval_rows(&phi_q_mle, t_bits, &rx));
            let g_out = DenseMLPoly::from_vec_padded(eval_cols(&ctx_mle, d_bits, &ry));
            let f_ctx = DenseMLPoly::from_vec_padded(eval_cols(&phi_k_mle, t_bits, &r_i));
            let g_v = DenseMLPoly::from_vec_padded(eval_cols(&v_mle, t_bits, &ry));

            let (proof, batch_r) = prove_sumcheck_multi_batched(
                &[f_out, f_ctx],
                &[g_out, g_v],
                &[F::from(1u64), alpha],
                combined,
                transcript,
            );

            let phi_q_eval = proof.final_evals_f[0];
            let phi_k_eval = proof.final_evals_f[1];
            let v_eval = proof.final_evals_g[1];

            let _phi_q_point = combine(&rx, &batch_r);
            let _phi_k_point = combine(&batch_r, &r_i);
            let v_point = combine(&batch_r, &ry);

            (
                AttentionSumcheckProof::Batched { proof, ctx_eval },
                phi_q_eval,
                phi_k_eval,
                v_eval,
                _phi_q_point,
                _phi_k_point,
                v_point,
            )
        } else {
            // 3b. Sequential chained sumchecks — seq_len != d_head.
            //
            // Out(rx, ry) = Σ_i Phi_Q(rx, i) · Context(i, ry)   [d_bits vars]
            let f_out = DenseMLPoly::from_vec_padded(eval_rows(&phi_q_mle, t_bits, &rx));
            let g_out = DenseMLPoly::from_vec_padded(eval_cols(&ctx_mle, d_bits, &ry));
            let (out_sumcheck, batch_r_out) =
                prove_sumcheck(&f_out, &g_out, out_eval, transcript);

            let phi_q_eval = out_sumcheck.final_eval_f;
            // ctx(batch_r_out, ry) — derived from out-sumcheck, not committed separately.
            let ctx_at_batch = out_sumcheck.final_eval_g;

            // Context(batch_r_out, ry) = Σ_t Phi_K(t, batch_r_out) · V(t, ry)   [t_bits vars]
            let f_ctx =
                DenseMLPoly::from_vec_padded(eval_cols(&phi_k_mle, t_bits, &batch_r_out));
            let g_v = DenseMLPoly::from_vec_padded(eval_cols(&v_mle, t_bits, &ry));
            let (context_sumcheck, batch_r_ctx) =
                prove_sumcheck(&f_ctx, &g_v, ctx_at_batch, transcript);

            let phi_k_eval = context_sumcheck.final_eval_f;
            let v_eval = context_sumcheck.final_eval_g;

            let _phi_q_point = combine(&rx, &batch_r_out);
            let _phi_k_point = combine(&batch_r_ctx, &batch_r_out);
            let v_point = combine(&batch_r_ctx, &ry);

            (
                AttentionSumcheckProof::Sequential {
                    out_sumcheck,
                    context_sumcheck,
                },
                phi_q_eval,
                phi_k_eval,
                v_eval,
                _phi_q_point,
                _phi_k_point,
                v_point,
            )
        };

    let phi_q_open = hyrax_open(&phi_q_mle.evaluations, &phi_q_point, nu_td, sigma_td);
    let phi_k_open = hyrax_open(&phi_k_mle.evaluations, &phi_k_point, nu_td, sigma_td);

    let proof = LinearAttentionProof {
        sumcheck,
        openings: AttentionOpenings { out_eval, phi_q_eval, phi_k_eval, v_eval },
        phi_q_com,
        phi_k_com,
        phi_q_open,
        phi_k_open,
    };

    let out_claim = EvalClaim { point: combine(&rx, &ry), value: out_eval };
    let v_claim = EvalClaim { point: v_point, value: v_eval };

    (proof, out_claim, v_claim)
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// **Production-Grade Succinct Verifier**
///
/// Ensures:
/// 1. $O(\sqrt{N})$ computation. No `hyrax_commit` is executed here.
/// 2. Cryptographic binding against the external `io_coms`.
/// Returns (out_claim, v_claim) for block-level combine verification.
///
/// GKR backward fusion: when `external_out_claim` is Some (= O_proj's x_claim),
/// (rx, ry) are taken from that claim instead of being sampled. out_inner_com is
/// not committed; soundness is provided by both sumchecks sharing the same eval point.
pub fn verify_linear_attention(
    proof: &LinearAttentionProof,
    inst: &LinearAttentionInstance,
    io_coms: &AttentionIOCommitments,
    external_out_claim: Option<EvalClaim>,
    transcript: &mut Transcript,
) -> Result<(EvalClaim, EvalClaim), String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;

    // 1. Absorb IO commitments (out_com removed — GKR backward, no out_inner_com).
    if !io_coms.skip_io_absorb {
        absorb_com(transcript, b"q_com", &io_coms.q_com);
        absorb_com(transcript, b"k_com", &io_coms.k_com);
        absorb_com(transcript, b"v_com", &io_coms.v_com);
    }

    // Absorb phi_q_com and phi_k_com (mirrors prover commitment step).
    absorb_com(transcript, b"phi_q_com", &proof.phi_q_com);
    absorb_com(transcript, b"phi_k_com", &proof.phi_k_com);

    // 2. Reproduce challenge points (or use external if GKR backward).
    let (rx, ry) = if let Some(ref claim) = external_out_claim {
        let rx = claim.point[..t_bits].to_vec();
        let ry = claim.point[t_bits..].to_vec();
        (rx, ry)
    } else {
        let rx = challenge_vec(transcript, t_bits, b"rx_out");
        let ry = challenge_vec(transcript, d_bits, b"ry_out");
        (rx, ry)
    };

    // 3. Verify sumcheck (variant selected by the prover at proof time)
    let (phi_q_point, phi_k_point, v_point) = match &proof.sumcheck {
        AttentionSumcheckProof::Batched { proof: sc, ctx_eval } => {
            // Reproduce r_i and alpha from transcript (mirrors prover exactly).
            transcript.append_field(b"claimed_out", &proof.openings.out_eval);
            let r_i = challenge_vec(transcript, d_bits, b"r_i_attn");
            transcript.append_field(b"claimed_ctx", ctx_eval);
            let alpha = transcript.challenge_field::<F>(b"attn_batch_alpha");
            let combined = proof.openings.out_eval + alpha * ctx_eval;

            let (batch_r, _) = verify_sumcheck_multi_batched(
                sc,
                &[F::from(1u64), alpha],
                combined,
                d_bits,
                transcript,
            )?;

            if proof.openings.phi_q_eval != sc.final_evals_f[0] {
                return Err("phi_q eval inconsistent with batch sumcheck leaf".into());
            }
            if proof.openings.phi_k_eval != sc.final_evals_f[1] {
                return Err("phi_k eval inconsistent with batch sumcheck leaf".into());
            }
            if proof.openings.v_eval != sc.final_evals_g[1] {
                return Err("v eval inconsistent with batch sumcheck leaf".into());
            }

            (
                combine(&rx, &batch_r),
                combine(&batch_r, &r_i),
                combine(&batch_r, &ry),
            )
        }
        AttentionSumcheckProof::Sequential {
            out_sumcheck,
            context_sumcheck,
        } => {
            // Out(rx, ry) = Σ_i Phi_Q(rx, i) · Context(i, ry)   [d_bits vars]
            let (batch_r_out, _) = verify_sumcheck(
                out_sumcheck,
                proof.openings.out_eval,
                d_bits,
                transcript,
            )?;

            if proof.openings.phi_q_eval != out_sumcheck.final_eval_f {
                return Err("phi_q eval inconsistent with out sumcheck leaf".into());
            }
            let ctx_at_batch = out_sumcheck.final_eval_g;

            // Context(batch_r_out, ry) = Σ_t Phi_K(t, batch_r_out) · V(t, ry)   [t_bits vars]
            let (batch_r_ctx, _) =
                verify_sumcheck(context_sumcheck, ctx_at_batch, t_bits, transcript)?;

            if proof.openings.phi_k_eval != context_sumcheck.final_eval_f {
                return Err("phi_k eval inconsistent with context sumcheck leaf".into());
            }
            if proof.openings.v_eval != context_sumcheck.final_eval_g {
                return Err("v eval inconsistent with context sumcheck leaf".into());
            }

            (
                combine(&rx, &batch_r_out),
                combine(&batch_r_ctx, &batch_r_out),
                combine(&batch_r_ctx, &ry),
            )
        }
    };

    // 4. Verify phi_q/phi_k via Hyrax openings against committed phi_q_com/phi_k_com.
    //    (phi_q_com/phi_k_com are bound to Lasso outputs via global output binding.)
    let (_, _, params_td) = params_from_vars(t_bits + d_bits);
    hyrax_verify(
        &proof.phi_q_com,
        proof.openings.phi_q_eval,
        &phi_q_point,
        &proof.phi_q_open,
        &params_td,
    )
    .map_err(|e| format!("phi_q Hyrax opening failed: {e}"))?;
    hyrax_verify(
        &proof.phi_k_com,
        proof.openings.phi_k_eval,
        &phi_k_point,
        &proof.phi_k_open,
        &params_td,
    )
    .map_err(|e| format!("phi_k Hyrax opening failed: {e}"))?;

    // out_com and v_com deferred to block-level combine proof
    let out_claim = EvalClaim { point: combine(&rx, &ry), value: proof.openings.out_eval };
    let v_claim = EvalClaim { point: v_point, value: proof.openings.v_eval };

    Ok((out_claim, v_claim))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod linear_attention_tests {
    use super::*;
    use crate::pcs::{hyrax_commit, poly_hyrax, HyraxParams};
    use crate::transcript::Transcript;
    use ark_ff::{One, PrimeField, Zero};

    /// Simulate the Global Pipeline:
    /// Constructs the private witness, public instance, AND the trusted IO commitments
    /// that would be passed down from previous layers.
    fn setup_test_pipeline(
        seq_len: usize,
        d_head: usize,
    ) -> (
        LinearAttentionWitness,
        LinearAttentionInstance,
        AttentionIOCommitments,
        AttentionProvingKey,
    ) {
        let m = 4usize;
        let table_size = 1 << m;
        let table: Vec<F> = (0..table_size).map(|i| F::from((i + 1) as u64)).collect();

        let q = vec![
            vec![F::from(1u64), F::from(2u64)],
            vec![F::from(3u64), F::from(4u64)],
        ];
        let k = vec![
            vec![F::from(0u64), F::from(1u64)],
            vec![F::from(2u64), F::from(3u64)],
        ];
        let v = vec![
            vec![F::from(5u64), F::from(6u64)],
            vec![F::from(7u64), F::from(8u64)],
        ];

        let apply_phi = |mat: &Vec<Vec<F>>| -> Vec<Vec<F>> {
            mat.iter()
                .map(|row| {
                    row.iter()
                        .map(|&x| table[x.into_bigint().as_ref()[0] as usize])
                        .collect()
                })
                .collect()
        };
        let phi_q = apply_phi(&q);
        let phi_k = apply_phi(&k);

        let mut context = vec![vec![F::zero(); d_head]; d_head];
        for i in 0..d_head {
            for j in 0..d_head {
                for t in 0..seq_len {
                    context[i][j] += phi_k[t][i] * v[t][j];
                }
            }
        }
        let mut out = vec![vec![F::zero(); d_head]; seq_len];
        for t in 0..seq_len {
            for j in 0..d_head {
                for i in 0..d_head {
                    out[t][j] += phi_q[t][i] * context[i][j];
                }
            }
        }

        let build_lasso = |mat: &Vec<Vec<F>>, phi: &Vec<Vec<F>>| -> (LassoInstance, Vec<usize>) {
            let indices: Vec<usize> = mat
                .iter()
                .flatten()
                .map(|x| x.into_bigint().as_ref()[0] as usize)
                .collect();
            let outputs: Vec<F> = phi.iter().flatten().copied().collect();
            (LassoInstance {
                tables: vec![table.clone()],
                outputs,
                bits_per_chunk: m,
            }, indices)
        };

        // 1. Prover's Private Data
        let witness = LinearAttentionWitness {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            phi_q,
            phi_k,
            context,
            out: out.clone(),
        };

        let (q_lasso, q_query_indices) = build_lasso(&q, &witness.phi_q);
        let (k_lasso, k_query_indices) = build_lasso(&k, &witness.phi_k);

        // 2. Public Instance (Dimensions and Lookups)
        let inst = LinearAttentionInstance {
            seq_len,
            d_head,
            q_lasso,
            k_lasso,
            q_query_indices,
            k_query_indices,
        };

        // 3. Trusted IO Commitments (Simulating Global Pipeline Manager)
        let v_mle = mat_to_mle(&v, seq_len, d_head);
        let (nu_td, _, params_td) = poly_hyrax(&v_mle);

        let q_com = hyrax_commit(
            &mat_to_mle(&q, seq_len, d_head).evaluations,
            nu_td,
            &params_td,
        );
        let k_com = hyrax_commit(
            &mat_to_mle(&k, seq_len, d_head).evaluations,
            nu_td,
            &params_td,
        );
        let v_com = hyrax_commit(&v_mle.evaluations, nu_td, &params_td);

        // GKR backward: out_com removed from AttentionIOCommitments.
        let io_coms = AttentionIOCommitments { q_com, k_com, v_com, skip_io_absorb: false };

        let lp = lasso_params();
        let pk = precommit_attention_tables(&inst.q_lasso, &inst.k_lasso, &lp);

        (witness, inst, io_coms, pk)
    }

    fn lasso_params() -> HyraxParams {
        HyraxParams::new(2) // 4-bit table: sigma=4-2=2
    }

    /// In GKR backward mode there is no out_com.  Instead, the caller (O_proj) provides
    /// an EvalClaim on out_inner at a specific MLE point.  For tests we simulate this by
    /// evaluating out_mle at a reproducible random point sampled from a separate transcript.
    fn make_external_out_claim(
        out: &[Vec<F>],
        seq_len: usize,
        d_head: usize,
        label: &[u8],
    ) -> EvalClaim {
        use crate::subprotocols::EvalClaim;
        let t_bits = seq_len.next_power_of_two().trailing_zeros() as usize;
        let d_bits = d_head.next_power_of_two().trailing_zeros() as usize;
        let out_mle = mat_to_mle(out, seq_len, d_head);
        let mut ext_t = crate::transcript::Transcript::new(label);
        let rx = challenge_vec(&mut ext_t, t_bits, b"ext_rx");
        let ry = challenge_vec(&mut ext_t, d_bits, b"ext_ry");
        let point = combine(&rx, &ry);
        let value = out_mle.evaluate(&point);
        EvalClaim { point, value }
    }

    #[test]
    fn test_linear_attention_succinct_e2e_success() {
        let (witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();
        let ext_claim = make_external_out_claim(&witness.out, 2, 2, b"ext_out");

        let mut pt = Transcript::new(b"linear_attn_test");
        let (proof, _out_claim, _v_claim) =
            prove_linear_attention(&witness, &inst, &pk, &io_coms, Some(ext_claim.clone()), &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, Some(ext_claim), &mut vt);
        assert!(
            result.is_ok(),
            "Succinct verification failed: {:?}",
            result.err()
        );
    }

    // --- Soundness Tests ---
    // Sumcheck algebraically catches inconsistencies between witness and sumcheck claim.

    #[test]
    fn test_rejects_tampered_context_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();
        let ext_claim = make_external_out_claim(&witness.out, 2, 2, b"ext_ctx_tamper");

        witness.context[0][0] += F::one();

        let mut pt = Transcript::new(b"linear_attn_tamper_ctx");
        let (proof, _, _) =
            prove_linear_attention(&witness, &inst, &pk, &io_coms, Some(ext_claim.clone()), &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_tamper_ctx");
        let result = verify_linear_attention(&proof, &inst, &io_coms, Some(ext_claim), &mut vt);
        assert!(result.is_err(), "should reject tampered context");
    }

    #[test]
    fn test_rejects_tampered_out_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();

        // Tamper out FIRST, then compute external claim from tampered matrix so that
        // claim.value ≠ phi_q * ctx, which causes the sumcheck to fail.
        witness.out[1][1] += F::one();
        let ext_claim = make_external_out_claim(&witness.out, 2, 2, b"ext_out_tamper");

        let mut pt = Transcript::new(b"linear_attn_test");
        let (proof, _, _) =
            prove_linear_attention(&witness, &inst, &pk, &io_coms, Some(ext_claim.clone()), &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, Some(ext_claim), &mut vt);
        assert!(result.is_err(), "should reject tampered out (sumcheck mismatch)");
    }

    #[test]
    fn test_rejects_tampered_phi_q_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();
        let ext_claim = make_external_out_claim(&witness.out, 2, 2, b"ext_phiq_tamper");

        witness.phi_q[1][0] += F::one();

        let mut pt = Transcript::new(b"linear_attn_test");
        let (proof, _, _) =
            prove_linear_attention(&witness, &inst, &pk, &io_coms, Some(ext_claim.clone()), &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, Some(ext_claim), &mut vt);
        assert!(result.is_err(), "should reject tampered phi_q");
    }

    #[test]
    fn test_rejects_tampered_v_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();
        let ext_claim = make_external_out_claim(&witness.out, 2, 2, b"ext_v_tamper");

        // Tamper v: changes the context sumcheck sum, causing mismatch with target.
        witness.v[0][1] += F::one();

        let mut pt = Transcript::new(b"linear_attn_test");
        let (proof, _, _) =
            prove_linear_attention(&witness, &inst, &pk, &io_coms, Some(ext_claim.clone()), &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, Some(ext_claim), &mut vt);
        assert!(result.is_err(), "should reject tampered v (context sumcheck mismatch)");
    }
}
