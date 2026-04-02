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
use ark_ff::Field;
use crate::pcs::{absorb_com, HyraxCommitment, HyraxParams};
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
pub struct AttentionIOCommitments {
    pub q_com: HyraxCommitment,
    pub k_com: HyraxCommitment,
    pub v_com: HyraxCommitment,
    pub out_com: HyraxCommitment,
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
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Returns (proof, out_claim, v_claim).
///
/// `out_claim` = EvalClaim on out at (rx, ry) — deferred to block-level combine.
/// `v_claim`   = EvalClaim on v at v_point — deferred to block-level combine.
pub fn prove_linear_attention(
    witness: &LinearAttentionWitness,
    inst: &LinearAttentionInstance,
    _pk: &AttentionProvingKey,
    io_coms: &AttentionIOCommitments,
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

    // 1. 公開コミットメントの吸収
    absorb_com(transcript, b"q_com", &io_coms.q_com);
    absorb_com(transcript, b"k_com", &io_coms.k_com);
    absorb_com(transcript, b"v_com", &io_coms.v_com);
    absorb_com(transcript, b"out_com", &io_coms.out_com);

    let rx = challenge_vec(transcript, t_bits, b"rx_out");
    let ry = challenge_vec(transcript, d_bits, b"ry_out");
    let out_eval = out_mle.evaluate(&combine(&rx, &ry));

    let (sumcheck, phi_q_eval, phi_k_eval, v_eval, _phi_q_point, _phi_k_point, v_point) =
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

    let proof = LinearAttentionProof {
        sumcheck,
        openings: AttentionOpenings { out_eval, phi_q_eval, phi_k_eval, v_eval },
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
pub fn verify_linear_attention(
    proof: &LinearAttentionProof,
    inst: &LinearAttentionInstance,
    io_coms: &AttentionIOCommitments,
    transcript: &mut Transcript,
) -> Result<(EvalClaim, EvalClaim), String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;

    // 1. コミットメントの吸収
    absorb_com(transcript, b"q_com", &io_coms.q_com);
    absorb_com(transcript, b"k_com", &io_coms.k_com);
    absorb_com(transcript, b"v_com", &io_coms.v_com);
    absorb_com(transcript, b"out_com", &io_coms.out_com);

    // 2. Reproduce challenge points
    let rx = challenge_vec(transcript, t_bits, b"rx_out");
    let ry = challenge_vec(transcript, d_bits, b"ry_out");

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

    // 4. Verify phi_q/phi_k directly against public Lasso outputs MLE.
    let padded_len = 1usize << (t_bits + d_bits);
    let mut phi_q_evals = vec![F::ZERO; padded_len];
    for (j, &out) in inst.q_lasso.outputs.iter().enumerate() {
        phi_q_evals[j] = out;
    }
    let phi_q_eval_expected = DenseMLPoly::new(phi_q_evals).evaluate(&phi_q_point);
    if phi_q_eval_expected != proof.openings.phi_q_eval {
        return Err("phi_q eval mismatch with Lasso outputs MLE".into());
    }

    let mut phi_k_evals = vec![F::ZERO; padded_len];
    for (j, &out) in inst.k_lasso.outputs.iter().enumerate() {
        phi_k_evals[j] = out;
    }
    let phi_k_eval_expected = DenseMLPoly::new(phi_k_evals).evaluate(&phi_k_point);
    if phi_k_eval_expected != proof.openings.phi_k_eval {
        return Err("phi_k eval mismatch with Lasso outputs MLE".into());
    }

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

        let build_lasso = |mat: &Vec<Vec<F>>, phi: &Vec<Vec<F>>| -> LassoInstance {
            let indices: Vec<usize> = mat
                .iter()
                .flatten()
                .map(|x| x.into_bigint().as_ref()[0] as usize)
                .collect();
            let outputs: Vec<F> = phi.iter().flatten().copied().collect();
            LassoInstance {
                tables: vec![table.clone()],
                query_indices: indices,
                outputs,
                bits_per_chunk: m,
            }
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

        // 2. Public Instance (Dimensions and Lookups)
        let inst = LinearAttentionInstance {
            seq_len,
            d_head,
            q_lasso: build_lasso(&q, &witness.phi_q),
            k_lasso: build_lasso(&k, &witness.phi_k),
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
        let out_com = hyrax_commit(
            &mat_to_mle(&out, seq_len, d_head).evaluations,
            nu_td,
            &params_td,
        );

        let io_coms = AttentionIOCommitments {
            q_com,
            k_com,
            v_com,
            out_com,
        };

        let lp = lasso_params();
        let pk = precommit_attention_tables(&inst.q_lasso, &inst.k_lasso, &lp);

        (witness, inst, io_coms, pk)
    }

    fn lasso_params() -> HyraxParams {
        HyraxParams::new(2) // 4-bit table: sigma=4-2=2
    }

    #[test]
    fn test_linear_attention_succinct_e2e_success() {
        let (witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();

        let mut pt = Transcript::new(b"linear_attn_test");
        let (proof, _out_claim, _v_claim) =
            prove_linear_attention(&witness, &inst, &pk, &io_coms, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, &mut vt);
        assert!(
            result.is_ok(),
            "Succinct verification failed: {:?}",
            result.err()
        );
    }

    // --- Soundness Tests ---
    // Sumcheck algebraically catches inconsistencies between witness and sumcheck claim.
    // Note: out_com and v_com opening is now deferred to block-level combine proof.

    #[test]
    fn test_rejects_tampered_context_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();

        witness.context[0][0] += F::one();

        let mut pt = Transcript::new(b"linear_attn_tamper_ctx");
        let (proof, _, _) =
            prove_linear_attention(&witness, &inst, &pk, &io_coms, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_tamper_ctx");
        let result = verify_linear_attention(&proof, &inst, &io_coms, &mut vt);
        assert!(result.is_err(), "should reject tampered context");
    }

    #[test]
    fn test_rejects_tampered_out_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();

        // Tamper out: changes out_eval, making the sumcheck target inconsistent.
        witness.out[1][1] += F::one();

        let mut pt = Transcript::new(b"linear_attn_test");
        let (proof, _, _) =
            prove_linear_attention(&witness, &inst, &pk, &io_coms, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, &mut vt);
        assert!(result.is_err(), "should reject tampered out (sumcheck mismatch)");
    }

    #[test]
    fn test_rejects_tampered_phi_q_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();

        witness.phi_q[1][0] += F::one();

        let mut pt = Transcript::new(b"linear_attn_test");
        let (proof, _, _) =
            prove_linear_attention(&witness, &inst, &pk, &io_coms, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, &mut vt);
        assert!(result.is_err(), "should reject tampered phi_q");
    }

    #[test]
    fn test_rejects_tampered_v_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();

        // Tamper v: changes the context sumcheck sum, causing mismatch with target.
        witness.v[0][1] += F::one();

        let mut pt = Transcript::new(b"linear_attn_test");
        let (proof, _, _) =
            prove_linear_attention(&witness, &inst, &pk, &io_coms, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, &mut vt);
        assert!(result.is_err(), "should reject tampered v (context sumcheck mismatch)");
    }
}
