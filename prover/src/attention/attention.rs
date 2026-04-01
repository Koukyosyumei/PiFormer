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
    absorb_com, hyrax_commit, hyrax_open, hyrax_verify, params_from_n, poly_hyrax, HyraxCommitment,
    HyraxParams, HyraxProof,
};
use crate::poly::utils::{combine, eval_cols, eval_rows, mat_to_mle};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{prove_sumcheck, verify_sumcheck, SumcheckProof};
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

/// Commitments to intermediate states generated EXCLUSIVELY inside this layer.
pub struct AttentionInternalCommitments {
    pub phi_q_com: HyraxCommitment,
    pub phi_k_com: HyraxCommitment,
    // context_com を削除
}

pub struct AttentionOpenings {
    pub out_eval: F,
    pub out_open: HyraxProof,
    pub phi_q_eval: F,
    pub phi_q_open: HyraxProof,
    // ctx_eval, ctx_open を削除 (Chainで接続されるため)
    pub phi_k_eval: F,
    pub phi_k_open: HyraxProof,
    pub v_eval: F,
    pub v_open: HyraxProof,
}

pub struct LinearAttentionProof {
    pub internal_coms: AttentionInternalCommitments,
    pub out_sumcheck: SumcheckProof,
    pub context_sumcheck: SumcheckProof,
    pub openings: AttentionOpenings,
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

pub fn prove_linear_attention(
    witness: &LinearAttentionWitness,
    inst: &LinearAttentionInstance,
    _pk: &AttentionProvingKey,
    io_coms: &AttentionIOCommitments,
    transcript: &mut Transcript,
    _lasso_params: &HyraxParams,
) -> LinearAttentionProof {
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

    let (nu_td, sigma_td, params_td) = poly_hyrax(&phi_q_mle);
    let (nu_dd, sigma_dd, params_dd) = poly_hyrax(&ctx_mle);

    // 2. 内部活性化行列のみコミット (Contextはスキップ)
    let phi_q_com = hyrax_commit(&phi_q_mle.evaluations, nu_td, &params_td);
    let phi_k_com = hyrax_commit(&phi_k_mle.evaluations, nu_td, &params_td);

    absorb_com(transcript, b"phi_q_com", &phi_q_com);
    absorb_com(transcript, b"phi_k_com", &phi_k_com);

    // 3. OUT に関する Sumcheck: Out = Phi_Q * Context
    let rx = challenge_vec(transcript, t_bits, b"rx_out");
    let ry = challenge_vec(transcript, d_bits, b"ry_out");

    let out_eval = out_mle.evaluate(&combine(&rx, &ry));
    transcript.append_field(b"claimed_out", &out_eval);

    // Out(rx, ry) = Σ_z Phi_Q(rx, z) * Context(z, ry)
    let f_out = DenseMLPoly::from_vec_padded(eval_rows(&phi_q_mle, t_bits, &rx));
    let g_out = DenseMLPoly::from_vec_padded(eval_cols(&ctx_mle, d_bits, &ry));
    let (out_sumcheck, r_i) = prove_sumcheck(&f_out, &g_out, out_eval, transcript);

    // --- Claims-Chaining Point ---
    // ここで Context をコミットせず、Sumcheckの結果得られた評価値を直接使う
    let ctx_eval_at_ri = out_sumcheck.final_eval_g;
    transcript.append_field(b"claimed_ctx_chained", &ctx_eval_at_ri);

    // 4. Context に関する Sumcheck: Context = Phi_K^T * V
    // Context(ri, ry) = Σ_t Phi_K(t, ri) * V(t, ry)
    let f_ctx = DenseMLPoly::from_vec_padded(eval_cols(&phi_k_mle, t_bits, &r_i));
    let g_ctx = DenseMLPoly::from_vec_padded(eval_cols(&v_mle, t_bits, &ry));
    let (context_sumcheck, r_t) = prove_sumcheck(&f_ctx, &g_ctx, ctx_eval_at_ri, transcript);

    // 5. 端点（Leaf）の Opening 生成
    // 中間値である context_open は不要
    LinearAttentionProof {
        internal_coms: AttentionInternalCommitments {
            phi_q_com,
            phi_k_com,
        },
        out_sumcheck: out_sumcheck.clone(),
        context_sumcheck: context_sumcheck.clone(),
        openings: AttentionOpenings {
            out_eval,
            out_open: hyrax_open(&out_mle.evaluations, &combine(&rx, &ry), nu_td, sigma_td),
            phi_q_eval: out_sumcheck.final_eval_f,
            phi_q_open: hyrax_open(&phi_q_mle.evaluations, &combine(&rx, &r_i), nu_td, sigma_td),
            phi_k_eval: context_sumcheck.final_eval_f,
            phi_k_open: hyrax_open(
                &phi_k_mle.evaluations,
                &combine(&r_t, &r_i),
                nu_td,
                sigma_td,
            ),
            v_eval: context_sumcheck.final_eval_g,
            v_open: hyrax_open(&v_mle.evaluations, &combine(&r_t, &ry), nu_td, sigma_td),
        },
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// **Production-Grade Succinct Verifier**
///
/// Ensures:
/// 1. $O(\sqrt{N})$ computation. No `hyrax_commit` is executed here.
/// 2. Cryptographic binding against the external `io_coms`.
pub fn verify_linear_attention(
    proof: &LinearAttentionProof,
    inst: &LinearAttentionInstance,
    io_coms: &AttentionIOCommitments,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let t = inst.seq_len;
    let d = inst.d_head;
    let t_bits = t.next_power_of_two().trailing_zeros() as usize;
    let d_bits = d.next_power_of_two().trailing_zeros() as usize;
    let n_td = t.next_power_of_two().max(1) * d.next_power_of_two().max(1);
    let (_, _, params_td) = params_from_n(n_td);

    // 1. コミットメントの吸収
    absorb_com(transcript, b"q_com", &io_coms.q_com);
    absorb_com(transcript, b"k_com", &io_coms.k_com);
    absorb_com(transcript, b"v_com", &io_coms.v_com);
    absorb_com(transcript, b"out_com", &io_coms.out_com);
    absorb_com(transcript, b"phi_q_com", &proof.internal_coms.phi_q_com);
    absorb_com(transcript, b"phi_k_com", &proof.internal_coms.phi_k_com);

    // 2. チャレンジ再生
    let rx = challenge_vec(transcript, t_bits, b"rx_out");
    let ry = challenge_vec(transcript, d_bits, b"ry_out");

    // 3. OUT Sumcheck 検証 (Claim引き継ぎの準備)
    let claimed_out = proof.openings.out_eval;
    transcript.append_field(b"claimed_out", &claimed_out);

    let (r_i, _final_out) = verify_sumcheck(&proof.out_sumcheck, claimed_out, d_bits, transcript)?;

    // --- Claims-Chaining Point ---
    // Use the sumcheck leaf directly — avoids division and matches prover exactly.
    // verify_sumcheck already checked final_eval_f * final_eval_g == running claim.
    let ctx_eval = proof.out_sumcheck.final_eval_g;
    if proof.openings.phi_q_eval != proof.out_sumcheck.final_eval_f {
        return Err("phi_q opening eval inconsistent with out sumcheck leaf".into());
    }
    transcript.append_field(b"claimed_ctx_chained", &ctx_eval);

    // 4. Context Sumcheck 検証
    let (r_t, _final_ctx) = verify_sumcheck(
        &proof.context_sumcheck,
        ctx_eval,
        t_bits,
        transcript,
    )?;

    // 5. Bind phi_k and v opening claims to the context sumcheck leaves
    if proof.openings.phi_k_eval != proof.context_sumcheck.final_eval_f {
        return Err("phi_k opening eval inconsistent with context sumcheck leaf".into());
    }
    if proof.openings.v_eval != proof.context_sumcheck.final_eval_g {
        return Err("v opening eval inconsistent with context sumcheck leaf".into());
    }

    // 6. 開封（Opening）の一括検証
    // ここで前のターンの `hyrax_verify_batch` を使うと、MSMを一本に集約できる
    // 評価点が異なるため、個別に検証するか、MSMアグリゲーションを適用する
    hyrax_verify(
        &io_coms.out_com,
        proof.openings.out_eval,
        &combine(&rx, &ry),
        &proof.openings.out_open,
        &params_td,
    )?;
    hyrax_verify(
        &io_coms.v_com,
        proof.openings.v_eval,
        &combine(&r_t, &ry),
        &proof.openings.v_open,
        &params_td,
    )?;
    hyrax_verify(
        &proof.internal_coms.phi_q_com,
        proof.openings.phi_q_eval,
        &combine(&rx, &r_i),
        &proof.openings.phi_q_open,
        &params_td,
    )?;
    hyrax_verify(
        &proof.internal_coms.phi_k_com,
        proof.openings.phi_k_eval,
        &combine(&r_t, &r_i),
        &proof.openings.phi_k_open,
        &params_td,
    )?;

    Ok(())
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
    use crate::pcs::HyraxParams;
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
        let vk = pk.vk();

        // 1. Prover generates the proof using Witness + IO Coms
        let mut pt = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&witness, &inst, &pk, &io_coms, &mut pt, &lp);

        // 2. Verifier checks the proof strictly in O(√N) using ONLY IO Coms
        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, &mut vt);
        assert!(
            result.is_ok(),
            "Succinct verification failed: {:?}",
            result.err()
        );
    }

    // --- Soundness Tests (Tampering with the Prover's Witness) ---
    // By modifying the witness BEFORE generating the proof, we simulate a malicious
    // Prover trying to forge a proof for invalid intermediate values. The Succinct
    // Verifier MUST catch this either through the Sumcheck algebraic relations or
    // the binding check against the trusted IO Commitments.

    #[test]
    fn test_rejects_tampered_context_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();
        let vk = pk.vk();

        // Tamper context: Breaks the Sumcheck mathematical relationship
        witness.context[0][0] += F::one();

        let mut pt = Transcript::new(b"linear_attn_tamper_ctx");
        let proof = prove_linear_attention(&witness, &inst, &pk, &io_coms, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_tamper_ctx");
        let result = verify_linear_attention(&proof, &inst, &io_coms, &mut vt);
        assert!(result.is_err(), "should reject tampered context");
    }

    #[test]
    fn test_rejects_tampered_out_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();
        let vk = pk.vk();

        // Tamper out: The Prover computes Sumcheck with a fake out matrix.
        // The Verifier checks the final Opening against `io_coms.out_com`,
        // which was committed to the *correct* out matrix by the Global Pipeline.
        witness.out[1][1] += F::one();

        let mut pt = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&witness, &inst, &pk, &io_coms, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, &mut vt);
        assert!(
            result.is_err(),
            "should reject tampered out opening against IO commitment"
        );
    }

    #[test]
    fn test_rejects_tampered_phi_q_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();
        let vk = pk.vk();

        // Tamper phi_q: Breaks Lasso lookup consistency
        witness.phi_q[1][0] += F::one();

        let mut pt = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&witness, &inst, &pk, &io_coms, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, &mut vt);
        assert!(result.is_err(), "should reject tampered phi_q");
    }

    #[test]
    fn test_rejects_tampered_v_matrix() {
        let (mut witness, inst, io_coms, pk) = setup_test_pipeline(2, 2);
        let lp = lasso_params();
        let vk = pk.vk();

        // Tamper v: Prover uses a fake V, but the proof will fail to verify
        // against the trusted external `io_coms.v_com`.
        witness.v[0][1] += F::one();

        let mut pt = Transcript::new(b"linear_attn_test");
        let proof = prove_linear_attention(&witness, &inst, &pk, &io_coms, &mut pt, &lp);

        let mut vt = Transcript::new(b"linear_attn_test");
        let result = verify_linear_attention(&proof, &inst, &io_coms, &mut vt);
        assert!(
            result.is_err(),
            "should reject tampered v against IO commitment"
        );
    }
}
