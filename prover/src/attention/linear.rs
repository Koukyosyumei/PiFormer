//! Arithmetic circuit for one head of linear attention.
//!
//! **Computations proved:**
//!   1. phiQ = φ(Q)                   — via Lasso lookup (per token × feature)
//!   2. phiK = φ(K)                   — via Lasso lookup
//!   3. context[i][j] = Σ_t phiK[t][i]·V[t][j]   — matrix multiply via sumcheck
//!   4. out[t][j]     = Σ_i phiQ[t][i]·context[i][j]  — matrix-vector via sumcheck
//!
//! Steps 3 & 4 are proved by the verifier selecting a random entry to audit
//! (Fiat-Shamir), then running a sumcheck over the inner-product dimension.
//! This is a standard "random-entry" opening reduction for matrix products.

use ark_ff::PrimeField;
use crate::field::F;
use crate::lookup::{LassoInstance, LassoProof, prove_lasso, verify_lasso};
use crate::poly::DenseMLPoly;
use crate::subprotocols::{SumcheckProof, prove_sumcheck, verify_sumcheck};
use crate::transcript::Transcript;

/// Full witness for proving one attention head.
pub struct LinearAttentionInstance {
    pub seq_len: usize,
    pub d_head: usize,
    /// Q, K, V matrices as field elements: shape (seq_len, d_head).
    pub q: Vec<Vec<F>>,
    pub k: Vec<Vec<F>>,
    pub v: Vec<Vec<F>>,
    /// φ(Q) and φ(K): same shape.
    pub phi_q: Vec<Vec<F>>,
    pub phi_k: Vec<Vec<F>>,
    /// context = φ(K)^T · V: shape (d_head, d_head).
    pub context: Vec<Vec<F>>,
    /// out = φ(Q) · context: shape (seq_len, d_head).
    pub out: Vec<Vec<F>>,
    /// Lasso instances for the φ lookups.
    pub q_lasso: LassoInstance,
    pub k_lasso: LassoInstance,
}

/// Proof for one attention head.
pub struct LinearAttentionProof {
    pub phi_q_proof: LassoProof,
    pub phi_k_proof: LassoProof,
    /// Sumcheck over T (seq dimension) for one audited context entry.
    pub context_sumcheck: SumcheckProof,
    /// Sumcheck over D (head-dim) for one audited output entry.
    pub out_sumcheck: SumcheckProof,
}

pub fn prove_linear_attention(
    inst: &LinearAttentionInstance,
    transcript: &mut Transcript,
) -> LinearAttentionProof {
    let T = inst.seq_len;
    let D = inst.d_head;

    // Step 1 & 2: Lasso for φ(Q) and φ(K)
    let phi_q_proof = prove_lasso(&inst.q_lasso, transcript);
    let phi_k_proof = prove_lasso(&inst.k_lasso, transcript);

    // Step 3: Prove one entry of context = φ(K)^T · V
    //   Verifier picks (i*, j*) via Fiat-Shamir.
    //   We prove: context[i*][j*] = Σ_{t=0}^{T-1} phiK[t][i*] · V[t][j*]
    //   via sumcheck over t.
    let r_ctx = transcript.challenge_field::<F>(b"ctx_query");
    let (i_star, j_star) = entry_from_challenge(r_ctx, D, D);

    let phiK_col: Vec<F> = inst.phi_k.iter().map(|row| row[i_star]).collect();
    let v_col:    Vec<F> = inst.v.iter().map(|row| row[j_star]).collect();
    let claimed_ctx = inst.context[i_star][j_star];

    let f_ctx = DenseMLPoly::from_vec_padded(phiK_col);
    let g_ctx = DenseMLPoly::from_vec_padded(v_col);
    transcript.append_field(b"claimed_ctx", &claimed_ctx);
    let (context_sumcheck, _) = prove_sumcheck(&f_ctx, &g_ctx, claimed_ctx, transcript);

    // Step 4: Prove one entry of out = φ(Q) · context
    //   Verifier picks (t*, j*) via Fiat-Shamir.
    //   We prove: out[t*][j_out] = Σ_{i=0}^{D-1} phiQ[t*][i] · context[i][j_out]
    let r_out = transcript.challenge_field::<F>(b"out_query");
    let (t_star, j_out) = entry_from_challenge(r_out, T, D);

    let phiQ_row:    Vec<F> = inst.phi_q[t_star].clone();
    let ctx_col_j: Vec<F> = inst.context.iter().map(|row| row[j_out]).collect();
    let claimed_out = inst.out[t_star][j_out];

    let f_out = DenseMLPoly::from_vec_padded(phiQ_row);
    let g_out = DenseMLPoly::from_vec_padded(ctx_col_j);
    transcript.append_field(b"claimed_out", &claimed_out);
    let (out_sumcheck, _) = prove_sumcheck(&f_out, &g_out, claimed_out, transcript);

    LinearAttentionProof { phi_q_proof, phi_k_proof, context_sumcheck, out_sumcheck }
}

pub fn verify_linear_attention(
    proof: &LinearAttentionProof,
    inst: &LinearAttentionInstance,
    transcript: &mut Transcript,
) -> Result<(), String> {
    let T = inst.seq_len;
    let D = inst.d_head;

    verify_lasso(&proof.phi_q_proof, &inst.q_lasso, transcript)
        .map_err(|e| format!("phi_q: {e}"))?;
    verify_lasso(&proof.phi_k_proof, &inst.k_lasso, transcript)
        .map_err(|e| format!("phi_k: {e}"))?;

    // Replicate the verifier's Fiat-Shamir choices
    let r_ctx = transcript.challenge_field::<F>(b"ctx_query");
    let (i_star, j_star) = entry_from_challenge(r_ctx, D, D);
    let claimed_ctx = inst.context[i_star][j_star];
    transcript.append_field(b"claimed_ctx", &claimed_ctx);

    let seq_bits = T.next_power_of_two().trailing_zeros() as usize;
    verify_sumcheck(&proof.context_sumcheck, claimed_ctx, seq_bits, transcript)
        .map_err(|e| format!("context sumcheck: {e}"))?;

    let r_out = transcript.challenge_field::<F>(b"out_query");
    let (t_star, j_out) = entry_from_challenge(r_out, T, D);
    let claimed_out = inst.out[t_star][j_out];
    transcript.append_field(b"claimed_out", &claimed_out);

    let d_bits = D.next_power_of_two().trailing_zeros() as usize;
    verify_sumcheck(&proof.out_sumcheck, claimed_out, d_bits, transcript)
        .map_err(|e| format!("out sumcheck: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Derive two bounded indices from a field challenge (mod rows, mod cols).
fn entry_from_challenge(r: F, rows: usize, cols: usize) -> (usize, usize) {
    let limb = PrimeField::into_bigint(r).as_ref()[0];
    let row = (limb as usize) % rows;
    let col = ((limb >> 32) as usize) % cols;
    (row, col)
}
