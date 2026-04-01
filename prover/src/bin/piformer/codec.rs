//! Binary codec for PiFormer key and proof files.
//!
//! File formats:
//!   *.pk   – full TransformerModelProvingKey (includes raw weights)
//!   *.vk   – slim TransformerModelVerifyingKey (G1 commitments only, no weights)
//!   *.bin  – proof bundle: ModelProof + public instances + lasso_sigma

use std::io::{self, Read, Write};

use ark_bn254::G1Affine;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};

use piformer_prover::{
    attention::{
        attention::{
            AttentionInternalCommitments, AttentionOpenings, AttentionProvingKey,
            AttentionSumcheckProof, AttentionVerifyingKey, LinearAttentionInstance,
            LinearAttentionProof,
        },
        layernorm::{
            LayerNormInternalCommitments, LayerNormOpenings, LayerNormProof, LayerNormVerifyingKey,
        },
        projection::{
            ProjectionOpenings, ProjectionProof, ProjectionProvingKey, ProjectionVerifyingKey,
        },
    },
    ffn::ffn::{
        FFNInstance, FFNInternalCommitments, FFNOpenings, FFNProof, FFNProvingKey, FFNVerifyingKey,
    },
    lookup::{
        lasso::{
            LassoInstance, LassoMultiProof, LassoMultiProvingKey, LassoMultiVerifyingKey,
            LassoProof, LassoProvingKey, LassoVerifyingKey,
        },
        range::RangeProof,
    },
    pcs::{HyraxCommitment, HyraxProof},
    poly::utils::TernaryValue,
    prover::{
        TransformerBlockProof, TransformerModelProof, TransformerModelProvingKey,
        TransformerModelVerifyingKey,
    },
    subprotocols::{
        combine::CombineProof,
        sumcheck::{RoundPoly, SumcheckProof, SumcheckProofMulti},
    },
    verifier::TransformerBlockVerifyingKey,
    F,
};

// ---------------------------------------------------------------------------
// Magic bytes & version
// ---------------------------------------------------------------------------
const PK_MAGIC: &[u8; 8] = b"PFMR_PK\0";
const VK_MAGIC: &[u8; 8] = b"PFMR_VK\0";
const PROOF_MAGIC: &[u8; 8] = b"PFMR_PR\0";
const VERSION: u8 = 1;

// ---------------------------------------------------------------------------
// Low-level primitives
// ---------------------------------------------------------------------------

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
fn write_usize<W: Write>(w: &mut W, v: usize) -> io::Result<()> {
    write_u64(w, v as u64)
}
fn read_usize<R: Read>(r: &mut R) -> io::Result<usize> {
    Ok(read_u64(r)? as usize)
}

fn ark_err(e: ark_serialize::SerializationError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

fn write_f<W: Write>(w: &mut W, f: &F) -> io::Result<()> {
    f.serialize_with_mode(&mut *w, Compress::No)
        .map_err(ark_err)
}
fn write_t<W: Write>(w: &mut W, t: &TernaryValue) -> io::Result<()> {
    let byte: u8 = match t {
        TernaryValue::ZERO => 0,
        TernaryValue::ONE => 1,
        TernaryValue::MINUSONE => 2,
    };
    w.write_all(&[byte])
}
fn read_t<R: Read>(r: &mut R) -> io::Result<TernaryValue> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    match buf[0] {
        0 => Ok(TernaryValue::ZERO),
        1 => Ok(TernaryValue::ONE),
        2 => Ok(TernaryValue::MINUSONE),
        b => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid ternary byte: {b}"),
        )),
    }
}
fn read_f<R: Read>(r: &mut R) -> io::Result<F> {
    F::deserialize_with_mode(&mut *r, Compress::No, Validate::No).map_err(ark_err)
}
fn write_g1<W: Write>(w: &mut W, g: &G1Affine) -> io::Result<()> {
    g.serialize_compressed(&mut *w).map_err(ark_err)
}
fn read_g1<R: Read>(r: &mut R) -> io::Result<G1Affine> {
    G1Affine::deserialize_compressed(&mut *r).map_err(ark_err)
}

fn write_vec<T, W: Write, F2: Fn(&mut W, &T) -> io::Result<()>>(
    w: &mut W,
    v: &[T],
    f: F2,
) -> io::Result<()> {
    write_u64(w, v.len() as u64)?;
    for x in v {
        f(w, x)?;
    }
    Ok(())
}
fn read_vec<T, R: Read, F2: Fn(&mut R) -> io::Result<T>>(r: &mut R, f: F2) -> io::Result<Vec<T>> {
    let n = read_u64(r)? as usize;
    (0..n).map(|_| f(r)).collect()
}

fn write_vec_f<W: Write>(w: &mut W, v: &[F]) -> io::Result<()> {
    write_vec(w, v, write_f)
}
fn write_vec_t<W: Write>(w: &mut W, v: &[TernaryValue]) -> io::Result<()> {
    write_vec(w, v, write_t)
}
fn read_vec_f<R: Read>(r: &mut R) -> io::Result<Vec<F>> {
    read_vec(r, read_f)
}
fn write_vec_g1<W: Write>(w: &mut W, v: &[G1Affine]) -> io::Result<()> {
    write_vec(w, v, write_g1)
}
fn read_vec_g1<R: Read>(r: &mut R) -> io::Result<Vec<G1Affine>> {
    read_vec(r, read_g1)
}
fn write_vec_vec_f<W: Write>(w: &mut W, v: &[Vec<F>]) -> io::Result<()> {
    write_vec(w, v, |w2, row| write_vec_f(w2, row))
}
fn write_vec_vec_t<W: Write>(w: &mut W, v: &[Vec<TernaryValue>]) -> io::Result<()> {
    write_vec(w, v, |w2, row| write_vec_t(w2, row))
}
fn read_vec_t<R: Read>(r: &mut R) -> io::Result<Vec<TernaryValue>> {
    read_vec(r, read_t)
}
fn read_vec_vec_t<R: Read>(r: &mut R) -> io::Result<Vec<Vec<TernaryValue>>> {
    read_vec(r, read_vec_t)
}
fn read_vec_vec_f<R: Read>(r: &mut R) -> io::Result<Vec<Vec<F>>> {
    read_vec(r, read_vec_f)
}
fn write_vec_usize<W: Write>(w: &mut W, v: &[usize]) -> io::Result<()> {
    write_vec(w, v, |w2, x| write_usize(w2, *x))
}
fn read_vec_usize<R: Read>(r: &mut R) -> io::Result<Vec<usize>> {
    read_vec(r, read_usize)
}
fn write_bool<W: Write>(w: &mut W, b: bool) -> io::Result<()> {
    w.write_all(&[b as u8])
}
fn read_bool<R: Read>(r: &mut R) -> io::Result<bool> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}

// ---------------------------------------------------------------------------
// Hyrax primitives
// ---------------------------------------------------------------------------

fn write_hyrax_proof<W: Write>(w: &mut W, p: &HyraxProof) -> io::Result<()> {
    write_vec_f(w, &p.w_prime)
}
fn read_hyrax_proof<R: Read>(r: &mut R) -> io::Result<HyraxProof> {
    Ok(HyraxProof {
        w_prime: read_vec_f(r)?,
    })
}

fn write_hyrax_commitment<W: Write>(w: &mut W, c: &HyraxCommitment) -> io::Result<()> {
    write_vec_g1(w, &c.row_coms)?;
    write_usize(w, c.nu)?;
    write_usize(w, c.sigma)
}
fn read_hyrax_commitment<R: Read>(r: &mut R) -> io::Result<HyraxCommitment> {
    Ok(HyraxCommitment {
        row_coms: read_vec_g1(r)?,
        nu: read_usize(r)?,
        sigma: read_usize(r)?,
    })
}

// Convenience: eval + proof pair
macro_rules! write_ep {
    ($w:expr, $eval:expr, $proof:expr) => {{
        write_f($w, $eval)?;
        write_hyrax_proof($w, $proof)?;
    }};
}
macro_rules! read_ep {
    ($r:expr) => {{
        (read_f($r)?, read_hyrax_proof($r)?)
    }};
}

// ---------------------------------------------------------------------------
// Sumcheck
// ---------------------------------------------------------------------------

fn write_round_poly<W: Write>(w: &mut W, rp: &RoundPoly) -> io::Result<()> {
    write_f(w, &rp.evals[0])?;
    write_f(w, &rp.evals[1])?;
    write_f(w, &rp.evals[2])
}
fn read_round_poly<R: Read>(r: &mut R) -> io::Result<RoundPoly> {
    Ok(RoundPoly {
        evals: [read_f(r)?, read_f(r)?, read_f(r)?],
    })
}

fn write_sumcheck_proof<W: Write>(w: &mut W, p: &SumcheckProof) -> io::Result<()> {
    write_vec(w, &p.round_polys, write_round_poly)?;
    write_f(w, &p.final_eval_f)?;
    write_f(w, &p.final_eval_g)
}
fn read_sumcheck_proof<R: Read>(r: &mut R) -> io::Result<SumcheckProof> {
    Ok(SumcheckProof {
        round_polys: read_vec(r, read_round_poly)?,
        final_eval_f: read_f(r)?,
        final_eval_g: read_f(r)?,
    })
}

// ---------------------------------------------------------------------------
// Lasso & Range
// ---------------------------------------------------------------------------

fn write_lasso_proof<W: Write>(w: &mut W, p: &LassoProof) -> io::Result<()> {
    write_vec_f(w, &p.sub_claims)?;
    write_vec(w, &p.sumcheck_proofs, write_sumcheck_proof)?;
    write_vec_f(w, &p.table_openings)?;
    write_vec(w, &p.hyrax_proofs, write_hyrax_proof)
}
fn read_lasso_proof<R: Read>(r: &mut R) -> io::Result<LassoProof> {
    Ok(LassoProof {
        sub_claims: read_vec_f(r)?,
        sumcheck_proofs: read_vec(r, read_sumcheck_proof)?,
        table_openings: read_vec_f(r)?,
        hyrax_proofs: read_vec(r, read_hyrax_proof)?,
    })
}

fn write_sumcheck_proof_multi<W: Write>(w: &mut W, p: &SumcheckProofMulti) -> io::Result<()> {
    write_vec(w, &p.round_polys, write_round_poly)?;
    write_vec_f(w, &p.final_evals_f)?;
    write_vec_f(w, &p.final_evals_g)
}
fn read_sumcheck_proof_multi<R: Read>(r: &mut R) -> io::Result<SumcheckProofMulti> {
    Ok(SumcheckProofMulti {
        round_polys: read_vec(r, read_round_poly)?,
        final_evals_f: read_vec_f(r)?,
        final_evals_g: read_vec_f(r)?,
    })
}

fn write_lasso_multi_proof<W: Write>(w: &mut W, p: &LassoMultiProof) -> io::Result<()> {
    write_f(w, &p.combined_grand_sum)?;
    write_sumcheck_proof_multi(w, &p.combined_sumcheck_proof)?;
    write_vec(w, &p.table_openings, |w, v: &Vec<F>| write_vec_f(w, v))?;
    write_hyrax_proof(w, &p.hyrax_proof)
}
fn read_lasso_multi_proof<R: Read>(r: &mut R) -> io::Result<LassoMultiProof> {
    Ok(LassoMultiProof {
        combined_grand_sum: read_f(r)?,
        combined_sumcheck_proof: read_sumcheck_proof_multi(r)?,
        table_openings: read_vec(r, |r| read_vec_f(r))?,
        hyrax_proof: read_hyrax_proof(r)?,
    })
}

fn write_range_proof<W: Write>(w: &mut W, p: &RangeProof) -> io::Result<()> {
    write_sumcheck_proof(w, &p.sumcheck)?;
    write_f(w, &p.claim_v)?;
    write_vec(w, &p.chunk_coms, write_hyrax_commitment)?;
    write_vec_f(w, &p.chunk_evals)?;
    write_vec(w, &p.chunk_opens, write_hyrax_proof)?;
    write_hyrax_commitment(w, &p.m_com)?;
    write_f(w, &p.m_eval)?;
    write_hyrax_proof(w, &p.m_open)
}
fn read_range_proof<R: Read>(r: &mut R) -> io::Result<RangeProof> {
    Ok(RangeProof {
        sumcheck: read_sumcheck_proof(r)?,
        claim_v: read_f(r)?,
        chunk_coms: read_vec(r, read_hyrax_commitment)?,
        chunk_evals: read_vec_f(r)?,
        chunk_opens: read_vec(r, read_hyrax_proof)?,
        m_com: read_hyrax_commitment(r)?,
        m_eval: read_f(r)?,
        m_open: read_hyrax_proof(r)?,
    })
}

// ---------------------------------------------------------------------------
// LayerNorm proof
// ---------------------------------------------------------------------------

fn write_ln_internal_coms<W: Write>(w: &mut W, c: &LayerNormInternalCommitments) -> io::Result<()> {
    write_hyrax_commitment(w, &c.sum_x_com)?;
    write_hyrax_commitment(w, &c.sq_sum_x_com)?;
    write_hyrax_commitment(w, &c.sum_x_sq_com)?;
    write_hyrax_commitment(w, &c.sigma_com)?;
    write_hyrax_commitment(w, &c.sigma_sq_com)?;
    write_hyrax_commitment(w, &c.sigma_y_com)?;
    write_hyrax_commitment(w, &c.gamma_x_com)
}
fn read_ln_internal_coms<R: Read>(r: &mut R) -> io::Result<LayerNormInternalCommitments> {
    Ok(LayerNormInternalCommitments {
        sum_x_com: read_hyrax_commitment(r)?,
        sq_sum_x_com: read_hyrax_commitment(r)?,
        sum_x_sq_com: read_hyrax_commitment(r)?,
        sigma_com: read_hyrax_commitment(r)?,
        sigma_sq_com: read_hyrax_commitment(r)?,
        sigma_y_com: read_hyrax_commitment(r)?,
        gamma_x_com: read_hyrax_commitment(r)?,
    })
}

fn write_ln_openings<W: Write>(w: &mut W, o: &LayerNormOpenings) -> io::Result<()> {
    // Group 1: at r_t
    write_f(w, &o.sum_x_at_rt)?;
    write_f(w, &o.sq_sum_x_at_rt)?;
    write_hyrax_proof(w, &o.rt_batch_proof)?;
    // Individual: x at combine(r_t, r_d_mean)
    write_ep!(w, &o.x_at_rt_rmean, &o.x_rt_rmean_proof);
    // Group 2: at r_sig_t
    write_f(w, &o.sum_x_at_rsig)?;
    write_f(w, &o.sq_sum_x_at_rsig)?;
    write_f(w, &o.sigma_at_rsig)?;
    write_f(w, &o.sigma_sq_at_rsig)?;
    write_f(w, &o.sum_x_sq_at_rsig)?;
    write_hyrax_proof(w, &o.rsig_batch_proof)?;
    // Group 3: at combine(r_y_t, r_y_d)
    write_f(w, &o.x_at_ry)?;
    write_f(w, &o.y_at_ry)?;
    write_f(w, &o.gamma_x_at_ry)?;
    write_f(w, &o.sigma_y_at_ry)?;
    write_hyrax_proof(w, &o.ry_td_batch_proof)?;
    // Group 4: at r_y_t
    write_f(w, &o.sum_x_at_ryt)?;
    write_f(w, &o.sigma_at_ryt)?;
    write_hyrax_proof(w, &o.ryt_batch_proof)
}
fn read_ln_openings<R: Read>(r: &mut R) -> io::Result<LayerNormOpenings> {
    // Group 1: at r_t
    let sum_x_at_rt = read_f(r)?;
    let sq_sum_x_at_rt = read_f(r)?;
    let rt_batch_proof = read_hyrax_proof(r)?;
    // Individual: x at combine(r_t, r_d_mean)
    let (x_at_rt_rmean, x_rt_rmean_proof) = read_ep!(r);
    // Group 2: at r_sig_t
    let sum_x_at_rsig = read_f(r)?;
    let sq_sum_x_at_rsig = read_f(r)?;
    let sigma_at_rsig = read_f(r)?;
    let sigma_sq_at_rsig = read_f(r)?;
    let sum_x_sq_at_rsig = read_f(r)?;
    let rsig_batch_proof = read_hyrax_proof(r)?;
    // Group 3: at combine(r_y_t, r_y_d)
    let x_at_ry = read_f(r)?;
    let y_at_ry = read_f(r)?;
    let gamma_x_at_ry = read_f(r)?;
    let sigma_y_at_ry = read_f(r)?;
    let ry_td_batch_proof = read_hyrax_proof(r)?;
    // Group 4: at r_y_t
    let sum_x_at_ryt = read_f(r)?;
    let sigma_at_ryt = read_f(r)?;
    let ryt_batch_proof = read_hyrax_proof(r)?;
    Ok(LayerNormOpenings {
        sum_x_at_rt,
        sq_sum_x_at_rt,
        rt_batch_proof,
        x_at_rt_rmean,
        x_rt_rmean_proof,
        sum_x_at_rsig,
        sq_sum_x_at_rsig,
        sigma_at_rsig,
        sigma_sq_at_rsig,
        sum_x_sq_at_rsig,
        rsig_batch_proof,
        x_at_ry,
        y_at_ry,
        gamma_x_at_ry,
        sigma_y_at_ry,
        ry_td_batch_proof,
        sum_x_at_ryt,
        sigma_at_ryt,
        ryt_batch_proof,
    })
}

fn write_ln_proof<W: Write>(w: &mut W, p: &LayerNormProof) -> io::Result<()> {
    write_ln_internal_coms(w, &p.internal_coms)?;
    write_sumcheck_proof(w, &p.mean_sumcheck)?;
    write_range_proof(w, &p.sigma_range_proof)?;
    write_range_proof(w, &p.y_range_proof)?;
    write_ln_openings(w, &p.openings)
}
fn read_ln_proof<R: Read>(r: &mut R) -> io::Result<LayerNormProof> {
    Ok(LayerNormProof {
        internal_coms: read_ln_internal_coms(r)?,
        mean_sumcheck: read_sumcheck_proof(r)?,
        sigma_range_proof: read_range_proof(r)?,
        y_range_proof: read_range_proof(r)?,
        openings: read_ln_openings(r)?,
    })
}

// ---------------------------------------------------------------------------
// Projection proof
// ---------------------------------------------------------------------------

fn write_proj_openings<W: Write>(w: &mut W, o: &ProjectionOpenings) -> io::Result<()> {
    write_f(w, &o.y_eval)?;
    write_f(w, &o.x_eval)?;
    write_ep!(w, &o.w_eval, &o.w_open);
    write_ep!(w, &o.bias_at_rj, &o.bias_opening_proof);
    Ok(())
}
fn read_proj_openings<R: Read>(r: &mut R) -> io::Result<ProjectionOpenings> {
    let y_eval = read_f(r)?;
    let x_eval = read_f(r)?;
    let (w_eval, w_open) = read_ep!(r);
    let (bias_at_rj, bias_opening_proof) = read_ep!(r);
    Ok(ProjectionOpenings {
        y_eval,
        x_eval,
        w_eval,
        w_open,
        bias_at_rj,
        bias_opening_proof,
    })
}

fn write_proj_proof<W: Write>(w: &mut W, p: &ProjectionProof) -> io::Result<()> {
    write_sumcheck_proof(w, &p.sumcheck)?;
    write_proj_openings(w, &p.openings)
}
fn read_proj_proof<R: Read>(r: &mut R) -> io::Result<ProjectionProof> {
    Ok(ProjectionProof {
        sumcheck: read_sumcheck_proof(r)?,
        openings: read_proj_openings(r)?,
    })
}

// ---------------------------------------------------------------------------
// Attention proof
// ---------------------------------------------------------------------------

fn write_attn_internal_coms<W: Write>(
    w: &mut W,
    c: &AttentionInternalCommitments,
) -> io::Result<()> {
    write_hyrax_commitment(w, &c.phi_q_com)?;
    write_hyrax_commitment(w, &c.phi_k_com)
    //write_hyrax_commitment(w, &c.context_com)
}
fn read_attn_internal_coms<R: Read>(r: &mut R) -> io::Result<AttentionInternalCommitments> {
    Ok(AttentionInternalCommitments {
        phi_q_com: read_hyrax_commitment(r)?,
        phi_k_com: read_hyrax_commitment(r)?,
        //context_com: read_hyrax_commitment(r)?,
    })
}

fn write_attn_openings<W: Write>(w: &mut W, o: &AttentionOpenings) -> io::Result<()> {
    write_f(w, &o.out_eval)?;
    write_ep!(w, &o.phi_q_eval, &o.phi_q_open);
    write_ep!(w, &o.phi_k_eval, &o.phi_k_open);
    write_f(w, &o.v_eval)?;
    Ok(())
}
fn read_attn_openings<R: Read>(r: &mut R) -> io::Result<AttentionOpenings> {
    let out_eval = read_f(r)?;
    let (phi_q_eval, phi_q_open) = read_ep!(r);
    let (phi_k_eval, phi_k_open) = read_ep!(r);
    let v_eval = read_f(r)?;
    Ok(AttentionOpenings {
        out_eval,
        phi_q_eval,
        phi_q_open,
        phi_k_eval,
        phi_k_open,
        v_eval,
    })
}

fn write_attn_proof<W: Write>(w: &mut W, p: &LinearAttentionProof) -> io::Result<()> {
    write_attn_internal_coms(w, &p.internal_coms)?;
    match &p.sumcheck {
        AttentionSumcheckProof::Batched { proof, ctx_eval } => {
            w.write_all(&[0u8])?; // tag: Batched
            write_sumcheck_proof_multi(w, proof)?;
            write_f(w, ctx_eval)?;
        }
        AttentionSumcheckProof::Sequential {
            out_sumcheck,
            context_sumcheck,
        } => {
            w.write_all(&[1u8])?; // tag: Sequential
            write_sumcheck_proof(w, out_sumcheck)?;
            write_sumcheck_proof(w, context_sumcheck)?;
        }
    }
    write_attn_openings(w, &p.openings)
}
fn read_attn_proof<R: Read>(r: &mut R) -> io::Result<LinearAttentionProof> {
    let internal_coms = read_attn_internal_coms(r)?;
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;
    let sumcheck = match tag[0] {
        0 => AttentionSumcheckProof::Batched {
            proof: read_sumcheck_proof_multi(r)?,
            ctx_eval: read_f(r)?,
        },
        1 => AttentionSumcheckProof::Sequential {
            out_sumcheck: read_sumcheck_proof(r)?,
            context_sumcheck: read_sumcheck_proof(r)?,
        },
        t => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown AttentionSumcheckProof tag: {}", t),
            ))
        }
    };
    Ok(LinearAttentionProof {
        internal_coms,
        sumcheck,
        openings: read_attn_openings(r)?,
    })
}

// ---------------------------------------------------------------------------
// FFN proof
// ---------------------------------------------------------------------------

fn write_ffn_internal_coms<W: Write>(w: &mut W, c: &FFNInternalCommitments) -> io::Result<()> {
    write_hyrax_commitment(w, &c.m_com)?;
    write_hyrax_commitment(w, &c.a_com)
}
fn read_ffn_internal_coms<R: Read>(r: &mut R) -> io::Result<FFNInternalCommitments> {
    Ok(FFNInternalCommitments {
        m_com: read_hyrax_commitment(r)?,
        a_com: read_hyrax_commitment(r)?,
    })
}

fn write_ffn_openings<W: Write>(w: &mut W, o: &FFNOpenings) -> io::Result<()> {
    write_ep!(w, &o.y_eval, &o.y_open);
    write_ep!(w, &o.a_eval, &o.a_open);
    write_ep!(w, &o.w2_eval, &o.w2_open);
    write_ep!(w, &o.m_eval, &o.m_open);
    write_ep!(w, &o.x_eval, &o.x_open);
    write_ep!(w, &o.w1_eval, &o.w1_open);
    Ok(())
}
fn read_ffn_openings<R: Read>(r: &mut R) -> io::Result<FFNOpenings> {
    let (y_eval, y_open) = read_ep!(r);
    let (a_eval, a_open) = read_ep!(r);
    let (w2_eval, w2_open) = read_ep!(r);
    let (m_eval, m_open) = read_ep!(r);
    let (x_eval, x_open) = read_ep!(r);
    let (w1_eval, w1_open) = read_ep!(r);
    Ok(FFNOpenings {
        y_eval,
        y_open,
        a_eval,
        a_open,
        w2_eval,
        w2_open,
        m_eval,
        m_open,
        x_eval,
        x_open,
        w1_eval,
        w1_open,
    })
}

fn write_ffn_proof<W: Write>(w: &mut W, p: &FFNProof) -> io::Result<()> {
    write_ffn_internal_coms(w, &p.internal_coms)?;
    write_sumcheck_proof(w, &p.y_sumcheck)?;
    write_sumcheck_proof(w, &p.m_sumcheck)?;
    write_ffn_openings(w, &p.openings)
}
fn read_ffn_proof<R: Read>(r: &mut R) -> io::Result<FFNProof> {
    Ok(FFNProof {
        internal_coms: read_ffn_internal_coms(r)?,
        y_sumcheck: read_sumcheck_proof(r)?,
        m_sumcheck: read_sumcheck_proof(r)?,
        openings: read_ffn_openings(r)?,
    })
}

// ---------------------------------------------------------------------------
// Block proof
// ---------------------------------------------------------------------------
// Combine proof
// ---------------------------------------------------------------------------

fn write_combine_proof<W: Write>(w: &mut W, p: &CombineProof) -> io::Result<()> {
    write_sumcheck_proof(w, &p.sumcheck)?;
    write_hyrax_proof(w, &p.hyrax_proof)
}
fn read_combine_proof<R: Read>(r: &mut R) -> io::Result<CombineProof> {
    Ok(CombineProof {
        sumcheck: read_sumcheck_proof(r)?,
        hyrax_proof: read_hyrax_proof(r)?,
    })
}

// ---------------------------------------------------------------------------
// Block proof
// ---------------------------------------------------------------------------

fn write_block_proof<W: Write>(w: &mut W, p: &TransformerBlockProof) -> io::Result<()> {
    write_ln_proof(w, &p.ln1_proof)?;
    write_proj_proof(w, &p.q_proj_proof)?;
    write_proj_proof(w, &p.k_proj_proof)?;
    write_proj_proof(w, &p.v_proj_proof)?;
    write_attn_proof(w, &p.attn_proof)?;
    write_proj_proof(w, &p.o_proj_proof)?;
    write_ln_proof(w, &p.ln2_proof)?;
    write_ffn_proof(w, &p.ffn_proof)?;
    write_hyrax_commitment(w, &p.x_norm1_com)?;
    write_hyrax_commitment(w, &p.q_com)?;
    write_hyrax_commitment(w, &p.k_com)?;
    write_hyrax_commitment(w, &p.v_com)?;
    write_hyrax_commitment(w, &p.out_inner_com)?;
    write_hyrax_commitment(w, &p.out_attn_com)?;
    write_hyrax_commitment(w, &p.x_norm2_com)?;
    write_hyrax_commitment(w, &p.out_ffn_com)?;
    write_combine_proof(w, &p.q_combine)?;
    write_combine_proof(w, &p.k_combine)?;
    write_combine_proof(w, &p.v_combine)?;
    write_combine_proof(w, &p.out_inner_combine)?;
    write_combine_proof(w, &p.x_norm1_combine)?;
    write_combine_proof(w, &p.out_attn_combine)
}
fn read_block_proof<R: Read>(r: &mut R) -> io::Result<TransformerBlockProof> {
    Ok(TransformerBlockProof {
        ln1_proof: read_ln_proof(r)?,
        q_proj_proof: read_proj_proof(r)?,
        k_proj_proof: read_proj_proof(r)?,
        v_proj_proof: read_proj_proof(r)?,
        attn_proof: read_attn_proof(r)?,
        o_proj_proof: read_proj_proof(r)?,
        ln2_proof: read_ln_proof(r)?,
        ffn_proof: read_ffn_proof(r)?,
        x_norm1_com: read_hyrax_commitment(r)?,
        q_com: read_hyrax_commitment(r)?,
        k_com: read_hyrax_commitment(r)?,
        v_com: read_hyrax_commitment(r)?,
        out_inner_com: read_hyrax_commitment(r)?,
        out_attn_com: read_hyrax_commitment(r)?,
        x_norm2_com: read_hyrax_commitment(r)?,
        out_ffn_com: read_hyrax_commitment(r)?,
        q_combine: read_combine_proof(r)?,
        k_combine: read_combine_proof(r)?,
        v_combine: read_combine_proof(r)?,
        out_inner_combine: read_combine_proof(r)?,
        x_norm1_combine: read_combine_proof(r)?,
        out_attn_combine: read_combine_proof(r)?,
    })
}

fn write_model_proof<W: Write>(w: &mut W, p: &TransformerModelProof) -> io::Result<()> {
    write_hyrax_commitment(w, &p.x_in_com)?;
    write_vec(w, &p.block_proofs, write_block_proof)?;
    write_ln_proof(w, &p.final_ln_proof)?;
    write_proj_proof(w, &p.lm_head_proof)?;
    write_hyrax_commitment(w, &p.final_ln_out_com)?;
    write_hyrax_commitment(w, &p.logits_com)?;
    write_lasso_multi_proof(w, &p.all_lasso_proof)
}
fn read_model_proof<R: Read>(r: &mut R) -> io::Result<TransformerModelProof> {
    Ok(TransformerModelProof {
        x_in_com: read_hyrax_commitment(r)?,
        block_proofs: read_vec(r, read_block_proof)?,
        final_ln_proof: read_ln_proof(r)?,
        lm_head_proof: read_proj_proof(r)?,
        final_ln_out_com: read_hyrax_commitment(r)?,
        logits_com: read_hyrax_commitment(r)?,
        all_lasso_proof: read_lasso_multi_proof(r)?,
    })
}

// ---------------------------------------------------------------------------
// Public instances (bundled with proof)
// ---------------------------------------------------------------------------

fn write_lasso_instance<W: Write>(w: &mut W, inst: &LassoInstance) -> io::Result<()> {
    write_vec(w, &inst.tables, |w2, t| write_vec_f(w2, t))?;
    write_vec_usize(w, &inst.query_indices)?;
    write_vec_f(w, &inst.outputs)?;
    write_usize(w, inst.bits_per_chunk)
}
fn read_lasso_instance<R: Read>(r: &mut R) -> io::Result<LassoInstance> {
    Ok(LassoInstance {
        tables: read_vec(r, read_vec_f)?,
        query_indices: read_vec_usize(r)?,
        outputs: read_vec_f(r)?,
        bits_per_chunk: read_usize(r)?,
    })
}

fn write_attn_instance<W: Write>(w: &mut W, inst: &LinearAttentionInstance) -> io::Result<()> {
    write_usize(w, inst.seq_len)?;
    write_usize(w, inst.d_head)?;
    write_lasso_instance(w, &inst.q_lasso)?;
    write_lasso_instance(w, &inst.k_lasso)
}
fn read_attn_instance<R: Read>(r: &mut R) -> io::Result<LinearAttentionInstance> {
    Ok(LinearAttentionInstance {
        seq_len: read_usize(r)?,
        d_head: read_usize(r)?,
        q_lasso: read_lasso_instance(r)?,
        k_lasso: read_lasso_instance(r)?,
    })
}

fn write_ffn_instance<W: Write>(w: &mut W, inst: &FFNInstance) -> io::Result<()> {
    write_lasso_instance(w, &inst.activation_lasso)
}
fn read_ffn_instance<R: Read>(r: &mut R) -> io::Result<FFNInstance> {
    Ok(FFNInstance {
        activation_lasso: read_lasso_instance(r)?,
    })
}

// ---------------------------------------------------------------------------
// Verifying key types
// ---------------------------------------------------------------------------

fn write_ln_vk<W: Write>(w: &mut W, vk: &LayerNormVerifyingKey) -> io::Result<()> {
    write_usize(w, vk.seq_len)?;
    write_usize(w, vk.d_head)?;
    write_vec_f(w, &vk.gamma)?;
    write_vec_f(w, &vk.beta)?;
    write_f(w, &vk.scale_gamma)?;
    write_f(w, &vk.scale_beta)
}
fn read_ln_vk<R: Read>(r: &mut R) -> io::Result<LayerNormVerifyingKey> {
    Ok(LayerNormVerifyingKey {
        seq_len: read_usize(r)?,
        d_head: read_usize(r)?,
        gamma: read_vec_f(r)?,
        beta: read_vec_f(r)?,
        scale_gamma: read_f(r)?,
        scale_beta: read_f(r)?,
    })
}

fn write_proj_vk<W: Write>(w: &mut W, vk: &ProjectionVerifyingKey) -> io::Result<()> {
    write_usize(w, vk.seq_len)?;
    write_usize(w, vk.d_in)?;
    write_usize(w, vk.d_out)?;
    write_hyrax_commitment(w, &vk.w_com)?;
    write_f(w, &vk.alpha)?;
    write_hyrax_commitment(w, &vk.bias_com)
}
fn read_proj_vk<R: Read>(r: &mut R) -> io::Result<ProjectionVerifyingKey> {
    Ok(ProjectionVerifyingKey {
        seq_len: read_usize(r)?,
        d_in: read_usize(r)?,
        d_out: read_usize(r)?,
        w_com: read_hyrax_commitment(r)?,
        alpha: read_f(r)?,
        bias_com: read_hyrax_commitment(r)?,
    })
}

fn write_proj_pk<W: Write>(w: &mut W, pk: &ProjectionProvingKey) -> io::Result<()> {
    write_proj_vk(w, &pk.vk)?;
    write_vec_vec_t(w, &pk.w)?;
    write_vec_f(w, &pk.bias)
}
fn read_proj_pk<R: Read>(r: &mut R) -> io::Result<ProjectionProvingKey> {
    Ok(ProjectionProvingKey {
        vk: read_proj_vk(r)?,
        w: read_vec_vec_t(r)?,
        bias: read_vec_f(r)?,
    })
}

fn write_lasso_vk<W: Write>(w: &mut W, vk: &LassoVerifyingKey) -> io::Result<()> {
    write_vec(w, &vk.table_coms, write_hyrax_commitment)
}
fn read_lasso_vk<R: Read>(r: &mut R) -> io::Result<LassoVerifyingKey> {
    Ok(LassoVerifyingKey {
        table_coms: read_vec(r, read_hyrax_commitment)?,
    })
}

fn write_lasso_pk<W: Write>(w: &mut W, pk: &LassoProvingKey) -> io::Result<()> {
    write_usize(w, pk.nu)?;
    write_vec(w, &pk.table_coms, write_hyrax_commitment)
}
fn read_lasso_pk<R: Read>(r: &mut R) -> io::Result<LassoProvingKey> {
    let nu = read_usize(r)?;
    Ok(LassoProvingKey {
        nu,
        table_coms: read_vec(r, read_hyrax_commitment)?,
    })
}

fn write_lasso_multi_vk<W: Write>(w: &mut W, vk: &LassoMultiVerifyingKey) -> io::Result<()> {
    write_vec(w, &vk.instance_table_coms, |w, v: &Vec<HyraxCommitment>| {
        write_vec(w, v, write_hyrax_commitment)
    })
}
fn read_lasso_multi_vk<R: Read>(r: &mut R) -> io::Result<LassoMultiVerifyingKey> {
    Ok(LassoMultiVerifyingKey {
        instance_table_coms: read_vec(r, |r| read_vec(r, read_hyrax_commitment))?,
    })
}

fn write_lasso_multi_pk<W: Write>(w: &mut W, pk: &LassoMultiProvingKey) -> io::Result<()> {
    write_usize(w, pk.nu)?;
    write_vec(w, &pk.instance_table_coms, |w, v: &Vec<HyraxCommitment>| {
        write_vec(w, v, write_hyrax_commitment)
    })
}
fn read_lasso_multi_pk<R: Read>(r: &mut R) -> io::Result<LassoMultiProvingKey> {
    let nu = read_usize(r)?;
    Ok(LassoMultiProvingKey {
        nu,
        instance_table_coms: read_vec(r, |r| read_vec(r, read_hyrax_commitment))?,
    })
}

fn write_attn_pk<W: Write>(w: &mut W, pk: &AttentionProvingKey) -> io::Result<()> {
    write_lasso_multi_pk(w, &pk.qk_lasso_pk)
}
fn read_attn_pk<R: Read>(r: &mut R) -> io::Result<AttentionProvingKey> {
    Ok(AttentionProvingKey {
        qk_lasso_pk: read_lasso_multi_pk(r)?,
    })
}

fn write_ffn_vk<W: Write>(w: &mut W, vk: &FFNVerifyingKey) -> io::Result<()> {
    write_usize(w, vk.seq_len)?;
    write_usize(w, vk.d_model)?;
    write_usize(w, vk.d_ff)?;
    write_hyrax_commitment(w, &vk.w1_com)?;
    write_hyrax_commitment(w, &vk.w2_com)?;
    write_lasso_vk(w, &vk.activation_lasso_vk)
}
fn read_ffn_vk<R: Read>(r: &mut R) -> io::Result<FFNVerifyingKey> {
    Ok(FFNVerifyingKey {
        seq_len: read_usize(r)?,
        d_model: read_usize(r)?,
        d_ff: read_usize(r)?,
        w1_com: read_hyrax_commitment(r)?,
        w2_com: read_hyrax_commitment(r)?,
        activation_lasso_vk: read_lasso_vk(r)?,
    })
}

fn write_ffn_pk<W: Write>(w: &mut W, pk: &FFNProvingKey) -> io::Result<()> {
    write_ffn_vk(w, &pk.vk)?;
    write_vec_vec_t(w, &pk.w1)?;
    write_vec_vec_t(w, &pk.w2)?;
    write_lasso_pk(w, &pk.activation_lasso_pk)
}
fn read_ffn_pk<R: Read>(r: &mut R) -> io::Result<FFNProvingKey> {
    Ok(FFNProvingKey {
        vk: read_ffn_vk(r)?,
        w1: read_vec_vec_t(r)?,
        w2: read_vec_vec_t(r)?,
        activation_lasso_pk: read_lasso_pk(r)?,
    })
}

// TransformerBlockVerifyingKey  (include_weights controls whether PK weight data is written)
fn write_block_vk<W: Write>(
    w: &mut W,
    bvk: &TransformerBlockVerifyingKey,
    include_weights: bool,
) -> io::Result<()> {
    write_usize(w, bvk.seq_len)?;
    write_usize(w, bvk.d_model)?;
    write_ln_vk(w, &bvk.ln1_vk)?;
    write_proj_vk(w, &bvk.q_vk)?;
    write_proj_vk(w, &bvk.k_vk)?;
    write_proj_vk(w, &bvk.v_vk)?;
    write_proj_vk(w, &bvk.o_vk)?;
    write_ln_vk(w, &bvk.ln2_vk)?;
    write_ffn_vk(w, &bvk.ffn_vk)?;
    // Always write attn_pk commitments: verifier needs them to replay transcript.
    write_attn_pk(w, &bvk.attn_pk)?;
    write_bool(w, include_weights)?;
    if include_weights {
        write_proj_pk(w, &bvk.q_pk)?;
        write_proj_pk(w, &bvk.k_pk)?;
        write_proj_pk(w, &bvk.v_pk)?;
        write_proj_pk(w, &bvk.o_pk)?;
        write_ffn_pk(w, &bvk.ffn_pk)?;
    }
    Ok(())
}
fn read_block_vk<R: Read>(r: &mut R) -> io::Result<TransformerBlockVerifyingKey> {
    let seq_len = read_usize(r)?;
    let d_model = read_usize(r)?;
    let ln1_vk = read_ln_vk(r)?;
    let q_vk = read_proj_vk(r)?;
    let k_vk = read_proj_vk(r)?;
    let v_vk = read_proj_vk(r)?;
    let o_vk = read_proj_vk(r)?;
    let ln2_vk = read_ln_vk(r)?;
    let ffn_vk = read_ffn_vk(r)?;
    // attn_pk commitments are always present (needed by verifier to replay transcript).
    let attn_pk = read_attn_pk(r)?;
    let has_weights = read_bool(r)?;
    let (q_pk, k_pk, v_pk, o_pk, ffn_pk) = if has_weights {
        (
            read_proj_pk(r)?,
            read_proj_pk(r)?,
            read_proj_pk(r)?,
            read_proj_pk(r)?,
            read_ffn_pk(r)?,
        )
    } else {
        // Stub PKs sufficient for verification (verifier never reads .w fields)
        let stub_proj = |vk: &ProjectionVerifyingKey| ProjectionProvingKey {
            vk: vk.clone(),
            w: vec![],
            bias: vec![],
        };
        let stub_ffn = |vk: &FFNVerifyingKey| FFNProvingKey {
            vk: vk.clone(),
            w1: vec![],
            w2: vec![],
            // activation_lasso_pk not needed for verification (verifier uses vk.activation_lasso_vk)
            activation_lasso_pk: LassoProvingKey {
                table_coms: vec![],
                nu: 0,
            },
        };
        (
            stub_proj(&q_vk),
            stub_proj(&k_vk),
            stub_proj(&v_vk),
            stub_proj(&o_vk),
            stub_ffn(&ffn_vk),
        )
    };
    Ok(TransformerBlockVerifyingKey {
        seq_len,
        d_model,
        ln1_vk,
        ln2_vk,
        q_vk,
        k_vk,
        v_vk,
        o_vk,
        ffn_vk,
        q_pk,
        k_pk,
        v_pk,
        o_pk,
        ffn_pk,
        attn_pk,
    })
}

// ---------------------------------------------------------------------------
// Top-level encode/decode (public API)
// ---------------------------------------------------------------------------

/// Encode the full proving key (includes raw weights).
pub fn encode_pk(pk: &TransformerModelProvingKey) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(PK_MAGIC);
    buf.push(VERSION);
    write_usize(&mut buf, pk.vk.num_blocks)?;
    write_usize(&mut buf, pk.vk.seq_len)?;
    write_usize(&mut buf, pk.vk.d_model)?;
    write_usize(&mut buf, pk.vk.vocab_size)?;
    write_ln_vk(&mut buf, &pk.vk.final_ln_vk)?;
    write_proj_vk(&mut buf, &pk.vk.lm_head_vk)?;
    for bpk in &pk.block_pks {
        write_block_vk(&mut buf, bpk, true)?;
    }
    write_proj_pk(&mut buf, &pk.lm_head_pk)?;
    Ok(buf)
}

/// Decode a proving key from bytes.
pub fn decode_pk(bytes: &[u8]) -> io::Result<TransformerModelProvingKey> {
    let mut r = bytes;
    check_magic(&mut r, PK_MAGIC)?;
    let _version = {
        let mut v = [0u8; 1];
        r.read_exact(&mut v)?;
        v[0]
    };
    let num_blocks = read_usize(&mut r)?;
    let seq_len = read_usize(&mut r)?;
    let d_model = read_usize(&mut r)?;
    let vocab_size = read_usize(&mut r)?;
    let final_ln_vk = read_ln_vk(&mut r)?;
    let lm_head_vk = read_proj_vk(&mut r)?;
    let mut block_pks = Vec::with_capacity(num_blocks);
    let mut block_vks = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let bvk = read_block_vk(&mut r)?;
        block_vks.push(bvk.clone());
        block_pks.push(bvk);
    }
    let lm_head_pk = read_proj_pk(&mut r)?;
    Ok(TransformerModelProvingKey {
        vk: TransformerModelVerifyingKey {
            num_blocks,
            seq_len,
            d_model,
            vocab_size,
            block_vks,
            final_ln_vk,
            lm_head_vk,
        },
        block_pks,
        lm_head_pk,
    })
}

/// Encode a slim verifying key (no raw weights, for distribution to verifiers).
pub fn encode_vk(vk: &TransformerModelVerifyingKey) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(VK_MAGIC);
    buf.push(VERSION);
    write_usize(&mut buf, vk.num_blocks)?;
    write_usize(&mut buf, vk.seq_len)?;
    write_usize(&mut buf, vk.d_model)?;
    write_usize(&mut buf, vk.vocab_size)?;
    write_ln_vk(&mut buf, &vk.final_ln_vk)?;
    write_proj_vk(&mut buf, &vk.lm_head_vk)?;
    for bvk in &vk.block_vks {
        write_block_vk(&mut buf, bvk, false)?;
    }
    Ok(buf)
}

/// Decode a slim verifying key (stubs out PK weight fields).
pub fn decode_vk(bytes: &[u8]) -> io::Result<TransformerModelVerifyingKey> {
    let mut r = bytes;
    check_magic(&mut r, VK_MAGIC)?;
    let _version = {
        let mut v = [0u8; 1];
        r.read_exact(&mut v)?;
        v[0]
    };
    let num_blocks = read_usize(&mut r)?;
    let seq_len = read_usize(&mut r)?;
    let d_model = read_usize(&mut r)?;
    let vocab_size = read_usize(&mut r)?;
    let final_ln_vk = read_ln_vk(&mut r)?;
    let lm_head_vk = read_proj_vk(&mut r)?;
    let mut block_vks = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        block_vks.push(read_block_vk(&mut r)?);
    }
    Ok(TransformerModelVerifyingKey {
        num_blocks,
        seq_len,
        d_model,
        vocab_size,
        block_vks,
        final_ln_vk,
        lm_head_vk,
    })
}

/// Encode a proof bundle (proof + public instances + lasso sigma).
pub fn encode_proof_bundle(
    proof: &TransformerModelProof,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
    lasso_sigma: usize,
) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.extend_from_slice(PROOF_MAGIC);
    buf.push(VERSION);
    write_model_proof(&mut buf, proof)?;
    write_attn_instance(&mut buf, inst_attn)?;
    write_ffn_instance(&mut buf, inst_ffn)?;
    write_usize(&mut buf, lasso_sigma)?;
    Ok(buf)
}

/// Decode a proof bundle.
pub fn decode_proof_bundle(
    bytes: &[u8],
) -> io::Result<(
    TransformerModelProof,
    LinearAttentionInstance,
    FFNInstance,
    usize,
)> {
    let mut r = bytes;
    check_magic(&mut r, PROOF_MAGIC)?;
    let _version = {
        let mut v = [0u8; 1];
        r.read_exact(&mut v)?;
        v[0]
    };
    let proof = read_model_proof(&mut r)?;
    let inst_attn = read_attn_instance(&mut r)?;
    let inst_ffn = read_ffn_instance(&mut r)?;
    let lasso_sigma = read_usize(&mut r)?;
    Ok((proof, inst_attn, inst_ffn, lasso_sigma))
}

fn check_magic<R: Read>(r: &mut R, expected: &[u8; 8]) -> io::Result<()> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "bad magic: expected {:?}, got {:?}",
                std::str::from_utf8(expected),
                std::str::from_utf8(&magic)
            ),
        ));
    }
    Ok(())
}
