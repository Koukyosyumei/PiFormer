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
            AttentionOpenings, AttentionProvingKey,
            AttentionSumcheckProof, LinearAttentionInstance,
            LinearAttentionProof,
        },
        layernorm::{
            LayerNormInternalCommitments, LayerNormOpenings, LayerNormProof, LayerNormVerifyingKey,
        },
        projection::{
            BatchedQKVProjectionOpenings, BatchedQKVProjectionProof, ProjectionOpenings,
            ProjectionProof, ProjectionProvingKey, ProjectionVerifyingKey,
        },
    },
    ffn::ffn::{
        FFNInstance, FFNOpenings, FFNProof, FFNProvingKey, FFNVerifyingKey,
    },
    lookup::{
        lasso::{
            LassoInstance, LassoMultiProof, LassoMultiProvingKey, LassoMultiVerifyingKey,
            LassoProof, LassoProvingKey, LassoVerifyingKey,
        },
        range::{GlobalRangeM, LogUpWitnessProof, RangeProof, RangeWitnessProof},
    },
    pcs::{HyraxCommitment, HyraxProof},
    poly::utils::TernaryValue,
    prover::{
        TransformerBlockProof, TransformerModelProof, TransformerModelProvingKey,
        TransformerModelVerifyingKey,
    },
    subprotocols::{
        sumcheck::{
            CubicRoundPoly, RoundPoly, SumcheckCubicProof, SumcheckCubicProofMulti, SumcheckProof,
            SumcheckProofMulti,
        },
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
    write_vec(w, &p.hyrax_proofs, write_hyrax_proof)?;
    write_vec(w, &p.l_k_coms, write_hyrax_commitment)?;
    write_vec_f(w, &p.l_k_evals)?;
    write_vec(w, &p.l_k_opens, write_hyrax_proof)
}
fn read_lasso_proof<R: Read>(r: &mut R) -> io::Result<LassoProof> {
    Ok(LassoProof {
        sub_claims: read_vec_f(r)?,
        sumcheck_proofs: read_vec(r, read_sumcheck_proof)?,
        table_openings: read_vec_f(r)?,
        hyrax_proofs: read_vec(r, read_hyrax_proof)?,
        l_k_coms: read_vec(r, read_hyrax_commitment)?,
        l_k_evals: read_vec_f(r)?,
        l_k_opens: read_vec(r, read_hyrax_proof)?,
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

fn write_cubic_round_poly<W: Write>(w: &mut W, rp: &CubicRoundPoly) -> io::Result<()> {
    write_f(w, &rp.evals[0])?;
    write_f(w, &rp.evals[1])?;
    write_f(w, &rp.evals[2])?;
    write_f(w, &rp.evals[3])
}
fn read_cubic_round_poly<R: Read>(r: &mut R) -> io::Result<CubicRoundPoly> {
    Ok(CubicRoundPoly {
        evals: [read_f(r)?, read_f(r)?, read_f(r)?, read_f(r)?],
    })
}

fn write_sumcheck_cubic_proof<W: Write>(w: &mut W, p: &SumcheckCubicProof) -> io::Result<()> {
    write_vec(w, &p.round_polys, write_cubic_round_poly)?;
    write_f(w, &p.final_eval_f)?;
    write_f(w, &p.final_eval_g)?;
    write_f(w, &p.final_eval_h)
}
fn read_sumcheck_cubic_proof<R: Read>(r: &mut R) -> io::Result<SumcheckCubicProof> {
    Ok(SumcheckCubicProof {
        round_polys: read_vec(r, read_cubic_round_poly)?,
        final_eval_f: read_f(r)?,
        final_eval_g: read_f(r)?,
        final_eval_h: read_f(r)?,
    })
}

fn write_sumcheck_cubic_proof_multi<W: Write>(
    w: &mut W,
    p: &SumcheckCubicProofMulti,
) -> io::Result<()> {
    write_vec(w, &p.round_polys, write_cubic_round_poly)?;
    write_vec_f(w, &p.final_evals_f)?;
    write_vec_f(w, &p.final_evals_g)?;
    write_vec_f(w, &p.final_evals_h)
}
fn read_sumcheck_cubic_proof_multi<R: Read>(r: &mut R) -> io::Result<SumcheckCubicProofMulti> {
    Ok(SumcheckCubicProofMulti {
        round_polys: read_vec(r, read_cubic_round_poly)?,
        final_evals_f: read_vec_f(r)?,
        final_evals_g: read_vec_f(r)?,
        final_evals_h: read_vec_f(r)?,
    })
}

fn write_lasso_multi_proof<W: Write>(w: &mut W, p: &LassoMultiProof) -> io::Result<()> {
    write_f(w, &p.combined_grand_sum)?;
    write_sumcheck_proof_multi(w, &p.combined_sumcheck_proof)?;
    write_vec(w, &p.table_openings, |w, v: &Vec<F>| write_vec_f(w, v))?;
    write_hyrax_proof(w, &p.hyrax_proof)?;
    write_vec(w, &p.output_opening_proofs, write_hyrax_proof)?;
    write_vec(w, &p.l_k_coms_multi, |w, v: &Vec<HyraxCommitment>| {
        write_vec(w, v, write_hyrax_commitment)
    })?;
    write_vec(w, &p.l_k_evals_multi, |w, v: &Vec<F>| write_vec_f(w, v))?;
    write_vec(w, &p.l_k_opens_multi, |w, v: &Vec<HyraxProof>| {
        write_vec(w, v, write_hyrax_proof)
    })
}
fn read_lasso_multi_proof<R: Read>(r: &mut R) -> io::Result<LassoMultiProof> {
    Ok(LassoMultiProof {
        combined_grand_sum: read_f(r)?,
        combined_sumcheck_proof: read_sumcheck_proof_multi(r)?,
        table_openings: read_vec(r, |r| read_vec_f(r))?,
        hyrax_proof: read_hyrax_proof(r)?,
        output_opening_proofs: read_vec(r, read_hyrax_proof)?,
        l_k_coms_multi: read_vec(r, |r| read_vec(r, read_hyrax_commitment))?,
        l_k_evals_multi: read_vec(r, |r| read_vec_f(r))?,
        l_k_opens_multi: read_vec(r, |r| read_vec(r, read_hyrax_proof))?,
    })
}

fn write_range_proof<W: Write>(w: &mut W, p: &RangeProof) -> io::Result<()> {
    write_sumcheck_proof(w, &p.sumcheck)?;
    write_f(w, &p.claim_v)?;
    write_vec(w, &p.chunk_coms, write_hyrax_commitment)?;
    write_vec_f(w, &p.chunk_evals)?;
    write_hyrax_proof(w, &p.chunk_batch_proof)?;
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
        chunk_batch_proof: read_hyrax_proof(r)?,
        m_com: read_hyrax_commitment(r)?,
        m_eval: read_f(r)?,
        m_open: read_hyrax_proof(r)?,
    })
}

fn write_logup_witness_proof<W: Write>(w: &mut W, p: &LogUpWitnessProof) -> io::Result<()> {
    write_vec(w, &p.h_coms, write_hyrax_commitment)?;
    write_vec(w, &p.combined_sumchecks, write_sumcheck_proof)?;
    write_vec_f(w, &p.combined_claims)?;
    write_vec_f(w, &p.h_at_rk)?;
    write_vec_f(w, &p.chunk_at_rk)?;
    write_vec(w, &p.h_open_proofs, write_hyrax_proof)?;
    write_vec(w, &p.chunk_open_proofs, write_hyrax_proof)
}
fn read_logup_witness_proof<R: Read>(r: &mut R) -> io::Result<LogUpWitnessProof> {
    Ok(LogUpWitnessProof {
        h_coms: read_vec(r, read_hyrax_commitment)?,
        combined_sumchecks: read_vec(r, read_sumcheck_proof)?,
        combined_claims: read_vec_f(r)?,
        h_at_rk: read_vec_f(r)?,
        chunk_at_rk: read_vec_f(r)?,
        h_open_proofs: read_vec(r, read_hyrax_proof)?,
        chunk_open_proofs: read_vec(r, read_hyrax_proof)?,
    })
}

fn write_range_witness_proof<W: Write>(w: &mut W, p: &RangeWitnessProof) -> io::Result<()> {
    write_sumcheck_proof(w, &p.sumcheck)?;
    write_f(w, &p.claim_v)?;
    write_vec(w, &p.chunk_coms, write_hyrax_commitment)?;
    write_vec_f(w, &p.chunk_evals)?;
    write_hyrax_proof(w, &p.chunk_batch_proof)?;
    write_logup_witness_proof(w, &p.logup)
}
fn read_range_witness_proof<R: Read>(r: &mut R) -> io::Result<RangeWitnessProof> {
    Ok(RangeWitnessProof {
        sumcheck: read_sumcheck_proof(r)?,
        claim_v: read_f(r)?,
        chunk_coms: read_vec(r, read_hyrax_commitment)?,
        chunk_evals: read_vec_f(r)?,
        chunk_batch_proof: read_hyrax_proof(r)?,
        logup: read_logup_witness_proof(r)?,
    })
}

fn write_global_range_m<W: Write>(w: &mut W, m: &GlobalRangeM) -> io::Result<()> {
    write_hyrax_commitment(w, &m.m_com)?;
    write_f(w, &m.m_eval)?;
    write_hyrax_proof(w, &m.m_open)?;
    write_sumcheck_proof(w, &m.logup_rhs_sumcheck)?;
    write_f(w, &m.logup_rhs_claim)?;
    write_f(w, &m.logup_m_at_rm2)?;
    write_hyrax_proof(w, &m.logup_m_open_rm2)
}
fn read_global_range_m<R: Read>(r: &mut R) -> io::Result<GlobalRangeM> {
    Ok(GlobalRangeM {
        m_com: read_hyrax_commitment(r)?,
        m_eval: read_f(r)?,
        m_open: read_hyrax_proof(r)?,
        logup_rhs_sumcheck: read_sumcheck_proof(r)?,
        logup_rhs_claim: read_f(r)?,
        logup_m_at_rm2: read_f(r)?,
        logup_m_open_rm2: read_hyrax_proof(r)?,
    })
}

// ---------------------------------------------------------------------------
// LayerNorm proof
// ---------------------------------------------------------------------------

fn write_ln_internal_coms<W: Write>(w: &mut W, c: &LayerNormInternalCommitments) -> io::Result<()> {
    write_hyrax_commitment(w, &c.sum_x_com)?;
    write_hyrax_commitment(w, &c.sigma_com)?;
    write_hyrax_commitment(w, &c.sq_sum_x_com)?;
    let has_sy = c.sigma_y_com.is_some();
    w.write_all(&[has_sy as u8])?;
    if let Some(ref sy) = c.sigma_y_com { write_hyrax_commitment(w, sy)?; }
    Ok(())
}
fn read_ln_internal_coms<R: Read>(r: &mut R) -> io::Result<LayerNormInternalCommitments> {
    let sum_x_com = read_hyrax_commitment(r)?;
    let sigma_com = read_hyrax_commitment(r)?;
    let sq_sum_x_com = read_hyrax_commitment(r)?;
    let mut flag = [0u8; 1];
    r.read_exact(&mut flag)?;
    let sigma_y_com = if flag[0] != 0 { Some(read_hyrax_commitment(r)?) } else { None };
    Ok(LayerNormInternalCommitments { sum_x_com, sigma_com, sq_sum_x_com, sigma_y_com })
}

fn write_ln_openings<W: Write>(w: &mut W, o: &LayerNormOpenings) -> io::Result<()> {
    // Group 1: at r_t
    write_f(w, &o.sum_x_at_rt)?;
    write_f(w, &o.sq_sum_x_at_rt)?;
    write_hyrax_proof(w, &o.rt_batch_proof)?;
    // Individual: x at combine(r_t, r_d_mean)
    write_ep!(w, &o.x_at_rt_rmean, &o.x_rt_rmean_proof);
    // Individual: x at r_final_q
    write_ep!(w, &o.x_at_r_final_q, &o.x_at_r_final_q_proof);
    // Group 2: at r_sig_t
    write_f(w, &o.sq_sum_x_at_rsig)?;
    write_f(w, &o.sigma_at_rsig)?;
    write_f(w, &o.sigma_sq_at_rsig)?;
    write_f(w, &o.sum_x_sq_at_rsig)?;
    write_hyrax_proof(w, &o.rsig_batch_proof)?;
    // Group 3: at combine(r_y_t, r_y_d)
    write_f(w, &o.x_at_ry)?;
    // y_at_ry: None in GKR mode, Some in conventional mode
    let has_y_at_ry = o.y_at_ry.is_some();
    w.write_all(&[has_y_at_ry as u8])?;
    if let Some(ref v) = o.y_at_ry { write_f(w, v)?; }
    write_f(w, &o.gamma_x_at_ry)?;
    write_f(w, &o.sigma_y_at_ry)?;
    write_hyrax_proof(w, &o.ry_td_batch_proof)?;
    // Group 4: at r_y_t
    write_f(w, &o.sum_x_at_ryt)?;
    write_f(w, &o.sigma_at_ryt)?;
    write_hyrax_proof(w, &o.ryt_batch_proof)?;
    // Group 5: at r_f_gx
    write_ep!(w, &o.x_at_rf_gx, &o.x_at_rf_gx_proof);
    // Group 6: at r_f_sy
    // Conventional mode: y_at_rf_sy/proof present; GKR mode: sigma_y_at_rf + sum_x present
    let has_y_at_rf_sy = o.y_at_rf_sy.is_some();
    w.write_all(&[has_y_at_rf_sy as u8])?;
    if has_y_at_rf_sy {
        write_ep!(w, o.y_at_rf_sy.as_ref().unwrap(), o.y_at_rf_sy_proof.as_ref().unwrap());
    } else {
        write_ep!(w, o.sigma_y_at_rf.as_ref().unwrap(), o.sigma_y_at_rf_proof.as_ref().unwrap());
        write_ep!(w, o.sum_x_at_rf_sy_t.as_ref().unwrap(), o.sum_x_at_rf_sy_t_proof.as_ref().unwrap());
    }
    write_ep!(w, &o.sigma_at_rf_sy_t, &o.sigma_at_rf_sy_t_proof);
    write_ep!(w, &o.sum_x_at_rf_sig, &o.sum_x_at_rf_sig_proof);
    // sigma_sq binding
    write_ep!(w, &o.sigma_at_rf_sigma_sq, &o.sigma_at_rf_sigma_sq_proof);
    Ok(())
}
fn read_ln_openings<R: Read>(r: &mut R) -> io::Result<LayerNormOpenings> {
    // Group 1: at r_t
    let sum_x_at_rt = read_f(r)?;
    let sq_sum_x_at_rt = read_f(r)?;
    let rt_batch_proof = read_hyrax_proof(r)?;
    // Individual: x at combine(r_t, r_d_mean)
    let (x_at_rt_rmean, x_rt_rmean_proof) = read_ep!(r);
    // Individual: x at r_final_q
    let (x_at_r_final_q, x_at_r_final_q_proof) = read_ep!(r);
    // Group 2: at r_sig_t
    let sq_sum_x_at_rsig = read_f(r)?;
    let sigma_at_rsig = read_f(r)?;
    let sigma_sq_at_rsig = read_f(r)?;
    let sum_x_sq_at_rsig = read_f(r)?;
    let rsig_batch_proof = read_hyrax_proof(r)?;
    // Group 3: at combine(r_y_t, r_y_d)
    let x_at_ry = read_f(r)?;
    let mut flag = [0u8; 1];
    r.read_exact(&mut flag)?;
    let y_at_ry = if flag[0] != 0 { Some(read_f(r)?) } else { None };
    let gamma_x_at_ry = read_f(r)?;
    let sigma_y_at_ry = read_f(r)?;
    let ry_td_batch_proof = read_hyrax_proof(r)?;
    // Group 4: at r_y_t
    let sum_x_at_ryt = read_f(r)?;
    let sigma_at_ryt = read_f(r)?;
    let ryt_batch_proof = read_hyrax_proof(r)?;
    // Group 5: at r_f_gx
    let (x_at_rf_gx, x_at_rf_gx_proof) = read_ep!(r);
    // Group 6: at r_f_sy
    r.read_exact(&mut flag)?;
    let (y_at_rf_sy, y_at_rf_sy_proof, sigma_y_at_rf, sigma_y_at_rf_proof,
         sum_x_at_rf_sy_t, sum_x_at_rf_sy_t_proof) = if flag[0] != 0 {
        let (v, p) = read_ep!(r);
        (Some(v), Some(p), None, None, None, None)
    } else {
        let (sy_v, sy_p) = read_ep!(r);
        let (sx_v, sx_p) = read_ep!(r);
        (None, None, Some(sy_v), Some(sy_p), Some(sx_v), Some(sx_p))
    };
    let (sigma_at_rf_sy_t, sigma_at_rf_sy_t_proof) = read_ep!(r);
    let (sum_x_at_rf_sig, sum_x_at_rf_sig_proof) = read_ep!(r);
    let (sigma_at_rf_sigma_sq, sigma_at_rf_sigma_sq_proof) = read_ep!(r);
    Ok(LayerNormOpenings {
        sum_x_at_rt,
        sq_sum_x_at_rt,
        rt_batch_proof,
        x_at_rt_rmean,
        x_rt_rmean_proof,
        x_at_r_final_q,
        x_at_r_final_q_proof,
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
        x_at_rf_gx,
        x_at_rf_gx_proof,
        y_at_rf_sy,
        y_at_rf_sy_proof,
        sigma_y_at_rf,
        sigma_y_at_rf_proof,
        sum_x_at_rf_sy_t,
        sum_x_at_rf_sy_t_proof,
        sigma_at_rf_sy_t,
        sigma_at_rf_sy_t_proof,
        sum_x_at_rf_sig,
        sum_x_at_rf_sig_proof,
        sigma_at_rf_sigma_sq,
        sigma_at_rf_sigma_sq_proof,
    })
}

fn write_ln_proof<W: Write>(w: &mut W, p: &LayerNormProof) -> io::Result<()> {
    write_ln_internal_coms(w, &p.internal_coms)?;
    write_sumcheck_proof(w, &p.mean_sumcheck)?;
    write_sumcheck_cubic_proof(w, &p.sq_sum_sumcheck)?;
    write_sumcheck_cubic_proof(w, &p.sum_x_sq_sumcheck)?;
    write_sumcheck_cubic_proof(w, &p.sigma_sq_sumcheck)?;
    write_sumcheck_cubic_proof_multi(w, &p.gamma_sigma_sumcheck)?;
    write_range_witness_proof(w, &p.sigma_range_proof)?;
    write_range_witness_proof(w, &p.y_range_proof)?;
    write_ln_openings(w, &p.openings)
}
fn read_ln_proof<R: Read>(r: &mut R) -> io::Result<LayerNormProof> {
    Ok(LayerNormProof {
        internal_coms: read_ln_internal_coms(r)?,
        mean_sumcheck: read_sumcheck_proof(r)?,
        sq_sum_sumcheck: read_sumcheck_cubic_proof(r)?,
        sum_x_sq_sumcheck: read_sumcheck_cubic_proof(r)?,
        sigma_sq_sumcheck: read_sumcheck_cubic_proof(r)?,
        gamma_sigma_sumcheck: read_sumcheck_cubic_proof_multi(r)?,
        sigma_range_proof: read_range_witness_proof(r)?,
        y_range_proof: read_range_witness_proof(r)?,
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

fn write_batched_qkv_openings<W: Write>(
    w: &mut W,
    o: &BatchedQKVProjectionOpenings,
) -> io::Result<()> {
    write_f(w, &o.q_eval)?;
    write_f(w, &o.k_eval)?;
    write_f(w, &o.v_eval)?;
    write_f(w, &o.x_eval)?;
    write_ep!(w, &o.w_q_eval, &o.w_q_open);
    write_ep!(w, &o.w_k_eval, &o.w_k_open);
    write_ep!(w, &o.w_v_eval, &o.w_v_open);
    write_ep!(w, &o.bias_q_eval, &o.bias_q_open);
    write_ep!(w, &o.bias_k_eval, &o.bias_k_open);
    write_ep!(w, &o.bias_v_eval, &o.bias_v_open);
    Ok(())
}
fn read_batched_qkv_openings<R: Read>(r: &mut R) -> io::Result<BatchedQKVProjectionOpenings> {
    let q_eval = read_f(r)?;
    let k_eval = read_f(r)?;
    let v_eval = read_f(r)?;
    let x_eval = read_f(r)?;
    let (w_q_eval, w_q_open) = read_ep!(r);
    let (w_k_eval, w_k_open) = read_ep!(r);
    let (w_v_eval, w_v_open) = read_ep!(r);
    let (bias_q_eval, bias_q_open) = read_ep!(r);
    let (bias_k_eval, bias_k_open) = read_ep!(r);
    let (bias_v_eval, bias_v_open) = read_ep!(r);
    Ok(BatchedQKVProjectionOpenings {
        q_eval,
        k_eval,
        v_eval,
        x_eval,
        w_q_eval,
        w_k_eval,
        w_v_eval,
        bias_q_eval,
        bias_k_eval,
        bias_v_eval,
        w_q_open,
        w_k_open,
        w_v_open,
        bias_q_open,
        bias_k_open,
        bias_v_open,
    })
}
fn write_batched_qkv_proof<W: Write>(
    w: &mut W,
    p: &BatchedQKVProjectionProof,
) -> io::Result<()> {
    write_sumcheck_proof(w, &p.sumcheck)?;
    write_batched_qkv_openings(w, &p.openings)
}
fn read_batched_qkv_proof<R: Read>(r: &mut R) -> io::Result<BatchedQKVProjectionProof> {
    Ok(BatchedQKVProjectionProof {
        sumcheck: read_sumcheck_proof(r)?,
        openings: read_batched_qkv_openings(r)?,
    })
}

// ---------------------------------------------------------------------------
// Attention proof
// ---------------------------------------------------------------------------

fn write_attn_openings<W: Write>(w: &mut W, o: &AttentionOpenings) -> io::Result<()> {
    write_f(w, &o.out_eval)?;
    write_f(w, &o.phi_q_eval)?;
    write_f(w, &o.phi_k_eval)?;
    write_f(w, &o.v_eval)?;
    Ok(())
}
fn read_attn_openings<R: Read>(r: &mut R) -> io::Result<AttentionOpenings> {
    Ok(AttentionOpenings {
        out_eval: read_f(r)?,
        phi_q_eval: read_f(r)?,
        phi_k_eval: read_f(r)?,
        v_eval: read_f(r)?,
    })
}

fn write_attn_proof<W: Write>(w: &mut W, p: &LinearAttentionProof) -> io::Result<()> {
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
    write_attn_openings(w, &p.openings)?;
    write_hyrax_commitment(w, &p.phi_q_com)?;
    write_hyrax_commitment(w, &p.phi_k_com)?;
    write_hyrax_proof(w, &p.phi_q_open)?;
    write_hyrax_proof(w, &p.phi_k_open)
}
fn read_attn_proof<R: Read>(r: &mut R) -> io::Result<LinearAttentionProof> {
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
        sumcheck,
        openings: read_attn_openings(r)?,
        phi_q_com: read_hyrax_commitment(r)?,
        phi_k_com: read_hyrax_commitment(r)?,
        phi_q_open: read_hyrax_proof(r)?,
        phi_k_open: read_hyrax_proof(r)?,
    })
}

// ---------------------------------------------------------------------------
// FFN proof
// ---------------------------------------------------------------------------

fn write_ffn_openings<W: Write>(w: &mut W, o: &FFNOpenings) -> io::Result<()> {
    write_f(w, &o.y_eval)?;
    write_f(w, &o.x_eval)?;
    write_f(w, &o.a_eval)?;
    write_ep!(w, &o.w2_eval, &o.w2_open);
    write_f(w, &o.m_eval)?;
    write_ep!(w, &o.w1_eval, &o.w1_open);
    Ok(())
}
fn read_ffn_openings<R: Read>(r: &mut R) -> io::Result<FFNOpenings> {
    let y_eval = read_f(r)?;
    let x_eval = read_f(r)?;
    let a_eval = read_f(r)?;
    let (w2_eval, w2_open) = read_ep!(r);
    let m_eval = read_f(r)?;
    let (w1_eval, w1_open) = read_ep!(r);
    Ok(FFNOpenings {
        y_eval,
        x_eval,
        a_eval,
        w2_eval,
        w2_open,
        m_eval,
        w1_eval,
        w1_open,
    })
}

fn write_ffn_proof<W: Write>(w: &mut W, p: &FFNProof) -> io::Result<()> {
    write_lasso_proof(w, &p.activation_lasso_proof)?;
    write_hyrax_commitment(w, &p.m_com)?;
    write_sumcheck_proof(w, &p.y_sumcheck)?;
    write_sumcheck_proof(w, &p.m_sumcheck)?;
    write_ffn_openings(w, &p.openings)?;
    write_hyrax_proof(w, &p.m_open)
}
fn read_ffn_proof<R: Read>(r: &mut R) -> io::Result<FFNProof> {
    Ok(FFNProof {
        activation_lasso_proof: read_lasso_proof(r)?,
        m_com: read_hyrax_commitment(r)?,
        y_sumcheck: read_sumcheck_proof(r)?,
        m_sumcheck: read_sumcheck_proof(r)?,
        openings: read_ffn_openings(r)?,
        m_open: read_hyrax_proof(r)?,
    })
}

// ---------------------------------------------------------------------------
// Block proof
// ---------------------------------------------------------------------------
// Combine proof
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Block proof
// ---------------------------------------------------------------------------

fn write_block_proof<W: Write>(w: &mut W, p: &TransformerBlockProof) -> io::Result<()> {
    write_ln_proof(w, &p.ln1_proof)?;
    write_ln_proof(w, &p.ln2_proof)?;
    write_global_range_m(w, &p.block_range_m)?;
    // FFN per-block
    write_lasso_proof(w, &p.ffn_lasso_proof)?;
    write_hyrax_commitment(w, &p.ffn_m_com)?;
    // Committed intermediate matrices
    write_hyrax_commitment(w, &p.x_norm1_com)?;
    write_hyrax_commitment(w, &p.q_com)?;
    write_hyrax_commitment(w, &p.k_com)?;
    write_hyrax_commitment(w, &p.v_com)?;
    write_hyrax_commitment(w, &p.out_attn_com)?;
    write_hyrax_commitment(w, &p.x_norm2_com)?;
    write_hyrax_commitment(w, &p.out_ffn_com)?;
    // Scalar evals at shared r_td
    write_f(w, &p.q_eval)?;
    write_f(w, &p.k_eval)?;
    write_f(w, &p.v_eval_rtd)?;
    write_f(w, &p.out_attn_eval)?;
    write_f(w, &p.out_ffn_eval)?;
    // Per-block QKV batch scalars
    write_f(w, &p.qkv_lambda)?;
    write_f(w, &p.qkv_mu)?;
    write_f(w, &p.qkv_w_q_eval)?;
    write_f(w, &p.qkv_w_k_eval)?;
    write_f(w, &p.qkv_w_v_eval)?;
    write_f(w, &p.qkv_bias_q_eval)?;
    write_f(w, &p.qkv_bias_k_eval)?;
    write_f(w, &p.qkv_bias_v_eval)?;
    // Per-block O-proj batch scalars
    write_f(w, &p.oproj_w_o_eval)?;
    write_f(w, &p.oproj_bias_o_eval)?;
    // Per-block FFN-M scalar
    write_f(w, &p.ffn_m_eval)?;
    // Attention phi_q/phi_k commitments + scalars
    write_hyrax_commitment(w, &p.attn_phi_q_com)?;
    write_hyrax_commitment(w, &p.attn_phi_k_com)?;
    write_f(w, &p.attn_out_eval)?;
    write_f(w, &p.attn_phi_q_eval)?;
    write_f(w, &p.attn_phi_k_eval)?;
    write_f(w, &p.attn_ctx_eval)?;
    write_f(w, &p.attn_v_eval)
}
fn read_block_proof<R: Read>(r: &mut R) -> io::Result<TransformerBlockProof> {
    Ok(TransformerBlockProof {
        ln1_proof: read_ln_proof(r)?,
        ln2_proof: read_ln_proof(r)?,
        block_range_m: read_global_range_m(r)?,
        // FFN per-block
        ffn_lasso_proof: read_lasso_proof(r)?,
        ffn_m_com: read_hyrax_commitment(r)?,
        // Committed intermediate matrices
        x_norm1_com: read_hyrax_commitment(r)?,
        q_com: read_hyrax_commitment(r)?,
        k_com: read_hyrax_commitment(r)?,
        v_com: read_hyrax_commitment(r)?,
        out_attn_com: read_hyrax_commitment(r)?,
        x_norm2_com: read_hyrax_commitment(r)?,
        out_ffn_com: read_hyrax_commitment(r)?,
        // Scalar evals at shared r_td
        q_eval: read_f(r)?,
        k_eval: read_f(r)?,
        v_eval_rtd: read_f(r)?,
        out_attn_eval: read_f(r)?,
        out_ffn_eval: read_f(r)?,
        // Per-block QKV batch scalars
        qkv_lambda: read_f(r)?,
        qkv_mu: read_f(r)?,
        qkv_w_q_eval: read_f(r)?,
        qkv_w_k_eval: read_f(r)?,
        qkv_w_v_eval: read_f(r)?,
        qkv_bias_q_eval: read_f(r)?,
        qkv_bias_k_eval: read_f(r)?,
        qkv_bias_v_eval: read_f(r)?,
        // Per-block O-proj batch scalars
        oproj_w_o_eval: read_f(r)?,
        oproj_bias_o_eval: read_f(r)?,
        // Per-block FFN-M scalar
        ffn_m_eval: read_f(r)?,
        // Attention phi_q/phi_k commitments + scalars
        attn_phi_q_com: read_hyrax_commitment(r)?,
        attn_phi_k_com: read_hyrax_commitment(r)?,
        attn_out_eval: read_f(r)?,
        attn_phi_q_eval: read_f(r)?,
        attn_phi_k_eval: read_f(r)?,
        attn_ctx_eval: read_f(r)?,
        attn_v_eval: read_f(r)?,
    })
}

fn write_model_proof<W: Write>(w: &mut W, p: &TransformerModelProof) -> io::Result<()> {
    write_hyrax_commitment(w, &p.x_in_com)?;
    write_vec(w, &p.block_proofs, write_block_proof)?;
    write_ln_proof(w, &p.final_ln_proof)?;
    write_proj_proof(w, &p.lm_head_proof)?;
    write_hyrax_commitment(w, &p.final_ln_out_com)?;
    write_hyrax_commitment(w, &p.logits_com)?;
    write_hyrax_proof(w, &p.lm_head_logits_open)?;
    write_lasso_multi_proof(w, &p.all_lasso_proof)?;
    write_global_range_m(w, &p.final_range_m)?;
    // Cross-block batch sumchecks
    write_sumcheck_proof_multi(w, &p.batch_qkv)?;
    write_sumcheck_proof_multi(w, &p.batch_oproj)?;
    write_sumcheck_proof_multi(w, &p.batch_ffn_y)?;
    write_sumcheck_proof_multi(w, &p.batch_ffn_m)?;
    write_sumcheck_proof_multi(w, &p.batch_attn_out)?;
    write_sumcheck_proof_multi(w, &p.batch_attn_ctx)?;
    // Global intermediate batch open
    write_hyrax_proof(w, &p.inter_batch_open)?;
    // 13 cross-block weight/activation batch opens
    write_hyrax_proof(w, &p.x_norm1_batch_open)?;
    write_hyrax_proof(w, &p.w_q_batch_open)?;
    write_hyrax_proof(w, &p.w_k_batch_open)?;
    write_hyrax_proof(w, &p.w_v_batch_open)?;
    write_hyrax_proof(w, &p.bias_q_batch_open)?;
    write_hyrax_proof(w, &p.bias_k_batch_open)?;
    write_hyrax_proof(w, &p.bias_v_batch_open)?;
    write_hyrax_proof(w, &p.w_o_batch_open)?;
    write_hyrax_proof(w, &p.bias_o_batch_open)?;
    write_hyrax_proof(w, &p.w2_batch_open)?;
    write_hyrax_proof(w, &p.w1_batch_open)?;
    write_hyrax_proof(w, &p.x_norm2_batch_open)?;
    write_hyrax_proof(w, &p.ffn_m_com_batch_open)?;
    // 3 attention batch opens
    write_hyrax_proof(w, &p.phi_q_batch_open)?;
    write_hyrax_proof(w, &p.phi_k_batch_open)?;
    write_hyrax_proof(w, &p.v_attn_batch_open)
}
fn read_model_proof<R: Read>(r: &mut R) -> io::Result<TransformerModelProof> {
    Ok(TransformerModelProof {
        x_in_com: read_hyrax_commitment(r)?,
        block_proofs: read_vec(r, read_block_proof)?,
        final_ln_proof: read_ln_proof(r)?,
        lm_head_proof: read_proj_proof(r)?,
        final_ln_out_com: read_hyrax_commitment(r)?,
        logits_com: read_hyrax_commitment(r)?,
        lm_head_logits_open: read_hyrax_proof(r)?,
        all_lasso_proof: read_lasso_multi_proof(r)?,
        final_range_m: read_global_range_m(r)?,
        // Cross-block batch sumchecks
        batch_qkv: read_sumcheck_proof_multi(r)?,
        batch_oproj: read_sumcheck_proof_multi(r)?,
        batch_ffn_y: read_sumcheck_proof_multi(r)?,
        batch_ffn_m: read_sumcheck_proof_multi(r)?,
        batch_attn_out: read_sumcheck_proof_multi(r)?,
        batch_attn_ctx: read_sumcheck_proof_multi(r)?,
        // Global intermediate batch open
        inter_batch_open: read_hyrax_proof(r)?,
        // 13 cross-block weight/activation batch opens
        x_norm1_batch_open: read_hyrax_proof(r)?,
        w_q_batch_open: read_hyrax_proof(r)?,
        w_k_batch_open: read_hyrax_proof(r)?,
        w_v_batch_open: read_hyrax_proof(r)?,
        bias_q_batch_open: read_hyrax_proof(r)?,
        bias_k_batch_open: read_hyrax_proof(r)?,
        bias_v_batch_open: read_hyrax_proof(r)?,
        w_o_batch_open: read_hyrax_proof(r)?,
        bias_o_batch_open: read_hyrax_proof(r)?,
        w2_batch_open: read_hyrax_proof(r)?,
        w1_batch_open: read_hyrax_proof(r)?,
        x_norm2_batch_open: read_hyrax_proof(r)?,
        ffn_m_com_batch_open: read_hyrax_proof(r)?,
        // 3 attention batch opens
        phi_q_batch_open: read_hyrax_proof(r)?,
        phi_k_batch_open: read_hyrax_proof(r)?,
        v_attn_batch_open: read_hyrax_proof(r)?,
    })
}

// ---------------------------------------------------------------------------
// Per-proof lookup outputs (bundled with proof)
// ---------------------------------------------------------------------------

fn lasso_bits_from_coms(coms: &[HyraxCommitment]) -> io::Result<usize> {
    let first = coms.first().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "empty lasso verifying key")
    })?;
    Ok(first.nu + first.sigma)
}

fn dummy_lasso_from_coms(coms: &[HyraxCommitment], outputs: Vec<F>) -> io::Result<LassoInstance> {
    Ok(LassoInstance {
        tables: vec![Vec::new(); coms.len()],
        outputs,
        bits_per_chunk: lasso_bits_from_coms(coms)?,
    })
}

fn write_lookup_outputs<W: Write>(
    w: &mut W,
    inst_attn: &LinearAttentionInstance,
    inst_ffn: &FFNInstance,
) -> io::Result<()> {
    write_vec_f(w, &inst_attn.q_lasso.outputs)?;
    write_vec_f(w, &inst_attn.k_lasso.outputs)?;
    write_vec_f(w, &inst_ffn.activation_lasso.outputs)
}

fn read_lookup_outputs<R: Read>(r: &mut R) -> io::Result<(Vec<F>, Vec<F>, Vec<F>)> {
    Ok((read_vec_f(r)?, read_vec_f(r)?, read_vec_f(r)?))
}

fn instances_from_vk_and_outputs(
    vk: &TransformerModelVerifyingKey,
    q_outputs: Vec<F>,
    k_outputs: Vec<F>,
    ffn_outputs: Vec<F>,
) -> io::Result<(LinearAttentionInstance, FFNInstance)> {
    let first_block = vk.block_vks.first().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "verifying key has no blocks")
    })?;
    let q_coms = first_block
        .attn_pk
        .qk_lasso_pk
        .instance_table_coms
        .get(0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing Q lasso table commitments"))?;
    let k_coms = first_block
        .attn_pk
        .qk_lasso_pk
        .instance_table_coms
        .get(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing K lasso table commitments"))?;
    let q_lasso = dummy_lasso_from_coms(q_coms, q_outputs)?;
    let k_lasso = dummy_lasso_from_coms(k_coms, k_outputs)?;
    let ffn_lasso = dummy_lasso_from_coms(
        &first_block.ffn_vk.activation_lasso_vk.table_coms,
        ffn_outputs,
    )?;
    Ok((
        LinearAttentionInstance {
            seq_len: vk.seq_len,
            d_head: vk.d_model,
            q_lasso,
            k_lasso,
            q_query_indices: Vec::new(),
            k_query_indices: Vec::new(),
        },
        FFNInstance {
            activation_lasso: ffn_lasso,
        },
    ))
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

/// Encode a proof bundle.
///
/// The bundle stores per-proof lookup output vectors, but not lookup tables.
/// Table commitments and metadata are reconstructed from the verifying key.
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
    write_lookup_outputs(&mut buf, inst_attn, inst_ffn)?;
    write_usize(&mut buf, lasso_sigma)?;
    Ok(buf)
}

/// Decode a proof bundle using the verifying key as the source of lookup-table metadata.
pub fn decode_proof_bundle(
    bytes: &[u8],
    vk: &TransformerModelVerifyingKey,
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
    let (q_outputs, k_outputs, ffn_outputs) = read_lookup_outputs(&mut r)?;
    let lasso_sigma = read_usize(&mut r)?;
    let (inst_attn, inst_ffn) =
        instances_from_vk_and_outputs(vk, q_outputs, k_outputs, ffn_outputs)?;
    Ok((proof, inst_attn, inst_ffn, lasso_sigma))
}

/// Decode only the proof and per-proof lookup output vectors for inspection.
pub fn decode_proof_bundle_public_parts(
    bytes: &[u8],
) -> io::Result<(TransformerModelProof, Vec<F>, Vec<F>, Vec<F>, usize)> {
    let mut r = bytes;
    check_magic(&mut r, PROOF_MAGIC)?;
    let _version = {
        let mut v = [0u8; 1];
        r.read_exact(&mut v)?;
        v[0]
    };
    let proof = read_model_proof(&mut r)?;
    let (q_outputs, k_outputs, ffn_outputs) = read_lookup_outputs(&mut r)?;
    let lasso_sigma = read_usize(&mut r)?;
    Ok((proof, q_outputs, k_outputs, ffn_outputs, lasso_sigma))
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
