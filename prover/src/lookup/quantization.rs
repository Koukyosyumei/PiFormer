use crate::field::F;
use crate::lookup::range::{
    prove_range_batched, verify_range_batched, GlobalRangeM, RangeProofWitness, RangeWitnessProof,
};
use crate::pcs::{
    absorb_com, hyrax_commit, hyrax_open_batch, hyrax_verify_batch, params_from_vars,
    HyraxBatchAccumulator, HyraxCommitment, HyraxProof,
};
use crate::poly::DenseMLPoly;
use crate::transcript::{challenge_vec, Transcript};
use ark_ff::PrimeField;

/// Exact quantization contract for lookup activations.
///
/// `scale_num / scale_den` is the effective integer-domain scale. If Python
/// stores activations as `raw = round(x * quant_scale)`, then
/// `scale_num / scale_den = quant_scale * activation_scale`.
#[derive(Clone, Debug)]
pub struct QuantizationParams {
    pub scale_num: u64,
    pub scale_den: u64,
}

#[derive(Clone)]
pub struct QuantizationProof {
    pub rem_coms: Vec<HyraxCommitment>,
    pub rem_range_proofs: Vec<RangeWitnessProof>,
    pub rem_range_m: GlobalRangeM,
    pub raw_evals: Vec<F>,
    pub rem_evals: Vec<F>,
    pub raw_open: HyraxProof,
    pub rem_open: HyraxProof,
}

fn field_to_i128(x: F) -> Result<i128, String> {
    let raw = x.into_bigint();
    let limbs = raw.as_ref();
    if limbs.get(2).copied().unwrap_or(0) == 0 && limbs.get(3).copied().unwrap_or(0) == 0 {
        let lo = limbs[0] as u128;
        let hi = limbs.get(1).copied().unwrap_or(0) as u128;
        let v = lo | (hi << 64);
        i128::try_from(v).map_err(|_| "field value does not fit i128".to_string())
    } else {
        let modulus = F::MODULUS;
        let m = modulus.as_ref();
        let mut out = [0u64; 4];
        let mut borrow = 0u128;
        for i in 0..4 {
            let mi = m.get(i).copied().unwrap_or(0) as u128;
            let ri = limbs.get(i).copied().unwrap_or(0) as u128;
            let sub = ri + borrow;
            if mi >= sub {
                out[i] = (mi - sub) as u64;
                borrow = 0;
            } else {
                out[i] = ((1u128 << 64) + mi - sub) as u64;
                borrow = 1;
            }
        }
        if borrow != 0 || out[2] != 0 || out[3] != 0 {
            return Err("negative field magnitude does not fit i128".to_string());
        }
        let mag_u = (out[0] as u128) | ((out[1] as u128) << 64);
        let mag = i128::try_from(mag_u)
            .map_err(|_| "negative field value does not fit i128".to_string())?;
        Ok(-mag)
    }
}

fn round_div_half_up(num: i128, den: i128) -> i128 {
    debug_assert!(den > 0);
    (num + den / 2).div_euclid(den)
}

pub fn lookup_zero_point_u64(table_count: usize, bits_per_chunk: usize) -> Result<u64, String> {
    if table_count <= 1 {
        return Ok(0);
    }
    let total_bits = table_count
        .checked_mul(bits_per_chunk)
        .ok_or_else(|| "lookup zero-point bit width overflow".to_string())?;
    if total_bits == 0 || total_bits > 63 {
        return Err(format!(
            "unsupported lookup zero-point bit width {total_bits}"
        ));
    }
    Ok(1u64 << (total_bits - 1))
}

pub fn quantized_lookup_index(
    raw: F,
    table_count: usize,
    bits_per_chunk: usize,
    params: &QuantizationParams,
) -> Result<usize, String> {
    if params.scale_num < 2 || params.scale_den == 0 {
        return Err("quantization scale numerator/denominator must be nonzero".to_string());
    }
    let total_bits = table_count
        .checked_mul(bits_per_chunk)
        .ok_or_else(|| "lookup bit width overflow".to_string())?;
    if total_bits == 0 || total_bits > 63 {
        return Err(format!("unsupported lookup bit width {total_bits}"));
    }
    let max_idx = (1u64 << total_bits) - 1;
    let zp = lookup_zero_point_u64(table_count, bits_per_chunk)? as i128;
    let raw_i = field_to_i128(raw)?;
    let scaled = raw_i
        .checked_mul(params.scale_den as i128)
        .ok_or_else(|| "quantization numerator overflow".to_string())?;
    let rounded = round_div_half_up(scaled, params.scale_num as i128);
    let idx = rounded
        .checked_add(zp)
        .ok_or_else(|| "quantization index overflow".to_string())?;
    if idx < 0 || idx > max_idx as i128 {
        return Err(format!(
            "quantized lookup index {idx} outside [0, {max_idx}]"
        ));
    }
    Ok(idx as usize)
}

fn validate_quant_params(params: &QuantizationParams) -> Result<(), String> {
    if params.scale_num < 2 || params.scale_den == 0 {
        return Err(
            "quantization scale numerator must be at least 2 and denominator nonzero".to_string(),
        );
    }
    if !params.scale_num.is_power_of_two() {
        return Err(format!(
            "quantization scale numerator must be a power of two for exact range proof, got {}",
            params.scale_num
        ));
    }
    Ok(())
}

fn range_bits_for_scale_num(scale_num: u64) -> usize {
    scale_num.trailing_zeros() as usize
}

fn indicator_eval(point: &[F], n: usize) -> F {
    DenseMLPoly::from_vec_padded(vec![F::from(1u64); n]).evaluate(point)
}

fn index_eval(indices: &[usize], point: &[F]) -> F {
    DenseMLPoly::from_vec_padded(indices.iter().map(|&i| F::from(i as u64)).collect())
        .evaluate(point)
}

fn quant_remainders(
    raw_values: &[F],
    indices: &[usize],
    table_count: usize,
    bits_per_chunk: usize,
    params: &QuantizationParams,
) -> Result<Vec<F>, String> {
    validate_quant_params(params)?;
    if raw_values.len() != indices.len() {
        return Err(format!(
            "raw/index length mismatch: got {} raw values and {} indices",
            raw_values.len(),
            indices.len()
        ));
    }
    let zp = lookup_zero_point_u64(table_count, bits_per_chunk)? as i128;
    let num = params.scale_num as i128;
    let den = params.scale_den as i128;
    let half = num / 2;
    raw_values
        .iter()
        .zip(indices.iter())
        .enumerate()
        .map(|(i, (&raw, &idx))| {
            let expected = quantized_lookup_index(raw, table_count, bits_per_chunk, params)?;
            if expected != idx {
                return Err(format!(
                    "lookup index mismatch at {i}: got {idx}, expected {expected}"
                ));
            }
            let q = idx as i128 - zp;
            let raw_i = field_to_i128(raw)?;
            let rem = raw_i
                .checked_mul(den)
                .and_then(|v| v.checked_add(half))
                .and_then(|v| v.checked_sub(num.checked_mul(q)?))
                .ok_or_else(|| format!("remainder computation overflow at {i}"))?;
            if rem < 0 || rem >= num {
                return Err(format!(
                    "quantization remainder out of range at {i}: {rem} not in [0,{num})"
                ));
            }
            Ok(F::from(rem as u64))
        })
        .collect()
}

pub fn prove_quantization_batch(
    label: &'static [u8],
    raw_mles: &[DenseMLPoly],
    raw_coms: &[HyraxCommitment],
    index_vectors: &[Vec<usize>],
    table_count: usize,
    bits_per_chunk: usize,
    rows: usize,
    cols: usize,
    params: &QuantizationParams,
    transcript: &mut Transcript,
) -> Result<QuantizationProof, String> {
    validate_quant_params(params)?;
    if raw_mles.len() != raw_coms.len() || raw_mles.len() != index_vectors.len() {
        return Err("quantization batch length mismatch".to_string());
    }
    let n = rows * cols;
    let num_vars = rows.next_power_of_two().trailing_zeros() as usize
        + cols.next_power_of_two().trailing_zeros() as usize;
    let (nu, sigma, params_h) = params_from_vars(num_vars);

    let mut rem_mles = Vec::with_capacity(raw_mles.len());
    let mut rem_coms = Vec::with_capacity(raw_mles.len());
    let mut rem_witnesses = Vec::with_capacity(raw_mles.len());
    for (raw_mle, indices) in raw_mles.iter().zip(index_vectors.iter()) {
        let raw_values = raw_mle.evaluations[..n].to_vec();
        let rem_values =
            quant_remainders(&raw_values, indices, table_count, bits_per_chunk, params)?;
        let rem_mle = DenseMLPoly::from_vec_padded(rem_values.clone());
        let rem_com = hyrax_commit(&rem_mle.evaluations, nu, &params_h);
        absorb_com(transcript, label, &rem_com);
        rem_witnesses.push(RangeProofWitness { values: rem_values });
        rem_mles.push(rem_mle);
        rem_coms.push(rem_com);
    }

    let rem_refs: Vec<&RangeProofWitness> = rem_witnesses.iter().collect();
    let (rem_range_proofs, rem_range_m, _) = prove_range_batched(
        &rem_refs,
        range_bits_for_scale_num(params.scale_num),
        transcript,
    )?;

    let r = challenge_vec(transcript, num_vars, b"quant_r");
    let raw_refs: Vec<&[F]> = raw_mles.iter().map(|m| m.evaluations.as_slice()).collect();
    let rem_refs_mle: Vec<&[F]> = rem_mles.iter().map(|m| m.evaluations.as_slice()).collect();
    let raw_evals: Vec<F> = raw_mles.iter().map(|m| m.evaluate(&r)).collect();
    let rem_evals: Vec<F> = rem_mles.iter().map(|m| m.evaluate(&r)).collect();
    let raw_open = hyrax_open_batch(&raw_refs, &r, nu, sigma, transcript);
    let rem_open = hyrax_open_batch(&rem_refs_mle, &r, nu, sigma, transcript);
    Ok(QuantizationProof {
        rem_coms,
        rem_range_proofs,
        rem_range_m,
        raw_evals,
        rem_evals,
        raw_open,
        rem_open,
    })
}

pub fn verify_quantization_batch(
    label: &'static [u8],
    proof: &QuantizationProof,
    raw_coms: &[HyraxCommitment],
    index_vectors: &[&[usize]],
    table_count: usize,
    bits_per_chunk: usize,
    rows: usize,
    cols: usize,
    params: &QuantizationParams,
    transcript: &mut Transcript,
    acc_range_sig: &mut HyraxBatchAccumulator,
    acc_range_y: &mut HyraxBatchAccumulator,
    acc_range_m: &mut HyraxBatchAccumulator,
) -> Result<(), String> {
    validate_quant_params(params)?;
    if proof.rem_coms.len() != raw_coms.len()
        || proof.raw_evals.len() != raw_coms.len()
        || proof.rem_evals.len() != raw_coms.len()
        || index_vectors.len() != raw_coms.len()
    {
        return Err("quantization proof length mismatch".to_string());
    }
    let n = rows * cols;
    let num_vars = rows.next_power_of_two().trailing_zeros() as usize
        + cols.next_power_of_two().trailing_zeros() as usize;
    for rem_com in &proof.rem_coms {
        absorb_com(transcript, label, rem_com);
    }
    let range_proof_refs: Vec<&RangeWitnessProof> = proof.rem_range_proofs.iter().collect();
    let rem_n_vars = n.next_power_of_two().trailing_zeros() as usize;
    verify_range_batched(
        &range_proof_refs,
        &proof.rem_range_m,
        &vec![rem_n_vars; proof.rem_range_proofs.len()],
        range_bits_for_scale_num(params.scale_num),
        transcript,
        acc_range_sig,
        acc_range_y,
        acc_range_m,
    )?;
    let r = challenge_vec(transcript, num_vars, b"quant_r");
    let (_, _, params_h) = params_from_vars(num_vars);
    hyrax_verify_batch(
        raw_coms,
        &proof.raw_evals,
        &r,
        &proof.raw_open,
        &params_h,
        transcript,
    )
    .map_err(|e| format!("quant raw opening: {e}"))?;
    hyrax_verify_batch(
        &proof.rem_coms,
        &proof.rem_evals,
        &r,
        &proof.rem_open,
        &params_h,
        transcript,
    )
    .map_err(|e| format!("quant remainder opening: {e}"))?;

    let one = indicator_eval(&r, n);
    let zp = F::from(lookup_zero_point_u64(table_count, bits_per_chunk)?);
    let num = F::from(params.scale_num);
    let den = F::from(params.scale_den);
    let half = F::from(params.scale_num / 2);
    for i in 0..raw_coms.len() {
        if index_vectors[i].len() != n {
            return Err(format!(
                "quant index vector length mismatch at {i}: got {}, expected {n}",
                index_vectors[i].len()
            ));
        }
        let idx_eval = index_eval(index_vectors[i], &r);
        let lhs = proof.raw_evals[i] * den + half * one;
        let rhs = num * (idx_eval - zp * one) + proof.rem_evals[i];
        if lhs != rhs {
            return Err(format!("quantization relation failed at batch item {i}"));
        }
    }
    Ok(())
}

pub fn verify_quantized_indices(
    label: &str,
    raw_values: &[F],
    indices: &[usize],
    table_count: usize,
    bits_per_chunk: usize,
    params: &QuantizationParams,
) -> Result<(), String> {
    if raw_values.len() != indices.len() {
        return Err(format!(
            "{label}: raw/index length mismatch: got {} raw values and {} indices",
            raw_values.len(),
            indices.len()
        ));
    }
    for (i, (&raw, &idx)) in raw_values.iter().zip(indices.iter()).enumerate() {
        let expected = quantized_lookup_index(raw, table_count, bits_per_chunk, params)
            .map_err(|e| format!("{label}[{i}]: {e}"))?;
        if idx != expected {
            return Err(format!(
                "{label}[{i}]: lookup index mismatch, got {idx}, expected {expected}"
            ));
        }
    }
    Ok(())
}
