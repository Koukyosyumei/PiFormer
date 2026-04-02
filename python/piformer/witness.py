"""
Integer forward pass and ZK witness generation for PiFormer.

Runs the model's forward pass entirely in Python arbitrary-precision integers,
producing a witness dict whose structure exactly matches the Rust
``json_io::JsonWitness`` schema.

Design contract
---------------
* All activations that enter a LayerNorm (x_in, x_mid, x_out) are
  arbitrary signed Python ints converted to BN254 field elements on output.
  Field arithmetic in the Rust prover reproduces the same arithmetic mod p.

* The phi activation clamps inputs to [0, 2^num_bits − 1], so Lasso
  query_indices are always small non-negative integers.

* LayerNorm witnesses (sum_x, sq_sum_x, y) are small non-negative Python
  ints computed via the Lasso inv-sqrt lookup, matching ``layernorm.rs``.

* Ternary weight matrices are integer-valued ({−alpha_int, 0, +alpha_int});
  their products with integer activations remain exact.

Only single-head attention (n_heads = 1) is supported.  Multi-head models
require the Rust prover to be extended accordingly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from .quant import (
    apply_phi_int,
    compute_ln_witness,
    extract_phi_tables_int,
    extract_ternary_weight_matrix,
    int_to_field_hex,
    mat_add_int,
    mat_mul_int,
    mat_to_json,
    mat_transpose,
    min_beta_floor,
    quantize_to_int,
    vec_to_json,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _project(x_int: List[List[int]], w_int: List[List[int]]) -> List[List[int]]:
    """Matrix multiply X (T × d_in) by W (d_in × d_out) → (T × d_out)."""
    return mat_mul_int(x_int, w_int)


def _add_bias_int(
    y: List[List[int]], bias: List[int] | None
) -> List[List[int]]:
    if bias is None:
        return y
    return [[y[i][j] + bias[j] for j in range(len(y[0]))] for i in range(len(y))]


def _phi_mat(
    m_int: List[List[int]],
    tables_int: List[List[int]],
    num_bits: int,
    c: int,
    phi_scale: float,
    quant_scale: int,
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Apply the decomposed lookup activation φ to every element of a matrix.

    1. Quantize each float → integer index in [0, 2^num_bits − 1].
       Here m_int values are already integers (scaled by quant_scale), so
       the effective float is m_int[i][j] / quant_scale.
    2. Look up φ(index) = Σ_k table_k[chunk_k(index)].

    Returns:
        (phi_out_matrix, query_indices_flat, outputs_flat)
    """
    bits_per_chunk = num_bits // c
    t, d = len(m_int), len(m_int[0])

    # Compute query indices: clamp-round quantize each element.
    # m_int is in "integer units"; effective float is m_int / quant_scale.
    q_indices_flat: List[int] = []
    for row in m_int:
        for v in row:
            x_float_approx = v / quant_scale if quant_scale != 0 else float(v)
            q_indices_flat.append(quantize_to_int(x_float_approx, phi_scale, num_bits))

    outputs_flat, _ = apply_phi_int(q_indices_flat, tables_int, bits_per_chunk)

    phi_out = [
        [outputs_flat[i * d + j] for j in range(d)]
        for i in range(t)
    ]
    return phi_out, q_indices_flat, outputs_flat


def _context_matrix(
    phi_k: List[List[int]], v: List[List[int]]
) -> List[List[int]]:
    """context = φ(K)ᵀ · V : (d_head, d_head)."""
    return mat_mul_int(mat_transpose(phi_k), v)


def _attn_out(
    phi_q: List[List[int]], context: List[List[int]]
) -> List[List[int]]:
    """out = φ(Q) · context : (T, d_head)."""
    return mat_mul_int(phi_q, context)


# ---------------------------------------------------------------------------
# LN weight preparation
# ---------------------------------------------------------------------------


def _prepare_ln_weights(
    gamma_float: List[float],
    beta_float: List[float],
    x_rows: List[List[int]],
    d: int,
    ln_scale: int,
    extra_beta_floor: int,
) -> Tuple[List[int], List[int]]:
    """
    Convert floating-point LayerNorm weights to non-negative integers that
    satisfy the Rust protocol's range-proof constraints for *x_rows*.

    1. gamma_int[j] = max(1, round(|gamma_float[j]| * ln_scale))
    2. beta_int[j]  = round(beta_float[j] * ln_scale) + adaptive_floor
                      + extra_beta_floor

    The adaptive floor is computed so that all y values produced by
    compute_ln_witness are non-negative.
    """
    gamma_int = [max(1, round(abs(g) * ln_scale)) for g in gamma_float]
    beta_base = [round(b * ln_scale) for b in beta_float]

    floor_needed = min_beta_floor(x_rows, gamma_int, d) + extra_beta_floor
    beta_int = [b + floor_needed for b in beta_base]
    return gamma_int, beta_int


# ---------------------------------------------------------------------------
# Block-level witness generation
# ---------------------------------------------------------------------------


def _gen_block_witness(
    x_in: List[List[int]],
    blk,  # PiFormerBlock
    quant_scale: int,
    ln_scale: int,
    extra_beta_floor: int,
) -> Tuple[Dict[str, Any], List[List[int]]]:
    """
    Generate the integer witness for one transformer block.

    Returns (block_witness_dict, x_out_int).
    The dict uses the same keys as ``json_io::JsonBlockWitness``.
    """
    d = len(x_in[0])
    t = len(x_in)

    attn = blk.attn
    ffn = blk.ffn

    if attn.n_heads != 1:
        raise ValueError(
            f"WitnessGenerator supports only n_heads=1 (got {attn.n_heads}). "
            "Multi-head attention requires extending the Rust prover."
        )

    # ---- Attention phi parameters ----
    phi_num_bits = attn.phi.num_bits
    phi_c = attn.phi.c
    phi_bits_per_chunk = attn.phi.bits_per_chunk
    phi_scale = attn.phi.scale
    attn_tables_int = extract_phi_tables_int(
        attn.phi.export_tables(), output_scale=quant_scale
    )

    # ---- FFN phi parameters ----
    ffn_num_bits = ffn.act.num_bits
    ffn_c = ffn.act.c
    ffn_bits_per_chunk = ffn.act.bits_per_chunk
    ffn_phi_scale = ffn.act.scale
    ffn_tables_int = extract_phi_tables_int(
        ffn.act.export_tables(), output_scale=quant_scale
    )

    # ---- LN1 ----
    ln1_gamma_int, ln1_beta_int = _prepare_ln_weights(
        blk.norm1.weight.detach().tolist(),
        blk.norm1.bias.detach().tolist(),
        x_in,
        d,
        ln_scale,
        extra_beta_floor,
    )
    ln1_y, ln1_sum_x, ln1_sq_sum_x = compute_ln_witness(
        x_in, ln1_gamma_int, ln1_beta_int, d
    )

    # ---- Q, K, V projections ----
    import torch
    w_q = extract_ternary_weight_matrix(
        attn.q_proj.weight.detach().tolist(),
        float(attn.q_proj.alpha.detach()),
        quant_scale=1,
    )
    w_k = extract_ternary_weight_matrix(
        attn.k_proj.weight.detach().tolist(),
        float(attn.k_proj.alpha.detach()),
        quant_scale=1,
    )
    w_v = extract_ternary_weight_matrix(
        attn.v_proj.weight.detach().tolist(),
        float(attn.v_proj.alpha.detach()),
        quant_scale=1,
    )
    # Weights are (out, in) in PyTorch; Rust expects X @ W where W is (d_in, d_out)
    w_q_T = mat_transpose(w_q)  # (d_model, d_model)
    w_k_T = mat_transpose(w_k)
    w_v_T = mat_transpose(w_v)

    q_raw = _project(ln1_y, w_q_T)
    k_raw = _project(ln1_y, w_k_T)
    v_raw = _project(ln1_y, w_v_T)

    # ---- φ activation on Q and K ----
    phi_q, q_indices, q_outputs = _phi_mat(
        q_raw, attn_tables_int, phi_num_bits, phi_c, phi_scale, quant_scale
    )
    phi_k, k_indices, k_outputs = _phi_mat(
        k_raw, attn_tables_int, phi_num_bits, phi_c, phi_scale, quant_scale
    )

    # ---- Linear attention ----
    context = _context_matrix(phi_k, v_raw)  # (d_head, d_head)
    attn_out = _attn_out(phi_q, context)     # (T, d_head)

    # ---- Output projection ----
    w_o = extract_ternary_weight_matrix(
        attn.out_proj.weight.detach().tolist(),
        float(attn.out_proj.alpha.detach()),
        quant_scale=1,
    )
    w_o_T = mat_transpose(w_o)
    o_bias = (
        [round(b) for b in attn.out_proj.bias.detach().tolist()]
        if attn.out_proj.bias is not None
        else None
    )
    out_attn = _add_bias_int(_project(attn_out, w_o_T), o_bias)

    # ---- Residual 1 ----
    x_mid = mat_add_int(x_in, out_attn)

    # ---- LN2 ----
    ln2_gamma_int, ln2_beta_int = _prepare_ln_weights(
        blk.norm2.weight.detach().tolist(),
        blk.norm2.bias.detach().tolist(),
        x_mid,
        d,
        ln_scale,
        extra_beta_floor,
    )
    ln2_y, ln2_sum_x, ln2_sq_sum_x = compute_ln_witness(
        x_mid, ln2_gamma_int, ln2_beta_int, d
    )

    # ---- FFN ----
    # FFN: Rust circuit proves M = X @ W1_ternary and Y = A @ W2_ternary
    # with no alpha scaling and no bias.  Use alpha=1.0 and skip bias here.
    w1 = extract_ternary_weight_matrix(
        ffn.fc1.weight.detach().tolist(),
        1.0,
        quant_scale=1,
    )
    w1_T = mat_transpose(w1)  # (d_model, d_ff)
    ffn_m = _project(ln2_y, w1_T)

    ffn_a, ffn_q_indices, ffn_q_outputs = _phi_mat(
        ffn_m, ffn_tables_int, ffn_num_bits, ffn_c, ffn_phi_scale, quant_scale
    )

    w2 = extract_ternary_weight_matrix(
        ffn.fc2.weight.detach().tolist(),
        1.0,
        quant_scale=1,
    )
    w2_T = mat_transpose(w2)  # (d_ff, d_model)
    ffn_y = _project(ffn_a, w2_T)

    # ---- Residual 2 ----
    x_out = mat_add_int(x_mid, ffn_y)

    # ---- Assemble lasso instances (stored in top-level witness) ----
    q_lasso = {
        "tables": [[int_to_field_hex(v) for v in tbl] for tbl in attn_tables_int],
        "query_indices": q_indices,
        "outputs": [int_to_field_hex(v) for v in q_outputs],
        "bits_per_chunk": phi_bits_per_chunk,
    }
    k_lasso = {
        "tables": [[int_to_field_hex(v) for v in tbl] for tbl in attn_tables_int],
        "query_indices": k_indices,
        "outputs": [int_to_field_hex(v) for v in k_outputs],
        "bits_per_chunk": phi_bits_per_chunk,
    }
    ffn_lasso = {
        "tables": [[int_to_field_hex(v) for v in tbl] for tbl in ffn_tables_int],
        "query_indices": ffn_q_indices,
        "outputs": [int_to_field_hex(v) for v in ffn_q_outputs],
        "bits_per_chunk": ffn_bits_per_chunk,
    }

    block_wit = {
        "x_in": mat_to_json(x_in),
        "ln1": {
            "x": mat_to_json(x_in),
            "y": mat_to_json(ln1_y),
            "sum_x": vec_to_json(ln1_sum_x),
            "sq_sum_x": vec_to_json(ln1_sq_sum_x),
        },
        "q_proj": {"x": mat_to_json(ln1_y), "y": mat_to_json(q_raw)},
        "k_proj": {"x": mat_to_json(ln1_y), "y": mat_to_json(k_raw)},
        "v_proj": {"x": mat_to_json(ln1_y), "y": mat_to_json(v_raw)},
        "attn": {
            "q": mat_to_json(q_raw),
            "k": mat_to_json(k_raw),
            "v": mat_to_json(v_raw),
            "phi_q": mat_to_json(phi_q),
            "phi_k": mat_to_json(phi_k),
            "context": mat_to_json(context),
            "out": mat_to_json(attn_out),
        },
        "o_proj": {"x": mat_to_json(attn_out), "y": mat_to_json(out_attn)},
        "x_mid": mat_to_json(x_mid),
        "ln2": {
            "x": mat_to_json(x_mid),
            "y": mat_to_json(ln2_y),
            "sum_x": vec_to_json(ln2_sum_x),
            "sq_sum_x": vec_to_json(ln2_sq_sum_x),
        },
        "ffn": {
            "x": mat_to_json(ln2_y),
            "m": mat_to_json(ffn_m),
            "a": mat_to_json(ffn_a),
            "y": mat_to_json(ffn_y),
        },
        "x_out": mat_to_json(x_out),
    }

    # Return metadata needed to build LassoInstances at the top level
    block_wit["_attn_lasso"] = (q_lasso, k_lasso)
    block_wit["_ffn_lasso"] = ffn_lasso
    block_wit["_ln1_weights"] = (ln1_gamma_int, ln1_beta_int)
    block_wit["_ln2_weights"] = (ln2_gamma_int, ln2_beta_int)

    return block_wit, x_out


# ---------------------------------------------------------------------------
# Top-level witness generator
# ---------------------------------------------------------------------------


class WitnessGenerator:
    """
    Generates a complete integer witness for the Rust ZK prover from a
    trained PiFormerModel and a token-id sequence.

    Parameters
    ----------
    quant_scale : int
        Scale applied to embedding floats → ints.
        E.g. ``64`` maps a float in [0, 4] to an integer in [0, 256].
    ln_scale : int
        Scale applied to LayerNorm gamma/beta floats → ints.
    extra_beta_floor : int
        Additional minimum added to every beta to provide headroom beyond the
        computed adaptive floor (safety margin).
    lasso_sigma : int
        Hyrax PCS sigma parameter used for Lasso commitments in the Rust prover.
        Must match the value used in ``HyraxParams::new(lasso_sigma)``.
    """

    def __init__(
        self,
        quant_scale: int = 64,
        ln_scale: int = 4,
        extra_beta_floor: int = 8,
        lasso_sigma: int = 4,
    ):
        self.quant_scale = quant_scale
        self.ln_scale = ln_scale
        self.extra_beta_floor = extra_beta_floor
        self.lasso_sigma = lasso_sigma

    def generate(self, model, token_ids: List[int]) -> Dict[str, Any]:
        """
        Run the integer forward pass and return the complete witness dict.

        Parameters
        ----------
        model : PiFormerModel
            Trained model in eval mode.
        token_ids : list of int
            Integer token ids, length = seq_len.

        Returns
        -------
        dict matching the ``json_io::JsonWitness`` schema, ready for
        ``json.dumps`` and loading by the Rust prover.
        """
        import torch

        model.eval()
        seq_len = len(token_ids)
        d_model = model.d_model

        # ---- Initial embeddings (float → integer) ----
        with torch.no_grad():
            tok_ids_t = torch.tensor([token_ids], dtype=torch.long)
            pos_ids = torch.arange(seq_len).unsqueeze(0)
            emb_float = (
                model.embedding(tok_ids_t) + model.pos_embedding(pos_ids)
            )
        # emb_float: (1, T, d_model) → (T, d_model)
        emb_float_2d = emb_float[0].tolist()
        x_in_int = [
            [max(1, round(v * self.quant_scale)) for v in row]
            for row in emb_float_2d
        ]

        # ---- Block forward passes ----
        block_witnesses: List[Dict] = []
        q_lasso_last = k_lasso_last = ffn_lasso_last = None
        x_cur = x_in_int

        block_ln_weights = []
        for blk in model.blocks:
            bwit, x_cur = _gen_block_witness(
                x_cur, blk, self.quant_scale, self.ln_scale, self.extra_beta_floor
            )
            q_lasso_last, k_lasso_last = bwit.pop("_attn_lasso")
            ffn_lasso_last = bwit.pop("_ffn_lasso")
            ln1_gamma, ln1_beta = bwit.pop("_ln1_weights")
            ln2_gamma, ln2_beta = bwit.pop("_ln2_weights")
            block_ln_weights.append((ln1_gamma, ln1_beta, ln2_gamma, ln2_beta))
            block_witnesses.append(bwit)

        # ---- Final LayerNorm ----
        final_gamma_int, final_beta_int = _prepare_ln_weights(
            model.norm.weight.detach().tolist(),
            model.norm.bias.detach().tolist(),
            x_cur,
            d_model,
            self.ln_scale,
            self.extra_beta_floor,
        )
        final_ln_y, final_ln_sum_x, final_ln_sq_sum_x = (
            compute_ln_witness(x_cur, final_gamma_int, final_beta_int, d_model)
        )

        # ---- LM Head projection ----
        w_head = extract_ternary_weight_matrix(
            model.head.weight.detach().tolist(),
            float(model.head.alpha.detach()),
            quant_scale=1,
        )
        w_head_T = mat_transpose(w_head)  # (d_model, vocab_size)
        lm_head_y = _project(final_ln_y, w_head_T)

        # ---- Use lasso instances from the last block (shared phi tables) ----
        # For models with all identical blocks, the first block's tables suffice.
        if q_lasso_last is None:
            raise RuntimeError("Model must have at least one block.")

        # Derive lasso_sigma from bits_per_chunk to match lasso.rs:
        #   nu = m // 2;  sigma = m - nu
        m = model.blocks[0].attn.phi.bits_per_chunk
        lasso_sigma = m - m // 2

        # Expose per-layer LN weights for use by the weight exporter, so the
        # verifying key gamma/beta matches what was used to compute witness y.
        self.block_ln_weights = block_ln_weights
        self.final_ln_weights = (final_gamma_int, final_beta_int)

        witness = {
            "lasso_sigma": lasso_sigma,
            "x_in": mat_to_json(x_in_int),
            "inst_attn": {
                "seq_len": seq_len,
                "d_head": d_model,
                "q_lasso": q_lasso_last,
                "k_lasso": k_lasso_last,
            },
            "inst_ffn": {
                "activation_lasso": ffn_lasso_last,
            },
            "blocks": block_witnesses,
            "final_ln": {
                "x": mat_to_json(x_cur),
                "y": mat_to_json(final_ln_y),
                "sum_x": vec_to_json(final_ln_sum_x),
                "sq_sum_x": vec_to_json(final_ln_sq_sum_x),
            },
            "lm_head": {
                "x": mat_to_json(final_ln_y),
                "y": mat_to_json(lm_head_y),
            },
        }
        return witness

    def predicted_token(self, witness: Dict[str, Any]) -> int:
        """Return the argmax token from the last-position lm_head output."""
        from .quant import field_hex_to_int

        lm_head_y = witness["lm_head"]["y"]
        last_pos = lm_head_y[-1]
        values = [field_hex_to_int(v) for v in last_pos]
        return max(range(len(values)), key=lambda i: values[i])
