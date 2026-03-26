"""
Export a trained PiFormerModel to JSON for consumption by the Rust prover.

Two export targets are supported:

1. ``export_weights_rust`` — writes a ``weights.json`` whose schema matches
   ``json_io::JsonWeights`` in the Rust prover.  All values are BN254 field
   elements encoded as 0x-prefixed 64-nibble hex strings.

2. ``export_witness_rust`` — runs the integer forward pass via
   ``WitnessGenerator`` and writes a ``witness.json`` whose schema matches
   ``json_io::JsonWitness``.

3. ``export_all`` — convenience wrapper that calls both.

Quantization conventions
------------------------
* LayerNorm weights: ``gamma_int[j] = max(1, round(weight[j] * ln_scale))``,
  ``beta_int[j]  = round(bias[j] * ln_scale) + beta_floor``.
  A per-layer ``beta_floor`` is auto-computed via ``min_beta_floor`` so all
  LayerNorm output values are non-negative (required by the Rust range proof).

* Projection weights: ternary {−alpha_int, 0, +alpha_int}.  The Rust circuit
  computes ``Y[i][j] = Σ_k X[i][k] · W[k][j]`` (W is d_in × d_out), while
  PyTorch stores weight as (out_features × in_features), so we transpose.

* φ activation tables: ``table_int[k][i] = max(0, round(table_float[k][i]))``.
  Tables are part of the witness / Lasso instance, not the weight file.

* Legacy ``export_model`` (old float format) is preserved for backward
  compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch.nn as nn

from .model import PiFormerModel
from .quant import (
    extract_ternary_weight_matrix,
    int_to_field_hex,
    mat_to_json,
    mat_transpose,
    min_beta_floor,
    vec_to_json,
)
from .witness import WitnessGenerator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ln_weights_int(
    ln: nn.LayerNorm,
    x_rows: Optional[List[List[int]]],
    d: int,
    ln_scale: int,
    extra_beta_floor: int,
) -> tuple[List[int], List[int]]:
    """Convert a PyTorch LayerNorm to integer (gamma, beta) vectors.

    gamma_int[j] = max(1, round(weight[j] * ln_scale))
    beta_int[j]  = round(bias[j] * ln_scale) + beta_floor

    ``beta_floor`` is computed adaptively from ``x_rows`` if provided,
    otherwise ``extra_beta_floor`` is used as a conservative default.
    """
    gamma_int = [max(1, round(abs(float(w)) * ln_scale)) for w in ln.weight]
    beta_raw = [round(float(b) * ln_scale) for b in ln.bias]

    if x_rows is not None:
        floor = min_beta_floor(x_rows, gamma_int, d) + extra_beta_floor
    else:
        floor = extra_beta_floor

    beta_int = [b + floor for b in beta_raw]
    return gamma_int, beta_int


def _proj_weight_int(
    linear: "TernaryLinear",  # noqa: F821
    quant_scale: int,
) -> List[List[int]]:
    """Extract integer (d_in × d_out) weight matrix from a TernaryLinear.

    PyTorch stores weight as (out_features × in_features); the Rust prover
    expects (d_in × d_out), so we transpose.
    """
    weight_float = linear.weight.detach().tolist()  # out × in
    alpha = float(linear.alpha)
    # Pass alpha=1.0 so entries are exactly {-1, 0, 1}; the Rust prover
    # expects the raw sign matrix, not values scaled by alpha.
    w_out_in = extract_ternary_weight_matrix(weight_float, 1.0)
    return mat_transpose(w_out_in)  # → in × out


# ---------------------------------------------------------------------------
# Rust-native weight export
# ---------------------------------------------------------------------------


def export_weights_rust(
    model: PiFormerModel,
    out_path: str,
    *,
    quant_scale: int = 64,
    ln_scale: int = 4,
    extra_beta_floor: int = 8,
    block_ln_weights: Optional[List] = None,
    final_ln_weights: Optional[tuple] = None,
) -> None:
    """Write ``weights.json`` in the format expected by the Rust prover.

    Args:
        model:              Trained PiFormerModel (eval mode recommended).
        out_path:           Destination file path.
        quant_scale:        Integer scale applied to TernaryLinear alpha values.
        ln_scale:           Integer scale applied to LayerNorm gamma/beta.
        extra_beta_floor:   Extra margin added to the auto-computed beta_floor.
        block_ln_weights:   Pre-computed list of (ln1_gamma, ln1_beta,
                            ln2_gamma, ln2_beta) per block.  When provided,
                            these are used directly instead of recomputing.
        final_ln_weights:   Pre-computed (gamma, beta) for the final LN layer.

    Raises:
        ValueError: if model.blocks[0].attn.n_heads != 1.
    """
    model.eval()
    d = model.d_model

    if model.blocks and model.blocks[0].attn.n_heads != 1:
        raise ValueError(
            "The Rust prover is single-head only.  "
            f"Got n_heads={model.blocks[0].attn.n_heads}. "
            "Either use n_heads=1 or extend the Rust prover."
        )

    n_layers = len(model.blocks)
    vocab_size = model.embedding.num_embeddings
    d_ff = model.blocks[0].ffn.fc1.out_features if n_layers > 0 else 0

    def _ln(ln: nn.LayerNorm) -> tuple[List[int], List[int]]:
        return _ln_weights_int(ln, None, d, ln_scale, extra_beta_floor)

    blocks_json = []
    for i, blk in enumerate(model.blocks):
        attn = blk.attn
        if block_ln_weights and i < len(block_ln_weights):
            ln1_gamma, ln1_beta, ln2_gamma, ln2_beta = block_ln_weights[i]
        else:
            ln1_gamma, ln1_beta = _ln(blk.norm1)
            ln2_gamma, ln2_beta = _ln(blk.norm2)

        q_w = _proj_weight_int(attn.q_proj, quant_scale)   # d × d
        k_w = _proj_weight_int(attn.k_proj, quant_scale)
        v_w = _proj_weight_int(attn.v_proj, quant_scale)
        o_w = _proj_weight_int(attn.out_proj, quant_scale)
        ffn_w1 = _proj_weight_int(blk.ffn.fc1, quant_scale)  # d × d_ff
        ffn_w2 = _proj_weight_int(blk.ffn.fc2, quant_scale)  # d_ff × d

        blocks_json.append({
            "ln1_gamma": vec_to_json(ln1_gamma),
            "ln1_beta":  vec_to_json(ln1_beta),
            "q_w":  q_w,
            "k_w":  k_w,
            "v_w":  v_w,
            "o_w":  o_w,
            "ln2_gamma": vec_to_json(ln2_gamma),
            "ln2_beta":  vec_to_json(ln2_beta),
            "ffn_w1": ffn_w1,
            "ffn_w2": ffn_w2,
        })

    if final_ln_weights is not None:
        final_ln_gamma, final_ln_beta = final_ln_weights
    else:
        final_ln_gamma, final_ln_beta = _ln(model.norm)
    lm_head_w = _proj_weight_int(model.head, quant_scale)  # d × vocab_size

    payload = {
        "num_blocks":       n_layers,
        "d_model":          d,
        "d_ff":             d_ff,
        "vocab_size":       vocab_size,
        "blocks":           blocks_json,
        "final_ln_gamma":   vec_to_json(final_ln_gamma),
        "final_ln_beta":    vec_to_json(final_ln_beta),
        "lm_head_w":        lm_head_w,
    }

    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"[export] weights.json → {out_path}  "
          f"(num_blocks={n_layers}, d_model={d}, d_ff={d_ff}, vocab={vocab_size})")


# ---------------------------------------------------------------------------
# Rust-native witness export
# ---------------------------------------------------------------------------


def export_witness_rust(
    model: PiFormerModel,
    token_ids: List[int],
    out_path: str,
    *,
    quant_scale: int = 64,
    ln_scale: int = 4,
    extra_beta_floor: int = 8,
    lasso_sigma: int = 4,
) -> None:
    """Run the integer forward pass and write ``witness.json``.

    Args:
        model:             Trained PiFormerModel (eval mode recommended).
        token_ids:         List of integer token ids (length = seq_len).
        out_path:          Destination file path.
        quant_scale:       Integer scale for activations / ternary weights.
        ln_scale:          Integer scale for LayerNorm gamma/beta.
        extra_beta_floor:  Extra safety margin added to auto beta_floor.
        lasso_sigma:       Hyrax sigma parameter used by the Lasso prover
                           (must match the value passed to ``piformer prove``).
    """
    gen = WitnessGenerator(
        quant_scale=quant_scale,
        ln_scale=ln_scale,
        extra_beta_floor=extra_beta_floor,
        lasso_sigma=lasso_sigma,
    )
    witness_dict = gen.generate(model, token_ids)
    Path(out_path).write_text(json.dumps(witness_dict, indent=2))
    print(f"[export] witness.json → {out_path}  "
          f"(seq_len={len(token_ids)}, lasso_sigma={lasso_sigma})")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def export_all(
    model: PiFormerModel,
    token_ids: List[int],
    *,
    weights_path: str = "weights.json",
    witness_path: str = "witness.json",
    quant_scale: int = 64,
    ln_scale: int = 4,
    extra_beta_floor: int = 8,
    lasso_sigma: int = 4,
) -> None:
    """Export both weights and witness in a single call.

    Generates the witness first (which computes actual LN gamma/beta from
    real activations), then exports weights using those same gamma/beta values
    so the verifying key is guaranteed to match the witness.

    Writes:
        ``weights_path``  — JSON weights for ``piformer setup``
        ``witness_path``  — JSON witness for ``piformer prove``
    """
    # Step 1: Generate the witness, capturing the per-layer LN weights used.
    gen = WitnessGenerator(
        quant_scale=quant_scale,
        ln_scale=ln_scale,
        extra_beta_floor=extra_beta_floor,
        lasso_sigma=lasso_sigma,
    )
    witness_dict = gen.generate(model, token_ids)
    Path(witness_path).write_text(json.dumps(witness_dict, indent=2))
    seq_len = len(token_ids)
    print(f"[export] witness.json → {witness_path}  "
          f"(seq_len={seq_len}, lasso_sigma={lasso_sigma})")

    # Step 2: Export weights using the same per-layer gamma/beta that the
    # witness used, so the verifying key is consistent with the witness.
    export_weights_rust(
        model, weights_path,
        quant_scale=quant_scale,
        ln_scale=ln_scale,
        extra_beta_floor=extra_beta_floor,
        block_ln_weights=gen.block_ln_weights,
        final_ln_weights=gen.final_ln_weights,
    )


# ---------------------------------------------------------------------------
# Legacy float export (backward compatibility)
# ---------------------------------------------------------------------------


def _export_ln_float(ln: nn.LayerNorm) -> dict:
    return {
        "weight": ln.weight.detach().tolist(),
        "bias": ln.bias.detach().tolist(),
    }


def export_model(model: PiFormerModel, output_path: str) -> None:
    """Serialize floating-point weights to JSON (legacy format).

    This function produces the *old* schema used before quantization alignment
    was implemented.  For the Rust prover use ``export_weights_rust`` instead.
    """
    model.eval()

    blocks_data = []
    for blk in model.blocks:
        attn = blk.attn
        blocks_data.append(
            {
                "attn": {
                    "n_heads": attn.n_heads,
                    "d_head": attn.d_head,
                    "q_proj": attn.q_proj.export_weights(),
                    "k_proj": attn.k_proj.export_weights(),
                    "v_proj": attn.v_proj.export_weights(),
                    "out_proj": attn.out_proj.export_weights(),
                    "phi_tables": attn.phi.export_tables(),
                    "phi_num_bits": attn.phi.num_bits,
                    "phi_c": attn.phi.c,
                    "phi_scale": attn.phi.scale,
                },
                "ffn": {
                    "fc1": blk.ffn.fc1.export_weights(),
                    "fc2": blk.ffn.fc2.export_weights(),
                    "phi_tables": blk.ffn.act.export_tables(),
                    "phi_num_bits": blk.ffn.act.num_bits,
                    "phi_c": blk.ffn.act.c,
                    "phi_scale": blk.ffn.act.scale,
                },
                "norm1": _export_ln_float(blk.norm1),
                "norm2": _export_ln_float(blk.norm2),
            }
        )

    payload = {
        "d_model": model.d_model,
        "embedding": model.embedding.weight.detach().tolist(),
        "pos_embedding": model.pos_embedding.weight.detach().tolist(),
        "blocks": blocks_data,
        "norm": _export_ln_float(model.norm),
        "head": model.head.export_weights(),
    }

    Path(output_path).write_text(json.dumps(payload, indent=2))
    print(f"[export] Saved model weights → {output_path}")
