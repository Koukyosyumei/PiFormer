"""
Export a trained PiFormerModel to JSON for consumption by the Rust prover.

Schema:
{
  "d_model": int,
  "embedding": [[float, ...]],              # vocab_size × d_model
  "pos_embedding": [[float, ...]],          # max_seq_len × d_model
  "blocks": [
    {
      "attn": {
        "n_heads": int,
        "d_head": int,
        "q_proj": {"weight": [[...]], "bias": null_or_list},
        "k_proj": {...},
        "v_proj": {...},
        "out_proj": {...},
        "phi_tables": [[...], [...]]        # c sub-tables, each of size 2^bits_per_chunk
      },
      "ffn": {
        "fc1": {"weight": [[...]], "bias": [...]},
        "fc2": {...},
        "phi_tables": [[...], [...]]
      },
      "norm1": {"weight": [...], "bias": [...]},
      "norm2": {...}
    },
    ...
  ],
  "norm": {"weight": [...], "bias": [...]},
  "head": {"weight": [[...]], "bias": null}
}
"""

import json
from pathlib import Path

import torch.nn as nn

from .model import PiFormerModel


def _export_ln(ln: nn.LayerNorm) -> dict:
    return {
        "weight": ln.weight.detach().tolist(),
        "bias": ln.bias.detach().tolist(),
    }


def export_model(model: PiFormerModel, output_path: str) -> None:
    """Serialize all quantized weights to JSON at *output_path*."""
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
                "norm1": _export_ln(blk.norm1),
                "norm2": _export_ln(blk.norm2),
            }
        )

    payload = {
        "d_model": model.d_model,
        "embedding": model.embedding.weight.detach().tolist(),
        "pos_embedding": model.pos_embedding.weight.detach().tolist(),
        "blocks": blocks_data,
        "norm": _export_ln(model.norm),
        "head": model.head.export_weights(),
    }

    Path(output_path).write_text(json.dumps(payload, indent=2))
    print(f"[export] Saved model weights → {output_path}")
