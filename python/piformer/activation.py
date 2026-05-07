"""
ZK-friendly activation via additively decomposed lookup tables.

φ(x) = Σ_{i=0}^{c-1} table_i[ (x_int >> (i * bits_per_chunk)) & mask ]

The additive decomposition matches Jolt's Lasso lookup argument:
each sub-table of size 2^(num_bits/c) can be committed separately,
giving O(c · 2^(num_bits/c)) total commitment cost vs O(2^num_bits) for a flat table.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredLookupActivation(nn.Module):
    """
    Learnable activation function implemented as a sum of sub-table lookups.

    Args:
        num_bits: Total quantization bits for the input (e.g. 8 or 16).
        c: Number of sub-tables (decomposition depth). Must divide num_bits.
        scale: Quantization scale; x_int = clamp(round(x / scale) + zero_point,
            0, 2^num_bits - 1).
        init: Initial table shape. ``"positive"`` is intended for attention
            kernels; ``"gelu"`` is intended for feed-forward activations.

    In the SNARK, each sub-table lookup is proved via a Lasso sumcheck argument.
    The additive decomposition lets us commit to c tables of size 2^(num_bits/c)
    instead of one table of size 2^num_bits.
    """

    def __init__(
        self,
        num_bits: int = 8,
        c: int = 2,
        scale: float = 1.0,
        init: str = "gelu",
    ):
        super().__init__()
        assert num_bits % c == 0, "num_bits must be divisible by c"
        init_kind = init
        if init_kind not in {"gelu", "positive"}:
            raise ValueError(f"unknown StructuredLookupActivation init: {init}")
        self.num_bits = num_bits
        self.c = c
        self.bits_per_chunk = num_bits // c
        self.chunk_size = 2 ** self.bits_per_chunk
        self.scale = scale
        self.zero_point = 1 << (num_bits - 1)
        self.init = init_kind

        # Initialize sub-tables to approximate GeLU / c shape so training converges faster.
        self.tables = nn.ParameterList()

        for i in range(c):
            indices = torch.arange(self.chunk_size, dtype=torch.float32)
            if i == c - 1: # 最上位テーブル
                # 上位ビットのインデックスがカバーする範囲の「代表値」でGeLUを計算
                # 例: 8bit, c=2なら 16刻みで値をサンプリング
                stride = self.chunk_size ** i
                x_idx = indices * stride + stride / 2
                x_approx = (x_idx - self.zero_point) * self.scale
                if init_kind == "positive":
                    init_values = F.elu(x_approx) + 1.0
                else:
                    init_values = F.gelu(x_approx).clamp_min(0.0)
            else:
                # 下位テーブルは 0 で初期化してギザギザを防ぐ
                init_values = torch.zeros(self.chunk_size)
            self.tables.append(nn.Parameter(init_values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantize to non-negative integer index.
        # The integer index path is non-differentiable today, so avoid building
        # a throwaway autograd graph for division/rounding/clamping.
        with torch.no_grad():
            x_int = (
                (x / self.scale).round()
                .add_(self.zero_point)
                .clamp_(0, 2 ** self.num_bits - 1)
                .long()
            )

        # 2. Decompose and sum sub-table lookups.
        # chunk_size is a power of two, so use bit-and / bit-shift instead of
        # % / //. We also seed the accumulator from the first lookup rather
        # than zeros_like(), saving one allocation + one add per call. This
        # gets called twice per attention layer (φ(Q), φ(K)) plus once per
        # FFN, so the kernel-count savings compound.
        mask = self.chunk_size - 1
        out = self.tables[0][x_int & mask]
        for i in range(1, self.c):
            out = out + self.tables[i][(x_int >> (i * self.bits_per_chunk)) & mask]

        if self.training:
            # Straight-through estimator: forward uses the ZK-friendly lookup,
            # backward lets gradients reach the projections before this layer.
            out = out + (x - x.detach())
        return out

    def export_tables(self) -> list[list[float]]:
        """Return detached sub-table values for the Rust prover."""
        return [t.detach().tolist() for t in self.tables]
