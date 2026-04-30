"""
Baseline transformer for architectural comparison against PiFormer.

Mirrors PiFormerModel's constructor signature so the comparison harness can
instantiate either model with the same hyperparameters. Differences from
PiFormer are exactly the three components we want to measure:

  * Softmax multi-head attention (vs. linear attention with kernel φ)
  * GELU activation                (vs. structured lookup activation)
  * Full-precision nn.Linear       (vs. ternary projections)

LayerNorm and the residual structure are identical in both models so the
attribution stays clean.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=self.causal)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


class BaselineFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class BaselineBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, causal: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SoftmaxAttention(d_model, n_heads, causal=causal)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = BaselineFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BaselineTransformer(nn.Module):
    """Standard pre-norm transformer with softmax attention, GELU FFN, dense weights."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        causal: bool = True,
        # Quantization kwargs accepted and ignored so the comparison harness
        # can call both constructors with identical signatures.
        **_unused,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.register_buffer(
            "position_ids",
            torch.arange(max_seq_len).unsqueeze(0),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [BaselineBlock(d_model, n_heads, d_ff, causal=causal) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = self.position_ids[:, :T]
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
