"""
PiFormer model: stack of linear-attention transformer blocks.

Each block consists of:
  - Pre-norm LayerNorm
  - LinearAttentionLayer  (ZK-friendly attention)
  - Pre-norm LayerNorm
  - PiFormerFFN           (ZK-friendly feed-forward with structured activation)
"""

import torch
import torch.nn as nn
from .attention import LinearAttentionLayer
from .projection import TernaryLinear
from .activation import StructuredLookupActivation


class PiFormerFFN(nn.Module):
    """
    Feed-forward block using TernaryLinear + StructuredLookupActivation.

    In the SNARK:
    - fc1, fc2 use constant (power-of-two) multiplications → only additions needed
    - act is proved via Lasso lookup
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_bits: int = 8,
        c: int = 2,
        scale: float = 0.1,
        max_exp: int = 4,
    ):
        super().__init__()
        self.fc1 = TernaryLinear(d_model, d_ff, max_exp=max_exp)
        self.act = StructuredLookupActivation(num_bits=num_bits, c=c, scale=scale)
        self.fc2 = TernaryLinear(d_ff, d_model, max_exp=max_exp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PiFormerBlock(nn.Module):
    """Single transformer block: norm → attn → residual → norm → ffn → residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_bits: int = 8,
        c: int = 2,
        scale: float = 0.1,
        max_exp: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LinearAttentionLayer(
            d_model, n_heads,
            num_bits=num_bits, c=c, scale=scale, max_exp=max_exp,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PiFormerFFN(
            d_model, d_ff,
            num_bits=num_bits, c=c, scale=scale, max_exp=max_exp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PiFormerModel(nn.Module):
    """
    Full PiFormer language model.

    Args:
        vocab_size: vocabulary size.
        d_model: model/embedding dimension.
        n_heads: attention heads (must divide d_model).
        n_layers: number of transformer blocks.
        d_ff: FFN hidden dimension.
        max_seq_len: maximum sequence length (for positional embeddings).
        num_bits: quantization bits for φ activations.
        c: sub-table decomposition depth for φ.
        scale: quantization scale for φ.
        max_exp: max power-of-two exponent for weight quantization.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        num_bits: int = 8,
        c: int = 2,
        scale: float = 0.1,
        max_exp: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                PiFormerBlock(
                    d_model, n_heads, d_ff,
                    num_bits=num_bits, c=c, scale=scale, max_exp=max_exp,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = TernaryLinear(d_model, vocab_size, max_exp=max_exp, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) integer token ids.
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
