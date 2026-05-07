"""
ZK-friendly linear attention layer.

Attention(Q, K, V) = φ(Q) (φ(K)^T V) / Z

- φ is StructuredLookupActivation (proved via Lasso in the SNARK)
- Projections use TernaryLinear (no general multiplications in circuit)
- No softmax: replaces exp+row-normalize with a kernel feature map
- Associativity: compute (φ(K)^T V) first → O(n d²) not O(n² d)
"""

import torch
import torch.nn as nn
from .activation import StructuredLookupActivation
from .projection import TernaryLinear


class LinearAttentionLayer(nn.Module):
    """
    Multi-head linear attention with structured lookup kernel.

    Args:
        d_model: model dimension.
        n_heads: number of attention heads.
        num_bits: quantization bits for φ.
        c: number of sub-tables for φ.
        scale: quantization scale for φ.
        max_exp: max power-of-two exponent for projections.
        eps: numerical stability floor for the normalizer Z.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_bits: int = 8,
        c: int = 2,
        scale: float = 0.1,
        max_exp: int = 4,
        eps: float = 1e-6,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps
        # Causal mode uses cumulative sums and is for training/eval comparison
        # only. The non-causal path is what the prover circuit and witness
        # currently mirror; do not enable causal=True for proof generation.
        self.causal = causal

        self.q_proj = TernaryLinear(d_model, d_model, max_exp=max_exp, bias=False)
        self.k_proj = TernaryLinear(d_model, d_model, max_exp=max_exp, bias=False)
        self.v_proj = TernaryLinear(d_model, d_model, max_exp=max_exp, bias=False)
        self.out_proj = TernaryLinear(d_model, d_model, max_exp=max_exp, bias=True)

        # Single strictly positive φ shared by Q and K projections. Linear
        # attention normalizers are unstable if the kernel features can go
        # negative or collapse to zero for half the signed input range.
        self.phi = StructuredLookupActivation(
            num_bits=num_bits, c=c, scale=scale, init="positive"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        Q = self.q_proj(x)  # (B, T, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads: (B, n_heads, T, d_head)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Apply φ element-wise
        phiQ = self.phi(Q)  # (B, n_heads, T, d_head)
        phiK = self.phi(K)

        if self.causal:
            # Per-position outer products φ(K_s) ⊗ V_s, cumulatively summed
            # over the sequence so position t only sees s ≤ t.
            kv = torch.einsum("bhnd,bhne->bhnde", phiK, V)   # (B, h, T, d, d)
            kv_cum = kv.cumsum(dim=2)
            out = torch.einsum("bhnd,bhnde->bhne", phiQ, kv_cum)

            k_cum = phiK.cumsum(dim=2)                       # (B, h, T, d)
            Z = (phiQ * k_cum).sum(dim=-1, keepdim=True).clamp(min=self.eps)
        else:
            # context = φ(K)^T · V  →  (B, n_heads, d_head, d_head)
            # Einsum: for each head, contract over the sequence dimension T.
            context = torch.einsum("bhnd,bhnm->bhdm", phiK, V)

            # out = φ(Q) · context  →  (B, n_heads, T, d_head)
            out = torch.einsum("bhnd,bhdm->bhnm", phiQ, context)

            # Normalizer Z_t = φ(Q_t) · Σ_s φ(K_s)
            k_sum = phiK.sum(dim=2, keepdim=True)            # (B, h, 1, d)
            Z = (phiQ * k_sum).sum(dim=-1, keepdim=True).clamp(min=self.eps)
        out = out / Z

        # Merge heads and final projection
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)
