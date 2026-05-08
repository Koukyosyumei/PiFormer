"""
Linear layer with weights constrained to {0, ±2^0, ±2^1, ..., ±2^max_exp}.

In the SNARK circuit, multiplying by 2^k is a left-shift (a field constant),
so these layers require only additions — no general field multiplications.

Training uses a straight-through estimator (STE) so gradients flow through
the quantization step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TernaryLinear(nn.Module):
    """
    重みを {-1, 0, 1} に制限した ZKPフレンドリーな線形層。
    MatMul-free ネットワークの核となる構造。
    """
    def __init__(self, in_features, out_features,
                 min_exp: int = 0, #-4,  # 2^-4 = 0.0625 まで表現
                 max_exp: int = 4, bias=True, alpha_init: float | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 実数値の重み（学習用）
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        # 学習可能なスケール因子 (alpha)
        # 3値重みにこれを掛けることで、モデルの表現力を維持する
        if alpha_init is None:
            alpha_init = 1.0 / math.sqrt(in_features)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # Gradual QAT knob: 0.0 → pure FP weights, 1.0 → pure ternary.
        # Trainers can ramp this from 0 to 1 over the course of training so the
        # FP weights first find a good basin, then are gently snapped to {-1,0,1}.
        self.register_buffer("quant_strength", torch.tensor(1.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _quantize_indices(self, w: torch.Tensor) -> torch.Tensor:
        """
        重みを {-1, 0, +1} に量子化する。

        Vectorized: a single fused expression instead of boolean-indexed
        assignment, which on GPU collapses ~8 kernel launches down to ~4.
        torch.sign(w) returns 0 for w==0, so multiplying by the >δ mask gives
        the correct {-1, 0, 1} result without any branch.
        """
        abs_w = w.abs()
        delta = abs_w.mean() * 0.7
        return torch.sign(w) * (abs_w > delta).to(w.dtype)

    def _quantize(self, w: torch.Tensor) -> torch.Tensor:
        """Return the scaled quantized weight used by the forward pass."""
        return self._quantize_indices(w) * self.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward value:  alpha * ((1-q) * w + q * sign-mask(w))
        # Backward:       identity STE into self.weight (the explicit
        #                 self.weight term is the only path with a gradient).
        # Both regimes are alpha-scaled so the effective weight magnitude is
        # consistent across the q-schedule and alpha receives gradient
        # throughout (including the q=0 warmup phase).
        # All ops are tensor-valued (no .item() / float() conversion) and there
        # is no data-dependent branching, so torch.compile captures one static
        # graph regardless of training/eval mode or the current value of q.
        w_idx = self._quantize_indices(self.weight.detach())
        w_blend = self.weight + self.quant_strength * (w_idx - self.weight.detach())
        return F.linear(x, w_blend * self.alpha, self.bias)

    @torch.no_grad()
    def export_weights(self) -> dict:
        """Rust / Jolt 側で利用するためのデータ書き出し"""
        # 量子化された {-1, 0, 1} のインデックスのみを書き出す
        w_indices = self._quantize_indices(self.weight)

        return {
            "weight_indices": w_indices.cpu().int().tolist(), # {-1, 0, 1} のみ
            "alpha": self.alpha.item(),
            "bias": self.bias.cpu().tolist() if self.bias is not None else None,
        }
