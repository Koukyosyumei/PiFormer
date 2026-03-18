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


class PowerOfTwoLinear(nn.Module):
    """
    Linear layer whose weights are quantized to power-of-two values.

    Args:
        in_features, out_features: standard Linear dims.
        max_exp: highest power; candidates are {0, ±1, ±2, ±4, ..., ±2^max_exp}.
        bias: whether to include a (non-quantized) bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_exp: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_exp = max_exp

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Candidate set: 0, ±1, ±2, ±4, ..., ±2^max_exp
        pos = [2.0 ** k for k in range(max_exp + 1)]
        candidates = sorted(set([0.0] + pos + [-v for v in pos]))
        self.register_buffer(
            "candidates", torch.tensor(candidates, dtype=torch.float32)
        )

    def _quantize(self, w: torch.Tensor) -> torch.Tensor:
        """Snap each weight to the nearest candidate."""
        diffs = (w.unsqueeze(-1) - self.candidates).abs()
        best = diffs.argmin(dim=-1)
        return self.candidates[best]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # STE: quantized value in forward, real gradient in backward
            w_q = self._quantize(self.weight).detach() + (
                self.weight - self.weight.detach()
            )
        else:
            w_q = self._quantize(self.weight)
        return F.linear(x, w_q, self.bias)

    def export_weights(self) -> dict:
        """Return quantized weights for the Rust prover."""
        w_q = self._quantize(self.weight).detach()
        return {
            "weight": w_q.tolist(),
            "bias": self.bias.detach().tolist() if self.bias is not None else None,
        }
