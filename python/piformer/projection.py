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


class TernaryLinear(nn.Module):
    """
    重みを {-1, 0, 1} に制限した ZKPフレンドリーな線形層。
    MatMul-free ネットワークの核となる構造。
    """
    def __init__(self, in_features, out_features,
                 min_exp: int = 0, #-4,  # 2^-4 = 0.0625 まで表現
                 max_exp: int = 4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 実数値の重み（学習用）
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        # 学習可能なスケール因子 (alpha)
        # 3値重みにこれを掛けることで、モデルの表現力を維持する
        self.alpha = nn.Parameter(torch.tensor(1.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _quantize(self, w: torch.Tensor) -> torch.Tensor:
        """
        重みを {-1, 0, 1} に量子化する。
        """
        # 1. 重みをスケーリング（平均を引いて中心を合わせる手法もあるが、ここではシンプルに）
        # alpha は量子化後の値の大きさを決定する

        # 2. 閾値の計算 (BitNet等の手法: 重みの絶対値の平均の一定割合を 0 にする)
        # ここではシンプルに、全体の平均を基準に 0 領域を作る
        delta = 0.7 * w.abs().mean()

        # 3. 三値化プロセス
        # w > delta  =>  1
        # w < -delta => -1
        # else       =>  0
        w_q = torch.zeros_like(w)
        w_q[w > delta] = 1
        w_q[w < -delta] = -1

        # 4. alpha を掛けて実スケールに戻す (W_final = alpha * {-1, 0, 1})
        # ZKP上では alpha は行列演算の「後」に1回掛けるだけ。
        return w_q * self.alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 学習時は STE (Straight-Through Estimator)
        if self.training:
            w_q = self._quantize(self.weight).detach() + (self.weight - self.weight.detach())
        else:
            w_q = self._quantize(self.weight)

        # 行列演算 (ZKP側では加減算のみになる)
        return F.linear(x, w_q, self.bias)

    @torch.no_grad()
    def export_weights(self) -> dict:
        """Rust / Jolt 側で利用するためのデータ書き出し"""
        # 量子化された {-1, 0, 1} のインデックスのみを書き出す
        delta = 0.7 * self.weight.abs().mean()
        w_indices = torch.zeros_like(self.weight)
        w_indices[self.weight > delta] = 1
        w_indices[self.weight < -delta] = -1

        return {
            "weight_indices": w_indices.cpu().int().tolist(), # {-1, 0, 1} のみ
            "alpha": self.alpha.item(),
            "bias": self.bias.cpu().tolist() if self.bias is not None else None,
        }
