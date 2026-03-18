# Efficient SNARKs for Linear Transformers via Structured Lookup Attention

We propose a modified attention and trasnformer algorithm for efficient SNARK.

- linear attention instead of softmax attention: $Attention(Q, K, V) = \phi(Q)(\phi(K)^T V)$
- We define the activation kernel function $\phi$ as the learnable and structured lookup table to obtain both efficiency and accuracy.
- Power-of-Two Linformer Projection: restrict the element of projection matricies to 2^{k} or \{0, 1, -1\}, whose multiplication is easier to compute.
- constraint Fusion inspried from zkGPT.


```
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredLookupActivation(nn.Module):
    def __init__(self, num_bits=16, c=2, scale=1.0):
        super().__init__()
        self.num_bits = num_bits
        self.c = c  # 分解数（c=2なら8bit+8bit）
        self.bits_per_table = num_bits // c
        self.table_size = 2**self.bits_per_table
        self.scale = scale # 量子化スケール [cite: 162]

        # サブテーブルを学習パラメータとして定義
        # 初期値としてGeLUなどの形状を持たせると学習が収束しやすい
        self.tables = nn.ParameterList([
            nn.Parameter(torch.randn(self.table_size) * 0.01) 
            for _ in range(c)
        ])

    def forward(self, x):
        # 1. 入力を指定のビット幅に量子化 [cite: 161, 555]
        # x_quant は 0 から 2^num_bits - 1 の整数
        x_quant = torch.clamp((x / self.scale).round(), 0, 2**self.num_bits - 1).long()

        # 2. インデックスの分解 (Structured Indexing) [cite: 212, 1000]
        # 例: 16bitを 8bit(high) と 8bit(low) に分ける
        indices = []
        temp_idx = x_quant
        for _ in range(self.c):
            indices.append(temp_idx % self.table_size)
            temp_idx = temp_idx // self.table_size
        
        # 3. 各サブテーブルからのルックアップと加算 (Additive Decomposition) [cite: 1253, 1261]
        # これが Lasso/Shout で O(c * n^(1/c)) の高速化を可能にする構造 [cite: 211, 802]
        output = 0
        for i in range(self.c):
            output = output + self.tables[i][indices[i]]
            
        return output

# --- 使用例 ---
# 16bit入力を2つの8bitテーブルに分解する活性化関数
phi = StructuredLookupActivation(num_bits=16, c=2, scale=0.01)

# 入力データ
input_tensor = torch.randn(1, 128) # (Batch, Dim)
output = phi(input_tensor)

print(f"Output shape: {output.shape}")
print(f"Table size (total entries): {phi.table_size * phi.c} (vs 65536 in unstructured)")
```

