# π-Former: Technical Design

> Succinct ZK Proofs of Transformer Inference via Structured Lookup Attention

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [ZK-Friendly Model Architecture](#2-zk-friendly-model-architecture)
3. [Proof System](#3-proof-system)
4. [Protocol Pseudocode](#4-protocol-pseudocode)
5. [Security Analysis](#5-security-analysis)
6. [Fixed-Point Encoding](#6-fixed-point-encoding)
7. [Complexity Summary](#7-complexity-summary)
8. [Comparison with Related Work](#8-comparison-with-related-work)
9. [Planned Extensions](#9-planned-extensions)
10. [Implementation Notes](#10-implementation-notes)
11. [File Formats](#11-file-formats)

---

## 1. Problem Statement

Let $\mathcal{M}$ be a transformer model with $L$ layers, embedding dimension $d$, and FFN width $d_{ff}$. Given a public input sequence $\mathbf{x} = (x_1, \ldots, x_T)$ and the model's frozen weights $\theta$, we want a succinct, non-interactive argument of knowledge:

$$\pi \leftarrow \mathsf{Prove}(\theta, \mathbf{x}, \mathbf{y}) \quad \text{such that} \quad \mathsf{Verify}(\theta, \mathbf{x}, \mathbf{y}, \pi) = 1 \iff \mathcal{M}_\theta(\mathbf{x}) = \mathbf{y}$$

The naïve approach of encoding the entire transformer in an R1CS circuit is prohibitively expensive:

1. **Softmax** requires `exp` and row-wise normalization — transcendental functions with no compact polynomial representation.
2. **Layer normalization** involves a square root and division by a running variance.
3. **Activation functions** (GeLU, SiLU) have no compact polynomial representation.
4. **Large tables**: a flat lookup for a 16-bit activation requires $2^{16}$ commitments.

π-Former addresses each bottleneck by co-designing the model and the proof system.

---

## 2. ZK-Friendly Model Architecture

### 2.1 Block Structure

Each transformer block computes:

```
X_norm1  = LayerNorm(X_in,        γ₁, β₁)
Q, K, V  = X_norm1 · W_Q/K/V      (projection)
Out_attn = φ(Q)(φ(K)ᵀV) · W_O    (linear attention + output projection)
X_mid    = X_in + Out_attn         (residual 1)
X_norm2  = LayerNorm(X_mid,       γ₂, β₂)
Out_ffn  = FFN(X_norm2)            (feed-forward network)
X_out    = X_mid + Out_ffn         (residual 2)
```

After all blocks, a final LayerNorm and language-model head projection produce the output logits.

### 2.2 Linear Attention

We replace the standard scaled dot-product attention

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

with the kernel (linear) attention formulation

$$\text{Attn}(Q, K, V) = \phi(Q)\,[\phi(K)^\top V] \tag{1}$$

where $\phi : \mathbb{F}^{d_h} \to \mathbb{F}^{d_h}$ is an element-wise feature map.

**Why this is SNARK-friendly:**

- The context matrix $C = \phi(K)^\top V \in \mathbb{F}^{d_h \times d_h}$ depends only on the key-value sequence, not on any query. It is proved once and reused for all queries.
- There are no exponentials, no row-wise divisions with unknown denominators, and no $\max$ operations.
- Associativity gives O(T d²) complexity instead of O(T² d), matching the proof cost.

### 2.3 Structured Lookup Activation

The kernel function $\phi$ is implemented as a **learnable, additively decomposed lookup table**:

$$\phi(x) = \sum_{i=0}^{c-1} T_i\!\left[\left\lfloor \frac{x_{\text{int}}}{2^{im}} \right\rfloor \bmod 2^m \right] \tag{2}$$

where:
- $x_{\text{int}} = \text{clamp}(\lfloor x / s \rfloor, 0, 2^B - 1)$ is the quantized input ($B$ bits, scale $s$).
- $B = c \cdot m$: total bits split into $c$ chunks of $m$ bits each.
- $T_0, \ldots, T_{c-1} \in \mathbb{F}^{2^m}$ are learnable sub-tables.

**Decomposition rationale:** A flat lookup for a $B$-bit input requires committing to a table of size $2^B$. The additive decomposition in Eq. (2) reduces this to $c$ tables of size $2^m = 2^{B/c}$, giving total commitment cost $O(c \cdot 2^{B/c})$ instead of $O(2^B)$. For $B=16, c=2$: 512 commitments instead of 65 536.

This structure exactly matches the **Lasso** lookup argument (§3.5), which is designed to exploit additive decompositions of this form.

**Initialization:** Sub-tables are initialized to approximate $\text{GeLU}(x)/c$ so that their sum closely approximates GeLU at the start of training, enabling stable convergence.

### 2.4 Power-of-Two Weight Quantization

Projection matrices $W_Q, W_K, W_V, W_O$ and FFN weight matrices are constrained to entries in

<<<<<<< HEAD
$$\mathcal{W} = \{0\} \cup \{\pm 2^k \mid k = 0, 1, \ldots, k_{\max}\}$$
=======
$$\mathcal{W} = \{0\} \cup \{-1, 0, 1\} \tag{3}$$
>>>>>>> main

A matrix-vector product $y = Wx$ with $W \in \mathcal{W}^{m \times n}$ in the circuit requires only additions and multiplications by field constants (left-shifts). There are no general field multiplications, so this is **free** in the sumcheck constraint system.

**Training:** We use a straight-through estimator (STE). The forward pass applies nearest-neighbor quantization to $\mathcal{W}$; the backward pass uses the identity function in place of the quantizer's zero gradient.

---

## 3. Proof System

### 3.1 Field

We work over the BN254 scalar field $\mathbb{F}_r$ (prime $r \approx 2^{254}$). All model values (weights, activations, hidden states) are represented as fixed-point integers encoded as field elements.

### 3.2 Hyrax Polynomial Commitment Scheme

π-Former uses **Hyrax**, a transparent (no trusted setup) multilinear PCS based on discrete-log hardness of the BN254 G1 group.

For a multilinear polynomial with $2^n = 2^{\nu+\sigma}$ evaluations, arranged as a $2^{\nu} \times 2^{\sigma}$ matrix $M$:

**Commit:**
$$C_i = \text{MSM}(\mathbf{g},\, M[i]) \quad \text{for each row } i \in [2^\nu]$$

where $\mathbf{g} = (g_0, \ldots, g_{2^\sigma - 1})$ are "nothing-up-my-sleeve" G1 generators derived via `SHA3("piformer-hyrax-gen" ‖ i)`.

**Open** at point $r = (r_L \| r_R)$ with $r_L \in \mathbb{F}^\nu$, $r_R \in \mathbb{F}^\sigma$:

$$w'_j = \sum_{i} L_i(r_L) \cdot M[i][j]$$

where $L_i$ are the multilinear Lagrange basis polynomials over $r_L$.

**Verify:**
1. $\sum_i L_i(r_L) \cdot C_i \stackrel{?}{=} \text{MSM}(\mathbf{g},\, w')$ — homomorphic commitment check.
2. $\langle R(r_R),\, w' \rangle \stackrel{?}{=} v$ — inner product check against claimed evaluation $v$.

**Complexity:** Prover O($\sqrt{N}$) group operations; verifier O($\sqrt{N}$) group operations + O($\log N$) field operations; proof size O($\sqrt{N}$) field elements.

### 3.3 Dense Multilinear Polynomials

A multilinear polynomial $\tilde{f} : \mathbb{F}_r^n \to \mathbb{F}_r$ over $n$ variables is uniquely determined by its evaluations on $\{0,1\}^n$, stored as a flat vector of $2^n$ field elements:

$$\text{evals}[i] = \tilde{f}(\text{bit}_0(i), \text{bit}_1(i), \ldots, \text{bit}_{n-1}(i)), \quad \text{bit}_k(i) = (i \gg k) \mathbin{\&} 1$$

**Key operations:**

- **Evaluate at $r \in \mathbb{F}_r^n$:** Repeated halving. In round $j$, fix variable $n-1-j$ to $r_j$. Cost: O($2^n$).
- **Fix first variable to $r$:** $\text{new}[i] = (1-r)\cdot\text{evals}[i] + r\cdot\text{evals}[i + 2^{n-1}]$. Cost: O($2^n$).
- **Equality polynomial** $\widetilde{\text{eq}}(r, \cdot)$: $\prod_{i}(r_i x_i + (1-r_i)(1-x_i))$.

### 3.4 Sumcheck Protocol

We use the sumcheck protocol for the statement

$$H = \sum_{x \in \{0,1\}^n} f(x) \cdot g(x) \tag{3}$$

where $f, g$ are dense MLEs (degree-2 round polynomials).

**Round $i$ ($i = 1,\ldots,n$):**

1. Prover sends univariate $g_i(X)$ represented by evaluations at $X \in \{0,1,2\}$.
2. Verifier checks $g_i(0) + g_i(1) = H_{i-1}$ and sends random $r_i \stackrel{\$}{\leftarrow} \mathbb{F}_r$.
3. Set $H_i = g_i(r_i)$.

At the end, the verifier holds a single opening claim $f(r) \cdot g(r) = H_n$, checked via two Hyrax openings.

**Complexity:** Prover O($n \cdot 2^n$) field operations; verifier O($n$) field operations plus two PCS openings.

### 3.5 Lasso Lookup Argument

Given sub-table $T_k \in \mathbb{F}_r^{2^m}$ and queries $\{(\text{idx}_j, v_j)\}_{j=1}^N$ with $v_j = T_k[\text{chunk}_k(\text{idx}_j)]$, correctness is proved via a batched MLE sumcheck.

**Step 1 — Commit.** Represent $T_k$ as a dense MLE $\widetilde{T}_k$ over $m$ variables.

**Step 2 — Batch.** Prover and verifier agree on random $\rho \stackrel{\$}{\leftarrow} \mathbb{F}_r$ (Fiat-Shamir). Define the selector polynomial:

$$L_k(x) = \sum_{j=1}^N \rho^j \cdot \widetilde{\text{eq}}\!\left(\text{bin}(\text{ch}_{k,j}),\, x\right) \tag{4}$$

where $\text{ch}_{k,j} = \text{chunk}_k(\text{idx}_j)$ and $\text{bin}(\cdot)$ is the $m$-bit binary encoding.

**Step 3 — Sumcheck.** Run the degree-2 sumcheck for

$$\sum_{x \in \{0,1\}^m} \widetilde{T}_k(x) \cdot L_k(x) = \sum_{j=1}^N \rho^j \cdot T_k[\text{ch}_{k,j}] \tag{5}$$

**Step 4 — Open.** After sumcheck terminates at random point $\mathbf{r}$, prover opens $\widetilde{T}_k(\mathbf{r})$ via Hyrax. The verifier checks $\widetilde{T}_k(\mathbf{r}) \cdot L_k(\mathbf{r}) = H_n$, where $L_k(\mathbf{r})$ is recomputed from public queries.

**Complexity (per sub-table):**
- Prover: O($N \cdot 2^m + m \cdot 2^m$) to build $L_k$ and run sumcheck.
- Verifier: O($N \cdot m$) to recompute $L_k(\mathbf{r})$, plus O($m$) sumcheck verification.

### 3.6 Range Proof

The range proof proves that field elements lie in $[0, 2^{32})$ using a chunked Lasso approach:

1. Split each value into two 16-bit chunks: $v = v_{lo} + 2^{16} \cdot v_{hi}$.
2. Commit to the chunk arrays $V_{lo}$, $V_{hi}$ and a multiplicity array $m$ via Hyrax.
3. Run a sumcheck to bind $V$ to the chunk arrays at a random evaluation point.
4. Check multiplicities against the identity table $T[i] = i$ of size $2^{16} = 65\,536$ using a LogUp-style argument.

This replaces expensive bit-decomposition circuits with two small-table Lasso instances, reducing proof cost significantly for the many range constraints in LayerNorm and the Y-output constraint fusion.

### 3.7 LayerNorm Circuit

LayerNorm is proved without any division gates using **constraint fusion**:

**Step 1 — Mean sumcheck.** Prove $\text{sum\_x}[i] = \sum_j x[i][j]$ for all rows at a random evaluation point $r_t$:

$$\text{sum\_x\_mle}(r_t) = \sum_{j \in \{0,1\}^{d_{\text{bits}}}} x_{\text{collapsed}}(j)$$

where $x_{\text{collapsed}}$ is $x_{\text{mle}}$ with the row variables fixed to $r_t$.

**Step 2 — Variance sumcheck.** Prove $\text{var\_x}[i] = \sum_j (d \cdot x[i][j] - \text{sum\_x}[i])^2$:

$$\text{var\_x\_mle}(r_t) = \sum_{j \in \{0,1\}^{d_{\text{bits}}}} h(j)^2, \quad h(j) = d \cdot x_{\text{collapsed}}(j) - \text{sum\_x\_mle}(r_t)$$

**Step 3 — Sigma range proof.** Prove $\sigma[i]$ is the integer floor square-root of $\text{var\_x}[i]$ by showing:

$$\text{var\_x}[i] - (d \cdot \sigma[i])^2 \geq 0 \quad \text{and} \quad (d \cdot \sigma[i] + d)^2 - 1 - \text{var\_x}[i] \geq 0$$

Both residuals are committed and range-checked via the chunked Lasso range proof.

**Step 4 — Y constraint fusion.** Prove the LayerNorm output $y[i][j]$ satisfies:

$$\gamma_j \cdot (d \cdot x[i][j] - \text{sum\_x}[i]) + \beta_j \cdot d \cdot \sigma[i] \approx d \cdot \sigma[i] \cdot y[i][j]$$

(up to integer rounding), verified at a single random point $(r_{y_t}, r_{y_d})$ using another range proof over the lo/hi residuals. The verifier evaluates $\gamma$ and $\beta$ directly from the public VK in O($d$) operations.

**All constraints are verified at a single random point** — the verifier runs O(1) equations, not O(T·d) loops.

### 3.8 Projection Circuit

Proves $Y = X \cdot W$ where $W$ is a committed weight matrix. A random-entry reduction selects one output entry $(r_t, r_d)$ via Fiat-Shamir; a single sumcheck over the contraction dimension proves that entry. Hyrax openings bind $X$, $Y$, and $W$ to the claimed evaluations.

### 3.9 Linear Attention Circuit

Proves the following claims for a single attention head:

| Step | Statement | Protocol |
|------|-----------|----------|
| 1 | $\Phi_Q[t][d] = \phi(Q[t][d])$ | Lasso per chunk |
| 2 | $\Phi_K[t][d] = \phi(K[t][d])$ | Lasso per chunk |
| 3 | $C[i][j] = \sum_t \Phi_K[t][i] \cdot V[t][j]$ | Sumcheck over $t$ |
| 4 | $\text{out}[t][j] = \sum_i \Phi_Q[t][i] \cdot C[i][j]$ | Sumcheck over $i$ |

Steps 3–4 use a random-entry reduction: the verifier draws Fiat-Shamir challenges to select one entry per matrix product, then a single sumcheck proves that entry. Soundness holds with probability $1 - d_h / |\mathbb{F}_r|$.

### 3.10 FFN Circuit

Proves $\text{out} = \phi(X \cdot W_1) \cdot W_2$:

1. **First projection** $M = X \cdot W_1$: sumcheck at a random entry.
2. **Activation** $A = \phi(M)$: Lasso lookup argument (same decomposed tables as the attention kernel).
3. **Second projection** $\text{out} = A \cdot W_2$: sumcheck at a random entry.

### 3.11 Constraint Fusion Across Layers

Before processing each layer, the verifier squeezes a random challenge $\lambda_\ell \stackrel{\$}{\leftarrow} \mathbb{F}_r$ from the Fiat-Shamir transcript. All proofs within layer $\ell$ are generated after this challenge is absorbed, binding them to a common transcript state. A dishonest prover cannot forge one sub-proof without invalidating all subsequent ones.

---

## 4. Protocol Pseudocode

This section gives a high-level view of the verifier pipeline. The verifier processes one transformer block by chaining sub-verifiers from output back toward input (GKR-style), then checks the initial input commitment and all side constraints in a final batched phase.

### 4.1 Main Block Verifier

```
Algorithm Verify_TransformerBlock:
  Input:
    C_W    : committed weight matrices {W_Q, W_K, W_V, W_O, W_1, W_2, γ, β}
    C_X_in : Hyrax commitment to the input X_in
    proof  : {ln1_proof, q_proj_proof, k_proj_proof, v_proj_proof,
              attn_proof, o_proj_proof, ln2_proof, ffn_proof,
              intermediate_commitments}
    inst   : {LinearAttentionInstance, FFNInstance}

  // Phase 1: GKR-style chaining (output → input)
  Verify_FFN(proof.ffn_proof, C_W.W1, C_W.W2, inst.ffn)
  Verify_LayerNorm(proof.ln2_proof, C_W.γ₂, C_W.β₂)
  Verify_Projection(proof.o_proj_proof, C_W.W_O)
  Verify_LinearAttention(proof.attn_proof, inst.attn)
  Verify_Projection(proof.q_proj_proof, C_W.W_Q)
  Verify_Projection(proof.k_proj_proof, C_W.W_K)
  Verify_Projection(proof.v_proj_proof, C_W.W_V)
  Verify_LayerNorm(proof.ln1_proof, C_W.γ₁, C_W.β₁)

  // Phase 2: Bind chaining start to committed input
  Hyrax_Verify(C_X_in, claimed_x_eval, point=r_in, proof.x_open_proof)

  return ACCEPT
```

### 4.2 LayerNorm Verifier

```
Function Verify_LayerNorm(proof, vk):
  d_f = F(d)

  // 1. Absorb IO commitments (x_com from pipeline, y_com from proof)
  Absorb(transcript, "x_com", io_coms.x_com)
  Absorb(transcript, "y_com", io_coms.y_com)
  Absorb(transcript, "sum_x_com", proof.sum_x_com)
  Absorb(transcript, "var_x_com", proof.var_x_com)
  Absorb(transcript, "sigma_com", proof.sigma_com)

  // 2. Row audit challenge
  r_t = ChallengeVec(transcript, t_bits, "layernorm_rt")
  Absorb(transcript, "claimed_mean", proof.openings.sum_x_at_rt)
  Absorb(transcript, "claimed_var",  proof.openings.var_x_at_rt)

  // 3. Mean sumcheck: sum_j x_collapsed[j] = sum_x_mle(r_t)
  (r_d_mean, final_mean) = Verify_Sumcheck(
      proof.mean_sumcheck, claim=proof.openings.sum_x_at_rt, num_vars=d_bits
  )
  Assert final_mean == proof.openings.x_at_rt_rmean * F(1)

  // 4. Variance sumcheck: sum_j h[j]^2 = var_x_mle(r_t)
  //    where h[j] = d * x_collapsed[j] - sum_x_mle(r_t)
  (r_d_var, final_var) = Verify_Sumcheck(
      proof.variance_sumcheck, claim=proof.openings.var_x_at_rt, num_vars=d_bits
  )
  h_eval = d_f * proof.openings.x_at_rt_rvar - proof.openings.sum_x_at_rt
  Assert final_var == h_eval * h_eval

  // 5. Sigma constraint fusion
  (r_sig, sig_eval) = Verify_RangeProof(proof.sigma_range_proof, bits=32)
  r_sig_t = r_sig[0..t_bits]
  r_sig_b = r_sig[t_bits]
  dsi = d_f * proof.openings.sigma_at_rsig
  lo_sig = proof.openings.var_x_at_rsig - dsi * dsi
  hi_sig = (dsi + d_f)*(dsi + d_f) - 1 - proof.openings.var_x_at_rsig
  Assert sig_eval == (1 - r_sig_b) * lo_sig + r_sig_b * hi_sig

  // 6. Y constraint fusion
  (r_y, y_eval) = Verify_RangeProof(proof.y_range_proof, bits=32)
  r_y_t = r_y[0..t_bits];  r_y_d = r_y[t_bits..t_bits+d_bits];  r_y_b = r_y[t_bits+d_bits]
  γ_r = Evaluate_MLE(vk.gamma, r_y_d)
  β_r = Evaluate_MLE(vk.beta,  r_y_d)
  sig_d = proof.openings.sigma_at_ryt * d_f
  expr  = vk.scale_γ * γ_r * (d_f * proof.openings.x_at_ry - proof.openings.sum_x_at_ryt)
        + vk.scale_β * β_r * sig_d
  lo_y = 2*expr - sig_d*(2*proof.openings.y_at_ry - 1)
  hi_y = sig_d*(2*proof.openings.y_at_ry + 1) - 1 - 2*expr
  Assert y_eval == (1 - r_y_b) * lo_y + r_y_b * hi_y

  // 7. Hyrax openings (binding intermediate commitments to random points)
  Hyrax_Verify(proof.sum_x_com, proof.openings.sum_x_at_rt, r_t, ...)
  Hyrax_Verify(proof.var_x_com, proof.openings.var_x_at_rt, r_t, ...)
  Hyrax_Verify(io_coms.x_com,   proof.openings.x_at_rt_rmean, (r_t, r_d_mean), ...)
  Hyrax_Verify(io_coms.x_com,   proof.openings.x_at_rt_rvar,  (r_t, r_d_var),  ...)
  Hyrax_Verify(proof.var_x_com, proof.openings.var_x_at_rsig,  r_sig_t, ...)
  Hyrax_Verify(proof.sigma_com, proof.openings.sigma_at_rsig,  r_sig_t, ...)
  Hyrax_Verify(io_coms.x_com,   proof.openings.x_at_ry,  (r_y_t, r_y_d), ...)
  Hyrax_Verify(io_coms.y_com,   proof.openings.y_at_ry,  (r_y_t, r_y_d), ...)
  Hyrax_Verify(proof.sum_x_com, proof.openings.sum_x_at_ryt, r_y_t, ...)
  Hyrax_Verify(proof.sigma_com, proof.openings.sigma_at_ryt,  r_y_t, ...)

  return OK
```

### 4.3 Linear Attention Verifier

```
Function Verify_LinearAttention(proof, inst, transcript):
  // 1. Verify phi(Q) lookup (Lasso, per chunk)
  Verify_Lasso(proof.q_lasso, inst.q_lasso, lasso_params)

  // 2. Verify phi(K) lookup (Lasso, per chunk)
  Verify_Lasso(proof.k_lasso, inst.k_lasso, lasso_params)

  // 3. Context sumcheck: C[i][j] = sum_t phi_K[t][i] * V[t][j]
  Absorb(transcript, "x_com", io_coms.phi_k_com)
  ...
  r_t = ChallengeVec(transcript, t_bits, "context_rt")
  (r_d_ctx, final_ctx) = Verify_Sumcheck(proof.context_sumcheck, ...)
  Assert final_ctx == proof.openings.phi_k_eval * proof.openings.v_eval
  Hyrax_Verify(phi_k_com, phi_k_eval, ...)
  Hyrax_Verify(v_com,     v_eval,     ...)

  // 4. Output sumcheck: out[t][j] = sum_i phi_Q[t][i] * C[i][j]
  r_t2 = ChallengeVec(transcript, t_bits, "out_rt")
  (r_d_out, final_out) = Verify_Sumcheck(proof.out_sumcheck, ...)
  Assert final_out == proof.openings.phi_q_eval * proof.openings.ctx_eval
  Hyrax_Verify(phi_q_com, phi_q_eval, ...)
  Hyrax_Verify(ctx_com,   ctx_eval,   ...)

  return OK
```

### 4.4 FFN Verifier

```
Function Verify_FFN(proof, inst, vk, transcript):
  // 1. First projection: M = X * W1
  r_t = ChallengeVec(transcript, t_bits, "ffn_rt")
  (r_d_m, final_m) = Verify_Sumcheck(proof.m_sumcheck, ...)
  Assert final_m == proof.openings.x_eval * proof.openings.w1_eval
  Hyrax_Verify(x_com,  x_eval,  ...)
  Hyrax_Verify(w1_com, w1_eval, ...)

  // 2. Activation: A = phi(M) via Lasso
  Verify_Lasso(proof.activation_lasso, inst.activation_lasso, lasso_params)

  // 3. Second projection: out = A * W2
  (r_d_out, final_out) = Verify_Sumcheck(proof.out_sumcheck, ...)
  Assert final_out == proof.openings.a_eval * proof.openings.w2_eval
  Hyrax_Verify(a_com,  a_eval,  ...)
  Hyrax_Verify(w2_com, w2_eval, ...)

  return OK
```

### 4.5 Range Proof Verifier

```
Function Verify_RangeProof(proof, num_vars, bits=32, transcript):
  // 1. Sumcheck to bind the virtual value array V to a random point r_v
  (r_v, final_v) = Verify_Sumcheck(proof.sumcheck, proof.claim_v, num_vars)
  Assert final_v == sum_{c} (2^(c*CHUNK_BITS)) * proof.chunk_evals[c]
  // (algebraic reconstruction from chunk evaluations)

  // 2. Check chunk commitments
  for c in 0..num_chunks:
    Hyrax_Verify(proof.chunk_coms[c], proof.chunk_evals[c], r_v, ...)

  // 3. LogUp multiplicity check: each chunk value appears in [0, 2^CHUNK_BITS)
  // (Verified by a GKR / grand-product argument against the identity table)
  ...

  return (r_v, proof.claim_v)
```

---

## 5. Security Analysis

### 5.1 Completeness

If the prover honestly executes the model and constructs all witnesses correctly, all sumcheck checks pass with probability 1 (the polynomial identities hold exactly over $\mathbb{F}_r$).

### 5.2 Soundness

**Sumcheck soundness:** By the Schwartz–Zippel lemma, a cheating prover can make a false degree-2 round polynomial pass with probability at most $2/|\mathbb{F}_r| \approx 2^{-253}$ per round. Over $n$ rounds: $2n/|\mathbb{F}_r|$.

**Hyrax soundness:** Under the discrete-log assumption in G1, Hyrax is computationally binding: the prover cannot open a committed polynomial to two different values at the same point.

**Lasso soundness:** Given sumcheck soundness and PCS binding, the only remaining attack is for the prover to claim a table evaluation inconsistent with the committed table. This is prevented by Hyrax binding.

**Range proof soundness:** The multiplicity check ensures the prover cannot supply chunk values outside $[0, 2^{16})$; combined with PCS binding this prevents false range claims.

**Constraint fusion:** The Fiat-Shamir transcript ensures all per-layer proofs are bound to the same random coins; forging any single sub-proof invalidates the entire transcript with overwhelming probability.

**Note on random-entry reduction:** Steps 3–4 of the attention circuit currently audit only one entry per matrix product. This is sound only if the verifier's challenge is truly random (Fiat-Shamir in the random-oracle model). For stronger soundness, one can batch over multiple random entries.

### 5.3 Zero Knowledge

The current implementation is **not zero-knowledge**: the prover reveals model weights and intermediate activations in the clear. Zero-knowledge can be added by:

- Committing to all witness polynomials before revealing evaluations.
- Applying a BlindFold-style ZK layer (Pedersen commitments + Nova folding over the sumcheck transcript) to hide all intermediate claims.

---

## 6. Fixed-Point Encoding

All real-valued tensors are represented as integers scaled by a factor $s$:

$$x_{\mathbb{F}} = \left\lfloor \frac{x}{s} \right\rfloor \bmod r \in \mathbb{F}_r$$

Arithmetic in $\mathbb{F}_r$ then simulates fixed-point integer arithmetic as long as no intermediate value overflows the field modulus. For $B$-bit activations and $k_{\max}$-bit weights, the maximum intermediate value in a matrix product is $T \cdot d_h \cdot 2^B \cdot 2^{k_{\max}}$, which must be kept below $r \approx 2^{254}$.

The Python training pipeline uses the same quantization so the exported integer tables and weights exactly match the field arithmetic in the Rust prover. This **eliminates the train–prove gap**.

---

## 7. Complexity Summary

Let $L$ = layers, $T$ = sequence length, $d$ = embedding dimension, $d_{ff}$ = FFN width, $m$ = bits per chunk, $c$ = chunk count, $N_{qk} = T \cdot d$ = total $\phi$ queries.

| Component | Prover time | Verifier time | Proof size |
|-----------|-------------|---------------|------------|
| Hyrax commit (size $N$) | O($N$) G1 MSM | — | O($\sqrt{N}$) G1 points |
| Hyrax open | O($\sqrt{N}$) | O($\sqrt{N}$) G1 + O($\log N$) $\mathbb{F}$ | O($\sqrt{N}$) $\mathbb{F}$ elems |
| Sumcheck ($n$ vars) | O($n \cdot 2^n$) $\mathbb{F}$ | O($n$) $\mathbb{F}$ | O($n$) $\mathbb{F}$ elems |
| Lasso (one sub-table) | O($N_{qk} \cdot 2^m + m \cdot 2^m$) | O($N_{qk} \cdot m$) | O($m$) $\mathbb{F}$ elems |
| LayerNorm (per block) | 2 sumchecks + 2 range proofs | O($d$) + O($\log T$) | O($\sqrt{Td}$) |
| Projection (per matrix) | O($T \cdot d$) | O($\log(Td)$) | O($\sqrt{Td}$) |
| Linear attention (per block) | O($T d^2$) | O($d \log T$) | O($d \log T$) |
| FFN (per block) | O($T d \cdot d_{ff}$) | O($\log(T d \cdot d_{ff})$) | O($\sqrt{T d_{ff}}$) |
| **Per block total** | O($T d^2 + c N_{qk} 2^m$) | O($d \log T$) | O($d \log T + c \cdot 2^m$) |
| **All $L$ blocks** | $\times L$ | $\times L$ | $\times L$ |

---

## 8. Comparison with Related Work

| System | Attention | Activation | PCS | ZK |
|--------|-----------|------------|-----|----|
| **zkGPT** | Softmax (approx.) | Lookup table | Plonky2 | ✓ |
| **zkLLM** | Softmax (tlookup) | Structured lookup | Custom IOP | ✗ |
| **π-Former** | Linear (exact) | Learned structured lookup | Hyrax | roadmap |

Key advantages of π-Former:

- **Exact computation:** linear attention is not an approximation of softmax; it is a different (learnable) attention mechanism trained to be useful and provable.
- **End-to-end co-design:** the model's weights and activation tables are trained to match the exact integer/field arithmetic used in the circuit, eliminating approximation errors.
- **Transparent setup:** Hyrax requires no trusted setup; all parameters are derived from a public hash function.
- **Extensible:** the lookup decomposition depth $c$ and chunk size $m$ are configurable trade-offs between model expressivity and proof cost.

---

## 9. Planned Extensions

### 9.1 Memory-Consistency Check for Lasso

The current Lasso argument proves that query outputs are consistent with a *claimed* table. To prevent a dishonest prover from using a different table per query, add an offline memory-consistency argument (Spice / Lasso grand-product check) that ties all queries back to a single committed table.

### 9.2 IVC for Autoregressive Generation

For autoregressive (token-by-token) inference, prove each step's attention incrementally using Nova-style incremental verifiable computation (IVC). The running context matrix $C = \phi(K)^\top V$ is the natural accumulator state.

### 9.3 Sparse Attention Exploitation

If the trained model learns a structured sparse pattern in $\phi(Q)\phi(K)^\top$ (e.g., local windows), the sumcheck over $T$ can be restricted to non-zero terms, reducing prover cost from O($T$) to O(nnz).

### 9.4 Shared Table Argument

If multiple layers share the same $\phi$ tables (weight tying), prove table correctness once and reference it from all layers via a single Lasso instance, reducing the per-layer proof cost for lookups.

### 9.5 Zero-Knowledge Layer

Add Pedersen-commitment-based blinding to all witness polynomials before the Fiat-Shamir transcript to achieve full zero-knowledge without changing the verifier's algebraic checks.

---

## 10. Implementation Notes

### Transcript Fiat-Shamir Convention

All challenges are derived from SHA3-256 with domain-separation labels. The hasher state is advanced by feeding each finalized hash back into the running state:

```
state ← SHA3-256(label ‖ data ‖ state_hash)
challenge ← F::from_le_bytes_mod_order(SHA3-256(state ‖ label))
```

The `--transcript-label` flag in the CLI must match between `prove` and `verify` (default: `"piformer"`).

### Bit-Ordering Convention

`DenseMLPoly::fix_first_variable(r)` fixes the highest-index variable first (bit $n-1$ of the evaluation index). Therefore sumcheck challenge $r_j$ corresponds to bit $n-1-j$ of the final evaluation index. When constructing $L_k(\mathbf{r})$ from bit decompositions of chunk indices (naturally LSB-first), the challenge vector must be reversed. See `lookup/lasso.rs` for the explicit fix.

### Hyrax Generator Derivation

Generators are derived deterministically as:

```rust
g_i = SHA3("piformer-hyrax-gen" ‖ i_as_le64) · G
```

where $G$ is the BN254 G1 generator. This provides "nothing-up-my-sleeve" independence between generators.

### Field Arithmetic

All field operations use the `ark-bn254` crate with the `ark-ff 0.4` API. The field element type `F = ark_bn254::Fr` is aliased in `src/field.rs`. The `PrimeField` trait provides `into_bigint()` for extracting limbs when needed for range decomposition.

### Prover / Verifier Key Separation

The CLI separates proving keys (`.pk`) from verifying keys (`.vk`):

- **Proving key** contains Hyrax G1 commitments **plus** the raw weight vectors (needed by the prover to compute witness-weight inner products).
- **Verifying key** contains only the G1 commitments (the prover never sends raw weights to the verifier).

The codec writes a `has_weights: bool` flag per sub-key. On `.vk` load, raw weight fields are stubbed with empty vectors; the verifier never accesses them.

---

## 11. File Formats

### Weights JSON (`*.json`)

```json
{
  "num_blocks": 1,
  "d_model": 2,
  "d_ff": 4,
  "vocab_size": 2,
  "blocks": [{
    "ln1_gamma": ["0x0000...0002", "0x0000...0002"],
    "ln1_beta":  ["0x0000...0005", "0x0000...0005"],
    "q_w": [["0x0000...0000", ...], ...],
    ...
  }],
  "final_ln_gamma": [...],
  "final_ln_beta":  [...],
  "lm_head_w": [...]
}
```

All field elements are serialized as lowercase hex strings with `0x` prefix, zero-padded to 64 hex digits.

### Witness JSON (`*.json`)

```json
{
  "seq_len": 2,
  "d_model": 2,
  "d_ff": 4,
  "vocab_size": 2,
  "lasso_sigma": 2,
  "x_in": [...],
  "block_witnesses": [{ ... }],
  "final_ln_wit": { ... },
  "lm_head_wit": { ... },
  "inst_attn": {
    "seq_len": 2,
    "d_head": 2,
    "q_lasso": { "tables": [...], "query_indices": [...], "outputs": [...], "bits_per_chunk": 4 },
    "k_lasso": { ... }
  },
  "inst_ffn": { "activation_lasso": { ... } }
}
```

### Binary Proving Key (`.pk`)

```
Magic:   b"PFMR_PK\0"  (8 bytes)
Version: u8             (1 byte)
num_blocks: u64 LE      (8 bytes)
seq_len:    u64 LE
d_model:    u64 LE
vocab_size: u64 LE
final_ln_vk: [LayerNormVK]
lm_head_pk:  [ProjectionPK with weights]
block_pks:   [num_blocks × TransformerBlockPK with weights]
```

Each field element occupies 32 bytes (uncompressed, `Compress::No`). Each G1Affine occupies 33 bytes (compressed). Lengths are prefixed as `u64 LE`.

### Binary Verifying Key (`.vk`)

Same layout as `.pk` but all `ProjectionPK` / `FFN_PK` weight fields are replaced by an empty vector (flag `has_weights = false`).

### Binary Proof Bundle (`.bin`)

```
Magic:       b"PFMR_PR\0"  (8 bytes)
Version:     u8
lasso_sigma: u64 LE
inst_attn:   [LinearAttentionInstance]
inst_ffn:    [FFNInstance]
proof:       [TransformerModelProof]
```

The proof contains one `TransformerBlockProof` per block, each holding all sub-proofs and intermediate Hyrax commitments needed by the verifier.
