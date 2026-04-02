# π-Former: Technical Design

> Succinct ZK Proofs of Transformer Inference via Structured Lookup Attention

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [ZK-Friendly Model Architecture](#2-zk-friendly-model-architecture)
3. [Proof System Primitives](#3-proof-system-primitives)
4. [Protocol Pseudocode](#4-protocol-pseudocode)
5. [Soundness Analysis](#5-soundness-analysis)
6. [Fixed-Point Encoding](#6-fixed-point-encoding)
7. [Complexity Summary](#7-complexity-summary)
8. [Comparison with Related Work](#8-comparison-with-related-work)
9. [Planned Extensions](#9-planned-extensions)
10. [Implementation Notes](#10-implementation-notes)
11. [File Formats](#11-file-formats)

---

## 1. Problem Statement

Let $\mathcal{M}$ be a transformer model with $L$ layers, embedding dimension $d$, FFN width $d_{ff}$, and sequence length $T$. Given a public input sequence $\mathbf{x} = (x_1, \ldots, x_T)$ and the model's frozen weights $\theta$, we want a succinct, non-interactive argument of knowledge:

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
X_norm1  = LayerNorm(X_in,     γ₁, β₁)
Q, K, V  = X_norm1 · W_{Q,K,V}          (batched projection)
Out_attn = φ(Q)(φ(K)ᵀV) · W_O           (linear attention + output projection)
X_mid    = X_in + Out_attn               (residual 1, homomorphic)
X_norm2  = LayerNorm(X_mid,    γ₂, β₂)
Out_ffn  = FFN(X_norm2)                  (feed-forward network)
X_out    = X_mid + Out_ffn               (residual 2, homomorphic)
```

After all $L$ blocks, a final LayerNorm and language-model head projection produce the output logits.

### 2.2 Linear Attention

We replace the standard scaled dot-product attention

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

with the kernel (linear) attention formulation

$$\text{Attn}(Q, K, V) = \phi(Q)\,[\phi(K)^\top V] \tag{1}$$

where $\phi : \mathbb{F}^{d_h} \to \mathbb{F}^{d_h}$ is an element-wise feature map.

**Why this is SNARK-friendly:**

- The context matrix $C = \phi(K)^\top V \in \mathbb{F}^{d_h \times d_h}$ depends only on the key-value sequence. It is computed once and reused for all queries.
- There are no exponentials, no row-wise divisions with unknown denominators, and no $\max$ operations.
- Associativity gives O($T d^2$) complexity instead of O($T^2 d$), matching the proof cost.

### 2.3 Structured Lookup Activation

The kernel function $\phi$ is implemented as a **learnable, additively decomposed lookup table**:

$$\phi(x) = \sum_{i=0}^{c-1} T_i\!\left[\left\lfloor \frac{x_{\text{int}}}{2^{im}} \right\rfloor \bmod 2^m \right] \tag{2}$$

where:
- $x_{\text{int}} = \text{clamp}(\lfloor x / s \rfloor, 0, 2^B - 1)$ is the quantized input ($B$ bits, scale $s$).
- $B = c \cdot m$: total bits split into $c$ chunks of $m$ bits each.
- $T_0, \ldots, T_{c-1} \in \mathbb{F}^{2^m}$ are learnable sub-tables.

**Decomposition rationale:** A flat lookup for a $B$-bit input requires committing to a table of size $2^B$. The additive decomposition reduces this to $c$ tables of size $2^m = 2^{B/c}$, giving total commitment cost $O(c \cdot 2^{B/c})$ instead of $O(2^B)$. For $B=16, c=2$: 512 commitments instead of 65 536.

This structure exactly matches the **Lasso** lookup argument (§3.5), which exploits additive decompositions of precisely this form.

**Initialization:** Sub-tables are initialized to approximate $\text{GeLU}(x)/c$ so that their sum closely approximates GeLU at the start of training, enabling stable convergence.

### 2.4 Power-of-Two Weight Quantization

Projection matrices $W_Q, W_K, W_V, W_O$ and FFN weight matrices are constrained to entries in

$$\mathcal{W} = \{0\} \cup \{\pm 2^k \mid k = 0, 1, \ldots, k_{\max}\}$$

A matrix-vector product $y = Wx$ with $W \in \mathcal{W}^{m \times n}$ in the circuit requires only additions and multiplications by field constants (left-shifts). There are no general field multiplications, so this is **free** in the sumcheck constraint system.

**Training:** A straight-through estimator (STE) is used. The forward pass applies nearest-neighbor quantization to $\mathcal{W}$; the backward pass uses the identity function in place of the quantizer's zero gradient.

---

## 3. Proof System Primitives

### 3.1 Field

We work over the BN254 scalar field $\mathbb{F}_r$ (prime $r \approx 2^{254}$). All model values (weights, activations, hidden states) are represented as fixed-point integers encoded as field elements.

### 3.2 Hyrax Polynomial Commitment Scheme

π-Former uses **Hyrax**, a transparent (no trusted setup) multilinear PCS based on the discrete-log assumption in the BN254 G1 group.

For a multilinear polynomial with $2^n = 2^{\nu+\sigma}$ evaluations, arranged as a $2^{\nu} \times 2^{\sigma}$ matrix $M$:

**Commit:**
$$C_i = \text{MSM}(\mathbf{g},\, M[i]) \quad \text{for each row } i \in [2^\nu]$$

where $\mathbf{g} = (g_0, \ldots, g_{2^\sigma - 1})$ are "nothing-up-my-sleeve" G1 generators derived via $g_j = \mathsf{SHA3}(\texttt{"piformer-hyrax-gen"} \| j) \cdot G$.

**Open** at point $r = (r_L \| r_R)$ with $r_L \in \mathbb{F}^\nu$, $r_R \in \mathbb{F}^\sigma$:
$$w'_j = \sum_{i \in [2^\nu]} L_i(r_L) \cdot M[i][j]$$
where $L_i$ are the multilinear Lagrange basis polynomials over $r_L$.

**Verify:**
1. $\sum_i L_i(r_L) \cdot C_i \stackrel{?}{=} \text{MSM}(\mathbf{g},\, w')$ — homomorphic commitment check.
2. $\langle R(r_R),\, w' \rangle \stackrel{?}{=} v$ — inner product check against claimed evaluation $v$.

**Multi-point batch:** For $K$ opening claims on (possibly different) commitments, a Fiat-Shamir challenge $\eta \stackrel{\$}{\leftarrow} \mathbb{F}_r$ is drawn and:
$$w'_{\text{agg}} = \sum_{i=1}^K \eta^{i-1} \cdot w'_i, \quad C_{\text{agg}} = \sum_{i=1}^K \eta^{i-1} \cdot C_{r_L,i}$$
Both checks reduce to a single MSM pair. This is realised via a **deferred batch accumulator** that collects all claims during verification and finalises with 2 MSMs per batch group.

**Complexity:** Prover O($\sqrt{N}$) G1 MSM; verifier O($\sqrt{N}$) G1 + O($\log N$) field; proof size O($\sqrt{N}$) field elements.

### 3.3 Dense Multilinear Polynomials

A multilinear polynomial $\tilde{f} : \mathbb{F}_r^n \to \mathbb{F}_r$ over $n$ variables is uniquely determined by its $2^n$ evaluations on $\{0,1\}^n$:

$$\mathsf{evals}[i] = \tilde{f}\!\left(\mathsf{bit}_{n-1}(i), \ldots, \mathsf{bit}_0(i)\right)$$

**Key operations:**
- **Evaluate at $r \in \mathbb{F}^n$:** Repeated half-folding. Cost O($2^n$) field ops.
- **Fix first variable to $r$:** $\mathsf{new}[i] = (1-r)\cdot\mathsf{evals}[i] + r\cdot\mathsf{evals}[i + 2^{n-1}]$. Cost O($2^n$).
- **Equality polynomial** $\widetilde{\mathsf{eq}}(r, x) = \prod_{i}(r_i x_i + (1-r_i)(1-x_i))$.

### 3.4 Sumcheck Protocol

We use the sumcheck protocol for the statement

$$H = \sum_{x \in \{0,1\}^n} f(x) \cdot g(x) \tag{3}$$

where $f, g$ are dense MLEs (degree-2 round polynomials).

**Round $i$ ($i = 1,\ldots,n$):**

1. Prover sends univariate $g_i(X)$ evaluated at $X \in \{0, 1, 2\}$.
2. Verifier checks $g_i(0) + g_i(1) = H_{i-1}$ and samples $r_i \stackrel{\$}{\leftarrow} \mathbb{F}_r$ (Fiat-Shamir).
3. Set $H_i = g_i(r_i)$.

At the end, the verifier holds a single opening claim $f(r) \cdot g(r) = H_n$, checked via two Hyrax openings.

**Variants used in π-Former:**
- **Quadratic (degree 2):** Products of two MLEs — used for projections and attention.
- **Cubic (degree 3):** Products of three MLEs — used inside LayerNorm variance.
- **Multi-batched:** $H = \sum_k \lambda_k \sum_x f_k(x) \cdot g_k(x)$, single shared sumcheck — used for batched QKV projection and constraint fusion in LayerNorm.

**Complexity:** Prover O($n \cdot 2^n$) field operations; verifier O($n$) field operations plus PCS openings.

### 3.5 Lasso Lookup Argument

Given sub-table $T_k \in \mathbb{F}_r^{2^m}$ and queries $\{(\mathsf{idx}_j, v_j)\}_{j=1}^N$ with $v_j = T_k[\mathsf{chunk}_k(\mathsf{idx}_j)]$, correctness is proved via a batched MLE sumcheck.

**Step 1 — Commit.** Represent $T_k$ as a dense MLE $\widetilde{T}_k$ over $m$ variables.

**Step 2 — Batch.** Sample $\rho \stackrel{\$}{\leftarrow} \mathbb{F}_r$ (Fiat-Shamir). Define the selector polynomial:

$$L_k(x) = \sum_{j=1}^N \rho^j \cdot \widetilde{\mathsf{eq}}\!\left(\mathsf{bin}(\mathsf{ch}_{k,j}),\, x\right) \tag{4}$$

where $\mathsf{ch}_{k,j} = \mathsf{chunk}_k(\mathsf{idx}_j)$.

**Step 3 — Sumcheck.** Run the degree-2 sumcheck for

$$\sum_{x \in \{0,1\}^m} \widetilde{T}_k(x) \cdot L_k(x) = \sum_{j=1}^N \rho^j \cdot T_k[\mathsf{ch}_{k,j}] \tag{5}$$

**Step 4 — Open.** After sumcheck terminates at random point $\mathbf{r}$, prover opens $\widetilde{T}_k(\mathbf{r})$ via Hyrax. Verifier checks $\widetilde{T}_k(\mathbf{r}) \cdot L_k(\mathbf{r}) = H_n$ where $L_k(\mathbf{r})$ is recomputed from public queries.

**Complexity (per sub-table):** Prover O($N \cdot 2^m + m \cdot 2^m$); verifier O($N \cdot m$) plus O($m$) sumcheck.

### 3.6 Global Batched Range Proof

The range proof proves that field elements lie in $[0, 2^{32})$ using a chunked Lasso approach. Because all range constraints in a transformer block check the same table $[0, 2^{32})$, the multiplicity commitments are **globally batched** across all witnesses.

**Per-witness setup.** Each witness $V \in \mathbb{F}^{2^n}$ is split into two 16-bit chunk arrays:
$$V[i] = V_{lo}[i] + 2^{16} \cdot V_{hi}[i]$$

**Two-phase protocol (for a batch of $B$ witnesses):**

*Phase 1 — Commit all chunks + shared multiplicity:*
1. For each witness $b$: commit $V^{(b)}_{lo}$, $V^{(b)}_{hi}$ via Hyrax → chunk commitments $\mathsf{cc}^{(b)}_0, \mathsf{cc}^{(b)}_1$.
2. Merge all chunk arrays into a global multiplicity array $m$: $m[v]$ counts the total occurrences of value $v$ across all witnesses.
3. Commit $m$ via Hyrax → single shared $m_{\mathsf{com}}$.

*Phase 2 — Per-witness sumcheck and openings:*
4. For each witness $b$: absorb claim $V^{(b)}(1,\ldots,1)$, run sumcheck to bind $V^{(b)}$ to random point $r_v^{(b)}$.
5. Verify chunk reconstruction: $V^{(b)}(r_v^{(b)}) = \mathsf{cc}^{(b)}_0(r_v^{(b)}) + 2^{16} \cdot \mathsf{cc}^{(b)}_1(r_v^{(b)})$ (Hyrax batch open).

*Shared multiplicity check:*
6. Sample $r_m \stackrel{\$}{\leftarrow} \mathbb{F}^{16}$; open $m(r_m)$ via Hyrax.
7. LogUp identity check: $\sum_{b,c} \mathsf{cc}^{(b)}_c(r_m) = m(r_m) \cdot T_{\mathsf{id}}(r_m)$ where $T_{\mathsf{id}}[i] = i$.

**Key saving:** Instead of $B$ independent $m_{\mathsf{com}}$ commitments ($B \times \sqrt{2^{16}}$ MSMs), there is exactly one $m_{\mathsf{com}}$. For $B=6$ (two LayerNorms per block, two range proofs each, plus final LayerNorm's two): saves 5 × $\sqrt{2^{16}}$ = 1280 MSMs.

### 3.7 LayerNorm Circuit

LayerNorm is proved without any division gates using **constraint fusion**:

**Step 1 — Mean sumcheck.** Prove $\mathsf{sum\_x}[i] = \sum_j x[i][j]$ at a random row evaluation point $r_t$:
$$\mathsf{sum\_x\_mle}(r_t) = \sum_{j \in \{0,1\}^{d_{\mathsf{bits}}}} x_{\mathsf{col}}(j)$$
where $x_{\mathsf{col}}$ is $x_{\mathsf{mle}}$ with row variables fixed to $r_t$.

**Step 2 — Variance sumcheck.** Prove $\mathsf{var\_x}[i] = \sum_j (d \cdot x[i][j] - \mathsf{sum\_x}[i])^2$ at $r_t$:
$$\mathsf{var\_x\_mle}(r_t) = \sum_{j \in \{0,1\}^{d_{\mathsf{bits}}}} h(j)^2, \quad h(j) = d \cdot x_{\mathsf{col}}(j) - \mathsf{sum\_x\_mle}(r_t)$$
(Cubic sumcheck over three copies of $h$.)

**Step 3 — Sigma range proof.** Prove $\sigma[i]$ is the integer floor square-root of $\mathsf{var\_x}[i]$:
$$\mathsf{var\_x}[i] - (d \cdot \sigma[i])^2 \geq 0 \quad \text{and} \quad (d \cdot \sigma[i] + d)^2 - 1 - \mathsf{var\_x}[i] \geq 0$$
Both residuals are range-checked via the global batched range proof (§3.6). The verifier reconstructs $\sigma(r_\sigma)$ from chunk evaluations: $\sigma(r_\sigma) = \mathsf{cc}_0(r_\sigma) + 2^{16} \cdot \mathsf{cc}_1(r_\sigma)$.

**Step 4 — Y constraint fusion.** Prove the LayerNorm output $y[i][j]$ satisfies:
$$\gamma_j \cdot (d \cdot x[i][j] - \mathsf{sum\_x}[i]) + \beta_j \cdot d \cdot \sigma[i] \approx d \cdot \sigma[i] \cdot y[i][j]$$
(up to integer rounding), verified at a single random point $(r_{y_t}, r_{y_d})$ via another range proof over the lo/hi residuals. The verifier evaluates $\gamma$ and $\beta$ directly from the public VK in O($d$) operations.

**All constraints are verified at a single random point** — the verifier runs O(1) equations, not O($T \cdot d$) loops.

### 3.8 Projection Circuit

Proves $Y = X \cdot W$ where $W$ is a committed weight matrix with entries in $\mathcal{W}$.

A single random entry $(r_t, r_d)$ is selected via Fiat-Shamir; a degree-2 sumcheck over the contraction index $k$ proves:
$$Y(r_t, r_d) = \sum_{k \in \{0,1\}^{k_{\mathsf{bits}}}} X(r_t, k) \cdot W(k, r_d)$$

Hyrax openings bind $X$, $Y$, and $W$ to their claimed MLE evaluations.

**Batched QKV variant:** A single sumcheck proves three projections simultaneously using Fiat-Shamir scalars $\lambda, \mu$:
$$\lambda \cdot \alpha_Q Y_Q(r_t, r_d) + \mu \cdot \alpha_K Y_K(r_t, r_d) + \alpha_V Y_V(r_t, r_d) = \sum_k X(r_t, k) \cdot (\lambda \alpha_Q W_Q(k,r_d) + \mu \alpha_K W_K(k,r_d) + \alpha_V W_V(k,r_d))$$
All three projections share the same sumcheck challenge $r_k$, reducing the sumcheck cost from $3\times$ to $1\times$.

### 3.9 Linear Attention Circuit

Proves the following claims for a single attention head:

| Step | Statement | Protocol |
|------|-----------|----------|
| 1 | $\Phi_Q[t][d] = \phi(Q[t][d])$ | Lasso (§3.5) per chunk |
| 2 | $\Phi_K[t][d] = \phi(K[t][d])$ | Lasso per chunk |
| 3 | $C[i][j] = \sum_t \Phi_K[t][i] \cdot V[t][j]$ | Degree-2 sumcheck over $t$ |
| 4 | $\mathsf{Out}[t][j] = \sum_i \Phi_Q[t][i] \cdot C[i][j]$ | Degree-2 sumcheck over $i$ |

Steps 3–4 use a random-entry reduction. The verifier draws $(r_{out}, r_i)$ via Fiat-Shamir and audits a single entry per matrix product.

**GKR backward fusion with O-projection:** The output projection $\mathsf{Out\_inner} = \mathsf{Out} \cdot W_O$ is proved *before* the attention sub-prover. The O-projection prover opens $\mathsf{Out\_inner}$ at a random point $(r_x, r_y)$ and returns this claim. The attention prover receives this as an *external claim* on $\mathsf{Out}$, eliminating the $\mathsf{out\_inner\_com}$ commitment entirely. Two independent sumchecks over different sub-problems share the same binding point, and a cheating prover who fakes either one would fail the other.

### 3.10 FFN Circuit

Proves $\mathsf{Out} = \phi(X \cdot W_1) \cdot W_2$:

**GKR backward ordering:**
1. **Activation** $A = \phi(M)$: Lasso lookup argument runs *first* (commits activation outputs to transcript).
2. **Second projection** $\mathsf{Out} = A \cdot W_2$: degree-2 sumcheck at a random entry. Verifier checks $A(r_k)$ from the Lasso outputs MLE — no separate $A_{\mathsf{com}}$ needed.
3. **First projection** $M = X \cdot W_1$: degree-2 sumcheck at a random entry. $M_{\mathsf{com}}$ is kept (field values of $M$ can be negative, unlike Lasso query indices).

This ordering eliminates $A_{\mathsf{com}}$ while maintaining transcript soundness.

### 3.11 Multi-Claim Combine Protocol

When the same polynomial $f$ is opened at multiple points $z_1, \ldots, z_k$ (e.g., $V_{\mathsf{com}}$ is claimed at both the V-projection output and the attention V-input), they are reduced to a single Hyrax opening via a GKR-style sumcheck:

1. Sample weights $\rho_1, \ldots, \rho_k \stackrel{\$}{\leftarrow} \mathbb{F}_r$ (Fiat-Shamir).
2. Build $G(x) = \sum_i \rho_i \cdot \widetilde{\mathsf{eq}}(z_i, x)$.
3. Run sumcheck for $\sum_x G(x) \cdot f(x) = \sum_i \rho_i \cdot v_i$.
4. At terminal point $r^*$: open $f(r^*)$ via Hyrax. Verifier computes $G(r^*)$ locally in O($k \cdot n$) field operations.

---

## 4. Protocol Pseudocode

This section gives complete, formal pseudocode for the π-Former prover and verifier. Notation: $\mathsf{FS}$ denotes a Fiat-Shamir transcript operation; $\mathsf{Com}(v)$ denotes Hyrax commitment; $\mathsf{Open}(f, r)$ denotes a Hyrax proof of $f(r)$.

### 4.1 Setup

```
Algorithm Setup(model_weights θ, params):
  Input:
    θ = {W_Q, W_K, W_V, W_O, W_1, W_2, γ_ℓ, β_ℓ} for ℓ = 1..L,
        final_ln weights, lm_head weights
    params = (T, d, d_ff, d_h, c, m, bits_per_chunk=16, total_bits=32)

  // Commit all static weight matrices
  For each weight matrix W in θ:
    W_com ← Com(W)
  
  // Commit activation sub-tables (shared across layers if weight-tied)
  For k = 0 to c-1:
    T_k_com ← Com(T_k)       // sub-table for activation φ
  
  // Derive Hyrax generators deterministically
  g_j ← SHA3("piformer-hyrax-gen" ‖ j) · G,  j = 0..√N
  
  // Publish
  VK ← { W_com_Q, W_com_K, W_com_V, W_com_O, W_com_1, W_com_2,
          γ_com_ℓ, β_com_ℓ, T_k_com, g_0..g_{√N} }
  PK ← VK ∪ { raw weight arrays }
  return (PK, VK)
```

### 4.2 End-to-End Prover

```
Algorithm Prove(PK, witness W, instances inst):
  Input:
    PK          : proving key (weights + commitments)
    W           : { x_in, block_witnesses[0..L-1],
                    final_ln_wit, lm_head_wit }
    inst        : { attn_inst[0..L-1], ffn_inst[0..L-1] }
  Output:
    π           : TransformerModelProof

  // 1. Initialize Fiat-Shamir transcript
  FS.init("piformer")
  x_in_com ← Com(W.x_in)
  FS.absorb("x_in_com", x_in_com)

  x_cur_com ← x_in_com
  block_proofs ← []

  // 2. Prove each transformer block
  For ℓ = 0 to L-1:
    (block_π_ℓ, x_out_com_ℓ) ← ProveBlock(
        PK.blocks[ℓ], W.block_witnesses[ℓ],
        inst.attn_inst[ℓ], inst.ffn_inst[ℓ],
        x_cur_com, FS)
    block_proofs.append(block_π_ℓ)
    x_cur_com ← x_out_com_ℓ

  // 3. Final LayerNorm range batch (σ and y for final_ln)
  final_rw ← ComputeRangeWitnesses(W.final_ln_wit, PK.final_ln_vk)
  (final_rps, final_range_m, final_rvs) ← ProveRangeBatched(
      [final_rw.σ_witness, final_rw.y_witness], bits=32, FS)

  // 4. Prove final LayerNorm
  final_ln_io_coms ← { x_com: x_cur_com,
                        y_com: Com(W.final_ln_wit.y) }
  final_ln_π ← ProveLayerNorm(
      W.final_ln_wit, final_ln_io_coms, PK.final_ln_vk,
      σ_range=(final_rps[0], final_rvs[0]),
      y_range=(final_rps[1], final_rvs[1]),
      FS)

  // 5. Prove LM head projection
  lm_head_io_coms ← { x_com: final_ln_io_coms.y_com,
                       y_com: Com(W.lm_head_wit.y) }
  lm_head_π ← ProveProjection(
      PK.lm_head_pk, W.lm_head_wit, lm_head_io_coms, FS)

  // 6. Finalise six deferred Hyrax batch accumulators
  //    Each outputs one batched MSM challenge + proof
  ln_acc_t.finalize(FS)       // LayerNorm row openings
  ln_acc_td.finalize(FS)      // LayerNorm (row,col) openings
  proj_acc_w.finalize(FS)     // Projection weight openings
  proj_acc_b.finalize(FS)     // Projection bias openings
  lmh_acc_w.finalize(FS)      // LM head weight openings
  lmh_acc_b.finalize(FS)      // LM head bias openings

  // 7. Global Lasso batch (all φ(Q) and φ(K) across all blocks)
  all_lasso_π ← ProveLassoMulti(
      [inst.attn_inst[ℓ].q_lasso, inst.attn_inst[ℓ].k_lasso
       for ℓ in 0..L-1],
      PK.lasso_pk, FS)

  return TransformerModelProof {
    x_in_com, block_proofs,
    final_range_m, final_ln_π, lm_head_π,
    acc_proofs: [ln_t, ln_td, proj_w, proj_b, lmh_w, lmh_b],
    all_lasso_π
  }
```

### 4.3 Block Prover

```
Algorithm ProveBlock(pk, wit, attn_inst, ffn_inst, x_in_com, FS):
  Input:
    pk      : TransformerBlockProvingKey
    wit     : { ln1_wit, qkv_wit, o_proj_wit, attn_wit, ln2_wit, ffn_wit }
    x_in_com: Hyrax commitment to X_in ∈ F^{T×d}
  Output:
    (π, x_out_com)

  // ── PHASE 0: Global range batch for this block ──────────────────────────
  // Commit all chunk arrays and one shared m_com BEFORE any sumchecks
  rw1 ← ComputeRangeWitnesses(wit.ln1_wit, pk.ln1_vk)
  rw2 ← ComputeRangeWitnesses(wit.ln2_wit, pk.ln2_vk)
  (block_rps, block_range_m, block_rvs) ← ProveRangeBatched(
      [rw1.σ_witness, rw1.y_witness, rw2.σ_witness, rw2.y_witness],
      bits=32, FS)
  // Transcript now contains: cc^(0)..cc^(3), m_com (5 commitments)
  // block_rps[i] = RangeWitnessProof per witness (no m_com)
  // block_rvs[i] = random evaluation point r_v^(i) ∈ F^{t_bits + d_bits + 1}

  // ── PHASE 1: LayerNorm 1 ────────────────────────────────────────────────
  x_norm1_com ← Com(wit.ln1_wit.y)
  ln1_io_coms ← { x_com: x_in_com, y_com: x_norm1_com }
  ln1_π ← ProveLayerNorm(wit.ln1_wit, ln1_io_coms, pk.ln1_vk,
               σ_range=(block_rps[0], block_rvs[0]),
               y_range=(block_rps[1], block_rvs[1]), FS)

  // ── PHASE 2: Batched QKV Projections ────────────────────────────────────
  q_com ← Com(wit.qkv_wit.q)
  k_com ← Com(wit.qkv_wit.k)
  v_com ← Com(wit.qkv_wit.v)
  qkv_io_coms ← { x_com: x_norm1_com, q_com, k_com, v_com }
  (qkv_π, q_claim, k_claim, v_proj_claim, x_norm1_claim) ←
      ProveQKVProjections(pk.qkv_pk, wit.qkv_wit, qkv_io_coms, FS)
  // Single sumcheck for three projections; shared r_k across Q, K, V

  // ── PHASE 3: Output Projection (GKR forward, before attention) ──────────
  out_attn_com ← Com(wit.o_proj_wit.y)
  o_proj_io_coms ← { x_com: None,         // x deferred (GKR backward)
                      y_com: out_attn_com }
  (o_proj_π, o_y_claim, o_x_claim) ←
      ProveProjection(pk.o_proj_pk, wit.o_proj_wit, o_proj_io_coms, FS)
  // o_x_claim = Out_inner evaluated at (r_x, r_y): used as external claim

  // ── PHASE 4: Linear Attention ────────────────────────────────────────────
  attn_io_coms ← { q_com, k_com, v_com }
  (attn_π, attn_out_claim, attn_v_claim) ←
      ProveLinearAttention(wit.attn_wit, attn_inst,
          attn_io_coms, external_out_claim=o_x_claim, FS)
  // o_x_claim binds attention's Out to the same point as O-proj
  // No out_inner_com emitted (GKR fusion)

  // ── PHASE 5: Residual 1 (homomorphic, zero proof cost) ──────────────────
  x_mid_com ← HyraxAdd(x_in_com, out_attn_com)
  // x_mid_com = Com(X_in + Out_attn) by Hyrax linearity; no proof needed

  // ── PHASE 6: LayerNorm 2 ────────────────────────────────────────────────
  x_norm2_com ← Com(wit.ln2_wit.y)
  ln2_io_coms ← { x_com: x_mid_com, y_com: x_norm2_com }
  ln2_π ← ProveLayerNorm(wit.ln2_wit, ln2_io_coms, pk.ln2_vk,
               σ_range=(block_rps[2], block_rvs[2]),
               y_range=(block_rps[3], block_rvs[3]), FS)

  // ── PHASE 7: FFN (GKR backward: Lasso → Y-proj → M-proj) ───────────────
  out_ffn_com ← Com(wit.ffn_wit.y)
  ffn_io_coms ← { x_com: x_norm2_com, y_com: out_ffn_com }
  (ffn_π, ffn_y_claim, ffn_x_claim) ←
      ProveFFN(pk.ffn_pk, wit.ffn_wit, ffn_inst, ffn_io_coms, FS)
  // a_com eliminated: Lasso runs first, A(r_k) verified from Lasso MLE

  // ── PHASE 8: Multi-claim Combine for V ──────────────────────────────────
  (v_combine_π, v_r, v_eval) ←
      ProveCombine(wit.qkv_wit.v, v_com,
                   [v_proj_claim, attn_v_claim], FS)

  // ── PHASE 9: Batch 7 Hyrax openings ─────────────────────────────────────
  // Openings deferred to block-level MSM batch (multi-point verifier)
  HyraxBatchOpen([
    (q_com,        q_claim.eval,       q_claim.point),
    (k_com,        k_claim.eval,       k_claim.point),
    (x_norm1_com,  x_norm1_claim.eval, x_norm1_claim.point),
    (out_attn_com, o_y_claim.eval,     o_y_claim.point),
    (x_norm2_com,  ln2_x_claim.eval,   ln2_x_claim.point),
    (out_ffn_com,  ffn_y_claim.eval,   ffn_y_claim.point),
    (v_com,        v_eval,             v_r)
  ], FS)

  // ── PHASE 10: Residual 2 (homomorphic) ───────────────────────────────────
  x_out_com ← HyraxAdd(x_mid_com, out_ffn_com)

  return (TransformerBlockProof {
    block_range_m,
    ln1_π, qkv_π, o_proj_π, attn_π, ln2_π, ffn_π, v_combine_π,
    x_norm1_com, q_com, k_com, v_com, out_attn_com, x_norm2_com, out_ffn_com
  }, x_out_com)
```

### 4.4 Range Proof Prover (Batched)

```
Algorithm ProveRangeBatched(witnesses[0..B-1], bits, FS):
  Input:
    witnesses[b] = { values: V^(b) ∈ F^{2^n_b} }   // B witnesses
    bits = 32, CHUNK_BITS = 16
  Output:
    (rps[0..B-1], global_m, r_vs[0..B-1])

  // ── Phase 1: Commit all chunks ───────────────────────────────────────────
  For b = 0 to B-1:
    V_lo^(b)[i] ← V^(b)[i] mod 2^16
    V_hi^(b)[i] ← V^(b)[i] >> 16
    cc^(b)_0    ← Com(V_lo^(b));  cc^(b)_1 ← Com(V_hi^(b))
    FS.absorb("chunk_com", cc^(b)_0, cc^(b)_1)

  // ── Phase 1b: Merge multiplicities and commit once ───────────────────────
  m[v] ← 0  for v in 0..2^16-1
  For b = 0 to B-1:
    For i in 0..2^n_b-1:
      m[V_lo^(b)[i]] += 1
      m[V_hi^(b)[i]] += 1
  m_com ← Com(m)
  FS.absorb("m_com", m_com)

  // ── Phase 2: Per-witness sumcheck + openings ─────────────────────────────
  For b = 0 to B-1:
    claim_v^(b) ← Σ_i V^(b)[i]   // sum over all evaluations
    FS.append("claim_v", claim_v^(b))
    (sc_π^(b), r_v^(b)) ← ProveSumcheck(V_mle^(b), ones_mle, claim_v^(b), FS)

    // Batch open both chunks at r_v^(b)
    ce^(b)_0 ← V_lo_mle^(b)(r_v^(b))
    ce^(b)_1 ← V_hi_mle^(b)(r_v^(b))
    Assert V^(b)(r_v^(b)) == ce^(b)_0 + 2^16 * ce^(b)_1
    chunk_batch_π^(b) ← HyraxOpenBatch([V_lo^(b), V_hi^(b)], r_v^(b), FS)

    rps[b] ← RangeWitnessProof {
      chunk_coms: [cc^(b)_0, cc^(b)_1],
      chunk_evals: [ce^(b)_0, ce^(b)_1],
      chunk_batch_π: chunk_batch_π^(b),
      sumcheck: sc_π^(b),
      claim_v: claim_v^(b)
    }

  // ── Phase 3: Shared multiplicity opening ────────────────────────────────
  r_m ← FS.challenge_vec("range_m_r", 16)
  m_eval ← m_mle(r_m)
  m_open ← HyraxOpen(m, r_m, FS)

  global_m ← GlobalRangeM { m_com, m_eval, m_open }
  return (rps, global_m, r_vs)
```

### 4.5 LayerNorm Prover

```
Algorithm ProveLayerNorm(wit, io_coms, vk, σ_range, y_range, FS):
  Input:
    wit     = { x, y, sum_x, sq_sum_x, σ }   // witness arrays
    io_coms = { x_com, y_com }                // from pipeline
    vk      = { γ, β, d, scale_γ, scale_β }
    σ_range = (RangeWitnessProof, r_σ)        // pre-committed, from global batch
    y_range = (RangeWitnessProof, r_y)        // pre-committed, from global batch

  // 1. Commit intermediate witnesses
  sum_x_com  ← Com(sum_x)
  sq_sum_com ← Com(sq_sum_x)
  σ_com      ← Com(σ)
  FS.absorb("x_com",      io_coms.x_com)
  FS.absorb("y_com",      io_coms.y_com)
  FS.absorb("sum_x_com",  sum_x_com)
  FS.absorb("sq_sum_com", sq_sum_com)
  FS.absorb("σ_com",      σ_com)

  // 2. Row audit challenge
  r_t ← FS.challenge_vec("layernorm_rt", t_bits)

  // 3. Mean sumcheck: Σ_j x_col[j] = sum_x_mle(r_t)
  x_col ← x_mle.fix_row(r_t)
  sum_x_rt ← sum_x_mle(r_t)
  FS.append("sum_x_at_rt", sum_x_rt)
  (mean_sc, r_d_mean) ← ProveSumcheck(x_col, ones_mle, sum_x_rt, FS)

  // 4. Variance sumcheck (cubic): Σ_j h[j]^2 = var_x_mle(r_t)
  //    h[j] = d * x_col[j] - sum_x_rt
  var_x_rt ← sq_sum_mle(r_t)
  FS.append("var_x_at_rt", var_x_rt)
  h_mle ← d * x_col - sum_x_rt
  (var_sc, r_d_var) ← ProveSumcheckCubic(h_mle, h_mle, h_mle,
                                           var_x_rt, FS)

  // 5. Sigma constraint verification (uses pre-committed σ_range)
  //    Verifier will reconstruct σ(r_σ) from chunk_evals
  (σ_rp, r_σ) ← σ_range
  r_σ_t ← r_σ[0..t_bits];  r_σ_b ← r_σ[t_bits+d_bits]
  var_x_rσ ← var_x_mle(r_σ_t)
  σ_rσ     ← σ_mle(r_σ_t)
  // (These evaluations are included in the proof for the verifier)

  // 6. Y constraint (multi-cubic batched sumcheck)
  (y_rp, r_y) ← y_range
  r_y_t ← r_y[0..t_bits];  r_y_d ← r_y[t_bits..t_bits+d_bits];  r_y_b ← r_y[...]
  γ_r ← γ_mle(r_y_d);  β_r ← β_mle(r_y_d)

  // Fuse γX and σY constraints into a single batched sumcheck:
  α ← FS.challenge("layernorm_alpha")
  // Claim: α * Σ_j eq_t(j)*γ(j)*x(j) + Σ_j eq_t(j)*σ(j)*y(j) = combined
  (gx_sy_sc, r_f) ← ProveSumcheckMultiBatched(
      [γ_mle ⊙ x_mle_col_ry, σ_mle_row_ry ⊙ y_mle_col_ry], α, FS)

  // 7. Collect Hyrax openings (deferred to batch accumulators)
  acc_t.add(sum_x_com,  sum_x_rt,  r_t)
  acc_t.add(sq_sum_com, var_x_rt,  r_t)
  acc_t.add(σ_com,      σ_rσ,      r_σ_t)
  acc_t.add(sum_x_com,  sum_x_rσ,  r_σ_t)
  acc_td.add(x_com,     x_rt_mean, (r_t, r_d_mean))
  acc_td.add(x_com,     x_rt_var,  (r_t, r_d_var))
  acc_td.add(x_com,     x_ry,      (r_y_t, r_y_d))
  acc_td.add(y_com,     y_ry,      (r_y_t, r_y_d))
  acc_t.add(sum_x_com,  sum_x_ryt, r_y_t)
  acc_t.add(σ_com,      σ_ryt,     r_y_t)

  return LayerNormProof {
    sum_x_com, sq_sum_com, σ_com,
    σ_range_proof: σ_rp,
    y_range_proof: y_rp,
    mean_sc, var_sc, gx_sy_sc,
    openings: { sum_x_rt, var_x_rt, x_rt_mean, x_rt_var,
                var_x_rσ, σ_rσ, x_ry, y_ry, sum_x_ryt, σ_ryt }
  }
```

### 4.6 End-to-End Verifier

```
Algorithm Verify(VK, x_in_com, logits_com, π):
  Input:
    VK           : verifying key (weight commitments + generators)
    x_in_com     : Hyrax commitment to input X_in
    logits_com   : expected commitment to output logits
    π            : TransformerModelProof
  Output:
    ACCEPT or REJECT

  // 1. Initialize transcript (must match prover exactly)
  FS.init("piformer")
  FS.absorb("x_in_com", x_in_com)

  x_cur_com ← x_in_com

  // 2. Verify each transformer block
  For ℓ = 0 to L-1:
    x_cur_com ← VerifyBlock(
        VK.blocks[ℓ], π.block_proofs[ℓ],
        x_cur_com, FS)

  // 3. Final LayerNorm range batch
  (final_r_vs, _) ← VerifyRangeBatched(
      [π.final_ln_π.σ_range_proof, π.final_ln_π.y_range_proof],
      π.final_range_m,
      [ln_σ_n, ln_y_n], bits=32, FS)

  // 4. Verify final LayerNorm
  final_ln_io_coms ← { x_com: x_cur_com, y_com: π.final_ln_π.y_com }
  VerifyLayerNorm(π.final_ln_π, final_ln_io_coms, VK.final_ln_vk,
                  σ_rv=final_r_vs[0], y_rv=final_r_vs[1],
                  FS, acc_t, acc_td)

  // 5. Verify LM head projection
  lm_head_io_coms ← { x_com: final_ln_io_coms.y_com, y_com: logits_com }
  VerifyProjection(π.lm_head_π, VK.lm_head_vk, lm_head_io_coms,
                   FS, acc_w, acc_b)

  // 6. Finalise batch accumulators (2 MSMs each)
  Assert acc_t.finalize(FS, π.acc_proofs.ln_t)     == ACCEPT
  Assert acc_td.finalize(FS, π.acc_proofs.ln_td)   == ACCEPT
  Assert acc_w.finalize(FS, π.acc_proofs.proj_w)   == ACCEPT
  Assert acc_b.finalize(FS, π.acc_proofs.proj_b)   == ACCEPT
  Assert acc_lmh_w.finalize(FS, π.acc_proofs.lmh_w) == ACCEPT
  Assert acc_lmh_b.finalize(FS, π.acc_proofs.lmh_b) == ACCEPT

  // 7. Global Lasso verification (all φ(Q), φ(K))
  Assert VerifyLassoMulti(
      [inst.attn_inst[ℓ].q_lasso, inst.attn_inst[ℓ].k_lasso for ℓ in 0..L-1],
      π.all_lasso_π, VK.lasso_vk, FS) == ACCEPT

  return ACCEPT
```

### 4.7 Block Verifier

```
Algorithm VerifyBlock(vk, π, x_in_com, FS):
  Input:
    vk           : TransformerBlockVerifyingKey
    π            : TransformerBlockProof
    x_in_com     : commitment to X_in
  Output:
    x_out_com    : commitment to X_out, or REJECT

  // ── PHASE 0: Global range batch ──────────────────────────────────────────
  // Transcript must absorb chunk_coms and m_com in same order as prover
  (block_r_vs, _) ← VerifyRangeBatched(
      [π.ln1_π.σ_range_proof, π.ln1_π.y_range_proof,
       π.ln2_π.σ_range_proof, π.ln2_π.y_range_proof],
      π.block_range_m,
      [ln_σ_n, ln_y_n, ln_σ_n, ln_y_n], bits=32, FS)
  ln1_σ_rv ← block_r_vs[0];  ln1_y_rv ← block_r_vs[1]
  ln2_σ_rv ← block_r_vs[2];  ln2_y_rv ← block_r_vs[3]

  // ── PHASE 1: LayerNorm 1 ─────────────────────────────────────────────────
  ln1_io_coms ← { x_com: x_in_com, y_com: π.x_norm1_com }
  VerifyLayerNorm(π.ln1_π, ln1_io_coms, vk.ln1_vk,
                  σ_rv=ln1_σ_rv, y_rv=ln1_y_rv,
                  FS, acc_t, acc_td)

  // ── PHASE 2: Batched QKV Projections ─────────────────────────────────────
  qkv_io_coms ← { x_com: π.x_norm1_com,
                   q_com: π.q_com, k_com: π.k_com, v_com: π.v_com }
  (q_claim, k_claim, v_proj_claim, x_norm1_claim) ←
      VerifyQKVProjections(π.qkv_π, vk.qkv_vk, qkv_io_coms, FS, acc_w, acc_b)

  // ── PHASE 3: Output Projection ───────────────────────────────────────────
  o_proj_io_coms ← { x_com: None, y_com: π.out_attn_com }
  (o_y_claim, o_x_claim) ←
      VerifyProjection(π.o_proj_π, vk.o_proj_vk, o_proj_io_coms,
                       FS, acc_w, acc_b)

  // ── PHASE 4: Linear Attention ─────────────────────────────────────────────
  attn_io_coms ← { q_com: π.q_com, k_com: π.k_com, v_com: π.v_com }
  (attn_out_claim, attn_v_claim) ←
      VerifyLinearAttention(π.attn_π, attn_inst,
          attn_io_coms, external_out_claim=o_x_claim, FS)
  // GKR fusion: o_x_claim binds attention's Out evaluation

  // ── PHASE 5: Residual 1 ──────────────────────────────────────────────────
  x_mid_com ← HyraxAdd(x_in_com, π.out_attn_com)

  // ── PHASE 6: LayerNorm 2 ─────────────────────────────────────────────────
  ln2_io_coms ← { x_com: x_mid_com, y_com: π.x_norm2_com }
  VerifyLayerNorm(π.ln2_π, ln2_io_coms, vk.ln2_vk,
                  σ_rv=ln2_σ_rv, y_rv=ln2_y_rv,
                  FS, acc_t, acc_td)

  // ── PHASE 7: FFN ─────────────────────────────────────────────────────────
  ffn_io_coms ← { x_com: π.x_norm2_com, y_com: π.out_ffn_com }
  (ffn_y_claim, ffn_x_claim) ←
      VerifyFFN(π.ffn_π, ffn_inst, vk.ffn_vk, ffn_io_coms, FS)

  // ── PHASE 8: Combine for V ───────────────────────────────────────────────
  (v_r, v_eval) ← VerifyCombineDeferred(
      π.v_combine_π, π.v_com,
      [v_proj_claim, attn_v_claim], FS)

  // ── PHASE 9: Block-level 7-point Hyrax batch ─────────────────────────────
  Assert HyraxVerifyMultiPoint([
    (π.q_com,        q_claim.eval,       q_claim.point),
    (π.k_com,        k_claim.eval,       k_claim.point),
    (π.x_norm1_com,  x_norm1_claim.eval, x_norm1_claim.point),
    (π.out_attn_com, o_y_claim.eval,     o_y_claim.point),
    (π.x_norm2_com,  ffn_x_claim.eval,   ffn_x_claim.point),
    (π.out_ffn_com,  ffn_y_claim.eval,   ffn_y_claim.point),
    (π.v_com,        v_eval,             v_r)
  ], VK.params, FS) == ACCEPT

  // ── PHASE 10: Residual 2 ─────────────────────────────────────────────────
  x_out_com ← HyraxAdd(x_mid_com, π.out_ffn_com)

  return x_out_com
```

### 4.8 Range Proof Verifier (Batched)

```
Algorithm VerifyRangeBatched(witness_proofs[0..B-1], global_m, n_vars[], bits, FS):
  Input:
    witness_proofs[b] = RangeWitnessProof {
      chunk_coms[0..1], chunk_evals[0..1],
      chunk_batch_π, sumcheck, claim_v
    }
    global_m = GlobalRangeM { m_com, m_eval, m_open }
  Output:
    (r_vs[0..B-1], m_eval)

  // 1. Absorb all chunk commitments (same order as prover Phase 1)
  For b = 0 to B-1:
    FS.absorb("chunk_com", witness_proofs[b].chunk_coms[0..1])
  FS.absorb("m_com", global_m.m_com)

  // 2. Per-witness: verify sumcheck and chunk reconstruction
  For b = 0 to B-1:
    wp ← witness_proofs[b]
    FS.append("claim_v", wp.claim_v)
    (r_v, final_v) ← VerifySumcheck(wp.sumcheck, wp.claim_v, n_vars[b], FS)
    
    // Check algebraic reconstruction
    v_reconstructed ← wp.chunk_evals[0] + 2^16 * wp.chunk_evals[1]
    Assert final_v == v_reconstructed

    // Batch Hyrax verification for chunks at r_v
    Assert HyraxVerifyBatch(
        wp.chunk_coms, wp.chunk_evals, r_v, wp.chunk_batch_π, FS
    ) == ACCEPT
    
    r_vs[b] ← r_v

  // 3. Shared multiplicity opening
  r_m ← FS.challenge_vec("range_m_r", 16)
  // Identity table T_id[i] = i: verifier reconstructs T_id_mle(r_m) locally
  T_id_rm ← Evaluate_IdentityTableMLE(r_m)
  Assert HyraxVerify(global_m.m_com, global_m.m_eval, r_m,
                     global_m.m_open, VK.params) == ACCEPT

  // 4. LogUp completeness check (over total chunk frequencies)
  //    Verifier checks the sum-of-chunks identity:
  //    Σ_{b,c} chunk_evals[b][c] * Lagrange(r_m)[...] == m_eval * T_id_rm
  //    (This is checked implicitly via the sumcheck + opening chain above)

  return (r_vs, global_m.m_eval)
```

### 4.9 LayerNorm Verifier

```
Algorithm VerifyLayerNorm(proof, io_coms, vk, σ_rv, y_rv, FS, acc_t, acc_td):
  // σ_rv, y_rv come from VerifyRangeBatched (already transcript-synchronized)

  // 1. Absorb commitments
  FS.absorb("x_com",      io_coms.x_com)
  FS.absorb("y_com",      io_coms.y_com)
  FS.absorb("sum_x_com",  proof.sum_x_com)
  FS.absorb("sq_sum_com", proof.sq_sum_com)
  FS.absorb("σ_com",      proof.σ_com)

  // 2. Row challenge
  r_t ← FS.challenge_vec("layernorm_rt", t_bits)
  FS.append("sum_x_at_rt", proof.openings.sum_x_rt)
  FS.append("var_x_at_rt", proof.openings.var_x_rt)

  // 3. Mean sumcheck
  (r_d_mean, final_mean) ← VerifySumcheck(
      proof.mean_sc, proof.openings.sum_x_rt, d_bits, FS)
  Assert final_mean == proof.openings.x_rt_mean   // x_col(r_d_mean)

  // 4. Variance sumcheck (cubic)
  (r_d_var, final_var) ← VerifySumcheckCubic(
      proof.var_sc, proof.openings.var_x_rt, d_bits, FS)
  h_eval ← d * proof.openings.x_rt_var - proof.openings.sum_x_rt
  Assert final_var == h_eval * h_eval * h_eval   // h^3 → cubic check

  // 5. Sigma constraint (uses σ_rv from range batch)
  r_σ_t ← σ_rv[0..t_bits];  r_σ_b ← σ_rv[t_bits+d_bits]
  σ_rσ  ← proof.openings.σ_rσ    // σ_mle(r_σ_t)
  // Reconstruct from chunk evals (σ committed as range proof chunks):
  σ_reconstructed ← proof.σ_range_proof.chunk_evals[0]
                   + 2^16 * proof.σ_range_proof.chunk_evals[1]
  Assert σ_rσ == σ_reconstructed

  dsi ← d * σ_rσ
  lo_σ ← proof.openings.var_x_rσ - dsi^2
  hi_σ ← (dsi + d)^2 - 1 - proof.openings.var_x_rσ
  σ_eval_expected ← (1 - r_σ_b) * lo_σ + r_σ_b * hi_σ
  Assert proof.σ_range_proof.claim_v == σ_eval_expected (verified via sumcheck)

  // 6. Y constraint fusion
  r_y_t ← y_rv[0..t_bits];  r_y_d ← y_rv[t_bits..t_bits+d_bits];  r_y_b ← y_rv[...]
  γ_r   ← Evaluate_MLE(vk.γ, r_y_d)    // O(d) verifier computation
  β_r   ← Evaluate_MLE(vk.β, r_y_d)
  y_ryt ← proof.openings.y_ry;          // y_mle(r_y_t, r_y_d)
  // Reconstruct y from chunk evals:
  y_reconstructed ← proof.y_range_proof.chunk_evals[0]
                   + 2^16 * proof.y_range_proof.chunk_evals[1]
  Assert y_ryt == y_reconstructed

  sig_d ← proof.openings.σ_ryt * d
  expr  ← vk.scale_γ * γ_r * (d * proof.openings.x_ry - proof.openings.sum_x_ryt)
         + vk.scale_β * β_r * sig_d
  lo_y ← 2*expr - sig_d*(2*y_ryt - 1)
  hi_y ← sig_d*(2*y_ryt + 1) - 1 - 2*expr
  y_eval_expected ← (1 - r_y_b) * lo_y + r_y_b * hi_y
  Assert proof.y_range_proof.claim_v == y_eval_expected

  // 7. Verify batched multi-cubic sumcheck (γX and σY fusion)
  α ← FS.challenge("layernorm_alpha")
  (r_f, final_gxsy) ← VerifySumcheckMultiBatched(
      proof.gx_sy_sc, combined_claim, FS)

  // 8. Defer Hyrax openings to batch accumulators
  acc_t.add(proof.sum_x_com,  proof.openings.sum_x_rt,  r_t)
  acc_t.add(proof.sq_sum_com, proof.openings.var_x_rt,  r_t)
  acc_t.add(proof.σ_com,      proof.openings.σ_rσ,      r_σ_t)
  acc_t.add(proof.sum_x_com,  proof.openings.sum_x_rσ,  r_σ_t)
  acc_td.add(io_coms.x_com,   proof.openings.x_rt_mean, (r_t, r_d_mean))
  acc_td.add(io_coms.x_com,   proof.openings.x_rt_var,  (r_t, r_d_var))
  acc_td.add(io_coms.x_com,   proof.openings.x_ry,      (r_y_t, r_y_d))
  acc_td.add(io_coms.y_com,   proof.openings.y_ry,      (r_y_t, r_y_d))
  acc_t.add(proof.sum_x_com,  proof.openings.sum_x_ryt, r_y_t)
  acc_t.add(proof.σ_com,      proof.openings.σ_ryt,     r_y_t)
```

---

## 5. Soundness Analysis

### 5.1 Completeness

**Theorem 1 (Completeness).** *If the prover honestly executes the transformer model $\mathcal{M}_\theta$ on input $\mathbf{x}$ and constructs all witnesses correctly, then $\mathsf{Verify}(\mathsf{VK}, x_{\mathsf{in\_com}}, \mathsf{logits\_com}, \pi) = \mathsf{ACCEPT}$ with probability 1.*

*Proof.* Every algebraic identity — sumcheck, range proof, and Hyrax opening — holds exactly over $\mathbb{F}_r$ when all values are correctly computed. The Fiat-Shamir transcript is deterministic; prover and verifier derive identical challenges. $\square$

---

### 5.2 Soundness of Individual Primitives

**Lemma 1 (Sumcheck Soundness).** *Let $f, g$ be degree-$\delta$ MLEs. A cheating prover can make the sumcheck for $H = \sum_{x \in \{0,1\}^n} f(x) \cdot g(x)$ accept with a false claim $H' \neq H$ with probability at most $n \delta / |\mathbb{F}_r|$.*

*Proof.* By the Schwartz–Zippel lemma, a cheating prover must supply a false round polynomial $g_i(X)$ that is a non-zero univariate of degree $\leq \delta$ for at least one round. The verifier's challenge $r_i \stackrel{\$}{\leftarrow} \mathbb{F}_r$ hits any specific false evaluation point with probability $\leq \delta / |\mathbb{F}_r|$. Union bound over $n$ rounds gives $n\delta / |\mathbb{F}_r|$. For $\delta = 2$, $n \leq 64$: bound $\leq 2^7 / 2^{254} = 2^{-247}$. $\square$

**Lemma 2 (Hyrax Binding).** *Under the discrete-log (DL) assumption in the BN254 G1 group, Hyrax is computationally binding: no PPT adversary can open the same commitment $C$ to two different polynomials at the same point.*

*Proof (sketch).* Hyrax reduces binding to the hardness of computing discrete logarithms. The commitment $C_i = \text{MSM}(\mathbf{g}, M[i])$ is a vector Pedersen commitment; opening a committed polynomial to two different evaluations at the same point would give a non-trivial linear relation among the generators $\mathbf{g}$, directly yielding a DL solution. $\square$

**Lemma 3 (Hyrax Multi-Point Batch Soundness).** *The random linear combination $\sum_i \eta^{i-1} \cdot w'_i$ preserves binding. A cheating prover who fakes any single opening $w'_j$ causes the combined commitment check to fail except with probability $(K-1)\delta / |\mathbb{F}_r|$ over the Fiat-Shamir challenge $\eta$.*

*Proof.* If $w'_j \neq \hat{w}'_j$ (honest value), then the combined vector $w'_{\mathsf{agg}} \neq \hat{w}'_{\mathsf{agg}}$ with probability $\geq 1 - (K-1)/|\mathbb{F}_r|$ over $\eta$ (since a single false entry makes the aggregated polynomial non-zero, and the polynomial in $\eta$ of degree $K-1$ has at most $K-1$ roots). $\square$

**Lemma 4 (Lasso Soundness).** *Given Hyrax binding (Lemma 2) and sumcheck soundness (Lemma 1), the only successful cheating strategy in the Lasso argument requires finding two different tables consistent with the same commitment — which reduces to breaking Hyrax binding.*

*Proof.* The Lasso sumcheck (Eq. 5) reduces to an opening claim $\widetilde{T}_k(\mathbf{r}) = v$ at a Fiat-Shamir random point $\mathbf{r}$. A cheating prover supplying the wrong table $\hat{T}_k \neq T_k$ must either (a) forge the sumcheck — caught by Lemma 1, or (b) claim an inconsistent $\widetilde{T}_k(\mathbf{r})$ — caught by Lemma 2. $\square$

**Lemma 5 (Range Proof Soundness).** *The batched range proof proves $V^{(b)}[i] \in [0, 2^{32})$ for all $b, i$ except with probability at most $O(n \cdot B / |\mathbb{F}_r|)$ (sumcheck error) plus negligible probability from Hyrax binding.*

*Proof.* The chunk reconstruction $V^{(b)}(r) = V_{lo}^{(b)}(r) + 2^{16} V_{hi}^{(b)}(r)$ is enforced at a Fiat-Shamir random point $r$ via the sumcheck (soundness $2n/|\mathbb{F}_r|$ by Lemma 1). Each chunk value is in $[0, 2^{16})$ because: (1) the multiplicity array $m$ is committed before any sumcheck, binding the prover to a fixed $m$; (2) the LogUp identity check $\sum_b \sum_c \sum_i \delta_{v, \mathsf{chunk}^{(b,c)}[i]} = m[v]$ enforces that all claimed chunk values actually appear in the identity table $T_{\mathsf{id}}[i] = i$ with the stated multiplicities; (3) Hyrax binding prevents $m_{\mathsf{com}}$ from being opened inconsistently. If any chunk value exceeded $2^{16}-1$, there would be no valid $m$ satisfying the LogUp identity. $\square$

---

### 5.3 Soundness of Structural Optimizations

**Theorem 2 (GKR Backward Fusion Soundness).** *Eliminating $\mathsf{out\_inner\_com}$ via the shared evaluation point between O-projection and attention does not reduce soundness.*

*Proof.* The O-projection prover outputs an evaluation claim $(r_x, r_y, v)$ asserting $\mathsf{Out\_inner}(r_x, r_y) = v$. The attention prover is given this claim as its output constraint and runs an independent sumcheck to prove:
$$\mathsf{Out}(r_x, r_y) = \sum_{i \in \{0,1\}^{d_h}} \Phi_Q(r_x, i) \cdot C(i, r_y)$$
The two sumchecks are over different sub-problems (contraction index $k$ in O-proj; head dimension $i$ in attention). A cheating prover who wishes to pass both checks must simultaneously satisfy two independent polynomial identity checks at the same random point $(r_x, r_y)$. Since the Fiat-Shamir challenges $(r_x, r_y)$ are derived from the transcript *after* all commitments, and the two checks are algebraically independent, the cheating probability is at most $2 \cdot (n \delta / |\mathbb{F}_r|)$ — the same as proving two separate sumchecks. $\square$

**Theorem 3 (Homomorphic Residual Soundness).** *The residual connections $X_{\mathsf{mid}} = X_{\mathsf{in}} + X_{\mathsf{out\_attn}}$ and $X_{\mathsf{out}} = X_{\mathsf{mid}} + X_{\mathsf{out\_ffn}}$, computed as $C_{\mathsf{mid}} = C_{\mathsf{in}} + C_{\mathsf{out\_attn}}$ by the verifier, are sound without any additional proof.*

*Proof.* Since $\mathsf{Com}(A) + \mathsf{Com}(B) = \mathsf{Com}(A + B)$ holds by the linearity of MSM (Pedersen-style commitment), the verifier's local computation of $C_{\mathsf{mid}}$ is deterministically correct given the binding of $C_{\mathsf{in}}$ and $C_{\mathsf{out\_attn}}$. Binding of these two commitments is guaranteed by Lemma 2. $\square$

**Theorem 4 (FFN GKR Ordering Soundness).** *Proving the activation Lasso before the Y-sumcheck (and eliminating $A_{\mathsf{com}}$) does not reduce soundness.*

*Proof.* The activation Lasso is committed to the transcript before the Y-sumcheck challenge $r_k$ is derived. Therefore the prover is committed to specific Lasso outputs before learning $r_k$. The verifier evaluates $A(r_k)$ from the Lasso outputs MLE, which is a public, deterministic function of the committed query outputs. A cheating prover who forges $A(r_k)$ must either forge the Lasso outputs (caught by Lemma 4) or forge the sumcheck (caught by Lemma 1). $\square$

**Theorem 5 (Batched QKV Soundness).** *The single shared-$r_k$ sumcheck for Q, K, V projections is as sound as three independent sumchecks.*

*Proof.* The verifier draws Fiat-Shamir scalars $\lambda, \mu \stackrel{\$}{\leftarrow} \mathbb{F}_r$ before the sumcheck challenge $r_k$. The combined claim is $\lambda \cdot Y_Q(r_t, r_d) + \mu \cdot Y_K(r_t, r_d) + Y_V(r_t, r_d)$. A cheating prover who fakes any single projection $\hat{Y}_Q \neq Y_Q$ changes the combined polynomial by $\lambda \cdot (\hat{Y}_Q - Y_Q)$, which is a non-zero polynomial in $\lambda$ of degree $\leq 1$. The verifier's random $\lambda$ fails to catch this with probability $\leq 1 / |\mathbb{F}_r| \approx 2^{-254}$. Combined with the sumcheck error: total soundness error $\leq 2 / |\mathbb{F}_r| + n\delta / |\mathbb{F}_r|$. $\square$

**Theorem 6 (Global Range Batch Soundness).** *Sharing one $m_{\mathsf{com}}$ across $B$ range witnesses is sound: a cheating prover cannot supply chunk values outside $[0, 2^{16})$ for any single witness.*

*Proof.* The key commitment-ordering invariant is that $m_{\mathsf{com}}$ is absorbed into the transcript *before* any per-witness sumcheck challenges $r_v^{(b)}$ are derived. Since $m$ aggregates all chunk occurrences across all witnesses, and the prover must commit to $m$ before seeing any $r_v^{(b)}$, the prover cannot adaptively choose the chunk values after learning the sumcheck challenges. Formally: suppose a cheating prover supplies $\hat{V}_{lo}^{(b_0)}$ with some entry $\hat{V}_{lo}^{(b_0)}[i] = v^* \geq 2^{16}$. For the LogUp check to pass, $m[v^*]$ must be positive, but $T_{\mathsf{id}}[v^*]$ does not exist (the identity table only has entries $0 \ldots 2^{16}-1$), so the LogUp identity $\sum_{b,c,i} \delta_{v^*, \mathsf{chunk}^{(b,c)}[i]} = m[v^*] \cdot T_{\mathsf{id}}[v^*]$ fails. The committed $m_{\mathsf{com}}$ binds the prover to the claimed multiplicities before all sumchecks (Lemma 2). $\square$

---

### 5.4 End-to-End Soundness Theorem

**Theorem 7 (π-Former Soundness).** *For any PPT adversary $\mathcal{A}$, the probability that $\mathsf{Verify}(\mathsf{VK}, x_{\mathsf{in\_com}}, y_{\mathsf{com}}, \pi) = \mathsf{ACCEPT}$ but $\mathcal{M}_\theta(\mathbf{x}) \neq \mathbf{y}$ is at most*

$$\varepsilon_{\mathsf{sound}} \leq L \cdot n_{\max} \cdot \delta_{\max} / |\mathbb{F}_r| + \varepsilon_{\mathsf{DL}}$$

*where $n_{\max}$ is the maximum number of sumcheck variables across all sub-protocols, $\delta_{\max} = 3$ is the maximum round polynomial degree, $L$ is the number of transformer blocks, and $\varepsilon_{\mathsf{DL}}$ is the DL advantage of $\mathcal{A}$ in the BN254 G1 group.*

*Proof sketch.* We proceed by hybrid argument over the $L$ blocks.

1. **Hyrax binding** (Lemma 2): If $\mathcal{A}$ can forge any committed evaluation, it breaks DL in G1.

2. **Block-level chaining:** In block $\ell$, the output commitment $C_{x_{\mathsf{out}}}^\ell$ is the *homomorphic sum* of three committed tensors. Binding of each component tensor (Lemma 2) implies binding of $C_{x_{\mathsf{out}}}^\ell$. The first block's input is $x_{\mathsf{in\_com}}$, which the verifier holds independently; binding proceeds inductively.

3. **Within each block:** The transcript state after block $\ell-1$ is a deterministic function of all commitments and proofs from blocks $0 \ldots \ell-1$. All challenges in block $\ell$ are Fiat-Shamir outputs from this state. Given Hyrax binding and sumcheck soundness:
   - The 4-element global range batch is sound (Theorem 6).
   - Each LayerNorm is sound (sumcheck error + range proof soundness, Lemma 5).
   - Batched QKV soundness (Theorem 5).
   - GKR fusion soundness (Theorem 2).
   - FFN GKR ordering soundness (Theorem 4).
   - Homomorphic residuals require no proof (Theorem 3).
   - Lasso soundness (Lemma 4) for all activation tables.

4. **Final layer + LM head:** Same argument as a single block.

5. **Global Lasso batch:** Indexed over all $Q, K$ lookups; soundness follows from Lemma 4 applied per sub-table, with the batch error bounded by the number of sub-tables times the per-table error.

6. **Union bound:** Summing over all $L$ blocks, sub-protocols, and sumcheck rounds, the total statistical error is bounded by $L \cdot n_{\max} \cdot \delta_{\max} / |\mathbb{F}_r|$. For $L=12$, $n_{\max} = 64$, $\delta_{\max} = 3$: error $\leq 2^4 \cdot 2^6 \cdot 2^2 / 2^{254} = 2^{-242}$. $\square$

---

### 5.5 Zero Knowledge

The current implementation is **not zero-knowledge**: weight commitments are public VK entries, and intermediate activation evaluations are revealed in sumcheck transcripts. Zero-knowledge can be added by:

- Blinding all witness polynomials with random masking terms before commitment.
- Applying a BlindFold-style ZK layer (Pedersen commitments with randomness + Nova folding over the sumcheck transcript) to hide all intermediate claims.

This is deferred to a future protocol version.

---

## 6. Fixed-Point Encoding

All real-valued tensors are represented as integers scaled by a factor $s$:

$$x_{\mathbb{F}} = \left\lfloor \frac{x}{s} \right\rfloor \bmod r \in \mathbb{F}_r$$

Arithmetic in $\mathbb{F}_r$ then simulates fixed-point integer arithmetic as long as no intermediate value overflows the field modulus. For $B$-bit activations and $k_{\max}$-bit weights, the maximum intermediate value in a matrix product is $T \cdot d_h \cdot 2^B \cdot 2^{k_{\max}}$, which must remain below $r \approx 2^{254}$.

The Python training pipeline uses the same quantization so the exported integer tables and weights exactly match the field arithmetic in the Rust prover. This **eliminates the train–prove gap**.

---

## 7. Complexity Summary

Let $L$ = layers, $T$ = sequence length, $d$ = embedding dimension, $d_{ff}$ = FFN width, $m$ = bits per chunk, $c$ = chunk count, $B_{rng}$ = range witnesses per block (4 for ln1+ln2).

| Component | Prover time | Verifier time | Proof size |
|-----------|-------------|---------------|------------|
| Hyrax commit (size $N$) | O($N$) G1 MSM | — | O($\sqrt{N}$) G1 points |
| Hyrax open (single) | O($\sqrt{N}$) G1 | O($\sqrt{N}$) G1 + O($\log N$) $\mathbb{F}$ | O($\sqrt{N}$) $\mathbb{F}$ |
| Hyrax $K$-point batch | O($K \sqrt{N}$) G1 | O($2\sqrt{N}$) G1 | O($K\sqrt{N}$) $\mathbb{F}$ |
| Sumcheck ($n$ vars, deg $\delta$) | O($n \delta \cdot 2^n$) $\mathbb{F}$ | O($n\delta$) $\mathbb{F}$ | O($n\delta$) $\mathbb{F}$ |
| Lasso (one sub-table, $N$ queries) | O($N \cdot 2^m + m \cdot 2^m$) | O($N \cdot m$) | O($m$) $\mathbb{F}$ |
| Range proof (1 witness, batched $m$) | O($\sqrt{2^{16}}$) G1 | O($\sqrt{2^{16}}$) G1 | O($\sqrt{2^{16}}$) $\mathbb{F}$ |
| Range proof global $m_\mathsf{com}$ | O($\sqrt{2^{16}}$) G1 **once** | O($\sqrt{2^{16}}$) G1 **once** | O($\sqrt{2^{16}}$) $\mathbb{F}$ |
| LayerNorm (per block) | 2 sumchecks + 2 range witnesses | O($d$) + O($\log T$) | O($\sqrt{Td}$) |
| Projection (per matrix) | O($T \cdot d$) | O($\log(Td)$) | O($\sqrt{Td}$) |
| Batched QKV (3 matrices) | O($T \cdot d$) (1×) | O($\log(Td)$) | O($\sqrt{Td}$) |
| Linear attention (per block) | O($T d_h^2$) | O($d_h \log T$) | O($d_h \log T$) |
| FFN (per block) | O($T d \cdot d_{ff}$) | O($\log(T d \cdot d_{ff})$) | O($\sqrt{T d_{ff}}$) |
| **Per block total** | O($T d^2 + c N_{qk} 2^m$) | O($d \log T$) | O($d \log T + c \cdot 2^m$) |
| **All $L$ blocks** | $\times L$ | $\times L$ | $\times L$ |

**Savings from structural optimizations:**

| Optimization | MSMs saved per block | Proof bytes saved |
|---|---|---|
| Batched QKV (1 sumcheck vs 3) | 2 × O($T d$) field ops | 2 sumcheck transcripts |
| GKR fusion (no out_inner_com) | $\sqrt{Td}$ G1 points | $\sqrt{Td}$ G1 points |
| Homomorphic residuals (no proof) | 2 × Hyrax opens | 2 Hyrax proofs |
| FFN GKR (no a_com) | $\sqrt{Td_{ff}}$ G1 points | $\sqrt{Td_{ff}}$ G1 points |
| Global range batch (1 m_com vs $B_{rng}$) | $(B_{rng}-1) \times \sqrt{2^{16}}$ G1 | $(B_{rng}-1) \times \sqrt{2^{16}}$ G1 |
| Deferred batch accumulators | $(K-1) \times 2$ MSMs per group | 0 (proof size unchanged) |

---

## 8. Comparison with Related Work

| System | Attention | Activation | PCS | ZK |
|--------|-----------|------------|-----|----|
| **zkGPT** | Softmax (approx.) | Lookup table | Plonky2 | ✓ |
| **zkLLM** | Softmax (tlookup) | Structured lookup | Custom IOP | ✗ |
| **π-Former** | Linear (exact) | Learned structured lookup | Hyrax | roadmap |

Key advantages of π-Former:

- **Exact computation:** linear attention is not an approximation of softmax; it is a different (learnable) attention mechanism trained to be useful and provable.
- **End-to-end co-design:** weights and activation tables are trained to match the exact integer/field arithmetic in the circuit, eliminating approximation errors.
- **Transparent setup:** Hyrax requires no trusted setup; all parameters are derived from a public hash function.
- **Structured optimizations:** GKR fusion, homomorphic residuals, global range batching, and batched QKV reduce concrete proof size and prover time without sacrificing soundness.
- **Extensible:** lookup decomposition depth $c$ and chunk size $m$ are configurable trade-offs between model expressivity and proof cost.

---

## 9. Planned Extensions

### 9.1 Memory-Consistency Check for Lasso

The current Lasso argument proves that query outputs are consistent with a *claimed* table. To prevent a dishonest prover from using a different table per query, add an offline memory-consistency argument (Spice / Lasso grand-product check) that ties all queries back to a single committed table across all invocations.

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
state     ← SHA3-256(label ‖ data ‖ state_hash)
challenge ← F::from_le_bytes_mod_order(SHA3-256(state ‖ label))
```

The `--transcript-label` flag in the CLI must match between `prove` and `verify` (default: `"piformer"`).

### Transcript Ordering Invariant

The global range batch for each block *must* be processed at the start of that block's sub-protocol sequence. All chunk commitments and the shared $m_{\mathsf{com}}$ are absorbed into the transcript before any per-witness sumcheck challenges are derived. Violating this ordering breaks soundness (§5.3, Theorem 6). In the implementation, `ProveRangeBatched` / `VerifyRangeBatched` is called at the very beginning of `prove_transformer_block` / `verify_transformer_block`, and `prove_layernorm` / `verify_layernorm` do *not* call range proof functions internally.

### Bit-Ordering Convention

`DenseMLPoly::fix_first_variable(r)` fixes the highest-index variable first (bit $n-1$ of the evaluation index). Therefore sumcheck challenge $r_j$ corresponds to bit $n-1-j$ of the final evaluation index. When constructing $L_k(\mathbf{r})$ from bit decompositions of chunk indices (naturally LSB-first), the challenge vector must be reversed. See `lookup/lasso.rs` for the explicit reversal.

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

- **Proving key** contains Hyrax G1 commitments **plus** the raw weight vectors (needed by the prover to compute witness–weight inner products during sumcheck).
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
Magic:      b"PFMR_PK\0"  (8 bytes)
Version:    u8             (1 byte)
num_blocks: u64 LE
seq_len:    u64 LE
d_model:    u64 LE
vocab_size: u64 LE
final_ln_vk:  [LayerNormVK]
lm_head_pk:   [ProjectionPK with weights]
block_pks:    [num_blocks × TransformerBlockPK with weights]
```

Each field element occupies 32 bytes (uncompressed). Each G1Affine occupies 33 bytes (compressed). Lengths are prefixed as `u64 LE`.

### Binary Verifying Key (`.vk`)

Same layout as `.pk` but all `ProjectionPK` / `FFN_PK` weight fields are replaced by empty vectors (`has_weights = false`).

### Binary Proof Bundle (`.bin`)

```
Magic:       b"PFMR_PR\0"  (8 bytes)
Version:     u8
lasso_sigma: u64 LE
inst_attn:   [LinearAttentionInstance]
inst_ffn:    [FFNInstance]
proof:       [TransformerModelProof]
```

`TransformerModelProof` contains one `TransformerBlockProof` per block, each holding all sub-proofs, intermediate Hyrax commitments, the block-level `GlobalRangeM`, and the 7-element batch opening proof. The final `GlobalRangeM` for the final LayerNorm is stored at the model level.
