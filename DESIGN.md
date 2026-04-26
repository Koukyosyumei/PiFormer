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

### 2.4 Ternary Weight Quantization

Projection matrices $W_Q, W_K, W_V, W_O$, FFN weight matrices, and the language-model head are constrained to entries in

$$\mathcal{T} = \{-1,\, 0,\, +1\}$$

with a single learnable scalar $\alpha \in \mathbb{F}_r$ per layer absorbing the magnitude. Each forward pass applies $Y = \alpha \cdot (X \cdot W) + b$ where $W \in \mathcal{T}^{m \times n}$. In the circuit, a matrix-vector product with $W \in \mathcal{T}^{m \times n}$ requires only additions and subtractions — no field multiplications — so the contraction is **free** in the sumcheck constraint system. The Rust prover represents weights as `Vec<Vec<TernaryValue>>` where `TernaryValue ∈ {ONE, ZERO, MINUSONE}`, and uses the specialized `eval_cols_ternary` helper to fold the equality polynomial against ternary entries with `+= / -= / skip` rather than field multiplications.

**Training:** A straight-through estimator (STE) is used. The forward pass applies a magnitude-thresholded mapping $w \mapsto \mathrm{sign}(w) \cdot [\,|w| > 0.7 \cdot \overline{|w|}\,]$; the backward pass uses the identity function in place of the quantizer's zero gradient. The learnable $\alpha$ is multiplied in *after* the matrix product, so the in-circuit weight stays ternary.

**Stand-alone enforcement.** A self-contained Lasso check (`attention/ternary_check.rs`) proves that every committed weight value is in $\{-1, 0, +1\}$ using a 4-element table $T = [0, 1, p-1, 0]$ keyed by an integer index $\mathsf{enc}(w) \in \{0,1,2\}$. This module is currently a stand-alone protocol with its own tests; a future revision will fold it into setup-time preprocessing so the verifying key contains a binding proof that $W \in \mathcal{T}^{m \times n}$.

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

Proves $Y = \alpha \cdot (X \cdot W) + b$ where $W$ is a committed ternary weight matrix ($W \in \mathcal{T}^{m \times n}$) and $\alpha \in \mathbb{F}_r$ is a per-layer scale absorbed into the sumcheck claim.

A single random entry $(r_t, r_d)$ is selected via Fiat-Shamir; a degree-2 sumcheck over the contraction index $k$ proves:
$$Y(r_t, r_d) - b(r_d) = \alpha \cdot \sum_{k \in \{0,1\}^{k_{\mathsf{bits}}}} X(r_t, k) \cdot W(k, r_d)$$

Hyrax openings bind $X$, $Y$, $W$, and $b$ to their claimed MLE evaluations. Because $W$ is ternary, the prover materialises only the cheap $\alpha \cdot W(\cdot, r_d)$ vector via `eval_cols_ternary`, never a full $\mathbb{F}_r$ weight matrix.

**Batched QKV variant:** A single sumcheck proves three projections simultaneously using Fiat-Shamir scalars $\lambda, \mu$:
$$\lambda \cdot (Y_Q(r_t, r_d) - b_Q(r_d)) + \mu \cdot (Y_K(r_t, r_d) - b_K(r_d)) + (Y_V(r_t, r_d) - b_V(r_d)) = \sum_k X(r_t, k) \cdot \big(\lambda \alpha_Q W_Q(k,r_d) + \mu \alpha_K W_K(k,r_d) + \alpha_V W_V(k,r_d)\big)$$
All three projections share the same sumcheck challenge $r_k$, reducing the sumcheck cost from $3\times$ to $1\times$.

**Cross-layer variant (`cross_layer/projection.rs`):** A stand-alone cubic sumcheck proves $Y_l = \alpha_l \cdot X_l \cdot W_l + b_l$ for **all** $L$ layers simultaneously by introducing a layer index variable $b \in \{0,1\}^{\log L}$. The verifier samples a single $(r_l, r_t, r_{\mathsf{out}})$ and the sumcheck runs over $(\log L + \log d_{\mathsf{in}})$ rounds with three multiplicands $f \cdot g \cdot h$ representing the layer-selector $\widetilde{\mathsf{eq}}(r_l, b)$, the activation $X(b, r_t, c)$, and the weight $W(b, c, r_{\mathsf{out}})$. This module is exercised by its own unit tests but is **not yet wired into the end-to-end prover**.

### 3.9 Linear Attention Circuit

The Rust prover proves the **un-normalized** linear-attention output

$$\mathsf{Out}_{\mathsf{attn}} = \phi(Q)\,\big(\phi(K)^\top V\big) \cdot W_O$$

for a single head ($n_{\mathsf{heads}} = 1$). The Python `LinearAttentionLayer` additionally divides by a normalizer $Z = \phi(Q) \cdot \sum_s \phi(K_s)$ for training stability; that division is **not** currently part of the SNARK circuit. To match the Rust prover at inference time, either disable $Z$ in the export step or extend the circuit with a row-normalization sumcheck (planned).

The four core claims for a single head are:

| Step | Statement | Protocol |
|------|-----------|----------|
| 1 | $\Phi_Q[t][d] = \phi(Q[t][d])$ | Lasso (§3.5) per chunk |
| 2 | $\Phi_K[t][d] = \phi(K[t][d])$ | Lasso per chunk |
| 3 | $C[i][j] = \sum_t \Phi_K[t][i] \cdot V[t][j]$ | Degree-2 sumcheck over $t$ |
| 4 | $\mathsf{Out}[t][j] = \sum_i \Phi_Q[t][i] \cdot C[i][j]$ | Degree-2 sumcheck over $i$ |

Steps 3–4 use a random-entry reduction. The verifier draws $(r_{out}, r_i)$ via Fiat-Shamir and audits a single entry per matrix product.

**Cross-block batching of attention.** The end-to-end model prover (§4.2) batches steps 3 and 4 across all $L$ blocks. Two `SumcheckProofMulti` instances — `batch_attn_out` and `batch_attn_ctx` — share one set of sumcheck challenges across all blocks, with per-block claims combined by Fiat-Shamir powers of $\eta$. Per-block opening claims for $\Phi_Q$, $\Phi_K$, $V$ at the shared evaluation points are folded into the global cross-block batch opens (§4.2).

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

### 4.2 End-to-End Prover (Cross-Block Batch Architecture)

The model prover does **not** prove blocks independently. Instead, after a per-block "Phase 1" that commits intermediate matrices and runs LayerNorm + range proofs, the prover runs a small number of *cross-block* batched sumchecks that cover one protocol type (e.g. QKV projection) across all $L$ blocks at once, with per-block claims combined by Fiat-Shamir powers of $\eta$.

```
Algorithm Prove(PK, witness W, instances inst):
  Input:
    PK          : proving key (weights + commitments)
    W           : { x_in, block_witnesses[0..L-1],
                    final_ln_wit, lm_head_wit }
    inst        : { attn_inst[0..L-1], ffn_inst[0..L-1] }
  Output:
    π           : TransformerModelProof

  // ── 1. Bind initial input ───────────────────────────────────────────────
  FS.init("piformer")
  x_in_com ← Com(W.x_in)
  FS.absorb("x_in_com", x_in_com)

  // ── 2. Phase 1 (per block): commit 7 intermediates, prove LN1+LN2 ──────
  x_cur_com ← x_in_com
  phase1[0..L-1] ← []
  For ℓ = 0 to L-1:
    p1 ← CommitBlockPhase1(W.block_witnesses[ℓ], x_cur_com, PK.block_pks[ℓ], FS)
    // p1 contains: x_norm1_com, q_com, k_com, v_com, out_attn_com,
    //              x_norm2_com, out_ffn_com, x_mid_com, ln1_proof, ln2_proof,
    //              block_range_m
    x_cur_com ← HyraxAdd(p1.x_mid_com, p1.out_ffn_com)
    phase1.append(p1)

  // ── 3. Derive global r_td after ALL Phase 1 commitments ────────────────
  r_td ← FS.challenge("gkr_r_td", t_bits + d_bits)
  (r_t, r_out) ← split(r_td, t_bits, d_bits)

  // ── 4. Cross-block QKV sumcheck ─────────────────────────────────────────
  For ℓ = 0 to L-1:
    Absorb (Wq, Wk, Wv, αq, αk, αv, bq, bk, bv) commitments for block ℓ
    (λ_ℓ, μ_ℓ) ← FS.challenges("qkv_lambda", "qkv_mu")
    Build f_ℓ(k) = X_norm1_ℓ(r_t, k)
    Build g_ℓ(k) = λ_ℓ αq Wq_ℓ(k, r_out) + μ_ℓ αk Wk_ℓ(k, r_out) + αv Wv_ℓ(k, r_out)
    target_ℓ = λ_ℓ (Q_ℓ - bq_ℓ) + μ_ℓ (K_ℓ - bk_ℓ) + (V_ℓ - bv_ℓ)
    FS.append claims (Q_ℓ, K_ℓ, V_ℓ at r_td)
  η_qkv ← FS.challenge("batch_eta_qkv")
  (batch_qkv, r_k_qkv) ← ProveSumcheckMulti({f_ℓ}, {g_ℓ}, powers(η_qkv), Σ η^ℓ target_ℓ, FS)

  // ── 5. Cross-block O-projection sumcheck ────────────────────────────────
  Per block: build f_ℓ(k) = α_O · X_inner_ℓ(r_t, k), g_ℓ(k) = Wo_ℓ(k, r_out)
  η_oproj ← FS.challenge("batch_eta_oproj")
  (batch_oproj, r_k_o) ← ProveSumcheckMulti(...)

  // ── 6. Cross-block Attention (out, then ctx) ────────────────────────────
  For ℓ = 0 to L-1:
    Commit phi_q_ℓ, phi_k_ℓ; absorb both
  // 6a. out_ℓ(r_t, r_k_o) = Σ_k phi_q_ℓ(r_t, k) · ctx_ℓ(k, r_k_o)
  (batch_attn_out, batch_r_attn_out) ← ProveSumcheckMulti({phi_q_ℓ@(r_t, ·)},
                                                           {ctx_ℓ@(·, r_k_o)}, ...)
  // 6b. ctx_ℓ(batch_r_attn_out, r_k_o) = Σ_t phi_k_ℓ(t, batch_r_attn_out) · v_ℓ(t, r_k_o)
  (batch_attn_ctx, batch_r_attn_ctx) ← ProveSumcheckMulti({phi_k_ℓ@(·, batch_r_attn_out)},
                                                           {v_ℓ@(·, r_k_o)}, ...)

  // ── 7. Per-block FFN: Lasso (commits A) → commit M ──────────────────────
  For ℓ = 0 to L-1:
    Absorb (W1, W2) commitments
    ffn_lasso_proof_ℓ ← ProveLasso(inst.ffn.activation_lasso, ...)  // commits A
    M_ℓ ← W.ffn.m;  m_com_ℓ ← Com(M_ℓ);  FS.absorb(m_com_ℓ)

  // Bind FFN lasso indices to a shared (rx, ry) point and open all M_ℓ at once
  Absorb concatenated lasso indices
  ffn_lasso_bind_point ← FS.challenge_vec(t_bits + f_bits)
  ffn_lasso_bind_open ← HyraxOpenBatch({M_ℓ}, ffn_lasso_bind_point, ...)

  // ── 8. Cross-block FFN-Y sumcheck: Y = A · W2 ───────────────────────────
  Per block: f_ℓ(k) = A_ℓ(r_t, k), g_ℓ(k) = W2_ℓ(k, r_out)
  (batch_ffn_y, r_k_fy) ← ProveSumcheckMulti(...)

  // ── 9. Cross-block FFN-M sumcheck: M = X_norm2 · W1 ─────────────────────
  rx_m ← FS.challenge_vec(t_bits, "ffn_rx_m")
  ry_m ← FS.challenge_vec(f_bits, "ffn_ry_m")
  Per block: f_ℓ(k) = X_norm2_ℓ(rx_m, k), g_ℓ(k) = W1_ℓ(k, ry_m)
  (batch_ffn_m, r_k_m) ← ProveSumcheckMulti(...)

  // ── 10. Final LayerNorm + LM head ───────────────────────────────────────
  final_rw ← ComputeRangeWitnesses(W.final_ln_wit, PK.final_ln_vk)
  (final_rps, final_range_m, final_rvs) ← ProveRangeBatched(
      [final_rw.σ_witness, final_rw.y_witness], bits=32, FS)
  final_ln_out_com ← Com(W.final_ln_wit.y)
  final_ln_π ← ProveLayerNorm(W.final_ln_wit, {x_cur_com, final_ln_out_com},
                              PK.final_ln_vk, final_rps[0..1], FS)
  logits_com ← Com(W.lm_head_wit.y)
  (lm_head_π, lm_y_claim, _) ← ProveProjection(PK.lm_head_pk, W.lm_head_wit,
                                                {x_com: final_ln_out_com}, FS, None)
  lm_head_logits_open ← HyraxOpen(W.lm_head_wit.y, lm_y_claim.point, ...)

  // ── 11. Advance transcript for accumulator μ-challenges ─────────────────
  For 10 deferred accumulators: FS.challenge("hyrax_group_mu")

  // ── 12. Global intermediate batch open ──────────────────────────────────
  // 5L matrices (Q, K, V, Out_attn, Out_ffn) opened at shared r_td
  inter_batch_open ← HyraxOpenBatch({Q_ℓ, K_ℓ, V_ℓ, Out_attn_ℓ, Out_ffn_ℓ}_{ℓ=0..L-1},
                                     r_td, ν_td, σ_td, FS)

  // ── 13. Cross-block batch opens for weights, biases, activations ────────
  // One HyraxOpenBatch per (matrix-type, evaluation-point) pair:
  //   x_norm1 @ (r_t, r_k_qkv);   Wq, Wk, Wv @ (r_k_qkv, r_out);   bq, bk, bv @ r_out
  //   Wo @ (r_k_o, r_out);        bo @ r_out
  //   W2 @ (r_k_fy, r_out);       W1 @ (r_k_m, ry_m);   x_norm2 @ (rx_m, r_k_m)
  //   M  @ (rx_m, ry_m)
  //   phi_q @ (r_t, batch_r_attn_out);   phi_k @ (batch_r_attn_ctx, batch_r_attn_out)
  //   v @ (batch_r_attn_ctx, r_k_o)      (per-block opening; v_attn_batch_open)
  // (See prover.rs §15 for the full list of 17 batch opens.)

  // ── 14. Global Lasso batch (all φ(Q) and φ(K) across all blocks) ────────
  all_lasso_π ← ProveLassoMulti(
      [inst.attn[ℓ].q_lasso, inst.attn[ℓ].k_lasso for ℓ in 0..L-1], PK.lasso_pk, FS)

  return TransformerModelProof {
    x_in_com, block_proofs[0..L-1],
    final_range_m, final_ln_proof, lm_head_proof,
    final_ln_out_com, logits_com, lm_head_logits_open,
    batch_qkv, batch_oproj, batch_ffn_y, batch_ffn_m,
    batch_attn_out, batch_attn_ctx,
    inter_batch_open,
    {x_norm1, w_q, w_k, w_v, bias_q, bias_k, bias_v,
     w_o, bias_o, w2, w1, x_norm2,
     ffn_m_com, ffn_lasso_bind, phi_q, phi_k, v_attn, qk_lasso_bind}_batch_open,
    all_lasso_π
  }
```

Each `TransformerBlockProof` carries the 7 intermediate commitments, the LN1/LN2 sub-proofs, the per-block range-multiplicity commitment, the FFN Lasso proof + `M` commitment, and the per-block scalar evaluations consumed by the cross-block batch sumchecks (`q_eval`, `k_eval`, `v_eval`, `attn_phi_q_eval`, `attn_phi_k_eval`, `attn_ctx_eval`, `attn_v_eval`, etc.).

### 4.3 Block Phase 1 (Per-Block Commit + LayerNorm)

The only truly per-block step is **Phase 1**: it commits the seven intermediate matrices, runs the global range batch for both LayerNorms, and proves LN1 / LN2. The remaining sub-protocols (QKV, O-projection, attention, FFN) are *not* run independently per block — they are batched cross-block in §4.2 steps 4–9.

```
Algorithm CommitBlockPhase1(wit, x_in_com, pk, FS):
  Input:
    pk      : TransformerBlockVerifyingKey (carries projection / attention PKs too)
    wit     : { ln1_wit, q_proj_wit, k_proj_wit, v_proj_wit,
                o_proj_wit, attn_wit, ln2_wit, ffn_wit }
    x_in_com: Hyrax commitment to X_in ∈ F^{T×d}
  Output:
    BlockPhase1Data
      { ln1_proof, ln2_proof, block_range_m,
        x_norm1_com, q_com, k_com, v_com,
        out_attn_com, x_norm2_com, out_ffn_com, x_mid_com }

  // ── 0. Commit 7 intermediate matrices ───────────────────────────────────
  x_norm1_com  ← Com(wit.ln1_wit.y)
  q_com        ← Com(wit.attn_wit.q)
  k_com        ← Com(wit.attn_wit.k)
  v_com        ← Com(wit.attn_wit.v)
  out_attn_com ← Com(wit.o_proj_wit.y)
  x_norm2_com  ← Com(wit.ln2_wit.y)
  out_ffn_com  ← Com(wit.ffn_wit.y)

  // ── 1. Global range batch for both LayerNorms in this block ────────────
  // 4 witnesses (ln1.σ, ln1.y, ln2.σ, ln2.y) share one m_com
  rw1 ← ComputeRangeWitnesses(wit.ln1_wit, pk.ln1_vk)
  rw2 ← ComputeRangeWitnesses(wit.ln2_wit, pk.ln2_vk)
  (block_rps, block_range_m, block_rvs) ← ProveRangeBatched(
      [rw1.σ_witness, rw1.y_witness, rw2.σ_witness, rw2.y_witness],
      bits=32, FS)

  // ── 2. LayerNorm 1 (uses x_norm1_com as y_com) ─────────────────────────
  ln1_io ← { x_com: x_in_com, y_com: x_norm1_com }
  ln1_proof ← ProveLayerNorm(wit.ln1_wit, ln1_io, pk.ln1_vk,
                              σ_range=(block_rps[0], block_rvs[0]),
                              y_range=(block_rps[1], block_rvs[1]), FS)

  // ── 3. Absorb q/k/v_com and out_attn_com explicitly ────────────────────
  // (the cross-block QKV / O-proj sumchecks rely on this transcript order)
  FS.absorb("q_com", q_com)
  FS.absorb("k_com", k_com)
  FS.absorb("v_com", v_com)
  FS.absorb("out_attn_com", out_attn_com)

  // ── 4. Residual 1 (homomorphic) ────────────────────────────────────────
  x_mid_com ← HyraxAdd(x_in_com, out_attn_com)

  // ── 5. LayerNorm 2 (uses x_norm2_com as y_com) ─────────────────────────
  ln2_io ← { x_com: x_mid_com, y_com: x_norm2_com }
  ln2_proof ← ProveLayerNorm(wit.ln2_wit, ln2_io, pk.ln2_vk,
                              σ_range=(block_rps[2], block_rvs[2]),
                              y_range=(block_rps[3], block_rvs[3]), FS)

  // ── 6. Absorb out_ffn_com (so transcript matches Phase 2 expectations) ─
  FS.absorb("y_com", out_ffn_com)

  return { ln1_proof, ln2_proof, block_range_m,
           x_norm1_com, q_com, k_com, v_com,
           out_attn_com, x_norm2_com, out_ffn_com, x_mid_com }
```

Residual 2 (`x_out_com = HyraxAdd(x_mid_com, out_ffn_com)`) is computed by the model-level prover after Phase 1 returns. The cross-block sumchecks in §4.2 (steps 4–9) consume the per-block intermediate commitments and the witness arrays directly; no further per-block sub-proofs are produced.

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
Algorithm Verify(VK, π, inst_attn, inst_ffn, public_x_in, public_logits):
  Input:
    VK              : verifying key (weight commitments + generators)
    π               : TransformerModelProof
    public_x_in     : verifier-provided input matrix
    public_logits   : verifier-provided expected output matrix
  Output:
    ACCEPT or REJECT

  // ── 1. Bind public I/O via re-commitment ────────────────────────────────
  Assert Com(public_x_in)   == π.x_in_com
  Assert Com(public_logits) == π.logits_com
  FS.init("piformer")
  FS.absorb("x_in_com", π.x_in_com)

  // ── 2. Phase 1 verify (per block): range + LN1 + LN2 ───────────────────
  Initialize accumulators: ln_acc_t, ln_acc_td, proj_acc_w, proj_acc_b,
                           lmh_acc_w, lmh_acc_b,
                           acc_range_sig, acc_range_y, acc_range_m, inter_acc
  x_cur_com ← π.x_in_com
  For ℓ = 0 to L-1:
    bp ← π.block_proofs[ℓ]
    block_r_vs ← VerifyRangeBatched(
        [bp.ln1_proof.σ_range_proof, bp.ln1_proof.y_range_proof,
         bp.ln2_proof.σ_range_proof, bp.ln2_proof.y_range_proof],
        bp.block_range_m, [ln_σ_n, ln_y_n, ln_σ_n, ln_y_n], 32, FS,
        acc_range_sig, acc_range_y, acc_range_m)
    VerifyLayerNorm(bp.ln1_proof, {x_cur_com, bp.x_norm1_com}, vk.ln1_vk,
                    block_r_vs[0..1], FS, ln_acc_t, ln_acc_td)
    FS.absorb(q_com, k_com, v_com, out_attn_com from bp)
    x_mid_com ← HyraxAdd(x_cur_com, bp.out_attn_com)
    VerifyLayerNorm(bp.ln2_proof, {x_mid_com, bp.x_norm2_com}, vk.ln2_vk,
                    block_r_vs[2..4], FS, ln_acc_t, ln_acc_td)
    FS.absorb(bp.out_ffn_com)
    x_cur_com ← HyraxAdd(x_mid_com, bp.out_ffn_com)

  // ── 3. Derive global r_td after ALL Phase 1 ────────────────────────────
  r_td ← FS.challenge("gkr_r_td", t_bits + d_bits)

  // ── 4–9. Six cross-block batch sumchecks (mirror prover §4.2) ──────────
  Verify batch_qkv      → recover r_k_qkv
  Verify batch_oproj    → recover r_k_o
  Verify batch_attn_out → recover batch_r_attn_out
  Verify batch_attn_ctx → recover batch_r_attn_ctx
  Verify per-block FFN Lasso proofs and ffn_lasso_bind opening
  Verify batch_ffn_y    → recover r_k_fy
  Verify batch_ffn_m    → recover r_k_m
  // Each sumcheck reduces L per-block claims to one batched final claim,
  // which is checked algebraically against per-block (W, b, X, ...) evals
  // produced by the corresponding cross-block batch open below.

  // ── 10. Final LayerNorm + LM head ──────────────────────────────────────
  (final_r_vs, _) ← VerifyRangeBatched(
      [π.final_ln_proof.σ_range_proof, π.final_ln_proof.y_range_proof],
      π.final_range_m, [ln_σ_n, ln_y_n], 32, FS, …)
  VerifyLayerNorm(π.final_ln_proof, {x_cur_com, π.final_ln_out_com},
                  VK.final_ln_vk, final_r_vs[0..1], FS, ln_acc_t, ln_acc_td)
  VerifyProjection(π.lm_head_proof, VK.lm_head_vk,
                   {x_com: π.final_ln_out_com}, FS, lmh_acc_w, lmh_acc_b)
  Assert HyraxVerify(π.logits_com, lm_y_claim, π.lm_head_logits_open, FS)

  // ── 11–13. Global intermediate batch open + 17 cross-block batch opens ─
  Assert HyraxVerifyBatch({Q_ℓ, K_ℓ, V_ℓ, Out_attn_ℓ, Out_ffn_ℓ}_ℓ at r_td,
                           π.inter_batch_open, …) == ACCEPT
  For each (matrices, point, batch_open) listed in §4.2 step 13:
      Assert HyraxVerifyBatch(matrices, point, batch_open, …) == ACCEPT

  // ── 14. Finalise deferred Hyrax accumulators (2 MSMs each) ─────────────
  Assert ln_acc_t.finalize(FS) == ACCEPT
  Assert ln_acc_td.finalize(FS) == ACCEPT
  Assert proj_acc_w.finalize(FS) == ACCEPT
  Assert proj_acc_b.finalize(FS) == ACCEPT
  Assert lmh_acc_w.finalize(FS) == ACCEPT
  Assert lmh_acc_b.finalize(FS) == ACCEPT
  Assert acc_range_sig.finalize(FS) == ACCEPT
  Assert acc_range_y.finalize(FS) == ACCEPT
  Assert acc_range_m.finalize(FS) == ACCEPT
  Assert inter_acc.finalize(FS) == ACCEPT

  // ── 15. Global Lasso verification (all φ(Q), φ(K) across all blocks) ───
  Assert VerifyLassoMulti(
      [inst_attn[ℓ].q_lasso, inst_attn[ℓ].k_lasso for ℓ in 0..L-1],
      π.all_lasso_proof, VK.lasso_vk, FS) == ACCEPT

  return ACCEPT
```

### 4.7 Block Phase 1 Verifier

The verifier walks each block in lockstep with the prover's Phase 1 (§4.3) — it never runs a stand-alone "block verifier" for the projection / attention / FFN sub-protocols, which are folded into the model-level cross-block sumchecks (§4.6 step 4–9).

```
Algorithm VerifyBlockPhase1(vk, bp, x_in_com, FS, accs):
  Input:
    vk           : TransformerBlockVerifyingKey
    bp           : TransformerBlockProof (only the Phase 1 fields are read here)
    x_in_com     : commitment to X_in for this block
  Output:
    x_out_com    : commitment to X_out, or REJECT (advances accumulators)

  // ── 0. Range batch (4 witnesses share one m_com) ───────────────────────
  block_r_vs ← VerifyRangeBatched(
      [bp.ln1_proof.σ_range_proof, bp.ln1_proof.y_range_proof,
       bp.ln2_proof.σ_range_proof, bp.ln2_proof.y_range_proof],
      bp.block_range_m, [ln_σ_n, ln_y_n, ln_σ_n, ln_y_n], 32, FS,
      accs.range_sig, accs.range_y, accs.range_m)

  // ── 1. LN1 ──────────────────────────────────────────────────────────────
  VerifyLayerNorm(bp.ln1_proof, {x_in_com, bp.x_norm1_com}, vk.ln1_vk,
                  block_r_vs[0..1], FS, accs.ln_t, accs.ln_td)

  // ── 2. Absorb Q/K/V/Out_attn coms (matches prover transcript) ──────────
  FS.absorb("q_com", bp.q_com)
  FS.absorb("k_com", bp.k_com)
  FS.absorb("v_com", bp.v_com)
  FS.absorb("out_attn_com", bp.out_attn_com)

  // ── 3. Residual 1 ──────────────────────────────────────────────────────
  x_mid_com ← HyraxAdd(x_in_com, bp.out_attn_com)

  // ── 4. LN2 ──────────────────────────────────────────────────────────────
  VerifyLayerNorm(bp.ln2_proof, {x_mid_com, bp.x_norm2_com}, vk.ln2_vk,
                  block_r_vs[2..4], FS, accs.ln_t, accs.ln_td)
  FS.absorb("y_com", bp.out_ffn_com)

  // ── 5. Residual 2 ──────────────────────────────────────────────────────
  return HyraxAdd(x_mid_com, bp.out_ffn_com)
```

After all $L$ blocks complete Phase 1, the verifier draws `r_td` from the transcript and proceeds to verify the six cross-block batch sumchecks (§4.6 step 4–9), using the per-block scalar evaluations stored in each `TransformerBlockProof` (`q_eval`, `k_eval`, `v_eval_rtd`, `out_attn_eval`, `out_ffn_eval`, `attn_phi_q_eval`, `attn_phi_k_eval`, `attn_ctx_eval`, `attn_v_eval`, `qkv_w_q_eval`, …) as the per-block claims being batched.

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

**Theorem 2 (Cross-Block Batch Sumcheck Soundness).** *Replacing $L$ independent per-block projection sumchecks (one per block, per protocol type) with a single `SumcheckProofMulti` whose round polynomials are random linear combinations $\sum_\ell \eta^{\ell-1} \cdot p_\ell(X)$ does not reduce soundness.*

*Proof.* The Fiat-Shamir challenge $\eta$ is drawn from the transcript *after* all per-block commitments and per-block claims have been absorbed. Therefore the prover is committed to the per-block polynomials $\{f_\ell, g_\ell\}_{\ell=0}^{L-1}$ and the per-block targets $\{H_\ell\}$ before learning $\eta$. If any single $\hat H_\ell \neq H_\ell$, the combined claim $\sum_\ell \eta^{\ell-1} \hat H_\ell$ differs from the honest combined claim $\sum_\ell \eta^{\ell-1} H_\ell$ by a non-zero polynomial in $\eta$ of degree $\leq L-1$, which the verifier's random $\eta$ catches except with probability $(L-1)/|\mathbb{F}_r|$. Conditional on the combined target being honest, the inner sumcheck has soundness $n\delta / |\mathbb{F}_r|$ by Lemma 1. The total error is therefore $\leq (L-1)/|\mathbb{F}_r| + n\delta / |\mathbb{F}_r|$, dominated by the latter for transformer-scale $L$. The same argument applies to all six cross-block sumchecks (`batch_qkv`, `batch_oproj`, `batch_attn_out`, `batch_attn_ctx`, `batch_ffn_y`, `batch_ffn_m`). $\square$

**Theorem 3 (Homomorphic Residual Soundness).** *The residual connections $X_{\mathsf{mid}} = X_{\mathsf{in}} + X_{\mathsf{out\_attn}}$ and $X_{\mathsf{out}} = X_{\mathsf{mid}} + X_{\mathsf{out\_ffn}}$, computed as $C_{\mathsf{mid}} = C_{\mathsf{in}} + C_{\mathsf{out\_attn}}$ by the verifier, are sound without any additional proof.*

*Proof.* Since $\mathsf{Com}(A) + \mathsf{Com}(B) = \mathsf{Com}(A + B)$ holds by the linearity of MSM (Pedersen-style commitment), the verifier's local computation of $C_{\mathsf{mid}}$ is deterministically correct given the binding of $C_{\mathsf{in}}$ and $C_{\mathsf{out\_attn}}$. Binding of these two commitments is guaranteed by Lemma 2. $\square$

**Theorem 4 (FFN Lasso-First Ordering Soundness).** *Running the activation Lasso before any cross-block FFN sumcheck challenges (so that $A$ is bound to the transcript prior to $r_{k,fy}$ and $r_{k,m}$) does not reduce soundness.*

*Proof.* The per-block FFN Lasso is absorbed into the transcript in Phase 7 of `Prove` (§4.2), strictly before the cross-block FFN-Y and FFN-M sumcheck challenges are derived. Therefore the prover is committed to specific Lasso outputs before learning the binding points. The verifier opens $A$ at the shared `ffn_lasso_bind_point` via a single batched Hyrax open (`ffn_lasso_bind_open`) and consumes those evaluations as public inputs to the cross-block FFN-Y batch sumcheck. A cheating prover who forges any $A_\ell(\cdot)$ value must either forge the Lasso outputs (caught by Lemma 4), forge the Hyrax opening (caught by Lemma 2), or forge the batched sumcheck (caught by Theorem 2). $\square$

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

| Optimization | Savings | Where |
|---|---|---|
| Cross-block batch sumchecks (1 sumcheck per type vs $L$) | $(L-1)$ sumcheck transcripts × 6 protocol types | §4.2 steps 4–9 |
| Batched QKV inside each cross-block sumcheck (1 vs 3) | 2 sumcheck transcripts per block | `batch_qkv` |
| Global intermediate batch open ($5L$ matrices at one point) | $5L - 1$ Hyrax opening proofs | `inter_batch_open` |
| Cross-block batch opens (one per matrix type, not per block) | $(L-1)$ Hyrax opens per type, $\times \sim 17$ types | §4.2 step 13 |
| FFN Lasso-first (no separate $A_{\mathsf{com}}$) | $\sqrt{Td_{ff}}$ G1 points | per block |
| Homomorphic residuals (no proof) | 2 Hyrax opens per block | per block |
| Global range batch (1 $m_{\mathsf{com}}$ vs $B_{rng}$) | $(B_{rng}-1) \times \sqrt{2^{16}}$ G1 | per block + final |
| Deferred batch accumulators (10 cross-cutting groups) | $(K-1) \times 2$ MSMs per group | model-level |
| Ternary weight encoding (no field mults in projection sumcheck) | $T \cdot d$ field mults → $T \cdot d$ adds per layer | every projection |

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

### Transcript Ordering Invariants

The implementation enforces several strict transcript-ordering rules that the soundness proofs rely on:

1. **Per-block range batch** is the very first action inside `commit_block_phase1`: all chunk commitments and the shared $m_{\mathsf{com}}$ are absorbed *before* any per-witness sumcheck challenges. `prove_layernorm` / `verify_layernorm` deliberately do *not* call range-proof functions themselves; the Phase 1 wrapper is responsible. Violating this ordering breaks Theorem 6.

2. **Phase 1 completes for ALL blocks** before the global `r_td = (r_t \| r_{\mathsf{out}})` challenge is sampled. This guarantees that every per-block intermediate commitment ($Q_\ell, K_\ell, V_\ell, \mathsf{Out\_attn}_\ell, \mathsf{Out\_ffn}_\ell, X_{\mathsf{norm}1,\ell}, X_{\mathsf{norm}2,\ell}$) is bound to the transcript before the cross-block batch sumcheck challenges are drawn.

3. **Per-block QKV/O-proj/FFN absorbs precede the corresponding batch $\eta$**. Inside the cross-block QKV loop, each block's $(W_Q, W_K, W_V, \alpha, b_Q, b_K, b_V)$ commitments and the per-block $(\lambda_\ell, \mu_\ell)$ challenges are absorbed before the model-level $\eta_{\mathsf{qkv}}$ is sampled. The same pattern holds for `batch_oproj`, `batch_ffn_y`, `batch_ffn_m`, `batch_attn_out`, and `batch_attn_ctx`.

4. **FFN Lasso commits $A$ before the FFN-Y/M batch challenges**. `prove_lasso` is invoked once per block before the corresponding $\eta_{\mathsf{ffn\_y}}$ challenge is drawn, ensuring the Theorem 4 ordering.

5. **Ten deferred μ-challenges** are advanced after the LM-head proof and before the global intermediate batch open. The verifier mirrors this `for _ in 0..10 { transcript.challenge_field(b"hyrax_group_mu") }` loop so that the Hyrax accumulators consume the same μ values the prover used.

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

`TransformerModelProof` carries:

- One `TransformerBlockProof` per block with the LN1/LN2 sub-proofs, the block-level `GlobalRangeM`, the seven intermediate Hyrax commitments, the per-block FFN Lasso proof and `M` commitment, the per-block `phi_q`/`phi_k` commitments, and the per-block scalar evaluations consumed by the cross-block batch sumchecks.
- Six model-level cross-block batched sumcheck proofs: `batch_qkv`, `batch_oproj`, `batch_attn_out`, `batch_attn_ctx`, `batch_ffn_y`, `batch_ffn_m`.
- One `inter_batch_open` covering the 5L intermediate matrices (Q, K, V, Out_attn, Out_ffn for each block) opened jointly at `r_td`.
- Roughly 17 cross-block batch opens for the per-type weight, bias, activation, and intermediate matrices at their respective shared evaluation points (`x_norm1`, `w_q`, `w_k`, `w_v`, `bias_q`, `bias_k`, `bias_v`, `w_o`, `bias_o`, `w2`, `w1`, `x_norm2`, `ffn_m_com`, `ffn_lasso_bind`, `phi_q`, `phi_k`, `v_attn`, `qk_lasso_bind`).
- One model-level `LassoMultiProof` (`all_lasso_proof`) covering all $\phi(Q)$ / $\phi(K)$ Lasso instances across every block.
- The final-LayerNorm sub-proof + `GlobalRangeM`, the LM-head projection sub-proof, the LM-head logits opening, and Hyrax commitments for the final-LN output and logits.
