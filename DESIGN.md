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

Each transformer block uses **sandwich normalization**: in addition to the two
classical pre-norms, the block applies a per-head LayerNorm to $Q$ and $K$
(QK-norm) and a post-attention LayerNorm to the merged attention output before
the residual sum. Concretely each block computes:

```
X_norm1  = LayerNorm(X_in,                     γ_ln1,  β_ln1)
Q, K, V  = X_norm1 · W_{Q,K,V}                                          (batched ternary projection)
Q_n      = LayerNorm(Q, γ_qn, β_qn)            (per-head QK-norm on Q)
K_n      = LayerNorm(K, γ_kn, β_kn)            (per-head QK-norm on K)
attn_num = φ(Q_n) · (φ(K_n)ᵀ V)                (un-normalized linear-attention numerator)
Z        = φ(Q_n) · Σ_s φ(K_n)_s               (per-token denominator, scalar per row)
attn_y   = floor(scale · attn_num / Z)         (fixed-point row normalization)
Out_attn = attn_y · W_O + b_O                                         (output projection, ternary)
attn_out = LayerNorm(Out_attn, γ_aon, β_aon)   (sandwich norm)
X_mid    = X_in + attn_out                                              (residual 1, homomorphic)
X_norm2  = LayerNorm(X_mid, γ_ln2, β_ln2)
Out_ffn  = FFN(X_norm2)                                                  (feed-forward network)
X_out    = X_mid + Out_ffn                                              (residual 2, homomorphic)
```

After all $L$ blocks a final LayerNorm and the LM head produce the output logits, so the model contains **$5L+1$ LayerNorms** in total.

The two normalizations that are not present in a textbook pre-norm transformer
are essential here:

1. **QK-norm** (`q_norm` / `k_norm`) pins the dynamic range of the kernel
   inputs $Q,K$ so that $\phi(Q)$, $\phi(K)$ and the denominator $Z$ stay
   inside the lookup-table range regardless of the single-$\alpha$ ternary
   projection scale. Without it, a poorly conditioned $\alpha$ collapses
   $\phi$ to a degenerate region and breaks the lookup index assumptions.
2. **Sandwich norm** (`attn_out_norm`) absorbs the unbounded scale of
   linear attention before it enters the residual stream. Softmax attention
   is row-stochastic, so its output is naturally bounded; the linear-attention
   numerator $\phi(Q)(\phi(K)^\top V)$ has no such bound, and without
   normalization the residual stream drifts in magnitude and destabilizes
   downstream LayerNorms.

Both are LayerNorms over the same primitive (§3.7), so they cost no new SNARK
machinery — they only enlarge the LN witness count, which is mostly absorbed
by the model-level batched LN proof of §4 and the bucketed range batch of
§3.6.

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

**Model-level multi-Lasso.** The end-to-end prover does not emit independent lookup proofs for every block. Instead, it uses two distinct mechanisms:

- For attention, all $\phi(Q_n)$ and $\phi(K_n)$ lookup instances across all blocks are batched into a single `LassoMultiProof` (`all_lasso_proof`) in committed-output mode: before the multi-Lasso batching challenges, the prover absorbs every `attn_phi_q_com` / `attn_phi_k_com`; a second sumcheck inside the multi-proof binds the combined lookup grand sum to those committed outputs.
- For FFN, the structure is rotated: a single cross-block `batch_ffn_y` sumcheck reduces "all $A_\ell(r) \cdot W_2(\cdot, r)$" to a terminal claim
  $$\sum_\ell w_\ell\,\widetilde A_\ell(r_t, r_{k_{fy}}) = c$$
  and a separate `LassoTerminalEvalProof` (`ffn_a_terminal_proof`) proves that this terminal claim is consistent with the Lasso lookup tables at the *same* terminal point. This is functionally equivalent to a committed-output multi-Lasso plus a Hyrax batch open of $\{A_\ell\}$, but folds them into one sub-protocol. The per-block `TransformerBlockProof.ffn_lasso_proof` field is an empty compatibility placeholder; only the model-level `ffn_a_terminal_proof` actually carries the FFN lookup proof.

The proof still carries raw lookup query indices (`ffn_lasso_query_indices` and `all_lasso_proof.all_query_indices`). Two complementary mechanisms tie them to the committed inputs:

1. **Quantization proof** (§3.5b) — binds each public index $\mathsf{idx}_j$ to the committed raw input value $\mathsf{raw}_j$ through the integer relation
   $$\mathsf{raw}_j \cdot s_{\mathrm{den}} + \lfloor s_{\mathrm{num}}/2 \rfloor = s_{\mathrm{num}} \cdot (\mathsf{idx}_j - \mathsf{zp}) + \mathsf{rem}_j, \qquad 0 \le \mathsf{rem}_j < s_{\mathrm{num}},$$
   committed as `ffn_quant_proof` / `qk_quant_proof`. This is the primary binding.
2. A redundant self-consistency check `ffn_lasso_bind_open` / `qk_lasso_bind_open` verifies, via a single Hyrax inner-product identity, that the proof's index-MLE opening vector is the one the verifier reconstructs from the public index list.

The FFN quantization proof binds against `ffn_m_com`; the QK quantization proof binds against the post-norm `q_norm_y_com` / `k_norm_y_com` (not the pre-norm `q_com` / `k_com`), since the Lasso queries are evaluated on $Q_n, K_n$.

### 3.5b Quantization Proof

The Lasso protocol works over *integer indices* $\mathsf{idx}_j \in [0, 2^B)$, while the committed input tensors (`q_norm_y_com`, `k_norm_y_com`, `ffn_m_com`) carry *signed* field-element values. The quantization proof bridges this gap. With per-layer quantization parameters $(s_{\mathrm{num}}, s_{\mathrm{den}})$ where $s_{\mathrm{num}}$ is a power of two (`scale_num.is_power_of_two()` is enforced) and zero-point $\mathsf{zp} = 2^{B-1}$ (centered encoding for $\ge 2$ sub-tables, $0$ for a single table), each raw value $r$ corresponds to the lookup index $\mathsf{idx} = \lfloor (r \cdot s_{\mathrm{den}} + s_{\mathrm{num}}/2) / s_{\mathrm{num}} \rfloor + \mathsf{zp}$.

For each lookup instance the prover:

1. Computes the remainder tensor $\mathsf{rem}_j = r_j s_{\mathrm{den}} + \lfloor s_{\mathrm{num}}/2 \rfloor - s_{\mathrm{num}}(\mathsf{idx}_j - \mathsf{zp})$ and commits it via Hyrax (`rem_coms`).
2. Adds $\mathsf{rem}$ to the global range batch with bit-width $\log_2 s_{\mathrm{num}}$ (so $0 \le \mathsf{rem}_j < s_{\mathrm{num}}$).
3. Samples a random $r$ ∈ $\mathbb{F}^{t_{\mathsf{bits}} + d_{\mathsf{bits}}}$ from the transcript.
4. Opens both the raw input commitment and the remainder commitment at $r$ via Hyrax batch opens.
5. The verifier checks the algebraic identity
   $$\widetilde r(\mathbf r) \cdot s_{\mathrm{den}} + \lfloor s_{\mathrm{num}}/2 \rfloor \cdot \widetilde 1(\mathbf r) \stackrel{?}{=} s_{\mathrm{num}} \cdot (\widetilde{\mathsf{idx}}(\mathbf r) - \mathsf{zp} \cdot \widetilde 1(\mathbf r)) + \widetilde{\mathsf{rem}}(\mathbf r)$$
   where $\widetilde{\mathsf{idx}}(\mathbf r)$ is computed by the verifier directly from the *public* index list. The indicator $\widetilde 1$ compensates for zero-padding when the matrix size is not a power of two.

Combined with the range proof of step 2, this argument is sound: a cheating prover who submits an index inconsistent with the committed raw value would have to find a non-zero remainder outside $[0, s_{\mathrm{num}})$, which fails the range proof, or violate the algebraic identity, which fails by Schwartz–Zippel over $r$. The two quantization proofs (FFN and QK) share machinery in `lookup/quantization.rs` and contribute two accumulators (`acc_quant_ffn`/`acc_quant_m` for FFN, `acc_quant_qk`/`acc_quant_qk_m` for QK) to the deferred Hyrax pool.

### 3.6 Global Batched Range Proof

The range proof proves that field elements lie in $[0, 2^{\mathsf{bits}})$ using
a chunked Lasso approach with `CHUNK_BITS = 16`. The current implementation
exposes two width buckets — $\mathsf{bits} = 32$ (FAST) and $\mathsf{bits} = 64$
(WIDE) — and routes every range witness to the narrowest bucket that contains
it (`choose_range_bits` in `prover/src/prover.rs`). LayerNorm $\sigma$/$y$
witnesses use 64-bit range proofs (`LAYERNORM_RANGE_BITS = 64`); the optional
attention-normalization residuals (§3.9) also use 64-bit range proofs
(`ATTN_NORM_RANGE_BITS = 64`).

**Per-witness setup.** Each witness $V \in \mathbb{F}^{2^n}$ is split into 16-bit
chunks. For 32-bit width:
$$V[i] = V_{lo}[i] + 2^{16} \cdot V_{hi}[i]$$
For 64-bit width the witness is split into four 16-bit chunks
$V_{c_0}, V_{c_1}, V_{c_2}, V_{c_3}$ with the analogous reconstruction.

**Two-phase protocol with per-bucket batching.** All range witnesses across
the entire model are collected into one list and partitioned by bit-width
bucket. Each non-empty bucket runs the protocol below independently, producing
one $m_{\mathsf{com}}$ per bucket. For $B$ witnesses in a single bucket:

*Phase 1 — Commit all chunks + shared multiplicity:*
1. For each witness $b$ and each chunk $c$: commit $V^{(b)}_c$ via Hyrax →
   chunk commitments $\mathsf{cc}^{(b)}_c$.
2. Merge all chunk arrays into a global multiplicity array $m$: $m[v]$ counts
   the total occurrences of value $v$ across all witnesses *and all chunks*
   in this bucket.
3. Commit $m$ via Hyrax → single shared $m_{\mathsf{com}}$.

*Phase 2 — Per-witness sumcheck and openings:*
4. For each witness $b$: absorb claim $V^{(b)}(1,\ldots,1)$, run sumcheck to
   bind $V^{(b)}$ to random point $r_v^{(b)}$.
5. Verify chunk reconstruction at $r_v^{(b)}$ via one Hyrax batch open over
   $\{\mathsf{cc}^{(b)}_c\}_c$.

*Shared multiplicity check:*
6. Sample $r_m \stackrel{\$}{\leftarrow} \mathbb{F}^{16}$; open $m(r_m)$ via Hyrax.
7. LogUp identity check: $\sum_{b,c} \mathsf{cc}^{(b)}_c(r_m) = m(r_m) \cdot T_{\mathsf{id}}(r_m)$ where $T_{\mathsf{id}}[i] = i$.

**Key saving.** Instead of one $m_{\mathsf{com}}$ per witness, there is exactly
one $m_{\mathsf{com}}$ per width bucket per model. For the LN bucket alone,
$B = 10L + 2$ (5 LayerNorms per block × 2 witnesses each, plus 2 for the
final LN); when normalized attention is enabled, the optional attn-norm bucket
adds another $2L$ witnesses (rem and diff per block, §3.9). The savings scale
with $L$: $(B-1)$ saved $m_{\mathsf{com}}$ MSMs per bucket.

### 3.7 LayerNorm Circuit

LayerNorm is proved without any division gates using **constraint fusion**.
The witness for one LN exposes seven per-row tensors:

| Symbol | Meaning |
|--------|---------|
| $x \in \mathbb{F}^{T \times d}$, $y \in \mathbb{F}^{T \times d}$ | LN input / output |
| $\mathsf{sum\_x}[i] = \sum_j x[i][j]$ | row sum |
| $\mathsf{sq\_sum\_x}[i] = \sum_j x[i][j]^2$ | row sum of squares |
| $\mathsf{sum\_x\_sq}[i] = \mathsf{sum\_x}[i]^2$ | square of the row sum |
| $\sigma[i]$ | integer floor of $\sqrt{\mathsf{var\_x}[i]}/d$ |
| $\sigma\mathsf{\_sq\_scaled}[i] = (d \cdot \sigma[i])^2$ | committed for the residual sumcheck |

The integer variance is recovered from these via
$$\mathsf{var\_x}[i] = d \cdot \bigl(d \cdot \mathsf{sq\_sum\_x}[i] - \mathsf{sum\_x\_sq}[i]\bigr).$$

The prover commits $\mathsf{sum\_x}$, $\mathsf{sq\_sum\_x}$, and $\sigma$ (three internal Hyrax commitments per LN); $\mathsf{sum\_x\_sq}$ and $\sigma\mathsf{\_sq\_scaled}$ appear inside the sigma-residual sumcheck and are bound algebraically rather than separately committed.

**Step 1 — Mean sumcheck (degree 2).** Prove $\mathsf{sum\_x}(r_t) = \sum_{j \in \{0,1\}^{d_{\mathsf{bits}}}} x_{\mathsf{col}}(j)$ where $x_{\mathsf{col}}$ is $\widetilde x$ with the row variables fixed to a transcript challenge $r_t \in \mathbb{F}^{t_{\mathsf{bits}}}$.

**Step 2 — Square-sum sumcheck (cubic).** Prove $\mathsf{sq\_sum\_x}(r_t) = \sum_{j} x_{\mathsf{col}}(j)^2$ via a degree-3 sumcheck over three copies of $x_{\mathsf{col}}$.

**Step 3 — Sigma-residual sumcheck (cubic multi-batched).** Sample $r_{\mathsf{sig}_t}$ (the row-prefix of the sigma range-proof challenge), then prove jointly
$$\mathsf{sum\_x\_sq}(r_{\mathsf{sig}_t}) = \sum_{i} \widetilde{\mathsf{eq}}(r_{\mathsf{sig}_t}, i) \cdot \mathsf{sum\_x}(i)^2,$$
$$\sigma\mathsf{\_sq\_scaled}(r_{\mathsf{sig}_t}) = \sum_{i} \widetilde{\mathsf{eq}}(r_{\mathsf{sig}_t}, i) \cdot (d \cdot \sigma(i))^2,$$
folded into a single cubic multi-batched sumcheck under a Fiat–Shamir scalar $\lambda_{\mathsf{sig}}$. This binds $\mathsf{sum\_x\_sq}$ and $\sigma\mathsf{\_sq\_scaled}$ to derivable squares of the already-committed $\mathsf{sum\_x}$ and $\sigma$, eliminating the need for separate commitments.

**Step 4 — Sigma range proof (floor-sqrt).** With $\mathsf{var\_x}(r_{\mathsf{sig}_t}) := d \cdot (d \cdot \mathsf{sq\_sum\_x}(r_{\mathsf{sig}_t}) - \mathsf{sum\_x\_sq}(r_{\mathsf{sig}_t}))$, the prover proves
$$\mathsf{var\_x} - (d\sigma)^2 \geq 0 \quad\text{and}\quad (d\sigma + d)^2 - 1 - \mathsf{var\_x} \geq 0$$
by range-checking the lo/hi residual pair via the global batched range proof (§3.6) with $\mathsf{LAYERNORM\_RANGE\_BITS} = 64$.

**Step 5 — Y constraint fusion (cubic multi-batched).** At a transcript point $(r_{y_t}, r_{y_d})$ derived from the global Y-range batch, fuse the two legs
$$\gamma \odot \widetilde x_{r_y} \quad\text{and}\quad \sigma_{r_y} \odot \widetilde y_{r_y}$$
into one cubic multi-batched sumcheck under a Fiat–Shamir scalar $\alpha$. The final Y range proof verifies
$$\gamma_j(d \cdot x[i][j] - \mathsf{sum\_x}[i]) + \beta_j(d\sigma[i]) \approx d\sigma[i] \cdot y[i][j]$$
up to integer rounding through lo/hi residual range proofs. The verifier evaluates $\gamma$ and $\beta$ directly from the public VK in $O(d)$ operations.

**Total per LN:** 1 quadratic + 3 cubic (multi-batched) sumchecks plus 2 range witnesses (σ and y), all verified at four shared random points.

**Model-level batched LN proof.** The end-to-end prover (§4.2) does **not**
emit one independent LayerNorm sub-proof per LN. Instead, a single call to
`prove_layernorms_batched` collects all $5L+1$ LN witnesses
(`ln1`, `q_norm`, `k_norm`, `attn_out_norm`, `ln2` per block, plus the final
LN) at the end of the prover and runs one batched cubic sumcheck for the
combined Y constraint, plus one shared mean / variance / sigma sumcheck per
LN folded by Fiat–Shamir powers. The verifier mirrors this with
`verify_layernorms_batched` (`prover/src/verifier.rs:1243`). This batch is
orthogonal to the cross-block batching of §4.2 — it folds claims of the
*same protocol type* across *different positions* within a block as well as
across blocks.

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

For a single head ($n_{\mathsf{heads}} = 1$), the linear-attention output that
the prover proves is

$$\mathsf{attn\_y}[t][j] \;=\; \left\lfloor \frac{\mathsf{ATTN\_NORM\_SCALE}\,\cdot\,\mathsf{attn\_num}[t][j]}{Z[t]} \right\rfloor, \qquad
\mathsf{Out}_{\mathsf{attn}} = \mathsf{attn\_y} \cdot W_O + b_O,$$

where $\mathsf{attn\_num} = \phi(Q_n)\,(\phi(K_n)^\top V)$ is the un-normalized
numerator and $Z[t] = \phi(Q_n)_t \cdot \sum_s \phi(K_n)_s$ is the per-row
denominator ($Q_n, K_n$ are the post-QK-norm tensors of §2.1). The integer
constant $\mathsf{ATTN\_NORM\_SCALE} = 64$ (`prover/src/prover.rs:121`)
re-introduces fractional precision before the floor division.

The five core claims for a single head are:

| Step | Statement | Protocol |
|------|-----------|----------|
| 1 | $\Phi_Q[t][d] = \phi(Q_n[t][d])$ | Lasso (§3.5), bound to `q_norm_y_com` |
| 2 | $\Phi_K[t][d] = \phi(K_n[t][d])$ | Lasso, bound to `k_norm_y_com` |
| 3 | $C[i][j] = \sum_t \Phi_K[t][i] \cdot V[t][j]$ | Degree-2 sumcheck over $t$ |
| 4 | $\mathsf{attn\_num}[t][j] = \sum_i \Phi_Q[t][i] \cdot C[i][j]$ | Degree-2 sumcheck over $i$ |
| 5 | $\mathsf{attn\_y}, Z$ correctly normalize $\mathsf{attn\_num}$ | Cubic multi-batched sumcheck + range (below) |

Steps 3–4 use a random-entry reduction at $(r_t, r_{k_o})$ and are folded
across blocks by `batch_attn_out` and `batch_attn_ctx` (§4.2). Lookup binding
to $Q_n, K_n$ (rather than the pre-norm $Q, K$) is critical: `q_com` and
`k_com` commit to $W_Q\!\cdot\!\mathsf{ln1\_y}$ before QK-norm, while
`q_norm_y_com` / `k_norm_y_com` commit to the post-QK-norm tensors fed to
$\phi$. The Lasso index-binding step opens *both* commitments at the
transcript-derived point (§4.2 step 14) so a cheating prover cannot mix
pre-norm and post-norm values.

**Step 5 — Floor-division proof.** When `attention_mode = "normalized_fixed"`
the witness includes the auxiliary tensors

| Tensor | Meaning |
|--------|---------|
| $\mathsf{attn\_num}[t][j]$ | un-normalized numerator (step 4 output) |
| $\mathsf{attn\_y}[t][j]$  | claimed normalized output (output of step 5) |
| $Z[t]$ | denominator (one scalar per row, broadcast) |
| $\mathsf{rem}[t][j]$ | remainder, $\in [0, Z[t])$ |
| $\mathsf{diff}[t][j]$ | $Z[t] - 1 - \mathsf{rem}[t][j]$, also $\in [0, Z[t])$ |

The prover commits all five (`attn_num_com`, `attn_norm_com`, `attn_z_com`,
`attn_rem_com`, `attn_diff_com`) and runs a single random-point cubic
multi-batched sumcheck at $r_{\mathsf{norm}} \in \mathbb{F}^{t_\mathsf{bits}+d_\mathsf{bits}}$
that enforces, per element,

$$\mathsf{ATTN\_NORM\_SCALE}\cdot\mathsf{attn\_num} \;-\; \mathsf{rem} \;-\; Z\!\cdot\!\mathsf{attn\_y} \;=\; 0,$$
$$\lambda\,\bigl(\,Z \;-\; 1 \;-\; \mathsf{rem} \;-\; \mathsf{diff}\,\bigr) \;=\; 0,$$

merged into one cubic claim by the Fiat–Shamir scalar $\lambda$
(`prover/src/prover.rs:1599`). Range proofs on `rem` and `diff` (added to the
64-bit bucket of §3.6) prove $\mathsf{rem}, \mathsf{diff} \geq 0$, which
combined with the second identity gives $0 \le \mathsf{rem} < Z$, the integer
floor-division witness.

**Z derivation.** A second sumcheck (`attn_z_sumcheck`) at $r_{\mathsf{norm},t}$
proves $Z[t] = \phi(Q_n)_t \cdot \sum_s \phi(K_n)_s$ from the same $\Phi_Q,
\Phi_K$ MLEs already committed, reusing `phi_q_com` and `phi_k_com`. In the
causal mode an additional cubic sumcheck binds $Z$ to the prefix sum
$\sum_{s \le t} \phi(K_n)_s$ instead.

**Cross-block batching of attention.** The end-to-end prover (§4.2) batches
steps 3 and 4 across all $L$ blocks. Two `SumcheckProofMulti` instances —
`batch_attn_out` and `batch_attn_ctx` — share one set of sumcheck challenges
across all blocks, with per-block claims combined by Fiat-Shamir powers of
$\eta$. The step-5 cubic sumcheck and the $Z$ sumcheck are themselves
multi-batched (one shared sumcheck across all blocks, $7\!\cdot\!L$ summand
slots in step 5). Per-block opening claims for $\Phi_Q, \Phi_K, V$,
$\mathsf{attn\_num}, \mathsf{attn\_y}, \mathsf{rem}, \mathsf{diff}$ at the
shared evaluation points are folded into the global cross-block batch opens
(`attn_norm_r_batch_open`, `attn_norm_attn_point_open`).

**Causal-mode protocol variants.** When `inst_attn.causal` is set, the
prover runs a different (but structurally analogous) set of sumchecks:

| Non-causal | Causal |
|------------|--------|
| `batch_attn_out` (degree-2 multi over $k$) | `batch_attn_out_causal` (cubic multi over $(t,k)$, third multiplicand $\widetilde{\mathsf{eq}}(r_t,\cdot)$ verifier-reproducible) |
| `batch_attn_ctx`  (degree-2 multi over $t$) | `batch_attn_ctx_causal` (cubic multi over $(s,a,b)$, third multiplicand $\mathrm{suffix}(s)\cdot\widetilde{\mathsf{eq}}_a(a)\cdot\widetilde{\mathsf{eq}}_b(b)$ verifier-reproducible) |
| `attn_z_sumcheck` + `attn_z_ksum_sumcheck` (two degree-2 multis) | `attn_z_causal_sumcheck` (one cubic multi over $(i,s,a)$, third multiplicand $\widetilde{\mathsf{eq}}_t(i)$) |

The causal context $C[i] = \sum_{s \le i} \phi(K_n)_s \cdot V_s$ is *not*
separately committed; its prefix value at the OUT-sumcheck terminal point
is recovered as the cubic CTX sumcheck's $f \cdot g$ leaf, and the leaf
constraint is verifier-reproducible because the prefix kernel
$\mathrm{suffix}(s) \widetilde{\mathsf{eq}}_a(a) \widetilde{\mathsf{eq}}_b(b)$ is
a public function of the OUT terminal point and $r_{k_o}$. This avoids any
proof-carried causal-context opening at the cost of one extra round of
sumcheck per cubic instance.

### 3.10 FFN Circuit

Proves $\mathsf{Out} = \phi(X \cdot W_1) \cdot W_2$:

**GKR backward ordering:**
1. **Activation** $A = \phi(M)$: the prover commits `ffn_m_com = Com(M)` and absorbs it into the transcript; $A$ itself is never independently committed in the current proof bundle.
2. **Second projection** $\mathsf{Out} = A \cdot W_2$: a cross-block `batch_ffn_y` degree-2 sumcheck reduces $\widetilde{Y}(r_t, r_{\mathsf{out}}) - \widetilde{b}(r_{\mathsf{out}}) = \alpha\sum_k A(r_t, k) \cdot W_2(k, r_{\mathsf{out}})$ across all blocks to a single terminal claim
   $$\textstyle\sum_\ell w_\ell\, \widetilde A_\ell(r_t, r_{k_{fy}}) \;=\; c$$
   where $w_\ell = \eta^\ell \cdot \widetilde W_{2,\ell}(r_{k_{fy}}, r_{\mathsf{out}})$. A `LassoTerminalEvalProof` (`ffn_a_terminal_proof`) then proves $c$ is consistent with applying the Lasso lookup tables to the per-block index streams, evaluated as an MLE at the terminal point. This fuses the equivalent of an `ffn_a_com` commitment, a Lasso multi-proof's output binding, and a Hyrax batch open of $\{A_\ell\}$ into one sub-protocol with no proof-carried $A_\ell$ outputs.
3. **First projection** $M = X_{\mathsf{n2}} \cdot W_1$: cross-block `batch_ffn_m` degree-2 sumcheck at shared random entry $(\mathsf{rx}_m, \mathsf{ry}_m)$. $M_{\mathsf{com}}$ is kept (field values of $M$ can be negative, unlike Lasso query indices, which is why the quantization proof of §3.5b is necessary to bridge the two).

`ffn_quant_proof` (§3.5b) binds the public FFN lookup indices to `ffn_m_com`, and `ffn_lasso_bind_open` is the auxiliary index-MLE self-consistency check.

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

The model prover does **not** prove blocks independently. Instead, after a
per-block "Phase 1" that commits intermediate matrices and absorbs all five
LayerNorm IO commitments into the transcript, the prover runs a small number
of *cross-block* batched sumchecks that cover one protocol type (e.g. QKV
projection) across all $L$ blocks at once, with per-block claims combined by
Fiat-Shamir powers of $\eta$. The LayerNorm sumchecks themselves are deferred
to a single model-end `prove_layernorms_batched` call that covers all $5L+1$
LNs in the model, and the global range proof is bucketed by bit-width over
*all* range witnesses (LN $\sigma$/$y$ plus optional attn-norm rem/diff)
across all blocks.

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

  // ── 2. Phase 1 (per block): commit intermediates, absorb 5 LN IO coms ──
  x_cur_com ← x_in_com
  phase1[0..L-1] ← []
  For ℓ = 0 to L-1:
    p1 ← CommitBlockPhase1(W.block_witnesses[ℓ], x_cur_com, PK.block_pks[ℓ], FS)
    // p1 contains:
    //   x_norm1_com, q_com, k_com, v_com,
    //   q_norm_y_com, k_norm_y_com,            (post-QK-norm tensors → φ inputs)
    //   attn_num_com?, attn_norm_com?,
    //   attn_z_com?, attn_rem_com?, attn_diff_com?,    (when normalized mode)
    //   out_attn_com,                          (W_O · attn_y + b_O)
    //   attn_out_norm_y_com,                   (sandwich-norm output)
    //   x_norm2_com, out_ffn_com, x_mid_com
    // The 5 LN IO commitments (ln1, q_norm, k_norm, attn_out_norm, ln2) and
    // the auxiliary attn-norm commitments are absorbed into FS in fixed order;
    // no per-block LN sumcheck is produced here — it is deferred to step 11b.
    // Residual 1 = HyraxAdd(x_cur_com, attn_out_norm_y_com)  (sandwich norm).
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
  // 6a. attn_num_ℓ(r_t, r_k_o) = Σ_k phi_q_ℓ(r_t, k) · ctx_ℓ(k, r_k_o)
  (batch_attn_out, batch_r_attn_out) ← ProveSumcheckMulti({phi_q_ℓ@(r_t, ·)},
                                                           {ctx_ℓ@(·, r_k_o)}, ...)
  // 6b. ctx_ℓ(batch_r_attn_out, r_k_o) = Σ_t phi_k_ℓ(t, batch_r_attn_out) · v_ℓ(t, r_k_o)
  (batch_attn_ctx, batch_r_attn_ctx) ← ProveSumcheckMulti({phi_k_ℓ@(·, batch_r_attn_out)},
                                                           {v_ℓ@(·, r_k_o)}, ...)

  // ── 6c. Cross-block attention normalization sumcheck (when has_attn_norm) ─
  // Proves, per element of (t, j) at one shared random point r_norm:
  //   ATTN_NORM_SCALE · attn_num - rem - Z · attn_y = 0
  //   λ ( Z - 1 - rem - diff ) = 0
  // merged into one cubic multi-batched sumcheck across all L blocks.
  r_norm  ← FS.challenge_vec("attn_norm_r", t_bits + d_bits)
  λ       ← FS.challenge("attn_norm_lambda")
  attn_norm_sumcheck ← ProveSumcheckCubicMultiBatched(
      [eq(r_norm,·)] × 7L,
      [n, r, z, z, 1, r, d]  per block (7L summand slots),
      [1, 1, y, 1, 1, 1, 1]  per block,
      weights = [SCALE, -1, -1, λ, -λ, -λ, -λ]  per block,
      target = 0, FS)
  // Then a separate sumcheck binds Z[t] = phi(Q_n)_t · Σ_s phi(K_n)_s
  // (or the causal prefix version) at r_norm_t; rem and diff are added to the
  // 64-bit range-batch witness list.
  attn_z_sumcheck ← ProveSumcheckMultiBatched(
      [phi_q_ℓ@(r_norm_t, ·)], [phi_k_sum_ℓ@(·)], η_z, claim_z, FS)

  // ── 7. Global FFN activation Lasso + M index binding ───────────────────
  For ℓ = 0 to L-1:
    Absorb (W1, W2) commitments
    A_ℓ ← W.ffn.a;  ffn_a_com_ℓ ← Com(A_ℓ)
    M_ℓ ← W.ffn.m;  ffn_m_com_ℓ ← Com(M_ℓ);  FS.absorb(ffn_m_com_ℓ)
    Collect activation lookup instance, query indices, and output binding
      (ffn_a_com_ℓ, A_ℓ)

  ffn_lasso_π ← ProveLassoMulti(
      [inst.ffn.activation_lasso for ℓ in 0..L-1],
      query_indices=[indices(M_ℓ)]_ℓ,
      output_bindings=[(ffn_a_com_ℓ, A_ℓ)]_ℓ,
      FS)

  // Bind FFN lasso indices to a shared (rx, ry) point and open all M_ℓ at once
  Absorb concatenated ffn_lasso_π query indices
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

  // ── 10. Final LayerNorm IO commitment ───────────────────────────────────
  final_ln_out_com ← Com(W.final_ln_wit.y)
  FS.absorb final_ln IO commitments (deferred sumcheck — see step 11b)

  // ── 11. Global range batch (all buckets, all witnesses in the model) ────
  // Witness list (in fixed order):
  //   block ℓ ∈ [0..L): ln1.σ, ln1.y, ln2.σ, ln2.y,
  //                     q_norm.σ, q_norm.y, k_norm.σ, k_norm.y,
  //                     attn_out_norm.σ, attn_out_norm.y          (10 per block)
  //   final_ln.σ, final_ln.y                                       (2)
  //   when has_attn_norm: per block: attn_norm.rem, attn_norm.diff (2L)
  // Each witness is routed to the narrowest of {32, 64} bit buckets.
  // One ProveRangeBatched call per non-empty bucket → one m_com per bucket.
  for bits ∈ {32, 64}:
      bucket ← witnesses with choose_range_bits(w) == bits
      if bucket non-empty:
          (rps[bucket], m[bits], rvs[bucket]) ← ProveRangeBatched(bucket, bits, FS)

  // ── 11b. Cross-LN batched proof (covers all 5L+1 LayerNorms) ───────────
  // One call to prove_layernorms_batched produces a single transcript that
  // bundles every LN's mean / variance / Y-fusion claim.  The LN witnesses
  // are presented in fixed order:
  //   for ℓ in 0..L: ln1, q_norm, k_norm, attn_out_norm, ln2
  //   then final_ln
  // The prover passes the σ/y RangeWitnessProofs from step 11 as inputs, so
  // the LN's σ-floor-sqrt and Y-residual checks reuse those range proofs.
  ln_batched_π ← ProveLayerNormsBatched(
      witnesses=[ln1_ℓ, q_norm_ℓ, k_norm_ℓ, attn_out_norm_ℓ, ln2_ℓ for ℓ]
                 ++ [final_ln],
      io_coms=..., σ_ranges=..., y_ranges=..., FS)

  // ── 12. LM head ─────────────────────────────────────────────────────────
  logits_com ← Com(W.lm_head_wit.y)
  (lm_head_π, lm_y_claim, _) ← ProveProjection(PK.lm_head_pk, W.lm_head_wit,
                                                {x_com: final_ln_out_com}, FS, None)
  lm_head_logits_open ← HyraxOpen(W.lm_head_wit.y, lm_y_claim.point, ...)

  // ── 13. Advance transcript for accumulator μ-challenges ─────────────────
  // The verifier draws 13 μ-challenges for the post-LM-head accumulator
  // pool: inter, ln_t, ln_td, proj_w, proj_b, lmh_w, lmh_b, rng_sig, rng_y,
  // rng_m, quant_ffn, quant_m, attn_norm.  It also reads two read-only ρ
  // fuse challenges (hyrax_fuse_td, hyrax_fuse_range_m) used by the
  // shared-params `finalize_many_with_mus` calls.
  For 13 deferred accumulators:
      FS.challenge("hyrax_group_mu")
  FS.challenge_readonly("hyrax_fuse_td")
  FS.challenge_readonly("hyrax_fuse_range_m")

  // ── 14. Global intermediate batch open ──────────────────────────────────
  // 5L matrices (Q, K, V, Out_attn, Out_ffn) opened at shared r_td.
  // (q_norm_y_com, k_norm_y_com, attn_out_norm_y_com, attn_num_com,
  //  attn_norm_com are opened as part of step 15 cross-block batches.)
  inter_batch_open ← HyraxOpenBatch({Q_ℓ, K_ℓ, V_ℓ, Out_attn_ℓ, Out_ffn_ℓ}_{ℓ=0..L-1},
                                     r_td, ν_td, σ_td, FS)

  // ── 15. Cross-block batch opens for weights, biases, activations ────────
  // One HyraxOpenBatch per (matrix-type, evaluation-point) pair:
  //   x_norm1 @ (r_t, r_k_qkv);   Wq, Wk, Wv @ (r_k_qkv, r_out);   bq, bk, bv @ r_out
  //   Wo @ (r_k_o, r_out);        bo @ r_out
  //   W2 @ (r_k_fy, r_out);       A @ (r_t, r_k_fy);
  //   W1 @ (r_k_m, ry_m);         x_norm2 @ (rx_m, r_k_m)
  //   M  @ (rx_m, ry_m)
  //   phi_q @ (r_t, batch_r_attn_out);   phi_k @ (batch_r_attn_ctx, batch_r_attn_out)
  //   v @ (batch_r_attn_ctx, r_k_o)      (per-block opening; v_attn_batch_open)
  //   (when has_attn_norm)
  //   [attn_num | attn_norm | rem | diff] @ r_norm                (one batch open)
  //   z_ℓ @ r_norm_t                                              (separate, smaller PCS params)
  //   [attn_num | attn_norm] @ (r_t, r_k_o)                       (binds attn_num to step-4 sumcheck)
  // (See prover.rs cross-block batch-open section for the full list.)

  // ── 14. Global attention Lasso batch (all φ(Q), φ(K) blocks) ───────────
  Absorb concatenated Q/K lasso indices
  qk_lasso_bind_point ← FS.challenge_vec(t_bits + d_bits)
  qk_lasso_bind_open ← HyraxOpenBatch({Q_ℓ, K_ℓ}_{ℓ=0..L-1},
                                       qk_lasso_bind_point, ...)
  all_lasso_π ← ProveLassoMulti(
      [inst.attn[ℓ].q_lasso, inst.attn[ℓ].k_lasso for ℓ in 0..L-1],
      query_indices=[indices(Q_ℓ), indices(K_ℓ)]_ℓ,
      output_bindings=[(phi_q_com_ℓ, ΦQ_ℓ), (phi_k_com_ℓ, ΦK_ℓ)]_ℓ,
      FS)

  return TransformerModelProof {
    x_in_com, block_proofs[0..L-1],
    final_range_m, final_ln_proof, lm_head_proof,
    final_ln_out_com, logits_com, lm_head_logits_open,
    batch_qkv, batch_oproj, batch_ffn_y, batch_ffn_m,
    batch_attn_out, batch_attn_ctx,
    inter_batch_open,
    {x_norm1, w_q, w_k, w_v, bias_q, bias_k, bias_v,
     w_o, bias_o, w2, w1, x_norm2,
     ffn_a, ffn_m_com, ffn_lasso_bind, phi_q, phi_k, v_attn,
     qk_lasso_bind}_batch_open,
    ffn_lasso_π, all_lasso_π
  }
```

Each `TransformerBlockProof` carries up to 15 commitments — 10 always-present (`x_norm1_com`, `q_com`, `k_com`, `v_com`, `q_norm_y_com`, `k_norm_y_com`, `out_attn_com`, `attn_out_norm_y_com`, `x_norm2_com`, `out_ffn_com`) plus 5 optional under normalized attention (`attn_num_com`, `attn_norm_com`, `attn_z_com`, `attn_rem_com`, `attn_diff_com`) — the `ffn_m_com`, the `phi_q`/`phi_k` commitments, and the per-block scalar evaluations consumed by the cross-block batch sumchecks (`q_eval`, `k_eval`, `v_eval_rtd`, `attn_phi_q_eval`, `attn_phi_k_eval`, `attn_ctx_eval`, etc.). No per-block LN sub-proof and no per-block FFN Lasso proof are stored — both are subsumed by the model-level batched protocols. The `TransformerBlockProof.ffn_lasso_proof` field is an empty compatibility placeholder; the real FFN activation argument is the model-level `ffn_a_terminal_proof` plus the QK `all_lasso_proof`.

### 4.3 Block Phase 1 (Per-Block Commit + LN IO Absorption)

The only truly per-block step is **Phase 1**: it commits the per-block
intermediate matrices and absorbs all five LN IO commitments (plus the
optional attention-normalization auxiliary commitments) into the transcript
in a fixed order. **No per-block LN sumcheck is emitted** in Phase 1 — the
LN sumchecks are deferred to the model-end batched call of §4.2 step 11b,
and the LN range proofs are deferred to the global bucketed range batch of
§4.2 step 11.

```
Algorithm CommitBlockPhase1(wit, x_in_com, pk, FS):
  Input:
    pk      : TransformerBlockVerifyingKey (carries projection / attention PKs)
    wit     : { ln1_wit, q_proj_wit, k_proj_wit, v_proj_wit,
                q_norm_wit, k_norm_wit, attn_wit, o_proj_wit,
                attn_out_norm_wit, ln2_wit, ffn_wit }
    x_in_com: Hyrax commitment to X_in ∈ F^{T×d}
  Output:
    BlockPhase1Data
      { x_norm1_com, q_com, k_com, v_com,
        q_norm_y_com, k_norm_y_com,
        attn_num_com?, attn_norm_com?,
        attn_z_com?, attn_rem_com?, attn_diff_com?,
        out_attn_com, attn_out_norm_y_com,
        x_norm2_com, out_ffn_com, x_mid_com }

  // ── 0. Commit intermediate matrices ────────────────────────────────────
  x_norm1_com         ← Com(wit.ln1_wit.y)
  q_com               ← Com(wit.q_proj_wit.y)            // pre-QK-norm Q
  k_com               ← Com(wit.k_proj_wit.y)            // pre-QK-norm K
  v_com               ← Com(wit.v_proj_wit.y)
  q_norm_y_com        ← Com(wit.q_norm_wit.y)            // post-QK-norm Q (φ input)
  k_norm_y_com        ← Com(wit.k_norm_wit.y)            // post-QK-norm K (φ input)
  if has_attn_norm:
      attn_num_com    ← Com(wit.attn_wit.out)            // un-normalized attn_num
      attn_norm_com   ← Com(wit.attn_wit.normalized_out) // attn_y after floor div
      attn_z_com      ← Com(wit.attn_wit.norm_z)         // per-row Z (length T)
      attn_rem_com    ← Com(wit.attn_wit.norm_rem)
      attn_diff_com   ← Com(wit.attn_wit.norm_diff)
  out_attn_com        ← Com(wit.o_proj_wit.y)            // W_O · attn_y + b_O
  attn_out_norm_y_com ← Com(wit.attn_out_norm_wit.y)     // sandwich-norm output
  x_norm2_com         ← Com(wit.ln2_wit.y)
  out_ffn_com         ← Com(wit.ffn_wit.y)

  // ── 1. Absorb LN1 IO commitments (sumcheck deferred to step 11b) ───────
  FS.absorb("x_com", x_in_com);  FS.absorb("y_com", x_norm1_com)

  // ── 2. Absorb projection / attention-norm commitments ──────────────────
  FS.absorb("q_com", q_com);  FS.absorb("k_com", k_com);  FS.absorb("v_com", v_com)
  if has_attn_norm:
      FS.absorb("attn_norm_com", attn_norm_com)
      FS.absorb("attn_num_com",  attn_num_com)
      FS.absorb("attn_z_com",    attn_z_com)
      FS.absorb("attn_rem_com",  attn_rem_com)
      FS.absorb("attn_diff_com", attn_diff_com)

  // ── 3. Absorb QK-norm IO commitments ───────────────────────────────────
  // q_norm: x = q_com (pre-norm),  y = q_norm_y_com (post-norm).
  FS.absorb(q_com); FS.absorb(q_norm_y_com)
  // k_norm: x = k_com (pre-norm),  y = k_norm_y_com (post-norm).
  FS.absorb(k_com); FS.absorb(k_norm_y_com)

  // ── 4. Absorb O-projection output ──────────────────────────────────────
  FS.absorb("out_attn_com", out_attn_com)

  // ── 5. Absorb attn_out_norm (sandwich norm) IO commitments ─────────────
  // x = out_attn_com (W_O · attn_y + b_O),  y = attn_out_norm_y_com.
  FS.absorb(out_attn_com); FS.absorb(attn_out_norm_y_com)

  // ── 6. Residual 1 — sandwich norm output added to the residual stream ──
  x_mid_com ← HyraxAdd(x_in_com, attn_out_norm_y_com)

  // ── 7. Absorb LN2 IO commitments ───────────────────────────────────────
  FS.absorb("x_com", x_mid_com);  FS.absorb("y_com", x_norm2_com)

  // ── 8. Absorb FFN output (matches downstream cross-block expectations) ─
  FS.absorb("y_com", out_ffn_com)

  return { x_norm1_com, q_com, k_com, v_com,
           q_norm_y_com, k_norm_y_com,
           attn_num_com, attn_norm_com,
           attn_z_com, attn_rem_com, attn_diff_com,
           out_attn_com, attn_out_norm_y_com,
           x_norm2_com, out_ffn_com, x_mid_com }
```

Residual 2 (`x_out_com = HyraxAdd(x_mid_com, out_ffn_com)`) is computed by the
model-level prover after Phase 1 returns. The cross-block sumchecks in §4.2
(steps 4–9) consume the per-block intermediate commitments and the witness
arrays directly; per-block LN sumchecks are folded into the model-level
batched LN proof of step 11b, and per-block range witnesses are folded into
the bucketed range batch of step 11.

### 4.4 Range Proof Prover (Batched, Bucketed by Bit-Width)

`ProveRangeBatched` is invoked once per bit-width bucket (currently $\{32, 64\}$);
the model-level prover builds the bucket lists from all LN $\sigma$/$y$
witnesses (10·L + 2) and, when `has_attn_norm`, the attention-normalization
remainders (2·L). The number of chunks $C = \lceil \mathsf{bits}/16 \rceil$
is 2 for the 32-bit bucket and 4 for the 64-bit bucket.

```
Algorithm ProveRangeBatched(witnesses[0..B-1], bits ∈ {32, 64}, FS):
  Input:
    witnesses[b] = { values: V^(b) ∈ F^{2^n_b} }    // B witnesses in this bucket
    CHUNK_BITS = 16,  C = bits / 16  ∈ {2, 4}
  Output:
    (rps[0..B-1], global_m, r_vs[0..B-1])

  // ── Phase 1: Commit all chunks ───────────────────────────────────────────
  For b = 0 to B-1:
    For c = 0 to C-1:
      V_c^(b)[i] ← (V^(b)[i] >> (16·c)) mod 2^16
      cc^(b)_c    ← Com(V_c^(b))
      FS.absorb("chunk_com", cc^(b)_c)

  // ── Phase 1b: Merge multiplicities and commit once per bucket ───────────
  m[v] ← 0  for v in 0..2^16-1
  For b = 0 to B-1, c = 0 to C-1, i = 0 to 2^{n_b}-1:
      m[V_c^(b)[i]] += 1
  m_com ← Com(m)
  FS.absorb("m_com", m_com)

  // ── Phase 2: Per-witness sumcheck + openings ─────────────────────────────
  For b = 0 to B-1:
    claim_v^(b) ← Σ_i V^(b)[i]
    FS.append("claim_v", claim_v^(b))
    (sc_π^(b), r_v^(b)) ← ProveSumcheck(V_mle^(b), ones_mle, claim_v^(b), FS)
    ce^(b)_c ← V_c_mle^(b)(r_v^(b))   for c = 0..C-1
    Assert V^(b)(r_v^(b)) == Σ_c 2^{16·c} · ce^(b)_c
    chunk_batch_π^(b) ← HyraxOpenBatch([V_c^(b)]_c, r_v^(b), FS)
    rps[b] ← RangeWitnessProof { chunk_coms, chunk_evals, chunk_batch_π,
                                  sumcheck: sc_π^(b), claim_v: claim_v^(b) }

  // ── Phase 3: Shared multiplicity opening ────────────────────────────────
  r_m   ← FS.challenge_vec("range_m_r", 16)
  m_open ← HyraxOpen(m, r_m, FS)
  return (rps, GlobalRangeM { m_com, m_mle(r_m), m_open }, r_vs)
```

The model-level proof carries one `GlobalRangeM` per non-empty bucket; the
verifier replays the bucket selection deterministically from the serialized
metadata.

### 4.5 LayerNorm Prover

The pseudocode below describes the LayerNorm protocol *for one LN witness*.
The end-to-end model prover does **not** invoke it once per LN; instead, it
collects all $5L+1$ LN witnesses (in fixed order: per block ln1, q_norm,
k_norm, attn_out_norm, ln2; then the final LN) and runs a single
`prove_layernorms_batched` call (§4.2 step 11b). That batched call shares
the row challenge $r_t$, the variance / Y-fusion sumchecks, and the deferred
Hyrax accumulator pushes across all LN witnesses, with per-LN claims
combined by Fiat–Shamir powers of an independent challenge. The single-LN
description below defines the constraint system; the batched call is its
$5L+1$-fold cross-instance random-linear combination, which is sound by the
same Schwartz–Zippel argument as cross-block batching (Theorem 2).

```
Algorithm ProveLayerNorm(wit, io_coms, vk, σ_range, y_range, FS):
  Input:
    wit     = { x, y, sum_x, sq_sum_x, sum_x_sq, σ, σ_sq_scaled }
    io_coms = { x_com, y_com }                // from pipeline
    vk      = { γ, β, d, scale_γ, scale_β }
    σ_range = (RangeWitnessProof, r_σ)        // pre-committed, from global batch
    y_range = (RangeWitnessProof, r_y)        // pre-committed, from global batch

  // 1. Commit intermediate witnesses (3 commitments per LN)
  sum_x_com    ← Com(sum_x)        // per-row sum
  sq_sum_x_com ← Com(sq_sum_x)     // per-row sum of squares
  σ_com        ← Com(σ)            // floor-sqrt witness
  FS.absorb("x_com",      io_coms.x_com)
  FS.absorb("y_com",      io_coms.y_com)
  FS.absorb("sum_x_com",  sum_x_com)
  FS.absorb("sq_sum_x_com", sq_sum_x_com)
  FS.absorb("sigma_com",  σ_com)

  // 2. Row audit challenge
  r_t ← FS.challenge_vec("layernorm_rt", t_bits)

  // 3. Mean sumcheck (deg 2): sum_x_mle(r_t) = Σ_j x_col(j)
  x_col ← x_mle.fix_row(r_t)
  sum_x_rt ← sum_x_mle(r_t)
  FS.append("sum_x_at_rt", sum_x_rt)
  (mean_sc, r_d_mean) ← ProveSumcheck(x_col, ones_mle, sum_x_rt, FS)

  // 4. Square-sum sumcheck (cubic): sq_sum_x_mle(r_t) = Σ_j x_col(j)^3
  //    (degree-3 sumcheck over three copies of x_col)
  sq_sum_x_rt ← sq_sum_x_mle(r_t)
  FS.append("sq_sum_x_at_rt", sq_sum_x_rt)
  (sq_sum_sc, r_d_var) ← ProveSumcheckCubic(x_col, x_col, x_col, sq_sum_x_rt, FS)

  // 5. Sigma-residual sumcheck (cubic multi-batched).  Binds
  //    sum_x_sq, σ_sq_scaled to derivable squares at r_sig_t.
  (σ_rp, r_σ) ← σ_range
  r_sig_t ← r_σ[0..t_bits]
  claim_sum_x_sq ← sum_x_sq_mle(r_sig_t)
  claim_σ_sq    ← σ_sq_scaled_mle(r_sig_t)
  FS.append("claim_sum_x_sq", claim_sum_x_sq)
  FS.append("claim_sigma_sq", claim_σ_sq)
  λ_sig ← FS.challenge("sigma_residual_batch_lambda")
  combined ← claim_sum_x_sq + λ_sig · claim_σ_sq
  (sig_res_sc, r_f_sig) ← ProveSumcheckCubicMultiBatched(
      [(eq(r_sig_t,·), sum_x, sum_x), (eq(r_sig_t,·), d·σ, d·σ)], λ_sig, FS)

  // 6. Y constraint fusion (cubic multi-batched): γ·X and σ·Y legs.
  (y_rp, r_y) ← y_range
  r_y_t ← r_y[0..t_bits];  r_y_d ← r_y[t_bits..t_bits+d_bits]
  γ_r   ← γ_mle(r_y_d);   β_r ← β_mle(r_y_d)
  α     ← FS.challenge("layernorm_alpha")
  (gx_sy_sc, r_f_y) ← ProveSumcheckCubicMultiBatched(
      [γ_leg, σ_leg], α, FS)

  // 7. Push Hyrax openings to deferred accumulators (acc_t / acc_td).

  return LayerNormProof {
    internal_coms: { sum_x_com, sq_sum_x_com, σ_com },
    mean_sc, sq_sum_sc, sig_res_sc, gx_sy_sc,
    sigma_range_proof, y_range_proof,
    openings: { many shared per-LN evaluations described in
                 LayerNormOpenings; the verifier opens sum_x / sq_sum_x
                 / σ at the four shared random points r_t, r_sig_t,
                 r_f_sig, r_y_t, r_f_y via batched Hyrax opens }
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

  // ── 1. Bind public input via re-commitment ──────────────────────────────
  Assert Com(public_x_in) == π.x_in_com
  FS.init("piformer")
  FS.absorb("x_in_com", π.x_in_com)

  // (Public output is bound later in step 12 by evaluating public_logits
  // as a multilinear extension at a transcript-derived point lm_y_point
  // and chaining that claim through the LM-head proof to final_ln_out_com.
  // No separate logits commitment is published.)

  // ── 2. Phase 1 verify (per block): absorb LN IO + projection commitments ─
  Initialize deferred accumulators (10 cross-cutting groups).
  x_cur_com ← π.x_in_com
  For ℓ = 0 to L-1:
    bp ← π.block_proofs[ℓ]
    // Absorb in the same fixed order as prover Phase 1 (§4.3): LN1 IO,
    // q_com / k_com / v_com, optional attn-norm aux commits, q_norm IO,
    // k_norm IO, out_attn_com, attn_out_norm IO, LN2 IO, out_ffn_com.
    AbsorbBlockPhase1Commits(bp, x_cur_com, FS)
    x_mid_com ← HyraxAdd(x_cur_com, bp.attn_out_norm_y_com)   // sandwich-norm residual
    x_cur_com ← HyraxAdd(x_mid_com, bp.out_ffn_com)

  // ── 3. Derive global r_td after ALL Phase 1 ────────────────────────────
  r_td ← FS.challenge("gkr_r_td", t_bits + d_bits)

  // ── 4–9. Six cross-block batch sumchecks (mirror prover §4.2) ──────────
  Verify batch_qkv      → recover r_k_qkv
  Verify batch_oproj    → recover r_k_o
  Verify batch_attn_out → recover batch_r_attn_out
  Verify batch_attn_ctx → recover batch_r_attn_ctx
  if has_attn_norm:
      Verify attn_norm_sumcheck → recover r_norm
      Verify attn_z_sumcheck    → recover r_norm_t-derived openings
  Verify global FFN Lasso committed to ffn_a_com and ffn_lasso_bind opening
  Verify batch_ffn_y    → recover r_k_fy
  Verify batch_ffn_m    → recover r_k_m

  // ── 10. Final LN IO absorption ─────────────────────────────────────────
  Absorb final_ln IO commitments (sumcheck deferred to step 11b)

  // ── 11. Global range batch verify (per non-empty bit-width bucket) ─────
  for bits ∈ {32, 64} that the prover used:
      VerifyRangeBatched(bucket_witness_proofs, π.range_m_for_bits, bits, FS,
                          accs.range_sig, accs.range_y, accs.range_m)

  // ── 11b. Cross-LN batched verify (5L+1 LayerNorms in one transcript) ────
  VerifyLayerNormsBatched(π.ln_batched_proof, all 5L+1 LN io_coms,
                           σ-range r_v's, y-range r_v's, FS,
                           ln_acc_t, ln_acc_td)

  // ── 12. LM head + public-output binding ────────────────────────────────
  // The verifier samples lm_y_point ∈ F^{t_bits + v_bits} from the transcript,
  // computes lm_y_value = ̃logits(lm_y_point) directly from public_logits,
  // and feeds (lm_y_point, lm_y_value) into the LM-head projection sumcheck.
  // VerifyProjectionGKR reduces this to an opening claim
  //   lm_x_claim = (point: r_x, value: v_x)  on  final_ln_out_com.
  // π.lm_head_input_open is verified via the ln_acc_td deferred accumulator,
  // which closes the chain back to the LayerNorm proof.
  lm_y_point ← FS.challenge_vec("lm_gkr_y", t_bits + v_bits)
  lm_y_value ← MLE_eval(public_logits, lm_y_point)
  lm_x_claim ← VerifyProjectionGKR(π.lm_head_proof, VK.lm_head_vk,
                                    {point: lm_y_point, value: lm_y_value},
                                    FS, lmh_acc_w, lmh_acc_b)
  ln_acc_td.add_verify(π.final_ln_out_com, lm_x_claim.value, lm_x_claim.point,
                        π.lm_head_input_open)

  // ── 13. Drain accumulator μ-challenges (13 groups, mirrors prover) ─────
  // Same as the prover side of step 13 above.  After the cross-block batch
  // opens of step 14–15 the verifier additionally draws 5 cross-block-batch
  // μ-challenges (cb_td, cb_qkvo_w, cb_qkvo_b, cb_wff, cb_mff) and, when
  // the QK quantization proof is present, 2 more (quant_qk, quant_qk_m).
  for _ in 0..13: FS.challenge("hyrax_group_mu")
  FS.challenge_readonly("hyrax_fuse_td")
  FS.challenge_readonly("hyrax_fuse_range_m")

  // ── 14–15. Global intermediate batch open + cross-block batch opens ───
  Assert HyraxVerifyBatch({Q_ℓ, K_ℓ, V_ℓ, Out_attn_ℓ, Out_ffn_ℓ}_ℓ at r_td,
                           π.inter_batch_open, …) == ACCEPT
  For each (matrices, point, batch_open) listed in §4.2 step 15:
      Assert HyraxVerifyBatch(matrices, point, batch_open, …) == ACCEPT
  if has_attn_norm:
      Assert HyraxVerifyBatch([attn_num | attn_norm | rem | diff]_ℓ at r_norm,
                               π.attn_norm_r_batch_open, …) == ACCEPT
      Assert HyraxVerifyBatch([z_ℓ] at r_norm_t,
                               π.attn_z_open, …) == ACCEPT
      Assert HyraxVerifyBatch([attn_num | attn_norm]_ℓ at (r_t, r_k_o),
                               π.attn_norm_attn_point_open, …) == ACCEPT

  // ── 16. Finalise deferred Hyrax accumulators (one MSM pair per group) ──
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

  // ── 15. Global Lasso verification ──────────────────────────────────────
  Assert VerifyLassoMultiCommittedOutputs(
      [inst_ffn.activation_lasso for ℓ in 0..L-1],
      π.ffn_lasso_proof, [bp.ffn_a_com]_ℓ, FS) == ACCEPT
  Assert ffn_lasso query indices are bound to [bp.ffn_m_com]_ℓ

  Assert qk_lasso query indices are bound to [bp.q_com, bp.k_com]_ℓ
  Assert VerifyLassoMultiCommittedOutputs(
      [inst_attn[ℓ].q_lasso, inst_attn[ℓ].k_lasso for ℓ in 0..L-1],
      π.all_lasso_proof, [bp.attn_phi_q_com, bp.attn_phi_k_com]_ℓ, FS) == ACCEPT

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

**Theorem 4 (Global FFN Lasso Soundness).** *Batching all FFN activation lookups into one committed-output `LassoMultiProof` does not reduce soundness, provided the output commitments and lookup-index bindings are absorbed before their Fiat-Shamir challenges.*

*Proof.* For each block, the prover commits `ffn_a_com` and `ffn_m_com` before the global FFN lookup proof and FFN-Y/M batch challenges. `prove_lasso_multi` absorbs all `ffn_a_com` commitments before sampling the multi-Lasso batching challenges `instance_batch_alpha` and `lookup_batch_rho`. The committed-output sumcheck then binds the combined lookup grand sum to the committed $A_\ell$ polynomials, and `ffn_a_batch_open` later binds the $A_\ell(r)$ evaluations used by `batch_ffn_y`. Separately, the proof's lookup query indices are absorbed before `ffn_lasso_bind_point`, and `ffn_lasso_bind_open` proves that those indices are the committed $M_\ell$ values evaluated at that point. A cheating prover must therefore either forge the Lasso sumcheck, open `ffn_a_com` / `ffn_m_com` inconsistently, or forge the FFN sumchecks; these are caught by Lemma 1, Lemma 2, and Theorem 2. $\square$

**Theorem 5 (Batched QKV Soundness).** *The single shared-$r_k$ sumcheck for Q, K, V projections is as sound as three independent sumchecks.*

*Proof.* The verifier draws Fiat-Shamir scalars $\lambda, \mu \stackrel{\$}{\leftarrow} \mathbb{F}_r$ before the sumcheck challenge $r_k$. The combined claim is $\lambda \cdot Y_Q(r_t, r_d) + \mu \cdot Y_K(r_t, r_d) + Y_V(r_t, r_d)$. A cheating prover who fakes any single projection $\hat{Y}_Q \neq Y_Q$ changes the combined polynomial by $\lambda \cdot (\hat{Y}_Q - Y_Q)$, which is a non-zero polynomial in $\lambda$ of degree $\leq 1$. The verifier's random $\lambda$ fails to catch this with probability $\leq 1 / |\mathbb{F}_r| \approx 2^{-254}$. Combined with the sumcheck error: total soundness error $\leq 2 / |\mathbb{F}_r| + n\delta / |\mathbb{F}_r|$. $\square$

**Theorem 6 (Global Range Batch Soundness).** *Sharing one $m_{\mathsf{com}}$ per bit-width bucket across all $B$ range witnesses (over all blocks and the final LN, plus optional attention-normalization residuals) is sound: a cheating prover cannot supply chunk values outside $[0, 2^{16})$ for any single witness.*

*Proof.* Each bucket is processed independently and the argument applies per bucket. The key commitment-ordering invariant is that $m_{\mathsf{com}}$ is absorbed into the transcript *before* any per-witness sumcheck challenge $r_v^{(b)}$ in the bucket is derived. Since $m$ aggregates all chunk occurrences across all witnesses (and all $C$ chunks per witness) in the bucket, and the prover must commit to $m$ before seeing any $r_v^{(b)}$, the prover cannot adaptively choose the chunk values after learning the sumcheck challenges. Formally: suppose a cheating prover supplies $\hat{V}_c^{(b_0)}[i] = v^* \geq 2^{16}$ for some witness $b_0$ and chunk index $c$. For the LogUp check to pass, $m[v^*]$ must be positive, but $T_{\mathsf{id}}[v^*]$ does not exist (the identity table only has entries $0 \ldots 2^{16}-1$), so the LogUp identity $\sum_{b,c,i} \delta_{v^*, \mathsf{chunk}^{(b,c)}[i]} = m[v^*] \cdot T_{\mathsf{id}}[v^*]$ fails. The committed $m_{\mathsf{com}}$ binds the prover to the claimed multiplicities before all sumchecks (Lemma 2). $\square$

**Theorem 6b (Model-level batched LayerNorm soundness).** *The single
`prove_layernorms_batched` call covering all $5L+1$ LayerNorms is as sound
as $5L+1$ independent LayerNorm sub-proofs run with independent challenges.*

*Proof.* The batched LN proof folds the $5L+1$ per-LN claims via Fiat–Shamir
powers of an independent batching challenge $\eta$, drawn after every per-LN
IO commitment ($x$, $y$, $\mathsf{sum\_x}$, $\mathsf{sq\_sum}$, $\sigma$) has
been absorbed into the transcript and after the bucketed range batch has
committed to all $\sigma$/$y$ chunks. The same Schwartz–Zippel argument as
Theorem 2 applies: a single false LN claim makes the combined polynomial in
$\eta$ non-zero of degree $\le 5L$, caught with probability $\ge 1 - 5L/|\mathbb{F}_r|$.
The shared inner sumchecks (mean, variance, Y-fusion) inherit
Lemma 1 soundness on the combined claim. $\square$

**Theorem 6d (Quantization proof soundness).** *For each lookup family, the
`prove_quantization_batch` sub-protocol binds every public lookup query index
$\mathsf{idx}_j$ to the committed raw input value $r_j$ via the integer
relation
$r_j \cdot s_{\mathrm{den}} + \lfloor s_{\mathrm{num}}/2 \rfloor =
s_{\mathrm{num}}(\mathsf{idx}_j - \mathsf{zp}) + \mathsf{rem}_j$
with $0 \le \mathsf{rem}_j < s_{\mathrm{num}}$, except with negligible
probability.*

*Proof sketch.* The prover commits the remainder tensor $\mathsf{rem}$ via
Hyrax before any quantization challenge is drawn; the global range batch
(Theorem 6) then forces $0 \le \mathsf{rem}_j < s_{\mathrm{num}}$ (with
$s_{\mathrm{num}}$ a power of two so the bit-width bucket is exact). At a
Fiat–Shamir random point $\mathbf r \in \mathbb F^{n}$ ($n = t_{\mathsf{bits}} + d_{\mathsf{bits}}$)
the verifier checks the algebraic identity over the four MLEs
$\widetilde r$, $\widetilde 1$, $\widetilde{\mathsf{idx}}$, $\widetilde{\mathsf{rem}}$.
By Schwartz–Zippel a non-zero residual polynomial in any of the witnesses
fails the identity with probability $\ge 1 - n/|\mathbb F_r|$. Hyrax binding
(Lemma 2) closes the loop: a cheating prover cannot open $\widetilde r$ or
$\widetilde{\mathsf{rem}}$ inconsistently with their commitments.
Together with Theorem 6, this ensures the unique-quotient property:
$\mathsf{idx}_j$ is forced to be the unique integer in $[0, 2^B)$ satisfying
the relation, i.e. $\mathsf{idx}_j = \lfloor (r_j s_{\mathrm{den}} + s_{\mathrm{num}}/2) / s_{\mathrm{num}} \rfloor + \mathsf{zp}$.
$\square$

**Theorem 6c (Attention-normalization sumcheck soundness).** *When
`has_attn_norm` is set, the cubic multi-batched sumcheck for*
$\mathsf{ATTN\_NORM\_SCALE} \cdot \mathsf{attn\_num} - \mathsf{rem} - Z\!\cdot\!\mathsf{attn\_y} = 0$
*and* $\lambda(Z - 1 - \mathsf{rem} - \mathsf{diff}) = 0$ *combined with the
range proofs on $\mathsf{rem}$, $\mathsf{diff}$ and the auxiliary $Z$
sumcheck soundly enforces the floor-division identity*
$\mathsf{attn\_y}[t][j] = \lfloor \mathsf{ATTN\_NORM\_SCALE}\cdot\mathsf{attn\_num}[t][j] / Z[t]\rfloor$
*at every $(t,j)$.*

*Proof sketch.* Range proofs (Theorem 6) constrain $\mathsf{rem} \ge 0$ and
$\mathsf{diff} \ge 0$. The second identity then gives
$\mathsf{rem} \le Z - 1$, i.e. $0 \le \mathsf{rem} < Z$. The first identity
expresses $Z\cdot\mathsf{attn\_y} = \mathsf{ATTN\_NORM\_SCALE}\cdot\mathsf{attn\_num} - \mathsf{rem}$
with the unique-quotient property of integer division given $0 \le \mathsf{rem} < Z$.
Both identities are checked at one Fiat–Shamir random point $r_{\mathsf{norm}}$
of $t_{\mathsf{bits}}+d_{\mathsf{bits}}$ variables, with cubic-sumcheck error
$3(t_{\mathsf{bits}}+d_{\mathsf{bits}})/|\mathbb{F}_r|$ (Lemma 1) and a
$\lambda$-batching error $1/|\mathbb{F}_r|$ (Lemma 3). The auxiliary
$Z$ sumcheck binds $Z[t]$ to $\phi(Q_n)_t \cdot \sum_s \phi(K_n)_s$ via the
already-committed $\phi(Q_n), \phi(K_n)$ MLEs (Lemmas 1 and 2). Hyrax
binding (Lemma 2) on `attn_num_com`, `attn_norm_com`, `attn_z_com`,
`attn_rem_com`, `attn_diff_com` closes the loop. $\square$

---

### 5.4 End-to-End Soundness Theorem

**Theorem 7 (π-Former Soundness).** *For any PPT adversary $\mathcal{A}$, the probability that $\mathsf{Verify}(\mathsf{VK}, x_{\mathsf{in\_com}}, y_{\mathsf{com}}, \pi) = \mathsf{ACCEPT}$ but $\mathcal{M}_\theta(\mathbf{x}) \neq \mathbf{y}$ is at most*

$$\varepsilon_{\mathsf{sound}} \leq L \cdot n_{\max} \cdot \delta_{\max} / |\mathbb{F}_r| + \varepsilon_{\mathsf{DL}}$$

*where $n_{\max}$ is the maximum number of sumcheck variables across all sub-protocols, $\delta_{\max} = 3$ is the maximum round polynomial degree, $L$ is the number of transformer blocks, and $\varepsilon_{\mathsf{DL}}$ is the DL advantage of $\mathcal{A}$ in the BN254 G1 group.*

*Proof sketch.* We proceed by hybrid argument over the $L$ blocks.

1. **Hyrax binding** (Lemma 2): If $\mathcal{A}$ can forge any committed evaluation, it breaks DL in G1.

2. **Block-level chaining:** In block $\ell$, the output commitment $C_{x_{\mathsf{out}}}^\ell$ is a two-step *homomorphic sum*: first $C_{x_{\mathsf{mid}}}^\ell = C_{x_{\mathsf{in}}}^\ell + C_{\mathsf{attn\_out\_norm}.y}^\ell$ (sandwich-norm residual), then $C_{x_{\mathsf{out}}}^\ell = C_{x_{\mathsf{mid}}}^\ell + C_{\mathsf{out\_ffn}}^\ell$. Binding of each component (Lemma 2) implies binding of the sums. The first block's input is `proof.x_in_com`, which the verifier recomputes from `public_x_in` and checks for equality; binding proceeds inductively.

3. **Within each block:** The transcript state after block $\ell-1$ is a deterministic function of all commitments and proofs from blocks $0 \ldots \ell-1$. All challenges in block $\ell$ are Fiat-Shamir outputs from this state. Given Hyrax binding and sumcheck soundness:
   - The bucketed global range batch over all witnesses (LN $\sigma$/$y$ + optional attn-norm rem/diff) is sound (Theorem 6).
   - Each of the five LayerNorms (ln1, q_norm, k_norm, attn_out_norm, ln2) is sound under the model-level batched LN proof (Theorem 6b) which composes the mean / variance / σ-floor / Y-fusion sub-claims with range-proof soundness (Lemma 5).
   - The sandwich-norm residual $C_{x_{\mathsf{mid}}} = C_{x_{\mathsf{in}}} + C_{\mathsf{attn\_out\_norm.y}}$ requires no proof (Theorem 3).
   - When `has_attn_norm`, the floor-division identity is sound (Theorem 6c).
   - Batched QKV soundness (Theorem 5).
   - GKR fusion soundness (Theorem 2) for the six cross-block sumchecks.
   - Global FFN Lasso soundness (Theorem 4).
   - Homomorphic residuals require no proof (Theorem 3).
   - Lasso soundness (Lemma 4) for all activation tables, with lookup-index-to-input binding via the quantization proofs (Theorem 6d), which tie the public indices in `ffn_lasso_query_indices` and `all_lasso_proof.all_query_indices` to the committed post-norm `q_norm_y_com`, `k_norm_y_com` and the committed `ffn_m_com` through the integer quantization relation. The φ inputs are exactly the LayerNorm outputs.

4. **Final layer + LM head:** Same argument as a single block.

5. **Global Lasso batches:** The FFN activation proof covers all $A=\phi(M)$ lookups, and the attention proof covers all $\phi(Q)$ / $\phi(K)$ lookups. Soundness follows from Lemma 4 plus the committed-output and committed-index bindings described in §3.5.

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

Let $L$ = layers, $T$ = sequence length, $d$ = embedding dimension, $d_{ff}$ = FFN width, $m$ = bits per chunk, $c$ = chunk count, $B_{rng}$ = range witnesses per block (10 for the five LayerNorms, plus 2 for the optional attn-norm rem/diff in normalized mode).

| Component | Prover time | Verifier time | Proof size |
|-----------|-------------|---------------|------------|
| Hyrax commit (size $N$) | O($N$) G1 MSM | — | O($\sqrt{N}$) G1 points |
| Hyrax open (single) | O($\sqrt{N}$) G1 | O($\sqrt{N}$) G1 + O($\log N$) $\mathbb{F}$ | O($\sqrt{N}$) $\mathbb{F}$ |
| Hyrax $K$-point batch | O($K \sqrt{N}$) G1 | O($2\sqrt{N}$) G1 | O($K\sqrt{N}$) $\mathbb{F}$ |
| Sumcheck ($n$ vars, deg $\delta$) | O($n \delta \cdot 2^n$) $\mathbb{F}$ | O($n\delta$) $\mathbb{F}$ | O($n\delta$) $\mathbb{F}$ |
| Lasso (one sub-table, $N$ queries) | O($N \cdot 2^m + m \cdot 2^m$) | O($N \cdot m$) | O($m$) $\mathbb{F}$ |
| Range proof (1 witness, batched $m$) | O($\sqrt{2^{16}}$) G1 | O($\sqrt{2^{16}}$) G1 | O($\sqrt{2^{16}}$) $\mathbb{F}$ |
| Range proof global $m_\mathsf{com}$ | O($\sqrt{2^{16}}$) G1 **once** | O($\sqrt{2^{16}}$) G1 **once** | O($\sqrt{2^{16}}$) $\mathbb{F}$ |
| LayerNorm (per LN, model-level batched across $5L+1$) | 3 sumchecks + 2 range witnesses, folded into one batched proof | O($d$) + O($\log T$) per LN | O($\sqrt{Td}$) per LN, shared transcript |
| Sandwich + QK norms (per block) | 3 extra LN witnesses (q_norm, k_norm, attn_out_norm) folded into the model-level LN batch | O($d$) + O($\log T$) per LN | absorbed by LN batch |
| Attention normalization (per model, normalized mode) | 1 cubic multi-batched sumcheck + 1 $Z$ sumcheck + 2L 64-bit range witnesses | O($\log T + \log d$) + 2 batch opens | O($\sqrt{T d}$) for `[num \| norm \| rem \| diff]` batch open + $Z$ open |
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
| Cross-block batch opens (one per tensor family, not per block) | $(L-1)$ Hyrax opens per committed/opened tensor family | §4.2 step 13 |
| Global FFN activation Lasso (one `LassoMultiProof` vs $L$ local proofs) | $(L-1)$ lookup sumcheck transcripts and table-opening batches | model-level `ffn_lasso_proof` |
| Committed-output Lasso (no proof-carried lookup outputs) | $O(N)$ field elements per lookup instance | `ffn_lasso_proof`, `all_lasso_proof` |
| Homomorphic residuals (no proof) | 2 Hyrax opens per block | per block |
| Global range batch (1 $m_{\mathsf{com}}$ per bucket vs per witness) | $(B - 1) \times \sqrt{2^{16}}$ G1 per bucket, where $B = 10L+2$ for the LN bucket plus $2L$ for the optional attn-norm bucket | model-level |
| Model-level batched LN proof (1 transcript vs $5L+1$) | $(5L+1 - 1)$ LN sumcheck transcripts | `prove_layernorms_batched` |
| Sandwich + QK norms reuse the LN primitive | no extra primitive — $5L+1$ LN witnesses fold into the LN batch + range bucket | per block |
| Attention normalization fused into one cubic multi-batched sumcheck | $7L$ summand slots in one transcript instead of $L$ separate normalization proofs | `attn_norm_sumcheck` |
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
- **Structured optimizations:** GKR-style fusion, homomorphic residuals, global range batching, global committed-output Lasso, and batched QKV reduce concrete proof size and prover time without sacrificing soundness.
- **Extensible:** lookup decomposition depth $c$ and chunk size $m$ are configurable trade-offs between model expressivity and proof cost.

---

## 9. Planned Extensions

### 9.1 Committed Lasso Indices

The current model proof still serializes lookup query indices and then binds them to committed tensors with one random-point Hyrax batch opening. A more succinct variant can replace the raw indices with the existing `LassoIndexProof` machinery: commit chunk polynomials for the lookup input tensor and prove the selector values used by Lasso with high-degree selector sumchecks. This reduces proof-carried index vectors at the cost of extra selector proofs, so it is currently kept as an optional protocol path rather than the default.

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

1. **Phase 1 absorbs all five LN IO commitments per block, in fixed order.** Each block absorbs `(x_in, x_norm1)` for LN1; `(q_com, k_com, v_com)` plus the optional `(attn_norm_com, attn_num_com, attn_z_com, attn_rem_com, attn_diff_com)`; `(q_com, q_norm_y_com)` for q_norm; `(k_com, k_norm_y_com)` for k_norm; `out_attn_com`; `(out_attn_com, attn_out_norm_y_com)` for attn_out_norm; `(x_mid_com, x_norm2_com)` for LN2; and finally `out_ffn_com`. No per-block LN sumcheck is emitted at this stage — it is deferred to the model-end batched LN call.

2. **Phase 1 completes for ALL blocks** before the global `r_td = (r_t \| r_{\mathsf{out}})` challenge is sampled. This guarantees that every per-block intermediate commitment ($Q_\ell, K_\ell, V_\ell$, $Q^{\mathrm{n}}_\ell, K^{\mathrm{n}}_\ell$, $\mathsf{Out\_attn}_\ell$, $\mathsf{attn\_out\_norm.y}_\ell$, $\mathsf{Out\_ffn}_\ell$, $X_{\mathsf{norm}1,\ell}, X_{\mathsf{norm}2,\ell}$, plus the optional attn-norm aux commits) is bound to the transcript before the cross-block batch sumcheck challenges are drawn.

3. **Per-block QKV/O-proj/FFN absorbs precede the corresponding batch $\eta$**. Inside the cross-block QKV loop, each block's $(W_Q, W_K, W_V, \alpha, b_Q, b_K, b_V)$ commitments and the per-block $(\lambda_\ell, \mu_\ell)$ challenges are absorbed before the model-level $\eta_{\mathsf{qkv}}$ is sampled. The same pattern holds for `batch_oproj`, `batch_ffn_y`, `batch_ffn_m`, `batch_attn_out`, and `batch_attn_ctx`.

4. **Attention-normalization sumcheck challenges are sampled after step-4 (`batch_attn_out`) terminates**, so $\mathsf{attn\_num}$ is already bound to its $(r_t, r_{k_o})$ evaluation point before $r_{\mathsf{norm}}$ is drawn. The cubic multi-batched sumcheck and the auxiliary $Z$ sumcheck use independent Fiat–Shamir challenges drawn after the `attn_num_com`, `attn_norm_com`, `attn_z_com`, `attn_rem_com`, `attn_diff_com` were absorbed in Phase 1.

5. **Bucketed range batch precedes the model-level LN batched sumcheck.** All chunk commitments and the per-bucket $m_{\mathsf{com}}$ values are absorbed *before* any LN $\sigma$/$y$ sumcheck challenge or any LN-batch cross-instance challenge is sampled. Violating this ordering breaks Theorem 6 / 6b.

6. **Committed-output Lasso absorbs output commitments before lookup batching challenges**. `prove_lasso_multi` / `verify_lasso_multi_committed_outputs` absorb `lasso_output_com` for every committed output before deriving `instance_batch_alpha` and `lookup_batch_rho`. This applies to `ffn_lasso_proof` (`ffn_a_com`) and `all_lasso_proof` (`attn_phi_q_com`, `attn_phi_k_com`).

7. **Lookup indices are transcript-bound before index-opening challenges**. The raw query-index vectors in both global Lasso proofs are absorbed before deriving `ffn_lasso_bind_r` or `qk_lasso_bind_r`. The corresponding Hyrax batch openings bind those index vectors to `ffn_m_com` (FFN) or to the post-norm `q_norm_y_com` / `k_norm_y_com` (attention) — *not* to the pre-norm `q_com` / `k_com`, since the Lasso queries are evaluated on the post-QK-norm tensors.

8. **Thirteen deferred μ-challenges** are advanced after the model-level batched LN proof and the LM-head proof, before the global intermediate batch open. The thirteen groups are `inter`, `ln_t`, `ln_td`, `proj_w`, `proj_b`, `lmh_w`, `lmh_b`, `rng_sig`, `rng_y`, `rng_m`, `quant_ffn`, `quant_m`, `attn_norm`. Two read-only ρ challenges (`hyrax_fuse_td`, `hyrax_fuse_range_m`) are then sampled to fuse subsets of the accumulators that share Hyrax parameters into a single MSM pair via `HyraxBatchAccumulator::finalize_many_with_mus`. After the cross-block batch opens of step 15 the prover and verifier draw an additional 5 μ-challenges (`cb_td`, `cb_qkvo_w`, `cb_qkvo_b`, `cb_wff`, `cb_mff`) and, when the QK quantization proof fires, 2 more (`quant_qk`, `quant_qk_m`).

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
Version:    u8             (1 byte; current key version = 5)
num_blocks: varint u64
seq_len:    varint u64
d_model:    varint u64
vocab_size: varint u64
final_ln_vk:  [LayerNormVK]
lm_head_pk:   [ProjectionPK with weights]
block_pks:    [num_blocks × TransformerBlockPK with weights]
```

Each field element occupies 32 bytes (uncompressed). Each G1Affine occupies 33 bytes (compressed). Vector lengths and integer dimensions are encoded as little-endian base-128 varints in current files.

### Binary Verifying Key (`.vk`)

Same layout as `.pk` but all `ProjectionPK` / `FFN_PK` weight fields are replaced by empty vectors (`has_weights = false`).

### Binary Proof Bundle (`.bin`)

```
Magic:       b"PFMR_PR\0"  (8 bytes)
Version:     u8             (current proof version = 22)
proof:       [TransformerModelProof]
lasso_sigma: varint u64
```

`TransformerModelProof` carries:

- One `TransformerBlockProof` per block with the per-block intermediate Hyrax commitments (`x_norm1_com`, `q_com`, `k_com`, `v_com`, `q_norm_y_com`, `k_norm_y_com`, optional `attn_num_com`/`attn_norm_com`/`attn_z_com`/`attn_rem_com`/`attn_diff_com`, `out_attn_com`, `attn_out_norm_y_com`, `x_norm2_com`, `out_ffn_com`), the `ffn_m_com`, the per-block `phi_q`/`phi_k` commitments, the per-block scalar evaluations consumed by the cross-block batch sumchecks, and a placeholder (empty) `ffn_lasso_proof` field retained for codec compatibility.
- A model-level `ln_batched_proof` (`prove_layernorms_batched` output) covering all $5L+1$ LayerNorms (per block: `ln1`, `q_norm`, `k_norm`, `attn_out_norm`, `ln2`; plus the final LN). LayerNorms are grouped by shape (`seq_len`, `d_head`) first; each shape group emits one set of (mean, sq_sum, sigma_residual, gamma_sigma) multi-batched sumchecks. The σ/y range proofs are carried inside the same struct in input order.
- A bucketed range batch (`ln_range_ms`: one `RangeBatchM` per non-empty bit-width bucket) covering all LN σ/y witnesses (routed via `choose_range_bits` to the narrowest of {32, 64} bits) and, when normalized attention is used, the attn-norm `rem`/`diff` residuals.
- When normalized attention is used: the cubic multi-batched `attn_norm_sumcheck`, the auxiliary `attn_z_sumcheck` + `attn_z_ksum_sumcheck` (non-causal) or `attn_z_causal_sumcheck` (causal), plus the corresponding batch opens `attn_norm_r_batch_open`, `attn_z_open`, `attn_norm_attn_point_open`, `attn_z_phi_q_open`, `attn_z_phi_k_open`. The `attn_norm_rem_range_proofs`, `attn_norm_diff_range_proofs`, and their per-witness bit-width vectors live as top-level fields so the verifier can route them into the right range buckets.
- Six (non-causal) or six causal-variant model-level cross-block batched sumcheck proofs: `batch_qkv`, `batch_oproj`, (`batch_attn_out` | `batch_attn_out_causal`), (`batch_attn_ctx` | `batch_attn_ctx_causal`), `batch_ffn_y`, `batch_ffn_m`. In causal mode additional `causal_phi_k_prefix_evals`, `causal_v_prefix_evals` scalar vectors are included.
- One `inter_batch_open` covering the 5L intermediate matrices (`q`, `k`, `v`, `out_attn`, `out_ffn` per block) opened jointly at `r_td`.
- Cross-block batch opens for the per-type weight, bias, activation, and intermediate matrices at their respective shared evaluation points: `x_norm1_batch_open`, `qkv_w_batch_open` (3L wq/wk/wv merged), `w_o_batch_open`, `qkvo_bias_batch_open` (4L bq/bk/bv/bo merged), `w2_batch_open`, `w1_batch_open`, `x_norm2_batch_open`, `ffn_m_com_batch_open`, `ffn_lasso_bind_open`, `phi_q_batch_open`, `phi_k_batch_open`, `v_attn_batch_open`, `qk_lasso_bind_open`.
- `ffn_lasso_query_indices`: the per-block FFN lookup index vectors in plaintext, used by the verifier to evaluate the index MLE for the quantization-proof and self-consistency check.
- `ffn_a_terminal_proof`: the model-level FFN Lasso terminal-eval proof binding the `batch_ffn_y` terminal claim to the Lasso table evaluations.
- `all_lasso_proof`: a committed-output `LassoMultiProof` covering all $\phi(Q_n)$ / $\phi(K_n)$ Lasso instances across every block.
- `ffn_quant_proof` and `qk_quant_proof`: the two quantization sub-proofs of §3.5b, each carrying remainder commitments, a range proof for the remainders, and raw/remainder openings at a shared transcript-derived random point.
- `final_ln_out_com` (Hyrax commitment to the final-LN output) and `lm_head_input_open` (Hyrax opening that closes the LM-head proof's input claim back to `final_ln_out_com`). The public output is *not* committed separately; it is bound by evaluating it as an MLE at the transcript-derived `lm_gkr_y` point.
- `lm_head_proof`: the LM-head ternary projection sumcheck plus its weight/bias openings.
