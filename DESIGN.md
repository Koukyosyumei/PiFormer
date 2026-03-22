# π-Former: Technical Design

> Efficient SNARKs for Linear Transformers via Structured Lookup Attention

---

## 1. Problem Statement

Let $\mathcal{M}$ be a transformer model with $L$ layers, $H$ attention heads per layer, embedding dimension $d$, and FFN width $d_{ff}$. Given a public input sequence $\mathbf{x} = (x_1, \ldots, x_T)$ and the model's (frozen) weights $\theta$, we want a succinct, non-interactive argument of knowledge:

$$\pi \leftarrow \mathsf{Prove}(\theta, \mathbf{x}, \mathbf{y}) \quad \text{such that} \quad \mathsf{Verify}(\theta, \mathbf{x}, \mathbf{y}, \pi) = 1 \iff \mathcal{M}_\theta(\mathbf{x}) = \mathbf{y}$$

The na&iuml;ve approach of encoding the entire transformer in an R1CS circuit is prohibitively expensive because:

1. **Softmax** requires `exp` and row-wise normalization — transcendental functions that require many multiplications in a lookup-table decomposition.
2. **Layer normalization** involves a square root and division by a running variance.
3. **Activation functions** (GeLU, SiLU) have no compact polynomial representation.
4. **Large tables**: encoding a flat lookup for a 16-bit activation requires $2^{16}$ commitments.

π-Former addresses each bottleneck by co-designing the model and the proof system.

---

## 2. ZK-Friendly Model Architecture

### 2.1 Linear Attention

We replace the standard scaled dot-product attention

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

with the kernel (linear) attention formulation

$$\text{Attn}(Q, K, V) = \frac{\phi(Q)\,[\phi(K)^\top V]}{\phi(Q) \cdot \Sigma} \tag{1}$$

where $\phi : \mathbb{R}^{d_h} \to \mathbb{R}^{d_h}$ is an element-wise feature map and $\Sigma = \sum_s \phi(K_s)$.

**Why this is SNARK-friendly:**

- The matrix $C = \phi(K)^\top V \in \mathbb{R}^{d_h \times d_h}$ depends only on the sequence, not on a query-specific row-softmax. It can be proved once and reused.
- The normalizer $Z_t = \phi(Q_t) \cdot \Sigma$ is a single inner product — one sumcheck proves it for any number of heads.
- There are no exponentials, no row-wise divisions with unknown denominators, and no $\text{max}$ operations.

**Associativity and complexity:** Because $[\phi(K)^\top V]$ does not depend on $Q$, we compute it first in $O(T d_h^2)$ multiplications. The output $\phi(Q)\,C$ is then $O(T d_h^2)$ as well, avoiding the $O(T^2 d_h)$ cost of full attention.

### 2.2 Structured Lookup Activation

The kernel function $\phi$ is implemented as a **learnable, additively decomposed lookup table**:

$$\phi(x) = \sum_{i=0}^{c-1} T_i\!\left[\left\lfloor \frac{x_{\text{int}}}{2^{im}} \right\rfloor \bmod 2^m \right] \tag{2}$$

where:
- $x_{\text{int}} = \text{clamp}(\lfloor x / s \rfloor, 0, 2^B - 1)$ is the quantized input ($B$ bits, scale $s$).
- $B = c \cdot m$: total bits split into $c$ chunks of $m$ bits each.
- $T_0, \ldots, T_{c-1} \in \mathbb{R}^{2^m}$ are learnable sub-tables.

**Decomposition rationale:** A flat lookup for a $B$-bit input requires committing to a table of size $2^B$. The additive decomposition in eq. (2) reduces this to $c$ tables of size $2^m = 2^{B/c}$, giving total commitment cost $O(c \cdot 2^{B/c})$ instead of $O(2^B)$. For $B=16, c=2$: 512 commitments instead of 65536.

This structure exactly matches the **Lasso** lookup argument (§3.2), which is designed to exploit additive decompositions of this form.

**Initialization:** Sub-tables are initialized to approximate $\text{GeLU}(x)/c$ so that their sum closely approximates GeLU at the start of training, enabling stable convergence.

### 2.3 Power-of-Two Weight Quantization

Projection matrices $W_Q, W_K, W_V, W_O$ and FFN weight matrices are constrained to entries in

$$\mathcal{W} = \{0\} \cup \{-1, 0, 1\} \tag{3}$$

A matrix-vector product $y = Wx$ with $W \in \mathcal{W}^{m \times n}$ in the circuit requires only additions and multiplications by field constants (left-shifts). There are no general field multiplications, so this is **free** in the sumcheck constraint system.

**Training:** We use a straight-through estimator (STE). The forward pass applies a nearest-neighbor quantization to $\mathcal{W}$; the backward pass uses the identity function in place of the quantizer's (zero) gradient:

$$\hat{W} = \text{quantize}(W) + (W - W).\text{detach}()$$

---

## 3. Proof System

### 3.1 Field and Polynomial Commitment

We work over the BN254 scalar field $\mathbb{F}_r$ (prime $r \approx 2^{254}$). All model values (weights, activations, hidden states) are represented as fixed-point integers encoded as field elements.

The polynomial commitment scheme (PCS) is currently a **trivial** scheme (the prover reveals polynomial evaluations directly). Production deployment replaces this with the **Dory** or an inner-product argument (IPA), giving:
- Prover time: $O(\sqrt{N})$ group operations for a table of size $N$.
- Proof size: $O(\log N)$ field elements.
- Verifier time: $O(\log N)$ operations.

### 3.2 Dense Multilinear Polynomials

A multilinear polynomial $\tilde{f} : \mathbb{F}_r^n \to \mathbb{F}_r$ over $n$ variables is uniquely determined by its evaluations on the Boolean hypercube $\{0,1\}^n$. We store these as a flat vector of $2^n$ field elements in little-endian order:

$$\text{evals}[i] = \tilde{f}(\text{bit}_0(i), \text{bit}_1(i), \ldots, \text{bit}_{n-1}(i)), \quad \text{bit}_k(i) = (i \gg k) \mathbin{\&} 1$$

**Key operations:**

- **Evaluate at $r \in \mathbb{F}_r^n$:** Repeated halving. In round $j$, fix variable $n-1-j$ to $r_j$. Cost: $O(2^n)$.
- **Fix first variable to $r$:** $\text{new}[i] = \text{evals}[i] \cdot (1-r) + \text{evals}[i + 2^{n-1}] \cdot r$. Cost: $O(2^n)$.
- **Equality polynomial** $\widetilde{\text{eq}}(r, \cdot)$: $\widetilde{\text{eq}}(r, x) = \prod_{i} (r_i x_i + (1-r_i)(1-x_i))$.

### 3.3 Sumcheck Protocol

We use the sum-check protocol of Lund et al. for the statement

$$H = \sum_{x \in \{0,1\}^n} f(x) \cdot g(x) \tag{4}$$

where $f, g$ are dense MLEs (products of two degree-1 polynomials give degree-2 round polynomials).

**Round $i$ ($i = 1,\ldots,n$):**

1. Prover sends the univariate $g_i(X) = \sum_{x_{i+1},\ldots,x_n \in \{0,1\}} f(r_1,\ldots,r_{i-1}, X, x_{i+1},\ldots,x_n) \cdot g(\cdots)$, represented by evaluations at $X \in \{0,1,2\}$.
2. Verifier checks $g_i(0) + g_i(1) = H_{i-1}$ and sends random $r_i \stackrel{\$}{\leftarrow} \mathbb{F}_r$.
3. Set $H_i = g_i(r_i)$.

At the end, the verifier holds a single opening claim $f(r_1,\ldots,r_n) \cdot g(r_1,\ldots,r_n) = H_n$, which it checks via a PCS opening.

**Complexity:** Prover $O(n \cdot 2^n)$ field operations; verifier $O(n)$ field operations plus two PCS openings.

**Implementation note (bit ordering):** `DenseMLPoly::fix_first_variable` fixes the highest-index variable first (i.e., $\text{bit}_{n-1}$), because indices $i < 2^{n-1}$ have $\text{bit}_{n-1}(i) = 0$ and indices $i \geq 2^{n-1}$ have $\text{bit}_{n-1}(i) = 1$. Consequently, the sumcheck challenge vector $\mathbf{r} = (r_0, r_1, \ldots, r_{n-1})$ satisfies $r_j \leftrightarrow \text{bit}_{n-1-j}$ of the evaluation index. The verifier must account for this when computing $L(\mathbf{r})$ by reversing $\mathbf{r}$ before pairing with the LSB-first bit decomposition.

### 3.4 Lasso Lookup Argument

Given sub-table $T_k \in \mathbb{F}_r^{2^m}$ and queries $\{(\text{idx}_j, v_j)\}_{j=1}^N$ with $v_j = T_k[\text{chunk}_k(\text{idx}_j)]$, we prove correctness via a **batched MLE evaluation sumcheck**.

**Step 1 — Commit.** Represent $T_k$ as a dense MLE $\widetilde{T}_k$ over $m$ variables.

**Step 2 — Batch.** Prover and verifier agree on a random challenge $\rho \stackrel{\$}{\leftarrow} \mathbb{F}_r$ (Fiat-Shamir). Define the **selector polynomial**:

$$L_k(x) = \sum_{j=1}^N \rho^j \cdot \widetilde{\text{eq}}\!\left(\text{bin}(\text{ch}_{k,j}),\, x\right) \tag{5}$$

where $\text{ch}_{k,j} = \text{chunk}_k(\text{idx}_j)$ and $\text{bin}(\cdot)$ is the $m$-bit binary encoding.

**Step 3 — Sumcheck.** Run the degree-2 sumcheck for

$$\sum_{x \in \{0,1\}^m} \widetilde{T}_k(x) \cdot L_k(x) = \sum_{j=1}^N \rho^j \cdot T_k[\text{ch}_{k,j}] \tag{6}$$

This is sound because $L_k$ has a spike of height $\rho^j$ exactly at $\text{bin}(\text{ch}_{k,j})$, so the sum equals the claimed batched evaluation.

**Step 4 — Open.** After the sumcheck terminates at random point $\mathbf{r}$, the prover opens $\widetilde{T}_k(\mathbf{r})$ via the PCS. The verifier checks $\widetilde{T}_k(\mathbf{r}) \cdot L_k(\mathbf{r}) = H_n$ where $L_k(\mathbf{r})$ is computed directly from the public queries.

**Security.** Soundness follows from sumcheck soundness (error $\leq nd/|\mathbb{F}_r|$ per round) combined with PCS binding. The full Lasso argument additionally requires a **memory-consistency check** (grand product argument) to ensure the prover cannot substitute entries; this is planned as a future step.

**Complexity (per sub-table):**
- Prover: $O(N \cdot 2^m + m \cdot 2^m)$ to build $L_k$ and run sumcheck.
- Verifier: $O(N \cdot m)$ to recompute $L_k(\mathbf{r})$, plus $O(m)$ sumcheck verification.

### 3.5 Linear Attention Circuit

For a single attention head with sequence length $T$ and head dimension $d_h$, the circuit proves four claims:

| Step | Statement | Protocol |
|------|-----------|----------|
| 1 | $\Phi_Q[t][d] = \phi(Q[t][d])$ for all $t, d$ | Lasso (§3.4) per chunk |
| 2 | $\Phi_K[t][d] = \phi(K[t][d])$ for all $t, d$ | Lasso per chunk |
| 3 | $C[i][j] = \sum_t \Phi_K[t][i] \cdot V[t][j]$ | Sumcheck over $t$ |
| 4 | $\text{out}[t][j] = \sum_i \Phi_Q[t][i] \cdot C[i][j]$ | Sumcheck over $i$ |

Steps 3 and 4 use a **random-entry reduction**: the verifier draws a Fiat-Shamir challenge to select one entry $(i^*, j^*)$ or $(t^*, j^*)$ to audit, then a single sumcheck proves that entry. Soundness holds with probability $1 - d_h/|\mathbb{F}_r|$ over the choice of random entry.

### 3.6 Constraint Fusion

Inspired by zkGPT, before processing layer $\ell$ the verifier squeezes a random challenge $\lambda_\ell \stackrel{\$}{\leftarrow} \mathbb{F}_r$ from the Fiat-Shamir transcript. All per-head claims within layer $\ell$ are implicitly bound to $\lambda_\ell$ because the transcript state is advanced before any head proof is generated. A dishonest prover cannot swap or selectively forge one head's proof without invalidating the transcript for all subsequent proofs.

---

## 4. Security Analysis

### 4.1 Completeness

If the prover honestly executes the model and constructs all witnesses correctly, all sumcheck checks pass with probability 1 (the polynomial identities hold exactly over $\mathbb{F}_r$).

### 4.2 Soundness

**Sumcheck soundness:** By the Schwartz–Zippel lemma, a cheating prover can make a false degree-2 round polynomial pass with probability at most $2/|\mathbb{F}_r| \approx 2^{-253}$ per round. Over $n$ rounds: $2n/|\mathbb{F}_r|$.

**Lasso soundness:** Given sumcheck soundness, the only remaining attack is for the prover to open a false evaluation $\widetilde{T}_k(\mathbf{r}) \neq T_k[\mathbf{r}\text{-indexed entry}]$. This is prevented by PCS binding.

**Constraint fusion:** The Fiat-Shamir transcript ensures all per-layer proofs are bound to the same random coins; forging any single head's proof invalidates the entire transcript with overwhelming probability.

**Note on the random-entry reduction (§3.5):** Steps 3–4 currently audit only one entry per matrix product. This is a *designated-verifier* reduction — it is sound only if the verifier's challenge is truly random (i.e., Fiat-Shamir in the random-oracle model). For stronger soundness, one can batch over multiple random entries.

### 4.3 Zero Knowledge

The current implementation is **not zero-knowledge**: the prover reveals model weights, intermediate activations, and polynomial evaluations in the clear. Zero-knowledge can be added by:

- Committing to all witness polynomials before revealing evaluations (standard technique).
- Applying the BlindFold ZK layer from Jolt (Pedersen commitments + Nova folding over the sumcheck transcript) to hide all intermediate claims.

---

## 5. Fixed-Point Encoding

All real-valued tensors are represented as integers scaled by a factor $s$:

$$x_{\mathbb{F}} = \left\lfloor \frac{x}{s} \right\rfloor \bmod r \in \mathbb{F}_r$$

Arithmetic in $\mathbb{F}_r$ then simulates fixed-point integer arithmetic as long as no intermediate value overflows the field modulus. For $B$-bit activations and $k_{\max}$-bit weights, the maximum intermediate value in a matrix product is $T \cdot d_h \cdot 2^B \cdot 2^{k_{\max}}$, which must be kept below $r \approx 2^{254}$.

The Python training pipeline uses the same quantization (`StructuredLookupActivation` with explicit `clamp` + `round`) so the exported integer tables and weights exactly match the field arithmetic in the Rust prover. This **eliminates the train–prove gap**.

---

## 6. Complexity Summary

Let $L$ = layers, $H$ = heads, $T$ = sequence length, $d_h$ = head dimension, $m$ = bits per chunk, $c$ = chunk count, $N_{qk} = T \cdot d_h$ = total $\phi$ queries per head.

| Component | Prover time | Verifier time | Proof size |
|-----------|------------|---------------|------------|
| Lasso (one sub-table) | $O(N_{qk} \cdot 2^m + m \cdot 2^m)$ | $O(N_{qk} \cdot m)$ | $O(m)$ field elems |
| Lasso ($c$ sub-tables, $2H$ sets) | $O(2cH \cdot (N_{qk} 2^m + m 2^m))$ | $O(2cH \cdot N_{qk} m)$ | $O(2cHm)$ |
| Context sumcheck ($H$ heads) | $O(H \cdot T)$ | $O(H \cdot \log T)$ | $O(H \log T)$ |
| Output sumcheck ($H$ heads) | $O(H \cdot d_h)$ | $O(H \cdot \log d_h)$ | $O(H \log d_h)$ |
| **Per layer total** | $O(H(cN_{qk}2^m + T + d_h))$ | $O(H(cN_{qk}m + \log T))$ | $O(H(cm + \log T))$ |
| **All $L$ layers** | $\times L$ | $\times L$ | $\times L$ |

Constraint fusion reduces the effective verifier overhead by sharing transcript state across heads without adding separate batching proofs.

---

## 7. Comparison with Related Work

| System | Attention | Activation | Proving backend | ZK |
|--------|-----------|-----------|----------------|-----|
| **zkGPT** | Softmax (approx.) | Lookup table | Plonky2 | ✓ |
| **zkLLM** | Softmax (tlookup) | Structured lookup | Custom IOP | ✗ |
| **π-Former** | Linear (exact) | Learned structured lookup | Lasso + Spartan | roadmap |

Key advantages of π-Former:
- **Exact computation**: linear attention is not an approximation of softmax; it is a different (learnable) attention mechanism trained to be useful and provable.
- **End-to-end co-design**: the model's weights and activation tables are trained to match the exact integer/field arithmetic used in the circuit, eliminating approximation errors.
- **Extensible**: the lookup decomposition depth $c$ and chunk size $m$ are configurable trade-offs between model expressivity and proof cost.

---

## 8. Planned Extensions

### 8.1 Real Polynomial Commitment Scheme

Replace the trivial PCS with an inner-product argument (e.g., Bulletproofs-style IPA or Dory). This adds $O(\log 2^m) = O(m)$ group operations to each table opening, keeping proof size polylogarithmic in the table size.

### 8.2 Memory-Consistency Check

The current Lasso argument proves that query outputs are consistent with a *claimed* table. To prevent a dishonest prover from using a different table per query, add an offline memory-consistency argument (Spice / Lasso grand-product check) that ties all queries back to a single committed table.

### 8.3 IVC for Autoregressive Generation

For autoregressive (token-by-token) inference, prove each step's attention incrementally using Nova-style incremental verifiable computation (IVC). The running context matrix $C = \phi(K)^\top V$ is the natural accumulator state.

### 8.4 Sparse Attention Exploitation

If the trained model learns a structured sparse pattern in $\phi(Q)\phi(K)^\top$ (e.g., local windows), the sumcheck over $T$ can be restricted to non-zero terms, reducing prover cost from $O(T)$ to $O(\text{nnz})$.

### 8.5 Shared Table Argument

If multiple layers share the same $\phi$ tables (weight tying), prove table correctness once and reference it from all layers via a single Lasso instance, reducing the per-layer proof cost for lookups.

---

## 9. Implementation Notes

### Transcript Fiat-Shamir Convention

All challenges are derived from SHA3-256 with domain-separation labels. The hasher state is advanced by feeding each finalized hash back into the running state, ensuring forward-security of the squeeze:

```
state ← SHA3-256(label ‖ data ‖ state_hash)
challenge ← F::from_le_bytes_mod_order(SHA3-256(state ‖ label))
```

### Bit-Ordering Convention

`DenseMLPoly::evaluate(r)` and `fix_first_variable(r)` fix variable $\text{bit}_{n-1}$ (the MSB in the evaluation index) first. Therefore, sumcheck challenge $r_j$ corresponds to $\text{bit}_{n-1-j}$ of the final evaluation index. When constructing $L_k(\mathbf{r})$ from bit decompositions of chunk indices (which are naturally in LSB-first order), the challenge vector must be reversed. See `lookup/lasso.rs` for the explicit fix.

### Field Arithmetic

All field operations use the `ark-bn254` crate with the `ark-ff 0.4` API. The field element trait bound used throughout is `ark_ff::PrimeField`, which provides `into_bigint()` for extracting limbs.
