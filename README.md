# π-Former

> **Succinct ZK Proofs of Transformer Inference via Structured Lookup Attention**

π-Former is a research prototype that makes transformer inference efficiently provable in a zero-knowledge SNARK. It replaces softmax attention and dense activations with ZK-friendly primitives — linear attention with a learnable, additively decomposed kernel — and proves the resulting computation using Hyrax PCS, the Spartan sumcheck IOP, and a Lasso-based lookup argument.

```
                ┌─────────────────────────────┐
                │   Python / PyTorch          │
                │   Training Pipeline         │
                │                             │
                │  PiFormerModel              │
                │   ├─ LinearAttentionLayer   │
                │   ├─ StructuredLookupAct    │
                │   └─ PowerOfTwoLinear       │
                │          │                  │
                │     export_model()          │
                └──────────┬──────────────────┘
                           │  weights.json
                ┌──────────▼──────────────────┐
                │   piformer CLI (Rust)        │
                │                             │
                │  piformer setup             │  ← offline preprocessing
                │  piformer prove             │  ← witness → proof.bin
                │  piformer verify            │  ← proof.bin → VALID/INVALID
                │  piformer inspect           │  ← human-readable dump
                │  piformer sample            │  ← end-to-end smoke test
                └─────────────────────────────┘
```

## Motivation

Proving transformer inference in a SNARK is expensive because:

1. **Softmax** requires `exp`, row-wise normalization, and division — none of which are polynomial operations.
2. **Dense activations** (GELU, SiLU) require large lookup tables when encoded naively.
3. **Layer normalization** involves a square root and division by a running variance.
4. **Matrix multiplications** with general weights require one multiplication gate per entry.

π-Former addresses all four with co-designed training and proving:

| Problem | π-Former Solution |
|---------|-------------------|
| Softmax | Linear attention: `φ(Q)(φ(K)ᵀV)` |
| Activations | Structured lookup φ, additively decomposed for Lasso |
| LayerNorm | Sumcheck for mean/variance + constraint-fused range proof |
| Dense weights | Power-of-two weights → shift-and-add only |

See [DESIGN.md](DESIGN.md) for the full technical treatment.

## Repository Layout

```
π-Former/
├── python/                     Python training pipeline
│   ├── piformer/
│   │   ├── activation.py       StructuredLookupActivation
│   │   ├── projection.py       PowerOfTwoLinear (STE training)
│   │   ├── attention.py        LinearAttentionLayer
│   │   ├── model.py            PiFormerBlock / PiFormerModel
│   │   └── export.py           JSON weight export for Rust
│   ├── train_demo.py           Demo training script
│   └── requirements.txt
│
├── prover/                     Rust SNARK prover/verifier (library + CLI)
│   └── src/
│       ├── lib.rs              Library root
│       ├── field.rs            BN254 Fr type alias
│       ├── transcript.rs       Fiat-Shamir (SHA3-256)
│       ├── poly/
│       │   ├── dense.rs        Dense multilinear polynomial
│       │   └── utils.rs        MLE helper utilities
│       ├── pcs/
│       │   └── mod.rs          Hyrax PCS (transparent, no trusted setup)
│       ├── subprotocols/
│       │   └── sumcheck.rs     Degree-2 sumcheck protocol
│       ├── lookup/
│       │   ├── lasso.rs        Lasso lookup argument
│       │   └── range.rs        32-bit range proof (chunked Lasso)
│       ├── attention/
│       │   ├── attention.rs    Linear attention circuit
│       │   ├── layernorm.rs    LayerNorm with constraint fusion
│       │   ├── projection.rs   Weight-matrix projection circuit
│       │   └── ternary_check.rs  Ternary weight validity proof
│       ├── ffn/
│       │   └── ffn.rs          Feed-forward network circuit
│       ├── setup.rs            Preprocessing → proving key / verifying key
│       ├── prover.rs           Full transformer block prover
│       ├── verifier.rs         Full transformer block verifier
│       └── bin/
│           ├── piformer.rs     CLI entry point
│           └── piformer/
│               ├── codec.rs    Binary serialization (.pk / .vk / .bin)
│               ├── json_io.rs  JSON serialization (weights / witness)
│               └── sample.rs   Synthetic model for smoke tests
│
├── paper/                      Reference papers
│   ├── zkGPT.pdf
│   └── zkLLM.pdf
├── reference/                  Reference implementations (submodules)
│   └── jolt/                   Jolt zkVM (Rust)
├── IDEA.md                     Initial design sketch
└── DESIGN.md                   Full technical specification
```

## Quickstart

### 1. End-to-End Smoke Test

Run a full setup → prove → verify cycle on a tiny synthetic model:

```bash
cd prover
cargo run --release --bin piformer -- sample --output-dir /tmp/piformer_sample
# Writes weights.json, model.pk, model.vk, witness.json, proof.bin
# ✓ Proof is VALID.
```

### 2. Inspect Generated Files

```bash
cargo run --release --bin piformer -- inspect /tmp/piformer_sample/model.vk
cargo run --release --bin piformer -- inspect /tmp/piformer_sample/proof.bin
```

### 3. Full Workflow with Your Own Model

```bash
# Offline setup (run once per model)
piformer setup \
  --weights model.json \
  --seq-len 32 \
  --pk model.pk \
  --vk model.vk

# Prove an inference run
piformer prove \
  --pk model.pk \
  --witness witness.json \
  --proof proof.bin

# Verify the proof
piformer verify \
  --vk model.vk \
  --proof proof.bin
```

### 4. Python Training (optional)

```bash
cd python
pip install -r requirements.txt
python train_demo.py
# → produces piformer_weights.json
```

### 5. Running Tests

```bash
cd prover && cargo test
```

## File Formats

| Extension | Contents |
|-----------|----------|
| `*.json`  | Human-readable weights or witness (field elements as hex strings) |
| `*.pk`    | Proving key — Hyrax G1 commitments + raw weight vectors |
| `*.vk`    | Verifying key — Hyrax G1 commitments only (no raw weights) |
| `*.bin`   | Proof bundle — proof + public instances (binary, magic-prefixed) |

All binary files carry a magic header (`PFMR_PK\0`, `PFMR_VK\0`, `PFMR_PR\0`) and a version byte for forward-compatibility detection.

## Model Architecture

### ZK-Friendly Attention

π-Former replaces softmax attention with linear (kernel) attention:

```
Attention(Q, K, V) = φ(Q) (φ(K)ᵀ V)

where:
  φ  = StructuredLookupActivation (learnable, Lasso-compatible)
  C  = φ(K)ᵀ V   (context matrix, computed once in O(T d²))
```

The context matrix `C` does not depend on any query, so it is computed once and reused across all positions. This avoids the O(T²d) cost of full attention and eliminates exponentials and row-wise divisions.

### Structured Lookup Activation

```python
φ(x) = Σ_{i=0}^{c-1}  table_i[ (x_int >> (i * m)) & mask ]
```

- Input `x` is quantized to a `B = c·m` bit integer.
- Decomposed into `c` chunks of `m` bits each.
- Each chunk indexes a learnable sub-table of size `2^m`.
- Tables are initialized to approximate GeLU and learned end-to-end.

This additive decomposition matches the Lasso lookup argument exactly: `c` sub-tables of size `2^m` instead of one monolithic table of size `2^B`. For `B=16, c=2`: 512 commitments instead of 65 536.

### Power-of-Two Weights

Projection matrices `W_Q, W_K, W_V, W_O` and FFN weights are constrained to entries in `{0, ±1, ±2, ±4, …, ±2^k}`. In the circuit, multiplying by `2^k` is a field constant (no multiplication gate), reducing all linear projections to additions. A straight-through estimator (STE) enables gradient flow during training.

### LayerNorm via Constraint Fusion

LayerNorm is proven without any division gates:

1. **Mean sumcheck** — proves `sum_x[i] = Σⱼ x[i][j]` using a degree-2 sumcheck.
2. **Variance sumcheck** — proves `var_x[i] = Σⱼ (d·x[i][j] − sum_x[i])²`.
3. **Range proofs** — verify `σ[i]` is the integer floor-square-root of `var_x[i]` and that output `y[i][j]` lies in the correct interval, both via a chunked Lasso range proof.

The verifier checks all constraints at a single random evaluation point (O(1) per layer), with no O(N) loops.

## Rust Prover Components

### Hyrax Polynomial Commitment Scheme (`pcs/`)

Transparent (no trusted setup), based on discrete-log hardness of BN254 G1:

- **Commit:** arrange the `2^n` evaluations in a `2^ν × 2^σ` matrix; commit each row with a multi-scalar multiplication (MSM).
- **Open at r = (r_L ‖ r_R):** prover sends `w′ = L(r_L)·M` (row-collapsed vector); verifier checks `Σᵢ Lᵢ·Cᵢ = MSM(gens, w′)` and `⟨R(r_R), w′⟩ = claimed_eval`.
- **Cost:** prover O(√N) group ops; verifier O(√N) group ops + O(log N) field ops.

### Dense Multilinear Polynomial (`poly/`)

`DenseMLPoly` stores `2^n` evaluations over `{0,1}^n` and supports:
- `evaluate(r)` — multilinear extension at arbitrary `r ∈ Fⁿ` via repeated halving, O(2ⁿ).
- `fix_first_variable(r)` — reduce n-variable poly to (n−1)-variable, O(2ⁿ).
- `eq_poly(r)` — equality polynomial `eq(r, ·)`.

### Sumcheck Protocol (`subprotocols/`)

Proves `H = Σ_{x ∈ {0,1}^n} f(x)·g(x)` for two MLEs. Each round polynomial has degree ≤ 2; represented by evaluations at `{0, 1, 2}` and verified via quadratic Lagrange interpolation. Prover cost O(n·2ⁿ); verifier cost O(n) plus two PCS openings.

### Lasso Lookup Argument (`lookup/lasso.rs`)

Proves `v_j = T[idx_j]` for a batch of queries via a batched MLE sumcheck over a selector polynomial built from Fiat-Shamir–randomized equality polynomials. Sub-table decomposition keeps table commitment cost at O(c · 2^m) instead of O(2^B).

### Range Proof (`lookup/range.rs`)

Proves field elements lie in `[0, 2^32)` by splitting each value into two 16-bit chunks, committing to them via Hyrax, and running a LogUp-style multiplicity check against the identity table `T[i] = i` of size 65 536.

### Linear Attention Circuit (`attention/attention.rs`)

Proves four claims for a single attention head:

| Step | Statement | Protocol |
|------|-----------|----------|
| 1 | `Φ_Q[t][d] = φ(Q[t][d])` for all `t, d` | Lasso per chunk |
| 2 | `Φ_K[t][d] = φ(K[t][d])` for all `t, d` | Lasso per chunk |
| 3 | `C[i][j] = Σ_t Φ_K[t][i] · V[t][j]` | Sumcheck over `t` |
| 4 | `out[t][j] = Σ_i Φ_Q[t][i] · C[i][j]` | Sumcheck over `i` |

## Cryptographic Parameters

| Parameter | Value |
|-----------|-------|
| Scalar field | BN254 Fr (~254-bit prime) |
| Hash / transcript | SHA3-256 (Fiat-Shamir) |
| PCS | Hyrax (transparent; DL hardness on BN254 G1) |
| Lookup argument | Lasso (batched MLE sumcheck) |
| Range proof | Chunked Lasso (16-bit chunks, identity table) |
| Non-interactive | Fiat-Shamir in the random-oracle model |

## Roadmap

- [ ] Fixed-point quantization alignment between Python training and Rust prover
- [ ] Full model export → Rust proof (load `piformer_weights.json` in the prover)
- [ ] Causal / autoregressive masking in linear attention
- [ ] Shared table argument across layers (same φ weights → prove once)
- [ ] IVC / Nova folding for streaming token-by-token proofs
- [ ] Benchmarks against zkGPT and zkLLM baselines
- [ ] On-chain (EVM) verifier

## References

- **Lasso** — Setty, Thaler, Wahby. *Unlocking the Lookup Singularity with Lasso.* EUROCRYPT 2024.
- **Spartan** — Setty. *Spartan: Efficient and general-purpose zkSNARKs without trusted setup.* CRYPTO 2020.
- **Hyrax** — Wahby et al. *Doubly-Efficient zkSNARKs Without Trusted Setup.* IEEE S&P 2018.
- **Jolt** — Arun et al. *Jolt: SNARKs for Virtual Machines via Lookups.* EUROCRYPT 2024.
- **zkGPT** — see `paper/zkGPT.pdf`
- **zkLLM** — see `paper/zkLLM.pdf`
- **Linear Transformers** — Katharopoulos et al. *Transformers are RNNs.* ICML 2020.

## License

MIT
