# π-Former

> **Efficient SNARKs for Linear Transformers via Structured Lookup Attention**

π-Former is a research prototype that makes transformer inference efficiently provable in a zero-knowledge SNARK. It replaces the standard softmax attention and dense activations with ZK-friendly primitives — linear attention with a learnable, additively decomposed kernel function — and proves the resulting computation using a Lasso-based lookup argument on top of the Spartan sumcheck IOP.

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
                           │  piformer_weights.json
                ┌──────────▼──────────────────┐
                │   Rust / SNARK Prover        │
                │                             │
                │  PiFormerProver             │
                │   ├─ Lasso lookup (φ)       │
                │   ├─ Sumcheck (matmul)      │
                │   └─ Constraint fusion      │
                │          │                  │
                │  PiFormerVerifier           │
                └─────────────────────────────┘
```

## Motivation

Proving transformer inference in a SNARK is expensive because:

1. **Softmax** requires `exp`, row-wise normalization, and division — none of which are polynomial operations.
2. **Dense activations** (GELU, SiLU) require large lookup tables when encoded naively.
3. **Matrix multiplications** with general weights require one multiplication gate per entry.

π-Former addresses all three with co-designed training and proving:

| Problem | π-Former Solution |
|---------|-------------------|
| Softmax | Linear attention: `φ(Q)(φ(K)ᵀV) / Z` |
| Activations | Structured lookup φ, additively decomposed for Lasso |
| Dense weights | Power-of-two weights → shift-and-add only |
| Proof size | Constraint fusion across layers/heads |

See [DESIGN.md](DESIGN.md) for the full technical treatment.

## Repository Layout

```
π-Former/
├── python/                  Python training pipeline
│   ├── piformer/
│   │   ├── activation.py    StructuredLookupActivation
│   │   ├── projection.py    PowerOfTwoLinear (STE training)
│   │   ├── attention.py     LinearAttentionLayer
│   │   ├── model.py         PiFormerBlock / PiFormerModel
│   │   └── export.py        JSON weight export for Rust
│   ├── train.py             Training script
│   └── requirements.txt
│
├── prover/                  Rust SNARK prover/verifier
│   └── src/
│       ├── field.rs         BN254 Fr field helpers
│       ├── transcript.rs    Fiat-Shamir (SHA3-256)
│       ├── poly/
│       │   └── dense.rs     Dense multilinear polynomial
│       ├── subprotocols/
│       │   └── sumcheck.rs  Degree-2 sumcheck protocol
│       ├── lookup/
│       │   └── lasso.rs     Lasso lookup argument
│       ├── attention/
│       │   └── linear.rs    Linear attention circuit
│       ├── prover.rs        PiFormerProver
│       ├── verifier.rs      PiFormerVerifier
│       └── bin/example.rs   End-to-end integration test
│
├── paper/                   Reference papers
│   ├── zkGPT.pdf
│   └── zkLLM.pdf
├── reference/               Reference implementations (submodule)
│   └── jolt/                Jolt zkVM (Rust)
├── IDEA.md                  Design sketch
└── DESIGN.md                Full technical specification
```

## Quickstart

### Training (Python)

```bash
cd python
pip install -r requirements.txt
python train.py
# → produces piformer_weights.json
```

### Proving & Verifying (Rust)

```bash
cd prover
cargo run --bin example
# → Proving linear attention...
# → Verifying...
# → ✓ Proof verified successfully!
```

### Running Tests

```bash
cd prover && cargo test
```

## Model Architecture

### ZK-Friendly Attention

π-Former replaces softmax attention with:

```
Attention(Q, K, V) = φ(Q) (φ(K)ᵀ V) / Z

where:
  φ     = StructuredLookupActivation (learnable, Lasso-compatible)
  Z     = φ(Q) · Σ_s φ(K_s)         (normalizer, no row-wise division)
```

The associativity of `φ(K)ᵀV` means the context matrix can be computed in `O(n d²)` rather than `O(n² d)`, and each step is a polynomial operation.

### Structured Lookup Activation

```python
φ(x) = Σ_{i=0}^{c-1}  table_i[ (x_int >> (i * m)) & mask ]
```

- Input `x` is quantized to a `num_bits`-bit integer
- Decomposed into `c` chunks of `m = num_bits/c` bits each
- Each chunk indexes a learnable sub-table of size `2^m`
- Tables are initialized to approximate GeLU and learned end-to-end

This additive decomposition matches the Lasso lookup argument structure exactly: `c` sub-tables of size `2^m` instead of one table of size `2^{num_bits}`.

### Power-of-Two Weights

Projection matrices `W_Q, W_K, W_V, W_O` and FFN weights are constrained to:

```
w ∈ { 0, ±1, ±2, ±4, ±8, ..., ±2^max_exp }
```

In the SNARK circuit, multiplying by `2^k` is a field constant (no multiplication gate), so the linear projections reduce to additions only. A straight-through estimator (STE) allows gradient flow during training.

## Rust Prover Components

### Dense Multilinear Polynomial

`DenseMLPoly` stores `2^n` evaluations over `{0,1}^n` and supports:
- `evaluate(r)` — multilinear extension at arbitrary `r ∈ Fⁿ` via repeated halving
- `fix_first_variable(r)` — reduce `n`-var poly to `(n-1)`-var (O(2ⁿ))
- `eq_poly(r)` — construct the equality polynomial `eq(r, ·)`

### Sumcheck Protocol

Proves `H = Σ_{x ∈ {0,1}^n} f(x)·g(x)` for two multilinear polynomials.
Each round polynomial has degree ≤ 2 (product of two MLEs); represented by evaluations at `{0,1,2}` and interpolated via quadratic Lagrange basis.

### Lasso Lookup Argument

For each sub-table `T_k` and queries `{(idx_j, val_j)}`:

1. Build selector polynomial `L_k(x) = Σ_j ρʲ · eq(binary(chunk_k(idx_j)), x)`
2. Run sumcheck: `Σ_{x ∈ {0,1}^m} T_k(x) · L_k(x) = Σ_j ρʲ · T_k[chunk_k(idx_j)]`
3. Open `T_k` at the random sumcheck point (trivial PCS; replace with Dory/IPA for production)

### Constraint Fusion

Before processing each layer, the prover squeezes a random challenge from the Fiat-Shamir transcript. This binds all per-head proofs in the layer together, so the verifier cannot accept a valid proof for one head alongside a bogus proof for another — without adding a separate batching protocol.

## Cryptographic Parameters

| Parameter | Value |
|-----------|-------|
| Scalar field | BN254 Fr (~254-bit prime) |
| Hash / transcript | SHA3-256 (Fiat-Shamir) |
| PCS (current) | Trivial (prover reveals evaluation) |
| PCS (planned) | Dory or inner-product argument |
| Lookup argument | Lasso (batched MLE sumcheck) |

## Roadmap

- [ ] Replace trivial PCS with IPA or Dory opening proofs
- [ ] Full model export → Rust proof (load `piformer_weights.json` in prover)
- [ ] Causal / autoregressive masking in linear attention
- [ ] IVC / Nova folding for streaming proofs (token-by-token)
- [ ] Benchmarks against zkGPT and zkLLM baselines
- [ ] Shared table argument across layers (same φ weights)
- [ ] Fixed-point quantization alignment between Python and Rust

## References

- **Lasso** — Setty, Thaler, Wahby. *Unlocking the Lookup Singularity with Lasso.* EUROCRYPT 2024.
- **Spartan** — Setty. *Spartan: Efficient and general-purpose zkSNARKs without trusted setup.* CRYPTO 2020.
- **Jolt** — Arun et al. *Jolt: SNARKs for Virtual Machines via Lookups.* EUROCRYPT 2024.
- **zkGPT** — see `paper/zkGPT.pdf`
- **zkLLM** — see `paper/zkLLM.pdf`
- **Linear Transformers** — Katharopoulos et al. *Transformers are RNNs.* ICML 2020.

## License

MIT
