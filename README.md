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
                │   └─ TernaryLinear          │
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
4. **General weight matrices** require one multiplication gate per entry.

π-Former addresses all four with co-designed training and proving:

| Problem | π-Former Solution |
|---------|-------------------|
| Softmax | Linear attention: `φ(Q)(φ(K)ᵀV)` |
| Activations | Structured lookup φ, additively decomposed for Lasso |
| LayerNorm | Sumcheck for mean + constraint-fused range proof |
| Dense weights | Ternary weights `{−1, 0, +1}` → addition/subtraction only |

See [DESIGN.md](DESIGN.md) for the full technical treatment.

## Repository Layout

```
π-Former/
├── python/                     Python training pipeline
│   ├── piformer/
│   │   ├── activation.py       StructuredLookupActivation
│   │   ├── projection.py       TernaryLinear (STE training)
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
│       │   └── utils.rs        MLE helpers + TernaryValue type
│       ├── pcs/
│       │   └── mod.rs          Hyrax PCS + HyraxBatchAccumulator
│       ├── subprotocols/
│       │   ├── sumcheck.rs     Degree-2 / multi-poly sumcheck
│       │   └── combine.rs      GKR combine proof (multi-claim → 1 opening)
│       ├── lookup/
│       │   ├── lasso.rs        Lasso lookup argument (batched multi-instance)
│       │   └── range.rs        32-bit range proof (chunked Lasso)
│       ├── attention/
│       │   ├── attention.rs    Linear attention circuit
│       │   ├── layernorm.rs    LayerNorm with constraint fusion
│       │   └── projection.rs   Ternary-weight projection circuit
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
└── DESIGN.md                   Full technical specification
```

## Quickstart

### 1. End-to-End Smoke Test

Run a full setup → prove → verify cycle on a tiny synthetic model:

```bash
cd prover
cargo run --release --bin piformer -- sample --output-dir /tmp/piformer_sample
# Writes weights.json, model.pk, model.vk, witness.json, public_input.json,
# public_output.json, proof.bin
# ✓  Proof is VALID.
```

### 2. Inspect Generated Files

```bash
cargo run --release --bin piformer -- inspect /tmp/piformer_sample/model.vk
cargo run --release --bin piformer -- inspect /tmp/piformer_sample/model.pk
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
  --proof proof.bin \
  --public-input public_input.json \
  --public-output public_output.json
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
| `*.pk`    | Proving key — Hyrax G1 commitments + raw ternary weight vectors |
| `*.vk`    | Verifying key — Hyrax G1 commitments only (no raw weights) |
| `*.bin`   | Proof bundle — proof + per-proof lookup outputs (binary, magic-prefixed) |

`verify` binds the proof to verifier-supplied public I/O by recomputing the
Hyrax commitments for `public_input.json` and `public_output.json`. Lookup table
metadata is taken from the verifying key, not from the proof bundle.

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

The context matrix `C` depends only on keys and values, not on any query, so it is computed once and reused across all positions. This avoids the O(T²d) cost of full attention and eliminates exponentials and row-wise divisions.

### Structured Lookup Activation

```python
φ(x) = Σ_{i=0}^{c-1}  table_i[ (x_int >> (i * m)) & mask ]
```

- Input `x` is quantized to a `B = c·m` bit integer.
- Decomposed into `c` chunks of `m` bits each.
- Each chunk indexes a learnable sub-table of size `2^m`.
- Tables are initialized to approximate GeLU and learned end-to-end.

This additive decomposition matches the Lasso lookup argument exactly: `c` sub-tables of size `2^m` instead of one monolithic table of size `2^B`. For `B=16, c=2`: 512 commitments instead of 65 536.

### Ternary Weights

All projection matrices (`W_Q, W_K, W_V, W_O`) and FFN weight matrices are constrained to entries in `{−1, 0, +1}` (represented as the `TernaryValue` enum in Rust). A matrix-vector product with ternary weights requires only additions and subtractions — no multiplication gates — making all linear projections essentially free in the sumcheck constraint system. A straight-through estimator (STE) enables gradient flow during training.

The prover exploits this during witness generation via `eval_cols_ternary`, which accumulates equality-polynomial evaluations with `+= / -= / skip` instead of field multiplications.

### LayerNorm via Constraint Fusion

LayerNorm is proven without any division gates:

1. **Mean sumcheck** — proves `sum_x[i] = Σⱼ x[i][j]` using a degree-2 sumcheck.
2. **Sigma range proof** — verifies `σ[i]` is the integer floor-square-root of `d·var_x[i]` by showing both residuals are non-negative, via a chunked Lasso range proof.
3. **Y constraint fusion** — verifies each output `y[i][j]` satisfies the scaled LayerNorm formula at a single random evaluation point using another range proof over lo/hi residuals.

The verifier checks all constraints at a single random evaluation point — O(1) per layer, with no O(T·d) loops.

## Key Implementation Features

### Ternary Weight Encoding

Weight matrices are stored as `Vec<Vec<TernaryValue>>` where `TernaryValue ∈ {ONE, ZERO, MINUSONE}`. The prover uses `eval_cols_ternary` to evaluate the weight MLE at a challenge point without materializing the full field matrix, and `convert_tm_to_fm` only when a full field representation is needed (e.g., for commitment).

### Batched Hyrax Openings (`HyraxBatchAccumulator`)

Multiple Hyrax openings at different points are accumulated into a single `HyraxBatchAccumulator`. The inner-product check is performed immediately on `add_verify` / `add_verify_batch`, but the MSM (the expensive part) is deferred to a single `finalize` call. This reduces `K` Hyrax verifications from `2K` MSMs to just `2` MSMs.

The verifier uses cross-layer accumulators: all LayerNorm openings in a model share one `acc_t` and one `acc_td`; all projection weight openings share `proj_acc_w` and `proj_acc_b`. These are finalized once at the end of the entire model verification.

### GKR Combine Proofs

Intermediate tensors (Q, K, V, out_inner, x_norm1, etc.) are referenced by multiple sub-provers at different evaluation points. A `CombineProof` bundles all claims on the same commitment into a single Hyrax opening via a short sumcheck. Eight combine proofs per block are then batch-verified in two MSMs using `hyrax_verify_multi_point`.

### Batched Global Lasso

All Lasso instances across all blocks (FFN activation lookups + Q/K attention kernel lookups) are batched into a single `LassoMultiProof` via `prove_lasso_multi` / `verify_lasso_multi`. This replaces `3L` independent Lasso proofs with one combined sumcheck and one Hyrax opening.

### Homomorphic Residual Connections

Residual additions (`X_mid = X_in + Out_attn`, `X_out = X_mid + Out_ffn`) are verified using the homomorphic property of Hyrax (Pedersen) commitments: `Com(A) + Com(B) = Com(A+B)`. The verifier computes `add_commitments` in O(√N) group operations with no prover assistance.

## Rust Prover Components

### Hyrax PCS (`pcs/`)

Transparent (no trusted setup), based on discrete-log hardness of BN254 G1. For a `2^n`-evaluation polynomial arranged as a `2^ν × 2^σ` matrix:

- **Commit:** one MSM per row → `2^ν` G1 points.
- **Open at r = (r_L ‖ r_R):** prover sends row-collapsed vector `w′`; verifier checks the MSM equation and inner product.
- **Batch:** `HyraxBatchAccumulator` defers all MSMs to a single `finalize` call.
- **Multi-point:** `hyrax_verify_multi_point` batches openings at different points into 2 MSMs.

### Sumcheck (`subprotocols/sumcheck.rs`)

Proves `H = Σ_{x ∈ {0,1}^n} f(x)·g(x)`. Each round polynomial has degree ≤ 2, represented by evaluations at `{0, 1, 2}`. Also supports `SumcheckProofMulti` for batching multiple `(f_i, g_i)` pairs with Fiat-Shamir-random linear combination.

### GKR Combine (`subprotocols/combine.rs`)

`prove_combine` / `verify_combine_deferred` reduce multiple evaluation claims on the same committed polynomial to a single Hyrax opening. The verifier defers the opening check; `hyrax_verify_multi_point` then handles all deferred openings in a batch.

### Lasso (`lookup/lasso.rs`)

`prove_lasso_multi` / `verify_lasso_multi` handle a vector of `LassoInstance` objects (each with its own sub-tables and query set) in a single combined sumcheck + one Hyrax opening, via Fiat-Shamir random linear combination across instances.

### Range Proof (`lookup/range.rs`)

Proves field elements lie in `[0, 2^32)` by splitting into two 16-bit chunks, committing via Hyrax, and running a LogUp-style multiplicity check against the identity table `T[i] = i` of size 65 536. `verify_range_deferred` / `verify_range_m_batch` defer and batch the multiplicity-commitment opening.

## Cryptographic Parameters

| Parameter | Value |
|-----------|-------|
| Scalar field | BN254 Fr (~254-bit prime) |
| Hash / transcript | SHA3-256 (Fiat-Shamir) |
| PCS | Hyrax (transparent; DL hardness on BN254 G1) |
| Lookup argument | Lasso (batched multi-instance) |
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
- [ ] Zero-knowledge layer (blinded witness polynomials)

## References

- **Lasso** — Setty, Thaler, Wahby. *Unlocking the Lookup Singularity with Lasso.* EUROCRYPT 2024.
- **Spartan** — Setty. *Spartan: Efficient and general-purpose zkSNARKs without trusted setup.* CRYPTO 2020.
- **Hyrax** — Wahby et al. *Doubly-Efficient zkSNARKs Without Trusted Setup.* IEEE S&P 2018.
- **Jolt** — Arun et al. *Jolt: SNARKs for Virtual Machines via Lookups.* EUROCRYPT 2024.
- **zkGPT** — see `paper/zkGPT.pdf`
- **zkLLM** — see `paper/zkLLM.pdf`
- **Linear Transformers** — Katharopoulos et al. *Transformers are RNNs.* ICML 2020.
- **BitNet** — Wang et al. *BitNet: Scaling 1-bit Transformers for Large Language Models.* 2023.

## License

MIT
