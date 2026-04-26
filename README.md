# π-Former

> **Succinct ZK Proofs of Transformer Inference via Structured Lookup Attention**

π-Former is a research prototype that makes transformer inference efficiently provable in a zero-knowledge SNARK. It replaces softmax attention and dense activations with ZK-friendly primitives — linear attention with a learnable, additively decomposed kernel and ternary projection weights — and proves the resulting computation using Hyrax PCS, the Spartan sumcheck IOP, and a Lasso-based lookup argument.

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
                │     export_all()            │
                └──────────┬──────────────────┘
                           │  weights.json
                           │  witness.json
                ┌──────────▼──────────────────┐
                │   piformer CLI (Rust)       │
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
| LayerNorm | Sumcheck for mean + cubic sumcheck for variance + chunked-Lasso range proof |
| Dense weights | Ternary weights `{−1, 0, +1}` with a single learnable `α` scale per layer |

See [DESIGN.md](DESIGN.md) for the full technical treatment.

## Repository Layout

```
π-Former/
├── python/                     Python training pipeline
│   ├── piformer/
│   │   ├── activation.py       StructuredLookupActivation
│   │   ├── projection.py       TernaryLinear (STE training)
│   │   ├── attention.py        LinearAttentionLayer
│   │   ├── model.py            PiFormerBlock / PiFormerFFN / PiFormerModel
│   │   ├── quant.py            Integer / field quantization helpers
│   │   ├── witness.py          Integer forward pass → JSON witness
│   │   └── export.py           weights.json + witness.json exporter
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
│       │   ├── sumcheck.rs     Degree-2/3 sumcheck + multi-poly batched variant
│       │   └── combine.rs      GKR combine proof (multi-claim → 1 opening)
│       ├── lookup/
│       │   ├── lasso.rs        Lasso lookup argument (batched multi-instance)
│       │   └── range.rs        32-bit range proof (chunked Lasso, shared m_com)
│       ├── attention/
│       │   ├── attention.rs    Linear attention sub-prover
│       │   ├── layernorm.rs    LayerNorm with constraint fusion
│       │   ├── projection.rs   Ternary-weight projection (single + batched QKV)
│       │   └── ternary_check.rs  Stand-alone Lasso check that weights ∈ {−1,0,1}
│       ├── ffn/
│       │   └── ffn.rs          Feed-forward network sub-prover
│       ├── cross_layer/
│       │   └── projection.rs   Stand-alone cross-layer projection sumcheck
│       ├── setup.rs            Preprocessing → proving key / verifying key
│       ├── prover.rs           Cross-block batch model prover
│       ├── verifier.rs         Cross-block batch model verifier
│       └── bin/
│           ├── piformer.rs     CLI entry point
│           └── piformer/
│               ├── codec.rs    Binary serialization (.pk / .vk / .bin)
│               ├── json_io.rs  JSON serialization (weights / witness)
│               └── sample.rs   Synthetic model for smoke tests
│
├── benchmark.py                End-to-end timing benchmark driver
├── demo.sh                     Convenience demo script
├── paper/                      Reference papers (zkGPT, zkLLM)
├── ref/                        Reference implementations
└── DESIGN.md                   Full technical specification
```

The two stand-alone modules — `attention/ternary_check.rs` and `cross_layer/projection.rs` — are self-contained protocols with their own tests. They are **not yet wired into the end-to-end prover** in `prover.rs`; they exist as building blocks for upcoming work (enforcing the ternary constraint inside the proof, and collapsing per-layer projection sumchecks into one cross-layer sumcheck).

## Quickstart

### 1. End-to-End Smoke Test

Run a full setup → prove → verify cycle on a tiny synthetic model:

```bash
cd prover
cargo run --release --bin piformer -- sample --output-dir /tmp/piformer_sample
# Writes weights.json, model.pk, model.vk, witness.json,
# public_input.json, public_output.json, proof.bin
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
# → produces piformer_weights.json and piformer_witness.json
```

The Rust prover is currently **single-head only** (`n_heads = 1`); training and export will refuse other configurations.

### 5. Benchmarking

```bash
python benchmark.py tiny           # one named config
python benchmark.py --all          # tiny → small → medium → large → gpt2-small
python benchmark.py small --seq-len 32 --no-build
```

`benchmark.py` builds a random model in PyTorch, exports `weights.json` / `witness.json`, and times `setup` / `prove` / `verify` via the Rust CLI.

### 6. Running Tests

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

The Python `LinearAttentionLayer` additionally divides by a normalizer `Z = φ(Q)·Σ_s φ(K_s)` for training stability. The Rust prover proves the **un-normalized** form `φ(Q)(φ(K)ᵀV) · W_O`; aligning the two requires either disabling `Z` at export time or extending the Rust circuit with a row-normalization sumcheck (planned).

### Structured Lookup Activation

```python
φ(x) = Σ_{i=0}^{c-1}  table_i[ (x_int >> (i * m)) & mask ]
```

- Input `x` is quantized to a `B = c·m` bit integer.
- Decomposed into `c` chunks of `m` bits each.
- Each chunk indexes a learnable sub-table of size `2^m`.
- Tables are initialized so their sum approximates GeLU and learned end-to-end.

This additive decomposition matches the Lasso lookup argument exactly: `c` sub-tables of size `2^m` instead of one monolithic table of size `2^B`. For `B=16, c=2`: 512 commitments instead of 65 536.

### Ternary Weights

All projection matrices (`W_Q, W_K, W_V, W_O`, FFN `W_1, W_2`, and the LM head) are constrained to entries in `{−1, 0, +1}` (represented as the `TernaryValue` enum in Rust), with a single learnable scalar `α` per layer absorbing the magnitude. A matrix-vector product with ternary weights requires only additions and subtractions — no multiplication gates — so all linear projections are essentially free in the sumcheck constraint system. A straight-through estimator (STE) enables gradient flow during training.

The prover exploits this during witness generation via `eval_cols_ternary`, which accumulates equality-polynomial evaluations with `+= / -= / skip` instead of field multiplications. `convert_tm_to_fm` is only used when a full field representation is needed (e.g., for commitment).

The dedicated `attention/ternary_check.rs` module proves the ternary constraint itself via a 4-element Lasso table `[0, 1, p−1, 0]`. It is currently a stand-alone protocol; future versions will fold it into the end-to-end prover so the `α`-scaled weights are constrained to `{−α, 0, +α}` inside the proof.

### LayerNorm via Constraint Fusion

LayerNorm is proven without any division gates:

1. **Mean sumcheck** — proves `sum_x[i] = Σⱼ x[i][j]` using a degree-2 sumcheck.
2. **Variance sumcheck** — proves `var_x[i] = Σⱼ (d·x[i][j] − sum_x[i])²` using a cubic sumcheck.
3. **Sigma range proof** — verifies `σ[i]` is the integer floor-square-root of the variance by showing both `(d·σ)² ≤ var` and `(d·σ + d)² > var` residuals are non-negative, via a chunked Lasso range proof.
4. **Y constraint fusion** — verifies each output `y[i][j]` satisfies the scaled LayerNorm formula `2·γ·(d·x − sum_x) + 2·β·d·σ ≈ 2·d·σ·y` (up to integer rounding) at a single random evaluation point via another range proof over the lo/hi residuals. The `γX` and `σY` legs are fused into a single multi-batched sumcheck.

The verifier checks all constraints at a single random evaluation point — O(1) per layer, with no O(T·d) loops.

## Key Implementation Features

### Cross-Block Batch Sumchecks

The end-to-end prover collapses the per-layer sumchecks into **one shared sumcheck per protocol type across all L blocks**, using a Fiat-Shamir random linear combination (`SumcheckProofMulti`). After Phase 1 (per-block range proofs, LN1, LN2, intermediate-matrix commitments), the model-level prover runs:

| Batch | Statement | Shared challenge |
|-------|-----------|------------------|
| `batch_qkv`      | `Q, K, V = X_norm1 · W_{Q,K,V}` (all blocks)            | `r_k_qkv` |
| `batch_oproj`    | `Out_attn = (φ(Q)φ(K)ᵀV) · W_O` output projection       | `r_k_o`   |
| `batch_attn_out` | `out_inner[t][j] = Σᵢ φ(Q)[t,i] · context[i,j]`         | `batch_r_attn_out` |
| `batch_attn_ctx` | `context[i][j] = Σ_t φ(K)[t,i] · V[t,j]`                | `batch_r_attn_ctx` |
| `batch_ffn_y`    | `Y_ffn = A · W_2`                                        | `r_k_fy`  |
| `batch_ffn_m`    | `M_ffn = X_norm2 · W_1`                                  | `r_k_m`   |

Each batch produces one logarithmic-length sumcheck transcript (instead of L copies) and one batched final-evaluation claim per block, opened later via the cross-block batch opens described below.

### Batched Hyrax Openings

Multiple Hyrax openings at the same point are batched into a single `hyrax_open_batch` (one MSM, deferred mu-challenge for inner-product check). Multiple openings at *different* points are batched into a `HyraxBatchAccumulator`: the inner-product check runs immediately on `add_verify`, and the MSM is deferred to a single `finalize` call. This reduces `K` Hyrax verifications from `2K` MSMs to just `2`.

The verifier maintains nine cross-cutting accumulators (`ln_acc_t`, `ln_acc_td`, `proj_acc_w`, `proj_acc_b`, `lmh_acc_w`, `lmh_acc_b`, `acc_range_sig`, `acc_range_y`, `acc_range_m`, plus `inter_acc` for per-block attention `v` openings) which are finalized once at the end of the entire model verification.

### Global Intermediate Open

After all cross-block sumchecks have fixed `r_td = (r_t ‖ r_out)`, a single `hyrax_open_batch` opens the **5L intermediate matrices** (`Q, K, V, Out_attn, Out_ffn` per block) at the shared point `r_td` — replacing 5L individual openings with one batched MSM.

### Batched Global Lasso

All Lasso instances across all blocks (FFN activation lookups + Q/K attention kernel lookups) are batched into a single `LassoMultiProof` via `prove_lasso_multi` / `verify_lasso_multi`. This replaces `3L` independent Lasso proofs with one combined sumcheck and one Hyrax opening.

### Homomorphic Residual Connections

Residual additions (`X_mid = X_in + Out_attn`, `X_out = X_mid + Out_ffn`) are verified using the homomorphic property of Hyrax (Pedersen) commitments: `Com(A) + Com(B) = Com(A+B)`. The verifier computes `add_commitments` in O(√N) group operations with no prover assistance.

### Global Range Batch

LayerNorm range constraints (`σ` and `y` residuals) for **all four** witnesses in a block share a single multiplicity-table commitment `m_com` against the identity table `T[i] = i` of size 65 536. This saves 3 × √(2^16) MSMs per block compared with one `m_com` per witness.

## Rust Prover Components

### Hyrax PCS (`pcs/`)

Transparent (no trusted setup), based on discrete-log hardness of BN254 G1. For a `2^n`-evaluation polynomial arranged as a `2^ν × 2^σ` matrix:

- **Commit:** one MSM per row → `2^ν` G1 points.
- **Open at r = (r_L ‖ r_R):** prover sends row-collapsed vector `w′`; verifier checks the MSM equation and inner product.
- **Batch (same point):** `hyrax_open_batch` / `hyrax_verify_batch` combines `K` openings into 2 MSMs.
- **Batch (different points):** `HyraxBatchAccumulator` defers MSMs to a single `finalize` call.

### Sumcheck (`subprotocols/sumcheck.rs`)

Proves `H = Σ_{x ∈ {0,1}^n} f(x)·g(x)`. Each round polynomial has degree ≤ 2, represented by evaluations at `{0, 1, 2}`. Cubic and multi-batched variants (`SumcheckCubicProof`, `SumcheckProofMulti`, `SumcheckCubicProofMulti`) extend the protocol to degree 3 and to Fiat-Shamir-random linear combinations of multiple `(f_i, g_i)` pairs — used for cross-block batching and LayerNorm variance/Y fusion.

### GKR Combine (`subprotocols/combine.rs`)

`prove_combine` / `verify_combine_deferred` reduce multiple evaluation claims on the same committed polynomial to a single Hyrax opening.

### Lasso (`lookup/lasso.rs`)

`prove_lasso_multi` / `verify_lasso_multi` handle a vector of `LassoInstance` objects (each with its own sub-tables and query set) in a single combined sumcheck plus one Hyrax opening, via Fiat-Shamir random linear combination across instances.

### Range Proof (`lookup/range.rs`)

Proves field elements lie in `[0, 2^32)` by splitting into two 16-bit chunks, committing via Hyrax, and running a LogUp-style multiplicity check against the identity table `T[i] = i` of size 65 536. The multiplicity commitment is shared across a batch of witnesses (the four LayerNorm witnesses in a block; or the two final-LN witnesses).

### Cross-Layer Projection (`cross_layer/projection.rs`)

A stand-alone cubic sumcheck that proves `Y_l = α_l · X_l · W_l + bias_l` for **all** L layers in one go (`log L + log d_in` rounds total). Currently exercised by its own unit tests; it is a candidate for replacing the per-layer projection sumchecks once stitching with the rest of the model is complete.

## Cryptographic Parameters

| Parameter | Value |
|-----------|-------|
| Scalar field | BN254 Fr (~254-bit prime) |
| Hash / transcript | SHA3-256 (Fiat-Shamir) |
| PCS | Hyrax (transparent; DL hardness on BN254 G1) |
| Lookup argument | Lasso (batched multi-instance) |
| Range proof | Chunked Lasso (16-bit chunks, identity table, shared `m_com`) |
| Non-interactive | Fiat-Shamir in the random-oracle model |

## Roadmap

- [ ] Multi-head attention support in the Rust prover
- [ ] Wire the cross-layer projection sumcheck into `prove`/`verify`
- [ ] Wire the in-circuit ternary-weight check into setup-time preprocessing
- [ ] Prove the linear-attention normalizer `Z` (or remove it from the Python forward pass)
- [ ] Causal / autoregressive masking in linear attention
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
