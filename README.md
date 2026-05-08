# π-Former

> **Succinct ZK proofs of transformer inference via structured lookup attention.**

π-Former is a research prototype for proving transformer inference with a succinct argument. It replaces softmax attention and dense activations with ZK-friendly primitives — linear attention with a learnable kernel, structured lookup activations, and ternary projection weights — and proves the resulting computation with Hyrax PCS, sumcheck IOPs, and Lasso-style lookup arguments over BN254.

The current prover uses a model-level batching architecture: block-local LayerNorm/range checks are followed by cross-block batched sumchecks, global Q/K and FFN activation Lasso proofs with committed-output binding, and batched Hyrax openings for repeated matrix claims.

See [DESIGN.md](DESIGN.md) for the full technical treatment.

## Approach

| Cost in standard transformers | π-Former replacement |
|-------------------------------|----------------------|
| Softmax (`exp`, normalization, division) | Linear attention `φ(Q)(φ(K)ᵀV)` |
| Dense activations (GELU/SiLU lookups) | Additively decomposed lookup `φ`, native to Lasso |
| LayerNorm (sqrt, division) | Sumcheck for mean/variance + chunked-Lasso range proof |
| Dense weight matrices | Ternary `{−1, 0, +1}` weights with one learnable scale `α` per layer |

## Quickstart

End-to-end smoke test on a synthetic model:

```bash
cd prover
cargo run --release --bin piformer -- sample --output-dir /tmp/piformer_sample
```

Full workflow:

```bash
piformer setup  --weights model.json --seq-len 32 --pk model.pk --vk model.vk
piformer prove  --pk model.pk --witness witness.json --proof proof.bin
piformer verify --vk model.vk --proof proof.bin \
                --public-input public_input.json --public-output public_output.json
```

Optional Python training pipeline:

```bash
cd python && pip install -r requirements.txt && python train_demo.py
```

The current end-to-end proof path uses no-saturation centered lookup
quantization: Python computes `index = round(raw / S) + zero_point` and rejects
out-of-range activations, while Rust verifies that relation against committed
Q/K/M tensors before the Lasso lookup. The benchmark and demo use zero
dense/lookup layers for fast proof timing and CLI smoke tests.

Tests and benchmarks:

```bash
cd prover && cargo test
python benchmark.py --all
```

## Repository Layout

```
python/    PyTorch training pipeline + JSON exporter
prover/    Rust SNARK prover/verifier (library + `piformer` CLI)
paper/     Reference papers (zkGPT, zkLLM)
DESIGN.md  Technical specification
```

## File Formats

| Extension | Contents |
|-----------|----------|
| `*.json`  | Weights or witness (field elements as hex strings) |
| `*.pk`    | Proving key (Hyrax commitments + raw ternary weights) |
| `*.vk`    | Verifying key (Hyrax commitments only) |
| `*.bin`   | Proof bundle (binary, magic-prefixed, versioned; lookup outputs live inside committed Lasso proofs) |

## Status

Research prototype. The Rust prover is currently single-head (`n_heads = 1`); training and export refuse other configurations. The proof is not fully zero-knowledge yet: public commitments and some intermediate evaluations are revealed, while lookup quantization is now proved with committed remainders/range proofs instead of public raw Q/K/M vectors. The cross-layer projection and stand-alone ternary-weight check exist as building blocks but are not yet wired into the end-to-end prover.

## License

Apache 2.0
