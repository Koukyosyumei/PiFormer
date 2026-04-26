# π-Former

> **Succinct ZK proofs of transformer inference via structured lookup attention.**

π-Former is a research prototype for proving transformer inference in a zero-knowledge SNARK. It replaces softmax attention and dense activations with ZK-friendly primitives — linear attention with a learnable kernel, structured lookup activations, and ternary projection weights — and proves the resulting computation with Hyrax PCS, the Spartan sumcheck IOP, and a Lasso-based lookup argument over BN254.

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
ref/       Reference implementations
DESIGN.md  Technical specification
```

## File Formats

| Extension | Contents |
|-----------|----------|
| `*.json`  | Weights or witness (field elements as hex strings) |
| `*.pk`    | Proving key (Hyrax commitments + raw ternary weights) |
| `*.vk`    | Verifying key (Hyrax commitments only) |
| `*.bin`   | Proof bundle (binary, magic-prefixed, versioned) |

## Status

Research prototype. The Rust prover is currently single-head (`n_heads = 1`); training and export refuse other configurations. The cross-layer projection and stand-alone ternary-weight check exist as building blocks but are not yet wired into the end-to-end prover.

## References

- Setty, Thaler, Wahby. *Unlocking the Lookup Singularity with Lasso.* EUROCRYPT 2024.
- Setty. *Spartan: Efficient and general-purpose zkSNARKs without trusted setup.* CRYPTO 2020.
- Wahby et al. *Doubly-Efficient zkSNARKs Without Trusted Setup.* IEEE S&P 2018.
- Arun et al. *Jolt: SNARKs for Virtual Machines via Lookups.* EUROCRYPT 2024.
- Katharopoulos et al. *Transformers are RNNs.* ICML 2020.
- Wang et al. *BitNet: Scaling 1-bit Transformers for Large Language Models.* 2023.
- zkGPT, zkLLM — see `paper/`.

## License

MIT
