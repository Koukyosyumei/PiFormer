# Quality Benchmark: PiFormer vs. Baseline Transformer

This benchmark trains PiFormer and a standard pre-norm transformer side-by-side on the same dataset with matched hyperparameters, then reports:

- Final validation loss / perplexity
- Parameter count
- Training throughput (tokens/sec) and peak GPU memory
- Inference-time / memory scaling across sequence lengths

The baseline differs from PiFormer in exactly three components — softmax attention, GELU FFN, dense `nn.Linear` — so the gap measures the cost of going ZK-friendly.

> **Scope.** Both models are non-causal (PiFormer's `LinearAttentionLayer` has no causal mask), so this is an architecture comparison, not a state-of-the-art LM benchmark.

## Google Colab

Open a fresh Colab notebook with a GPU runtime (T4 is fine), then run the cells below.

```python
# Cell 1 — clone & install
!git clone https://github.com/<your-fork>/PiFormer.git
%cd PiFormer/python
!pip install -q -r requirements.txt
```

```python
# Cell 2 — run the comparison (~5–15 min on T4 at default settings)
!python -m benchmark.run_compare \
    --dataset tinyshakespeare \
    --d_model 128 --n_heads 4 --n_layers 4 --d_ff 512 \
    --seq_len 128 --batch_size 64 --steps 3000 \
    --inference_seq_lens 64,128,256,512,1024,2048 \
    --out benchmark_results.json
```

```python
# Cell 3 — plots
!pip install -q matplotlib
!python -m benchmark.plot benchmark_results.json
from IPython.display import Image
display(Image("benchmark_loss.png"))
display(Image("benchmark_inference.png"))
```

## Local

```bash
cd python
python -m benchmark.run_compare --device cuda --steps 3000
python -m benchmark.plot benchmark_results.json
```

## Useful flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | `tinyshakespeare` | Also supports `wikitext2` (needs `pip install datasets`). |
| `--models` | `baseline,piformer` | Comma-separated subset to run. |
| `--steps` | `3000` | Training steps (per model). |
| `--seq_len` / `--batch_size` | `128` / `64` | Training shape. |
| `--d_model` / `--n_heads` / `--n_layers` / `--d_ff` | `128 / 4 / 4 / 512` | Model size. Both models share these. |
| `--num_bits` / `--c` / `--scale` / `--max_exp` | `8 / 2 / 0.1 / 4` | PiFormer quantization knobs (ignored by baseline). |
| `--inference_seq_lens` | `64,128,256,512,1024` | Lengths swept for forward-pass timing. |
| `--skip_inference` | off | Skip the scaling sweep. |

## What to look for

- **Quality gap.** PiFormer's val perplexity should track the baseline closely at this scale; a large gap suggests the quantization knobs (`num_bits`, `max_exp`, `scale`) need tuning.
- **Throughput at long T.** Linear attention's O(T·d²) compute beats softmax's O(T²·d) once `T > d`. Expect the gap to widen on the inference-scaling plot beyond `T ≈ d_model`.
- **Peak memory.** Linear attention stores the d×d context matrix, not the T×T attention matrix — memory should grow linearly in T rather than quadratically.

## Files

- `baseline.py` — `BaselineTransformer` (softmax + GELU + dense `nn.Linear`).
- `data.py` — Dataset loaders (TinyShakespeare auto-downloaded, optional WikiText-2).
- `run_compare.py` — Trainer + inference-scaling driver. Writes a JSON report.
- `plot.py` — Renders training curves and inference-scaling plots from the JSON.
