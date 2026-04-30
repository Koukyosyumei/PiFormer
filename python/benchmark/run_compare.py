"""
Train PiFormer and a vanilla baseline transformer side-by-side on the same
dataset with the same hyperparameters. Reports validation loss / perplexity,
parameter count, training throughput, peak GPU memory, and inference-time
scaling vs. sequence length.

Usage (Colab GPU)
-----------------
    !pip install torch
    !git clone https://github.com/<you>/PiFormer.git
    %cd PiFormer/python
    !python -m benchmark.run_compare --steps 3000 --device cuda

Useful flags
------------
    --dataset {tinyshakespeare, wikitext2}   default: tinyshakespeare
    --d_model 128 --n_heads 4 --n_layers 4 --d_ff 512
    --seq_len 128 --batch_size 64 --steps 3000 --lr 3e-4
    --inference_seq_lens 64,128,256,512,1024
    --models piformer,baseline               choose subset to run
    --out results.json
    --no_compile                             skip torch.compile
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from piformer.model import PiFormerModel
from piformer.projection import TernaryLinear

from .baseline import BaselineTransformer
from .data import get_dataset, random_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def eval_loss(model, val_ids, batch_size, seq_len, device, n_batches: int = 20):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = random_batch(val_ids, batch_size, seq_len, device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        total += loss.item()
    model.train()
    return total / n_batches


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def maybe_compile_model(name: str, model: nn.Module, args, device: torch.device) -> nn.Module:
    """Compile models for long CUDA runs, where PiFormer's small kernels benefit most."""
    if args.no_compile:
        return model
    if device.type != "cuda":
        print(f"  torch.compile: skipped for {name} (CUDA device required)")
        return model
    if not hasattr(torch, "compile"):
        print(f"  torch.compile: skipped for {name} (not available in this PyTorch)")
        return model

    print(f"  torch.compile: compiling {name} (mode={args.compile_mode})")
    return torch.compile(model, mode=args.compile_mode)


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def copy_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in unwrap_compiled_model(model).state_dict().items()
    }


@torch.no_grad()
def init_ternary_from_dense(dst: TernaryLinear, weight: torch.Tensor, bias: torch.Tensor | None):
    """Initialize a ternary layer from a dense layer using one scalar alpha."""
    weight = weight.to(device=dst.weight.device, dtype=dst.weight.dtype)
    abs_w = weight.abs()
    selected = abs_w > (0.7 * abs_w.mean())
    if selected.any():
        alpha = abs_w[selected].mean()
    else:
        alpha = abs_w.mean().clamp_min(1e-6)

    dst.weight.copy_(weight)
    dst.alpha.copy_(alpha.to(device=dst.alpha.device, dtype=dst.alpha.dtype))
    if dst.bias is not None and bias is not None:
        dst.bias.copy_(bias.to(device=dst.bias.device, dtype=dst.bias.dtype))


@torch.no_grad()
def init_piformer_from_baseline_state(model: PiFormerModel, state: dict[str, torch.Tensor]):
    """Warm-start PiFormer from a trained baseline where the architectures overlap."""
    model.embedding.weight.copy_(state["embedding.weight"].to(model.embedding.weight.device))
    model.pos_embedding.weight.copy_(
        state["pos_embedding.weight"].to(model.pos_embedding.weight.device)
    )
    model.norm.weight.copy_(state["norm.weight"].to(model.norm.weight.device))
    model.norm.bias.copy_(state["norm.bias"].to(model.norm.bias.device))
    init_ternary_from_dense(model.head, state["head.weight"], None)

    for i, block in enumerate(model.blocks):
        block.norm1.weight.copy_(state[f"blocks.{i}.norm1.weight"].to(block.norm1.weight.device))
        block.norm1.bias.copy_(state[f"blocks.{i}.norm1.bias"].to(block.norm1.bias.device))
        block.norm2.weight.copy_(state[f"blocks.{i}.norm2.weight"].to(block.norm2.weight.device))
        block.norm2.bias.copy_(state[f"blocks.{i}.norm2.bias"].to(block.norm2.bias.device))

        init_ternary_from_dense(
            block.attn.q_proj,
            state[f"blocks.{i}.attn.q_proj.weight"],
            None,
        )
        init_ternary_from_dense(
            block.attn.k_proj,
            state[f"blocks.{i}.attn.k_proj.weight"],
            None,
        )
        init_ternary_from_dense(
            block.attn.v_proj,
            state[f"blocks.{i}.attn.v_proj.weight"],
            None,
        )
        init_ternary_from_dense(
            block.attn.out_proj,
            state[f"blocks.{i}.attn.out_proj.weight"],
            state[f"blocks.{i}.attn.out_proj.bias"],
        )
        init_ternary_from_dense(
            block.ffn.fc1,
            state[f"blocks.{i}.ffn.fc1.weight"],
            state[f"blocks.{i}.ffn.fc1.bias"],
        )
        init_ternary_from_dense(
            block.ffn.fc2,
            state[f"blocks.{i}.ffn.fc2.weight"],
            state[f"blocks.{i}.ffn.fc2.bias"],
        )


def make_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """AdamW with decay only on ordinary matrix weights."""
    decay_params = []
    no_decay_params = []

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if (
                param_name.endswith("bias")
                or param_name == "alpha"
                or "tables" in full_name
                or isinstance(module, (nn.LayerNorm, nn.Embedding))
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
    )


def learning_rate_for_step(step: int, steps: int, args) -> float:
    if args.warmup_steps > 0 and step <= args.warmup_steps:
        return args.lr * step / args.warmup_steps
    if args.min_lr_ratio >= 1.0 or steps <= args.warmup_steps:
        return args.lr

    progress = (step - args.warmup_steps) / max(1, steps - args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    return args.lr * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)


def train_one(
    name: str,
    model: nn.Module,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    args,
    device: torch.device,
    steps: int,
):
    print(f"\n=== Training {name} ===")
    print(f"  params: {count_params(model):,}")
    print(f"  steps:  {steps}")

    optim = make_optimizer(model, args)
    history = {"step": [], "train_loss": [], "val_loss": []}

    reset_peak_memory(device)
    model.train()

    # Warmup so we don't time autotuning.
    for _ in range(5):
        x, y = random_batch(train_ids, args.batch_size, args.seq_len, device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        loss.backward()
        optim.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    running = 0.0
    log_start = 1
    for step in range(1, steps + 1):
        lr = learning_rate_for_step(step, steps, args)
        for group in optim.param_groups:
            group["lr"] = lr

        x, y = random_batch(train_ids, args.batch_size, args.seq_len, device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} produced non-finite loss at step {step}: {loss}")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        running += loss.item()

        if step % args.log_every == 0 or step == steps:
            logged_steps = step - log_start + 1
            avg = running / logged_steps
            log_start = step + 1
            running = 0.0
            val = eval_loss(model, val_ids, args.batch_size, args.seq_len, device)
            history["step"].append(step)
            history["train_loss"].append(avg)
            history["val_loss"].append(val)
            print(f"  step {step:5d}  train={avg:.4f}  val={val:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    final_val = eval_loss(model, val_ids, args.batch_size, args.seq_len, device, n_batches=50)
    return {
        "name": name,
        "params": count_params(model),
        "steps": steps,
        "train_seconds": elapsed,
        "steps_per_sec": steps / elapsed,
        "tokens_per_sec": steps * args.batch_size * args.seq_len / elapsed,
        "final_val_loss": final_val,
        "final_val_ppl": float(torch.exp(torch.tensor(final_val))),
        "peak_mem_mb": peak_memory_mb(device),
        "history": history,
    }


@torch.no_grad()
def inference_scaling(
    name: str,
    model: nn.Module,
    seq_lens,
    vocab_size: int,
    batch_size: int,
    device: torch.device,
):
    """Forward-pass latency and peak memory for a sweep of sequence lengths."""
    print(f"\n=== Inference scaling: {name} ===")
    model.eval()
    rows = []
    for T in seq_lens:
        try:
            reset_peak_memory(device)
            x = torch.randint(0, vocab_size, (batch_size, T), device=device)
            for _ in range(3):  # warmup
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = 10
            for _ in range(iters):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / iters * 1000
            mem = peak_memory_mb(device)
            print(f"  T={T:5d}  {ms:8.2f} ms/iter   peak_mem={mem:.1f} MB")
            rows.append({"seq_len": T, "ms_per_iter": ms, "peak_mem_mb": mem})
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  T={T:5d}  OOM")
                rows.append({"seq_len": T, "ms_per_iter": None, "peak_mem_mb": None, "oom": True})
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="tinyshakespeare",
                   choices=["tinyshakespeare", "wikitext2"])
    p.add_argument("--cache_dir", default="./.benchmark_cache")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)

    # PiFormer-only knobs
    p.add_argument("--num_bits", type=int, default=8)
    p.add_argument("--c", type=int, default=2)
    p.add_argument("--scale", type=float, default=0.1)
    p.add_argument("--max_exp", type=int, default=4)

    # Training
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000,
                   help="Default training steps. Overridden per-model by "
                        "--baseline_steps / --piformer_steps when set.")
    p.add_argument("--baseline_steps", type=int, default=None,
                   help="Training steps for the baseline transformer "
                        "(falls back to --steps).")
    p.add_argument("--piformer_steps", type=int, default=None,
                   help="Training steps for PiFormer (falls back to --steps).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="AdamW weight decay for ordinary matrix weights only.")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--warmup_steps", type=int, default=100,
                   help="Linear LR warmup steps; set 0 to disable.")
    p.add_argument("--min_lr_ratio", type=float, default=0.1,
                   help="Final LR as a fraction of --lr for cosine decay; set 1 for constant LR.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Clip global gradient norm during training; set <=0 to disable.")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_compile", action="store_true",
                   help="Skip torch.compile. By default, CUDA runs compile each model.")
    p.add_argument("--compile_mode", default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode used when compilation is enabled.")
    p.add_argument("--piformer_init_from_baseline", action="store_true",
                   help="Warm-start PiFormer from the trained baseline run.")

    # Models to run
    p.add_argument("--models", default="baseline,piformer",
                   help="Comma-separated subset of {baseline,piformer}.")

    # Inference scaling
    p.add_argument("--inference_seq_lens", default="64,128,256,512,1024",
                   help="Comma-separated sequence lengths for forward-pass timing.")
    p.add_argument("--inference_batch_size", type=int, default=8)
    p.add_argument("--skip_inference", action="store_true")

    p.add_argument("--out", default="benchmark_results.json")
    p.add_argument("--loss_plot", default=None,
                   help="Path for ggplot loss-curve PNG. Defaults to <out>_loss.png. "
                        "Pass an empty string to skip plotting.")
    return p.parse_args()


def plot_loss_curves(results: dict, out_path: Path) -> None:
    """Plot train/val loss vs. step for each trained model using plotnine (ggplot)."""
    import pandas as pd
    from plotnine import (
        ggplot, aes, geom_line, labs, theme_bw, scale_linetype_manual,
    )

    rows = []
    for name, r in results["models"].items():
        h = r["history"]
        for step, train_loss, val_loss in zip(h["step"], h["train_loss"], h["val_loss"]):
            rows.append({"model": name, "step": step, "split": "train", "loss": train_loss})
            rows.append({"model": name, "step": step, "split": "val", "loss": val_loss})

    if not rows:
        print("No training history to plot.")
        return

    df = pd.DataFrame(rows)
    plot = (
        ggplot(df, aes(x="step", y="loss", color="model", linetype="split"))
        + geom_line(size=0.8)
        + scale_linetype_manual(values={"train": "dashed", "val": "solid"})
        + labs(
            title="Training curves: baseline vs. PiFormer",
            x="step",
            y="cross-entropy loss",
            color="model",
            linetype="split",
        )
        + theme_bw()
    )
    plot.save(str(out_path), width=7, height=4, dpi=140, verbose=False)
    print(f"wrote {out_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    cache_dir = Path(args.cache_dir)
    train_ids, val_ids, vocab_size, _decode = get_dataset(args.dataset, cache_dir)
    print(f"Dataset: {args.dataset}   vocab={vocab_size}   "
          f"train_tokens={len(train_ids):,}   val_tokens={len(val_ids):,}")

    common = dict(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=max(args.seq_len,
                        max(int(s) for s in args.inference_seq_lens.split(","))),
    )
    piformer_kwargs = dict(
        num_bits=args.num_bits, c=args.c, scale=args.scale, max_exp=args.max_exp
    )

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {"args": vars(args), "models": {}, "inference": {}}
    baseline_state_for_init = None

    if args.piformer_init_from_baseline and "baseline" not in selected:
        raise ValueError("--piformer_init_from_baseline requires --models to include baseline")

    baseline_steps = args.baseline_steps if args.baseline_steps is not None else args.steps
    piformer_steps = args.piformer_steps if args.piformer_steps is not None else args.steps

    if "baseline" in selected:
        torch.manual_seed(args.seed)
        model = BaselineTransformer(**common).to(device)
        model = maybe_compile_model("baseline", model, args, device)
        results["models"]["baseline"] = train_one(
            "baseline", model, train_ids, val_ids, args, device, baseline_steps
        )
        if not args.skip_inference:
            seq_lens = [int(s) for s in args.inference_seq_lens.split(",")]
            results["inference"]["baseline"] = inference_scaling(
                "baseline", model, seq_lens, vocab_size,
                args.inference_batch_size, device,
            )
        if args.piformer_init_from_baseline:
            print("  saving trained baseline weights for PiFormer initialization")
            baseline_state_for_init = copy_state_to_cpu(model)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "piformer" in selected:
        torch.manual_seed(args.seed)
        model = PiFormerModel(**common, **piformer_kwargs).to(device)
        if args.piformer_init_from_baseline:
            print("  initializing PiFormer from trained baseline weights")
            init_piformer_from_baseline_state(model, baseline_state_for_init)
        model = maybe_compile_model("piformer", model, args, device)
        results["models"]["piformer"] = train_one(
            "piformer", model, train_ids, val_ids, args, device, piformer_steps
        )
        if not args.skip_inference:
            seq_lens = [int(s) for s in args.inference_seq_lens.split(",")]
            results["inference"]["piformer"] = inference_scaling(
                "piformer", model, seq_lens, vocab_size,
                args.inference_batch_size, device,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'model':<10} {'params':>10} {'val_loss':>10} {'val_ppl':>10} "
          f"{'tok/s':>10} {'peak_MB':>10}")
    for name, r in results["models"].items():
        print(f"{name:<10} {r['params']:>10,} {r['final_val_loss']:>10.4f} "
              f"{r['final_val_ppl']:>10.2f} {r['tokens_per_sec']:>10.0f} "
              f"{r['peak_mem_mb']:>10.1f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")

    if args.loss_plot != "":
        loss_plot_path = Path(
            args.loss_plot if args.loss_plot is not None
            else Path(args.out).with_suffix("").as_posix() + "_loss.png"
        )
        try:
            plot_loss_curves(results, loss_plot_path)
        except ImportError as e:
            print(f"Skipping loss plot — install plotnine to enable: {e}")


if __name__ == "__main__":
    main()
