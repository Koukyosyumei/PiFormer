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
import time
from contextlib import nullcontext
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


# ---------------------------------------------------------------------------
# Gradual ternary QAT schedule
# ---------------------------------------------------------------------------
#
# Training a ternary network from scratch tends to underfit because the
# {-1, 0, +1} constraint is too aggressive for the optimizer to find a good
# basin. Instead we ramp every TernaryLinear's `quant_strength` from 0 → 1
# across three phases:
#
#   phase 1: warmup    (q = 0)        full-precision pretrain
#   phase 2: ramp      (q: 0 → 1)     gentle interpolation to ternary
#   phase 3: finetune  (q = 1)        pure ternary, optionally lower LR
#
# Phase boundaries are expressed as fractions of args.steps. The default split
# (0.4 / 0.4 / 0.2) gives the FP weights time to settle, then ramps long
# enough that any single step's quantization shock is small.

def _ternary_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, TernaryLinear)]


def _set_quant_strength(modules, q: float):
    for m in modules:
        m.quant_strength.fill_(q)


def _qat_schedule(step: int, total_steps: int, warmup_frac: float, ramp_frac: float):
    """Return (q, phase) for a given step. Phase ∈ {'warmup','ramp','finetune'}."""
    warmup_end = int(round(total_steps * warmup_frac))
    ramp_end = int(round(total_steps * (warmup_frac + ramp_frac)))
    if step <= warmup_end:
        return 0.0, "warmup"
    if step <= ramp_end:
        span = max(1, ramp_end - warmup_end)
        return (step - warmup_end) / span, "ramp"
    return 1.0, "finetune"


def train_one(
    name: str,
    model: nn.Module,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    args,
    device: torch.device,
):
    print(f"\n=== Training {name} ===")
    print(f"  params: {count_params(model):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = {"step": [], "train_loss": [], "val_loss": [], "quant_strength": []}

    # Identify ternary layers and decide whether to schedule quant_strength.
    tern_modules = _ternary_modules(model)
    use_qat = bool(tern_modules) and args.ternary_qat
    if use_qat:
        print(
            f"  gradual QAT: warmup={args.ternary_warmup_frac:.2f} "
            f"ramp={args.ternary_ramp_frac:.2f} "
            f"finetune_lr_mult={args.ternary_finetune_lr_mult:g} "
            f"({len(tern_modules)} TernaryLinear layers)"
        )
        # Start in the warmup phase (q=0) so weights train at full precision.
        _set_quant_strength(tern_modules, 0.0)
    elif tern_modules:
        # QAT disabled: keep the original behavior (always fully ternary).
        _set_quant_strength(tern_modules, 1.0)

    finetune_lr_applied = False

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
        optim.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    running = 0.0
    cur_q = 1.0
    cur_phase = "finetune"
    for step in range(1, args.steps + 1):
        if use_qat:
            cur_q, new_phase = _qat_schedule(
                step, args.steps,
                args.ternary_warmup_frac, args.ternary_ramp_frac,
            )
            _set_quant_strength(tern_modules, cur_q)
            # Drop LR once when entering the finetune phase, so the now-snapped
            # ternary weights settle without big optimizer steps disturbing them.
            if (
                not finetune_lr_applied
                and new_phase == "finetune"
                and args.ternary_finetune_lr_mult != 1.0
            ):
                for g in optim.param_groups:
                    g["lr"] *= args.ternary_finetune_lr_mult
                finetune_lr_applied = True
                print(f"  [step {step}] entering finetune phase: lr → "
                      f"{optim.param_groups[0]['lr']:.2e}")
            cur_phase = new_phase

        x, y = random_batch(train_ids, args.batch_size, args.seq_len, device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        running += loss.item()

        if step % args.log_every == 0 or step == args.steps:
            avg = running / args.log_every
            running = 0.0
            val = eval_loss(model, val_ids, args.batch_size, args.seq_len, device)
            history["step"].append(step)
            history["train_loss"].append(avg)
            history["val_loss"].append(val)
            history["quant_strength"].append(cur_q)
            qat_tag = f"  q={cur_q:.2f} [{cur_phase}]" if use_qat else ""
            print(f"  step {step:5d}  train={avg:.4f}  val={val:.4f}{qat_tag}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Pin to fully ternary weights for final evaluation and downstream
    # inference benchmarks — that's the model we'll actually ship.
    if tern_modules:
        _set_quant_strength(tern_modules, 1.0)

    final_val = eval_loss(model, val_ids, args.batch_size, args.seq_len, device, n_batches=50)
    return {
        "name": name,
        "params": count_params(model),
        "train_seconds": elapsed,
        "steps_per_sec": args.steps / elapsed,
        "tokens_per_sec": args.steps * args.batch_size * args.seq_len / elapsed,
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
    p.add_argument("--causal", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Use autoregressive attention. Applies a causal mask to "
                        "the baseline and prefix linear attention to PiFormer.")

    # PiFormer-only knobs
    p.add_argument("--num_bits", type=int, default=8)
    p.add_argument("--c", type=int, default=2)
    p.add_argument("--scale", type=float, default=0.1)
    p.add_argument("--max_exp", type=int, default=4)

    # Training
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)

    # Gradual ternary QAT schedule (only affects models with TernaryLinear)
    p.add_argument("--ternary_qat", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Enable warmup → ramp → finetune schedule for ternary weights. "
                        "Use --no-ternary_qat to fall back to ternary-from-scratch.")
    p.add_argument("--ternary_warmup_frac", type=float, default=0.4,
                   help="Fraction of steps with full-precision weights (q=0).")
    p.add_argument("--ternary_ramp_frac", type=float, default=0.4,
                   help="Fraction of steps over which q ramps 0 → 1.")
    p.add_argument("--ternary_finetune_lr_mult", type=float, default=0.1,
                   help="LR multiplier applied once at the start of the q=1 finetune phase.")

    # Models to run
    p.add_argument("--models", default="baseline,piformer",
                   help="Comma-separated subset of {baseline,piformer}.")

    # Inference scaling
    p.add_argument("--inference_seq_lens", default="64,128,256,512,1024",
                   help="Comma-separated sequence lengths for forward-pass timing.")
    p.add_argument("--inference_batch_size", type=int, default=8)
    p.add_argument("--skip_inference", action="store_true")

    p.add_argument("--out", default="benchmark_results.json")
    return p.parse_args()


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
    print(f"Attention mode: {'causal' if args.causal else 'bidirectional'}")

    common = dict(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        causal=args.causal,
        max_seq_len=max(args.seq_len,
                        max(int(s) for s in args.inference_seq_lens.split(","))),
    )
    piformer_kwargs = dict(
        num_bits=args.num_bits, c=args.c, scale=args.scale, max_exp=args.max_exp
    )

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {"args": vars(args), "models": {}, "inference": {}}

    if "baseline" in selected:
        torch.manual_seed(args.seed)
        model = BaselineTransformer(**common).to(device)
        results["models"]["baseline"] = train_one(
            "baseline", model, train_ids, val_ids, args, device
        )
        if not args.skip_inference:
            seq_lens = [int(s) for s in args.inference_seq_lens.split(",")]
            results["inference"]["baseline"] = inference_scaling(
                "baseline", model, seq_lens, vocab_size,
                args.inference_batch_size, device,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "piformer" in selected:
        torch.manual_seed(args.seed)
        model = PiFormerModel(**common, **piformer_kwargs).to(device)
        results["models"]["piformer"] = train_one(
            "piformer", model, train_ids, val_ids, args, device
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


if __name__ == "__main__":
    main()
