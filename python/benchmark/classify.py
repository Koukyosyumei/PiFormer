"""
Bidirectional benchmark: text-classification-flavored task that fits the
non-causal linear-attention math (PiFormer's prover-compatible mode).

Task — bucket-majority classification
-------------------------------------
Random sequences over a vocabulary of size V; labels are the index (in
0..num_classes) of the V/num_classes-sized vocab bucket whose tokens occur
most often in the sequence. The label depends on the entire sequence, so
the model must aggregate over the full sequence — exactly the regime where
non-causal attention is the natural fit and what the current PiFormer
prover circuit (sum over all t) is designed for.

Usage
-----
    python -m benchmark.classify --steps 1500
    python -m benchmark.classify --models piformer --steps 1500 --device cpu

Both models run with causal=False so the comparison is apples-to-apples and
PiFormer is in its prover-compatible math.
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

from .baseline import BaselineTransformer
from .data import load_trec
from .run_compare import (
    count_params,
    maybe_compile_model,
    peak_memory_mb,
    reset_peak_memory,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_bucket_majority_dataset(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    num_classes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vocab_size % num_classes != 0:
        raise ValueError("vocab_size must be divisible by num_classes")
    bucket = vocab_size // num_classes
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)
    bucket_id = x // bucket
    counts = torch.zeros(n_samples, num_classes, dtype=torch.long)
    counts.scatter_add_(1, bucket_id, torch.ones_like(bucket_id))
    y = counts.argmax(dim=1)
    return x, y


def iter_minibatches(x, y, batch_size, device, shuffle=True, seed=None):
    n = x.size(0)
    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        order = torch.randperm(n, generator=g)
    else:
        order = torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield x[idx].to(device), y[idx].to(device)


# ---------------------------------------------------------------------------
# Classifier wrapper — reuses the encoder stack of either model
# ---------------------------------------------------------------------------

class PooledClassifier(nn.Module):
    """Mean-pool the encoder's pre-head hidden state, then linear classify.

    Works for both PiFormerModel and BaselineTransformer because they share
    the same encoder structure (embedding + pos_embedding + blocks + norm).

    If pad_id is set, the mean is taken only over non-pad positions so short
    sequences aren't diluted by padding tokens.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_model: int,
        num_classes: int,
        pad_id: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, num_classes)
        self.pad_id = pad_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        m = self.encoder
        B, T = input_ids.shape
        pos = m.position_ids[:, :T]
        x = m.embedding(input_ids) + m.pos_embedding(pos)
        for block in m.blocks:
            x = block(x)
        x = m.norm(x)
        if self.pad_id is None:
            pooled = x.mean(dim=1)
        else:
            mask = (input_ids != self.pad_id).to(x.dtype).unsqueeze(-1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_accuracy(model, x, y, batch_size, device) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for xb, yb in iter_minibatches(x, y, batch_size, device, shuffle=False):
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb, reduction="sum")
        loss_sum += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    model.train()
    return correct / total, loss_sum / total


def train_one(
    name: str,
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    args,
    device: torch.device,
):
    print(f"\n=== Training {name} ===")
    print(f"  params: {count_params(model):,}")
    print(f"  steps:  {args.steps}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = {"step": [], "train_loss": [], "val_loss": [], "val_acc": []}

    reset_peak_memory(device)
    model.train()

    train_iter = iter_minibatches(train_x, train_y, args.batch_size, device, seed=args.seed)
    t0 = time.perf_counter()
    running = 0.0
    log_start = 1
    for step in range(1, args.steps + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter_minibatches(
                train_x, train_y, args.batch_size, device, seed=args.seed + step
            )
            xb, yb = next(train_iter)

        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        running += loss.item()

        if step % args.log_every == 0 or step == args.steps:
            avg = running / (step - log_start + 1)
            log_start = step + 1
            running = 0.0
            val_acc, val_loss = eval_accuracy(model, val_x, val_y, args.batch_size, device)
            history["step"].append(step)
            history["train_loss"].append(avg)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(
                f"  step {step:5d}  train={avg:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc * 100:.2f}%"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    final_acc, final_loss = eval_accuracy(model, val_x, val_y, args.batch_size, device)
    return {
        "name": name,
        "params": count_params(model),
        "train_seconds": elapsed,
        "steps_per_sec": args.steps / elapsed,
        "final_val_loss": final_loss,
        "final_val_acc": final_acc,
        "peak_mem_mb": peak_memory_mb(device),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_accuracy_curves(results: dict, out_path: Path) -> None:
    """Plot val accuracy vs. step for each trained model using plotnine."""
    import pandas as pd
    from plotnine import (
        ggplot, aes, geom_line, geom_hline, labs, theme_bw,
        scale_y_continuous,
    )

    rows = []
    for name, r in results["models"].items():
        h = r["history"]
        for step, acc in zip(h["step"], h["val_acc"]):
            rows.append({"model": name, "step": step, "val_acc": acc * 100.0})

    if not rows:
        print("No training history to plot.")
        return

    df = pd.DataFrame(rows)
    base_rate_pct = results.get("base_rate", 0.0) * 100.0
    plot = (
        ggplot(df, aes(x="step", y="val_acc", color="model"))
        + geom_line(size=0.8)
        + geom_hline(yintercept=base_rate_pct, linetype="dashed", color="grey")
        + scale_y_continuous(limits=[0, 100])
        + labs(
            title="Validation accuracy vs. step",
            x="step",
            y="val accuracy (%)",
            color="model",
        )
        + theme_bw()
    )
    plot.save(str(out_path), width=7, height=4, dpi=140, verbose=False)
    print(f"wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "trec"],
                   help="synthetic: bucket-majority (no download). "
                        "trec: TREC question classification, 6 coarse classes.")
    p.add_argument("--cache_dir", default="./.benchmark_cache")

    # Task (synthetic only — for trec these come from the dataset)
    p.add_argument("--vocab_size", type=int, default=16)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--n_train", type=int, default=20_000)
    p.add_argument("--n_val", type=int, default=2_000)

    # Model
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=128)

    # PiFormer-only knobs
    p.add_argument("--num_bits", type=int, default=8)
    p.add_argument("--c", type=int, default=2)
    p.add_argument("--scale", type=float, default=0.1)
    p.add_argument("--max_exp", type=int, default=4)

    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--compile_mode", default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"])

    p.add_argument("--models", default="baseline,piformer",
                   help="Comma-separated subset of {baseline,piformer}.")
    p.add_argument("--out", default="classify_results.json")
    p.add_argument("--acc_plot", default=None,
                   help="Path for ggplot val-accuracy PNG. Defaults to <out>_acc.png. "
                        "Pass an empty string to skip plotting.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    if args.dataset == "synthetic":
        print(
            f"Task: bucket-majority over V={args.vocab_size}, "
            f"K={args.num_classes}, T={args.seq_len}"
        )
        train_x, train_y = make_bucket_majority_dataset(
            args.n_train, args.seq_len, args.vocab_size, args.num_classes, args.seed
        )
        val_x, val_y = make_bucket_majority_dataset(
            args.n_val, args.seq_len, args.vocab_size, args.num_classes, args.seed + 1
        )
        vocab_size = args.vocab_size
        num_classes = args.num_classes
        seq_len = args.seq_len
        pad_id = None
    else:  # trec
        print(f"Task: TREC question classification (coarse, 6 classes), char-level")
        train_x, train_y, val_x, val_y, vocab_size, num_classes, pad_id = load_trec(
            Path(args.cache_dir), seq_len=args.seq_len
        )
        seq_len = args.seq_len
        print(f"  vocab_size={vocab_size}  pad_id={pad_id}  seq_len={seq_len}")

    base_rate = (val_y.bincount(minlength=num_classes).max().item() / val_y.numel())
    print(f"Train: {len(train_x):,}   Val: {len(val_x):,}   "
          f"majority-class baseline acc: {base_rate * 100:.2f}%")

    common = dict(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=seq_len,
    )
    piformer_kwargs = dict(
        num_bits=args.num_bits, c=args.c, scale=args.scale, max_exp=args.max_exp,
        causal=False,  # bidirectional task; matches the prover circuit
    )

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {"args": vars(args), "models": {}, "base_rate": base_rate}

    if "baseline" in selected:
        torch.manual_seed(args.seed)
        encoder = BaselineTransformer(**common, causal=False)
        model = PooledClassifier(encoder, args.d_model, num_classes, pad_id=pad_id).to(device)
        model = maybe_compile_model("baseline", model, args, device)
        results["models"]["baseline"] = train_one(
            "baseline", model, train_x, train_y, val_x, val_y, args, device
        )
        del model, encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "piformer" in selected:
        torch.manual_seed(args.seed)
        encoder = PiFormerModel(**common, **piformer_kwargs)
        model = PooledClassifier(encoder, args.d_model, num_classes, pad_id=pad_id).to(device)
        model = maybe_compile_model("piformer", model, args, device)
        results["models"]["piformer"] = train_one(
            "piformer", model, train_x, train_y, val_x, val_y, args, device
        )
        del model, encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    task_label = "bucket-majority" if args.dataset == "synthetic" else "TREC (coarse)"
    print(f"SUMMARY  ({task_label} classification, causal=False)")
    print("=" * 60)
    print(f"{'model':<10} {'params':>10} {'val_loss':>10} {'val_acc':>10} "
          f"{'steps/s':>10} {'peak_MB':>10}")
    for name, r in results["models"].items():
        print(f"{name:<10} {r['params']:>10,} {r['final_val_loss']:>10.4f} "
              f"{r['final_val_acc'] * 100:>9.2f}% {r['steps_per_sec']:>10.1f} "
              f"{r['peak_mem_mb']:>10.1f}")
    print(f"{'baseline_majority':<10} {'-':>10} {'-':>10} "
          f"{base_rate * 100:>9.2f}% {'-':>10} {'-':>10}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")

    if args.acc_plot != "":
        acc_plot_path = Path(
            args.acc_plot if args.acc_plot is not None
            else Path(args.out).with_suffix("").as_posix() + "_acc.png"
        )
        try:
            plot_accuracy_curves(results, acc_plot_path)
        except ImportError as e:
            print(f"Skipping accuracy plot — install plotnine to enable: {e}")


if __name__ == "__main__":
    main()
