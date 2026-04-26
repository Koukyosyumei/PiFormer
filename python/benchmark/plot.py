"""
Plot benchmark_results.json: training curves and inference-time scaling.

    python -m benchmark.plot benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", help="JSON written by run_compare.py")
    p.add_argument("--out_prefix", default="benchmark")
    args = p.parse_args()

    import matplotlib.pyplot as plt  # imported here so the trainer has no dep

    data = json.loads(Path(args.results).read_text())

    # Loss curves -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in data["models"].items():
        h = r["history"]
        ax.plot(h["step"], h["val_loss"], label=f"{name} (val)")
        ax.plot(h["step"], h["train_loss"], "--", alpha=0.5, label=f"{name} (train)")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    out = f"{args.out_prefix}_loss.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # Inference scaling -----------------------------------------------------
    if data.get("inference"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for name, rows in data["inference"].items():
            xs = [r["seq_len"] for r in rows if r.get("ms_per_iter") is not None]
            ys = [r["ms_per_iter"] for r in rows if r.get("ms_per_iter") is not None]
            ms = [r["peak_mem_mb"] for r in rows if r.get("peak_mem_mb") is not None]
            ax1.plot(xs, ys, "o-", label=name)
            ax2.plot(xs, ms, "o-", label=name)
        for ax, ylabel, title in [
            (ax1, "ms / forward pass", "Inference latency"),
            (ax2, "peak GPU mem (MB)", "Inference memory"),
        ]:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("sequence length")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(alpha=0.3, which="both")
            ax.legend()
        out = f"{args.out_prefix}_inference.png"
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
