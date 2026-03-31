#!/usr/bin/env python3
"""
PiFormer Benchmark
==================
Creates a PiFormer model of a given named size with random weights, exports
the weights/witness, then times setup / prove / verify via the Rust CLI.

Usage
-----
    python benchmark.py [MODEL_NAME] [OPTIONS]

    MODEL_NAME  One of: tiny, small, medium, large, gpt2-small
                Default: tiny

Options
-------
    --seq-len N        Sequence length (default: model-dependent)
    --out-dir DIR      Scratch directory (default: benchmark_out/<model>)
    --no-build         Skip `cargo build` (assumes binary already built)
    --all              Run all predefined model sizes sequentially

Examples
--------
    python benchmark.py tiny
    python benchmark.py gpt2-small --seq-len 32
    python benchmark.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Model configurations
# NOTE: n_heads is fixed to 1 — the Rust prover currently supports single-head
# attention only. Dimensions are chosen to be powers of 2.
# ---------------------------------------------------------------------------
CONFIGS: dict[str, dict] = {
    "tiny": {
        "d_model": 8,
        "n_heads": 1,
        "n_layers": 1,
        "d_ff": 16,
        "vocab_size": 32,
        "seq_len": 8,
        "num_bits": 8,
        "c": 2,
        "scale": 0.1,
        "max_exp": 4,
        "description": "Tiny (demo-scale)",
    },
    "small": {
        "d_model": 64,
        "n_heads": 1,
        "n_layers": 2,
        "d_ff": 256,
        "vocab_size": 256,
        "seq_len": 16,
        "num_bits": 8,
        "c": 2,
        "scale": 0.1,
        "max_exp": 4,
        "description": "Small",
    },
    "medium": {
        "d_model": 128,
        "n_heads": 1,
        "n_layers": 4,
        "d_ff": 512,
        "vocab_size": 512,
        "seq_len": 32,
        "num_bits": 8,
        "c": 2,
        "scale": 0.1,
        "max_exp": 4,
        "description": "Medium",
    },
    "large": {
        "d_model": 256,
        "n_heads": 1,
        "n_layers": 6,
        "d_ff": 1024,
        "vocab_size": 1024,
        "seq_len": 32,
        "num_bits": 8,
        "c": 2,
        "scale": 0.1,
        "max_exp": 4,
        "description": "Large",
    },
    # GPT-2 Small has d_model=768, 12 layers, d_ff=3072; nearest power-of-2
    # equivalent with n_heads=1.
    "gpt2-small": {
        "d_model": 512,
        "n_heads": 1,
        "n_layers": 12,
        "d_ff": 2048,
        "vocab_size": 50257,
        "seq_len": 32,
        "num_bits": 8,
        "c": 2,
        "scale": 0.1,
        "max_exp": 4,
        "description": "GPT-2 Small (power-of-2 approximation, n_heads=1)",
    },
}

REPO_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = REPO_ROOT / "python"
PROVER_DIR = REPO_ROOT / "prover"
BINARY = PROVER_DIR / "target" / "release" / "piformer"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def _run_timed(cmd: list[str], label: str) -> float:
    """Run *cmd*, stream its output, and return wall-clock seconds."""
    print(f"\n[{label}] $ {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"[{label}] FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)
    return elapsed


def _build_rust() -> None:
    print("\n[build] cargo build --release")
    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "piformer"],
        cwd=PROVER_DIR,
        text=True,
    )
    if result.returncode != 0:
        print("[build] FAILED", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[build] binary: {BINARY}")


def _model_params(cfg: dict) -> int:
    """Rough parameter count (embedding excluded, just transformer blocks + head)."""
    d, d_ff, L, V = cfg["d_model"], cfg["d_ff"], cfg["n_layers"], cfg["vocab_size"]
    per_block = (
        4 * d * d          # Q, K, V, O projections
        + d * d_ff         # FFN W1
        + d_ff * d         # FFN W2
        + 2 * d            # LN1 gamma+beta
        + 2 * d            # LN2 gamma+beta
    )
    lm_head = d * V
    embedding = V * d + cfg["seq_len"] * d  # token + positional
    return L * per_block + lm_head + embedding


# ---------------------------------------------------------------------------
# Python model creation + export
# ---------------------------------------------------------------------------

def _export(cfg: dict, out_dir: Path) -> float:
    """Create a random-weight PiFormerModel and export weights + witness."""

    # Build the Python inline script so we don't need the repo on PYTHONPATH
    seq_len = cfg["seq_len"]
    token_ids = list(range(seq_len % cfg["vocab_size"]))  # deterministic dummy input
    # Make sure token_ids have the right length
    token_ids = (token_ids * (seq_len // len(token_ids) + 1))[:seq_len]

    py_code = f"""
import sys
sys.path.insert(0, {str(PYTHON_DIR)!r})

import torch
from piformer.model import PiFormerModel
from piformer.export import export_all

cfg = {cfg!r}
torch.manual_seed(42)
model = PiFormerModel(
    vocab_size=cfg["vocab_size"],
    d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],
    n_layers=cfg["n_layers"],
    d_ff=cfg["d_ff"],
    max_seq_len=cfg["seq_len"],
    num_bits=cfg["num_bits"],
    c=cfg["c"],
    scale=cfg["scale"],
    max_exp=cfg["max_exp"],
)
model.eval()

token_ids = {token_ids!r}
export_all(
    model, token_ids,
    weights_path={str(out_dir / "weights.json")!r},
    witness_path={str(out_dir / "witness.json")!r},
    quant_scale=64,
    ln_scale=4,
    extra_beta_floor=8,
    lasso_sigma=4,
)
"""

    print(f"\n[export] building model: {cfg['description']}")
    print(f"         d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, "
          f"d_ff={cfg['d_ff']}, seq_len={cfg['seq_len']}, vocab={cfg['vocab_size']}")
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, "-c", py_code], text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print("[export] FAILED", file=sys.stderr)
        sys.exit(result.returncode)
    return elapsed


# ---------------------------------------------------------------------------
# Main benchmark logic for one model
# ---------------------------------------------------------------------------

def benchmark_one(name: str, cfg: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = out_dir / "weights.json"
    witness = out_dir / "witness.json"
    pk      = out_dir / "model.pk"
    vk      = out_dir / "model.vk"
    proof   = out_dir / "proof.bin"

    _section(f"Benchmark: {name}  ({cfg['description']})")
    print(f"  Approx parameters : ~{_model_params(cfg):,}")
    print(f"  Output directory  : {out_dir}")

    t_export = _export(cfg, out_dir)

    t_setup = _run_timed(
        [str(BINARY), "setup",
         "--weights", str(weights),
         "--seq-len", str(cfg["seq_len"]),
         "--pk", str(pk),
         "--vk", str(vk)],
        "setup",
    )

    t_prove = _run_timed(
        [str(BINARY), "prove",
         "--pk", str(pk),
         "--witness", str(witness),
         "--proof", str(proof)],
        "prove",
    )

    t_verify = _run_timed(
        [str(BINARY), "verify",
         "--vk", str(vk),
         "--proof", str(proof)],
        "verify",
    )

    sizes = {
        "pk":    pk.stat().st_size    if pk.exists()    else 0,
        "vk":    vk.stat().st_size    if vk.exists()    else 0,
        "proof": proof.stat().st_size if proof.exists() else 0,
    }

    return {
        "model":      name,
        "description": cfg["description"],
        "d_model":    cfg["d_model"],
        "n_layers":   cfg["n_layers"],
        "d_ff":       cfg["d_ff"],
        "seq_len":    cfg["seq_len"],
        "vocab_size": cfg["vocab_size"],
        "approx_params": _model_params(cfg),
        "t_export_s":  round(t_export,  3),
        "t_setup_s":   round(t_setup,   3),
        "t_prove_s":   round(t_prove,   3),
        "t_verify_s":  round(t_verify,  3),
        "pk_bytes":    sizes["pk"],
        "vk_bytes":    sizes["vk"],
        "proof_bytes": sizes["proof"],
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def print_table(results: list[dict]) -> None:
    _section("Results summary")
    header = (
        f"{'Model':<14} {'d_model':>7} {'layers':>6} {'d_ff':>6} "
        f"{'seq':>4} {'params':>10}  "
        f"{'export':>8} {'setup':>8} {'prove':>8} {'verify':>8}  "
        f"{'proof size':>10}"
    )
    print(header)
    print("─" * len(header))
    for r in results:
        row = (
            f"{r['model']:<14} "
            f"{r['d_model']:>7} "
            f"{r['n_layers']:>6} "
            f"{r['d_ff']:>6} "
            f"{r['seq_len']:>4} "
            f"{r['approx_params']:>10,}  "
            f"{r['t_export_s']:>7.2f}s "
            f"{r['t_setup_s']:>7.2f}s "
            f"{r['t_prove_s']:>7.2f}s "
            f"{r['t_verify_s']:>7.2f}s  "
            f"{_fmt_bytes(r['proof_bytes']):>10}"
        )
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PiFormer ZK proving and verification time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="tiny",
        choices=list(CONFIGS.keys()),
        metavar="MODEL_NAME",
        help=f"Model size to benchmark. Choices: {', '.join(CONFIGS)}. Default: tiny",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        metavar="N",
        help="Override sequence length",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Scratch directory for generated files",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip `cargo build` (assumes binary is already built)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all predefined model sizes sequentially",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write results as JSON to this file",
    )
    args = parser.parse_args()

    # ── Prerequisites ────────────────────────────────────────────────────────
    _section("Prerequisites")

    py_ver = sys.version.split()[0]
    print(f"  python  : {py_ver}")

    try:
        import torch  # noqa: F401
        print(f"  torch   : {torch.__version__}")
    except ImportError:
        print("ERROR: PyTorch not installed. "
              f"Run: pip install -r {PYTHON_DIR}/requirements.txt", file=sys.stderr)
        sys.exit(1)

    if not args.no_build:
        _build_rust()
    elif not BINARY.exists():
        print(f"ERROR: binary not found at {BINARY}. "
              "Remove --no-build to build it first.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"  piformer: {BINARY} (pre-built)")

    # ── Select models ────────────────────────────────────────────────────────
    names = list(CONFIGS.keys()) if args.all else [args.model]

    base_out = args.out_dir or (REPO_ROOT / "benchmark_out")
    results: list[dict] = []

    for name in names:
        cfg = dict(CONFIGS[name])
        if args.seq_len is not None:
            cfg["seq_len"] = args.seq_len
        out_dir = base_out / name
        r = benchmark_one(name, cfg, out_dir)
        results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_table(results)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\n[results] JSON written to {args.json}")

    print()


if __name__ == "__main__":
    main()
