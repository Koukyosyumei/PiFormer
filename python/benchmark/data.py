"""
Dataset loaders for the PiFormer-vs-baseline benchmark.

Two famous small datasets are supported:

  * ``tinyshakespeare`` (default) — Karpathy's char-level corpus (~1.1 MB,
    auto-downloaded). Ubiquitous in transformer benchmarks (e.g. NanoGPT).
  * ``wikitext2`` — WikiText-2 word-level (via HuggingFace ``datasets`` if
    available). Larger, slower; pass ``--dataset wikitext2`` to use it.

Both produce ``(train_ids, val_ids, vocab_size, decode_fn)``.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Callable

import torch

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def _ensure_tinyshakespeare(cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "tinyshakespeare.txt"
    if not path.exists():
        print(f"Downloading TinyShakespeare → {path}")
        urllib.request.urlretrieve(TINYSHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


def load_tinyshakespeare(cache_dir: Path, val_frac: float = 0.1):
    text = _ensure_tinyshakespeare(cache_dir)
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = int(len(data) * val_frac)
    train_ids = data[:-n_val]
    val_ids = data[-n_val:]

    def decode(ids):
        return "".join(itos[int(i)] for i in ids)

    return train_ids, val_ids, len(chars), decode


def load_wikitext2(cache_dir: Path):
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "wikitext2 requires `pip install datasets`. "
            "Falling back: pass --dataset tinyshakespeare."
        ) from e

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=str(cache_dir))
    train_text = "\n".join(ds["train"]["text"])
    val_text = "\n".join(ds["validation"]["text"])

    # Use char-level tokenization for simplicity (BPE would add HF tokenizer dep).
    chars = sorted(set(train_text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    # Map any val-only chars to a dedicated <unk> id.
    unk = len(chars)
    itos = {i: ch for i, ch in enumerate(chars)}
    itos[unk] = "?"

    def encode(t: str):
        return torch.tensor([stoi.get(c, unk) for c in t], dtype=torch.long)

    train_ids = encode(train_text)
    val_ids = encode(val_text)

    def decode(ids):
        return "".join(itos[int(i)] for i in ids)

    return train_ids, val_ids, len(chars) + 1, decode


def get_dataset(name: str, cache_dir: Path):
    name = name.lower()
    if name == "tinyshakespeare":
        return load_tinyshakespeare(cache_dir)
    if name == "wikitext2":
        return load_wikitext2(cache_dir)
    raise ValueError(f"unknown dataset: {name}")


def random_batch(
    ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
):
    """Sample a batch of (input, target) pairs for next-token prediction."""
    starts = torch.randint(0, len(ids) - seq_len - 1, (batch_size,))
    x = torch.stack([ids[s : s + seq_len] for s in starts])
    y = torch.stack([ids[s + 1 : s + 1 + seq_len] for s in starts])
    return x.to(device), y.to(device)
