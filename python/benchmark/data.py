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

TREC_TRAIN_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
TREC_TEST_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"
TREC_COARSE_LABELS = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]


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


def _ensure_trec(cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / "trec_train_5500.label"
    test_path = cache_dir / "trec_TREC_10.label"
    for path, url in [(train_path, TREC_TRAIN_URL), (test_path, TREC_TEST_URL)]:
        if not path.exists():
            print(f"Downloading {url} → {path}")
            urllib.request.urlretrieve(url, path)
    return train_path, test_path


def _parse_trec(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label_field, _, text = line.partition(" ")
            coarse = label_field.split(":")[0]
            pairs.append((text.strip(), coarse))
    return pairs


def load_trec(cache_dir: Path, seq_len: int = 128):
    """TREC question classification (coarse, 6 classes), char-level encoded.

    Returns (train_x, train_y, val_x, val_y, vocab_size, num_classes, pad_id).
    Reserved ids: 0 = PAD, 1 = UNK. seq_len is fixed; longer questions are
    truncated, shorter ones right-padded with PAD.
    """
    train_path, test_path = _ensure_trec(cache_dir)
    train_pairs = _parse_trec(train_path)
    test_pairs = _parse_trec(test_path)

    label_map = {l: i for i, l in enumerate(TREC_COARSE_LABELS)}
    pad_id, unk_id = 0, 1
    chars = sorted({c for text, _ in train_pairs for c in text})
    stoi = {c: i + 2 for i, c in enumerate(chars)}
    vocab_size = len(chars) + 2

    def encode(text: str) -> list[int]:
        ids = [stoi.get(c, unk_id) for c in text[:seq_len]]
        ids += [pad_id] * (seq_len - len(ids))
        return ids

    def encode_pairs(pairs):
        x = torch.tensor([encode(t) for t, _ in pairs], dtype=torch.long)
        y = torch.tensor([label_map[lbl] for _, lbl in pairs], dtype=torch.long)
        return x, y

    train_x, train_y = encode_pairs(train_pairs)
    val_x, val_y = encode_pairs(test_pairs)
    return train_x, train_y, val_x, val_y, vocab_size, len(TREC_COARSE_LABELS), pad_id


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
