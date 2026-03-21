"""
PiFormer demo: learn to compute sum-mod-V of an integer sequence.

Task
----
  Input:   a sequence of integers  x_0, x_1, ..., x_{T-1}  in {0, ..., V-1}
  Target:  at EVERY position, predict  (x_0 + x_1 + ... + x_{T-1}) mod V

  Example (V = 8, T = 8):
    input  → [3, 1, 5, 2, 7, 0, 4, 6]   (sum = 28, 28 mod 8 = 4)
    target → [4, 4, 4, 4, 4, 4, 4, 4]

Why this task fits PiFormer
---------------------------
Linear attention computes a global context  C = φ(K)ᵀ V  that aggregates
ALL key-value pairs in the sequence — perfect for computing a global sum.
The target is the same at every position (no causal requirement), so the
non-causal architecture is used correctly: the model may attend to the full
input when predicting each position.

The correct answer is unique for each input sequence, so the model cannot
cheat by memorising a constant output.

Expected outcome (V = 8, T = 8, ~18 K parameters)
---------------------------------------------------
  Random-guess loss  ≈  log(V) ≈ 2.08
  Trained loss       →  < 0.10
  Accuracy           →  ~ 100 %

Usage
-----
  cd python
  pip install -r requirements.txt
  python train_demo.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from piformer import PiFormerModel
from piformer.projection import PowerOfTwoLinear

# ── Hyper-parameters ──────────────────────────────────────────────────────────
VOCAB_SIZE  = 8       # tokens are 0 … V-1; output classes also 0 … V-1
SEQ_LEN     = 8       # sequence length
D_MODEL     = 32
N_HEADS     = 2
N_LAYERS    = 2
D_FF        = 64
BATCH_SIZE  = 128
NUM_STEPS   = 3000
LR          = 3e-3
# ZK-friendly architecture parameters
NUM_BITS    = 4       # φ uses 2 sub-tables of size 2^2 = 4
C           = 2
SCALE       = 0.5
MAX_EXP     = 3
LOG_EVERY   = 300
# ─────────────────────────────────────────────────────────────────────────────


def sum_mod_batch(batch_size: int, seq_len: int, vocab_size: int):
    """
    Generate (input_sequences, target_labels) pairs for the sum-mod-V task.

    targets[b, t] = (sum of inputs[b, :]) % vocab_size  for all t.
    The label is the same at every sequence position, so the loss at all T
    positions pushes toward the correct global sum.
    """
    inputs  = torch.randint(0, vocab_size, (batch_size, seq_len))
    sums    = inputs.sum(dim=-1) % vocab_size           # (B,)
    targets = sums.unsqueeze(1).expand(-1, seq_len)     # (B, T)
    return inputs, targets


@torch.no_grad()
def evaluate(model: nn.Module, vocab_size: int, n_batches: int = 20) -> dict:
    """Average cross-entropy loss and per-sequence accuracy on fresh batches."""
    model.eval()
    total_loss = seq_correct = seq_total = tok_correct = tok_total = 0
    for _ in range(n_batches):
        x, y = sum_mod_batch(BATCH_SIZE, SEQ_LEN, vocab_size)
        logits  = model(x)                              # (B, T, V)
        loss    = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        preds   = logits.argmax(dim=-1)                 # (B, T)
        # per-sequence: correct if ALL positions match (since all targets equal)
        seq_correct += (preds == y).all(dim=-1).sum().item()
        seq_total   += y.shape[0]
        tok_correct += (preds == y).sum().item()
        tok_total   += y.numel()
        total_loss  += loss.item()
    model.train()
    return {
        "loss":     total_loss / n_batches,
        "seq_acc":  seq_correct / seq_total,   # whole-sequence accuracy
        "tok_acc":  tok_correct / tok_total,   # per-token accuracy
    }


def show_sample(model: nn.Module, vocab_size: int, n: int = 6):
    """Print example predictions to confirm the model is computing sum mod V."""
    model.eval()
    x, y = sum_mod_batch(n, SEQ_LEN, vocab_size)
    with torch.no_grad():
        preds = model(x).argmax(dim=-1)
    model.train()

    print("\nSample predictions  (input → pred | truth   [input sum])")
    print("─" * 62)
    for i in range(n):
        inp  = x[i].tolist()
        pred = preds[i, 0].item()          # same at all positions
        gold = y[i, 0].item()
        total = sum(inp)
        ok   = "✓" if pred == gold else "✗"
        print(f"  {ok}  {inp}  →  {pred}  |  {gold}   (Σ={total})")
    print("─" * 62)


def main():
    torch.manual_seed(42)

    model = PiFormerModel(
        vocab_size  = VOCAB_SIZE,
        d_model     = D_MODEL,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        d_ff        = D_FF,
        max_seq_len = SEQ_LEN + 4,
        num_bits    = NUM_BITS,
        c           = C,
        scale       = SCALE,
        max_exp     = MAX_EXP,
    )
    # PowerOfTwoLinear initialises weights with std=0.02 (all snap to 0).
    # Reinitialise with std=1.0 so weights start near ±1 and the model is
    # non-trivial from step 1.
    for m in model.modules():
        if isinstance(m, PowerOfTwoLinear):
            nn.init.normal_(m.weight, std=1.0)

    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 62)
    print("  PiFormer  ·  Sum-mod-V Demo")
    print("=" * 62)
    print(f"  Parameters : {n_params:,}")
    print(f"  Task       : predict (Σ tokens) mod {VOCAB_SIZE}")
    print(f"               seq_len={SEQ_LEN}, vocab={VOCAB_SIZE}")
    print(f"  Baseline   : random-guess loss = {math.log(VOCAB_SIZE):.3f}")
    print("=" * 62)
    print(f"{'step':>6}  {'loss':>8}  {'tok-acc':>8}  {'seq-acc':>8}")
    print("─" * 40)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_STEPS, eta_min=LR * 0.02
    )

    running_loss = 0.0
    model.train()
    for step in range(1, NUM_STEPS + 1):
        x, y   = sum_mod_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        logits = model(x)                               # (B, T, V)
        loss   = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if step % LOG_EVERY == 0:
            m = evaluate(model, VOCAB_SIZE)
            print(
                f"  {step:5d}  {m['loss']:8.4f}  "
                f"{m['tok_acc']*100:7.1f}%  {m['seq_acc']*100:7.1f}%"
            )
            running_loss = 0.0

    print("─" * 40)

    final = evaluate(model, VOCAB_SIZE, n_batches=100)
    print(
        f"\nFinal  loss={final['loss']:.4f}  "
        f"tok-acc={final['tok_acc']*100:.1f}%  "
        f"seq-acc={final['seq_acc']*100:.1f}%"
    )

    show_sample(model, VOCAB_SIZE)


if __name__ == "__main__":
    main()
