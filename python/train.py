"""
Minimal PiFormer training script.

Replace the `random_batch` function with your real dataset.
The model and training loop are production-quality.

Usage:
    cd python
    pip install -r requirements.txt
    python train.py
"""

import torch
import torch.nn as nn
from torch.optim import AdamW

from piformer import PiFormerModel, export_model

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
VOCAB_SIZE = 256
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
MAX_SEQ_LEN = 64
BATCH_SIZE = 8
SEQ_LEN = 32
NUM_STEPS = 500
LR = 3e-4
NUM_BITS = 8       # quantization bits for φ
C = 2              # sub-table decomposition depth
SCALE = 0.1        # quantization scale for φ
MAX_EXP = 4        # max power-of-two exponent for weights
EXPORT_PATH = "piformer_weights.json"


# ---------------------------------------------------------------------------
# Dataset (placeholder: replace with real data)
# ---------------------------------------------------------------------------
def random_batch(batch_size: int, seq_len: int, vocab_size: int):
    ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
    return ids[:, :-1], ids[:, 1:]   # input, target (next-token prediction)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    model = PiFormerModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        num_bits=NUM_BITS,
        c=C,
        scale=SCALE,
        max_exp=MAX_EXP,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PiFormerModel: {total_params:,} parameters")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS)

    model.train()
    for step in range(1, NUM_STEPS + 1):
        input_ids, targets = random_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

        logits = model(input_ids)                          # (B, T, vocab_size)
        loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"step {step:4d}/{NUM_STEPS} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

    print("\nTraining complete.")
    export_model(model, EXPORT_PATH)


if __name__ == "__main__":
    main()
