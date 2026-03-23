"""
PiFormer training demo.

Trains a tiny character-level language model, then exports both the quantized
weight file (weights.json) and a sample witness (witness.json) in the format
expected by the Rust prover.

Usage:
    python train_demo.py

After training you can run:
    cd ../prover
    cargo run --release --bin piformer -- setup \
        --weights piformer_weights.json --seq-len 8 \
        --pk model.pk --vk model.vk
    cargo run --release --bin piformer -- prove \
        --pk model.pk --witness piformer_witness.json --proof proof.bin
    cargo run --release --bin piformer -- verify \
        --vk model.vk --proof proof.bin

Note: n_heads must be 1 — the Rust prover is single-head only.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from piformer.model import PiFormerModel
from piformer.export import export_all


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.data = [self.char_to_idx[ch] for ch in data]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# 2. Training setup
# ---------------------------------------------------------------------------

text_data = (
    "to be, or not to be, that is the question: "
    "whether 'tis nobler in the mind to suffer"
) * 100

SEQ_LEN = 8   # keep small so the Rust prover's seq_len matches

dataset = CharDataset(text_data, SEQ_LEN)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# n_heads=1 is required by the single-head Rust prover
model = PiFormerModel(
    vocab_size=dataset.vocab_size,
    d_model=8,
    n_heads=1,
    n_layers=1,
    d_ff=16,
    max_seq_len=SEQ_LEN,
    num_bits=8,
    c=2,
    scale=0.1,
    max_exp=4,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------

print(f"Starting training on {device}…")
model.train()
for epoch in range(10):
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1:2d}/10  loss={total_loss / len(loader):.4f}")

# ---------------------------------------------------------------------------
# 4. Quick generation test
# ---------------------------------------------------------------------------

model.eval()
with torch.no_grad():
    start_str = "to be"
    input_ids = torch.tensor(
        [dataset.char_to_idx[ch] for ch in start_str]
    ).unsqueeze(0).to(device)

    generated = start_str
    for _ in range(3):
        logits = model(input_ids)
        next_token = logits[0, -1, :].argmax().item()
        generated += dataset.idx_to_char[next_token]
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]]).to(device)], dim=1
        )

    print(f"\nGenerated: '{generated}'")

# ---------------------------------------------------------------------------
# 5. Export weights + witness for the Rust prover
# ---------------------------------------------------------------------------

# Pick a short prompt whose length matches SEQ_LEN so the prover's seq_len
# parameter is consistent.
sample_prompt = "to be, o"[:SEQ_LEN]
token_ids = [dataset.char_to_idx[ch] for ch in sample_prompt]

export_all(
    model,
    token_ids,
    weights_path="piformer_weights.json",
    witness_path="piformer_witness.json",
    quant_scale=64,
    ln_scale=4,
    extra_beta_floor=8,
    lasso_sigma=4,
)

print(
    "\nExport complete.\n"
    "  piformer_weights.json  ← use with: piformer setup --weights\n"
    "  piformer_witness.json  ← use with: piformer prove  --witness\n"
)
