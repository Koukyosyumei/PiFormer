import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from piformer.model import PiFormerModel

# --- 1. 超軽量データセットの作成 (Tiny Shakespeare Subset) ---
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

# --- 2. 学習設定 ---
# ダミーデータ（または小さなテキスト）
text_data = "to be, or not to be, that is the question: whether 'tis nobler in the mind to suffer" * 100
seq_len = 32
dataset = CharDataset(text_data, seq_len)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# モデルの初期化 (非常に小さく設定)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PiFormerModel(
    vocab_size=dataset.vocab_size,
    d_model=64,
    n_heads=2,
    n_layers=2,
    d_ff=128,
    max_seq_len=seq_len,
    num_bits=8,
    c=2,
    scale=0.1,
    max_exp=4
).to(device)

# 最適化（STEを使用しているため、通常のAdamでOK）
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# --- 3. 学習ループ ---
print(f"Starting training on {device}...")
model.train()
for epoch in range(10):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x) # (B, T, vocab_size)

        # Flatten for loss
        loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")

# --- 4. 生成テスト (学習できているかの確認) ---
model.eval()
with torch.no_grad():
    start_str = "to be"
    input_ids = torch.tensor([dataset.char_to_idx[ch] for ch in start_str]).unsqueeze(0).to(device)

    generated = start_str
    for _ in range(20):
        logits = model(input_ids)
        next_token = logits[0, -1, :].argmax().item()
        generated += dataset.idx_to_char[next_token]
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)

    print(f"\nGenerated text: '{generated}'")