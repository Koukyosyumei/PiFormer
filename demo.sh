#!/usr/bin/env bash
# =============================================================================
# PiFormer end-to-end demo
#
# Pipeline:
#   1. (Python) Train a tiny character-level language model
#   2. (Python) Export quantized weights.json + witness.json
#   3. (Rust)   piformer setup   — commit to weights, produce .pk / .vk
#   4. (Rust)   piformer prove   — generate ZK proof from witness
#   5. (Rust)   piformer verify  — check the proof
#
# Usage:
#   bash demo.sh [--out-dir DIR] [--epochs N] [--skip-train]
#
#   --out-dir DIR     Scratch directory for all generated files (default: demo_out)
#   --epochs N        Training epochs (default: 10)
#   --skip-train      Reuse existing weights.json / witness.json in OUT_DIR
#
# Requirements:
#   - Python 3.9+ with PyTorch installed  (pip install -r python/requirements.txt)
#   - Rust toolchain  (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
OUT_DIR="demo_out"
EPOCHS=10
SKIP_TRAIN=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)  OUT_DIR="$2";  shift 2 ;;
        --epochs)   EPOCHS="$2";   shift 2 ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

WEIGHTS="$OUT_DIR/weights.json"
WITNESS="$OUT_DIR/witness.json"
PK="$OUT_DIR/model.pk"
VK="$OUT_DIR/model.vk"
PROOF="$OUT_DIR/proof.bin"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$REPO_ROOT/python"
PROVER_DIR="$REPO_ROOT/prover"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step() { echo; echo "==> $*"; }
ok()   { echo "    OK: $*"; }

# ---------------------------------------------------------------------------
# 0. Validate tools
# ---------------------------------------------------------------------------
step "Checking prerequisites"

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found." >&2; exit 1
fi
ok "python3 $(python3 --version 2>&1 | awk '{print $2}')"

if ! command -v cargo &>/dev/null; then
    echo "ERROR: cargo not found. Install Rust from https://rustup.rs" >&2; exit 1
fi
ok "cargo $(cargo --version)"

python3 -c "import torch" 2>/dev/null || {
    echo "ERROR: PyTorch not installed. Run: pip install -r $PYTHON_DIR/requirements.txt" >&2
    exit 1
}
ok "torch $(python3 -c 'import torch; print(torch.__version__)')"

mkdir -p "$OUT_DIR"
ok "Output directory: $OUT_DIR"

# ---------------------------------------------------------------------------
# 1. Build the Rust CLI (release mode)
# ---------------------------------------------------------------------------
step "Building Rust CLI (release)"
cargo build --release --manifest-path "$PROVER_DIR/Cargo.toml" --bin piformer 2>&1 \
    | grep -E "^(error|warning\[|Compiling|Finished)" || true
PIFORMER="$PROVER_DIR/target/release/piformer"
ok "Binary: $PIFORMER"

# ---------------------------------------------------------------------------
# 2. Train and export (Python)
# ---------------------------------------------------------------------------
if [[ "$SKIP_TRAIN" -eq 1 && -f "$WEIGHTS" && -f "$WITNESS" ]]; then
    step "Skipping training — reusing existing files"
    ok "weights: $WEIGHTS"
    ok "witness: $WITNESS"
else
    step "Training PiFormer (epochs=$EPOCHS)"

    python3 - <<PYEOF
import sys, os
sys.path.insert(0, "$PYTHON_DIR")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from piformer.model import PiFormerModel
from piformer.export import export_all

# ---- Tiny dataset ----
class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        chars = sorted(set(data))
        self.vocab_size = len(chars)
        self.c2i = {c: i for i, c in enumerate(chars)}
        self.i2c = {i: c for i, c in enumerate(chars)}
        self.data = [self.c2i[c] for c in data]
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

text = (
    "to be, or not to be, that is the question: "
    "whether 'tis nobler in the mind to suffer"
) * 100

SEQ_LEN = 8
dataset = CharDataset(text, SEQ_LEN)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

print(f"  Training on {device} for $EPOCHS epoch(s) …")
model.train()
for epoch in range($EPOCHS):
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x).view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"  Epoch {epoch+1:2d}/$EPOCHS  loss={total/len(loader):.4f}")

# ---- Quick generation test ----
model.eval()
with torch.no_grad():
    start = "to be"
    ids = torch.tensor([[dataset.c2i[c] for c in start]]).to(device)
    gen = start
    for _ in range(3):
        nxt = model(ids)[0, -1].argmax().item()
        gen += dataset.i2c[nxt]
        ids = torch.cat([ids, torch.tensor([[nxt]]).to(device)], dim=1)
print(f"  Generated: '{gen}'")

# ---- Export ----
prompt   = "to be, o"[:SEQ_LEN]
tok_ids  = [dataset.c2i[c] for c in prompt]
export_all(
    model, tok_ids,
    weights_path="$WEIGHTS",
    witness_path="$WITNESS",
    quant_scale=64,
    ln_scale=4,
    extra_beta_floor=8,
    lasso_sigma=4,
)
PYEOF

    ok "weights.json written: $WEIGHTS"
    ok "witness.json written: $WITNESS"
fi

# Sanity-check: ternary weights must be plain ints {-1, 0, 1}, not hex strings
step "Validating exported weight format"
python3 - <<PYEOF
import json, sys
with open("$WEIGHTS") as f:
    w = json.load(f)
blk = w["blocks"][0]
for field in ("q_w", "k_w", "v_w", "o_w", "ffn_w1", "ffn_w2"):
    sample = blk[field][0][0]
    if not isinstance(sample, int):
        print(f"ERROR: {field}[0][0] is {type(sample).__name__!r}, expected int", file=sys.stderr)
        sys.exit(1)
    if sample not in (-1, 0, 1):
        print(f"ERROR: {field}[0][0] = {sample}, expected {{-1, 0, 1}}", file=sys.stderr)
        sys.exit(1)
lm = w["lm_head_w"][0][0]
if not isinstance(lm, int) or lm not in (-1, 0, 1):
    print(f"ERROR: lm_head_w[0][0] = {lm!r}, expected int in {{-1,0,1}}", file=sys.stderr)
    sys.exit(1)
print("  Weight format OK: ternary fields are plain ints in {-1, 0, 1}")
PYEOF
ok "Format check passed"

# ---------------------------------------------------------------------------
# 3. piformer setup — offline preprocessing (key generation)
# ---------------------------------------------------------------------------
step "piformer setup  (key generation)"
"$PIFORMER" setup \
    --weights "$WEIGHTS" \
    --seq-len 8 \
    --pk "$PK" \
    --vk "$VK"
ok "Proving key:    $PK  ($(du -sh "$PK" | cut -f1))"
ok "Verifying key:  $VK  ($(du -sh "$VK" | cut -f1))"

# ---------------------------------------------------------------------------
# 4. piformer prove — generate ZK proof
# ---------------------------------------------------------------------------
step "piformer prove  (proof generation)"
"$PIFORMER" prove \
    --pk "$PK" \
    --witness "$WITNESS" \
    --proof "$PROOF"
ok "Proof:          $PROOF  ($(du -sh "$PROOF" | cut -f1))"

# ---------------------------------------------------------------------------
# 5. piformer verify — verify the proof
# ---------------------------------------------------------------------------
step "piformer verify  (proof verification)"
"$PIFORMER" verify \
    --vk "$VK" \
    --proof "$PROOF"

# ---------------------------------------------------------------------------
# 6. Inspect generated files
# ---------------------------------------------------------------------------
step "Inspecting generated files"
"$PIFORMER" inspect "$VK"
"$PIFORMER" inspect "$PROOF"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "  PiFormer demo complete!"
echo "  All files in: $OUT_DIR/"
ls -lh "$OUT_DIR/"
echo "============================================================"
