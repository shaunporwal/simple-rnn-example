# ── imports ─────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pathlib
import argparse  # Added argparse for command-line arguments

# ── device ──────────────────────────────────────────────────────
DEVICE = "cpu"


# ── dataset (character-level) ───────────────────────────────────
class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 10):
        self.chars = sorted(list(set(text)))
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.i2c = {i: c for c, i in self.c2i.items()}
        self.vocab_size = len(self.chars)
        self.seq_len = seq_len
        self.data = torch.tensor([self.c2i[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]  # input, target


# ── model ───────────────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, vocab_size, hidden=256, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)  # [B, T] -> [B, T, H]
        out, hidden = self.gru(x, hidden)
        out = self.head(out)  # logits
        return out, hidden


# ── training helper ─────────────────────────────────────────────
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits, _ = model(xb)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item()
    return total / len(loader)


# ── text generation ─────────────────────────────────────────────
@torch.no_grad()
def generate(model, start, ds, max_len=500, temperature=0.8):
    model.eval()
    hidden = None
    indices = [ds.c2i[c] for c in start]
    inp = torch.tensor([indices], device=DEVICE)
    for _ in range(max_len):
        logits, hidden = model(inp, hidden)
        next_logits = logits[0, -1] / temperature
        probs = F.softmax(next_logits, dim=0)
        idx = torch.multinomial(probs, 1).item()
        indices.append(idx)
        inp = torch.tensor([[idx]], device=DEVICE)
    return "".join(ds.i2c[i] for i in indices)


# ── main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Train a character-level RNN model.")
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "-sl", "--seq_len", type=int, default=10, help="Sequence length for training"
    )
    parser.add_argument(
        "-ml",
        "--max_len",
        type=int,
        default=500,
        help="Maximum length for text generation",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=30, help="Number of training epochs"
    )
    args = parser.parse_args()

    # 1.  load your own text file
    path = pathlib.Path("story.txt")

    raw_text = path.read_text().lower()
    ds = CharDataset(raw_text, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 2.  build model
    model = GRUModel(ds.vocab_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 3.  quick training loop
    for epoch in range(args.epochs):
        loss = train_epoch(model, dl, opt, loss_fn)
        print(f"epoch {epoch + 1} loss {loss:.3f}")

    # 4.  ask for a prompt and generate
    while True:
        prompt = input("\nYour prompt (blank to quit): ")
        if not prompt.strip():
            break
        print("\n--- continuation ---")
        print(generate(model, prompt.lower(), ds, max_len=args.max_len))
