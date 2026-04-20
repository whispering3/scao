"""
Quick benchmark: SCAO vs AdamW on a small GPT-like transformer.
================================================================

Usage:
    python benchmarks/compare_adamw_scao.py [--steps 200] [--device cpu]

Reports:
  - Loss curve every 10 steps
  - Total wall-clock time
  - Memory usage (if CUDA)
  - Final perplexity proxy
"""

from __future__ import annotations

import argparse
import math
import time
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scao import SCAO


# ---------------------------------------------------------------------------
# Tiny GPT-like model
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, seq_len: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", causal_mask.view(1, 1, seq_len, seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_head: int = 4,
        seq_len: int = 64,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, seq_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_benchmark(
    optimizer_name: str,
    steps: int = 200,
    device: str = "cpu",
    batch_size: int = 8,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)

    vocab_size = 256
    seq_len = 64
    d_model = 128

    model = TinyGPT(vocab_size=vocab_size, d_model=d_model, seq_len=seq_len).to(device)
    print(f"  Model parameters: {model.num_params:,}")

    # Random token sequences as synthetic data
    data = torch.randint(0, vocab_size, (1000, seq_len + 1), device=device)
    inputs = data[:, :-1]
    labels = data[:, 1:]
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=True)

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=0.1
        )
    elif optimizer_name == "scao":
        optimizer = SCAO(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.1,
            warmup_steps=20,
            precond_freq=20,
            k_min=8,
            k_max=64,
            tau=1.0,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    step = 0
    data_iter = iter(loader)

    while step < steps:
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)

        optimizer.zero_grad()
        logits = model(xb)  # (B, T, vocab)
        loss = loss_fn(logits.reshape(-1, vocab_size), yb.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        step += 1

        if step % 10 == 0 or step == 1:
            print(f"  [{optimizer_name}] step {step:4d}/{steps}  loss={loss.item():.4f}")

    elapsed = time.perf_counter() - t0

    result = {
        "optimizer": optimizer_name,
        "final_loss": losses[-1],
        "avg_loss_last_20": sum(losses[-20:]) / min(20, len(losses)),
        "total_time_s": elapsed,
        "steps_per_sec": steps / elapsed,
    }

    if device == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1e6

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SCAO vs AdamW benchmark")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    print(f"\n{'='*60}")
    print(f"  SCAO vs AdamW Benchmark  |  device={device}  |  steps={args.steps}")
    print(f"{'='*60}\n")

    results = []
    for opt_name in ["adamw", "scao"]:
        print(f"\n--- {opt_name.upper()} ---")
        r = run_benchmark(
            opt_name,
            steps=args.steps,
            device=device,
            batch_size=args.batch_size,
        )
        results.append(r)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  {r['optimizer'].upper()}")
        print(f"    Final loss:          {r['final_loss']:.4f}")
        print(f"    Avg loss (last 20):  {r['avg_loss_last_20']:.4f}")
        print(f"    Total time:          {r['total_time_s']:.1f}s")
        print(f"    Steps/sec:           {r['steps_per_sec']:.1f}")
        if "peak_memory_mb" in r:
            print(f"    Peak memory:         {r['peak_memory_mb']:.0f} MB")

    if len(results) == 2:
        a, s = results[0], results[1]
        speedup = a["avg_loss_last_20"] / max(s["avg_loss_last_20"], 1e-9)
        print(f"\n  SCAO loss ratio vs AdamW: {speedup:.3f}x")
        print(f"  (>1 means SCAO reached lower loss in same steps)")
    print()


if __name__ == "__main__":
    main()
