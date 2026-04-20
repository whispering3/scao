"""
GPT Convergence Benchmark: SCAO vs AdamW vs Shampoo
====================================================

Compares three optimizers on a TinyGPT (~10M params) character-level language model
with synthetic data, reporting:

  • Loss curve at every step
  • Convergence speed  (first step reaching <threshold>)
  • Area Under Loss Curve (AUC) — lower is better
  • Steps-per-second throughput
  • Peak memory (if CUDA)
  • ASCII loss chart in terminal
  • CSV export for plotting

All optimizers implemented from scratch — no external dependencies beyond PyTorch.

Usage:
    python scao/benchmarks/gpt_comparison.py
    python scao/benchmarks/gpt_comparison.py --steps 300 --device cuda
    python scao/benchmarks/gpt_comparison.py --steps 100 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scao import SCAO


# ===========================================================================
# Baseline optimizers (no external deps)
# ===========================================================================

class DiagonalShampoo(torch.optim.Optimizer):
    """
    Diagonal/scalar Shampoo — per-layer diagonal preconditioner using
    accumulated gradient statistics.  Equivalent to Adagrad but with
    an explicit EMA decay like AdamW.

    This is a simplified Shampoo without full Kronecker factors, serving
    as a "second-order baseline" that is much cheaper than full Shampoo.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["G2"] = torch.zeros_like(g)   # diag Hessian approx

                state["step"] += 1
                G2 = state["G2"]

                # EMA of squared gradient (diagonal curvature)
                G2.mul_(beta).addcmul_(g, g, value=1.0 - beta)

                # Diagonal preconditioner: p ← p - lr * G2^{-1/2} * g
                precond_g = g / (G2.sqrt().add_(eps))

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(precond_g.to(p.dtype), alpha=-lr)

        return loss


class Muon(torch.optim.Optimizer):
    """
    Muon — Momentum + Orthogonal Update (simplified).
    Applies Nesterov momentum then orthogonalises the update matrix via
    Newton-Schulz iterations (same trick as the paper).

    Reference: Kosson et al., "Muon: An Optimizer for Hidden Layers", 2024.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _zeropower_via_newtonschulz(G: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Newton-Schulz iterations to compute G / ||G||_F  (approx polar factor).
        Works on 2-D matrices; falls back to normalised G for 1-D.
        """
        if G.ndim < 2:
            norm = G.norm().clamp(min=1e-8)
            return G / norm
        # Normalise to spectral neighbourhood of 1
        norm = G.norm(p="fro").clamp(min=1e-8)
        X = G / norm
        # 5th-order poly iterations  X ← a*X + b*X*(X^T*X) + c*X*(X^T*X)^2
        a, b, c = (3.4445, -4.7750, 2.0315)
        for _ in range(steps):
            A = X.T @ X
            X = a * X + b * (X @ A) + c * (X @ (A @ A))
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state["buf"] = torch.zeros_like(g)

                buf = state["buf"]
                buf.mul_(mu).add_(g)          # Nesterov buffer
                g_nes = g.add(buf, alpha=mu)  # Nesterov gradient

                # Reshape to 2-D for orthogonalisation
                orig_shape = g_nes.shape
                if g_nes.ndim >= 2:
                    g2d = g_nes.view(g_nes.shape[0], -1)
                    ortho = self._zeropower_via_newtonschulz(g2d, ns)
                    update = ortho.view(orig_shape)
                else:
                    update = self._zeropower_via_newtonschulz(g_nes, ns)

                # Scale to match RMS of the gradient
                rms = g_nes.norm() / (g_nes.numel() ** 0.5 + 1e-8)
                update = update * rms

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr)

        return loss


# ===========================================================================
# Model
# ===========================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, seq_len: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask.view(1, 1, seq_len, seq_len))

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
        return self.proj((attn @ v).transpose(1, 2).reshape(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, seq_len: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, seq_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """~10M parameter GPT-style language model."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model:    int = 256,
        n_layers:   int = 6,
        n_head:     int = 8,
        seq_len:    int = 128,
    ):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks    = nn.ModuleList(
            [TransformerBlock(d_model, n_head, seq_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

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


# ===========================================================================
# Benchmark runner
# ===========================================================================

def make_optimizer(name: str, model: nn.Module) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    if name == "scao":
        return SCAO(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.1,
            warmup_steps=50,
            precond_freq=20,
            k_min=8,
            k_max=64,
            tau=1.0,
        )

    if name == "diag_shampoo":
        return DiagonalShampoo(model.parameters(), lr=3e-4, weight_decay=0.1)

    if name == "muon":
        return Muon(model.parameters(), lr=3e-3, weight_decay=0.1)

    raise ValueError(f"Unknown optimizer: {name}")


def run(
    opt_name:   str,
    steps:      int,
    device:     str,
    batch_size: int,
    seed:       int,
    threshold:  float,
    vocab_size: int,
    seq_len:    int,
    d_model:    int,
    n_layers:   int,
    n_head:     int,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    model = TinyGPT(
        vocab_size=vocab_size, d_model=d_model,
        n_layers=n_layers, n_head=n_head, seq_len=seq_len,
    ).to(device)

    optimizer = make_optimizer(opt_name, model)

    # Synthetic character-level data
    data   = torch.randint(0, vocab_size, (2000, seq_len + 1), device=device)
    inputs = data[:, :-1]
    labels = data[:, 1:]
    loader = DataLoader(
        TensorDataset(inputs, labels),
        batch_size=batch_size, shuffle=True,
    )

    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    converge_step: int | None = None

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    data_iter = iter(loader)

    for step in range(1, steps + 1):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits.reshape(-1, vocab_size), yb.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        lv = loss.item()
        losses.append(lv)

        if converge_step is None and lv < threshold:
            converge_step = step

    elapsed = time.perf_counter() - t0

    auc = sum(losses) / len(losses)
    result: dict[str, Any] = {
        "optimizer":        opt_name,
        "steps":            steps,
        "losses":           losses,
        "final_loss":       losses[-1],
        "avg_last_20":      sum(losses[-20:]) / min(20, len(losses)),
        "auc":              auc,
        "converge_step":    converge_step,
        "total_time_s":     elapsed,
        "steps_per_sec":    steps / elapsed,
        "num_params":       model.num_params,
    }
    if device == "cuda":
        result["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1e6

    return result


# ===========================================================================
# Reporting
# ===========================================================================

COLORS = {
    "adamw":        "\033[94m",   # blue
    "scao":         "\033[92m",   # green
    "diag_shampoo": "\033[93m",   # yellow
    "muon":         "\033[95m",   # magenta
    "reset":        "\033[0m",
}


def _color(name: str, text: str) -> str:
    return f"{COLORS.get(name, '')}{text}{COLORS['reset']}"


def ascii_loss_chart(results: list[dict], width: int = 72, height: int = 20) -> None:
    """Print a compact ASCII loss curve for all optimizers."""
    all_losses = [l for r in results for l in r["losses"]]
    y_min = min(all_losses) * 0.98
    y_max = max(all_losses[: len(results[0]["losses"]) // 5]) * 1.02  # zoom on early

    steps = len(results[0]["losses"])
    stride = max(1, steps // width)

    print(f"\n  Loss Curve  (y: {y_min:.3f}–{y_max:.3f}, first {steps} steps)")
    print("  " + "─" * (width + 2))

    symbols = {"adamw": "·", "scao": "●", "diag_shampoo": "▲", "muon": "■"}

    grid: list[list[str]] = [[" "] * (width + 1) for _ in range(height)]

    for r in results:
        sym = symbols.get(r["optimizer"], "x")
        losses = r["losses"]
        for col in range(width):
            step_idx = min(col * stride, len(losses) - 1)
            lv = losses[step_idx]
            # Clamp to chart range
            lv_clamped = max(y_min, min(y_max, lv))
            row = int((y_max - lv_clamped) / (y_max - y_min + 1e-9) * (height - 1))
            row = max(0, min(height - 1, row))
            grid[row][col] = sym

    for i, row in enumerate(grid):
        y_val = y_max - i * (y_max - y_min) / (height - 1)
        label = f"{y_val:7.3f} │"
        print("  " + label + "".join(row))

    print("  " + " " * 9 + "└" + "─" * width)
    step_labels = " " * 9 + "  1" + " " * (width // 2 - 3) + f"{steps // 2}" + \
                  " " * (width // 2 - 5) + f"{steps}"
    print("  " + step_labels)

    legend = "  Legend: " + "  ".join(
        f"{symbols[r['optimizer']]}={r['optimizer']}" for r in results
    )
    print(legend)


def print_summary(results: list[dict], threshold: float) -> None:
    w = 72
    print(f"\n{'═' * w}")
    print("  BENCHMARK SUMMARY — TinyGPT  (lower loss = better)")
    print(f"{'─' * w}")
    print(f"  {'Optimizer':<18} {'FinalLoss':>10} {'AvgLast20':>10} "
          f"{'AUC':>8} {'ConvStep':>10} {'Steps/s':>8}")
    print(f"{'─' * w}")

    # Sort by AUC
    sorted_r = sorted(results, key=lambda r: r["auc"])
    best_auc = sorted_r[0]["auc"]

    for r in sorted_r:
        name   = r["optimizer"]
        marker = " ✓" if r["auc"] == best_auc else "  "
        conv   = str(r["converge_step"]) if r["converge_step"] else f">{r['steps']}"
        print(
            _color(name,
                f"  {name:<18} {r['final_loss']:10.4f} {r['avg_last_20']:10.4f} "
                f"{r['auc']:8.4f} {conv:>10} {r['steps_per_sec']:8.1f}{marker}"
            )
        )

    print(f"{'─' * w}")
    print(f"  Model: {results[0]['num_params']:,} params | "
          f"Threshold: {threshold:.3f} | AUC = mean loss over all steps")

    # Relative comparisons (SCAO vs AdamW)
    scao_r  = next((r for r in results if r["optimizer"] == "scao"), None)
    adamw_r = next((r for r in results if r["optimizer"] == "adamw"), None)
    if scao_r and adamw_r:
        auc_ratio     = adamw_r["auc"]      / max(scao_r["auc"],     1e-9)
        speed_ratio   = scao_r["steps_per_sec"] / max(adamw_r["steps_per_sec"], 1e-9)
        print(f"\n  SCAO vs AdamW:")
        print(f"    AUC ratio      : {auc_ratio:.3f}x  (>1 means SCAO converges faster)")
        print(f"    Throughput     : {speed_ratio:.3f}x  (1 = same speed)")
        if scao_r.get("peak_memory_mb") and adamw_r.get("peak_memory_mb"):
            mem_ratio = scao_r["peak_memory_mb"] / max(adamw_r["peak_memory_mb"], 1)
            print(f"    Memory overhead: {mem_ratio:.2f}x")
    print(f"{'═' * w}\n")


def save_csv(results: list[dict], path: str) -> None:
    steps = len(results[0]["losses"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["step"] + [r["optimizer"] for r in results])
        for i in range(steps):
            row = [i + 1] + [r["losses"][i] for r in results]
            writer.writerow(row)
    print(f"  Loss curves saved to: {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="GPT optimizer comparison benchmark")
    parser.add_argument("--steps",      type=int,   default=200,
                        help="Number of training steps (default: 200)")
    parser.add_argument("--device",     type=str,   default="cpu",
                        help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--threshold",  type=float, default=4.5,
                        help="Loss threshold for convergence step metric")
    parser.add_argument("--optimizers", type=str,   default="adamw,scao,diag_shampoo",
                        help="Comma-separated list of optimizers to compare")
    parser.add_argument("--csv",        type=str,   default=None,
                        help="Optional path to save loss curve CSV")
    # Model size knobs
    parser.add_argument("--d-model",   type=int,   default=256)
    parser.add_argument("--n-layers",  type=int,   default=6)
    parser.add_argument("--n-head",    type=int,   default=8)
    parser.add_argument("--seq-len",   type=int,   default=128)
    parser.add_argument("--vocab-size",type=int,   default=256)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA not available, falling back to CPU.")
        device = "cpu"

    opt_names = [o.strip() for o in args.optimizers.split(",")]

    print(f"\n{'═' * 72}")
    print(f"  GPT Benchmark  |  device={device}  |  steps={args.steps}")
    print(f"  Optimizers: {', '.join(opt_names)}")
    print(f"{'═' * 72}\n")

    results = []
    for name in opt_names:
        print(f"  Running {name.upper()} ...", flush=True)
        r = run(
            opt_name=name,
            steps=args.steps,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            threshold=args.threshold,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_head=args.n_head,
        )
        results.append(r)
        print(f"    → done  final_loss={r['final_loss']:.4f}  "
              f"time={r['total_time_s']:.1f}s  {r['steps_per_sec']:.1f} steps/s")

    ascii_loss_chart(results)
    print_summary(results, args.threshold)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
