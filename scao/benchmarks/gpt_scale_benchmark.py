"""
Multi-Scale GPT Benchmark: SCAO vs AdamW vs Shampoo
=====================================================

Trains a GPT-style model at configurable parameter scales and measures:
  - Validation perplexity (PPL)
  - Wall-clock time to target PPL
  - Peak GPU/CPU memory
  - Tokens per second

This is the primary benchmark for the SCAO paper's Table 1.

Usage (quick CPU smoke test):
    python scao/benchmarks/gpt_scale_benchmark.py --scale tiny --steps 200

Usage (full GPU run for paper):
    python scao/benchmarks/gpt_scale_benchmark.py --scale 125m --device cuda
    python scao/benchmarks/gpt_scale_benchmark.py --scale 350m --device cuda
    python scao/benchmarks/gpt_scale_benchmark.py --scale 1b   --device cuda

Scales:
  tiny  :   1M params  (d=128,  L=4,  H=4)   — CPU validation
  small :  10M params  (d=256,  L=6,  H=8)   — CPU validation
  medium:  50M params  (d=512,  L=8,  H=8)   — needs GPU
  125m  : 125M params  (d=768,  L=12, H=12)  — GPT-2 small equivalent
  350m  : 350M params  (d=1024, L=24, H=16)  — GPT-2 medium equivalent
  1b    : 1.3B params  (d=2048, L=24, H=16)  — GPT-3 small equivalent

Requirements:
    pip install datasets tokenizers
    # For GPU runs: CUDA >= 11.8, at least 24 GB VRAM for 350m, 80 GB for 1b
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Iterator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scao import SCAO

# ---------------------------------------------------------------------------
# Scale configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    n_head: int
    d_ff: int           # feed-forward hidden dim (usually 4 * d_model)
    seq_len: int
    label: str          # human-readable name

SCALE_CONFIGS: dict[str, ModelConfig] = {
    "tiny":   ModelConfig(128,  4,  4,  512,  128, "1M"),
    "5m":     ModelConfig(192,  6,  4,  768,  128, "5M"),
    "small":  ModelConfig(256,  6,  8, 1024,  256, "10M"),
    "medium": ModelConfig(512,  8,  8, 2048,  512, "50M"),
    "125m":   ModelConfig(768,  12, 12, 3072, 1024, "125M"),
    "350m":   ModelConfig(1024, 24, 16, 4096, 1024, "350M"),
    "1b":     ModelConfig(2048, 24, 16, 8192, 1024, "1.3B"),
}

# ---------------------------------------------------------------------------
# Model definition (GPT-2 architecture)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, seq_len: int) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # Reshape to multi-head
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention with causal mask
        scale = self.d_head ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, seq_len: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-2 style transformer language model."""

    def __init__(self, cfg: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_head, cfg.d_ff, cfg.seq_len)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        # Weight tying (GPT-2 style)
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def load_data(
    seq_len: int,
    batch_size: int,
    device: str,
    vocab_size: int = 256,
) -> tuple:
    """
    Load WikiText-2 (character-level) or fall back to synthetic data.
    Returns (train_iter_fn, val_loader, actual_vocab_size).
    train_iter_fn() returns a fresh iterator over the training set.
    """
    from torch.utils.data import DataLoader, TensorDataset

    print("  Loading data ...", end=" ", flush=True)
    try:
        from datasets import load_dataset  # type: ignore[import]
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(ds["train"]["text"])       # type: ignore[index]
        val_text   = "\n".join(ds["validation"]["text"])  # type: ignore[index]
        print(f"WikiText-2 ({len(train_text):,} chars train)", flush=True)
    except Exception as exc:
        print(f"fallback to synthetic ({exc})", flush=True)
        n = 50_000
        data = torch.randint(0, vocab_size, (n,), dtype=torch.long)
        split = int(n * 0.9)
        train_text = val_text = None
        # synthetic path:
        train_ds = TensorDataset(
            data[:split - seq_len].unfold(0, seq_len, seq_len),
            data[1:split - seq_len + 1].unfold(0, seq_len, seq_len),
        )
        val_ds = TensorDataset(
            data[split:split + (len(data) - split - seq_len)].unfold(0, seq_len, seq_len),
            data[split + 1:split + 1 + (len(data) - split - seq_len)].unfold(0, seq_len, seq_len),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)
        return lambda: iter(train_loader), val_loader, vocab_size

    # Character-level tokenisation
    chars = sorted(set(train_text + val_text))
    actual_vocab = min(len(chars), vocab_size)
    chars = chars[:actual_vocab]
    ch2idx = {c: i for i, c in enumerate(chars)}

    def encode(text: str) -> torch.Tensor:
        return torch.tensor([ch2idx.get(c, 0) for c in text], dtype=torch.long)

    def make_loader(text: str, shuffle: bool) -> DataLoader:
        tokens  = encode(text)
        n_full  = (len(tokens) - 1) // (seq_len + 1)
        tokens  = tokens[: n_full * (seq_len + 1) + 1]
        chunks  = tokens.unfold(0, seq_len + 1, seq_len + 1)
        inputs  = chunks[:, :-1].to(device)
        labels  = chunks[:, 1:].to(device)
        return DataLoader(
            TensorDataset(inputs, labels),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

    train_loader = make_loader(train_text, True)
    val_loader   = make_loader(val_text,   False)
    return lambda: iter(train_loader), val_loader, actual_vocab


# ---------------------------------------------------------------------------
# Diagonal Shampoo baseline (identical to gpt_comparison.py)
# ---------------------------------------------------------------------------

class DiagonalShampoo(torch.optim.Optimizer):
    """Adagrad-style diagonal second-order baseline."""

    def __init__(self, params, lr: float = 1e-3, eps: float = 1e-8,
                 weight_decay: float = 0.0, rho: float = 0.999) -> None:
        super().__init__(params, dict(lr=lr, eps=eps, weight_decay=weight_decay, rho=rho))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if not state:
                    state["H"] = torch.zeros_like(g)
                    state["t"] = 0
                state["t"] += 1
                t = state["t"]
                H = state["H"]
                H.mul_(group["rho"]).addcmul_(g, g, value=1.0 - group["rho"])
                bias = 1.0 - group["rho"] ** t
                Hd = (H / bias).pow(0.25).add_(group["eps"])
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(g / Hd, alpha=-group["lr"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, vocab_size: int,
             max_batches: int = 50) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, count = 0.0, 0
    for i, (xb, yb) in enumerate(loader):
        if i >= max_batches:
            break
        logits = model(xb)
        loss = loss_fn(logits.reshape(-1, vocab_size), yb.reshape(-1))
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_single(
    opt_name:     str,
    cfg:          ModelConfig,
    steps:        int,
    device:       str,
    batch_size:   int,
    seed:         int,
    train_iter_fn,
    val_loader,
    actual_vocab: int,
    lr:           float,
    diag_lr:      float,
    warmup_steps: int,
) -> dict:
    torch.manual_seed(seed)

    model = GPT(cfg, actual_vocab).to(device)
    n_params = model.num_params

    # --- Optimizer ---
    eff_lr = diag_lr if opt_name == "diag_shampoo" else lr
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=eff_lr, weight_decay=0.1, betas=(0.9, 0.95),
        )
    elif opt_name in ("scao", "scao_int8"):
        # precond_freq: update every ~2% of steps, min 10.
        # Stable eigenvectors (infrequent updates, high rho) outperform fresh-but-noisy
        # estimates (more frequent updates, lower rho) for short training runs.
        # min_precond_updates=2: Phase 2 starts at max(warmup, 2×p_freq).
        # k_max=d//2: capture up to 50% of spectral mass per dimension.
        # epsilon_sparse=0.01: discard only 1% → higher effective rank than default 0.05.
        # tau=1.0: curvature-aware clipping for training stability.
        p_freq = max(10, steps // 50)
        # SCAO benefits from a slightly higher LR than AdamW: the preconditioner
        # scales down high-curvature directions so the effective step size is
        # slightly smaller than AdamW's; compensating with +17% LR restores parity.
        scao_lr = eff_lr * 1.17
        optimizer = SCAO(
            model.parameters(),
            lr=scao_lr,
            weight_decay=0.1,
            warmup_steps=warmup_steps,
            precond_freq=p_freq,
            min_precond_updates=2,
            k_min=4,
            k_max=min(256, cfg.d_model // 2),
            epsilon_sparse=0.01,
            tau=1.0,
            betas=(0.9, 0.95),
            use_int8_ema=(opt_name == "scao_int8"),
        )
    elif opt_name == "diag_shampoo":
        optimizer = DiagonalShampoo(
            model.parameters(), lr=eff_lr, weight_decay=0.01,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # LR: linear warmup then cosine decay
    lr_warmup_steps = warmup_steps
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=lr_warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps - lr_warmup_steps, 1), eta_min=lr * 0.1,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[lr_warmup_steps],
    )

    loss_fn = nn.CrossEntropyLoss()
    train_losses: list[float] = []
    val_losses:   list[float] = []
    timestamps:   list[float] = []   # wall-clock seconds at each train step
    val_ppls:     list[tuple[float, float]] = []  # (wall_clock, ppl)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # CPU memory: start tracing Python heap + Torch C++ allocations via
    # tracemalloc.  This captures optimizer state tensors (float32 moments,
    # preconditioner blocks) allocated after the training start.
    tracemalloc.start()

    t0 = time.perf_counter()
    data_iter = train_iter_fn()
    val_every = max(10, steps // 20)

    for step in range(1, steps + 1):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = train_iter_fn()
            xb, yb = next(data_iter)

        optimizer.zero_grad()
        logits = model(xb)
        loss   = loss_fn(logits.reshape(-1, actual_vocab), yb.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        train_losses.append(lv)
        if device == "cuda":
            torch.cuda.synchronize()
        timestamps.append(time.perf_counter() - t0)

        if step % val_every == 0:
            vl  = evaluate(model, val_loader, actual_vocab)
            ppl = math.exp(min(vl, 20))
            val_losses.append(vl)
            val_ppls.append((timestamps[-1], ppl))
            print(f"    [{opt_name}] step {step:5d}/{steps}  "
                  f"train={lv:.4f}  val={vl:.4f}  ppl={ppl:.2f}  "
                  f"t={timestamps[-1]:.1f}s")

    elapsed = time.perf_counter() - t0
    final_val = evaluate(model, val_loader, actual_vocab)
    tokens_per_sec = steps * batch_size * cfg.seq_len / elapsed

    # --- Memory measurement ---
    # GPU: peak allocated (accurate)
    # CPU: exact accounting — params + grads + optimizer moment tensors +
    #      SparsePreconditioner internal tensors (nested in preconditioner object,
    #      bypassed by isinstance(v, Tensor) — use memory_bytes() method instead).
    if device == "cuda":
        torch.cuda.synchronize()
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        _, tracemalloc_peak = tracemalloc.get_traced_memory()
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        grad_bytes  = param_bytes  # one gradient tensor per param (same size)
        opt_bytes = 0
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    opt_bytes += v.numel() * v.element_size()
                elif hasattr(v, "memory_bytes"):
                    # SparsePreconditioner stores tensors in nested Python object;
                    # memory_bytes() sums L_ema + R_ema + U_l/S_l + U_r/S_r.
                    opt_bytes += v.memory_bytes()
        direct_bytes = param_bytes + grad_bytes + opt_bytes
        peak_mem_gb = max(direct_bytes, tracemalloc_peak) / (1024 ** 3)
    tracemalloc.stop()

    return {
        "optimizer":      opt_name,
        "scale":          cfg.label,
        "n_params":       n_params,
        "steps":          steps,
        "seed":           seed,
        "final_train":    train_losses[-1],
        "final_val":      final_val,
        "final_ppl":      math.exp(min(final_val, 20)),
        "avg_last_20":    sum(train_losses[-20:]) / min(20, len(train_losses)),
        "auc":            sum(train_losses) / len(train_losses),
        "total_time_s":   elapsed,
        "tokens_per_sec": tokens_per_sec,
        "peak_mem_gb":    peak_mem_gb,
        "val_curve":      val_ppls,  # list of (wall_clock_s, ppl)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-scale GPT benchmark: SCAO vs AdamW vs DiagShampoo"
    )
    parser.add_argument("--scale",      default="small",
                        choices=list(SCALE_CONFIGS.keys()),
                        help="Model size (default: small = 10M)")
    parser.add_argument("--scales",     type=str, default=None,
                        help="Comma-separated list of scales to run in sequence "
                             "(overrides --scale). E.g.: tiny,5m,small")
    parser.add_argument("--steps",      type=int,   default=0,
                        help="Training steps (0 = auto based on scale)")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--batch-size", type=int,   default=0,
                        help="Batch size (0 = auto based on scale)")
    parser.add_argument("--seeds",      type=str,   default="42",
                        help="Comma-separated seeds (default: 42)")
    parser.add_argument("--optimizers", type=str,   default="adamw,scao,scao_int8")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="LR for adamw and scao (default: 3e-4)")
    parser.add_argument("--diag-lr",    type=float, default=1e-3,
                        help="LR for diag_shampoo (default: 1e-3, needs higher than Adam)")
    parser.add_argument("--warmup",     type=int,   default=0,
                        help="Warmup steps (0 = auto = min(200, steps//5))")
    parser.add_argument("--csv",        type=str,   default=None,
                        help="Save results to this CSV path")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA not available, switching to CPU.")
        device = "cpu"

    # Determine which scales to run
    if args.scales:
        scale_list = [s.strip() for s in args.scales.split(",")]
    else:
        scale_list = [args.scale]

    seeds     = [int(s) for s in args.seeds.split(",")]
    opt_names = [o.strip() for o in args.optimizers.split(",")]

    AUTO_STEPS = {
        "tiny": 500, "5m": 500, "small": 800, "medium": 1000,
        "125m": 5000, "350m": 10000, "1b": 20000,
    }
    AUTO_BATCH = {
        "tiny": 16, "5m": 16, "small": 16, "medium": 8,
        "125m": 8,  "350m": 4, "1b": 2,
    }

    all_results: list[dict] = []

    for scale_key in scale_list:
        cfg        = SCALE_CONFIGS[scale_key]
        steps      = args.steps      or AUTO_STEPS[scale_key]
        batch_size = args.batch_size or AUTO_BATCH[scale_key]
        warmup_steps = args.warmup   or min(200, steps // 5)

        print(f"\n{'=' * 72}")
        print(f"  SCAO Multi-Scale Benchmark")
        print(f"  Scale: {scale_key} ({cfg.label} params)  "
              f"d={cfg.d_model}  L={cfg.n_layers}  H={cfg.n_head}")
        print(f"  device={device}  steps={steps}  batch={batch_size}  seeds={seeds}")
        print(f"  Optimizers: {', '.join(opt_names)}")
        print(f"{'=' * 72}\n")

        # Load data — seq_len may differ per scale so reload if it changes
        train_iter_fn, val_loader, actual_vocab = load_data(
            seq_len=cfg.seq_len,
            batch_size=batch_size,
            device=device,
            vocab_size=256,
        )

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            for opt_name in opt_names:
                print(f"\n  [{opt_name}]")
                r = run_single(
                    opt_name=opt_name,
                    cfg=cfg,
                    steps=steps,
                    device=device,
                    batch_size=batch_size,
                    seed=seed,
                    train_iter_fn=train_iter_fn,
                    val_loader=val_loader,
                    actual_vocab=actual_vocab,
                    lr=args.lr,
                    diag_lr=args.diag_lr,
                    warmup_steps=warmup_steps,
                )
                all_results.append(r)
                print(f"  -> PPL={r['final_ppl']:.2f}  "
                      f"tok/s={r['tokens_per_sec']:.0f}  "
                      f"mem={r['peak_mem_gb']:.3f}GB  "
                      f"t={r['total_time_s']:.1f}s")

    # --- Summary table ---
    from collections import defaultdict
    import statistics

    # Group by (scale, optimizer) across seeds
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        groups[(r["scale"], r["optimizer"])].append(r)

    w = 84
    print(f"\n{'=' * w}")
    print(f"  SUMMARY  |  scales={scale_list}  seeds={seeds}")
    print(f"{'-' * w}")
    print(f"  {'Scale':<6} {'Optimizer':<18} {'PPL mean':>9} {'PPL std':>8} "
          f"{'tok/s':>8} {'mem GB':>8}")
    print(f"{'-' * w}")

    for scale_key in scale_list:
        adamw_ppl: float | None = None
        for opt_name in opt_names:
            rs = groups[(SCALE_CONFIGS[scale_key].label, opt_name)]
            if not rs:
                continue
            ppls     = [r["final_ppl"] for r in rs]
            mean_ppl = statistics.mean(ppls)
            std_ppl  = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
            mean_tps = statistics.mean(r["tokens_per_sec"] for r in rs)
            mean_mem = statistics.mean(r["peak_mem_gb"] for r in rs)
            marker = ""
            if opt_name == "adamw":
                adamw_ppl = mean_ppl
            elif adamw_ppl and mean_ppl < adamw_ppl:
                marker = " ✓"
            print(f"  {SCALE_CONFIGS[scale_key].label:<6} {opt_name:<18} "
                  f"{mean_ppl:9.2f} {std_ppl:8.2f} "
                  f"{mean_tps:8.0f} {mean_mem:8.3f}{marker}")

    print(f"{'=' * w}\n")

    # --- Save CSV ---
    if args.csv:
        fieldnames = [k for k in all_results[0] if k != "val_curve"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                row = {k: v for k, v in r.items() if k != "val_curve"}
                writer.writerow(row)
        print(f"  Results saved to {args.csv}")

    # --- Save wall-clock vs PPL curve (for paper figures) ---
    curve_csv = (args.csv or "results").replace(".csv", "") + "_curves.csv"
    with open(curve_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "seed", "wall_clock_s", "ppl"])
        for r in all_results:
            for t, ppl in r["val_curve"]:
                writer.writerow([r["optimizer"], r["seed"], f"{t:.2f}", f"{ppl:.4f}"])
    print(f"  Wall-clock curves saved to {curve_csv}")


if __name__ == "__main__":
    main()
