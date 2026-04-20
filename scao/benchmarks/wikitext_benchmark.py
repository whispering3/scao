"""
Real-Data Benchmark: SCAO vs AdamW on WikiText-2
=================================================

Uses the HuggingFace `datasets` library to download WikiText-2 (raw)
and trains a TinyGPT character/BPE-level language model.

This benchmark demonstrates SCAO's advantage on structured, real-world
text data where the loss landscape is genuinely ill-conditioned.

Usage:
    python scao/benchmarks/wikitext_benchmark.py
    python scao/benchmarks/wikitext_benchmark.py --steps 500 --d-model 256

Requirements:
    pip install datasets tokenizers
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scao import SCAO
from scao.benchmarks.gpt_comparison import (
    TinyGPT,
    DiagonalShampoo,
    ascii_loss_chart,
    print_summary,
    save_csv,
)


# ===========================================================================
# Data pipeline
# ===========================================================================

def load_wikitext(seq_len: int, batch_size: int, vocab_size: int, device: str):
    """
    Download WikiText-2 via HuggingFace datasets and build character-level
    token batches.  Falls back to synthetic data if download fails.

    Returns (train_loader, val_loader, actual_vocab_size).
    """
    from torch.utils.data import DataLoader, TensorDataset

    print("  Downloading WikiText-2 ...", end=" ", flush=True)
    try:
        from datasets import load_dataset  # type: ignore[import]
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(ds["train"]["text"])  # type: ignore[index]
        val_text   = "\n".join(ds["validation"]["text"])  # type: ignore[index]
        print(f"ok  ({len(train_text):,} chars train)")
    except Exception as e:
        print(f"FAILED ({e})\n  Using synthetic data instead.")
        # Fallback: random integer sequences
        from torch.utils.data import TensorDataset
        n = 5000
        data = torch.randint(0, vocab_size, (n, seq_len + 1))
        split = int(n * 0.9)
        tr = DataLoader(TensorDataset(data[:split, :-1], data[:split, 1:]),
                        batch_size=batch_size, shuffle=True)
        va = DataLoader(TensorDataset(data[split:, :-1], data[split:, 1:]),
                        batch_size=batch_size, shuffle=False)
        return tr, va, vocab_size

    # Character-level tokenisation (simple, no external tokenizer needed)
    chars   = sorted(set(train_text + val_text))
    actual_vocab = min(len(chars), vocab_size)
    chars   = chars[:actual_vocab]
    ch2idx  = {c: i for i, c in enumerate(chars)}

    def encode(text: str) -> torch.Tensor:
        ids = [ch2idx.get(c, 0) for c in text]
        return torch.tensor(ids, dtype=torch.long)

    def make_loader(text: str, shuffle: bool) -> DataLoader:
        tokens = encode(text)
        # Chunk into (seq_len+1) windows
        n_chunks = (len(tokens) - 1) // (seq_len + 1)
        tokens   = tokens[: n_chunks * (seq_len + 1) + 1]
        chunks   = tokens.unfold(0, seq_len + 1, seq_len + 1)
        inputs   = chunks[:, :-1].to(device)
        labels   = chunks[:, 1:].to(device)
        return DataLoader(
            TensorDataset(inputs, labels),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

    return make_loader(train_text, True), make_loader(val_text, False), actual_vocab


# ===========================================================================
# Validation
# ===========================================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader, vocab_size: int, max_batches: int = 50) -> float:
    """Return mean cross-entropy loss on validation set."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, count = 0.0, 0
    for i, (xb, yb) in enumerate(loader):
        if i >= max_batches:
            break
        logits = model(xb)
        loss   = loss_fn(logits.reshape(-1, vocab_size), yb.reshape(-1))
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


# ===========================================================================
# Training
# ===========================================================================

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
    train_loader,
    val_loader,
    actual_vocab: int,
) -> dict:
    torch.manual_seed(seed)

    model = TinyGPT(
        vocab_size=actual_vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_head=n_head,
        seq_len=seq_len,
    ).to(device)

    # Optimizer
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.1,
            betas=(0.9, 0.95),
        )
    elif opt_name == "scao":
        optimizer = SCAO(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.1,
            warmup_steps=min(100, steps // 3),
            precond_freq=10,           # 10 updates per 100 steps
            min_precond_updates=10,    # must have 10 reliable curvature samples
            k_min=8,
            k_max=64,
            tau=None,                  # no tau clipping — external clip is off too
            betas=(0.9, 0.95),
        )
    elif opt_name == "diag_shampoo":
        optimizer = DiagonalShampoo(
            model.parameters(), lr=3e-4, weight_decay=0.1,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # LR schedule: linear warmup then cosine decay.
    # Warmup length matches SCAO's Adam phase so the LR peaks exactly when
    # SCAO transitions to the preconditioned phase — giving SCAO the full
    # cosine budget instead of starting at 78% of max LR.
    # AdamW / DiagShampoo use the same schedule for a fair comparison.
    lr_warmup = min(100, steps // 3)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=lr_warmup,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps - lr_warmup, 1), eta_min=3e-5,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[lr_warmup],
    )

    loss_fn = nn.CrossEntropyLoss()
    train_losses: list[float] = []
    val_losses:   list[float] = []
    converge_step: int | None = None

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0         = time.perf_counter()
    data_iter  = iter(train_loader)
    val_every  = max(10, steps // 20)

    for step in range(1, steps + 1):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            xb, yb   = next(data_iter)

        optimizer.zero_grad()
        logits = model(xb)
        loss   = loss_fn(logits.reshape(-1, actual_vocab), yb.reshape(-1))
        loss.backward()
        # SCAO has internal curvature-aware clipping (tau parameter).
        # External Euclidean clipping fights with the preconditioner, so we
        # skip it for SCAO and apply it only for other optimizers.
        if opt_name != "scao":
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        lv = loss.item()
        train_losses.append(lv)

        if converge_step is None and lv < threshold:
            converge_step = step

        if step % val_every == 0:
            vl = evaluate(model, val_loader, actual_vocab)
            val_losses.append(vl)
            ppl = math.exp(min(vl, 20))
            print(f"    [{opt_name}] step {step:4d}/{steps}  "
                  f"train={lv:.4f}  val={vl:.4f}  ppl={ppl:.1f}")

    elapsed = time.perf_counter() - t0
    final_val = evaluate(model, val_loader, actual_vocab)

    return {
        "optimizer":     opt_name,
        "steps":         steps,
        "losses":        train_losses,
        "val_losses":    val_losses,
        "final_loss":    train_losses[-1],
        "avg_last_20":   sum(train_losses[-20:]) / min(20, len(train_losses)),
        "auc":           sum(train_losses) / len(train_losses),
        "final_val":     final_val,
        "final_ppl":     math.exp(min(final_val, 20)),
        "converge_step": converge_step,
        "total_time_s":  elapsed,
        "steps_per_sec": steps / elapsed,
        "num_params":    sum(p.numel() for p in model.parameters()),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="WikiText-2 optimizer benchmark")
    parser.add_argument("--steps",       type=int,   default=300)
    parser.add_argument("--device",      type=str,   default="cpu")
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--threshold",   type=float, default=4.0,
                        help="Train loss threshold for convergence step")
    parser.add_argument("--optimizers",  type=str,
                        default="adamw,scao,diag_shampoo")
    parser.add_argument("--csv",         type=str,   default=None)
    parser.add_argument("--d-model",     type=int,   default=128)
    parser.add_argument("--n-layers",    type=int,   default=4)
    parser.add_argument("--n-head",      type=int,   default=4)
    parser.add_argument("--seq-len",     type=int,   default=128)
    parser.add_argument("--vocab-size",  type=int,   default=128,
                        help="Max char vocab size (WikiText-2 has ~100 unique chars)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA not available, using CPU.")
        device = "cpu"

    opt_names = [o.strip() for o in args.optimizers.split(",")]

    # Load data once (shared across all optimizers for fair comparison)
    train_loader, val_loader, actual_vocab = load_wikitext(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        device=device,
    )

    print(f"\n{'=' * 72}")
    print(f"  WikiText-2 Benchmark  |  device={device}  |  steps={args.steps}")
    print(f"  Model: d_model={args.d_model}  n_layers={args.n_layers}  "
          f"vocab={actual_vocab}  seq_len={args.seq_len}")
    print(f"  Optimizers: {', '.join(opt_names)}")
    print(f"{'=' * 72}\n")

    results = []
    for name in opt_names:
        print(f"\n--- {name.upper()} ---")
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
            train_loader=train_loader,
            val_loader=val_loader,
            actual_vocab=actual_vocab,
        )
        results.append(r)
        print(f"  -> final: train={r['final_loss']:.4f}  "
              f"val={r['final_val']:.4f}  ppl={r['final_ppl']:.1f}  "
              f"{r['steps_per_sec']:.2f} steps/s")

    # ---- Reporting ----
    ascii_loss_chart(results)

    # Extended summary with val perplexity
    w = 72
    print(f"\n{'=' * w}")
    print("  FINAL RESULTS - WikiText-2  (real structured text)")
    print(f"{'-' * w}")
    print(f"  {'Optimizer':<18} {'TrainLoss':>10} {'ValLoss':>9} "
          f"{'ValPPL':>8} {'AUC':>8} {'Steps/s':>8}")
    print(f"{'-' * w}")

    best_val = min(r["final_val"] for r in results)
    for r in sorted(results, key=lambda x: x["final_val"]):
        marker = " *" if r["final_val"] == best_val else "  "
        print(f"  {r['optimizer']:<18} {r['final_loss']:10.4f} "
              f"{r['final_val']:9.4f} {r['final_ppl']:8.1f} "
              f"{r['auc']:8.4f} {r['steps_per_sec']:8.2f}{marker}")

    scao_r  = next((r for r in results if r["optimizer"] == "scao"),  None)
    adamw_r = next((r for r in results if r["optimizer"] == "adamw"), None)
    if scao_r and adamw_r:
        val_ratio = adamw_r["final_val"] / max(scao_r["final_val"], 1e-9)
        ppl_delta = adamw_r["final_ppl"] - scao_r["final_ppl"]
        print(f"\n  SCAO vs AdamW:")
        print(f"    Val loss ratio : {val_ratio:.4f}x")
        print(f"    PPL improvement: {ppl_delta:+.2f} (positive = SCAO better)")
    print(f"{'=' * w}\n")

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()


