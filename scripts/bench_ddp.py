#!/usr/bin/env python3
"""
bench_ddp.py — Multi-GPU SCAO vs AdamW benchmark with DDP
==========================================================
Launched with torchrun (recommended) or torch.multiprocessing.spawn.

Usage — single node, all visible GPUs:
    torchrun --nproc_per_node=auto scripts/bench_ddp.py

Usage — single node, specific GPU count:
    torchrun --nproc_per_node=2 scripts/bench_ddp.py --scale 125m --steps 1000

Usage — multi-node (2 nodes, 4 GPUs each):
    # On node 0 (master):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\
             --master_addr=<NODE0_IP> --master_port=29500 \\
             scripts/bench_ddp.py --scale 125m

    # On node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\
             --master_addr=<NODE0_IP> --master_port=29500 \\
             scripts/bench_ddp.py --scale 125m

CPU-only test (gloo backend):
    torchrun --nproc_per_node=2 scripts/bench_ddp.py \\
             --device cpu --steps 20 --scale tiny

Outputs (rank 0 only):
    results_ddp_<scale>.csv        — per-step metrics
    results_ddp_<scale>_curves.csv — wall-clock vs PPL curves
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scao import SCAO
from scao.benchmarks.gpt_scale_benchmark import (
    SCALE_CONFIGS,
    GPTModel,
    load_data,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU SCAO DDP benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scale", default="125m",
        choices=list(SCALE_CONFIGS.keys()),
        help="Model scale to benchmark",
    )
    parser.add_argument(
        "--optimizers", default="adamw,scao,scao_int8",
        help="Comma-separated list of optimizers to benchmark",
    )
    parser.add_argument("--steps", type=int, default=0,
                        help="Training steps (0 = auto per scale)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Per-GPU batch size (0 = auto per scale)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=0,
                        help="Warmup steps (0 = auto)")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated seeds")
    parser.add_argument("--device", default="",
                        help="Device: cuda or cpu (default: cuda if available)")
    parser.add_argument("--out_dir", default="",
                        help="Output directory for CSV files")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DDP setup / teardown
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, str]:
    """Initialise the process group and return (rank, world_size, device)."""
    # torchrun sets LOCAL_RANK / RANK / WORLD_SIZE automatically.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        backend = "gloo"
        device = "cpu"

    dist.init_process_group(backend=backend)
    return rank, world_size, device


def teardown_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Training loop for one (optimizer, seed) combination
# ---------------------------------------------------------------------------

def _make_optimizer(name: str, params, lr: float, warmup: int) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.1)
    if name == "scao":
        return SCAO(params, lr=lr, weight_decay=0.1,
                    warmup_steps=warmup, async_precond=False)
    if name == "scao_int8":
        return SCAO(params, lr=lr, weight_decay=0.1,
                    warmup_steps=warmup, async_precond=False, use_int8_ema=True)
    raise ValueError(f"Unknown optimizer: {name!r}")


def run_one(
    *,
    opt_name: str,
    seed: int,
    cfg,
    steps: int,
    batch_size: int,
    warmup_steps: int,
    lr: float,
    device: str,
    rank: int,
    world_size: int,
) -> tuple[dict, list[tuple[float, float]]]:
    """Train one (optimizer, seed) combination; return (summary_dict, curve)."""
    torch.manual_seed(seed + rank)

    # Build model and wrap with DDP
    train_iter_fn, val_loader, vocab_size = load_data(
        seq_len=cfg.seq_len, batch_size=batch_size, device=device
    )
    model = GPTModel(cfg, vocab_size=vocab_size).to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(device.split(":")[-1])] if "cuda" in device else None,
        )

    optimizer = _make_optimizer(opt_name, model.parameters(), lr, warmup_steps)

    criterion = nn.CrossEntropyLoss()
    train_iter = train_iter_fn()
    curve: list[tuple[float, float]] = []  # (wall_clock_s, ppl)
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = train_iter_fn()
            xb, yb = next(train_iter)

        logits = model(xb)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % max(1, steps // 20) == 0 and rank == 0:
            ppl = loss.exp().item()
            curve.append((time.perf_counter() - t0, ppl))

    # Validation PPL (rank 0 only)
    val_loss = 0.0
    val_batches = 0
    if rank == 0:
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += criterion(logits.view(-1, logits.size(-1)), yb.view(-1)).item()
                val_batches += 1
        model.train()

    val_ppl = (val_loss / max(val_batches, 1))
    val_ppl = 2 ** val_ppl if val_ppl > 0 else float("inf")

    # Peak memory
    if "cuda" in device:
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_mem_gb = 0.0

    # Tokens per second
    elapsed = time.perf_counter() - t0
    tokens_per_sec = steps * batch_size * world_size * cfg.seq_len / elapsed

    summary = {
        "scale": cfg.label,
        "optimizer": opt_name,
        "seed": seed,
        "world_size": world_size,
        "final_ppl": val_ppl,
        "tokens_per_sec": tokens_per_sec,
        "peak_mem_gb": peak_mem_gb,
        "elapsed_s": elapsed,
    }
    return summary, curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_ddp()

    if args.device:
        device = args.device

    cfg = SCALE_CONFIGS[args.scale]
    AUTO_STEPS  = {"tiny": 200, "5m": 500, "small": 800, "medium": 1000,
                   "125m": 3000, "350m": 6000, "1b": 15000}
    AUTO_BATCH  = {"tiny": 8, "5m": 8, "small": 8, "medium": 4,
                   "125m": 4, "350m": 2, "1b": 1}

    steps      = args.steps      or AUTO_STEPS[args.scale]
    batch_size = args.batch_size or AUTO_BATCH[args.scale]
    warmup_steps = args.warmup   or min(200, steps // 5)
    opt_names  = [o.strip() for o in args.optimizers.split(",")]
    seeds      = [int(s) for s in args.seeds.split(",")]

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  SCAO DDP Benchmark — scale={args.scale}  world_size={world_size}")
        print(f"  device={device}  steps={steps}  batch/gpu={batch_size}")
        print(f"  optimizers={opt_names}  seeds={seeds}")
        print(f"{'='*70}\n")

    all_results: list[dict] = []
    all_curves: list[tuple] = []

    for opt_name in opt_names:
        for seed in seeds:
            if rank == 0:
                print(f"  {opt_name:18s} seed={seed} ...")

            summary, curve = run_one(
                opt_name=opt_name,
                seed=seed,
                cfg=cfg,
                steps=steps,
                batch_size=batch_size,
                warmup_steps=warmup_steps,
                lr=args.lr,
                device=device,
                rank=rank,
                world_size=world_size,
            )

            if rank == 0:
                all_results.append(summary)
                for t, ppl in curve:
                    all_curves.append((opt_name, seed, t, ppl))
                print(f"    → PPL={summary['final_ppl']:.2f}  "
                      f"tok/s={summary['tokens_per_sec']:.0f}  "
                      f"mem={summary['peak_mem_gb']:.3f} GB")

    # ── Write results (rank 0 only) ──────────────────────────────────────────
    if rank == 0:
        out_dir = Path(args.out_dir) if args.out_dir else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"results_ddp_{args.scale}.csv"
        if all_results:
            fieldnames = list(all_results[0].keys())
            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(all_results)
            print(f"\n  Results → {csv_path}")

        curves_path = out_dir / f"results_ddp_{args.scale}_curves.csv"
        with curves_path.open("w", newline="") as f:
            w2 = csv.writer(f)
            w2.writerow(["optimizer", "seed", "wall_clock_s", "ppl"])
            for row in all_curves:
                w2.writerow(row)
        print(f"  Curves  → {curves_path}")

    teardown_ddp()


if __name__ == "__main__":
    main()
