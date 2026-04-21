#!/usr/bin/env python3
"""
bench_125m_350m.py — SCAO vs AdamW benchmarks at 125M and 350M parameters
==========================================================================
Convenience wrapper around scao/benchmarks/gpt_scale_benchmark.py.
Runs both scales with three optimizer variants:
  - adamw        : baseline
  - scao         : SCAO with float32 EMA accumulators
  - scao_int8    : SCAO with int8 EMA accumulators (4× EMA memory reduction)

Results are saved to:
  results_125m_350m.csv        — per-seed summary rows
  results_125m_350m_curves.csv — wall-clock vs PPL curves (for paper figures)
  report_125m_350m.txt         — human-readable summary

Usage:
    # Quick CPU smoke test (tiny steps, small batch):
    python scripts/bench_125m_350m.py --device cpu --steps 50

    # Full GPU run (requires ≥24 GB VRAM for 350m):
    python scripts/bench_125m_350m.py --device cuda

    # Single scale:
    python scripts/bench_125m_350m.py --device cuda --scales 125m

    # Multiple seeds for statistical significance:
    python scripts/bench_125m_350m.py --device cuda --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the full benchmark infrastructure from gpt_scale_benchmark.py
from scao.benchmarks.gpt_scale_benchmark import (   # noqa: E402
    SCALE_CONFIGS,
    load_data,
    run_single,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCAO vs AdamW at 125M and 350M parameter scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument(
        "--scales", default="125m,350m",
        help="Comma-separated scales to run (default: 125m,350m)",
    )
    parser.add_argument(
        "--optimizers", default="adamw,scao,scao_int8",
        help="Comma-separated optimizer names (default: adamw,scao,scao_int8)",
    )
    parser.add_argument("--device",  default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--steps",   type=int, default=0,
                        help="Training steps (0 = auto per scale)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Batch size (0 = auto per scale)")
    parser.add_argument("--seeds",   default="42",
                        help="Comma-separated random seeds (default: 42)")
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--warmup",  type=int, default=0,
                        help="Warmup steps (0 = auto)")
    parser.add_argument("--seq_len", type=int, default=0,
                        help="Override model seq_len (0 = use scale default). "
                             "Useful for CPU smoke tests, e.g. --seq_len 64")
    parser.add_argument("--csv",     default="results_125m_350m.csv")
    parser.add_argument("--report",  default="report_125m_350m.txt")
    parser.add_argument(
        "--out_dir", default="",
        help="Directory to write CSV and report files (default: current directory)",
    )
    return parser.parse_args()


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    args = parse_args()

    # Resolve output paths: prefix with --out_dir when provided
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not Path(args.csv).is_absolute():
            args.csv = str(out_dir / args.csv)
        if not Path(args.report).is_absolute():
            args.report = str(out_dir / args.report)

    scales     = [s.strip() for s in args.scales.split(",")]
    opt_names  = [o.strip() for o in args.optimizers.split(",")]
    seeds      = [int(s) for s in args.seeds.split(",")]
    device     = args.device

    # Per-scale defaults (matches gpt_scale_benchmark AUTO_STEPS / AUTO_BATCH)
    AUTO_STEPS = {"tiny": 500, "5m": 500, "small": 800, "medium": 1000,
                  "125m": 5000, "350m": 10000, "1b": 20000}
    AUTO_BATCH = {"tiny": 16,  "5m": 16,  "small": 16,  "medium": 8,
                  "125m": 8,   "350m": 4,  "1b": 2}

    header = "=" * 72
    print(f"\n{header}")
    print(f"  SCAO Large-Scale Benchmark: {', '.join(scales)}")
    print(f"  device={device}  optimizers={', '.join(opt_names)}  seeds={seeds}")
    print(f"{header}\n")

    all_results: list[dict] = []
    wall_curves: list[tuple] = []  # (opt, seed, t, ppl)

    for scale_key in scales:
        if scale_key not in SCALE_CONFIGS:
            print(f"  WARN: unknown scale '{scale_key}', skipping.")
            continue

        cfg        = SCALE_CONFIGS[scale_key]
        steps      = args.steps      or AUTO_STEPS[scale_key]
        batch_size = args.batch_size or AUTO_BATCH[scale_key]
        warmup_steps = args.warmup   or min(200, steps // 5)

        # Allow CPU smoke tests with reduced sequence length
        import copy
        if args.seq_len > 0:
            cfg = copy.copy(cfg)
            cfg = type(cfg)(
                d_model=cfg.d_model, n_layers=cfg.n_layers, n_head=cfg.n_head,
                d_ff=cfg.d_ff, seq_len=args.seq_len, label=cfg.label,
            )

        print(f"\n{'─' * 72}")
        print(f"  Scale: {scale_key} ({cfg.label} params)"
              f"  d={cfg.d_model}  L={cfg.n_layers}  H={cfg.n_head}")
        print(f"  steps={steps}  batch={batch_size}  warmup={warmup_steps}")
        print(f"{'─' * 72}")

        train_iter_fn, val_loader, actual_vocab = load_data(
            seq_len=cfg.seq_len,
            batch_size=batch_size,
            device=device,
            vocab_size=256,
        )

        for seed in seeds:
            for opt_name in opt_names:
                print(f"\n  [{scale_key}] seed={seed}  optimizer={opt_name}")
                t_start = time.perf_counter()
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
                    diag_lr=args.lr,
                    warmup_steps=warmup_steps,
                )
                all_results.append(r)
                for t, ppl in r["val_curve"]:
                    wall_curves.append((opt_name, seed, t, ppl))

                elapsed = time.perf_counter() - t_start
                print(f"  → PPL={r['final_ppl']:.2f}  "
                      f"tok/s={r['tokens_per_sec']:.0f}  "
                      f"mem={r['peak_mem_gb']:.3f} GB  "
                      f"wall={elapsed:.0f}s")

    if not all_results:
        print("No results collected.")
        return

    # ── Summary table ─────────────────────────────────────────────────────
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        groups[(r["scale"], r["optimizer"])].append(r)

    w = 88
    lines: list[str] = []
    lines.append("=" * w)
    lines.append(f"  SCAO Large-Scale Benchmark Results")
    lines.append(f"  Scales: {scales}   Seeds: {seeds}   Device: {device}")
    lines.append("─" * w)
    lines.append(
        f"  {'Scale':<6} {'Optimizer':<18} {'PPL':>8} {'±':>6} "
        f"{'tok/s':>8} {'mem GB':>8} {'vs AdamW':>10}"
    )
    lines.append("─" * w)

    for scale_key in scales:
        if scale_key not in SCALE_CONFIGS:
            continue
        cfg = SCALE_CONFIGS[scale_key]
        adamw_ppl: float | None = None
        for opt_name in opt_names:
            rs = groups.get((cfg.label, opt_name), [])
            if not rs:
                continue
            ppls     = [r["final_ppl"] for r in rs]
            mean_ppl = statistics.mean(ppls)
            std_ppl  = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
            mean_tps = statistics.mean(r["tokens_per_sec"] for r in rs)
            mean_mem = statistics.mean(r["peak_mem_gb"] for r in rs)

            vs_col = ""
            if opt_name == "adamw":
                adamw_ppl = mean_ppl
            elif adamw_ppl is not None:
                delta_pct = (mean_ppl - adamw_ppl) / adamw_ppl * 100
                marker = "✓ " if mean_ppl < adamw_ppl else "  "
                vs_col = f"{marker}{delta_pct:+.1f}%"

            lines.append(
                f"  {cfg.label:<6} {opt_name:<18} {mean_ppl:8.2f} {std_ppl:6.2f} "
                f"{mean_tps:8.0f} {mean_mem:8.3f} {vs_col:>10}"
            )
    lines.append("=" * w)

    # int8 memory savings summary
    for scale_key in scales:
        if scale_key not in SCALE_CONFIGS:
            continue
        cfg = SCALE_CONFIGS[scale_key]
        scao_rs     = groups.get((cfg.label, "scao"), [])
        scao_i8_rs  = groups.get((cfg.label, "scao_int8"), [])
        if scao_rs and scao_i8_rs:
            mem_fp32 = statistics.mean(r["peak_mem_gb"] for r in scao_rs)
            mem_int8 = statistics.mean(r["peak_mem_gb"] for r in scao_i8_rs)
            saving   = (mem_fp32 - mem_int8) / mem_fp32 * 100 if mem_fp32 > 0 else 0
            lines.append(
                f"  {cfg.label}: int8 EMA saves {saving:.1f}% peak memory "
                f"({mem_fp32:.3f} → {mem_int8:.3f} GB)"
            )
    lines.append("")

    report_str = "\n".join(lines)
    print(f"\n{report_str}")

    # ── Write report ──────────────────────────────────────────────────────
    report_path = Path(args.report)
    report_path.write_text(report_str, encoding="utf-8")
    print(f"  Report saved → {report_path}")

    # ── Write CSVs ────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    fieldnames = [k for k in all_results[0] if k != "val_curve"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: v for k, v in r.items() if k != "val_curve"})
    print(f"  Results CSV → {csv_path}")

    curve_path = Path(str(csv_path).replace(".csv", "_curves.csv"))
    with curve_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "seed", "wall_clock_s", "ppl"])
        for opt_name, seed, t, ppl in wall_curves:
            writer.writerow([opt_name, seed, f"{t:.2f}", f"{ppl:.4f}"])
    print(f"  Curve CSV   → {curve_path}")


if __name__ == "__main__":
    main()
