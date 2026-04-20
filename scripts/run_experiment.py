#!/usr/bin/env python3
"""
run_experiment.py — Reproducible SCAO experiment runner
========================================================

Loads a YAML config, trains the specified model with SCAO and baseline
optimizers, and writes results to logs/.

Usage
-----
    python scripts/run_experiment.py --config configs/base.yaml
    python scripts/run_experiment.py --config configs/gpt_small.yaml --seeds 42 123 7
    python scripts/run_experiment.py --config configs/base.yaml --optimizers adamw scao

All runs are fully deterministic for a given seed.
Results are saved to logs/<experiment_name>/<seed>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml  # type: ignore[import]
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from scao import SCAO


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    if not _HAS_YAML:
        raise ImportError("pip install pyyaml to use YAML configs")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Resolve _base_ inheritance
    if "_base_" in cfg:
        base_path = Path(path).parent / cfg.pop("_base_")
        base = load_config(str(base_path))
        base.update(cfg)
        cfg = base
    return cfg


def build_optimizer(name: str, params, cfg: dict) -> torch.optim.Optimizer:
    oc = cfg.get("optimizer", {})
    lr = float(oc.get("lr", 3e-4))
    wd = float(oc.get("weight_decay", 0.1))
    betas = tuple(oc.get("betas", [0.9, 0.95]))
    eps = float(oc.get("eps", 1e-8))

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    elif name == "scao":
        return SCAO(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=wd,
            precond_freq=int(oc.get("precond_freq", 10)),
            min_precond_updates=int(oc.get("min_precond_updates", 10)),
            warmup_steps=int(oc.get("warmup_steps", 100)),
            k_min=int(oc.get("k_min", 8)),
            k_max=int(oc.get("k_max", 128)),
            rho=float(oc.get("rho", 0.999)),
            tau=oc.get("tau", None),
            max_precond_dim=int(oc.get("max_precond_dim", 4096)),
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def save_run(out_dir: Path, opt_name: str, seed: int, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{opt_name}_seed{seed}.json"
    with open(fname, "w") as f:
        # Convert tensors / non-serialisable types
        clean = {k: (v.tolist() if hasattr(v, "tolist") else v)
                 for k, v in result.items()}
        json.dump(clean, f, indent=2)
    print(f"  Saved -> {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--optimizers", nargs="+", default=["adamw", "scao"])
    parser.add_argument("--seeds",     nargs="+", type=int, default=[42])
    parser.add_argument("--steps",     type=int, default=None,
                        help="Override config steps")
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--out-dir",   default="logs/")
    args = parser.parse_args()

    cfg = load_config(args.config) if _HAS_YAML else {}
    tc  = cfg.get("training", {})
    steps = args.steps or tc.get("steps", 300)
    device = args.device

    out_root = Path(args.out_dir) / Path(args.config).stem
    print(f"\n{'='*60}")
    print(f"  SCAO Experiment Runner")
    print(f"  Config : {args.config}")
    print(f"  Steps  : {steps}")
    print(f"  Seeds  : {args.seeds}")
    print(f"  Opts   : {args.optimizers}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    all_results = []
    for seed in args.seeds:
        for opt_name in args.optimizers:
            print(f"--- {opt_name.upper()} (seed={seed}) ---")
            set_seed(seed)

            # Minimal toy model for smoke test
            # Replace with real model from configs/
            model = nn.Sequential(
                nn.Linear(64, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 10),
            ).to(device)

            opt = build_optimizer(opt_name, model.parameters(), cfg)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

            losses, t0 = [], time.perf_counter()
            for step in range(1, steps + 1):
                x = torch.randn(32, 64, device=device)
                y = torch.randint(0, 10, (32,), device=device)
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                if opt_name != "scao":
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
                if step % max(1, steps // 10) == 0:
                    print(f"  step {step:4d}/{steps}  loss={loss.item():.4f}")

            elapsed = time.perf_counter() - t0
            result = {
                "optimizer": opt_name, "seed": seed,
                "final_loss": losses[-1],
                "avg_last10pct": sum(losses[-max(1, steps//10):]) / max(1, steps//10),
                "steps_per_sec": steps / elapsed,
                "total_time_s": elapsed,
                "losses": losses,
            }
            save_run(out_root / f"seed{seed}", opt_name, seed, result)
            all_results.append(result)

    # Summary CSV
    summary_path = out_root / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "seed", "final_loss",
                                               "avg_last10pct", "steps_per_sec"])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
