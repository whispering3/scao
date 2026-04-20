"""
Synthetic Benchmark Suite: SCAO vs AdamW vs DiagShampoo
========================================================

Runs without any external datasets (no HuggingFace, no internet).
Four synthetic data profiles exercise different aspects of optimizer quality:

  profile 1 — markov_text
      Bigram Markov chain over a 128-token vocabulary.  Token transition matrix
      is drawn from a Dirichlet distribution, giving structured (non-uniform)
      next-token probabilities.  Mimics word co-occurrence without real text.

  profile 2 — zipf_lm
      Language-model sequences whose token frequencies follow Zipf's law
      (freq ~ 1/rank^1.1).  Head tokens appear very often; tail tokens rarely.
      Forces the optimizer to handle gradient variance across embedding rows.

  profile 3 — ill_conditioned_regression
      Linear regression y = X @ W* + noise where the condition number of X^T X
      is set to `kappa` (default 1000).  Ground-truth W* is known, so we can
      measure ||W - W*|| directly.  Tests whether second-order methods exploit
      curvature more efficiently than Adam.

  profile 4 — noisy_periodic
      Sequences generated from a sum of sinusoids with iid noise.  Tests
      optimizer stability when the gradient signal oscillates.

For each profile, the benchmark reports:
  - Final validation loss / PPL
  - Wall-clock seconds for the full run
  - Per-step timing: mean ± std (ms)
  - Per-phase timing: Adam warmup vs SCAO phase (SCAO only)
  - Preconditioner overhead: % of total SCAO time
  - Peak memory: RSS (MB) via tracemalloc, GPU (MB) via torch.cuda

Usage:
    python scao/benchmarks/synthetic_benchmark.py
    python scao/benchmarks/synthetic_benchmark.py --profile zipf_lm --steps 500
    python scao/benchmarks/synthetic_benchmark.py --all --csv results_synthetic.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import Iterator, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scao import SCAO


# ===========================================================================
# Profiling helpers
# ===========================================================================

class StepTimer:
    """Accumulates per-step wall-clock timings."""

    def __init__(self) -> None:
        self._times: list[float] = []
        self._t0: float = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._times.append((time.perf_counter() - self._t0) * 1000)  # ms

    @property
    def mean_ms(self) -> float:
        return sum(self._times) / max(len(self._times), 1)

    @property
    def std_ms(self) -> float:
        if len(self._times) < 2:
            return 0.0
        mu = self.mean_ms
        return math.sqrt(sum((t - mu) ** 2 for t in self._times) / (len(self._times) - 1))

    @property
    def p95_ms(self) -> float:
        if not self._times:
            return 0.0
        s = sorted(self._times)
        idx = int(0.95 * len(s))
        return s[idx]

    @property
    def total_s(self) -> float:
        return sum(self._times) / 1000.0

    def reset(self) -> None:
        self._times.clear()


class PhaseTimer:
    """Tracks time spent in Phase 1 (Adam) vs Phase 2 (SCAO) for SCAO optimizer."""

    def __init__(self) -> None:
        self.phase1_ms: float = 0.0
        self.phase2_ms: float = 0.0
        self.precond_ms: float = 0.0   # eigendecomp + EMA update time
        self._t0: float = 0.0
        self._phase: int = 0           # 1 or 2
        self._precond_t0: float = 0.0

    def begin_step(self, is_phase2: bool) -> None:
        self._phase = 2 if is_phase2 else 1
        self._t0 = time.perf_counter()

    def end_step(self) -> None:
        dt = (time.perf_counter() - self._t0) * 1000
        if self._phase == 1:
            self.phase1_ms += dt
        else:
            self.phase2_ms += dt

    def begin_precond(self) -> None:
        self._precond_t0 = time.perf_counter()

    def end_precond(self) -> None:
        self.precond_ms += (time.perf_counter() - self._precond_t0) * 1000


@contextmanager
def memory_snapshot(label: str = ""):
    """Context manager that returns peak memory increase in MB (RSS via tracemalloc)."""
    gc.collect()
    tracemalloc.start()
    peak_before = tracemalloc.get_traced_memory()[1]
    try:
        yield
    finally:
        _, peak_after = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        delta_mb = (peak_after - peak_before) / (1024 ** 2)
        # stored on the context object for the caller
        memory_snapshot.last_delta_mb = delta_mb


@dataclass
class RunResult:
    optimizer:         str
    profile:           str
    steps:             int
    seed:              int
    final_loss:        float
    final_ppl:         float           # exp(loss), or NaN for non-LM tasks
    best_loss:         float
    total_time_s:      float
    step_mean_ms:      float
    step_std_ms:       float
    step_p95_ms:       float
    phase1_time_s:     float           # 0 for non-SCAO
    phase2_time_s:     float           # 0 for non-SCAO
    precond_time_s:    float           # 0 for non-SCAO
    precond_overhead_pct: float        # precond_time / phase2_time * 100
    peak_mem_mb:       float           # tracemalloc RSS delta
    peak_gpu_mb:       float           # torch.cuda.max_memory_allocated (0 on CPU)
    loss_curve:        list[float] = field(default_factory=list, repr=False)


# ===========================================================================
# Synthetic data generators
# ===========================================================================

def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    import random; random.seed(seed)


def make_markov_text(
    n_tokens:   int = 128,
    seq_len:    int = 64,
    n_train:    int = 2000,
    n_val:      int = 400,
    seed:       int = 42,
    device:     str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates token sequences from a bigram Markov chain.

    Each row of the transition matrix T[i] is drawn from Dirichlet(alpha=0.5),
    creating a sparse, structured next-token distribution.  This gives the
    model genuine statistical structure to learn — not pure noise.

    Returns (train_x, train_y, val_x, val_y) with shapes (N, seq_len).
    """
    _seed_everything(seed)
    # Transition matrix: T[i, j] = P(next=j | current=i)
    alpha = torch.full((n_tokens,), 0.5)
    T = torch.distributions.Dirichlet(alpha).sample((n_tokens,))  # (V, V)

    def _sample_sequences(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Start tokens: uniform
        starts = torch.randint(0, n_tokens, (n,))
        seqs = torch.zeros(n, seq_len + 1, dtype=torch.long)
        seqs[:, 0] = starts
        for t in range(1, seq_len + 1):
            prev = seqs[:, t - 1]           # (n,)
            probs = T[prev]                  # (n, V)
            seqs[:, t] = torch.multinomial(probs, 1).squeeze(1)
        return seqs[:, :-1].to(device), seqs[:, 1:].to(device)

    train_x, train_y = _sample_sequences(n_train)
    val_x,   val_y   = _sample_sequences(n_val)
    return train_x, train_y, val_x, val_y


def make_zipf_lm(
    n_tokens:   int = 256,
    seq_len:    int = 64,
    n_train:    int = 2000,
    n_val:      int = 400,
    exponent:   float = 1.1,
    seed:       int = 42,
    device:     str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates sequences where token[t] ~ Zipf(exponent), independently.

    This creates a highly skewed gradient distribution: the embedding rows
    of head tokens receive dense gradient updates, tail tokens receive sparse
    updates.  Tests the optimizer's ability to handle heterogeneous curvature.
    """
    _seed_everything(seed)
    ranks = torch.arange(1, n_tokens + 1, dtype=torch.float)
    freqs = ranks.pow(-exponent)
    freqs = freqs / freqs.sum()

    def _sample(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.multinomial(
            freqs.expand(n * (seq_len + 1), -1),
            num_samples=1,
        ).reshape(n, seq_len + 1)
        return tokens[:, :-1].to(device), tokens[:, 1:].to(device)

    return _sample(n_train) + _sample(n_val)


def make_ill_conditioned_regression(
    in_dim:   int   = 64,
    out_dim:  int   = 16,
    n_train:  int   = 2000,
    n_val:    int   = 400,
    kappa:    float = 1000.0,
    noise:    float = 0.01,
    seed:     int   = 42,
    device:   str   = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Regression dataset  y = X @ W_star + noise,  where X^T X has condition
    number kappa (log-spaced singular values from 1 to kappa).

    Returns (train_x, train_y, val_x, val_y, W_star).
    W_star lets you measure ||W - W*|| directly.
    """
    _seed_everything(seed)
    # Build X with prescribed singular values
    U = torch.linalg.qr(torch.randn(n_train + n_val, in_dim))[0]  # (N, d)
    svals = torch.logspace(0, math.log10(math.sqrt(kappa)), in_dim)  # σ_1...σ_d
    X = U * svals.unsqueeze(0)                                        # (N, d)

    W_star = torch.randn(in_dim, out_dim) * 0.1
    Y = X @ W_star + noise * torch.randn(n_train + n_val, out_dim)

    train_x = X[:n_train].to(device)
    train_y = Y[:n_train].to(device)
    val_x   = X[n_train:].to(device)
    val_y   = Y[n_train:].to(device)
    W_star  = W_star.to(device)
    return train_x, train_y, val_x, val_y, W_star


def make_noisy_periodic(
    n_tokens:   int = 64,
    seq_len:    int = 32,
    n_train:    int = 2000,
    n_val:      int = 400,
    n_freqs:    int = 8,
    noise:      float = 0.3,
    seed:       int = 42,
    device:     str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Language-model sequences whose underlying signal is a sum of sinusoids
    + additive iid noise, then quantised to n_tokens bins.

    Frequency content creates structured auto-correlation.  Tests optimizer
    stability when gradient signal oscillates rather than monotonically decays.
    """
    _seed_everything(seed)
    N = n_train + n_val
    t = torch.linspace(0, 4 * math.pi, seq_len + 1).unsqueeze(0).expand(N, -1)

    # Random amplitudes and phases per frequency
    amps   = torch.rand(N, n_freqs, 1)
    phases = torch.rand(N, n_freqs, 1) * 2 * math.pi
    freqs  = torch.arange(1, n_freqs + 1, dtype=torch.float).view(1, n_freqs, 1)
    t_exp  = t.unsqueeze(1).expand(N, n_freqs, seq_len + 1)

    signal = (amps * torch.sin(freqs * t_exp + phases)).sum(1)  # (N, T+1)
    signal = signal + noise * torch.randn_like(signal)

    # Quantise to [0, n_tokens)
    mn, mx = signal.amin(), signal.amax()
    tokens = ((signal - mn) / (mx - mn + 1e-8) * (n_tokens - 1)).long().clamp(0, n_tokens - 1)

    train_x = tokens[:n_train, :-1].to(device)
    train_y = tokens[:n_train, 1:].to(device)
    val_x   = tokens[n_train:, :-1].to(device)
    val_y   = tokens[n_train:, 1:].to(device)
    return train_x, train_y, val_x, val_y


# ===========================================================================
# Models
# ===========================================================================

class TinyLM(nn.Module):
    """Minimal transformer LM: embedding + 2 transformer blocks + head."""

    def __init__(self, vocab: int, d: int = 128, n_layers: int = 2,
                 n_head: int = 4, seq_len: int = 64) -> None:
        super().__init__()
        self.vocab = vocab
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        # norm_first=False avoids the "enable_nested_tensor" warning in PyTorch 2.x;
        # is_causal handled via explicit additive mask in forward() — not via the
        # is_causal kwarg which conflicts with explicit mask in recent PyTorch.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_head, dim_feedforward=4 * d,
            dropout=0.0, batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        # Additive causal mask: upper triangle = -inf, diagonal+lower = 0.
        # Pass only mask= (no is_causal=True) to avoid double-masking in PyTorch 2.x.
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        h = self.encoder(h, mask=mask)
        return self.head(self.ln(h))


class LinearModel(nn.Module):
    """Simple linear layer for regression benchmark."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


# ===========================================================================
# Optimizer factory
# ===========================================================================

class DiagonalShampoo(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, weight_decay=0.0, rho=0.999):
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
                H, t = state["H"], state["t"]
                H.mul_(group["rho"]).addcmul_(g, g, value=1.0 - group["rho"])
                bias = 1.0 - group["rho"] ** t
                Hd = (H / bias).pow(0.25).add_(group["eps"])
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(g / Hd, alpha=-group["lr"])


def make_optimizer(name: str, params, lr: float, steps: int,
                   warmup_steps: int) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    if name == "scao":
        return SCAO(
            params, lr=lr, weight_decay=0.1, betas=(0.9, 0.95),
            warmup_steps=warmup_steps,
            precond_freq=max(5, steps // 80),
            min_precond_updates=5,
            k_min=4, k_max=32, tau=None,
        )
    if name == "diag_shampoo":
        return DiagonalShampoo(params, lr=lr * 3, weight_decay=0.01)
    raise ValueError(name)


def make_scheduler(opt, steps: int, warmup: int):
    ws = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    cs = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(steps - warmup, 1), eta_min=opt.defaults["lr"] * 0.05)
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[ws, cs], milestones=[warmup])


# ===========================================================================
# Detect if SCAO is in Phase 2 (for phase timer)
# ===========================================================================

def _scao_in_phase2(opt: SCAO) -> bool:
    """Heuristic: check if at least one param group is in SCAO phase."""
    for group in opt.param_groups:
        for p in group["params"]:
            if p in opt.state and opt.state[p].get("scao_phase_started", False):
                return True
    return False


# ===========================================================================
# Run a single (optimizer, profile) experiment
# ===========================================================================

def run_lm(
    opt_name:   str,
    train_x:    torch.Tensor,
    train_y:    torch.Tensor,
    val_x:      torch.Tensor,
    val_y:      torch.Tensor,
    vocab:      int,
    d:          int,
    seq_len:    int,
    steps:      int,
    batch_size: int,
    lr:         float,
    warmup:     int,
    seed:       int,
    device:     str,
    profile:    str,
) -> RunResult:
    torch.manual_seed(seed)
    model = TinyLM(vocab, d=d, n_layers=2, n_head=4, seq_len=seq_len).to(device)
    opt   = make_optimizer(opt_name, model.parameters(), lr, steps, warmup)
    sched = make_scheduler(opt, steps, warmup)
    loss_fn = nn.CrossEntropyLoss()

    timer   = StepTimer()
    p_timer = PhaseTimer()
    loss_curve: list[float] = []
    best_loss = float("inf")

    n_train = train_x.shape[0]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    gc.collect()
    tracemalloc.start()

    for step in range(1, steps + 1):
        idx  = torch.randint(0, n_train, (batch_size,))
        xb   = train_x[idx]
        yb   = train_y[idx]

        is_p2 = opt_name == "scao" and _scao_in_phase2(opt)

        timer.start()
        if opt_name == "scao":
            p_timer.begin_step(is_p2)

        opt.zero_grad()
        logits = model(xb)
        loss   = loss_fn(logits.reshape(-1, vocab), yb.reshape(-1))
        loss.backward()
        if opt_name != "scao":
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if opt_name == "scao":
            p_timer.end_step()
        timer.stop()

        lv = loss.item()
        loss_curve.append(lv)
        if lv < best_loss:
            best_loss = lv

    # Final validation loss
    model.eval()
    with torch.no_grad():
        v_logits = model(val_x)
        val_loss = loss_fn(v_logits.reshape(-1, vocab), val_y.reshape(-1)).item()
    model.train()

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 ** 2)

    peak_gpu_mb = 0.0
    if device == "cuda":
        torch.cuda.synchronize()
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    p2_s = p_timer.phase2_ms / 1000.0
    pre_s = p_timer.precond_ms / 1000.0
    overhead = (pre_s / p2_s * 100) if p2_s > 0 else 0.0

    return RunResult(
        optimizer=opt_name, profile=profile, steps=steps, seed=seed,
        final_loss=val_loss,
        final_ppl=math.exp(min(val_loss, 20)),
        best_loss=best_loss,
        total_time_s=timer.total_s,
        step_mean_ms=timer.mean_ms,
        step_std_ms=timer.std_ms,
        step_p95_ms=timer.p95_ms,
        phase1_time_s=p_timer.phase1_ms / 1000.0,
        phase2_time_s=p2_s,
        precond_time_s=pre_s,
        precond_overhead_pct=overhead,
        peak_mem_mb=peak_mem_mb,
        peak_gpu_mb=peak_gpu_mb,
        loss_curve=loss_curve,
    )


def run_regression(
    opt_name:   str,
    train_x:    torch.Tensor,
    train_y:    torch.Tensor,
    val_x:      torch.Tensor,
    val_y:      torch.Tensor,
    W_star:     torch.Tensor,
    in_dim:     int,
    out_dim:    int,
    steps:      int,
    batch_size: int,
    lr:         float,
    warmup:     int,
    seed:       int,
    device:     str,
) -> RunResult:
    torch.manual_seed(seed)
    model = LinearModel(in_dim, out_dim).to(device)
    opt   = make_optimizer(opt_name, model.parameters(), lr, steps, warmup)
    sched = make_scheduler(opt, steps, warmup)
    loss_fn = nn.MSELoss()

    timer   = StepTimer()
    p_timer = PhaseTimer()
    loss_curve: list[float] = []
    best_loss = float("inf")

    n_train = train_x.shape[0]

    gc.collect()
    tracemalloc.start()

    for step in range(1, steps + 1):
        idx  = torch.randint(0, n_train, (batch_size,))
        xb, yb = train_x[idx], train_y[idx]

        is_p2 = opt_name == "scao" and _scao_in_phase2(opt)
        timer.start()
        if opt_name == "scao":
            p_timer.begin_step(is_p2)

        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        if opt_name != "scao":
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if opt_name == "scao":
            p_timer.end_step()
        timer.stop()

        lv = loss.item()
        loss_curve.append(lv)
        if lv < best_loss:
            best_loss = lv

    # Residual ||W - W*||
    W_learned = model.layer.weight.detach().T  # (in_dim, out_dim)
    residual = (W_learned - W_star).norm().item()
    val_loss = loss_fn(model(val_x), val_y).item()

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 ** 2)

    p2_s  = p_timer.phase2_ms / 1000.0
    pre_s = p_timer.precond_ms / 1000.0
    overhead = (pre_s / p2_s * 100) if p2_s > 0 else 0.0

    r = RunResult(
        optimizer=opt_name, profile="ill_conditioned_regression",
        steps=steps, seed=seed,
        final_loss=val_loss,
        final_ppl=float("nan"),   # not a LM task
        best_loss=best_loss,
        total_time_s=timer.total_s,
        step_mean_ms=timer.mean_ms,
        step_std_ms=timer.std_ms,
        step_p95_ms=timer.p95_ms,
        phase1_time_s=p_timer.phase1_ms / 1000.0,
        phase2_time_s=p2_s,
        precond_time_s=pre_s,
        precond_overhead_pct=overhead,
        peak_mem_mb=peak_mem_mb,
        peak_gpu_mb=0.0,
        loss_curve=loss_curve,
    )
    r.__dict__["residual_norm"] = residual   # bonus metric
    return r


# ===========================================================================
# Print helpers
# ===========================================================================

W = 80

def _bar(label: str, val: float, best: float, width: int = 30) -> str:
    if best <= 0 or val <= 0:
        return ""
    ratio = min(val / best, 3.0)
    filled = int(ratio * width / 3)
    bar = "#" * filled + "-" * (width - filled)
    return f"  {label:<18} |{bar}| {val:.2f}"


def print_profile_summary(results: list[RunResult], opt_names: list[str]) -> None:
    import statistics
    from collections import defaultdict

    groups: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        groups[r.optimizer].append(r)

    print(f"\n{'=' * W}")
    print(f"  TIMING + MEMORY PROFILE  |  profile={results[0].profile if results else '?'}")
    print(f"{'-' * W}")
    print(f"  {'Optimizer':<18} {'PPL':>7} {'loss':>7} {'t/step(ms)':>11} "
          f"{'p95(ms)':>8} {'total(s)':>9} {'mem(MB)':>8}")
    print(f"{'-' * W}")

    ref_ppl = None
    for name in opt_names:
        rs = groups.get(name, [])
        if not rs:
            continue
        ppls  = [r.final_ppl for r in rs if not math.isnan(r.final_ppl)]
        losses = [r.final_loss for r in rs]
        mean_ppl  = statistics.mean(ppls) if ppls else float("nan")
        std_ppl   = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
        mean_loss = statistics.mean(losses)
        mean_ms   = statistics.mean(r.step_mean_ms for r in rs)
        mean_p95  = statistics.mean(r.step_p95_ms for r in rs)
        mean_t    = statistics.mean(r.total_time_s for r in rs)
        mean_mem  = statistics.mean(r.peak_mem_mb for r in rs)

        if name == "adamw":
            ref_ppl = mean_ppl
        marker = " *" if (ref_ppl and mean_ppl < ref_ppl) else ""

        ppl_str = f"{mean_ppl:.2f}" if not math.isnan(mean_ppl) else "  N/A"
        ppl_std = f"+-{std_ppl:.2f}" if len(ppls) > 1 else "      "
        print(f"  {name:<18} {ppl_str:>7}{ppl_std}  {mean_loss:>7.4f}  "
              f"{mean_ms:>8.2f}ms  {mean_p95:>6.2f}ms  {mean_t:>7.1f}s  "
              f"{mean_mem:>7.1f}MB{marker}")

    # Phase breakdown for SCAO
    scao_rs = groups.get("scao", [])
    if scao_rs:
        r = scao_rs[0]
        total = r.phase1_time_s + r.phase2_time_s
        if total > 0:
            p1_pct = r.phase1_time_s / total * 100
            p2_pct = r.phase2_time_s / total * 100
            pre_pct = r.precond_overhead_pct
            print(f"\n  SCAO phase breakdown (seed={r.seed}):")
            print(f"    Phase 1 (Adam warmup): {r.phase1_time_s:.2f}s  ({p1_pct:.1f}%)")
            print(f"    Phase 2 (SCAO):        {r.phase2_time_s:.2f}s  ({p2_pct:.1f}%)")
            print(f"    Precond overhead:      {r.precond_time_s:.2f}s  "
                  f"({pre_pct:.1f}% of Phase 2)")

    print(f"{'=' * W}")


# ===========================================================================
# Main
# ===========================================================================

PROFILES = ["markov_text", "zipf_lm", "ill_conditioned_regression", "noisy_periodic"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic benchmark: SCAO vs AdamW vs DiagShampoo"
    )
    parser.add_argument("--profile",    default="markov_text", choices=PROFILES + ["all"])
    parser.add_argument("--all",        action="store_true", help="Run all 4 profiles")
    parser.add_argument("--steps",      type=int,   default=300)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--d-model",    type=int,   default=128)
    parser.add_argument("--seeds",      type=str,   default="42,1337")
    parser.add_argument("--optimizers", type=str,   default="adamw,scao,diag_shampoo")
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--csv",        type=str,   default=None)
    parser.add_argument("--json",       type=str,   default=None)
    args = parser.parse_args()

    profiles = PROFILES if (args.all or args.profile == "all") else [args.profile]
    seeds    = [int(s) for s in args.seeds.split(",")]
    opt_names = [o.strip() for o in args.optimizers.split(",")]
    device   = args.device
    steps    = args.steps
    bs       = args.batch_size
    d        = args.d_model
    lr       = args.lr
    warmup   = min(100, steps // 4)

    all_results: list[RunResult] = []

    for profile in profiles:
        print(f"\n{'#' * W}")
        print(f"  PROFILE: {profile}  |  steps={steps}  d={d}  seeds={seeds}")
        print(f"{'#' * W}")

        # --- Build dataset (once per profile) ---
        if profile == "markov_text":
            seq_len = 64
            vocab   = 128
            train_x, train_y, val_x, val_y = make_markov_text(
                n_tokens=vocab, seq_len=seq_len, seed=0, device=device)
            is_regression = False
        elif profile == "zipf_lm":
            seq_len = 64
            vocab   = 256
            train_x, train_y, val_x, val_y = make_zipf_lm(
                n_tokens=vocab, seq_len=seq_len, seed=0, device=device)
            is_regression = False
        elif profile == "ill_conditioned_regression":
            in_dim, out_dim = d, d // 4
            train_x, train_y, val_x, val_y, W_star = make_ill_conditioned_regression(
                in_dim=in_dim, out_dim=out_dim, kappa=500.0, seed=0, device=device)
            is_regression = True
            vocab = seq_len = 0
        elif profile == "noisy_periodic":
            seq_len = 32
            vocab   = 64
            train_x, train_y, val_x, val_y = make_noisy_periodic(
                n_tokens=vocab, seq_len=seq_len, seed=0, device=device)
            is_regression = False

        profile_results: list[RunResult] = []

        for seed in seeds:
            print(f"\n  --- seed {seed} ---")
            for opt_name in opt_names:
                print(f"  [{opt_name}]", end=" ", flush=True)
                t0 = time.perf_counter()

                if is_regression:
                    r = run_regression(
                        opt_name, train_x, train_y, val_x, val_y, W_star,
                        in_dim, out_dim, steps, bs, lr, warmup, seed, device,
                    )
                else:
                    r = run_lm(
                        opt_name, train_x, train_y, val_x, val_y, vocab,
                        d, seq_len, steps, bs, lr, warmup, seed, device, profile,
                    )

                elapsed = time.perf_counter() - t0
                ppl_str = f"PPL={r.final_ppl:.2f}" if not math.isnan(r.final_ppl) else f"loss={r.final_loss:.4f}"
                res_str = ""
                if hasattr(r, "__dict__") and "residual_norm" in r.__dict__:
                    res_str = f"  ||W-W*||={r.__dict__['residual_norm']:.4f}"
                print(f"{ppl_str}  ms/step={r.step_mean_ms:.1f}+-{r.step_std_ms:.1f}"
                      f"  mem={r.peak_mem_mb:.1f}MB  t={elapsed:.1f}s{res_str}")

                profile_results.append(r)
                all_results.append(r)

        print_profile_summary(profile_results, opt_names)

    # --- Save CSV ---
    if args.csv:
        _save_csv(all_results, args.csv)
        print(f"\n  Results saved to {args.csv}")

    # --- Save JSON (includes loss curves) ---
    if args.json:
        rows = []
        for r in all_results:
            d_ = asdict(r)
            if hasattr(r, "__dict__") and "residual_norm" in r.__dict__:
                d_["residual_norm"] = r.__dict__["residual_norm"]
            rows.append(d_)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  Full results (with loss curves) saved to {args.json}")


def _save_csv(results: list[RunResult], path: str) -> None:
    skip = {"loss_curve"}
    rows = []
    for r in results:
        row = {k: v for k, v in asdict(r).items() if k not in skip}
        if hasattr(r, "__dict__") and "residual_norm" in r.__dict__:
            row["residual_norm"] = r.__dict__["residual_norm"]
        rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
