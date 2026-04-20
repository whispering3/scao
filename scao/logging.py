"""
SCAO Logging & Monitoring
=========================
Zero-cost callback system for observing optimizer internals during training.

Usage
-----
    from scao import SCAO
    from scao.logging import SCAOLogger, ConsoleLogger, TensorBoardLogger

    opt = SCAO(model.parameters(), lr=1e-3)

    # Console logging every 100 steps
    opt.add_callback(ConsoleLogger(log_every=100))

    # TensorBoard logging
    opt.add_callback(TensorBoardLogger(writer, log_every=50))

    # Custom callback
    def my_callback(metrics: dict):
        wandb.log(metrics)
    opt.add_callback(my_callback)

Metrics dict keys
-----------------
    step              : int   global optimizer step
    scao/rank_mean    : float mean preconditioner rank across layers
    scao/rank_min     : int   minimum rank
    scao/rank_max     : int   maximum rank
    scao/layers       : int   number of preconditioned layers
    scao/L_norm_mean  : float mean Frobenius norm of L_ema (curvature health)
    scao/precond_freq : int   configured precond_freq
"""

from __future__ import annotations

import math
from typing import Callable

MetricsDict = dict[str, object]
Callback = Callable[[MetricsDict], None]


# ---------------------------------------------------------------------------
# Built-in loggers
# ---------------------------------------------------------------------------

class ConsoleLogger:
    """
    Prints SCAO metrics to stdout every `log_every` steps.

    Args:
        log_every: frequency of logging (default: 100 steps)
        prefix: string prepended to each log line
    """

    def __init__(self, log_every: int = 100, prefix: str = "[SCAO]") -> None:
        self.log_every = log_every
        self.prefix = prefix

    def __call__(self, metrics: MetricsDict) -> None:
        step = metrics.get("step", 0)
        if isinstance(step, int) and step % self.log_every != 0:
            return
        parts = [f"step={step}"]
        for k, v in metrics.items():
            if k == "step":
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(f"{self.prefix} " + "  ".join(parts))


class TensorBoardLogger:
    """
    Logs SCAO metrics to a TensorBoard SummaryWriter every `log_every` steps.

    Args:
        writer: ``torch.utils.tensorboard.SummaryWriter`` instance
        log_every: frequency of logging (default: 50 steps)
        tag_prefix: prefix for all scalar tags
    """

    def __init__(self, writer, log_every: int = 50, tag_prefix: str = "") -> None:
        self.writer = writer
        self.log_every = log_every
        self.tag_prefix = tag_prefix

    def __call__(self, metrics: MetricsDict) -> None:
        step = metrics.get("step", 0)
        if isinstance(step, int) and step % self.log_every != 0:
            return
        for k, v in metrics.items():
            if k == "step":
                continue
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                tag = f"{self.tag_prefix}{k}" if self.tag_prefix else k
                self.writer.add_scalar(tag, float(v), global_step=step)


class WandbLogger:
    """
    Logs SCAO metrics to Weights & Biases every `log_every` steps.

    Args:
        log_every: frequency of logging (default: 50 steps)
    """

    def __init__(self, log_every: int = 50) -> None:
        self.log_every = log_every

    def __call__(self, metrics: MetricsDict) -> None:
        step = metrics.get("step", 0)
        if isinstance(step, int) and step % self.log_every != 0:
            return
        try:
            import wandb  # type: ignore[import]
            wandb.log({k: v for k, v in metrics.items() if k != "step"},
                      step=step)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Metrics collection (called internally by SCAO.step())
# ---------------------------------------------------------------------------

def collect_metrics(optimizer) -> MetricsDict:
    """
    Collect current optimizer metrics from a SCAO instance.
    Called automatically if any callbacks are registered.

    Returns a MetricsDict with all standard SCAO keys.
    """
    from .optimizer import SCAO  # avoid circular import at module level

    step = 0
    ranks: list[int] = []
    l_norms: list[float] = []

    for state in optimizer.state.values():
        if "step" in state:
            step = max(step, state["step"])
        if "preconditioner" in state:
            prec = state["preconditioner"]
            ranks.append(prec.k)
            if prec.use_kronecker:
                l_norms.append(prec.L_ema.norm(p="fro").item())

    metrics: MetricsDict = {"step": step}
    if ranks:
        metrics["scao/rank_mean"] = sum(ranks) / len(ranks)
        metrics["scao/rank_min"] = min(ranks)
        metrics["scao/rank_max"] = max(ranks)
        metrics["scao/layers"] = len(ranks)
    if l_norms:
        metrics["scao/L_norm_mean"] = sum(l_norms) / len(l_norms)

    metrics["scao/precond_freq"] = optimizer.defaults.get("precond_freq", -1)

    return metrics
