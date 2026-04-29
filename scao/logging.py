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
Standard (v1):
    step              : int   global optimizer step
    scao/rank_mean    : float mean preconditioner rank across layers
    scao/rank_min     : int   minimum rank
    scao/rank_max     : int   maximum rank
    scao/layers       : int   number of preconditioned layers
    scao/L_norm_mean  : float mean Frobenius norm of L_ema (curvature health)
                              NOTE: only available when use_int8_ema=False.
                              When int8 EMA is active (scao_40b / scao_125b
                              presets), this key is omitted rather than
                              returning a stale dequantized estimate.
    scao/precond_freq : int   configured precond_freq

New in v2:
    noise_std         : float current gradient noise injection std (annealed)
    global_norm_ema   : float slow EMA of mean per-layer gradient norm
                              (used by R2 dynamic sparsity and R5 adaptive rank)
    warmup_active     : bool  True while the adaptive warmup (R1) is still running
    scao/warmup_exited_early : bool  True if R1 exited warmup before warmup_steps
    scao/actual_warmup_steps : int   step at which warmup actually ended
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
            if isinstance(v, bool):
                # Log booleans as 0/1 scalars so they plot as a flat line
                scalar = float(v)
            elif isinstance(v, (int, float)):
                scalar = float(v)
            else:
                continue
            if math.isnan(scalar):
                continue
            tag = f"{self.tag_prefix}{k}" if self.tag_prefix else k
            self.writer.add_scalar(tag, scalar, global_step=step)


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
    Called automatically by SCAO.step() if any callbacks are registered.

    v3 changes
    ----------
    - ``scao/L_norm_mean`` is now only included when ``use_int8_ema=False``.
      Previously this line would raise AttributeError for the scao_40b /
      scao_125b presets because those store L_ema as int8 (``L_ema_q`` +
      ``L_ema_scale``) and do not have a ``L_ema`` float32 attribute.
      Rather than dequantizing on every logging step (expensive and lossy),
      we simply skip the key when int8 EMA is active.

    - Adds ``scao/warmup_exited_early`` and ``scao/actual_warmup_steps``
      from the adaptive warmup scheduler (R1), so you can see in your
      training logs whether the warmup ended early and at which step.

    Returns a MetricsDict with all available keys for this run configuration.
    """
    step = 0
    ranks: list[int] = []
    l_norms: list[float] = []

    for state in optimizer.state.values():
        if "step" in state:
            step = max(step, state["step"])
        if "preconditioner" in state:
            prec = state["preconditioner"]
            ranks.append(prec.k)
            
            # Safely collect L_ema norm if available (for curvature health diagnostics)
            if getattr(prec, "use_block_diagonal", False):
                # For block-diagonal, average the norm across sub-blocks
                b_norms = []
                for blk in prec._blocks:
                    if blk.use_kronecker:
                        if getattr(blk, "use_int8_ema", False):
                            b_norms.append(blk.L_ema_scale * 127.0)
                        else:
                            b_norms.append(blk.L_ema.norm(p="fro").item())
                if b_norms:
                    l_norms.append(sum(b_norms) / len(b_norms))
            elif getattr(prec, "use_kronecker", False):
                if getattr(prec, "use_int8_ema", False):
                    # For int8, scale * 127 gives a rough bound of the frobenius norm
                    l_norms.append(prec.L_ema_scale * 127.0)
                else:
                    l_norms.append(prec.L_ema.norm(p="fro").item())

    metrics: MetricsDict = {"step": step}

    if ranks:
        metrics["scao/rank_mean"] = sum(ranks) / len(ranks)
        metrics["scao/rank_min"]  = min(ranks)
        metrics["scao/rank_max"]  = max(ranks)
        metrics["scao/layers"]    = len(ranks)

    if l_norms:
        metrics["scao/L_norm_mean"] = sum(l_norms) / len(l_norms)

    metrics["scao/precond_freq"] = optimizer.param_groups[0].get("precond_freq", -1)

    # v3: adaptive warmup diagnostics (R1)
    ws = getattr(optimizer, "_warmup_scheduler", None)
    if ws is not None:
        metrics["scao/warmup_exited_early"]  = ws.exited_early
        metrics["scao/actual_warmup_steps"]  = ws.actual_warmup_steps

    return metrics