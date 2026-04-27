"""
SCAO distributed training utilities.
=====================================
Helpers for ZeRO-3 (DeepSpeed) and FSDP (PyTorch ≥ 2.0) compatibility.

Usage with FSDP
---------------
    from scao.distributed import wrap_scao_for_fsdp

    model = FSDP(model, ...)
    optimizer = wrap_scao_for_fsdp(SCAO(model.parameters(), lr=1e-3))

Usage with DeepSpeed ZeRO-3
----------------------------
SCAO is compatible with ZeRO-3 out of the box as long as you do NOT enable
stage 3 optimizer state partitioning for preconditioner tensors (they must
live on the rank that owns the corresponding parameters).

In your DeepSpeed config, set:
    "zero_optimization": {
        "stage": 3,
        "stage3_param_persistence_threshold": 1e4
    }
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from .optimizer import SCAO


def _collect_kronecker_tensors(
    prec,
    tensors_to_sync: list[torch.Tensor],
    int8_ema_precs: list,
) -> None:
    """
    Collect tensors from a single Kronecker preconditioner for distributed sync.

    Eigenfactors (U_l, S_l, U_r, S_r) are always float32 and added to
    ``tensors_to_sync`` for standard all-reduce.  EMA accumulators are
    routed to ``tensors_to_sync`` (fp32 path) or ``int8_ema_precs``
    (int8 path, requires dequantize → reduce → requantize).
    """
    for t in (prec.U_l, prec.S_l, prec.U_r, prec.S_r):
        tensors_to_sync.append(t)
    if getattr(prec, "use_int8_ema", False):
        int8_ema_precs.append((prec, "L"))
        int8_ema_precs.append((prec, "R"))
    else:
        tensors_to_sync.extend([prec.L_ema, prec.R_ema])


def sync_preconditioners(optimizer: "SCAO", process_group=None) -> None:
    """
    All-reduce preconditioner eigenfactors across all ranks.

    In FSDP/ZeRO setups where parameters are sharded, each rank computes
    curvature from its local gradient shard.  This function averages those
    estimates so every rank has a globally consistent preconditioner.

    Call this AFTER optimizer.step() and BEFORE the next forward pass, or
    better: schedule it on the preconditioner update cadence (precond_freq).

    Supports both fp32 and int8 EMA accumulators (``use_int8_ema=True``).
    Int8 tensors are dequantized to float32 before the all-reduce, then
    re-quantized on each rank after averaging.

    Args:
        optimizer:  SCAO optimizer instance.
        process_group: optional distributed process group (default: global group).
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    world_size = dist.get_world_size(group=process_group)
    if world_size == 1:
        return

    from .utils import dequantize_sym_int8, quantize_sym_int8

    handles = []
    tensors_to_sync: list[torch.Tensor] = []
    int8_ema_precs: list = []  # list of (prec, "L"|"R") for int8 EMA path

    for state in optimizer.state.values():
        if "preconditioner" not in state:
            continue
        prec = state["preconditioner"]
        if getattr(prec, "use_block_diagonal", False):
            # Block-diagonal: sync each sub-block independently
            for blk in prec._blocks:
                if blk.use_kronecker:
                    _collect_kronecker_tensors(blk, tensors_to_sync, int8_ema_precs)
                else:
                    tensors_to_sync.append(blk.diag_ema)
        elif prec.use_kronecker:
            _collect_kronecker_tensors(prec, tensors_to_sync, int8_ema_precs)
        else:
            tensors_to_sync.append(prec.diag_ema)

    # Batch all-reduce of float32 tensors (eigenfactors + non-int8 EMAs)
    for t in tensors_to_sync:
        h = dist.all_reduce(t, op=dist.ReduceOp.SUM, group=process_group, async_op=True)
        handles.append((h, t))

    for h, t in handles:
        h.wait()
        t.div_(world_size)

    # All-reduce int8 EMA accumulators: dequantize → reduce → average → requantize
    for prec, side in int8_ema_precs:
        if side == "L":
            ema_f32 = dequantize_sym_int8(prec.L_ema_q, prec.L_ema_scale)
            dist.all_reduce(ema_f32, op=dist.ReduceOp.SUM, group=process_group)
            ema_f32.div_(world_size)
            prec.L_ema_q, prec.L_ema_scale = quantize_sym_int8(ema_f32)
        else:
            ema_f32 = dequantize_sym_int8(prec.R_ema_q, prec.R_ema_scale)
            dist.all_reduce(ema_f32, op=dist.ReduceOp.SUM, group=process_group)
            ema_f32.div_(world_size)
            prec.R_ema_q, prec.R_ema_scale = quantize_sym_int8(ema_f32)


def wrap_scao_for_fsdp(optimizer: "SCAO") -> "SCAO":
    """
    Register a post-step hook that synchronises preconditioners after each
    curvature update in FSDP training.

    Returns the same optimizer (mutation in place), for chaining:
        optimizer = wrap_scao_for_fsdp(SCAO(...))
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: F401
    except ImportError:
        warnings.warn("torch.distributed.fsdp not available; FSDP wrapping skipped.")
        return optimizer

    original_step = optimizer.step

    def patched_step(closure=None):
        result = original_step(closure)
        # Synchronise preconditioners on curvature-update steps
        step = next(
            (s.get("step", 0) for s in optimizer.state.values() if "step" in s),
            0,
        )
        if step % optimizer.defaults["precond_freq"] == 0:
            optimizer.synchronize_precond()
            sync_preconditioners(optimizer)
        return result

    optimizer.step = patched_step  # type: ignore[method-assign]
    return optimizer
