"""
SCAO CUDA acceleration layer.
=============================
Provides faster implementations of the critical hot paths.
Falls back gracefully to pure-PyTorch if CUDA is not available or
the CUDA extension has not been compiled.

To compile the CUDA extension:
    cd scao/cuda
    python setup.py build_ext --inplace

Or install from the project root:
    pip install -e ".[cuda]"

Public API
----------
fused_kronecker_precond(U_l, s_l_inv4, U_r, s_r_inv4, G) -> Tensor
    Identity + low-rank Kronecker correction:
        P     = U_l^T @ G @ U_r          (k×k)
        delta = (s_l_inv4[:,None] * s_r_inv4[None,:] - 1) * P
        out   = G + U_l @ delta @ U_r^T
    Uses the fused CUDA kernel when k <= 128 and all tensors are on CUDA.
    Falls back to pure PyTorch otherwise.

low_rank_precond_mm(U, s, G, left=True) -> Tensor
    Efficient U diag(s) U^T @ G  (left=True) or G @ U diag(s) U^T (left=False).
    Uses tiled CUDA kernels when available; falls back to torch.mm chain.

int8_ema_update(ema_q, ema_scale, new_val, rho) -> (Tensor, float)
    Fused dequantize → rho*old + new_val → requantize int8 EMA update.
    Falls back to the same logic in pure PyTorch (float32 intermediates).
"""

from __future__ import annotations

import os
import warnings
from typing import Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Extension load
# ---------------------------------------------------------------------------

_ext = None
_ext_load_attempted = False


def _load_ext() -> object | None:
    global _ext, _ext_load_attempted
    if _ext_load_attempted:
        return _ext
    _ext_load_attempted = True
    try:
        # Import the compiled C++ extension.
        # It is usually named _scao_cuda and placed in this directory.
        from . import _scao_cuda  # type: ignore[import]
        _ext = _scao_cuda
    except ImportError:
        msg = (
            "SCAO: compiled CUDA extension 'scao.cuda._scao_cuda' not found. "
            "Run 'python setup.py build_ext --inplace' inside scao/cuda/ to build it. "
            "Falling back to pure-PyTorch implementations (slower)."
        )
        if os.environ.get("SCAO_FORCE_CUDA_EXT", "0") == "1":
            raise RuntimeError(msg) from None
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        _ext = None
    return _ext


# ---------------------------------------------------------------------------
# Pure-PyTorch fallbacks
# ---------------------------------------------------------------------------

def _fused_kronecker_precond_pytorch(
    U_l: Tensor,
    s_l_inv4: Tensor,
    U_r: Tensor,
    s_r_inv4: Tensor,
    G: Tensor,
) -> Tensor:
    """
    Pure-PyTorch identity + low-rank Kronecker correction.

    Numerically equivalent to the CUDA kernel but materialises the (m,n)
    correction tensor.  Acceptable for CPU or small k.

        P     = U_l^T @ G @ U_r                          # (k, k)
        delta = (s_l_inv4[:,None] * s_r_inv4[None,:] - 1) * P
        out   = G + U_l @ delta @ U_r^T
    """
    P = (U_l.T @ G) @ U_r                                    # (k, k)
    scale = s_l_inv4.unsqueeze(1) * s_r_inv4.unsqueeze(0)    # (k, k)
    delta = (scale - 1.0) * P                                 # (k, k)
    correction = U_l @ delta @ U_r.T                          # (m, n)
    return G + correction


def _low_rank_precond_mm_pytorch(
    U: Tensor,
    s: Tensor,
    G: Tensor,
    left: bool = True,
) -> Tensor:
    """
    Pure-PyTorch U diag(s) U^T @ G  (left=True)
    or  G @ U diag(s) U^T           (left=False).
    """
    if left:
        # (U * s) @ U^T @ G  — fuse scale into U to avoid extra alloc
        Us = U * s.unsqueeze(0)   # (m, k)
        return Us @ (U.T @ G)     # (m, n)
    else:
        # G @ U diag(s) U^T
        Us = U * s.unsqueeze(0)   # (n, k)
        return (G @ Us) @ U.T     # (m, n)


def _int8_ema_update_pytorch(
    ema_q: Tensor,
    ema_scale: float,
    new_val: Tensor,
    rho: float,
) -> Tuple[Tensor, float]:
    """
    Pure-PyTorch int8 EMA update.

    Equivalent to the two-pass CUDA kernel:
        updated  = rho * dequantize(ema_q, ema_scale) + new_val
        new_scale = max(|updated|) / 127
        q_new    = clamp(round(updated / new_scale), -127, 127).int8()
    """
    fp32 = ema_q.to(torch.float32).mul_(ema_scale)
    updated = fp32.mul_(rho).add_(new_val)
    abs_max = updated.abs().max().item()
    new_scale = abs_max / 127.0 if abs_max > 1e-30 else 1.0
    inv_scale = 1.0 / new_scale if new_scale > 1e-30 else 0.0
    q_new = updated.mul_(inv_scale).round_().clamp_(-127, 127).to(torch.int8)
    return q_new, new_scale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_kronecker_precond(
    U_l: Tensor,
    s_l_inv4: Tensor,
    U_r: Tensor,
    s_r_inv4: Tensor,
    G: Tensor,
) -> Tensor:
    """
    Apply fused Kronecker identity+correction preconditioner to gradient G.

    Routing:
      - CUDA extension available AND k <= 128 AND all tensors on CUDA
        → ``_scao_cuda.fused_kronecker_precond`` (kernel 3)
      - Otherwise
        → pure-PyTorch fallback

    Args:
        U_l:      (m, k) left eigenvectors, float32
        s_l_inv4: (k,)   left eigenvalues^{-1/4}, float32
        U_r:      (n, k) right eigenvectors, float32
        s_r_inv4: (k,)   right eigenvalues^{-1/4}, float32
        G:        (m, n) gradient matrix, any float dtype

    Returns:
        Preconditioned gradient, same shape and dtype as G.
    """
    ext = _load_ext()
    k = U_l.shape[1]
    use_cuda_kernel = (
        ext is not None
        and G.is_cuda
        and k <= 128
    )
    if use_cuda_kernel:
        try:
            return ext.fused_kronecker_precond(U_l, s_l_inv4, U_r, s_r_inv4, G)
        except (AttributeError, RuntimeError):
            pass
    return _fused_kronecker_precond_pytorch(U_l, s_l_inv4, U_r, s_r_inv4, G)


def low_rank_precond_mm(
    U: Tensor,
    s: Tensor,
    G: Tensor,
    left: bool = True,
) -> Tensor:
    """
    Efficient low-rank preconditioned matmul.

    left=True  → U diag(s) U^T @ G
    left=False → G @ U diag(s) U^T

    Routing:
      - CUDA extension available AND tensors on CUDA
        → ``_scao_cuda.low_rank_precond_mm`` (kernels 1+2)
      - Otherwise
        → pure-PyTorch fallback

    Args:
        U:    (d, k) eigenvectors
        s:    (k,)   eigenvalue scale factors
        G:    (m, n) gradient matrix
        left: side on which to apply the preconditioner

    Returns:
        Preconditioned gradient, same shape as G.
    """
    ext = _load_ext()
    use_cuda_kernel = (
        ext is not None
        and G.is_cuda
        and U.is_cuda
    )
    if use_cuda_kernel:
        try:
            return ext.low_rank_precond_mm(U, s, G, left)
        except (AttributeError, RuntimeError):
            pass
    return _low_rank_precond_mm_pytorch(U, s, G, left)


def int8_ema_update(
    ema_q: Tensor,
    ema_scale: float,
    new_val: Tensor,
    rho: float,
) -> Tuple[Tensor, float]:
    """
    Fused int8 EMA update: dequantize → rho*old + new_val → requantize.

    Routing:
      - CUDA extension available AND tensors on CUDA
        → ``_scao_cuda.int8_ema_update`` (kernels 4a+4b, two-pass atomic)
      - Otherwise
        → pure-PyTorch fallback

    Args:
        ema_q:     (N,) int8 tensor — current quantized EMA
        ema_scale: float — current dequantization scale
        new_val:   (N,) float32 tensor — alpha * outer_product_flat
        rho:       EMA decay coefficient

    Returns:
        (q_new, new_scale): updated int8 EMA tensor and new scale factor
    """
    ext = _load_ext()
    use_cuda_kernel = (
        ext is not None
        and ema_q.is_cuda
        and new_val.is_cuda
    )
    if use_cuda_kernel:
        try:
            return ext.int8_ema_update(ema_q, ema_scale, new_val, rho)
        except (AttributeError, RuntimeError):
            pass
    return _int8_ema_update_pytorch(ema_q, ema_scale, new_val, rho)
