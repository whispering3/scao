"""
SCAO CUDA acceleration layer (optional).
=========================================
Provides faster implementations of the critical hot paths.
Falls back gracefully to pure-PyTorch if CUDA is not available or
the CUDA extension has not been compiled.

To compile the CUDA extension:
    cd scao/cuda
    python setup.py build_ext --inplace

Or install from the project root:
    pip install -e ".[cuda]"

Kernels exposed
---------------
low_rank_precond_mm(U, s, G, left)
    2-pass tiled matmul: U diag(s) U^T G.
    O(k·m·n) vs the old O(k·m²·n) per-element kernel.

fused_kronecker_precond(U_l, s_l_inv4, U_r, s_r_inv4, G)
    Full identity+correction precond in one GPU launch (k ≤ 128).
    Avoids materialising the (m,n) correction tensor.

int8_ema_update(ema_q, ema_scale, new_val, rho)
    Fused dequantize → EMA update → requantize for int8 curvature accumulators.
"""

from __future__ import annotations

import torch
from torch import Tensor

_cuda_ext = None  # Will hold the compiled C++/CUDA extension if available


def _try_load_cuda_ext():
    global _cuda_ext
    if _cuda_ext is not None:
        return _cuda_ext
    try:
        from scao.cuda import _scao_cuda  # type: ignore[import]
        _cuda_ext = _scao_cuda
    except ImportError:
        _cuda_ext = None
    return _cuda_ext


# ---------------------------------------------------------------------------
# Low-rank matrix-matrix multiply  (U diag(s) U^T) @ G
# ---------------------------------------------------------------------------

def low_rank_precond_mm(
    U: Tensor,
    s_inv_quarter: Tensor,
    G: Tensor,
    side: str = "left",
) -> Tensor:
    """
    Compute  U * diag(s_inv_quarter) * (U^T * G)   if side='left'
    or       G * U * diag(s_inv_quarter) * U^T      if side='right'

    Uses fused CUDA kernel if available, otherwise pure PyTorch.

    Args:
        U: (m, k) orthonormal matrix
        s_inv_quarter: (k,) vector of S^{-1/4} scaling factors
        G: (m, n) gradient matrix
        side: 'left' or 'right'

    Returns:
        Result tensor of same shape as G.
    """
    ext = _try_load_cuda_ext()
    if ext is not None and G.is_cuda:
        try:
            return ext.low_rank_precond_mm(U, s_inv_quarter, G, side == "left")
        except (AttributeError, RuntimeError):
            pass  # Fall through to PyTorch fallback

    # Pure PyTorch fallback
    if side == "left":
        # (k, n) = (m, k)^T @ (m, n)
        proj = U.T @ G
        # scale rows
        proj = proj * s_inv_quarter.unsqueeze(1)
        # (m, n) = (m, k) @ (k, n)
        return U @ proj
    else:
        # (m, k) = (m, n) @ (n, k)
        proj = G @ U
        # scale cols
        proj = proj * s_inv_quarter.unsqueeze(0)
        # (m, n) = (m, k) @ (k, n)^T
        return proj @ U.T


# ---------------------------------------------------------------------------
# Fused both-sides Kronecker precond (identity + correction)
# G_out = G + U_l @ ((s_l⊗s_r - 1) * (U_l^T@G@U_r)) @ U_r^T
# ---------------------------------------------------------------------------

def fused_kronecker_precond(
    U_l: Tensor,
    s_l_inv4: Tensor,
    U_r: Tensor,
    s_r_inv4: Tensor,
    G: Tensor,
) -> Tensor:
    """
    Full identity+correction Kronecker precond step, fused in one CUDA kernel.

    G_out = G + U_l @ delta @ U_r^T
    where delta[p,q] = (s_l_inv4[p]*s_r_inv4[q] - 1) * (U_l^T @ G @ U_r)[p,q]

    Falls back to pure PyTorch for k > 128 or when CUDA extension is not
    compiled.

    Args:
        U_l:      (m, k) left eigenvectors
        s_l_inv4: (k,)   left S^{-1/4} factors
        U_r:      (n, k) right eigenvectors
        s_r_inv4: (k,)   right S^{-1/4} factors
        G:        (m, n) gradient matrix (float32 or bfloat16)

    Returns:
        G_out: (m, n) preconditioned gradient
    """
    k = U_l.shape[1]
    ext = _try_load_cuda_ext()
    if ext is not None and G.is_cuda and k <= 128:
        try:
            return ext.fused_kronecker_precond(U_l, s_l_inv4, U_r, s_r_inv4, G)
        except (AttributeError, RuntimeError):
            pass

    # Pure PyTorch fallback: identity + low-rank correction
    G_proj   = (U_l.T @ G) @ U_r                                          # (k, k)
    G_scaled = s_l_inv4.unsqueeze(1) * G_proj * s_r_inv4.unsqueeze(0)    # (k, k)
    return G + U_l @ (G_scaled - G_proj) @ U_r.T                          # (m, n)


# ---------------------------------------------------------------------------
# int8 EMA update (dequantize → rho*old + alpha*new → requantize)
# ---------------------------------------------------------------------------

def int8_ema_update(
    ema_q: Tensor,
    ema_scale: float,
    new_val: Tensor,
    rho: float,
) -> tuple[Tensor, float]:
    """
    Fused int8 EMA update on CUDA.

    Computes: ema_new = rho * dequantize(ema_q, ema_scale) + new_val
    Then requantizes ema_new to int8 and returns (ema_q_new, new_scale).

    Falls back to pure Python when CUDA extension is not compiled.

    Args:
        ema_q:     (N,) int8 quantized EMA tensor (flat)
        ema_scale: current dequantization scale (float)
        new_val:   (N,) float32 new contribution = alpha * outer_product.view(-1)
        rho:       EMA decay coefficient

    Returns:
        (ema_q_new, new_scale): updated int8 tensor and its scale
    """
    ext = _try_load_cuda_ext()
    if ext is not None and ema_q.is_cuda and new_val.is_cuda:
        try:
            return ext.int8_ema_update(ema_q, ema_scale, new_val, rho)
        except (AttributeError, RuntimeError):
            pass

    # Pure Python fallback
    updated = rho * ema_q.float() * ema_scale + new_val
    abs_max = updated.abs().max().item()
    new_scale = abs_max / 127.0 if abs_max > 1e-30 else 1.0
    q = (updated / new_scale).round().clamp(-127, 127).to(torch.int8)
    return q, new_scale


# ---------------------------------------------------------------------------
# Batched eigendecomposition with truncation
# ---------------------------------------------------------------------------

def truncated_eigh(
    A: Tensor,
    k: int,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """
    Compute the top-k eigenvalues and eigenvectors of symmetric PSD matrix A.

    Uses CUDA LOBPCG if available; falls back to full eigh + slice.

    Args:
        A: (m, m) symmetric PSD matrix
        k: number of top eigenpairs to return

    Returns:
        (eigenvalues, eigenvectors) of shapes (k,) and (m, k),
        eigenvalues in descending order.
    """
    ext = _try_load_cuda_ext()
    m = A.shape[0]

    if k >= m:
        # Full decomposition
        S, U = torch.linalg.eigh(A)
        return S.flip(0).clamp(min=0.0), U.flip(1)

    if ext is not None and A.is_cuda:
        try:
            S, U = ext.truncated_eigh(A, k)
            return S, U
        except (AttributeError, RuntimeError):
            pass

    # PyTorch fallback: full eigh + truncate (LOBPCG is less stable for small k)
    try:
        S, U = torch.linalg.eigh(A)
    except torch.linalg.LinAlgError:
        S, U = torch.linalg.eigh(A + eps * torch.eye(m, device=A.device, dtype=A.dtype))

    S = S.flip(0).clamp(min=0.0)
    U = U.flip(1)
    return S[:k], U[:, :k]
