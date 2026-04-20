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
