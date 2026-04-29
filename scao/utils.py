"""
Utility functions for SCAO: matrix operations, rank selection, Newton-Schulz iterations.
"""

from __future__ import annotations
import math
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Matrix root via Newton-Schulz iteration (faster than eigendecomposition for
# large matrices; GPU-friendly, no CPU round-trip).
# ---------------------------------------------------------------------------


def newton_schulz_root_inv(A: Tensor, steps: int = 10, eps: float = 1e-8) -> Tensor:
    """
    Compute A^{-1/4} via coupled Newton-Schulz iterations.

    Uses the Schur-Newton method. Works on GPU without leaving device.
    Input A must be symmetric positive semi-definite (m x m).

    Returns:
        X ≈ A^{-1/4}  (m x m)
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be square"
    m = A.shape[0]
    dtype = A.dtype
    device = A.device

    # Normalise: X0 = A / ||A||_F so that spectral radius ≈ 1
    norm = A.norm(p="fro").clamp(min=eps)
    X = A / norm
    # Allocate identity once and reuse across both Newton-Schulz loops.
    eye = torch.eye(m, dtype=dtype, device=device)
    Y = eye.clone()

    for _ in range(steps):
        # Coupled iterations: X_{k+1} = (3*X - X*Y*X) / 2
        #                      Y_{k+1} = (3*I - Y*X) * Y / 2
        XY = X @ Y
        X_new = (3.0 * X - XY @ X) * 0.5
        Y_new = (3.0 * eye - XY) @ Y * 0.5
        X, Y = X_new, Y_new

    # X ≈ A^{-1/2}; we need A^{-1/4} = (A^{-1/2})^{1/2}
    # One more iteration with Y = I and X = result
    X2 = X
    Y2 = eye.clone()
    for _ in range(steps):
        X2Y2 = X2 @ Y2
        X2 = (3.0 * X2 - X2Y2 @ X2) * 0.5
        Y2 = (3.0 * eye - X2Y2) @ Y2 * 0.5

    # Scale back: (A/||A||)^{-1/4} = ||A||^{1/4} * A^{-1/4}
    scale = norm.pow(0.25)
    return X2 * scale


def matrix_power_neg_quarter(
    A: Tensor,
    use_newton_schulz: bool = False,
    ns_steps: int = 10,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute A^{-1/4} for symmetric PSD matrix A.

    Two modes:
      - use_newton_schulz=False (default): eigendecomposition via torch.linalg.eigh,
        numerically exact but requires CPU round-trip on older CUDA.
      - use_newton_schulz=True: GPU-native Newton-Schulz iteration,
        approximate but stays on device throughout.

    Args:
        A: symmetric PSD matrix (m x m)
        use_newton_schulz: whether to use NS iteration
        ns_steps: NS iteration count
        eps: floor for eigenvalues to avoid division by zero

    Returns:
        A^{-1/4} (m x m)
    """
    if use_newton_schulz:
        return newton_schulz_root_inv(A, steps=ns_steps, eps=eps)

    # Eigendecomposition path (exact)
    # eigh returns eigenvalues in ascending order
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = eigvals.clamp(min=eps)
    inv_quarter = eigvals.pow(-0.25)
    return (eigvecs * inv_quarter.unsqueeze(0)) @ eigvecs.T


def low_rank_matrix_power_neg_quarter(
    U: Tensor,
    S: Tensor,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """
    Given low-rank factorization A ≈ U * diag(S) * U^T,
    return factors (U, S^{-1/4}) such that A^{-1/4} ≈ U * diag(S^{-1/4}) * U^T
    (within the subspace spanned by U).

    Args:
        U: (m, k) orthonormal columns
        S: (k,) positive eigenvalues

    Returns:
        (U, S_inv_quarter) both (m,k) and (k,)
    """
    S_inv_quarter = S.clamp(min=eps).pow(-0.25)
    return U, S_inv_quarter


# ---------------------------------------------------------------------------
# Adaptive rank selection based on eigenspectrum entropy
# ---------------------------------------------------------------------------

def adaptive_rank(
    eigenvalues: Tensor,
    epsilon: float = 0.05,
    k_min: int = 4,
    k_max: int = 128,
) -> int:
    """
    Select minimum rank k such that the top-k eigenvalues capture at least
    (1 - epsilon) fraction of the total spectral mass.

    The eigenvalues must be sorted in *descending* order.

    Args:
        eigenvalues: (m,) tensor of non-negative eigenvalues, descending order
        epsilon: allowed spectral mass fraction to discard (default 0.05 → 95% capture)
        k_min: minimum allowed rank
        k_max: maximum allowed rank

    Returns:
        k (int)
    """
    total = eigenvalues.sum().item()
    if total < 1e-12:
        return k_min

    threshold = (1.0 - epsilon) * total
    cumsum = torch.cumsum(eigenvalues, dim=0)
    indices = (cumsum >= threshold).nonzero(as_tuple=True)[0]
    k = int(indices[0].item()) + 1 if indices.numel() > 0 else len(eigenvalues)

    return int(min(k_max, max(k_min, k)))


def spectral_entropy(eigenvalues: Tensor, eps: float = 1e-12) -> float:
    """
    Shannon entropy of the normalised eigenspectrum (for diagnostics).
    """
    total = eigenvalues.sum().clamp(min=eps)
    p = eigenvalues / total
    # H = -sum(p * log(p))  ignoring zeros
    mask = p > eps
    H = -(p[mask] * p[mask].log()).sum().item()
    return H


# ---------------------------------------------------------------------------
# int8 symmetric quantization helpers (for quantized EMA accumulators)
# ---------------------------------------------------------------------------

def quantize_sym_int8(x: Tensor) -> tuple[Tensor, float]:
    """
    Symmetric per-tensor int8 quantization.

    Maps x → (q, scale) such that  x ≈ q.float() * scale,
    with q ∈ [-127, 127] (int8).  Uses scale = max(|x|) / 127.

    Args:
        x: float32 tensor of any shape

    Returns:
        q:     int8 tensor of same shape
        scale: float, dequantization multiplier
    """
    abs_max = x.abs().max().item()
    if abs_max < 1e-30:
        return torch.zeros_like(x, dtype=torch.int8), 1.0
    scale = abs_max / 127.0
    q = (x / scale).round_().clamp_(-127, 127).to(torch.int8)
    return q, scale


def dequantize_sym_int8(q: Tensor, scale: float) -> Tensor:
    """
    Dequantize int8 tensor back to float32.

    Args:
        q:     int8 tensor
        scale: dequantization multiplier (from quantize_sym_int8)

    Returns:
        float32 tensor ≈ original pre-quantization values
    """
    return q.to(torch.float32).mul_(scale)




def to_2d(g: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """
    Reshape gradient tensor to 2D matrix (m, n).
    Returns (g_2d, original_shape).
    """
    shape = g.shape
    if g.ndim == 1:
        return g.unsqueeze(0), shape   # (1, n)
    if g.ndim == 2:
        return g, shape
    # For conv weights (out, in, kH, kW) → (out, in*kH*kW)
    return g.reshape(shape[0], -1), shape


def from_2d(g_2d: Tensor, original_shape: tuple[int, ...]) -> Tensor:
    """
    Reshape 2D preconditioned gradient back to original shape.
    """
    return g_2d.reshape(original_shape)
