"""
SparsePreconditioner
====================
Per-layer Kronecker-factored sparse preconditioner for SCAO.

Maintains low-rank approximations of the left (L) and right (R) curvature
factor EMAs:
    L_t ≈ U_l diag(S_l) U_l^T   (m × m approximated at rank k)
    R_t ≈ U_r diag(S_r) U_r^T   (n × n approximated at rank k)

The preconditioned gradient is:
    G_precond = U_l diag(S_l^{-1/4}) (U_l^T G U_r) diag(S_r^{-1/4}) U_r^T

For 1-D parameters (biases, LayerNorm), falls back to Adam-style diagonal.
For parameters where max(m, n) > max_precond_dim, uses sparse block-diagonal
preconditioning: the gradient matrix is partitioned into contiguous blocks
along the larger dimension, each of size ≤ max_precond_dim, and an independent
Kronecker preconditioner is applied to each block.  This preserves curvature
information across the full gradient while keeping the per-block eigendecomp
cost bounded at O(max_precond_dim³).

Mixed-precision note
--------------------
All curvature statistics (L_ema, R_ema) and eigenfactors (U_l, S_l, U_r, S_r)
are stored in **float32** regardless of the parameter dtype.  This is required
because ``torch.linalg.eigh`` does not support bfloat16 or float16.
Gradients are cast to float32 before statistics updates, and preconditioned
gradients are cast back to the original parameter dtype before returning.

torch.compile note
------------------
``update_curvature()`` and ``_update_eigenfactors()`` are decorated with
``@torch.compiler.disable`` because they contain Python control flow
(try/except, adaptive_rank loop) that cannot be traced.  They are called
infrequently (every T_precond steps) so graph-breaks there are harmless.

``precondition()`` and ``natural_grad_norm()`` contain only tensor ops and are
compile-friendly when invoked from a ``torch.compile``-d training step.
"""

from __future__ import annotations
import torch
from torch import Tensor

from .utils import (
    adaptive_rank,
    to_2d,
    from_2d,
    low_rank_matrix_power_neg_quarter,
    quantize_sym_int8,
    dequantize_sym_int8,
)
from .cuda import fused_kronecker_precond as _cuda_fused_precond

# float32 is the minimum precision required for stable eigendecomposition.
# bfloat16/float16 are NOT supported by torch.linalg.eigh on CPU.
_PRECOND_DTYPE = torch.float32


class SparsePreconditioner:
    """
    Low-rank Kronecker preconditioner for a single parameter tensor.

    State tensors (all on the same device as the parameter):
        L_ema     : (m, m) - full EMA of left curvature factor (temp, during update)
        R_ema     : (n, n) - full EMA of right curvature factor (temp, during update)
        U_l       : (m, k) - left eigenvectors
        S_l       : (k,)   - left eigenvalues (descending)
        U_r       : (n, k) - right eigenvectors
        S_r       : (k,)   - right eigenvalues (descending)
        k         : current rank (int)
    """

    def __init__(
        self,
        param: Tensor,
        epsilon_sparse: float = 0.05,
        k_min: int = 8,
        k_max: int = 128,
        rho: float = 0.999,
        max_precond_dim: int = 4096,
        use_newton_schulz: bool = False,
        use_int8_ema: bool = False,
    ) -> None:
        shape = param.shape
        device = param.device
        param_dtype = param.dtype  # original dtype — used to cast outputs back

        # Reshape to 2D to determine (m, n)
        p2d, _ = to_2d(param)
        m, n = p2d.shape

        self.original_shape = shape
        self.m = m
        self.n = n
        self.epsilon_sparse = epsilon_sparse
        self.k_min = k_min
        self.k_max = min(k_max, min(m, n) // 2) if min(m, n) > 2 * k_min else k_min
        self.rho = rho
        self.use_newton_schulz = use_newton_schulz
        # use_int8_ema: store L_ema/R_ema as int8 tensors with float32 scale factors.
        # Reduces EMA memory from (m²+n²)*4 bytes to (m²+n²)*1 + 8 bytes — 4× reduction.
        # Only applied to the Kronecker path (block-diagonal delegates to sub-preconditioners).
        self.use_int8_ema = use_int8_ema
        self.device = device
        # param_dtype: dtype of the parameter (may be bf16/fp16)
        self.param_dtype = param_dtype

        # Decide whether to use Kronecker preconditioner, block-diagonal, or diagonal fallback.
        # - Kronecker: standard case, max(m,n) ≤ max_precond_dim
        # - Block-diagonal: large layers, max(m,n) > max_precond_dim, splits the larger
        #   dimension into blocks of size ≤ max_precond_dim and applies an independent
        #   Kronecker preconditioner per block.  Matches the paper's Algorithm 1 claim.
        # - Diagonal: only for 1-D params (biases, LayerNorm scales)
        self.use_block_diagonal = (
            param.ndim >= 2
            and m > 1
            and n > 1
            and max(m, n) > max_precond_dim
        )
        self.use_kronecker = (
            param.ndim >= 2
            and m > 1
            and n > 1
            and max(m, n) <= max_precond_dim
        )

        # For large matrices (min dimension > 512), default to Newton-Schulz
        # iterations instead of eigendecomposition.  NS stays on-device (no
        # CPU transfer), is O(m²k) per step vs O(m³) for eigh, and handles
        # bfloat16 natively — essential for 300M+ parameter transformer layers
        # where weight matrices can be 1024×4096 or larger.
        if self.use_kronecker and not use_newton_schulz:
            use_newton_schulz = min(m, n) > 512

        if self.use_block_diagonal:
            # Block-diagonal preconditioning:
            # Split the gradient along the larger dimension into B ≤ max_precond_dim
            # blocks.  Each block gets its own independent Kronecker preconditioner.
            # This matches the paper's Algorithm 1 for large-matrix layers.
            large_dim = m if m >= n else n
            small_dim = n if m >= n else m
            self._block_dim = 0 if m >= n else 1  # 0 = split rows, 1 = split cols
            block_size = max_precond_dim
            n_full = large_dim // block_size
            remainder = large_dim % block_size
            self._block_sizes: list[int] = [block_size] * n_full
            if remainder > 0:
                self._block_sizes.append(remainder)
            self._blocks: list[SparsePreconditioner] = []
            for bs in self._block_sizes:
                fake_shape = (bs, small_dim) if self._block_dim == 0 else (small_dim, bs)
                fake_param = param.new_empty(fake_shape)
                blk = SparsePreconditioner(
                    param=fake_param,
                    epsilon_sparse=epsilon_sparse,
                    k_min=k_min,
                    k_max=k_max,
                    rho=rho,
                    max_precond_dim=max_precond_dim,
                    use_newton_schulz=use_newton_schulz,
                    use_int8_ema=use_int8_ema,
                )
                self._blocks.append(blk)
            # k tracks the average rank across blocks
            self.k: int = self._blocks[0].k if self._blocks else k_min

        elif self.use_kronecker:
            k_init = min(k_min, min(m, n))
            self.k: int = k_init

            # Curvature EMAs and eigenfactors are ALWAYS stored in float32.
            # This is required because linalg.eigh is not implemented for
            # bfloat16 or float16 on CPU (and is numerically unstable at
            # reduced precision on GPU).
            #
            # Fix 1 — initialize L/R as scaled identity, NOT zeros.
            # A zero init means the EMA accumulates for T_precond steps with
            # L ≈ 0, so when the preconditioner first fires the matrix is
            # rank-1 and ill-conditioned, causing a hard divergence at step
            # T_precond.  Initializing with eps * I gives a well-conditioned
            # starting point: eigenvalues begin near 1e-4, S^{-1/4} ≈ 10,
            # which is safely handled by the adaptive-eps regularization.
            _eye_eps = 1e-4
            L_init = torch.eye(m, device=device, dtype=_PRECOND_DTYPE) * _eye_eps
            R_init = torch.eye(n, device=device, dtype=_PRECOND_DTYPE) * _eye_eps

            if use_int8_ema:
                # Store EMA as int8 + float scale for 4× memory reduction.
                # (m²+n²) bytes int8 vs (m²+n²)*4 bytes fp32.
                self.L_ema_q, self.L_ema_scale = quantize_sym_int8(L_init)
                self.R_ema_q, self.R_ema_scale = quantize_sym_int8(R_init)
                # float32 shadow used only during eigendecomposition — not stored between steps
            else:
                self.L_ema = L_init
                self.R_ema = R_init

            self.U_l = torch.eye(m, k_init, device=device, dtype=_PRECOND_DTYPE)
            self.S_l = torch.ones(k_init, device=device, dtype=_PRECOND_DTYPE)
            self.U_r = torch.eye(n, k_init, device=device, dtype=_PRECOND_DTYPE)
            self.S_r = torch.ones(k_init, device=device, dtype=_PRECOND_DTYPE)
        else:
            # Diagonal fallback: maintain per-element variance estimate in float32
            self.diag_ema = torch.zeros(shape, device=device, dtype=_PRECOND_DTYPE)
            self._diag_bias_factor: float = 1e-8  # updated in update_curvature
            self.k = 1

        # Step counter for this preconditioner (updated externally)
        self.precond_step: int = 0

    # ------------------------------------------------------------------
    # Curvature accumulation (called every T_precond optimizer steps)
    # ------------------------------------------------------------------

    @torch.no_grad()
    @torch.compiler.disable
    def update_curvature(self, grad: Tensor) -> None:
        """
        Update the EMA of curvature factors given the current gradient.
        Then recompute the low-rank eigenfactors and select the new rank.

        This is the *expensive* operation — call every T_precond steps only.
        Decorated with @torch.compiler.disable because it contains Python
        control flow (try/except, rank-selection loop) that cannot be traced.

        Args:
            grad: raw gradient of the parameter (same shape as param).
                  Any dtype is accepted — will be cast to float32 internally.
        """
        self.precond_step += 1

        # Cast gradient to float32 for numerically stable statistics.
        # This is the key fix for bfloat16/float16 parameter support.
        g2d_f32, _ = to_2d(grad.to(_PRECOND_DTYPE))

        # Block-diagonal: delegate each slice to its own sub-preconditioner.
        if self.use_block_diagonal:
            offset = 0
            for bs, blk in zip(self._block_sizes, self._blocks):
                if self._block_dim == 0:
                    g_blk = g2d_f32[offset:offset + bs, :]
                else:
                    g_blk = g2d_f32[:, offset:offset + bs]
                blk.update_curvature(g_blk)
                offset += bs
            self.k = sum(b.k for b in self._blocks) // len(self._blocks)
            return

        if not self.use_kronecker:
            sq = g2d_f32.squeeze(0) if g2d_f32.shape[0] == 1 else g2d_f32
            self.diag_ema.mul_(self.rho).add_((1.0 - self.rho) * sq.reshape_as(self.diag_ema).pow(2))
            # Bias-correct the diagonal EMA so early estimates are not ~0.
            # Same principle as Adam's bias correction for exp_avg_sq.
            bias_factor = 1.0 - self.rho ** self.precond_step
            self._diag_bias_factor = max(bias_factor, 1e-8)
            return

        alpha = 1.0 - self.rho
        if self.use_int8_ema:
            # int8 path: dequantize → update → requantize
            L_fp32 = dequantize_sym_int8(self.L_ema_q, self.L_ema_scale)
            L_new  = L_fp32.mul_(self.rho).add_(alpha * (g2d_f32 @ g2d_f32.T))
            self.L_ema_q, self.L_ema_scale = quantize_sym_int8(L_new)

            R_fp32 = dequantize_sym_int8(self.R_ema_q, self.R_ema_scale)
            R_new  = R_fp32.mul_(self.rho).add_(alpha * (g2d_f32.T @ g2d_f32))
            self.R_ema_q, self.R_ema_scale = quantize_sym_int8(R_new)
        else:
            self.L_ema.mul_(self.rho).add_(alpha * (g2d_f32 @ g2d_f32.T))
            self.R_ema.mul_(self.rho).add_(alpha * (g2d_f32.T @ g2d_f32))

        # EMA bias correction: the EMA starts at 0 so early estimates are
        # (1-rho^t) * true_value instead of true_value. Divide by (1-rho^t)
        # to debias — same trick as Adam's bias correction for exp_avg_sq.
        # Without this, S_l eigenvalues are ~0 for the first ~1/(1-rho) steps,
        # making S_l^{-1/4} blow up and the preconditioned gradient useless.
        bias_factor = 1.0 - self.rho ** self.precond_step
        self._update_eigenfactors(bias_factor)

    @torch.no_grad()
    @torch.compiler.disable
    def _update_eigenfactors(self, bias_factor: float = 1.0) -> None:
        """
        Eigendecompose L_ema and R_ema, select adaptive rank, store U/S factors.

        Args:
            bias_factor: EMA bias correction factor = (1 - rho^precond_step).
                         Dividing L_ema and R_ema by this factor debiases the
                         EMA so that eigenvalues reflect true curvature magnitude
                         from the first update onward.  Default 1.0 (no correction).

        All computations are in float32 (_PRECOND_DTYPE).
        Decorated with @torch.compiler.disable (try/except + Python loops).
        """
        eps = 1e-8
        # Dequantize if int8 EMA is active
        if self.use_int8_ema:
            L_fp32 = dequantize_sym_int8(self.L_ema_q, self.L_ema_scale)
            R_fp32 = dequantize_sym_int8(self.R_ema_q, self.R_ema_scale)
        else:
            L_fp32 = self.L_ema
            R_fp32 = self.R_ema

        # Apply bias correction so eigenvalues reflect true curvature scale
        L_debiased = L_fp32 / max(bias_factor, eps)
        R_debiased = R_fp32 / max(bias_factor, eps)

        # Fix 3 — adaptive Tikhonov regularization before inversion.
        # eps_precond = 1e-4 * trace(L) / m ensures the matrix is always
        # well-conditioned relative to the current curvature scale.  This
        # prevents blow-up of S^{-1/4} in low-signal directions and is
        # more robust than a fixed eps across different model sizes/dtypes.
        eps_l = max(eps, 1e-4 * L_debiased.trace().item() / self.m)
        eps_r = max(eps, 1e-4 * R_debiased.trace().item() / self.n)
        L_reg = L_debiased + eps_l * torch.eye(self.m, device=self.device, dtype=_PRECOND_DTYPE)
        R_reg = R_debiased + eps_r * torch.eye(self.n, device=self.device, dtype=_PRECOND_DTYPE)

        # torch.linalg.eigh calls LAPACK which internally consumes the global
        # PyTorch RNG state (it draws random starting vectors for iterative
        # solvers).  Preserving and restoring the state ensures that curvature
        # updates do NOT perturb the data-loading shuffle order — critical for
        # fair comparison against AdamW in benchmarks.
        rng_state = torch.get_rng_state()

        # --- Left factor ---
        try:
            S_l_full, U_l_full = torch.linalg.eigh(L_reg)
        except torch.linalg.LinAlgError:
            S_l_full, U_l_full = torch.linalg.eigh(
                L_reg + eps * torch.eye(self.m, device=self.device, dtype=_PRECOND_DTYPE)
            )
        S_l_full = S_l_full.flip(0).clamp(min=0.0)
        U_l_full = U_l_full.flip(1)

        # --- Right factor ---
        try:
            S_r_full, U_r_full = torch.linalg.eigh(R_reg)
        except torch.linalg.LinAlgError:
            S_r_full, U_r_full = torch.linalg.eigh(
                R_reg + eps * torch.eye(self.n, device=self.device, dtype=_PRECOND_DTYPE)
            )
        S_r_full = S_r_full.flip(0).clamp(min=0.0)
        U_r_full = U_r_full.flip(1)

        # Restore RNG state so eigh does not corrupt data-shuffling seeds.
        torch.set_rng_state(rng_state)

        # --- Adaptive rank selection ---
        k_l = adaptive_rank(S_l_full, self.epsilon_sparse, self.k_min, self.k_max)
        k_r = adaptive_rank(S_r_full, self.epsilon_sparse, self.k_min, self.k_max)
        # Take the smaller of the two to keep Kronecker structure symmetric
        k_new = min(k_l, k_r)
        # Apply momentum on rank change: avoid oscillating between ranks
        k_new = max(k_new, self.k - 1)   # allow rank to drop by at most 1 per update
        k_new = min(k_new, self.k + 4)   # allow rank to grow by at most 4 per update
        k_new = int(min(self.k_max, max(self.k_min, k_new)))
        self.k = k_new

        # Store truncated eigenfactors
        self.U_l = U_l_full[:, :k_new].contiguous()
        self.S_l = S_l_full[:k_new].contiguous()
        self.U_r = U_r_full[:, :k_new].contiguous()
        self.S_r = S_r_full[:k_new].contiguous()

    # ------------------------------------------------------------------
    # Preconditioning  (called every optimizer step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def precondition(self, grad: Tensor) -> Tensor:
        """
        Apply the sparse preconditioner to the gradient.

        Uses identity + low-rank Kronecker correction:

            G_proj    = U_l^T G U_r                          # (k, k) projection
            G_scaled  = diag(S_l^{-1/4}) G_proj diag(S_r^{-1/4})  # curvature scaling
            G_precond = G + U_l (G_scaled - G_proj) U_r^T    # identity + correction

        This is equivalent to the preconditioner:
            P = I + U_l (diag(S_l^{-1/4}) - I) U_l^T ⊗ U_r (diag(S_r^{-1/4}) - I) U_r^T

        which acts as identity on the complement and applies curvature scaling within
        the low-rank subspace.  Crucially, no gradient signal is discarded — the
        previous formulation (pure projection) silently zeroed the complement,
        discarding ~(1 - k/m)(1 - k/n) fraction of the gradient at small rank.

        For the diagonal fallback:
            g_precond = g / (diag_ema^{1/4} + eps)

        Inputs of any float dtype (fp32, bf16, fp16) are accepted.
        Internal computations run in float32; the result is cast back to
        the original gradient dtype before returning.

        Args:
            grad: gradient tensor (same shape as param)

        Returns:
            Preconditioned gradient (same shape and dtype as grad)
        """
        orig_dtype = grad.dtype
        g2d, orig_shape = to_2d(grad.to(_PRECOND_DTYPE))

        # Block-diagonal: precondition each slice independently, then reassemble.
        if self.use_block_diagonal:
            result = torch.zeros_like(g2d)
            offset = 0
            for bs, blk in zip(self._block_sizes, self._blocks):
                if self._block_dim == 0:
                    g_blk = g2d[offset:offset + bs, :]
                    result[offset:offset + bs, :] = blk.precondition(g_blk).to(_PRECOND_DTYPE)
                else:
                    g_blk = g2d[:, offset:offset + bs]
                    result[:, offset:offset + bs] = blk.precondition(g_blk).to(_PRECOND_DTYPE)
                offset += bs
            return from_2d(result, orig_shape).to(orig_dtype)

        if not self.use_kronecker:
            bias = getattr(self, "_diag_bias_factor", 1.0)
            denom = (self.diag_ema / bias).reshape_as(g2d).pow(0.25).add_(1e-8)
            result = g2d / denom
            return from_2d(result, orig_shape).to(orig_dtype)

        eps = 1e-8
        _, S_l_inv4 = low_rank_matrix_power_neg_quarter(self.U_l, self.S_l, eps)
        _, S_r_inv4 = low_rank_matrix_power_neg_quarter(self.U_r, self.S_r, eps)

        # Use fused CUDA kernel when available (k ≤ 128) — avoids materialising
        # the (m,n) correction tensor and reduces memory bandwidth.
        # Falls back to pure PyTorch (inside _cuda_fused_precond) otherwise.
        G_precond = _cuda_fused_precond(self.U_l, S_l_inv4, self.U_r, S_r_inv4, g2d)

        return from_2d(G_precond, orig_shape).to(orig_dtype)

    # ------------------------------------------------------------------
    # Curvature-aware gradient norm  (for natural gradient clipping)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def natural_grad_norm(self, grad: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Compute the approximate natural gradient norm:

            ||g||_F^2 ≈ g^T (U_l S_l^{-1/2} U_l^T ⊗ U_r S_r^{-1/2} U_r^T) g

        Computation is in float32; returns a float32 scalar.

        Returns:
            Scalar tensor with the natural gradient norm.
        """
        g2d, _ = to_2d(grad.to(_PRECOND_DTYPE))

        # Block-diagonal: sum squared norms over blocks, then sqrt.
        if self.use_block_diagonal:
            sq_norms: list[Tensor] = []
            offset = 0
            for bs, blk in zip(self._block_sizes, self._blocks):
                if self._block_dim == 0:
                    g_blk = g2d[offset:offset + bs, :]
                else:
                    g_blk = g2d[:, offset:offset + bs]
                sq_norms.append(blk.natural_grad_norm(g_blk, eps=eps).pow(2))
                offset += bs
            return torch.stack(sq_norms).sum().sqrt()

        if not self.use_kronecker:
            denom = self.diag_ema.reshape_as(g2d).pow(0.5).add_(eps)
            return (g2d.pow(2) / denom).sum().sqrt()

        S_l_inv2 = self.S_l.clamp(min=eps).pow(-0.5)
        S_r_inv2 = self.S_r.clamp(min=eps).pow(-0.5)

        G_proj = (self.U_l.T @ g2d) @ self.U_r   # (k, k)
        G_scaled = S_l_inv2.unsqueeze(1) * G_proj * S_r_inv2.unsqueeze(0)

        return G_scaled.norm(p="fro")

    # ------------------------------------------------------------------
    # State dict helpers (for checkpointing)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        d: dict = {
            "precond_step": self.precond_step,
            "k": self.k,
            "use_kronecker": self.use_kronecker,
            "use_block_diagonal": self.use_block_diagonal,
        }
        if self.use_block_diagonal:
            d["blocks"] = [blk.state_dict() for blk in self._blocks]
        elif self.use_kronecker:
            if self.use_int8_ema:
                d.update({
                    "L_ema_q": self.L_ema_q,
                    "L_ema_scale": self.L_ema_scale,
                    "R_ema_q": self.R_ema_q,
                    "R_ema_scale": self.R_ema_scale,
                })
            else:
                d.update({
                    "L_ema": self.L_ema,
                    "R_ema": self.R_ema,
                })
            d.update({
                "U_l": self.U_l,
                "S_l": self.S_l,
                "U_r": self.U_r,
                "S_r": self.S_r,
            })
        else:
            d["diag_ema"] = self.diag_ema
        return d

    def memory_bytes(self) -> int:
        """Return total bytes occupied by all internal state tensors."""
        total = 0
        if self.use_block_diagonal:
            return sum(blk.memory_bytes() for blk in self._blocks)
        if self.use_kronecker:
            if self.use_int8_ema:
                # int8 EMA: 1 byte/element + 4 bytes scale per factor
                total += self.L_ema_q.numel() + 4  # int8 + scale float
                total += self.R_ema_q.numel() + 4
            else:
                total += self.L_ema.numel() * self.L_ema.element_size()
                total += self.R_ema.numel() * self.R_ema.element_size()
            for t in (self.U_l, self.S_l, self.U_r, self.S_r):
                total += t.numel() * t.element_size()
        else:
            total += self.diag_ema.numel() * self.diag_ema.element_size()
        return total

    def load_state_dict(self, state: dict) -> None:
        self.precond_step = state["precond_step"]
        self.k = state["k"]
        if self.use_block_diagonal:
            for blk, blk_state in zip(self._blocks, state["blocks"]):
                blk.load_state_dict(blk_state)
        elif self.use_kronecker:
            if self.use_int8_ema:
                self.L_ema_q = state["L_ema_q"].to(device=self.device, dtype=torch.int8)
                self.L_ema_scale = float(state["L_ema_scale"])
                self.R_ema_q = state["R_ema_q"].to(device=self.device, dtype=torch.int8)
                self.R_ema_scale = float(state["R_ema_scale"])
            else:
                self.L_ema.copy_(state["L_ema"])
                self.R_ema.copy_(state["R_ema"])
            # Eigenfactors are always stored in _PRECOND_DTYPE (float32)
            self.U_l = state["U_l"].to(device=self.device, dtype=_PRECOND_DTYPE)
            self.S_l = state["S_l"].to(device=self.device, dtype=_PRECOND_DTYPE)
            self.U_r = state["U_r"].to(device=self.device, dtype=_PRECOND_DTYPE)
            self.S_r = state["S_r"].to(device=self.device, dtype=_PRECOND_DTYPE)
        else:
            self.diag_ema.copy_(state["diag_ema"])


def _broadcast_precond(
    precond: "SparsePreconditioner",
    process_group: "torch.distributed.ProcessGroup | None" = None,
) -> None:
    """
    Broadcast all preconditioner state tensors from rank 0 to all ranks.

    Handles all three preconditioner modes (Kronecker, block-diagonal, diagonal)
    and both EMA storage formats (float32 and int8).  Also syncs the step counter
    and adaptive rank ``k`` so that subsequent updates remain numerically identical
    across all ranks.

    Args:
        precond: the SparsePreconditioner instance to synchronise.
        process_group: optional process group (default: the global default group).

    Notes:
        This function is called by ``SCAO.sync_preconditioner()``.  It is not
        intended to be called directly unless you manage the distributed state
        yourself.
    """
    import torch.distributed as dist

    # Sync step counter from rank 0.
    step_t = torch.tensor([precond.precond_step], dtype=torch.int64, device=precond.device)
    dist.broadcast(step_t, src=0, group=process_group)
    precond.precond_step = int(step_t.item())

    if precond.use_block_diagonal:
        for blk in precond._blocks:
            _broadcast_precond(blk, process_group)
        return

    if precond.use_kronecker:
        # Sync the adaptive rank k; non-rank-0 processes must resize tensors if
        # the checkpoint was saved at a different rank than their current state.
        k_t = torch.tensor([precond.k], dtype=torch.int64, device=precond.device)
        dist.broadcast(k_t, src=0, group=process_group)
        k_new = int(k_t.item())

        if k_new != precond.k:
            precond.k = k_new
            precond.U_l = torch.empty(precond.m, k_new, dtype=_PRECOND_DTYPE, device=precond.device)
            precond.S_l = torch.empty(k_new, dtype=_PRECOND_DTYPE, device=precond.device)
            precond.U_r = torch.empty(precond.n, k_new, dtype=_PRECOND_DTYPE, device=precond.device)
            precond.S_r = torch.empty(k_new, dtype=_PRECOND_DTYPE, device=precond.device)

        # Broadcast EMA accumulators.
        if precond.use_int8_ema:
            dist.broadcast(precond.L_ema_q, src=0, group=process_group)
            dist.broadcast(precond.R_ema_q, src=0, group=process_group)
            # Scale factors are Python floats; wrap as tensors for broadcast.
            for attr in ("L_ema_scale", "R_ema_scale"):
                t = torch.tensor([getattr(precond, attr)], device=precond.device)
                dist.broadcast(t, src=0, group=process_group)
                setattr(precond, attr, float(t.item()))
        else:
            dist.broadcast(precond.L_ema, src=0, group=process_group)
            dist.broadcast(precond.R_ema, src=0, group=process_group)

        # Broadcast eigenfactors (in-place: tensors already have the right shape).
        dist.broadcast(precond.U_l, src=0, group=process_group)
        dist.broadcast(precond.S_l, src=0, group=process_group)
        dist.broadcast(precond.U_r, src=0, group=process_group)
        dist.broadcast(precond.S_r, src=0, group=process_group)

    else:
        # Diagonal fallback
        dist.broadcast(precond.diag_ema, src=0, group=process_group)

