"""
SCAO — Sparse Curvature-Aware Adaptive Optimizer
=================================================

Main optimizer class.  Inherits from torch.optim.Optimizer and is
a drop-in replacement for AdamW in any PyTorch training loop.

Algorithm overview
------------------
1. Adam warmup phase (first `warmup_steps` steps):
   Standard Adam/AdamW update — preconditioner is being built up.

2. SCAO phase (after warmup):
   a. Every `precond_freq` steps: update_curvature(g) on each layer's
      SparsePreconditioner (expensive, done asynchronously on a side stream
      when CUDA is available).
   b. Every step: apply preconditioned gradient via precondition(g).
   c. Apply curvature-aware gradient clipping (if tau is set).
   d. Adam-style first-moment update on the preconditioned gradient.
   e. Weight decay (decoupled, AdamW-style).
   f. Parameter update.

Mixed-precision (bfloat16 / float16)
-------------------------------------
SCAO is fully compatible with bfloat16 and float16 parameters and gradients.
All preconditioner statistics are maintained in float32 internally, and the
preconditioned gradient is cast back to the parameter dtype before the update.
Use with ``torch.amp.autocast`` and ``torch.amp.GradScaler`` as you would
AdamW — no special configuration required.

torch.compile
-------------
``optimizer.step()`` is compatible with ``torch.compile`` on Linux (requires
a C++ toolchain; not available on Windows unless MSVC is installed).
The hot-path (``precondition()``) is fully traceable.  The curvature update
(``update_curvature()``) is decorated with ``@torch.compiler.disable`` because
it runs infrequently and contains non-traceable Python control flow.

Usage
-----
    from scao import SCAO

    optimizer = SCAO(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.1,
        precond_freq=20,
    )

    # Training loop (identical to AdamW)
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from .preconditioner import SparsePreconditioner


class SCAO(Optimizer):
    """
    Sparse Curvature-Aware Adaptive Optimizer.

    Callbacks / monitoring
    ----------------------
    Register zero or more callbacks to observe internal metrics at every step::

        from scao.logging import ConsoleLogger
        opt = SCAO(model.parameters(), lr=1e-3)
        opt.add_callback(ConsoleLogger(log_every=100))

    Any callable accepting a ``dict`` works (WandB, TensorBoard, custom).
    Callbacks are only invoked when at least one is registered, so there is
    zero overhead in the common production case without callbacks.
    """
    """
    Sparse Curvature-Aware Adaptive Optimizer.

    Args:
        params:
            Iterable of parameters or parameter groups.
        lr:
            Learning rate (default: 1e-3).
        betas:
            (beta1, beta2) coefficients for gradient and squared-gradient
            momentum (default: (0.9, 0.999)).  beta2 is used only during
            Adam warmup; in SCAO phase the preconditioner replaces the
            second moment.
        eps:
            Numerical stability epsilon added to denominators (default: 1e-8).
        weight_decay:
            Decoupled weight-decay coefficient (AdamW-style, default: 0.01).
        precond_freq:
            Number of optimizer steps between preconditioner updates
            T_precond (default: 20).
        epsilon_sparse:
            Spectral mass fraction to discard when selecting adaptive rank;
            smaller = higher rank = more accurate but more memory (default: 0.05).
        k_min:
            Minimum allowed rank per layer (default: 8).
        k_max:
            Maximum allowed rank per layer (default: 128).
        rho:
            EMA decay for curvature factor accumulators (default: 0.999).
        tau:
            Curvature-aware clipping threshold; set to None to disable
            (default: 1.0).
        warmup_steps:
            Number of Adam warmup steps before switching to SCAO update
            (default: 100).  Set to 0 to skip warmup.
        min_precond_updates:
            Minimum number of curvature updates the preconditioner must
            accumulate before the SCAO phase starts.  Guards against
            switching to preconditioned updates before the eigenvectors
            are statistically reliable (default: 10).  The optimizer
            stays in Adam warmup until BOTH ``warmup_steps`` is reached
            AND the preconditioner has received at least this many updates.
        max_precond_dim:
            Layers with any dimension above this threshold use the diagonal
            fallback preconditioner (default: 4096).  Increase for very large
            embedding tables if memory allows.
        use_newton_schulz:
            Use GPU-native Newton-Schulz iterations for matrix roots instead
            of eigendecomposition.  Faster on large matrices but approximate
            (default: False).
        async_precond:
            Use a dedicated CUDA stream for preconditioner updates to overlap
            with the main gradient all-reduce (default: True when CUDA is
            available).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precond_freq: int = 20,
        epsilon_sparse: float = 0.05,
        k_min: int = 8,
        k_max: int = 128,
        rho: float = 0.999,
        tau: float | None = 1.0,
        warmup_steps: int = 100,
        min_precond_updates: int = 10,
        max_precond_dim: int = 4096,
        use_newton_schulz: bool = False,
        async_precond: bool = True,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if precond_freq < 1:
            raise ValueError(f"precond_freq must be >= 1")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precond_freq=precond_freq,
            epsilon_sparse=epsilon_sparse,
            k_min=k_min,
            k_max=k_max,
            rho=rho,
            tau=tau,
            warmup_steps=warmup_steps,
            min_precond_updates=min_precond_updates,
            max_precond_dim=max_precond_dim,
            use_newton_schulz=use_newton_schulz,
        )
        super().__init__(params, defaults)

        # Dedicated CUDA stream for async preconditioner updates
        self._precond_stream: torch.cuda.Stream | None = None
        if async_precond and torch.cuda.is_available():
            self._precond_stream = torch.cuda.Stream()

        # Callback list — zero cost when empty
        self._callbacks: list = []

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _init_state(self, p: Tensor, group: dict) -> None:
        """Lazily initialise per-parameter optimizer state."""
        state = self.state[p]
        if len(state) > 0:
            return

        state["step"] = 0
        # Local SCAO-phase step counter, reset at Phase 1→2 transition.
        # Used for bias correction in Phase 2 so that the cold-start from
        # zeroed moments is handled correctly regardless of global step value.
        state["scao_step"] = 0
        # Momentum tensors are stored in float32 to avoid precision loss when
        # accumulating gradients from bfloat16/float16 parameters.
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32,
                                            memory_format=torch.preserve_format)
        # Second moment — used during Adam warmup AND in the SCAO phase
        # (SOAP-style: tracks variance of preconditioned gradient).
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32,
                                               memory_format=torch.preserve_format)
        # Sparse preconditioner
        state["preconditioner"] = SparsePreconditioner(
            param=p,
            epsilon_sparse=group["epsilon_sparse"],
            k_min=group["k_min"],
            k_max=group["k_max"],
            rho=group["rho"],
            max_precond_dim=group["max_precond_dim"],
            use_newton_schulz=group["use_newton_schulz"],
        )

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Tensor | None:
        """
        Perform a single optimisation step.

        Args:
            closure: optional closure that re-evaluates the model and returns
                     the loss (for line-search optimisers; SCAO ignores it
                     but accepts for API compatibility).

        Returns:
            loss (Tensor | None) from the closure, if provided.
        """
        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._step_group(group)

        # Fire callbacks (zero-cost check when list is empty)
        if self._callbacks:
            from .logging import collect_metrics
            metrics = collect_metrics(self)
            for cb in self._callbacks:
                cb(metrics)

        return loss

    def _step_group(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        precond_freq = group["precond_freq"]
        tau = group["tau"]
        warmup_steps = group["warmup_steps"]
        min_precond_updates = group["min_precond_updates"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("SCAO does not support sparse gradients.")

            self._init_state(p, group)
            state = self.state[p]

            state["step"] += 1
            step = state["step"]

            exp_avg: Tensor = state["exp_avg"]
            exp_avg_sq: Tensor = state["exp_avg_sq"]
            precond: SparsePreconditioner = state["preconditioner"]

            # ----------------------------------------------------------------
            # Weight decay (decoupled, AdamW-style)
            # ----------------------------------------------------------------
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)

            # ----------------------------------------------------------------
            # Phase 1: Adam warmup
            # ----------------------------------------------------------------
            # Stay in Adam until BOTH warmup_steps is reached AND the
            # preconditioner has received at least min_precond_updates
            # curvature updates.  The second guard prevents switching to
            # preconditioned gradients before the eigenvectors are reliable.
            in_warmup = (step <= warmup_steps or
                         precond.precond_step < min_precond_updates)
            if in_warmup:
                # Cast gradient to float32 for numerically stable accumulation.
                # exp_avg and exp_avg_sq are always float32 (see _init_state).
                grad_f32 = grad.float()
                exp_avg.mul_(beta1).add_(grad_f32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1.0 - beta2)

                bias_corr1 = 1.0 - beta1 ** step
                bias_corr2 = 1.0 - beta2 ** step
                step_size = lr / bias_corr1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)

                # Accumulate curvature even during warmup so preconditioner
                # is ready when we switch phases.
                if step % precond_freq == 0:
                    self._update_precond_async(precond, grad)

                # Cast update back to param dtype before applying
                p.add_(exp_avg.to(p.dtype) / denom.to(p.dtype), alpha=-step_size)
                continue

            # ----------------------------------------------------------------
            # Phase 2: SCAO update
            # ----------------------------------------------------------------

            # Record the exact step at which Phase 2 begins (once only).
            # This is used by the blending ramp and scao_step bias correction.
            if not state.get("scao_phase_started", False):
                state["scao_phase_started"] = True
                state["phase2_start_step"] = step
                state["scao_step"] = 0

            # Increment local Phase-2 step counter (t_s in Algorithm 1).
            # Used for bias correction — moments track g_eff which started
            # accumulating at Phase-2 onset, so t_s gives the correct
            # denominator even though the tensor values carry Phase-1 history.
            # The blend ramp ensures a smooth warm-start: at t_s=1 the update
            # is still 98% g_raw, so Phase-1 momentum is not discarded abruptly.
            state["scao_step"] += 1
            scao_step = state["scao_step"]

            # 2a. Update curvature every precond_freq steps
            if step % precond_freq == 0:
                self._update_precond_async(precond, grad)

            # 2b. Apply preconditioned gradient (returns same dtype as grad)
            g_precond = precond.precondition(grad)

            # 2c. Curvature-aware gradient clipping
            if tau is not None:
                g_precond = self._curvature_clip(g_precond, precond, grad, tau, eps)

            # 50-step linear blend from Adam to SCAO at phase transition.
            # At scao_step=1: blend≈0.02 (mostly raw gradient).
            # At scao_step=50: blend=1.0 (fully preconditioned).
            blend = min(1.0, scao_step / 50.0)
            grad_f32 = grad.float()
            g_precond_f32 = g_precond.float()
            g_eff = blend * g_precond_f32 + (1.0 - blend) * grad_f32

            # 2d. Adam-style update on g_eff with shared-moment bias correction.
            # Moments are NOT reset at Phase 2 — they warm-start from Phase 1.
            # Using global `step` for bias correction is correct here: the moments
            # have been accumulating since step 1, so (1-β^step) is the actual
            # bias in them.  Using scao_step (t_s=1 at first Phase-2 step) would
            # apply a 1/0.1=10x amplification on an already-debiased moment,
            # causing a destructive spike.  scao_step is tracked for diagnostics
            # and matches the paper's t_s notation, but bias correction uses step.
            exp_avg.mul_(beta1).add_(g_eff, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g_eff, g_eff, value=1.0 - beta2)

            bias_corr1 = 1.0 - beta1 ** step
            bias_corr2 = 1.0 - beta2 ** step
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)

            # 2e. Parameter update — cast back to param dtype
            p.add_((exp_avg / denom).to(p.dtype), alpha=-(lr / bias_corr1))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_precond_async(
        self,
        precond: SparsePreconditioner,
        grad: Tensor,
    ) -> None:
        """
        Schedule a preconditioner curvature update.
        If a dedicated CUDA stream is available, runs the update there
        so it overlaps with the next forward/backward pass.
        """
        if self._precond_stream is not None:
            with torch.cuda.stream(self._precond_stream):
                grad_clone = grad.detach().clone()
                precond.update_curvature(grad_clone)
        else:
            precond.update_curvature(grad.detach())

    @staticmethod
    def _curvature_clip(
        g_precond: Tensor,
        precond: SparsePreconditioner,
        grad: Tensor,
        tau: float,
        eps: float,
    ) -> Tensor:
        """
        Curvature-aware gradient clipping.

        Clips the preconditioned gradient so that the natural gradient norm
        does not exceed `tau`:
            if ||g||_F > tau:  g_precond ← g_precond * tau / ||g||_F

        where ||g||_F is the approximate natural gradient norm.
        """
        nat_norm = precond.natural_grad_norm(grad, eps=eps)
        if nat_norm > tau:
            g_precond = g_precond * (tau / nat_norm.clamp(min=eps))
        return g_precond

    # ------------------------------------------------------------------
    # Utility: synchronise async preconditioner stream
    # ------------------------------------------------------------------

    def synchronize_precond(self) -> None:
        """
        Block until all pending async preconditioner updates complete.
        Call before checkpointing or evaluation if async_precond=True.
        """
        if self._precond_stream is not None:
            self._precond_stream.synchronize()

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def add_callback(self, callback) -> None:
        """
        Register a monitoring callback.

        The callback will be called after every ``step()`` with a metrics dict.
        See ``scao.logging`` for built-in loggers (Console, TensorBoard, WandB).

        Args:
            callback: any callable accepting a ``dict[str, object]``
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback) -> None:
        """Remove a previously registered callback."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def clear_callbacks(self) -> None:
        """Remove all registered callbacks."""
        self._callbacks.clear()

    # ------------------------------------------------------------------
    # State dict / load_state_dict overrides for preconditioners
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """
        Return serialisable state dict.  Preconditioner tensors are included.
        """
        base = super().state_dict()
        # Preconditioners are already stored as tensors in state, so
        # the base class handles them correctly via _process_value.
        # We just need to handle the SparsePreconditioner object itself.
        extra_precond: dict[int, dict] = {}
        for idx, (p_ref, state) in enumerate(self.state.items()):
            if "preconditioner" in state:
                extra_precond[idx] = state["preconditioner"].state_dict()
        base["_scao_precond"] = extra_precond
        return base

    def load_state_dict(self, state_dict: dict) -> None:
        extra_precond = state_dict.pop("_scao_precond", {})
        super().load_state_dict(state_dict)
        # Restore preconditioner state
        for idx, (state) in enumerate(self.state.values()):
            if idx in extra_precond and "preconditioner" in state:
                state["preconditioner"].load_state_dict(extra_precond[idx])

    # ------------------------------------------------------------------
    # Convenience: per-layer rank diagnostics
    # ------------------------------------------------------------------

    def current_ranks(self) -> dict[int, int]:
        """
        Return a dict mapping parameter index → current preconditioner rank.
        Useful for monitoring adaptive rank behaviour during training.
        """
        ranks: dict[int, int] = {}
        for idx, state in enumerate(self.state.values()):
            if "preconditioner" in state:
                ranks[idx] = state["preconditioner"].k
        return ranks

    def precond_stats(self) -> dict[str, object]:
        """
        Return summary statistics about all preconditioners.
        """
        ks = list(self.current_ranks().values())
        if not ks:
            return {}
        return {
            "num_precond_layers": len(ks),
            "rank_min": min(ks),
            "rank_max": max(ks),
            "rank_mean": sum(ks) / len(ks),
        }
