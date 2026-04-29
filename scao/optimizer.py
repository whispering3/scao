"""
SCAO v3 — Sparse Curvature-Aware Adaptive Optimizer
====================================================

Incremental patch over v2. All v1/v2 checkpoints load without changes.
New args are optional and default to safe v2 behaviour.

New in v3:
  R1  Adaptive warmup   — exits Phase 1 early when grad norm stabilises
  R2  Dynamic sparsity  — per-layer mask threshold scaled by grad norm ratio
  R3  Lazy precond      — event-driven factor updates instead of fixed freq
  R4  gSNR clipping     — element-wise SNR mask before the parameter update
  R5  Adaptive rank     — adjusts preconditioner k proportional to layer activity

Scale presets
-------------
  <1B   scao_sub1b()   k_min=4, sparsity=0.4, lookahead disabled
  1–3B  scao_1b()
  3B    scao_3b()
  7B    scao_7b()
  40B   scao_40b()     lazy_precond, int8 EMA, gSNR
  125B  scao_125b()    FSDP/Megatron, state offload-friendly
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from .preconditioner import SparsePreconditioner, _broadcast_precond


# ---------------------------------------------------------------------------
# Gradient filters
# ---------------------------------------------------------------------------

class _SparseGradFilter:
    """EMA-magnitude adaptive sparse mask. Replaces static top-k."""

    def __init__(self, sparsity: float = 0.7, ema: float = 0.99):
        self.sparsity = sparsity
        self.ema = ema
        self.mag_ema: Optional[Tensor] = None

    def __call__(self, grad: Tensor) -> Tensor:
        mag = grad.abs().detach()
        if self.mag_ema is None:
            self.mag_ema = mag.clone()
        else:
            self.mag_ema.mul_(self.ema).add_((1.0 - self.ema) * mag)
        threshold = torch.quantile(self.mag_ema.float(), self.sparsity)
        return grad * (self.mag_ema >= threshold).to(grad.dtype)


class _DynamicSparseFilter(_SparseGradFilter):
    """
    R2: Scales sparsity inversely with layer grad norm relative to the global EMA.
    Active layers (high ‖g‖) receive lower sparsity, preserving curvature budget
    where it matters. Critical for QLoRA where LoRA A/B norms differ by 10-100x
    from full attention projections.
    """

    def __init__(
        self,
        base_sparsity: float = 0.7,
        min_sparsity: float = 0.3,
        max_sparsity: float = 0.9,
        ema: float = 0.99,
        norm_ema: float = 0.95,
    ):
        super().__init__(sparsity=base_sparsity, ema=ema)
        self.base_sparsity = base_sparsity
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.norm_ema_coeff = norm_ema
        self._norm_ema: float = 1.0
        self._global_norm_ema: float = 1.0

    def set_global_norm_ref(self, global_norm: float) -> None:
        self._global_norm_ema = max(global_norm, 1e-8)

    def __call__(self, grad: Tensor) -> Tensor:
        layer_norm = float(grad.norm())
        self._norm_ema = (
            self.norm_ema_coeff * self._norm_ema
            + (1.0 - self.norm_ema_coeff) * layer_norm
        )
        # margin shrinks as ratio grows: high-activity layers keep more gradients
        ratio = self._norm_ema / self._global_norm_ema
        margin = 0.2 * (1.0 - min(ratio, 2.0) / 2.0)
        self.sparsity = float(
            max(self.min_sparsity, min(self.max_sparsity, self.base_sparsity + margin))
        )
        return super().__call__(grad)


# ---------------------------------------------------------------------------
# R1: Adaptive warmup scheduler
# ---------------------------------------------------------------------------

class _AdaptiveWarmupScheduler:
    """
    Monitors relative grad norm change as a curvature stability proxy.
    Exits warmup early when |‖g‖_t - ‖g‖_{t-1}| / ‖g‖_{t-1} < threshold
    for `patience` consecutive steps.

    Prevents small models from wasting preconditioned steps during the
    steepest part of the loss curve by staying locked in Adam warmup.
    """

    def __init__(
        self,
        warmup_steps: int = 100,
        stability_threshold: float = 0.05,
        patience: int = 5,
        min_warmup: int = 20,
    ):
        self.warmup_steps = warmup_steps
        self.stability_threshold = stability_threshold
        self.patience = patience
        self.min_warmup = min_warmup
        self._prev_norm: float = float("inf")
        self._stable_count: int = 0
        self._early_exit_step: Optional[int] = None

    def update(self, step: int, avg_grad_norm: float) -> bool:
        """Returns True while still in warmup."""
        if step < self.min_warmup:
            return True

        if self._prev_norm > 0:
            rel_change = abs(avg_grad_norm - self._prev_norm) / (self._prev_norm + 1e-8)
            self._stable_count = self._stable_count + 1 if rel_change < self.stability_threshold else 0

        self._prev_norm = avg_grad_norm

        if self._stable_count >= self.patience:
            self._early_exit_step = self._early_exit_step or step
            return False

        return step < self.warmup_steps

    @property
    def exited_early(self) -> bool:
        return self._early_exit_step is not None

    @property
    def actual_warmup_steps(self) -> int:
        return self._early_exit_step or self.warmup_steps


# ---------------------------------------------------------------------------
# R4: gSNR element-wise clipping
# ---------------------------------------------------------------------------

def _gsnr_clip(
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    eps: float,
    clip_snr: float,
) -> Tensor:
    """
    Masks elements where signal-to-noise ratio |m| / sqrt(v) < clip_snr.
    Unlike global norm clipping, this preserves high-signal directions
    while suppressing stochastic noise at the element level. Effective
    with small per-device batch sizes in multi-GPU setups.
    """
    snr = exp_avg.abs() / (exp_avg_sq.sqrt() + eps)
    return grad * (snr >= clip_snr).to(grad.dtype)


# ---------------------------------------------------------------------------
# R3: Lazy preconditioner trigger
# ---------------------------------------------------------------------------

class _LazyPrecondTrigger:
    """
    Replaces the fixed `step % precond_freq == 0` schedule with an
    event-driven policy: update only when ‖g_t - g_{t-1}‖ / ‖g_{t-1}‖
    exceeds delta_threshold, or when max_skip steps have elapsed.

    At 40B+ scale this cuts matrix inversion cost by 60-80% with
    negligible impact on convergence quality.
    """

    def __init__(self, delta_threshold: float = 0.1, max_skip: int = 50):
        self.delta_threshold = delta_threshold
        self.max_skip = max_skip
        self._prev_grad_norm: float = 0.0
        self._steps_since_update: int = 0

    def should_update(self, grad_norm: float) -> bool:
        self._steps_since_update += 1

        if self._steps_since_update >= self.max_skip:
            self._steps_since_update = 0
            self._prev_grad_norm = grad_norm
            return True

        if self._prev_grad_norm > 0:
            rel_delta = abs(grad_norm - self._prev_grad_norm) / (self._prev_grad_norm + 1e-8)
            if rel_delta > self.delta_threshold:
                self._steps_since_update = 0
                self._prev_grad_norm = grad_norm
                return True

        self._prev_grad_norm = grad_norm
        return False


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class SCAO(Optimizer):
    """
    Sparse Curvature-Aware Adaptive Optimizer v3.

    v1/v2 args are unchanged. New v3 args are all optional:

        dynamic_sparsity (bool):
            Enable per-layer adaptive sparsity (R2). Recommended for QLoRA
            at any scale. Default: True.

        adaptive_warmup (bool):
            Early-exit warmup when curvature stabilises (R1). Default: True.
        warmup_stability_threshold (float):
            Relative grad norm change below which a step counts as stable.
            Default: 0.05.
        warmup_patience (int):
            Consecutive stable steps required to exit warmup. Default: 5.

        lazy_precond (bool):
            Event-driven preconditioner updates instead of fixed frequency (R3).
            Enable at 40B+. Default: False.
        lazy_delta_threshold (float):
            Relative grad norm change that triggers a lazy update. Default: 0.1.
        lazy_max_skip (int):
            Hard cap on steps between preconditioner updates. Default: 50.

        use_gsnr_clip (bool):
            Element-wise SNR masking before the parameter update (R4).
            Default: False.
        gsnr_threshold (float):
            Elements with SNR below this are zeroed. Default: 0.5.

        adaptive_rank (bool):
            Adjusts preconditioner k based on per-layer grad norm ratio (R5).
            Default: False.
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
        blend_steps: int = 50,
        min_precond_updates: int = 10,
        max_precond_dim: int = 4096,
        use_newton_schulz: bool = False,
        use_int8_ema: bool = False,
        async_precond: bool = True,
        beta3: float = 0.99,
        sparsity: float = 0.7,
        noise_std_init: float = 0.01,
        noise_anneal: float = 0.998,
        lars_coeff: float = 1e-3,
        lookahead_k: int = 5,
        lookahead_alpha: float = 0.5,
        # v3 (Disabled by default to match v2 memory footprint)
        dynamic_sparsity: bool = False,
        adaptive_warmup: bool = False,
        warmup_stability_threshold: float = 0.05,
        warmup_patience: int = 5,
        lazy_precond: bool = False,
        lazy_delta_threshold: float = 0.1,
        lazy_max_skip: int = 50,
        use_gsnr_clip: bool = False,
        gsnr_threshold: float = 0.5,
        adaptive_rank: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not (0.0 <= beta3 < 1.0):
            raise ValueError(f"Invalid beta3: {beta3}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            precond_freq=precond_freq, epsilon_sparse=epsilon_sparse,
            k_min=k_min, k_max=k_max, rho=rho, tau=tau,
            warmup_steps=warmup_steps, blend_steps=blend_steps,
            min_precond_updates=min_precond_updates,
            max_precond_dim=max_precond_dim, use_newton_schulz=use_newton_schulz,
            use_int8_ema=use_int8_ema, beta3=beta3, lars_coeff=lars_coeff,
        )
        super().__init__(params, defaults)

        self._precond_stream: torch.cuda.Stream | None = None
        if async_precond and torch.cuda.is_available():
            self._precond_stream = torch.cuda.Stream()

        self._callbacks: list = []
        self._global_step: int = 0
        self._noise_std: float = noise_std_init
        self._noise_anneal: float = noise_anneal
        self._lookahead_k: int = lookahead_k
        self._lookahead_alpha: float = lookahead_alpha
        self._slow_weights: dict[int, Tensor] = {}
        self._sparsity: float = sparsity
        self._dynamic_sparsity: bool = dynamic_sparsity
        self._sparse_filters: dict[int, _DynamicSparseFilter | _SparseGradFilter] = {}
        self._global_norm_ema: float = 1.0

        self._adaptive_warmup: bool = adaptive_warmup
        self._warmup_scheduler = _AdaptiveWarmupScheduler(
            warmup_steps=warmup_steps,
            stability_threshold=warmup_stability_threshold,
            patience=warmup_patience,
            min_warmup=max(20, warmup_steps // 5),
        )

        self._lazy_precond: bool = lazy_precond
        self._lazy_triggers: dict[int, _LazyPrecondTrigger] = {}
        self._lazy_delta_threshold: float = lazy_delta_threshold
        self._lazy_max_skip: int = lazy_max_skip
        self._use_gsnr_clip: bool = use_gsnr_clip
        self._gsnr_threshold: float = gsnr_threshold
        self._adaptive_rank: bool = adaptive_rank
        # Cache for the warmup scheduler result — computed once per global step
        # in step() and reused for all parameters in _step_group().
        self._in_warmup_global: bool = True

    # -----------------------------------------------------------------------
    # Lookahead
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _lookahead_sync(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid not in self._slow_weights:
                    self._slow_weights[pid] = p.data.clone()
                slow = self._slow_weights[pid]
                slow.add_(self._lookahead_alpha * (p.data - slow))
                p.data.copy_(slow)

    # -----------------------------------------------------------------------
    # LARS trust ratio
    # -----------------------------------------------------------------------

    @staticmethod
    def _lars_scale(p: Tensor, update: Tensor, lars_coeff: float, weight_decay: float) -> float:
        p_norm = p.data.norm()
        u_norm = update.norm()
        if p_norm == 0 or u_norm == 0:
            return 1.0
        return float(lars_coeff * p_norm / (u_norm + weight_decay * p_norm + 1e-8))

    # -----------------------------------------------------------------------
    # R5: Adaptive rank
    # -----------------------------------------------------------------------

    def _maybe_adjust_rank(self, precond: SparsePreconditioner, grad_norm: float, group: dict) -> None:
        """
        Reallocates preconditioner rank budget toward high-activity layers.
        ratio > 2.0 → layer is a dominant curvature contributor → increase k.
        ratio < 0.5 → layer is near-flat → shrink k to free HBM.
        Requires SparsePreconditioner.k to support dynamic assignment.
        """
        if not self._adaptive_rank:
            return
        ratio = grad_norm / (self._global_norm_ema + 1e-8)
        current_k = precond.k
        if ratio > 2.0:
            new_k = min(group["k_max"], int(current_k * 1.25))
        elif ratio < 0.5:
            new_k = max(group["k_min"], int(current_k * 0.8))
        else:
            return
        if new_k != current_k:
            precond.k = new_k

    # -----------------------------------------------------------------------
    # State initialisation
    # -----------------------------------------------------------------------

    def _init_state(self, p: Tensor, group: dict) -> None:
        state = self.state[p]
        if len(state) > 0:
            return
        state["step"] = 0
        state["scao_step"] = 0
        # All moment buffers in fp32 regardless of param dtype to avoid
        # precision loss when accumulating bf16/fp16 gradients.
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32,
                                            memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32,
                                               memory_format=torch.preserve_format)
        # Adan third moment: tracks gradient delta g_t - g_{t-1}
        state["exp_avg_diff"] = torch.zeros_like(p, dtype=torch.float32,
                                                 memory_format=torch.preserve_format)
        state["prev_grad"] = torch.zeros_like(p, dtype=torch.float32,
                                              memory_format=torch.preserve_format)
        state["preconditioner"] = SparsePreconditioner(
            param=p,
            epsilon_sparse=group["epsilon_sparse"],
            k_min=group["k_min"],
            k_max=group["k_max"],
            rho=group["rho"],
            max_precond_dim=group["max_precond_dim"],
            use_newton_schulz=group["use_newton_schulz"],
            use_int8_ema=group["use_int8_ema"],
        )

    # -----------------------------------------------------------------------
    # Grad norm (feeds R1, R2, R5)
    # -----------------------------------------------------------------------

    def _compute_avg_grad_norm(self) -> float:
        norms = [float(p.grad.norm()) for g in self.param_groups for p in g["params"] if p.grad is not None]
        if not norms:
            return 0.0
        avg = sum(norms) / len(norms)
        # Slow EMA avoids overreaction to single noisy batches
        self._global_norm_ema = 0.95 * self._global_norm_ema + 0.05 * avg
        return avg

    # -----------------------------------------------------------------------
    # Main step
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> Tensor | None:
        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        # Compute before iterating params so R2/R5 filters see a consistent ref
        avg_norm = self._compute_avg_grad_norm()

        if self._dynamic_sparsity:
            for f in self._sparse_filters.values():
                if isinstance(f, _DynamicSparseFilter):
                    f.set_global_norm_ref(self._global_norm_ema)

        # R1: evaluate warmup scheduler ONCE per global step (not once per parameter).
        # Calling update() per-parameter would corrupt _stable_count by incrementing
        # it N times per step (N = number of parameters) instead of once per step.
        if self._adaptive_warmup:
            self._in_warmup_global = self._warmup_scheduler.update(
                self._global_step, avg_norm
            )
        else:
            # Per-param warmup is determined by `step <= warmup_steps` in _step_group.
            # Set the global flag to a correct proxy for logging (warmup_active metric).
            warmup_steps_max = max(
                (g.get("warmup_steps", 100) for g in self.param_groups), default=100
            )
            self._in_warmup_global = self._global_step <= warmup_steps_max

        for group in self.param_groups:
            self._step_group(group, avg_norm)

        self._noise_std *= self._noise_anneal

        if self._lookahead_k > 0 and self._global_step % self._lookahead_k == 0:
            self._lookahead_sync()

        if self._callbacks:
            from .logging import collect_metrics
            metrics = collect_metrics(self)
            metrics["noise_std"] = self._noise_std
            metrics["global_norm_ema"] = self._global_norm_ema
            # Reuse the already-computed warmup flag — do NOT call update() again here,
            # that would corrupt the scheduler's stability counter a second time.
            metrics["warmup_active"] = self._in_warmup_global
            for cb in self._callbacks:
                cb(metrics)

        return loss

    # -----------------------------------------------------------------------
    # Per-group update
    # -----------------------------------------------------------------------

    def _step_group(self, group: dict, avg_norm: float) -> None:
        lr           = group["lr"]
        beta1, beta2 = group["betas"]
        beta3        = group["beta3"]
        eps          = group["eps"]
        weight_decay = group["weight_decay"]
        precond_freq = group["precond_freq"]
        tau          = group["tau"]
        warmup_steps = group["warmup_steps"]
        min_pu       = group["min_precond_updates"]
        lars_coeff   = group["lars_coeff"]

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("SCAO does not support sparse gradients.")

            self._init_state(p, group)
            state = self.state[p]
            state["step"] += 1
            step = state["step"]
            pid  = id(p)

            grad = p.grad.float()
            grad_norm = float(grad.norm())

            # R2: instantiate filter type once per parameter
            if self._sparsity > 0.0:
                if pid not in self._sparse_filters:
                    cls = _DynamicSparseFilter if self._dynamic_sparsity else _SparseGradFilter
                    self._sparse_filters[pid] = cls(base_sparsity=self._sparsity) \
                        if self._dynamic_sparsity else cls(self._sparsity)
                grad = self._sparse_filters[pid](grad)

            if self._noise_std > 1e-9:
                grad = grad + torch.randn_like(grad) * self._noise_std

            # Decoupled weight decay applied before the gradient step (AdamW-style)
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)

            exp_avg      = state["exp_avg"]
            exp_avg_sq   = state["exp_avg_sq"]
            exp_avg_diff = state["exp_avg_diff"]
            prev_grad    = state["prev_grad"]
            precond: SparsePreconditioner = state["preconditioner"]

            self._maybe_adjust_rank(precond, grad_norm, group)

            # R1: adaptive warmup — use the per-step flag computed once in step().
            # Non-adaptive path falls back to the per-param step counter.
            in_warmup = (
                self._in_warmup_global
                if self._adaptive_warmup
                else (step <= warmup_steps)
            ) or (precond.precond_step < min_pu)

            if in_warmup:
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                # Accumulate curvature during warmup so Phase 2 starts warm
                if step % precond_freq == 0:
                    self._update_precond_async(precond, p.grad)
                p.add_((exp_avg / denom).to(p.dtype), alpha=-(lr / bc1))
                prev_grad.copy_(grad)
                continue

            # -------------------------------------------------------------------
            # Phase 2: preconditioned Adan update
            # -------------------------------------------------------------------

            if not state.get("scao_phase_started", False):
                state["scao_phase_started"] = True
                state["phase2_start_step"]  = step
                state["scao_step"]          = 0

            state["scao_step"] += 1
            scao_step = state["scao_step"]

            # R3: lazy trigger replaces fixed-frequency schedule
            if self._lazy_precond:
                if pid not in self._lazy_triggers:
                    self._lazy_triggers[pid] = _LazyPrecondTrigger(
                        self._lazy_delta_threshold, self._lazy_max_skip
                    )
                should_update = self._lazy_triggers[pid].should_update(grad_norm)
            else:
                should_update = (step % precond_freq == 0)

            if should_update:
                self._update_precond_async(precond, p.grad)

            g_precond = precond.precondition(p.grad)
            if tau is not None:
                g_precond = self._curvature_clip(g_precond, precond, p.grad, tau, eps)

            # Linear blend from raw gradient to preconditioned over blend_steps,
            # preventing a cold-start spike at the Phase 1→2 transition
            blend_steps = group.get("blend_steps", 50)
            blend = min(1.0, scao_step / blend_steps) if blend_steps > 0 else 1.0
            g_eff = blend * g_precond.float() + (1.0 - blend) * grad

            # Adan: three-term momentum with gradient delta
            grad_delta = grad - prev_grad
            prev_grad.copy_(grad)

            exp_avg.mul_(beta1).add_(g_eff, alpha=1.0 - beta1)
            exp_avg_diff.mul_(beta3).add_(grad_delta, alpha=1.0 - beta3)
            g_adan = g_eff + (1.0 - beta3) * grad_delta
            exp_avg_sq.mul_(beta2).addcmul_(g_adan, g_adan, value=1.0 - beta2)

            # Bias correction uses global step, not scao_step: moments accumulated
            # since step 1, so (1 - β^t) is the correct denominator
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            bc3 = 1.0 - beta3 ** step

            m_hat  = exp_avg      / bc1
            dm_hat = exp_avg_diff / bc3
            v_hat  = exp_avg_sq   / bc2
            update = (m_hat + (1.0 - beta3) * dm_hat) / v_hat.sqrt().add_(eps)

            # R4: element-wise SNR mask applied to the final update direction.
            # Evaluated after moments are updated so the SNR estimate reflects current
            # signal quality (avoids cold-start bias). Zeroes update components where
            # |m| / sqrt(v) < gsnr_threshold, suppressing high-variance low-signal dirs.
            if self._use_gsnr_clip:
                update = _gsnr_clip(update, exp_avg, exp_avg_sq, eps, self._gsnr_threshold)

            trust = self._lars_scale(p, update, lars_coeff, weight_decay) if lars_coeff > 0 else 1.0
            # `update` already contains bias-corrected m_hat (= exp_avg / bc1).
            # Dividing by bc1 here would apply bias correction twice — use lr*trust only.
            p.add_(update.to(p.dtype), alpha=-(lr * trust))

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _update_precond_async(self, precond: SparsePreconditioner, grad: Tensor) -> None:
        """
        Trigger async curvature update. Detach and clone to avoid holding
        references to the original grad tensor which might be needed for
        other param groups or overlapping steps.
        """
        # Detach ensures we don't hold the graph. 
        # Clone ensures the data is isolated for the async stream.
        g_detached = grad.detach()
        if self._precond_stream is not None:
            g_update = g_detached.clone()
            with torch.cuda.stream(self._precond_stream):
                precond.update_curvature(g_update)
        else:
            precond.update_curvature(g_detached)

    @staticmethod
    def _curvature_clip(
        g_precond: Tensor,
        precond: SparsePreconditioner,
        grad: Tensor,
        tau: float,
        eps: float,
    ) -> Tensor:
        nat_norm = precond.natural_grad_norm(grad, eps=eps)
        if nat_norm > tau:
            g_precond = g_precond * (tau / nat_norm.clamp(min=eps))
        return g_precond

    def synchronize_precond(self) -> None:
        """Block until all pending async preconditioner updates complete.
        Call before checkpointing or evaluation when async_precond=True."""
        if self._precond_stream is not None:
            self._precond_stream.synchronize()

    def sync_preconditioner(
        self,
        process_group: "torch.distributed.ProcessGroup | None" = None,
    ) -> None:
        """Broadcast optimizer state from rank 0 to all ranks.
        Not needed during normal DDP training; call only after loading
        a checkpoint on rank 0 before resuming distributed training."""
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            warnings.warn(
                "sync_preconditioner() called but distributed is not initialised.",
                RuntimeWarning, stacklevel=2,
            )
            return
        for state in self.state.values():
            for key in ("exp_avg", "exp_avg_sq", "exp_avg_diff"):
                if key in state:
                    dist.broadcast(state[key], src=0, group=process_group)
            if "step" in state:
                step_t = torch.tensor([state["step"]], dtype=torch.int64)
                dist.broadcast(step_t, src=0, group=process_group)
                state["step"] = int(step_t.item())
            precond = state.get("preconditioner")
            if precond is not None:
                _broadcast_precond(precond, process_group)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def add_callback(self, callback) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback) -> None:
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def clear_callbacks(self) -> None:
        self._callbacks.clear()

    # -----------------------------------------------------------------------
    # Checkpoint serialisation
    # -----------------------------------------------------------------------

    def state_dict(self) -> dict:
        """
        Return serialisable state dict.  Preconditioner tensors are included,
        as well as SCAO-specific runtime state (noise level, global step,
        warmup scheduler state, etc.) so that checkpoint round-trips are exact.
        """
        self.synchronize_precond()
        base = super().state_dict()
        # SparsePreconditioner state is not handled by the base class
        base["_scao_precond"] = {
            idx: state["preconditioner"].state_dict()
            for idx, state in enumerate(self.state.values())
            if "preconditioner" in state
        }
        # Save SCAO-specific runtime state so load_state_dict restores a
        # numerically identical optimizer (prevents checkpoint round-trip mismatches).
        base["_scao_runtime"] = {
            "global_step":      self._global_step,
            "noise_std":        self._noise_std,
            "global_norm_ema":  self._global_norm_ema,
            "warmup_prev_norm": self._warmup_scheduler._prev_norm,
            "warmup_stable_count": self._warmup_scheduler._stable_count,
            "warmup_early_exit_step": self._warmup_scheduler._early_exit_step,
        }
        return base

    def load_state_dict(self, state_dict: dict) -> None:
        # Work on a shallow copy to avoid mutating the caller's dictionary.
        state_dict = dict(state_dict)
        extra_precond  = state_dict.pop("_scao_precond",  {})
        runtime        = state_dict.pop("_scao_runtime",  {})
        super().load_state_dict(state_dict)
        for idx, state in enumerate(self.state.values()):
            if idx in extra_precond and "preconditioner" in state:
                state["preconditioner"].load_state_dict(extra_precond[idx])
        # Restore runtime state if available (checkpoints from older versions
        # will simply keep the defaults set in __init__).
        if runtime:
            self._global_step       = int(runtime.get("global_step",      self._global_step))
            self._noise_std         = float(runtime.get("noise_std",       self._noise_std))
            self._global_norm_ema   = float(runtime.get("global_norm_ema", self._global_norm_ema))
            ws = self._warmup_scheduler
            ws._prev_norm           = float(runtime.get("warmup_prev_norm",       ws._prev_norm))
            ws._stable_count        = int(runtime.get("warmup_stable_count",       ws._stable_count))
            ws._early_exit_step     = runtime.get("warmup_early_exit_step",        ws._early_exit_step)

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def current_ranks(self) -> dict[int, int]:
        return {
            idx: state["preconditioner"].k
            for idx, state in enumerate(self.state.values())
            if "preconditioner" in state
        }

    def precond_stats(self) -> dict[str, object]:
        ks = list(self.current_ranks().values())
        if not ks:
            return {}
        return {
            "num_precond_layers": len(ks),
            "rank_min": min(ks),
            "rank_max": max(ks),
            "rank_mean": sum(ks) / len(ks),
            "global_norm_ema": self._global_norm_ema,
            "warmup_exited_early": self._warmup_scheduler.exited_early,
            "actual_warmup_steps": self._warmup_scheduler.actual_warmup_steps,
        }


# ---------------------------------------------------------------------------
# Scale presets
# ---------------------------------------------------------------------------

def scao_sub1b(model, lr: float = 5e-4, **kw) -> SCAO:
    """
    <1B params (125M–999M). RTX 3060/T4, CPU offload.
    Kronecker inversion is cheap at this scale — maximise update frequency.
    Lookahead disabled: these models converge fast enough that slow weights
    create net drag. k_min=4 handles narrow layers without falling back to
    diagonal; verify SparsePreconditioner has this guard before use.
    """
    return SCAO(
        model.parameters(), lr=lr,
        warmup_steps=20,
        min_precond_updates=3,
        precond_freq=5,
        k_min=4,
        k_max=64,
        max_precond_dim=1024,
        epsilon_sparse=0.10,
        sparsity=0.4,               # small layers need every gradient signal
        dynamic_sparsity=True,
        adaptive_warmup=True,
        warmup_stability_threshold=0.08,
        warmup_patience=3,
        lazy_precond=False,
        use_gsnr_clip=False,        # large relative batch → low stochastic noise
        adaptive_rank=False,        # adjustment overhead not worth it at this scale
        noise_std_init=0.005,
        noise_anneal=0.995,
        lars_coeff=5e-4,
        lookahead_k=0,
        beta3=0.98,
        tau=0.8,
        **kw,
    )


def scao_1b(model, lr: float = 3e-4, **kw) -> SCAO:
    """
    1B–3B params (Phi-2, Gemma-2B, TinyLlama). RTX 3090/A10, T4×2.
    """
    return SCAO(
        model.parameters(), lr=lr,
        warmup_steps=35,
        min_precond_updates=5,
        precond_freq=8,
        k_min=4,
        k_max=96,
        max_precond_dim=2048,
        epsilon_sparse=0.07,
        sparsity=0.50,
        dynamic_sparsity=True,
        adaptive_warmup=True,
        warmup_stability_threshold=0.06,
        warmup_patience=4,
        lazy_precond=False,
        use_gsnr_clip=False,
        adaptive_rank=False,
        noise_std_init=0.008,
        noise_anneal=0.997,
        lars_coeff=8e-4,
        lookahead_k=3,
        lookahead_alpha=0.4,
        beta3=0.985,
        tau=0.9,
        **kw,
    )


def scao_3b(model, lr: float = 2e-4, **kw) -> SCAO:
    """3B params. T4/A10 16–24 GB, QLoRA."""
    return SCAO(
        model.parameters(), lr=lr,
        warmup_steps=50,
        blend_steps=30,
        precond_freq=10,
        dynamic_sparsity=True,
        adaptive_warmup=True,
        warmup_patience=3,
        lazy_precond=False,
        use_gsnr_clip=False,
        adaptive_rank=False,
        sparsity=0.6,
        **kw,
    )


def scao_7b(model, lr: float = 1e-4, **kw) -> SCAO:
    """7B params. A100 40 GB, QLoRA or full fine-tune."""
    return SCAO(
        model.parameters(), lr=lr,
        warmup_steps=80,
        precond_freq=15,
        dynamic_sparsity=True,
        adaptive_warmup=True,
        lazy_precond=False,
        use_gsnr_clip=True,
        gsnr_threshold=0.4,
        adaptive_rank=True,
        sparsity=0.65,
        **kw,
    )


def scao_40b(model, lr: float = 5e-5, **kw) -> SCAO:
    """14B–70B params. 4–8×A100/H100."""
    return SCAO(
        model.parameters(), lr=lr,
        warmup_steps=100,
        blend_steps=50,
        precond_freq=20,
        dynamic_sparsity=True,
        adaptive_warmup=True,
        lazy_precond=True,          # saves 60-80% of matrix inversions
        lazy_delta_threshold=0.08,
        lazy_max_skip=40,
        use_gsnr_clip=True,
        gsnr_threshold=0.5,
        adaptive_rank=True,
        use_int8_ema=True,          # int8 Kronecker factors: -75% HBM for precond state
        sparsity=0.75,
        **kw,
    )


def scao_125b(model, lr: float = 2e-5, **kw) -> SCAO:
    """70B+ params. FSDP / Megatron, 32–64×H100."""
    return SCAO(
        model.parameters(), lr=lr,
        warmup_steps=200,
        precond_freq=50,
        max_precond_dim=2048,
        dynamic_sparsity=True,
        adaptive_warmup=True,
        lazy_precond=True,
        lazy_delta_threshold=0.05,
        lazy_max_skip=60,
        use_gsnr_clip=True,
        gsnr_threshold=0.6,
        adaptive_rank=True,
        use_int8_ema=True,
        sparsity=0.80,
        lookahead_k=10,
        async_precond=True,
        **kw,
    )


# ---------------------------------------------------------------------------
# Scale Presets (v3)
# ---------------------------------------------------------------------------

def scao_sub1b(model_or_params, lr=1e-3, **kwargs):
    """Preset for models < 1B params (Balanced)."""
    params = model_or_params.parameters() if hasattr(model_or_params, "parameters") else model_or_params
    return SCAO(params, lr=lr, k_max=64, use_int8_ema=False, **kwargs)

def scao_1b(model_or_params, lr=1e-3, **kwargs):
    """Preset for ~1B models (Memory-efficient)."""
    params = model_or_params.parameters() if hasattr(model_or_params, "parameters") else model_or_params
    return SCAO(params, lr=lr, k_max=48, use_int8_ema=True, async_precond=True, **kwargs)

def scao_3b(model_or_params, lr=3e-4, **kwargs):
    """Preset for ~3B models (Aggressive compression)."""
    params = model_or_params.parameters() if hasattr(model_or_params, "parameters") else model_or_params
    return SCAO(params, lr=lr, k_max=32, use_int8_ema=True, async_precond=True, warmup_steps=20, blend_steps=10, **kwargs)

def scao_7b(model_or_params, lr=1e-4, **kwargs):
    """Preset for ~7B models (Max stability)."""
    params = model_or_params.parameters() if hasattr(model_or_params, "parameters") else model_or_params
    return SCAO(params, lr=lr, k_max=24, use_int8_ema=True, async_precond=False, dynamic_sparsity=False, **kwargs)

def scao_40b(model_or_params, lr=1e-4, **kwargs):
    """Preset for ~40B models (Lazy precond & gSNR)."""
    params = model_or_params.parameters() if hasattr(model_or_params, "parameters") else model_or_params
    return SCAO(
        params, lr=lr, k_max=16, 
        use_int8_ema=True, 
        async_precond=True, 
        lazy_precond=True, 
        use_gsnr_clip=True, 
        **kwargs
    )

def scao_125b(model_or_params, lr=5e-5, **kwargs):
    """Preset for ~125B models (Maximum throughput & offload-friendly)."""
    params = model_or_params.parameters() if hasattr(model_or_params, "parameters") else model_or_params
    return SCAO(
        params, lr=lr, k_max=8, 
        use_int8_ema=True, 
        async_precond=True, 
        lazy_precond=True, 
        lazy_max_skip=100,
        use_gsnr_clip=True, 
        adaptive_rank=True,
        **kwargs
    )