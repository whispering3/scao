"""
Unit tests for SCAO optimizer.

Tests:
  1. API compatibility: drop-in for AdamW
  2. Convergence on a convex quadratic (toy sanity check)
  3. Warmup phase behaves like Adam
  4. Preconditioner rank adaptation
  5. Curvature-aware clipping
  6. Checkpoint save/load round-trip
  7. Weight decay correctness
  8. 1-D parameter (bias) fallback path
"""

from __future__ import annotations

import math
import copy

import pytest
import torch
import torch.nn as nn

# Add project root to path when running directly
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scao import SCAO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_quadratic_params(n: int = 64, ill_cond: float = 100.0, device="cpu"):
    """
    Minimise  0.5 * x^T A x - b^T x  where A is ill-conditioned.
    Optimal solution: x* = A^{-1} b.
    Returns (param, loss_fn) where loss_fn() computes the scalar loss.
    """
    torch.manual_seed(42)
    # Build ill-conditioned Hessian A = Q diag(eigenvalues) Q^T
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    eigvals = torch.logspace(0, math.log10(ill_cond), n)  # 1 … ill_cond
    A = (Q * eigvals.unsqueeze(0)) @ Q.T
    A = A.to(device)
    b = torch.randn(n, device=device)
    x_star = torch.linalg.solve(A, b)

    x = nn.Parameter(torch.zeros(n, device=device))

    def loss_fn():
        return 0.5 * (x @ A @ x) - b @ x

    return x, x_star, loss_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSCAOBasic:
    def test_instantiation_defaults(self):
        model = nn.Linear(32, 16)
        opt = SCAO(model.parameters(), lr=1e-3)
        assert opt is not None

    def test_instantiation_custom_hyperparams(self):
        model = nn.Linear(32, 16)
        opt = SCAO(
            model.parameters(),
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=0.1,
            precond_freq=10,
            k_min=4,
            k_max=32,
            rho=0.99,
            tau=2.0,
            warmup_steps=50,
        )
        assert opt.defaults["precond_freq"] == 10

    def test_invalid_lr_raises(self):
        model = nn.Linear(8, 4)
        with pytest.raises(ValueError):
            SCAO(model.parameters(), lr=-1e-3)

    def test_invalid_beta_raises(self):
        model = nn.Linear(8, 4)
        with pytest.raises(ValueError):
            SCAO(model.parameters(), betas=(1.0, 0.999))

    def test_step_returns_none_without_closure(self):
        model = nn.Linear(8, 4)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0)
        result = opt.step()
        assert result is None

    def test_step_with_closure(self):
        model = nn.Linear(8, 4)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0)
        x = torch.randn(2, 8)

        def closure():
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss

        loss_val = opt.step(closure)
        assert loss_val is not None


class TestSCAOConvergence:
    """Verify that SCAO converges on a simple ill-conditioned quadratic."""

    def test_convergence_quadratic_warmup_only(self):
        """With warmup_steps=200 (all Adam), should converge like Adam."""
        x, x_star, loss_fn = simple_quadratic_params(n=32, ill_cond=10.0)
        # noise_std_init=0.0, sparsity=0.0: disable stochastic features so
        # convergence is deterministic and comparable to a clean Adam baseline.
        # adaptive_warmup=False: stay in Adam warmup for all warmup_steps (500),
        # otherwise the scheduler can exit early and switch to the Adan/SCAO update.
        opt = SCAO([x], lr=1e-2, warmup_steps=500, weight_decay=0.0, tau=None,
                   noise_std_init=0.0, sparsity=0.0, adaptive_warmup=False)

        for _ in range(500):
            opt.zero_grad()
            l = loss_fn()
            l.backward()
            opt.step()

        err = (x - x_star).norm().item()
        assert err < 0.5, f"Did not converge in warmup-only mode; error={err:.4f}"

    def test_convergence_scao_phase(self):
        """SCAO phase should converge to near-optimal on an ill-conditioned quadratic."""
        x, x_star, loss_fn = simple_quadratic_params(n=32, ill_cond=50.0)
        opt = SCAO(
            [x],
            lr=1e-2,
            warmup_steps=50,
            precond_freq=5,
            weight_decay=0.0,
            tau=None,
            lars_coeff=0.0,       # LARS trust-ratio is designed for large-scale training;
                                   # on a small 32-dim quadratic ||p||/||update|| ≈ 0.35,
                                   # which shrinks the effective lr by ~3000x and stalls convergence
            k_min=4,
            k_max=16,
            epsilon_sparse=0.1,
            noise_std_init=0.0,   # disable noise for deterministic convergence test
            sparsity=0.0,         # disable sparsity filter for clean gradient signal
            adaptive_warmup=False, # warmup exits exactly at warmup_steps; avoids early exit on stable quadratic
        )

        for _ in range(600):
            opt.zero_grad()
            l = loss_fn()
            l.backward()
            opt.step()

        err = (x - x_star).norm().item()
        assert err < 1.0, f"SCAO did not converge; residual error={err:.4f}"

    def test_scao_faster_than_adam_on_ill_conditioned(self):
        """
        SCAO should reach a target loss in fewer steps than Adam on an
        ill-conditioned problem.
        """
        n, ill_cond = 32, 200.0
        target_loss = -5.0

        def steps_to_target(optimizer_cls, **kwargs):
            x, _, loss_fn = simple_quadratic_params(n=n, ill_cond=ill_cond)
            opt = optimizer_cls([x], **kwargs)
            for step in range(2000):
                opt.zero_grad()
                l = loss_fn()
                l.backward()
                opt.step()
                if l.item() < target_loss:
                    return step
            return 2000

        adam_steps = steps_to_target(
            torch.optim.Adam, lr=1e-2, weight_decay=0.0
        )
        scao_steps = steps_to_target(
            SCAO,
            lr=1e-2,
            warmup_steps=20,
            precond_freq=5,
            weight_decay=0.0,
            tau=None,
            k_min=4,
            k_max=16,
        )
        # SCAO should be at least as good as Adam (allow 10% slack)
        assert scao_steps <= adam_steps * 1.1, (
            f"SCAO ({scao_steps} steps) slower than Adam ({adam_steps} steps)"
        )


class TestSCAOWarmup:
    def test_warmup_matches_adam_update_direction(self):
        """
        During warmup the update should be in the same direction as Adam.
        We compare the parameter change direction.
        """
        torch.manual_seed(0)
        model_scao = nn.Linear(16, 8, bias=False)
        model_adam = copy.deepcopy(model_scao)

        x = torch.randn(4, 16)
        y = torch.randn(4, 8)

        opt_scao = SCAO(model_scao.parameters(), lr=1e-3, warmup_steps=100,
                        weight_decay=0.0, tau=None, noise_std_init=0.0, sparsity=0.0)
        opt_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3,
                                    betas=(0.9, 0.999), weight_decay=0.0)

        loss_fn = nn.MSELoss()

        scao_deltas, adam_deltas = [], []
        for _ in range(3):
            for opt, model, deltas in [(opt_scao, model_scao, scao_deltas),
                                        (opt_adam, model_adam, adam_deltas)]:
                w_before = model.weight.data.clone()
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                deltas.append((model.weight.data - w_before).sign().flatten())

        for s, a in zip(scao_deltas, adam_deltas):
            agreement = (s == a).float().mean().item()
            assert agreement > 0.7, f"Warmup direction disagrees with Adam: {agreement:.2%}"


class TestPreconditioner:
    def test_rank_adaptation(self):
        """Rank should adapt to the curvature complexity."""
        model = nn.Linear(64, 64, bias=False)
        opt = SCAO(
            model.parameters(),
            lr=1e-3,
            warmup_steps=0,
            precond_freq=1,
            k_min=4,
            k_max=32,
        )
        x = torch.randn(16, 64)
        for _ in range(5):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()

        ranks = opt.current_ranks()
        assert len(ranks) > 0
        for k in ranks.values():
            assert k >= 4, f"rank below k_min: {k}"
            assert k <= 32, f"rank above k_max: {k}"

    def test_precond_stats(self):
        model = nn.Linear(32, 32)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0, precond_freq=1)
        opt.zero_grad()
        model(torch.randn(4, 32)).sum().backward()
        opt.step()
        stats = opt.precond_stats()
        assert "rank_mean" in stats
        assert stats["num_precond_layers"] >= 1

    def test_1d_param_fallback(self):
        """Bias (1-D) should use the diagonal fallback path without errors."""
        model = nn.Linear(16, 8, bias=True)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0, precond_freq=1)
        x = torch.randn(4, 16)
        for _ in range(3):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()  # Should not raise


class TestClipping:
    def test_clipping_reduces_large_grad(self):
        """Large gradients should be clipped."""
        model = nn.Linear(16, 8, bias=False)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0,
                   precond_freq=1, tau=1e-6)  # very aggressive clip

        # Inject huge gradient
        opt.zero_grad()
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 1e6

        # Step should complete without NaN
        opt.step()
        for p in model.parameters():
            assert not p.isnan().any(), "NaN after aggressive clipping"

    def test_no_clipping_when_tau_none(self):
        """With tau=None, no clipping is applied."""
        model = nn.Linear(16, 8, bias=False)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0,
                   precond_freq=1, tau=None)
        opt.zero_grad()
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step()  # No error expected


class TestWeightDecay:
    def test_weight_decay_shrinks_params(self):
        model = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)

        # lr must be non-zero: AdamW-style decay is p *= (1 - lr * wd)
        opt = SCAO(model.parameters(), lr=1e-2, weight_decay=0.1,
                   warmup_steps=0, noise_std_init=0.0)
        opt.zero_grad()
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()

        for p in model.parameters():
            assert (p < 1.0).all(), "Weight decay should have shrunk params"


class TestCheckpointing:
    def test_save_load_roundtrip(self):
        """State dict save/load should produce identical optimizer state."""
        torch.manual_seed(7)
        model = nn.Linear(32, 16)
        # sparsity=0.0, lookahead_k=0, noise_std_init=0.0: disable stochastic
        # features to ensure the checkpoint test is deterministic (noise uses the
        # RNG and would produce different values for opt vs opt2 post-load).
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=5, precond_freq=2,
                   sparsity=0.0, lookahead_k=0, noise_std_init=0.0)

        x = torch.randn(4, 32)
        for _ in range(10):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()

        # Capture state and a copy of parameters after 10 steps
        state = copy.deepcopy(opt.state_dict())
        model2 = copy.deepcopy(model)

        # opt  → model   (continues from checkpoint)
        # opt2 → model2  (restored from same checkpoint)
        opt2 = SCAO(model2.parameters(), lr=1e-3, warmup_steps=5, precond_freq=2,
                    sparsity=0.0, lookahead_k=0, noise_std_init=0.0)
        opt2.load_state_dict(state)

        # Both models + optimizers should produce the SAME next step
        torch.manual_seed(99)
        x_new = torch.randn(4, 32)

        for m_, o_ in [(model, opt), (model2, opt2)]:
            o_.zero_grad()
            m_(x_new).sum().backward()
            o_.step()

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), "Checkpoint round-trip mismatch"


class TestMultiLayerModel:
    def test_transformer_block(self):
        """End-to-end test on a mini transformer attention block."""
        d_model, n_head = 64, 4
        attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        opt = SCAO(attn.parameters(), lr=1e-4, warmup_steps=10, precond_freq=5)

        x = torch.randn(2, 8, d_model)
        for _ in range(15):
            opt.zero_grad()
            out, _ = attn(x, x, x)
            out.sum().backward()
            opt.step()

        # Verify no NaN or Inf in parameters
        for p in attn.parameters():
            assert not p.isnan().any(), "NaN in transformer params"
            assert not p.isinf().any(), "Inf in transformer params"

class TestMixedPrecision:
    """bfloat16 and float16 parameter support."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_reduced_precision_params(self, dtype):
        """SCAO must handle reduced-precision parameters without error or NaN."""
        model = nn.Linear(32, 16).to(dtype)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=5, precond_freq=2)
        x = torch.randn(4, 32, dtype=dtype)

        for _ in range(10):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()

        for p in model.parameters():
            assert not p.isnan().any(), f"NaN in {dtype} params"
            assert not p.isinf().any(), f"Inf in {dtype} params"
            assert p.dtype == dtype, "Parameter dtype changed"

    def test_amp_autocast_gradscaler(self):
        """SCAO must be compatible with torch.amp GradScaler."""
        model = nn.Linear(32, 16)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=5, precond_freq=2)
        scaler = torch.amp.GradScaler("cpu")
        x = torch.randn(4, 32)

        for _ in range(5):
            opt.zero_grad()
            with torch.amp.autocast("cpu"):
                loss = model(x).sum()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        for p in model.parameters():
            assert not p.isnan().any()

    def test_preconditioner_state_always_float32(self):
        """Internal preconditioner state must be float32 regardless of param dtype."""
        model = nn.Linear(16, 8).to(torch.bfloat16)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0, precond_freq=1)
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()

        for state in opt.state.values():
            if "preconditioner" in state:
                prec = state["preconditioner"]
                if prec.use_kronecker:
                    assert prec.L_ema.dtype == torch.float32
                    assert prec.U_l.dtype == torch.float32
                    assert prec.S_l.dtype == torch.float32


# ---------------------------------------------------------------------------
# LR Scheduler compatibility
# ---------------------------------------------------------------------------

class TestLRScheduler:
    """Verify SCAO works correctly with common PyTorch LR schedulers."""

    def _run_with_scheduler(self, scheduler_factory, steps=20):
        model = nn.Linear(16, 8)
        opt = SCAO(model.parameters(), lr=1e-2, warmup_steps=3, precond_freq=2)
        x = torch.randn(4, 16)
        scheduler = scheduler_factory(opt)
        for _ in range(steps):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()
            scheduler.step()
        for p in model.parameters():
            assert not p.isnan().any(), "NaN detected"
            assert not p.isinf().any(), "Inf detected"
        return opt.param_groups[0]["lr"]

    def test_cosine_annealing(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        final_lr = self._run_with_scheduler(
            lambda opt: CosineAnnealingLR(opt, T_max=20, eta_min=1e-5)
        )
        assert final_lr <= 1e-2

    def test_step_lr(self):
        from torch.optim.lr_scheduler import StepLR
        final_lr = self._run_with_scheduler(lambda opt: StepLR(opt, step_size=5, gamma=0.5))
        assert final_lr < 1e-2

    def test_linear_warmup_decay(self):
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        def factory(opt):
            warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=5)
            decay = LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=15)
            return SequentialLR(opt, schedulers=[warmup, decay], milestones=[5])
        final_lr = self._run_with_scheduler(factory, steps=20)
        assert final_lr < 1e-2

    def test_onecycle_lr(self):
        from torch.optim.lr_scheduler import OneCycleLR
        model = nn.Linear(16, 8)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=3, precond_freq=2)
        x = torch.randn(4, 16)
        scheduler = OneCycleLR(opt, max_lr=1e-2, total_steps=20)
        for _ in range(20):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()
            scheduler.step()
        for p in model.parameters():
            assert not p.isnan().any()

    def test_reduce_on_plateau(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        model = nn.Linear(16, 8)
        opt = SCAO(model.parameters(), lr=1e-2, warmup_steps=3, precond_freq=2)
        x = torch.randn(4, 16)
        scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=3, min_lr=1e-5)
        for _ in range(15):
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            opt.step()
            scheduler.step(loss.item())
        for p in model.parameters():
            assert not p.isnan().any()


# ---------------------------------------------------------------------------
# Logging / Callback hooks
# ---------------------------------------------------------------------------

class TestLoggingHooks:
    """Verify the callback system fires correctly and carries expected metrics."""

    def test_callback_fires_on_each_step(self):
        model = nn.Linear(16, 8)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=2, precond_freq=2)
        x = torch.randn(4, 16)
        calls = []
        opt.add_callback(calls.append)
        for _ in range(6):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()
        assert len(calls) == 6

    def test_callback_metrics_keys(self):
        model = nn.Linear(16, 8)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0, precond_freq=1)
        x = torch.randn(4, 16)
        received = {}
        opt.add_callback(received.update)
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        assert "step" in received
        assert "scao/layers" in received

    def test_no_callbacks_by_default(self):
        model = nn.Linear(16, 8)
        opt = SCAO(model.parameters(), lr=1e-3)
        assert opt._callbacks == []

    def test_remove_callback(self):
        model = nn.Linear(16, 8)
        opt = SCAO(model.parameters(), lr=1e-3, precond_freq=2)
        x = torch.randn(4, 16)
        calls = []
        cb = calls.append
        opt.add_callback(cb)
        opt.remove_callback(cb)
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        assert len(calls) == 0

    def test_console_logger(self, capsys):
        from scao.logging import ConsoleLogger
        model = nn.Linear(16, 8)
        opt = SCAO(model.parameters(), lr=1e-3, precond_freq=2)
        x = torch.randn(4, 16)
        opt.add_callback(ConsoleLogger(log_every=1))
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        captured = capsys.readouterr().out
        assert "[SCAO]" in captured


class TestTorchCompile:
    """torch.compile compatibility (skipped on Windows without MSVC)."""

    @pytest.mark.skipif(
        __import__("platform").system() == "Windows",
        reason="torch.compile requires MSVC/cl.exe on Windows; skip in dev",
    )
    def test_compile_step(self):
        """optimizer.step() should be compilable on supported platforms."""
        model = nn.Linear(32, 16)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0, precond_freq=5)
        x = torch.randn(4, 32)

        try:
            compiled_model = torch.compile(model)
        except Exception:
            pytest.skip("torch.compile not available in this environment")

        for _ in range(6):
            opt.zero_grad()
            compiled_model(x).sum().backward()
            opt.step()

        for p in model.parameters():
            assert not p.isnan().any()


# ---------------------------------------------------------------------------
# Block-diagonal preconditioning (large-layer path)
# ---------------------------------------------------------------------------

class TestBlockDiagonal:
    """Verify block-diagonal preconditioning activates for large matrices."""

    def test_block_diagonal_activates(self):
        """
        A parameter with max(m,n) > max_precond_dim should use use_block_diagonal,
        not the diagonal fallback.
        """
        from scao.preconditioner import SparsePreconditioner
        # 128x16 param, max_precond_dim=64 → split into 2 blocks of 64 rows
        p = torch.randn(128, 16)
        prec = SparsePreconditioner(p, max_precond_dim=64)
        assert prec.use_block_diagonal, "Expected block-diagonal for large param"
        assert not prec.use_kronecker, "Should not use single Kronecker for large param"
        assert len(prec._blocks) == 2
        assert prec._block_dim == 0
        for blk in prec._blocks:
            assert blk.use_kronecker, "Each sub-block should use Kronecker"

    def test_block_diagonal_no_nan(self):
        """Block-diagonal preconditioner must produce finite outputs."""
        from scao.preconditioner import SparsePreconditioner
        torch.manual_seed(0)
        p = torch.randn(128, 16)
        prec = SparsePreconditioner(p, max_precond_dim=64, k_min=4)
        g = torch.randn_like(p)
        prec.update_curvature(g)
        g_out = prec.precondition(g)
        assert g_out.shape == g.shape
        assert not g_out.isnan().any()
        assert not g_out.isinf().any()

    def test_block_diagonal_state_dict_roundtrip(self):
        """state_dict / load_state_dict must be invertible for block-diagonal."""
        from scao.preconditioner import SparsePreconditioner
        torch.manual_seed(1)
        p = torch.randn(128, 16)
        prec = SparsePreconditioner(p, max_precond_dim=64, k_min=4)
        g = torch.randn_like(p)
        for _ in range(3):
            prec.update_curvature(g)
        g1 = prec.precondition(g).clone()

        state = prec.state_dict()
        prec2 = SparsePreconditioner(p, max_precond_dim=64, k_min=4)
        prec2.load_state_dict(state)
        g2 = prec2.precondition(g)
        assert torch.allclose(g1, g2, atol=1e-5)

    def test_optimizer_with_large_layer(self):
        """SCAO optimizer step must work end-to-end on a model with large layers."""
        model = nn.Linear(256, 64)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=5, precond_freq=2,
                   max_precond_dim=64)
        x = torch.randn(4, 256)
        for _ in range(15):
            opt.zero_grad()
            model(x).sum().backward()
            opt.step()
        for p in model.parameters():
            assert not p.isnan().any()
            assert not p.isinf().any()


# ---------------------------------------------------------------------------
# Distributed sync (no-op when dist not initialized)
# ---------------------------------------------------------------------------

class TestDistributed:
    """Verify distributed utilities are safe in single-process mode."""

    def test_sync_preconditioners_noop_without_dist(self):
        """sync_preconditioners should be a no-op when dist is not initialized."""
        from scao.distributed import sync_preconditioners
        model = nn.Linear(16, 8)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0, precond_freq=1)
        x = torch.randn(4, 16)
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        sync_preconditioners(opt)  # must not raise

    def test_wrap_scao_for_fsdp_returns_optimizer(self):
        """wrap_scao_for_fsdp must return a callable optimizer."""
        from scao.distributed import wrap_scao_for_fsdp
        import warnings
        model = nn.Linear(16, 8)
        opt = SCAO(model.parameters(), lr=1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wrapped = wrap_scao_for_fsdp(opt)
        assert wrapped is not None

    def test_sync_preconditioners_block_diagonal(self):
        """sync_preconditioners must not crash on block-diagonal preconditioners."""
        from scao.distributed import sync_preconditioners
        model = nn.Linear(256, 64)
        opt = SCAO(model.parameters(), lr=1e-3, warmup_steps=0, precond_freq=1,
                   max_precond_dim=64)
        x = torch.randn(4, 256)
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        sync_preconditioners(opt)  # dist not initialized → silent no-op

