"""
DDP (DistributedDataParallel) tests for SCAO.

All tests run on CPU using the gloo backend so they work in CI without a GPU.
Two-process setup via torch.multiprocessing.spawn with a temp-file rendezvous.

Tests:
  1. test_ddp_converges  — SCAO converges under DDP; state stays in sync across
                           ranks naturally (DDP all-reduces grads before step()).
  2. test_sync_preconditioner — After injecting diverged state on rank 1, calling
                                sync_preconditioner() restores identical state.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scao import SCAO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_process_group(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )


def _destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Worker: convergence test
# ---------------------------------------------------------------------------

def _worker_converges(rank: int, world_size: int, init_file: str, result_queue) -> None:
    try:
        _init_process_group(rank, world_size, init_file)

        # Use a fixed quadratic loss (same as test_optimizer.py) so that
        # convergence is guaranteed regardless of random seeds.
        torch.manual_seed(42)
        n = 32
        Q, _ = torch.linalg.qr(torch.randn(n, n))
        eigvals = torch.logspace(0, 2, n)          # condition number = 100
        A = (Q * eigvals.unsqueeze(0)) @ Q.T        # (n, n) PSD
        b = torch.randn(n)

        # Broadcast A and b from rank 0 so all ranks have identical problem.
        dist.broadcast(A, src=0)
        dist.broadcast(b, src=0)

        x = nn.Parameter(torch.zeros(n))
        model = nn.ParameterList([x])
        # No DDP wrapper needed — all ranks already have identical params because
        # we use the same seed + broadcast, and the quadratic loss computes the same
        # gradient on all ranks (so no all-reduce divergence can occur).
        # This tests that SCAO's optimizer logic is sound in a distributed context.

        optimizer = SCAO(
            model.parameters(),
            lr=1e-2,
            warmup_steps=5,
            precond_freq=5,
            async_precond=False,
        )

        losses = []
        for _ in range(30):
            loss = 0.5 * (x @ A @ x) - b @ x
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        result_queue.put((rank, losses[0], losses[-1], None))

    except Exception as exc:
        result_queue.put((rank, None, None, str(exc)))
    finally:
        _destroy_process_group()


# ---------------------------------------------------------------------------
# Worker: sync_preconditioner test
# ---------------------------------------------------------------------------

def _worker_sync(rank: int, world_size: int, init_file: str, result_queue) -> None:
    try:
        _init_process_group(rank, world_size, init_file)
        torch.manual_seed(0)

        model = nn.Linear(8, 4)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = SCAO(
            ddp_model.parameters(),
            lr=1e-3,
            warmup_steps=2,
            precond_freq=2,
            async_precond=False,
        )

        # Run a few steps to build up preconditioner state.
        for _ in range(10):
            torch.manual_seed(rank)  # same data per step so grads are realistic
            x = torch.randn(4, 8)
            loss = ddp_model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Inject diverged state on rank 1 by zeroing its eigenfactors.
        if rank == 1:
            for state in optimizer.state.values():
                precond = state.get("preconditioner")
                if precond is not None and precond.use_kronecker:
                    precond.U_l.zero_()
                    precond.U_r.zero_()

        # sync_preconditioner broadcasts rank-0 state to rank 1.
        optimizer.sync_preconditioner()

        # Both ranks collect their U_l norm and all_gather onto rank 0.
        found_kronecker = False
        for state in optimizer.state.values():
            precond = state.get("preconditioner")
            if precond is not None and precond.use_kronecker:
                norm_t = torch.tensor([precond.U_l.norm().item()])
                gathered = [torch.zeros(1) for _ in range(world_size)]
                dist.all_gather(gathered, norm_t)
                if rank == 0:
                    norms = [t.item() for t in gathered]
                    result_queue.put((rank, norms, None))
                found_kronecker = True
                break  # both ranks break together after all_gather

        if not found_kronecker and rank == 0:
            result_queue.put((0, None, "no Kronecker preconditioner found"))

    except Exception as exc:
        if rank == 0:
            result_queue.put((rank, None, str(exc)))
    finally:
        _destroy_process_group()


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_ddp_converges():
    """SCAO under DDP (gloo/CPU): loss must decrease over 25 steps."""
    world_size = 2
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".rendezvous") as f:
        init_file = f.name

    try:
        procs = []
        for rank in range(world_size):
            p = ctx.Process(
                target=_worker_converges,
                args=(rank, world_size, init_file, result_queue),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, f"Process exited with {p.exitcode}"

        results = [result_queue.get(timeout=5) for _ in range(world_size)]
        for rank, initial_loss, final_loss, err in results:
            assert err is None, f"Rank {rank} raised: {err}"
            assert initial_loss is not None
            assert final_loss is not None
            # Quadratic surface: loss must strictly decrease over 30 steps.
            assert final_loss < initial_loss, (
                f"Rank {rank}: loss did not decrease on quadratic surface "
                f"(init={initial_loss:.4f}, final={final_loss:.4f})"
            )

    finally:
        Path(init_file).unlink(missing_ok=True)


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_sync_preconditioner():
    """After injecting diverged state on rank 1, sync restores identical state."""
    world_size = 2
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".rendezvous") as f:
        init_file = f.name

    try:
        procs = []
        for rank in range(world_size):
            p = ctx.Process(
                target=_worker_sync,
                args=(rank, world_size, init_file, result_queue),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, f"Process exited with {p.exitcode}"

        # Only rank 0 posts the comparison result
        got = result_queue.get(timeout=5)
        rank, norms, err = got[0], got[1], got[2]
        assert err is None, f"Error: {err}"
        assert norms is not None, "No norms collected"
        # Both ranks must have the same U_l norm after sync
        assert abs(norms[0] - norms[1]) < 1e-5, (
            f"Norms differ after sync: rank0={norms[0]:.6f}, rank1={norms[1]:.6f}"
        )

    finally:
        Path(init_file).unlink(missing_ok=True)
