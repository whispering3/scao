"""
SCAO - Sparse Curvature-Aware Adaptive Optimizer
=================================================
A second-order PyTorch optimizer that delivers Shampoo-quality
preconditioned gradients at near-AdamW memory and throughput cost.

Quickstart
----------
    import scao

    optimizer = scao.SCAO(model.parameters(), lr=3.5e-4)

    # or, explicit:
    from scao import SCAO
    optimizer = SCAO(
        model.parameters(),
        lr=3.5e-4,
        weight_decay=0.1,
        warmup_steps=100,
        precond_freq=10,
    )

Paper
-----
SCAO: Sparse Curvature-Aware Adaptive Optimization for Large-Scale Models
NeurIPS 2026 (under review)
"""

from .optimizer import SCAO
from .preconditioner import SparsePreconditioner
from .utils import matrix_power_neg_quarter, adaptive_rank
from . import logging as scao_logging

__version__ = "0.1.1"
__author__ = "SCAO Authors"
__license__ = "Apache-2.0"

__all__ = [
    # Main API — this is all most users need
    "SCAO",
    # Advanced / internals
    "SparsePreconditioner",
    "matrix_power_neg_quarter",
    "adaptive_rank",
    "scao_logging",
    # Metadata
    "__version__",
]

