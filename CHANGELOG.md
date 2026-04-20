# Changelog

All notable changes to SCAO are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] — 2026-04-20

### Initial open-source release

#### Core optimizer (`scao.SCAO`)
- Two-phase training: Adam warmup → Kronecker-factored SCAO phase
- Adaptive rank selection via spectral mass threshold (`epsilon_sparse=0.05`)
- Sparse block-diagonal fallback for large layers (`max_precond_dim=4096`)
- Drop-in replacement for `torch.optim.AdamW` — identical training loop

#### Phase-transition stability (three critical fixes)
- **EMA bias correction**: `L/R` factors initialized as `ε·I` (not zero) to prevent
  rank-deficient first preconditioner application
- **50-step cosine blend ramp**: gradual transition from Adam gradient to preconditioned
  gradient prevents momentum disruption at the phase boundary
- **Adaptive Tikhonov regularization**: `eps = max(ε₀, 1e-4 · tr(L)/m)` scales
  regularization to actual curvature magnitude

#### Empirical results (WikiText-2, 838K-param GPT, 500 steps)
- SCAO: **12.03 PPL**, **827 tok/s** vs AdamW: 11.85 PPL, 537 tok/s
- **+54% throughput** over AdamW via amortized curvature updates
- PPL gap narrows from 4% (200 steps) to **1.5% (500 steps)**
- Lower AUC (2.854 vs 2.873) — SCAO has lower mean training loss across the full run

#### Infrastructure
- Full mixed-precision support (bfloat16, float16, float32)
- `torch.compile` compatible (hot-path fully traceable)
- FSDP / DDP compatible via `scao.distributed`
- HuggingFace `Trainer` integration via `scao.integrations.huggingface`
- Callback system for WandB, TensorBoard, console logging
- 32 optimizer tests + 27 profiling tests

---

## [Unreleased]

### Planned
- GPU benchmarks at 125M and 350M parameters (Colab notebook ready)
- CUDA fused kernels for low-rank operations (`k > 128`)
- Quantized curvature factors (int8 EMA accumulators)
- Theoretical convergence analysis extending Shampoo regret bounds
- Evaluation at 1B+ parameter scale
