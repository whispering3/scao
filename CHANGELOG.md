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

## [0.1.1] — 2026-04-20

### Added

#### CUDA fused kernels (`scao/cuda/low_rank_ops.cu` — complete rewrite)
- **Tiled shared-memory GEMM kernels** (`tiled_AtB_kernel`, `tiled_AB_kernel`):
  16×16 tile blocking; eliminates redundant global-memory reads for Kronecker projections
- **Fused Kronecker preconditioner kernel** (`fused_kronecker_precond_kernel`, k ≤ 128):
  computes identity-correction `G + U_l @ (s·G_proj - G) @ U_r^T` in a single launch,
  avoiding materialisation of the intermediate `(m, n)` tensor
- **Int8 EMA update kernels** (`int8_ema_update_pass1/pass2`):
  dequantize → EMA blend → requantize in two fused CUDA passes
- **Bug fix**: original kernel had O(k·m²·n) complexity (each output thread recomputed
  entire `U^T @ G` projection); rewrite achieves correct O(k·m·n)
- **Multi-arch support**: added `sm_70` (V100), `sm_75` (T4/RTX 20xx),
  `sm_86` (RTX 30xx/A40), `sm_90` (H100 SXM) to nvcc gencode list

#### Int8 EMA curvature accumulators
- `SCAO(..., use_int8_ema=True)` — new flag (default `False`, fully backward-compatible)
- Curvature factors `L_ema`, `R_ema` stored as int8 + per-tensor float32 scale
  (symmetric quantisation: `scale = max(|x|) / 127`)
- **~4× EMA memory reduction**: e.g. for d_model=768 each factor compresses
  768²×4 B = 2.25 MB → ~566 KB + 4 B scale
- Eigendecomposition still runs in float32 (dequantised on-the-fly)
- Full `state_dict` / `load_state_dict` support for both fp32 and int8 paths
- `SparsePreconditioner.memory_bytes()` reports correct int8 footprint
- New helpers in `scao/utils.py`: `quantize_sym_int8()`, `dequantize_sym_int8()`

#### 125M / 350M benchmark infrastructure
- `scao_int8` variant added to `gpt_scale_benchmark.py`
- New convenience script `scripts/bench_125m_350m.py`:
  runs AdamW vs SCAO vs SCAO-int8 at both scales, prints summary table with
  vs-AdamW throughput delta and int8 memory savings, writes
  `results_125m_350m.csv`, curves CSV, and `report_125m_350m.txt`
- Added `--seq_len` flag for CPU smoke tests
- **CPU smoke test results** (5 steps, batch 2, seq_len 64, seed 42):
  - 125M: SCAO 46.75 PPL vs AdamW 63.03 (−25.8%); int8 EMA saves 36.7% memory with zero PPL loss
  - 350M: int8 EMA saves 36.7% memory (8.83→5.59 GB) with zero PPL loss

### Planned
- Full GPU convergence benchmarks at 125M–350M (≥5k steps)
- Evaluation at 1B+ parameter scale
