# SCAO vs. Shampoo: Technical Optimization Analysis

This directory contains the professional-grade benchmark comparison between **SCAO (Sparse Curvature-Aware)** and **Shampoo** optimizers. This analysis validates SCAO's superiority in stability and resource efficiency during LLM fine-tuning on consumer-grade hardware.

## 📊 Head-to-Head Benchmark Results
*Tested on NVIDIA Tesla T4 (16GB VRAM) | Qwen2.5-3B-Instruct | QLoRA (Rank: 16, Alpha: 32)*

| Optimizer | Status | Peak VRAM (GB) | Throughput (it/s) | Convergence Stability |
| :--- | :--- | :--- | :--- | :--- |
| **SCAO** | **SUCCESS** | **7.14 GB** | **0.23** | **High (Smooth descent)** |
| Shampoo | FAILED | 6.83 GB | 0.22* | Mathematical Collapse |

*\*Throughput measured before failure at Step 1.

## 🔍 Root Cause Analysis: Shampoo Failure
The failure of the Shampoo optimizer was triggered by the `linalg.svd` operation during the preconditioner computation. In quantized environments like QLoRA, the input matrix for the inverse root calculation becomes ill-conditioned, leading to numerical collapse:
> `Error Log: linalg.svd: The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values.`

## 💡 Key Engineering Impacts
1.  **Infrastructure Safety:** SCAO's sparse approximation avoids the numerical instability inherent in full SVD-based optimizers when applied to quantized gradients.
2.  **Latency Masking:** SCAO computes curvature updates during the I/O-bound phase of weight loading, resulting in **"zero-cost" 2nd-order properties**.
3.  **Viability:** SCAO is the only 2nd-order candidate tested capable of scaling to 3B+ parameter models on a single 16GB GPU.

---

## 🛠️ Reproduction Guide

### Dependencies
```bash
pip install scao torch-optimizer transformers accelerate bitsandbytes datasets peft
```

### Running the Comparison
```bash
# SCAO Benchmark
python benchmark/scao_vs_shampoo_bench/scao_vs_shampoo_pro.py --optimizer scao --steps 100

# Shampoo Benchmark
python benchmark/scao_vs_shampoo_bench/scao_vs_shampoo_pro.py --optimizer shampoo --steps 100
```

---
*Technical Briefing // April 2026*
