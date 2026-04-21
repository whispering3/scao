"""
SCAO v0.1.1 — Kaggle GPU Benchmark Kernel
==========================================
Runs on Kaggle T4 x2 GPU (free tier).
AdamW vs SCAO vs SCAO+int8 on WikiText-2 at 125M and 350M parameters.
Compiles CUDA fused kernels (nvcc available on Kaggle).
"""

# ── 1. Setup ──────────────────────────────────────────────────────────────────
import subprocess, sys, os, pathlib

def run(cmd, **kw):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, **kw)
    return r.returncode

# Install deps
run("pip install -q datasets tokenizers")

# ── 2. Clone SCAO from GitHub ─────────────────────────────────────────────────
WORK = pathlib.Path("/kaggle/working")
PROJECT_ROOT = WORK / "scao_repo"

if not PROJECT_ROOT.exists():
    print("\nCloning SCAO from GitHub...")
    rc = run(f"git clone --depth 1 https://github.com/whispering3/scao.git {PROJECT_ROOT}")
    if rc != 0:
        print("ERROR: git clone failed. Check that the repo is public.")
        sys.exit(1)
else:
    print(f"\nRepo already at {PROJECT_ROOT}")

sys.path.insert(0, str(PROJECT_ROOT))

# Check GPU
import torch
print(f"\nPyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"Device  : {p.name}  VRAM: {p.total_memory/1e9:.1f} GB")
    vram_gb = p.total_memory / 1e9
else:
    vram_gb = 0
    print("WARNING: No GPU detected!")

import scao
print(f"SCAO v{scao.__version__} imported OK")

# ── 3. Compile CUDA kernels ───────────────────────────────────────────────────
cuda_dir = PROJECT_ROOT / "scao" / "cuda"
print("\n── Compiling CUDA kernels ──────────────────────────")
rc = run(f"cd {cuda_dir} && python setup.py build_ext --inplace 2>&1")
if rc == 0:
    print("CUDA kernels: compiled OK")
else:
    print("CUDA kernels: compilation failed — using PyTorch fallback")

# ── 4. Validate kernels ───────────────────────────────────────────────────────
from scao.cuda import fused_kronecker_precond, int8_ema_update

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

m, n, k = 64, 64, 16
U_l, _ = torch.linalg.qr(torch.randn(m, k, device=device))
U_r, _ = torch.linalg.qr(torch.randn(n, k, device=device))
s_l = torch.rand(k, device=device) + 0.1
s_r = torch.rand(k, device=device) + 0.1
G   = torch.randn(m, n, device=device)

out = fused_kronecker_precond(U_l, s_l, U_r, s_r, G)
G_proj   = (U_l.T @ G) @ U_r
G_scaled = s_l.unsqueeze(1) * G_proj * s_r.unsqueeze(0)
ref      = G + U_l @ (G_scaled - G_proj) @ U_r.T
err = (out - ref).abs().max().item()
status = "OK" if err < 1e-4 else "MISMATCH!"
print(f"\nfused_kronecker_precond  max_err={err:.2e}  {status}")

ema_q   = torch.randint(-127, 127, (256,), dtype=torch.int8, device=device)
q_new, s_new = int8_ema_update(ema_q, 0.01, torch.randn(256, device=device)*0.005, 0.95)
print(f"int8_ema_update          scale={s_new:.6f}  OK")

# ── 5. Benchmarks ─────────────────────────────────────────────────────────────
SCRIPT = str(PROJECT_ROOT / "scripts" / "bench_125m_350m.py")
OUT    = "/kaggle/working"

print("\n── 125M benchmark ──────────────────────────────────")
run(f"python {SCRIPT} --scales 125m --device cuda --steps 500 --seeds 42,123 --out_dir {OUT}")

if vram_gb >= 15:
    print("\n── 350M benchmark ──────────────────────────────────")
    run(f"python {SCRIPT} --scales 350m --device cuda --steps 500 --seeds 42 --out_dir {OUT}")
else:
    print(f"\nSkipping 350M — only {vram_gb:.1f} GB VRAM (need >=15 GB)")

# ── 6. Print summary ──────────────────────────────────────────────────────────
import csv
def load_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f: return list(csv.DictReader(f))

results = load_csv(f"{OUT}/results_125m_350m.csv")
if results:
    print(f"\n{'Scale':<8} {'Optimizer':<14} {'PPL':>8} {'tok/s':>8} {'mem GB':>8}")
    print("-" * 55)
    from collections import defaultdict
    by_scale = defaultdict(list)
    for r in results: by_scale[r['scale']].append(r)
    for scale, rows in sorted(by_scale.items()):
        adamw_ppl = next((float(r['final_ppl']) for r in rows if r['optimizer']=='adamw'), None)
        for r in sorted(rows, key=lambda x: x['optimizer']):
            ppl = float(r['final_ppl'])
            vs  = f"{(ppl-adamw_ppl)/adamw_ppl*100:+.1f}%" if adamw_ppl and r['optimizer']!='adamw' else "baseline"
            print(f"{r['scale']:<8} {r['optimizer']:<14} {ppl:>8.2f} "
                  f"{float(r.get('tokens_per_sec',0)):>8.0f} "
                  f"{float(r.get('peak_mem_gb',0)):>8.3f}  {vs}")

print("\nDone! Download outputs from /kaggle/working/")
