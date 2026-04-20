#!/usr/bin/env bash
# run_experiment.sh — Reproduce all SCAO paper results in one command
#
# Usage:
#   bash scripts/run_experiment.sh             # full suite (CPU, tiny model)
#   bash scripts/run_experiment.sh --quick     # smoke test (100 steps, 1 seed)
#   bash scripts/run_experiment.sh --gpu       # GPU run at 125m scale
#
# Requirements:
#   pip install datasets tokenizers   (HuggingFace data loader)
#   pip install -e ".[dev]"           (SCAO package, dev mode)
#
# Results are written to:
#   results_paper_<scale>.csv
#   results_paper_<scale>_curves.csv

set -e
QUICK=0
GPU=0
for arg in "$@"; do
  [[ "$arg" == "--quick" ]] && QUICK=1
  [[ "$arg" == "--gpu"   ]] && GPU=1
done

SEEDS="42,123"
STEPS=500
SCALE="tiny"
DEVICE="cpu"

if [[ $QUICK -eq 1 ]]; then SEEDS="42"; STEPS=100; fi
if [[ $GPU   -eq 1 ]]; then DEVICE="cuda"; SCALE="125m"; STEPS=5000; fi

echo "============================================================"
echo " SCAO Experiment Suite"
echo " scale=${SCALE}  device=${DEVICE}  seeds=${SEEDS}  steps=${STEPS}"
echo "============================================================"

# Primary benchmark: SCAO vs AdamW vs DiagShampoo
echo ""
echo "--- GPT-scale benchmark ---"
python scao/benchmarks/gpt_scale_benchmark.py \
  --scale  "$SCALE"  \
  --device "$DEVICE" \
  --steps  "$STEPS"  \
  --seeds  "$SEEDS"  \
  --optimizers "adamw,scao,diag_shampoo" \
  --csv    "results_paper_${SCALE}.csv"

echo ""
echo "All done. Results in results_paper_${SCALE}.csv"
echo "Open scripts/scao_colab_benchmark.ipynb for GPU benchmarks at 125M+ scale."
