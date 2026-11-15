#!/bin/bash
# run_all_phases.sh — Execute all benchmark phases sequentially
# Must be run from the project root directory

set -e
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PROJECT_DIR="/mnt/e/New"
VENV_DIR="/mnt/e/sequence-model-benchmark/.venv"

cd "$PROJECT_DIR"

# Activate the existing venv from the sibling project
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "============================================"
echo "  Python: $(python3 --version)"
echo "  Working Dir: $(pwd)"
echo "============================================"

# Check torch + CUDA
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_mem / (1024**2)), 'MB')
"

echo ""
echo "========== PHASE 2: DECODING BENCHMARK =========="
python3 -m benchmark.benchmark_decode 2>&1 || echo "[Phase 2 encountered an error]"

echo ""
echo "========== PHASE 3: BATCH SCALING BENCHMARK =========="
python3 -m benchmark.benchmark_batch 2>&1 || echo "[Phase 3 encountered an error]"

echo ""
echo "========== PHASE 4: QUALITY EVALUATION =========="
python3 -m evaluation.evaluate_quality 2>&1 || echo "[Phase 4 encountered an error]"

echo ""
echo "========== PHASE 5: RETRIEVAL EVALUATION =========="
python3 -m evaluation.evaluate_retrieval 2>&1 || echo "[Phase 5 encountered an error]"

echo ""
echo "========== PHASE 6: VISUALIZATION =========="
python3 -m analysis.visualization 2>&1 || echo "[Phase 6 encountered an error]"

echo ""
echo "========== ALL PHASES COMPLETE =========="
echo "Results in: results/raw/"
ls -la results/raw/ 2>/dev/null
echo ""
echo "Graphs in: results/graphs/"
ls -la results/graphs/ 2>/dev/null
