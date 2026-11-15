#!/bin/bash
# run_all.sh — Execute all IEEE benchmark phases sequentially in WSL2
set -o pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /mnt/e/New

echo "============================================================"
echo "  IEEE Benchmark Suite — Full Execution"
echo "  $(date)"
echo "============================================================"
echo ""

python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    p = torch.cuda.get_device_properties(0)
    print('VRAM:', round(p.total_memory/(1024**2)), 'MB')
"

echo ""
echo "========== SYSTEM METADATA =========="
python3 -m benchmark.system_monitor
echo ""

echo "========== PHASE 2: DECODING BENCHMARK =========="
python3 -m benchmark.benchmark_decode 2>&1
PHASE2_RC=$?
echo "Phase 2 exit code: $PHASE2_RC"
echo ""

echo "========== PHASE 3: BATCH SCALING BENCHMARK =========="
python3 -m benchmark.benchmark_batch 2>&1
PHASE3_RC=$?
echo "Phase 3 exit code: $PHASE3_RC"
echo ""

echo "========== PHASE 4: QUALITY EVALUATION =========="
python3 -m evaluation.evaluate_quality 2>&1
PHASE4_RC=$?
echo "Phase 4 exit code: $PHASE4_RC"
echo ""

echo "========== PHASE 5: RETRIEVAL EVALUATION =========="
python3 -m evaluation.evaluate_retrieval 2>&1
PHASE5_RC=$?
echo "Phase 5 exit code: $PHASE5_RC"
echo ""

echo "========== PHASE 6: VISUALIZATION =========="
python3 -m analysis.visualization 2>&1
PHASE6_RC=$?
echo "Phase 6 exit code: $PHASE6_RC"
echo ""

echo "============================================================"
echo "  ALL PHASES COMPLETE — $(date)"
echo "============================================================"
echo ""
echo "=== CSV Results ==="
ls -la results/raw/ 2>/dev/null
echo ""
echo "=== Generated Graphs ==="
ls -la results/graphs/ 2>/dev/null
echo ""
echo "Exit codes: Phase2=$PHASE2_RC Phase3=$PHASE3_RC Phase4=$PHASE4_RC Phase5=$PHASE5_RC Phase6=$PHASE6_RC"
