#!/bin/bash
set -e
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /mnt/e/New
source /home/yugpa/benchmark-env/bin/activate

echo "============================================================"
echo "  Executing Phase 5: Long-Context Retrieval Benchmark"
echo "  $(date)"
echo "============================================================"

python3 -m evaluation.evaluate_retrieval
echo "Phase 5 Complete. Exit code: $?"
