#!/bin/bash
set -e
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /mnt/e/New
source /home/yugpa/benchmark-env/bin/activate

echo "============================================================"
echo "  Executing Phase 2: Decoding Benchmark"
echo "  $(date)"
echo "============================================================"

python3 -m benchmark.benchmark_decode
echo "Phase 2 Complete. Exit code: $?"
