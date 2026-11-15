#!/bin/bash
set -e
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /mnt/e/New
source /home/yugpa/benchmark-env/bin/activate

echo "============================================================"
echo "  Executing Phase 4: Language Modelling Quality & BPB"
echo "  $(date)"
echo "============================================================"

python3 -m evaluation.evaluate_quality
echo "Phase 4 Complete. Exit code: $?"
