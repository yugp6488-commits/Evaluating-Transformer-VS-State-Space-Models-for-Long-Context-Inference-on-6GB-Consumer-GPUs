#!/bin/bash
# =============================================================================
# Strict 6GB VRAM Ceiling — Full Benchmark Re-execution
# IEEE Study: Evaluating Transformer and State Space Models
# for Long-Context Inference on 6GB Consumer GPUs
#
# Enforces set_per_process_memory_fraction(5800/total) on every script.
# =============================================================================
set -e

export OMP_NUM_THREADS=4

# Activate the dedicated benchmark environment
source /home/yugpa/benchmark-env/bin/activate

cd /mnt/e/New

echo "============================================================"
echo "  Strict 6GB VRAM Ceiling — Phase 1 (Baseline Prefill)"
echo "  $(date)"
echo "============================================================"
python run_benchmark.py

echo ""
echo "============================================================"
echo "  Strict 6GB VRAM Ceiling — Phase 2 (Decode Scaling)"
echo "  $(date)"
echo "============================================================"
python -m benchmark.benchmark_decode

echo ""
echo "============================================================"
echo "  Strict 6GB VRAM Ceiling — Phase 3 (Batch Scaling)"
echo "  $(date)"
echo "============================================================"
python -m benchmark.benchmark_batch

echo ""
echo "============================================================"
echo "  Strict 6GB VRAM Ceiling — Phase 4 (Quality PPL & BPB)"
echo "  $(date)"
echo "============================================================"
python -m evaluation.evaluate_quality

echo ""
echo "============================================================"
echo "  Strict 6GB VRAM Ceiling — Phase 5 (Retrieval)"
echo "  $(date)"
echo "============================================================"
python -m evaluation.evaluate_retrieval

echo ""
echo "============================================================"
echo "  All Phases Complete with Strict 6GB VRAM Cap"
echo "  $(date)"
echo "============================================================"
