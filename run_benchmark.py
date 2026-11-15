"""
run_benchmark.py — Phase 1: Prefill Latency and Memory Profiling

IEEE Benchmark Study: Evaluating Transformer and State Space Models
for Long-Context Inference on 6GB Consumer GPUs

Measures prefill latency, throughput, peak allocated VRAM, peak reserved VRAM,
and persistent inference-state memory across context lengths [512..8192] in FP32
precision for Pythia-410M and Mamba-370M.

Hardware Target:
    NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB GDDR6 dedicated physical VRAM)
    PyTorch Allocator Ceiling: 5,800 MB (via torch.cuda.set_per_process_memory_fraction)

Outputs:
    - phase1_prefill.csv (and hardware_results_validated.csv for compatibility)

Usage:
    OMP_NUM_THREADS=4 python run_benchmark.py [--output-dir <path>] [--smoke-test]
"""

import argparse
import csv
import gc
import os
import statistics
import sys
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams

from benchmark.system_monitor import (
    enforce_vram_ceiling,
    check_vram_headroom,
    estimate_transformer_kv_cache_mb,
    estimate_mamba_state_mb,
    estimate_activation_overhead_mb,
    STRICT_VRAM_LIMIT_MB,
)

# ---------------------------------------------------------------------------
# Enforce PyTorch allocator ceiling BEFORE any CUDA allocations
# ---------------------------------------------------------------------------
# NOTE: torch.cuda.set_per_process_memory_fraction limits PyTorch's internal
# caching allocator (c10::cuda::CUDACachingAllocator). It raises a clean OOM
# if tensor allocations exceed the configured ceiling.
# It does NOT cap CUDA driver context, non-PyTorch cudaMalloc, or OS memory.
# See RESEARCH_LIMITATIONS.md §2.4 for full scope discussion.
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    enforce_vram_ceiling(STRICT_VRAM_LIMIT_MB)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192]
BATCH_SIZES = [1]
REPEATS = 3
SHAPE_WARMUPS = 1
DEFAULT_OUTPUT_DIR = "results/validated_results"

CSV_HEADER = [
    "Model",
    "Batch",
    "Sequence_Length",
    "Peak_Allocated_MB",      # PyTorch caching allocator peak (max_memory_allocated)
    "Peak_Reserved_MB",       # PyTorch allocator pool peak (max_memory_reserved)
    "Peak_minus_Weights_MB",  # Peak allocated minus static weight memory
    "State_VRAM_MB",          # Allocated-memory delta of cache without logits
    "State_Logical_MB",       # sum(numel * element_size) of cache tensors
    "Latency_sec",            # Mean prefill latency in seconds
    "Latency_std_sec",        # Standard deviation of prefill latency
    "ms_per_prefill_token",   # Latency per processed input token in ms
    "Prefill_Throughput_tps", # Tokens per second (batch * seq_len / latency)
    "Strict_VRAM_Cap_MB",     # Configured allocator ceiling (5800 MB)
    "Within_Allocator_Cap",   # Whether peak allocations stayed under ceiling
]
N_METRICS = len(CSV_HEADER) - 3


def mb_allocated() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0


def mb_reserved() -> float:
    return torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0


def cache_size_mb(obj) -> Tuple[float, float]:
    seen, alloc, logical = set(), 0, 0

    def walk(x):
        nonlocal alloc, logical
        if torch.is_tensor(x):
            logical += x.numel() * x.element_size()
            try:
                st = x.untyped_storage()
                if st.data_ptr() not in seen:
                    seen.add(st.data_ptr())
                    alloc += st.nbytes()
            except Exception:
                pass
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)

    walk(obj)
    return alloc / (1024 ** 2), logical / (1024 ** 2)


def full_forward(model, input_ids, seq_len: int, batch: int, is_mamba: bool):
    if is_mamba:
        ip = InferenceParams(max_seqlen=seq_len, max_batch_size=batch)
        return model(input_ids, inference_params=ip)
    return model(input_ids, use_cache=True)


def measure_state(model, seq_len: int, batch: int, is_mamba: bool) -> Tuple[float, float]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ids = torch.randint(0, 1000, (batch, seq_len), dtype=torch.long, device=DEVICE)
    base = mb_allocated()

    with torch.no_grad():
        if is_mamba:
            ip = InferenceParams(max_seqlen=seq_len, max_batch_size=batch)
            model.backbone(ids, inference_params=ip)
            state = ip.key_value_memory_dict
        else:
            out = model.gpt_neox(ids, use_cache=True)
            state = out.past_key_values
            if hasattr(state, "to_legacy_cache"):
                state = state.to_legacy_cache()
            del out

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    delta = mb_allocated() - base
    _, logical_mb = cache_size_mb(state)
    del state, ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return delta, logical_mb


def warm_up(model, is_mamba: bool = False):
    with torch.no_grad():
        dummy = torch.randint(0, 1000, (1, 64), dtype=torch.long, device=DEVICE)
        out = full_forward(model, dummy, 64, 1, is_mamba)
        del out
        if is_mamba:
            ip = InferenceParams(max_seqlen=64, max_batch_size=1)
            model.backbone(dummy, inference_params=ip)
        else:
            model.gpt_neox(dummy, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def oom_row(model_name: str, batch: int, seq_len: int) -> List:
    return [model_name, batch, seq_len] + ["OOM_6GB_CAP"] * N_METRICS


def measure_prefill(
    model_name: str,
    model,
    seq_len: int,
    batch: int,
    weights_mb: float,
    is_mamba: bool = False,
) -> Optional[List]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pre-flight VRAM guard: weights are already in cur_alloc, do not double-count
    if is_mamba:
        est_state = estimate_mamba_state_mb(48, 1024, 16, batch)
    else:
        est_state = estimate_transformer_kv_cache_mb(24, 16, 64, seq_len, batch)
    est_act = estimate_activation_overhead_mb(seq_len, batch, 1024)
    additional_needed_mb = est_state + est_act
    fits, cur_alloc, avail = check_vram_headroom(additional_needed_mb)
    if not fits:
        print(f"  {model_name:12} | {batch:<5} | {seq_len:<7} | VRAM_LIMIT_EXCEEDED (6GB Cap) "
              f"— need additional ~{additional_needed_mb:.0f} MB, only {avail:.0f} MB available")
        return [model_name, batch, seq_len] + ["VRAM_LIMIT_EXCEEDED"] * N_METRICS

    input_ids = torch.randint(0, 1000, (batch, seq_len), dtype=torch.long, device=DEVICE)

    try:
        with torch.no_grad():
            for _ in range(SHAPE_WARMUPS):
                out = full_forward(model, input_ids, seq_len, batch, is_mamba)
                del out
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(REPEATS):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                out = full_forward(model, input_ids, seq_len, batch, is_mamba)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                del out

            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        del input_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        state_mb, state_logical_mb = measure_state(model, seq_len, batch, is_mamba)

        latency = statistics.mean(times)
        latency_std = statistics.stdev(times) if len(times) > 1 else 0.0
        total_tokens = batch * seq_len
        ms_per_token = (latency / total_tokens) * 1000 if total_tokens > 0 else 0.0
        throughput = total_tokens / latency if latency > 0 else 0.0
        within_cap = peak_alloc_mb <= STRICT_VRAM_LIMIT_MB

        row = [
            model_name,
            batch,
            seq_len,
            round(peak_alloc_mb, 2),
            round(peak_reserved_mb, 2),
            round(peak_alloc_mb - weights_mb, 2),
            round(state_mb, 2),
            round(state_logical_mb, 2),
            round(latency, 4),
            round(latency_std, 4),
            round(ms_per_token, 4),
            round(throughput, 2),
            STRICT_VRAM_LIMIT_MB,
            within_cap,
        ]

        print(f"  {model_name:12} | {batch:<5} | {seq_len:<7} | {peak_alloc_mb:<10.2f} | "
              f"{peak_reserved_mb:<10.2f} | {peak_alloc_mb - weights_mb:<11.2f} | {state_mb:<9.2f} | "
              f"{state_logical_mb:<10.2f} | {latency:<8.4f} | {latency_std:<7.4f} | "
              f"{throughput:<9.2f} | Cap={STRICT_VRAM_LIMIT_MB:.0f} | {within_cap}")
        return row

    except torch.cuda.OutOfMemoryError:
        print(f"  {model_name:12} | {batch:<5} | {seq_len:<7} | OOM_6GB_CAP (strict 6GB ceiling enforced)")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return oom_row(model_name, batch, seq_len)


def run_model(model_name: str, hf_name: str, is_mamba: bool, seq_lengths: List[int], repeats: int) -> List[List]:
    print(f"\nLoading {hf_name} ...")
    if is_mamba:
        model = MambaLMHeadModel.from_pretrained(hf_name, device=DEVICE, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float32).to(DEVICE)

    model.eval()
    warm_up(model, is_mamba)
    weights_mb = mb_allocated()
    print(f"Static weight memory: {weights_mb:.2f} MB")

    rows = []
    failed = False
    for b in BATCH_SIZES:
        for l in seq_lengths:
            if failed:
                row = oom_row(model_name, b, l)
                rows.append(row)
                print(f"  {model_name:12} | {b:<5} | {l:<7} | OOM_6GB_CAP (skipped)")
                continue

            row = measure_prefill(model_name, model, l, b, weights_mb, is_mamba)
            rows.append(row)
            if any("OOM" in str(x) or "LIMIT" in str(x) for x in row):
                failed = True

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Prefill Latency and Memory Benchmark")
    parser.add_argument("--smoke-test", action="store_true", help="Run single configuration smoke test")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save CSV outputs")
    args = parser.parse_args()

    smoke_test = args.smoke_test
    output_dir = args.output_dir
    seq_lengths = [512] if smoke_test else SEQ_LENGTHS
    repeats = 1 if smoke_test else REPEATS

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  Phase 1: Prefill Latency & Memory Profiling (FP32, Strict 6GB VRAM Cap)")
    print("  IEEE Study: Long-Context Inference on 6GB Consumer GPUs")
    if smoke_test:
        print("  *** RUNNING IN SMOKE-TEST MODE (512 tokens, 1 repeat, Pythia only) ***")
    print(f"  Target Output Directory: {output_dir}")
    print("=" * 80)

    print(f"  {'Model':12} | {'Batch':<5} | {'Tokens':<7} | {'PeakAlloc':<10} | "
          f"{'PeakRsrv':<10} | {'Peak-Wts':<11} | {'State(MB)':<9} | {'Logical':<10} | "
          f"{'Time(s)':<8} | {'Std(s)':<7} | {'Tput(t/s)':<9} | Ceiling")
    print("-" * 135)

    all_results = []

    # 1. TEST TRANSFORMER (PYTHIA)
    pythia_rows = run_model("Pythia-410M", "EleutherAI/pythia-410m", is_mamba=False, seq_lengths=seq_lengths, repeats=repeats)
    all_results.extend(pythia_rows)

    # 2. TEST STATE SPACE MODEL (MAMBA)
    if not smoke_test:
        mamba_rows = run_model("Mamba-370M", "state-spaces/mamba-370m", is_mamba=True, seq_lengths=seq_lengths, repeats=repeats)
        all_results.extend(mamba_rows)

    # Export results
    phase1_csv = os.path.join(output_dir, "phase1_prefill.csv")
    with open(phase1_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_results)

    # Also save hardware_results_validated.csv for backwards compatibility
    compat_csv = os.path.join(output_dir, "hardware_results_validated.csv")
    with open(compat_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_results)

    print(f"\nPhase 1 benchmark complete. Results saved to '{phase1_csv}' and '{compat_csv}'.")


if __name__ == "__main__":
    main()
