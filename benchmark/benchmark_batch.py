"""
benchmark/benchmark_batch.py — Phase 3: Batch Size & Memory Scaling

IEEE Benchmark Study: Evaluating Transformer and State Space Models
for Long-Context Inference on 6GB Consumer GPUs

Evaluates batch sizes [1, 2, 4] across context lengths [512..8192]
for Pythia-410M and Mamba-370M. Records peak allocated VRAM, reserved
VRAM, static weight memory, additional overhead, persistent state
memory, latency, and throughput. OOM conditions are handled gracefully with
strict garbage collection and tensor deallocation.

Throughput Definition:
    Throughput (tokens/sec) = (batch_size * sequence_length) / latency_sec

Hardware Target:
    NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB GDDR6 dedicated VRAM)
    PyTorch Allocator Ceiling: 5,800 MB (via torch.cuda.set_per_process_memory_fraction)

Outputs:
    - phase3_batch.csv (and batch_scaling_results.csv for compatibility)

Usage:
    OMP_NUM_THREADS=4 python -m benchmark.benchmark_batch [--output-dir <path>] [--smoke-test]
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

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    MambaLMHeadModel = None
    InferenceParams = None

from benchmark.system_monitor import (
    log_system_metadata,
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
if torch.cuda.is_available():
    enforce_vram_ceiling(STRICT_VRAM_LIMIT_MB)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192]
BATCH_SIZES = [1, 2, 4]
REPEATS = 3
SHAPE_WARMUPS = 1
DEFAULT_OUTPUT_DIR = "results/validated_results"

CSV_HEADER = [
    "Model",
    "Batch",
    "Sequence_Length",
    "Peak_Allocated_VRAM_MB",
    "Peak_Reserved_VRAM_MB",
    "Static_Weight_MB",
    "Additional_Overhead_MB",
    "Persistent_State_MB",
    "State_Logical_MB",
    "Throughput_tokens_per_sec",
    "Latency_sec",
    "Latency_std_sec",
    "ms_per_token",
    "Strict_VRAM_Cap_MB",
    "Within_Allocator_Cap",
]
N_METRICS = len(CSV_HEADER) - 3


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
        if InferenceParams is None:
            raise RuntimeError("mamba_ssm is required to run Mamba models.")
        ip = InferenceParams(max_seqlen=seq_len, max_batch_size=batch)
        return model(input_ids, inference_params=ip)
    return model(input_ids, use_cache=True)


def measure_state_memory(
    model, seq_len: int, batch: int, is_mamba: bool
) -> Tuple[float, float]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ids = torch.randint(0, 1000, (batch, seq_len), dtype=torch.long, device=DEVICE)
    base = mb_allocated()
    state = None

    try:
        with torch.no_grad():
            if is_mamba:
                if InferenceParams is None:
                    raise RuntimeError("mamba_ssm is required to run Mamba models.")
                ip = InferenceParams(max_seqlen=seq_len, max_batch_size=batch)
                model.backbone(ids, inference_params=ip)
                state = ip.key_value_memory_dict
            else:
                out = model.gpt_neox(ids, use_cache=True)
                state = out.past_key_values
                if hasattr(state, "to_legacy_cache"):
                    state = state.to_legacy_cache()
                del out

        sync_cuda()
        delta = mb_allocated() - base
        _, logical_mb = cache_size_mb(state)
        return delta, logical_mb
    finally:
        del state, ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def warm_up(model, is_mamba: bool = False) -> None:
    with torch.no_grad():
        dummy = torch.randint(0, 1000, (1, 64), dtype=torch.long, device=DEVICE)
        out = full_forward(model, dummy, 64, 1, is_mamba)
        del out
        if is_mamba:
            if InferenceParams is not None:
                ip = InferenceParams(max_seqlen=64, max_batch_size=1)
                model.backbone(dummy, inference_params=ip)
        else:
            model.gpt_neox(dummy, use_cache=True)
    sync_cuda()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def oom_row(model_name: str, batch: int, seq_len: int) -> List:
    return [model_name, batch, seq_len] + ["OOM_6GB_CAP"] * N_METRICS


def measure_batch_scaling(
    model_name: str,
    model,
    seq_len: int,
    batch: int,
    weights_mb: float,
    is_mamba: bool,
    repeats: int,
) -> List:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pre-flight VRAM guard: weights are already in cur_alloc
    if is_mamba:
        est_state = estimate_mamba_state_mb(48, 1024, 16, batch)
    else:
        est_state = estimate_transformer_kv_cache_mb(24, 16, 64, seq_len, batch)
    est_act = estimate_activation_overhead_mb(seq_len, batch, 1024)
    additional_needed_mb = est_state + est_act
    fits, cur_alloc, avail = check_vram_headroom(additional_needed_mb)
    if not fits:
        print(
            f"  {model_name:12} | {batch:5} | {seq_len:7} | "
            f"VRAM_LIMIT_EXCEEDED (6GB Cap) — need additional ~{additional_needed_mb:.0f} MB, "
            f"only {avail:.0f} MB available"
        )
        return [model_name, batch, seq_len] + ["VRAM_LIMIT_EXCEEDED"] * N_METRICS

    input_ids = None
    try:
        input_ids = torch.randint(
            0, 1000, (batch, seq_len), dtype=torch.long, device=DEVICE
        )

        with torch.no_grad():
            for _ in range(SHAPE_WARMUPS):
                out = full_forward(model, input_ids, seq_len, batch, is_mamba)
                del out
            sync_cuda()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            times = []
            for _ in range(repeats):
                sync_cuda()
                start = time.perf_counter()
                out = full_forward(model, input_ids, seq_len, batch, is_mamba)
                sync_cuda()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                del out

            peak_alloc_mb = mb_allocated() if not torch.cuda.is_available() else torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved_mb = mb_reserved() if not torch.cuda.is_available() else torch.cuda.max_memory_reserved() / (1024**2)

        del input_ids
        input_ids = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        state_mb, state_logical_mb = measure_state_memory(
            model, seq_len, batch, is_mamba
        )

        latency = statistics.mean(times)
        latency_std = statistics.stdev(times) if len(times) > 1 else 0.0
        total_tokens = batch * seq_len
        throughput = total_tokens / latency if latency > 0 else 0.0
        ms_per_token = (latency / total_tokens) * 1000 if total_tokens > 0 else 0.0
        overhead_mb = max(peak_alloc_mb - weights_mb - state_mb, 0.0)
        within_cap = peak_alloc_mb <= STRICT_VRAM_LIMIT_MB

        row = [
            model_name,
            batch,
            seq_len,
            round(peak_alloc_mb, 2),
            round(peak_reserved_mb, 2),
            round(weights_mb, 2),
            round(overhead_mb, 2),
            round(state_mb, 2),
            round(state_logical_mb, 2),
            round(throughput, 2),
            round(latency, 4),
            round(latency_std, 4),
            round(ms_per_token, 4),
            STRICT_VRAM_LIMIT_MB,
            within_cap,
        ]
        return row

    except torch.cuda.OutOfMemoryError:
        if input_ids is not None:
            del input_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return oom_row(model_name, batch, seq_len)
    except Exception as e:
        if "out of memory" in str(e).lower():
            if input_ids is not None:
                del input_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return oom_row(model_name, batch, seq_len)
        raise e


def run_model(
    model_name: str,
    hf_name: str,
    is_mamba: bool,
    batch_sizes: List[int],
    seq_lengths: List[int],
    repeats: int,
) -> List[List]:
    print(f"\n{'='*75}")
    print(f"  Loading {model_name} ({hf_name}) ...")
    print(f"{'='*75}")

    if is_mamba:
        if MambaLMHeadModel is None:
            print("  [ERROR] mamba_ssm package is not installed. Skipping Mamba.")
            return [oom_row(model_name, b, s) for b in batch_sizes for s in seq_lengths]
        model = MambaLMHeadModel.from_pretrained(
            hf_name, device=DEVICE, dtype=torch.float32
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=torch.float32
        ).to(DEVICE)

    model.eval()
    warm_up(model, is_mamba)
    weights_mb = mb_allocated()
    print(f"  Static weight memory: {weights_mb:.2f} MB")

    print(
        f"\n  {'Model':12} | {'Batch':>5} | {'SeqLen':>7} | {'PeakAlloc':>10} | "
        f"{'PeakRsrv':>10} | {'Weights':>8} | {'Overhead':>9} | "
        f"{'State':>8} | {'Logical':>8} | {'Tput(t/s)':>10} | "
        f"{'Lat(s)':>8} | {'Std(s)':>7} | {'ms/tok':>7}"
    )
    print(f"  {'-'*132}")

    rows = []
    for batch in batch_sizes:
        oom_encountered = False
        for seq_len in seq_lengths:
            if oom_encountered:
                row = oom_row(model_name, batch, seq_len)
                rows.append(row)
                print(
                    f"  {model_name:12} | {batch:5} | {seq_len:7} | "
                    f"OOM_6GB_CAP (skipped — subsequent length after OOM at batch {batch})"
                )
                continue

            result = measure_batch_scaling(
                model_name, model, seq_len, batch, weights_mb, is_mamba, repeats
            )
            rows.append(result)
            is_failed = any("OOM" in str(x) or "LIMIT" in str(x) for x in result)
            if is_failed:
                oom_encountered = True
                print(
                    f"  {model_name:12} | {batch:5} | {seq_len:7} | "
                    f"OOM_6GB_CAP (strict 6GB ceiling enforced)"
                )
            else:
                within_cap = result[14] if len(result) > 14 else "N/A"
                print(
                    f"  {result[0]:12} | {result[1]:5} | {result[2]:7} | "
                    f"{result[3]:10.2f} | {result[4]:10.2f} | {result[5]:8.2f} | "
                    f"{result[6]:9.2f} | {result[7]:8.2f} | {result[8]:8.2f} | "
                    f"{result[9]:10.2f} | {result[10]:8.4f} | {result[11]:7.4f} | "
                    f"{result[12]:7.4f} | Cap={STRICT_VRAM_LIMIT_MB:.0f} | {within_cap}"
                )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Batch Size & Memory Scaling Benchmark")
    parser.add_argument("--smoke-test", action="store_true", help="Run single configuration smoke test")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save CSV outputs")
    args = parser.parse_args()

    smoke_test = args.smoke_test
    output_dir = args.output_dir
    batch_sizes = [1] if smoke_test else BATCH_SIZES
    seq_lengths = [512] if smoke_test else SEQ_LENGTHS
    repeats = 1 if smoke_test else REPEATS

    os.makedirs(output_dir, exist_ok=True)
    log_system_metadata(os.path.join(output_dir, "system_metadata.csv"))

    print("\n" + "=" * 75)
    print("  Phase 3: Batch Size & Memory Scaling Benchmark")
    print("  IEEE Study: Long-Context Inference on 6GB Consumer GPUs")
    if smoke_test:
        print("  *** RUNNING IN SMOKE-TEST MODE (Batch=1, SeqLen=512, 1 repeat, Pythia only) ***")
    print(f"  Target Output Directory: {output_dir}")
    print("=" * 75)

    all_rows = []

    # Model 1: Pythia-410M (Transformer baseline)
    pythia_rows = run_model("Pythia-410M", "EleutherAI/pythia-410m", is_mamba=False,
                            batch_sizes=batch_sizes, seq_lengths=seq_lengths, repeats=repeats)
    all_rows.extend(pythia_rows)

    # Model 2: Mamba-370M (SSM baseline)
    if not smoke_test:
        mamba_rows = run_model("Mamba-370M", "state-spaces/mamba-370m", is_mamba=True,
                               batch_sizes=batch_sizes, seq_lengths=seq_lengths, repeats=repeats)
        all_rows.extend(mamba_rows)

    # Export to CSV
    phase3_csv = os.path.join(output_dir, "phase3_batch.csv")
    with open(phase3_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_rows)

    compat_csv = os.path.join(output_dir, "batch_scaling_results.csv")
    with open(compat_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_rows)

    print(f"\nPhase 3 complete. Results exported to: '{phase3_csv}' and '{compat_csv}'")
    print(f"Total configurations evaluated: {len(all_rows)}")
    oom_count = sum(1 for r in all_rows if "OOM" in r)
    print(f"Total OOM events handled safely: {oom_count}")


if __name__ == "__main__":
    main()
