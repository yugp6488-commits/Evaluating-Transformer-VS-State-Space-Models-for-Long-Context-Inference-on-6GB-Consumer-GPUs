"""
benchmark/benchmark_decode.py — Phase 2: Token-by-Token Decoding Benchmark

IEEE Benchmark Study: Evaluating Transformer and State Space Models
for Long-Context Inference on 6GB Consumer GPUs

Measures prefill latency, per-token decode latency, decode throughput,
peak allocated VRAM, and peak reserved VRAM after prefill across
sequence lengths [512..8192] for Pythia-410M and Mamba-370M.

Hardware Target:
    NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB GDDR6 dedicated VRAM)
    PyTorch Allocator Ceiling: 5,800 MB (via torch.cuda.set_per_process_memory_fraction)

Decoding Policy:
    Generation length: exactly 128 tokens.
    First-token argmax latency is captured in Prefill_Latency_sec.
    Decode_Latency_per_token_ms instruments the remaining 127 token-by-token steps.

Outputs:
    - phase2_decode.csv (and decode_results.csv for compatibility)

Usage:
    OMP_NUM_THREADS=4 python -m benchmark.benchmark_decode [--output-dir <path>] [--smoke-test]
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
BATCH_SIZE = 1
MAX_NEW_TOKENS = 128  # Frozen at exactly 128 generated tokens
REPEATS = 3
DEFAULT_OUTPUT_DIR = "results/validated_results"

CSV_HEADER = [
    "Model",
    "Prefill_Length",
    "Prefill_Latency_sec",
    "Prefill_Latency_std_sec",
    "Decode_Latency_per_token_ms",
    "Decode_Latency_std_ms",
    "Decode_Throughput_tokens_per_sec",
    "Peak_Allocated_MB",
    "Peak_Reserved_MB",
    "Total_Generated_Tokens",
    "Strict_VRAM_Cap_MB",
    "Within_Allocator_Cap",
]
N_METRICS = len(CSV_HEADER) - 2


def oom_row(model_name: str, seq_len: int) -> List:
    return [model_name, seq_len] + ["OOM_6GB_CAP"] * N_METRICS


def warm_up_model(model, is_mamba: bool) -> None:
    with torch.no_grad():
        dummy = torch.randint(0, 1000, (1, 64), dtype=torch.long, device=DEVICE)
        if is_mamba:
            ip = InferenceParams(max_seqlen=64, max_batch_size=1)
            model(dummy, inference_params=ip)
        else:
            model(dummy, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decode_transformer(model, input_ids: torch.Tensor, max_new_tokens: int) -> Tuple[float, List[float], int]:
    """Prefill then decode token-by-token for a Transformer model."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_time = time.perf_counter() - prefill_start

    decode_times = []
    generated_tokens = [next_token]
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - t0)
            generated_tokens.append(next_token)

    del past_key_values, outputs
    return prefill_time, decode_times, len(generated_tokens)


def decode_mamba(model, input_ids: torch.Tensor, seq_len: int, max_new_tokens: int) -> Tuple[float, List[float], int]:
    """Prefill then decode token-by-token for a Mamba SSM model."""
    total_len = seq_len + max_new_tokens

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    with torch.no_grad():
        ip = InferenceParams(max_seqlen=total_len, max_batch_size=1)
        outputs = model(input_ids, inference_params=ip)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        ip.seqlen_offset = seq_len
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_time = time.perf_counter() - prefill_start

    decode_times = []
    generated_tokens = [next_token]
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(next_token, inference_params=ip)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            ip.seqlen_offset += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - t0)
            generated_tokens.append(next_token)

    del ip, outputs
    return prefill_time, decode_times, len(generated_tokens)


def measure_decode(
    model_name: str,
    model,
    seq_len: int,
    is_mamba: bool,
    repeats: int,
) -> List:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pre-flight VRAM guard: model weights already in cur_alloc
    if is_mamba:
        est_state = estimate_mamba_state_mb(48, 1024, 16, BATCH_SIZE)
    else:
        est_state = estimate_transformer_kv_cache_mb(24, 16, 64, seq_len + MAX_NEW_TOKENS, BATCH_SIZE)
    est_act = estimate_activation_overhead_mb(seq_len, BATCH_SIZE, 1024)
    additional_needed_mb = est_state + est_act
    fits, cur_alloc, avail = check_vram_headroom(additional_needed_mb)
    if not fits:
        print(f"  {model_name:12} | {seq_len:10} | VRAM_LIMIT_EXCEEDED (6GB Cap) "
              f"— need additional ~{additional_needed_mb:.0f} MB, only {avail:.0f} MB available")
        return [model_name, seq_len] + ["VRAM_LIMIT_EXCEEDED"] * N_METRICS

    input_ids = torch.randint(0, 1000, (BATCH_SIZE, seq_len), dtype=torch.long, device=DEVICE)

    try:
        prefill_times = []
        all_decode_times = []

        for _ in range(repeats):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            if is_mamba:
                pf_time, dec_times, n_gen = decode_mamba(model, input_ids, seq_len, MAX_NEW_TOKENS)
            else:
                pf_time, dec_times, n_gen = decode_transformer(model, input_ids, MAX_NEW_TOKENS)

            prefill_times.append(pf_time)
            all_decode_times.extend(dec_times)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        within_cap = peak_alloc_mb <= STRICT_VRAM_LIMIT_MB

        pf_mean = statistics.mean(prefill_times)
        pf_std = statistics.stdev(prefill_times) if len(prefill_times) > 1 else 0.0

        dec_mean_ms = statistics.mean(all_decode_times) * 1000 if all_decode_times else 0.0
        dec_std_ms = statistics.stdev(all_decode_times) * 1000 if len(all_decode_times) > 1 else 0.0
        dec_throughput = 1000.0 / dec_mean_ms if dec_mean_ms > 0 else 0.0

        del input_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        row = [
            model_name,
            seq_len,
            round(pf_mean, 4),
            round(pf_std, 4),
            round(dec_mean_ms, 4),
            round(dec_std_ms, 4),
            round(dec_throughput, 2),
            round(peak_alloc_mb, 2),
            round(peak_reserved_mb, 2),
            n_gen,
            STRICT_VRAM_LIMIT_MB,
            within_cap,
        ]
        return row

    except torch.cuda.OutOfMemoryError:
        del input_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return oom_row(model_name, seq_len)


def run_model(model_name: str, hf_name: str, is_mamba: bool, seq_lengths: List[int], repeats: int) -> List[List]:
    print(f"\n{'='*70}")
    print(f"  Loading {hf_name} ...")
    print(f"{'='*70}")

    if is_mamba:
        model = MambaLMHeadModel.from_pretrained(hf_name, device=DEVICE, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float32).to(DEVICE)

    model.eval()
    warm_up_model(model, is_mamba)

    print(f"  {'Model':12} | {'PrefillLen':>10} | {'PF Lat(s)':>9} | "
          f"{'PF Std(s)':>9} | {'Dec ms/tok':>10} | {'Dec Std':>8} | "
          f"{'Dec t/s':>8} | {'PeakAlloc':>9} | {'PeakRsrv':>9} | {'GenTok':>6}")
    print(f"  {'-'*112}")

    rows = []
    oom_cascade = False
    for seq_len in seq_lengths:
        if oom_cascade:
            row = oom_row(model_name, seq_len)
            rows.append(row)
            print(f"  {model_name:12} | {seq_len:10} | OOM_6GB_CAP (skipped)")
            continue

        result = measure_decode(model_name, model, seq_len, is_mamba, repeats)
        rows.append(result)
        is_failed = any("OOM" in str(x) or "LIMIT" in str(x) for x in result)
        if is_failed:
            oom_cascade = True
            print(f"  {model_name:12} | {seq_len:10} | OOM_6GB_CAP (strict 6GB ceiling enforced)")
        else:
            within_cap = result[11] if len(result) > 11 else "N/A"
            print(f"  {result[0]:12} | {result[1]:10} | {result[2]:9.4f} | "
                  f"{result[3]:9.4f} | {result[4]:10.4f} | {result[5]:8.4f} | "
                  f"{result[6]:8.2f} | {result[7]:9.2f} | {result[8]:9.2f} | {result[9]:6} | "
                  f"Cap={STRICT_VRAM_LIMIT_MB:.0f} | {within_cap}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Token-by-Token Decoding Benchmark")
    parser.add_argument("--smoke-test", action="store_true", help="Run single configuration smoke test")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save CSV outputs")
    args = parser.parse_args()

    smoke_test = args.smoke_test
    output_dir = args.output_dir
    seq_lengths = [512] if smoke_test else SEQ_LENGTHS
    repeats = 1 if smoke_test else REPEATS

    os.makedirs(output_dir, exist_ok=True)
    log_system_metadata(os.path.join(output_dir, "system_metadata.csv"))

    print("\n" + "=" * 70)
    print("  Phase 2: Token-by-Token Decoding Benchmark (MAX_NEW_TOKENS=128)")
    if smoke_test:
        print("  *** RUNNING IN SMOKE-TEST MODE (512 tokens, 1 repeat, Pythia only) ***")
    print(f"  Target Output Directory: {output_dir}")
    print("=" * 70)

    all_rows = []

    pythia_rows = run_model("Pythia-410M", "EleutherAI/pythia-410m", is_mamba=False, seq_lengths=seq_lengths, repeats=repeats)
    all_rows.extend(pythia_rows)

    if not smoke_test:
        mamba_rows = run_model("Mamba-370M", "state-spaces/mamba-370m", is_mamba=True, seq_lengths=seq_lengths, repeats=repeats)
        all_rows.extend(mamba_rows)

    phase2_csv = os.path.join(output_dir, "phase2_decode.csv")
    with open(phase2_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_rows)

    compat_csv = os.path.join(output_dir, "decode_results.csv")
    with open(compat_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_rows)

    print(f"\nPhase 2 complete. Results saved to '{phase2_csv}' and '{compat_csv}'.")


if __name__ == "__main__":
    main()
