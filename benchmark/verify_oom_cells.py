#!/usr/bin/env python3
"""
benchmark/verify_oom_cells.py — Targeted OOM Cell Verification for Phase 3.

Executes exactly the 7 Phase 3 configurations that were labelled as OOM:
  Pythia-410M: (Batch 1, 8192), (2, 4096), (2, 8192), (4, 2048), (4, 4096), (4, 8192)
  Mamba-370M:  (Batch 4, 8192)

Constraints:
  - Frozen configuration: FP32 precision, strict 5800.0 MB PyTorch allocator cap.
  - Cascade skip DISABLED (every cell executes independently).
  - Pre-flight estimate (check_vram_headroom) DISABLED.
  - torch.cuda.OutOfMemoryError caught per cell.
  - Peak stats reset and cache emptied between cells.
  - Outputs results/final_experiment/phase3_oom_verification.csv.
"""

import csv
import gc
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    MambaLMHeadModel = None
    InferenceParams = None

from benchmark.system_monitor import enforce_vram_ceiling, STRICT_VRAM_LIMIT_MB

# ---------------------------------------------------------------------------
# Enforce PyTorch allocator ceiling BEFORE any CUDA allocations
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    enforce_vram_ceiling(STRICT_VRAM_LIMIT_MB)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "results" / "final_experiment" / "phase3_oom_verification.csv"


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def full_forward(model, input_ids, seq_len: int, batch: int, is_mamba: bool):
    if is_mamba:
        if InferenceParams is None:
            raise RuntimeError("mamba_ssm is required to run Mamba models.")
        ip = InferenceParams(max_seqlen=seq_len, max_batch_size=batch)
        return model(input_ids, inference_params=ip)
    return model(input_ids, use_cache=True)


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


def test_cell(model, model_name: str, batch: int, seq_len: int, is_mamba: bool):
    # Strict cache cleanup and peak stats reset prior to execution
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    input_ids = None
    outcome = "UNKNOWN"
    peak_mb = 0.0
    err_msg = ""

    print(f"  [RUNNING] {model_name:12} | Batch={batch} | SeqLen={seq_len:5} ...", end="", flush=True)

    try:
        input_ids = torch.randint(0, 1000, (batch, seq_len), dtype=torch.long, device=DEVICE)
        sync_cuda()

        with torch.no_grad():
            out = full_forward(model, input_ids, seq_len, batch, is_mamba)
            sync_cuda()
            del out

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        outcome = "SUCCESS"
        err_msg = "Completed forward pass without error"
        print(f" -> SUCCESS (Peak: {peak_mb:.2f} MB)")

    except torch.cuda.OutOfMemoryError as e:
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        outcome = "OOM_EXCEPTION_CAUGHT"
        err_msg = str(e).replace("\n", " ").strip()
        print(f" -> OOM_EXCEPTION_CAUGHT (Peak: {peak_mb:.2f} MB)")

    except Exception as e:
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        if "out of memory" in str(e).lower():
            outcome = "OOM_EXCEPTION_CAUGHT"
        else:
            outcome = "ERROR"
        err_msg = str(e).replace("\n", " ").strip()
        print(f" -> {outcome} (Peak: {peak_mb:.2f} MB): {err_msg[:60]}...")

    finally:
        if input_ids is not None:
            del input_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "Model": model_name,
        "Batch": batch,
        "Sequence_Length": seq_len,
        "Outcome": outcome,
        "Peak_Allocated_MB_Before_Failure": round(peak_mb, 2),
        "Error_Message": err_msg,
    }


def main():
    print("=" * 80)
    print("  PHASE 3 OOM CELL VERIFICATION SUITE")
    print(f"  Ceiling Enforced: {STRICT_VRAM_LIMIT_MB} MB")
    print(f"  Target Device:    {DEVICE}")
    print("=" * 80)

    results = []

    # 1. Pythia-410M Cells
    pythia_cells = [
        (1, 8192),
        (2, 4096),
        (2, 8192),
        (4, 2048),
        (4, 4096),
        (4, 8192),
    ]

    print("\n--- Testing Pythia-410M (6 Target Cells) ---")
    model_pythia = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m", torch_dtype=torch.float32
    ).to(DEVICE)
    model_pythia.eval()
    warm_up(model_pythia, is_mamba=False)
    print(f"  Pythia-410M loaded. Static weight memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

    for b, s in pythia_cells:
        res = test_cell(model_pythia, "Pythia-410M", b, s, is_mamba=False)
        results.append(res)

    del model_pythia
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Mamba-370M Cells
    mamba_cells = [
        (4, 8192),
    ]

    print("\n--- Testing Mamba-370M (1 Target Cell) ---")
    if MambaLMHeadModel is None:
        print("  [ERROR] mamba_ssm not installed.")
        results.append({
            "Model": "Mamba-370M",
            "Batch": 4,
            "Sequence_Length": 8192,
            "Outcome": "ERROR",
            "Peak_Allocated_MB_Before_Failure": 0.0,
            "Error_Message": "mamba_ssm package not installed",
        })
    else:
        model_mamba = MambaLMHeadModel.from_pretrained(
            "state-spaces/mamba-370m", device=DEVICE, dtype=torch.float32
        )
        model_mamba.eval()
        warm_up(model_mamba, is_mamba=True)
        print(f"  Mamba-370M loaded. Static weight memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

        for b, s in mamba_cells:
            res = test_cell(model_mamba, "Mamba-370M", b, s, is_mamba=True)
            results.append(res)

        del model_mamba
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write output CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Model", "Batch", "Sequence_Length", "Outcome", "Peak_Allocated_MB_Before_Failure", "Error_Message"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print(f"  Verification Complete. Saved to: {OUTPUT_FILE}")
    print("=" * 80)

    # Print Summary
    n_oom = sum(1 for r in results if r["Outcome"] == "OOM_EXCEPTION_CAUGHT")
    n_succ = sum(1 for r in results if r["Outcome"] == "SUCCESS")
    n_other = len(results) - n_oom - n_succ

    print(f"\nSummary:")
    print(f"  Total target cells tested: {len(results)}")
    print(f"  Real OOM exceptions caught: {n_oom} / {len(results)}")
    print(f"  Unexpected successes:       {n_succ} / {len(results)}")
    if n_other > 0:
        print(f"  Other errors:               {n_other} / {len(results)}")


if __name__ == "__main__":
    main()
