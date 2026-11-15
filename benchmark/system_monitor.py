"""
benchmark/system_monitor.py

Captures hardware metadata and framework versions for reproducible
CSV logging in the IEEE benchmark study:
"Evaluating Transformer and State Space Models for Long-Context Inference on 6GB Consumer GPUs"

Includes strict 6 GB dedicated VRAM ceiling enforcement and proactive
pre-flight VRAM headroom guards to prevent PCIe shared-memory spillover
under Windows WDDM / WSL2.
"""

import csv
import gc
import os
import platform
from datetime import datetime
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# PyTorch Allocator Ceiling Constants & Scope Clarification
# ---------------------------------------------------------------------------
# ARCHITECTURAL NOTICE (IEEE Reproducibility):
# torch.cuda.set_per_process_memory_fraction() limits ONLY PyTorch's internal
# caching allocator (c10::cuda::CUDACachingAllocator). It guarantees that
# allocations requested through PyTorch tensor operations cannot exceed the
# designated fraction of device memory.
#
# IMPORTANT LIMITATIONS — What the allocator cap does NOT control:
# 1. CUDA driver/context overhead (~200-400 MB) initialized by the runtime
#    upon first CUDA call. This is a direct cudaMalloc by the driver.
# 2. Direct cudaMalloc() calls made by external C/C++ libraries or
#    third-party CUDA extensions that bypass PyTorch's caching allocator
#    (e.g., Triton kernels in mamba-ssm may allocate outside PyTorch).
# 3. OS compositor / desktop display memory managed by Windows WDDM.
# 4. Shared GPU memory (system RAM addressable over PCIe) managed by the
#    WDDM driver, which is distinct from dedicated VRAM.
#
# CONSEQUENCE:
# Setting the allocator cap to 5800 MB on a 6144 MB GPU limits PyTorch
# tensor allocations to 5800 MB. However, total physical dedicated VRAM
# usage may still exceed 5800 MB due to driver context and non-PyTorch
# allocations. The 344 MB margin is a best-effort buffer for these
# overheads, NOT a hard guarantee that total GPU memory stays below 6144 MB.
#
# MEMORY TERMINOLOGY (used throughout CSV columns):
# - Peak_Allocated_MB: PyTorch caching allocator peak (torch.cuda.max_memory_allocated)
# - Peak_Reserved_MB: PyTorch allocator pool peak (torch.cuda.max_memory_reserved)
#   Reserved >= Allocated because the allocator holds free blocks in its pool.
# - Physical Dedicated VRAM: The 6144 MB of GDDR6 on the GPU die.
# - Shared GPU Memory: System RAM addressable by the GPU over PCIe (up to
#   half of system RAM under WDDM). Used transparently by the driver when
#   dedicated VRAM is exhausted. NOT the same as dedicated VRAM.
# ---------------------------------------------------------------------------
STRICT_VRAM_LIMIT_MB = 5800.0  # PyTorch allocator cap
PHYSICAL_VRAM_MB = 6144.0      # RTX 3050 Laptop physical dedicated VRAM
VALIDATED_DIR = "results/validated_results"


def enforce_vram_ceiling(limit_mb: float = STRICT_VRAM_LIMIT_MB) -> float:
    """Set a per-process PyTorch allocator ceiling.

    Must be called BEFORE any CUDA tensors are allocated (immediately after
    confirming torch.cuda.is_available()).

    Scope & Limitations:
        This restricts PyTorch's c10::cuda::CUDACachingAllocator. When PyTorch
        tensor allocations reach limit_mb, the allocator raises a clean
        torch.cuda.OutOfMemoryError instead of requesting more GPU memory.

        This does NOT guarantee that total physical dedicated VRAM usage stays
        below any particular threshold. CUDA driver context, non-PyTorch
        cudaMalloc calls, and OS display overhead are not controlled by this
        setting. The 344 MB margin between 5800 MB and 6144 MB is a best-effort
        buffer for these overheads.

    Args:
        limit_mb: Maximum PyTorch allocator budget in MB (default 5800.0).

    Returns:
        The exact fraction applied to torch.cuda.set_per_process_memory_fraction.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0

    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    fraction = min(1.0, limit_mb / total_vram_mb)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)

    print(f"  [VRAM CAP] Enforced {limit_mb:.0f} MB PyTorch allocator ceiling "
          f"(fraction={fraction:.4f} of {total_vram_mb:.0f} MB reported). "
          f"Note: Caps PyTorch caching allocator; driver context is excluded.")
    return fraction


def check_vram_headroom(
    additional_estimated_mb: float,
    limit_mb: float = STRICT_VRAM_LIMIT_MB,
) -> Tuple[bool, float, float]:
    """Proactive pre-flight VRAM guard — checks if additional memory fits headroom.

    CRITICAL FIX (No Double-Counting):
    Model weights are ALREADY loaded and allocated in GPU memory before any
    benchmark sequence length or batch configuration is evaluated. Therefore,
    torch.cuda.memory_allocated() already includes static model weights.

    This function checks whether ONLY the ADDITIONAL memory (dynamic KV-cache/SSM
    recurrent state + transient activation tensors + input buffers) can fit
    within the remaining headroom up to limit_mb:

        current_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        available_mb = limit_mb - current_allocated_mb
        fits = additional_estimated_mb <= available_mb

    DO NOT add model weight memory to additional_estimated_mb. Passing
    (weights + state + activations) would double-count weights against available_mb.

    Args:
        additional_estimated_mb: Additional dynamic memory needed for the run
            (state memory + activation overhead).
        limit_mb: Allocator ceiling in MB (default 5800.0).

    Returns:
        (fits, current_allocated_mb, available_mb) where fits is True if
        additional_estimated_mb <= available_mb.
    """
    import torch

    if not torch.cuda.is_available():
        return True, 0.0, limit_mb

    current_allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    available_mb = max(limit_mb - current_allocated_mb, 0.0)
    fits = additional_estimated_mb <= available_mb

    return fits, round(current_allocated_mb, 2), round(available_mb, 2)


def estimate_transformer_kv_cache_mb(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 4,
) -> float:
    """Estimate Transformer KV-cache VRAM in MB.

    KV-cache = 2 × layers × batch × heads × seq_len × head_dim × dtype_bytes
    (factor of 2 for both K and V tensors).
    """
    kv_bytes = 2 * num_layers * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    return kv_bytes / (1024 ** 2)


def estimate_mamba_state_mb(
    num_layers: int,
    d_model: int,
    ssm_state_size: int,
    batch_size: int,
    dtype_bytes: int = 4,
) -> float:
    """Estimate Mamba SSM recurrent state VRAM in MB.

    State = layers × batch × d_inner × ssm_state_size × dtype_bytes
    (d_inner ≈ 2 × d_model for Mamba's expand factor of 2).
    """
    d_inner = 2 * d_model
    state_bytes = num_layers * batch_size * d_inner * ssm_state_size * dtype_bytes
    return state_bytes / (1024 ** 2)


def estimate_activation_overhead_mb(
    seq_len: int, batch_size: int, d_model: int, dtype_bytes: int = 4
) -> float:
    """Rough estimate of transient activation memory during a forward pass."""
    # Activations ≈ batch × seq_len × d_model × ~8 intermediate tensors × dtype
    act_bytes = batch_size * seq_len * d_model * 8 * dtype_bytes
    return act_bytes / (1024 ** 2)


def get_system_metadata() -> Dict[str, str]:
    """Collect hardware and software metadata for reproducibility.

    Captures GPU name, VRAM, system RAM, CPU utilization, OS details,
    and framework versions (PyTorch, CUDA, cuDNN, Transformers, Mamba-SSM).
    """
    import torch

    meta = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_count"] = str(torch.cuda.device_count())
        props = torch.cuda.get_device_properties(0)
        total_vram_bytes = props.total_memory
        meta["gpu_total_vram_mb"] = f"{total_vram_bytes / (1024**2):.2f}"
        meta["cuda_version"] = torch.version.cuda or "N/A"
        meta["cudnn_version"] = (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else "N/A"
        )
        meta["cudnn_enabled"] = str(torch.backends.cudnn.enabled)
    else:
        meta["gpu_name"] = "N/A (CPU execution)"
        meta["gpu_count"] = "0"
        meta["gpu_total_vram_mb"] = "0.0"
        meta["cuda_version"] = "N/A"
        meta["cudnn_version"] = "N/A"
        meta["cudnn_enabled"] = "N/A"

    try:
        import transformers

        meta["transformers_version"] = transformers.__version__
    except ImportError:
        meta["transformers_version"] = "N/A"

    try:
        import mamba_ssm

        meta["mamba_ssm_version"] = getattr(mamba_ssm, "__version__", "installed")
    except (ImportError, AttributeError):
        meta["mamba_ssm_version"] = "N/A"

    try:
        import psutil

        mem = psutil.virtual_memory()
        meta["system_ram_total_gb"] = f"{mem.total / (1024**3):.2f}"
        meta["system_ram_available_gb"] = f"{mem.available / (1024**3):.2f}"
        meta["cpu_percent"] = f"{psutil.cpu_percent(interval=0.2):.1f}"
        meta["cpu_count_physical"] = str(psutil.cpu_count(logical=False))
        meta["cpu_count_logical"] = str(psutil.cpu_count(logical=True))
    except ImportError:
        meta["system_ram_total_gb"] = "N/A (psutil not installed)"
        meta["system_ram_available_gb"] = "N/A"
        meta["cpu_percent"] = "N/A"
        meta["cpu_count_physical"] = "N/A"
        meta["cpu_count_logical"] = "N/A"

    return meta


def get_gpu_memory_snapshot() -> Dict[str, float]:
    """Capture current GPU memory state in MB.

    Memory terminology:
        - allocated_mb: Active tensor memory managed by PyTorch's allocator.
        - reserved_mb: Total memory held by PyTorch's allocator pool (includes
          free blocks not yet returned to CUDA). Always >= allocated.
        - max_allocated_mb / max_reserved_mb: Peak values since last reset.
        - strict_vram_cap_mb: PyTorch allocator ceiling. Does NOT represent
          a cap on total physical GPU memory usage.
        - within_allocator_cap: Whether peak PyTorch allocations stayed within
          the configured allocator ceiling.
    """
    import torch

    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
            "free_mb": 0.0,
            "strict_vram_cap_mb": STRICT_VRAM_LIMIT_MB,
            "within_6gb_cap": True,
        }

    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_alloc = torch.cuda.max_memory_allocated()
    max_res = torch.cuda.max_memory_reserved()

    peak_alloc_mb = round(max_alloc / (1024**2), 2)
    peak_reserved_mb = round(max_res / (1024**2), 2)

    return {
        "allocated_mb": round(allocated / (1024**2), 2),
        "reserved_mb": round(reserved / (1024**2), 2),
        "max_allocated_mb": peak_alloc_mb,
        "max_reserved_mb": peak_reserved_mb,
        "free_mb": round((total_mem - allocated) / (1024**2), 2),
        "strict_vram_cap_mb": STRICT_VRAM_LIMIT_MB,
        "within_6gb_cap": peak_alloc_mb <= STRICT_VRAM_LIMIT_MB,
    }


def log_system_metadata(filepath: str = "results/validated_results/system_metadata.csv") -> str:
    """Write system metadata to CSV for experiment reproducibility."""
    meta = get_system_metadata()
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(meta)

    return filepath


def print_system_info() -> None:
    """Pretty-print system metadata to stdout."""
    meta = get_system_metadata()
    print("=" * 65)
    print("  SYSTEM METADATA (Reproducibility Record)")
    print("=" * 65)
    for key, value in meta.items():
        print(f"  {key:30s} : {value}")
    print("=" * 65)


if __name__ == "__main__":
    print_system_info()
    path = log_system_metadata()
    print(f"\nMetadata saved to: {path}")
