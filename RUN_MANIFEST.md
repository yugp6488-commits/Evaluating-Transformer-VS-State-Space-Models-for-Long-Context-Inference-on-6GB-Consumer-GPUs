# RUN_MANIFEST.md — Execution Manifest & Cryptographic Checksums

**Project:** Evaluating Transformer and State Space Models for Long-Context Inference on 6GB Consumer GPUs  
**Target Publication:** IEEE Transactions on Machine Learning Systems / IEEE Access  
**Execution Timestamp:** 2026-11-15T16:40:00  
**Host Environment:** WSL2 Ubuntu 22.04 LTS on Windows 11 Workstation  
**Single Source of Truth:** `results/final_experiment/`  
**Derived Paper Tables:** `results/paper_tables/`  

---

## 1. Hardware & Software Telemetry

| Parameter | Reported Value | Source / Instrumentation |
|---|---|---|
| **GPU Model** | NVIDIA GeForce RTX 3050 6GB Laptop GPU | `torch.cuda.get_device_name(0)` |
| **Physical Dedicated VRAM** | 6,143.50 MB (~6,144 MB GDDR6) | `torch.cuda.get_device_properties(0).total_memory` |
| **PyTorch Allocator Ceiling** | 5,800.0 MB (fraction = 0.9440) | `torch.cuda.set_per_process_memory_fraction` |
| **Driver Overhead Buffer** | 343.50 MB | Dedicated margin for WDDM / CUDA runtime context |
| **CUDA Driver Version** | 12.1 | `torch.version.cuda` |
| **cuDNN Version** | 90100 (enabled: True) | `torch.backends.cudnn.version()` |
| **PyTorch Version** | 2.5.1+cu121 | `torch.__version__` |
| **Transformers Version** | 4.45.2 | `transformers.__version__` |
| **Mamba-SSM Version** | 2.2.2 | `mamba_ssm.__version__` |
| **Python Version** | 3.10.12 | `platform.python_version()` |
| **Linux Kernel** | 6.6.114.1-microsoft-standard-WSL2 | `platform.platform()` |
| **System RAM (WSL2-Visible)** | 11.54 GB (10.82 GB available) | `psutil.virtual_memory()` |
| **CPU Configuration** | 8 physical cores / 16 logical threads | `psutil.cpu_count()` |
| **Thread Constraints Enforced** | `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` | Environment variables |

> [!NOTE]
> **TODO for Authors:** Host RAM as configured by the author: confirm before submission.

---

## 2. Cryptographic Checksums (Authoritative Experimental Data & Derived Tables)

Computed via SHA256 across `results/final_experiment/` and `results/paper_tables/`:

### 2.1 Single Source of Truth (`results/final_experiment/`)
| File Path | SHA256 Hash |
|---|---|
| `results/final_experiment/environment_metadata.csv` | `4c80fd544854eaf9c3d66eab84d4103a63a53a2e04d43f480bcbc24a776f060f` |
| `results/final_experiment/phase1_prefill.csv` | `79340d5e012f5a19a1b274ab01a0af0e9d830584c0c72036fe0540b15bee9b33` |
| `results/final_experiment/phase2_decode.csv` | `6f2aafb1fa01a664c5c3e23ea929ce893856a42ee7228fb4aa68157318f79f08` |
| `results/final_experiment/phase3_batch.csv` | `97be74b27274b96c3e3b1287d90500a52a92dbd84d1ad05eeb9b80e8a6b94f3f` |
| `results/final_experiment/phase4_quality_per_document.csv` | `39471a1fcab5f92b0c19f4a53ca9dd1807067d11f62fa1703bffc1a2deffa58d` |
| `results/final_experiment/phase4_quality_summary.csv` | `e02e96a26b7261ca678c60e7fed52dbaa5c2686866d5285a48c9bf0b67540325` |
| `results/final_experiment/phase5_retrieval_trials.csv` | `1f905ec7daec0091596904cade6bfaaa45d35eaf7d0cd77b24124178264407dc` |
| `results/final_experiment/phase5_retrieval_summary.csv` | `36cda3a3d93805a7f6a78244145a744e3166f7521bba744ed3f2572e388f113d` |

### 2.2 Derived Paper Tables (`results/paper_tables/`)
| File Path | SHA256 Hash |
|---|---|
| `results/paper_tables/phase1_prefill_table.csv` | `dbfbc481238e5b562fbe855a98be06e4542c9a03ad3d07e810ce1abadc8a2992` |
| `results/paper_tables/phase2_decode_table.csv` | `55f5d77915bdcf6e6afa810ad6a6e5d78594c6f10c5fd668f726229d58aae56b` |
| `results/paper_tables/phase3_batch_table.csv` | `6b915827feb9954403c35b891ec637d562c33b13349da77a9c0dad72165b2ad5` |
| `results/paper_tables/phase4_per_document_table.csv` | `fe2350d1334680a55bc87d063b52598dab16e1a98927c63c7414ebd10d88a0d8` |
| `results/paper_tables/phase4_summary_table.csv` | `9eec050f7b5959797756448ba15b71a8bd37358d877e8b9979d973948e936d31` |
| `results/paper_tables/phase5_retrieval_table.csv` | `6e5342bedbc4f3cd34313567e44bc93c50ac286ac8cedd7eb11b1bd5e59354da` |
| `results/paper_tables/derived_claims.json` | `84d304f3fe2254cc594b8893504b3899d8f6edee68d1040a9eab25c6ddc89b1b` |
| `results/paper_tables/tables.md` | `958f6aec21bab9dad9463e24acaae4e78edd178339a257ed061e9cc2647c1823` |

---

## 3. Archival Record

All historical raw baseline files and smoke test directories have been moved into `results/_archive/` and are not used in the paper.
