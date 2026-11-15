# Experimental Results and Verification Appendix

**Project:** Evaluating Transformer and State Space Models for Long-Context Inference on 6GB Consumer GPUs  
**Target Publication:** IEEE Transactions on Machine Learning Systems / IEEE Access  
**Single Source of Truth:** `results/final_experiment/*.csv`  
**Derived Tables & Claims:** `results/paper_tables/*.csv`, `results/paper_tables/derived_claims.json`, `results/paper_tables/tables.md`  

---

## 1. Hardware and Runtime Environment

All benchmarks were executed on consumer workstation hardware under virtualized Linux. All telemetry values below are extracted directly from `results/final_experiment/environment_metadata.csv`:

* **Dedicated GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (GA107 Ampere, 2048 CUDA cores)
* **Dedicated Physical VRAM:** 6,143.50 MB GDDR6 (`torch.cuda.get_device_properties(0).total_memory`)
* **PyTorch Allocator Ceiling:** 5,800.0 MB (`torch.cuda.set_per_process_memory_fraction(0.9440, device=0)`)
* **Host Runtime Layer:** WSL2 on Windows 11 (Linux kernel 6.6.114.1-microsoft-standard-WSL2)
* **WSL2-Visible System RAM:** 11.54 GB total, 10.82 GB available at benchmark initialization
* **CUDA Driver / Runtime:** CUDA 12.1, cuDNN 90100 (enabled)
* **Core Framework Versions:** PyTorch 2.5.1+cu121, Hugging Face Transformers 4.45.2, Mamba-SSM 2.2.2, Python 3.10.12
* **CPU Execution Bounds:** 8 physical cores, 16 logical threads, constrained via `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`

> [!NOTE]
> **TODO for Authors:** Host RAM as configured by the author: confirm before submission.

### Note on Allocator Ceiling vs. Intrinsic Hardware Limits
The 8,192-token Pythia out-of-memory error is caused by the 5,800.0 MB PyTorch allocator cap (`torch.cuda.set_per_process_memory_fraction`), which leaves 343.50 MB headroom for Windows/WDDM display context and driver operations. It is not an intrinsic 6,144 MB hardware boundary, but an enforced allocator boundary to prevent uninstrumented PCIe paging into shared system RAM.

---

## 2. Phase 1: Prefill Latency and Memory Scaling (Batch=1, FP32)

Measurements reflect single-sequence prompt ingestion from 512 to 8,192 tokens in FP32 precision.

| Model | Batch | Sequence_Length | Status | Latency_sec | Latency_std_sec | Prefill_Throughput_tps | Peak_Allocated_MB | Peak_Reserved_MB | State_VRAM_MB | State_Logical_MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pythia-410M | 1 | 512 | Success | 0.0929 | 0.0010 | 5514.17 | 1942.63 | 1980.00 | 192.00 | 96.00 |
| Pythia-410M | 1 | 1024 | Success | 0.1995 | 0.0028 | 5133.65 | 2234.88 | 2302.00 | 384.00 | 192.00 |
| Pythia-410M | 1 | 2048 | Success | 0.4655 | 0.0065 | 4399.54 | 2820.39 | 2970.00 | 768.00 | 384.00 |
| Pythia-410M | 1 | 4096 | Success | 1.3924 | 0.0227 | 2941.75 | 3988.41 | 4258.00 | 1536.00 | 768.00 |
| Pythia-410M | 1 | 8192 | OOM at 5,800 MB cap | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| Mamba-370M | 1 | 512 | Success | 0.1061 | 0.0008 | 4824.63 | 1533.05 | 1664.00 | 7.50 | 7.50 |
| Mamba-370M | 1 | 1024 | Success | 0.2075 | 0.0020 | 4934.86 | 1633.26 | 1798.00 | 7.50 | 7.50 |
| Mamba-370M | 1 | 2048 | Success | 0.3956 | 0.0010 | 5177.59 | 1833.68 | 2072.00 | 7.50 | 7.50 |
| Mamba-370M | 1 | 4096 | Success | 0.7826 | 0.0016 | 5233.77 | 2234.88 | 2620.00 | 7.50 | 7.50 |
| Mamba-370M | 1 | 8192 | Success | 1.5533 | 0.0099 | 5273.80 | 3036.91 | 3696.00 | 7.50 | 7.50 |

### Findings & Memory Analysis
* **Prefill Throughput:** Pythia prefill throughput declines from 5514.17 tps at 512 tokens to 2941.75 tps at 4,096 tokens, before encountering an OOM at 8,192 tokens. Mamba prefill throughput scales from 4824.63 tps at 512 tokens to 5273.80 tps at 8,192 tokens. A possible explanation, not tested here, is that sequential memory access patterns in the parallel associative scan retain favorable execution characteristics at longer sequence lengths.
* **Persistent State vs. Peak Allocation:** Only Mamba's persistent state is constant at 7.50 MB across all context lengths (comprising 6.00 MB SSM hidden state and 1.50 MB 1D convolution state). However, Mamba's peak allocated memory still grows with context length, increasing from 1,533.05 MB at 512 tokens to 3,036.91 MB at 8,192 tokens due to transient scan activations.
* **Pythia State VRAM vs. State Logical:** In Pythia, measured State_VRAM is exactly 2x State_Logical (e.g., 192.00 MB vs. 96.00 MB at 512 tokens, 384.00 MB vs. 192.00 MB at 1,024 tokens, up to 1536.00 MB vs. 768.00 MB at 4,096 tokens). This 2x ratio represents the cache-concatenation allocation overhead where key and value states are duplicated or buffered during autoregressive layer concatenation.

---

## 3. Phase 2: Token-by-Token Autoregressive Decoding (128 Tokens, Batch=1, FP32)

Decoding latency and throughput were measured for generating 128 new tokens following prompt prefill.

| Model | Prefill_Length | Status | Prefill_Latency_sec | Decode_Latency_per_token_ms | Decode_Latency_std_ms | Decode_Throughput_tps | Peak_Allocated_MB | Peak_Reserved_MB | Total_Generated_Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pythia-410M | 512 | Success | 0.0963 | 17.3332 | 2.0970 | 57.69 | 2037.85 | 2080.00 | 128 |
| Pythia-410M | 1024 | Success | 0.2018 | 20.9163 | 1.2878 | 47.81 | 2424.11 | 2522.00 | 128 |
| Pythia-410M | 2048 | Success | 0.4879 | 27.7644 | 0.6684 | 36.02 | 3197.62 | 3330.00 | 128 |
| Pythia-410M | 4096 | Success | 1.4524 | 42.7899 | 1.0303 | 23.37 | 4741.63 | 4960.00 | 128 |
| Pythia-410M | 8192 | OOM at 5,800 MB cap | OOM | OOM | OOM | OOM | OOM | OOM | 0 |
| Mamba-370M | 512 | Success | 0.1065 | 23.3987 | 7.0673 | 42.74 | 1533.05 | 1664.00 | 128 |
| Mamba-370M | 1024 | Success | 0.2096 | 23.5770 | 2.6444 | 42.41 | 1633.26 | 1798.00 | 128 |
| Mamba-370M | 2048 | Success | 0.3957 | 23.1607 | 2.0367 | 43.18 | 1833.68 | 2072.00 | 128 |
| Mamba-370M | 4096 | Success | 0.7730 | 22.8304 | 1.7850 | 43.80 | 2234.88 | 2620.00 | 128 |
| Mamba-370M | 8192 | Success | 1.5827 | 23.3307 | 2.0951 | 42.86 | 3036.91 | 3696.00 | 128 |

### Findings & Latency Invariance
* **Pythia Decoding Growth:** Pythia per-token decode latency increases from 17.3332 ms at 512 tokens to 42.7899 ms at 4,096 tokens, representing a 2.47x latency growth. A possible explanation, not tested here, is that the linear expansion of KV-cache memory lookups across attention heads increases per-step memory bus traffic.
* **Mamba Flat Decoding Profile:** Mamba per-token decode latency remains constant across prompt lengths, with a mean of 23.2595 ms and standard deviation of 0.2825 ms across sequence lengths from 512 to 8,192 tokens. A flat decode latency of approximately 23 ms/token is observed; a possible explanation, not tested here, is per-step kernel launch overhead on mobile CUDA runtimes.
* **Cross-Over Point:** The first context length where Mamba decode throughput exceeds Pythia is 2,048 tokens (Mamba: 43.18 tps vs. Pythia: 36.02 tps).

---

## 4. Phase 3: Multi-Batch Scaling and OOM Boundaries (FP32)

Batch sizes 1, 2, and 4 were evaluated across sequence lengths 512 to 8,192 tokens. Status is derived strictly from whether the configuration succeeded or triggered the allocator limit.

| Model | Batch | Sequence_Length | Status | Latency_sec | Latency_std_sec | Throughput_tps | Peak_Allocated_VRAM_MB | Peak_Reserved_VRAM_MB | Persistent_State_MB | State_Logical_MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pythia-410M | 1 | 512 | Success | 0.0936 | 0.0007 | 5469.59 | 1942.63 | 1980.00 | 192.00 | 96.00 |
| Pythia-410M | 1 | 1024 | Success | 0.1992 | 0.0020 | 5139.62 | 2234.88 | 2302.00 | 384.00 | 192.00 |
| Pythia-410M | 1 | 2048 | Success | 0.4732 | 0.0072 | 4327.59 | 2820.39 | 2970.00 | 768.00 | 384.00 |
| Pythia-410M | 1 | 4096 | Success | 1.4775 | 0.0252 | 2772.27 | 3988.41 | 4258.00 | 1536.00 | 768.00 |
| Pythia-410M | 1 | 8192 | OOM at 5,800 MB cap | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| Pythia-410M | 2 | 512 | Success | 0.1835 | 0.0013 | 5580.82 | 2234.88 | 2300.00 | 384.00 | 192.00 |
| Pythia-410M | 2 | 1024 | Success | 0.3933 | 0.0088 | 5206.73 | 2820.39 | 2970.00 | 768.00 | 384.00 |
| Pythia-410M | 2 | 2048 | Success | 1.1278 | 0.0109 | 3631.78 | 3988.41 | 4258.00 | 1536.00 | 768.00 |
| Pythia-410M | 2 | 4096 | OOM at 5,800 MB cap | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| Pythia-410M | 2 | 8192 | OOM at 5,800 MB cap | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| Pythia-410M | 4 | 512 | Success | 0.3651 | 0.0084 | 5609.91 | 2820.39 | 2970.00 | 768.00 | 384.00 |
| Pythia-410M | 4 | 1024 | Success | 1.0777 | 0.0137 | 3800.62 | 3988.41 | 4258.00 | 1536.00 | 768.00 |
| Pythia-410M | 4 | 2048 | OOM at 5,800 MB cap | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| Pythia-410M | 4 | 4096 | OOM at 5,800 MB cap | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| Pythia-410M | 4 | 8192 | OOM at 5,800 MB cap | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| Mamba-370M | 1 | 512 | Success | 0.1069 | 0.0005 | 4787.47 | 1533.05 | 1664.00 | 7.50 | 7.50 |
| Mamba-370M | 1 | 1024 | Success | 0.2101 | 0.0027 | 4875.00 | 1633.26 | 1798.00 | 7.50 | 7.50 |
| Mamba-370M | 1 | 2048 | Success | 0.3975 | 0.0070 | 5152.76 | 1833.68 | 2072.00 | 7.50 | 7.50 |
| Mamba-370M | 1 | 4096 | Success | 0.7765 | 0.0133 | 5275.06 | 2234.88 | 2620.00 | 7.50 | 7.50 |
| Mamba-370M | 1 | 8192 | Success | 1.5473 | 0.0441 | 5294.31 | 3036.91 | 3696.00 | 7.50 | 7.50 |
| Mamba-370M | 2 | 512 | Success | 0.2203 | 0.0020 | 4647.91 | 1640.76 | 1806.00 | 15.00 | 15.00 |
| Mamba-370M | 2 | 1024 | Success | 0.4289 | 0.0072 | 4774.74 | 1841.18 | 2080.00 | 15.00 | 15.00 |
| Mamba-370M | 2 | 2048 | Success | 0.8695 | 0.0055 | 4711.02 | 2242.38 | 2628.00 | 15.00 | 15.00 |
| Mamba-370M | 2 | 4096 | Success | 1.7506 | 0.0448 | 4679.55 | 3044.41 | 3704.00 | 15.00 | 15.00 |
| Mamba-370M | 2 | 8192 | Success | 3.5012 | 0.0978 | 4679.56 | 4646.97 | 4762.00 | 15.00 | 15.00 |
| Mamba-370M | 4 | 512 | Success | 0.4265 | 0.0006 | 4801.39 | 1856.18 | 2096.00 | 30.00 | 30.00 |
| Mamba-370M | 4 | 1024 | Success | 0.8547 | 0.0024 | 4792.53 | 2257.38 | 2644.00 | 30.00 | 30.00 |
| Mamba-370M | 4 | 2048 | Success | 1.7387 | 0.0113 | 4711.52 | 3059.41 | 3718.00 | 30.00 | 30.00 |
| Mamba-370M | 4 | 4096 | Success | 3.5307 | 0.1410 | 4640.45 | 4661.97 | 4776.00 | 30.00 | 30.00 |
| Mamba-370M | 4 | 8192 | OOM at 5,800 MB cap | OOM | OOM | OOM | OOM | OOM | OOM | OOM |

### Failure Envelope & Serving-Envelope Ratios
* **OOM Failure Counts:** Out of 15 configurations per model, Pythia encountered 6 OOM failures, whereas Mamba encountered 1 OOM failure (Batch 4, Length 8,192).
* **Serving Envelope Max Context Ratios:**
  - **Batch 1:** Pythia maximum sequence length = 4,096 tokens; Mamba maximum = 8,192 tokens. Serving envelope ratio = 2.0x.
  - **Batch 2:** Pythia maximum sequence length = 2,048 tokens; Mamba maximum = 8,192 tokens. Serving envelope ratio = 4.0x.
  - **Batch 4:** Pythia maximum sequence length = 1,024 tokens; Mamba maximum = 4,096 tokens. Serving envelope ratio = 4.0x.

---

## 5. Phase 4: Language Modelling Quality on PG-19 Test Split

Evaluated on 10 book extracts from the authentic PG-19 test split (`emozilla/pg19`) truncated to 25,000 characters per document, using sliding-window chunking (`max_length=2048`, `stride=512`).

### Summary Metrics
| Model | Num_Documents | Total_Scored_Tokens | Total_Scored_UTF8_Bytes | Chunks_Scored | Chunks_OOM_Skipped | Total_NLL_nats | Perplexity | BPB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pythia-410M | 10 | 64494 | 248843 | 101 | 0 | 177518.50 | 15.6815 | 1.0292 |
| Mamba-370M | 10 | 64494 | 248843 | 101 | 0 | 170398.83 | 14.0425 | 0.9879 |

### Per-Document Metrics
| Document_Index | Model | Dataset | Total_Text_Bytes | Scored_Tokens | Scored_UTF8_Bytes | Chunks_Scored | Chunks_OOM_Skipped | Total_NLL_nats | Perplexity | BPB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Pythia-410M | PG-19 (test) | 25000 | 5879 | 24998 | 9 | 0 | 15148.31 | 13.1534 | 0.8742 |
| 2 | Pythia-410M | PG-19 (test) | 25000 | 6709 | 24997 | 11 | 0 | 22187.90 | 27.3081 | 1.2806 |
| 3 | Pythia-410M | PG-19 (test) | 25000 | 6429 | 24997 | 10 | 0 | 18994.07 | 19.1909 | 1.0962 |
| 4 | Pythia-410M | PG-19 (test) | 25000 | 5954 | 24997 | 9 | 0 | 19537.01 | 26.6110 | 1.1276 |
| 5 | Pythia-410M | PG-19 (test) | 23870 | 7140 | 23867 | 11 | 0 | 17276.18 | 11.2417 | 1.0443 |
| 6 | Pythia-410M | PG-19 (test) | 25000 | 6987 | 24999 | 11 | 0 | 17039.05 | 11.4579 | 0.9833 |
| 7 | Pythia-410M | PG-19 (test) | 25000 | 6750 | 24997 | 11 | 0 | 17167.69 | 12.7224 | 0.9908 |
| 8 | Pythia-410M | PG-19 (test) | 25000 | 5720 | 24997 | 9 | 0 | 16483.72 | 17.8458 | 0.9514 |
| 9 | Pythia-410M | PG-19 (test) | 25000 | 6536 | 24997 | 10 | 0 | 19086.80 | 18.5461 | 1.1016 |
| 10 | Pythia-410M | PG-19 (test) | 25000 | 6390 | 24997 | 10 | 0 | 14597.77 | 9.8205 | 0.8425 |
| 1 | Mamba-370M | PG-19 (test) | 25000 | 5879 | 24998 | 9 | 0 | 14497.98 | 11.7760 | 0.8367 |
| 2 | Mamba-370M | PG-19 (test) | 25000 | 6709 | 24997 | 11 | 0 | 21410.27 | 24.3194 | 1.2357 |
| 3 | Mamba-370M | PG-19 (test) | 25000 | 6429 | 24997 | 10 | 0 | 18441.69 | 17.6109 | 1.0644 |
| 4 | Mamba-370M | PG-19 (test) | 25000 | 5954 | 24997 | 9 | 0 | 18868.58 | 23.7851 | 1.0890 |
| 5 | Mamba-370M | PG-19 (test) | 23870 | 7140 | 23867 | 11 | 0 | 16329.60 | 9.8459 | 0.9871 |
| 6 | Mamba-370M | PG-19 (test) | 25000 | 6987 | 24999 | 11 | 0 | 16433.93 | 10.5073 | 0.9484 |
| 7 | Mamba-370M | PG-19 (test) | 25000 | 6750 | 24997 | 11 | 0 | 16579.32 | 11.6604 | 0.9569 |
| 8 | Mamba-370M | PG-19 (test) | 25000 | 5720 | 24997 | 9 | 0 | 15569.79 | 15.2106 | 0.8986 |
| 9 | Mamba-370M | PG-19 (test) | 25000 | 6536 | 24997 | 10 | 0 | 18362.21 | 16.5999 | 1.0598 |
| 10 | Mamba-370M | PG-19 (test) | 25000 | 6390 | 24997 | 10 | 0 | 13905.46 | 8.8121 | 0.8025 |

### Direct Comparability of PPL and BPB Role
Both Pythia-410M and Mamba-370M share the identical GPT-NeoX-20B BPE tokenization scheme, resulting in an identical total of 64,494 scored tokens and 248,843 scored UTF-8 bytes across all 10 evaluated documents. Because the token sequences and denominators are identical, perplexity (PPL) is directly comparable between the two models in this experiment. Bits Per Byte (BPB) is reported as a supplementary byte-normalised metric rather than an essential adjustment for mismatched vocabularies.

---

## 6. Phase 5: Long-Context Synthetic Retrieval (450 Trials)

Evaluated across 3 synthetic tasks (Passkey Retrieval, Needle-in-a-Haystack, Associative Recall), 5 sequence lengths (512 to 8,192 tokens), 5 insertion positions (0%, 25%, 50%, 75%, 100%), and 3 trials per position (15 trials per cell). Both models were pre-trained with a 2,048-token context window; sequence lengths above 2,048 are designated as extrapolation for both models.

| Task | Target_Seq_Len | Regime | Pythia_Exact_Match | Pythia_95_CI | Pythia_Partial_Credit | Mamba_Exact_Match | Mamba_95_CI | Mamba_Partial_Credit | Comparison |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| passkey | 512 | In-Distribution | 0.8667 | [0.6212, 0.9626] | 0.8667 (partial credit) | 0.4000 | [0.1982, 0.6425] | 0.4000 (partial credit) | CIs overlap |
| passkey | 1024 | In-Distribution | 1.0000 | [0.7961, 1.0000] | 1.0000 (partial credit) | 0.4667 | [0.2481, 0.6988] | 0.4667 (partial credit) | CIs disjoint |
| passkey | 2048 | In-Distribution | 1.0000 | [0.7961, 1.0000] | 1.0000 (partial credit) | 0.3333 | [0.1518, 0.5829] | 0.3333 (partial credit) | CIs disjoint |
| passkey | 4096 | Extrapolation | 0.0000 | [0.0000, 0.2039] | 0.0000 (partial credit) | 0.5333 | [0.3012, 0.7519] | 0.5333 (partial credit) | CIs disjoint |
| passkey | 8192 | Extrapolation | OOM (n=15 failed) | N/A (OOM) | N/A (OOM) | 0.0000 | [0.0000, 0.2039] | 0.0000 (partial credit) | CIs disjoint |
| needle_in_haystack | 512 | In-Distribution | 1.0000 | [0.7961, 1.0000] | 1.0000 (partial credit) | 1.0000 | [0.7961, 1.0000] | 1.0000 (partial credit) | CIs overlap |
| needle_in_haystack | 1024 | In-Distribution | 1.0000 | [0.7961, 1.0000] | 1.0000 (partial credit) | 1.0000 | [0.7961, 1.0000] | 1.0000 (partial credit) | CIs overlap |
| needle_in_haystack | 2048 | In-Distribution | 1.0000 | [0.7961, 1.0000] | 1.0000 (partial credit) | 0.8000 | [0.5481, 0.9295] | 0.9600 (partial credit) | CIs overlap |
| needle_in_haystack | 4096 | Extrapolation | 0.0000 | [0.0000, 0.2039] | 0.0000 (partial credit) | 0.8000 | [0.5481, 0.9295] | 0.8400 (partial credit) | CIs disjoint |
| needle_in_haystack | 8192 | Extrapolation | OOM (n=15 failed) | N/A (OOM) | N/A (OOM) | 0.0667 | [0.0119, 0.2982] | 0.3067 (partial credit) | CIs disjoint |
| associative_recall | 512 | In-Distribution | 0.0667 | [0.0119, 0.2982] | 0.0667 (partial credit) | 0.2667 | [0.1090, 0.5195] | 0.2667 (partial credit) | CIs overlap |
| associative_recall | 1024 | In-Distribution | 0.0000 | [0.0000, 0.2039] | 0.0000 (partial credit) | 0.4000 | [0.1982, 0.6425] | 0.4000 (partial credit) | CIs overlap |
| associative_recall | 2048 | In-Distribution | 0.0000 | [0.0000, 0.2039] | 0.0000 (partial credit) | 0.2000 | [0.0705, 0.4519] | 0.2000 (partial credit) | CIs overlap |
| associative_recall | 4096 | Extrapolation | 0.0000 | [0.0000, 0.2039] | 0.0000 (partial credit) | 0.0000 | [0.0000, 0.2039] | 0.0000 (partial credit) | CIs overlap |
| associative_recall | 8192 | Extrapolation | OOM (n=15 failed) | N/A (OOM) | N/A (OOM) | 0.0000 | [0.0000, 0.2039] | 0.0000 (partial credit) | CIs disjoint |
| ALL_TASKS_AGGREGATE | 0 | In-Distribution (<=2048) | 0.6593 (n=135, OOM=0) | [0.5759, 0.7338] | 0.6593 (partial credit) | 0.5407 (n=135, OOM=0) | [0.4567, 0.6225] | 0.5585 (partial credit) | CIs overlap |
| ALL_TASKS_AGGREGATE | 0 | Extrapolation (>2048) | 0.0000 (n=45, OOM=45) | [0.0000, 0.0787] | 0.0000 (partial credit) | 0.2333 (n=90, OOM=0) | [0.1580, 0.3305] | 0.2800 (partial credit) | CIs disjoint |

### Statistical Retrieval Findings & Significance Analysis
* **In-Distribution Performance (<=2048 tokens):**
  - Pythia demonstrates high accuracy on passkey (0.8667 at 512, 1.0000 at 1,024 and 2,048) and needle-in-a-haystack (1.0000 across 512, 1,024, and 2,048).
  - Mamba is weaker on passkey retrieval, achieving accuracies of 0.4000 at 512, 0.4667 at 1,024, and 0.3333 at 2,048 tokens.
  - On needle-in-a-haystack, Mamba achieves 1.0000 at 512 and 1,024, and 0.8000 at 2,048 tokens.
  - Across all tasks in-distribution, Pythia aggregate accuracy is 0.6593 (95% CI: [0.5759, 0.7338]) while Mamba is 0.5407 (95% CI: [0.4567, 0.6225]). Because these Wilson confidence intervals overlap, this aggregate difference is not statistically significant.
* **Extrapolation Performance (>2048 tokens):**
  - Pythia fails completely on all tasks at 4,096 tokens (0.0000 exact match across passkey, needle, and associative recall) and triggers an OOM at 8,192 tokens across all 15 trials per task.
  - Mamba maintains an accuracy of 0.5333 on passkey and 0.8000 on needle-in-a-haystack at 4,096 tokens, before collapsing at 8,192 tokens (0.0000 on passkey, 0.0667 on needle, 0.0000 on associative recall).
  - Across all tasks in extrapolation, Pythia aggregate accuracy is 0.0000 (95% CI: [0.0000, 0.0787], with 45 trials failing due to OOM) while Mamba aggregate accuracy is 0.2333 (95% CI: [0.1580, 0.3305]). Because these confidence intervals are disjoint, the difference in extrapolation accuracy is statistically significant.
* **Associative Recall Difficulty:** Associative recall is poor for both models across all context lengths (<=0.4000 for Mamba and <=0.0667 for Pythia), indicating that zero-shot key-value association without task-specific fine-tuning is challenging for both architectures.
* **Interpretation Rule:** Differences with overlapping Wilson confidence intervals are not considered statistically significant, and comparative advantages are claimed only when confidence intervals are disjoint.

---

## 7. Scope and Limitations

To enable rigorous evaluation by peer reviewers, the operational scope and boundary conditions of this study are explicitly documented:

1. **Text Sample Truncation:** Quality evaluation in Phase 4 is conducted on 10 book extracts from the PG-19 test split, each truncated to 25,000 characters. While this yields 64,494 scored tokens and 248,843 scored bytes per model, results reflect this sample scale.
2. **Pre-training Data Contamination:** PG-19 books originate from Project Gutenberg, which is contained within The Pile corpus on which both Pythia-410M and Mamba-370M were pre-trained. Train/test contamination cannot be excluded.
3. **Single Run for Language Modelling Quality:** Phase 4 quality metrics derive from a single deterministic evaluation pass across the 10 selected documents.
4. **Retrieval Sample Size:** Phase 5 retrieval evaluates n=15 trials per configuration cell (5 context depth positions x 3 repeated trials with randomized seeds).
5. **Synthetic Task Domain:** Retrieval tests evaluate synthetic token extraction tasks (5-digit passkeys, single-fact needle insertion, key-value record matching) and do not evaluate complex multi-hop reasoning or document summarization.
6. **Precision Format:** All benchmark runs were executed strictly in FP32 precision to prevent quantization artifacts from confounding architectural memory scaling comparisons.
7. **Kernel Implementations:** Mamba uses the hardware-accelerated fused CUDA/Triton kernels provided by `mamba_ssm` via `MambaLMHeadModel.from_pretrained()`.
8. **Decode Latency Characteristics:** The flat Mamba decode latency profile (~23 ms/token) is reported as an empirical observation. A possible explanation, not tested here, is per-step kernel launch overhead on mobile CUDA runtimes.
