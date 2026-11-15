# Consolidated Experimental Tables
**Single Source of Truth:** `results/final_experiment/*.csv`  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6,143.50 MB VRAM) under WSL2  
**Enforced Allocator Ceiling:** 5,800.0 MB  

## Table 1: Phase 1 Prefill Latency and Memory Scaling (Batch=1, FP32)
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

---

## Table 2: Phase 2 Autoregressive Decoding (128 Tokens Generated, Batch=1, FP32)
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

---

## Table 3: Phase 3 Multi-Batch Scaling and OOM Boundaries (FP32)
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

---

## Table 4a: Phase 4 PG-19 Test Split Language Modelling Summary
| Model | Num_Documents | Total_Scored_Tokens | Total_Scored_UTF8_Bytes | Chunks_Scored | Chunks_OOM_Skipped | Total_NLL_nats | Perplexity | BPB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pythia-410M | 10 | 64494 | 248843 | 101 | 0 | 177518.50 | 15.6815 | 1.0292 |
| Mamba-370M | 10 | 64494 | 248843 | 101 | 0 | 170398.83 | 14.0425 | 0.9879 |

---

## Table 4b: Phase 4 PG-19 Test Split Per-Document Metrics
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

---

## Table 5: Phase 5 Long-Context Synthetic Retrieval Exact-Match Accuracy with Wilson 95% CIs
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
