# CHANGELOG.md — Final Paper Deliverables Consistency & Audit Pass

**Project:** Evaluating Transformer and State Space Models for Long-Context Inference on 6GB Consumer GPUs  
**Target Publication:** IEEE Transactions on Machine Learning Systems / IEEE Access  
**Single Source of Truth:** `results/final_experiment/*.csv`  
**Execution Timestamp:** 2026-11-15T16:43:00  

---

## 1. Ground Rules Enforced

* **Single Source of Truth:** All numerical values derive strictly and programmatically from `results/final_experiment/*.csv`. Zero numbers were typed from memory or inferred from legacy runs.
* **Preservation of Raw Measurements:** No data was fabricated, smoothed, or altered. OOM is reported as a failure, never as a valid zero.
* **Precision Standards:** Percentages formatted to two decimal places; PPL, BPB, and latencies formatted to four decimal places.
* **Plain Text Math:** Formatted as plain text (e.g., "2.47x"), with zero duplicated LaTeX/plain artifacts.

---

## 2. Tasks Executed & Files Modified

### Task 1: Built `analysis/generate_paper_tables.py`
* **File Created:** `analysis/generate_paper_tables.py`
* **Output Deliverables:**
  - `results/paper_tables/phase1_prefill_table.csv`
  - `results/paper_tables/phase2_decode_table.csv`
  - `results/paper_tables/phase3_batch_table.csv` (Status derived strictly as "Success" vs. "OOM at 5,800 MB cap")
  - `results/paper_tables/phase4_per_document_table.csv`
  - `results/paper_tables/phase4_summary_table.csv` (recomputed from per-document rows; no Peak VRAM column)
  - `results/paper_tables/phase5_retrieval_table.csv` (exact match over SUCCESS trials, Wilson 95% CIs, partial credit labeled, Pythia-vs-Mamba comparison column)
  - `results/paper_tables/derived_claims.json`
  - `results/paper_tables/tables.md` (consolidated markdown tables)

### Task 2: Rewrote `FINAL_RESULTS_AUDIT.md`
* **File Updated:** `FINAL_RESULTS_AUDIT.md` (and artifact copy)
* **Modifications:**
  - Retitled to "Experimental Results and Verification Appendix".
  - Removed "Independent Auditor / Senior Engineer" framing.
  - Regenerated all tables from Task 1 outputs.
  - Phase 5 text rewritten to match empirical data: Pythia strong in-distribution (<=2048), collapses at 4096 (0.0000), OOM at 8192; Mamba weaker on passkey, holds needle at 0.80 across 2048 and 4096, collapses at 8192; associative recall poor for both (<=0.40).
  - Explicitly stated that differences with overlapping Wilson CIs are not statistically significant, and only claimed differences with disjoint CIs.
  - Converted causal claims ("GPU SRAM caches", "attention pairwise matching", "kernel overhead") to sentences beginning "A possible explanation, not tested here, is ...".
  - Clarified memory: only Mamba's persistent state is constant (7.50 MB); peak allocated memory grows with context (1,533.05 MB -> 3,036.91 MB in Phase 1). Explained Pythia State_VRAM (192 MB) vs State_Logical (96 MB) as 2x cache-concatenation allocation overhead.
  - Added "Scope and limitations" section covering all 8 specified boundaries.
  - Updated hardware telemetry to measured WSL2 values: GPU 6,143.50 MB, WSL-visible RAM 11.54 GB, torch 2.5.1+cu121, transformers 4.45.2, mamba_ssm 2.2.2, Python 3.10.12. Added author host RAM TODO note. Clarified that Pythia 8,192 OOM is caused by the 5,800.0 MB PyTorch allocator cap.

### Task 3: Fixed `RESEARCH_LIMITATIONS.md`
* **File Updated:** `RESEARCH_LIMITATIONS.md` (and `IMPLEMENTATION.md`)
* **Modifications:**
  - Section 4 rewritten: confirmed both models use identical GPT-NeoX-20B BPE tokenization (identical 64,494 scored tokens, identical 248,843 bytes). Removed claims that PPL cannot be directly compared; kept BPB as a supplementary byte-normalised metric.
  - Section 2.1: documented historical uncapped 6,326.44 MB baseline in `hardware_results.csv` as a separate, uncapped, non-comparable run that may have involved shared memory paging; excluded from headline paper results.
  - Section 5: designated context lengths > 2,048 tokens as extrapolation for both models (both trained at 2,048). Clarified that CSV `Is_Extrapolation` column reflects Mamba logging, while paper tables label both models.
  - RAM inconsistency resolved: updated to measured WSL2-visible RAM (11.54 GB) and added author host RAM TODO note.

### Task 4: Updated Analysis & Visualization Pipeline
* **Files Updated:** `analysis/validate_results.py`, `analysis/visualization.py`
* **Modifications:**
  - In `analysis/validate_results.py`: added rigorous assertions for all Task 1 expected values (PPL, BPB, tokens, bytes, retrieval cell accuracies, OOM counts, decode latency growth, and serving ratios).
  - In `analysis/visualization.py`: updated Phase 5 retrieval to aggregate exact match over SUCCESS trials, mark OOMs explicitly without fabricating valid points, plot Wilson 95% CIs as error bars (`ax.errorbar`), mark context > 2048 as extrapolation for both models, and display OOM in heatmap cells.
  - Regenerated all 12 figures in `results/graphs/` (300 DPI raster PNG and vector PDF).

### Task 5: Repository Cleanup & Archival
* **Files/Directories Moved:**
  - Moved `results/raw/`, `results/validated_results/`, `results/smoke_test_run1_backup/`, `results/smoke_test_run2_backup/`, and `results/smoke_test_run3_backup/` into `results/_archive/`.
  - Added `results/_archive/README.md` clarifying that archived runs are not used in the paper.
* **Pycache Deletion:** Recursively removed all `__pycache__` directories and `*.pyc` files.
* **Manifests & Hashes:**
  - Updated `compute_hashes.py` and `RUN_MANIFEST.md` so that cryptographic SHA256 checksums strictly cover `results/final_experiment/` and `results/paper_tables/`.
* **Root Documentation:** Created root `README.md` documenting single source of truth and one-command reproduction (`python analysis/generate_paper_tables.py && python analysis/validate_results.py`).
* **Superseded Notices:** Prefixed `FINAL_VALIDATION_REPORT.md` and `FINAL_AUDIT_REPORT.md` with explicit superseded notes.

### Task 6: Final Automated Verification
* **File Created:** `tests/verify_final_audit.py`
* **Verification Execution:** Executed `tests/verify_final_audit.py` and `analysis/validate_results.py`.

---

## 3. Exact Assertions and Verification Outcomes

All programmatic assertions executed and **PASSED (100% Success)**:

1. **Phase 2 Autoregressive Decoding Assertions:**
   - Pythia decode latency growth 512 -> 4,096 tokens: **2.47x** (exact ratio: 2.4687, `abs(growth - 2.47) < 0.02` passed).
   - Mamba decode mean latency: **23.2595 ms**, standard deviation: **0.2825 ms** (passed).
   - First sequence length where Mamba decode throughput exceeds Pythia: **2,048 tokens** (Mamba: 43.18 tps vs. Pythia: 36.02 tps, passed).
   - Generated tokens: **128 tokens** for all successful configurations (passed).

2. **Phase 1 Prefill Throughput Assertions:**
   - Pythia prefill throughput: **5514.17 tps** at 512 tokens to **2941.75 tps** at 4,096 tokens (passed).
   - Mamba prefill throughput: **4824.63 tps** at 512 tokens to **5273.80 tps** at 8,192 tokens (passed).

3. **Phase 3 Serving Envelope & OOM Assertions:**
   - Pythia OOM failure count: **6 of 15 configurations** (passed).
   - Mamba OOM failure count: **1 of 15 configurations** (Batch 4, Length 8,192) (passed).
   - Batch 1 max context ratio: **2.0x** (Mamba 8,192 vs. Pythia 4,096, passed).
   - Batch 2 max context ratio: **4.0x** (Mamba 8,192 vs. Pythia 2,048, passed).
   - Batch 4 max context ratio: **4.0x** (Mamba 4,096 vs. Pythia 1,024, passed).

4. **Phase 4 Language Modelling Quality Assertions (PG-19 Test Split):**
   - Pythia Total NLL: **177518.50 nats**, Perplexity: **15.6815**, BPB: **1.0292** (passed).
   - Mamba Total NLL: **170398.83 nats**, Perplexity: **14.0425**, BPB: **0.9879** (passed).
   - Scored Tokens: exactly **64,494 tokens** for both models (passed).
   - Scored Bytes: exactly **248,843 bytes** for both models (passed).

5. **Phase 5 Long-Context Synthetic Retrieval Assertions (Task x Length, n=15 per cell):**
   - Pythia Passkey: `512: 0.8667`, `1024: 1.0000`, `2048: 1.0000`, `4096: 0.0000`, `8192: OOM` (passed).
   - Pythia Needle-in-a-Haystack: `512: 1.0000`, `1024: 1.0000`, `2048: 1.0000`, `4096: 0.0000`, `8192: OOM` (passed).
   - Pythia Associative Recall: `512: 0.0667`, `1024: 0.0000`, `2048: 0.0000`, `4096: 0.0000`, `8192: OOM` (passed).
   - Mamba Passkey: `512: 0.4000`, `1024: 0.4667`, `2048: 0.3333`, `4096: 0.5333`, `8192: 0.0000` (passed).
   - Mamba Needle-in-a-Haystack: `512: 1.0000`, `1024: 1.0000`, `2048: 0.8000`, `4096: 0.8000`, `8192: 0.0667` (passed).
   - Mamba Associative Recall: `512: 0.2667`, `1024: 0.4000`, `2048: 0.2000`, `4096: 0.0000`, `8192: 0.0000` (passed).

6. **Forbidden Strings Scan (`tests/verify_final_audit.py`):**
   - Scanned `FINAL_RESULTS_AUDIT.md` for `"91.1"`, `"3,983.34"`, `"177,522"`, `"170,400"`, `"OOM (Skipped)"`, `"Independent"`, `"cross-tokenizer"`, `"100.0% perfect"`, `"retained higher retrieval fidelity"`.
   - Result: **Zero occurrences found (PASS)**.

7. **Phase 5 Table Parity Check (`tests/verify_final_audit.py`):**
   - Compared every cell of Table 5 in `FINAL_RESULTS_AUDIT.md` against `results/paper_tables/phase5_retrieval_table.csv` across all columns (`Task`, `Target_Seq_Len`, `Regime`, `Pythia_Exact_Match`, `Pythia_95_CI`, `Pythia_Partial_Credit`, `Mamba_Exact_Match`, `Mamba_95_CI`, `Mamba_Partial_Credit`, `Comparison`).
   - Result: **17 of 17 rows match with exact numerical parity (PASS)**.
