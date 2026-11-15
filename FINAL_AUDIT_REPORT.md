> [!NOTE]
> **SUPERSEDED:** This historical audit report is preserved for development traceability. All authoritative paper deliverables, verified tables, and final metrics derive strictly from `results/final_experiment/*.csv` and are summarized in `FINAL_RESULTS_AUDIT.md` (Experimental Results and Verification Appendix) and `results/paper_tables/`.

# FINAL_AUDIT_REPORT.md — Scientific, Mathematical & Implementation Audit

**Project:** Evaluating Transformer and State Space Models for Long-Context Inference on 6GB Consumer GPUs  
**Target Publication:** IEEE Transactions on Machine Learning Systems / IEEE Access  
**Evaluation Target:** `EleutherAI/pythia-410m` (Transformer) vs. `state-spaces/mamba-370m` (Selective SSM)  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MB GDDR6 physical VRAM, 5800 MB PyTorch allocator cap) under WSL2 Ubuntu 22.04  
**Audit Date:** 2026-11-15  
**Auditor:** Senior Machine Learning Systems Research Engineer  

---

## 1. Executive Summary

This audit constitutes a comprehensive, evidence-based review of the entire benchmark and evaluation pipeline. Each script, mathematical formulation, configuration file, data artifact, and documentation document was systematically inspected against IEEE reproducible research standards.

A script returning exit code 0 or passing a single-configuration smoke test is **not** evidence of scientific correctness. Multiple subtle, compounding issues were identified and classified into **Critical**, **Major**, and **Minor** categories.

All identified issues are detailed below with their root cause, scientific implications, mandatory corrections, verification protocols, and final resolution status.

---

## 2. Issue Inventory & Categorization

| ID | File & Function | Severity | Summary of Issue | Resolution Status |
|---|---|---|---|---|
| **ISSUE-01** | `evaluation/evaluate_quality.py`<br>`evaluate_model()` | **Critical** | OOM handling failed to advance `prev_end = end`, creating inconsistent target token indexing compared to validator. | Corrected |
| **ISSUE-02** | `evaluation/evaluate_quality.py`<br>`load_evaluation_samples()` | **Critical** | Synthetic literary fallback allowed in full evaluation path rather than halting on missing authentic PG-19. | Corrected |
| **ISSUE-03** | `evaluation/evaluate_retrieval.py`<br>`evaluate_task()` | **Critical** | Accuracy averaged across all trials including OOMs (scoring 0.0) instead of computing valid match rate and reporting OOM separately. | Corrected |
| **ISSUE-04** | `evaluation/evaluate_retrieval.py`<br>Trial Preservation | **Critical** | Individual trial outcomes collapsed into a single semicolon string rather than exporting a discrete `trials.csv`. | Corrected |
| **ISSUE-05** | `evaluation/evaluate_quality.py`<br>Per-document output | **Critical** | Only aggregated summary rows exported; no per-document CSV (`phase4_quality_per_document.csv`) exported. | Corrected |
| **ISSUE-06** | `evaluation/bits_per_byte.py`<br>First-Token Explanation | **Major** | Previous report conjectured first token was `'The'` (3 bytes) or BOM; empirical reality is text starts with `'ST'` (2 bytes). | Empirically Verified & Resolved |
| **ISSUE-07** | `run_benchmark.py`<br>`CSV_HEADER` & measurement | **Major** | Phase 1 prefill script captured only allocated VRAM, omitting reserved VRAM (`max_memory_reserved`). | Corrected |
| **ISSUE-08** | `benchmark/benchmark_decode.py`<br>`CSV_HEADER` & measurement | **Major** | Phase 2 decode script captured only allocated VRAM, omitting reserved VRAM (`max_memory_reserved`). | Corrected |
| **ISSUE-09** | `config/experiments.yaml`<br>Configuration Discrepancies | **Major** | Discrepancy between YAML (`max_new_tokens: 50`, `deepmind/pg19`, `results/raw`) and codebase (`128`, `emozilla/pg19`, `results/validated_results`). | Frozen in `experiments_frozen.yaml` |
| **ISSUE-10** | `analysis/visualization.py`<br>Data source directories | **Major** | Visualization script hardcoded to old folders, unable to cleanly ingest `results/final_experiment/`. | Corrected with CLI flag |
| **ISSUE-11** | All benchmark scripts<br>CLI parameterization | **Minor** | Output directories were hardcoded rather than accepting `--output-dir` arguments. | Corrected |
| **ISSUE-12** | `IMPLEMENTATION.md` & `RESEARCH_LIMITATIONS.md`<br>Milestone Status | **Minor** | Documentation previously blurred the distinction between completed smoke tests and completed full experiments. | Corrected |

---

## 3. Detailed Audit Findings

### ISSUE-01 (Critical): OOM `prev_end` Semantic Inconsistency in Language Modeling Quality
* **File & Function:** `evaluation/evaluate_quality.py` (`evaluate_model`) vs. `evaluation/bits_per_byte.py` (`verify_pg19_evaluation_path`).
* **Exact Problem:** In `evaluate_quality.py`, inside `except torch.cuda.OutOfMemoryError:`, execution proceeded via `continue` before reaching `prev_end = end` (line 346). In `verify_pg19_evaluation_path()`, when an OOM was simulated, it executed `prev_end = end; continue`.
* **Why Scientifically Problematic:** If chunk $k$ OOMs and `prev_end` is not advanced to `end`, chunk $k+1$ evaluates `trg_begin = max(begin, prev_end_old)`. Because `begin = begin + stride`, if `stride < max_length`, `trg_begin` points into tokens from the failed chunk. In the presence of an OOM, this causes an inconsistent target span, violating Invariant 2 (Sliding-Window Correctness) and Invariant 3 (OOM Safety).
* **Required Correction:** In `evaluate_quality.py`, when catching an OOM, explicitly assign `prev_end = end` before advancing the loop, record the chunk as skipped, and verify that the target tokens and byte counts for that chunk are strictly excluded from both numerator and denominator.
* **Validation Method:** Controlled integration test in `tests/test_nll_byte_integration.py` simulating OOM on chunk 1 and verifying that chunk 2 evaluates only its non-overlapping target span without duplicates or errors.
* **Final Status:** Corrected in `evaluation/evaluate_quality.py`.

---

### ISSUE-02 (Critical): Synthetic Literary Fallback Substitution Policy
* **File & Function:** `evaluation/evaluate_quality.py` (`load_evaluation_samples`).
* **Exact Problem:** When network streaming from Hugging Face failed, the script silently substituted a 3-paragraph synthetic text multiplied 15 times. Although labeled as synthetic in the CSV, Stage 2 requirement E states: *"Use the authentic PG-19 test split. If the dataset is unavailable, stop and report the problem. Do not silently substitute synthetic data."*
* **Why Scientifically Problematic:** Synthetic fallback text possesses artificially low perplexity due to exact 15x repetition and does not represent long-context literary distribution. An IEEE systems study cannot substitute synthetic text for full PG-19 evaluation.
* **Required Correction:** When running full experiments (non-smoke-test), require authentic PG-19 from `emozilla/pg19` (or `deepmind/pg19`). If both network streaming and local disk cache fail, raise `RuntimeError` and halt immediately.
* **Validation Method:** Verify that `load_evaluation_samples()` returns 10 authentic PG-19 book extracts with total character count > 200,000 characters.
* **Final Status:** Corrected in `evaluation/evaluate_quality.py`.

---

### ISSUE-03 (Critical): Retrieval Accuracy Formulation under OOM / Limit Conditions
* **File & Function:** `evaluation/evaluate_retrieval.py` (`evaluate_task`).
* **Exact Problem:** `avg_accuracy` was calculated as `sum(t["score"] for t in trial_results) / len(trial_results)`. Trials that encountered OOM or VRAM limit were assigned a score of 0.0 and factored into the denominator.
* **Why Scientifically Problematic:** Stage 6 explicitly mandates:
  $$\text{Accuracy} = \frac{\text{successful valid trials}}{\text{total valid trials}}$$
  Conflating OOM failures with algorithmic retrieval errors skews accuracy. An OOM is a systems failure, not an attention retrieval failure.
* **Required Correction:** Partition trials into valid ($N_{valid}$) and invalid ($N_{invalid}$, OOM/VRAM exceeded). Compute `Accuracy = sum(score) / N_valid` for valid trials. Record `N_valid`, `N_invalid`, and `Exact_Match_Rate` separately.
* **Validation Method:** Unit test `tests/test_retrieval_metrics.py` verifying that a mixture of successful trials and OOM trials produces the exact ratio of valid successes over valid trials.
* **Final Status:** Corrected in `evaluation/evaluate_retrieval.py`.

---

### ISSUE-04 (Critical): Trial-Level Granularity Loss in Retrieval CSV
* **File & Function:** `evaluation/evaluate_retrieval.py` (`CSV_HEADER`, `main`).
* **Exact Problem:** Individual trial records were joined into a single semicolon-delimited string (`t0:exact=True,score=1.00...`) within a single summary row.
* **Why Scientifically Problematic:** Stage 6 requires preserving all trial details: requested context length, actual processed tokens, requested position, actual position, seed, expected, generated, exact match, and status. Downstream statistical audit cannot run ANOVA or confidence interval recalculation on string-crammed fields.
* **Required Correction:** Output two separate CSV files:
  1. `phase5_retrieval_trials.csv`: One row per trial ($5 \text{ lengths} \times 5 \text{ positions} \times 3 \text{ tasks} \times 3 \text{ trials} = 225 \text{ rows per model}$).
  2. `phase5_retrieval_summary.csv`: Aggregated accuracy and exact-match rates.
* **Validation Method:** Independent CSV reader checks that every trial row is populated and parses cleanly.
* **Final Status:** Corrected in `evaluation/evaluate_retrieval.py`.

---

### ISSUE-05 (Critical): Missing Per-Document Quality Outputs
* **File & Function:** `evaluation/evaluate_quality.py` (`main`).
* **Exact Problem:** Quality evaluation only wrote a single aggregate row per model to `quality_results.csv`.
* **Why Scientifically Problematic:** Per-document variance, document-specific token counts, byte counts, and individual NLL totals were lost.
* **Required Correction:** Export both:
  1. `phase4_quality_per_document.csv`: Document index, model, document byte count, scored tokens, scored bytes, NLL, PPL, BPB, chunks scored, chunks skipped.
  2. `phase4_quality_summary.csv`: Macro- and micro-averaged aggregate metrics.
* **Validation Method:** Verify that `phase4_quality_per_document.csv` has exactly 10 rows per model for a 10-sample run.
* **Final Status:** Corrected in `evaluation/evaluate_quality.py`.

---

### ISSUE-06 (Major): First-Token Discrepancy Investigation
* **File & Function:** `tests/investigate_first_token.py`, `evaluation/bits_per_byte.py`.
* **Exact Problem:** The previous session report conjectured that the 2-byte difference between 25,000 total document bytes and 24,998 scored bytes was caused by `'The'` (3 bytes) or a Byte Order Mark (BOM).
* **Empirical Reality:**
  - Running `investigate_first_token.py` directly on the first book of `emozilla/pg19` test split revealed:
    - Text starts with: `'ST. PAUL***\n\n\nE-text prepared by Josephine Paolucci...'`
    - Token 0 ID: `1267`
    - Token 0 vocabulary representation: `'ST'`
    - Token 0 raw bytes: `b'ST'`
    - Token 0 byte length: **2 bytes**
    - First 10 raw bytes of original text: `b'ST. PAUL**'`
    - BOM present: **False**
  - **Resolution:** Token 0 is `'ST'`, which is exactly 2 bytes long. Because token 0 serves as un-scored initial prompt context in causal language modeling, its 2 bytes are correctly excluded from the target denominator. Thus, $25,000 - 2 = 24,998$ bytes scored! The conjecture about `'The'` was factually erroneous and has been corrected.
* **Final Status:** Empirically verified and documented.

---

### ISSUE-07 & ISSUE-08 (Major): Omission of Peak Reserved Memory in Phase 1 & Phase 2
* **File & Function:** `run_benchmark.py` and `benchmark/benchmark_decode.py`.
* **Exact Problem:** Both scripts recorded only `torch.cuda.max_memory_allocated()`, omitting `torch.cuda.max_memory_reserved()`.
* **Why Scientifically Problematic:** PyTorch allocates memory in cached memory pools. While `allocated` represents active tensor memory, `reserved` represents the physical allocation requested by PyTorch from the CUDA driver. On a 6GB GPU, allocator fragmentation can cause reserved memory to hit the cap even when allocated memory is lower.
* **Required Correction:** Add `Peak_Allocated_MB` and `Peak_Reserved_MB` columns to both Phase 1 and Phase 2 CSV headers and instrumentation loops.
* **Validation Method:** Verify that CSVs contain both columns and that `Peak_Reserved_MB >= Peak_Allocated_MB`.
* **Final Status:** Corrected in both scripts.

---

### ISSUE-09 (Major): Experiment Configuration Parameter Drift
* **File & Function:** `config/experiments.yaml` vs runner scripts.
* **Exact Problem:**
  - `config/experiments.yaml` had `decoding.max_new_tokens: 50`, while `benchmark_decode.py` had `MAX_NEW_TOKENS = 128`.
  - `experiments.yaml` had `evaluation.quality.dataset: deepmind/pg19`, while the working parquet stream is `emozilla/pg19`.
  - `experiments.yaml` listed legacy output files under `results/raw/`, whereas validated scripts write to `results/validated_results/` and full runs must write to `results/final_experiment/`.
* **Why Scientifically Problematic:** Parameter drift prevents automated reproduction and creates ambiguity in published methods.
* **Required Correction:** Create `config/experiments_frozen.yaml` freezing:
  - `decoding.max_new_tokens: 128`
  - `quality.dataset: emozilla/pg19`
  - `quality.max_samples: 10`
  - Output directory: `results/final_experiment/`
  - All seeds, warm-ups, and repetition counts.
* **Validation Method:** Configuration consistency check test.
* **Final Status:** Frozen and documented in Stage 7.

---

### ISSUE-10 & ISSUE-11 (Major / Minor): Pipeline Output Parameterization
* **File & Function:** `run_benchmark.py`, `benchmark_decode.py`, `benchmark_batch.py`, `evaluate_quality.py`, `evaluate_retrieval.py`, `visualization.py`.
* **Exact Problem:** Output directories were hardcoded to `results/validated_results/`, preventing cleanly routing outputs to `results/final_experiment/` without code edits.
* **Required Correction:** Add an optional `--output-dir` argument (defaulting to `results/final_experiment`) to all scripts, with fallback to script defaults when omitted.
* **Validation Method:** Run with `--output-dir results/final_experiment` and verify clean CSV output.
* **Final Status:** Corrected across all scripts.

---

## 4. Preservation of Legacy Data

The following existing directories are strictly preserved and untouched:
- `results/raw/`: Original legacy baseline data (unconstrained VRAM, Pythia 8192 reached 6326.44 MB).
- `results/smoke_test_run1_backup/`: First single-configuration smoke test backup.
- `results/smoke_test_run2_backup/`: Second smoke test backup.
- `results/smoke_test_run3_backup/`: Third smoke test backup.
- `results/validated_results/`: Validated smoke test outputs from prior corrections.
- Root legacy files: `hardware_results.csv`, `hardware_results_strict6gb.csv`.

All final full experimental runs will be written exclusively to:
`results/final_experiment/`

---

## 5. Audit Conclusion

The codebase had solid foundations but suffered from 5 critical scientific bugs in metric calculation, OOM handling, dataset fallback, and trial preservation. With these issues identified and resolved, the pipeline is ready for rigorous validation gate execution (Stage 8) and subsequent full experiment execution (Stage 9).
