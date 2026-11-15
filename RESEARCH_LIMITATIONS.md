# RESEARCH_LIMITATIONS.md — Academic Validity & Boundary Analysis

## "Evaluating Transformer and State Space Models for Long-Context Inference on 6GB Consumer GPUs"
**Publication Domain:** IEEE Transactions on Machine Learning Systems / IEEE Computer Architecture Letters  
**Status:** Canonical Scientific Constraints Document  

---

## 1. Executive Scientific Overview

This document rigorously delineates the physical, theoretical, architectural, and procedural boundaries governing our comparative evaluation of **Pythia-410M** (Transformer baseline) and **Mamba-370M** (Selective State Space Model baseline). Acknowledging these constraints is essential for maintaining scientific integrity, avoiding over-generalized conclusions, and enabling reproducible verification by independent peer reviewers.

---

## 2. Hardware Constraints & Consumer Workstation Realities

### 2.1 6GB Physical VRAM Ceiling & Uncapped Historical Baselines
* **Hardware Target:** NVIDIA GeForce RTX 3050 Laptop GPU (6,143.50 MB GDDR6 dedicated VRAM configuration, 5,800.0 MB PyTorch allocator cap).
* **Memory Headroom & Oversubscription:**
  - In our historical uncapped baseline (`hardware_results.csv`), Pythia-410M reported a PyTorch allocator peak of **6,326.44 MB** at a context length of 8,192 tokens (Batch=1, FP32). This run was conducted without an allocator ceiling (`set_per_process_memory_fraction`). Because allocations exceeded the physical 6,143.50 MB boundary, CUDA may have satisfied requests by paging over PCIe into WDDM shared system RAM, introducing latency artifacts. This historical measurement represents a separate, uncapped, non-comparable run that is preserved for audit transparency but excluded from headline paper results.
  - In all controlled production experiments (`results/final_experiment/`), PyTorch's caching allocator is strictly capped at 5,800.0 MB. Under this cap, Pythia-410M deterministically triggers an out-of-memory error at 8,192 tokens rather than paging over PCIe.
  - At batch sizes $B \ge 2$, Pythia-410M experiences deterministic `torch.cuda.OutOfMemoryError` at sequences $\ge 4,096$ tokens, whereas Mamba-370M remains within lower memory bounds due to its constant recurrent state footprint (7.50 MB).

### 2.2 Virtualization & WSL2 Overhead
* **Runtime Layer:** Ubuntu 22.04 LTS running under Windows Subsystem for Linux (WSL2).
* **System RAM:** WSL2-visible system RAM is 11.54 GB (measured dynamically via `psutil.virtual_memory()`).
> [!NOTE]
> **TODO for Authors:** Host RAM as configured by the author: confirm before submission.
* **Kernel & Driver Translation:** WSL2 relies on Microsoft's `dxgkrnl` virtual GPU (vGPU) para-virtualization bus to interface with the Windows WDDM display driver.
* **CPU Contention Safeguards:** Default OpenMP implementations spawn thread pools equal to logical hyper-threaded cores ($16$ threads on an 8-core/16-thread host). In virtualized Linux, this produces severe thread over-subscription, context-switching overhead, and false timing variance.
* **Enforced Mitigation:** All experiments enforce `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`. Researchers reproducing these results must verify this environment variable.

### 2.3 Thermal Throttling in Mobile Form Factors
* Laptop chassis exhibit strict Thermal Design Power (TDP) envelopes (typically 45W–60W Dynamic Boost for the RTX 3050 mobile GPU).
* Extended benchmarking runs induce thermal throttling, causing core clock frequencies to downclock from boost states (~1600 MHz) to base states (~1200 MHz).
* **Mitigation Protocol:** 
  1. All timing blocks enforce pre-measurement warm-ups (`shape_warmups=1`).
  2. Latency is reported as mean $\pm$ standard deviation across $N=3$ repeated trials.
  3. Garbage collection and CUDA cache flushing (`torch.cuda.empty_cache()`) are executed between every parameter sweep.

### 2.4 VRAM Allocator Ceiling vs. System-Level GPU Memory Scope
* **Allocator Mechanism:** PyTorch's `torch.cuda.set_per_process_memory_fraction()` configures the threshold for the internal `c10::cuda::CUDACachingAllocator`. When PyTorch tensor allocations reach this ceiling, the allocator raises a clean `torch.cuda.OutOfMemoryError`.
* **Scope & Architectural Boundaries:**
  - `set_per_process_memory_fraction()` limits **only PyTorch tensor allocations**.
  - It does **not** cap driver/CUDA runtime context overhead (~200–400 MB) initialized upon the first CUDA call.
  - It does **not** intercept direct `cudaMalloc()` calls by external C/C++ or third-party CUDA extensions that bypass PyTorch's allocator.
  - It does **not** control operating system compositor or desktop display memory managed by Windows WDDM.
* **Methodological Justification for 5800 MB Setting:** Setting the fraction to $5800 / 6144 \approx 0.9441$ limits PyTorch tensor allocations to 5800 MB. The 344 MB margin is a best-effort buffer intended to keep total GPU memory usage within the 6144 MB physical dedicated boundary, but this is **not guaranteed** — total dedicated VRAM usage may still exceed 5800 MB due to non-PyTorch allocations.
* **Memory Terminology:**
  - **Peak Allocated (PyTorch):** The peak memory managed by PyTorch's caching allocator (`torch.cuda.max_memory_allocated`).
  - **Peak Reserved (PyTorch):** The peak memory pool held by the allocator, including free blocks (`torch.cuda.max_memory_reserved`). Always ≥ Allocated.
  - **Physical Dedicated VRAM:** The 6144 MB of GDDR6 on the GPU die.
  - **Shared GPU Memory:** System RAM addressable by the GPU over PCIe (up to half of system RAM under WDDM). The driver may page into this transparently when dedicated VRAM is exhausted. This is distinct from dedicated VRAM and is not measured by PyTorch's memory APIs.

---

## 3. Model Architecture & Parameter Symmetry

### 3.1 Structural Equivalence vs. Parameter Discrepancy
To isolate attention-versus-recurrence characteristics, models must be as closely size-matched as public checkpoints permit:

$$\Delta_{\text{params}} = \frac{405\text{M} - 370\text{M}}{405\text{M}} \approx 8.64\%$$

* **Pythia-410M (405M parameters):** 24 transformer layers, hidden dimension $d=1024$, 16 attention heads, 25% Rotary Position Embedding (RoPE).
* **Mamba-370M (370M parameters):** 48 selective SSM layers, model dimension $d=1024$, state expansion factor $E=2$, recurrent state dimension $d_{state}=16$, convolutional kernel $d_{conv}=4$.
* **Implication:** The 8.6% parameter difference favored Pythia-410M in raw capacity. Any efficiency or scaling advantages observed in Mamba occur *despite* Pythia's slightly larger capacity, strengthening the validity of Mamba's memory efficiency findings.

### 3.2 Pre-training Domain Uniformity
* Both models were trained from scratch on **The Pile** dataset (EleutherAI, 825 GB of diverse text).
* Pre-training data distribution is held constant, minimizing domain-shift confounding variables that frequently plague cross-architecture comparisons in the literature.

---

## 4. Tokenizer Architecture & Evaluation Normalization

### 4.1 Tokenizer Vocabulary Equivalence
Both Pythia-410M and Mamba-370M utilize the identical GPT-NeoX-20B Byte-Pair Encoding (BPE) vocabulary and tokenizer implementation. Empirical verification on the PG-19 test split confirms exact tokenization parity across both models:
* Identical total scored token count: exactly **64,494 tokens** across all 10 evaluated book extracts.
* Identical total scored UTF-8 bytes: exactly **248,843 bytes** across all 10 evaluated book extracts.
* Identical per-document token sequences and byte lengths.

### 4.2 Direct Comparability of Perplexity (PPL)
Because both models evaluate identical token sequences with identical denominators, token-level perplexity (PPL) is directly comparable between Pythia-410M and Mamba-370M in this study:

$$\text{PPL} = \exp\left( \frac{1}{N_{\text{tokens}}} \sum_{i=1}^{N_{\text{tokens}}} -\ln P(w_i \mid w_{<i}) \right)$$

No vocabulary mismatch or token segmentation divergence exists between these models.

### 4.3 Supplementary Bits Per Byte (BPB) Normalization
Bits Per Byte (BPB) is reported alongside PPL as a supplementary, byte-normalised information-theoretic metric:

$$\text{BPB} = \frac{\sum_{i=1}^{N_{\text{tokens}}} -\ln P(w_i \mid w_{<i})}{\text{Length}_{\text{UTF-8}}(\text{Text}) \cdot \ln(2)}$$

* **Role:** BPB provides an intuitive measure of Shannon compression efficiency relative to raw uncompressed text bytes, reinforcing the direct perplexity comparisons.

---

## 5. Long-Context Extrapolation Boundaries

### 5.1 Pre-training Length Ceiling ($N_{\text{train}} = 2048$)
Both Pythia-410M and Mamba-370M were pre-trained with a fixed context window of **2,048 tokens**. Evaluating contexts at $N \in \{4096, 8192\}$ tests **out-of-distribution length extrapolation** for both architectures.

```
    [0 tokens] --------------- [2048 tokens] ------------------------ [8192 tokens]
             |<- In-Distribution Range ->|<- Extrapolation / Out-of-Distribution ->|
                                         |   (Applies to BOTH Pythia and Mamba)   |
```

* **Note on Dataset Flag:** In the raw trial exports (`phase5_retrieval_trials.csv`), the `Is_Extrapolation` boolean column reflects whether context exceeded the pre-training window during Mamba logging. In all paper tables and analyses, context lengths $> 2048$ are explicitly designated as extrapolation for **both** models.

### 5.2 Transformer Extrapolation Failure Modes (RoPE Degradation)
* Pythia applies Rotary Position Embeddings (RoPE) to 25% of each head dimension.
* When sequence lengths exceed 2,048 tokens without position interpolation (e.g., YaRN or NTK-aware scaling), attention logits encounter unseen rotary angle rotations.
* In Phase 5 retrieval benchmarks, Pythia experiences a complete collapse at 4,096 tokens (0.0000 exact match across all tasks) and triggers an OOM at 8,192 tokens.

### 5.3 State Space Model Extrapolation Modes (Recurrent State Bottleneck)
* Mamba contains **no explicit positional embeddings**. Positional order is encoded implicitly via the sequential recurrence equation:

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t$$

* **Theoretical Horizon vs. Practical Capacity:** In principle, an SSM can unroll indefinitely. However, Mamba-370M compresses all historical sequence context into a fixed-size latent recurrent state consisting of:
  - SSM State: $L \times d_{\text{inner}} \times d_{\text{state}} \times 4\text{ bytes} = 48 \times 2048 \times 16 \times 4 = 6,291,456\text{ bytes} = 6.00\text{ MB}$
  - Conv State: $L \times d_{\text{inner}} \times d_{\text{conv}} \times 4\text{ bytes} = 48 \times 2048 \times 4 \times 4 = 1,572,864\text{ bytes} = 1.50\text{ MB}$
  - Total Recurrent State: **7.50 MB** (constant $\mathcal{O}(1)$ footprint regardless of context length $N$).

* At $N = 8192$, the model must compress $8,192 \times 1024$ tokens of historical context into this identical 7.50 MB state buffer—a **$4\times$ information compression ratio** relative to its 2,048-token training regime.
* In Phase 5 retrieval benchmarks, Mamba maintains moderate recall at 4,096 tokens (0.5333 passkey, 0.8000 needle), but collapses at 8,192 tokens (0.0000 passkey, 0.0667 needle, 0.0000 associative recall).

---

## 6. Synthetic vs. Naturalistic Retrieval Limitations

### 6.1 Synthetic Task Scope
Phase 5 evaluates long-context fidelity via three synthetic retrieval protocols:
1. **Passkey Retrieval:** High-entropy 5-digit key lookup amidst repetitive distractor sentences.
2. **Needle-in-a-Haystack:** Factual relation retrieval at specified context depths ($0\%, 25\%, 50\%, 75\%, 100\%$).
3. **Associative Recall:** Key-value binding resolution (`RECORD: key => value`).

### 6.2 Ecological Validity Caveat
* While synthetic tasks cleanly isolate token retention and recall depth, they do not evaluate complex multi-hop reasoning, narrative summarization, or cross-document discourse synthesis.
* High passkey retrieval accuracy does not necessarily imply equivalent performance on naturalistic long-document comprehension benchmarks (e.g., L-Eval or SCROLLS).

### 6.3 Evaluation Sample Size
* In Phase 4, computational bounds on 6GB VRAM necessitate restricting the PG-19 held-out evaluation to **10 long-form text samples** evaluated via sliding window (`max_length=2048`, `stride=512`).
* While statistically representative of literary text, confidence intervals reflect this sample size.

---

## 7. Determinism & Scientific Reproducibility Controls

To ensure that independent investigators obtain identical results within hardware tolerances, the following reproducibility controls are embedded into the test suite:

1. **Fixed PRNG Seeding:** All synthetic prompt generators, sentence shufflers, and passkey generators use deterministic seeds keyed to trial index, task, length, and position:
   $$\text{Seed} = 42 + (\text{Trial} \times 10000) + N_{\text{seq}} + \lfloor 100 \times \text{Depth} \rfloor$$
2. **Identical Cross-Model Prompts:** Prompt strings are generated independently of tokenizer instances. Both Pythia and Mamba process byte-for-byte identical textual prompts.
3. **Hardware Telemetry Logging:** Every benchmark execution automatically appends hardware state (VRAM allocated/reserved, system RAM, CPU load, driver versions) to `results/raw/system_metadata.csv`.
4. **Greedy Decoding:** All text generation operates with temperature $T=0.0$ (argmax greedy decoding) to eliminate stochastic sampling variance.
