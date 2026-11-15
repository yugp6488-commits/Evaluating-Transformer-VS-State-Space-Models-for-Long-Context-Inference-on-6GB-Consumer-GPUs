# Evaluating Transformers and State Space Models on 6 GB GPUs

This repository contains the benchmark code, frozen experiment outputs, validation artifacts, and research notes for a study of long-context inference on memory-constrained consumer GPUs. It compares **Pythia-410M**, a decoder-only Transformer, with **Mamba-370M**, a selective state space model.

The study measures prefill and decode performance, batch and memory scaling, language-model quality, and synthetic long-context retrieval. The checked-in production results were collected in FP32 on an NVIDIA RTX 3050 Laptop GPU under WSL2, with a 5,800 MB PyTorch allocator ceiling.

## Main Findings

These are measurements from one GPU and one software environment, not general performance guarantees.

| Area | Observed result |
| --- | --- |
| Prefill and memory | At batch size 1, Mamba completed the 8,192-token run with 3,036.91 MB peak allocated memory. Pythia reached the configured allocator limit at 8,192 tokens. |
| Decode | Across prompt lengths 512 to 8,192, Mamba decode latency stayed near 23 ms/token. Pythia increased from 17.33 ms/token at 512 tokens to 42.79 ms/token at 4,096 tokens; the 8,192-token run hit the allocator limit. |
| Batch scaling | Of 15 tested batch/context configurations, Pythia had 6 allocator OOMs and Mamba had 1. |
| PG-19 quality | On 10 test-set book extracts, Pythia measured 15.6815 perplexity / 1.0292 BPB; Mamba measured 14.0425 / 0.9879. |
| Synthetic retrieval | Results vary by task and length. At lengths above 2,048 tokens, both models are outside their reported pretraining context; the evaluation is an extrapolation test, not evidence of general long-document reasoning. |

The allocator limit is a PyTorch allocation guard with a margin for runtime and display use. It is not the GPU's physical VRAM capacity, nor does it account for every non-PyTorch allocation. See [RESEARCH_LIMITATIONS.md](RESEARCH_LIMITATIONS.md) for scope and interpretation details.

## Models and Evaluation

| Model | Architecture | Parameters | Evaluated context lengths |
| --- | --- | ---: | --- |
| Pythia-410M (`EleutherAI/pythia-410m`) | Decoder-only Transformer | ~405M | 512, 1,024, 2,048, 4,096, 8,192 |
| Mamba-370M (`state-spaces/mamba-370m`) | Selective state space model | ~370M | 512, 1,024, 2,048, 4,096, 8,192 |

Both checkpoints use the GPT-NeoX tokenizer in the evaluated setup. The benchmark phases are:

1. Prefill latency, throughput, and state/memory measurements.
2. Autoregressive decoding after prompts of varying lengths (128 generated tokens).
3. Batch sizes 1, 2, and 4, including recorded out-of-memory outcomes.
4. PG-19 language-model scoring with sliding windows; reports perplexity and bits per byte (BPB).
5. Passkey retrieval, needle-in-a-haystack, and associative recall at multiple context lengths and insertion positions.

## Repository Guide

| Path | Contents |
| --- | --- |
| [config/experiments_frozen.yaml](config/experiments_frozen.yaml) | Frozen experiment parameters and target environment. |
| [benchmark/](benchmark/) | Decode and batch benchmark implementations plus system telemetry. |
| [evaluation/](evaluation/) | PG-19 quality/BPB and synthetic retrieval evaluation. |
| [tests/](tests/) | Validation and byte-level evaluation tests. |
| [results/final_experiment/](results/final_experiment/) | Authoritative production CSV outputs; treat these as the canonical results. |
| [results/paper_tables/](results/paper_tables/) | Derived paper tables and claims. |
| [results/graphs/](results/graphs/) | Checked-in figures. |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Detailed protocol, phase descriptions, and CSV schemas. |
| [FINAL_RESULTS_AUDIT.md](FINAL_RESULTS_AUDIT.md) | Result audit and interpretation of the production measurements. |
| [RUN_MANIFEST.md](RUN_MANIFEST.md) | Environment metadata and checksums for production outputs. |
| [RESEARCH_LIMITATIONS.md](RESEARCH_LIMITATIONS.md) | Experimental boundaries and limitations. |

## Reproduce the Benchmarks

The benchmark code targets Linux or WSL2 with a CUDA-capable NVIDIA GPU. The recorded environment used Python 3.10.12, CUDA 12.1, PyTorch 2.5.1+cu121, Transformers 4.45.2, and Mamba-SSM 2.2.2. Other environments may require compatible package and kernel versions.

From the repository root in WSL2, create an environment and install the listed dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install **Mamba-SSM 2.2.2** and its required CUDA extensions separately, following the upstream installation instructions for the installed PyTorch/CUDA combination. The dataset and model checkpoints are downloaded from Hugging Face when evaluation scripts run, so network access and sufficient disk space are required.

Set the thread limits used by the experiment, then run a phase from the repository root:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Phase 1 writes to a separate directory so checked-in results are preserved.
python run_benchmark.py --output-dir results/rerun

# Phases 2–5
python -m benchmark.benchmark_decode
python -m benchmark.benchmark_batch
python -m evaluation.evaluate_quality
python -m evaluation.evaluate_retrieval
```

Benchmark commands may create or update output files. Preserve the checked-in canonical data by backing up results or directing outputs to a separate location where supported. The frozen production data is in `results/final_experiment/`; a rerun is a new measurement and should not be presented as the original run. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the phase protocol and output schemas.

Run the validation suite with:

```bash
python -m unittest discover -s tests
```

## Reading the Results

- Start with [results/paper_tables/tables.md](results/paper_tables/tables.md) for the consolidated tables.
- Use `results/final_experiment/*.csv` as the authoritative raw production outputs; derived tables are not a replacement for them.
- Confirm the tested environment and file checksums in [RUN_MANIFEST.md](RUN_MANIFEST.md).
- Contexts longer than 2,048 tokens are extrapolation for both models according to the study's pretraining-context boundary.
- The PG-19 evaluation covers 10 long-form samples, and retrieval tasks are synthetic. These results do not establish performance on all workloads or natural long-document reasoning.

## License and Citation

No license or citation metadata is currently included in this repository. Add the appropriate license and bibliographic record before redistributing the code or citing the study.
