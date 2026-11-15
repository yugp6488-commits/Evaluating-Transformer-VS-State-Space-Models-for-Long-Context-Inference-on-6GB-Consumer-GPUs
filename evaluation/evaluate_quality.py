"""
evaluation/evaluate_quality.py — Phase 4: Language Modelling Quality

IEEE Benchmark Study: Evaluating Transformer and State Space Models
for Long-Context Inference on 6GB Consumer GPUs

Evaluates token-level perplexity and Bits Per Byte (BPB) on a held-out
subset of the PG-19 dataset for both Pythia-410M and Mamba-370M using
their respective tokenizers and sliding-window chunking.

BPB Byte Alignment Invariant:
    The UTF-8 byte denominator counts ONLY the text corresponding to
    the target tokens that were actually scored (i.e., tokens whose
    labels != -100 and that were not skipped due to OOM). Bytes from
    overlap-prefix tokens (masked as -100) and OOM-skipped chunks
    are excluded. This guarantees that the NLL numerator and the byte
    denominator measure the same text span.

Outputs:
    - phase4_quality_per_document.csv (per-document breakdown)
    - phase4_quality_summary.csv (aggregate macro/micro summary)

Usage:
    OMP_NUM_THREADS=4 python -m evaluation.evaluate_quality [--output-dir <path>] [--smoke-test]
"""

import argparse
import csv
import gc
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:
    MambaLMHeadModel = None

from benchmark.system_monitor import (
    log_system_metadata,
    enforce_vram_ceiling,
    STRICT_VRAM_LIMIT_MB,
)
from evaluation.bits_per_byte import (
    TokenizerByteMapper,
    compute_bpb,
    compute_masked_token_nll,
    compute_perplexity,
    count_utf8_bytes,
    verify_pg19_evaluation_path,
)

# ---------------------------------------------------------------------------
# Enforce PyTorch allocator ceiling BEFORE any CUDA allocations
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    enforce_vram_ceiling(STRICT_VRAM_LIMIT_MB)

# ---------------------------------------------------------------------------
# Constants & Configuration Defaults
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 10
MAX_LENGTH = 2048
STRIDE = 512
DEFAULT_OUTPUT_DIR = "results/validated_results"

PER_DOC_HEADER = [
    "Document_Index",
    "Model",
    "Dataset",
    "Total_Text_Bytes",
    "Total_Tokens",
    "Scored_Tokens",
    "Skipped_Tokens",
    "Scored_UTF8_Bytes",
    "Skipped_UTF8_Bytes",
    "Chunks_Scored",
    "Chunks_OOM_Skipped",
    "Perplexity",
    "BPB",
    "Total_NLL_nats",
    "Avg_NLL_per_token",
    "Strict_VRAM_Cap_MB",
    "Within_Allocator_Cap",
]

SUMMARY_HEADER = [
    "Model",
    "Dataset",
    "Num_Samples",
    "Max_Length",
    "Stride",
    "Total_Scored_Tokens",
    "Total_Scored_UTF8_Bytes",
    "Total_Text_UTF8_Bytes",
    "Chunks_Scored",
    "Chunks_OOM_Skipped",
    "Perplexity",
    "BPB",
    "Total_NLL_nats",
    "Avg_NLL_per_token",
    "Strict_VRAM_Cap_MB",
    "Within_Allocator_Cap",
]

MODEL_CONFIGS = [
    {
        "name": "Pythia-410M",
        "hf_id": "EleutherAI/pythia-410m",
        "tokenizer_id": "EleutherAI/pythia-410m",
        "family": "transformer",
    },
    {
        "name": "Mamba-370M",
        "hf_id": "state-spaces/mamba-370m",
        "tokenizer_id": "EleutherAI/gpt-neox-20b",
        "family": "ssm",
    },
]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Dataset Loading (Authentic PG-19 strictly required for full runs)
# ---------------------------------------------------------------------------
def load_evaluation_samples(
    max_samples: int = MAX_SAMPLES,
    allow_synthetic: bool = False,
) -> Tuple[List[str], str, bool]:
    """Load text samples from PG-19 test split.

    Strictly halts if authentic PG-19 cannot be downloaded, unless
    allow_synthetic=True is explicitly passed.

    Returns:
        (samples, dataset_name, is_synthetic)
    """
    samples = []

    # 1. Primary: Stream parquet export of PG-19 test split (emozilla/pg19)
    try:
        from datasets import load_dataset
        print("  Attempting to stream authentic PG-19 test split (emozilla/pg19)...")
        dataset = load_dataset(
            "emozilla/pg19",
            split="test",
            streaming=True,
        )
        for item in dataset:
            text = item.get("text", "")
            if len(text) > 1000:
                samples.append(text[:25000])  # Cap sample length for bounded memory
                if len(samples) >= max_samples:
                    break
        if samples:
            print(f"  [AUTHENTIC] Successfully loaded {len(samples)} authentic PG-19 book extracts.")
            return samples, "PG-19 (test)", False
    except Exception as e:
        print(f"  [INFO] emozilla/pg19 streaming failed: {e}")

    # 2. Secondary: Stream deepmind/pg19
    try:
        from datasets import load_dataset
        print("  Attempting to stream deepmind/pg19...")
        dataset = load_dataset(
            "deepmind/pg19",
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
        for item in dataset:
            text = item.get("text", "")
            if len(text) > 1000:
                samples.append(text[:25000])
                if len(samples) >= max_samples:
                    break
        if samples:
            print(f"  [AUTHENTIC] Successfully loaded {len(samples)} extracts from deepmind/pg19.")
            return samples, "PG-19 (test)", False
    except Exception as e:
        print(f"  [INFO] deepmind/pg19 streaming failed: {e}")

    # 3. Halt if authentic PG-19 unavailable and synthetic not explicitly authorized
    if not allow_synthetic:
        raise RuntimeError(
            "AUTHENTIC_PG19_UNAVAILABLE: Could not load authentic PG-19 test split from emozilla/pg19 or deepmind/pg19. "
            "Under IEEE scientific standards, silent substitution of synthetic text is strictly prohibited. "
            "Please check network connectivity or provide local cached PG-19 data."
        )

    # 4. Standardized literary fallback — ONLY IF EXPLICITLY AUTHORIZED
    print("  " + "!" * 70)
    print("  [CRITICAL NOTICE] Authentic PG-19 could not be downloaded.")
    print("  Explicit synthetic fallback authorized. Recording dataset as:")
    print("  'Synthetic-Literary-Fallback (Non-PG19)'.")
    print("  " + "!" * 70)

    base_texts = [
        (
            "The Project Gutenberg eBook of The History of Antiquity, by Max Duncker. "
            "The dawn of civilization in the fertile valleys of the Nile and the Euphrates "
            "marked the transition from nomadic pastoralism to organized agrarian societies. "
            "Astronomical observations carved upon clay tablets recorded solar eclipses, "
            "planetary conjunctions, and seasonal flooding cycles with remarkable mathematical "
            "regularity. Scribes documented royal genealogies, administrative tax assessments, "
            "and military campaigns across the Levant, establishing the earliest bureaucratic "
            "archives known to archaeological scholarship. The interplay between religious "
            "hierarchies and dynastic rulers shaped monumental architecture, temples, and "
            "fortifications that endured across millennia despite foreign invasions and societal "
            "collapse. Subsequent Hellenistic translations preserved fragments of these earlier "
            "chronicles, transmitting ancient Near Eastern cosmology into European philosophy."
        ),
        (
            "The Principles of Political Economy and Taxation, by David Ricardo. "
            "The value of a commodity, or the quantity of any other commodity for which it will "
            "exchange, depends on the relative quantity of labour which is necessary for its "
            "production, and not on the greater or less compensation which is paid for that labour. "
            "If the quantity of labour realized in commodities regulate their exchangeable value, "
            "every increase of the quantity of labour must augment the value of that commodity on "
            "which it is exercised, as every diminution must lower it. In determining the laws "
            "which regulate rent, wages, and profits, we must consider the natural progress of wealth "
            "and population, the productivity of agricultural soil of varying fertility, and the "
            "accumulation of capital in industrial manufacturing."
        ),
        (
            "A Treatise on Electricity and Magnetism, by James Clerk Maxwell. "
            "The most important aspect of any physical phenomenon is its relation to the energy "
            "which it embodies or transforms. In electrostatic fields, the distribution of potential "
            "satisfies Laplace's differential equation in regions free of charge, and Poisson's "
            "equation in the presence of continuous volume charge density. When magnetic induction "
            "varies with time, electromotive forces are established in accordance with Faraday's "
            "law of induction. The introduction of the displacement current establishes the symmetric "
            "propagation of transverse electromagnetic waves through dielectric media at the velocity "
            "of light, demonstrating the electromagnetic nature of radiant luminous energy."
        ),
    ]

    for i in range(max_samples):
        base = base_texts[i % len(base_texts)]
        multiplied = (f"Document {i + 1}. " + base + "\n\n") * 15
        samples.append(multiplied)

    return samples, "Synthetic-Literary-Fallback (Non-PG19)", True


# ---------------------------------------------------------------------------
# Sliding-Window Evaluation with Authoritative Byte Mapping
# ---------------------------------------------------------------------------
def evaluate_model(
    model_name: str,
    model,
    tokenizer,
    texts: List[str],
    dataset_name: str,
    max_length: int = MAX_LENGTH,
    stride: int = STRIDE,
    is_mamba: bool = False,
) -> Tuple[Dict, List[List]]:
    """Evaluate Perplexity and Bits Per Byte over texts using sliding window.

    Authoritative Byte Alignment Invariants:
        1. Original normalized UTF-8 text is authoritative byte source.
        2. TokenizerByteMapper maps every token ID to exact raw UTF-8 bytes.
        3. Sliding-window overlap context tokens (masked with label=-100) contribute 0 bytes.
        4. Token 0 in the first chunk (un-scored prompt context) contributes 0 bytes.
        5. OOM-skipped chunks contribute 0 tokens and 0 bytes.
        6. Each scored target token contributes exactly once to both NLL and bytes.
        7. If exact byte alignment cannot be verified, BPB is reported as None (N/A).

    Returns:
        (summary_dict, per_document_rows)
    """
    byte_mapper = TokenizerByteMapper(tokenizer)
    bpb_available = byte_mapper.is_exact
    if not bpb_available:
        print(f"  [NOTICE] Tokenizer {tokenizer.name_or_path} does not support exact byte mapping. BPB will be reported as unavailable.")

    total_nll = 0.0
    total_scored_tokens = 0
    total_scored_bytes = 0
    total_text_bytes = 0
    total_chunks_scored = 0
    total_chunks_oom_skipped = 0

    per_doc_rows = []

    for idx, text in enumerate(texts):
        orig_bytes = text.encode("utf-8")
        doc_text_bytes = len(orig_bytes)
        total_text_bytes += doc_text_bytes

        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(DEVICE)
        token_ids_list = encodings.input_ids[0].tolist()
        seq_len = input_ids.size(1)

        if seq_len <= 1:
            continue

        doc_bpb_available = bpb_available
        if doc_bpb_available:
            doc_is_exact = byte_mapper.verify_sequence_bytes(token_ids_list, orig_bytes)
            if not doc_is_exact:
                print(f"  [WARNING] Sample {idx+1}: Token sequence failed bit-for-bit byte reconstruction. Disabling BPB.")
                doc_bpb_available = False

        doc_nll = 0.0
        doc_scored_tokens = 0
        doc_skipped_tokens = 0
        doc_scored_bytes = 0
        doc_skipped_bytes = 0
        doc_chunks_scored = 0
        doc_chunks_oom = 0

        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_begin = max(begin, prev_end)
            trg_offset = trg_begin - begin

            chunk_ids = input_ids[:, begin:end]
            labels = chunk_ids.clone()
            if trg_offset > 0:
                labels[:, :trg_offset] = -100

            chunk_target_abs_indices = [
                begin + r for r in range(trg_offset, end - begin)
                if (begin + r) > 0  # Token 0 is prompt context, un-scored
            ]

            try:
                sync_cuda()
                with torch.no_grad():
                    outputs = model(chunk_ids)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                nll, n_tok = compute_masked_token_nll(logits, labels, ignore_index=-100)
                doc_nll += nll
                doc_scored_tokens += n_tok

                assert len(chunk_target_abs_indices) == n_tok, (
                    f"Target token count mismatch: {len(chunk_target_abs_indices)} indices vs {n_tok} scored in NLL."
                )

                if doc_bpb_available:
                    chunk_bytes_count, _, is_valid = byte_mapper.get_scored_target_bytes(
                        token_ids_list, chunk_target_abs_indices
                    )
                    if not is_valid:
                        doc_bpb_available = False
                        print(f"  [WARNING] Sample {idx+1}, chunk [{begin}:{end}]: Byte mapping invalid. Disabling BPB.")
                    else:
                        if n_tok > 0:
                            assert chunk_bytes_count > 0, (
                                f"Validation failure: {n_tok} scored tokens yielded 0 bytes."
                            )
                        doc_scored_bytes += chunk_bytes_count

                doc_chunks_scored += 1
                prev_end = end

            except torch.cuda.OutOfMemoryError:
                print(f"    [OOM] OutOfMemoryError at sample {idx+1}, chunk [{begin}:{end}]. Skipping chunk.")
                doc_chunks_oom += 1
                doc_skipped_tokens += len(chunk_target_abs_indices)
                if doc_bpb_available:
                    sk_bytes, _, _ = byte_mapper.get_scored_target_bytes(token_ids_list, chunk_target_abs_indices)
                    doc_skipped_bytes += sk_bytes
                # CRITICAL SCIENTIFIC FIX: Advance prev_end so skipped chunk does not cause duplicate targets
                prev_end = end
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if end >= seq_len:
                    break
                continue
            finally:
                for varname in ("logits", "outputs"):
                    if varname in locals():
                        try:
                            del locals()[varname]
                        except Exception:
                            pass
                del chunk_ids, labels
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if end >= seq_len:
                break

        del input_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_nll += doc_nll
        total_scored_tokens += doc_scored_tokens
        total_scored_bytes += doc_scored_bytes
        total_chunks_scored += doc_chunks_scored
        total_chunks_oom_skipped += doc_chunks_oom

        doc_ppl = compute_perplexity(doc_nll, doc_scored_tokens)
        doc_bpb = compute_bpb(doc_nll, doc_scored_bytes) if doc_bpb_available else None

        peak_doc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        doc_within_cap = peak_doc_mb <= STRICT_VRAM_LIMIT_MB

        doc_row = [
            idx + 1,
            model_name,
            dataset_name,
            doc_text_bytes,
            seq_len,
            doc_scored_tokens,
            doc_skipped_tokens,
            doc_scored_bytes if doc_bpb_available else "N/A",
            doc_skipped_bytes if doc_bpb_available else "N/A",
            doc_chunks_scored,
            doc_chunks_oom,
            round(doc_ppl, 4),
            round(doc_bpb, 4) if doc_bpb is not None else "N/A",
            round(doc_nll, 4),
            round(doc_nll / doc_scored_tokens, 4) if doc_scored_tokens > 0 else 0.0,
            STRICT_VRAM_LIMIT_MB,
            doc_within_cap,
        ]
        per_doc_rows.append(doc_row)

        running_ppl = compute_perplexity(total_nll, total_scored_tokens) if total_scored_tokens > 0 else float("inf")
        running_bpb = compute_bpb(total_nll, total_scored_bytes) if bpb_available else None
        bpb_str = f"{running_bpb:.4f}" if running_bpb is not None else "N/A"
        scored_bytes_str = f"{total_scored_bytes:,}" if bpb_available else "N/A"
        print(
            f"    Sample {idx + 1:2d}/{len(texts)}: Scored Tokens={total_scored_tokens:,} | "
            f"Scored Bytes={scored_bytes_str} | Running PPL={running_ppl:.2f} | Running BPB={bpb_str}"
        )

    # --- Global Consistency validation ---
    if bpb_available:
        if total_scored_tokens > 0 and total_scored_bytes == 0:
            print("  [VALIDATION ERROR] Scored tokens > 0 but scored bytes == 0. Disabling BPB.")
            bpb_available = False
        elif total_scored_tokens == 0 and total_scored_bytes > 0:
            print("  [VALIDATION ERROR] Scored bytes > 0 but scored tokens == 0. Disabling BPB.")
            bpb_available = False
        elif total_scored_bytes > total_text_bytes:
            print(f"  [VALIDATION WARNING] Scored bytes ({total_scored_bytes:,}) > total text bytes ({total_text_bytes:,}).")
            bpb_available = False

    ppl = compute_perplexity(total_nll, total_scored_tokens)
    bpb = compute_bpb(total_nll, total_scored_bytes) if bpb_available else None
    avg_nll = total_nll / total_scored_tokens if total_scored_tokens > 0 else 0.0

    summary_dict = {
        "total_scored_tokens": total_scored_tokens,
        "total_scored_bytes": total_scored_bytes if bpb_available else "N/A",
        "total_text_bytes": total_text_bytes,
        "chunks_scored": total_chunks_scored,
        "chunks_oom_skipped": total_chunks_oom_skipped,
        "total_nll": total_nll,
        "avg_nll": avg_nll,
        "perplexity": ppl,
        "bpb": bpb,
        "bpb_available": bpb_available,
    }

    return summary_dict, per_doc_rows


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4: Language Modelling Quality Evaluation")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick single-sample test")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save CSV outputs")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES, help="Number of PG-19 test samples")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic fallback if PG-19 unavailable")
    args = parser.parse_args()

    smoke_test = args.smoke_test
    output_dir = args.output_dir
    max_samples = 1 if smoke_test else args.max_samples
    max_length = 512 if smoke_test else MAX_LENGTH
    stride = 256 if smoke_test else STRIDE

    os.makedirs(output_dir, exist_ok=True)
    log_system_metadata(os.path.join(output_dir, "system_metadata.csv"))

    print("\n" + "=" * 75)
    print("  Phase 4: Language Modelling Quality Evaluation (PPL & BPB)")
    print("  IEEE Study: Long-Context Inference on 6GB Consumer GPUs")
    if smoke_test:
        print("  *** RUNNING IN SMOKE-TEST MODE (1 sample, max_length=512, Pythia only) ***")
    print(f"  Target Output Directory: {output_dir}")
    print("=" * 75)

    texts, dataset_name, is_synthetic = load_evaluation_samples(
        max_samples=max_samples,
        allow_synthetic=args.allow_synthetic or smoke_test,
    )
    print(f"  Evaluation corpus: {dataset_name} ({len(texts)} texts, {sum(len(t) for t in texts):,} characters).")

    all_summary_rows = []
    all_per_doc_rows = []

    configs_to_run = [MODEL_CONFIGS[0]] if smoke_test else MODEL_CONFIGS

    for config in configs_to_run:
        print(f"\n  {'='*65}")
        print(f"  Evaluating Model: {config['name']} ({config['hf_id']})")
        print(f"  Tokenizer: {config['tokenizer_id']}")
        print(f"  {'='*65}")

        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_id"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Pre-flight BPB and byte-alignment verification on evaluation text
        if texts:
            verif = verify_pg19_evaluation_path(
                text=texts[0],
                tokenizer=tokenizer,
                max_length=max_length,
                stride=stride,
            )
            if verif["is_valid"]:
                print(f"  [BYTE-ALIGNMENT VERIFIED] All 4 BPB invariants satisfied for {config['name']} "
                      f"(Tokens={verif['scored_tokens']:,}, Bytes={verif['scored_bytes']:,}).")
            else:
                print(f"  [BYTE-ALIGNMENT WARNING] Invariants not satisfied: {verif['errors']}")

        if config["family"] == "ssm":
            if MambaLMHeadModel is None:
                print("  [ERROR] mamba_ssm package not available. Skipping Mamba evaluation.")
                continue
            model = MambaLMHeadModel.from_pretrained(
                config["hf_id"], device=DEVICE, dtype=torch.float32
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config["hf_id"], torch_dtype=torch.float32
            ).to(DEVICE)

        model.eval()

        summary, doc_rows = evaluate_model(
            model_name=config["name"],
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            dataset_name=dataset_name,
            max_length=max_length,
            stride=stride,
            is_mamba=(config["family"] == "ssm"),
        )
        all_per_doc_rows.extend(doc_rows)

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        within_cap = peak_mb <= STRICT_VRAM_LIMIT_MB

        bpb_val = round(summary["bpb"], 4) if summary["bpb"] is not None else "N/A"

        row = [
            config["name"],
            dataset_name,
            len(texts),
            max_length,
            stride,
            summary["total_scored_tokens"],
            summary["total_scored_bytes"],
            summary["total_text_bytes"],
            summary["chunks_scored"],
            summary["chunks_oom_skipped"],
            round(summary["perplexity"], 4),
            bpb_val,
            round(summary["total_nll"], 4),
            round(summary["avg_nll"], 4),
            STRICT_VRAM_LIMIT_MB,
            within_cap,
        ]
        all_summary_rows.append(row)

        bpb_display = f"{summary['bpb']:.4f}" if summary['bpb'] is not None else "N/A (exact alignment unavailable)"
        print(f"\n  --- {config['name']} Summary ---")
        print(f"    Dataset            : {dataset_name}")
        print(f"    Perplexity (PPL)   : {summary['perplexity']:.4f}")
        print(f"    Bits Per Byte      : {bpb_display}")
        print(f"    Scored Tokens      : {summary['total_scored_tokens']:,}")
        print(f"    Scored UTF-8 Bytes : {summary['total_scored_bytes']}")
        print(f"    Total Text Bytes   : {summary['total_text_bytes']:,}")
        print(f"    Chunks Scored      : {summary['chunks_scored']}")
        print(f"    Chunks OOM Skipped : {summary['chunks_oom_skipped']}")
        print(f"    Avg NLL/token      : {summary['avg_nll']:.4f} nats")
        print(f"    Peak Allocator VRAM: {peak_mb:.2f} MB (Cap={STRICT_VRAM_LIMIT_MB:.0f} MB | {within_cap})")

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Export per-document CSV
    per_doc_csv = os.path.join(output_dir, "phase4_quality_per_document.csv")
    with open(per_doc_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PER_DOC_HEADER)
        writer.writerows(all_per_doc_rows)

    # Export summary CSV
    summary_csv = os.path.join(output_dir, "phase4_quality_summary.csv")
    with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(all_summary_rows)

    # Also keep legacy filename quality_results.csv in output_dir for compatibility
    compat_csv = os.path.join(output_dir, "quality_results.csv")
    with open(compat_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(all_summary_rows)

    print(f"\nPhase 4 complete.")
    print(f"  Per-document results: '{per_doc_csv}' ({len(all_per_doc_rows)} rows)")
    print(f"  Summary results     : '{summary_csv}' ({len(all_summary_rows)} rows)")


if __name__ == "__main__":
    main()
