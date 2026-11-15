"""
evaluation/evaluate_retrieval.py — Phase 5: Long-Context Retrieval

IEEE Benchmark Study: Evaluating Transformer and State Space Models
for Long-Context Inference on 6GB Consumer GPUs

Implements three standardized synthetic retrieval tasks:
  1. Passkey Retrieval (exact token memory)
  2. Needle-in-a-Haystack (fact localization in distraction)
  3. Associative Recall (key-value binding and retrieval)
Across sequence lengths [512..8192] and needle positions [0%, 25%, 50%, 75%, 100%].

Guarantees:
  - Exact token-length prompt construction verified per tokenizer
  - Actual needle position measured and recorded as percentage of complete prompt
  - Accuracy calculated strictly as: successful_valid_trials / total_valid_trials
  - Invalid (OOM / VRAM_LIMIT_EXCEEDED) trials tracked and reported separately
  - Both trial-level (phase5_retrieval_trials.csv) and summary (phase5_retrieval_summary.csv) exported
  - Context lengths > 2048 explicitly tagged as Is_Extrapolation=True for Mamba

Usage:
    OMP_NUM_THREADS=4 python -m evaluation.evaluate_retrieval [--output-dir <path>] [--smoke-test]
"""

import argparse
import csv
import gc
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    MambaLMHeadModel = None
    InferenceParams = None

from benchmark.system_monitor import (
    log_system_metadata,
    enforce_vram_ceiling,
    check_vram_headroom,
    estimate_transformer_kv_cache_mb,
    estimate_mamba_state_mb,
    estimate_activation_overhead_mb,
    STRICT_VRAM_LIMIT_MB,
)

# ---------------------------------------------------------------------------
# Enforce PyTorch allocator ceiling BEFORE any CUDA allocations
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    enforce_vram_ceiling(STRICT_VRAM_LIMIT_MB)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192]
POSITIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
SEED = 42
NUM_TRIALS = 3
MAMBA_TRAINING_LIMIT = 2048
MAX_NEW_TOKENS = 20
DEFAULT_OUTPUT_DIR = "results/validated_results"

TRIALS_HEADER = [
    "Model",
    "Task",
    "Target_Seq_Len",
    "Actual_Tokens",
    "Requested_Pos_Pct",
    "Actual_Needle_Pos_Pct",
    "Trial_Index",
    "Trial_Seed",
    "Is_Extrapolation",
    "Expected_Answer",
    "Predicted_Answer",
    "Exact_Match",
    "Partial_Score",
    "Status",  # SUCCESS, OOM, VRAM_LIMIT_EXCEEDED
]

SUMMARY_HEADER = [
    "Model",
    "Task",
    "Sequence_Length",
    "Target_Position_Pct",
    "Actual_Needle_Position_Pct",
    "Actual_Processed_Tokens",
    "Is_Extrapolation",
    "Num_Total_Trials",
    "Num_Valid_Trials",
    "Num_Invalid_Trials",
    "Valid_Accuracy",
    "Exact_Match_Rate",
    "Strict_VRAM_Cap_MB",
    "Within_Allocator_Cap",
]

FILLER_SENTENCES = [
    "The quick brown fox jumps over the lazy dog in the tranquil afternoon. ",
    "A journey of a thousand miles begins with a single deliberate step forward. ",
    "To be or not to be, that remains the enduring philosophical question. ",
    "All that glitters is not gold, yet it frequently catches the human eye. ",
    "Knowledge represents power, and wisdom guides the ethical application of knowledge. ",
    "The early bird catches the worm, but the second mouse secures the cheese. ",
    "In the quiet corridors of the archival library, forgotten histories sleep. ",
    "Time and tide wait for no man, nor do they pause for human contemplation. ",
    "Every silver cloud carries subtle light if observed from the proper angle. ",
    "Rigorous practice fosters mastery, though ultimate perfection remains aspirational. ",
    "The autumn leaves drifted silently onto the cobblestone pathways of the ancient city. ",
    "Atmospheric pressure decreases systematically as altitude increases above sea level. ",
    "Electromagnetic radiation travels through the vacuum of space at the speed of light. ",
    "Computational efficiency depends profoundly on memory hierarchy and cache locality. ",
    "The crystalline structure of minerals exhibits precise geometric symmetry under analysis. ",
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

TASK_NAMES = ["passkey", "needle_in_haystack", "associative_recall"]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Exact Token-Length Prompt Builders Using Model's Own Tokenizer
# ---------------------------------------------------------------------------
def build_exact_token_prompt(
    tokenizer,
    task_name: str,
    rng: random.Random,
    target_len: int,
    position_pct: float,
) -> Tuple[torch.Tensor, str, str, int, float]:
    """Construct retrieval prompt matching target_len EXACTLY using model's tokenizer.

    Returns:
        (input_ids_tensor, prompt_text, expected_answer, actual_token_count, actual_needle_position_pct)
    """
    if task_name == "passkey":
        passkey = str(rng.randint(10000, 99999))
        needle = f" The secret passkey is {passkey}. Remember this number."
        question = "\n\nWhat is the secret passkey mentioned above? The passkey is:"
        expected = passkey
    elif task_name == "needle_in_haystack":
        cities = ["Kyoto", "Zurich", "Reykjavik", "Valparaiso", "Montreal", "Florence"]
        animals = ["snow leopard", "blue whale", "peregrine falcon", "golden eagle", "red panda"]
        city = rng.choice(cities)
        animal = rng.choice(animals)
        needle = f" The special fact is: the {animal} lives in {city}."
        expected = f"{animal} lives in {city}"
        question = "\n\nAccording to the text above, what is the special fact? The special fact is: the"
    elif task_name == "associative_recall":
        keys = ["sigma", "omega", "lambda", "kappa", "psi", "delta", "phi"]
        key_word = rng.choice(keys)
        value_num = str(rng.randint(100, 999))
        needle = f" RECORD: key={key_word} maps to value={value_num}."
        expected = value_num
        question = f"\n\nWhat value does key '{key_word}' map to? The value is:"
    else:
        raise ValueError(f"Unknown task: {task_name}")

    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    fixed_len = len(needle_ids) + len(question_ids)

    filler_len = max(target_len - fixed_len, 0)
    before_len = int(round(filler_len * position_pct))
    after_len = filler_len - before_len

    filler_pool_text = "".join(FILLER_SENTENCES) * ((target_len // 200) + 15)
    filler_pool_ids = tokenizer.encode(filler_pool_text, add_special_tokens=False)

    max_offset = max(len(filler_pool_ids) - filler_len - 10, 1)
    offset = rng.randint(0, max_offset)
    filler_before = filler_pool_ids[offset : offset + before_len]
    filler_after = filler_pool_ids[offset + before_len : offset + before_len + after_len]

    final_token_ids = filler_before + needle_ids + filler_after + question_ids

    # Enforce exact token count
    if len(final_token_ids) > target_len:
        excess = len(final_token_ids) - target_len
        if len(filler_after) >= excess:
            filler_after = filler_after[:-excess]
        elif len(filler_before) >= excess:
            filler_before = filler_before[excess:]
        final_token_ids = filler_before + needle_ids + filler_after + question_ids
    elif len(final_token_ids) < target_len:
        deficit = target_len - len(final_token_ids)
        filler_before = filler_pool_ids[:deficit] + filler_before
        final_token_ids = filler_before + needle_ids + filler_after + question_ids

    actual_tokens = len(final_token_ids)
    assert actual_tokens == target_len, f"Token mismatch: {actual_tokens} vs {target_len}"

    needle_start_idx = len(filler_before)
    assert final_token_ids[needle_start_idx : needle_start_idx + len(needle_ids)] == needle_ids, (
        "Needle token sequence verification failed in final prompt."
    )
    actual_needle_pct = round(needle_start_idx / actual_tokens, 4)

    prompt_text = tokenizer.decode(final_token_ids)
    input_ids = torch.tensor([final_token_ids], dtype=torch.long, device=DEVICE)

    return input_ids, prompt_text, expected, actual_tokens, actual_needle_pct


# ---------------------------------------------------------------------------
# Generation & Accuracy Checking
# ---------------------------------------------------------------------------
def generate_greedy(
    model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int, is_mamba: bool
) -> str:
    seq_len = input_ids.size(1)

    with torch.no_grad():
        if is_mamba:
            if InferenceParams is None:
                raise RuntimeError("mamba_ssm is required to run Mamba.")
            total_len = seq_len + max_new_tokens
            ip = InferenceParams(max_seqlen=total_len, max_batch_size=1)

            sync_cuda()
            outputs = model(input_ids, inference_params=ip)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ip.seqlen_offset = seq_len

            generated = [next_token]
            for _ in range(max_new_tokens - 1):
                sync_cuda()
                outputs = model(next_token, inference_params=ip)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                ip.seqlen_offset += 1
                generated.append(next_token)

            gen_ids = torch.cat(generated, dim=1)
            del ip
        else:
            sync_cuda()
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values
            logits = outputs.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            generated = [next_token]
            for _ in range(max_new_tokens - 1):
                sync_cuda()
                outputs = model(next_token, past_key_values=past_kv, use_cache=True)
                past_kv = outputs.past_key_values
                logits = outputs.logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated.append(next_token)

            gen_ids = torch.cat(generated, dim=1)
            del past_kv

    result_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    del gen_ids, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result_text


def check_retrieval_accuracy(
    generated_text: str, expected: str
) -> Tuple[bool, float]:
    gen_lower = generated_text.lower().strip()
    exp_lower = expected.lower().strip()

    exact_match = exp_lower in gen_lower
    if exact_match:
        score = 1.0
    else:
        exp_words = exp_lower.split()
        matched = sum(1 for w in exp_words if w in gen_lower)
        score = matched / len(exp_words) if exp_words else 0.0

    return exact_match, score


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------
def evaluate_task(
    model,
    tokenizer,
    model_name: str,
    task_name: str,
    seq_len: int,
    position_pct: float,
    is_mamba: bool,
    num_trials: int = NUM_TRIALS,
) -> Tuple[Dict, List[List]]:
    """Execute multiple trials for a specific task, length, and position.

    Returns:
        (summary_dict, trial_rows)
    """
    is_extrapolation = model_name == "Mamba-370M" and seq_len > MAMBA_TRAINING_LIMIT
    trial_records = []
    trial_rows = []

    for trial in range(num_trials):
        trial_seed = SEED + trial * 10000 + seq_len + int(position_pct * 100)
        rng = random.Random(trial_seed)

        input_ids, prompt, expected, actual_tokens, actual_needle_pct = build_exact_token_prompt(
            tokenizer=tokenizer,
            task_name=task_name,
            rng=rng,
            target_len=seq_len,
            position_pct=position_pct,
        )

        # Pre-flight VRAM guard: weights already in cur_alloc
        if is_mamba:
            est_state = estimate_mamba_state_mb(48, 1024, 16, 1)
        else:
            est_state = estimate_transformer_kv_cache_mb(24, 16, 64, actual_tokens, 1)
        est_act = estimate_activation_overhead_mb(actual_tokens, 1, 1024)
        additional_needed_mb = est_state + est_act
        fits, cur_alloc, avail = check_vram_headroom(additional_needed_mb)

        if not fits:
            trial_record = {
                "trial": trial,
                "exact_match": False,
                "score": 0.0,
                "expected": expected,
                "generated": "VRAM_LIMIT_EXCEEDED",
                "actual_tokens": actual_tokens,
                "actual_needle_pct": actual_needle_pct,
                "status": "VRAM_LIMIT_EXCEEDED",
            }
            trial_records.append(trial_record)
            trial_rows.append([
                model_name, task_name, seq_len, actual_tokens, position_pct, actual_needle_pct,
                trial, trial_seed, is_extrapolation, expected, "VRAM_LIMIT_EXCEEDED", False, 0.0, "VRAM_LIMIT_EXCEEDED"
            ])
            continue

        try:
            generated = generate_greedy(
                model, tokenizer, input_ids, MAX_NEW_TOKENS, is_mamba
            )
            exact, score = check_retrieval_accuracy(generated, expected)
            cleaned_gen = generated[:80].replace("\n", " ")
            trial_record = {
                "trial": trial,
                "exact_match": exact,
                "score": score,
                "expected": expected,
                "generated": cleaned_gen,
                "actual_tokens": actual_tokens,
                "actual_needle_pct": actual_needle_pct,
                "status": "SUCCESS",
            }
            trial_records.append(trial_record)
            trial_rows.append([
                model_name, task_name, seq_len, actual_tokens, position_pct, actual_needle_pct,
                trial, trial_seed, is_extrapolation, expected, cleaned_gen, exact, round(score, 4), "SUCCESS"
            ])
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trial_record = {
                "trial": trial,
                "exact_match": False,
                "score": 0.0,
                "expected": expected,
                "generated": "OOM",
                "actual_tokens": actual_tokens,
                "actual_needle_pct": actual_needle_pct,
                "status": "OOM",
            }
            trial_records.append(trial_record)
            trial_rows.append([
                model_name, task_name, seq_len, actual_tokens, position_pct, actual_needle_pct,
                trial, trial_seed, is_extrapolation, expected, "OOM", False, 0.0, "OOM"
            ])

    # Scientific Accuracy Formulation: successful valid trials / total valid trials
    valid_trials = [t for t in trial_records if t["status"] == "SUCCESS"]
    invalid_trials = [t for t in trial_records if t["status"] in ("OOM", "VRAM_LIMIT_EXCEEDED")]

    valid_accuracy = (
        sum(t["score"] for t in valid_trials) / len(valid_trials)
        if valid_trials
        else 0.0
    )
    exact_matches = sum(1 for t in valid_trials if t["exact_match"])
    exact_match_rate = (exact_matches / len(valid_trials)) if valid_trials else 0.0

    avg_tokens = (
        sum(t["actual_tokens"] for t in trial_records) / len(trial_records)
        if trial_records
        else seq_len
    )
    mean_needle_pct = (
        sum(t["actual_needle_pct"] for t in trial_records) / len(trial_records)
        if trial_records
        else position_pct
    )

    summary_dict = {
        "actual_tokens": int(avg_tokens),
        "actual_needle_pct": round(mean_needle_pct, 4),
        "is_extrapolation": is_extrapolation,
        "num_total_trials": len(trial_records),
        "num_valid_trials": len(valid_trials),
        "num_invalid_trials": len(invalid_trials),
        "valid_accuracy": round(valid_accuracy, 4),
        "exact_match_rate": round(exact_match_rate, 4),
    }

    return summary_dict, trial_rows


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 5: Long-Context Retrieval Evaluation")
    parser.add_argument("--smoke-test", action="store_true", help="Run single configuration smoke test")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save CSV outputs")
    args = parser.parse_args()

    smoke_test = args.smoke_test
    output_dir = args.output_dir
    seq_lengths = [512] if smoke_test else SEQ_LENGTHS
    positions = [0.5] if smoke_test else POSITIONS
    num_trials = 1 if smoke_test else NUM_TRIALS
    tasks_to_run = ["passkey"] if smoke_test else TASK_NAMES
    configs_to_run = [MODEL_CONFIGS[0]] if smoke_test else MODEL_CONFIGS

    os.makedirs(output_dir, exist_ok=True)
    log_system_metadata(os.path.join(output_dir, "system_metadata.csv"))

    print("\n" + "=" * 75)
    print("  Phase 5: Long-Context Retrieval Evaluation")
    print("  IEEE Study: Long-Context Inference on 6GB Consumer GPUs")
    if smoke_test:
        print("  *** RUNNING IN SMOKE-TEST MODE (Task=passkey, SeqLen=512, Pos=0.5, 1 trial, Pythia only) ***")
    print(f"  Target Output Directory: {output_dir}")
    print("=" * 75)

    all_trial_rows = []
    all_summary_rows = []

    for config in configs_to_run:
        print(f"\n  {'='*65}")
        print(f"  Model: {config['name']} ({config['hf_id']})")
        print(f"  Family: {config['family']}")
        print(f"  {'='*65}")

        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_id"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

        for task_name in tasks_to_run:
            print(f"\n    Task: {task_name.upper()}")
            for seq_len in seq_lengths:
                for pos_pct in positions:
                    extrap_tag = (
                        " [EXTRAPOLATION]"
                        if (config["name"] == "Mamba-370M" and seq_len > MAMBA_TRAINING_LIMIT)
                        else ""
                    )

                    summary, trial_rows = evaluate_task(
                        model=model,
                        tokenizer=tokenizer,
                        model_name=config["name"],
                        task_name=task_name,
                        seq_len=seq_len,
                        position_pct=pos_pct,
                        is_mamba=(config["family"] == "ssm"),
                        num_trials=num_trials,
                    )
                    all_trial_rows.extend(trial_rows)

                    summary_row = [
                        config["name"],
                        task_name,
                        seq_len,
                        pos_pct,
                        summary["actual_needle_pct"],
                        summary["actual_tokens"],
                        summary["is_extrapolation"],
                        summary["num_total_trials"],
                        summary["num_valid_trials"],
                        summary["num_invalid_trials"],
                        summary["valid_accuracy"],
                        summary["exact_match_rate"],
                        STRICT_VRAM_LIMIT_MB,
                        True,
                    ]
                    all_summary_rows.append(summary_row)

                    print(
                        f"      SeqLen={seq_len:5d} | Pos={pos_pct*100:5.1f}% | "
                        f"ActualNeedlePct={summary['actual_needle_pct']*100:5.1f}% | "
                        f"Tokens={summary['actual_tokens']:5d} | "
                        f"ValidAcc={summary['valid_accuracy']:.2f} | ExactRate={summary['exact_match_rate']:.2f} | "
                        f"Valid/Total={summary['num_valid_trials']}/{summary['num_total_trials']}"
                        f"{extrap_tag}"
                    )

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Export trials CSV
    trials_csv = os.path.join(output_dir, "phase5_retrieval_trials.csv")
    with open(trials_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TRIALS_HEADER)
        writer.writerows(all_trial_rows)

    # Export summary CSV
    summary_csv = os.path.join(output_dir, "phase5_retrieval_summary.csv")
    with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(all_summary_rows)

    # Compatibility export
    compat_csv = os.path.join(output_dir, "retrieval_results.csv")
    with open(compat_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(all_summary_rows)

    print(f"\nPhase 5 complete.")
    print(f"  Trials saved to : '{trials_csv}' ({len(all_trial_rows)} rows)")
    print(f"  Summary saved to: '{summary_csv}' ({len(all_summary_rows)} rows)")


if __name__ == "__main__":
    main()
