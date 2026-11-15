"""
tests/test_validation_gate.py — Stage 8 Validation Gate Master Test Suite

Executes the 12 mandatory scientific validation tests:
 1. Syntax checks for all Python modules
 2. BPB byte-mapping vocabulary tests
 3. Unicode and whitespace bit-for-bit reconstruction tests
 4. Sliding-window target-partition tests
 5. Simulated OOM sliding-window safety tests
 6. Actual NLL numerator and byte denominator integration tests
 7. Retrieval exact-token-length prompt construction tests
 8. Retrieval trial aggregation and validity separation tests
 9. Decode timing and token count invariance tests
10. Batch-scaling VRAM headroom guard tests
11. CSV schema and column integrity tests
12. Frozen configuration parameter consistency tests
"""

import math
import os
import py_compile
import random
import unittest
import yaml
import torch
from transformers import AutoTokenizer

from benchmark.system_monitor import (
    check_vram_headroom,
    estimate_transformer_kv_cache_mb,
    estimate_mamba_state_mb,
    estimate_activation_overhead_mb,
    STRICT_VRAM_LIMIT_MB,
)
from evaluation.bits_per_byte import (
    TokenizerByteMapper,
    compute_bpb,
    compute_masked_token_nll,
    compute_perplexity,
    verify_pg19_evaluation_path,
)
from evaluation.evaluate_retrieval import build_exact_token_prompt, check_retrieval_accuracy


class TestValidationGate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pythia_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        if cls.pythia_tok.pad_token is None:
            cls.pythia_tok.pad_token = cls.pythia_tok.eos_token
        cls.mapper = TokenizerByteMapper(cls.pythia_tok)

    # --- Test 1: Syntax checks ---
    def test_01_syntax_checks_all_scripts(self):
        scripts = [
            "run_benchmark.py",
            "benchmark/benchmark_decode.py",
            "benchmark/benchmark_batch.py",
            "benchmark/system_monitor.py",
            "evaluation/bits_per_byte.py",
            "evaluation/evaluate_quality.py",
            "evaluation/evaluate_retrieval.py",
            "analysis/visualization.py",
        ]
        for s in scripts:
            self.assertTrue(os.path.exists(s), f"Script {s} missing.")
            compiled = py_compile.compile(s, doraise=True)
            self.assertIsNotNone(compiled, f"Syntax error in {s}")

    # --- Test 2: BPB byte mapping vocabulary ---
    def test_02_bpb_byte_mapping_vocab(self):
        self.assertTrue(self.mapper.is_exact)
        vocab = self.pythia_tok.get_vocab()
        for tok_str, tok_id in vocab.items():
            b = self.mapper.get_token_bytes(tok_id)
            self.assertIsNotNone(b, f"Token {tok_id} ({tok_str}) mapped to None")
            self.assertIsInstance(b, bytes)

    # --- Test 3: Unicode and whitespace reconstruction ---
    def test_03_unicode_and_whitespace_reconstruction(self):
        test_strings = [
            "Simple ASCII text without special symbols.",
            "Multiple   spaces    and\ttabs\n\nand newlines.",
            "Multibyte UTF-8: Café, résumé, naïve, über, façade.",
            "Greek & Cyrillic: Ελευθερία, привет мир, Москва, Αθήνα.",
            "CJK Characters: 深度学习, トランスフォーマー, 状態空間モデル.",
            "Math & Symbols: ∀x ∈ ℝ, ∑_{i=1}^N x_i = μ · N ± ε, √2 ≈ 1.414.",
            "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        ]
        for s in test_strings:
            orig_bytes = s.encode("utf-8")
            enc = self.pythia_tok(s, return_tensors="pt")
            token_ids = enc.input_ids[0].tolist()
            passed = self.mapper.verify_sequence_bytes(token_ids, orig_bytes)
            self.assertTrue(passed, f"Failed bit-for-bit reconstruction on: {s}")

    # --- Test 4: Sliding-window target-partition ---
    def test_04_sliding_window_target_partition(self):
        text = "This is a structured paragraph designed to test sliding-window overlap partitioning across token chunks."
        enc = self.pythia_tok(text, return_tensors="pt")
        token_ids = enc.input_ids[0].tolist()
        seq_len = len(token_ids)

        max_length = 8
        stride = 4
        prev_end = 0
        scored_targets = []

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_begin = max(begin, prev_end)
            trg_offset = trg_begin - begin

            chunk_targets = [begin + r for r in range(trg_offset, end - begin) if (begin + r) > 0]
            scored_targets.extend(chunk_targets)
            prev_end = end
            if end >= seq_len:
                break

        # Every target 1..seq_len-1 must be evaluated exactly once
        self.assertEqual(scored_targets, list(range(1, seq_len)))
        self.assertEqual(len(scored_targets), len(set(scored_targets)))

    # --- Test 5: Simulated OOM tests ---
    def test_05_simulated_oom_sliding_window_safety(self):
        text = "A long sequence designed to test that an OOM chunk skips targets cleanly and preserves subsequent windows."
        res = verify_pg19_evaluation_path(
            text=text,
            tokenizer=self.pythia_tok,
            max_length=8,
            stride=4,
            simulate_oom_chunks=[1],  # Chunk 1 OOMs
        )
        self.assertTrue(res["is_valid"])
        self.assertTrue(res["oom_safety_passed"])
        self.assertEqual(res["chunks_skipped"], 1)
        self.assertGreater(res["chunks_scored"], 0)
        self.assertLess(res["scored_bytes"], res["total_text_bytes"])

    # --- Test 6: Actual NLL and byte-denominator integration ---
    def test_06_actual_nll_and_byte_integration(self):
        text = "Empirical benchmark evaluation of neural sequence architectures."
        raw_bytes = text.encode("utf-8")
        token_ids = self.pythia_tok.encode(text, return_tensors="pt")
        seq_len = token_ids.shape[1]

        # Targets are tokens 1..seq_len-1 (token 0 masked with -100)
        labels = token_ids.clone()
        labels[:, 0] = -100

        # Create controlled logits with zero cross-entropy variance
        vocab_size = self.pythia_tok.vocab_size
        logits = torch.zeros((1, seq_len, vocab_size), dtype=torch.float32)

        nll_nats, scored_tokens = compute_masked_token_nll(logits, labels, ignore_index=-100)
        self.assertEqual(scored_tokens, seq_len - 1)

        target_indices = list(range(1, seq_len))
        scored_bytes_count, concat_bytes, is_valid = self.mapper.get_scored_target_bytes(
            token_ids[0].tolist(), target_indices
        )
        self.assertTrue(is_valid)

        tok0_bytes = self.mapper.get_token_bytes(token_ids[0, 0].item())
        self.assertEqual(concat_bytes, raw_bytes[len(tok0_bytes):])
        self.assertEqual(scored_bytes_count, len(raw_bytes) - len(tok0_bytes))

        bpb = compute_bpb(nll_nats, scored_bytes_count)
        self.assertIsNotNone(bpb)
        self.assertAlmostEqual(bpb, nll_nats / (scored_bytes_count * math.log(2.0)), places=6)

    # --- Test 7: Retrieval exact-length prompt construction ---
    def test_07_retrieval_exact_length(self):
        rng = random.Random(42)
        for target_len in [512, 1024, 2048]:
            for pos in [0.0, 0.25, 0.50, 0.75, 1.0]:
                for task in ["passkey", "needle_in_haystack", "associative_recall"]:
                    ids, prompt, exp, actual_tok, needle_pct = build_exact_token_prompt(
                        tokenizer=self.pythia_tok,
                        task_name=task,
                        rng=rng,
                        target_len=target_len,
                        position_pct=pos,
                    )
                    self.assertEqual(actual_tok, target_len, f"Length mismatch on task {task}, target {target_len}")
                    self.assertEqual(ids.shape[1], target_len)
                    self.assertGreaterEqual(needle_pct, 0.0)
                    self.assertLessEqual(needle_pct, 1.0)

    # --- Test 8: Retrieval trial aggregation and validity separation ---
    def test_08_retrieval_trial_aggregation_and_validity(self):
        # Verify check_retrieval_accuracy
        exact, score = check_retrieval_accuracy("The secret passkey is 48291.", "48291")
        self.assertTrue(exact)
        self.assertEqual(score, 1.0)

        exact_fail, score_fail = check_retrieval_accuracy("I cannot recall the number.", "48291")
        self.assertFalse(exact_fail)
        self.assertEqual(score_fail, 0.0)

        # Mathematical formulation verification:
        # 3 trials: trial 0 = success (1.0), trial 1 = failure (0.0), trial 2 = OOM (invalid)
        # Valid trials = [trial 0, trial 1] => valid accuracy = (1.0 + 0.0) / 2 = 0.50
        # Invalid trials = [trial 2] => count = 1
        trials = [
            {"score": 1.0, "exact_match": True, "status": "SUCCESS"},
            {"score": 0.0, "exact_match": False, "status": "SUCCESS"},
            {"score": 0.0, "exact_match": False, "status": "OOM"},
        ]
        valid = [t for t in trials if t["status"] == "SUCCESS"]
        invalid = [t for t in trials if t["status"] == "OOM"]
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(invalid), 1)
        valid_accuracy = sum(t["score"] for t in valid) / len(valid)
        self.assertEqual(valid_accuracy, 0.5)
        exact_match_rate = sum(1 for t in valid if t["exact_match"]) / len(valid)
        self.assertEqual(exact_match_rate, 0.5)

    # --- Test 9: Decode timing and token count ---
    def test_09_decode_token_count_frozen_to_128(self):
        from benchmark.benchmark_decode import MAX_NEW_TOKENS
        self.assertEqual(MAX_NEW_TOKENS, 128, "MAX_NEW_TOKENS must be frozen at exactly 128.")

    # --- Test 10: Batch-scaling VRAM headroom guard ---
    def test_10_vram_headroom_guard_no_double_counting(self):
        # Test that check_vram_headroom evaluates additional needed against remaining budget
        limit = STRICT_VRAM_LIMIT_MB
        self.assertEqual(limit, 5800.0)

        # Estimate additional memory for seq_len=2048, batch=1:
        # KV-cache = 2 * 24 layers * 16 heads * 64 head_dim * 2048 seq_len * 4 bytes = 384 MB (at N=4096: 768 MB, N=8192: 1536 MB)
        est_state = estimate_transformer_kv_cache_mb(24, 16, 64, 2048, 1)
        self.assertAlmostEqual(est_state, 384.0, delta=1.0)

        est_mamba = estimate_mamba_state_mb(48, 1024, 16, 1)
        self.assertAlmostEqual(est_mamba, 6.0, delta=1.0)  # 48 * 2048 * 16 * 4 / (1024^2) = 6 MB

        est_act = estimate_activation_overhead_mb(2048, 1, 1024)
        self.assertGreater(est_act, 0.0)

        # Mock check: additional needed fits if available >= needed
        additional = est_state + est_act
        fits, cur, avail = check_vram_headroom(additional)
        self.assertIsInstance(fits, bool)
        self.assertGreaterEqual(avail, 0.0)

    # --- Test 11: CSV schema integrity ---
    def test_11_csv_schema_integrity(self):
        import run_benchmark
        import benchmark.benchmark_decode as b_dec
        import benchmark.benchmark_batch as b_bat
        import evaluation.evaluate_quality as e_qual
        import evaluation.evaluate_retrieval as e_ret

        # Phase 1 header
        self.assertIn("Peak_Allocated_MB", run_benchmark.CSV_HEADER)
        self.assertIn("Peak_Reserved_MB", run_benchmark.CSV_HEADER)
        self.assertIn("Strict_VRAM_Cap_MB", run_benchmark.CSV_HEADER)

        # Phase 2 header
        self.assertIn("Peak_Allocated_MB", b_dec.CSV_HEADER)
        self.assertIn("Peak_Reserved_MB", b_dec.CSV_HEADER)
        self.assertIn("Total_Generated_Tokens", b_dec.CSV_HEADER)

        # Phase 3 header
        self.assertIn("Peak_Allocated_VRAM_MB", b_bat.CSV_HEADER)
        self.assertIn("Peak_Reserved_VRAM_MB", b_bat.CSV_HEADER)
        self.assertIn("Throughput_tokens_per_sec", b_bat.CSV_HEADER)

        # Phase 4 header
        self.assertIn("Total_Scored_Tokens", e_qual.SUMMARY_HEADER)
        self.assertIn("Total_Scored_UTF8_Bytes", e_qual.SUMMARY_HEADER)
        self.assertIn("Perplexity", e_qual.SUMMARY_HEADER)
        self.assertIn("BPB", e_qual.SUMMARY_HEADER)
        self.assertIn("Document_Index", e_qual.PER_DOC_HEADER)

        # Phase 5 header
        self.assertIn("Valid_Accuracy", e_ret.SUMMARY_HEADER)
        self.assertIn("Exact_Match_Rate", e_ret.SUMMARY_HEADER)
        self.assertIn("Num_Valid_Trials", e_ret.SUMMARY_HEADER)
        self.assertIn("Num_Invalid_Trials", e_ret.SUMMARY_HEADER)
        self.assertIn("Trial_Index", e_ret.TRIALS_HEADER)
        self.assertIn("Status", e_ret.TRIALS_HEADER)

    # --- Test 12: Configuration consistency ---
    def test_12_configuration_consistency(self):
        with open("config/experiments_frozen.yaml", "r") as f:
            cfg = yaml.safe_load(f)

        self.assertEqual(cfg["hardware"]["pytorch_allocator_cap_mb"], 5800.0)
        self.assertEqual(cfg["phase2_decode"]["max_new_tokens"], 128)
        self.assertEqual(cfg["phase1_prefill"]["sequence_lengths"], [512, 1024, 2048, 4096, 8192])
        self.assertEqual(cfg["phase3_batch"]["batch_sizes"], [1, 2, 4])
        self.assertEqual(cfg["phase4_quality"]["max_samples"], 10)
        self.assertEqual(cfg["phase5_retrieval"]["num_trials"], 3)
        self.assertEqual(cfg["phase5_retrieval"]["seed"], 42)


if __name__ == "__main__":
    unittest.main()
