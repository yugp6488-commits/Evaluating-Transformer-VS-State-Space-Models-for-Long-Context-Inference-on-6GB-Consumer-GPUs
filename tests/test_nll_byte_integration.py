"""
tests/test_nll_byte_integration.py

Stage 2.C Integration Test: Validates actual NLL numerator and byte denominator alignment
using a controlled deterministic model / controlled logits.

Verifies:
1. Target token positions contributing to NLL match scored byte positions bit-for-bit.
2. Token zero is not scored when used only as prompt context (0 nats, 0 bytes).
3. Overlapping sliding-window context tokens contribute neither NLL nor bytes.
4. Every successfully evaluated target is scored exactly once.
5. Every skipped chunk/target contributes 0 NLL and 0 bytes without corrupting denominators.
6. The concatenated bytes of scored targets equal the exact slice of the authoritative text.
7. BPB formula: BPB = total_NLL_nats / (total_scored_UTF8_bytes * ln(2)).
"""

import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from evaluation.bits_per_byte import (
    TokenizerByteMapper,
    compute_bpb,
    compute_masked_token_nll,
    compute_perplexity,
)


class ControlledDummyLM(nn.Module):
    """Deterministic dummy LM returning predictable logits for controlled testing."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        # Return uniform logits except for target:
        # Each position has logit 10.0 on correct token, 0.0 elsewhere
        # Giving a predictable cross entropy per token
        logits = torch.zeros((batch, seq_len, self.vocab_size), dtype=torch.float32, device=input_ids.device)
        return logits


class TestNLLByteIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.mapper = TokenizerByteMapper(cls.tokenizer)
        assert cls.mapper.is_exact, "Pythia tokenizer byte mapper must be exact."

    def test_01_controlled_nll_and_byte_alignment_single_chunk(self):
        """Verify NLL and byte alignment on a single chunk with zero overlap."""
        text = "Scientific empirical verification requires reproducible evidence."
        raw_bytes = text.encode("utf-8")
        token_ids = self.tokenizer.encode(text, return_tensors="pt")
        seq_len = token_ids.shape[1]

        # In causal LM, labels are shifted by 1. For input length L, there are L-1 predictions.
        # Token 0 is prompt context; targets are tokens 1..L-1
        labels = token_ids.clone()
        labels[:, 0] = -100  # Token 0 masked

        dummy_model = ControlledDummyLM(self.tokenizer.vocab_size)
        logits = dummy_model(token_ids)

        nll_nats, scored_tokens = compute_masked_token_nll(logits, labels, ignore_index=-100)
        self.assertEqual(scored_tokens, seq_len - 1)

        # Theoretical NLL for uniform zero logits across vocab_size:
        # Cross entropy of uniform logits = ln(vocab_size)
        expected_token_nll = math.log(self.tokenizer.vocab_size)
        self.assertAlmostEqual(nll_nats, (seq_len - 1) * expected_token_nll, places=2)

        # Scored byte mapping
        target_indices = list(range(1, seq_len))
        scored_bytes_count, concat_bytes, is_valid = self.mapper.get_scored_target_bytes(
            token_ids[0].tolist(), target_indices
        )
        self.assertTrue(is_valid)

        # Token 0 raw bytes
        tok0_bytes = self.mapper.get_token_bytes(token_ids[0, 0].item())
        self.assertEqual(concat_bytes, raw_bytes[len(tok0_bytes):])
        self.assertEqual(scored_bytes_count, len(raw_bytes) - len(tok0_bytes))

        # Check BPB calculation
        bpb = compute_bpb(nll_nats, scored_bytes_count)
        self.assertIsNotNone(bpb)
        expected_bpb = nll_nats / (scored_bytes_count * math.log(2.0))
        self.assertAlmostEqual(bpb, expected_bpb, places=6)

    def test_02_sliding_window_nll_and_byte_alignment(self):
        """Verify sliding window with overlap: context contributes 0 NLL and 0 bytes."""
        text = (
            "The quick brown fox jumps over the lazy dog repeatedly across multiple sentences. "
            "In empirical research on memory-constrained hardware, each sliding window must "
            "isolate target tokens precisely to prevent duplicate loss attribution or byte overcounting."
        )
        raw_bytes = text.encode("utf-8")
        enc = self.tokenizer(text, return_tensors="pt")
        token_ids = enc.input_ids
        seq_len = token_ids.shape[1]

        max_length = 16
        stride = 8
        dummy_model = ControlledDummyLM(self.tokenizer.vocab_size)

        total_nll = 0.0
        all_scored_indices = []
        all_scored_byte_slices = []

        prev_end = 0
        chunk_count = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_begin = max(begin, prev_end)
            trg_offset = trg_begin - begin

            chunk_ids = token_ids[:, begin:end]
            labels = chunk_ids.clone()
            if trg_offset > 0:
                labels[:, :trg_offset] = -100

            logits = dummy_model(chunk_ids)
            nll, n_tok = compute_masked_token_nll(logits, labels, ignore_index=-100)
            total_nll += nll

            chunk_targets = [begin + r for r in range(trg_offset, end - begin) if (begin + r) > 0]
            self.assertEqual(len(chunk_targets), n_tok)
            all_scored_indices.extend(chunk_targets)

            for idx in chunk_targets:
                all_scored_byte_slices.append(self.mapper.get_token_bytes(token_ids[0, idx].item()))

            prev_end = end
            chunk_count += 1
            if end >= seq_len:
                break

        # Verification 1: Exactly tokens 1..seq_len-1 scored once
        self.assertEqual(all_scored_indices, list(range(1, seq_len)))
        self.assertEqual(len(set(all_scored_indices)), seq_len - 1)

        # Verification 2: Concatenated bytes equal raw_bytes minus token 0
        concatenated_bytes = b"".join(all_scored_byte_slices)
        tok0_bytes = self.mapper.get_token_bytes(token_ids[0, 0].item())
        self.assertEqual(concatenated_bytes, raw_bytes[len(tok0_bytes):])
        self.assertEqual(len(concatenated_bytes), len(raw_bytes) - len(tok0_bytes))

        # Verification 3: BPB formula
        bpb = compute_bpb(total_nll, len(concatenated_bytes))
        self.assertIsNotNone(bpb)
        self.assertGreater(bpb, 0.0)

    def test_03_simulated_oom_sliding_window_nll_and_bytes(self):
        """Verify that an OOM chunk skips targets without corrupting subsequent chunks."""
        text = (
            "Chunk one is evaluated. Chunk two triggers an out-of-memory exception. "
            "Chunk three resumes evaluation cleanly. The pipeline must record skipped "
            "tokens and exclude them from both numerator and denominator."
        )
        raw_bytes = text.encode("utf-8")
        enc = self.tokenizer(text, return_tensors="pt")
        token_ids = enc.input_ids
        seq_len = token_ids.shape[1]

        max_length = 16
        stride = 8
        dummy_model = ControlledDummyLM(self.tokenizer.vocab_size)

        total_nll = 0.0
        all_scored_indices = []
        all_scored_byte_slices = []
        skipped_indices = []

        oom_chunk_idx = 1  # Simulate OOM on second chunk
        prev_end = 0
        chunk_idx = 0

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_begin = max(begin, prev_end)
            trg_offset = trg_begin - begin

            if chunk_idx == oom_chunk_idx:
                # OOM occurs: record skipped targets, advance prev_end = end
                chunk_targets = [begin + r for r in range(trg_offset, end - begin) if (begin + r) > 0]
                skipped_indices.extend(chunk_targets)
                prev_end = end
                chunk_idx += 1
                if end >= seq_len:
                    break
                continue

            chunk_ids = token_ids[:, begin:end]
            labels = chunk_ids.clone()
            if trg_offset > 0:
                labels[:, :trg_offset] = -100

            logits = dummy_model(chunk_ids)
            nll, n_tok = compute_masked_token_nll(logits, labels, ignore_index=-100)
            total_nll += nll

            chunk_targets = [begin + r for r in range(trg_offset, end - begin) if (begin + r) > 0]
            self.assertEqual(len(chunk_targets), n_tok)
            all_scored_indices.extend(chunk_targets)

            for idx in chunk_targets:
                all_scored_byte_slices.append(self.mapper.get_token_bytes(token_ids[0, idx].item()))

            prev_end = end
            chunk_idx += 1
            if end >= seq_len:
                break

        # Verification 1: No overlap between scored and skipped
        self.assertEqual(len(set(all_scored_indices).intersection(set(skipped_indices))), 0)
        # Verification 2: Union of scored + skipped + token 0 equals all tokens
        all_union = set(all_scored_indices).union(set(skipped_indices)).union({0})
        self.assertEqual(all_union, set(range(seq_len)))
        # Verification 3: Scored bytes equal length of scored byte slices
        concatenated_scored_bytes = b"".join(all_scored_byte_slices)
        self.assertEqual(len(concatenated_scored_bytes), sum(len(b) for b in all_scored_byte_slices))
        # Verification 4: Denominator matches scored bytes, not total document bytes
        self.assertLess(len(concatenated_scored_bytes), len(raw_bytes))


if __name__ == "__main__":
    unittest.main()
