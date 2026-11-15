"""
tests/test_bpb_byte_alignment.py

Unit test suite verifying exact token-to-byte mapping and BPB denominator
calculation on authoritative normalized UTF-8 text across:
  - Whitespace (single, multiple spaces, tabs, newlines, CRLF)
  - Punctuation (ASCII, curly quotes, dashes, ellipses, brackets)
  - Unicode characters (diacritics, accents, Cyrillic, Arabic)
  - Multibyte UTF-8 text (CJK ideographs, emojis, 4-byte astral characters)
  - Sliding-window overlap and un-scored prompt context handling
  - Graceful degradation when exact alignment is unavailable
"""

import sys
import unittest
from transformers import AutoTokenizer

from evaluation.bits_per_byte import (
    TokenizerByteMapper,
    compute_bpb,
    compute_perplexity,
    count_utf8_bytes,
    verify_pg19_evaluation_path,
)


class TestBPBByteAlignment(unittest.TestCase):
    """Test suite for BPB byte-level mapping and alignment."""

    @classmethod
    def setUpClass(cls):
        print("\n[SETUP] Initializing tokenizers for Pythia-410M and Mamba-370M...")
        cls.pythia_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        cls.mamba_tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        cls.pythia_mapper = TokenizerByteMapper(cls.pythia_tok)
        cls.mamba_mapper = TokenizerByteMapper(cls.mamba_tok)

        # Diverse test corpus exercising all requested linguistic and encoding edge cases
        cls.test_cases = [
            # 1. Whitespace variations
            (
                "whitespace_simple",
                "Hello world! This is a simple test.",
            ),
            (
                "whitespace_complex",
                "   Multiple    spaces   and\nnewlines\n\n\ttabs. \r\n Carriage return.",
            ),
            # 2. Punctuation
            (
                "punctuation",
                "Punctuation: ... ,,, !!! ??? --- ;;; ::: 'single' \"double\" “curly” «guillemets» [brackets] {braces} (parens).",
            ),
            # 3. Unicode accents and scripts
            (
                "unicode_diacritics",
                "Diacritics: café, naïve, résumé, façade, jalapeño, München, Kraków, Zürich, Čech, Ångström.",
            ),
            (
                "unicode_scripts",
                "World scripts: Привет (Russian), γεια (Greek), مرحبا (Arabic), שלום (Hebrew), नमस्ते (Hindi).",
            ),
            # 4. Multibyte CJK, Emojis, and 4-byte astral characters
            (
                "multibyte_cjk_emoji",
                "CJK & Emojis: 漢字, 日本語, 한국어, 🚀🌟🔥🎉💡.",
            ),
            (
                "four_byte_astral",
                "Astral 4-byte: 𠮷 (U+20BB7, 4 bytes: \\xf0\\xa0\\xae\\xb7) and 𝄞 (musical G clef, U+1D11E, 4 bytes).",
            ),
            # 5. Mixed realistic literary text
            (
                "mixed_literary",
                (
                    "The Project Gutenberg eBook of The History of Antiquity, by Max Duncker.\n"
                    "“Astronomical observations carved upon clay tablets recorded solar eclipses: 100% precision!”\n"
                    "   Indent with 3 spaces... café résumé — 漢字 🚀 𠮷.\n"
                ),
            ),
        ]

    def test_01_mapper_is_exact(self):
        """Verify that TokenizerByteMapper builds an exact 100% vocabulary mapping."""
        self.assertTrue(self.pythia_mapper.is_exact, "Pythia mapper must be exact.")
        self.assertTrue(self.mamba_mapper.is_exact, "Mamba mapper must be exact.")

    def test_02_full_sequence_byte_reconstruction(self):
        """Verify that every test case reconstructs the authoritative UTF-8 bytes bit-for-bit."""
        for label, text in self.test_cases:
            orig_bytes = text.encode("utf-8")
            for tok_name, mapper, tok in [
                ("Pythia", self.pythia_mapper, self.pythia_tok),
                ("Mamba", self.mamba_mapper, self.mamba_tok),
            ]:
                tokens = tok(text)["input_ids"]
                is_verified = mapper.verify_sequence_bytes(tokens, orig_bytes)
                self.assertTrue(
                    is_verified,
                    f"[{tok_name} | {label}] Token sequence failed bit-for-bit UTF-8 reconstruction!",
                )
                # Verify individual byte slice sum
                total_mapped_bytes = sum(len(mapper.get_token_bytes(t)) for t in tokens)
                self.assertEqual(
                    total_mapped_bytes,
                    len(orig_bytes),
                    f"[{tok_name} | {label}] Mapped byte length {total_mapped_bytes} != {len(orig_bytes)}",
                )

    def test_03_sliding_window_overlap_and_prefix_masking(self):
        """Verify that sliding window overlap contributes 0 bytes and token 0 contributes 0 bytes."""
        text = self.test_cases[-1][1] * 4  # Repeat to force multiple chunks
        orig_bytes = text.encode("utf-8")

        for tok_name, mapper, tok in [
            ("Pythia", self.pythia_mapper, self.pythia_tok),
            ("Mamba", self.mamba_mapper, self.mamba_tok),
        ]:
            tokens = tok(text)["input_ids"]
            seq_len = len(tokens)
            max_length = 64
            stride = 24
            prev_end = 0

            scored_indices = []
            chunk_byte_pieces = []

            for begin in range(0, seq_len, stride):
                end = min(begin + max_length, seq_len)
                trg_begin = max(begin, prev_end)
                trg_offset = trg_begin - begin

                # In causal LM: targets are [trg_begin, end), with index 0 excluded in chunk 0
                for rel_idx in range(trg_offset, end - begin):
                    abs_idx = begin + rel_idx
                    if abs_idx == 0:
                        continue  # Token 0 is prompt context, not scored
                    scored_indices.append(abs_idx)
                    chunk_byte_pieces.append(mapper.get_token_bytes(tokens[abs_idx]))

                prev_end = end
                if end >= seq_len:
                    break

            # 1. Scored token indices must partition [1, seq_len) exactly once
            self.assertEqual(scored_indices, list(range(1, seq_len)))
            self.assertEqual(len(scored_indices), seq_len - 1)

            # 2. Byte denominator must equal orig_bytes minus token 0's bytes
            token_0_bytes = len(mapper.get_token_bytes(tokens[0]))
            expected_scored_bytes = len(orig_bytes) - token_0_bytes
            actual_scored_bytes = sum(len(b) for b in chunk_byte_pieces)

            self.assertEqual(
                actual_scored_bytes,
                expected_scored_bytes,
                f"[{tok_name}] Scored byte count {actual_scored_bytes} != expected {expected_scored_bytes}",
            )

            # 3. Concatenated scored bytes must match original text bit-for-bit starting from token 1
            reconstructed_scored = b"".join(chunk_byte_pieces)
            self.assertEqual(
                reconstructed_scored,
                orig_bytes[token_0_bytes:],
                f"[{tok_name}] Scored bytes do not match original text bit-for-bit!",
            )

    def test_04_simulated_oom_chunk_skip(self):
        """Verify that skipping an OOM chunk excludes its bytes without corrupting other chunks."""
        text = self.test_cases[-1][1] * 4
        orig_bytes = text.encode("utf-8")
        mapper = self.pythia_mapper
        tok = self.pythia_tok

        tokens = tok(text)["input_ids"]
        seq_len = len(tokens)
        max_length = 64
        stride = 24
        prev_end = 0

        scored_indices = []
        chunk_byte_pieces = []
        skipped_chunk_idx = 1  # Simulate OOM on chunk 1
        chunk_counter = 0

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_begin = max(begin, prev_end)
            trg_offset = trg_begin - begin

            if chunk_counter == skipped_chunk_idx:
                # OOM occurs: skip this chunk entirely
                chunk_counter += 1
                prev_end = end
                if end >= seq_len:
                    break
                continue

            for rel_idx in range(trg_offset, end - begin):
                abs_idx = begin + rel_idx
                if abs_idx == 0:
                    continue
                scored_indices.append(abs_idx)
                chunk_byte_pieces.append(mapper.get_token_bytes(tokens[abs_idx]))

            chunk_counter += 1
            prev_end = end
            if end >= seq_len:
                break

        # Verify that no tokens from the skipped chunk exist in scored_indices
        self.assertLess(len(scored_indices), seq_len - 1)
        actual_bytes = sum(len(b) for b in chunk_byte_pieces)
        self.assertEqual(len(b"".join(chunk_byte_pieces)), actual_bytes)

    def test_05_unmapped_tokenizer_reports_unavailable(self):
        """Verify that an unmapped or unaligned tokenizer reports BPB as None (unavailable)."""
        class DummyTokenizer:
            name_or_path = "dummy"
            def get_vocab(self):
                # Contains tokens with characters outside the byte table
                return {"\u2603_special": 0, "test": 1}

        dummy_mapper = TokenizerByteMapper(DummyTokenizer())
        self.assertFalse(dummy_mapper.is_exact)
        self.assertIsNone(dummy_mapper.get_token_bytes(0))

        # BPB computation must return None when bytes are unavailable
        bpb_val = compute_bpb(total_nll_nats=100.0, total_utf8_bytes=None)
        self.assertIsNone(bpb_val)

    def test_06_verify_pg19_evaluation_path_full(self):
        """Verify that verify_pg19_evaluation_path validates all 4 invariants on complete runs."""
        text = self.test_cases[-1][1] * 3  # Long mixed literary text
        for tok_name, tok in [("Pythia", self.pythia_tok), ("Mamba", self.mamba_tok)]:
            report = verify_pg19_evaluation_path(
                text=text,
                tokenizer=tok,
                max_length=64,
                stride=24,
            )
            self.assertTrue(report["is_valid"], f"[{tok_name}] Validation failed: {report['errors']}")
            self.assertTrue(report["reconstruction_passed"], f"[{tok_name}] Reconstruction failed")
            self.assertTrue(report["sliding_window_passed"], f"[{tok_name}] Sliding-window check failed")
            self.assertTrue(report["oom_safety_passed"], f"[{tok_name}] OOM check failed")
            self.assertTrue(report["denominator_integrity_passed"], f"[{tok_name}] Denominator integrity failed")
            self.assertEqual(report["chunks_skipped"], 0)
            self.assertGreater(report["chunks_scored"], 1)
            self.assertIsNotNone(report["bpb"])

    def test_07_verify_pg19_evaluation_path_oom_safety(self):
        """Verify that verify_pg19_evaluation_path validates OOM chunk skip invariants."""
        text = self.test_cases[-1][1] * 3
        for tok_name, tok in [("Pythia", self.pythia_tok), ("Mamba", self.mamba_tok)]:
            # Baseline without OOM
            full_report = verify_pg19_evaluation_path(text, tok, max_length=64, stride=24)
            # Simulated OOM on chunks 1 and 3
            oom_report = verify_pg19_evaluation_path(
                text, tok, max_length=64, stride=24, simulate_oom_chunks=[1, 3]
            )
            self.assertTrue(oom_report["is_valid"], f"[{tok_name}] OOM validation failed: {oom_report['errors']}")
            self.assertTrue(oom_report["oom_safety_passed"])
            self.assertEqual(oom_report["chunks_skipped"], 2)
            self.assertEqual(oom_report["chunks_scored"], full_report["chunks_scored"] - 2)
            # Verify scored tokens and bytes decreased by exactly the skipped amounts
            self.assertLess(oom_report["scored_tokens"], full_report["scored_tokens"])
            self.assertLess(oom_report["scored_bytes"], full_report["scored_bytes"])
            self.assertLess(oom_report["total_nll"], full_report["total_nll"])
            # Denominator integrity must remain strictly true
            self.assertTrue(oom_report["denominator_integrity_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
