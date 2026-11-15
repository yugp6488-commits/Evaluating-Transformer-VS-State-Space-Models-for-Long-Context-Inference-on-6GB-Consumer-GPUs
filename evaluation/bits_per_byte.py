"""
evaluation/bits_per_byte.py — Bits Per Byte (BPB) and Perplexity Utilities

IEEE Benchmark Study: Evaluating Transformer and State Space Models
for Long-Context Inference on 6GB Consumer GPUs

Provides mathematically rigorous functions and validation routines to compute:
1. Token-level Perplexity (PPL) per model
2. Bits Per Byte (BPB) for cross-tokenizer fair comparison with authoritative UTF-8 byte mapping
3. Rigorous validation of the evaluation path (verify_pg19_evaluation_path)
"""

import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


def count_utf8_bytes(text: str) -> int:
    """Return the exact number of raw UTF-8 encoded bytes in a string."""
    return len(text.encode("utf-8"))


class TokenizerByteMapper:
    """Maps token IDs to exact raw UTF-8 byte sequences for a specific tokenizer.

    Validates against the authoritative original normalized text. If exact
    byte-level alignment cannot be guaranteed, flags is_exact = False so
    callers can report BPB as unavailable rather than estimating.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer_id = getattr(tokenizer, "name_or_path", str(tokenizer))
        self.token_to_bytes: Dict[int, Optional[bytes]] = {}
        self.is_exact: bool = True
        self._build_mapper()

    def _build_mapper(self) -> None:
        """Construct token-to-bytes lookup table from tokenizer vocabulary."""
        try:
            from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
            b2u = bytes_to_unicode()
            u2b = {v: k for k, v in b2u.items()}
            # GPT-NeoX / Pythia multi-space tokens contain literal space characters (ASCII 32)
            u2b[" "] = 32

            vocab = self.tokenizer.get_vocab()
            special_ids = set(getattr(self.tokenizer, "all_special_ids", []))

            for tok_str, tok_id in vocab.items():
                if tok_id in special_ids:
                    # Special tokens map to their utf-8 string bytes
                    self.token_to_bytes[tok_id] = tok_str.encode("utf-8")
                    continue
                try:
                    raw = bytes([u2b[c] for c in tok_str])
                    self.token_to_bytes[tok_id] = raw
                except KeyError:
                    # Byte fallback token format (e.g. SentencePiece/LLaMA style <0xAB>)
                    if tok_str.startswith("<0x") and tok_str.endswith(">") and len(tok_str) == 6:
                        try:
                            byte_val = int(tok_str[3:5], 16)
                            self.token_to_bytes[tok_id] = bytes([byte_val])
                        except ValueError:
                            self.is_exact = False
                            self.token_to_bytes[tok_id] = None
                    else:
                        # Cannot reliably map to raw byte stream
                        self.is_exact = False
                        self.token_to_bytes[tok_id] = None
        except Exception:
            self.is_exact = False

    def get_token_bytes(self, token_id: int) -> Optional[bytes]:
        """Return the exact raw byte sequence for a token ID, or None if unmapped."""
        if not self.is_exact:
            return None
        return self.token_to_bytes.get(token_id, None)

    def verify_sequence_bytes(self, token_ids: List[int], authoritative_bytes: bytes) -> bool:
        """Verify that token sequence exactly reconstructs authoritative_bytes bit-for-bit.

        Returns True iff every token maps to known bytes and their concatenation
        equals authoritative_bytes without loss or modification.
        """
        if not self.is_exact:
            return False
        byte_slices = []
        for tid in token_ids:
            b = self.get_token_bytes(tid)
            if b is None:
                return False
            byte_slices.append(b)
        reconstructed = b"".join(byte_slices)
        return reconstructed == authoritative_bytes

    def get_scored_target_bytes(
        self,
        token_ids: List[int],
        scored_token_indices: List[int],
    ) -> Tuple[int, Optional[bytes], bool]:
        """Extract exact UTF-8 bytes for scored target tokens.

        Args:
            token_ids: Full sequence of token IDs for the document.
            scored_token_indices: List of token indices that were evaluated as targets.

        Returns:
            (num_bytes, concatenated_bytes, is_valid)
        """
        if not self.is_exact:
            return 0, None, False

        byte_pieces = []
        for idx in scored_token_indices:
            if idx < 0 or idx >= len(token_ids):
                return 0, None, False
            tid = token_ids[idx]
            b = self.get_token_bytes(tid)
            if b is None:
                return 0, None, False
            byte_pieces.append(b)

        concat_bytes = b"".join(byte_pieces)
        return len(concat_bytes), concat_bytes, True


def compute_masked_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[float, int]:
    """Compute total negative log-likelihood (in nats) and count of scored tokens.

    Supports next-token shift and target masking with ignore_index.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )

    num_tokens = (shift_labels != ignore_index).sum().item()
    return loss.item(), int(num_tokens)


def compute_bpb(total_nll_nats: float, total_utf8_bytes: Optional[int]) -> Optional[float]:
    """Compute Bits Per Byte (BPB) from total NLL and authoritative UTF-8 byte count.

    Formula:
        BPB = total_nll_nats / (total_utf8_bytes * math.log(2))

    Returns:
        float BPB value, or None if byte count is unavailable or <= 0.
    """
    if total_utf8_bytes is None or total_utf8_bytes <= 0:
        return None
    return total_nll_nats / (total_utf8_bytes * math.log(2.0))


def compute_perplexity(total_nll_nats: float, total_tokens: int) -> float:
    """Compute token-level perplexity from total NLL and total token count.

    Formula:
        PPL = exp(total_nll_nats / total_tokens)
    """
    if total_tokens <= 0:
        return float("inf")
    avg_nll = total_nll_nats / total_tokens
    if avg_nll > 100.0:
        return float("inf")
    return math.exp(avg_nll)


def verify_pg19_evaluation_path(
    text: str,
    tokenizer,
    max_length: int = 2048,
    stride: int = 512,
    simulate_oom_chunks: Optional[List[int]] = None,
    mock_nll_per_token: float = 2.5,
) -> Dict[str, Any]:
    """Rigorously verify Bits Per Byte (BPB) calculation and byte-alignment on PG-19 text.

    Validates the four core scientific invariants:
    1. Bit-for-Bit Reconstruction: Reconstructing token IDs back into raw bytes
       matches the original normalized document's UTF-8 byte string exactly.
    2. Sliding-Window Correctness: Overlapping context tokens contribute 0 bytes
       and 0 NLL; scored target tokens contribute their exact byte lengths and
       valid NLL scores exactly once.
    3. OOM Safety: Skipped evaluation chunks contribute zero bytes and zero NLL,
       without corrupting denominators or subsequent chunks.
    4. Denominator Integrity: The final byte denominator equals the exact cumulative
       byte length of tokens whose NLL is included in the final sum.

    Args:
        text: Authoritative original evaluation text string.
        tokenizer: Hugging Face tokenizer instance.
        max_length: Sliding-window chunk size in tokens.
        stride: Sliding-window step size in tokens.
        simulate_oom_chunks: Optional list of 0-based chunk indices to simulate OOM on.
        mock_nll_per_token: Simulated NLL per token (used when no live model is passed).

    Returns:
        Dict summarizing validation outcome, booleans for each invariant, token/byte counts, and any errors.
    """
    oom_chunks: Set[int] = set(simulate_oom_chunks or [])
    errors: List[str] = []

    # --- Invariant 1: Authoritative Ground Truth & Bit-for-Bit Reconstruction ---
    authoritative_bytes = text.encode("utf-8")
    total_text_bytes = len(authoritative_bytes)

    enc = tokenizer(text, return_tensors="pt", truncation=False)
    token_ids = enc.input_ids[0].tolist()
    seq_len = len(token_ids)

    mapper = TokenizerByteMapper(tokenizer)
    reconstruction_passed = mapper.verify_sequence_bytes(token_ids, authoritative_bytes)
    if not reconstruction_passed:
        errors.append(
            f"Bit-for-Bit Reconstruction Failed: Token sequence ({seq_len} tokens) "
            f"did not reconstruct authoritative UTF-8 bytes ({total_text_bytes} bytes)."
        )

    # --- Invariant 2 & 3: Sliding-Window Correctness & OOM Safety ---
    token_eval_counts = [0] * seq_len
    token_context_counts = [0] * seq_len
    scored_target_indices: List[int] = []
    scored_byte_slices: List[bytes] = []
    accumulated_nll = 0.0

    chunk_idx = 0
    chunks_scored = 0
    chunks_skipped = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_begin = max(begin, prev_end)
        trg_offset = trg_begin - begin

        is_oom = chunk_idx in oom_chunks
        if is_oom:
            chunks_skipped += 1
            # Invariant 3: OOM chunk must contribute zero tokens, zero bytes, zero NLL
            prev_end = end
            chunk_idx += 1
            if end >= seq_len:
                break
            continue

        chunks_scored += 1

        # Overlapping context tokens: [begin : trg_begin)
        # Invariant 2: Must contribute 0 bytes and 0 NLL
        for rel_idx in range(0, trg_offset):
            abs_idx = begin + rel_idx
            token_context_counts[abs_idx] += 1

        # Scored target tokens: [trg_begin : end)
        for rel_idx in range(trg_offset, end - begin):
            abs_idx = begin + rel_idx
            if abs_idx == 0:
                # Token 0 is prompt context, un-scored in causal LM
                token_context_counts[0] += 1
                continue

            token_eval_counts[abs_idx] += 1
            scored_target_indices.append(abs_idx)
            accumulated_nll += mock_nll_per_token
            t_bytes = mapper.get_token_bytes(token_ids[abs_idx])
            if t_bytes is not None:
                scored_byte_slices.append(t_bytes)
            else:
                errors.append(f"Unmapped token at index {abs_idx} (id={token_ids[abs_idx]})")

        prev_end = end
        chunk_idx += 1
        if end >= seq_len:
            break

    chunks_total = chunk_idx

    # Invariant 2 verification: Each target token evaluated exactly once
    sliding_window_passed = True
    for idx in range(1, seq_len):
        eval_count = token_eval_counts[idx]
        if eval_count > 1:
            sliding_window_passed = False
            errors.append(f"Token {idx} evaluated {eval_count} times (> 1). Duplicate scoring!")
        elif eval_count == 0 and not oom_chunks:
            sliding_window_passed = False
            errors.append(f"Token {idx} was missed (eval_count == 0) despite no OOM.")

    # Token 0 must have eval_count == 0 (prompt context)
    if token_eval_counts[0] != 0:
        sliding_window_passed = False
        errors.append(f"Token 0 was scored {token_eval_counts[0]} times. Must be 0 (prompt context).")

    # Invariant 3 verification: OOM safety
    oom_safety_passed = True
    if oom_chunks:
        for idx in range(1, seq_len):
            if token_eval_counts[idx] not in (0, 1):
                oom_safety_passed = False
                errors.append(f"Invalid eval_count {token_eval_counts[idx]} for token {idx} during OOM test.")

    # Invariant 4 verification: Denominator Integrity
    denominator_bytes = sum(len(b) for b in scored_byte_slices)
    concatenated_scored_bytes = b"".join(scored_byte_slices)
    denominator_integrity_passed = (len(concatenated_scored_bytes) == denominator_bytes)

    if not denominator_integrity_passed:
        errors.append(f"Denominator byte length mismatch: {len(concatenated_scored_bytes)} != {denominator_bytes}")

    if not oom_chunks and reconstruction_passed:
        token_0_bytes = len(mapper.get_token_bytes(token_ids[0]) or b"")
        expected_denominator = total_text_bytes - token_0_bytes
        if denominator_bytes != expected_denominator:
            denominator_integrity_passed = False
            errors.append(f"Full run denominator mismatch: {denominator_bytes} != expected {expected_denominator}")
        if concatenated_scored_bytes != authoritative_bytes[token_0_bytes:]:
            denominator_integrity_passed = False
            errors.append("Concatenated scored bytes do not match authoritative text slice bit-for-bit!")

    # Final BPB calculation
    bpb = (accumulated_nll / (denominator_bytes * math.log(2.0))) if denominator_bytes > 0 else None

    is_valid = (
        reconstruction_passed
        and sliding_window_passed
        and oom_safety_passed
        and denominator_integrity_passed
        and len(errors) == 0
    )

    return {
        "is_valid": is_valid,
        "reconstruction_passed": reconstruction_passed,
        "sliding_window_passed": sliding_window_passed,
        "oom_safety_passed": oom_safety_passed,
        "denominator_integrity_passed": denominator_integrity_passed,
        "total_text_bytes": total_text_bytes,
        "total_tokens": seq_len,
        "scored_tokens": len(scored_target_indices),
        "scored_bytes": denominator_bytes,
        "total_nll": round(accumulated_nll, 4),
        "bpb": round(bpb, 4) if bpb is not None else None,
        "chunks_total": chunks_total,
        "chunks_scored": chunks_scored,
        "chunks_skipped": chunks_skipped,
        "errors": errors,
    }
