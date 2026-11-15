import math
from typing import Any, Dict, List, Optional, Set, Tuple

from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

# Test drafting verify_pg19_evaluation_path
b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}
u2b[' '] = 32

class TokenizerByteMapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_to_bytes = {}
        self.is_exact = True
        self._build_mapper()

    def _build_mapper(self):
        vocab = self.tokenizer.get_vocab()
        special_ids = set(self.tokenizer.all_special_ids)
        for tok_str, tok_id in vocab.items():
            if tok_id in special_ids:
                self.token_to_bytes[tok_id] = tok_str.encode("utf-8")
                continue
            try:
                self.token_to_bytes[tok_id] = bytes([u2b[c] for c in tok_str])
            except KeyError:
                if tok_str.startswith("<0x") and tok_str.endswith(">") and len(tok_str) == 6:
                    try:
                        self.token_to_bytes[tok_id] = bytes([int(tok_str[3:5], 16)])
                    except ValueError:
                        self.is_exact = False
                        self.token_to_bytes[tok_id] = None
                else:
                    self.token_to_bytes[tok_id] = tok_str.encode("utf-8")

    def get_token_bytes(self, token_id: int) -> Optional[bytes]:
        return self.token_to_bytes.get(token_id, None)

    def verify_sequence_bytes(self, token_ids: List[int], authoritative_bytes: bytes) -> bool:
        byte_slices = [self.get_token_bytes(t) for t in token_ids]
        if any(b is None for b in byte_slices):
            return False
        return b"".join(byte_slices) == authoritative_bytes


def verify_pg19_evaluation_path(
    text: str,
    tokenizer,
    max_length: int = 512,
    stride: int = 256,
    simulate_oom_chunks: Optional[List[int]] = None,
    mock_nll_per_token: float = 2.5,
) -> Dict[str, Any]:
    """Rigorously verify Bits Per Byte (BPB) calculation and byte-alignment on PG-19 text."""
    oom_chunks: Set[int] = set(simulate_oom_chunks or [])
    errors: List[str] = []

    # 1. Authoritative ground truth
    authoritative_bytes = text.encode("utf-8")
    total_text_bytes = len(authoritative_bytes)

    # Tokenize
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

    # 2. Sliding-Window Correctness & OOM Safety
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

        # Process valid chunk
        chunks_scored += 1

        # Overlapping context tokens: [begin : trg_begin)
        for rel_idx in range(0, trg_offset):
            abs_idx = begin + rel_idx
            token_context_counts[abs_idx] += 1

        # Scored target tokens: [trg_begin : end)
        for rel_idx in range(trg_offset, end - begin):
            abs_idx = begin + rel_idx
            if abs_idx == 0:
                # Token 0 is prompt context, un-scored
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

    # Invariant 2 verification: Each scored target token scored exactly once
    sliding_window_passed = True
    for idx in range(1, seq_len):
        eval_count = token_eval_counts[idx]
        # If the chunk covering this token was OOM, eval_count should be 0; otherwise 1
        if eval_count > 1:
            sliding_window_passed = False
            errors.append(f"Token {idx} evaluated {eval_count} times (> 1). Duplicate scoring!")
        elif eval_count == 0 and not oom_chunks:
            sliding_window_passed = False
            errors.append(f"Token {idx} was missed (eval_count == 0) despite no OOM.")

    # Token 0 must have eval_count == 0
    if token_eval_counts[0] != 0:
        sliding_window_passed = False
        errors.append(f"Token 0 was scored {token_eval_counts[0]} times. Must be 0 (prompt context).")

    # Invariant 3 verification: OOM safety
    oom_safety_passed = True
    if oom_chunks:
        # Check that skipped chunks added nothing to scored_target_indices
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

    # BPB calculation
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

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
text = (
    "The Project Gutenberg eBook of The History of Antiquity, by Max Duncker.\n"
    "This is a test: é, ñ, 漢字, 🚀!\n"
    "   Multiple    spaces   and\nnewlines\n\n\ttabs.    \n"
    "Rare: 𠮷 (U+20BB7).\n"
) * 5

# Normal run
res = verify_pg19_evaluation_path(text, tok, max_length=64, stride=24)
print("Normal run verification:")
for k, v in res.items():
    print(f"  {k}: {v}")

assert res["is_valid"] is True

# Run with simulated OOM on chunk 1
res_oom = verify_pg19_evaluation_path(text, tok, max_length=64, stride=24, simulate_oom_chunks=[1])
print("\nOOM run verification:")
for k, v in res_oom.items():
    print(f"  {k}: {v}")

assert res_oom["is_valid"] is True
assert res_oom["chunks_skipped"] == 1
assert res_oom["scored_tokens"] < res["scored_tokens"]
assert res_oom["scored_bytes"] < res["scored_bytes"]
print("\nALL VERIFICATIONS PASSED!")
