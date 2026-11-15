from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

def map_tokens_to_bytes(text, tok):
    """Map each token to its byte span in text.encode('utf-8')."""
    orig_bytes = text.encode("utf-8")
    enc = tok(text, return_offsets_mapping=True)
    tokens = enc.input_ids
    offsets = enc.offset_mapping

    # Build char-to-byte index mapping
    char_to_byte = [0] * (len(text) + 1)
    b_idx = 0
    for c_idx, ch in enumerate(text):
        char_to_byte[c_idx] = b_idx
        b_idx += len(ch.encode("utf-8"))
    char_to_byte[len(text)] = b_idx

    # Convert char offsets to byte spans
    # Note: When multiple tokens cover the same character (e.g. multibyte UTF-8 split),
    # we need each byte in orig_bytes to be owned by exactly one token.
    token_byte_spans = []
    for s, e in offsets:
        token_byte_spans.append((char_to_byte[s], char_to_byte[e]))

    return orig_bytes, tokens, token_byte_spans, offsets

test_texts = [
    "The Project Gutenberg eBook of The History of Antiquity, by Max Duncker.\nThis is a test: é, ñ, 漢字, 🚀!",
    "Testing rare: 𠮷 (CJK Unified Ideographs Extension B, U+20BB7, 4 bytes)",
    "   Multiple    spaces   and\nnewlines\n\n\ttabs."
]

for t in test_texts:
    orig_bytes, tokens, spans, offsets = map_tokens_to_bytes(t, tok)
    print(f"\nText: {repr(t[:40])} (total bytes: {len(orig_bytes)}, tokens: {len(tokens)})")
    # Let's inspect spans
    for i, (tok_id, (bs, be), (cs, ce)) in enumerate(zip(tokens, spans, offsets)):
        print(f"  tok {i:2d} ({tok_id:5d}): char=({cs:2d},{ce:2d}) byte=({bs:2d},{be:2d}) text={repr(t[cs:ce])}")
