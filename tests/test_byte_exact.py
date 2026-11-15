from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}
u2b[' '] = 32

def token_to_raw_bytes(tok, token_id):
    tok_str = tok.convert_ids_to_tokens(token_id)
    return bytes([u2b[c] for c in tok_str])

test_cases = [
    "Hello world! This is a simple test.",
    "   Multiple    spaces   and\nnewlines\n\n\ttabs.    ",
    "Punctuation: ... ,,, !!! ??? --- ;;; ::: 'quotes' \"double\" (parens) [brackets] {braces}",
    "Unicode: café, naïve, résumé, façade, jalapeño, München, Kraków, Zürich.",
    "Multibyte & Emojis: 🚀🌟🔥🎉 and 漢字, 日本語, 한국어, العربية, עברית.",
    "Rare 4-byte: 𠮷 (CJK Extension B, U+20BB7, 4 bytes: \\xf0\\xa0\\xae\\xb7)",
    "Mixed edge case: \n  Hello 🚀 world! 漢字 café... \t 12345! \n  𠮷 end.  \n"
]

for tid in ["EleutherAI/pythia-410m", "EleutherAI/gpt-neox-20b"]:
    tok = AutoTokenizer.from_pretrained(tid)
    print(f"\n==================== Testing {tid} ====================")
    for text in test_cases:
        orig_bytes = text.encode("utf-8")
        tokens = tok(text)["input_ids"]
        
        # Map every token to its exact byte span
        byte_slices = [token_to_raw_bytes(tok, t) for t in tokens]
        reconstructed = b"".join(byte_slices)
        
        assert reconstructed == orig_bytes, f"FAILED on {repr(text[:30])}"
        print(f"  PASS: {len(orig_bytes):3d} bytes == {sum(len(b) for b in byte_slices):3d} bytes ({len(tokens):2d} tokens) | text={repr(text[:35])}...")
