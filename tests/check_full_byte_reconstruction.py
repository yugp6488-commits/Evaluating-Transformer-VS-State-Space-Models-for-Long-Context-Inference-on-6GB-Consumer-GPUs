from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

for tid in ["EleutherAI/pythia-410m", "EleutherAI/gpt-neox-20b"]:
    tok = AutoTokenizer.from_pretrained(tid)
    test_texts = [
        "The Project Gutenberg eBook of The History of Antiquity, by Max Duncker.\nThis is a test: é, ñ, 漢字, 🚀!",
        "   Multiple    spaces   and\nnewlines\n\n\ttabs.",
        "Unicode: café, naïve, résumé, façade, jalapeño.",
        "Multibyte & Emojis: 🚀🌟🔥🎉 and 漢字, 日本語, 한국어, العربية.",
        "Testing rare: 𠮷 (CJK Unified Ideographs Extension B, U+20BB7, 4 bytes)"
    ]
    
    for text in test_texts:
        orig_bytes = text.encode("utf-8")
        enc = tok(text, return_tensors=None)
        tokens = enc["input_ids"]
        
        # Reconstruct byte stream token by token
        token_byte_slices = []
        for t in tokens:
            t_str = tok.convert_ids_to_tokens(t)
            # Map each char in t_str back to its raw byte
            t_bytes = bytes([byte_decoder[c] for c in t_str])
            token_byte_slices.append(t_bytes)
            
        reconstructed_bytes = b"".join(token_byte_slices)
        assert reconstructed_bytes == orig_bytes, f"Mismatch on {text[:20]}!"
        print(f"[{tid.split('/')[-1]}] PERFECT MATCH: {len(orig_bytes)} bytes == {sum(len(b) for b in token_byte_slices)} bytes ({len(tokens)} tokens)")
