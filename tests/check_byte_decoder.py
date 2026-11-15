from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

# Test with tokens 22692, 237, 211
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

for tid in [22692, 237, 211]:
    tok_str = tok.convert_ids_to_tokens(tid)
    raw_bytes = bytes([byte_decoder[c] for c in tok_str])
    print(f"tid={tid}, tok_str={repr(tok_str)}, raw_bytes={raw_bytes}, n_bytes={len(raw_bytes)}")

# Combine them!
all_raw = b"".join(bytes([byte_decoder[c] for c in tok.convert_ids_to_tokens(tid)]) for tid in [22692, 237, 211])
print("Combined raw bytes:", all_raw)
print("Decoded text:", all_raw.decode("utf-8"))
print("Matches original ' 🚀'?", all_raw == " 🚀".encode("utf-8"))
