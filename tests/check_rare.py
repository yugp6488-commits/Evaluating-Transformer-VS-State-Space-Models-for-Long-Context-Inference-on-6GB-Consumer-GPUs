from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

# Test an unusual character or byte sequence
text = "Testing rare: 𠮷 (CJK Unified Ideographs Extension B, U+20BB7, 4 bytes)"
enc = tok(text, return_offsets_mapping=True)

print("Tokens:", enc.input_ids)
print("Offsets:", enc.offset_mapping)
for i, (tok_id, (s, e)) in enumerate(zip(enc.input_ids, enc.offset_mapping)):
    tok_str = tok.decode([tok_id])
    span_str = text[s:e]
    print(f"Token {i}: id={tok_id}, slice={repr(span_str)}, decoded={repr(tok_str)}, span=({s}, {e})")
