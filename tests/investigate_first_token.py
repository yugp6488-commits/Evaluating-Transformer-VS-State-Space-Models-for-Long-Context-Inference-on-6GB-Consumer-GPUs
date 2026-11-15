from evaluation.evaluate_quality import load_evaluation_samples
from transformers import AutoTokenizer
from evaluation.bits_per_byte import TokenizerByteMapper

print("Loading 1 sample...")
texts, dataset_name, is_synthetic = load_evaluation_samples(1)
text = texts[0]

print(f"Dataset Name: {dataset_name}")
print(f"Is Synthetic: {is_synthetic}")
print(f"Total Chars: {len(text)}")

raw_bytes = text.encode("utf-8")
print(f"Total UTF-8 Bytes: {len(raw_bytes)}")
print(f"First 100 chars: {repr(text[:100])}")
print(f"First 100 bytes: {raw_bytes[:100]}")
has_bom = raw_bytes.startswith(b'\xef\xbb\xbf')
print(f"Starts with UTF-8 BOM: {has_bom}")

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
enc = tok(text, return_tensors="pt", truncation=False)
token_ids = enc.input_ids[0].tolist()

print(f"\nTotal tokens: {len(token_ids)}")
print(f"First 10 token IDs: {token_ids[:10]}")

mapper = TokenizerByteMapper(tok)
print(f"Mapper is_exact: {mapper.is_exact}")

# Let's inspect tokens 0 to 10
print("\nFirst 10 tokens inspection:")
cum_bytes = 0
for i in range(10):
    tid = token_ids[i]
    tok_str = tok.convert_ids_to_tokens(tid)
    tok_bytes = mapper.get_token_bytes(tid)
    b_len = len(tok_bytes) if tok_bytes else 0
    cum_bytes += b_len
    print(f"Token {i:2d}: id={tid:5d} | repr={repr(tok_str):12s} | bytes={tok_bytes} | len={b_len} | cum_bytes={cum_bytes} | single_decode={repr(tok.decode([tid]))}")

# Now examine what happens with the first token vs remaining tokens
print("\n--- Sliding Window Target Calculation ---")
max_length = 512
stride = 256
prev_end = 0

scored_indices = []
scored_byte_pieces = []
for begin in range(0, len(token_ids), stride):
    end = min(begin + max_length, len(token_ids))
    trg_begin = max(begin, prev_end)
    trg_offset = trg_begin - begin

    for r in range(trg_offset, end - begin):
        abs_idx = begin + r
        if abs_idx == 0:
            continue
        scored_indices.append(abs_idx)
        scored_byte_pieces.append(mapper.get_token_bytes(token_ids[abs_idx]))

    prev_end = end
    if end >= len(token_ids):
        break

scored_bytes = sum(len(b) for b in scored_byte_pieces)
print(f"Total scored tokens: {len(scored_indices)}")
print(f"Total scored bytes: {scored_bytes}")
print(f"Difference (total_bytes - scored_bytes): {len(raw_bytes) - scored_bytes}")
print(f"Token 0 bytes: {mapper.get_token_bytes(token_ids[0])} (len={len(mapper.get_token_bytes(token_ids[0]))})")
