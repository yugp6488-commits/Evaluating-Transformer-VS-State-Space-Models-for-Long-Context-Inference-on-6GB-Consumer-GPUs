import math
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

# Build byte decoder
b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}
u2b[' '] = 32

def build_tokenizer_byte_mapper(tokenizer):
    """Build and validate a mapping from token IDs to exact UTF-8 byte sequences.
    
    Returns (mapper_fn, is_exact):
        mapper_fn(token_id) -> bytes
        is_exact: True if every token in vocab maps to exact bytes, False otherwise.
    """
    vocab = tokenizer.get_vocab()
    token_to_bytes = {}
    is_exact = True
    
    for tok_str, tok_id in vocab.items():
        try:
            raw = bytes([u2b[c] for c in tok_str])
            token_to_bytes[tok_id] = raw
        except KeyError:
            is_exact = False
            token_to_bytes[tok_id] = None
            
    if not is_exact:
        return None, False
        
    def get_token_bytes(token_id: int) -> bytes:
        return token_to_bytes.get(token_id, b"")
        
    return get_token_bytes, True

# Test with a mock sliding window on a full text
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
get_bytes, is_exact = build_tokenizer_byte_mapper(tok)
print("Pythia tokenizer is_exact byte mapper:", is_exact)

sample_text = (
    "The Project Gutenberg eBook of The History of Antiquity, by Max Duncker.\n"
    "This is a test: é, ñ, 漢字, 🚀!\n"
    "   Multiple    spaces   and\nnewlines\n\n\ttabs.    \n"
    "Rare: 𠮷 (U+20BB7).\n"
) * 5

orig_bytes = sample_text.encode("utf-8")
enc = tok(sample_text, return_tensors=None)
token_ids = enc["input_ids"]
seq_len = len(token_ids)
print(f"Sample text length: {len(sample_text)} chars, {len(orig_bytes)} bytes, {seq_len} tokens")

# Full token byte verification against original text:
all_token_bytes = [get_bytes(t) for t in token_ids]
full_reconstructed = b"".join(all_token_bytes)
assert full_reconstructed == orig_bytes, "Full byte reconstruction mismatch!"
print("Full byte reconstruction verified: EXACT MATCH!")

# Now simulate sliding window chunking with overlap (max_length=50, stride=20)
max_length = 50
stride = 20
prev_end = 0

scored_tokens = []
scored_bytes_list = []

for begin in range(0, seq_len, stride):
    end = min(begin + max_length, seq_len)
    trg_begin = max(begin, prev_end)
    chunk_tokens = token_ids[begin:end]
    trg_offset = trg_begin - begin
    
    # In causal LM:
    # chunk target tokens are at indices [trg_offset, len(chunk_tokens))
    # except token 0 in chunk 0 (which has no predecessor)
    chunk_target_indices = []
    chunk_bytes = []
    for rel_idx in range(trg_offset, len(chunk_tokens)):
        abs_idx = begin + rel_idx
        if abs_idx == 0:
            continue  # Token 0 is prompt context, un-scored
        chunk_target_indices.append(abs_idx)
        t_bytes = get_bytes(chunk_tokens[rel_idx])
        chunk_bytes.append(t_bytes)
        
    scored_tokens.extend(chunk_target_indices)
    scored_bytes_list.extend(chunk_bytes)
    prev_end = end
    if end >= seq_len:
        break

print(f"Total scored tokens: {len(scored_tokens)} (expected: {seq_len - 1})")
assert len(scored_tokens) == seq_len - 1
assert scored_tokens == list(range(1, seq_len))

total_scored_bytes = sum(len(b) for b in scored_bytes_list)
token_0_bytes = len(get_bytes(token_ids[0]))
expected_scored_bytes = len(orig_bytes) - token_0_bytes

print(f"Total scored bytes: {total_scored_bytes} (expected: {expected_scored_bytes}, orig: {len(orig_bytes)})")
assert total_scored_bytes == expected_scored_bytes
assert b"".join(scored_bytes_list) == orig_bytes[token_0_bytes:]
print("SLIDING WINDOW BYTE ALIGNMENT: 100% MATHEMATICALLY AND BIT-EXACT!")
