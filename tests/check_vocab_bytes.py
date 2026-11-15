from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

# Test how to get raw bytes for any token:
# Method A: tok.decode([token_id])? We saw that for partial bytes it produces replacement char .
# Method B: tok.convert_ids_to_tokens(token_id)?
# Let's inspect the entire vocab of pythia-410m and gpt-neox-20b!
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}

vocab = tok.get_vocab()
print(f"Vocab size: {len(vocab)}")

unmapped = []
mapped_count = 0
for tok_str, tok_id in vocab.items():
    # Can we convert tok_str to raw bytes?
    # Some tokens are special, e.g. <|endoftext|>, or multi-space tokens '   '
    # Let's see:
    try:
        raw = bytes([u2b[c] for c in tok_str])
        mapped_count += 1
    except KeyError as e:
        unmapped.append((tok_str, tok_id, str(e)))

print(f"Mapped directly via u2b: {mapped_count}")
print(f"Unmapped count: {len(unmapped)}")
print(f"Sample unmapped (first 20): {unmapped[:20]}")
