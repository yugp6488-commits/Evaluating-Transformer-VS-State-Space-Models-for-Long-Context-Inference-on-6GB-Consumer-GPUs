from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}
# Space character (0x20) in b2u is mapped to 'Ġ'
# But in NeoX multi-space tokens, space is literally ' '
# Let's map ' ' to 0x20!
u2b[' '] = 32

vocab = tok.get_vocab()
still_unmapped = []
for tok_str, tok_id in vocab.items():
    try:
        raw = bytes([u2b[c] for c in tok_str])
    except KeyError as e:
        still_unmapped.append((tok_str, tok_id, str(e)))

print("Still unmapped count:", len(still_unmapped))
print("Still unmapped:", still_unmapped)
