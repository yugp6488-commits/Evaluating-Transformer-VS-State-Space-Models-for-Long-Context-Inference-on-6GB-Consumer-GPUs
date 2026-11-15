from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}
u2b[' '] = 32

vocab = tok.get_vocab()
still_unmapped = []
for tok_str, tok_id in vocab.items():
    try:
        raw = bytes([u2b[c] for c in tok_str])
    except KeyError as e:
        still_unmapped.append((tok_str, tok_id, str(e)))

print("Mamba tokenizer still unmapped count:", len(still_unmapped))
print("Still unmapped:", still_unmapped)
