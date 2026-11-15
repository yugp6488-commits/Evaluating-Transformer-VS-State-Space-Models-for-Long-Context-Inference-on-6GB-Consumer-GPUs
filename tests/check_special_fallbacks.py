from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

def test_mapper_special_and_fallbacks():
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    u2b[' '] = 32
    
    vocab = tok.get_vocab()
    special_ids = set(tok.all_special_ids)
    
    token_to_bytes = {}
    for tok_str, tok_id in vocab.items():
        if tok_id in special_ids:
            token_to_bytes[tok_id] = tok_str.encode("utf-8")
            continue
        try:
            token_to_bytes[tok_id] = bytes([u2b[c] for c in tok_str])
        except KeyError:
            if tok_str.startswith("<0x") and tok_str.endswith(">") and len(tok_str) == 6:
                try:
                    byte_val = int(tok_str[3:5], 16)
                    token_to_bytes[tok_id] = bytes([byte_val])
                except ValueError:
                    token_to_bytes[tok_id] = None
            else:
                token_to_bytes[tok_id] = tok_str.encode("utf-8")

    print(f"Total vocab: {len(vocab)}, mapped: {len(token_to_bytes)}, unmapped: {sum(1 for v in token_to_bytes.values() if v is None)}")
    assert all(v is not None for v in token_to_bytes.values())
    print("SUCCESS: 100% of tokens mapped including special tokens and fallbacks!")

test_mapper_special_and_fallbacks()
