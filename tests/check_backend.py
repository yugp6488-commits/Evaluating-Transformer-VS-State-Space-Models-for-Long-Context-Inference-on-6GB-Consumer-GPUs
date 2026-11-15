from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

# Test decode of single token vs sequence of tokens
# How does HuggingFace tokenizer decode token IDs to bytes?
# In tokenizers (the Rust library underlying fast tokenizers):
# tok.backend_tokenizer has a decoder!
backend = tok.backend_tokenizer
print("Backend decoder:", backend.decoder)

# Test decoding a single token with clean_up_tokenization_spaces=False
tids = [50275, 29454, 50274, 31748, 172, 243, 108, 117]
for tid in tids:
    dec = tok.decode([tid], clean_up_tokenization_spaces=False)
    tok_str = tok.convert_ids_to_tokens(tid)
    print(f"tid={tid:5d}: decode={repr(dec):10s} tok_str={repr(tok_str):10s}")

# What if we decode [172, 243, 108, 117] together?
print("Decoding [172, 243, 108, 117] together:", repr(tok.decode([172, 243, 108, 117])))
