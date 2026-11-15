from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
text = "   Multiple    spaces   and\nnewlines\n\n\ttabs."
enc = tok(text)
for tid in enc.input_ids:
    s = tok.convert_ids_to_tokens(tid)
    print(f"tid={tid}, repr={repr(s)}")
