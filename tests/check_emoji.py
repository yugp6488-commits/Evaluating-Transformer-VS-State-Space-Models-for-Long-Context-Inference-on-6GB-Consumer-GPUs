from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

text = " 🚀"
enc = tok(text, return_offsets_mapping=True)
print("text:", repr(text))
print("tokens:", enc.input_ids)
print("offsets:", enc.offset_mapping)
for tid in enc.input_ids:
    print(f"id={tid}, repr={repr(tok.decode([tid]))}, convert_ids_to_tokens={repr(tok.convert_ids_to_tokens(tid))}")

full_text = "The Project Gutenberg eBook of The History of Antiquity, by Max Duncker.\nThis is a test: é, ñ, 漢字, 🚀!"
enc_full = tok(full_text, return_offsets_mapping=True)
for i in range(34, len(enc_full.input_ids)):
    tid = enc_full.input_ids[i]
    off = enc_full.offset_mapping[i]
    sub = full_text[off[0]:off[1]]
    print(f"i={i}, tid={tid}, off={off}, sub={repr(sub)}, dec={repr(tok.decode([tid]))}, tok={repr(tok.convert_ids_to_tokens(tid))}")
