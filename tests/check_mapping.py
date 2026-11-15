from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

text = "The Project Gutenberg eBook of The History of Antiquity, by Max Duncker.\nThis is a test: é, ñ, 漢字, 🚀!"
enc = tok(text, return_offsets_mapping=True)

input_ids = enc.input_ids
offsets = enc.offset_mapping

print(f"Total tokens: {len(input_ids)}")
print(f"Total chars: {len(text)}")
print(f"Total bytes: {len(text.encode('utf-8'))}")

# Convert char offsets to byte offsets
char_to_byte = [0] * (len(text) + 1)
b_idx = 0
for c_idx, ch in enumerate(text):
    char_to_byte[c_idx] = b_idx
    b_idx += len(ch.encode("utf-8"))
char_to_byte[len(text)] = b_idx

print("Char to byte length:", len(char_to_byte), "Max byte:", char_to_byte[-1])

# For each token, what is its byte span in the original text?
token_byte_spans = []
for i, (tok_id, (c_start, c_end)) in enumerate(zip(input_ids, offsets)):
    b_start = char_to_byte[c_start]
    b_end = char_to_byte[c_end]
    token_byte_spans.append((b_start, b_end))
    if i < 10 or i > len(input_ids) - 5:
        print(f"Token {i:2d} (id={tok_id:5d}): char=({c_start:2d}, {c_end:2d}) -> byte=({b_start:2d}, {b_end:2d}) "
              f"text={repr(text[c_start:c_end])} raw_bytes={repr(text.encode('utf-8')[b_start:b_end])}")
