from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

texts = [
    "The Project Gutenberg eBook of The History of Antiquity, by Max Duncker.\nThis is a test: é, ñ, 漢字, 🚀!",
    "Testing rare: 𠮷 (CJK Unified Ideographs Extension B, U+20BB7, 4 bytes)",
    "   Multiple    spaces   and\nnewlines\n\n\ttabs."
]

for t in texts:
    enc = tok(t, return_offsets_mapping=True)
    offsets = enc.offset_mapping
    print(f"\nChecking text len={len(t)} chars:")
    for i in range(len(offsets) - 1):
        curr_s, curr_e = offsets[i]
        next_s, next_e = offsets[i+1]
        if curr_e != next_s:
            print(f"  Boundary mismatch at token {i}->{i+1}: tok[{i}]=({curr_s},{curr_e}, {repr(t[curr_s:curr_e])}) vs tok[{i+1}]=({next_s},{next_e}, {repr(t[next_s:next_e])})")
    print(f"  Start offset of tok 0: {offsets[0][0]}")
    print(f"  End offset of tok N-1: {offsets[-1][1]} (text len={len(t)})")
