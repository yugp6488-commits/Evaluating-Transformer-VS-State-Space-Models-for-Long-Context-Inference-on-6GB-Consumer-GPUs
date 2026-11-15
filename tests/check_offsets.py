import sys
from transformers import AutoTokenizer

for tid in ["EleutherAI/pythia-410m", "EleutherAI/gpt-neox-20b"]:
    print(f"Testing {tid}...")
    tok = AutoTokenizer.from_pretrained(tid)
    print(f"  is_fast: {tok.is_fast}")
    
    test_texts = [
        "Hello world! This is a simple test.",
        "   Multiple    spaces   and\nnewlines\n\n\ttabs.",
        "Punctuation: ... ,,, !!! ??? --- ;;; ::: 'quotes' \"double\"",
        "Unicode: café, naïve, résumé, façade, jalapeño.",
        "Multibyte & Emojis: 🚀🌟🔥🎉 and 漢字, 日本語, 한국어, العربية.",
        "Mixed: \n  Hello 🚀 world! 漢字 café... \t 12345!  \n"
    ]
    
    for text in test_texts:
        enc = tok(text, return_offsets_mapping=True)
        tokens = enc.input_ids
        offsets = enc.offset_mapping
        print(f"  Text: {repr(text[:30])}... (len={len(text)} chars, {len(text.encode('utf-8'))} bytes)")
        print(f"  Tokens: {len(tokens)}, Offsets: {len(offsets)}")
        
        # Check reconstruction from offsets
        # For each token, extract text[start:end]
        reconstructed_spans = []
        for i, (start, end) in enumerate(offsets):
            span_text = text[start:end]
            reconstructed_spans.append((start, end, span_text))
        
        # Check coverage
        covered = [False] * len(text)
        for start, end in offsets:
            for c in range(start, end):
                covered[c] = True
        
        uncovered = [i for i, c in enumerate(covered) if not c]
        if uncovered:
            print(f"    WARNING: Uncovered char indices: {uncovered}")
        else:
            print("    All characters covered by token offsets!")
            
        print("  Sample offsets (first 5):", [(s, e, text[s:e]) for s, e in offsets[:5]])
