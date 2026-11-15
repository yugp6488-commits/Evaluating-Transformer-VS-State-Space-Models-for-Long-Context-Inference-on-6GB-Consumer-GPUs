#!/usr/bin/env python3
"""
verify_tables_match.py - Verification script for Task 6.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ref_df = pd.read_csv(ROOT / "results" / "paper_tables" / "phase5_retrieval_table.csv")

with open(ROOT / "FINAL_RESULTS_AUDIT.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

table_lines = []
in_t5 = False
for line in lines:
    if "## 6. Phase 5:" in line:
        in_t5 = True
        continue
    if in_t5:
        if line.startswith("### "):
            break
        if line.strip().startswith("|") and not line.strip().startswith("| ---"):
            table_lines.append(line.strip())

header = [c.strip() for c in table_lines[0].split("|")[1:-1]]
data_rows = []
for l in table_lines[1:]:
    vals = [c.strip() for c in l.split("|")[1:-1]]
    data_rows.append(vals)

parsed_df = pd.DataFrame(data_rows, columns=header)
print(f"Parsed table rows: {len(parsed_df)}, Ref table rows: {len(ref_df)}")

mismatches = []
for i in range(len(ref_df)):
    r_ref = ref_df.iloc[i]
    r_parsed = parsed_df.iloc[i]
    for col in ["Task", "Pythia_Exact_Match", "Pythia_95_CI", "Mamba_Exact_Match", "Mamba_95_CI", "Comparison"]:
        v_ref = str(r_ref[col]).replace('"', '').strip()
        v_parsed = str(r_parsed[col]).replace('"', '').strip()
        if v_ref != v_parsed:
            mismatches.append((i, col, v_ref, v_parsed))

if mismatches:
    print(f"FAILED mismatches: {mismatches}")
    exit(1)
else:
    print("SUCCESS: Every single cell in Phase 5 table matches generate_paper_tables.py perfectly!")
