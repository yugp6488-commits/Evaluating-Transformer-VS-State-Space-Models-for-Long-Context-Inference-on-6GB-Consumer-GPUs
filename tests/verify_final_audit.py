#!/usr/bin/env python3
"""
verify_final_audit.py - Automated Master Verification Gate for Task 6.
"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
audit_file = ROOT / "FINAL_RESULTS_AUDIT.md"
ref_p5_file = ROOT / "results" / "paper_tables" / "phase5_retrieval_table.csv"

def test_forbidden_strings():
    print("[TEST 1] Scanning FINAL_RESULTS_AUDIT.md for forbidden strings...")
    assert audit_file.exists(), f"Missing {audit_file}"
    content = audit_file.read_text(encoding="utf-8")

    forbidden = [
        "91.1",
        "3,983.34",
        "177,522",
        "170,400",
        "OOM (Skipped)",
        "cross-tokenizer",
        "100.0% perfect",
        "retained higher retrieval fidelity",
    ]

    for item in forbidden:
        assert item not in content, f"Forbidden string '{item}' found in FINAL_RESULTS_AUDIT.md!"

    # Case-insensitive check for 'independent'
    assert not re.search(r"\bindependent\b", content, re.IGNORECASE), "Forbidden word 'independent' found in FINAL_RESULTS_AUDIT.md!"
    print("  -> PASSED: Zero forbidden strings found.")


def test_phase5_table_parity():
    print("[TEST 2] Verifying Phase 5 table parity against generate_paper_tables.py...")
    assert ref_p5_file.exists(), f"Missing {ref_p5_file}"
    ref_df = pd.read_csv(ref_p5_file)

    lines = audit_file.read_text(encoding="utf-8").splitlines()
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
    assert len(parsed_df) == len(ref_df), f"Row count mismatch: {len(parsed_df)} vs {len(ref_df)}"

    for i in range(len(ref_df)):
        r_ref = ref_df.iloc[i]
        r_parsed = parsed_df.iloc[i]
        for col in ["Task", "Target_Seq_Len", "Regime", "Pythia_Exact_Match", "Pythia_95_CI", "Pythia_Partial_Credit", "Mamba_Exact_Match", "Mamba_95_CI", "Mamba_Partial_Credit", "Comparison"]:
            v_ref = str(r_ref[col]).replace('"', '').strip()
            v_parsed = str(r_parsed[col]).replace('"', '').strip()
            assert v_ref == v_parsed, f"Mismatch at row {i}, column {col}: '{v_parsed}' != '{v_ref}'"

    print("  -> PASSED: Every single cell in Phase 5 table matches generate_paper_tables.py output.")


def main():
    print("=" * 70)
    print("  RUNNING AUTOMATED FINAL VERIFICATION (TASK 6)")
    print("=" * 70)
    test_forbidden_strings()
    test_phase5_table_parity()
    print("=" * 70)
    print("  ALL VERIFICATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
