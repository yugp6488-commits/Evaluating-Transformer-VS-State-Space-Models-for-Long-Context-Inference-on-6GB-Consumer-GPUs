#!/usr/bin/env python3
"""
compute_hashes.py - Computes SHA256 checksums for authoritative paper data and tables.
Covers results/final_experiment/ and results/paper_tables/.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

target_files = [
    # Authoritative Single Source of Truth
    "results/final_experiment/environment_metadata.csv",
    "results/final_experiment/phase1_prefill.csv",
    "results/final_experiment/phase2_decode.csv",
    "results/final_experiment/phase3_batch.csv",
    "results/final_experiment/phase4_quality_per_document.csv",
    "results/final_experiment/phase4_quality_summary.csv",
    "results/final_experiment/phase5_retrieval_trials.csv",
    "results/final_experiment/phase5_retrieval_summary.csv",
    # Derived Paper Tables
    "results/paper_tables/phase1_prefill_table.csv",
    "results/paper_tables/phase2_decode_table.csv",
    "results/paper_tables/phase3_batch_table.csv",
    "results/paper_tables/phase4_per_document_table.csv",
    "results/paper_tables/phase4_summary_table.csv",
    "results/paper_tables/phase5_retrieval_table.csv",
    "results/paper_tables/derived_claims.json",
    "results/paper_tables/tables.md",
]

def main():
    print(f"{'SHA256 Hash':64}  {'File Path'}")
    print("-" * 100)
    for rel_path in target_files:
        p = ROOT / rel_path
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            print(f"{h}  {rel_path}")
        else:
            print(f"{'MISSING':64}  {rel_path}")

if __name__ == "__main__":
    main()
