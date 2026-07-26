"""Data-loader smoke test against the real mounted competition data.

    python scripts/smoke_test_loader.py

Loads, but never trains. Exercises four representative cases:
  1. one train well
  2. one test well
  3. the well with the highest GR missingness
  4. the well with the longest hidden suffix
Then validates sample_submission.csv against itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap
bootstrap()

import pandas as pd

from src.data import (
    discover_wells,
    iter_wells,
    load_well,
    summarize_well,
    validate_split,
    well_metadata,
)
from src.paths import SAMPLE_SUBMISSION, available, require_competition_data
from src.submission import validate_submission

KEYS = ["well_id", "split", "n_rows", "n_visible", "n_hidden", "visible_fraction",
        "prediction_start_row", "prediction_start_source", "clean_prefix_split",
        "gr_missing_frac", "gr_longest_gap", "has_target_column", "has_typewell",
        "n_markers_present", "markers_absent"]


def show(title: str, rec: dict) -> None:
    print(f"\n--- {title} ---")
    for k in KEYS:
        if k in rec:
            v = rec[k]
            print(f"  {k:<26} {v:.4g}" if isinstance(v, float) else f"  {k:<26} {v}")


def main() -> int:
    require_competition_data()

    train_ids = sorted(discover_wells("train"))
    test_ids = sorted(discover_wells("test"))
    print(f"discovered {len(train_ids)} train wells, {len(test_ids)} test wells")
    if not train_ids:
        print("ERROR: no train wells discovered")
        return 1

    # 1. one train well
    show("1. train well", summarize_well(load_well(train_ids[0], "train")))

    # 2. one test well
    if test_ids:
        show("2. test well", summarize_well(load_well(test_ids[0], "test")))
    else:
        print("\n--- 2. test well --- none discovered")

    # scan train metadata once, reuse for cases 3 and 4
    print("\nscanning train wells for edge cases (streamed, one well resident)...")
    meta = well_metadata("train")

    if "gr_missing_frac" in meta and meta["gr_missing_frac"].notna().any():
        row = meta.loc[meta["gr_missing_frac"].idxmax()]
        show("3. highest GR missingness", row.to_dict())
    else:
        print("\n--- 3. highest GR missingness --- no GR column resolved")

    if "n_hidden" in meta and meta["n_hidden"].notna().any():
        row = meta.loc[meta["n_hidden"].idxmax()]
        show("4. longest hidden suffix", row.to_dict())

    # structural issues
    issues = pd.concat([validate_split("train"), validate_split("test")], ignore_index=True)
    print(f"\n--- structural issues: {len(issues)} ---")
    if len(issues):
        print(issues["issue"].value_counts().to_string())
        leaks = issues[issues["issue"].str.startswith("TEST_LEAK")]
        if len(leaks):
            print("\n!! POTENTIAL TEST LEAK !!")
            print(leaks.to_string(index=False))

    # 5. submission contract
    print("\n--- 5. sample_submission self-validation ---")
    if available(SAMPLE_SUBMISSION):
        rep = validate_submission(SAMPLE_SUBMISSION, SAMPLE_SUBMISSION)
        print(rep)
        print("(placeholder/constant warnings are expected for the sample itself)")
    else:
        print(f"  not mounted: {SAMPLE_SUBMISSION}")

    print("\nSMOKE TEST COMPLETE — no model was trained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
