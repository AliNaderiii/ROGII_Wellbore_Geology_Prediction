"""Section 3 — Sample submission audit -> reports/sample_submission_audit.md."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.discovery import discover_wells
from src.paths import REPORTS_DIR, SAMPLE_SUBMISSION, TEST_DIR, ensure_reports_dir
from src.submission import (
    SUBMISSION_FILENAME,
    audit_sample_submission,
    validate_submission,
    _split_id,
)

ensure_reports_dir()


def main() -> None:
    if not SAMPLE_SUBMISSION.exists():
        raise SystemExit(f"Missing {SAMPLE_SUBMISSION}")

    spec = audit_sample_submission()
    df = pd.read_csv(SAMPLE_SUBMISSION)
    self_check = validate_submission(df, spec)

    per_well = Counter(_split_id(i)[0] for i in spec.id_order)
    test_wells = set(discover_wells(TEST_DIR, "test"))
    covered = set(per_well) & test_wells
    unmatched = set(per_well) - test_wells

    lines = [
        "# Sample Submission Audit",
        "",
        f"Source: `{SAMPLE_SUBMISSION}`",
        "",
        "## Contract",
        "",
        f"- Columns: `{spec.columns}`",
        f"- Row count: **{spec.n_rows:,}**",
        f"- ID column: `{spec.id_column}`",
        f"- Prediction column(s): `{spec.value_columns}`",
        f"- Prediction dtype in sample: `{spec.value_dtype}` (submit float)",
        f"- Required output filename: `{SUBMISSION_FILENAME}`",
        "",
        "## ID structure",
        "",
        f"- Example ID: `{spec.id_order[0] if spec.id_order else 'n/a'}`",
        f"- Inferred pattern: `{spec.id_pattern}`",
        f"- ID decomposes into well + monotonically increasing row index: "
        f"**{spec.id_is_wellid_plus_index}**",
        f"- Distinct wells in submission: **{len(per_well)}**",
        f"- Test wells found on disk: **{len(test_wells)}**",
        f"- Submission wells matched to a test well directory: **{len(covered)}**",
        f"- Submission wells with no matching test file: "
        f"{sorted(unmatched)[:10] if unmatched else 'none'}",
        "",
        "## Rows per well",
        "",
        "| well | rows |",
        "|---|---|",
    ]
    lines += [f"| {w} | {n:,} |" for w, n in sorted(per_well.items())]
    lines += [
        "",
        "## Relationship between ID and row index",
        "",
        "The row order in `sample_submission.csv` is the contract. The validator",
        "in `src/submission.py` enforces exact ID equality **and** exact ordering,",
        "so predictions must be reindexed onto `spec.id_order` before writing.",
        "",
        "## Self-check (sample validated against its own spec)",
        "",
        "```",
        str(self_check),
        "```",
        "",
        "## Reusable validator",
        "",
        "```python",
        "from src.submission import audit_sample_submission, validate_submission, write_submission",
        "",
        "spec = audit_sample_submission()",
        "report = validate_submission(pred_df, spec, plausible_range=(-500, 500))",
        "report.raise_if_failed()",
        "write_submission(pred_df, 'submission.csv')",
        "```",
        "",
    ]
    out = REPORTS_DIR / "sample_submission_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
