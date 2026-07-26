"""Section 3 — Sample submission audit -> reports/sample_submission_audit.md."""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:  # executed as a loose file, not as a package
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap
bootstrap()

import pandas as pd

from src.discovery import discover_wells
from src.paths import (
    REPORTS_DIR,
    SAMPLE_SUBMISSION,
    SUBMISSION_FILENAME,
    TEST_DIR,
    ensure_reports_dir,
)
from src.submission import validate_submission


def split_id(raw: str) -> tuple[str, str | None]:
    """Split an id into (well part, trailing integer) when it looks composite."""
    m = re.match(r"^(.*?)[_\-:]?(\d+)$", str(raw))
    if m:
        return m.group(1).rstrip("_-:"), m.group(2)
    return str(raw), None


def main() -> None:
    ensure_reports_dir()
    if not SAMPLE_SUBMISSION.exists():
        raise SystemExit(f"Missing {SAMPLE_SUBMISSION}")

    df = pd.read_csv(SAMPLE_SUBMISSION)
    cols = [str(c) for c in df.columns]
    id_col = cols[0]
    value_cols = cols[1:]
    ids = df[id_col].astype(str).tolist()

    parts = [split_id(i) for i in ids]
    wells = [w for w, _ in parts]
    tails = [t for _, t in parts]
    per_well = Counter(wells)
    composite = all(t is not None for t in tails) and len(per_well) < len(ids)
    monotone = False
    if composite:
        tmp = pd.DataFrame({"w": wells, "t": [int(t) for t in tails]})
        monotone = bool(tmp.groupby("w")["t"].apply(lambda s: s.is_monotonic_increasing).all())

    pattern = re.sub(r"\d+", "<int>", ids[0]) if ids else "n/a"

    test_wells = set(discover_wells(TEST_DIR, "test"))
    unmatched = sorted(set(per_well) - test_wells)

    # the sample must validate against itself (sanity check on the validator)
    self_check = validate_submission(SAMPLE_SUBMISSION, SAMPLE_SUBMISSION)

    dtypes = {c: str(pd.to_numeric(df[c], errors="coerce").dtype) for c in value_cols}

    lines = [
        "# Sample Submission Audit",
        "",
        f"Source: `{SAMPLE_SUBMISSION}`",
        "",
        "## Contract",
        "",
        f"- Columns: `{cols}`",
        f"- Row count: **{len(df):,}**",
        f"- ID column: `{id_col}`",
        f"- Prediction column(s): `{value_cols}`",
        f"- Prediction dtype(s): `{dtypes}` (submit float)",
        f"- Required output filename: `{SUBMISSION_FILENAME}`",
        "",
        "## ID structure",
        "",
        f"- Example ID: `{ids[0] if ids else 'n/a'}`",
        f"- Inferred pattern: `{pattern}`",
        f"- ID decomposes into well + monotonically increasing row index: **{composite and monotone}**",
        f"- Distinct wells in submission: **{len(per_well)}**",
        f"- Test wells found on disk: **{len(test_wells)}**",
        f"- Submission wells matched to a test well: **{len(set(per_well) & test_wells)}**",
        f"- Submission wells with no matching test file: {unmatched[:10] if unmatched else 'none'}",
        f"- Duplicate IDs in sample: **{int(df[id_col].duplicated().sum())}**",
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
        "The row order of `sample_submission.csv` is the contract. The validator",
        "in `src/submission.py` enforces exact ID equality **and** exact ordering,",
        "so predictions must be reindexed onto the sample's id sequence before",
        "writing.",
        "",
        "## Self-check (sample validated against itself)",
        "",
        "```",
        str(self_check),
        "```",
        "",
        "> The `not_sample_placeholder` / `not_constant` warnings above are",
        "> expected here: the sample *is* the placeholder. They exist to catch a",
        "> real submission that forgot to write predictions.",
        "",
        "## Reusable validator",
        "",
        "```bash",
        "python -m src.submission \\",
        "  --submission submission.csv \\",
        f"  --sample-submission {SAMPLE_SUBMISSION}",
        "```",
        "",
        "```python",
        "from src.submission import validate_submission",
        "report = validate_submission('submission.csv')",
        "report.raise_if_failed()",
        "```",
        "",
    ]
    out = REPORTS_DIR / "sample_submission_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
