"""Submission validation for the ROGII wellbore geology task.

Library use
-----------
    from src.submission import validate_submission
    report = validate_submission("submission.csv", SAMPLE_SUBMISSION)
    print(report)          # human readable
    report.passed          # bool
    report.to_dict()       # structured

Command line
------------
    python -m src.submission \
      --submission submission.csv \
      --sample-submission /kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv

Exit status is 0 on PASS and 1 on FAIL, so it can gate a pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import SAMPLE_SUBMISSION, SUBMISSION_FILENAME

REQUIRED_COLUMNS = ["id", "tvt"]


# ------------------------------------------------------------------ report --

@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"  # "error" | "warning"

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL" if self.severity == "error" else "WARN"


@dataclass
class ValidationReport:
    submission_path: str = ""
    sample_path: str = ""
    n_rows: int = 0
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", severity: str = "error") -> None:
        self.checks.append(Check(name, passed, detail, severity))

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    # aliases
    @property
    def ok(self) -> bool:
        return self.passed

    @property
    def result(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise ValueError(
                "Submission FAILED validation:\n  - "
                + "\n  - ".join(f"{c.name}: {c.detail}" for c in self.errors)
            )

    def to_dict(self) -> dict:
        return {
            "result": self.result,
            "submission": self.submission_path,
            "sample_submission": self.sample_path,
            "n_rows": self.n_rows,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks
            ],
        }

    def __str__(self) -> str:
        width = max((len(c.name) for c in self.checks), default=10)
        lines = [f"Submission validation: {self.result}",
                 f"  file   : {self.submission_path}",
                 f"  sample : {self.sample_path}",
                 f"  rows   : {self.n_rows:,}",
                 ""]
        for c in self.checks:
            lines.append(f"  [{c.status:4}] {c.name:<{width}}  {c.detail}")
        lines.append("")
        lines.append(
            f"  {len(self.errors)} error(s), {len(self.warnings)} warning(s) -> {self.result}"
        )
        return "\n".join(lines)


# -------------------------------------------------------------- validation --

def _load(obj, label: str) -> tuple[pd.DataFrame | None, str, str | None]:
    """Return (frame, display_path, error). Never raises on a bad path."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy(), f"<in-memory {label}>", None
    path = Path(obj)
    if not path.exists():
        return None, str(path), f"{label} file not found: {path}"
    try:
        return pd.read_csv(path), str(path), None
    except Exception as exc:
        return None, str(path), f"could not read {label}: {type(exc).__name__}: {exc}"


def validate_submission(
    submission_path,
    sample_submission_path=SAMPLE_SUBMISSION,
    *,
    require_exact_order: bool = True,
    plausible_range: tuple[float, float] | None = None,
) -> ValidationReport:
    """Validate a submission against sample_submission.csv.

    Accepts paths or DataFrames. Always returns a report; it does not raise on
    invalid input, so callers can inspect every failure at once.
    """
    rep = ValidationReport()

    sub, sub_disp, sub_err = _load(submission_path, "submission")
    sample, samp_disp, samp_err = _load(sample_submission_path, "sample_submission")
    rep.submission_path, rep.sample_path = sub_disp, samp_disp

    if samp_err:
        rep.add("sample_submission_readable", False, samp_err)
        return rep
    rep.add("sample_submission_readable", True, f"{len(sample):,} reference rows")

    if sub_err:
        rep.add("submission_readable", False, sub_err)
        return rep
    rep.add("submission_readable", True, "")
    rep.n_rows = len(sub)

    # -- filename -----------------------------------------------------------
    if not isinstance(submission_path, pd.DataFrame):
        name = Path(submission_path).name
        rep.add(
            "output_filename",
            name == SUBMISSION_FILENAME,
            f"file is '{name}', Kaggle expects '{SUBMISSION_FILENAME}'"
            if name != SUBMISSION_FILENAME else f"'{name}'",
            severity="warning",
        )

    # -- columns ------------------------------------------------------------
    sub_cols = [str(c) for c in sub.columns]
    samp_cols = [str(c) for c in sample.columns]
    expected = samp_cols if samp_cols else REQUIRED_COLUMNS
    rep.add(
        "exact_columns",
        sub_cols == expected,
        f"got {sub_cols}, expected {expected}",
    )
    unexpected = [c for c in sub_cols if c not in expected]
    rep.add("no_unexpected_columns", not unexpected, f"unexpected: {unexpected}" if unexpected else "none")
    missing_cols = [c for c in expected if c not in sub_cols]
    rep.add("no_missing_columns", not missing_cols, f"missing: {missing_cols}" if missing_cols else "none")
    if missing_cols:
        return rep  # nothing further is meaningful

    id_col, val_col = expected[0], expected[1] if len(expected) > 1 else "tvt"

    # -- row count ----------------------------------------------------------
    rep.add(
        "row_count",
        len(sub) == len(sample),
        f"{len(sub):,} rows, expected {len(sample):,}",
    )

    # -- ids ----------------------------------------------------------------
    sub_ids = sub[id_col].astype(str)
    samp_ids = sample[id_col].astype(str)

    dup_mask = sub_ids.duplicated()
    rep.add(
        "no_duplicate_ids",
        not dup_mask.any(),
        f"{int(dup_mask.sum())} duplicate ids, e.g. {sub_ids[dup_mask].unique()[:3].tolist()}"
        if dup_mask.any() else "none",
    )

    sub_set, samp_set = set(sub_ids), set(samp_ids)
    missing_ids = samp_set - sub_set
    unknown_ids = sub_set - samp_set
    rep.add(
        "no_missing_ids",
        not missing_ids,
        f"{len(missing_ids)} ids absent, e.g. {sorted(missing_ids)[:3]}" if missing_ids else "none",
    )
    rep.add(
        "no_unknown_ids",
        not unknown_ids,
        f"{len(unknown_ids)} ids not in sample, e.g. {sorted(unknown_ids)[:3]}" if unknown_ids else "none",
    )

    if require_exact_order:
        same_order = len(sub) == len(sample) and sub_ids.tolist() == samp_ids.tolist()
        detail = "matches sample_submission"
        if not same_order and not missing_ids and not unknown_ids and len(sub) == len(sample):
            first = next(
                (i for i, (a, b) in enumerate(zip(sub_ids, samp_ids)) if a != b), None
            )
            detail = f"same id set but different order; first mismatch at row {first}"
        elif not same_order:
            detail = "id sequence differs from sample_submission"
        rep.add("id_order", same_order, detail)

    # -- duplicated rows ----------------------------------------------------
    dup_rows = int(sub.duplicated().sum())
    rep.add("no_duplicate_rows", dup_rows == 0, f"{dup_rows} fully duplicated rows" if dup_rows else "none")

    # -- prediction column --------------------------------------------------
    raw = sub[val_col]
    numeric = pd.to_numeric(raw, errors="coerce")
    non_numeric = int((numeric.isna() & raw.notna()).sum())
    rep.add(
        "numeric_dtype",
        non_numeric == 0,
        f"{non_numeric} non-numeric values in '{val_col}'" if non_numeric else f"'{val_col}' is numeric",
    )

    n_nan = int(raw.isna().sum())
    rep.add("no_nan_predictions", n_nan == 0, f"{n_nan} NaN values" if n_nan else "none")

    arr = numeric.to_numpy(dtype="float64", na_value=0.0)
    n_inf = int(np.isinf(arr).sum())
    rep.add("no_infinite_predictions", n_inf == 0, f"{n_inf} infinite values" if n_inf else "none")

    # -- accidental prefix overwrite ---------------------------------------
    # A constant column, or one that exactly reproduces the sample's placeholder,
    # means predictions were never written.
    if len(sample) and val_col in sample.columns and len(sub) == len(sample):
        samp_vals = pd.to_numeric(sample[val_col], errors="coerce")
        identical = bool(np.allclose(arr, samp_vals.to_numpy(dtype="float64", na_value=0.0), equal_nan=True))
        rep.add(
            "not_sample_placeholder",
            not identical,
            "values are identical to sample_submission — predictions not written?"
            if identical else "differs from placeholder",
            severity="warning",
        )
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite):
        constant = bool(finite.nunique() == 1)
        rep.add(
            "not_constant",
            not constant,
            f"all predictions equal {finite.iloc[0]}" if constant else f"{finite.nunique():,} distinct values",
            severity="warning",
        )

    if plausible_range is not None and len(finite):
        lo, hi = plausible_range
        out = int(((finite < lo) | (finite > hi)).sum())
        rep.add(
            "plausible_range",
            out == 0,
            f"{out} values outside [{lo}, {hi}]" if out else f"within [{lo}, {hi}]",
            severity="warning",
        )
    return rep


@dataclass
class SampleSpec:
    """The submission contract, learned from sample_submission.csv."""
    path: str
    columns: list[str]
    id_column: str
    value_columns: list[str]
    n_rows: int
    id_order: list[str]
    id_pattern: str
    wells: list[str]
    rows_per_well: dict[str, int]
    id_is_wellid_plus_index: bool
    value_dtypes: dict[str, str]
    n_duplicate_ids: int

    def describe(self) -> str:
        return (
            f"{self.n_rows:,} rows | columns={self.columns} | "
            f"id='{self.id_column}' pattern='{self.id_pattern}' | "
            f"{len(self.wells)} well(s) | dtypes={self.value_dtypes}"
        )


def _split_id(raw: str) -> tuple[str, str | None]:
    """Split an id into (well part, trailing integer) when it looks composite."""
    import re
    m = re.match(r"^(.*?)[_\-:]?(\d+)$", str(raw))
    if m:
        return m.group(1).rstrip("_-:"), m.group(2)
    return str(raw), None


def audit_sample_submission(sample_submission_path=SAMPLE_SUBMISSION) -> SampleSpec:
    """Learn the submission contract from the official sample file."""
    import re
    from collections import Counter

    path = Path(sample_submission_path)
    if not path.exists():
        raise FileNotFoundError(f"sample submission not found: {path}")
    df = pd.read_csv(path)

    cols = [str(c) for c in df.columns]
    id_col = cols[0]
    value_cols = cols[1:]
    ids = df[id_col].astype(str).tolist()

    parts = [_split_id(i) for i in ids]
    wells = [w for w, _ in parts]
    tails = [t for _, t in parts]
    per_well = Counter(wells)

    composite = bool(ids) and all(t is not None for t in tails) and len(per_well) < len(ids)
    monotone = False
    if composite:
        tmp = pd.DataFrame({"w": wells, "t": [int(t) for t in tails]})
        monotone = bool(tmp.groupby("w")["t"].apply(lambda s: s.is_monotonic_increasing).all())

    return SampleSpec(
        path=str(path),
        columns=cols,
        id_column=id_col,
        value_columns=value_cols,
        n_rows=len(df),
        id_order=ids,
        id_pattern=re.sub(r"\d+", "<int>", ids[0]) if ids else "n/a",
        wells=list(dict.fromkeys(wells)),
        rows_per_well=dict(sorted(per_well.items())),
        id_is_wellid_plus_index=composite and monotone,
        value_dtypes={c: str(pd.to_numeric(df[c], errors="coerce").dtype) for c in value_cols},
        n_duplicate_ids=int(df[id_col].duplicated().sum()),
    )


def build_submission(
    predictions, sample_submission_path=SAMPLE_SUBMISSION
) -> pd.DataFrame:
    """Assemble a correctly ordered submission frame from an id -> value mapping.

    Guarantees the sample's exact id order, which is the single most common
    source of a silently wrong submission.
    """
    spec = audit_sample_submission(sample_submission_path)
    val_col = spec.value_columns[0] if spec.value_columns else "tvt"

    if isinstance(predictions, pd.DataFrame):
        if spec.id_column not in predictions.columns:
            raise KeyError(f"predictions must contain an '{spec.id_column}' column")
        src_val = val_col if val_col in predictions.columns else predictions.columns[-1]
        mapping = dict(zip(predictions[spec.id_column].astype(str), predictions[src_val]))
    elif isinstance(predictions, pd.Series):
        mapping = {str(k): v for k, v in predictions.items()}
    else:
        mapping = {str(k): v for k, v in dict(predictions).items()}

    missing = [i for i in spec.id_order if i not in mapping]
    if missing:
        raise KeyError(
            f"{len(missing)} sample ids have no prediction, e.g. {missing[:5]}"
        )
    return pd.DataFrame({
        spec.id_column: spec.id_order,
        val_col: [mapping[i] for i in spec.id_order],
    })


def write_submission(df: pd.DataFrame, path=None, *, sample_submission_path=SAMPLE_SUBMISSION) -> Path:
    """Validate then write. Refuses to write an invalid submission."""
    out = Path(path) if path is not None else Path(SUBMISSION_FILENAME)
    rep = validate_submission(df, sample_submission_path)
    rep.raise_if_failed()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


# --------------------------------------------------------------------- CLI --

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m src.submission",
        description="Validate a submission file against sample_submission.csv",
    )
    p.add_argument("--submission", required=True, help="path to submission.csv")
    p.add_argument("--sample-submission", default=str(SAMPLE_SUBMISSION),
                   help="path to the official sample_submission.csv")
    p.add_argument("--allow-any-order", action="store_true",
                   help="do not require the id order to match the sample")
    p.add_argument("--json", action="store_true", help="emit the structured report as JSON")
    args = p.parse_args(argv)

    rep = validate_submission(
        args.submission,
        args.sample_submission,
        require_exact_order=not args.allow_any_order,
    )
    print(json.dumps(rep.to_dict(), indent=2) if args.json else str(rep))
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
