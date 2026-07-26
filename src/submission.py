"""Reusable submission validator.

Usage
-----
    from src.submission import audit_sample_submission, validate_submission

    spec = audit_sample_submission()          # learns the contract from the sample
    report = validate_submission(my_df, spec) # raises/returns issues
    report.raise_if_failed()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import SAMPLE_SUBMISSION

SUBMISSION_FILENAME = "submission.csv"


@dataclass
class SubmissionSpec:
    columns: list[str]
    id_column: str
    value_columns: list[str]
    n_rows: int
    id_order: list[str]
    id_pattern: str
    wells: list[str]
    id_is_wellid_plus_index: bool
    value_dtype: str = "float64"

    def describe(self) -> str:
        return (
            f"columns={self.columns}, rows={self.n_rows}, "
            f"id_col='{self.id_column}', value_cols={self.value_columns}, "
            f"wells={len(self.wells)}, id_pattern='{self.id_pattern}'"
        )


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError("Submission invalid:\n  - " + "\n  - ".join(self.errors))

    def __str__(self) -> str:
        out = ["OK" if self.ok else "FAILED"]
        out += [f"  ERROR   {e}" for e in self.errors]
        out += [f"  WARNING {w}" for w in self.warnings]
        return "\n".join(out)


def _split_id(raw: str) -> tuple[str, str | None]:
    """Split an id into (well part, trailing integer) if it looks composite."""
    m = re.match(r"^(.*?)[_\-:]?(\d+)$", str(raw))
    if m:
        return m.group(1).rstrip("_-:"), m.group(2)
    return str(raw), None


def audit_sample_submission(path: Path | None = None) -> SubmissionSpec:
    path = path or SAMPLE_SUBMISSION
    df = pd.read_csv(path)
    cols = list(map(str, df.columns))
    id_col = cols[0]
    value_cols = cols[1:]

    ids = df[id_col].astype(str).tolist()
    wells, tails = zip(*(_split_id(i) for i in ids)) if ids else ((), ())
    uniq_wells = list(dict.fromkeys(wells))

    # Does the numeric tail restart per well and increase monotonically?
    composite = all(t is not None for t in tails) and len(uniq_wells) < len(ids)
    monotone_per_well = False
    if composite:
        tmp = pd.DataFrame({"w": wells, "t": [int(t) for t in tails]})
        monotone_per_well = bool(
            tmp.groupby("w")["t"].apply(lambda s: s.is_monotonic_increasing).all()
        )

    sample_id = ids[0] if ids else ""
    pattern = re.sub(r"\d+", "<int>", sample_id)

    dtype = "float64"
    if value_cols:
        dtype = str(pd.to_numeric(df[value_cols[0]], errors="coerce").dtype)

    return SubmissionSpec(
        columns=cols,
        id_column=id_col,
        value_columns=value_cols,
        n_rows=len(df),
        id_order=ids,
        id_pattern=pattern,
        wells=uniq_wells,
        id_is_wellid_plus_index=composite and monotone_per_well,
        value_dtype=dtype,
    )


def validate_submission(
    sub: pd.DataFrame | Path | str,
    spec: SubmissionSpec | None = None,
    *,
    require_exact_order: bool = True,
    finite_only: bool = True,
    plausible_range: tuple[float, float] | None = None,
) -> ValidationReport:
    spec = spec or audit_sample_submission()
    rep = ValidationReport()

    if isinstance(sub, (str, Path)):
        p = Path(sub)
        if p.name != SUBMISSION_FILENAME:
            rep.warnings.append(f"file is named '{p.name}', Kaggle expects '{SUBMISSION_FILENAME}'")
        sub = pd.read_csv(p)

    cols = list(map(str, sub.columns))
    if cols != spec.columns:
        rep.errors.append(f"columns {cols} != expected {spec.columns}")
        return rep

    if len(sub) != spec.n_rows:
        rep.errors.append(f"row count {len(sub)} != expected {spec.n_rows}")

    ids = sub[spec.id_column].astype(str).tolist()
    if ids and ids[0] != spec.id_order[0]:
        rep.warnings.append(f"first id '{ids[0]}' != sample first id '{spec.id_order[0]}'")
    if sub[spec.id_column].duplicated().any():
        rep.errors.append("duplicate ids present")
    missing = set(spec.id_order) - set(ids)
    extra = set(ids) - set(spec.id_order)
    if missing:
        rep.errors.append(f"{len(missing)} ids from the sample are missing (e.g. {sorted(missing)[:3]})")
    if extra:
        rep.errors.append(f"{len(extra)} ids not present in the sample (e.g. {sorted(extra)[:3]})")
    if require_exact_order and not missing and not extra and ids != spec.id_order:
        rep.errors.append("id ordering differs from sample_submission")

    for col in spec.value_columns:
        v = pd.to_numeric(sub[col], errors="coerce")
        if v.isna().any():
            rep.errors.append(f"column '{col}' has {int(v.isna().sum())} NaN/non-numeric values")
        if finite_only and np.isinf(v.to_numpy(dtype="float64", na_value=0.0)).any():
            rep.errors.append(f"column '{col}' contains infinities")
        if plausible_range is not None:
            lo, hi = plausible_range
            out = int(((v < lo) | (v > hi)).sum())
            if out:
                rep.warnings.append(f"column '{col}' has {out} values outside [{lo}, {hi}]")
        if not pd.api.types.is_float_dtype(v):
            rep.warnings.append(f"column '{col}' is not float dtype")

    return rep


def write_submission(df: pd.DataFrame, path: Path | str = SUBMISSION_FILENAME) -> Path:
    spec = audit_sample_submission()
    rep = validate_submission(df, spec)
    rep.raise_if_failed()
    p = Path(path)
    df.to_csv(p, index=False)
    return p


if __name__ == "__main__":
    spec = audit_sample_submission()
    print(spec.describe())
    print(validate_submission(pd.read_csv(SAMPLE_SUBMISSION), spec))
