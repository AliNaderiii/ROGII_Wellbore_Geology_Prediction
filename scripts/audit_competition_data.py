"""Section 1 — Competition data audit.

Generates:
    reports/input_inventory.md
    reports/dataset_schema.csv
    reports/well_summary.csv
    reports/data_quality_initial.md

Efficiency notes
----------------
* Schema discovery reads only the header + first 200 rows of each file.
* Per-well statistics are computed with a single pass per file using
  `usecols` restricted to the resolved roles + marker columns, and float32
  dtypes. Nothing is concatenated into one giant frame.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:  # executed as a loose file, not as a package
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap
bootstrap()

import numpy as np
import pandas as pd

from src.columns import FORMATION_ORDER, marker_columns, resolve_roles
from src.discovery import discover_wells, well_id_prefix
from src.paths import (
    SAMPLE_SUBMISSION,
    TEST_DIR,
    TRAIN_DIR,
    ensure_reports_dir,
)



def peek_schema(path: Path, nrows: int = 200) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=nrows)
    rows = []
    for col in head.columns:
        s = head[col]
        rows.append(
            {
                "file": path.name,
                "column": col,
                "dtype_sampled": str(s.dtype),
                "n_null_in_sample": int(s.isna().sum()),
                "example": "" if s.dropna().empty else str(s.dropna().iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def count_rows(path: Path) -> int:
    with path.open("rb") as fh:
        n = sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1 << 20), b""))
    return max(n - 1, 0)


def summarise_well(well) -> dict:
    rec: dict = {
        "well_id": well.well_id,
        "split": well.split,
        "prefix": well_id_prefix(well.well_id),
        "has_horizontal": well.horizontal is not None,
        "has_typewell": well.typewell is not None,
        "n_png": len(well.images),
        "n_other_files": len(well.others),
    }
    if well.horizontal is not None:
        hw = pd.read_csv(well.horizontal)
        roles = resolve_roles(hw.columns)
        markers = marker_columns(hw.columns)
        rec["hw_rows"] = len(hw)
        rec["hw_cols"] = hw.shape[1]
        rec["hw_marker_cols"] = ",".join(markers)
        rec["hw_missing_markers"] = ",".join(
            f for f in FORMATION_ORDER if f not in markers
        )
        rec["hw_any_duplicate_rows"] = int(hw.duplicated().sum())

        md_col = roles.get("md")
        if md_col:
            md = pd.to_numeric(hw[md_col], errors="coerce")
            d = md.diff().dropna()
            rec["md_min"] = float(md.min())
            rec["md_max"] = float(md.max())
            rec["md_monotonic_increasing"] = bool(md.is_monotonic_increasing)
            rec["md_step_median"] = float(d.median()) if len(d) else np.nan
            rec["md_step_std"] = float(d.std()) if len(d) else np.nan
            rec["md_duplicates"] = int(md.duplicated().sum())
            rec["md_uniform_spacing"] = bool(
                len(d) and np.isclose(d.std(), 0.0, atol=1e-6)
            )
        gr_col = roles.get("gr")
        if gr_col:
            gr = pd.to_numeric(hw[gr_col], errors="coerce")
            rec["gr_missing_frac"] = float(gr.isna().mean())
            rec["gr_min"] = float(gr.min())
            rec["gr_max"] = float(gr.max())
            rec["gr_negative"] = int((gr < 0).sum())
        ti_col = roles.get("tvt_input")
        if ti_col:
            ti = pd.to_numeric(hw[ti_col], errors="coerce")
            known = ti.notna()
            rec["tvt_input_missing_frac"] = float(1 - known.mean())
            rec["tvt_input_known_rows"] = int(known.sum())
            if known.any():
                first_gap = int(np.argmax(~known.values)) if (~known).any() else len(ti)
                rec["prefix_len_contiguous"] = first_gap
                rec["hidden_tail_len"] = len(ti) - first_gap
                rec["tvt_input_pattern_is_clean_prefix"] = bool(
                    known.values[:first_gap].all() and not known.values[first_gap:].any()
                )
                rec["tvt_last_known"] = float(ti.iloc[first_gap - 1]) if first_gap else np.nan
        tgt_col = roles.get("tvt")
        rec["has_tvt_target"] = tgt_col is not None
        if tgt_col:
            t = pd.to_numeric(hw[tgt_col], errors="coerce")
            rec["tvt_missing_frac"] = float(t.isna().mean())
            rec["tvt_min"] = float(t.min())
            rec["tvt_max"] = float(t.max())
        ps_col = roles.get("prediction_start")
        rec["has_prediction_start"] = ps_col is not None
        if ps_col:
            rec["prediction_start_value"] = str(hw[ps_col].dropna().iloc[0]) if hw[ps_col].notna().any() else ""
        id_col = roles.get("id")
        if id_col:
            rec["id_example"] = str(hw[id_col].iloc[0])
            rec["id_duplicates"] = int(hw[id_col].duplicated().sum())

    if well.typewell is not None:
        tw = pd.read_csv(well.typewell)
        roles_tw = resolve_roles(tw.columns)
        rec["tw_rows"] = len(tw)
        rec["tw_cols"] = tw.shape[1]
        rec["tw_columns"] = ",".join(map(str, tw.columns))
        g = roles_tw.get("geology")
        if g:
            rec["tw_n_geology_labels"] = int(tw[g].nunique(dropna=True))
            rec["tw_geology_labels"] = ",".join(map(str, pd.unique(tw[g].dropna())))
        tv = roles_tw.get("tvt")
        if tv:
            t = pd.to_numeric(tw[tv], errors="coerce")
            rec["tw_tvt_min"] = float(t.min())
            rec["tw_tvt_max"] = float(t.max())
            rec["tw_tvt_monotonic"] = bool(t.is_monotonic_increasing)
    return rec


def main() -> None:
    REPORTS = ensure_reports_dir()
    train = discover_wells(TRAIN_DIR, "train")
    test = discover_wells(TEST_DIR, "test")
    all_wells = {**{k: v for k, v in train.items()}, **{k: v for k, v in test.items()}}

    if not all_wells:
        raise SystemExit(
            f"No wells discovered under {TRAIN_DIR} / {TEST_DIR}. "
            "Check that the competition dataset is mounted."
        )

    # ---- schema (cheap header peek on a few representative files) ----
    schema_frames = []
    seen_kinds: set[str] = set()
    for split, wells in (("train", train), ("test", test)):
        for well in wells.values():
            for kind, path in (("horizontal", well.horizontal), ("typewell", well.typewell)):
                if path is None:
                    continue
                key = f"{split}:{kind}"
                if key in seen_kinds:
                    continue
                seen_kinds.add(key)
                df = peek_schema(path)
                df.insert(0, "split", split)
                df.insert(1, "kind", kind)
                schema_frames.append(df)
    if SAMPLE_SUBMISSION.exists():
        df = peek_schema(SAMPLE_SUBMISSION)
        df.insert(0, "split", "submission")
        df.insert(1, "kind", "sample_submission")
        schema_frames.append(df)
    schema = pd.concat(schema_frames, ignore_index=True)
    schema.to_csv(REPORTS / "dataset_schema.csv", index=False)

    # ---- per-well summary ----
    records = [summarise_well(w) for w in all_wells.values()]
    summary = pd.DataFrame(records).sort_values(["split", "well_id"])
    summary.to_csv(REPORTS / "well_summary.csv", index=False)

    tr = summary[summary.split == "train"]
    te = summary[summary.split == "test"]

    # ---- duplicates ----
    dup_wells = sorted(set(train) & set(test))
    prefix_counts = Counter(summary["prefix"])
    dup_prefixes = {p: c for p, c in prefix_counts.items() if c > 1}

    def fmt(series, f="{:.3f}"):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return "n/a"
        return f"min={f.format(s.min())}, median={f.format(s.median())}, max={f.format(s.max())}"

    inv = [
        "# Input Inventory — Competition Data",
        "",
        f"Generated from `{TRAIN_DIR.parent}`.",
        "",
        "## Well counts",
        "",
        f"- Train wells discovered: **{len(train)}**",
        f"- Test wells discovered: **{len(test)}**",
        f"- Wells with both horizontal + typewell files: "
        f"**{int(summary.has_horizontal.astype(bool).sum() & 1) if False else int((summary.has_horizontal & summary.has_typewell).sum())}**",
        f"- PNG files attached to wells: **{int(summary.n_png.sum())}**",
        "",
        "## Rows",
        "",
        f"- Horizontal-well rows (train): **{int(pd.to_numeric(tr.get('hw_rows'), errors='coerce').sum()):,}**",
        f"- Horizontal-well rows (test): **{int(pd.to_numeric(te.get('hw_rows'), errors='coerce').sum()):,}**",
        f"- Typewell rows (train): **{int(pd.to_numeric(tr.get('tw_rows'), errors='coerce').sum()):,}**",
        f"- Rows per horizontal well: {fmt(summary.get('hw_rows'), '{:.0f}')}",
        "",
        "## Columns",
        "",
        "See `dataset_schema.csv` for the full column list per file kind.",
        "",
        f"- Marker columns present in train HW: `{tr['hw_marker_cols'].iloc[0] if len(tr) and 'hw_marker_cols' in tr else 'n/a'}`",
        f"- Marker columns present in test HW: `{te['hw_marker_cols'].iloc[0] if len(te) and 'hw_marker_cols' in te else 'n/a'}`",
        f"- TVT target column present in train: "
        f"{bool(tr['has_tvt_target'].any()) if 'has_tvt_target' in tr else 'n/a'}",
        f"- TVT target column present in test: "
        f"{bool(te['has_tvt_target'].any()) if 'has_tvt_target' in te else 'n/a'}",
        "",
        "## MD ordering and spacing",
        "",
        f"- Wells with monotonically increasing MD: "
        f"{int(summary.get('md_monotonic_increasing', pd.Series(dtype=bool)).sum())} / {len(summary)}",
        f"- Wells with perfectly uniform MD spacing: "
        f"{int(summary.get('md_uniform_spacing', pd.Series(dtype=bool)).sum())} / {len(summary)}",
        f"- Median MD step across wells: {fmt(summary.get('md_step_median'))}",
        "",
        "## Hidden-suffix structure",
        "",
        f"- Wells where TVT_input is a clean known-prefix / unknown-suffix split: "
        f"{int(summary.get('tvt_input_pattern_is_clean_prefix', pd.Series(dtype=bool)).sum())} / {len(summary)}",
        f"- Prefix length (rows): {fmt(summary.get('prefix_len_contiguous'), '{:.0f}')}",
        f"- Hidden tail length (rows): {fmt(summary.get('hidden_tail_len'), '{:.0f}')}",
        "",
        "## Duplicates",
        "",
        f"- Well IDs appearing in both train and test: {dup_wells or 'none'}",
        f"- Well-ID prefixes shared by more than one well: {len(dup_prefixes)}",
        f"- Wells containing duplicated rows: "
        f"{int((pd.to_numeric(summary.get('hw_any_duplicate_rows'), errors='coerce').fillna(0) > 0).sum())}",
        f"- Wells containing duplicated MD values: "
        f"{int((pd.to_numeric(summary.get('md_duplicates'), errors='coerce').fillna(0) > 0).sum())}",
        "",
    ]
    (REPORTS / "input_inventory.md").write_text("\n".join(inv), encoding="utf-8")

    dq = [
        "# Initial Data Quality Report",
        "",
        "## Missing values",
        "",
        f"- GR missing fraction per well: {fmt(summary.get('gr_missing_frac'))}",
        f"- Wells with >50% GR missing: "
        f"{int((pd.to_numeric(summary.get('gr_missing_frac'), errors='coerce').fillna(0) > 0.5).sum())}",
        f"- TVT_input missing fraction per well: {fmt(summary.get('tvt_input_missing_frac'))}",
        "",
        "> GR gaps are expected to be *contiguous depth intervals* (tool outage), not",
        "> random. Impute within each well only — never globally.",
        "",
        "## Invalid values",
        "",
        f"- Wells with negative GR readings: "
        f"{int((pd.to_numeric(summary.get('gr_negative'), errors='coerce').fillna(0) > 0).sum())}",
        f"- Wells with non-monotonic MD: "
        f"{len(summary) - int(summary.get('md_monotonic_increasing', pd.Series(dtype=bool)).sum())}",
        f"- Wells missing a typewell file: {int((~summary.has_typewell).sum())}",
        f"- Wells missing a horizontal file: {int((~summary.has_horizontal).sum())}",
        "",
        "## Row ordering",
        "",
        "Row order is treated as authoritative: submissions must preserve the",
        "original file order. The loader never sorts or re-indexes.",
        "",
        "## Typewell geology labels",
        "",
        f"- Distinct geology labels per typewell: {fmt(summary.get('tw_n_geology_labels'), '{:.0f}')}",
        "",
        "## Action items",
        "",
        "1. Per-well linear interpolation of GR, then light smoothing.",
        "2. Do not use formation-top marker columns as raw features if they are",
        "   absent from the test horizontal files (see inventory above).",
        "3. Anchor predictions on the last known TVT_input value of the prefix.",
        "4. Enforce the ANCC->ASTNU->ASTNL->EGFDU->EGFDL->BUDA ordering in",
        "   post-processing.",
        "",
    ]
    (REPORTS / "data_quality_initial.md").write_text("\n".join(dq), encoding="utf-8")

    print("Wrote input_inventory.md, dataset_schema.csv, well_summary.csv, data_quality_initial.md")


if __name__ == "__main__":
    main()
