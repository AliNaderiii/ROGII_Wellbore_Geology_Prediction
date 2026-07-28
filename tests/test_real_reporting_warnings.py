"""Regression: `src.real_reporting` must not emit a Pandas FutureWarning.

The original defect was `Series.fillna(False)` on an *object*-dtype column.
`alignment_ok` and `alignment_cache_hit` round-trip through CSV as `object`
whenever any row is blank (the Ridge rows carry no dip diagnostics), and pandas
deprecated the silent downcast that `.fillna` then performs:

    FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill
    is deprecated and will change in a future version.

These tests run the report writer with warnings promoted to errors, on a table
whose dtypes reproduce the real completed run, so the warning cannot come back
unnoticed. They also pin the *semantics* of the fix: a blank flag counts as
False, never as a silent success.
"""
from __future__ import annotations

import importlib
import sys
import warnings

import numpy as np
import pandas as pd



def _mod(name):
    key = f"src.{name}"
    return importlib.reload(sys.modules[key]) if key in sys.modules else importlib.import_module(key)


def _mixed_dtype_well_table(tmp_path):
    """A well-level table matching the real run's dtypes.

    Ridge rows leave the dip-only diagnostic columns blank, which is exactly
    what forces `alignment_ok`/`alignment_cache_hit` to object dtype after the
    CSV round-trip that the report writer performs.
    """
    rows = []
    for protocol in ("same_well_masked", "unseen_well"):
        for rank in range(4):
            common = {
                "protocol": protocol, "well_id": f"{protocol[:2]}{rank}",
                "n_points": 100 + rank, "suffix_len": 100 + rank,
                "prefix_len": 500 + rank, "gr_missing_frac": 0.01 * rank,
                "sse": (rank + 1) * 100.0, "rmse": float(rank + 1),
                "max_abs_error": float(rank + 2), "bias": 0.0, "fold": rank,
                "has_typewell": True, "target_min": -10.0, "target_max": 10.0,
                "target_range": 20.0, "scored_exact_suffix": True,
                "trajectory_curvature_deg_per_1000ft": float(rank),
                "alignment_confidence_mean": 0.5, "alignment_confidence_p10": 0.4,
            }
            # Ridge: no dip diagnostics at all -> blanks in the CSV.
            rows.append({"model": "ridge", **common})
            rows.append({"model": "ridge_spatial", **common, "sse": common["sse"] * 1.01})
            # The rejected model: some wells failed alignment, one has no flag.
            rows.append({
                "model": "dip_constrained_alignment", **common,
                "sse": common["sse"] * 9.0,
                "fallback_points": rank,
                "alignment_ok": None if rank == 3 else bool(rank % 2),
                "alignment_failure_reason": "" if rank % 2 else "no_valid_gr_typewell_match",
                "alignment_cache_hit": None if rank == 3 else bool(rank % 2),
            })
    pd.DataFrame(rows).to_csv(tmp_path / "well_level_validation.csv", index=False)
    pd.DataFrame(
        [
            {"protocol": p, "model": m, "global_rmse": 1.0, "median_well_rmse": 1.0,
             "worst10_well_rmse": 1.0}
            for p in ("same_well_masked", "unseen_well")
            for m in ("ridge", "ridge_spatial")
        ]
    ).to_csv(tmp_path / "validation_results.csv", index=False)
    return tmp_path


def test_write_real_analysis_emits_no_future_warning(tmp_path):
    real = _mod("real_reporting")
    root = _mixed_dtype_well_table(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        written = real.write_real_analysis(root)
    assert {"dip_constrained_alignment_real.md", "protocol_comparison_real.md"} <= {
        p.name for p in written
    }


def test_the_object_dtype_path_is_actually_exercised(tmp_path):
    """Guard the guard: if the fixture stopped producing object dtype, the
    warning test above would pass for the wrong reason."""
    root = _mixed_dtype_well_table(tmp_path)
    well = pd.read_csv(root / "well_level_validation.csv")
    assert well["alignment_ok"].dtype == object
    assert well["alignment_cache_hit"].dtype == object
    assert well["alignment_ok"].isna().any()


def test_bool_column_treats_missing_as_false_not_true():
    real = _mod("real_reporting")
    frame = pd.DataFrame({"flag": [True, False, None, np.nan, "True", "false", 1, 0]})
    out = real._bool_column(frame, "flag")
    assert out.dtype == bool
    assert out.tolist() == [True, False, False, False, True, False, True, False]


def test_bool_column_handles_a_genuinely_boolean_column():
    real = _mod("real_reporting")
    frame = pd.DataFrame({"flag": pd.Series([True, False, True], dtype=bool)})
    out = real._bool_column(frame, "flag")
    assert out.dtype == bool
    assert out.tolist() == [True, False, True]


def test_bool_column_defaults_to_false_for_a_missing_column():
    real = _mod("real_reporting")
    frame = pd.DataFrame({"other": [1, 2, 3]})
    out = real._bool_column(frame, "absent")
    assert out.dtype == bool
    assert not out.any()
    assert len(out) == 3


def test_failure_counts_use_the_coerced_flags(tmp_path):
    """A well whose alignment_ok is blank must be counted as a failure."""
    real = _mod("real_reporting")
    root = _mixed_dtype_well_table(tmp_path)
    real.write_real_analysis(root)
    ablation = pd.read_csv(root / "dip_constrained_alignment_ablation.csv")
    # Per protocol: ranks 0 and 2 are False, rank 3 is blank -> 3 failures.
    assert set(ablation["alignment_failure_wells"]) == {3}
    # Only rank 1 has a True cache hit.
    assert set(ablation["cache_hit_wells"]) == {1}


def test_blank_failure_reason_is_labelled_unspecified(tmp_path):
    real = _mod("real_reporting")
    root = _mixed_dtype_well_table(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        real.write_real_analysis(root)
    text = (root / "dip_constrained_alignment_real.md").read_text()
    assert "unspecified" in text
    assert "no_valid_gr_typewell_match" in text
