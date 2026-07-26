"""Tests for the submission validator (src/submission.py)."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _sub():
    import src.submission
    return importlib.reload(src.submission)


def _good(sample_submission_path) -> pd.DataFrame:
    df = pd.read_csv(sample_submission_path)
    df["tvt"] = np.linspace(10.0, 30.0, len(df))
    return df


def _failed(report, check_name: str) -> bool:
    return any(c.name == check_name and not c.passed for c in report.checks)


def test_valid_submission_passes(sample_submission_path, tmp_path):
    m = _sub()
    p = tmp_path / "submission.csv"
    _good(sample_submission_path).to_csv(p, index=False)
    rep = m.validate_submission(p, sample_submission_path)
    assert rep.passed, str(rep)
    assert rep.result == "PASS"
    assert rep.to_dict()["n_errors"] == 0


def test_duplicate_ids_fail(sample_submission_path, tmp_path):
    m = _sub()
    df = _good(sample_submission_path)
    df.loc[1, "id"] = df.loc[0, "id"]
    rep = m.validate_submission(df, sample_submission_path)
    assert not rep.passed
    assert _failed(rep, "no_duplicate_ids")


def test_wrong_order_fails(sample_submission_path):
    m = _sub()
    df = _good(sample_submission_path).sample(frac=1.0, random_state=7)
    rep = m.validate_submission(df, sample_submission_path)
    assert not rep.passed
    assert _failed(rep, "id_order")
    # the same set in a different order is only an ordering problem
    assert not _failed(rep, "no_missing_ids")
    assert not _failed(rep, "no_unknown_ids")


def test_order_can_be_waived(sample_submission_path):
    m = _sub()
    df = _good(sample_submission_path).sample(frac=1.0, random_state=7)
    rep = m.validate_submission(df, sample_submission_path, require_exact_order=False)
    assert rep.passed


def test_nan_predictions_fail(sample_submission_path):
    m = _sub()
    df = _good(sample_submission_path)
    df.loc[3, "tvt"] = np.nan
    rep = m.validate_submission(df, sample_submission_path)
    assert not rep.passed
    assert _failed(rep, "no_nan_predictions")


def test_infinite_predictions_fail(sample_submission_path):
    m = _sub()
    df = _good(sample_submission_path)
    df.loc[5, "tvt"] = np.inf
    rep = m.validate_submission(df, sample_submission_path)
    assert not rep.passed
    assert _failed(rep, "no_infinite_predictions")


def test_unknown_ids_fail(sample_submission_path):
    m = _sub()
    df = _good(sample_submission_path)
    df.loc[0, "id"] = "NOT_A_REAL_ID_999"
    rep = m.validate_submission(df, sample_submission_path)
    assert not rep.passed
    assert _failed(rep, "no_unknown_ids")
    assert _failed(rep, "no_missing_ids")


def test_wrong_row_count_fails(sample_submission_path):
    m = _sub()
    df = _good(sample_submission_path).iloc[:-3]
    rep = m.validate_submission(df, sample_submission_path)
    assert not rep.passed
    assert _failed(rep, "row_count")


def test_wrong_columns_fail(sample_submission_path):
    m = _sub()
    df = _good(sample_submission_path).rename(columns={"tvt": "prediction"})
    rep = m.validate_submission(df, sample_submission_path)
    assert not rep.passed
    assert _failed(rep, "exact_columns")
    assert _failed(rep, "no_missing_columns")


def test_unexpected_extra_column_fails(sample_submission_path):
    m = _sub()
    df = _good(sample_submission_path)
    df["debug_feature"] = 1.0
    rep = m.validate_submission(df, sample_submission_path)
    assert not rep.passed
    assert _failed(rep, "no_unexpected_columns")


def test_non_numeric_predictions_fail(sample_submission_path):
    m = _sub()
    df = _good(sample_submission_path).astype({"tvt": object})
    df.loc[2, "tvt"] = "not_a_number"
    rep = m.validate_submission(df, sample_submission_path)
    assert not rep.passed
    assert _failed(rep, "numeric_dtype")


def test_placeholder_overwrite_warns_not_fails(sample_submission_path):
    """An unmodified sample is structurally valid but almost certainly a mistake."""
    m = _sub()
    rep = m.validate_submission(sample_submission_path, sample_submission_path)
    assert rep.passed                      # no structural error
    assert _failed(rep, "not_sample_placeholder")
    assert _failed(rep, "not_constant")
    assert len(rep.warnings) >= 2


def test_missing_submission_file_reports_cleanly(tmp_path, sample_submission_path):
    m = _sub()
    rep = m.validate_submission(tmp_path / "does_not_exist.csv", sample_submission_path)
    assert not rep.passed
    assert _failed(rep, "submission_readable")


def test_filename_warning(sample_submission_path, tmp_path):
    m = _sub()
    p = tmp_path / "my_preds.csv"
    _good(sample_submission_path).to_csv(p, index=False)
    rep = m.validate_submission(p, sample_submission_path)
    assert rep.passed                      # wrong name is a warning, not an error
    assert _failed(rep, "output_filename")


def test_write_submission_refuses_invalid(sample_submission_path, tmp_path):
    m = _sub()
    df = _good(sample_submission_path)
    df.loc[0, "tvt"] = np.nan
    with pytest.raises(ValueError, match="FAILED validation"):
        m.write_submission(df, tmp_path / "submission.csv",
                           sample_submission_path=sample_submission_path)
    assert not (tmp_path / "submission.csv").exists()


def test_write_submission_writes_valid(sample_submission_path, tmp_path):
    m = _sub()
    out = m.write_submission(_good(sample_submission_path), tmp_path / "submission.csv",
                             sample_submission_path=sample_submission_path)
    assert out.exists()
    assert m.validate_submission(out, sample_submission_path).passed


def test_cli_returns_exit_codes(sample_submission_path, tmp_path, capsys):
    m = _sub()
    good = tmp_path / "submission.csv"
    _good(sample_submission_path).to_csv(good, index=False)
    assert m.main(["--submission", str(good),
                   "--sample-submission", str(sample_submission_path)]) == 0
    assert "PASS" in capsys.readouterr().out

    bad = tmp_path / "bad.csv"
    df = _good(sample_submission_path)
    df.loc[0, "tvt"] = np.nan
    df.to_csv(bad, index=False)
    assert m.main(["--submission", str(bad),
                   "--sample-submission", str(sample_submission_path)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_json_output(sample_submission_path, tmp_path, capsys):
    import json
    m = _sub()
    p = tmp_path / "submission.csv"
    _good(sample_submission_path).to_csv(p, index=False)
    m.main(["--submission", str(p), "--sample-submission", str(sample_submission_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"] == "PASS"
    assert payload["n_errors"] == 0
    assert isinstance(payload["checks"], list)
