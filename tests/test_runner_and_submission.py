"""End-to-end runner execution, real-path resolution and submission schema."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod(name):
    """Import (or reload after the `mount` fixture patched the environment)."""
    import importlib

    key = f"src.{name}"
    if key in sys.modules:
        return importlib.reload(sys.modules[key])
    return importlib.import_module(key)


# ------------------------------------------------------- path resolution ----

def test_kaggle_competition_paths_are_the_documented_ones():
    """The runner must target the real Kaggle mount when it exists."""
    import src.paths as paths

    assert str(paths.KAGGLE_COMPETITION_ROOT) == (
        "/kaggle/input/competitions/rogii-wellbore-geology-prediction"
    )
    assert paths.KAGGLE_COMPETITION_ROOT.name == paths.COMPETITION_SLUG
    assert str(paths.KAGGLE_REPORTS_DIR) == "/kaggle/working/reports"


def test_competition_root_prefers_env_then_kaggle(tmp_path, monkeypatch):
    import importlib

    import src.paths as paths

    comp = tmp_path / "comp"
    (comp / "train").mkdir(parents=True)
    (comp / "test").mkdir(parents=True)
    monkeypatch.setenv("ROGII_COMPETITION_ROOT", str(comp))
    reloaded = importlib.reload(paths)
    assert reloaded.COMPETITION_ROOT == comp.resolve()
    assert reloaded.COMPETITION_ROOT_SOURCE == "env:ROGII_COMPETITION_ROOT"
    assert reloaded.TRAIN_DIR == comp.resolve() / "train"
    assert reloaded.TEST_DIR == comp.resolve() / "test"


def test_discovery_finds_wells_under_the_kaggle_layout(mount):
    """train/ and test/ are discovered through the identical code path."""
    d = _mod("data")
    train = d.discover_wells("train")
    test = d.discover_wells("test")
    assert train and test
    assert all(w.horizontal is not None for w in train.values())
    assert all(w.horizontal is not None for w in test.values())


# ------------------------------------------------------------ the runner ----

def _run(mount, extra_args, reports):
    comp = mount / "input" / "competitions" / "rogii-wellbore-geology-prediction"
    env = {
        **os.environ,
        "ROGII_COMPETITION_ROOT": str(comp),
        "ROGII_REPORTS_DIR": str(reports),
        "ROGII_REPO_ROOT": str(ROOT),
        "PYTHONPATH": str(ROOT),
    }
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_validation.py"), *extra_args],
        capture_output=True, text=True, env=env, timeout=900,
    )


def test_runner_produces_every_required_report(mount, tmp_path):
    reports = tmp_path / "reports"
    proc = _run(
        mount,
        ["--n-splits", "2", "--models", "hold_last,linear_extrap", "--quiet"],
        reports,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in (
        "feature_manifest.csv",
        "validation_results.csv",
        "well_level_validation.csv",
        "stratified_validation.csv",
        "validation_failures.csv",
        "baseline_report.md",
        "validation_protocol_run.md",
        "run_environment.json",
    ):
        assert (reports / name).exists(), f"missing {name}"


def test_runner_reports_both_protocols_separately(mount, tmp_path):
    v = _mod("validation")
    reports = tmp_path / "reports"
    proc = _run(mount, ["--n-splits", "2", "--models", "hold_last", "--quiet"], reports)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    res = pd.read_csv(reports / "validation_results.csv")
    protocols = set(res["protocol"])
    assert v.PROTOCOL_A in protocols
    assert v.PROTOCOL_B in protocols
    # never merged into a single unexplained row
    assert len(res[res["protocol"] == v.PROTOCOL_A]) >= 1
    assert len(res[res["protocol"] == v.PROTOCOL_B]) >= 1


def test_runner_records_required_metrics(mount, tmp_path):
    reports = tmp_path / "reports"
    proc = _run(mount, ["--n-splits", "2", "--models", "hold_last", "--quiet"], reports)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    res = pd.read_csv(reports / "validation_results.csv")
    for col in (
        "global_rmse", "mean_well_rmse", "median_well_rmse",
        "worst10_well_rmse", "n_wells", "n_points",
    ):
        assert col in res.columns, col

    strat = pd.read_csv(reports / "stratified_validation.csv")
    assert set(strat["stratify_by"]) >= {
        "hidden_suffix_length", "gr_missingness", "prefix_length"
    }

    env = json.loads((reports / "run_environment.json").read_text())
    for key in (
        "runtime_seconds", "peak_rss_mb", "n_failures",
        "n_wells_validated", "n_points_evaluated", "cross_fitted",
    ):
        assert key in env, key
    assert env["cross_fitted"] is True
    assert env["blocked_wells_in_validation"] == 0


def test_runner_expect_train_guard_rejects_a_partial_mount(mount, tmp_path):
    proc = _run(
        mount,
        ["--expect-train", "773", "--models", "hold_last", "--quiet"],
        tmp_path / "reports",
    )
    assert proc.returncode != 0
    assert "expected 773 train wells" in (proc.stdout + proc.stderr)


def test_runner_never_writes_a_blocked_well_into_results(mount, tmp_path):
    v = _mod("validation")
    reports = tmp_path / "reports"
    proc = _run(mount, ["--n-splits", "2", "--models", "hold_last", "--quiet"], reports)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    well = pd.read_csv(reports / "well_level_validation.csv")
    assert not set(well["well_id"].astype(str)) & set(v.BLOCKED_WELL_IDS)


def test_synthetic_label_stamps_the_banner(mount, tmp_path):
    reports = tmp_path / "reports"
    proc = _run(
        mount,
        ["--n-splits", "2", "--models", "hold_last", "--quiet",
         "--label", "SYNTHETIC PIPELINE VERIFICATION ONLY"],
        reports,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = (reports / "baseline_report.md").read_text()
    assert "SYNTHETIC PIPELINE VERIFICATION ONLY" in report
    assert "NOT competition validation results" in report


def test_unlabelled_run_carries_no_synthetic_banner(mount, tmp_path):
    reports = tmp_path / "reports"
    proc = _run(mount, ["--n-splits", "2", "--models", "hold_last", "--quiet"], reports)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SYNTHETIC" not in (reports / "baseline_report.md").read_text()


# --------------------------------------------------------- submission -------

def test_submission_schema_matches_sample(mount, sample_submission_path, tmp_path):
    sub = _mod("submission")
    spec = sub.audit_sample_submission(sample_submission_path)
    assert spec.n_rows > 0
    preds = {i: 0.5 for i in spec.id_order}
    frame = sub.build_submission(preds, sample_submission_path)
    rep = sub.validate_submission(frame, sample_submission_path)
    assert rep.passed, str(rep)
    assert list(frame.columns) == spec.columns
    assert frame[spec.id_column].tolist() == spec.id_order


def test_submission_rejects_wrong_row_count(mount, sample_submission_path):
    sub = _mod("submission")
    spec = sub.audit_sample_submission(sample_submission_path)
    frame = pd.DataFrame(
        {spec.id_column: spec.id_order[:-1],
         spec.value_columns[0]: [0.1] * (len(spec.id_order) - 1)}
    )
    assert not sub.validate_submission(frame, sample_submission_path).passed


def test_submission_rejects_nan_predictions(mount, sample_submission_path):
    sub = _mod("submission")
    spec = sub.audit_sample_submission(sample_submission_path)
    vals = [0.1] * len(spec.id_order)
    vals[0] = np.nan
    frame = pd.DataFrame({spec.id_column: spec.id_order, spec.value_columns[0]: vals})
    assert not sub.validate_submission(frame, sample_submission_path).passed


def test_submission_rejects_reordered_ids(mount, sample_submission_path):
    sub = _mod("submission")
    spec = sub.audit_sample_submission(sample_submission_path)
    ids = list(reversed(spec.id_order))
    frame = pd.DataFrame(
        {spec.id_column: ids, spec.value_columns[0]: [0.1] * len(ids)}
    )
    assert not sub.validate_submission(frame, sample_submission_path).passed


def test_submission_ids_cover_only_the_hidden_region(mount, sample_submission_path):
    """The sample submission scores the suffix, so ids must start at the boundary."""
    sub = _mod("submission")
    d = _mod("data")
    spec = sub.audit_sample_submission(sample_submission_path)
    test_files = d.discover_wells("test")
    for well_id, n_rows in spec.rows_per_well.items():
        well = d.load_well(test_files[well_id])
        assert n_rows == int(well.hidden_mask.sum()), well_id
