"""Tests for the final submission pipeline (scripts/build_final_submission.py).

Uses the synthetic mount fixture (never real data) to verify production-quality
behaviour: dynamic path resolution, forbidden-feature guard, prefix preservation,
ID order, audit metadata, fail-closed handling and reproducibility.
"""

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
SCRIPT = ROOT / "scripts" / "build_final_submission.py"


def _run_final(mount, output_dir, extra_args=None):
    """Run build_final_submission.py as subprocess with synthetic mount env."""
    comp = mount / "input" / "competitions" / "rogii-wellbore-geology-prediction"
    env = {
        **os.environ,
        "ROGII_COMPETITION_ROOT": str(comp),
        "ROGII_DATASETS_ROOT": str(mount / "input" / "datasets"),
        "ROGII_REPORTS_DIR": str(mount / "working" / "reports"),
        "ROGII_REPO_ROOT": str(ROOT),
        "PYTHONPATH": str(ROOT),
    }
    cmd = [sys.executable, str(SCRIPT), "--output-dir", str(output_dir)]
    if extra_args:
        cmd += extra_args
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)


def test_final_pipeline_produces_valid_submission(mount, tmp_path):
    out = tmp_path / "final"
    proc = _run_final(mount, out, ["--expect-train", "9", "--expect-test", "2"])
    assert proc.returncode == 0, proc.stdout + proc.stderr

    sub_path = out / "submission.csv"
    audit_path = out / "submission_audit.json"
    assert sub_path.exists()
    assert audit_path.exists()

    sample_path = mount / "input" / "competitions" / "rogii-wellbore-geology-prediction" / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    sub = pd.read_csv(sub_path)

    # exact columns
    assert list(sub.columns) == ["id", "tvt"]
    # exact row count
    assert len(sub) == len(sample)
    # exact ID order
    assert sub["id"].tolist() == sample["id"].tolist()
    # no duplicate IDs
    assert sub["id"].duplicated().sum() == 0
    # no missing IDs
    assert set(sub["id"]) == set(sample["id"])
    # no NaN / inf / finite numeric
    assert sub["tvt"].isna().sum() == 0
    vals = pd.to_numeric(sub["tvt"], errors="coerce")
    assert vals.isna().sum() == 0
    assert np.all(np.isfinite(vals.to_numpy()))
    # not constant placeholder
    assert vals.nunique() > 1


def test_final_pipeline_audit_metadata(mount, tmp_path):
    out = tmp_path / "final"
    proc = _run_final(mount, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    audit = json.loads((out / "submission_audit.json").read_text())
    # Required flags
    assert audit["data_source"] == "real_kaggle"
    assert audit["synthetic_generator_used"] is False
    assert audit["final_model"] == "Ridge Default"
    assert audit["alignment_features"] is False
    assert audit["spatial_features"] is False
    assert audit["final_submission_authorized"] is True
    assert audit["external_artifacts_used"] is False

    # Counts
    assert audit["n_train_wells_discovered"] == 9  # TRW001..009 in mount fixture
    assert audit["n_eligible_training_wells"] == 9  # none of those are blocked (blocked IDs are 000d7d20 etc)
    # In mount fixture, TRW001..009 are eligible; blocked IDs are not present in train.
    # For generic mount we built 9 train wells, 2 test wells.
    assert audit["n_test_wells_discovered"] == 2
    assert audit["n_submission_rows"] == audit["n_predicted_hidden_rows"]
    assert audit["n_submission_rows"] == 300  # 2 wells x 150 rows in mount fixture

    # Feature safety
    feat_list = audit["feature_list"]
    assert isinstance(feat_list, list) and len(feat_list) > 0
    forbidden = ["align_tvt", "align_score", "pf_", "beam_", "nbr_", "geology", "ANCC", "BUDA"]
    for f in feat_list:
        for bad in forbidden:
            assert bad.lower() not in f.lower(), f"forbidden {bad} in {f}"

    # Prediction stats
    for k in ("prediction_min", "prediction_max", "prediction_mean", "prediction_std"):
        assert k in audit and np.isfinite(audit[k])

    # Validation status
    assert audit["submission_validation_status"] == "PASS"
    assert audit["validation_status"] == "PASS"

    # Blocked wells excluded
    assert set(audit["public_duplicate_test_wells_excluded"]) == {"000d7d20", "00bbac68", "00e12e8b"}

    # Model config
    assert audit["model_configuration"]["alignment_features"] is False
    assert audit["model_configuration"]["spatial"] is None


def test_final_pipeline_excludes_blocked_wells(mount, tmp_path):
    """Ensure blocked IDs are not used for training even if present."""
    # In synthetic mount, blocked IDs are not present in train, but we test that
    # final pipeline's guard would reject them if they were.
    # We verify that eligible count == total - blocked (if any).
    import src.data as data_mod
    # Reload after mount env (conftest already reloaded)
    train_files = data_mod.discover_wells("train")
    # Add a fake blocked entry to check filter
    from src.validation import BLOCKED_WELL_IDS, filter_blocked
    all_ids = list(train_files.keys()) + list(BLOCKED_WELL_IDS)
    filtered = filter_blocked(all_ids)
    assert not set(filtered) & set(BLOCKED_WELL_IDS)
    assert len(filtered) == len(train_files)


def test_final_pipeline_expect_test_optional(mount, tmp_path):
    """--expect-test is optional; without it pipeline must still succeed."""
    out = tmp_path / "final_no_expect"
    proc = _run_final(mount, out)  # no expect args
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # With correct expect-test it should also succeed
    out2 = tmp_path / "final_with_expect"
    proc2 = _run_final(mount, out2, ["--expect-test", "2"])
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    # With wrong expect-test it must fail closed
    out3 = tmp_path / "final_wrong_expect"
    proc3 = _run_final(mount, out3, ["--expect-test", "3"])
    assert proc3.returncode != 0


def test_final_pipeline_hidden_suffix_only(mount, tmp_path):
    """Submission rows must correspond exactly to hidden suffix rows per well."""
    out = tmp_path / "final"
    proc = _run_final(mount, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Load test wells and verify hidden counts match sample per-well counts
    import src.data as data_mod
    test_files = data_mod.discover_wells("test")
    sample_path = mount / "input" / "competitions" / "rogii-wellbore-geology-prediction" / "sample_submission.csv"
    sample = pd.read_csv(sample_path)

    # per-well counts from sample
    def split_id(s):
        import re
        m = re.match(r"^(.*?)[_\-:](\d+)$", str(s))
        if m:
            return m.group(1).rstrip("_-:"), m.group(2)
        return str(s), None
    from collections import Counter
    wells = [split_id(i)[0] for i in sample["id"]]
    per_well_sample = Counter(wells)

    for wid, n_sample in per_well_sample.items():
        well = data_mod.load_well(test_files[wid])
        assert n_sample == int(well.hidden_mask.sum()), f"well {wid} sample {n_sample} != hidden {well.hidden_mask.sum()}"


def test_final_pipeline_no_geology_or_markers(mount, tmp_path):
    out = tmp_path / "final"
    proc = _run_final(mount, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    audit = json.loads((out / "submission_audit.json").read_text())
    feat = audit["feature_list"]
    for bad in ["geology", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]:
        for f in feat:
            assert bad.lower() not in f.lower(), f"formation/geology feature leaked: {f} contains {bad}"


def test_final_pipeline_reproducibility(mount, tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    proc1 = _run_final(mount, out1)
    proc2 = _run_final(mount, out2)
    assert proc1.returncode == 0 and proc2.returncode == 0, proc1.stdout + proc1.stderr + proc2.stdout + proc2.stderr
    df1 = pd.read_csv(out1 / "submission.csv")
    df2 = pd.read_csv(out2 / "submission.csv")
    assert df1["id"].tolist() == df2["id"].tolist()
    # Ridge is deterministic; predictions must match exactly
    assert np.allclose(df1["tvt"].to_numpy(), df2["tvt"].to_numpy(), atol=1e-9)


def test_final_pipeline_fail_closed_when_mount_absent(tmp_path):
    """Without a competition mount the script must fail closed (non-zero exit)."""
    env = {**os.environ, "ROGII_COMPETITION_ROOT": str(tmp_path / "nonexistent"), "PYTHONPATH": str(ROOT)}
    # Ensure paths will not find Kaggle dir
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode != 0
    assert "Competition data" in proc.stdout + proc.stderr or "not found" in proc.stdout + proc.stderr.lower()


def test_final_pipeline_dynamic_path_resolution(mount, tmp_path):
    """Script must use env ROGII_COMPETITION_ROOT, not a hardcoded /kaggle path."""
    # mount fixture sets a temporary competition root distinct from /kaggle
    comp = mount / "input" / "competitions" / "rogii-wellbore-geology-prediction"
    assert comp.exists()
    # The script succeeded in previous tests using that temporary path, proving dynamic resolution
    out = tmp_path / "final"
    proc = _run_final(mount, out)
    assert proc.returncode == 0
    # Audit should record dataset_root that matches the temporary mount, not the hardcoded Kaggle path
    audit = json.loads((out / "submission_audit.json").read_text())
    assert str(comp) in audit["dataset_root"] or audit["dataset_root"] == str(comp)
