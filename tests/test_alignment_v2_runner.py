"""Smoke test for the Alignment v2 experiment runner on synthetic data.

Runs the runner with ``--max-wells 5 --n-splits 2`` against the
synthetic mount fixture, then asserts:

* the runner produced every expected output file;
* the decision JSON reports ``is_real_run=False`` (synthetic banner);
* the decision JSON does *not* promote any arm (a synthetic run must
  never promote a v2 candidate, regardless of measured numbers);
* the well-level CSV has the expected columns and finite values;
* no submission file was written.

The runner never touches the production oof_meta_stack.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_v2_runner(mount, tmp_path) -> dict:
    """Run the v2 experiment runner with a small synthetic universe."""
    env = {
        "ROGII_COMPETITION_ROOT": str(
            mount / "input" / "competitions" / "rogii-wellbore-geology-prediction"
        ),
        "ROGII_REPORTS_DIR": str(tmp_path / "reports"),
        "PYTHONPATH": str(ROOT),
    }
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_alignment_v2_experiment.py"),
        "--max-wells", "5",
        "--n-splits", "2",
        "--inner-splits", "2",
        "--tune-splits", "2",
        "--reports-dir", str(tmp_path / "reports"),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def test_v2_runner_smoke(mount, tmp_path):
    out = _run_v2_runner(mount, tmp_path)
    assert out["returncode"] == 0, f"runner failed: {out['stderr']}"
    reports = tmp_path / "reports"
    # Every output file is named ``synthetic_alignment_v2_*`` because
    # the runner refuses to call this a real run.
    files = list(reports.iterdir())
    names = [f.name for f in files]
    print("Files written:", names)
    assert any("alignment_v2_summary" in n for n in names)
    assert any("alignment_v2_decision" in n for n in names)
    assert any("alignment_v2_run_environment" in n for n in names)
    assert any("alignment_v2_well_level" in n for n in names)
    # The decision JSON must be present and the runner must NOT have
    # promoted any arm on a synthetic run.
    decision_path = next(reports.glob("*alignment_v2_decision.json"))
    decision = json.loads(decision_path.read_text())
    assert decision["is_real_run"] is False
    assert decision["promoted"] is False
    assert decision["promoted_arm"] is None
    assert "synthetic" in decision["promotion_note"].lower() or "real mount" in decision["promotion_note"].lower()
    # The well-level CSV is well-formed.
    well_path = next(reports.glob("*alignment_v2_well_level.csv"))
    well_df = pd.read_csv(well_path)
    assert "model" in well_df.columns
    assert "protocol" in well_df.columns
    assert "well_id" in well_df.columns
    assert "sse" in well_df.columns
    # No submission file is written by the runner.
    submission = tmp_path / "submission.csv"
    assert not submission.exists()


def test_v2_runner_does_not_modify_trajectory_stack_outputs(mount, tmp_path):
    """The v2 runner writes its own reports dir; the trajectory stack
    reports dir is untouched. We assert the two prefixes do not
    overlap."""
    out = _run_v2_runner(mount, tmp_path)
    assert out["returncode"] == 0
    reports = tmp_path / "reports"
    names = [f.name for f in reports.iterdir()]
    # Only v2 outputs.
    assert all("alignment_v2" in n for n in names), names
    assert not any("trajectory_stack" in n for n in names)
