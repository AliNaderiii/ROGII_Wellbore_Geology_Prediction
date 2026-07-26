"""Tests for scripts/run_all_audits.py."""
from __future__ import annotations

import importlib

import pytest


def _orch():
    import scripts.run_all_audits
    return importlib.reload(scripts.run_all_audits)


def test_aborts_clearly_without_competition_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ROGII_COMPETITION_ROOT", str(tmp_path / "unmounted"))
    monkeypatch.setenv("ROGII_REPORTS_DIR", str(tmp_path / "reports"))
    import src.paths
    importlib.reload(src.paths)
    m = _orch()
    assert m.main() == 2
    err = capsys.readouterr().err
    assert "AUDIT ABORTED" in err
    assert "not mounted" in err


def test_full_run_generates_reports(mount, capsys):
    m = _orch()
    result = m.run_all()
    assert result["ok"], result["steps"]
    names = {p.rsplit("/", 1)[-1] for p in result["reports"]}
    # sections 1, 3 and 7 must always be produced from this fixture
    assert {"input_inventory.md", "dataset_schema.csv", "well_summary.csv",
            "data_quality_initial.md"} <= names
    assert "sample_submission_audit.md" in names
    assert "decision_table.md" in names
    out = capsys.readouterr().out
    assert "REPORTS GENERATED" in out


def test_optional_steps_skip_when_inputs_absent(mount, capsys):
    """No .pptx and no auxiliary datasets in the fixture -> skip, not fail."""
    m = _orch()
    result = m.run_all()
    statuses = {s["step"]: s["status"] for s in result["steps"]}
    assert statuses["task presentation audit"] == "SKIPPED"
    assert statuses["external artifact + leakage audit"] == "SKIPPED"
    assert result["ok"]


def test_decision_table_not_overwritten(mount):
    m = _orch()
    from src.paths import REPORTS_DIR
    m.run_all(verbose=False)
    dt = REPORTS_DIR / "decision_table.md"
    dt.write_text("# my curated decisions\n", encoding="utf-8")
    m.run_all(verbose=False)
    assert dt.read_text() == "# my curated decisions\n"


def test_find_repo_root_without_file(mount, monkeypatch):
    m = _orch()
    monkeypatch.delenv("ROGII_REPO_ROOT", raising=False)
    root = m.find_repo_root()
    assert (root / "src" / "paths.py").exists()


def test_reports_go_to_configured_directory(mount):
    m = _orch()
    from src.paths import REPORTS_DIR
    result = m.run_all(verbose=False)
    assert result["reports_dir"] == str(REPORTS_DIR)
    assert "working/reports" in str(REPORTS_DIR)
    assert all(p.startswith(str(REPORTS_DIR)) for p in result["reports"])
