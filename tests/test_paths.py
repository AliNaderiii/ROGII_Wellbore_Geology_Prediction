"""Tests for path resolution (src/paths.py)."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

KAGGLE = "/kaggle/input/competitions/rogii-wellbore-geology-prediction"


def _paths(monkeypatch, **env):
    for k in ("ROGII_COMPETITION_ROOT", "ROGII_REPORTS_DIR", "ROGII_DATASETS_ROOT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import src.paths
    return importlib.reload(src.paths)


def test_kaggle_defaults_are_exact(monkeypatch):
    """With no override, every path must be the documented Kaggle mount."""
    p = _paths(monkeypatch)
    assert str(p.COMPETITION_ROOT) == KAGGLE
    assert str(p.TRAIN_DIR) == f"{KAGGLE}/train"
    assert str(p.TEST_DIR) == f"{KAGGLE}/test"
    assert str(p.SAMPLE_SUBMISSION) == f"{KAGGLE}/sample_submission.csv"
    assert str(p.TASK_PPTX) == f"{KAGGLE}/AI_wellbore_geology_prediction_task_en.pptx"
    assert str(p.KAGGLE_REPORTS_DIR) == "/kaggle/working/reports"


def test_env_override_takes_priority(monkeypatch, tmp_path):
    comp = tmp_path / "comp"
    (comp / "train").mkdir(parents=True)
    (comp / "test").mkdir(parents=True)
    p = _paths(monkeypatch, ROGII_COMPETITION_ROOT=str(comp))
    assert p.COMPETITION_ROOT == comp.resolve()
    assert p.COMPETITION_ROOT_SOURCE == "env:ROGII_COMPETITION_ROOT"
    assert p.TRAIN_DIR == comp.resolve() / "train"
    p.require_competition_data()          # must not raise


def test_reports_dir_defaults_to_kaggle_working_when_present(monkeypatch):
    """On Kaggle, /kaggle/working exists, so reports go there."""
    import src.paths
    real_exists = Path.exists

    def fake_exists(self):
        if str(self) == "/kaggle/working":
            return True
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    p = _paths(monkeypatch)
    assert str(p.REPORTS_DIR) == "/kaggle/working/reports"


def test_reports_dir_falls_back_locally(monkeypatch):
    """Off Kaggle there is no /kaggle/working, so never write into it."""
    p = _paths(monkeypatch)
    assert "/kaggle/working" not in str(p.REPORTS_DIR)
    assert p.REPORTS_DIR == p.REPO_ROOT / "reports"


def test_reports_dir_env_override(monkeypatch, tmp_path):
    p = _paths(monkeypatch, ROGII_REPORTS_DIR=str(tmp_path / "r"))
    assert p.REPORTS_DIR == tmp_path / "r"
    assert p.ensure_reports_dir().is_dir()


def test_require_competition_data_message(monkeypatch, tmp_path):
    p = _paths(monkeypatch, ROGII_COMPETITION_ROOT=str(tmp_path / "absent"))
    with pytest.raises(p.CompetitionDataMissing) as exc:
        p.require_competition_data()
    msg = str(exc.value)
    assert "not mounted" in msg
    assert "ROGII_COMPETITION_ROOT" in msg


def test_require_sample_submission_optional_flag(monkeypatch, tmp_path):
    comp = tmp_path / "c"
    (comp / "train").mkdir(parents=True)
    (comp / "test").mkdir(parents=True)
    p = _paths(monkeypatch, ROGII_COMPETITION_ROOT=str(comp))
    p.require_competition_data()                       # fine without the sample
    with pytest.raises(p.CompetitionDataMissing, match="sample submission"):
        p.require_competition_data(need_sample_submission=True)


def test_describe_paths_reports_status(monkeypatch, tmp_path):
    comp = tmp_path / "c"
    (comp / "train").mkdir(parents=True)
    (comp / "test").mkdir(parents=True)
    p = _paths(monkeypatch, ROGII_COMPETITION_ROOT=str(comp))
    text = p.describe_paths()
    assert "COMPETITION_ROOT" in text and "TASK_PPTX" in text
    assert "OK" in text and "MISSING" in text          # pptx absent here
