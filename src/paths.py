"""Canonical paths for the ROGII wellbore geology prediction task.

Every script in this repo imports paths from here so there is exactly one
place to change if the mount layout differs.
"""
from __future__ import annotations

import os
from pathlib import Path

# The canonical Kaggle mount. KAGGLE_ROOT is honoured only so the test fixture
# in tests/ can exercise this code off-Kaggle; it is unset in production.
_KAGGLE = Path(os.environ.get("KAGGLE_ROOT", "/kaggle"))

COMPETITION_ROOT = Path(
    "/kaggle/input/competitions/rogii-wellbore-geology-prediction"
) if _KAGGLE == Path("/kaggle") else (
    _KAGGLE / "input" / "competitions" / "rogii-wellbore-geology-prediction"
)

TRAIN_DIR = COMPETITION_ROOT / "train"
TEST_DIR = COMPETITION_ROOT / "test"
SAMPLE_SUBMISSION = COMPETITION_ROOT / "sample_submission.csv"
TASK_PPTX = COMPETITION_ROOT / "AI_wellbore_geology_prediction_task_en.pptx"

DATASETS_ROOT = _KAGGLE / "input" / "datasets"
KOOLBOX_DIR = DATASETS_ROOT / "phongnguyn23021656" / "koolbox-offline"
CLAUDE_MODELS_DIR = DATASETS_ROOT / "fleongg" / "rogii-claude-models-pub"
ARTIFACTS_DIR = DATASETS_ROOT / "ravaghi" / "wellbore-geology-prediction-artifacts"

EXTERNAL_RESOURCES = {
    "koolbox-offline": KOOLBOX_DIR,
    "rogii-claude-models-pub": CLAUDE_MODELS_DIR,
    "wellbore-geology-prediction-artifacts": ARTIFACTS_DIR,
}

# Repository-local outputs (works both on Kaggle and in a git checkout).
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"


def ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def available(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False
