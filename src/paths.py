"""Canonical paths for the ROGII wellbore geology prediction task.

Single source of truth. Safe to import from a Kaggle Notebook cell: nothing at
module level depends on ``__file__`` (which is undefined inside a notebook
cell), and no directory is created on import.

Overrides, used only by the test-suite fixtures, never in production:

* ``ROGII_KAGGLE_ROOT``  -> relocates ``/kaggle``
* ``ROGII_REPORTS_DIR``  -> relocates the reports output directory
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Kaggle mount ---------------------------------------------------------
KAGGLE_ROOT = Path(os.environ.get("ROGII_KAGGLE_ROOT", "/kaggle"))

COMPETITION_ROOT = KAGGLE_ROOT / "input" / "competitions" / "rogii-wellbore-geology-prediction"
TRAIN_DIR = COMPETITION_ROOT / "train"
TEST_DIR = COMPETITION_ROOT / "test"
SAMPLE_SUBMISSION = COMPETITION_ROOT / "sample_submission.csv"
TASK_PPTX = COMPETITION_ROOT / "AI_wellbore_geology_prediction_task_en.pptx"

# --- auxiliary mounted datasets ------------------------------------------
DATASETS_ROOT = KAGGLE_ROOT / "input" / "datasets"
KOOLBOX_DIR = DATASETS_ROOT / "phongnguyn23021656" / "koolbox-offline"
CLAUDE_MODELS_DIR = DATASETS_ROOT / "fleongg" / "rogii-claude-models-pub"
ARTIFACTS_DIR = DATASETS_ROOT / "ravaghi" / "wellbore-geology-prediction-artifacts"

EXTERNAL_RESOURCES = {
    "koolbox-offline": KOOLBOX_DIR,
    "rogii-claude-models-pub": CLAUDE_MODELS_DIR,
    "wellbore-geology-prediction-artifacts": ARTIFACTS_DIR,
}

# --- outputs --------------------------------------------------------------
WORKING_DIR = KAGGLE_ROOT / "working"
REPORTS_DIR = Path(os.environ.get("ROGII_REPORTS_DIR", str(WORKING_DIR / "reports")))
SUBMISSION_FILENAME = "submission.csv"
SUBMISSION_PATH = WORKING_DIR / SUBMISSION_FILENAME


def ensure_reports_dir() -> Path:
    """Create and return the reports directory (call sites only, never import)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def available(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


class CompetitionDataMissing(RuntimeError):
    """Raised when the competition mount is absent or incomplete."""


def require_competition_data(*, need_sample_submission: bool = False) -> None:
    """Fail fast, and loudly, if the competition dataset is not mounted."""
    problems: list[str] = []
    if not available(COMPETITION_ROOT):
        problems.append(f"competition root not found: {COMPETITION_ROOT}")
    else:
        if not available(TRAIN_DIR):
            problems.append(f"train directory not found: {TRAIN_DIR}")
        if not available(TEST_DIR):
            problems.append(f"test directory not found: {TEST_DIR}")
        if need_sample_submission and not available(SAMPLE_SUBMISSION):
            problems.append(f"sample submission not found: {SAMPLE_SUBMISSION}")
    if problems:
        raise CompetitionDataMissing(
            "Competition data is not mounted.\n  - "
            + "\n  - ".join(problems)
            + "\n\nAttach the competition dataset to the notebook, or set "
            "ROGII_KAGGLE_ROOT to point at a directory laid out like /kaggle."
        )
