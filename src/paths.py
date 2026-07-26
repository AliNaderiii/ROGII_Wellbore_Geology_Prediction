"""Canonical paths for the ROGII wellbore geology prediction task.

Resolution priority for the competition root:

1. ``$ROGII_COMPETITION_ROOT``      — explicit override (CI, local dev, tests)
2. ``/kaggle/input/competitions/rogii-wellbore-geology-prediction``  — Kaggle
3. a local repository fallback (``<repo>/data/...``) — local development only

Reports default to ``/kaggle/working/reports`` on Kaggle, and to
``<repo>/reports`` when the Kaggle working directory does not exist, so local
runs never try to write into a non-existent ``/kaggle``.

Safe to import from a Kaggle Notebook cell: nothing here needs ``__file__``
(undefined in a cell). ``REPORTS_DIR`` itself *is* created at import time
(via ``mkdir(parents=True, exist_ok=True)``) so downstream code and path
validation can always rely on it existing; this is the only directory
created as a side effect of importing this module.
"""
from __future__ import annotations

import os
from pathlib import Path

COMPETITION_SLUG = "rogii-wellbore-geology-prediction"

# The exact Kaggle mount points, kept as named constants so they can be
# asserted against in tests and printed in reports.
KAGGLE_ROOT = Path("/kaggle")
KAGGLE_INPUT = KAGGLE_ROOT / "input"
KAGGLE_WORKING = KAGGLE_ROOT / "working"
KAGGLE_COMPETITION_ROOT = KAGGLE_INPUT / "competitions" / COMPETITION_SLUG
KAGGLE_REPORTS_DIR = KAGGLE_WORKING / "reports"


def _repo_root() -> Path:
    """Best-effort repository root, without requiring ``__file__``."""
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # pasted into a notebook cell
        cwd = Path.cwd().resolve()
        for base in (cwd, *cwd.parents):
            if (base / "src" / "paths.py").exists():
                return base
        return cwd


REPO_ROOT = _repo_root()


def _resolve_competition_root() -> tuple[Path, str]:
    """Return (path, how_it_was_chosen)."""
    env = os.environ.get("ROGII_COMPETITION_ROOT")
    if env:
        return Path(env).expanduser().resolve(), "env:ROGII_COMPETITION_ROOT"

    if KAGGLE_COMPETITION_ROOT.exists():
        return KAGGLE_COMPETITION_ROOT, "kaggle-mount"

    # Local development fallbacks, only used when nothing above exists.
    for candidate in (
        REPO_ROOT / "data" / "competitions" / COMPETITION_SLUG,
        REPO_ROOT / "data" / COMPETITION_SLUG,
        REPO_ROOT / "data",
    ):
        if candidate.exists():
            return candidate.resolve(), "local-fallback"

    # Nothing exists yet: still report the Kaggle path, so error messages name
    # the location the user actually expects.
    return KAGGLE_COMPETITION_ROOT, "kaggle-default (not present)"


COMPETITION_ROOT, COMPETITION_ROOT_SOURCE = _resolve_competition_root()

TRAIN_DIR = COMPETITION_ROOT / "train"
TEST_DIR = COMPETITION_ROOT / "test"
SAMPLE_SUBMISSION = COMPETITION_ROOT / "sample_submission.csv"
TASK_PPTX = COMPETITION_ROOT / "AI_wellbore_geology_prediction_task_en.pptx"


def _resolve_reports_dir() -> Path:
    env = os.environ.get("ROGII_REPORTS_DIR")
    if env:
        return Path(env).expanduser()
    if KAGGLE_WORKING.exists():
        return KAGGLE_REPORTS_DIR
    return REPO_ROOT / "reports"


REPORTS_DIR = _resolve_reports_dir()
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- auxiliary mounted datasets ------------------------------------------
DATASETS_ROOT = Path(os.environ.get("ROGII_DATASETS_ROOT", str(KAGGLE_INPUT / "datasets")))
KOOLBOX_DIR = DATASETS_ROOT / "phongnguyn23021656" / "koolbox-offline"
CLAUDE_MODELS_DIR = DATASETS_ROOT / "fleongg" / "rogii-claude-models-pub"
ARTIFACTS_DIR = DATASETS_ROOT / "ravaghi" / "wellbore-geology-prediction-artifacts"

EXTERNAL_RESOURCES = {
    "koolbox-offline": KOOLBOX_DIR,
    "rogii-claude-models-pub": CLAUDE_MODELS_DIR,
    "wellbore-geology-prediction-artifacts": ARTIFACTS_DIR,
}

SUBMISSION_FILENAME = "submission.csv"
SUBMISSION_PATH = (KAGGLE_WORKING if KAGGLE_WORKING.exists() else REPO_ROOT) / SUBMISSION_FILENAME


def ensure_reports_dir() -> Path:
    """Return the reports directory, (re)creating it if needed.

    ``REPORTS_DIR`` is already created at import time, so this is mostly a
    convenience for call sites that want an explicit, idempotent guarantee
    (e.g. after deleting the directory mid-run).
    """
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
            + f"\n\nResolved via: {COMPETITION_ROOT_SOURCE}\n"
            "Attach the competition dataset to the notebook, or set "
            "ROGII_COMPETITION_ROOT to the directory containing train/ and test/."
        )


def describe_paths() -> str:
    """Human-readable path resolution summary, printed by the orchestrator."""
    rows = [
        ("COMPETITION_ROOT", COMPETITION_ROOT, available(COMPETITION_ROOT)),
        ("TRAIN_DIR", TRAIN_DIR, available(TRAIN_DIR)),
        ("TEST_DIR", TEST_DIR, available(TEST_DIR)),
        ("SAMPLE_SUBMISSION", SAMPLE_SUBMISSION, available(SAMPLE_SUBMISSION)),
        ("TASK_PPTX", TASK_PPTX, available(TASK_PPTX)),
        ("REPORTS_DIR", REPORTS_DIR, available(REPORTS_DIR)),
    ]
    width = max(len(n) for n, _, _ in rows)
    lines = [f"path resolution ({COMPETITION_ROOT_SOURCE}):"]
    lines += [
        f"  {name:<{width}}  {'OK     ' if ok else 'MISSING'}  {path}"
        for name, path, ok in rows
    ]
    return "\n".join(lines)
