"""Make `src` importable without relying on __file__.

``__file__`` is undefined when code is pasted into a Kaggle Notebook cell, so
every script imports this helper instead of computing its own path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def find_repo_root() -> Path:
    override = os.environ.get("ROGII_REPO_ROOT")
    if override:
        return Path(override).resolve()

    candidates: list[Path] = []
    try:  # present when executed as a file, absent inside a notebook cell
        here = Path(__file__).resolve()
        candidates.append(here.parent.parent)
    except NameError:
        pass
    cwd = Path.cwd().resolve()
    candidates += [cwd, *cwd.parents]

    for base in candidates:
        if (base / "src" / "paths.py").exists():
            return base
    raise RuntimeError(
        "Could not locate the repository root (no src/paths.py found). "
        "Set ROGII_REPO_ROOT to the checkout directory."
    )


def bootstrap() -> Path:
    root = find_repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
