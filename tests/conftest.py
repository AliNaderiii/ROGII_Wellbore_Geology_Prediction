"""Synthetic fixtures. Used ONLY by tests — never to produce real reports."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def _hw_frame(
    n: int = 200,
    *,
    prefix: int = 50,
    gr_missing: slice | None = None,
    with_target: bool = True,
    with_markers: bool = True,
    internal_gap: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    md = np.arange(n, dtype=float) * 1.0 + 8000.0
    tvt = np.cumsum(rng.normal(0, 0.05, n)) + 20.0
    gr = 60 + 25 * np.sin(md / 40) + rng.normal(0, 5, n)
    if gr_missing is not None:
        gr[gr_missing] = np.nan

    tvt_input = tvt.copy()
    tvt_input[prefix:] = np.nan
    if internal_gap:
        tvt_input[10:20] = np.nan  # hole *inside* the visible prefix

    df = pd.DataFrame({
        "MD": md,
        "X": 1000 + md * 0.3,
        "Y": 2000 + md * 0.1,
        "Z": -md * 0.9,
        "GR": gr,
        "TVT_input": tvt_input,
    })
    if with_markers:
        for i, f in enumerate(FORMATIONS):
            df[f] = 10.0 + 8 * i + rng.normal(0, 0.1, n)
    if with_target:
        df["TVT"] = tvt
    return df


def _tw_frame(seed: int = 0) -> pd.DataFrame:
    tvt = np.linspace(0, 60, 200)
    return pd.DataFrame({
        "TVT": tvt,
        "GR": 50 + 30 * np.sin(tvt / 6),
        "Geology": [FORMATIONS[min(int(t // 10), 5)] for t in tvt],
    })


def write_well(directory: Path, well_id: str, hw: pd.DataFrame, tw: pd.DataFrame | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    hw.to_csv(directory / f"{well_id}__horizontal_well.csv", index=False)
    if tw is not None:
        tw.to_csv(directory / f"{well_id}__typewell.csv", index=False)


@pytest.fixture
def mount(tmp_path: Path, monkeypatch) -> Path:
    """A synthetic /kaggle-like mount with a deliberate variety of wells."""
    root = tmp_path / "kaggle"
    comp = root / "input" / "competitions" / "rogii-wellbore-geology-prediction"
    train, test = comp / "train", comp / "test"

    # 1. ordinary train well
    write_well(train, "TRW001", _hw_frame(seed=1), _tw_frame())
    # 2. train well with high GR missingness (>50%, contiguous)
    write_well(train, "TRW002", _hw_frame(gr_missing=slice(0, 130), seed=2), _tw_frame())
    # 3. train well with a long hidden suffix (small prefix)
    write_well(train, "TRW003", _hw_frame(n=400, prefix=20, seed=3), _tw_frame())
    # 4. train well with an internal TVT_input gap
    write_well(train, "TRW004", _hw_frame(internal_gap=True, seed=4), _tw_frame())
    # 5. train well with no typewell
    write_well(train, "TRW005", _hw_frame(seed=5), None)
    # 6-9. long train wells: prefixes big enough to host a masked suffix, which
    #      is what the same-well masked protocol requires (>= 200 rows + mask).
    #      Four of them, so GroupKFold can form non-empty folds in that mode.
    for i, seed in enumerate((6, 7, 8, 9)):
        write_well(
            train,
            f"TRW00{6 + i}",
            _hw_frame(n=2000, prefix=1400, seed=seed),
            _tw_frame(),
        )

    # test wells: no target, no marker columns (mirrors the reported test schema)
    rows = []
    for i in (1, 2):
        wid = f"TSW00{i}"
        hw = _hw_frame(n=200, prefix=50, with_target=False, with_markers=False, seed=10 + i)
        write_well(test, wid, hw, _tw_frame())
        rows += [{"id": f"{wid}_{r}", "tvt": 0.0} for r in range(50, 200)]
    pd.DataFrame(rows).to_csv(comp / "sample_submission.csv", index=False)

    monkeypatch.setenv("ROGII_COMPETITION_ROOT", str(comp))
    monkeypatch.setenv("ROGII_DATASETS_ROOT", str(root / "input" / "datasets"))
    monkeypatch.setenv("ROGII_REPORTS_DIR", str(root / "working" / "reports"))
    monkeypatch.setenv("ROGII_REPO_ROOT", str(ROOT))

    # src.paths caches module-level constants -> reload after patching env.
    # Order matters: modules are reloaded dependency-first so that classes
    # defined in a reloaded module (e.g. WellFiles, InferenceTask) are the same
    # objects the dependent modules hold references to. Reloading out of order
    # produces two live copies of a class and confusing isinstance failures.
    import importlib

    import src.paths
    importlib.reload(src.paths)
    for mod in (
        "src.discovery",
        "src.columns",
        "src.data",
        "src.manifest",
        "src.tasks",
        "src.features",
        "src.spatial",
        "src.baselines",
        "src.validation",
        "src.reporting",
        "src.submission",
    ):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    return root


@pytest.fixture
def sample_submission_path(mount: Path) -> Path:
    return (
        mount / "input" / "competitions" / "rogii-wellbore-geology-prediction"
        / "sample_submission.csv"
    )
