"""Section 8 — Baseline data pipeline.

Design rules enforced here:

* Wells are discovered dynamically (`src.discovery`); no well ID is hardcoded.
* Row order from disk is preserved end to end (`row_index` is materialised).
* Visible (prefix) vs hidden (suffix) regions are labelled explicitly.
* Target availability is validated, not assumed.
* Train and test go through exactly the same code path.
* Pure pandas/numpy, offline, one file at a time — memory friendly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.columns import FORMATION_ORDER, marker_columns, resolve_roles
from src.discovery import WellFiles, discover_wells
from src.paths import TEST_DIR, TRAIN_DIR


@dataclass
class WellData:
    well_id: str
    split: str
    hw: pd.DataFrame                 # horizontal well, original row order
    tw: pd.DataFrame | None          # typewell reference
    roles: dict[str, str] = field(default_factory=dict)
    markers: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------
    def col(self, role: str) -> pd.Series | None:
        name = self.roles.get(role)
        return self.hw[name] if name else None

    @property
    def visible_mask(self) -> np.ndarray:
        return self.hw["is_visible"].to_numpy()

    @property
    def hidden_mask(self) -> np.ndarray:
        return ~self.visible_mask


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _mark_regions(hw: pd.DataFrame, roles: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Add `row_index`, `is_visible`; return region metadata.

    Visible = TVT_input is known. If an explicit Prediction Start column
    exists it takes precedence, because that is the organisers' own boundary.
    """
    hw = hw.copy()
    hw["row_index"] = np.arange(len(hw), dtype=np.int64)

    ps_col = roles.get("prediction_start")
    ti_col = roles.get("tvt_input")

    start: int | None = None
    source = "none"

    if ps_col is not None:
        s = hw[ps_col]
        if pd.api.types.is_bool_dtype(s) or set(pd.unique(s.dropna())) <= {0, 1, True, False}:
            flags = s.fillna(0).astype(bool).to_numpy()
            if flags.any():
                start, source = int(np.argmax(flags)), "prediction_start_flag"
        else:
            val = pd.to_numeric(s.dropna(), errors="coerce")
            md = pd.to_numeric(hw[roles["md"]], errors="coerce") if roles.get("md") else None
            if len(val) and md is not None:
                thr = float(val.iloc[0])
                over = (md >= thr).to_numpy()
                if over.any():
                    start, source = int(np.argmax(over)), "prediction_start_md"

    if start is None and ti_col is not None:
        known = pd.to_numeric(hw[ti_col], errors="coerce").notna().to_numpy()
        if known.all():
            start, source = len(hw), "tvt_input_all_known"
        elif not known.any():
            start, source = 0, "tvt_input_none_known"
        else:
            start, source = int(np.argmax(~known)), "tvt_input_first_gap"

    if start is None:
        start, source = len(hw), "fallback_all_visible"

    visible = np.zeros(len(hw), dtype=bool)
    visible[:start] = True
    hw["is_visible"] = visible

    meta = {
        "prediction_start_row": int(start),
        "prediction_start_source": source,
        "n_rows": int(len(hw)),
        "n_visible": int(visible.sum()),
        "n_hidden": int((~visible).sum()),
    }

    if ti_col is not None:
        known = pd.to_numeric(hw[ti_col], errors="coerce").notna().to_numpy()
        meta["clean_prefix_split"] = bool(known[:start].all() and not known[start:].any())
        meta["n_tvt_input_known"] = int(known.sum())
        meta["tvt_last_known"] = (
            float(pd.to_numeric(hw[ti_col], errors="coerce").iloc[start - 1]) if start > 0 else np.nan
        )
    if roles.get("md"):
        md = pd.to_numeric(hw[roles["md"]], errors="coerce")
        meta["md_monotonic"] = bool(md.is_monotonic_increasing)
        d = md.diff().dropna()
        meta["md_step_median"] = float(d.median()) if len(d) else np.nan
        meta["md_at_prediction_start"] = float(md.iloc[start]) if start < len(md) else np.nan
    return hw, meta


def load_well(files: WellFiles, *, require_typewell: bool = False) -> WellData:
    if files.horizontal is None:
        raise FileNotFoundError(f"{files.well_id}: no horizontal well file")
    hw = _load_csv(files.horizontal)
    roles = resolve_roles(hw.columns)
    markers = marker_columns(hw.columns)

    if "md" not in roles:
        raise ValueError(f"{files.well_id}: cannot resolve an MD column from {list(hw.columns)}")

    hw, meta = _mark_regions(hw, roles)

    tw = None
    if files.typewell is not None:
        tw = _load_csv(files.typewell)
    elif require_typewell:
        raise FileNotFoundError(f"{files.well_id}: no typewell file")

    tgt = roles.get("tvt")
    meta.update({
        "well_id": files.well_id,
        "split": files.split,
        "has_typewell": tw is not None,
        "n_typewell_rows": int(len(tw)) if tw is not None else 0,
        "has_target_column": tgt is not None,
        "n_target_known": int(pd.to_numeric(hw[tgt], errors="coerce").notna().sum()) if tgt else 0,
        "markers_present": ",".join(markers),
        "markers_absent": ",".join(f for f in FORMATION_ORDER if f not in markers),
        "n_png": len(files.images),
    })
    meta["target_available_on_hidden"] = bool(
        tgt is not None
        and meta["n_hidden"] > 0
        and pd.to_numeric(hw.loc[~hw["is_visible"], tgt], errors="coerce").notna().any()
    )
    return WellData(files.well_id, files.split, hw, tw, roles, markers, meta)


def iter_wells(
    split: str = "train",
    *,
    directory: Path | None = None,
    limit: int | None = None,
) -> Iterator[WellData]:
    """Stream wells one at a time — never holds the whole dataset in memory."""
    directory = directory or (TRAIN_DIR if split == "train" else TEST_DIR)
    wells = discover_wells(directory, split)
    for i, (_, files) in enumerate(sorted(wells.items())):
        if limit is not None and i >= limit:
            break
        if files.horizontal is None:
            continue
        yield load_well(files)


def well_metadata(split: str = "train", limit: int | None = None) -> pd.DataFrame:
    """Well-level metadata table for both splits, built by streaming."""
    return pd.DataFrame([w.meta for w in iter_wells(split, limit=limit)])


def validate_split(split: str, limit: int | None = None) -> pd.DataFrame:
    """Assertions that must hold before modelling; returns the issue table."""
    issues = []
    for w in iter_wells(split, limit=limit):
        m = w.meta
        if not m.get("md_monotonic", True):
            issues.append((w.well_id, "md_not_monotonic", ""))
        if m.get("clean_prefix_split") is False:
            issues.append((w.well_id, "tvt_input_not_clean_prefix", ""))
        if not m["has_typewell"]:
            issues.append((w.well_id, "missing_typewell", ""))
        if split == "train" and not m["has_target_column"]:
            issues.append((w.well_id, "missing_target_column", ""))
        if split == "test" and m["target_available_on_hidden"]:
            issues.append((w.well_id, "TEST_LEAK_target_present_on_hidden_rows", "investigate"))
        if m["n_hidden"] == 0:
            issues.append((w.well_id, "no_hidden_region", ""))
    return pd.DataFrame(issues, columns=["well_id", "issue", "note"])
