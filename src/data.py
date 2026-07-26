"""Baseline data loader for the ROGII wellbore geology task.

Loading only — this module never trains, scores, or writes a model.

Public API
----------
    discover_wells(split)            -> dict[well_id, WellFiles]
    load_horizontal_well(path|files) -> DataFrame (original row order kept)
    load_typewell(path|files)        -> DataFrame | None
    load_well(well_id, split)        -> WellData
    identify_visible_prefix(hw, roles)  -> (mask, info)
    identify_hidden_suffix(hw, roles)   -> (mask, info)
    summarize_well(well)             -> dict
    iter_wells(split)                -> Iterator[WellData]   (one well at a time)

Guarantees
----------
* Well IDs are discovered from the filesystem; none are hardcoded.
* Original CSV row order is preserved end to end and never sorted.
* Wells are streamed one at a time — no global concatenation.
* Optional columns (TVT target, markers, Prediction Start, typewell) may be
  absent without raising; their absence is recorded in metadata instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from src.columns import FORMATION_ORDER, marker_columns, resolve_roles
from src.discovery import WellFiles
from src.discovery import discover_wells as _discover_files
from src.paths import TEST_DIR, TRAIN_DIR, require_competition_data


def _split_dir(split: str) -> Path:
    s = split.lower()
    if s == "train":
        return TRAIN_DIR
    if s == "test":
        return TEST_DIR
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


# --------------------------------------------------------------- discovery --

def discover_wells(
    split: str = "train",
    *,
    directory: Path | None = None,
    check_mount: bool = True,
) -> dict[str, WellFiles]:
    """Dynamically discover wells for a split. No hardcoded well IDs."""
    if directory is None:
        if check_mount:
            require_competition_data()
        directory = _split_dir(split)
    return _discover_files(Path(directory), split)


def list_well_ids(split: str = "train", **kw) -> list[str]:
    return sorted(discover_wells(split, **kw))


# ------------------------------------------------------------------ loading --

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_horizontal_well(source: Path | str | WellFiles) -> pd.DataFrame:
    """Load a horizontal-well CSV, preserving original row order.

    Adds a ``row_index`` column recording the on-disk position, so the order
    survives any later filtering or merging.
    """
    path = source.horizontal if isinstance(source, WellFiles) else Path(source)
    if path is None:
        raise FileNotFoundError("no horizontal-well file for this well")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"horizontal-well file not found: {path}")
    df = _read_csv(path)
    if "row_index" not in df.columns:
        df.insert(0, "row_index", np.arange(len(df), dtype=np.int64))
    return df


def load_typewell(source: Path | str | WellFiles | None) -> pd.DataFrame | None:
    """Load a typewell CSV. Returns None when absent (it is optional)."""
    if source is None:
        return None
    path = source.typewell if isinstance(source, WellFiles) else Path(source)
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return _read_csv(path)


# ------------------------------------------------- visible / hidden regions --

def _prediction_start_row(hw: pd.DataFrame, roles: dict[str, str]) -> tuple[int, str]:
    """First row of the hidden suffix, plus how it was determined."""
    n = len(hw)
    ps_col = roles.get("prediction_start")
    if ps_col is not None and ps_col in hw.columns:
        s = hw[ps_col]
        non_null = s.dropna()
        uniq = set(pd.unique(non_null))
        looks_boolean = pd.api.types.is_bool_dtype(s) or uniq <= {0, 1, True, False, "0", "1"}
        if looks_boolean and len(non_null):
            flags = s.fillna(0).astype(float).astype(bool).to_numpy()
            if flags.any():
                return int(np.argmax(flags)), "prediction_start_flag"
        elif len(non_null) and roles.get("md") in hw.columns:
            thr = pd.to_numeric(non_null, errors="coerce").dropna()
            md = pd.to_numeric(hw[roles["md"]], errors="coerce")
            if len(thr):
                over = (md >= float(thr.iloc[0])).to_numpy()
                if over.any():
                    return int(np.argmax(over)), "prediction_start_md"

    ti_col = roles.get("tvt_input")
    if ti_col is not None and ti_col in hw.columns:
        known = pd.to_numeric(hw[ti_col], errors="coerce").notna().to_numpy()
        if known.all():
            return n, "tvt_input_all_known"
        if not known.any():
            return 0, "tvt_input_none_known"
        return int(np.argmax(~known)), "tvt_input_first_gap"

    return n, "fallback_all_visible"


def identify_visible_prefix(
    hw: pd.DataFrame, roles: dict[str, str] | None = None
) -> tuple[np.ndarray, dict]:
    """Boolean mask of the known-TVT prefix, plus diagnostic info.

    ``info['clean_prefix_split']`` is False when TVT_input has an *internal*
    gap, i.e. known values reappear after the first gap. That is a data-quality
    signal the caller must not ignore.
    """
    roles = roles if roles is not None else resolve_roles(hw.columns)
    n = len(hw)
    start, source = _prediction_start_row(hw, roles)

    mask = np.zeros(n, dtype=bool)
    mask[:start] = True

    info: dict = {
        "prediction_start_row": int(start),
        "prediction_start_source": source,
        "n_rows": int(n),
        "n_visible": int(mask.sum()),
        "n_hidden": int(n - mask.sum()),
        "visible_fraction": float(mask.mean()) if n else np.nan,
    }

    ti_col = roles.get("tvt_input")
    if ti_col is not None and ti_col in hw.columns:
        ti = pd.to_numeric(hw[ti_col], errors="coerce")
        known = ti.notna().to_numpy()
        info["n_tvt_input_known"] = int(known.sum())
        info["clean_prefix_split"] = bool(known[:start].all() and not known[start:].any())
        info["internal_tvt_input_gap"] = bool(not known[:start].all())
        info["known_after_prediction_start"] = int(known[start:].sum())
        info["tvt_last_known"] = float(ti.iloc[start - 1]) if start > 0 and known[start - 1] else np.nan
    else:
        info["clean_prefix_split"] = None
        info["internal_tvt_input_gap"] = None

    md_col = roles.get("md")
    if md_col is not None and md_col in hw.columns:
        md = pd.to_numeric(hw[md_col], errors="coerce")
        info["md_at_prediction_start"] = float(md.iloc[start]) if start < n else np.nan
        info["md_monotonic"] = bool(md.is_monotonic_increasing)
    return mask, info


def identify_hidden_suffix(
    hw: pd.DataFrame, roles: dict[str, str] | None = None
) -> tuple[np.ndarray, dict]:
    """Boolean mask of the hidden suffix (complement of the visible prefix)."""
    visible, info = identify_visible_prefix(hw, roles)
    return ~visible, info


# -------------------------------------------------------------- well object --

@dataclass
class WellData:
    well_id: str
    split: str
    hw: pd.DataFrame
    tw: pd.DataFrame | None = None
    roles: dict[str, str] = field(default_factory=dict)
    markers: dict[str, str] = field(default_factory=dict)
    tw_roles: dict[str, str] = field(default_factory=dict)
    region_info: dict = field(default_factory=dict)
    files: WellFiles | None = None

    @property
    def visible_mask(self) -> np.ndarray:
        return self.hw["is_visible"].to_numpy()

    @property
    def hidden_mask(self) -> np.ndarray:
        return self.hw["is_hidden"].to_numpy()

    @property
    def prefix(self) -> pd.DataFrame:
        return self.hw.loc[self.visible_mask]

    @property
    def suffix(self) -> pd.DataFrame:
        return self.hw.loc[self.hidden_mask]

    def col(self, role: str) -> pd.Series | None:
        name = self.roles.get(role)
        return self.hw[name] if name and name in self.hw.columns else None

    @property
    def has_target(self) -> bool:
        return self.roles.get("tvt") is not None

    # ---------------------------------------------------------- leakage --
    # In train wells the full TVT curve is present, including the hidden
    # region. Those values are the *label*; feeding them to a model as an
    # input feature reproduces the classic "predicting from the answer"
    # failure. These accessors make the safe path the easy path.

    LEAKY_ROLES = ("tvt",)

    def inference_features(
        self, *, drop: tuple[str, ...] = ("TVT", "tvt")
    ) -> pd.DataFrame:
        """Columns safe to use as model inputs.

        Drops the TVT target column entirely. TVT_input is retained, but it is
        NaN on the hidden region by construction, so it cannot leak.
        """
        cols = [c for c in self.hw.columns if c not in set(drop)]
        tgt = self.roles.get("tvt")
        if tgt and tgt in cols:
            cols.remove(tgt)
        return self.hw[cols]

    def target(self, region: str = "hidden") -> pd.Series | None:
        """The TVT label for a region. Explicit, so its use is always visible."""
        tgt = self.roles.get("tvt")
        if tgt is None:
            return None
        if region == "hidden":
            return self.hw.loc[self.hidden_mask, tgt]
        if region == "visible":
            return self.hw.loc[self.visible_mask, tgt]
        if region == "all":
            return self.hw[tgt]
        raise ValueError("region must be 'hidden', 'visible' or 'all'")

    def assert_no_target_leakage(self, frame: pd.DataFrame) -> None:
        """Raise if `frame` still carries the TVT target column."""
        tgt = self.roles.get("tvt")
        if tgt and tgt in frame.columns:
            raise ValueError(
                f"{self.well_id}: feature frame still contains the target column "
                f"{tgt!r}. Use .inference_features() instead."
            )


def load_well(
    well: str | WellFiles,
    split: str = "train",
    *,
    directory: Path | None = None,
    require_typewell: bool = False,
) -> WellData:
    """Load one well (horizontal + typewell) with regions identified."""
    if isinstance(well, WellFiles):
        files = well
        split = files.split or split
    else:
        found = discover_wells(split, directory=directory)
        if well not in found:
            raise KeyError(
                f"well {well!r} not found in split {split!r} "
                f"({len(found)} wells discovered)"
            )
        files = found[well]

    if files.horizontal is None:
        raise FileNotFoundError(f"{files.well_id}: horizontal-well file is missing")

    hw = load_horizontal_well(files)
    roles = resolve_roles(hw.columns)
    if "md" not in roles:
        raise ValueError(
            f"{files.well_id}: cannot resolve an MD column from {list(hw.columns)}"
        )
    markers = marker_columns(hw.columns)

    visible, info = identify_visible_prefix(hw, roles)
    hw["is_visible"] = visible
    hw["is_hidden"] = ~visible

    tw = load_typewell(files)
    if tw is None and require_typewell:
        raise FileNotFoundError(f"{files.well_id}: typewell file is missing")
    tw_roles = resolve_roles(tw.columns) if tw is not None else {}

    return WellData(
        well_id=files.well_id,
        split=files.split or split,
        hw=hw,
        tw=tw,
        roles=roles,
        markers=markers,
        tw_roles=tw_roles,
        region_info=info,
        files=files,
    )


# ------------------------------------------------------------------ summary --

def summarize_well(well: WellData) -> dict:
    """Flat, well-level metadata record — safe when optional columns absent."""
    hw, roles = well.hw, well.roles
    rec: dict = {
        "well_id": well.well_id,
        "split": well.split,
        "n_rows": int(len(hw)),
        "n_columns": int(hw.shape[1]),
        "has_typewell": well.tw is not None,
        "n_typewell_rows": int(len(well.tw)) if well.tw is not None else 0,
        "n_png": len(well.files.images) if well.files else 0,
    }
    rec.update(well.region_info)

    rec["markers_present"] = ",".join(well.markers)
    rec["markers_absent"] = ",".join(f for f in FORMATION_ORDER if f not in well.markers)
    rec["n_markers_present"] = len(well.markers)

    # missing-value report over every resolved role
    for role in ("md", "x", "y", "z", "gr", "tvt_input", "tvt"):
        col = roles.get(role)
        rec[f"has_{role}"] = col is not None
        if col is not None and col in hw.columns:
            s = pd.to_numeric(hw[col], errors="coerce")
            rec[f"{role}_missing_frac"] = float(s.isna().mean())
            rec[f"{role}_min"] = float(s.min()) if s.notna().any() else np.nan
            rec[f"{role}_max"] = float(s.max()) if s.notna().any() else np.nan
        else:
            rec[f"{role}_missing_frac"] = np.nan

    md_col = roles.get("md")
    if md_col:
        md = pd.to_numeric(hw[md_col], errors="coerce")
        d = md.diff().dropna()
        rec["md_step_median"] = float(d.median()) if len(d) else np.nan
        rec["md_step_std"] = float(d.std()) if len(d) else np.nan
        rec["md_duplicates"] = int(md.duplicated().sum())
        rec["md_uniform_spacing"] = bool(len(d) and np.isclose(d.std(), 0.0, atol=1e-9))
        rec["md_step_min"] = float(d.min()) if len(d) else np.nan
        rec["md_step_max"] = float(d.max()) if len(d) else np.nan
        # the competition grid is nominally one foot; flag any departure
        rec["md_step_is_one_foot"] = bool(
            len(d) and np.allclose(d.to_numpy(), 1.0, atol=1e-6)
        )
        rec["md_has_gaps"] = bool(len(d) and float(d.max()) > 1.5 * float(d.median()))

    gr_col = roles.get("gr")
    if gr_col:
        gr = pd.to_numeric(hw[gr_col], errors="coerce")
        na = gr.isna().to_numpy()
        rec["gr_missing_frac"] = float(na.mean())
        rec["gr_high_missingness"] = bool(na.mean() > 0.5)
        # longest contiguous gap: a physical tool outage, not random noise
        longest = cur = 0
        for v in na:
            cur = cur + 1 if v else 0
            longest = max(longest, cur)
        rec["gr_longest_gap"] = int(longest)
        rec["gr_negative"] = int((gr < 0).sum())

    tgt = roles.get("tvt")
    rec["has_target_column"] = tgt is not None
    if tgt is not None:
        t = pd.to_numeric(hw[tgt], errors="coerce")
        rec["n_target_known"] = int(t.notna().sum())
        rec["target_available_on_hidden"] = bool(
            rec.get("n_hidden", 0) > 0 and t[well.hidden_mask].notna().any()
        )
    else:
        rec["n_target_known"] = 0
        rec["target_available_on_hidden"] = False

    if well.tw is not None:
        g = well.tw_roles.get("geology")
        if g:
            rec["tw_n_geology_labels"] = int(well.tw[g].nunique(dropna=True))
        tv = well.tw_roles.get("tvt")
        if tv:
            t = pd.to_numeric(well.tw[tv], errors="coerce")
            rec["tw_tvt_min"] = float(t.min()) if t.notna().any() else np.nan
            rec["tw_tvt_max"] = float(t.max()) if t.notna().any() else np.nan
    rec["duplicate_rows"] = int(hw.drop(columns=["row_index"], errors="ignore").duplicated().sum())
    return rec


# ---------------------------------------------------------------- streaming --

def iter_wells(
    split: str = "train",
    *,
    directory: Path | None = None,
    limit: int | None = None,
    well_ids: Iterable[str] | None = None,
) -> Iterator[WellData]:
    """Yield wells one at a time; never holds the full dataset in memory."""
    found = discover_wells(split, directory=directory)
    keys = sorted(found) if well_ids is None else [w for w in well_ids if w in found]
    for i, wid in enumerate(keys):
        if limit is not None and i >= limit:
            break
        if found[wid].horizontal is None:
            continue
        yield load_well(found[wid])


def well_metadata(
    split: str = "train", *, directory: Path | None = None, limit: int | None = None
) -> pd.DataFrame:
    """Well-level metadata table, built by streaming (one well resident)."""
    return pd.DataFrame(
        [summarize_well(w) for w in iter_wells(split, directory=directory, limit=limit)]
    )


def validate_split(
    split: str, *, directory: Path | None = None, limit: int | None = None
) -> pd.DataFrame:
    """Structural checks that must hold before modelling."""
    issues: list[tuple[str, str, str]] = []
    for w in iter_wells(split, directory=directory, limit=limit):
        s = summarize_well(w)
        if s.get("md_monotonic") is False:
            issues.append((w.well_id, "md_not_monotonic", "row order may be unreliable"))
        if s.get("clean_prefix_split") is False:
            issues.append((w.well_id, "tvt_input_internal_gap",
                           f"{s.get('known_after_prediction_start', 0)} known rows after start"))
        if not s["has_typewell"]:
            issues.append((w.well_id, "missing_typewell", "typewell prior unavailable"))
        if split == "train" and not s["has_target_column"]:
            issues.append((w.well_id, "missing_target_column", "cannot be used for supervision"))
        if split == "test" and s["target_available_on_hidden"]:
            issues.append((w.well_id, "TEST_LEAK_target_on_hidden_rows", "investigate before use"))
        if s.get("md_duplicates", 0):
            issues.append((w.well_id, "duplicate_md_values",
                           f"{s['md_duplicates']} repeated MD readings"))
        if s.get("md_step_is_one_foot") is False:
            issues.append((w.well_id, "md_step_not_one_foot",
                           f"median step {s.get('md_step_median')}, "
                           f"min {s.get('md_step_min')}, max {s.get('md_step_max')}"))
        if s.get("n_hidden", 0) == 0:
            issues.append((w.well_id, "no_hidden_region", "nothing to predict"))
        if s.get("gr_high_missingness"):
            issues.append((w.well_id, "high_gr_missingness",
                           f"{s.get('gr_missing_frac', float('nan')):.1%} missing, "
                           f"longest gap {s.get('gr_longest_gap')}"))
    return pd.DataFrame(issues, columns=["well_id", "issue", "note"])
