"""Prediction tasks — the unit every baseline consumes.

A ``WellTask`` is a *boundary-relative* view of one well: everything known at
inference time, plus the index range that must be predicted. Its defining
property is structural, not documentary:

    the TVT label for the region under prediction is not reachable
    from the object a model is given.

``WellTask.inputs()`` returns an ``InferenceTask`` that carries no target at
all, and the harness only ever hands models that object. A model therefore
cannot read the answer even by mistake — there is nothing to read.

Two boundary modes are supported:

``real``
    The competition boundary. Known TVT is ``TVT_input`` on the visible
    prefix; the region to predict is the hidden suffix; truth comes from the
    ``TVT`` label column (train wells only).

``masked``
    A synthetic boundary moved *earlier inside the visible prefix*, so both
    the inputs and the truth come from ``TVT_input``. This is the same-well
    masked-suffix protocol, and because it never touches the label column it
    is the only protocol that could also be run on a test well.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np
import pandas as pd

from src.data import WellData, iter_wells

MIN_PREFIX_ROWS = 200


class TaskConstructionError(RuntimeError):
    """Raised when a well cannot form a usable prediction task."""


@dataclass(frozen=True)
class InferenceTask:
    """Everything a model may see. Deliberately has no target field."""

    well_id: str
    split: str
    mode: str
    start: int  # first row index to predict
    stop: int  # one past the last row to predict
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt_known: np.ndarray  # NaN from `start` onwards, by construction
    tw_tvt: np.ndarray | None
    tw_gr: np.ndarray | None
    tw_geology: np.ndarray | None

    @property
    def n_rows(self) -> int:
        return int(self.md.size)

    @property
    def n_predict(self) -> int:
        return int(self.stop - self.start)

    @property
    def prefix_len(self) -> int:
        return int(self.start)

    @property
    def index(self) -> np.ndarray:
        return np.arange(self.start, self.stop)

    @property
    def anchor_tvt(self) -> float:
        """Last known TVT before the boundary."""
        known = self.tvt_known[: self.start]
        finite = known[np.isfinite(known)]
        return float(finite[-1]) if finite.size else float("nan")

    @property
    def anchor_row(self) -> int:
        known = np.isfinite(self.tvt_known[: self.start])
        return int(np.max(np.flatnonzero(known))) if known.any() else -1

    @property
    def dmd(self) -> np.ndarray:
        """Feet drilled past the boundary, for the predicted rows."""
        row = self.anchor_row if self.anchor_row >= 0 else max(self.start - 1, 0)
        return self.md[self.start : self.stop] - self.md[row]

    @property
    def gr_missing_frac(self) -> float:
        return float(np.mean(~np.isfinite(self.gr))) if self.gr.size else 1.0

    @property
    def has_typewell(self) -> bool:
        return (
            self.tw_tvt is not None
            and self.tw_gr is not None
            and np.isfinite(self.tw_tvt).sum() > 2
        )

    def assert_no_target(self) -> None:
        """Structural check: nothing on this object exposes the label."""
        if hasattr(self, "target"):  # pragma: no cover - defensive
            raise AssertionError(f"{self.well_id}: InferenceTask exposes a target")
        if np.isfinite(self.tvt_known[self.start : self.stop]).any():
            raise AssertionError(
                f"{self.well_id}: tvt_known has finite values inside the "
                "prediction region — the boundary was built incorrectly"
            )


@dataclass(frozen=True)
class WellTask:
    """An ``InferenceTask`` plus the truth used only for scoring."""

    inputs_: InferenceTask
    target: np.ndarray | None  # length == n_predict, or None at inference

    @property
    def well_id(self) -> str:
        return self.inputs_.well_id

    @property
    def mode(self) -> str:
        return self.inputs_.mode

    def inputs(self) -> InferenceTask:
        """The only object a model is ever given."""
        return self.inputs_

    def scored(self) -> np.ndarray:
        if self.target is None:
            raise TaskConstructionError(f"{self.well_id}: task carries no truth")
        return self.target


def _finite(arr) -> np.ndarray:
    return np.asarray(pd.to_numeric(pd.Series(arr), errors="coerce"), dtype="float64")


def _column(well: WellData, role: str) -> np.ndarray | None:
    name = well.roles.get(role)
    if name is None or name not in well.hw.columns:
        return None
    return _finite(well.hw[name].to_numpy())


def _typewell_arrays(well: WellData):
    if well.tw is None:
        return None, None, None
    tvt_col = well.tw_roles.get("tvt") or well.tw_roles.get("tvt_input")
    gr_col = well.tw_roles.get("gr")
    geo_col = well.tw_roles.get("geology")
    tvt = _finite(well.tw[tvt_col].to_numpy()) if tvt_col else None
    gr = _finite(well.tw[gr_col].to_numpy()) if gr_col else None
    geo = well.tw[geo_col].to_numpy() if geo_col and geo_col in well.tw.columns else None
    return tvt, gr, geo


def make_task(
    well: WellData,
    mode: str = "real",
    *,
    min_prefix: int = MIN_PREFIX_ROWS,
    min_predict: int = 25,
) -> WellTask:
    """Build a prediction task for one well.

    Parameters
    ----------
    mode
        ``"real"`` uses the competition boundary; ``"masked"`` moves the
        boundary earlier inside the visible prefix so that truth comes from
        ``TVT_input`` and the label column is never touched.
    """
    n = len(well.hw)
    md = _column(well, "md")
    if md is None:
        raise TaskConstructionError(f"{well.well_id}: no MD column")

    x = _column(well, "x")
    y = _column(well, "y")
    z = _column(well, "z")
    gr = _column(well, "gr")
    tvt_input = _column(well, "tvt_input")
    nan = np.full(n, np.nan)
    x = nan.copy() if x is None else x
    y = nan.copy() if y is None else y
    z = nan.copy() if z is None else z
    gr = nan.copy() if gr is None else gr
    if tvt_input is None:
        raise TaskConstructionError(f"{well.well_id}: no TVT_input column")

    real_start = int(well.region_info.get("prediction_start_row", n))
    real_start = max(0, min(real_start, n))

    if mode == "real":
        start, stop = real_start, n
        tgt_col = well.roles.get("tvt")
        target = None
        if tgt_col is not None and tgt_col in well.hw.columns:
            target = _finite(well.hw[tgt_col].to_numpy())[start:stop]
        known = tvt_input.copy()
        known[start:] = np.nan

    elif mode == "masked":
        prefix_len = real_start
        real_suffix = n - real_start
        budget = prefix_len - min_prefix
        if budget < min_predict:
            raise TaskConstructionError(
                f"{well.well_id}: visible prefix of {prefix_len} rows is too "
                f"short to mask a suffix (needs > {min_prefix + min_predict})"
            )
        masked_len = int(np.clip(real_suffix, min_predict, budget))
        start = prefix_len - masked_len
        stop = prefix_len
        target = tvt_input[start:stop].copy()
        known = tvt_input.copy()
        known[start:] = np.nan
    else:
        raise ValueError(f"mode must be 'real' or 'masked', got {mode!r}")

    if stop - start < min_predict:
        raise TaskConstructionError(
            f"{well.well_id}: only {stop - start} rows to predict in mode {mode!r}"
        )
    if not np.isfinite(known[:start]).any():
        raise TaskConstructionError(f"{well.well_id}: no known TVT before the boundary")

    tw_tvt, tw_gr, tw_geo = _typewell_arrays(well)

    task = InferenceTask(
        well_id=well.well_id,
        split=well.split,
        mode=mode,
        start=int(start),
        stop=int(stop),
        md=md,
        x=x,
        y=y,
        z=z,
        gr=gr,
        tvt_known=known,
        tw_tvt=tw_tvt,
        tw_gr=tw_gr,
        tw_geology=tw_geo,
    )
    task.assert_no_target()
    if target is not None and target.size != task.n_predict:  # pragma: no cover
        raise TaskConstructionError(f"{well.well_id}: truth length mismatch")
    return WellTask(inputs_=task, target=target)


def iter_tasks(
    split: str = "train",
    mode: str = "real",
    *,
    well_ids=None,
    limit: int | None = None,
    directory=None,
    on_error: str = "skip",
) -> Iterator[WellTask]:
    """Stream tasks, one well resident in memory at a time."""
    for well in iter_wells(split, well_ids=well_ids, limit=limit, directory=directory):
        try:
            yield make_task(well, mode)
        except TaskConstructionError:
            if on_error == "raise":
                raise
            continue


def task_descriptor(task: WellTask | InferenceTask) -> dict:
    """Stratification metadata for reporting (no target information)."""
    inp = task.inputs() if isinstance(task, WellTask) else task
    return {
        "well_id": inp.well_id,
        "split": inp.split,
        "mode": inp.mode,
        "n_rows": inp.n_rows,
        "prefix_len": inp.prefix_len,
        "suffix_len": inp.n_predict,
        "gr_missing_frac": inp.gr_missing_frac,
        "anchor_tvt": inp.anchor_tvt,
        "has_typewell": inp.has_typewell,
        "md_start": float(inp.md[inp.start]) if inp.n_rows else float("nan"),
    }


def without_typewell(task: InferenceTask) -> InferenceTask:
    """Ablation helper: the same task with the typewell prior removed."""
    return replace(task, tw_tvt=None, tw_gr=None, tw_geology=None)
