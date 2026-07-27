"""Cross-fitting guarantees.

The bug these tests exist to prevent: a model fitted on the same wells it is
scored on reports memorisation as skill. A LightGBM with hundreds of trees can
reproduce a well's individual trajectory almost exactly, so an in-sample RMSE
can be arbitrarily good while saying nothing about unseen-well performance.

Both validation protocols must therefore fit on fold-train wells and score on
fold-validation wells, with no overlap, and the harness must refuse to run if
that is ever violated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.validation as _v


def _mod(name):
    import importlib
    import sys

    return importlib.reload(sys.modules[f"src.{name}"])


def _tasks(mount, ids, mode="real"):
    d = _mod("data")
    tasks_mod = _mod("tasks")
    files = d.discover_wells("train")
    out = []
    for wid in ids:
        try:
            out.append(tasks_mod.make_task(d.load_well(files[wid]), mode))
        except tasks_mod.TaskConstructionError:
            pass
    return out


def _builder(mount):
    """A task_builder closure like the one the runner passes in."""
    d = _mod("data")
    tasks_mod = _mod("tasks")
    files = d.discover_wells("train")

    def build(ids, mode):
        tasks, skipped = [], []
        for wid in ids:
            entry = files.get(wid)
            if entry is None:
                skipped.append((wid, "missing"))
                continue
            try:
                tasks.append(tasks_mod.make_task(d.load_well(entry), mode))
            except tasks_mod.TaskConstructionError as exc:
                skipped.append((wid, str(exc)))
        return tasks, skipped

    return build


# ------------------------------------------------------- protocol identity --

def test_protocol_names_are_distinct_and_invalid_is_marked():
    v = _mod("validation")
    assert v.PROTOCOL_A != v.PROTOCOL_B
    assert v.PROTOCOL_INVALID.startswith("INVALID")
    assert v.VALID_PROTOCOLS == (v.PROTOCOL_A, v.PROTOCOL_B)
    assert v.PROTOCOL_INVALID not in v.VALID_PROTOCOLS


# --------------------------------------------------- the cross-fit guarantee --

@pytest.mark.parametrize("mode,protocol_attr", [("real", "PROTOCOL_B"), ("masked", "PROTOCOL_A")])
def test_no_well_is_both_fitted_and_scored(mount, mode, protocol_attr):
    """The central guarantee, asserted by spying on what each model is fitted on."""
    v = _mod("validation")
    protocol = getattr(v, protocol_attr)
    ids = [f"TRW00{i}" for i in (1, 2, 3, 4, 6, 7, 8, 9)]

    fitted_on: dict[int, set[str]] = {}
    scored_on: dict[int, set[str]] = {}
    call = {"n": 0}

    class Spy:
        name = "spy"
        needs_alignment = False
        uses_spatial = False

        def __init__(self, **kw):
            self.fold = call["n"]

        def fit(self, tasks, **kw):
            fitted_on.setdefault(self.fold, set()).update(t.well_id for t in tasks)
            call["n"] += 1
            return self

        def predict(self, task, feats=None):
            scored_on.setdefault(self.fold, set()).add(task.well_id)
            return np.zeros(task.n_predict)

    folds = v.make_group_folds(ids, n_splits=2, seed=0)
    v.run_cross_fitted_protocol(
        protocol=protocol,
        mode=mode,
        factories={"spy": Spy},
        folds=folds,
        task_builder=_builder(mount),
    )

    assert fitted_on, "no model was fitted at all"
    for fold_idx, trained in fitted_on.items():
        scored = scored_on.get(fold_idx, set())
        assert scored, f"fold {fold_idx} scored nothing"
        assert not (trained & scored), (
            f"fold {fold_idx}: wells {sorted(trained & scored)} were both "
            "fitted and scored"
        )


def test_cross_fit_driver_raises_on_deliberate_overlap(mount):
    """If a Fold is subverted, the driver must still catch the overlap."""
    v = _mod("validation")
    build = _builder(mount)

    class FakeFold:
        index = 0
        train_ids = ["TRW001", "TRW003"]
        valid_ids = ["TRW001"]  # deliberate overlap

    with pytest.raises(v.CrossFitLeakage, match="both"):
        v.run_cross_fitted_protocol(
            protocol=v.PROTOCOL_B,
            mode="real",
            factories={},
            folds=[FakeFold()],
            task_builder=build,
        )


def test_every_well_is_scored_exactly_once_across_folds(mount):
    v = _mod("validation")
    ids = [f"TRW00{i}" for i in (1, 2, 3, 4, 6, 7, 8, 9)]
    folds = v.make_group_folds(ids, n_splits=5, seed=3)

    run = v.run_cross_fitted_protocol(
        protocol=v.PROTOCOL_B,
        mode="real",
        factories={"hold_last": _mod("baselines").BASELINES["hold_last"]},
        folds=folds,
        task_builder=_builder(mount),
    )
    scored = [r.well_id for r in run.well_results]
    assert len(scored) == len(set(scored)), "a well was scored twice"


def test_in_sample_diagnostic_is_labelled_invalid(mount):
    v = _mod("validation")
    tasks = _tasks(mount, ["TRW001", "TRW003"], "real")
    run = v.run_in_sample_diagnostic(
        factories={"hold_last": _mod("baselines").BASELINES["hold_last"]},
        tasks=tasks,
    )
    assert run.protocol == v.PROTOCOL_INVALID
    assert all(r.protocol == v.PROTOCOL_INVALID for r in run.well_results)


def test_in_sample_beats_honest_for_a_memorising_model(mount):
    """Demonstrates *why* cross-fitting matters, using a deliberate memoriser."""
    v = _mod("validation")
    ids = [f"TRW00{i}" for i in (1, 2, 3, 4, 6, 7, 8, 9)]

    class Memoriser:
        """Stores each well's answer at fit time and replays it."""

        name = "memoriser"
        needs_alignment = False
        uses_spatial = False

        def __init__(self, **kw):
            self.book: dict[str, np.ndarray] = {}

        def fit(self, tasks, **kw):
            for t in tasks:
                if t.target is not None:
                    self.book[t.well_id] = np.asarray(t.target, dtype="float64")
            return self

        def predict(self, task, feats=None):
            got = self.book.get(task.well_id)
            if got is not None and got.size == task.n_predict:
                return got  # perfect, and entirely fake
            anchor = task.anchor_tvt
            return np.full(task.n_predict, anchor if np.isfinite(anchor) else 0.0)

    tasks = _tasks(mount, ids, "real")
    insample = v.run_in_sample_diagnostic(factories={"m": Memoriser}, tasks=tasks)
    honest = v.run_cross_fitted_protocol(
        protocol=v.PROTOCOL_B,
        mode="real",
        factories={"m": Memoriser},
        folds=v.make_group_folds(ids, n_splits=2, seed=0),
        task_builder=_builder(mount),
    )

    ins = v.summarize(pd.DataFrame([r.__dict__ for r in insample.well_results]))
    hon = v.summarize(pd.DataFrame([r.__dict__ for r in honest.well_results]))
    ins_rmse = float(ins["global_rmse"].iloc[0])
    hon_rmse = float(hon["global_rmse"].iloc[0])

    assert ins_rmse == pytest.approx(0.0, abs=1e-9), "memoriser should be perfect in-sample"
    assert hon_rmse > 1e-6, (
        "cross-fitted protocol failed to expose a pure memoriser — "
        "this is exactly the bug the harness must prevent"
    )


# ------------------------------------------------------------ spatial LOWO --

def test_spatial_donors_exclude_validation_wells(mount):
    """Donors come from fold-train wells; no validation well may donate."""
    v = _mod("validation")
    spatial = _mod("spatial")
    ids = [f"TRW00{i}" for i in (1, 2, 3, 4, 6, 7, 8, 9)]
    folds = v.make_group_folds(ids, n_splits=2, seed=0)

    run = v.run_cross_fitted_protocol(
        protocol=v.PROTOCOL_B,
        mode="real",
        factories={"ridge": _mod("baselines").BASELINES["ridge"]},
        folds=folds,
        task_builder=_builder(mount),
        spatial_config=spatial.SpatialConfig(k=4, radius=1e9),
        spatial_models=("ridge",),
    )
    assert run.spatial_notes, "spatial prior was never built"
    for note in run.spatial_notes:
        assert note["n_donor_wells"] > 0
        assert note["self_exclusion"]


def test_spatial_prior_rejects_a_validation_donor_directly(mount):
    spatial = _mod("spatial")
    tasks = _tasks(mount, ["TRW001", "TRW003"], "real")
    prior = spatial.SpatialPrior().fit(tasks)
    with pytest.raises(spatial.SpatialLeakage):
        prior.assert_disjoint(["TRW001"])


# ----------------------------------------------------------- failure count --

def test_failures_are_counted_not_swallowed(mount):
    v = _mod("validation")
    ids = [f"TRW00{i}" for i in (1, 3, 6, 7)]

    class Broken:
        name = "broken"
        needs_alignment = False
        uses_spatial = False

        def __init__(self, **kw):
            pass

        def fit(self, tasks, **kw):
            return self

        def predict(self, task, feats=None):
            raise ValueError("deliberate failure")

    run = v.run_cross_fitted_protocol(
        protocol=v.PROTOCOL_B,
        mode="real",
        factories={"broken": Broken},
        folds=v.make_group_folds(ids, n_splits=3, seed=0),
        task_builder=_builder(mount),
    )
    assert run.n_failures > 0
    assert all(f["stage"] in {"task", "fit", "predict"} for f in run.failures)
    assert any("deliberate failure" in f["error"] for f in run.failures)
    assert not run.well_results, "a failing model must not produce scores"


def test_blocked_wells_rejected_inside_the_driver(mount):
    v = _mod("validation")

    class FakeFold:
        index = 0
        train_ids = ["TRW001", "000d7d20"]
        valid_ids = ["TRW003"]

    with pytest.raises(v.BlockedWellError):
        v.run_cross_fitted_protocol(
            protocol=v.PROTOCOL_B,
            mode="real",
            factories={},
            folds=[FakeFold()],
            task_builder=_builder(mount),
        )
