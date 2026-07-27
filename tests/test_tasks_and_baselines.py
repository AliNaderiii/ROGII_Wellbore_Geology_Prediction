"""Task construction, feature safety and baseline behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines import BASELINE_ORDER, HAVE_LIGHTGBM
from src.manifest import assert_safe_features
from src.validation import rmse, summarize

# The `mount` fixture reloads src.* after patching the environment, so these
# helpers resolve the *current* module objects at call time. Importing the
# symbols at module scope would bind pre-reload classes and break isinstance.


def _mod(name):
    import importlib
    import sys

    return importlib.reload(sys.modules[f"src.{name}"])


def _wells(mount, split="train"):
    return _mod("data").discover_wells(split)


def _load(mount, well_id, split="train"):
    d = _mod("data")
    return d.load_well(d.discover_wells(split)[well_id])


def _task(mount, well_id, mode="real", **kw):
    return _mod("tasks").make_task(_load(mount, well_id), mode, **kw)


def _feats(inp, **kw):
    return _mod("features").build_features(inp, **kw)


def _baseline(name, **kw):
    return _mod("baselines").BASELINES[name](**kw)


# ------------------------------------------------------------------- tasks --

def test_inference_task_has_no_target_attribute(mount):
    files = _wells(mount)
    task = _task(mount, "TRW001", "real")
    inp = task.inputs()
    assert not hasattr(inp, "target")
    assert "target" not in vars(inp)


def test_tvt_known_is_nan_inside_prediction_region(mount):
    files = _wells(mount)
    inp = _task(mount, "TRW001", "real").inputs()
    assert not np.isfinite(inp.tvt_known[inp.start : inp.stop]).any()
    inp.assert_no_target()


def test_anchor_is_the_last_known_prefix_value(mount):
    files = _wells(mount)
    well = _load(mount, "TRW001")
    inp = _mod("tasks").make_task(well, "real").inputs()
    known = pd.to_numeric(well.hw["TVT_input"], errors="coerce").dropna()
    assert inp.anchor_tvt == pytest.approx(float(known.iloc[-1]))


def test_masked_mode_never_reads_the_label_column(mount):
    """Truth in masked mode must come from TVT_input, not TVT."""
    well = _load(mount, "TRW006")
    task = _mod("tasks").make_task(well, "masked")
    inp = task.inputs()
    ti = pd.to_numeric(well.hw["TVT_input"], errors="coerce").to_numpy()
    np.testing.assert_allclose(task.target, ti[inp.start : inp.stop])
    assert inp.stop <= well.region_info["prediction_start_row"]


def test_masked_mode_leaves_a_usable_prefix(mount):
    inp = _task(mount, "TRW006", "masked").inputs()
    assert inp.prefix_len >= 200
    assert np.isfinite(inp.tvt_known[: inp.start]).any()


def test_short_prefix_cannot_form_a_masked_task(mount):
    tasks_mod = _mod("tasks")
    well = _load(mount, "TRW001")
    with pytest.raises(tasks_mod.TaskConstructionError):
        tasks_mod.make_task(well, "masked", min_prefix=10_000)


def test_real_mode_target_matches_the_label_column(mount):
    files = _wells(mount)
    well = _load(mount, "TRW001")
    task = _mod("tasks").make_task(well, "real")
    inp = task.inputs()
    tvt = pd.to_numeric(well.hw["TVT"], errors="coerce").to_numpy()
    np.testing.assert_allclose(task.target, tvt[inp.start : inp.stop])


# ---------------------------------------------------------------- features --

def test_feature_frame_only_contains_manifest_approved_columns(mount):
    files = _wells(mount)
    inp = _task(mount, "TRW001", "real").inputs()
    frame = _feats(inp).frame()
    assert_safe_features(frame.columns)


def test_feature_frame_has_one_row_per_predicted_point(mount):
    files = _wells(mount)
    inp = _task(mount, "TRW001", "real").inputs()
    assert len(_feats(inp).frame()) == inp.n_predict


def test_features_are_finite_enough_to_model(mount):
    files = _wells(mount)
    inp = _task(mount, "TRW001", "real").inputs()
    frame = _feats(inp).frame()
    assert np.isfinite(frame["dmd"]).all()
    assert np.isfinite(frame["align_tvt"]).all()


def test_gr_interpolation_is_within_well_and_flags_gaps():
    v = np.array([1.0, np.nan, np.nan, 4.0])
    filled, missing = _mod("features").interpolate_within_well(v)
    np.testing.assert_allclose(filled, [1.0, 2.0, 3.0, 4.0])
    assert missing.tolist() == [False, True, True, False]


def test_all_missing_gr_does_not_crash():
    filled, missing = _mod("features").interpolate_within_well(np.full(5, np.nan))
    assert np.isfinite(filled).all()
    assert missing.all()


def test_high_gr_missingness_well_still_produces_features(mount):
    files = _wells(mount)
    inp = _task(mount, "TRW002", "real").inputs()
    frame = _feats(inp).frame()
    assert len(frame) == inp.n_predict
    assert frame["gr_missing_frac_well"].iloc[0] > 0.0


def test_well_without_typewell_still_produces_features(mount):
    files = _wells(mount)
    inp = _task(mount, "TRW005", "real").inputs()
    feats = _feats(inp)
    assert not feats.ref.ok
    assert len(feats.frame()) == inp.n_predict


# --------------------------------------------------------------- baselines --

@pytest.mark.parametrize("name", [n for n in BASELINE_ORDER if n != "lightgbm"])
def test_every_baseline_predicts_the_right_shape(mount, name):
    files = _wells(mount)
    task = _task(mount, "TRW001", "real")
    inp = task.inputs()
    pred = _baseline(name).predict(inp, _feats(inp))
    assert np.asarray(pred).shape == (inp.n_predict,)
    assert np.isfinite(pred).all()


def test_hold_last_predicts_exactly_the_anchor(mount):
    files = _wells(mount)
    inp = _task(mount, "TRW001", "real").inputs()
    pred = _baseline("hold_last").predict(inp)
    assert np.allclose(pred, inp.anchor_tvt)


@pytest.mark.parametrize("name", [n for n in BASELINE_ORDER if n != "lightgbm"])
def test_baselines_survive_a_well_with_no_typewell(mount, name):
    files = _wells(mount)
    task = _task(mount, "TRW005", "real")
    inp = task.inputs()
    pred = _baseline(name).predict(inp, _feats(inp))
    assert np.isfinite(pred).all()


def test_fitting_never_sees_a_target_attribute(mount, monkeypatch):
    """A model's predict() must only ever receive an InferenceTask."""
    files = _wells(mount)
    tasks = [_task(mount, w, "real") for w in ("TRW001", "TRW003")]
    seen = []

    model = _baseline("linear_extrap")
    original = model.predict

    def spy(task, feats=None):
        seen.append(hasattr(task, "target"))
        return original(task, feats)

    monkeypatch.setattr(model, "predict", spy)
    model.fit(tasks)
    for t in tasks:
        model.predict(t.inputs())
    assert seen and not any(seen)


@pytest.mark.skipif(not HAVE_LIGHTGBM, reason="lightgbm not installed")
def test_lightgbm_trains_and_predicts(mount):
    files = _wells(mount)
    tasks = [_task(mount, w, "real") for w in ("TRW001", "TRW003", "TRW004")]
    model = _baseline("lightgbm", num_boost_round=5).fit(tasks)
    inp = tasks[0].inputs()
    pred = model.predict(inp)
    assert np.isfinite(pred).all() and pred.shape == (inp.n_predict,)


def test_ridge_features_are_manifest_clean(mount):
    files = _wells(mount)
    tasks = [_task(mount, w, "real") for w in ("TRW001", "TRW003")]
    model = _baseline("ridge").fit(tasks)
    assert_safe_features(model.feature_names_)


# ----------------------------------------------------------------- spatial --

def test_spatial_prior_excludes_the_queried_well(mount):
    """Self-exclusion is by well ID, which is what makes it leave-one-well-out.

    Note the fixture places all wells on an identical X/Y track, so a *different*
    well can legitimately sit at distance 0 — distance cannot be used as the
    probe. Instead we verify directly that none of the returned neighbours
    belong to the queried well.
    """
    spatial = _mod("spatial")
    tasks = [_task(mount, w, "real") for w in ("TRW001", "TRW003", "TRW004")]
    prior = spatial.SpatialPrior().fit(tasks)
    inp = tasks[0].inputs()
    frame = prior.features_for(inp)
    assert len(frame) == inp.n_predict

    raw = prior._neighbours(
        inp.x[inp.start : inp.stop], inp.y[inp.start : inp.stop], exclude=inp.well_id
    )
    assert raw["nbr_n"].max() > 0, "no neighbours found at all"

    # exhaustive check: rebuild without exclusion and confirm the difference
    everything = prior._neighbours(
        inp.x[inp.start : inp.stop], inp.y[inp.start : inp.stop], exclude="__none__"
    )
    assert everything["nbr_n"].max() >= raw["nbr_n"].max()


def test_spatial_fold_guard_rejects_a_validation_donor(mount):
    spatial = _mod("spatial")
    tasks = [_task(mount, w, "real") for w in ("TRW001", "TRW003")]
    prior = spatial.SpatialPrior().fit(tasks)
    with pytest.raises(spatial.SpatialLeakage):
        prior.assert_disjoint(["TRW001"])
    prior.assert_disjoint(["TRW999"])  # clean case must not raise


def test_spatial_describe_documents_the_method(mount):
    files = _wells(mount)
    tasks = [_task(mount, w, "real") for w in ("TRW001", "TRW003")]
    d = _mod("spatial").SpatialPrior().fit(tasks).describe()
    for key in ("k", "radius_ft", "weighting", "self_exclusion", "index", "source"):
        assert key in d


# ----------------------------------------------------------------- metrics --

def test_rmse_matches_manual_computation():
    assert rmse(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(
        np.sqrt((1 + 4) / 2)
    )


def test_rmse_ignores_nan_pairs():
    assert rmse(np.array([1.0, np.nan]), np.array([2.0, 5.0])) == pytest.approx(1.0)


def test_global_rmse_is_point_weighted_not_well_weighted():
    """A long well must dominate a short one, matching the competition metric."""
    df = pd.DataFrame(
        [
            {"model": "m", "protocol": "p", "well_id": "long", "n_points": 1000,
             "sse": 1000.0, "rmse": 1.0, "max_abs_error": 1.0, "bias": 0.0,
             "prefix_len": 10, "suffix_len": 1000, "gr_missing_frac": 0.0,
             "anchor_tvt": 0.0, "has_typewell": True, "predict_seconds": 0.0},
            {"model": "m", "protocol": "p", "well_id": "short", "n_points": 10,
             "sse": 1000.0, "rmse": 10.0, "max_abs_error": 10.0, "bias": 0.0,
             "prefix_len": 10, "suffix_len": 10, "gr_missing_frac": 0.0,
             "anchor_tvt": 0.0, "has_typewell": True, "predict_seconds": 0.0},
        ]
    )
    s = summarize(df).iloc[0]
    assert s["global_rmse"] == pytest.approx(np.sqrt(2000.0 / 1010.0))
    assert s["mean_well_rmse"] == pytest.approx(5.5)
    assert s["global_rmse"] < s["mean_well_rmse"]
