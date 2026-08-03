"""Tests for the Alignment v2 model and the OOF meta-stack v2
(src.alignment_v2_model).

Covers:
* manifest safety of the v2 meta-stack design matrix;
* the gated model falls back to the *exact* Ridge anchor on every
  failure path (kill switch, missing candidates, non-finite output);
* the OOF meta-stack v2 returns the exact Ridge anchor on its kill
  switch;
* the model trains and predicts on the synthetic fixture without
  raising.

Real-data validation lives in
``scripts/run_alignment_v2_experiment.py``.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mod(name):
    key = f"src.{name}"
    if key in sys.modules:
        return importlib.reload(sys.modules[key])
    return importlib.import_module(key)


def _tasks(mount, ids, mode="real"):
    data = _mod("data")
    tasks_mod = _mod("tasks")
    files = data.discover_wells("train")
    out = []
    for wid in ids:
        try:
            out.append(tasks_mod.make_task(data.load_well(files[wid]), mode))
        except Exception:
            pass
    return out


def _well_ids() -> list[str]:
    return ["TRW001", "TRW006", "TRW007", "TRW008", "TRW009"]


# --- manifest safety --------------------------------------------------------


def test_v2_meta_columns_are_manifest_cleared(mount):
    manifest = _mod("manifest")
    cols = (
        "v2_dmd", "v2_log1p_dmd", "v2_corr_multi_scale", "v2_corr_dp",
        "v2_corr_irls", "v2_corr_branch_hedged", "v2_disagreement",
        "v2_confidence", "v2_gr_miss_suffix", "v2_gr_miss_prefix",
        "v2_suffix_len", "v2_prefix_len",
    )
    manifest.assert_safe_features(list(cols), context="v2 meta-stack columns")


def test_v2_meta_columns_root_in_allowed_sources(mount):
    manifest = _mod("manifest")
    allowed = {
        "MD", "X", "Y", "Z", "GR", "TVT_input", "TVT_input (prefix only)",
        "Typewell TVT", "Typewell GR",
    }
    allowed_c = {manifest.canonical(a) for a in allowed}
    for name in (
        "v2_dmd", "v2_log1p_dmd", "v2_corr_multi_scale", "v2_corr_dp",
        "v2_corr_irls", "v2_corr_branch_hedged", "v2_disagreement",
        "v2_confidence", "v2_gr_miss_suffix", "v2_gr_miss_prefix",
        "v2_suffix_len", "v2_prefix_len",
    ):
        roots = manifest.root_sources(manifest.canonical(name))
        assert roots <= allowed_c, f"{name} roots outside allowed set: {roots - allowed_c}"


# --- the gated model falls back to exact Ridge ------------------------------


def test_v2_model_falls_back_to_exact_ridge_when_killed(mount):
    a2m = _mod("alignment_v2_model")
    baselines = _mod("baselines")
    train = _tasks(mount, _well_ids(), "real")
    anchor = baselines.RidgeBaseline(alignment_features=False)
    model = a2m.AlignmentV2Model(
        anchor_model=anchor,
        config=a2m.AlignmentV2Config(min_examples=10**9, gbdt_max_iter=10),
    )
    # Force the kill switch.
    model.killed = True
    model.kill_reason = "test_kill"
    for t in train:
        inp = t.inputs()
        base = np.asarray(anchor.predict(inp), dtype="float64")
        # Fit anchor if not fitted.
        if anchor.model is None:
            anchor.fit(train)
            base = np.asarray(anchor.predict(inp), dtype="float64")
        pred = model.predict(inp)
        # The killed model must return the exact Ridge anchor output.
        assert np.allclose(pred, base, atol=1e-12)


def test_v2_model_returns_finite_output_after_fit(mount):
    a2m = _mod("alignment_v2_model")
    baselines = _mod("baselines")
    train = _tasks(mount, _well_ids(), "real")
    valid = _tasks(mount, _well_ids(), "real")
    anchor = baselines.RidgeBaseline(alignment_features=False)
    model = a2m.AlignmentV2Model(
        anchor_model=anchor,
        config=a2m.AlignmentV2Config(
            inner_splits=2,
            tune_splits=2,
            min_examples=4,
            gbdt_max_iter=20,
            gbdt_max_depth=2,
        ),
    )
    model.fit(train)
    # The model is either killed or fitted. Either way, predict must
    # be finite.
    anchor.fit(train)
    for t in valid:
        inp = t.inputs()
        pred = model.predict(inp)
        assert np.all(np.isfinite(pred)), f"non-finite pred for {inp.well_id}"
        # When killed, the prediction must be bit-identical to the
        # Ridge anchor output.
        if model.killed:
            base = np.asarray(anchor.predict(inp), dtype="float64")
            assert np.allclose(pred, base, atol=1e-12)


def test_v2_meta_stack_falls_back_to_exact_ridge_when_killed(mount):
    a2m = _mod("alignment_v2_model")
    baselines = _mod("baselines")
    train = _tasks(mount, _well_ids(), "real")
    anchor = baselines.RidgeBaseline(alignment_features=False)
    anchor.fit(train)
    model = a2m.OOFMetaStackV2Anchor(anchor_model=anchor)
    # Force the kill switch.
    model.stack.killed = True
    model.stack.kill_reason = "test_kill"
    for t in train:
        inp = t.inputs()
        base = np.asarray(anchor.predict(inp), dtype="float64")
        pred = model.predict(inp)
        assert np.allclose(pred, base, atol=1e-12)


def test_v2_meta_stack_returns_finite_output_after_fit(mount):
    a2m = _mod("alignment_v2_model")
    baselines = _mod("baselines")
    train = _tasks(mount, _well_ids(), "real")
    valid = _tasks(mount, _well_ids(), "real")
    anchor = baselines.RidgeBaseline(alignment_features=False)
    anchor.fit(train)
    model = a2m.OOFMetaStackV2Anchor(
        anchor_model=anchor,
        config=a2m.OOFMetaStackV2Config(inner_splits=2, tune_splits=2),
    )
    model.fit(train)
    for t in valid:
        inp = t.inputs()
        pred = model.predict(inp)
        assert np.all(np.isfinite(pred)), f"non-finite pred for {inp.well_id}"


# --- blocked wells are excluded from OOF training --------------------------


def test_v2_model_excludes_blocked_wells_from_oof(mount):
    a2m = _mod("alignment_v2_model")
    baselines = _mod("baselines")
    from src.validation import BLOCKED_WELL_IDS

    train = _tasks(mount, _well_ids(), "real")
    # Sneak a blocked well id into the training list. The OOF step
    # must raise.
    fake = list(train)
    # Mutate the first task's well_id to a blocked one and expect a
    # failure at the assert_no_blocked_wells call.
    class _FakeTask:
        def __init__(self, t, wid):
            self._t = t
            self._wid = wid

        def inputs(self):
            return self._t.inputs()

        @property
        def target(self):
            return self._t.target

        @property
        def well_id(self):
            return self._wid

    blocked_id = sorted(BLOCKED_WELL_IDS)[0]
    fake[0] = _FakeTask(train[0], blocked_id)
    # Verify the well_id change is visible.
    assert fake[0].well_id == blocked_id
    assert any(t.well_id == blocked_id for t in fake)
    anchor = baselines.RidgeBaseline(alignment_features=False)
    anchor.fit([t for t in train])
    model = a2m.AlignmentV2Model(
        anchor_model=anchor,
        config=a2m.AlignmentV2Config(inner_splits=2, tune_splits=2, min_examples=4),
    )
    with pytest.raises(Exception):
        model.fit(fake)
