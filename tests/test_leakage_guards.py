"""Leakage guards. These are the tests that matter most.

Each one asserts that a *specific* way of cheating is impossible, not merely
discouraged. If any of these fail, no result from this repository is
trustworthy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.manifest as _manifest_mod
import src.validation as _validation_mod
from src.manifest import TRAIN_ONLY_MARKERS

# Symbols are resolved through the module at call time, never bound at import.
#
# `importlib.reload` (used by the `mount` fixture in conftest) updates a
# module's dict *in place*. A function object imported before the reload
# therefore keeps executing against the refreshed globals and raises the NEW
# exception class, while a test holding the OLD class object fails to catch it.
# Going through the module every time makes these tests order-independent.


def _m():
    return _manifest_mod


def _v():
    return _validation_mod


def assert_safe_features(cols):
    return _m().assert_safe_features(cols)


def assert_no_blocked_wells(ids, context="test"):
    return _v().assert_no_blocked_wells(ids, context=context)


PUBLIC_TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]


# ------------------------------------------------------ blocked test wells --

def test_blocked_ids_are_exactly_the_three_public_test_wells():
    assert _v().BLOCKED_WELL_IDS == frozenset(PUBLIC_TEST_WELLS)


@pytest.mark.parametrize("well", PUBLIC_TEST_WELLS)
def test_guard_raises_for_each_public_test_well(well):
    with pytest.raises(_v().BlockedWellError):
        assert_no_blocked_wells(["ok1", well, "ok2"], context="unit test")


def test_guard_passes_for_clean_universe():
    assert_no_blocked_wells(["a", "b", "c"], context="unit test")


def test_guard_message_names_the_offender():
    with pytest.raises(_v().BlockedWellError, match="000d7d20"):
        assert_no_blocked_wells(["000d7d20"], context="unit test")


def test_filter_blocked_removes_public_wells():
    out = _v().filter_blocked(["w1", *PUBLIC_TEST_WELLS, "w2"])
    assert out == ["w1", "w2"]


def test_blocked_wells_cannot_reach_any_fold():
    ids = [f"w{i:03d}" for i in range(30)] + PUBLIC_TEST_WELLS
    folds = _v().make_group_folds(ids, n_splits=5, seed=0)
    for fold in folds:
        assert not set(fold.train_ids) & _v().BLOCKED_WELL_IDS
        assert not set(fold.valid_ids) & _v().BLOCKED_WELL_IDS


def test_fold_construction_is_a_partition():
    ids = [f"w{i:03d}" for i in range(37)]
    folds = _v().make_group_folds(ids, n_splits=5, seed=1)
    seen: list[str] = []
    for fold in folds:
        assert not set(fold.train_ids) & set(fold.valid_ids)
        seen += fold.valid_ids
    assert sorted(seen) == sorted(ids), "every well validated exactly once"


def test_fold_rejects_blocked_id_injected_after_construction():
    Fold = _v().Fold

    with pytest.raises(_v().BlockedWellError):
        Fold(index=0, train_ids=["a", "00bbac68"], valid_ids=["b"])
    with pytest.raises(_v().BlockedWellError):
        Fold(index=0, train_ids=["a"], valid_ids=["b", "00e12e8b"])


# ------------------------------------------------------------- feature gate --

def test_target_column_is_rejected():
    with pytest.raises(_m().FeatureLeakage, match="TVT"):
        assert_safe_features(["dmd", "dz", "TVT"])


@pytest.mark.parametrize("marker", TRAIN_ONLY_MARKERS)
def test_each_train_only_marker_is_rejected(marker):
    with pytest.raises(_m().FeatureLeakage, match=marker):
        assert_safe_features(["dmd", marker])


def test_tvt_input_is_not_a_row_feature():
    """TVT_input is USE_PREFIX_ONLY: it must not appear in a model matrix."""
    with pytest.raises(_m().FeatureLeakage):
        assert_safe_features(["dmd", "TVT_input"])


def test_unaudited_column_is_rejected():
    """The manifest is a whitelist: unknown columns cannot slip through."""
    with pytest.raises(_m().FeatureLeakage, match="mystery"):
        assert_safe_features(["dmd", "mystery_feature"])


def test_typewell_geology_is_rejected_as_train_only():
    """Geology exists in train typewells and NOT in test typewells.

    Full coverage of the corrected row lives in
    ``tests/test_typewell_schema_manifest.py``; this asserts the guard itself.
    """
    with pytest.raises(_m().FeatureLeakage, match="TRAIN-ONLY"):
        assert_safe_features(["dmd", "Typewell Geology"])
    with pytest.raises(_m().FeatureLeakage):
        assert_safe_features(["dmd", "Geology"])


def test_safe_features_pass():
    assert_safe_features(["MD", "X", "Y", "Z", "GR", "Typewell TVT", "Typewell GR"])


def test_case_and_separator_insensitive_rejection():
    for variant in ("tvt", "TVT", "Tvt", "target"):
        with pytest.raises(_m().FeatureLeakage):
            assert_safe_features([variant])


# ----------------------------------------------------------------- manifest --

def test_manifest_contains_every_required_entry():
    names = {f.feature_name for f in _m().MANIFEST}
    required = {
        "MD", "X", "Y", "Z", "GR", "TVT_input", "TVT",
        "Typewell TVT", "Typewell GR", "Typewell Geology",
        *TRAIN_ONLY_MARKERS,
    }
    assert required <= names


def test_manifest_has_all_required_columns():
    frame = _m().manifest_frame()
    for col in (
        "feature_name", "source", "train_availability", "test_availability",
        "available_after_prediction_start", "target_derived",
        "safe_for_inference", "decision", "leakage_risk", "notes",
    ):
        assert col in frame.columns


def test_manifest_marks_markers_as_train_only_and_rejected():
    frame = _m().manifest_frame().set_index("feature_name")
    for marker in TRAIN_ONLY_MARKERS:
        assert frame.loc[marker, "decision"] == "REJECT"
        assert frame.loc[marker, "test_availability"].lower().startswith("no")


def test_manifest_marks_tvt_as_target_and_unsafe():
    row = _m().manifest_frame().set_index("feature_name").loc["TVT"]
    assert row["decision"] == "TARGET"
    assert row["safe_for_inference"] == "no"


def test_manifest_marks_typewell_geology_train_only_and_unsafe():
    """Absent from the test typewell schema -> TRAIN_ANALYSIS_ONLY, never USE."""
    row = _m().manifest_frame().set_index("feature_name").loc["Typewell Geology"]
    assert row["decision"] == "TRAIN_ANALYSIS_ONLY"
    assert row["train_availability"] == "yes"
    assert row["test_availability"] == "no"
    assert row["safe_for_inference"] == "false"
    assert row["leakage_risk"] == "HIGH for final inference"


def test_no_manifest_entry_is_train_only_and_inference_safe():
    """The invariant the Kaggle schema audit caught being violated."""
    assert _m().validate_manifest() == []
    for spec in _m().MANIFEST:
        if spec.train_only:
            assert not spec.claims_inference_safe, spec.feature_name
            assert spec.feature_name not in _m().safe_inference_features()


def test_manifest_marks_the_seven_safe_raw_features_usable():
    frame = _m().manifest_frame().set_index("feature_name")
    for name in ("MD", "X", "Y", "Z", "GR", "Typewell TVT", "Typewell GR"):
        assert frame.loc[name, "decision"] == "USE", name
        assert frame.loc[name, "safe_for_inference"] == "yes", name


def test_no_rejected_feature_is_also_safe():
    assert not set(_m().safe_inference_features()) & set(_m().rejected_features())


def test_manifest_verification_detects_a_schema_change():
    """If markers ever appear in test, the check must notice."""
    train_cols = ["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT", *TRAIN_ONLY_MARKERS]
    honest = _m().verify_manifest_against_data(train_cols, ["MD", "X", "Y", "Z", "GR", "TVT_input"])
    assert honest.set_index("feature_name").loc["ANCC", "agrees"]

    changed = _m().verify_manifest_against_data(train_cols, train_cols)
    assert not changed.set_index("feature_name").loc["ANCC", "agrees"]
