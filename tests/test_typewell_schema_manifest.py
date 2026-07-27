"""Regression tests for the Typewell schema asymmetry.

The Kaggle schema audit established:

    TRAIN typewell columns: ['TVT', 'GR', 'Geology']
    TEST  typewell columns: ['TVT', 'GR']

The manifest previously claimed ``Typewell Geology`` was available in *both*
splits, with ``decision=USE_ALIGNMENT_ONLY``. That is a train/serve skew: any
alignment check, post-processing rule or calibration built on the column would
run during validation and be unavailable at inference.

These tests pin the corrected behaviour so the regression cannot return. They
are deliberately written against the literal audited column lists rather than
against a fixture, so they keep testing the real schema fact.
"""
from __future__ import annotations

import importlib

import pytest

import src.manifest as _manifest_mod

# The audited schemas, stated literally. If the organisers ever change them,
# these constants — and the manifest — must be updated together, deliberately.
TRAIN_TYPEWELL_COLUMNS = ["TVT", "GR", "Geology"]
TEST_TYPEWELL_COLUMNS = ["TVT", "GR"]

TRAIN_HW_COLUMNS = [
    "MD", "X", "Y", "Z", "GR", "TVT_input", "TVT",
    "ANCC", "ASTNL", "ASTNU", "BUDA", "EGFDL", "EGFDU",
]
TEST_HW_COLUMNS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]

#: Requirement 6 — the only raw columns admissible at inference.
EXPECTED_SAFE_RAW = [
    "MD", "X", "Y", "Z", "GR", "TVT_input", "Typewell TVT", "Typewell GR",
]


def _m():
    """Resolve through the module so the `mount` fixture's reload is honoured."""
    return _manifest_mod


def _verify(**kw):
    return _m().verify_manifest_against_data(
        TRAIN_HW_COLUMNS, TEST_HW_COLUMNS,
        train_tw_columns=TRAIN_TYPEWELL_COLUMNS,
        test_tw_columns=TEST_TYPEWELL_COLUMNS,
        **kw,
    )


# ------------------------------------------------- the corrected manifest --

def test_typewell_geology_row_matches_the_required_correction():
    """Every field of the corrected manifest row, asserted individually."""
    row = _m().manifest_frame().set_index("feature_name").loc["Typewell Geology"]

    assert row["source"].startswith("typewell CSV")
    assert row["train_availability"] == "yes"
    assert row["test_availability"] == "no"
    assert row["available_after_prediction_start"] == "no"
    assert row["target_derived"] == "false, but train-only geological metadata"
    assert row["safe_for_inference"] == "false"
    assert row["decision"] == "TRAIN_ANALYSIS_ONLY"
    assert row["leakage_risk"] == "HIGH for final inference"

    notes = row["notes"]
    assert "present in train" in notes
    assert "absent from test" in notes
    assert "EDA" in notes
    assert "never enter the final Test feature matrix" in notes


def test_typewell_geology_is_no_longer_use_alignment_only():
    """The stale decision must not survive anywhere in the manifest."""
    decisions = {f.decision for f in _m().MANIFEST}
    assert "USE_ALIGNMENT_ONLY" not in decisions
    assert "USE_ALIGNMENT_ONLY" not in _m().DECISIONS


def test_typewell_geology_is_train_only_and_excluded_from_inference():
    spec = next(f for f in _m().MANIFEST if f.feature_name == "Typewell Geology")
    assert spec.in_train
    assert not spec.in_test
    assert spec.train_only
    assert not spec.claims_inference_safe
    assert "Typewell Geology" not in _m().safe_inference_features()
    assert "Typewell Geology" in _m().rejected_features()
    assert "Typewell Geology" in _m().train_only_features()
    assert "Typewell Geology" in _m().train_analysis_only_features()


# ------------------------------------------- schema verification behaviour --

def test_verification_sees_geology_in_train_but_not_test():
    """The core regression: the column must be observed asymmetrically.

    Before the fix a name-mangling bug ("Typewell " + canonical("Geology") ->
    "Typewell Typewell Geology") reported the column as absent from *both*
    splits, which hid the asymmetry behind a symmetric mismatch.
    """
    row = _verify().set_index("feature_name").loc["Typewell Geology"]
    assert row["observed_in_train"] is True or row["observed_in_train"]
    assert not row["observed_in_test"]
    assert row["observed_train_only"]
    assert row["agrees"], "corrected manifest must agree with the audited schema"
    assert not row["train_only_but_marked_available"]


def test_verification_agrees_with_the_audited_schemas_everywhere():
    frame = _verify()
    disagreements = frame.loc[~frame["agrees"], "feature_name"].tolist()
    assert not disagreements, f"manifest disagrees for: {disagreements}"


def test_assert_manifest_matches_data_passes_on_the_audited_schemas():
    frame = _m().assert_manifest_matches_data(
        TRAIN_HW_COLUMNS, TEST_HW_COLUMNS,
        train_tw_columns=TRAIN_TYPEWELL_COLUMNS,
        test_tw_columns=TEST_TYPEWELL_COLUMNS,
    )
    assert len(frame) > 0
    assert not frame["train_only_but_marked_available"].any()


def test_audited_schema_constants_match_the_documented_audit():
    assert list(_m().AUDITED_TRAIN_TYPEWELL_COLUMNS) == TRAIN_TYPEWELL_COLUMNS
    assert list(_m().AUDITED_TEST_TYPEWELL_COLUMNS) == TEST_TYPEWELL_COLUMNS
    _m().assert_audited_schemas()  # must not raise


def test_typewell_column_canonicalisation_is_not_double_prefixed():
    """`Typewell ` + canonical(c) was the original name-mangling defect."""
    assert _m().canonical_typewell("Geology") == "Typewell Geology"
    assert _m().canonical_typewell("TVT") == "Typewell TVT"
    assert _m().canonical_typewell("GR") == "Typewell GR"
    assert _m().canonical_typewell("Typewell Geology") == "Typewell Geology"


def test_observed_schema_reports_the_two_typewell_schemas_separately():
    train, test = _m().observed_schema(
        TRAIN_HW_COLUMNS, TEST_HW_COLUMNS,
        train_tw_columns=TRAIN_TYPEWELL_COLUMNS,
        test_tw_columns=TEST_TYPEWELL_COLUMNS,
    )
    assert "Typewell Geology" in train
    assert "Typewell Geology" not in test
    assert {"Typewell TVT", "Typewell GR"} <= train
    assert {"Typewell TVT", "Typewell GR"} <= test


# -------------------------------------------------------- fail-loud checks --

def test_train_only_feature_marked_available_in_test_is_rejected_loudly():
    """Requirement 4: the runner's gate must raise, not warn.

    Simulated by telling the verifier that Geology *is* present in the test
    typewell — i.e. the false claim the old manifest made. Because the manifest
    now says test=no, this shows up as a disagreement and must raise.
    """
    with pytest.raises(_m().SchemaVerificationError) as exc:
        _m().assert_manifest_matches_data(
            TRAIN_HW_COLUMNS, TEST_HW_COLUMNS,
            train_tw_columns=TRAIN_TYPEWELL_COLUMNS,
            test_tw_columns=TRAIN_TYPEWELL_COLUMNS,  # wrong on purpose
        )
    assert "Typewell Geology" in str(exc.value)


def test_stale_manifest_marking_geology_test_available_is_caught_by_schema_check():
    """Requirement 7, reproducing the exact pre-fix state.

    The old row (``test_availability=yes``, ``decision=USE_ALIGNMENT_ONLY``) is
    *internally* self-consistent — nothing about the document alone reveals the
    error. It is only a lie relative to the data. So the schema gate, not the
    document gate, is what has to catch it, and it must raise rather than warn.
    """
    import dataclasses

    m = _m()
    stale = dataclasses.replace(
        next(f for f in m.MANIFEST if f.feature_name == "Typewell Geology"),
        train_availability="yes",
        test_availability="yes",
        available_after_prediction_start="yes",
        safe_for_inference="yes",
        decision="USE",
        leakage_risk="low",
    )
    patched = tuple(
        stale if f.feature_name == "Typewell Geology" else f for f in m.MANIFEST
    )

    original = m.MANIFEST
    try:
        m.MANIFEST = patched
        # The document alone looks fine — that is the trap.
        assert m.validate_manifest() == []
        # Against the real schemas it must fail, loudly and by name.
        with pytest.raises(m.SchemaVerificationError) as exc:
            m.assert_manifest_matches_data(
                TRAIN_HW_COLUMNS,
                TEST_HW_COLUMNS,
                train_tw_columns=TRAIN_TYPEWELL_COLUMNS,
                test_tw_columns=TEST_TYPEWELL_COLUMNS,
            )
        message = str(exc.value)
        assert "Typewell Geology" in message
        assert "TRAIN only" in message
        assert "refusing to proceed" in message

        flagged = m.verify_manifest_against_data(
            TRAIN_HW_COLUMNS,
            TEST_HW_COLUMNS,
            train_tw_columns=TRAIN_TYPEWELL_COLUMNS,
            test_tw_columns=TEST_TYPEWELL_COLUMNS,
        ).set_index("feature_name").loc["Typewell Geology"]
        assert flagged["train_only_but_marked_available"]
    finally:
        m.MANIFEST = original

    # The real manifest is untouched and still correct.
    assert _m().validate_manifest() == []
    _m().assert_audited_schemas()


def test_a_manifest_claiming_a_train_only_feature_is_safe_cannot_be_built():
    """The dataclass itself refuses the inconsistent row."""
    with pytest.raises(_m().ManifestInconsistency):
        _m().FeatureSpec(
            feature_name="Typewell Geology",
            source="typewell CSV",
            train_availability="yes",
            test_availability="no",
            available_after_prediction_start="no",
            target_derived="no",
            safe_for_inference="yes",
            decision="USE",          # the defect
            leakage_risk="none",
            notes="deliberately inconsistent",
        )


def test_validate_manifest_flags_a_train_only_feature_marked_usable():
    """A hand-built inconsistent manifest is caught by validate_manifest."""
    m = _m()
    # Build the row via object.__setattr__ so __post_init__ cannot veto it —
    # this simulates a manifest edited carelessly rather than constructed.
    bad = object.__new__(m.FeatureSpec)
    for field, value in {
        "feature_name": "Typewell Geology",
        "source": "typewell CSV",
        "train_availability": "yes",
        "test_availability": "no",
        "available_after_prediction_start": "no",
        "target_derived": "no",
        "safe_for_inference": "yes",
        "decision": "USE",
        "leakage_risk": "none",
        "notes": "",
        "tier": "raw",
        "used_by": "",
        "parents": (),
    }.items():
        object.__setattr__(bad, field, value)

    problems = m.validate_manifest((bad,))
    assert problems
    assert any("Typewell Geology" in p for p in problems)
    with pytest.raises(m.ManifestInconsistency):
        m.assert_manifest_valid((bad,))


def test_derived_feature_cannot_launder_a_train_only_parent():
    """A USE feature whose parent is train-only must be rejected."""
    m = _m()
    child = m.FeatureSpec(
        feature_name="geology_onehot",
        source="derived from Typewell Geology",
        train_availability="yes (computed)",
        test_availability="yes (computed)",
        available_after_prediction_start="yes",
        target_derived="no",
        safe_for_inference="yes",
        decision="USE",
        leakage_risk="none",
        notes="one-hot of the typewell formation label",
        tier="derived",
        parents=("Typewell Geology",),
    )
    problems = m.validate_manifest(m.MANIFEST + (child,))
    assert any("geology_onehot" in p and "Typewell Geology" in p for p in problems)


# ------------------------------------------- the final inference feature set --

def test_final_inference_features_only_use_the_approved_raw_sources():
    """Requirement 6, checked transitively rather than by name."""
    provenance = _m().assert_inference_provenance()
    roots = {r for rs in provenance.values() for r in rs}
    assert roots <= set(EXPECTED_SAFE_RAW)
    assert "Typewell Geology" not in roots
    assert not roots & set(_m().TRAIN_ONLY_MARKERS)
    assert "TVT" not in roots


def test_safe_raw_inference_sources_are_exactly_the_approved_eight():
    assert list(_m().SAFE_RAW_INFERENCE_SOURCES) == EXPECTED_SAFE_RAW


def test_tvt_input_is_prefix_only_not_a_row_feature():
    """TVT_input is an approved *source* but never a matrix column."""
    assert "TVT_input" in _m().SAFE_RAW_INFERENCE_SOURCES
    assert "TVT_input" not in _m().safe_inference_features()
    assert "TVT_input" in _m().prefix_only_features()
    with pytest.raises(_m().FeatureLeakage):
        _m().assert_safe_features(["dmd", "TVT_input"])


@pytest.mark.parametrize(
    "spelling",
    ["Geology", "geology", "GEOLOGY", "Typewell Geology", "typewell_geology",
     "Typewell_Geology", "tw_geology", "Formation", "formation", "facies"],
)
def test_every_geology_spelling_is_excluded(spelling):
    assert _m().canonical(spelling) == "Typewell Geology"
    with pytest.raises(_m().FeatureLeakage, match="TRAIN-ONLY"):
        _m().assert_safe_features(["dmd", spelling])


@pytest.mark.parametrize("marker", ["ANCC", "ASTNL", "ASTNU", "BUDA", "EGFDL", "EGFDU"])
def test_formation_markers_are_excluded_because_absent_in_test(marker):
    """Requirement 8: markers are out, and for the documented reason."""
    spec = next(f for f in _m().MANIFEST if f.feature_name == marker)
    assert spec.train_only
    assert spec.decision == "REJECT"
    assert marker not in _m().safe_inference_features()
    with pytest.raises(_m().FeatureLeakage):
        _m().assert_safe_features(["dmd", marker])


def test_the_built_feature_matrix_passes_the_inference_gate():
    """The columns the code actually produces, not just the manifest."""
    from src.features import FEATURE_COLUMNS
    from src.spatial import SPATIAL_COLUMNS

    _m().assert_inference_matrix(FEATURE_COLUMNS, context="test")
    _m().assert_inference_matrix(
        list(FEATURE_COLUMNS) + list(SPATIAL_COLUMNS), context="test"
    )
    canon = {_m().canonical(c) for c in FEATURE_COLUMNS}
    assert "Typewell Geology" not in canon
    assert not canon & set(_m().TRAIN_ONLY_MARKERS)
    assert "TVT" not in canon


def test_assert_inference_matrix_rejects_geology():
    with pytest.raises(_m().FeatureLeakage):
        _m().assert_inference_matrix(["dmd", "dz", "Typewell Geology"])


# ----------------------------------------------------- loader-level checks --

def test_task_carries_geology_for_analysis_but_features_never_read_it(mount):
    """Geology may be loaded for EDA; it must not reach the feature matrix."""
    import src.data as data_mod
    import src.features as features_mod
    import src.tasks as tasks_mod

    data = importlib.reload(data_mod)
    tasks = importlib.reload(tasks_mod)
    features = importlib.reload(features_mod)

    well = data.load_well("TRW001", "train")
    task = tasks.make_task(well, "real").inputs()

    # The fixture's train typewell mirrors the real one: TVT, GR, Geology.
    assert list(well.tw.columns) == TRAIN_TYPEWELL_COLUMNS
    assert task.tw_geology is not None, "train-side analysis still gets the labels"

    frame = features.build_features(task, alignment=False).frame()
    canon = {_manifest_mod.canonical(c) for c in frame.columns}
    assert "Typewell Geology" not in canon
    features.validate_feature_frame(frame)


def test_test_split_typewell_has_no_geology_column(mount):
    """The fixture must mirror the real test schema, or these tests prove nothing."""
    import src.data as data_mod

    data = importlib.reload(data_mod)
    well = data.load_well("TSW001", "test")
    assert list(well.tw.columns) == TEST_TYPEWELL_COLUMNS
    assert "geology" not in well.tw_roles


def test_verification_against_the_fixture_mount_agrees(mount):
    """End-to-end: real loader output, real verification, no disagreement."""
    import src.data as data_mod

    data = importlib.reload(data_mod)
    tr = data.load_well("TRW001", "train")
    te = data.load_well("TSW001", "test")

    frame = _manifest_mod.assert_manifest_matches_data(
        tr.hw.columns, te.hw.columns,
        train_tw_columns=list(tr.tw.columns),
        test_tw_columns=list(te.tw.columns),
    )
    row = frame.set_index("feature_name").loc["Typewell Geology"]
    assert row["observed_in_train"] and not row["observed_in_test"]
    assert not frame["train_only_but_marked_available"].any()
