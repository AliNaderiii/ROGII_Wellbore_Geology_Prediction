"""Rejection of the direct dip-constrained alignment model.

The completed real validation run (770 eligible wells, both protocols,
cross-fitted by well ID) showed the direct alignment trajectory is decisively
worse than Ridge:

    same_well_masked  Ridge 29.452  vs  alignment 277.654   (+248.202 RMSE)
    unseen_well       Ridge 14.441  vs  alignment  96.545   (+82.104 RMSE)

These tests make that decision structural rather than remembered: the model is
marked REJECTED, the recorded evidence has to stay consistent with the run, and
any attempt to route it into a final predictor or an ensemble branch fails.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest


def _mod(name):
    key = f"src.{name}"
    return importlib.reload(sys.modules[key]) if key in sys.modules else importlib.import_module(key)


# The figures the rejection rests on, transcribed once from the completed run.
REAL_RESULT = {
    "same_well_masked": {
        "ridge": 29.452,
        "alignment": 277.654,
        "delta": 248.202,
        "mean_confidence": 0.1577,
        "fallback_fraction": 0.675,
    },
    "unseen_well": {
        "ridge": 14.441,
        "alignment": 96.545,
        "delta": 82.104,
        "mean_confidence": 0.3197,
        "fallback_fraction": 0.340,
    },
}


# ------------------------------------------------------- registry contents --

def test_direct_alignment_is_marked_rejected():
    ms = _mod("model_status")
    status = ms.status_of("dip_constrained_alignment")
    assert status.status == ms.REJECTED
    assert status.is_rejected
    assert ms.is_rejected("dip_constrained_alignment")
    assert "dip_constrained_alignment" in ms.rejected_models()
    # The rejection must carry its evidence, not just a verdict.
    assert status.source_run
    assert "770" in status.source_run


def test_recorded_evidence_matches_the_completed_validation_run():
    ms = _mod("model_status")
    status = ms.status_of("dip_constrained_alignment")
    for protocol, expected in REAL_RESULT.items():
        e = status.evidence_for(protocol)
        assert e.n_wells == 770
        assert e.ridge_global_rmse == pytest.approx(expected["ridge"], abs=1e-3)
        assert e.model_global_rmse == pytest.approx(expected["alignment"], abs=1e-3)
        assert e.delta_vs_ridge == pytest.approx(expected["delta"], abs=1e-3)
        assert e.mean_confidence == pytest.approx(expected["mean_confidence"], abs=1e-4)
        assert e.fallback_fraction == pytest.approx(expected["fallback_fraction"], abs=1e-3)
        # The rejection criterion itself: strictly worse than Ridge.
        assert e.worse_than_ridge


def test_rejection_holds_under_both_protocols_not_just_one():
    ms = _mod("model_status")
    status = ms.status_of("dip_constrained_alignment")
    protocols = {e.protocol for e in status.evidence}
    assert protocols == {ms.PROTOCOL_A, ms.PROTOCOL_B}
    assert all(e.worse_than_ridge for e in status.evidence)


def test_protocol_names_agree_with_the_validation_module():
    ms = _mod("model_status")
    validation = _mod("validation")
    assert ms.PROTOCOL_A == validation.PROTOCOL_A
    assert ms.PROTOCOL_B == validation.PROTOCOL_B


def test_ridge_is_not_rejected_and_unknown_models_are_never_approved():
    ms = _mod("model_status")
    assert not ms.is_rejected("ridge")
    unknown = ms.status_of("particle_filter")
    assert unknown.status == ms.CANDIDATE
    assert not unknown.is_rejected
    # Crucially, "not rejected" must not read as "approved".
    assert unknown.status != ms.APPROVED


# --------------------------------------------------------- enforcement path --

def test_promoting_the_rejected_model_raises():
    ms = _mod("model_status")
    with pytest.raises(ms.RejectedModelError) as exc:
        ms.assert_not_rejected(["ridge", "dip_constrained_alignment"], context="final ensemble")
    assert "final ensemble" in str(exc.value)
    assert "dip_constrained_alignment" in str(exc.value)


def test_enforcement_accepts_model_objects_not_only_names(mount):
    ms = _mod("model_status")
    baselines = _mod("baselines")
    rejected = baselines.BASELINES["dip_constrained_alignment"]()
    with pytest.raises(ms.RejectedModelError):
        ms.assert_not_rejected([rejected], context="final predictor")
    # A clean roster passes.
    ms.assert_not_rejected([baselines.BASELINES["ridge"]()], context="final predictor")


def test_a_clean_model_roster_passes_enforcement():
    ms = _mod("model_status")
    ms.assert_not_rejected(["ridge", "lightgbm", "hold_last"], context="final ensemble")


def test_ablation_branches_are_all_permitted():
    """No ablation branch may be a rejected model."""
    ms = _mod("model_status")
    ablation = _mod("ablation")
    ms.assert_not_rejected(ablation.BRANCH_ORDER, context="ablation branches")
    assert not any(ms.is_rejected(b) for b in ablation.BRANCH_ORDER)


def test_rejected_model_still_runs_as_a_diagnostic(mount):
    """Rejection blocks *promotion*, not measurement.

    The model must remain runnable so the failure can keep being reproduced;
    the gate is on the final/ensemble path only.
    """
    ms = _mod("model_status")
    data = _mod("data")
    tasks = _mod("tasks")
    features = _mod("features")
    baselines = _mod("baselines")
    files = data.discover_wells("train")
    inp = tasks.make_task(data.load_well(files["TRW006"]), "masked").inputs()
    feats = features.build_features(inp, alignment=False, dip_alignment=True)
    model = baselines.DipConstrainedGRTypewellAlignment()
    pred = model.predict(inp, feats)
    assert pred.shape == (inp.n_predict,)
    assert np.isfinite(pred).all()
    assert ms.is_rejected(model.name)


# ------------------------------------------------------------- report layer --

def test_status_table_reports_the_rejection_per_protocol():
    ms = _mod("model_status")
    table = pd.DataFrame(ms.status_table())
    rows = table[table["model"] == "dip_constrained_alignment"]
    assert set(rows["protocol"]) == {ms.PROTOCOL_A, ms.PROTOCOL_B}
    assert (rows["status"] == ms.REJECTED).all()
    assert (rows["delta_vs_ridge"] > 0).all()


def test_alignment_report_states_the_rejection(tmp_path):
    """The generated A/B report must carry the REJECTED verdict, not just numbers."""
    real = _mod("real_reporting")
    rows = []
    for protocol in ("same_well_masked", "unseen_well"):
        for rank in range(3):
            common = {
                "protocol": protocol, "well_id": f"{protocol[:2]}{rank}",
                "n_points": 100, "suffix_len": 100, "prefix_len": 500,
                "gr_missing_frac": 0.0, "sse": 100.0, "rmse": 1.0,
                "max_abs_error": 1.0, "bias": 0.0, "fold": rank,
                "has_typewell": True, "target_min": -10.0, "target_max": 10.0,
                "target_range": 20.0, "scored_exact_suffix": True,
                "trajectory_curvature_deg_per_1000ft": 1.0,
                "alignment_confidence_mean": 0.2, "alignment_confidence_p10": 0.1,
                "alignment_ok": True, "alignment_failure_reason": "",
                "fallback_points": 0, "alignment_cache_hit": False,
            }
            rows.append({"model": "ridge", **common})
            rows.append({"model": "dip_constrained_alignment", **common, "sse": 900.0})
    pd.DataFrame(rows).to_csv(tmp_path / "well_level_validation.csv", index=False)
    pd.DataFrame(
        [
            {"protocol": p, "model": m, "global_rmse": 1.0, "median_well_rmse": 1.0,
             "worst10_well_rmse": 1.0}
            for p in ("same_well_masked", "unseen_well")
            for m in ("ridge", "ridge_spatial")
        ]
    ).to_csv(tmp_path / "validation_results.csv", index=False)

    real.write_real_analysis(tmp_path)
    text = (tmp_path / "dip_constrained_alignment_real.md").read_text()
    assert "REJECTED" in text
    assert "assert_not_rejected" in text
    # The report must say the model is barred from ensembles, in words.
    assert "ensemble" in text.lower()


# ------------------------------------------------------------ diagnostics --

def test_diagnostics_never_read_the_target_into_a_feature(mount, monkeypatch):
    """The diagnostic script must build features from a target-free task.

    A spy on `build_features` asserts every call receives an object with no
    `target` attribute, which is what structurally prevents a diagnostic from
    becoming a leak.
    """
    import scripts.diagnose_dip_alignment as diag

    data = _mod("data")
    tasks = _mod("tasks")
    features = _mod("features")
    files = data.discover_wells("train")
    task = tasks.make_task(data.load_well(files["TRW006"]), "masked")

    seen = []
    original = features.build_features

    def spy(t, **kw):
        seen.append(hasattr(t, "target"))
        return original(t, **kw)

    monkeypatch.setattr(diag, "build_features", spy)
    row = diag.diagnose_well(task, "same_well_masked")
    assert row is not None
    assert seen and not any(seen)


def test_diagnostics_answer_every_numbered_question(mount):
    """All ten analysis questions must resolve to a computed row."""
    import scripts.diagnose_dip_alignment as diag

    data = _mod("data")
    tasks = _mod("tasks")
    files = data.discover_wells("train")
    rows = []
    for wid in ("TRW006", "TRW007", "TRW008"):
        row = diag.diagnose_well(tasks.make_task(data.load_well(files[wid]), "real"), "unseen_well")
        if row is not None:
            rows.append(row)
    assert rows
    summary = diag.summarize(pd.DataFrame(rows))
    assert set(summary["question"]) == {
        "Q1", "Q1b", "Q2", "Q3", "Q4", "Q4b", "Q5", "Q6", "Q7", "Q8", "Q9", "Q9b"
    }
    # Every question must carry a measured value, not a placeholder.
    assert summary["measures"].str.len().gt(0).all()


def test_diagnostics_report_the_hard_coded_z_coefficient(mount):
    """The -1 dTVT/dZ assumption must be surfaced against its empirical value."""
    import scripts.diagnose_dip_alignment as diag

    data = _mod("data")
    tasks = _mod("tasks")
    files = data.discover_wells("train")
    row = diag.diagnose_well(tasks.make_task(data.load_well(files["TRW006"]), "real"), "unseen_well")
    assert row["assumed_dtvt_dz"] == -1.0
    assert np.isfinite(row["empirical_dtvt_dz"])
    assert row["dtvt_dz_error"] == pytest.approx(row["empirical_dtvt_dz"] + 1.0)


def test_diagnostics_survive_a_well_with_no_typewell(mount):
    import scripts.diagnose_dip_alignment as diag

    data = _mod("data")
    tasks = _mod("tasks")
    files = data.discover_wells("train")
    row = diag.diagnose_well(tasks.make_task(data.load_well(files["TRW005"]), "real"), "unseen_well")
    # Either the well is skipped or it is reported as having no typewell —
    # never silently credited with one.
    assert row is None or row["has_typewell"] is False
