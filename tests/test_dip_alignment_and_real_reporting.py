"""Safety and reporting checks for the isolated dip-constrained experiment."""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd


def _mod(name):
    key = f"src.{name}"
    return importlib.reload(sys.modules[key]) if key in sys.modules else importlib.import_module(key)


def test_dip_alignment_uses_inference_task_and_hits_target_free_cache(mount, tmp_path):
    data = _mod("data")
    tasks = _mod("tasks")
    features = _mod("features")
    cache_mod = _mod("cache")
    files = data.discover_wells("train")
    task = tasks.make_task(data.load_well(files[sorted(files)[0]]), "real").inputs()
    task.assert_no_target()
    cache = cache_mod.FeatureCache(tmp_path / "alignment-cache")

    first = features.build_features(
        task,
        alignment=False,
        dip_alignment=True,
        alignment_cache=cache,
        cache_context={"dataset_version": "test", "fold": 0, "protocol": "unseen_well"},
    )
    assert first.dip_align["track"].shape == (task.n_rows,)
    assert first.dip_align["confidence"].shape == (task.n_rows,)
    assert not first.dip_align["cache_hit"]
    # A second build sees the same target-free artifact.  Its key includes the
    # boundary, so it cannot mix a real suffix with a same-well mask.
    second = features.build_features(
        task,
        alignment=False,
        dip_alignment=True,
        alignment_cache=cache,
        cache_context={"dataset_version": "test", "fold": 0, "protocol": "unseen_well"},
    )
    assert second.dip_align["cache_hit"]
    assert cache.stats.hits >= 1
    assert cache.stats.writes >= 1


def test_dip_alignment_model_reports_confidence_and_fallback_without_target_access(mount):
    data = _mod("data")
    tasks = _mod("tasks")
    features = _mod("features")
    baselines = _mod("baselines")
    files = data.discover_wells("train")
    # TRW006 has the fixture's long visible prefix required by masked mode.
    task = tasks.make_task(data.load_well(files["TRW006"]), "masked")
    inp = task.inputs()
    assert not hasattr(inp, "target")
    feats = features.build_features(inp, alignment=False, dip_alignment=True)
    model = baselines.DipConstrainedGRTypewellAlignment()
    pred = model.predict(inp, feats)
    diagnostic = model.prediction_diagnostics(inp, feats, pred)
    assert pred.shape == (inp.n_predict,)
    assert np.isfinite(pred).all()
    assert "alignment_confidence_mean" in diagnostic
    assert "fallback_points" in diagnostic
    expected_fallback = (
        inp.n_predict
        if not feats.dip_align["ok"]
        else int(np.count_nonzero(feats.dip_align["confidence"][inp.start : inp.stop] < model.min_confidence))
    )
    assert diagnostic["fallback_points"] == expected_fallback
    # Typewell Geology is carried by InferenceTask for train-side analysis, but
    # cannot influence the alignment: perturbing it leaves the target-free
    # feature track unchanged.
    from dataclasses import replace
    poisoned = replace(inp, tw_geology=np.full(10, "FORBIDDEN"))
    poisoned_feats = features.build_features(poisoned, alignment=False, dip_alignment=True)
    np.testing.assert_allclose(
        feats.dip_align["dip_prediction"], poisoned_feats.dip_align["dip_prediction"], equal_nan=True
    )
    np.testing.assert_allclose(
        feats.dip_align["track"], poisoned_feats.dip_align["track"], equal_nan=True
    )


def test_real_reporting_keeps_protocols_separate_and_writes_requested_files(tmp_path):
    real = _mod("real_reporting")
    rows = []
    for protocol in ("same_well_masked", "unseen_well"):
        for rank in range(3):
            common = {
                "protocol": protocol, "well_id": f"{protocol[:2]}{rank}",
                "n_points": 100 + rank, "suffix_len": 100 + rank,
                "prefix_len": 500 + rank, "gr_missing_frac": 0.01 * rank,
                "sse": (rank + 1) * 100.0, "rmse": float(rank + 1),
                "max_abs_error": float(rank + 2), "bias": 0.0, "fold": rank,
                "has_typewell": True, "target_min": -10.0, "target_max": 10.0,
                "target_range": 20.0, "scored_exact_suffix": True,
                "trajectory_curvature_deg_per_1000ft": float(rank),
                "alignment_confidence_mean": 0.5, "alignment_confidence_p10": 0.4,
                "alignment_ok": True, "alignment_failure_reason": "",
                "fallback_points": 0, "alignment_cache_hit": False,
            }
            rows.append({"model": "ridge", **common})
            rows.append({"model": "ridge_spatial", **common, "sse": common["sse"] * 1.01})
            rows.append({"model": "dip_constrained_alignment", **common, "fallback_points": 4})
    well = pd.DataFrame(rows)
    well.to_csv(tmp_path / "well_level_validation.csv", index=False)
    # Result rows are sufficient for the spatial A/B; their exact values are
    # deliberately generated in this test fixture, not a competition claim.
    result = []
    for protocol in ("same_well_masked", "unseen_well"):
        for model in ("ridge", "ridge_spatial"):
            result.append({
                "protocol": protocol, "model": model, "global_rmse": 1.0,
                "median_well_rmse": 1.0, "worst10_well_rmse": 1.0,
            })
    pd.DataFrame(result).to_csv(tmp_path / "validation_results.csv", index=False)
    pd.DataFrame([
        {"protocol": "same_well_masked", "feature": "nbr_n", "n_prediction_rows": 100,
         "n_populated": 100, "non_constant": True, "n_unique_finite": 3}
    ]).to_csv(tmp_path / "spatial_feature_diagnostics.csv", index=False)

    written = real.write_real_analysis(tmp_path)
    names = {p.name for p in written}
    assert {
        "protocol_comparison_real.md", "error_analysis_real.csv",
        "gr_missingness_error_real.csv", "suffix_length_error_real.csv",
        "worst_wells_real.csv", "spatial_ablation_real.md",
        "dip_constrained_alignment_ablation.csv", "dip_constrained_alignment_real.md",
    } <= names
    worst = pd.read_csv(tmp_path / "worst_wells_real.csv")
    assert set(worst["protocol"]) == {"same_well_masked", "unseen_well"}
    # The report must not contain a made-up combined protocol.
    text = (tmp_path / "protocol_comparison_real.md").read_text()
    assert "never averages them" in text
