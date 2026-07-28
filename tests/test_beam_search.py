"""Safety, cache and Ridge-integration tests for Beam Search features."""
from __future__ import annotations

from dataclasses import replace
import importlib
import sys

import numpy as np


def _mod(name):
    key = f"src.{name}"
    return importlib.reload(sys.modules[key]) if key in sys.modules else importlib.import_module(key)


def _task(mount, well_id="TRW001", mode="real"):
    data = _mod("data")
    tasks = _mod("tasks")
    return tasks.make_task(data.load_well(data.discover_wells("train")[well_id]), mode)


def _config():
    return _mod("beam_search").BeamSearchConfig(beam_width=12, branch_factor=5, stride=32)


def test_beam_search_outputs_only_manifested_target_free_features(mount):
    beam = _mod("beam_search")
    manifest = _mod("manifest")
    inp = _task(mount).inputs()
    out = beam.beam_search_features(inp, config=_config(), fold_id=2, protocol="unseen_well")

    assert list(out.frame.columns) == list(beam.BEAM_FEATURE_COLUMNS)
    assert out.frame.shape == (inp.n_predict, len(beam.BEAM_FEATURE_COLUMNS))
    assert np.isfinite(out.frame.to_numpy()).all()
    manifest.assert_safe_features(out.frame.columns)
    assert not hasattr(inp, "target")
    assert not np.isfinite(inp.tvt_known[inp.start : inp.stop]).any()


def test_beam_search_reports_required_diagnostics(mount):
    beam = _mod("beam_search")
    out = beam.beam_search_features(_task(mount).inputs(), config=_config())
    required = {
        "confidence_mean", "branch_spread_mean", "path_smoothness",
        "fallback_status", "fallback_fraction", "failure_reason", "cache_hit",
    }
    assert required <= set(out.diagnostics)
    assert 0.0 <= out.diagnostics["confidence_mean"] <= 1.0
    assert out.diagnostics["branch_spread_mean"] >= 0.0
    assert out.diagnostics["path_smoothness"] >= 0.0


def test_beam_search_never_reads_typewell_geology(mount):
    beam = _mod("beam_search")
    inp = _task(mount).inputs()

    class ExplodesOnUse:
        def __array__(self, *args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("Typewell Geology was read")

    guarded = replace(inp, tw_geology=ExplodesOnUse())
    out = beam.beam_search_features(guarded, config=_config())
    assert len(out.frame) == guarded.n_predict


def test_beam_search_falls_back_without_typewell(mount):
    beam = _mod("beam_search")
    inp = replace(_task(mount).inputs(), tw_tvt=None, tw_gr=None)
    out = beam.beam_search_features(inp, config=_config())
    assert out.diagnostics["fallback_status"] is True
    assert out.diagnostics["fallback_fraction"] == 1.0
    assert out.diagnostics["failure_reason"] == "missing_or_invalid_typewell"
    assert out.frame["beam_fallback"].eq(1.0).all()


def test_beam_search_cache_is_scoped_by_well_and_fold(mount, tmp_path):
    cache_mod = _mod("cache")
    beam = _mod("beam_search")
    inp = _task(mount).inputs()
    cache = cache_mod.FeatureCache(tmp_path / "beam-cache")

    first = beam.beam_search_features(
        inp, config=_config(), cache=cache, fold_id=0, protocol="unseen_well"
    )
    second = beam.beam_search_features(
        inp, config=_config(), cache=cache, fold_id=0, protocol="unseen_well"
    )
    other_fold = beam.beam_search_features(
        inp, config=_config(), cache=cache, fold_id=1, protocol="unseen_well"
    )
    assert first.diagnostics["cache_hit"] is False
    assert second.diagnostics["cache_hit"] is True
    assert other_fold.diagnostics["cache_hit"] is False
    assert cache.stats.writes == 2
    np.testing.assert_allclose(first.frame, second.frame)
    np.testing.assert_allclose(first.frame, other_fold.frame)


def test_beam_search_is_device_independent(mount):
    beam = _mod("beam_search")
    inp = _task(mount).inputs()
    cpu = beam.beam_search_features(inp, config=_config(), device="cpu", fold_id=3)
    gpu_request = beam.beam_search_features(inp, config=_config(), device="gpu", fold_id=3)
    np.testing.assert_allclose(cpu.frame, gpu_request.frame)
    assert cpu.diagnostics["execution_device"] == "cpu"
    assert gpu_request.diagnostics["execution_device"] == "cpu"


def test_beam_search_is_an_optional_ridge_feature_not_a_baseline(mount):
    beam = _mod("beam_search")
    baselines = _mod("baselines")
    assert "beam_search" not in baselines.BASELINES

    train = [_task(mount, w) for w in ("TRW006", "TRW007")]
    generator = beam.BeamSearchFeatureGenerator(config=_config(), fold_id=0)
    model = baselines.RidgeBaseline(beam_search=generator).fit(train)
    inp = _task(mount, "TRW008").inputs()
    pred = model.predict(inp)
    assert pred.shape == (inp.n_predict,)
    assert np.isfinite(pred).all()
    assert set(beam.BEAM_FEATURE_COLUMNS) <= set(model.feature_names_)
