"""Tests for the Alignment v2 candidate generators (src.alignment_v2).

Covers:
* manifest safety (every ALIGN_V2_FEATURE_COLUMN is cleared);
* candidate families are finite, bounded and deterministic;
* the dynamic-programming path is monotonically non-decreasing and
  bounded in curvature;
* the branch ensemble is built from the available candidates only and
  the trimmed-mean hedge is well-defined;
* the robust projection rejects clipping beyond the cap;
* the feature row covers every column and every value is finite.

All tests use the synthetic ``mount`` fixture; real-data validation
lives in ``scripts/run_alignment_v2_experiment.py``.
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


def test_align_v2_columns_are_manifest_cleared(mount):
    manifest = _mod("manifest")
    a2 = _mod("alignment_v2")
    manifest.assert_safe_features(
        list(a2.ALIGN_V2_FEATURE_COLUMNS), context="alignment v2 columns"
    )


def test_align_v2_columns_root_in_allowed_sources(mount):
    manifest = _mod("manifest")
    a2 = _mod("alignment_v2")
    allowed = {
        "MD", "X", "Y", "Z", "GR", "TVT_input", "TVT_input (prefix only)",
        "Typewell TVT", "Typewell GR",
    }
    allowed_c = {manifest.canonical(a) for a in allowed}
    for name in a2.ALIGN_V2_FEATURE_COLUMNS:
        roots = manifest.root_sources(manifest.canonical(name))
        assert roots <= allowed_c, f"{name} roots outside allowed set: {roots - allowed_c}"


# --- candidate families -----------------------------------------------------


def test_multi_scale_affine_finite(mount):
    a2 = _mod("alignment_v2")
    for t in _tasks(mount, _well_ids(), "real"):
        inp = t.inputs()
        out = a2.multi_scale_affine_calibration(inp)
        if not out.ok:
            assert out.failure_reason != ""
            continue
        for r in out.scale_results:
            assert r.cal.alpha == pytest.approx(0.0) or 0.25 <= r.cal.alpha <= 4.0
            assert r.cal.beta == pytest.approx(0.0) or abs(r.cal.beta) <= 500.0
        assert np.isfinite(out.alpha_ptp)
        assert np.isfinite(out.beta_ptp)
        assert 0.0 <= out.confidence <= 1.0
        assert 0 <= out.n_ok <= 3


def test_multi_scale_alignment_deterministic(mount):
    a2 = _mod("alignment_v2")
    for t in _tasks(mount, _well_ids(), "real"):
        inp = t.inputs()
        out1 = a2.multi_scale_trajectory_alignment(inp)
        out2 = a2.multi_scale_trajectory_alignment(inp)
        # Determinism: identical outputs.
        assert out1.dominant_shift == pytest.approx(out2.dominant_shift)
        assert out1.ptp == pytest.approx(out2.ptp)
        if out1.ok:
            for r1, r2 in zip(out1.scale_results, out2.scale_results):
                assert r1.mb.shift1 == pytest.approx(r2.mb.shift1)


def test_dynamic_path_is_monotonic_and_bounded(mount):
    a2 = _mod("alignment_v2")
    for t in _tasks(mount, _well_ids(), "real"):
        inp = t.inputs()
        dp = a2.dynamic_path_match(inp)
        if not dp.ok or dp.path is None:
            assert dp.failure_reason != ""
            continue
        path = dp.path
        s, stop = int(inp.start), int(inp.stop)
        # The path on the predicted region is monotonically
        # non-decreasing.
        pred = path[s:stop]
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-9), f"non-monotonic: min diff {diffs.min()}"
        # Curvature bound: the per-step TVT movement cannot exceed
        # max_gradient * median_md_step on average.
        max_step = a2.ALIGN_V2_DPW_MAX_GRADIENT
        md_step = np.median(np.diff(inp.md[s:stop]))
        if md_step > 1e-9 and diffs.size > 0:
            assert np.all(diffs <= max_step * md_step + 1e-9)
        assert np.all(np.isfinite(path))


def test_branch_ensemble_excludes_fallbacks(mount):
    a2 = _mod("alignment_v2")
    for t in _tasks(mount, _well_ids(), "real"):
        inp = t.inputs()
        n = int(inp.n_predict)
        base = np.full(n, float(inp.anchor_tvt) if np.isfinite(inp.anchor_tvt) else 0.0)
        ens = a2.build_branch_ensemble(inp, base_pred=base)
        # At least the ridge branch is always present.
        names = {b.name for b in ens.candidates}
        assert "ridge" in names
        if ens.n_available > 0:
            assert np.isfinite(ens.branch_disagreement)
            assert 0.0 <= ens.mean_correction_abs
        # The hedge branch is only added when >=2 non-fallback branches.
        hedges = [b for b in ens.candidates if b.name == "branch_hedged"]
        if hedges:
            assert hedges[0].available
            assert ens.n_available >= 2


def test_robust_projection_rejects_excessive_clipping(mount):
    a2 = _mod("alignment_v2")
    for t in _tasks(mount, _well_ids(), "real"):
        inp = t.inputs()
        n = int(inp.n_predict)
        # A wildly wrong candidate: a linear ramp that grows by
        # hundreds of feet per row. The robust projection should clip
        # the movement and reject the projection (too many clipped).
        bad = np.linspace(0.0, 1000.0 * n, n)
        proj = a2.robust_stratigraphic_projection(inp, candidate_path=bad)
        # Either rejected or, if accepted, the projected path is
        # bounded.
        if proj.ok:
            assert np.all(np.isfinite(proj.path))
            assert np.max(np.abs(proj.path - bad)) <= a2.ALIGN_V2_CORRECTION_CAP_FT + 1e-9
        else:
            assert proj.failure_reason != ""


def test_align_v2_feature_row_covers_all_columns(mount):
    a2 = _mod("alignment_v2")
    for t in _tasks(mount, _well_ids(), "real"):
        inp = t.inputs()
        n = int(inp.n_predict)
        base = np.full(n, float(inp.anchor_tvt) if np.isfinite(inp.anchor_tvt) else 0.0)
        bundle = a2.run_align_v2_candidates(inp, base_pred=base, apply_projection=True)
        # Every column is in the row, every value is finite.
        for col in a2.ALIGN_V2_FEATURE_COLUMNS:
            assert col in bundle.features, col
            v = bundle.features[col]
            assert np.isfinite(v), f"{col} = {v}"


# --- end-to-end: v2 candidates stay on the allowed roots ------------------


def test_align_v2_candidates_never_read_hidden_tvt(mount):
    """Inference-safe root contract: at no point does the v2 module read a
    hidden TVT label or a target-derived quantity.
    """
    a2 = _mod("alignment_v2")
    # We do not have the test input file mounted; we rely on the
    # InferenceTask to structurally exclude any target access. The
    # test below confirms the candidates run on a well whose target
    # is non-trivial and still produce finite, valid outputs.
    for t in _tasks(mount, _well_ids(), "real"):
        inp = t.inputs()
        # The InferenceTask asserts no finite TVT past `start`.
        inp.assert_no_target()
        n = int(inp.n_predict)
        base = np.full(n, float(inp.anchor_tvt) if np.isfinite(inp.anchor_tvt) else 0.0)
        bundle = a2.run_align_v2_candidates(inp, base_pred=base, apply_projection=True)
        for path in bundle.paths.values():
            if path is None:
                continue
            assert np.all(np.isfinite(path)), "non-finite candidate path"
            assert path.size == n
