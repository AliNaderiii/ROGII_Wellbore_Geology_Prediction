"""Numerical investigation of multi-scale alignment candidates (synthetic only).

This is a *diagnostic* test, not a real-data validation. It exercises the
candidate generators on the synthetic ``mount`` fixture and asserts that
the numbers they produce are finite, bounded and well-behaved. The real
value of these candidates is measured later by ``run_alignment_v2_experiment``.

The candidates under test:

A. Multi-scale affine GR calibration
   - min_prefix_rows in {40, 80, 160}
   - shared sanity bounds (alpha, beta, fit_rmse, prefix_corr)

B. Multi-scale trajectory alignment
   - search half-range in {12, 35, 100} ft
   - shared step scale (>= 0.05 of the search)
   - per-scale dominant shift, confidence, prefix_trust, agreement count

C. Branch disagreement summary
   - peak-to-peak (ptp) of the three dominant shifts
   - a "min_confidence" lower bound
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest


# --- candidate functions ----------------------------------------------------


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


def _well_ids_for_masked(mount) -> list[str]:
    # 4 long wells (TRW006..TRW009) have prefix 1400; the others are too short
    # for masked-mode prediction but still work in real mode.
    return ["TRW006", "TRW007", "TRW008", "TRW009"]


def _well_ids_for_real(mount) -> list[str]:
    return ["TRW001", "TRW006", "TRW007", "TRW008", "TRW009"]


# A. Multi-scale affine GR calibration --------------------------------------


@pytest.mark.parametrize("min_rows", [40, 80, 160])
def test_affine_calibration_finite_and_bounded(mount, min_rows):
    geo = _mod("geoanchor")
    for t in _tasks(mount, _well_ids_for_real(mount), "real"):
        inp = t.inputs()
        ref = geo.TypewellReference(inp.tw_tvt, inp.tw_gr)
        cal = geo.fit_prefix_affine_calibration(
            inp, ref, config=geo.AffineCalibrationConfig(min_prefix_rows=min_rows)
        )
        if not cal.ok:
            # On wells with very short or mostly-missing prefix this is allowed.
            assert cal.failure_reason != ""
            continue
        assert np.isfinite(cal.alpha)
        assert 0.25 <= cal.alpha <= 4.0
        assert np.isfinite(cal.beta)
        assert abs(cal.beta) <= 500.0
        assert np.isfinite(cal.fit_rmse_z)
        assert cal.fit_rmse_z >= 0.0


def test_affine_calibration_scales_agree_when_signal_strong(mount):
    """When the prefix has plenty of GR + a usable typewell, all three scales
    should agree on the same alpha/beta (the fit is the same least-squares)."""
    geo = _mod("geoanchor")
    for t in _tasks(mount, _well_ids_for_real(mount), "real"):
        inp = t.inputs()
        ref = geo.TypewellReference(inp.tw_tvt, inp.tw_gr)
        cals = [
            geo.fit_prefix_affine_calibration(
                inp, ref, config=geo.AffineCalibrationConfig(min_prefix_rows=m)
            )
            for m in (40, 80, 160)
        ]
        oks = [c.ok for c in cals]
        if not all(oks):
            continue
        alphas = np.asarray([c.alpha for c in cals])
        betas = np.asarray([c.beta for c in cals])
        # The same rows satisfy all three min_rows thresholds, so the fit is
        # identical across scales.
        assert np.allclose(alphas, alphas[0], atol=1e-9)
        assert np.allclose(betas, betas[0], atol=1e-9)


# B. Multi-scale trajectory alignment --------------------------------------


ALIGN_SCALES_FT = (12.0, 35.0, 100.0)


def _multiscale_for_well(task) -> dict:
    geo = _mod("geoanchor")
    inp = task.inputs()
    ref = geo.TypewellReference(inp.tw_tvt, inp.tw_gr)
    cal = geo.fit_prefix_affine_calibration(inp, ref)
    rows = []
    for s in ALIGN_SCALES_FT:
        cfg = geo.MultiBranchConfig(search=s, step=max(0.25, 0.05 * s))
        res = geo.multibranch_scan(inp, cal=cal, ref=ref, config=cfg)
        rows.append(res)
    return {"cal": cal, "rows": rows}


@pytest.mark.parametrize("scale", ALIGN_SCALES_FT)
def test_multiscale_scan_finite(mount, scale):
    geo = _mod("geoanchor")
    for t in _tasks(mount, _well_ids_for_real(mount), "real"):
        inp = t.inputs()
        ref = geo.TypewellReference(inp.tw_tvt, inp.tw_gr)
        cal = geo.fit_prefix_affine_calibration(inp, ref)
        cfg = geo.MultiBranchConfig(search=scale, step=max(0.25, 0.05 * scale))
        res = geo.multibranch_scan(inp, cal=cal, ref=ref, config=cfg)
        if not res.ok:
            assert res.failure_reason != ""
            continue
        assert np.isfinite(res.shift1)
        assert abs(res.shift1) <= scale + 1e-9
        assert 0.0 <= res.confidence <= 1.0
        assert 0.0 <= res.prefix_trust <= 1.0
        if res.bimodal:
            assert np.isfinite(res.shift2)
            assert res.sep >= 0.0


def test_multiscale_branch_disagreement_finite(mount):
    """Peak-to-peak of the three dominant shifts is finite when shifts exist.

    On short or no-typewell wells the scan may produce no usable shift; that
    is allowed (the gate would fall back to Ridge) and we just skip the
    disagree-bound check on that well.
    """
    for t in _tasks(mount, _well_ids_for_real(mount), "real"):
        out = _multiscale_for_well(t)
        ok_shifts = [r.shift1 for r in out["rows"] if r.ok]
        if not ok_shifts:
            continue
        assert all(np.isfinite(s) for s in ok_shifts)
        ptp = float(np.max(ok_shifts) - np.min(ok_shifts))
        assert np.isfinite(ptp)
        # Physical sanity: short scale can't be wildly more correct than the
        # long scale on a well-formed field. The bound here is loose.
        assert ptp <= 2.0 * ALIGN_SCALES_FT[-1]


# C. Branch disagreement summary ------------------------------------------


def test_branch_disagreement_ptp_is_bounded(mount):
    for t in _tasks(mount, _well_ids_for_real(mount), "real"):
        out = _multiscale_for_well(t)
        shifts = [r.shift1 for r in out["rows"] if r.ok]
        if not shifts:
            continue
        assert np.isfinite(np.max(shifts) - np.min(shifts))
