"""Leakage, geometry and gate-rule tests for the GeoAnchor experiment.

The experiment exists in the first place because four public notebooks
demonstrated that GR calibration, branch hedging and well-level gating are
*worth controlling*. These tests pin down the two things that must never
slip while those ideas are exercised here:

1. Nothing target-derived reaches a feature, a gate, or a score that was
   produced by a model fitted on the well being scored.
2. Every correction path has a working fallback: killed gate -> anchor,
   failed rule -> anchor, exception -> anchor.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mod(name):
    key = f"src.{name}"
    if key in sys.modules:
        return importlib.reload(sys.modules[key])
    return importlib.import_module(key)


def _tasks(mount, ids, mode):
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


# ---------------------------------------------------------------------------
# In-memory well construction (no mount needed for most tests)
# ---------------------------------------------------------------------------


def _make_task(
    *,
    n=700,
    start=500,
    seed=0,
    gr_fn=None,
    tw_shift=None,
    with_tw=True,
    gr_missing=(),
):
    """Build a deterministic InferenceTask directly (no filesystem).

    ``gr_fn`` maps (row_index, tvt) -> GR and lets a test plant a known offset
    into the typewell's reference space.
    """
    tasks_mod = _mod("tasks")
    data = _mod("data")

    rng = np.random.default_rng(seed)
    tvt_tw = np.arange(-100.0, 100.0, 0.5)
    gr_tw = 70.0 + 20.0 * np.sin(tvt_tw / 7.0) + 8.0 * np.sin(tvt_tw / 2.3 + 1.0)
    if tw_shift is not None:  # two identical marker beds, separated
        gr_tw = gr_tw + 30.0 * np.exp(-0.5 * ((tvt_tw - tw_shift) / 1.5) ** 2)

    md = 9000.0 + np.arange(n, dtype=float)
    tvt = np.cumsum(rng.normal(0, 0.02, n)) + 10.0
    if gr_fn is None:
        gr = np.interp(tvt, tvt_tw, gr_tw) + rng.normal(0, 1.0, n)
    else:
        gr = gr_fn(np.arange(n), tvt)
    for block in gr_missing:
        gr[block] = np.nan
    tvt_input = tvt.copy()
    tvt_input[start:] = np.nan

    hw = pd.DataFrame(
        {
            "MD": md,
            "X": 3000.0 + md * 0.2,
            "Y": 4000.0 + md * 0.05,
            "Z": -2000.0 - md * 0.9,
            "GR": gr,
            "TVT_input": tvt_input,
            "TVT": tvt,
        }
    )
    tw = pd.DataFrame({"TVT": tvt_tw, "GR": gr_tw}) if with_tw else None
    visible = np.zeros(n, dtype=bool)
    visible[:start] = True
    hw["is_visible"] = visible
    hw["is_hidden"] = ~visible
    roles = {"md": "MD", "x": "X", "y": "Y", "z": "Z", "gr": "GR", "tvt_input": "TVT_input", "tvt": "TVT"}
    well = data.WellData(
        well_id="TESTWELL",
        split="train",
        hw=hw,
        tw=tw,
        roles=roles,
        markers={},
        tw_roles={"tvt": "TVT", "gr": "GR"} if with_tw else {},
        region_info={"prediction_start_row": start},
    )
    return tasks_mod.make_task(well, "real")


# ---------------------------------------------------------------------------
# Manifest safety
# ---------------------------------------------------------------------------


def test_arm_and_gate_columns_are_manifest_cleared():
    manifest = _mod("manifest")
    ga = _mod("geoanchor")
    manifest.assert_safe_features(ga.AFFINE_CAL_FEATURE_COLUMNS, context="arm B columns")
    manifest.assert_safe_features(ga.MB_FEATURE_COLUMNS, context="arm C columns")
    manifest.assert_safe_features(ga.GATE_FEATURE_COLUMNS, context="gate columns")


def test_generator_frames_match_declared_columns(mount):
    ga = _mod("geoanchor")
    task = _tasks(mount, ["TRW006"], "real")[0].inputs()
    for gen_cls in (ga.AffineCalibrationFeatureGenerator, ga.MultiBranchFeatureGenerator):
        out = gen_cls().generate(task)
        assert list(out.frame.columns) == list(gen_cls.feature_columns)
        assert len(out.frame) == task.n_predict
        assert np.isfinite(out.frame.to_numpy()).all()


def test_features_ignore_the_tvt_label(mount):
    """Perturbing the TVT label must change nothing the generators emit."""
    data = _mod("data")
    tasks_mod = _mod("tasks")
    ga = _mod("geoanchor")
    hw = data.load_well(data.discover_wells("train")["TRW006"])
    task_a = tasks_mod.make_task(hw, "real").inputs()
    hw2 = hw
    hw2 = data.WellData(
        well_id=hw.well_id, split=hw.split,
        hw=hw.hw.assign(TVT=hw.hw["TVT"] * 100.0 + 7.0),
        tw=hw.tw, roles=hw.roles, markers=hw.markers, tw_roles=hw.tw_roles,
        region_info=hw.region_info,
    )
    task_b = tasks_mod.make_task(hw2, "real").inputs()
    for gen_cls in (ga.AffineCalibrationFeatureGenerator, ga.MultiBranchFeatureGenerator):
        fa = gen_cls().generate(task_a).frame
        fb = gen_cls().generate(task_b).frame
        pd.testing.assert_frame_equal(fa, fb)


# ---------------------------------------------------------------------------
# Idea 1: affine calibration
# ---------------------------------------------------------------------------


def test_affine_calibration_recovers_known_gain_and_offset():
    ga = _mod("geoanchor")
    rng = np.random.default_rng(3)
    tw_tvt = np.arange(-100.0, 100.0, 0.5)
    gr_tw = 70.0 + 20.0 * np.sin(tw_tvt / 7.0)
    n, start = 700, 500
    # The prefix must span enough of the typewell for the fit to be determined:
    # drift ~30 ft across the visible prefix.
    rng2 = np.random.default_rng(4)
    tvt = 5.0 + np.cumsum(rng2.normal(0.06, 0.02, n))
    # Plant an exact affine relationship: G_hw = 2 * G_tw + 15 (+ noise).
    gr = 2.0 * np.interp(tvt, tw_tvt, gr_tw) + 15.0 + rng.normal(0, 0.5, n)
    task = _make_task_with_tw(n=n, start=start, seed=4, gr=gr, tvt=tvt, tw_tvt=tw_tvt, gr_tw=gr_tw)
    cal = ga.fit_prefix_affine_calibration(task.inputs())
    assert cal.ok, cal.failure_reason
    assert cal.alpha == pytest.approx(2.0, abs=0.05)
    assert cal.beta == pytest.approx(15.0, abs=0.5)
    assert cal.fit_rmse_z < 0.5


def _make_task_with_tw(*, n, start, seed, gr, tw_tvt, gr_tw, tvt=None):
    tasks_mod = _mod("tasks")
    data = _mod("data")
    md = 9000.0 + np.arange(n, dtype=float)
    if tvt is None:
        rng = np.random.default_rng(seed)
        tvt = np.cumsum(rng.normal(0, 0.02, n)) + 10.0
    tvt = np.asarray(tvt, dtype="float64")
    tvt_input = tvt.copy()
    tvt_input[start:] = np.nan
    hw = pd.DataFrame(
        {
            "MD": md, "X": 3000 + md * 0.2, "Y": 4000 + md * 0.05,
            "Z": -2000 - md * 0.9, "GR": gr, "TVT_input": tvt_input, "TVT": tvt,
        }
    )
    hw["is_visible"] = np.arange(n) < start
    hw["is_hidden"] = np.arange(n) >= start
    well = data.WellData(
        well_id="TW_WELL", split="train", hw=hw, tw=pd.DataFrame({"TVT": tw_tvt, "GR": gr_tw}),
        roles={"md": "MD", "x": "X", "y": "Y", "z": "Z", "gr": "GR", "tvt_input": "TVT_input", "tvt": "TVT"},
        markers={}, tw_roles={"tvt": "TVT", "gr": "GR"},
        region_info={"prediction_start_row": start},
    )
    return tasks_mod.make_task(well, "real")


def test_affine_calibration_is_prefix_only():
    ga = _mod("geoanchor")
    task = _make_task(seed=5).inputs()
    cal_a = ga.fit_prefix_affine_calibration(task)
    sabotaged = replace(task, gr=np.where(np.arange(task.n_rows) >= task.start, 9999.0, task.gr))
    cal_b = ga.fit_prefix_affine_calibration(sabotaged)
    assert (cal_a.ok, cal_b.ok) == (True, True)
    assert cal_a.alpha == pytest.approx(cal_b.alpha)
    assert cal_a.beta == pytest.approx(cal_b.beta)
    assert cal_a.fit_rmse_z == pytest.approx(cal_b.fit_rmse_z)
    assert cal_a.prefix_corr == pytest.approx(cal_b.prefix_corr)


def test_affine_calibration_degrades_safely():
    ga = _mod("geoanchor")
    # no typewell at all
    task = _make_task(with_tw=False).inputs()
    cal = ga.fit_prefix_affine_calibration(task)
    assert not cal.ok and cal.failure_reason
    # GR entirely missing
    rng = np.random.default_rng(9)
    bad = _make_task(gr_fn=lambda i, t: np.full(len(i), np.nan))
    cal2 = ga.fit_prefix_affine_calibration(bad.inputs())
    assert not cal2.ok
    out = ga.AffineCalibrationFeatureGenerator().generate(bad.inputs())
    assert np.isfinite(out.frame.to_numpy()).all()
    assert float(out.frame["acal_ok"].iloc[0]) == 0.0


# ---------------------------------------------------------------------------
# Ideas 2/3: multi-branch scan
# ---------------------------------------------------------------------------


def test_scan_finds_planted_datum_shift():
    ga = _mod("geoanchor")
    n, start = 900, 500
    tw_tvt = np.arange(-100.0, 100.0, 0.5)
    gr_tw = tw_tvt.copy()  # strictly monotone typewell: GR pins down TVT exactly
    rng = np.random.default_rng(11)
    tvt = 5.0 + np.cumsum(rng.normal(0.06, 0.02, n))
    shift = 5.0
    # Prefix GR matches the typewell at the true TVT (identity); hidden GR is
    # pinned at anchor + shift, a planted constant datum move.
    gr = tvt.copy()
    anchor = tvt[start - 1]
    gr[start:] = anchor + shift
    task = _make_task_with_tw(n=n, start=start, seed=11, gr=gr, tvt=tvt, tw_tvt=tw_tvt, gr_tw=gr_tw).inputs()
    res = ga.multibranch_scan(task)
    assert res.ok, res.failure_reason
    assert res.shift1 == pytest.approx(shift, abs=1.5)
    assert res.confidence > 0.0


def test_scan_reported_truncated_when_gr_outage_too_large():
    ga = _mod("geoanchor")
    task = _make_task(gr_missing=(slice(400, 700),)).inputs()
    res = ga.multibranch_scan(task)
    assert not res.ok
    out = ga.MultiBranchFeatureGenerator().generate(task)
    assert np.isfinite(out.frame.to_numpy()).all()
    assert float(out.frame["mb_ok"].iloc[0]) == 0.0


# ---------------------------------------------------------------------------
# Idea 4: nested prefix pseudo-holdout
# ---------------------------------------------------------------------------


def test_nested_task_is_strictly_inside_the_visible_prefix():
    ga = _mod("geoanchor")
    parent = _make_task(n=900, start=500, seed=1).inputs()
    nested = ga.nested_pseudo_task(parent)
    assert nested is not None
    n_inp = nested.inputs
    assert n_inp.stop == parent.start
    assert n_inp.start < parent.start
    assert n_inp.n_predict == min(parent.n_predict, parent.start - ga.NESTED_MIN_PREFIX)
    assert not np.isfinite(n_inp.tvt_known[n_inp.start :]).any()
    n_inp.assert_no_target()
    # Truth comes from rows the parent treated as visible.
    np.testing.assert_array_equal(nested.truth, parent.tvt_known[n_inp.start : parent.start])


def test_nested_task_returns_none_when_prefix_too_small():
    ga = _mod("geoanchor")
    parent = _make_task(n=400, start=250, seed=1).inputs()
    # budget = 250 - 200 = 50 >= 25 -> works
    assert ga.nested_pseudo_task(parent) is not None
    parent_tight = _make_task(n=300, start=220, seed=1).inputs()
    # budget = 220 - 200 = 20 < 25 -> cannot host a pseudo-holdout
    assert ga.nested_pseudo_task(parent_tight) is None


# ---------------------------------------------------------------------------
# Idea 7: bounded candidate corrections
# ---------------------------------------------------------------------------


class _FakePathGenerator:
    """Deterministic stand-in for the PF/Beam generators."""

    feature_columns = ("fake_shift",)

    def __init__(self, family, shift, conf=0.8, fb_frac=0.0, reason=""):
        self.family = family
        self.shift = np.asarray(shift, dtype="float64")
        self.conf = conf
        self.fb_frac = fb_frac
        self.reason = reason

    def generate(self, task):
        pf = _mod("particle_filter")
        n = task.n_predict
        shift = (
            np.full(n, float(self.shift[0]), dtype="float64")
            if self.shift.size == 1
            else self.shift[:n]
        )
        frame = pd.DataFrame({f"{self.family}_shift": shift})
        diag = {
            "confidence_mean": self.conf,
            "confidence_p10": self.conf,
            "branch_spread_mean": 0.1,
            "fallback_status": self.fb_frac > 0.0,
            "fallback_fraction": self.fb_frac,
            "failure_reason": self.reason,
        }
        return pf.PathFeatureOutput(frame=frame, diagnostics=diag)


def test_candidate_corrections_are_bounded_and_referenced():
    ga = _mod("geoanchor")
    task = _make_task(n=900, start=500, seed=2).inputs()
    base = np.zeros(task.n_predict)
    pf = _FakePathGenerator("pf", np.full(task.n_predict, 100.0))  # way past the cap
    beam = _FakePathGenerator("beam", np.full(task.n_predict, -100.0))
    cands = ga.generate_candidate_corrections(task, base, pf=pf, beam=beam)
    assert cands["pf"].available and cands["beam"].available
    cap = ga.CORRECTION_CAP_FT
    assert np.max(np.abs(cands["pf"].prediction - base)) == pytest.approx(cap)
    assert np.min(cands["beam"].prediction - base) == pytest.approx(-cap)
    mean = cands["pf_beam_mean"].prediction
    assert np.max(np.abs(mean - base)) <= cap + 1e-9


def test_candidate_unavailable_on_hard_failure_or_low_support():
    ga = _mod("geoanchor")
    task = _make_task(n=900, start=500, seed=2).inputs()
    base = np.zeros(task.n_predict)
    hard = _FakePathGenerator("pf", np.zeros(task.n_predict), reason="missing_or_invalid_typewell")
    soft_low = _FakePathGenerator("beam", np.zeros(task.n_predict), fb_frac=0.95)
    cands = ga.generate_candidate_corrections(task, base, pf=hard, beam=soft_low)
    assert not cands["pf"].available
    assert not cands["beam"].available
    assert not cands["pf_beam_mean"].available


# ---------------------------------------------------------------------------
# Idea 5: gate rules
# ---------------------------------------------------------------------------


_BASE_LEVEL = 3.0


def _gate_ready_model(mount, *, shifts, confs=0.8, thr=None, hgb_gain=1.0, killed=False):
    """GatedRidgeAnchor with fake path generators and a constant anchor model."""
    ga = _mod("geoanchor")
    gate_cfg = ga.GateConfig(inner_splits=2, tune_splits=2, seed=0)
    pf = _FakePathGenerator("pf", shifts["pf"], conf=confs)
    beam = _FakePathGenerator("beam", shifts["beam"], conf=confs)
    model = ga.GatedRidgeAnchor(pf=pf, beam=beam, gate_config=gate_cfg, protocol="unseen_well", fold=0)

    class _ConstAnchor:
        def predict(self, task, feats=None):
            return np.full(task.n_predict, _BASE_LEVEL)

    model.anchor_model = _ConstAnchor()
    gate = ga.WellLevelGate(pf=pf, beam=beam, config=gate_cfg, protocol="unseen_well", fold=0)
    gate.killed = bool(killed)
    gate.kill_reason = "test_killed" if killed else ""
    gate.thresholds = thr or ga.GateThresholds(margin=0.0, conf_thr=0.0, sep_cap=np.inf)
    gate.model = True  # sentinel: predict_improvements is patched below
    model.gate = gate

    def _predict_improvements(task, **kw):
        return {"pf_beam_mean": hgb_gain + 0.3, "pf": hgb_gain, "beam": hgb_gain - 0.1}

    gate.predict_improvements = _predict_improvements
    return model


def _rule_task():
    """Constant-zero well: anchor=0 and pseudo truth=0.

    The fake anchor model predicts a constant ``_BASE_LEVEL`` (=3), so its
    pseudo RMSE is 3^2 = 9 SE and its pseudo tail risk is 9. A zero-shift
    candidate hits the truth exactly; its pseudo SE/tail are ~0. This gives a
    clean, deterministic way to flip each gate rule independently.
    """
    n, start = 700, 600
    tw_tvt = np.arange(-100.0, 100.0, 0.5)
    gr_tw = 70.0 + 20.0 * np.sin(tw_tvt / 7.0)
    tvt = np.zeros(n)
    tvt_input = tvt.copy()
    tvt_input[start:] = np.nan
    gr = np.interp(tvt, tw_tvt, gr_tw)
    tasks_mod = _mod("tasks")
    data = _mod("data")
    md = 9000.0 + np.arange(n, dtype=float)
    hw = pd.DataFrame(
        {
            "MD": md, "X": 3000 + md * 0.2, "Y": 4000 + md * 0.05,
            "Z": -2000 - md * 0.9, "GR": gr, "TVT_input": tvt_input, "TVT": tvt,
        }
    )
    hw["is_visible"] = np.arange(n) < start
    hw["is_hidden"] = np.arange(n) >= start
    well = data.WellData(
        well_id="RULEWELL", split="train", hw=hw,
        tw=pd.DataFrame({"TVT": tw_tvt, "GR": gr_tw}),
        roles={"md": "MD", "x": "X", "y": "Y", "z": "Z", "gr": "GR", "tvt_input": "TVT_input", "tvt": "TVT"},
        markers={}, tw_roles={"tvt": "TVT", "gr": "GR"},
        region_info={"prediction_start_row": start},
    )
    return tasks_mod.make_task(well, "real").inputs()


def test_gate_applies_when_all_rules_pass(mount):
    task = _rule_task()
    n = task.n_predict
    shifts = {"pf": np.zeros(n), "beam": np.zeros(n)}
    model = _gate_ready_model(mount, shifts=shifts)
    pred = model.predict(task)
    assert np.allclose(pred, task.anchor_tvt)  # zero shift applied at the outer anchor
    assert not np.allclose(pred, _BASE_LEVEL)  # and not the anchor prediction
    log = model.gate_log[-1]
    assert log["outcome"] == "applied_pf_beam_mean"
    assert log["reason"] == "all_rules_passed"


def test_gate_falls_back_when_pseudo_holdout_not_improved(mount):
    task = _rule_task()
    n = task.n_predict
    shifts = {"pf": np.full(n, _BASE_LEVEL), "beam": np.full(n, _BASE_LEVEL)}
    model = _gate_ready_model(mount, shifts=shifts)
    pred = model.predict(task)
    assert np.allclose(pred, _BASE_LEVEL)  # the anchor prediction, untouched
    assert model.gate_log[-1]["outcome"] == "fallback"
    assert "pseudo_holdout_not_improved" in model.gate_log[-1]["reason"]


def test_gate_falls_back_when_confidence_below_threshold(mount):
    ga = _mod("geoanchor")
    task = _rule_task()
    n = task.n_predict
    shifts = {"pf": np.zeros(n), "beam": np.zeros(n)}
    thr = ga.GateThresholds(margin=0.0, conf_thr=0.95, sep_cap=np.inf)
    model = _gate_ready_model(mount, shifts=shifts, thr=thr, confs=0.5)
    model.predict(task)
    assert model.gate_log[-1]["outcome"] == "fallback"
    assert "alignment_confidence_below_threshold" in model.gate_log[-1]["reason"]


def test_gate_falls_back_when_branch_disagreement_unacceptable(mount):
    ga = _mod("geoanchor")
    task = _rule_task()
    n = task.n_predict
    shifts = {"pf": np.zeros(n), "beam": np.full(n, 10.0)}
    thr = ga.GateThresholds(margin=0.0, conf_thr=0.0, sep_cap=1.0)
    model = _gate_ready_model(mount, shifts=shifts, thr=thr)
    model.predict(task)
    assert model.gate_log[-1]["outcome"] == "fallback"
    assert "branch_disagreement_unacceptable" in model.gate_log[-1]["reason"]


def test_gate_falls_back_when_tail_risk_increases(mount):
    task = _rule_task()
    n = task.n_predict
    shift = np.full(n, 2.9)
    shift[int(0.9 * n):] = -3.1  # marginal mean win (SE 8.53 < 9), worse tail
    shifts = {"pf": shift, "beam": shift}
    model = _gate_ready_model(mount, shifts=shifts)
    model.predict(task)
    assert model.gate_log[-1]["outcome"] == "fallback"
    assert "worst_tail_risk_increased" in model.gate_log[-1]["reason"]


def test_gate_falls_back_when_killed(mount):
    task = _rule_task()
    n = task.n_predict
    shifts = {"pf": np.zeros(n), "beam": np.zeros(n)}
    model = _gate_ready_model(mount, shifts=shifts, killed=True)
    pred = model.predict(task)
    assert np.allclose(pred, _BASE_LEVEL)
    assert model.gate_log[-1]["outcome"] == "fallback"
    assert model.gate_log[-1]["gate_killed"]


def test_gate_never_accepts_validation_wells_in_fit(mount):
    ga = _mod("geoanchor")
    tasks = _tasks(mount, ["TRW006", "TRW007", "TRW008", "TRW009"], "real")
    pf = _FakePathGenerator("pf", np.zeros(10))
    beam = _FakePathGenerator("beam", np.zeros(10))
    gate = ga.WellLevelGate(pf=pf, beam=beam, config=ga.GateConfig(inner_splits=2, tune_splits=2))
    with pytest.raises(ga.CrossFitLeakage):
        gate.fit(tasks, validation_ids={"TRW006"})


def _protocol_driver_smoke(mount):
    ga = _mod("geoanchor")
    data = _mod("data")
    tasks_mod = _mod("tasks")
    v = _mod("validation")
    files = data.discover_wells("train")

    def task_builder(ids, mode):
        tasks, skipped = [], []
        for wid in ids:
            try:
                tasks.append(tasks_mod.make_task(data.load_well(files[wid]), mode))
            except tasks_mod.TaskConstructionError as exc:
                skipped.append((wid, str(exc)))
        return tasks, skipped

    return ga, v, files, task_builder


def test_driver_runs_both_protocols_with_all_arms_paired(mount):
    ga, v, files, task_builder = _protocol_driver_smoke(mount)
    ids = sorted(files)
    folds = v.make_group_folds(ids, n_splits=2, seed=0)
    memo = {}
    cfg = ga.GateConfig(inner_splits=2, tune_splits=2, seed=0)
    results = {}
    for protocol, mode in ((v.PROTOCOL_A, "masked"), (v.PROTOCOL_B, "real")):
        run = ga.run_geoanchor_protocol(
            protocol=protocol, mode=mode, folds=folds, task_builder=task_builder,
            memo=memo, seed=0, gate_config=cfg,
        )
        results[protocol] = run
        # Task-stage skips are expected (short wells cannot host a masked
        # suffix); fit/predict failures are not.
        hard = [f for f in run.failures if f.get("stage") in {"fit", "predict"}]
        assert not hard, hard[:3]
        df = pd.DataFrame([vars(r) for r in run.well_results])
        assert set(df["model"]) == set(ga.ARM_ORDER)
        counts = df.groupby("model")["well_id"].nunique()
        assert counts.nunique() == 1  # every arm scored the same wells (paired)
    assert results[v.PROTOCOL_A].gate_logs or results[v.PROTOCOL_B].gate_logs
    assert results[v.PROTOCOL_B].gate_fit_infos


def test_gate_trains_only_on_fold_training_wells(mount):
    """Spy on the well IDs the gate is fitted on; none may be validation wells."""
    ga, v, files, task_builder = _protocol_driver_smoke(mount)
    seen: list[set] = []
    original = ga.WellLevelGate.fit

    def spy(self, train_tasks, **kw):
        seen.append({t.well_id for t in train_tasks})
        return original(self, train_tasks, **kw)

    ga.WellLevelGate.fit = spy
    try:
        ids = sorted(files)
        folds = v.make_group_folds(ids, n_splits=2, seed=0)
        run = ga.run_geoanchor_protocol(
            protocol=v.PROTOCOL_B, mode="real", folds=folds, task_builder=task_builder,
            memo={}, seed=0, gate_config=ga.GateConfig(inner_splits=2, tune_splits=2),
        )
    finally:
        ga.WellLevelGate.fit = original
    assert seen, "gate.fit was never called"
    for fold, train_seen in zip(folds, seen):
        assert train_seen <= set(fold.train_ids)
        assert not (train_seen & set(fold.valid_ids))


def test_killed_gate_matches_ridge_default_exactly(mount):
    """With too few OOF examples the gate must die and arm E must equal arm A."""
    ga, v, files, task_builder = _protocol_driver_smoke(mount)
    ids = sorted(files)
    folds = v.make_group_folds(ids, n_splits=2, seed=0)
    run = ga.run_geoanchor_protocol(
        protocol=v.PROTOCOL_B, mode="real", folds=folds, task_builder=task_builder,
        memo={}, seed=0, gate_config=ga.GateConfig(inner_splits=2, tune_splits=2),
    )
    df = pd.DataFrame([vars(r) for r in run.well_results])
    infos = pd.DataFrame(run.gate_fit_infos)
    if infos["killed"].all():
        a = df[df["model"] == ga.ARM_A].set_index("well_id")["rmse"].sort_index()
        e = df[df["model"] == ga.ARM_E].set_index("well_id")["rmse"].sort_index()
        np.testing.assert_allclose(a.to_numpy(), e.to_numpy(), rtol=1e-9)


def test_geoanchor_determinism(mount):
    ga, v, files, task_builder = _protocol_driver_smoke(mount)
    ids = sorted(files)
    folds = v.make_group_folds(ids, n_splits=2, seed=7)
    cfg = ga.GateConfig(inner_splits=2, tune_splits=2, seed=7)
    runs = []
    for _ in range(2):
        run = ga.run_geoanchor_protocol(
            protocol=v.PROTOCOL_B, mode="real", folds=folds, task_builder=task_builder,
            memo={}, seed=7, gate_config=cfg,
        )
        df = pd.DataFrame([vars(r) for r in run.well_results])
        runs.append(df.sort_values(["model", "well_id"])["rmse"].to_numpy())
    np.testing.assert_allclose(runs[0], runs[1], rtol=0, atol=0)
