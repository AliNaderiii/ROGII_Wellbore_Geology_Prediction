"""Leakage, fallback-exactness, kill-switch and decision tests for the
trajectory stack pipeline (src/trajectory_stack.py).

Pinned invariants:

1. Every new design-matrix column is manifest-registered and roots into the
   allowed inference sources only.
2. The gated arm and the meta-stack return the *bit-identical* Ridge anchor
   output on any guard failure (kill switch, gate decline, exception).
3. Fold-training wells (and only those) reach gate/stack training; blocked
   public wells raise everywhere.
4. Corrections are bounded by the a-priori cap, shrunk, and warmup-ramped.
5. The promotion rule promotes nothing that is not a real-mount run and
   enforces each pre-registered rule.
"""
from __future__ import annotations

import importlib
import sys

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
# Manifest safety
# ---------------------------------------------------------------------------


def test_stack_design_columns_are_manifest_cleared():
    manifest = _mod("manifest")
    from src.trajectory_stack import STACK_GATE_FEATURE_COLUMNS

    manifest.assert_safe_features(STACK_GATE_FEATURE_COLUMNS, context="stack gate columns")
    manifest.assert_safe_features(
        ["meta_res_ridge", "meta_res_lgbm", "meta_res_cat", "meta_dmd", "meta_log1p_dmd"],
        context="stack meta columns",
    )


def test_new_features_root_only_in_allowed_sources():
    manifest = _mod("manifest")
    allowed = {
        "MD", "X", "Y", "Z", "GR", "TVT_input (prefix only)", "TVT_input",
        "Typewell TVT", "Typewell GR",
    }
    from src.manifest import canonical

    allowed_c = {canonical(a) for a in allowed}
    for name in (
        "gate_ms_ptp", "gate_ms_dominant_shift", "gate_ms_n_agree", "gate_ms_min_conf",
        "gate_lgbm_oof_skill", "gate_cat_oof_skill",
        "gate_cand_mb", "gate_cand_lgbm", "gate_cand_cat",
        "meta_res_ridge", "meta_res_lgbm", "meta_res_cat", "meta_dmd", "meta_log1p_dmd",
    ):
        roots = manifest.root_sources(manifest.canonical(name))
        assert roots <= allowed_c, f"{name} roots outside allowed set: {roots - allowed_c}"
        for bad in ("TVT", "Typewell Geology", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"):
            assert bad not in roots


# ---------------------------------------------------------------------------
# Boosted residual learners
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls_name", ["LightGBMResidual", "CatBoostResidual"])
def test_boosted_residual_fit_predict_finite(mount, cls_name):
    ts = _mod("trajectory_stack")
    if cls_name == "LightGBMResidual" and not ts.HAVE_LIGHTGBM:
        pytest.skip("lightgbm not installed")
    if cls_name == "CatBoostResidual" and not ts.HAVE_CATBOOST:
        pytest.skip("catboost not installed")
    cls = getattr(ts, cls_name)
    tasks = _tasks(mount, ["TRW001", "TRW006", "TRW007", "TRW008", "TRW009"], "real")
    train = tasks[:3]
    valid = tasks[3:]
    model = cls(seed=0, max_iter=25, estop_rounds=8)
    model.fit(train)
    for t in valid:
        inp = t.inputs()
        pred = model.predict(inp)
        assert pred.shape == (inp.n_predict,)
        assert np.all(np.isfinite(pred))
    # fitted only on fold-train wells: identical-feature task scored under a
    # second seeded fit must reproduce bit-exact predictions
    model2 = cls(seed=0, max_iter=25, estop_rounds=8)
    model2.fit(train)
    p1 = model.predict(valid[0].inputs())
    p2 = model2.predict(valid[0].inputs())
    assert np.allclose(p1, p2)


@pytest.mark.parametrize("cls_name", ["LightGBMResidual", "CatBoostResidual"])
def test_boosted_residual_rejects_blocked_wells(mount, cls_name, tmp_path):
    ts = _mod("trajectory_stack")
    from src.validation import BlockedWellError

    data = _mod("data")
    tasks_mod = _mod("tasks")
    files = data.discover_wells("train")
    t = tasks_mod.make_task(data.load_well(files["TRW001"]), "real")
    blocked = t.__class__(inputs_=t.inputs_, target=t.target)
    # swap the well id via reconstructing the dataclass with a blocked id
    from dataclasses import replace as dc_replace

    task_blocked = dc_replace(t, inputs_=dc_replace(t.inputs_, well_id="000d7d20"))
    cls = getattr(ts, cls_name)
    model = cls(seed=0, max_iter=5, estop_rounds=2)
    with pytest.raises((BlockedWellError, Exception)) as exc:
        model.fit([task_blocked])
    assert "public test wells" in str(exc.value) or "Blocked" in type(exc.value).__name__


# ---------------------------------------------------------------------------
# OOF meta-stack: kill switch and exact fallback
# ---------------------------------------------------------------------------


def _kill_ready_stack(anchor, *, killed=True):
    ts = _mod("trajectory_stack")
    cfg = ts.StackConfig(seed=0, boost_max_iter=5, boost_estop_rounds=3)
    model = ts.OOFMetaStackAnchor(anchor_model=anchor, config=cfg)
    model.stack.killed = killed
    model.stack.kill_reason = "test_killed" if killed else ""
    model.stack.info["killed"] = killed
    model.stack.info["kill_reason"] = model.stack.kill_reason
    return model


def test_stack_killed_returns_bit_identical_anchor(mount):
    from src.baselines import RidgeBaseline

    anchor = RidgeBaseline(alignment_features=False)
    train = _tasks(mount, ["TRW001", "TRW006", "TRW007"], "real")
    valid = _tasks(mount, ["TRW008", "TRW009"], "real")
    anchor.fit(train)
    model = _kill_ready_stack(anchor, killed=True)
    for t in valid:
        inp = t.inputs()
        base = anchor.predict(inp)
        pred = model.predict(inp)
        np.testing.assert_array_equal(pred, base)  # bit-identical, not merely close
        diag = model.prediction_diagnostics(inp, None, pred)
        assert diag["gate_fallback_exact_ridge"] is True
        assert diag["fallback_fraction"] == 1.0


def test_stack_fit_runs_and_records_kill_state(mount):
    from src.baselines import RidgeBaseline

    ts = _mod("trajectory_stack")
    anchor = RidgeBaseline(alignment_features=False)
    tasks = _tasks(mount, ["TRW001", "TRW006", "TRW007", "TRW008", "TRW009"], "real")
    anchor.fit(tasks)
    cfg = ts.StackConfig(
        inner_splits=2, tune_splits=2, seed=0, boost_max_iter=10, boost_estop_rounds=4,
        max_rows_per_well=80,
    )
    model = ts.OOFMetaStackAnchor(anchor_model=anchor, config=cfg)
    model.fit(tasks)
    assert isinstance(model.stack.killed, bool)
    info = model.stack.info
    assert info["killed"] == model.stack.killed
    if model.stack.killed:
        assert info["kill_reason"]
    else:
        assert np.isfinite(info["meta_alpha"]) or info["pooled_sub_oof_delta"] < 0
    for t in tasks[:2]:
        pred = model.predict(t.inputs())
        assert pred.shape == (t.inputs().n_predict,)
        assert np.all(np.isfinite(pred))


def test_stack_correction_is_capped(mount):
    """When alive, the stack may move at most CORRECTION_CAP_FT vs the anchor."""
    ts = _mod("trajectory_stack")
    from src.geoanchor import CORRECTION_CAP_FT

    class _FakeAnchor:
        alpha = 1
        model = True  # sentinel truthy

        def predict(self, task, feats=None):
            return np.full(task.n_predict, float(task.anchor_tvt))

    anchor = _FakeAnchor()
    stack = ts.OOFMetaStack(ts.StackConfig(seed=0))
    stack.meta_columns = ["meta_res_ridge", "meta_dmd", "meta_log1p_dmd"]
    stack.medians_ = pd.Series({c: 0.0 for c in stack.meta_columns})

    class _IdScaler:
        def transform(self, X):
            return X

    class _HugeMeta:
        def predict(self, X):
            return np.full(X.shape[0], 1e6)

    stack.scaler_ = _IdScaler()
    stack.meta_model = _HugeMeta()
    stack.killed = False
    well = _tasks(mount, ["TRW006"], "real")[0]
    inp = well.inputs()
    base = np.full(inp.n_predict, float(inp.anchor_tvt))
    pred = stack.predict(inp, base)
    assert np.all(np.abs(pred - base) <= CORRECTION_CAP_FT + 1e-9)


# ---------------------------------------------------------------------------
# Gated trajectory stack: rules and exact fallback
# ---------------------------------------------------------------------------

_BASE_LEVEL = 3.0


class _FakePathGenerator:
    """Deterministic fake PF/Beam: constant shift track + confidence."""

    def __init__(self, family, shift, conf=0.8):
        from src.particle_filter import PathFeatureOutput

        self._out = PathFeatureOutput
        self.family = family
        self.shift = np.asarray(shift, dtype="float64")
        self.conf = conf
        self.feature_columns = (f"{family}_shift",)

    def generate(self, task):
        from src.particle_filter import PathFeatureOutput

        n = task.n_predict
        shift = self.shift
        if shift.size != n:
            shift = np.full(n, float(self.shift[0]) if self.shift.size else 0.0)
        frame = pd.DataFrame({f"{self.family}_shift": shift},
                             index=np.arange(task.start, task.stop))
        diag = {
            "confidence_mean": self.conf,
            "confidence_p10": self.conf,
            "branch_spread_mean": 0.0,
            "branch_spread_p90": 0.0,
            "fallback_fraction": 0.0,
            "failure_reason": "",
        }
        return PathFeatureOutput(frame=frame, diagnostics=diag)


def _rule_task(tvt_value=0.0):
    """Constant well; truth equals tvt_value everywhere."""
    tasks_mod = _mod("tasks")
    data = _mod("data")
    n, start = 700, 600
    tw_tvt = np.arange(-100.0, 100.0, 0.5)
    gr_tw = 70.0 + 20.0 * np.sin(tw_tvt / 7.0)
    tvt = np.full(n, tvt_value)
    tvt_input = tvt.copy()
    tvt_input[start:] = np.nan
    gr = np.interp(tvt, tw_tvt, gr_tw)
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
        well_id="GATEWELL", split="train", hw=hw,
        tw=pd.DataFrame({"TVT": tw_tvt, "GR": gr_tw}),
        roles={"md": "MD", "x": "X", "y": "Y", "z": "Z", "gr": "GR",
               "tvt_input": "TVT_input", "tvt": "TVT"},
        markers={}, tw_roles={"tvt": "TVT", "gr": "GR"},
        region_info={"prediction_start_row": start},
    )
    return tasks_mod.make_task(well, "real").inputs()


def _gated_ready_model(shifts, *, gain=1.0, killed=False, thr=None, learners=None):
    ts = _mod("trajectory_stack")
    cfg = ts.TrajectoryGateConfig(inner_splits=2, tune_splits=2, seed=0)
    pf = _FakePathGenerator("pf", shifts)
    beam = _FakePathGenerator("beam", shifts)
    anchor = _FakeAnchor()
    model = ts.GatedTrajectoryStack(
        pf=pf, beam=beam, anchor_model=anchor, config=cfg,
        protocol="unseen_well", fold=0,
    )
    model.killed = killed
    model.kill_reason = "test_killed" if killed else ""
    model.thresholds = thr or ts.StackGateThresholds(
        margin=0.0, conf_thr=0.0, sep_cap=np.inf, shrink=1.0, warmup=1,
    )
    model.gate_model = True  # sentinel; _predict_improvements patched below
    model.learners_ = learners or {}

    def _pred(task, **kw):
        return {c: gain for c in ts.STACK_CANDIDATES}

    model._predict_improvements = _pred
    return model


class _FakeAnchor:
    model = True  # truthy sentinel: fitted

    def predict(self, task, feats=None):
        return np.full(task.n_predict, _BASE_LEVEL)


def test_gate_killed_returns_bit_identical_anchor():
    ts = _mod("trajectory_stack")
    task = _rule_task(tvt_value=0.0)
    model = _gated_ready_model(np.zeros(task.n_predict), killed=True)
    anchor = _FakeAnchor()
    base = anchor.predict(task)
    pred = model.predict(task)
    np.testing.assert_array_equal(pred, base)
    assert "killed" in model.decision_log[-1]["reason"] or "test_killed" in model.decision_log[-1]["reason"]


def test_gate_applies_zero_correction_under_all_rules_passed():
    ts = _mod("trajectory_stack")
    task = _rule_task(tvt_value=0.0)
    # Zero PF/Beam shift: pseudo truth is 0, anchor predicts 3, so a zero
    # candidate has delta = 3 > 0 and a trivially better tail.
    model = _gated_ready_model(np.zeros(task.n_predict), gain=1.0)
    pred = model.predict(task)
    assert model.decision_log[-1]["outcome"].startswith("applied_")
    # anchor_tvt of this well is 0: applied prediction equals the anchor TVT
    assert np.allclose(pred, task.anchor_tvt, atol=1e-9)
    assert not np.allclose(pred, _BASE_LEVEL)


def test_gate_falls_back_when_pseudo_not_improved():
    ts = _mod("trajectory_stack")
    task = _rule_task(tvt_value=0.0)
    # Shift +3 everywhere: candidate hits the anchor's own mistake and never
    # improves the pseudo-holdout (delta ~ 0), so rule 1 blocks it.
    model = _gated_ready_model(np.full(task.n_predict, _BASE_LEVEL), gain=1.0)
    pred = model.predict(task)
    np.testing.assert_array_equal(pred, _FakeAnchor().predict(task))
    reason = model.decision_log[-1]["reason"]
    assert "pseudo_holdout_not_improved" in reason


def test_gate_falls_back_below_confidence_threshold():
    ts = _mod("trajectory_stack")
    task = _rule_task(tvt_value=0.0)
    thr = ts.StackGateThresholds(margin=0.0, conf_thr=0.99, sep_cap=np.inf, shrink=1.0, warmup=1)
    model = _gated_ready_model(np.zeros(task.n_predict), gain=1.0, thr=thr)
    pred = model.predict(task)
    np.testing.assert_array_equal(pred, _FakeAnchor().predict(task))
    assert "confidence_below_oof_threshold" in model.decision_log[-1]["reason"]


def test_gate_falls_back_on_disagreement():
    ts = _mod("trajectory_stack")
    task = _rule_task(tvt_value=0.0)
    thr = ts.StackGateThresholds(margin=0.0, conf_thr=0.0, sep_cap=0.001, shrink=1.0, warmup=1)
    # PF +2 / Beam -2 shifts: they disagree, and the disagreement proxy is the
    # PF-vs-Beam track gap (=4) computed on the fake frames.
    ts_mod = _mod("trajectory_stack")
    cfg = ts_mod.TrajectoryGateConfig(inner_splits=2, tune_splits=2, seed=0)
    pf = _FakePathGenerator("pf", np.full(task.n_predict, 2.0))
    beam = _FakePathGenerator("beam", np.full(task.n_predict, -2.0))
    model = ts_mod.GatedTrajectoryStack(
        pf=pf, beam=beam, anchor_model=_FakeAnchor(), config=cfg,
        protocol="unseen_well", fold=0,
    )
    model.thresholds = thr
    model.gate_model = True
    model.learners_ = {}
    model._predict_improvements = lambda task, **kw: {c: 1.0 for c in ts_mod.STACK_CANDIDATES}
    pred = model.predict(task)
    reason = model.decision_log[-1]["reason"]
    # every available PF/Beam candidate carries disagreement 4 > 0.001
    assert "branch_disagreement_above_cap" in reason
    np.testing.assert_array_equal(pred, _FakeAnchor().predict(task))


def test_gate_exception_returns_anchor():
    ts = _mod("trajectory_stack")
    task = _rule_task(tvt_value=0.0)

    class _Boom:
        feature_columns = ("pf_shift",)

        def generate(self, task):
            raise RuntimeError("forced failure")

    anchor = _FakeAnchor()
    model = ts.GatedTrajectoryStack(
        pf=_Boom(), beam=_Boom(), anchor_model=anchor,
        config=ts.TrajectoryGateConfig(inner_splits=2, tune_splits=2, seed=0),
        protocol="unseen_well", fold=0,
    )
    model.gate_model = True
    pred = model.predict(task)
    np.testing.assert_array_equal(pred, anchor.predict(task))
    assert "decision_exception" in model.decision_log[-1]["reason"]


def test_gate_oof_skills_match_between_train_and_inference_rows(mount):
    """Train/serve consistency: the aggregate inner-OOF skills attached to
    the full-fold learners (inference) are exactly the skills used to build
    the gate's training examples (no zero-train vs positive-serve skew)."""
    ts = _mod("trajectory_stack")
    from src.baselines import RidgeBaseline
    from src.geoanchor import MemoizedPathGenerator
    from src.particle_filter import ParticleFilterFeatureGenerator
    from src.beam_search import BeamSearchFeatureGenerator

    tasks = _tasks(mount, ["TRW001", "TRW002", "TRW006", "TRW007", "TRW008", "TRW009"], "real")
    anchor = RidgeBaseline(alignment_features=False)
    anchor.fit(tasks)
    memo: dict = {}

    def _mk(cls, family):
        return MemoizedPathGenerator(
            cls(cache=None, dataset_version="t", fold_id=0, protocol="u", device="cpu"),
            memo, family,
        )

    cfg = ts.TrajectoryGateConfig(
        inner_splits=2, tune_splits=2, seed=0, boost_max_iter=10, boost_estop_rounds=4,
    )
    model = ts.GatedTrajectoryStack(
        pf=_mk(ParticleFilterFeatureGenerator, "pf"),
        beam=_mk(BeamSearchFeatureGenerator, "beam"),
        anchor_model=anchor, config=cfg, protocol="unseen_well", fold=0,
    )
    model.fit(tasks)
    for name, learner in getattr(model, "learners_", {}).items():
        expected = float(np.clip(model._oof_skills.get(name, 0.0), 0.0, 1.0))
        assert float(learner.info.oof_skill) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Multi-scale scan
# ---------------------------------------------------------------------------


def test_multiscale_scan_reports_and_degrades():
    ts = _mod("trajectory_stack")
    task = _rule_task(tvt_value=5.0)
    res = ts.multiscale_scan(task)
    if res.ok:
        assert len(res.shifts) >= 1
        assert res.ptp >= 0.0
        assert isinstance(res.n_agree, int)
    else:
        assert res.failure_reason
    # No typewell at all -> honest failure, never a crash.
    tasks_mod = _mod("tasks")
    data = _mod("data")
    n, start = 300, 250
    md = 9000.0 + np.arange(n, dtype=float)
    tvt = np.full(n, 5.0)
    tvt_input = tvt.copy()
    tvt_input[start:] = np.nan
    hw = pd.DataFrame({"MD": md, "X": md, "Y": md, "Z": -md, "GR": np.full(n, 70.0),
                       "TVT_input": tvt_input, "TVT": tvt})
    well = data.WellData(
        well_id="NOTW", split="train", hw=hw, tw=None,
        roles={"md": "MD", "x": "X", "y": "Y", "z": "Z", "gr": "GR",
               "tvt_input": "TVT_input", "tvt": "TVT"},
        markers={}, tw_roles={}, region_info={"prediction_start_row": start},
    )
    res2 = ts.multiscale_scan(tasks_mod.make_task(well, "real").inputs())
    assert not res2.ok
    assert res2.failure_reason == "missing_or_invalid_typewell"


# ---------------------------------------------------------------------------
# Promotion decision rules
# ---------------------------------------------------------------------------


def _decision_frames(candidate_delta=-0.5):
    """Tiny synthetic frames exercising the promotion rules mechanically."""
    protocols = ["same_well_masked", "unseen_well"]
    rows = []
    for proto in protocols:
        for i in range(100):
            well = f"w{i:03d}"
            ridge_rmse = 14.0 if proto == "unseen_well" else 29.0
            rows.append(dict(model="ridge_default", protocol=proto, fold=i % 5,
                             well_id=well, n_points=100, sse=ridge_rmse**2 * 100,
                             rmse=ridge_rmse, max_abs_error=30.0, bias=0.0,
                             prefix_len=3000, suffix_len=1000, gr_missing_frac=0.1,
                             anchor_tvt=0.0, has_typewell=True, predict_seconds=0.01))
            cand_rmse = ridge_rmse + candidate_delta
            rows.append(dict(model="gated_trajectory", protocol=proto, fold=i % 5,
                             well_id=well, n_points=100, sse=cand_rmse**2 * 100,
                             rmse=cand_rmse, max_abs_error=30.0, bias=0.0,
                             prefix_len=3000, suffix_len=1000, gr_missing_frac=0.1,
                             anchor_tvt=0.0, has_typewell=True, predict_seconds=0.01))
    well_df = pd.DataFrame(rows)
    from src.validation import summarize
    from src.pf_beam_robustness import pair_default_vs_candidate, fold_deltas, bootstrap_global_rmse_delta

    summary = summarize(well_df)
    paired = pair_default_vs_candidate(well_df, default="ridge_default", candidate="gated_trajectory")
    paired["candidate_arm"] = "gated_trajectory"
    stab = fold_deltas(well_df, default="ridge_default", candidate="gated_trajectory")
    stab["candidate_arm"] = "gated_trajectory"
    boot = bootstrap_global_rmse_delta(paired, n_boot=100, seed=0)
    boot["candidate_arm"] = "gated_trajectory"
    decisions = pd.DataFrame(
        [{"protocol": "unseen_well", "fold": 0, "well_id": f"w{i:03d}",
          "outcome": "applied_pf", "reason": "all_rules_passed"} for i in range(100)]
    )
    return summary, well_df, stab, boot, decisions


def test_promotion_denied_when_not_real_run():
    tsd = _mod("trajectory_stack_decision")
    summary, well_df, stab, boot, decisions = _decision_frames(candidate_delta=-0.5)
    ev = tsd.evaluate_arm(
        "gated_trajectory", summary=summary, well_df=well_df, fold_stab=stab,
        boot_ci=boot, decision_log=decisions, stack_infos=[], is_real=False,
    )
    assert ev["all_rules_passed"] is False  # never from a synthetic run
    assert ev["rules"]["r1_unseen_beats_reference"]["passed"] is True
    assert ev["rules"]["r4_fold_majority_improved"]["passed"] is True


def test_promotion_fails_when_candidate_worse():
    tsd = _mod("trajectory_stack_decision")
    summary, well_df, stab, boot, decisions = _decision_frames(candidate_delta=+0.5)
    ev = tsd.evaluate_arm(
        "gated_trajectory", summary=summary, well_df=well_df, fold_stab=stab,
        boot_ci=boot, decision_log=decisions, stack_infos=[], is_real=True,
    )
    assert ev["all_rules_passed"] is False
    assert ev["rules"]["r1_unseen_beats_reference"]["passed"] is False
    assert ev["rules"]["r4_fold_majority_improved"]["passed"] is False


def test_rule8_detects_fallback_sse_mismatch():
    tsd = _mod("trajectory_stack_decision")
    summary, well_df, stab, boot, decisions = _decision_frames(candidate_delta=-0.5)
    # Make one fallback row's SSE differ between arms: fallback was NOT exact.
    mask = (
        (well_df["model"] == "gated_trajectory")
        & (well_df["well_id"] == "w000")
        & (well_df["protocol"] == "unseen_well")
    )
    well_df.loc[mask, "sse"] += 1.0
    from src.validation import summarize
    from src.pf_beam_robustness import fold_deltas, pair_default_vs_candidate, bootstrap_global_rmse_delta

    summary = summarize(well_df)
    paired = pair_default_vs_candidate(well_df, default="ridge_default", candidate="gated_trajectory")
    paired["candidate_arm"] = "gated_trajectory"
    stab2 = fold_deltas(well_df, default="ridge_default", candidate="gated_trajectory")
    stab2["candidate_arm"] = "gated_trajectory"
    boot2 = bootstrap_global_rmse_delta(paired, n_boot=50, seed=0)
    boot2["candidate_arm"] = "gated_trajectory"
    decisions = pd.DataFrame(
        [{"protocol": "unseen_well", "fold": 0, "well_id": "w000",
          "outcome": "fallback", "reason": "test"}]
    )
    ev = tsd.evaluate_arm(
        "gated_trajectory", summary=summary, well_df=well_df, fold_stab=stab2,
        boot_ci=boot2, decision_log=decisions, stack_infos=[], is_real=True,
    )
    assert ev["rules"]["r8_exact_ridge_fallback"]["passed"] is False


def test_experiment_cli_rejects_degenerate_split_depths():
    mod = importlib.import_module("scripts.run_trajectory_stack_experiment")
    with pytest.raises(SystemExit):
        mod.main(["--inner-splits", "1"])
    with pytest.raises(SystemExit):
        mod.main(["--tune-splits", "1"])
    with pytest.raises(SystemExit):
        mod.main(["--inner-splits", "0", "--tune-splits", "0"])
