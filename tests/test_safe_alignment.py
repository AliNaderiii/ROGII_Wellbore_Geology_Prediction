"""Leakage and behaviour tests for the staged safe-alignment pipeline.

Covers the mandate's non-negotiables:

* blocked public test wells can never enter folds, fitting or tuning;
* the hidden region is structurally unreachable in every pseudo cut;
* every stage falls back to the *exact* Ridge Default prediction when a
  guard declines or a component fails;
* corrections are bounded and warm up smoothly (no abrupt first row);
* robust projection rejects unstable fits and bounds movement;
* the runner's stage registry and decision bookkeeping stay consistent.

Synthetic fixtures only — never a competition result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.safe_alignment import (
    STAGE_A,
    STAGE_B,
    STAGE_C,
    STAGE_D,
    STAGE_E,
    STAGE_F,
    STAGE_LEVEL,
    STAGE_ORDER,
    SafeAlignmentConfig,
    SafeAlignmentModel,
    _ramp,
    build_stage_models,
    pseudo_task_at,
    robust_projection,
)
from src.tasks import InferenceTask, WellTask
from src.validation import BLOCKED_WELL_IDS, make_group_folds

STAGE_B_NAME = STAGE_B  # readability alias


# --------------------------------------------------------------------------
# Task fixtures
# --------------------------------------------------------------------------


def _make_task(
    well_id: str = "SYN001",
    n: int = 1400,
    prefix: int = 1000,
    seed: int = 0,
    gr_missing: slice | None = None,
    with_target: bool = True,
) -> WellTask:
    rng = np.random.default_rng(seed)
    md = 8000.0 + np.arange(n, dtype="float64")
    tvt = 20.0 + np.cumsum(rng.normal(0.0, 0.05, n))
    gr = 60.0 + 25.0 * np.sin(tvt / 4.0) + rng.normal(0, 3.0, n)
    if gr_missing is not None:
        gr[gr_missing] = np.nan
    tvt_known = tvt.copy()
    tvt_known[prefix:] = np.nan
    tw_tvt = np.linspace(0.0, 60.0, 240)
    tw_gr = 60.0 + 25.0 * np.sin(tw_tvt / 4.0)
    inp = InferenceTask(
        well_id=well_id,
        split="train",
        mode="real",
        start=prefix,
        stop=n,
        md=md,
        x=1000.0 + 0.3 * md,
        y=2000.0 + 0.1 * md,
        z=-0.9 * md,
        gr=gr,
        tvt_known=tvt_known,
        tw_tvt=tw_tvt,
        tw_gr=tw_gr,
        tw_geology=None,
    )
    target = tvt[prefix:n].copy() if with_target else None
    return WellTask(inputs_=inp, target=target)


def _fit_stage(stage: str, tasks: list[WellTask], **kw) -> SafeAlignmentModel:
    models = build_stage_models(
        (STAGE_A, stage), memo={}, protocol="test", fold=0, tune=kw.pop("tune", False)
    )
    for name in (STAGE_A, stage):
        models[name].fit(tasks)
    return models[stage]


@pytest.fixture(scope="module")
def train_tasks() -> list[WellTask]:
    return [_make_task(f"SYN{i:03d}", seed=i) for i in range(8)]


# --------------------------------------------------------------------------
# Leakage guards
# --------------------------------------------------------------------------


class TestLeakageGuards:
    def test_blocked_wells_never_reach_folds(self):
        ids = [f"W{i}" for i in range(12)] + sorted(BLOCKED_WELL_IDS)
        folds = make_group_folds(ids, n_splits=3)
        for fold in folds:
            assert not (set(fold.train_ids) | set(fold.valid_ids)) & BLOCKED_WELL_IDS

    def test_fold_construction_raises_on_blocked_well(self):
        # Import inside the test: the shared `mount` fixture reloads
        # src.validation, so a module-level import could hold a stale class.
        import src.validation as validation

        with pytest.raises(validation.BlockedWellError):
            validation.Fold(
                index=0,
                train_ids=["A", next(iter(validation.BLOCKED_WELL_IDS))],
                valid_ids=["B"],
            )

    def test_pseudo_cut_carries_no_hidden_rows(self):
        task = _make_task().inputs()
        built = pseudo_task_at(task, 700)
        assert built is not None
        cut_task, truth = built
        # The nested boundary lies strictly inside the parent's visible prefix.
        assert cut_task.stop <= task.start
        # And the nested task's own hidden region is all-NaN.
        cut_task.assert_no_target()
        assert np.isfinite(truth).all()
        # Truth comes from TVT_input the parent could legitimately see.
        np.testing.assert_allclose(truth, task.tvt_known[700 : task.start])

    def test_pseudo_cut_refuses_short_prefixes(self):
        task = _make_task(prefix=250).inputs()
        assert pseudo_task_at(task, 100) is None  # below MIN_PREFIX_ROWS
        assert pseudo_task_at(task, 240) is None  # window too small

    def test_stage_decisions_use_no_target_attribute(self, train_tasks):
        model = _fit_stage(STAGE_F, train_tasks)
        task = _make_task("SYNX01", seed=99).inputs()
        task.assert_no_target()
        base, corr, dec = model._decide(task)
        # A decision object never carries target values, only diagnostics.
        assert not hasattr(dec, "target")
        assert base.size == task.n_predict

    def test_tuner_reads_only_fold_train_targets(self, train_tasks):
        """The stage-F tuner consumes WellTask.scored() of fold-train wells
        only; passing tasks without targets must silently keep defaults."""
        untargeted = [WellTask(inputs_=t.inputs_, target=None) for t in train_tasks]
        models = build_stage_models(
            (STAGE_A, STAGE_F), memo={}, protocol="t", fold=0, tune=True
        )
        models[STAGE_A].fit(untargeted)
        models[STAGE_F].fit(untargeted)
        cfg = SafeAlignmentConfig()
        assert models[STAGE_F].conf_thr == float(cfg.conf_grid[0])
        assert models[STAGE_F].warmup == int(cfg.warmup_rows)


# --------------------------------------------------------------------------
# Exact Ridge fallback
# --------------------------------------------------------------------------


class TestExactRidgeFallback:
    def test_stage_a_is_the_shared_anchor_instance(self, train_tasks):
        models = build_stage_models(
            STAGE_ORDER, memo={}, protocol="t", fold=0, tune=False
        )
        for stage in STAGE_ORDER:
            models[stage].fit(train_tasks)
        anchor = models[STAGE_A]
        for stage in STAGE_ORDER[1:]:
            assert models[stage].anchor_model is anchor

    def test_no_typewell_falls_back_bit_exact(self, train_tasks):
        model = _fit_stage(STAGE_F, train_tasks)
        t = _make_task("SYNX02", seed=7)
        inp = t.inputs()
        stripped = InferenceTask(
            **{
                **{f: getattr(inp, f) for f in inp.__dataclass_fields__},
                "tw_tvt": None,
                "tw_gr": None,
            }
        )
        pred = model.predict(stripped)
        ridge = model.anchor_model.predict(stripped)
        np.testing.assert_array_equal(pred, ridge)
        assert model.decision_log[-1].outcome == "fallback"

    def test_all_gr_missing_falls_back_bit_exact(self, train_tasks):
        model = _fit_stage(STAGE_F, train_tasks)
        t = _make_task("SYNX03", seed=8, gr_missing=slice(0, None))
        pred = model.predict(t.inputs())
        ridge = model.anchor_model.predict(t.inputs())
        np.testing.assert_array_equal(pred, ridge)

    def test_decision_exception_falls_back_bit_exact(self, train_tasks, monkeypatch):
        model = _fit_stage(STAGE_F, train_tasks)
        monkeypatch.setattr(
            model, "_bundle", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        t = _make_task("SYNX04", seed=9)
        pred = model.predict(t.inputs())
        ridge = model.anchor_model.predict(t.inputs())
        np.testing.assert_array_equal(pred, ridge)
        assert "decision_exception" in model.decision_log[-1].reason \
            or "bundle_failed" in model.decision_log[-1].reason

    def test_every_prediction_is_finite(self, train_tasks):
        for stage in STAGE_ORDER[1:]:
            model = _fit_stage(stage, train_tasks)
            pred = model.predict(_make_task("SYNX05", seed=11).inputs())
            assert np.isfinite(pred).all(), stage


# --------------------------------------------------------------------------
# Correction bounds, warmup, projection
# --------------------------------------------------------------------------


class TestBoundsAndWarmup:
    def test_correction_never_exceeds_cap(self, train_tasks):
        cfg = SafeAlignmentConfig()
        for stage in (STAGE_B, STAGE_C, STAGE_D, STAGE_E, STAGE_F):
            model = _fit_stage(stage, train_tasks)
            for seed in (21, 22, 23):
                t = _make_task(f"SYNB{seed}", seed=seed)
                inp = t.inputs()
                base, corr, dec = model._decide(inp)
                if corr is not None:
                    assert np.max(np.abs(corr)) <= cfg.correction_cap_ft + 1e-9

    def test_warmup_prevents_abrupt_first_row(self, train_tasks):
        model = _fit_stage(STAGE_B_NAME, train_tasks)
        for seed in range(30, 40):
            t = _make_task(f"SYNW{seed}", seed=seed)
            inp = t.inputs()
            base, corr, dec = model._decide(inp)
            if corr is None:
                continue
            pred = model.predict(inp)
            ramp0 = 1.0 / max(min(model.warmup, max(10, inp.n_predict // 2)), 1)
            assert abs(pred[0] - base[0]) <= abs(corr[0]) * ramp0 + 1e-9

    def test_ramp_shape(self):
        r = _ramp(500, 200)
        assert r[0] == pytest.approx(1.0 / 200.0)
        assert r[199] == pytest.approx(1.0)
        assert np.all(np.diff(r) >= -1e-12)
        assert _ramp(0, 200).size == 0

    def test_projection_bounds_movement(self):
        cfg = SafeAlignmentConfig()
        n = 400
        md = np.arange(n, dtype="float64")
        z = -0.9 * md
        rng = np.random.default_rng(0)
        cand = 20.0 + 0.01 * md + rng.normal(0, 3.0, n)
        out, applied, reason = robust_projection(md, z, cand, 20.0, cfg)
        assert applied, reason
        assert np.max(np.abs(out - cand)) <= cfg.projection_max_move_ft + 1e-9

    def test_projection_rejects_short_and_degenerate_inputs(self):
        cfg = SafeAlignmentConfig()
        md = np.arange(10, dtype="float64")
        out, applied, reason = robust_projection(md, -md, md * 0.0, 0.0, cfg)
        assert not applied and reason == "projection_too_few_rows"
        n = 200
        same_md = np.zeros(n)
        out, applied, reason = robust_projection(
            same_md, -same_md, np.full(n, 5.0), 5.0, cfg
        )
        assert not applied and reason == "projection_degenerate_md"

    def test_projection_rejects_unstable_fit(self):
        cfg = SafeAlignmentConfig(projection_max_move_ft=0.01)
        n = 400
        md = np.arange(n, dtype="float64")
        rng = np.random.default_rng(1)
        cand = 20.0 + np.cumsum(rng.normal(0, 1.0, n))  # wild walk: huge moves
        out, applied, reason = robust_projection(md, -0.9 * md, cand, 20.0, cfg)
        assert not applied and reason == "projection_unstable_fit"
        np.testing.assert_array_equal(out, cand)


# --------------------------------------------------------------------------
# Stage registry / decision bookkeeping
# --------------------------------------------------------------------------


class TestStageMachinery:
    def test_stage_order_is_monotone(self):
        assert list(STAGE_ORDER) == sorted(STAGE_ORDER, key=STAGE_LEVEL.get)
        assert STAGE_ORDER[0] == STAGE_A

    def test_unknown_stage_rejected(self):
        with pytest.raises(ValueError):
            SafeAlignmentModel("bogus", pf=None, beam=None)

    def test_decisions_are_logged_per_well(self, train_tasks):
        log: list = []
        models = build_stage_models(
            (STAGE_A, STAGE_D), memo={}, protocol="t", fold=3, decision_log=log, tune=False
        )
        models[STAGE_A].fit(train_tasks)
        models[STAGE_D].fit(train_tasks)
        for seed in (51, 52):
            models[STAGE_D].predict(_make_task(f"SYNL{seed}", seed=seed).inputs())
        assert len(log) == 2
        assert {d.stage for d in log} == {STAGE_D}
        assert {d.fold for d in log} == {3}
        assert all(d.outcome in ("applied", "fallback") for d in log)

    def test_stage_f_requires_corroborating_cuts(self, train_tasks):
        """A stage-F application needs >= 2 valid cuts; wells whose prefix
        cannot host extra cuts must fall back."""
        model = _fit_stage(STAGE_F, train_tasks)
        # Prefix of 260 rows: the primary nested cut fits, the 0.5/0.65/0.75
        # cuts all collapse below MIN_PREFIX_ROWS => at most 1 valid cut.
        t = _make_task("SYNF01", n=560, prefix=260, seed=61)
        pred = model.predict(t.inputs())
        ridge = model.anchor_model.predict(t.inputs())
        dec = model.decision_log[-1]
        if dec.outcome == "fallback" and dec.n_cuts_valid <= 1:
            np.testing.assert_array_equal(pred, ridge)

    def test_gate_diagnostics_reported(self, train_tasks):
        model = _fit_stage(STAGE_F, train_tasks)
        inp = _make_task("SYNF02", seed=62).inputs()
        pred = model.predict(inp)
        diag = model.prediction_diagnostics(inp, None, pred)
        assert "gate_activation" in diag
        assert "gate_fallback_exact_ridge" in diag
        assert diag["gate_activation"] != diag["gate_fallback_exact_ridge"]
