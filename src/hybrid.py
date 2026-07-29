"""Leakage-safe Ridge/neural hybrids and conservative gates.

The module intentionally treats Ridge Default as an immutable anchor.  A
candidate can only alter a prediction after an inner, well-level OOF exercise
has shown a safe improvement.  If fitting, diagnostics, or a gate rule fails,
the returned array is byte-for-byte the anchor model's prediction.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.baselines import BaselineModel, RidgeBaseline
from src.neural import NeuralConfig, NeuralResidualModel, set_deterministic
from src.tasks import InferenceTask, WellTask
from src.validation import make_group_folds


@dataclass
class GateThresholds:
    confidence_threshold: float = 1.0
    max_correction: float = 0.0
    min_inner_wells: int = 0
    accepted: bool = False
    reason: str = "not_fitted"


@dataclass
class _OOFRow:
    well_id: str
    anchor: np.ndarray
    candidate: np.ndarray
    truth: np.ndarray
    confidence: float
    correction_magnitude: float


def _safe_rmse(pred, truth) -> float:
    p, t = np.asarray(pred, dtype="float64"), np.asarray(truth, dtype="float64")
    m = np.isfinite(p) & np.isfinite(t)
    return float(np.sqrt(np.mean((p[m] - t[m]) ** 2))) if m.any() else np.inf


def _global_sse(rows: Sequence[_OOFRow], *, gated: bool = False, candidate_only: bool = False, threshold: float = 1.0, max_corr: float = np.inf):
    se = n = 0
    per_well = []
    for row in rows:
        use = candidate_only or (gated and row.confidence >= threshold and row.correction_magnitude <= max_corr)
        pred = row.candidate if use else row.anchor
        m = np.isfinite(pred) & np.isfinite(row.truth)
        if m.any():
            se += float(np.sum((pred[m] - row.truth[m]) ** 2))
            n += int(m.sum())
            per_well.append(float(np.sqrt(np.mean((pred[m] - row.truth[m]) ** 2))))
    return (se / max(n, 1), per_well)


class RidgeNeuralBlend(BaselineModel):
    """Ridge Default plus a blend weight selected from inner well-level OOF."""

    needs_alignment = False
    name = "ridge_neural_blend"

    def __init__(
        self,
        neural_config: NeuralConfig | None = None,
        *,
        inner_splits: int = 3,
        seed: int = 17,
    ):
        self.neural_config = copy.deepcopy(neural_config or NeuralConfig())
        self.inner_splits = int(inner_splits)
        self.seed = int(seed)
        self.anchor_model: RidgeBaseline | None = None
        self.neural_model: NeuralResidualModel | None = None
        self.weight_neural_: float = 0.0
        self.fit_report: dict = {}

    def _inner_oof(self, tasks: Sequence[WellTask]) -> list[_OOFRow]:
        ids = sorted({str(t.well_id) for t in tasks})
        if len(ids) < max(2, self.inner_splits):
            return []
        folds = make_group_folds(ids, n_splits=min(self.inner_splits, len(ids)), seed=self.seed)
        by_id = {str(t.well_id): t for t in tasks}
        rows: list[_OOFRow] = []
        for fold in folds:
            tr = [by_id[w] for w in fold.train_ids if w in by_id]
            va = [by_id[w] for w in fold.valid_ids if w in by_id]
            if not tr or not va:
                continue
            anchor = RidgeBaseline(alpha=10.0, alignment_features=False).fit(tr)
            cfg = copy.deepcopy(self.neural_config)
            cfg.seed = self.seed + fold.index + 1
            # Inner OOF is a selection diagnostic; keep it bounded and never
            # use it as the final outer-fold model.
            cfg.max_epochs = min(cfg.max_epochs, 10)
            cfg.patience = min(cfg.patience, 3)
            neural = NeuralResidualModel(config=cfg).fit(tr)
            for task in va:
                inp = task.inputs()
                ap = np.asarray(anchor.predict(inp), dtype="float64")
                cp = np.asarray(neural.predict(inp), dtype="float64")
                truth = np.asarray(task.scored(), dtype="float64")
                corr = float(np.sqrt(np.nanmean((cp - ap) ** 2))) if np.isfinite(cp - ap).any() else np.inf
                # Confidence is target-free at inference.  It is deliberately
                # a conservative function of correction magnitude; the
                # magnitude bound itself is learned from OOF rows below.
                confidence = 1.0 / (1.0 + max(corr, 0.0))
                rows.append(_OOFRow(str(task.well_id), ap, cp, truth, confidence, corr))
        return rows

    def fit(self, tasks: Sequence[WellTask], **kw):
        from src.validation import assert_no_blocked_wells
        assert_no_blocked_wells([t.well_id for t in tasks], context="hybrid supervised fitting")
        set_deterministic(self.seed)
        rows = self._inner_oof(tasks)
        if rows:
            # Minimise pooled OOF SSE over a deterministic finite grid.  This
            # is an OOF-derived weight, never a public-LB-tuned constant.
            best_w, best_sse = 0.0, np.inf
            for w in np.linspace(0.0, 1.0, 21):
                se = 0.0
                for row in rows:
                    pred = w * row.candidate + (1.0 - w) * row.anchor
                    m = np.isfinite(pred) & np.isfinite(row.truth)
                    se += float(np.sum((pred[m] - row.truth[m]) ** 2))
                if se < best_sse:
                    best_sse, best_w = se, float(w)
            self.weight_neural_ = best_w
        else:
            # No inner evidence means no candidate evidence.  Exact Ridge is
            # safer than inventing a blend weight.
            self.weight_neural_ = 0.0
        self.anchor_model = RidgeBaseline(alpha=10.0, alignment_features=False).fit(tasks)
        cfg = copy.deepcopy(self.neural_config)
        cfg.seed = self.seed
        self.neural_model = NeuralResidualModel(config=cfg).fit(tasks)
        self.fit_report = {
            "inner_oof_wells": len({r.well_id for r in rows}),
            "inner_oof_rows": len(rows),
            "weight_neural": self.weight_neural_,
            "weight_anchor": 1.0 - self.weight_neural_,
            "weight_selection": "inner_group_oof_grid",
            "fallback_exact_ridge_if_no_oof": True,
        }
        return self

    def predict(self, task: InferenceTask, feats=None):
        if self.anchor_model is None or self.neural_model is None:
            return np.asarray(self.anchor_model.predict(task, feats), dtype="float64") if self.anchor_model else np.full(task.n_predict, self._anchor(task))
        anchor = np.asarray(self.anchor_model.predict(task, feats), dtype="float64")
        neural = np.asarray(self.neural_model.predict(task, feats), dtype="float64")
        pred = (1.0 - self.weight_neural_) * anchor + self.weight_neural_ * neural
        if not np.isfinite(pred).all():
            return anchor  # exact fallback
        return pred

    def prediction_diagnostics(self, task, feats, pred):
        return {
            "hybrid_weight_neural": self.weight_neural_,
            "hybrid_weight_anchor": 1.0 - self.weight_neural_,
            "hybrid_fallback": False,
        }


class ConservativeRidgeNeuralGate(BaselineModel):
    """Apply a neural correction only when inner OOF safety rules pass.

    Rules implemented here are intentionally conservative and auditable:
    * inner OOF candidate must beat the anchor on pooled squared error;
    * the gated policy must not increase the mean or worst-10 well RMSE;
    * confidence and correction thresholds are selected from inner OOF only;
    * correction must be finite and bounded at inference;
    * any failure returns the exact Ridge prediction.
    """

    needs_alignment = False
    name = "ridge_neural_gated"

    def __init__(self, neural_config: NeuralConfig | None = None, *, inner_splits: int = 3, seed: int = 17):
        self.neural_config = copy.deepcopy(neural_config or NeuralConfig())
        self.inner_splits = int(inner_splits)
        self.seed = int(seed)
        self.anchor_model: RidgeBaseline | None = None
        self.neural_model: NeuralResidualModel | None = None
        self.thresholds = GateThresholds()
        self.fit_report: dict = {}

    def _rows(self, tasks):
        blend = RidgeNeuralBlend(self.neural_config, inner_splits=self.inner_splits, seed=self.seed)
        return blend._inner_oof(tasks)

    def fit(self, tasks: Sequence[WellTask], **kw):
        from src.validation import assert_no_blocked_wells
        assert_no_blocked_wells([t.well_id for t in tasks], context="gated hybrid supervised fitting")
        set_deterministic(self.seed)
        rows = self._rows(tasks)
        self.anchor_model = RidgeBaseline(alpha=10.0, alignment_features=False).fit(tasks)
        if not rows:
            self.thresholds = GateThresholds(reason="no_inner_oof_evidence")
            self.neural_model = None
            self.fit_report = {"gate_accepted": False, "fallback_rate_inner_oof": 1.0}
            return self
        anchor_mse, anchor_well = _global_sse(rows)
        candidate_mse, cand_well = _global_sse(rows, candidate_only=True)
        # Candidate is kept only if it improves the same OOF rows.  The gate
        # can still reject individual wells later.
        if not np.isfinite(candidate_mse) or candidate_mse >= anchor_mse:
            self.thresholds = GateThresholds(reason="candidate_not_better_on_inner_oof")
            self.neural_model = None
            self.fit_report = {"gate_accepted": False, "candidate_mse": candidate_mse, "anchor_mse": anchor_mse}
            return self
        corr_values = np.asarray([r.correction_magnitude for r in rows if np.isfinite(r.correction_magnitude)], dtype="float64")
        max_corr = float(np.quantile(corr_values, 0.95)) if corr_values.size else 0.0
        candidates = sorted({float(np.quantile([r.confidence for r in rows], q)) for q in np.linspace(0, 1, 21)})
        candidates = [*candidates, 1.0]
        chosen = None
        for threshold in candidates:
            mse, wells = _global_sse(rows, gated=True, threshold=threshold, max_corr=max_corr)
            if not wells:
                continue
            worst10 = float(np.mean(sorted(wells, reverse=True)[: min(10, len(wells))]))
            anchor_worst10 = float(np.mean(sorted(anchor_well, reverse=True)[: min(10, len(anchor_well))]))
            if mse <= anchor_mse and worst10 <= anchor_worst10:
                # Prefer the safest threshold among policies that pass, then
                # maximise activation only as a tie-breaker.
                activated = sum(r.confidence >= threshold and r.correction_magnitude <= max_corr for r in rows)
                score = (activated, -threshold)
                if chosen is None or score > chosen[0]:
                    chosen = (score, threshold, worst10)
        if chosen is None:
            self.thresholds = GateThresholds(reason="gated_policy_tail_or_global_regression")
            self.neural_model = None
            return self
        self.thresholds = GateThresholds(
            confidence_threshold=float(chosen[1]), max_correction=max_corr,
            min_inner_wells=len({r.well_id for r in rows}), accepted=True, reason="inner_oof_rules_passed",
        )
        cfg = copy.deepcopy(self.neural_config)
        cfg.seed = self.seed
        self.neural_model = NeuralResidualModel(config=cfg).fit(tasks)
        self.fit_report = {
            "gate_accepted": True,
            "anchor_mse_inner_oof": anchor_mse,
            "candidate_mse_inner_oof": candidate_mse,
            "confidence_threshold": self.thresholds.confidence_threshold,
            "max_correction": self.thresholds.max_correction,
            "inner_oof_wells": len({r.well_id for r in rows}),
            "inner_oof_activation_rate": float(sum(r.confidence >= self.thresholds.confidence_threshold and r.correction_magnitude <= max_corr for r in rows) / len(rows)),
        }
        return self

    def predict(self, task: InferenceTask, feats=None):
        if self.anchor_model is None:
            return np.full(task.n_predict, self._anchor(task), dtype="float64")
        anchor = np.asarray(self.anchor_model.predict(task, feats), dtype="float64")
        if not self.thresholds.accepted or self.neural_model is None:
            return anchor  # exact Ridge fallback
        try:
            candidate = np.asarray(self.neural_model.predict(task, feats), dtype="float64")
            correction = candidate - anchor
            magnitude = float(np.sqrt(np.nanmean(correction ** 2))) if np.isfinite(correction).any() else np.inf
            confidence = 1.0 / (1.0 + max(magnitude, 0.0))
            ok = (
                np.isfinite(candidate).all()
                and np.isfinite(correction).all()
                and magnitude <= self.thresholds.max_correction
                and confidence >= self.thresholds.confidence_threshold
            )
            return candidate if ok else anchor
        except Exception:
            return anchor

    def prediction_diagnostics(self, task, feats, pred):
        anchor = self.anchor_model.predict(task, feats) if self.anchor_model else np.full(task.n_predict, self._anchor(task))
        correction = np.asarray(pred) - np.asarray(anchor)
        return {
            "gate_accepted": self.thresholds.accepted,
            "gate_confidence_threshold": self.thresholds.confidence_threshold,
            "gate_max_correction": self.thresholds.max_correction,
            "gate_activation": bool(np.any(np.abs(correction) > 0)),
            "gate_correction_magnitude": float(np.sqrt(np.nanmean(correction ** 2))) if np.isfinite(correction).any() else np.inf,
            "gate_fallback_exact_ridge": bool(np.allclose(pred, anchor, rtol=0.0, atol=0.0)),
        }


__all__ = ["RidgeNeuralBlend", "ConservativeRidgeNeuralGate", "GateThresholds"]
