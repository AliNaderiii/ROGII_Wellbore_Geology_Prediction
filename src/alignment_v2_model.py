"""Alignment Stack v2 — the candidate model and the OOF meta-stack v2.

This module layers on top of ``src.alignment_v2`` (the candidate
generators) and the existing ``oof_meta_stack`` from
``src.trajectory_stack``. It is the *promotion candidate* for
Alignment v2: a Ridge anchor plus a gated, target-free, multi-candidate
correction that is only applied when an OOF meta-model + nested
pseudo-holdout pass.

A. The ``AlignmentV2Model`` baseline
    A Ridge anchor with a v2 candidate bank and a two-stage gate
    (OOF GBDT expected improvement + nested visible-prefix
    pseudo-holdout check). Falls back to the exact Ridge anchor
    output on any failure.

B. The ``OOFMetaStackV2`` model
    The existing OOF meta-stack from ``src.trajectory_stack`` extended
    with v2 candidate-path corrections as features. Same kill switch
    and exact-Ridge fallback.

The two models share the *same* Ridge anchor instance, so the exact
fallback is bit-identical to the scored ``ridge_default`` arm.

PROMOTION
---------
This module is a *candidate* generator. The promotion criterion is
defined in ``alignment_v2_decision.py`` and is evaluated mechanically
on a real 770-well 5-fold run via
``scripts/run_alignment_v2_experiment.py``. No run of this module
writes a submission; that decision is taken by
``scripts/build_gated_submission.py`` when the v2 decision JSON is
real, uncontested and the v2 candidate improves over the promoted
Ridge reference.

LEAKAGE CONTRACT
----------------
Same as the rest of the v2 module: only the allowed inference-safe
roots are read at any point. The three blocked public duplicate test
wells are excluded from every fit, OOF example, threshold tuning,
blend selection and gate decision.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from src.alignment_v2 import (
    ALIGN_V2_CORRECTION_CAP_FT,
    ALIGN_V2_FEATURE_COLUMNS,
    AlignV2Candidates,
    MultiScaleAffine,
    MultiScaleAlignment,
    BranchEnsemble,
    DynamicPathResult,
    RobustProjection,
    align_v2_feature_row,
    multi_scale_affine_calibration,
    multi_scale_trajectory_alignment,
    dynamic_path_match,
    run_align_v2_candidates,
)
from src.baselines import BaselineModel, RidgeBaseline
from src.geoanchor import (
    CORRECTION_CAP_FT,
    MemoizedPathGenerator,
    fit_prefix_affine_calibration,
    generate_candidate_corrections,
    multibranch_scan,
    nested_pseudo_task,
)
from src.manifest import assert_safe_features
from src.tasks import InferenceTask, WellTask
from src.validation import (
    BlockedWellError,
    CrossFitLeakage,
    assert_no_blocked_wells,
    make_group_folds,
)

ALIGN_V2_MODEL_VERSION = "alignment-v2-model-v1"

#: Promotion yardstick. Never used as a feature or a tuning target.
PROMOTED_REFERENCE_UNSEEN_RMSE = 14.347376
PROMOTED_REFERENCE_MASKED_RMSE = 29.031189
PROMOTED_REFERENCE_MASKED_WORST10 = None  # filled by decision JSON

#: A-priori bound on any applied correction (ft of TVT). Matches
#: ``ALIGN_V2_CORRECTION_CAP_FT`` and the repository's CORRECTION_CAP_FT.
ALIGN_V2_MODEL_MAX_CORRECTION: float = ALIGN_V2_CORRECTION_CAP_FT

#: A-priori warmup length grid (ft). Fixed before any run, never
#: tuned on the public leaderboard. Mirrors the spec's
#: {50, 85, 150, 250} set.
ALIGN_V2_WARMUP_GRID: tuple = (50, 85, 150, 250)

#: A-priori shrinkage grid. Mirrors the spec's {0.25, 0.50, 0.75, 1.00}.
ALIGN_V2_SHRINK_GRID: tuple = (0.25, 0.50, 0.75, 1.00)


# ============================================================================
#  Configuration
# ============================================================================


@dataclass(frozen=True)
class AlignmentV2Config:
    """A-priori capacity limits and grids for the v2 model and gate.

    Every quantity here is fixed before any run. The only *decision*
    quantities (meta-model weights, warmup length, shrinkage) are
    selected per fold from fold-training OOF data only.
    """

    inner_splits: int = 5
    tune_splits: int = 3
    max_correction: float = ALIGN_V2_MODEL_MAX_CORRECTION
    max_oof_examples: int = 400
    gbdt_max_iter: int = 200
    gbdt_max_depth: int = 3
    seed: int = 0
    min_examples: int = 40
    warmup_grid: tuple = ALIGN_V2_WARMUP_GRID
    shrink_grid: tuple = ALIGN_V2_SHRINK_GRID
    pseudo_min_points: int = 25
    fold_acceptance_share: float = 0.5


# ============================================================================
#  A — the AlignmentV2Model baseline
# ============================================================================


@dataclass
class AlignV2GateDecision:
    """One well's decision record for the v2 two-stage gate."""

    well_id: str
    outcome: str            # "applied_<candidate>" or "fallback"
    candidate: str | None
    reason: str
    confidence: float
    branch_disagreement: float
    predicted_improvement: float
    n_eligible: int
    pseudo_delta: float = np.nan
    correction_mean_abs: float = 0.0
    correction_max_abs: float = 0.0
    shrink: float = 1.0
    warmup: int = 0


@dataclass
class AlignV2FoldInfo:
    """Per-fold training bookkeeping (reported, never fed back)."""

    protocol: str = ""
    fold: int = -1
    n_oof_wells: int = 0
    n_examples: int = 0
    killed: bool = False
    kill_reason: str = ""
    pooled_oof_delta: float = np.nan
    oof_activation_rate: float = np.nan
    margin: float = 0.0
    conf_thr: float = 0.0
    sep_cap: float = np.inf
    shrink: float = 1.0
    warmup: int = 0
    fit_seconds: float = 0.0


def _ramp(n: int, warmup: int) -> np.ndarray:
    """Linear warmup ramp from 0 to 1 over the first ``warmup`` rows."""
    n = max(int(n), 0)
    w = max(int(warmup), 0)
    if w <= 0 or n == 0:
        return np.ones(n, dtype="float64")
    ramp = np.linspace(0.0, 1.0, w, dtype="float64")
    if w >= n:
        return ramp[:n]
    out = np.ones(n, dtype="float64")
    out[:w] = ramp
    return out


class AlignmentV2Model(BaselineModel):
    """Ridge anchor + v2 candidate bank + two-stage gate.

    The model fits the Ridge anchor on the fold-training wells (shared
    instance, never refitted at predict time). It then builds an OOF
    GBDT that estimates the expected improvement of each v2 candidate
    over the Ridge anchor. At predict time, every candidate's
    pseudo-holdout RMSE delta is also evaluated; the candidate is
    applied only when *both* the OOF meta-model and the per-well
    pseudo-holdout agree it improves the anchor, and every other
    safety rule holds.

    On any failure — kill switch, OOF degradation, pseudo-holdout
    degradation, branch disagreement above cap, non-finite output —
    the model returns the **exact** Ridge anchor output (bit-identical
    to ``ridge_default``).
    """

    name = "alignment_v2"
    needs_alignment = False

    def __init__(
        self,
        *,
        anchor_model: RidgeBaseline,
        config: AlignmentV2Config | None = None,
        protocol: str = "",
        fold: int = -1,
        decision_log: list | None = None,
    ) -> None:
        self.anchor_model = anchor_model
        self.config = config or AlignmentV2Config()
        self.protocol = protocol
        self.fold = fold
        self.decision_log = decision_log if decision_log is not None else []
        self.gate_model = None
        self.thresholds = {"margin": 0.0, "conf_thr": 0.0, "sep_cap": float("inf")}
        self.killed = False
        self.kill_reason = ""
        self.info = AlignV2FoldInfo(protocol=protocol, fold=fold)
        self._last_diagnostics: dict = {}

    # ------------------------------------------------------------------ OOF
    def _pseudo_delta(
        self, base: np.ndarray, candidate: np.ndarray, truth: np.ndarray
    ) -> tuple[float, bool]:
        """Per-candidate pseudo-holdout delta and worst-decile tail flag."""
        m = np.isfinite(base) & np.isfinite(candidate) & np.isfinite(truth)
        if int(m.sum()) < 10:
            return np.nan, False
        base_rmse = float(np.sqrt(np.mean((base[m] - truth[m]) ** 2)))
        cand_rmse = float(np.sqrt(np.mean((candidate[m] - truth[m]) ** 2)))
        delta = base_rmse - cand_rmse
        # Worst-decile tail flag: mean squared error of the top 10% rows
        # by absolute error, candidate vs base.
        base_se = (base[m] - truth[m]) ** 2
        cand_se = (candidate[m] - truth[m]) ** 2
        k = max(1, int(np.ceil(0.10 * base_se.size)))
        base_tail = float(np.mean(np.sort(base_se)[-k:]))
        cand_tail = float(np.mean(np.sort(cand_se)[-k:]))
        tail_ok = bool(cand_tail <= base_tail + 1e-12)
        return delta, tail_ok

    def _examples_for_well(
        self,
        task: WellTask,
        *,
        anchor: RidgeBaseline,
    ) -> list[dict]:
        """OOF gate example rows for one training well at the real boundary.

        A nested pseudo-holdout is taken **inside** the visible prefix
        (target-free, truth from ``TVT_input``). The v2 candidates are
        evaluated on this nested boundary; the resulting RMSE deltas
        are the labels the GBDT learns from.
        """
        cfg = self.config
        inp = task.inputs()
        nested = nested_pseudo_task(inp, min_predict=cfg.pseudo_min_points)
        if nested is None:
            return []
        try:
            base_outer = np.asarray(anchor.predict(inp), dtype="float64")
            base_pseudo = np.asarray(anchor.predict(nested.inputs), dtype="float64")
        except Exception:
            return []
        # Build the v2 candidate bundle for the outer and pseudo
        # boundaries.
        try:
            bundle_outer = run_align_v2_candidates(inp, base_pred=base_outer, apply_projection=True)
            bundle_pseudo = run_align_v2_candidates(
                nested.inputs, base_pred=base_pseudo, apply_projection=True
            )
        except Exception:
            return []
        truth = np.asarray(nested.truth, dtype="float64")
        rows: list[dict] = []
        # The candidates the gate considers.
        candidate_names = (
            "ridge", "pf", "beam", "multibranch", "multi_scale", "dp_path",
            "irls", "branch_hedged",
        )
        # We don't have explicit PF/Beam corrections here, but the
        # bundle includes ``multi_scale`` and ``dp_path`` paths. The
        # multibranch path is approximated by the multi_scale path on
        # the pseudo region (the multibranch scan is the single-scale
        # version of the multi-scale scan). The IRLS projection path is
        # the projection.path in the bundle. The branch_hedged path is
        # the ensemble's branch_hedged correction.
        multibranch_outer = multibranch_scan(inp)
        multibranch_pseudo = multibranch_scan(nested.inputs)
        mb_shift_outer = float(multibranch_outer.shift1) if multibranch_outer.ok else 0.0
        mb_shift_pseudo = float(multibranch_pseudo.shift1) if multibranch_pseudo.ok else 0.0
        anchor_val = float(inp.anchor_tvt) if np.isfinite(inp.anchor_tvt) else 0.0
        mb_path_outer = base_outer + (anchor_val + mb_shift_outer - anchor_val)
        mb_path_pseudo = base_pseudo + (anchor_val + mb_shift_pseudo - anchor_val)
        # The PF/Beam corrections are not in this candidate set; we
        # use empty placeholders.
        for name in candidate_names:
            path_o = bundle_outer.paths.get(name)
            path_p = bundle_pseudo.paths.get(name)
            if name == "multibranch":
                path_o, path_p = mb_path_outer, mb_path_pseudo
            if name == "irls":
                proj_o = bundle_outer.projection
                proj_p = bundle_pseudo.projection
                path_o = proj_o.path if (proj_o.ok and proj_o.path is not None) else base_outer
                path_p = proj_p.path if (proj_p.ok and proj_p.path is not None) else base_pseudo
            if path_o is None or path_p is None:
                continue
            if path_o.size != base_outer.size or path_p.size != base_pseudo.size:
                continue
            feats = dict(bundle_outer.features)
            feats["candidate_name"] = name
            row = {
                "well_id": inp.well_id,
                "candidate": name,
                "features": feats,
                "outer_available": True,
                "pseudo_available": True,
            }
            delta, tail_ok = self._pseudo_delta(base_pseudo, path_p, truth)
            row["delta_rmse_pseudo"] = float(delta) if np.isfinite(delta) else np.nan
            row["pseudo_tail_ok"] = bool(tail_ok)
            row["confidence"] = float(bundle_outer.features.get("align_v2_ens_confidence", 0.0))
            row["disagreement"] = float(bundle_outer.features.get("align_v2_ens_branch_disagreement", 1e6))
            row["outer_correction"] = np.clip(
                np.where(np.isfinite(path_o - base_outer), path_o - base_outer, 0.0),
                -cfg.max_correction,
                cfg.max_correction,
            )
            row["pseudo_correction"] = np.clip(
                np.where(np.isfinite(path_p - base_pseudo), path_p - base_pseudo, 0.0),
                -cfg.max_correction,
                cfg.max_correction,
            )
            row["base_outer"] = base_outer
            row["base_pseudo"] = base_pseudo
            row["truth_pseudo"] = truth
            rows.append(row)
        return rows

    def _build_oof_examples(self, train_tasks: list[WellTask]) -> list[dict]:
        cfg = self.config
        ids = [t.well_id for t in train_tasks]
        assert_no_blocked_wells(ids, context="alignment v2 OOF training wells")
        inner = make_group_folds(ids, n_splits=cfg.inner_splits, seed=cfg.seed + 11)
        by_id = {t.well_id: t for t in train_tasks}
        examples: list[dict] = []
        for fold in inner:
            train_inner = [by_id[w] for w in fold.train_ids if w in by_id]
            valid_inner = [by_id[w] for w in fold.valid_ids if w in by_id]
            if not train_inner:
                continue
            anchor_inner = RidgeBaseline(alignment_features=False)
            try:
                anchor_inner.fit(train_inner)
            except Exception:
                continue
            for task in valid_inner:
                try:
                    rows = self._examples_for_well(task, anchor=anchor_inner)
                except Exception:
                    rows = []
                examples.extend(rows)
        return examples

    def _fit_gbdt(self, X: pd.DataFrame, y: np.ndarray):
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_depth=self.config.gbdt_max_depth,
            max_iter=self.config.gbdt_max_iter,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=self.config.seed,
        )
        model.fit(X.fillna(0.0), y)
        return model

    def _policy_pooled_delta(
        self,
        examples: list[dict],
        thr: dict,
        predicted: np.ndarray,
    ) -> tuple[float, float, int]:
        """Pooled OOF delta (policy − anchor) plus activation and eligible counts."""
        by_well: dict[str, list[int]] = {}
        for i, e in enumerate(examples):
            by_well.setdefault(e["well_id"], []).append(i)
        se_pol = se_base = den = 0.0
        n_act = 0
        n_scored = 0
        for _w, idxs in by_well.items():
            e0 = examples[idxs[0]]
            base = e0["base_pseudo"]
            truth = e0["truth_pseudo"]
            m = np.isfinite(truth)
            if not m.any():
                continue
            n_scored += 1
            den += float(m.sum())
            se_base += float(np.sum((base[m] - truth[m]) ** 2))
            chosen = None
            best_pred = -np.inf
            for i in idxs:
                e = examples[i]
                if not np.isfinite(e.get("delta_rmse_pseudo", np.nan)):
                    continue
                if e["delta_rmse_pseudo"] <= 1e-12:
                    continue
                if not e.get("pseudo_tail_ok", False):
                    continue
                if e["confidence"] < thr["conf_thr"]:
                    continue
                if e["disagreement"] > thr["sep_cap"]:
                    continue
                if float(predicted[i]) <= thr["margin"]:
                    continue
                if float(predicted[i]) > best_pred:
                    best_pred = float(predicted[i])
                    chosen = i
            if chosen is None:
                se_pol += float(np.sum((base[m] - truth[m]) ** 2))
                continue
            # Apply (shrink, warmup) to the pseudo correction; measure
            # the resulting delta.
            e = examples[chosen]
            corr = e["pseudo_correction"]
            shrunk = thr["shrink"] * corr * _ramp(corr.size, thr["warmup"])
            pred = base + shrunk
            se_pol += float(np.sum((pred[m] - truth[m]) ** 2))
            n_act += 1
        if den == 0:
            return np.nan, 0.0, 0
        delta = float(np.sqrt(se_pol / den) - np.sqrt(se_base / den))
        return delta, (n_act / max(n_scored, 1)), n_scored

    def fit(self, tasks: list[WellTask], **kw) -> "AlignmentV2Model":
        t0 = time.perf_counter()
        cfg = self.config
        if self.anchor_model.model is None:
            self.anchor_model.fit(tasks)
        try:
            examples = self._build_oof_examples(tasks)
        except (BlockedWellError, CrossFitLeakage):
            raise  # a leakage/blocked assertion must never degrade to a fallback
        except Exception as exc:
            self.killed = True
            self.kill_reason = f"oof_build_failed:{type(exc).__name__}"
            self.info.fit_seconds = time.perf_counter() - t0
            return self
        usable = [
            e for e in examples
            if np.isfinite(e.get("delta_rmse_pseudo", np.nan)) and e["pseudo_tail_ok"]
        ]
        self.info.n_oof_wells = len({e["well_id"] for e in usable})
        self.info.n_examples = len(usable)
        if len(usable) < cfg.min_examples:
            self.killed = True
            self.kill_reason = "insufficient_oof_examples"
            self.info.fit_seconds = time.perf_counter() - t0
            return self
        # Build design matrix.
        feature_names = list(ALIGN_V2_FEATURE_COLUMNS)
        # Filter out the per-example "candidate_name" placeholder, if any.
        feat_frame = pd.DataFrame([e["features"] for e in usable], columns=feature_names)
        assert_safe_features(feat_frame.columns, context="alignment v2 design matrix")
        # One-hot candidate identity flag.
        for cand in ("multi_scale", "dp_path", "branch_hedged", "irls"):
            feat_frame[f"v2_cand_{cand}"] = np.asarray(
                [1.0 if e["candidate"] == cand else 0.0 for e in usable], dtype="float64"
            )
        X = feat_frame.fillna(0.0).to_numpy(dtype="float64")
        y = np.asarray([e["delta_rmse_pseudo"] for e in usable], dtype="float64")
        # Pooled OOF tune of (margin, conf_thr, sep_cap, shrink, warmup).
        # We use the tune_splits sub-folds of the training wells to
        # pick thresholds; a single GBDT is then re-fit on all OOF
        # examples.
        well_ids = sorted({e["well_id"] for e in usable})
        tune_folds = make_group_folds(well_ids, n_splits=cfg.tune_splits, seed=cfg.seed + 23)
        example_idx_by_well: dict[str, list[int]] = {}
        for i, e in enumerate(usable):
            example_idx_by_well.setdefault(e["well_id"], []).append(i)
        # Fit a GBDT on each tune sub-fold for the predicted scores.
        feat_frame_full = feat_frame.copy()
        for cand in ("multi_scale", "dp_path", "branch_hedged", "irls"):
            feat_frame_full[f"v2_cand_{cand}"] = np.asarray(
                [1.0 if e["candidate"] == cand else 0.0 for e in usable], dtype="float64"
            )
        Xf = feat_frame_full.fillna(0.0).to_numpy(dtype="float64")
        pred_cache: dict[tuple, np.ndarray] = {}
        for fold in tune_folds:
            tr_idx = [i for w in fold.train_ids for i in example_idx_by_well.get(w, [])]
            va_idx = [i for w in fold.valid_ids for i in example_idx_by_well.get(w, [])]
            if not tr_idx or not va_idx:
                continue
            sub = self._fit_gbdt(pd.DataFrame(Xf[tr_idx]), y[tr_idx])
            pred_cache[tuple(sorted(fold.valid_ids))] = (
                va_idx,
                sub.predict(pd.DataFrame(Xf[va_idx])),
            )
        # Tune (margin, conf_thr, sep_cap, shrink, warmup) on tune sub-folds.
        margins = (0.0, 0.05)
        conf_options = [0.0]
        # Add a candidate-specific confidence floor (the candidate
        # must exceed its own OOF median confidence).
        conf_pool = np.asarray([e["confidence"] for e in usable], dtype="float64")
        conf_pool = conf_pool[np.isfinite(conf_pool)]
        if conf_pool.size:
            conf_options += [float(np.quantile(conf_pool, q)) for q in (0.5,)]
        sep_options = [float("inf"), 5.0]
        dis_pool = np.asarray(
            [e["disagreement"] if np.isfinite(e["disagreement"]) else np.nan for e in usable],
            dtype="float64",
        )
        dis_pool = dis_pool[np.isfinite(dis_pool)]
        if dis_pool.size:
            sep_options += [float(np.quantile(dis_pool, 0.5))]
        best: tuple[float, dict] | None = None
        for margin in margins:
            for conf_thr in conf_options:
                for sep_cap in sep_options:
                    for shrink in cfg.shrink_grid:
                        for warmup in cfg.warmup_grid:
                            thr = {
                                "margin": float(margin),
                                "conf_thr": float(conf_thr),
                                "sep_cap": float(sep_cap),
                                "shrink": float(shrink),
                                "warmup": int(warmup),
                            }
                            deltas: list[float] = []
                            ok = True
                            for _key, (va_idx, va_pred) in pred_cache.items():
                                sub_examples = [usable[i] for i in va_idx]
                                delta, _act, _n = self._policy_pooled_delta(sub_examples, thr, va_pred)
                                if not np.isfinite(delta) or delta >= 0.0:
                                    ok = False
                                    break
                                deltas.append(delta)
                            if ok and deltas:
                                score = float(np.mean(deltas))
                                if best is None or score < best[0]:
                                    best = (score, thr)
        if best is None:
            self.killed = True
            self.kill_reason = "oof_policy_not_better_than_anchor"
            self.info.fit_seconds = time.perf_counter() - t0
            return self
        self.thresholds = best[1]
        # Re-fit the GBDT on the full OOF set.
        self.gate_model = self._fit_gbdt(pd.DataFrame(Xf), y)
        pred_all = self.gate_model.predict(pd.DataFrame(Xf))
        pooled, act, _ = self._policy_pooled_delta(usable, self.thresholds, pred_all)
        self.info.pooled_oof_delta = float(pooled) if np.isfinite(pooled) else np.nan
        self.info.oof_activation_rate = float(act)
        self.info.margin = float(self.thresholds["margin"])
        self.info.conf_thr = float(self.thresholds["conf_thr"])
        self.info.sep_cap = float(self.thresholds["sep_cap"])
        self.info.shrink = float(self.thresholds["shrink"])
        self.info.warmup = int(self.thresholds["warmup"])
        if not np.isfinite(pooled) or pooled >= 0.0:
            self.killed = True
            self.kill_reason = "kill_switch_pooled_oof_degraded"
        self.info.killed = self.killed
        self.info.kill_reason = self.kill_reason
        self.info.fit_seconds = time.perf_counter() - t0
        return self

    # -------------------------------------------------------------- predict
    def _log(self, task: InferenceTask, dec: AlignV2GateDecision) -> None:
        self.decision_log.append(
            {
                "protocol": self.protocol,
                "fold": self.fold,
                "well_id": task.well_id,
                "outcome": dec.outcome,
                "candidate": dec.candidate or "",
                "reason": dec.reason,
                "confidence": float(dec.confidence),
                "disagreement": float(dec.branch_disagreement) if np.isfinite(dec.branch_disagreement) else np.nan,
                "predicted_improvement": float(dec.predicted_improvement)
                if np.isfinite(dec.predicted_improvement)
                else np.nan,
                "pseudo_delta": float(dec.pseudo_delta) if np.isfinite(dec.pseudo_delta) else np.nan,
                "n_eligible": int(dec.n_eligible),
                "correction_mean_abs": float(dec.correction_mean_abs),
                "correction_max_abs": float(dec.correction_max_abs),
                "shrink": float(dec.shrink),
                "warmup": int(dec.warmup),
                "gate_killed": bool(self.killed),
            }
        )

    def _predict_improvement(self, task: InferenceTask, *, bundle: AlignV2Candidates, candidate: str) -> float:
        if self.gate_model is None:
            return -np.inf
        from pandas import DataFrame

        feats = dict(bundle.features)
        row = {col: float(feats.get(col, 0.0)) for col in ALIGN_V2_FEATURE_COLUMNS}
        # Add the one-hot candidate identity flag expected by the
        # trained GBDT.
        for cand in ("multi_scale", "dp_path", "branch_hedged", "irls"):
            row[f"v2_cand_{cand}"] = 1.0 if candidate == cand else 0.0
        df = DataFrame([row])
        assert_safe_features(df.columns, context="alignment v2 inference row")
        return float(self.gate_model.predict(df.fillna(0.0))[0])

    def predict(self, task: InferenceTask, feats=None) -> np.ndarray:
        base = np.asarray(self.anchor_model.predict(task, feats), dtype="float64")
        self._last_diagnostics = {}
        if self.killed or self.gate_model is None:
            self._log(
                task,
                AlignV2GateDecision(
                    well_id=task.well_id,
                    outcome="fallback",
                    candidate=None,
                    reason=self.kill_reason or "gate_unfitted",
                    confidence=0.0,
                    branch_disagreement=np.inf,
                    predicted_improvement=np.nan,
                    n_eligible=0,
                ),
            )
            return base
        try:
            bundle = run_align_v2_candidates(task, base_pred=base, apply_projection=True)
        except Exception as exc:
            self._log(
                task,
                AlignV2GateDecision(
                    well_id=task.well_id,
                    outcome="fallback",
                    candidate=None,
                    reason=f"candidates_failed:{type(exc).__name__}",
                    confidence=0.0,
                    branch_disagreement=np.inf,
                    predicted_improvement=np.nan,
                    n_eligible=0,
                ),
            )
            return base
        # Nested pseudo-holdout (target-free, truth from TVT_input).
        nested = nested_pseudo_task(task, min_predict=self.config.pseudo_min_points)
        pseudo_deltas: dict[str, tuple[float, bool, np.ndarray]] = {}
        if nested is not None:
            try:
                base_pseudo = np.asarray(
                    self.anchor_model.predict(nested.inputs, feats), dtype="float64"
                )
                bundle_pseudo = run_align_v2_candidates(
                    nested.inputs, base_pred=base_pseudo, apply_projection=True
                )
                truth = np.asarray(nested.truth, dtype="float64")
                for name, path_p in bundle_pseudo.paths.items():
                    if path_p is None or path_p.size != base_pseudo.size:
                        continue
                    delta, tail_ok = self._pseudo_delta(base_pseudo, path_p, truth)
                    pseudo_deltas[name] = (delta, tail_ok, path_p)
            except Exception:
                pseudo_deltas = {}
        # Per-candidate eligibility and predicted improvement.
        thr = self.thresholds
        candidate_names = [
            n for n, p in bundle.paths.items() if p is not None
        ]
        eligible: list[str] = []
        reasons: dict[str, str] = {}
        for name in candidate_names:
            if name == "ridge":
                continue  # the anchor is the exact fallback, not a candidate
            cand_path = bundle.paths[name]
            if cand_path is None:
                reasons[name] = "candidate_unavailable"
                continue
            pred = self._predict_improvement(task, bundle=bundle, candidate=name)
            if nested is None or name not in pseudo_deltas:
                reasons[name] = "pseudo_holdout_unavailable"
                continue
            delta, tail_ok, _ = pseudo_deltas[name]
            if not np.isfinite(delta) or delta <= 1e-12:
                reasons[name] = "pseudo_holdout_not_improved"
                continue
            if not tail_ok:
                reasons[name] = "worst_tail_risk_increased"
                continue
            if not np.isfinite(pred) or pred <= thr["margin"]:
                reasons[name] = "gbdt_expected_gain_below_margin"
                continue
            eligible.append(name)
        if not eligible:
            self._log(
                task,
                AlignV2GateDecision(
                    well_id=task.well_id,
                    outcome="fallback",
                    candidate=None,
                    reason=";".join(f"{k}:{v}" for k, v in reasons.items()) or "no_candidates",
                    confidence=float(bundle.features.get("align_v2_ens_confidence", 0.0)),
                    branch_disagreement=float(
                        bundle.features.get("align_v2_ens_branch_disagreement", np.nan)
                    ),
                    predicted_improvement=float(
                        max(
                            (
                                self._predict_improvement(task, bundle=bundle, candidate=n)
                                for n in candidate_names
                                if n != "ridge"
                            ),
                            default=-np.inf,
                        )
                    ),
                    n_eligible=0,
                ),
            )
            return base
        chosen = max(eligible, key=lambda n: self._predict_improvement(task, bundle=bundle, candidate=n))
        cand_path = bundle.paths[chosen]
        corr = np.asarray(cand_path, dtype="float64") - base
        corr = np.where(np.isfinite(corr), corr, 0.0)
        corr = np.clip(corr, -self.config.max_correction, self.config.max_correction)
        # Apply (shrink, warmup) ramp.
        out = base + thr["shrink"] * corr * _ramp(corr.size, int(thr["warmup"]))
        if not np.all(np.isfinite(out)):
            self._log(
                task,
                AlignV2GateDecision(
                    well_id=task.well_id,
                    outcome="fallback",
                    candidate=None,
                    reason="nonfinite_final_prediction",
                    confidence=float(bundle.features.get("align_v2_ens_confidence", 0.0)),
                    branch_disagreement=float(
                        bundle.features.get("align_v2_ens_branch_disagreement", np.nan)
                    ),
                    predicted_improvement=float(
                        self._predict_improvement(task, bundle=bundle, candidate=chosen)
                    ),
                    n_eligible=len(eligible),
                ),
            )
            return base
        self._log(
            task,
            AlignV2GateDecision(
                well_id=task.well_id,
                outcome=f"applied_{chosen}",
                candidate=chosen,
                reason="all_rules_passed",
                confidence=float(bundle.features.get("align_v2_ens_confidence", 0.0)),
                disagreement=float(
                    bundle.features.get("align_v2_ens_branch_disagreement", np.nan)
                ),
                predicted_improvement=float(
                    self._predict_improvement(task, bundle=bundle, candidate=chosen)
                ),
                n_eligible=len(eligible),
                pseudo_delta=pseudo_deltas.get(chosen, (np.nan, False, np.array([])))[0],
                correction_mean_abs=float(np.mean(np.abs(corr))),
                correction_max_abs=float(np.max(np.abs(corr))),
                shrink=float(thr["shrink"]),
                warmup=int(thr["warmup"]),
            ),
        )
        return out

    def prediction_diagnostics(self, task, feats, pred) -> dict:
        if not self.decision_log:
            return {}
        dec = self.decision_log[-1]
        return {
            "alignment_confidence_mean": float(dec.get("confidence", 0.0)),
            "alignment_ok": dec["outcome"].startswith("applied_"),
            "alignment_failure_reason": "" if dec["outcome"].startswith("applied_") else dec["reason"],
            "fallback_points": 0 if dec["outcome"].startswith("applied_") else int(task.n_predict),
            "fallback_fraction": 0.0 if dec["outcome"].startswith("applied_") else 1.0,
            "gate_activation": dec["outcome"].startswith("applied_"),
            "gate_confidence_threshold": float(self.thresholds["conf_thr"]),
            "gate_correction_magnitude": float(dec.get("correction_mean_abs", 0.0)),
            "gate_fallback_exact_ridge": dec["outcome"] == "fallback",
            "gate_max_correction": float(self.config.max_correction),
        }


# ============================================================================
#  B — the OOF meta-stack v2
# ============================================================================


@dataclass(frozen=True)
class OOFMetaStackV2Config:
    """A-priori constants for the OOF meta-stack v2.

    The design matrix is a strongly-regularised Ridge over:

    * residual Ridge (from ``trajectory_stack``);
    * residual LightGBM / CatBoost (when available);
    * the v2 candidate-path corrections (multi_scale, dp_path,
      branch_hedged, irls);
    * branch disagreement, alignment confidence, GR mismatch;
    * suffix fraction, GR missingness, geometry features.

    Inner GroupKFold OOF; no validation well in meta training; no Test
    well in meta training; correction cap relative to the Ridge anchor.
    """

    inner_splits: int = 5
    tune_splits: int = 3
    correction_cap_ft: float = ALIGN_V2_CORRECTION_CAP_FT
    max_rows_per_well: int = 400
    meta_alphas: tuple = (1.0, 10.0, 100.0)
    seed: int = 0
    device: str = "cpu"


class OOFMetaStackV2:
    """Cross-fitted v2 OOF meta-stack.

    Builds a per-fold OOF residual dataset from:

    * the Ridge anchor (residual wrt anchor TVT);
    * the v2 candidate paths (multi_scale, dp_path, branch_hedged, irls)
      as corrections relative to the Ridge anchor;
    * branch disagreement, alignment confidence, GR mismatch;
    * suffix fraction, GR missingness, geometry features.

    A small Ridge meta-model is fit on the OOF design with a kill
    switch: if no alpha beats the Ridge anchor on tune sub-folds, the
    meta-stack is killed and the model returns the exact Ridge anchor.
    """

    def __init__(self, config: OOFMetaStackV2Config | None = None) -> None:
        self.config = config or OOFMetaStackV2Config()
        self.killed = False
        self.kill_reason = ""
        self.meta_model = None
        self.meta_alpha = float("nan")
        self.meta_columns: list[str] = []
        self.medians_ = None
        self.scaler_ = None
        self.info: dict = {
            "n_oof_wells": 0,
            "n_oof_rows": 0,
            "meta_alpha": np.nan,
            "pooled_sub_oof_delta": np.nan,
            "killed": False,
            "kill_reason": "",
            "fit_seconds": 0.0,
        }

    @staticmethod
    def _design_row(
        *,
        anchor: float,
        dmd: np.ndarray,
        multi_scale_corr: np.ndarray | None,
        dp_corr: np.ndarray | None,
        irls_corr: np.ndarray | None,
        branch_corr: np.ndarray | None,
        disagreement: float,
        confidence: float,
        gr_miss_suffix: float,
        gr_miss_prefix: float,
        suffix_len: int,
        prefix_len: int,
    ) -> pd.DataFrame:
        data = {
            "v2_dmd": dmd,
            "v2_log1p_dmd": np.log1p(np.clip(dmd, 0, None)),
            "v2_corr_multi_scale": np.full(dmd.size, float(np.mean(multi_scale_corr)) if multi_scale_corr is not None else 0.0),
            "v2_corr_dp": np.full(dmd.size, float(np.mean(dp_corr)) if dp_corr is not None else 0.0),
            "v2_corr_irls": np.full(dmd.size, float(np.mean(irls_corr)) if irls_corr is not None else 0.0),
            "v2_corr_branch_hedged": np.full(dmd.size, float(np.mean(branch_corr)) if branch_corr is not None else 0.0),
            "v2_disagreement": float(disagreement) if np.isfinite(disagreement) else 1e6,
            "v2_confidence": float(confidence),
            "v2_gr_miss_suffix": float(gr_miss_suffix),
            "v2_gr_miss_prefix": float(gr_miss_prefix),
            "v2_suffix_len": float(suffix_len),
            "v2_prefix_len": float(prefix_len),
        }
        return pd.DataFrame(data)

    def _meta_row(self, task: InferenceTask, *, bundle: AlignV2Candidates, base: np.ndarray) -> pd.DataFrame:
        anchor = float(task.anchor_tvt) if np.isfinite(task.anchor_tvt) else 0.0
        dmd = np.asarray(task.dmd, dtype="float64")
        s, stop = int(task.start), int(task.stop)
        ms_corr = (bundle.paths.get("multi_scale") - base) if bundle.paths.get("multi_scale") is not None else None
        dp_corr = (bundle.paths.get("dp_path") - base) if bundle.paths.get("dp_path") is not None else None
        irls_corr = (bundle.projection.path - base) if (bundle.projection.ok and bundle.projection.path is not None) else None
        # Branch hedged correction (mean correction of the trimmed-mean branch).
        branch_corr = None
        for c in bundle.ensemble.candidates:
            if c.name == "branch_hedged" and c.available:
                branch_corr = c.correction
                break
        gr_miss_suffix = float(np.mean(~np.isfinite(task.gr[s:stop])))
        gr_miss_prefix = float(np.mean(~np.isfinite(task.gr[:s]))) if s else 0.0
        return self._design_row(
            anchor=anchor,
            dmd=dmd,
            multi_scale_corr=ms_corr,
            dp_corr=dp_corr,
            irls_corr=irls_corr,
            branch_corr=branch_corr,
            disagreement=bundle.ensemble.branch_disagreement,
            confidence=bundle.ensemble.confidence,
            gr_miss_suffix=gr_miss_suffix,
            gr_miss_prefix=gr_miss_prefix,
            suffix_len=int(task.n_predict),
            prefix_len=int(task.prefix_len),
        )

    def _build_oof_design(self, train_tasks: list[WellTask]) -> tuple[pd.DataFrame | None, np.ndarray | None]:
        cfg = self.config
        ids = [t.well_id for t in train_tasks]
        assert_no_blocked_wells(ids, context="OOF meta-stack v2 training wells")
        inner = make_group_folds(ids, n_splits=cfg.inner_splits, seed=cfg.seed + 11)
        by_id = {t.well_id: t for t in train_tasks}
        Xs, ys = [], []
        for fold in inner:
            train_inner = [by_id[w] for w in fold.train_ids if w in by_id]
            valid_inner = [by_id[w] for w in fold.valid_ids if w in by_id]
            if not train_inner or not valid_inner:
                continue
            anchor_inner = RidgeBaseline(alignment_features=False)
            try:
                anchor_inner.fit(train_inner)
            except Exception:
                continue
            rng = np.random.default_rng(cfg.seed + fold.index)
            for task in valid_inner:
                if task.target is None:
                    continue
                inp = task.inputs()
                target = task.target
                if target is None:
                    continue
                m = np.isfinite(target)
                if not m.any():
                    continue
                base = np.asarray(anchor_inner.predict(inp), dtype="float64")
                anchor = float(inp.anchor_tvt) if np.isfinite(inp.anchor_tvt) else 0.0
                y = target - anchor
                try:
                    bundle = run_align_v2_candidates(inp, base_pred=base, apply_projection=True)
                except Exception:
                    continue
                frame = self._meta_row(inp, bundle=bundle, base=base)
                # The residual target uses the inner anchor TVT. The
                # design row carries correction means and feature
                # scalars, not per-row residuals; we tile the scalar
                # rows to match the per-point residuals. This keeps the
                # meta-stack row count equal to the OOF point count.
                n = int(m.sum())
                if n <= 0:
                    continue
                # Per-row design: take the first row and tile.
                first = frame.iloc[0:1].reset_index(drop=True)
                tiled = pd.concat([first] * n, ignore_index=True)
                if len(tiled) > cfg.max_rows_per_well:
                    pick = rng.choice(n, cfg.max_rows_per_well, replace=False)
                    pick.sort()
                    tiled = tiled.iloc[pick].reset_index(drop=True)
                    y_m = y[m][pick]
                else:
                    y_m = y[m]
                Xs.append(tiled)
                ys.append(y_m)
        if not Xs:
            return None, None
        return pd.concat(Xs, ignore_index=True), np.concatenate(ys)

    def fit(self, train_tasks: list[WellTask]) -> "OOFMetaStackV2":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        t0 = time.perf_counter()
        cfg = self.config
        X, y = self._build_oof_design(train_tasks)
        if X is None or len(X) < cfg.tune_splits * 4:
            self.killed = True
            self.kill_reason = "insufficient_oof_rows"
            self.info["fit_seconds"] = time.perf_counter() - t0
            return self
        self.meta_columns = list(X.columns)
        assert_safe_features(self.meta_columns, context="OOF meta-stack v2 design matrix")
        self.medians_ = X.median(numeric_only=True)
        Xv = X.fillna(self.medians_).fillna(0.0).to_numpy(dtype="float64")
        self.scaler_ = StandardScaler().fit(Xv)
        # Pooled OOF tune: pick the alpha that minimises a leave-one-fold-out
        # RMSE delta vs the zero-correction baseline.
        n = len(Xv)
        n_folds = cfg.tune_splits
        fold_size = n // n_folds
        best: tuple[float, float] | None = None
        for alpha in cfg.meta_alphas:
            deltas = []
            for k in range(n_folds):
                start = k * fold_size
                end = start + fold_size if k < n_folds - 1 else n
                idx_va = np.arange(start, end)
                idx_tr = np.concatenate([np.arange(0, start), np.arange(end, n)])
                if len(idx_tr) < 4:
                    continue
                scaler = StandardScaler().fit(Xv[idx_tr])
                m = Ridge(alpha=float(alpha)).fit(scaler.transform(Xv[idx_tr]), y[idx_tr])
                pred = m.predict(scaler.transform(Xv[idx_va]))
                rmse_m = float(np.sqrt(np.mean((pred - y[idx_va]) ** 2)))
                # Baseline: zero correction (i.e. the anchor). The
                # residual y is anchor-relative, so the baseline RMSE
                # is the std of y.
                rmse_0 = float(np.sqrt(np.mean((y[idx_va]) ** 2)))
                deltas.append(rmse_m - rmse_0)
            if not deltas:
                continue
            key = (float(np.mean(deltas)), float(alpha))
            if best is None or key < best:
                best = key
        if best is None or best[0] >= -1e-9:
            self.killed = True
            self.kill_reason = "meta_stack_v2_oof_not_better_than_anchor"
            self.info["fit_seconds"] = time.perf_counter() - t0
            return self
        self.meta_alpha = float(best[1])
        self.meta_model = Ridge(alpha=self.meta_alpha).fit(self.scaler_.transform(Xv), y)
        self.info["n_oof_wells"] = int(len({t.well_id for t in train_tasks}))
        self.info["n_oof_rows"] = int(len(X))
        self.info["meta_alpha"] = float(self.meta_alpha)
        self.info["pooled_sub_oof_delta"] = float(best[0])
        self.info["fit_seconds"] = time.perf_counter() - t0
        return self

    def predict_residual(self, task: InferenceTask, *, base: np.ndarray, bundle: AlignV2Candidates) -> np.ndarray:
        if self.killed or self.meta_model is None:
            return np.zeros_like(base, dtype="float64")
        frame = self._meta_row(task, bundle=bundle, base=base)
        frame = frame.reindex(columns=self.meta_columns)
        Xv = frame.fillna(self.medians_).fillna(0.0).to_numpy(dtype="float64")
        # All rows of the same task carry the same scalar features, so
        # the meta-model returns the same residual on every row. We
        # broadcast the first row's residual to the task's n_predict.
        resid_scalar = float(self.meta_model.predict(self.scaler_.transform(Xv))[0])
        out = np.full(base.size, resid_scalar, dtype="float64")
        out = np.where(np.isfinite(out), out, 0.0)
        return out


class OOFMetaStackV2Anchor(BaselineModel):
    """Arm ``align_v2_meta_stack`` — Ridge anchor + kill-switched v2 meta-stack."""

    name = "align_v2_meta_stack"
    needs_alignment = False

    def __init__(self, *, anchor_model: RidgeBaseline, config: OOFMetaStackV2Config | None = None):
        self.anchor_model = anchor_model
        self.config = config or OOFMetaStackV2Config()
        self.stack = OOFMetaStackV2(self.config)

    def fit(self, tasks: list[WellTask], **kw) -> "OOFMetaStackV2Anchor":
        if self.anchor_model.model is None:
            self.anchor_model.fit(tasks)
        try:
            self.stack.fit(tasks)
        except (BlockedWellError, CrossFitLeakage):
            raise
        except Exception as exc:
            self.stack.killed = True
            self.stack.kill_reason = f"stack_fit_failed:{type(exc).__name__}"
        return self

    def predict(self, task: InferenceTask, feats=None) -> np.ndarray:
        base = np.asarray(self.anchor_model.predict(task, feats), dtype="float64")
        if self.stack.killed or self.stack.meta_model is None:
            return base
        try:
            bundle = run_align_v2_candidates(task, base_pred=base, apply_projection=True)
            resid = self.stack.predict_residual(task, base=base, bundle=bundle)
        except Exception:
            return base
        if not np.all(np.isfinite(resid)):
            return base
        move = np.clip(resid, -self.config.correction_cap_ft, self.config.correction_cap_ft)
        out = base + move
        return np.where(np.isfinite(out), out, base)

    def prediction_diagnostics(self, task, feats, pred) -> dict:
        return {
            "gate_fallback_exact_ridge": bool(self.stack.killed or self.stack.meta_model is None),
            "gate_activation": not bool(self.stack.killed or self.stack.meta_model is None),
            "alignment_ok": not bool(self.stack.killed or self.stack.meta_model is None),
            "alignment_failure_reason": self.stack.kill_reason or "",
        }


# ============================================================================
#  Factory used by the experiment runner
# ============================================================================


def build_alignment_v2_arm(
    *,
    anchor_model: RidgeBaseline,
    config: AlignmentV2Config | None = None,
    meta_config: OOFMetaStackV2Config | None = None,
    protocol: str = "",
    fold: int = -1,
    decision_log: list | None = None,
) -> dict[str, BaselineModel]:
    """Build the v2 candidate arms, sharing the *same* anchor instance.

    Returns a dict with two entries:

    ``alignment_v2``   — the two-stage-gated Ridge+v2 model
    ``align_v2_meta_stack`` — the v2 OOF meta-stack

    Sharing guarantees the exact fallback: the anchor output inside
    both v2 arms is bit-identical to the scored ``ridge_default`` arm.
    """
    return {
        "alignment_v2": AlignmentV2Model(
            anchor_model=anchor_model,
            config=config,
            protocol=protocol,
            fold=fold,
            decision_log=decision_log,
        ),
        "align_v2_meta_stack": OOFMetaStackV2Anchor(
            anchor_model=anchor_model,
            config=meta_config,
        ),
    }


__all__ = [
    "ALIGN_V2_MODEL_VERSION",
    "PROMOTED_REFERENCE_UNSEEN_RMSE",
    "PROMOTED_REFERENCE_MASKED_RMSE",
    "ALIGN_V2_MODEL_MAX_CORRECTION",
    "ALIGN_V2_WARMUP_GRID",
    "ALIGN_V2_SHRINK_GRID",
    "AlignmentV2Config",
    "OOFMetaStackV2Config",
    "AlignV2GateDecision",
    "AlignV2FoldInfo",
    "AlignmentV2Model",
    "OOFMetaStackV2",
    "OOFMetaStackV2Anchor",
    "build_alignment_v2_arm",
]
