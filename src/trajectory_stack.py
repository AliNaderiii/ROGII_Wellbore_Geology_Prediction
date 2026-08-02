"""Trajectory stack pipeline — booster residuals, OOF meta-stack, gated stack.

Implements the remaining letters of the production pipeline on top of the
verified repository components, **without modifying any validated code**:

    A/L  Ridge Default anchor + exact fallback      (src.baselines.RidgeBaseline)
    B    PF/Beam bounded candidate corrections      (src.geoanchor)
    C    Visible-prefix affine GR calibration       (src.geoanchor)
    D    Multi-scale GR datum scan + branch disagreement (this module)
    E    Robust IRLS stratigraphic projection       (src.safe_alignment)
    F    LightGBM anchored-residual model           (this module)
    G    CatBoost anchored-residual model           (this module)
    H    OOF Ridge meta-stack                       (this module)
    I    Visible-prefix pseudo-holdout gate         (this module)
    J    Tail-risk / worst-well guard               (this module)
    K    Warmup correction ramp                     (this module)

Every learned component is fitted on fold-training wells only, every decision
threshold is selected from fold-training OOF diagnostics only, and every arm
returns the *exact* Ridge Default prediction whenever any guard declines:

    prediction = Ridge_anchor + guarded_correction

Leakage contract (same as the rest of the repository): inference reads MD,
X, Y, Z, GR, the visible ``TVT_input`` prefix, Typewell TVT and Typewell GR.
Nothing here reads Typewell Geology, formation markers, hidden TVT, external
artifacts, Koolbox, duplicate-well lookups, or any public-leaderboard signal.
The three blocked public wells are excluded upstream by ``src.validation``
and re-asserted here wherever training sets are built.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.baselines import BaselineModel, RidgeBaseline
from src.features import TypewellReference, validate_feature_frame
from src.geoanchor import (
    CORRECTION_CAP_FT,
    CandidateCorrection,
    GateFitInfo,
    MultiBranchConfig,
    _family_confidence,
    _finite_rmse,
    _tail_mean_se,
    fit_prefix_affine_calibration,
    generate_candidate_corrections,
    multibranch_scan,
    nested_pseudo_task,
)
from src.device import CPU_RESOLUTION, DeviceResolution
from src.manifest import assert_safe_features
from src.safe_alignment import _ramp
from src.tasks import InferenceTask, WellTask
from src.validation import (
    CrossFitLeakage,
    assert_no_blocked_wells,
    make_group_folds,
)

TRAJECTORY_STACK_VERSION = "trajectory-stack-v1"

# Reference used only as a *promotion yardstick* (never as a feature or a
# tuning target): the verified real 770-well Ridge Default unseen_well RMSE.
RIDGE_REFERENCE_UNSEEN_RMSE = 14.422911

# --------------------------------------------------------------------------
# Arm registry
# --------------------------------------------------------------------------

ARM_RIDGE = "ridge_default"          # A/L — anchor and exact fallback
ARM_LGBM = "lgbm_residual"           # F — ungated evidence arm
ARM_CAT = "catboost_residual"        # G — ungated evidence arm
ARM_STACK = "oof_meta_stack"         # H — kill-switched OOF meta-stack
ARM_GATED = "gated_trajectory"       # the promotion candidate (I/J/K over A–H)

ARM_ORDER = (ARM_RIDGE, ARM_LGBM, ARM_CAT, ARM_STACK, ARM_GATED)

ARM_LABELS = {
    ARM_RIDGE: "A/L. Ridge Default (anchor + exact fallback)",
    ARM_LGBM: "F. LightGBM anchored-residual (evidence arm)",
    ARM_CAT: "G. CatBoost anchored-residual (evidence arm)",
    ARM_STACK: "H. OOF Ridge meta-stack (kill-switched)",
    ARM_GATED: "Conservative gated trajectory stack (promotion candidate)",
}

#: Gate candidate bank. PF/Beam/mean are the repository's target-free tracks;
#: mb_hedged is the trust-shrunk bimodal datum candidate; lgbm_row / cat_row
#: are the bounded residual-model tracks.
STACK_CANDIDATES = ("pf", "beam", "pf_beam_mean", "mb_hedged", "lgbm_row", "cat_row")

#: A-priori multi-scale scan half-ranges (ft). Fixed before any run: the scan
#: asks the same bounded datum question at three tolerances; consistency
#: across scales is the disagreement evidence.
MULTISCALE_RANGES_FT = (8.0, 15.0, 25.0)

#: Distance (ft) within which two scales are said to agree on the datum.
MULTISCALE_AGREE_FT = 1.5

# Available boosting libraries are detected, not assumed. An unavailable
# library makes its arm honestly "unavailable"; it never substitutes another
# model.
try:  # pragma: no cover - environment dependent
    import lightgbm as _lgb

    HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover
    _lgb = None
    HAVE_LIGHTGBM = False

try:  # pragma: no cover - environment dependent
    from catboost import CatBoostRegressor, Pool

    HAVE_CATBOOST = True
except Exception:  # pragma: no cover
    CatBoostRegressor = None
    Pool = None
    HAVE_CATBOOST = False


# ==========================================================================
# D — multi-scale GR datum scan
# ==========================================================================


@dataclass
class MultiScaleResult:
    """Datum shift of the GR/Typewell scan at several half-ranges.

    All values are target-free diagnostics of GR-vs-Typewell-GR agreement.
    ``ptp`` (peak-to-peak of per-scale shifts) is the multi-scale branch
    disagreement: when the answer depends on how far the scan is allowed to
    move, the alignment is unstable and the gate must distrust it.
    """

    ok: bool
    failure_reason: str = ""
    shifts: tuple[float, ...] = ()
    confidences: tuple[float, ...] = ()
    ptp: float = np.inf
    dominant_shift: float = 0.0
    n_agree: int = 0
    min_confidence: float = 0.0


def multiscale_scan(
    task: InferenceTask,
    *,
    cal=None,
    ref: TypewellReference | None = None,
    ranges: tuple[float, ...] = MULTISCALE_RANGES_FT,
) -> MultiScaleResult:
    """Run the bounded datum scan at several half-ranges and compare answers."""
    ref = ref or TypewellReference(task.tw_tvt, task.tw_gr)
    if not ref.ok:
        return MultiScaleResult(ok=False, failure_reason="missing_or_invalid_typewell")
    shifts: list[float] = []
    confs: list[float] = []
    for r in ranges:
        mb = multibranch_scan(task, cal=cal, ref=ref, config=MultiBranchConfig(search=float(r)))
        if not mb.ok:
            continue
        shifts.append(float(mb.shift1))
        confs.append(float(mb.confidence))
    if not shifts:
        return MultiScaleResult(ok=False, failure_reason="all_scales_failed")
    arr = np.asarray(shifts, dtype="float64")
    dominant = float(np.median(arr))
    n_agree = int(np.count_nonzero(np.abs(arr - dominant) <= MULTISCALE_AGREE_FT))
    return MultiScaleResult(
        ok=True,
        shifts=tuple(float(s) for s in shifts),
        confidences=tuple(float(c) for c in confs),
        ptp=float(np.max(arr) - np.min(arr)),
        dominant_shift=dominant,
        n_agree=n_agree,
        min_confidence=float(np.min(confs)),
    )


# ==========================================================================
# F/G — boosted anchored-residual learners (fold-fitted, early-stopped)
# ==========================================================================


@dataclass
class BoostFitInfo:
    """Per-fit bookkeeping for a boosted residual learner."""

    library: str = ""
    n_wells: int = 0
    n_rows: int = 0
    n_eval_wells: int = 0
    best_iteration: int = 0
    eval_rmse: float = np.nan
    available: bool = True
    unavailable_reason: str = ""
    fit_seconds: float = 0.0
    #: Fold-training OOF residual skill, filled by the stack/gate when the
    #: learner is cross-fitted inside a fold: (anchor_oof - learner_oof) /
    #: anchor_oof, clipped to [0, 1]. Target-free at inference time.
    oof_skill: float = 0.0
    #: Device provenance, mirrored into every model report row.
    device_requested: str = "cpu"
    device_selected: str = "cpu"
    lgbm_device: str = "cpu"
    catboost_device: str = "cpu"
    gpu_fallback_reason: str = ""


class _BoostedResidual(RidgeBaseline):
    """Boosted tree on the anchored residual, on the default feature matrix.

    Inherits the validated plumbing of ``RidgeBaseline``: identical 28-column
    no-alignment design matrix, identical fold-specific median imputation,
    identical anchored-residual target, identical typewell clip. Only the fit
    changes: a well-disjoint inner holdout (carved from the fold-training
    wells) drives early stopping, and every seed is fixed.
    """

    _lib = "base"

    def __init__(
        self,
        *,
        seed: int = 0,
        max_iter: int = 400,
        estop_rounds: int = 50,
        eval_fraction: float = 0.2,
        min_wells_for_eval: int = 16,
        thread_count: int = 4,
        device: DeviceResolution | None = None,
    ):
        super().__init__(alignment_features=False)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.estop_rounds = int(estop_rounds)
        self.eval_fraction = float(eval_fraction)
        self.min_wells_for_eval = int(min_wells_for_eval)
        self.thread_count = int(thread_count)
        #: Where this learner trains. Defaults to the pure-CPU resolution so
        #: every existing caller keeps its exact previous behaviour.
        self.device = device or CPU_RESOLUTION
        self.model = None
        self.info = BoostFitInfo(library=self._lib, **self._device_info_fields())

    def _device_info_fields(self) -> dict:
        dev = getattr(self, "device", None) or CPU_RESOLUTION
        return dev.as_report()

    @property
    def effective_device(self) -> str:
        """``cpu``/``gpu`` for *this* library (not the run-wide selection)."""
        dev = self.device or CPU_RESOLUTION
        return dev.catboost_device if self._lib == "catboost" else dev.lgbm_device

    # — implemented per library ------------------------------------------------
    def _fit_booster(self, Xtr, ytr, Xev, yev):  # pragma: no cover - abstract
        raise NotImplementedError

    def _booster_unavailable_reason(self) -> str:
        return f"{self._lib}_not_installed"

    def _have_library(self) -> bool:  # pragma: no cover - overridden
        return True

    # — shared fit plumbing ------------------------------------------------------
    def fit(self, tasks, **kw):
        t0 = time.perf_counter()
        self.info = BoostFitInfo(library=self._lib, **self._device_info_fields())
        assert_no_blocked_wells([t.well_id for t in tasks], context=f"{self.name} fit")
        if not self._have_library():
            raise RuntimeError(
                f"{self.name}: {self._booster_unavailable_reason()}; the harness "
                "records this arm as unavailable rather than substituting a model"
            )
        X, y, groups = self._training_arrays(tasks)
        if X is None:
            self.info.available = False
            self.info.unavailable_reason = "no_training_rows"
            self.info.fit_seconds = time.perf_counter() - t0
            return self
        self.medians_ = X.median(numeric_only=True)
        Xv = self._clean(X)
        wells = np.asarray(pd.unique(groups))
        self.info.n_wells = int(wells.size)
        self.info.n_rows = int(Xv.shape[0])

        rng = np.random.default_rng(self.seed + 3)
        n_eval = 0
        eval_ids: np.ndarray = np.array([], dtype=object)
        if wells.size >= self.min_wells_for_eval:
            n_eval = max(2, int(round(self.eval_fraction * wells.size)))
            eval_ids = rng.permutation(wells)[:n_eval]
            mask_ev = np.isin(groups, eval_ids)
        else:
            mask_ev = np.zeros(Xv.shape[0], dtype=bool)
        self.info.n_eval_wells = int(eval_ids.size)
        Xtr, ytr = Xv[~mask_ev], y[~mask_ev]
        if Xtr.shape[0] < 50:
            Xtr, ytr = Xv, y
            mask_ev = np.zeros(Xv.shape[0], dtype=bool)
        Xev, yev = (Xv[mask_ev], y[mask_ev]) if mask_ev.any() else (None, None)
        self.model = self._fit_booster(Xtr, ytr, Xev, yev)
        self.info.fit_seconds = time.perf_counter() - t0
        return self

    def predict(self, task, feats=None):
        anchor = self._anchor(task)
        if self.model is None:
            return np.full(task.n_predict, anchor)
        X = self._features(task, feats).reindex(columns=self.feature_names_)
        resid = self.model.predict(self._clean(X))
        return self._clip_to_typewell(task, anchor + np.asarray(resid, dtype="float64"))


class LightGBMResidual(_BoostedResidual):
    """F. LightGBM on the anchored residual, well-disjoint early stopping."""

    name = ARM_LGBM
    _lib = "lightgbm"

    def _have_library(self) -> bool:
        return HAVE_LIGHTGBM

    def _fit_booster(self, Xtr, ytr, Xev, yev):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": self.seed,
            "deterministic": True,
        }
        # Device parameters last: they own device_type/n_jobs/gpu_use_dp.
        params.update((self.device or CPU_RESOLUTION).lgbm_params(self.thread_count))
        dtrain = _lgb.Dataset(Xtr, label=ytr, free_raw_data=True)
        callbacks = []
        valid_sets = None
        if Xev is not None and len(yev) >= 50:
            deval = _lgb.Dataset(Xev, label=yev, reference=dtrain, free_raw_data=True)
            valid_sets = [deval]
            callbacks.append(_lgb.early_stopping(self.estop_rounds, verbose=False))
        model = _lgb.train(
            params,
            dtrain,
            num_boost_round=self.max_iter,
            valid_sets=valid_sets,
            callbacks=callbacks or None,
        )
        self.info.best_iteration = int(model.best_iteration or self.max_iter)
        if valid_sets:
            score = model.best_score.get("valid_0", {}).get("rmse")
            self.info.eval_rmse = float(score) if score is not None else np.nan
        return model


class CatBoostResidual(_BoostedResidual):
    """G. CatBoost on the anchored residual, well-disjoint early stopping."""

    name = ARM_CAT
    _lib = "catboost"

    def _have_library(self) -> bool:
        return HAVE_CATBOOST

    def _fit_booster(self, Xtr, ytr, Xev, yev):
        params = dict(
            iterations=self.max_iter,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            random_seed=self.seed,
            verbose=False,
        )
        params.update((self.device or CPU_RESOLUTION).catboost_params(self.thread_count))
        train_pool = Pool(Xtr, ytr)
        eval_pool = None
        if Xev is not None and len(yev) >= 50:
            eval_pool = Pool(Xev, yev)
            params["od_type"] = "Iter"
            params["od_wait"] = self.estop_rounds
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
        if eval_pool is not None:
            self.info.best_iteration = int(model.get_best_iteration() or self.max_iter)
            self.info.eval_rmse = float(
                model.get_best_score().get("validation", {}).get("RMSE", np.nan)
            )
        else:
            self.info.best_iteration = self.max_iter
        return model


def build_residual_learners(
    *,
    seed: int,
    use_lightgbm: bool = True,
    use_catboost: bool = True,
    device: DeviceResolution | None = None,
    **boost_kw,
) -> dict:
    """The residual learner set for the stack/gate. Unavailable libraries are
    skipped at *fit* time with an honest failure record (never substituted)."""
    out: dict[str, _BoostedResidual] = {}
    device = device or CPU_RESOLUTION
    if use_lightgbm and HAVE_LIGHTGBM:
        out["lgbm"] = LightGBMResidual(seed=seed, device=device, **boost_kw)
    if use_catboost and HAVE_CATBOOST:
        out["cat"] = CatBoostResidual(seed=seed, device=device, **boost_kw)
    return out


# ==========================================================================
# H — OOF Ridge meta-stack
# ==========================================================================

#: Registered meta-design columns (see src/manifest.py registrations).
META_RES_COLUMNS = ("meta_res_ridge", "meta_res_lgbm", "meta_res_cat")
META_GEOM_COLUMNS = ("meta_dmd", "meta_log1p_dmd")


@dataclass(frozen=True)
class StackConfig:
    """A-priori constants for the OOF meta-stack. Decision-level quantities
    (meta alpha, kill switch) are selected per fold from fold-training OOF
    rows only; the grids below are fixed before any run."""

    inner_splits: int = 5
    tune_splits: int = 3
    meta_alphas: tuple = (1.0, 10.0, 100.0)
    correction_cap_ft: float = CORRECTION_CAP_FT
    max_rows_per_well: int = 400
    boost_max_iter: int = 400
    boost_estop_rounds: int = 50
    boost_threads: int = 4
    use_lightgbm: bool = True
    use_catboost: bool = True
    #: CPU/GPU resolution for the boosted residual learners only.
    device: DeviceResolution = CPU_RESOLUTION
    seed: int = 0


def _task_row_geom(task: InferenceTask) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(task.dmd, dtype="float64")
    return d, np.log1p(np.clip(d, 0, None))


class OOFMetaStack:
    """Cross-fitted stacker of anchored-residual learners.

    Inner GroupKFold over the *fold-training* wells produces OOF residual
    predictions from each learner (Ridge and the boosters). A small meta-Ridge
    on ``[res_ridge, res_lgbm?, res_cat?, dmd, log1p_dmd]`` is fitted on those
    OOF rows only. The ridge column lets the meta-model fall back to pure
    Ridge numerically, and the kill switch below restores it *exactly*:

    * alpha is chosen on tune-subfolds of the same OOF rows (pooled RMSE of
      the stack must beat the pooled RMSE of the OOF ridge column);
    * if no alpha beats the anchor strictly, ``killed=True`` and every call
      to :meth:`predict` returns the anchor prediction unchanged.
    """

    def __init__(self, config: StackConfig | None = None) -> None:
        self.config = config or StackConfig()
        self.killed = False
        self.kill_reason = ""
        self.meta_model = None
        self.meta_alpha = float("nan")
        self.meta_columns: list[str] = []
        self.learners_: dict[str, _BoostedResidual] = {}
        self.medians_: pd.Series | None = None
        self.scaler_ = None
        self.info = {
            "n_oof_wells": 0,
            "n_oof_rows": 0,
            "meta_alpha": np.nan,
            "pooled_sub_oof_delta": np.nan,
            "killed": False,
            "kill_reason": "",
            "learners_used": [],
            "fit_seconds": 0.0,
        }

    # ------------------------------------------------------------------ OOF
    def _meta_row(self, task: InferenceTask, preds: dict[str, np.ndarray]) -> pd.DataFrame:
        anchor = task.anchor_tvt
        anchor = anchor if np.isfinite(anchor) else 0.0
        dmd, log1p_dmd = _task_row_geom(task)
        data: dict[str, np.ndarray] = {}
        if "ridge" in preds:
            data["meta_res_ridge"] = np.asarray(preds["ridge"], dtype="float64") - anchor
        if "lgbm" in preds:
            data["meta_res_lgbm"] = np.asarray(preds["lgbm"], dtype="float64") - anchor
        if "cat" in preds:
            data["meta_res_cat"] = np.asarray(preds["cat"], dtype="float64") - anchor
        data["meta_dmd"] = dmd
        data["meta_log1p_dmd"] = log1p_dmd
        return pd.DataFrame(data)

    def _build_oof_design(self, train_tasks: list[WellTask]):
        ids = [t.well_id for t in train_tasks]
        assert_no_blocked_wells(ids, context="meta-stack OOF training wells")
        inner = make_group_folds(ids, n_splits=self.config.inner_splits, seed=self.config.seed + 11)
        by_id = {t.well_id: t for t in train_tasks}
        Xs, ys, ws = [], [], []
        boost_kw = dict(
            max_iter=self.config.boost_max_iter,
            estop_rounds=self.config.boost_estop_rounds,
            thread_count=self.config.boost_threads,
            device=self.config.device,
        )
        for fold in inner:
            train_inner = [by_id[w] for w in fold.train_ids if w in by_id]
            valid_inner = [by_id[w] for w in fold.valid_ids if w in by_id]
            if not train_inner or not valid_inner:
                continue
            ridge_inner = RidgeBaseline(alignment_features=False)
            ridge_inner.fit(train_inner)
            learners_inner = build_residual_learners(
                seed=self.config.seed + 100 + fold.index,
                use_lightgbm=self.config.use_lightgbm,
                use_catboost=self.config.use_catboost,
                **boost_kw,
            )
            fitted: dict[str, _BoostedResidual] = {}
            for name, learner in learners_inner.items():
                try:
                    learner.fit(train_inner)
                    fitted[name] = learner
                except Exception:
                    continue  # an unavailable library never blocks the fold
            rng = np.random.default_rng(self.config.seed + fold.index)
            for task in valid_inner:
                inp = task.inputs()
                target = task.target
                if target is None:
                    continue
                anchor = inp.anchor_tvt
                anchor = anchor if np.isfinite(anchor) else 0.0
                y = np.asarray(target, dtype="float64") - anchor
                mask = np.isfinite(y)
                if not mask.any():
                    continue
                preds = {"ridge": np.asarray(ridge_inner.predict(inp), dtype="float64")}
                for name, learner in fitted.items():
                    try:
                        preds[name] = np.asarray(learner.predict(inp), dtype="float64")
                    except Exception:
                        pass
                frame = self._meta_row(inp, preds)
                frame, y_m = frame[mask], y[mask]
                if len(frame) > self.config.max_rows_per_well:
                    pick = rng.choice(len(frame), self.config.max_rows_per_well, replace=False)
                    pick.sort()
                    frame, y_m = frame.iloc[pick], y_m[pick]
                Xs.append(frame)
                ys.append(y_m)
                ws.append(np.full(len(frame), inp.well_id))
        if not Xs:
            return None, None, None
        return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)

    # ------------------------------------------------------------------ fit
    def fit(self, train_tasks: list[WellTask]) -> "OOFMetaStack":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        t0 = time.perf_counter()
        cfg = self.config
        X, y, wells = self._build_oof_design(train_tasks)
        if X is None or len(np.unique(wells)) < cfg.tune_splits * 2:
            self.killed = True
            self.kill_reason = "insufficient_oof_rows"
            self._record(t0)
            return self
        self.info["n_oof_wells"] = int(len(np.unique(wells)))
        self.info["n_oof_rows"] = int(len(X))

        self.meta_columns = list(X.columns)
        assert_safe_features(self.meta_columns, context="meta-stack design matrix")
        validate_feature_frame(X)
        self.medians_ = X.median(numeric_only=True)
        Xv = X.fillna(self.medians_).fillna(0.0).to_numpy(dtype="float64")
        self.scaler_ = StandardScaler().fit(Xv)

        # --- alpha selection + kill switch on tune-subfolds -------------------
        uniq = np.unique(wells)
        sub = make_group_folds([str(w) for w in uniq], n_splits=cfg.tune_splits, seed=cfg.seed + 23)
        ridge_col = X.columns.get_loc("meta_res_ridge")
        best: tuple[float, float] | None = None  # (mean_delta, alpha)
        for alpha in cfg.meta_alphas:
            deltas = []
            for fold in sub:
                tr = np.isin(wells, np.asarray(fold.train_ids, dtype=object))
                va = np.isin(wells, np.asarray(fold.valid_ids, dtype=object))
                if not tr.any() or not va.any():
                    continue
                sub_scaler = StandardScaler().fit(Xv[tr])
                meta = Ridge(alpha=float(alpha)).fit(sub_scaler.transform(Xv[tr]), y[tr])
                pred_va = meta.predict(sub_scaler.transform(Xv[va]))
                rmse_stack = float(np.sqrt(np.mean((pred_va - y[va]) ** 2)))
                rmse_ridge = float(np.sqrt(np.mean((Xv[va, ridge_col] - y[va]) ** 2)))
                deltas.append(rmse_stack - rmse_ridge)
            if not deltas:
                continue
            key = (float(np.mean(deltas)), float(alpha))  # deterministic tie-break
            if best is None or key < best:
                best = key
        if best is None or best[0] >= -1e-9:
            self.killed = True
            self.kill_reason = "meta_stack_oof_not_better_than_anchor"
            self._record(t0)
            return self

        self.meta_alpha = float(best[1])
        self.meta_model = Ridge(alpha=self.meta_alpha).fit(self.scaler_.transform(Xv), y)
        self.info["pooled_sub_oof_delta"] = float(best[0])

        # --- full-fold learners for inference --------------------------------
        boost_kw = dict(
            max_iter=cfg.boost_max_iter,
            estop_rounds=cfg.boost_estop_rounds,
            thread_count=cfg.boost_threads,
            device=cfg.device,
        )
        learners = build_residual_learners(
            seed=cfg.seed + 900,
            use_lightgbm=cfg.use_lightgbm,
            use_catboost=cfg.use_catboost,
            **boost_kw,
        )
        self.learners_ = {}
        for name, learner in learners.items():
            try:
                learner.fit(train_tasks)
                self.learners_[name] = learner
            except Exception as exc:
                learner.info.available = False
                learner.info.unavailable_reason = f"fit_failed:{type(exc).__name__}"
        self.info["learners_used"] = sorted(self.learners_)
        self.info["meta_alpha"] = float(self.meta_alpha)
        self._record(t0)
        return self

    def _record(self, t0: float) -> None:
        self.info["killed"] = bool(self.killed)
        self.info["kill_reason"] = str(self.kill_reason)
        self.info["fit_seconds"] = float(time.perf_counter() - t0)

    # -------------------------------------------------------------- predict
    def stacked_residual_columns(self, task: InferenceTask, anchor_pred: np.ndarray) -> pd.DataFrame:
        preds = {"ridge": np.asarray(anchor_pred, dtype="float64")}
        for name, learner in self.learners_.items():
            preds[name] = np.asarray(learner.predict(task), dtype="float64")
        frame = self._meta_row(task, preds)
        return frame.reindex(columns=self.meta_columns)

    def predict(self, task: InferenceTask, anchor_pred: np.ndarray) -> np.ndarray:
        """Stacked prediction; the caller substitutes the exact anchor on kill."""
        frame = self.stacked_residual_columns(task, anchor_pred)
        Xv = frame.fillna(self.medians_).fillna(0.0).to_numpy(dtype="float64")
        resid = self.meta_model.predict(self.scaler_.transform(Xv))
        anchor = task.anchor_tvt
        anchor = anchor if np.isfinite(anchor) else 0.0
        pred = anchor + np.asarray(resid, dtype="float64")
        pred = np.where(np.isfinite(pred), pred, anchor_pred)
        move = np.clip(pred - anchor_pred, -self.config.correction_cap_ft, self.config.correction_cap_ft)
        out = anchor_pred + move
        return np.where(np.isfinite(out), out, anchor_pred)


class OOFMetaStackAnchor(BaselineModel):
    """Arm H — Ridge anchor + kill-switched OOF meta-stack."""

    name = ARM_STACK
    needs_alignment = False

    def __init__(self, *, anchor_model: RidgeBaseline, config: StackConfig | None = None):
        self.anchor_model = anchor_model
        self.config = config or StackConfig()
        self.stack = OOFMetaStack(self.config)
        self._last_diagnostics: dict = {}

    def fit(self, tasks: list[WellTask], **kw) -> "OOFMetaStackAnchor":
        if self.anchor_model.model is None:
            self.anchor_model.fit(tasks)
        try:
            self.stack.fit(tasks)
        except Exception as exc:
            self.stack.killed = True
            self.stack.kill_reason = f"stack_fit_failed:{type(exc).__name__}"
            self.stack.info["killed"] = True
            self.stack.info["kill_reason"] = self.stack.kill_reason
        return self

    def predict(self, task: InferenceTask, feats=None) -> np.ndarray:
        base = np.asarray(self.anchor_model.predict(task, feats), dtype="float64")
        stack = self.stack
        if stack.killed or stack.meta_model is None:
            self._last_diagnostics = {
                "gate_activation": False,
                "gate_fallback_exact_ridge": True,
                "fallback_points": int(task.n_predict),
                "fallback_fraction": 1.0,
                "alignment_ok": False,
                "alignment_failure_reason": stack.kill_reason or "stack_unfitted",
                "gate_confidence_threshold": np.nan,
            }
            return base
        try:
            pred = stack.predict(task, base)
        except Exception as exc:
            self._last_diagnostics = {
                "gate_activation": False,
                "gate_fallback_exact_ridge": True,
                "fallback_points": int(task.n_predict),
                "fallback_fraction": 1.0,
                "alignment_ok": False,
                "alignment_failure_reason": f"stack_predict_failed:{type(exc).__name__}",
                "gate_confidence_threshold": np.nan,
            }
            return base
        if not np.all(np.isfinite(pred)):
            return base
        corr = pred - base
        self._last_diagnostics = {
            "gate_activation": bool(np.any(np.abs(corr) > 1e-12)),
            "gate_fallback_exact_ridge": False,
            "fallback_points": int(np.count_nonzero(np.abs(corr) <= 1e-12)),
            "fallback_fraction": float(np.mean(np.abs(corr) <= 1e-12)),
            "alignment_ok": True,
            "alignment_failure_reason": "",
            "gate_correction_magnitude": float(np.mean(np.abs(corr))),
            "gate_confidence_threshold": np.nan,
        }
        return pred

    def prediction_diagnostics(self, task, feats, pred) -> dict:
        return dict(self._last_diagnostics)


# ==========================================================================
# Gate candidate bank — PF/Beam + bimodal hedge + residual learners
# ==========================================================================


def _residual_candidate(
    name: str,
    task: InferenceTask,
    base_pred: np.ndarray,
    learner: _BoostedResidual | None,
    corr_ref: np.ndarray | None,
    correction_cap: float,
) -> CandidateCorrection:
    """A bounded residual-model correction with a fold-OOF confidence scalar."""
    if learner is None or getattr(learner, "model", None) is None:
        return CandidateCorrection(name, None, 0.0, np.inf, False, "learner_unfitted")
    try:
        pred = np.asarray(learner.predict(task), dtype="float64")
    except Exception:
        return CandidateCorrection(name, None, 0.0, np.inf, False, "learner_predict_failed")
    if pred.size != base_pred.size or not np.isfinite(pred).any():
        return CandidateCorrection(name, None, 0.0, np.inf, False, "learner_nonfinite")
    corrected = base_pred + np.clip(pred - base_pred, -correction_cap, correction_cap)
    corrected = np.where(np.isfinite(corrected), corrected, base_pred)
    if corr_ref is not None:
        disagreement = float(np.mean(np.abs(corrected - base_pred - corr_ref)))
        disagreement = float(np.clip(disagreement, 0.0, 2.0 * correction_cap))
    else:
        disagreement = float("inf")
    confidence = float(np.clip(getattr(learner.info, "oof_skill", 0.0), 0.0, 1.0))
    return CandidateCorrection(name, corrected, confidence, disagreement, True, "")


def trajectory_candidates(
    task: InferenceTask,
    base_pred: np.ndarray,
    *,
    pf,
    beam,
    mb=None,
    learners: dict[str, _BoostedResidual] | None = None,
    correction_cap: float = CORRECTION_CAP_FT,
) -> dict[str, CandidateCorrection]:
    """The full gated candidate bank (all target-free to compute).

    Produces the repository PF/Beam candidates, the trust-shrunk bimodal
    datum hedge, and the residual-learner tracks. Every correction is
    referenced to and capped around ``base_pred`` (the Ridge anchor output).
    """
    out = generate_candidate_corrections(
        task, base_pred, pf=pf, beam=beam, mb=mb, correction_cap=correction_cap
    )

    # ---- bimodal hedge: trust-shrunk constant datum on the anchor ----------
    if mb is not None and mb.ok and np.isfinite(mb.shift1):
        hedged_shift = mb.w1 * mb.shift1 + (1.0 - mb.w1) * mb.shift2
        corrected = base_pred + np.clip(hedged_shift, -correction_cap, correction_cap)
        corrected = np.where(np.isfinite(corrected), corrected, base_pred)
        out["mb_hedged"] = CandidateCorrection(
            "mb_hedged",
            corrected,
            float(np.clip(mb.confidence * max(mb.prefix_trust, 0.0), 0.0, 1.0)),
            float(mb.sep) if mb.bimodal else 0.0,
            True,
            "",
        )
    else:
        reason = mb.failure_reason if mb is not None else "multibranch_not_provided"
        out["mb_hedged"] = CandidateCorrection("mb_hedged", None, 0.0, np.inf, False, str(reason))

    available = [c for c in out.values() if c.available and c.prediction is not None]
    corr_mean = None
    if available:
        corr_mean = np.mean([c.prediction - base_pred for c in available], axis=0)

    learners = learners or {}
    for name, key in (("lgbm_row", "lgbm"), ("cat_row", "cat")):
        out[name] = _residual_candidate(
            name, task, base_pred, learners.get(key), corr_mean, correction_cap
        )
    return out


# ==========================================================================
# I/J/K/L — the gated trajectory stack
# ==========================================================================

#: Extended gate design rows. Provenance is registered in src/manifest.py;
#: every row derives strictly from the allowed inference roots.
STACK_GATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "gate_prefix_len",
    "gate_suffix_len",
    "gate_gr_missing_suffix",
    "gate_prefix_gr_missing",
    "gate_tvt_std_prefix",
    "gate_tvt_range_prefix",
    "gate_tvt_slope_300",
    "gate_anchor",
    "gate_acal_alpha",
    "gate_acal_beta",
    "gate_acal_fit_rmse",
    "gate_acal_prefix_corr",
    "gate_mb_shift1",
    "gate_mb_sep",
    "gate_mb_cost_gap",
    "gate_mb_confidence",
    "gate_mb_bimodal",
    "gate_mb_prefix_trust",
    "gate_pf_confidence",
    "gate_pf_spread",
    "gate_pf_fallback",
    "gate_beam_confidence",
    "gate_beam_spread",
    "gate_beam_fallback",
    "gate_track_disagreement",
    "gate_ms_ptp",
    "gate_ms_dominant_shift",
    "gate_ms_n_agree",
    "gate_ms_min_conf",
    "gate_lgbm_oof_skill",
    "gate_cat_oof_skill",
    "gate_cand_pf",
    "gate_cand_beam",
    "gate_cand_mean",
    "gate_cand_mb",
    "gate_cand_lgbm",
    "gate_cand_cat",
)


@dataclass(frozen=True)
class TrajectoryGateConfig:
    """A-priori grids and capacity limits for the stack gate.

    Everything the gate *decides* (margin, confidence threshold, disagreement
    cap, shrink factor, warmup length) is tuned per outer fold from
    fold-training OOF examples only; the grids are fixed before any run and
    never touched by any leaderboard signal.
    """

    inner_splits: int = 5
    tune_splits: int = 3
    max_correction: float = CORRECTION_CAP_FT
    margins: tuple = (0.0, 0.05)
    confidence_levels: tuple = (0.0, 0.5)
    disagreement_levels: tuple = (1.0, 0.5)
    shrink_grid: tuple = (0.5, 0.75, 1.0)
    warmup_grid: tuple = (100, 200, 400)
    min_examples: int = 40
    gbdt_max_iter: int = 150
    gbdt_max_depth: int = 3
    boost_max_iter: int = 400
    boost_estop_rounds: int = 50
    boost_threads: int = 4
    use_lightgbm: bool = True
    use_catboost: bool = True
    #: CPU/GPU resolution for the boosted residual learners only.
    device: DeviceResolution = CPU_RESOLUTION
    min_pseudo_points: int = 25
    seed: int = 0


def _well_prefix_scalars_local(task: InferenceTask) -> dict[str, float]:
    from src.geoanchor import _well_prefix_scalars

    return _well_prefix_scalars(task)


def stack_gate_feature_row(
    task: InferenceTask,
    *,
    cal,
    mb,
    ms: MultiScaleResult,
    pf_out,
    beam_out,
    learners: dict[str, _BoostedResidual],
    candidate: str,
) -> dict[str, float]:
    """One design row of the stack gate (well, candidate) at one boundary."""
    from src.geoanchor import CORRECTION_MIN_OBSERVED_FRAC

    prefix = _well_prefix_scalars_local(task)
    gr_miss_suffix = float(np.mean(~np.isfinite(task.gr[task.start : task.stop])))
    gr_miss_prefix = float(np.mean(~np.isfinite(task.gr[: task.start])))
    anchor = task.anchor_tvt
    anchor = anchor if np.isfinite(anchor) else 0.0
    pf_track = anchor + np.asarray(pf_out.frame["pf_shift"], dtype="float64")
    beam_track = anchor + np.asarray(beam_out.frame["beam_shift"], dtype="float64")
    pf_fallback = float(pf_out.diagnostics.get("fallback_fraction", 1.0))
    beam_fallback = float(beam_out.diagnostics.get("fallback_fraction", 1.0))
    both_informed = (
        pf_fallback <= 1.0 - CORRECTION_MIN_OBSERVED_FRAC
        and beam_fallback <= 1.0 - CORRECTION_MIN_OBSERVED_FRAC
    )
    observers = slice(0, min(600, task.n_predict))
    if both_informed:
        disagreement = float(np.mean(np.abs(pf_track[observers] - beam_track[observers])))
    elif mb is not None and mb.ok:
        disagreement = float(mb.sep)
    else:
        disagreement = np.inf
    lgbm = learners.get("lgbm")
    cat = learners.get("cat")
    row = {
        "gate_prefix_len": float(task.prefix_len),
        "gate_suffix_len": float(task.n_predict),
        "gate_gr_missing_suffix": float(gr_miss_suffix),
        "gate_prefix_gr_missing": float(gr_miss_prefix),
        "gate_tvt_std_prefix": prefix["std"],
        "gate_tvt_range_prefix": prefix["range"],
        "gate_tvt_slope_300": prefix["slope300"],
        "gate_anchor": float(anchor),
        "gate_acal_alpha": float(cal.alpha) if cal.ok else 0.0,
        "gate_acal_beta": float(cal.beta / max(cal.sd_tw, 1e-9)) if cal.ok else 0.0,
        "gate_acal_fit_rmse": float(cal.fit_rmse_z) if cal.ok else 0.0,
        "gate_acal_prefix_corr": float(cal.prefix_corr)
        if cal.ok and np.isfinite(cal.prefix_corr)
        else 0.0,
        "gate_mb_shift1": float(mb.shift1) if (mb is not None and mb.ok) else 0.0,
        "gate_mb_sep": float(mb.sep) if (mb is not None and mb.ok) else 0.0,
        "gate_mb_cost_gap": float(mb.cost_gap) if (mb is not None and mb.ok) else 0.0,
        "gate_mb_confidence": float(mb.confidence) if (mb is not None and mb.ok) else 0.0,
        "gate_mb_bimodal": 1.0 if (mb is not None and mb.ok and mb.bimodal) else 0.0,
        "gate_mb_prefix_trust": float(mb.prefix_trust) if (mb is not None and mb.ok) else 0.0,
        "gate_pf_confidence": _family_confidence(pf_out.diagnostics),
        "gate_pf_spread": float(pf_out.diagnostics.get("branch_spread_mean", 0.0) or 0.0),
        "gate_pf_fallback": float(pf_fallback),
        "gate_beam_confidence": _family_confidence(beam_out.diagnostics),
        "gate_beam_spread": float(beam_out.diagnostics.get("branch_spread_mean", 0.0) or 0.0),
        "gate_beam_fallback": float(beam_fallback),
        "gate_track_disagreement": float(disagreement) if np.isfinite(disagreement) else 1e6,
        "gate_ms_ptp": float(ms.ptp) if ms.ok and np.isfinite(ms.ptp) else 1e6,
        "gate_ms_dominant_shift": float(ms.dominant_shift) if ms.ok else 0.0,
        "gate_ms_n_agree": float(ms.n_agree) if ms.ok else 0.0,
        "gate_ms_min_conf": float(ms.min_confidence) if ms.ok else 0.0,
        "gate_lgbm_oof_skill": float(np.clip(lgbm.info.oof_skill, 0.0, 1.0)) if lgbm else 0.0,
        "gate_cat_oof_skill": float(np.clip(cat.info.oof_skill, 0.0, 1.0)) if cat else 0.0,
        "gate_cand_pf": 1.0 if candidate == "pf" else 0.0,
        "gate_cand_beam": 1.0 if candidate == "beam" else 0.0,
        "gate_cand_mean": 1.0 if candidate == "pf_beam_mean" else 0.0,
        "gate_cand_mb": 1.0 if candidate == "mb_hedged" else 0.0,
        "gate_cand_lgbm": 1.0 if candidate == "lgbm_row" else 0.0,
        "gate_cand_cat": 1.0 if candidate == "cat_row" else 0.0,
    }
    return {k: float(v) for k, v in row.items()}


@dataclass
class StackGateThresholds:
    margin: float = 0.0
    conf_thr: float = 0.0
    sep_cap: float = np.inf
    shrink: float = 1.0
    warmup: int = 200
    tuned_on: int = 0
    reason: str = "default"


@dataclass
class StackGateDecision:
    outcome: str            # "applied_<candidate>" or "fallback"
    candidate: str | None
    reason: str
    confidence: float
    disagreement: float
    predicted_improvement: float
    n_eligible: int
    pseudo_delta: float = np.nan
    correction_mean_abs: float = 0.0
    correction_max_abs: float = 0.0
    shrink: float = 1.0
    warmup: int = 0


def _apply_pipeline(base: np.ndarray, corr: np.ndarray, shrink: float, warmup: int) -> np.ndarray:
    """Full correction pipeline: clip, shrink, warmup ramp. Pure function."""
    corr = np.where(np.isfinite(corr), corr, 0.0)
    out = base + shrink * corr * _ramp(base.size, warmup)
    return np.where(np.isfinite(out), out, base)


class GatedTrajectoryStack(BaselineModel):
    """The promotion candidate: Ridge anchor + gated, shrunken correction.

    A correction is applied to a fold-validation (or Test) well only when
    **all** rules pass; otherwise the well receives the exact Ridge Default
    prediction from the shared anchor instance (bit-identical to arm A):

    1. the candidate improves a visible-prefix pseudo-holdout (target-free);
    2. its confidence clears the fold-OOF-tuned threshold;
    3. its branch disagreement clears the fold-OOF-tuned cap;
    4. worst-decile pseudo-holdout squared error does not increase;
    5. the applied correction is bounded by the a-priori cap;
    6. the tuned policy improved pooled OOF on fold-training wells (kill
       switch — otherwise the gate is disabled for the whole fold);
    7. the final prediction is finite (any non-finite row → exact Ridge).
    """

    name = ARM_GATED
    needs_alignment = False

    def __init__(
        self,
        *,
        pf,
        beam,
        anchor_model: RidgeBaseline,
        config: TrajectoryGateConfig | None = None,
        protocol: str = "",
        fold: int = -1,
        decision_log: list | None = None,
        boost_kw: dict | None = None,
    ):
        self.pf = pf
        self.beam = beam
        self.anchor_model = anchor_model
        self.config = config or TrajectoryGateConfig()
        self.protocol = protocol
        self.fold = fold
        self.decision_log = decision_log if decision_log is not None else []
        self.boost_kw = boost_kw or {}
        self.learners_: dict[str, _BoostedResidual] = {}
        self.thresholds = StackGateThresholds()
        self.killed = False
        self.kill_reason = ""
        self.gate_model = None
        self.info = GateFitInfo(protocol=protocol, fold=fold)
        self._last_diagnostics: dict = {}
        self._oof_skills: dict[str, float] = {}

    # ------------------------------------------------------------- helpers
    def _boost_config(self, seed: int) -> dict:
        return dict(
            seed=seed,
            max_iter=self.config.boost_max_iter,
            estop_rounds=self.config.boost_estop_rounds,
            thread_count=self.config.boost_threads,
            device=self.config.device,
        )

    @staticmethod
    def _pseudo_stats(corr: np.ndarray, base: np.ndarray, truth: np.ndarray,
                      shrink: float, warmup: int) -> tuple[float, bool]:
        """Rule-1 delta and rule-4 tail flag for one candidate on a pseudo cut."""
        pred = _apply_pipeline(base, corr, shrink, warmup)
        base_rmse = _finite_rmse(base, truth)
        cand_rmse = _finite_rmse(pred, truth)
        if not (np.isfinite(base_rmse) and np.isfinite(cand_rmse)):
            return np.nan, False
        base_tail = _tail_mean_se(base, truth)
        cand_tail = _tail_mean_se(pred, truth)
        tail_ok = bool(
            np.isfinite(base_tail)
            and np.isfinite(cand_tail)
            and cand_tail <= base_tail + 1e-12
        )
        return float(base_rmse - cand_rmse), tail_ok

    # ---------------------------------------------------------- gate train
    def _examples_for_task(
        self,
        task_outer: WellTask,
        *,
        ridge_inner: RidgeBaseline,
        learners_inner: dict[str, _BoostedResidual],
    ) -> list[dict]:
        """OOF gate examples for one fold-training well."""
        cfg = self.config
        inp_outer = task_outer.inputs()
        nested = nested_pseudo_task(inp_outer, min_predict=cfg.min_pseudo_points)
        if nested is None:
            return []
        ref = TypewellReference(inp_outer.tw_tvt, inp_outer.tw_gr)
        cal = fit_prefix_affine_calibration(inp_outer, ref)
        mb = multibranch_scan(inp_outer, cal=cal, ref=ref)
        ms = multiscale_scan(inp_outer, cal=cal, ref=ref)
        pf_outer = self.pf.generate(inp_outer)
        beam_outer = self.beam.generate(inp_outer)

        base_outer = np.asarray(ridge_inner.predict(inp_outer), dtype="float64")
        cands_outer = trajectory_candidates(
            inp_outer, base_outer, pf=self.pf, beam=self.beam, mb=mb,
            learners=learners_inner, correction_cap=cfg.max_correction,
        )
        pseudo = nested.inputs
        base_pseudo = np.asarray(ridge_inner.predict(pseudo), dtype="float64")
        cands_pseudo = trajectory_candidates(
            pseudo, base_pseudo, pf=self.pf, beam=self.beam, mb=mb,
            learners=learners_inner, correction_cap=cfg.max_correction,
        )
        truth = np.asarray(nested.truth, dtype="float64")
        if not np.isfinite(truth).any():
            return []
        rows: list[dict] = []
        for name in STACK_CANDIDATES:
            cand_o = cands_outer.get(name)
            cand_p = cands_pseudo.get(name)
            if cand_o is None or cand_p is None:
                continue
            feats = stack_gate_feature_row(
                inp_outer, cal=cal, mb=mb, ms=ms, pf_out=pf_outer,
                beam_out=beam_outer, learners=learners_inner, candidate=name,
            )
            example = {
                "well_id": inp_outer.well_id,
                "candidate": name,
                "features": feats,
                "confidence": float(cand_o.confidence),
                "disagreement": float(cand_o.disagreement)
                if np.isfinite(cand_o.disagreement)
                else np.inf,
                "outer_available": bool(cand_o.available),
                "pseudo_available": bool(cand_p.available and cand_p.prediction is not None),
                "n_pseudo_points": int(np.isfinite(truth).sum()),
            }
            if example["pseudo_available"]:
                corr = np.asarray(cand_p.prediction, dtype="float64") - base_pseudo
                example["pseudo_corr"] = np.where(np.isfinite(corr), corr, 0.0)
                example["pseudo_base"] = base_pseudo
                example["pseudo_truth"] = truth
                d1, t1 = self._pseudo_stats(example["pseudo_corr"], base_pseudo, truth, 1.0, 0)
                example["delta_rmse_pseudo"] = d1
            else:
                example["delta_rmse_pseudo"] = np.nan
            rows.append(example)
        return rows

    def _build_oof_examples(self, train_tasks: list[WellTask]) -> list[dict]:
        ids = [t.well_id for t in train_tasks]
        assert_no_blocked_wells(ids, context="stack gate OOF training wells")
        inner = make_group_folds(ids, n_splits=self.config.inner_splits, seed=self.config.seed + 11)
        by_id = {t.well_id: t for t in train_tasks}
        examples: list[dict] = []
        skipped = 0
        se_l = {"lgbm": 0.0, "cat": 0.0}
        se_r = 0.0
        n_eval = 0
        # ------------------------------------------------------------ pass 1
        # Fit inner folds once; score the residual skills on the pooled inner
        # OOF rows. The inner skill aggregates are then attached to every
        # inner learner *before* gate examples are built (pass 2), so the
        # `gate_*_oof_skill` columns and residual-candidate confidences carry
        # the same quantity at train time that the full-fold learners carry
        # at inference time (no train/serve skew on those columns).
        fitted_by_fold: list = []
        for fold in inner:
            train_inner = [by_id[w] for w in fold.train_ids if w in by_id]
            valid_inner = [by_id[w] for w in fold.valid_ids if w in by_id]
            if not train_inner or not valid_inner:
                continue
            ridge_inner = RidgeBaseline(alignment_features=False)
            ridge_inner.fit(train_inner)
            learners_inner = build_residual_learners(
                use_lightgbm=self.config.use_lightgbm,
                use_catboost=self.config.use_catboost,
                **self._boost_config(self.config.seed + 100 + fold.index),
            )
            fitted_inner: dict[str, _BoostedResidual] = {}
            for name, learner in learners_inner.items():
                try:
                    learner.fit(train_inner)
                    fitted_inner[name] = learner
                except Exception:
                    continue
            for task in valid_inner:
                truth = task.target
                if truth is None:
                    continue
                inp = task.inputs()
                m = np.isfinite(truth)
                if not m.any():
                    continue
                base = np.asarray(ridge_inner.predict(inp), dtype="float64")
                se_r += float(np.sum((base[m] - truth[m]) ** 2))
                for name, learner in fitted_inner.items():
                    pred = np.asarray(learner.predict(inp), dtype="float64")
                    se_l[name] += float(np.sum((pred[m] - truth[m]) ** 2))
                n_eval += int(m.sum())
            fitted_by_fold.append((fold, ridge_inner, fitted_inner, valid_inner))
        self._oof_skills = {"lgbm": 0.0, "cat": 0.0}
        if n_eval > 0 and se_r > 0:
            r_ridge = float(np.sqrt(se_r / n_eval))
            if r_ridge > 0:
                for name in ("lgbm", "cat"):
                    if se_l.get(name, 0.0) > 0:
                        skill = (r_ridge - float(np.sqrt(se_l[name] / n_eval))) / r_ridge
                        self._oof_skills[name] = float(np.clip(skill, 0.0, 1.0))
        for _fold, _ridge, fitted_inner, _valid in fitted_by_fold:
            for name, learner in fitted_inner.items():
                learner.info.oof_skill = float(
                    np.clip(self._oof_skills.get(name, 0.0), 0.0, 1.0)
                )
        # ------------------------------------------------------------ pass 2
        for fold, ridge_inner, fitted_inner, valid_inner in fitted_by_fold:
            for task in valid_inner:
                try:
                    rows = self._examples_for_task(
                        task, ridge_inner=ridge_inner, learners_inner=fitted_inner
                    )
                except Exception:
                    rows = []
                if not rows:
                    skipped += 1
                examples.extend(rows)
        self.info.n_pseudo_skipped += skipped
        return examples

    @staticmethod
    def _eligible_example(e: dict, thr: StackGateThresholds, predicted: float) -> bool:
        return (
            e["outer_available"]
            and e["pseudo_available"]
            and np.isfinite(e.get("delta_rmse_pseudo", np.nan))
            and e["confidence"] >= thr.conf_thr
            and e["disagreement"] <= thr.sep_cap
            and predicted > thr.margin
        )

    def _policy_eval(self, examples: list[dict], thr: StackGateThresholds,
                     predictions: np.ndarray) -> tuple[float, float]:
        """Pooled pseudo OOF global-RMSE delta (policy − anchor) + activation."""
        by_well: dict[str, list[int]] = {}
        for i, e in enumerate(examples):
            by_well.setdefault(e["well_id"], []).append(i)
        se_pol = se_base = den = 0.0
        n_act = n_scored = 0
        for _w, idxs in by_well.items():
            e0 = examples[idxs[0]]
            if not e0["pseudo_available"]:
                continue
            base = e0["pseudo_base"]
            truth = e0["pseudo_truth"]
            m = np.isfinite(truth)
            if not m.any():
                continue
            n_scored += 1
            den += float(m.sum())
            se_base += float(np.sum((base[m] - truth[m]) ** 2))
            chosen = None
            best_pred = -np.inf
            for i in idxs:
                if not self._eligible_example(examples[i], thr, float(predictions[i])):
                    continue
                if float(predictions[i]) > best_pred:
                    best_pred = float(predictions[i])
                    chosen = i
            if chosen is None:
                se_pol += float(np.sum((base[m] - truth[m]) ** 2))
                continue
            e = examples[chosen]
            # Rule 1 + 4 under the tuned (shrink, warmup).
            delta, tail_ok = self._pseudo_stats(
                e["pseudo_corr"], base, truth, thr.shrink, thr.warmup
            )
            if not np.isfinite(delta) or delta <= 1e-12 or not tail_ok:
                se_pol += float(np.sum((base[m] - truth[m]) ** 2))
                continue
            pred = _apply_pipeline(base, e["pseudo_corr"], thr.shrink, thr.warmup)
            se_pol += float(np.sum((pred[m] - truth[m]) ** 2))
            n_act += 1
        if den == 0:
            return np.nan, 0.0
        delta = float(np.sqrt(se_pol / den) - np.sqrt(se_base / den))
        return delta, (n_act / max(n_scored, 1))

    def fit(self, tasks: list[WellTask], **kw) -> "GatedTrajectoryStack":
        from sklearn.ensemble import HistGradientBoostingRegressor

        t0 = time.perf_counter()
        if self.anchor_model.model is None:
            self.anchor_model.fit(tasks)

        # Residual learners fitted on the whole fold (inference-time tracks).
        self.learners_ = {}
        boost_learners = build_residual_learners(
            use_lightgbm=self.config.use_lightgbm,
            use_catboost=self.config.use_catboost,
            **self._boost_config(self.config.seed + 900),
        )
        for name, learner in boost_learners.items():
            try:
                learner.fit(tasks)
                self.learners_[name] = learner
            except Exception as exc:
                learner.info.available = False
                learner.info.unavailable_reason = f"fit_failed:{type(exc).__name__}"

        try:
            examples = self._build_oof_examples(tasks)
        except Exception as exc:
            self.killed = True
            self.kill_reason = f"gate_examples_failed:{type(exc).__name__}"
            self._record_fit(t0)
            return self
        # Attach the inner-OOF residual skills to the full-fold learners:
        # these are the target-free confidences their candidates carry.
        for name, learner in self.learners_.items():
            learner.info.oof_skill = float(np.clip(self._oof_skills.get(name, 0.0), 0.0, 1.0))
        usable = [
            e
            for e in examples
            if e["outer_available"] and e["pseudo_available"]
            and np.isfinite(e.get("delta_rmse_pseudo", np.nan))
        ]
        self.info.n_oof_wells = len({e["well_id"] for e in usable})
        self.info.n_examples = len(usable)
        if len(usable) < self.config.min_examples:
            self.killed = True
            self.kill_reason = "insufficient_oof_examples"
            self._record_fit(t0)
            return self

        X = pd.DataFrame([e["features"] for e in usable], columns=STACK_GATE_FEATURE_COLUMNS)
        assert_safe_features(X.columns, context="stack gate design matrix")
        validate_feature_frame(X)
        y = np.asarray([e["delta_rmse_pseudo"] for e in usable], dtype="float64")

        def _fit_gbdt(Xtr, ytr):
            model = HistGradientBoostingRegressor(
                max_depth=self.config.gbdt_max_depth,
                max_iter=self.config.gbdt_max_iter,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=self.config.seed,
            )
            model.fit(Xtr, ytr)
            return model

        # ---- threshold / shrink / warmup tuning on tune sub-folds -----------
        well_ids = sorted({e["well_id"] for e in usable})
        tune_folds = make_group_folds(well_ids, n_splits=self.config.tune_splits, seed=self.config.seed + 23)
        conf_pool = np.asarray([e["confidence"] for e in usable], dtype="float64")
        conf_pool = conf_pool[np.isfinite(conf_pool)]
        dis_pool = np.asarray(
            [e["disagreement"] if np.isfinite(e["disagreement"]) else np.nan for e in usable],
            dtype="float64",
        )
        dis_pool = dis_pool[np.isfinite(dis_pool)]
        conf_options = (
            sorted({0.0} | {float(np.quantile(conf_pool, q)) for q in self.config.confidence_levels})
            if conf_pool.size
            else [0.0]
        )
        sep_options = (
            sorted({float("inf")} | {float(np.quantile(dis_pool, q)) for q in self.config.disagreement_levels})
            if dis_pool.size
            else [float("inf")]
        )
        example_idx_by_well: dict[str, list[int]] = {}
        for i, e in enumerate(usable):
            example_idx_by_well.setdefault(e["well_id"], []).append(i)
        pred_cache: dict[tuple, tuple] = {}
        for fold in tune_folds:
            tr_idx = [i for w in fold.train_ids for i in example_idx_by_well.get(w, [])]
            va_idx = [i for w in fold.valid_ids for i in example_idx_by_well.get(w, [])]
            if not tr_idx or not va_idx:
                continue
            sub = _fit_gbdt(X.iloc[tr_idx], y[tr_idx])
            pred_cache[tuple(sorted(fold.valid_ids))] = (va_idx, sub.predict(X.iloc[va_idx]))

        best: tuple[float, StackGateThresholds] | None = None
        for margin in self.config.margins:
            for conf_thr in conf_options:
                for sep_cap in sep_options:
                    for shrink in self.config.shrink_grid:
                        for warmup in self.config.warmup_grid:
                            thr = StackGateThresholds(
                                margin=margin,
                                conf_thr=float(conf_thr),
                                sep_cap=float(sep_cap),
                                shrink=float(shrink),
                                warmup=int(warmup),
                                tuned_on=len(usable),
                                reason="tuned_subcf",
                            )
                            deltas, ok = [], True
                            for key, (va_idx, va_pred) in pred_cache.items():
                                sub_examples = [usable[i] for i in va_idx]
                                delta, _act = self._policy_eval(sub_examples, thr, va_pred)
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
            self._record_fit(t0)
            return self

        self.thresholds = best[1]
        self.gate_model = _fit_gbdt(X, y)
        pred_all = self.gate_model.predict(X)
        pooled, act = self._policy_eval(usable, self.thresholds, pred_all)
        self.info.pooled_oof_delta = float(pooled) if np.isfinite(pooled) else np.nan
        self.info.oof_activation_rate = float(act)
        if not np.isfinite(pooled) or pooled >= 0.0:
            self.killed = True
            self.kill_reason = "kill_switch_pooled_oof_degraded"
        self._record_fit(t0)
        return self

    def _record_fit(self, t0: float) -> None:
        self.info.killed = bool(self.killed)
        self.info.kill_reason = str(self.kill_reason)
        self.info.margin = float(self.thresholds.margin)
        self.info.conf_thr = float(self.thresholds.conf_thr)
        self.info.sep_cap = float(self.thresholds.sep_cap)
        self.info.fit_seconds = float(time.perf_counter() - t0)

    def _predict_improvements(
        self, task: InferenceTask, *, cal, mb, ms, pf_out, beam_out
    ) -> dict[str, float]:
        if self.gate_model is None:
            return {c: -np.inf for c in STACK_CANDIDATES}
        rows = [
            stack_gate_feature_row(
                task, cal=cal, mb=mb, ms=ms, pf_out=pf_out, beam_out=beam_out,
                learners=self.learners_, candidate=c,
            )
            for c in STACK_CANDIDATES
        ]
        X = pd.DataFrame(rows, columns=STACK_GATE_FEATURE_COLUMNS)
        assert_safe_features(X.columns, context="stack gate inference row")
        pred = self.gate_model.predict(X)
        return {c: float(p) for c, p in zip(STACK_CANDIDATES, pred)}

    def _log(self, task: InferenceTask, dec: StackGateDecision) -> None:
        self.decision_log.append(
            {
                "protocol": self.protocol,
                "fold": self.fold,
                "well_id": task.well_id,
                "outcome": dec.outcome,
                "candidate": dec.candidate or "",
                "reason": dec.reason,
                "confidence": float(dec.confidence),
                "disagreement": float(dec.disagreement) if np.isfinite(dec.disagreement) else np.nan,
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

    def predict(self, task: InferenceTask, feats=None) -> np.ndarray:
        base = np.asarray(self.anchor_model.predict(task, feats), dtype="float64")
        self._last_diagnostics = {}
        if self.killed or self.gate_model is None:
            self._log(task, StackGateDecision(
                "fallback", None, self.kill_reason or "gate_unfitted",
                0.0, np.inf, np.nan, 0,
            ))
            return base
        thr = self.thresholds
        try:
            ref = TypewellReference(task.tw_tvt, task.tw_gr)
            cal = fit_prefix_affine_calibration(task, ref)
            mb = multibranch_scan(task, cal=cal, ref=ref)
            ms = multiscale_scan(task, cal=cal, ref=ref)
            pf_out = self.pf.generate(task)
            beam_out = self.beam.generate(task)
            cands = trajectory_candidates(
                task, base, pf=self.pf, beam=self.beam, mb=mb,
                learners=self.learners_, correction_cap=self.config.max_correction,
            )
            predicted = self._predict_improvements(
                task, cal=cal, mb=mb, ms=ms, pf_out=pf_out, beam_out=beam_out
            )
        except Exception as exc:
            self._log(task, StackGateDecision(
                "fallback", None, f"decision_exception:{type(exc).__name__}",
                0.0, np.inf, np.nan, 0,
            ))
            return base

        # Rule 1 + 4 evidence: the nested visible-prefix pseudo-holdout.
        nested = nested_pseudo_task(task, min_predict=self.config.min_pseudo_points)
        pseudo_stats: dict[str, tuple[float, bool]] = {}
        if nested is not None:
            try:
                base_pseudo = np.asarray(self.anchor_model.predict(nested.inputs), dtype="float64")
                cands_pseudo = trajectory_candidates(
                    nested.inputs, base_pseudo, pf=self.pf, beam=self.beam, mb=mb,
                    learners=self.learners_, correction_cap=self.config.max_correction,
                )
                truth = np.asarray(nested.truth, dtype="float64")
                for name, cand_p in cands_pseudo.items():
                    if not cand_p.available or cand_p.prediction is None:
                        pseudo_stats[name] = (-np.inf, False)
                        continue
                    corr = np.asarray(cand_p.prediction, dtype="float64") - base_pseudo
                    pseudo_stats[name] = self._pseudo_stats(
                        np.where(np.isfinite(corr), corr, 0.0),
                        base_pseudo, truth, thr.shrink, thr.warmup,
                    )
            except Exception:
                pseudo_stats = {}

        eligible: list[str] = []
        reasons: dict[str, str] = {}
        for name in STACK_CANDIDATES:
            cand = cands.get(name)
            if cand is None or not cand.available or cand.prediction is None:
                reasons[name] = f"candidate_unavailable:{cand.failure_reason if cand else 'missing'}"
                continue
            if nested is None:
                reasons[name] = "pseudo_holdout_unavailable"
                continue
            delta, tail_ok = pseudo_stats.get(name, (np.nan, False))
            if not np.isfinite(delta) or delta <= 1e-12:
                reasons[name] = "pseudo_holdout_not_improved"       # rule 1
                continue
            if not tail_ok:
                reasons[name] = "worst_tail_risk_increased"          # rule 4
                continue
            if cand.confidence < thr.conf_thr:
                reasons[name] = "confidence_below_oof_threshold"     # rule 2
                continue
            if not np.isfinite(cand.disagreement) or cand.disagreement > thr.sep_cap:
                reasons[name] = "branch_disagreement_above_cap"      # rule 3
                continue
            if not np.isfinite(predicted.get(name, -np.inf)) or predicted[name] <= thr.margin:
                reasons[name] = "gbdt_expected_gain_below_margin"
                continue
            eligible.append(name)

        if not eligible:
            chosen_reason = ";".join(f"{k}:{v}" for k, v in reasons.items()) or "no_candidates"
            self._log(task, StackGateDecision(
                "fallback", None, chosen_reason,
                float(max((c.confidence for c in cands.values()), default=0.0)),
                float(max((c.disagreement for c in cands.values() if np.isfinite(c.disagreement)), default=np.nan)),
                float(max((p for p in predicted.values() if np.isfinite(p)), default=np.nan)),
                0,
            ))
            return base

        chosen = max(eligible, key=lambda c: predicted[c])
        cand = cands[chosen]
        corr = np.asarray(cand.prediction, dtype="float64") - base
        corr = np.clip(np.where(np.isfinite(corr), corr, 0.0), -self.config.max_correction,
                       self.config.max_correction)                       # rule 5
        pred = _apply_pipeline(base, corr, thr.shrink, thr.warmup)
        if not np.all(np.isfinite(pred)):                                # rule 7
            self._log(task, StackGateDecision(
                "fallback", None, "nonfinite_final_prediction",
                float(cand.confidence), float(cand.disagreement), float(predicted[chosen]),
                len(eligible),
            ))
            return base
        self._log(task, StackGateDecision(
            f"applied_{chosen}", chosen, "all_rules_passed",
            float(cand.confidence),
            float(cand.disagreement) if np.isfinite(cand.disagreement) else np.nan,
            float(predicted[chosen]),
            len(eligible),
            pseudo_delta=pseudo_stats.get(chosen, (np.nan, False))[0],
            correction_mean_abs=float(np.mean(np.abs(corr))),
            correction_max_abs=float(np.max(np.abs(corr))),
            shrink=float(thr.shrink),
            warmup=int(thr.warmup),
        ))
        return pred

    def prediction_diagnostics(self, task, feats, pred) -> dict:
        out = dict(self._last_diagnostics)
        if self.decision_log:
            dec = self.decision_log[-1]
            out.setdefault("gate_activation", dec["outcome"].startswith("applied_"))
            out.setdefault("gate_fallback_exact_ridge", dec["outcome"] == "fallback")
            out.setdefault("gate_confidence_threshold", float(self.thresholds.conf_thr))
            out.setdefault("gate_max_correction", float(self.config.max_correction))
            out.setdefault("gate_correction_magnitude", float(dec.get("correction_mean_abs", 0.0)))
            out.setdefault("alignment_ok", dec["outcome"].startswith("applied_"))
            out.setdefault("alignment_failure_reason", "" if dec["outcome"].startswith("applied_") else dec["reason"])
            out.setdefault("fallback_points", 0 if dec["outcome"].startswith("applied_") else int(task.n_predict))
            out.setdefault("fallback_fraction", 0.0 if dec["outcome"].startswith("applied_") else 1.0)
            out.setdefault("alignment_confidence_mean", float(dec.get("confidence", 0.0)))
        return out


# ==========================================================================
# Factories for the experiment runner
# ==========================================================================


def build_stack_models(
    arms: Iterable[str],
    *,
    anchor_model: RidgeBaseline,
    pf_factory,
    beam_factory,
    stack_config: StackConfig | None = None,
    gate_config: TrajectoryGateConfig | None = None,
    protocol: str = "",
    fold: int = -1,
    decision_log: list | None = None,
    boost_kw: dict | None = None,
    device: DeviceResolution | None = None,
) -> dict[str, BaselineModel]:
    """Build the requested arms, sharing the *same* anchor instance.

    Sharing guarantees the exact fallback: the anchor output used inside a
    gated arm is bit-identical to the scored ``ridge_default`` arm.
    """
    arms = tuple(dict.fromkeys(arms))
    models: dict[str, BaselineModel] = {}
    boost_kw = dict(boost_kw or {})
    # An explicit `device=` wins; otherwise honour a device already inside
    # boost_kw, else stay on the pure-CPU default.
    device = device or boost_kw.pop("device", None) or CPU_RESOLUTION
    boost_kw.pop("device", None)
    for arm in arms:
        if arm == ARM_RIDGE:
            models[arm] = anchor_model
        elif arm == ARM_LGBM:
            models[arm] = LightGBMResidual(device=device, **boost_kw)
        elif arm == ARM_CAT:
            models[arm] = CatBoostResidual(device=device, **boost_kw)
        elif arm == ARM_STACK:
            models[arm] = OOFMetaStackAnchor(
                anchor_model=anchor_model,
                config=stack_config,
            )
        elif arm == ARM_GATED:
            models[arm] = GatedTrajectoryStack(
                pf=pf_factory(),
                beam=beam_factory(),
                anchor_model=anchor_model,
                config=gate_config,
                protocol=protocol,
                fold=fold,
                decision_log=decision_log,
                boost_kw=boost_kw,
            )
        else:
            raise KeyError(f"unknown stack arm {arm!r}; known: {ARM_ORDER}")
    return models
