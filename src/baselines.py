"""The seven approved baselines.

Contract
--------
Every model implements::

    fit(tasks)     -> self     # tasks: list[WellTask], training wells only
    predict(task)  -> ndarray  # task: InferenceTask (no target reachable)

``predict`` receives an ``InferenceTask``, which has no target attribute, so a
model physically cannot read the answer. Feature matrices additionally pass
``assert_safe_features`` before any learned model consumes them, so a
train-only marker or the TVT column would raise rather than train.

Implemented baselines plus one explicitly isolated controlled experiment:

1. HoldLastTVT                         — constant continuation of the anchor
2. LinearExtrapolation                  — anchor + fitted prefix slope * dMD
3. GeometricProjection                  — TVT + Z projection with a fitted dip response
4. GRTypewellMatching                   — per-row search for the best-matching typewell TVT
5. NormalizedCrossCorrelation           — windowed NCC alignment, bias-corrected
6. RidgeBaseline                        — ridge on the anchored residual
7. LightGBMBaseline                     — gradient boosting on the anchored residual
8. DipConstrainedGRTypewellAlignment    — isolated GR/typewell A/B experiment

Models 6-7 learn the *residual* TVT - tvt_last, so hold-last is exactly the
zero prediction and any learned signal is an improvement over it by
construction.  Model 8 is intentionally not a Ridge feature or an ensemble:
it is compared directly with the unchanged Ridge baseline.
"""
from __future__ import annotations

import time
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    WellFeatures,
    build_features,
    calibrate_gr_to_reference,
    validate_feature_frame,
)
from src.tasks import InferenceTask, WellTask

try:  # optional dependency; the harness reports honestly when absent
    import lightgbm as lgb

    HAVE_LIGHTGBM = True
except Exception:  # pragma: no cover
    lgb = None
    HAVE_LIGHTGBM = False


# --------------------------------------------------------------------------


class BaselineModel:
    """Interface shared by every baseline."""

    name = "base"
    needs_alignment = False
    # Separate from the established NCC-style alignment because the
    # dip-constrained experiment must not change the Ridge feature matrix.
    needs_dip_alignment = False
    uses_spatial = False

    def fit(self, tasks: list[WellTask], **kw) -> "BaselineModel":
        return self

    def predict(self, task: InferenceTask, feats: WellFeatures | None = None) -> np.ndarray:
        raise NotImplementedError

    def prediction_diagnostics(
        self, task: InferenceTask, feats: WellFeatures | None, pred: np.ndarray
    ) -> dict:
        """Optional target-free prediction diagnostics for the report layer."""
        return {}

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _anchor(task: InferenceTask) -> float:
        a = task.anchor_tvt
        return float(a) if np.isfinite(a) else 0.0

    @staticmethod
    def _clip_to_typewell(task: InferenceTask, pred: np.ndarray) -> np.ndarray:
        """Keep predictions inside the reference section when one exists."""
        if task.tw_tvt is None:
            return pred
        finite = task.tw_tvt[np.isfinite(task.tw_tvt)]
        if finite.size < 2:
            return pred
        return np.clip(pred, finite.min(), finite.max())


# ------------------------------------------------------- 1. hold-last TVT --


class HoldLastTVT(BaselineModel):
    """Predict the last known TVT forever. Assumes the well stays in zone."""

    name = "hold_last"

    def predict(self, task, feats=None):
        return np.full(task.n_predict, self._anchor(task))


# ------------------------------------------------- 2. linear extrapolation --


@dataclass
class LinearExtrapolation(BaselineModel):
    """Continue the prefix's TVT trend, with a fitted damping factor.

    A raw slope extrapolated over thousands of feet diverges, so the slope is
    damped by a factor fitted on the training tasks. ``damping`` is the only
    learned parameter and it is fitted by 1-D search on fold-train wells only.
    """

    span: float = 300.0
    damping: float = 1.0
    max_span: float = 5000.0
    name: str = field(default="linear_extrap", init=False)

    def _slope(self, task: InferenceTask) -> float:
        s = task.start
        md, tvt = task.md[:s], task.tvt_known[:s]
        known = np.isfinite(tvt)
        if known.sum() < 3:
            return 0.0
        anchor_md = md[task.anchor_row] if task.anchor_row >= 0 else md[-1]
        w = known & (md >= anchor_md - self.span)
        if w.sum() < 3:
            w = known
        x, y = md[w], tvt[w]
        xd = x - x.mean()
        denom = float((xd * xd).sum())
        if denom <= 0:
            return 0.0
        return float((xd * (y - y.mean())).sum() / denom)

    def fit(self, tasks, **kw):
        grid = np.linspace(0.0, 1.0, 21)
        best, best_err = 1.0, np.inf
        cache = []
        for t in tasks:
            if t.target is None:
                continue
            inp = t.inputs()
            cache.append((self._slope(inp), inp.dmd, self._anchor(inp), t.target))
        if not cache:
            return self
        for d in grid:
            se = tot = 0.0
            for slope, dmd, anchor, y in cache:
                step = np.clip(dmd, 0, self.max_span)
                pred = anchor + d * slope * step
                se += float(np.nansum((pred - y) ** 2))
                tot += int(np.isfinite(y).sum())
            err = se / max(tot, 1)
            if err < best_err:
                best_err, best = err, float(d)
        self.damping = best
        return self

    def predict(self, task, feats=None):
        slope = self._slope(task)
        step = np.clip(task.dmd, 0, self.max_span)
        pred = self._anchor(task) + self.damping * slope * step
        return self._clip_to_typewell(task, pred)


# --------------------------------------------- 3. TVT + Z geometric project --


@dataclass
class GeometricProjection(BaselineModel):
    """Project TVT using the wellbore's own vertical movement.

    Physically: TVT changes when the bit's vertical movement differs from the
    stratigraphic surface's. Writing the surface's apparent dip along the hole
    as a constant ``dip``,

        TVT(md) = TVT_anchor - (dZ(md) - dip * dMD)

    ``beta`` scales how much of the observed dZ transfers into TVT (1.0 = the
    formation is flat and every foot of vertical movement crosses section);
    both parameters are fitted on fold-train wells only.
    """

    beta: float = 1.0
    dip: float = 0.0
    name: str = field(default="geom_projection", init=False)

    @staticmethod
    def _dz(task: InferenceTask) -> np.ndarray:
        row = task.anchor_row if task.anchor_row >= 0 else max(task.start - 1, 0)
        z0 = task.z[row]
        if not np.isfinite(z0):
            finite = np.flatnonzero(np.isfinite(task.z[: task.start]))
            z0 = task.z[finite[-1]] if finite.size else np.nan
        dz = task.z[task.start : task.stop] - z0
        return np.nan_to_num(dz)

    def _prefix_dip(self, task: InferenceTask) -> float:
        """Apparent dip implied by the prefix: d(TVT + Z)/dMD."""
        s = task.start
        md, tvt, z = task.md[:s], task.tvt_known[:s], task.z[:s]
        m = np.isfinite(tvt) & np.isfinite(z)
        if m.sum() < 10:
            return 0.0
        anchor_md = md[task.anchor_row] if task.anchor_row >= 0 else md[-1]
        w = m & (md >= anchor_md - 1000.0)
        if w.sum() < 10:
            w = m
        x = md[w]
        y = tvt[w] + z[w]
        xd = x - x.mean()
        denom = float((xd * xd).sum())
        if denom <= 0:
            return 0.0
        return float((xd * (y - y.mean())).sum() / denom)

    def fit(self, tasks, **kw):
        cache = []
        for t in tasks:
            if t.target is None:
                continue
            inp = t.inputs()
            cache.append(
                (self._dz(inp), inp.dmd, self._prefix_dip(inp), self._anchor(inp), t.target)
            )
        if not cache:
            return self
        best, best_err = (1.0, 0.0), np.inf
        for beta in np.linspace(0.0, 1.2, 13):
            for lam in (0.0, 0.5, 1.0):
                se = tot = 0.0
                for dz, dmd, pdip, anchor, y in cache:
                    pred = anchor - beta * dz + lam * pdip * dmd
                    se += float(np.nansum((pred - y) ** 2))
                    tot += int(np.isfinite(y).sum())
                err = se / max(tot, 1)
                if err < best_err:
                    best_err, best = err, (float(beta), float(lam))
        self.beta, self._lam = best[0], best[1]
        return self

    def predict(self, task, feats=None):
        lam = getattr(self, "_lam", 1.0)
        pred = (
            self._anchor(task)
            - self.beta * self._dz(task)
            + lam * self._prefix_dip(task) * task.dmd
        )
        return self._clip_to_typewell(task, pred)


# ----------------------------------------------- 4. GR / typewell matching --


@dataclass
class GRTypewellMatching(BaselineModel):
    """Match the lateral GR to the typewell GR, point by point.

    For each predicted row, search the reference log near the running estimate
    for the TVT whose GR (and local GR trend) best matches the observation, and
    penalise movement away from the previous row so the solution stays a
    continuous trajectory rather than a scatter of independent picks. Falls
    back to hold-last where GR is missing or the typewell is unusable.
    """

    search: float = 25.0
    continuity: float = 6.0
    smooth: int = 101
    name: str = field(default="gr_typewell_match", init=False)
    needs_alignment: bool = field(default=False, init=False)

    def fit(self, tasks, **kw):
        cache = []
        for t in tasks[:60]:
            if t.target is None:
                continue
            inp = t.inputs()
            feats = build_features(inp, alignment=False)
            if feats.ref.ok:
                cache.append((inp, feats, t.target))
        if not cache:
            return self
        best, best_err = self.continuity, np.inf
        for cont in (2.0, 6.0, 15.0, 40.0):
            self.continuity = cont
            se = tot = 0.0
            for inp, feats, y in cache:
                pred = self.predict(inp, feats)
                se += float(np.nansum((pred - y) ** 2))
                tot += int(np.isfinite(y).sum())
            err = se / max(tot, 1)
            if err < best_err:
                best_err, best = err, cont
        self.continuity = best
        return self

    def predict(self, task, feats=None):
        feats = feats or build_features(task, alignment=False)
        anchor = self._anchor(task)
        n = task.n_predict
        if not feats.ref.ok:
            return np.full(n, anchor)

        ref = feats.ref
        # Calibrate the lateral log into reference-GR space first: the two logs
        # are different tools in different holes, so comparing raw (or
        # separately standardised) values is meaningless. The calibration is
        # fitted on the known prefix only.
        signal = calibrate_gr_to_reference(task, ref, feats.gr["gr_z"])
        missing = feats.gr["gr_is_missing"] > 0.5
        gr_s = pd.Series(signal).rolling(self.smooth, center=True, min_periods=5).mean().to_numpy()

        grid, ref_z = ref.grid, ref.gr_z
        ref_s = pd.Series(ref_z).rolling(
            max(3, int(self.smooth * 0.25)), center=True, min_periods=1
        ).mean().to_numpy()

        out = np.empty(n)
        cur = anchor
        step = max(1, n // 400)  # decimate the search; interpolate between
        picks_idx, picks_val = [], []
        for i in range(0, n, step):
            row = task.start + i
            if missing[row] or not np.isfinite(gr_s[row]):
                picks_idx.append(i)
                picks_val.append(cur)
                continue
            lo = max(ref.tvt_min, cur - self.search)
            hi = min(ref.tvt_max, cur + self.search)
            if hi <= lo:
                picks_idx.append(i)
                picks_val.append(cur)
                continue
            cand = np.arange(lo, hi + ref.step, ref.step)
            vals = np.interp(cand, grid, ref_s)
            cost = (vals - gr_s[row]) ** 2 + ((cand - cur) / max(self.continuity, 1e-6)) ** 2
            cur = float(cand[int(np.argmin(cost))])
            picks_idx.append(i)
            picks_val.append(cur)

        if len(picks_idx) < 2:
            return np.full(n, anchor)
        out = np.interp(np.arange(n), np.asarray(picks_idx), np.asarray(picks_val))
        out = pd.Series(out).rolling(51, center=True, min_periods=1).mean().to_numpy()
        # Re-reference to the anchor: the search starts from the known TVT, so
        # its increment is the trustworthy part, while its absolute level
        # inherits any residual calibration error.
        out = anchor + (out - out[0])
        return self._clip_to_typewell(task, out)


# ------------------------------------- 5. normalized cross-correlation -----


@dataclass
class NormalizedCrossCorrelation(BaselineModel):
    """Windowed NCC of the lateral GR against the typewell GR.

    Uses the bias-corrected ``align_tvt`` track: the correlation is computed on
    windows, calibrated against the known prefix, then blended toward the
    anchor wherever the correlation score is weak, so a confident match is
    trusted and a poor one degrades to hold-last instead of hallucinating.
    """

    min_score: float = 0.35
    name: str = field(default="ncc_alignment", init=False)
    needs_alignment: bool = field(default=True, init=False)

    def fit(self, tasks, **kw):
        cache = []
        for t in tasks[:40]:
            if t.target is None:
                continue
            inp = t.inputs()
            feats = build_features(inp, alignment=True)
            if feats.align["_align_ok"]:
                cache.append((inp, feats, t.target))
        if not cache:
            return self
        best, best_err = self.min_score, np.inf
        for thr in (0.2, 0.35, 0.5, 0.65):
            self.min_score = thr
            se = tot = 0.0
            for inp, feats, y in cache:
                pred = self.predict(inp, feats)
                se += float(np.nansum((pred - y) ** 2))
                tot += int(np.isfinite(y).sum())
            err = se / max(tot, 1)
            if err < best_err:
                best_err, best = err, thr
        self.min_score = best
        return self

    def predict(self, task, feats=None):
        feats = feats if feats is not None else build_features(task, alignment=True)
        anchor = self._anchor(task)
        sl = slice(task.start, task.stop)
        if not feats.align["_align_ok"]:
            return np.full(task.n_predict, anchor)

        tvt = np.asarray(feats.align["align_tvt"][sl], dtype="float64")
        score = np.asarray(feats.align["align_score"][sl], dtype="float64")

        # Trust the correlation's *movement*, not its absolute level.
        #
        # The alignment is already bias-corrected against the known prefix, but
        # any residual level error is a constant offset applied to the whole
        # hidden region, which is exactly the error the anchor does not have.
        # So the track is re-referenced to its own value at the boundary and
        # only the increment since then is added to the anchor. A confident
        # correlation contributes its full movement; a weak one decays the
        # prediction back to hold-last.
        start_val = np.asarray(feats.align["align_tvt"], dtype="float64")[task.start]
        if not np.isfinite(start_val):
            return np.full(task.n_predict, anchor)
        delta = np.nan_to_num(tvt - start_val, nan=0.0)
        w = np.clip((score - self.min_score) / max(1e-6, 1.0 - self.min_score), 0.0, 1.0)
        pred = anchor + w * delta
        return self._clip_to_typewell(task, pred)


# ----------------------------------- 6. isolated dip-constrained A/B model --


@dataclass
class DipConstrainedGRTypewellAlignment(BaselineModel):
    """GR/typewell alignment constrained by visible-prefix apparent dip.

    This is a direct, inference-safe continuation model, not an additional
    Ridge feature and not an ensemble.  Its low-confidence or failed-match
    fallback is the X/Y/Z + visible-`TVT_input` dip projection computed in
    ``src.features``.  It never reads ``TVT``, a formation marker or Typewell
    Geology.
    """

    min_confidence: float = 0.20
    full_confidence: float = 0.65
    name: str = field(default="dip_constrained_alignment", init=False)
    needs_alignment: bool = field(default=False, init=False)
    needs_dip_alignment: bool = field(default=True, init=False)

    def predict(self, task, feats=None):
        # Direct callers may have built the standard feature bundle before this
        # experimental model was selected. Rebuild only in that case; the
        # cross-fitted evaluator supplies the shared dip bundle directly.
        if feats is None or feats.dip_align.get("failure_reason") == "not_requested":
            feats = build_features(task, alignment=False, dip_alignment=True)
        d = feats.dip_align
        sl = slice(task.start, task.stop)
        fallback = np.asarray(d["dip_prediction"][sl], dtype="float64")
        fallback = np.where(np.isfinite(fallback), fallback, self._anchor(task))
        if not d["ok"]:
            return self._clip_to_typewell(task, fallback)

        track = np.asarray(d["track"], dtype="float64")
        boundary = track[task.start] if task.start < track.size else np.nan
        if not np.isfinite(boundary):
            return self._clip_to_typewell(task, fallback)
        # Re-reference the GR correlation to the known boundary.  This admits
        # GR/typewell *movement* but cannot carry a latent absolute TVT offset
        # across Prediction Start.
        aligned = self._anchor(task) + np.nan_to_num(track[sl] - boundary, nan=0.0)
        confidence = np.asarray(d["confidence"][sl], dtype="float64")
        denom = max(self.full_confidence - self.min_confidence, 1e-6)
        weight = np.clip((confidence - self.min_confidence) / denom, 0.0, 1.0)
        pred = weight * aligned + (1.0 - weight) * fallback
        return self._clip_to_typewell(task, pred)

    def prediction_diagnostics(self, task, feats, pred) -> dict:
        if feats is None:
            return {}
        d = feats.dip_align
        sl = slice(task.start, task.stop)
        confidence = np.asarray(d["confidence"][sl], dtype="float64")
        fallback = (not bool(d["ok"])) | (confidence < self.min_confidence)
        return {
            "alignment_confidence_mean": float(np.nanmean(confidence)) if confidence.size else np.nan,
            "alignment_confidence_p10": float(np.nanquantile(confidence, 0.10)) if confidence.size else np.nan,
            "alignment_ok": bool(d["ok"]),
            "alignment_failure_reason": str(d["failure_reason"]),
            "alignment_cache_hit": bool(d.get("cache_hit", False)),
            "fallback_points": int(np.count_nonzero(fallback)),
            "fallback_fraction": float(np.mean(fallback)) if confidence.size else np.nan,
            "dip_fit_r2": float(d["dip_r2"]),
        }


# ----------------------------------------------------------- 6/7 learned ---


def _design_matrix(feats: WellFeatures, task: InferenceTask) -> pd.DataFrame:
    frame = feats.frame()
    validate_feature_frame(frame)
    return frame


class _LearnedBaseline(BaselineModel):
    """Shared plumbing: build X/y on the anchored residual, then fit."""

    max_rows_per_well = 400

    def __init__(self, *, spatial: "object | None" = None):
        self.spatial = spatial
        self.feature_names_: list[str] = []
        self.uses_spatial = spatial is not None

    def _features(self, task: InferenceTask, feats: WellFeatures | None) -> pd.DataFrame:
        feats = feats if feats is not None else build_features(task, alignment=self.needs_alignment)
        X = _design_matrix(feats, task)
        if self.spatial is not None:
            extra = self.spatial.features_for(task)
            X = pd.concat([X.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        return X

    def prediction_diagnostics(self, task, feats, pred) -> dict:
        """Expose the existing GR/typewell alignment quality for error analysis.

        This is reporting-only; it does not change Ridge's design matrix or
        prediction.  ``align_score`` is the calibrated match discriminability
        used by the established NCC feature, while the GR correlation is a
        separate visible-prefix diagnostic.
        """
        if feats is None:
            return {}
        score = np.asarray(feats.align["align_score"][task.start : task.stop], dtype="float64")
        return {
            "alignment_confidence_mean": float(np.nanmean(score)) if score.size else np.nan,
            "alignment_confidence_p10": float(np.nanquantile(score, 0.10)) if score.size else np.nan,
            "alignment_ok": bool(feats.align.get("_align_ok", False)),
            "alignment_failure_reason": "" if feats.align.get("_align_ok", False) else "ncc_alignment_unavailable",
            "typewell_gr_correlation": float(feats.typewell_gr_prefix_correlation),
        }

    def _training_arrays(self, tasks: list[WellTask]):
        Xs, ys, groups = [], [], []
        rng = np.random.default_rng(0)
        for t in tasks:
            if t.target is None:
                continue
            inp = t.inputs()
            feats = build_features(inp, alignment=self.needs_alignment)
            X = self._features(inp, feats)
            y = np.asarray(t.target, dtype="float64") - self._anchor(inp)
            m = np.isfinite(y)
            if not m.any():
                continue
            X, y = X[m], y[m]
            if len(X) > self.max_rows_per_well:
                pick = rng.choice(len(X), self.max_rows_per_well, replace=False)
                pick.sort()
                X, y = X.iloc[pick], y[pick]
            Xs.append(X)
            ys.append(y)
            groups.append(np.full(len(X), t.well_id))
        if not Xs:
            return None, None, None
        X = pd.concat(Xs, ignore_index=True)
        y = np.concatenate(ys)
        g = np.concatenate(groups)
        self.feature_names_ = list(X.columns)
        return X, y, g


@dataclass(init=False)
class RidgeBaseline(_LearnedBaseline):
    """Ridge regression on the anchored residual."""

    name = "ridge"
    needs_alignment = True

    def __init__(self, alpha: float = 10.0, *, spatial=None):
        super().__init__(spatial=spatial)
        self.alpha = alpha
        self.model = None
        self.medians_: pd.Series | None = None

    def _clean(self, X: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None:
            self.medians_ = X.median(numeric_only=True)
        filled = X.fillna(self.medians_).fillna(0.0)
        return filled.to_numpy(dtype="float64")

    def fit(self, tasks, **kw):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X, y, _ = self._training_arrays(tasks)
        if X is None:
            return self
        self.medians_ = X.median(numeric_only=True)
        Xv = self._clean(X)
        self.scaler_ = StandardScaler().fit(Xv)
        self.model = Ridge(alpha=self.alpha).fit(self.scaler_.transform(Xv), y)
        return self

    def predict(self, task, feats=None):
        anchor = self._anchor(task)
        if self.model is None:
            return np.full(task.n_predict, anchor)
        X = self._features(task, feats)
        X = X.reindex(columns=self.feature_names_)
        resid = self.model.predict(self.scaler_.transform(self._clean(X)))
        return self._clip_to_typewell(task, anchor + resid)


@dataclass(init=False)
class LightGBMBaseline(_LearnedBaseline):
    """LightGBM on the anchored residual."""

    name = "lightgbm"
    needs_alignment = True

    def __init__(self, *, spatial=None, **params):
        super().__init__(spatial=spatial)
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "num_threads": 0,
            "seed": 0,
        }
        self.params.update(params)
        self.num_boost_round = int(self.params.pop("num_boost_round", 400))
        self.model = None

    def fit(self, tasks, **kw):
        if not HAVE_LIGHTGBM:
            raise RuntimeError(
                "lightgbm is not installed; the harness records this model as "
                "unavailable rather than substituting a different one"
            )
        X, y, _ = self._training_arrays(tasks)
        if X is None:
            return self
        ds = lgb.Dataset(X, label=y, free_raw_data=True)
        requested = os.environ.get("ROGII_LIGHTGBM_DEVICE", "cpu")
        params = dict(self.params)
        if requested == "gpu":
            params["device_type"] = "gpu"
        try:
            self.model = lgb.train(params, ds, num_boost_round=self.num_boost_round)
            self.execution_mode = requested
            self.gpu_fallback_reason = ""
        except Exception as exc:
            if requested != "gpu":
                raise
            # Kaggle images often ship a CPU-only wheel. Retry transparently
            # and retain the reason for the runtime report/log.
            self.gpu_fallback_reason = f"LightGBM GPU unavailable: {type(exc).__name__}: {exc}"
            self.model = lgb.train(self.params, ds, num_boost_round=self.num_boost_round)
            self.execution_mode = "cpu_fallback"
        return self

    def predict(self, task, feats=None):
        anchor = self._anchor(task)
        if self.model is None:
            return np.full(task.n_predict, anchor)
        X = self._features(task, feats).reindex(columns=self.feature_names_)
        resid = self.model.predict(X)
        return self._clip_to_typewell(task, anchor + np.asarray(resid, dtype="float64"))

    def importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame(columns=["feature", "gain"])
        return (
            pd.DataFrame(
                {
                    "feature": self.model.feature_name(),
                    "gain": self.model.feature_importance("gain"),
                }
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )


# --------------------------------------------------------------- registry --

BASELINES = {
    "hold_last": HoldLastTVT,
    "linear_extrap": LinearExtrapolation,
    "geom_projection": GeometricProjection,
    "gr_typewell_match": GRTypewellMatching,
    "ncc_alignment": NormalizedCrossCorrelation,
    "dip_constrained_alignment": DipConstrainedGRTypewellAlignment,
    "ridge": RidgeBaseline,
    "lightgbm": LightGBMBaseline,
}

BASELINE_ORDER = list(BASELINES)


def build_baseline(name: str, **kw) -> BaselineModel:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; known: {BASELINE_ORDER}")
    return BASELINES[name](**kw)


def predict_with_timing(model: BaselineModel, task: InferenceTask, feats=None):
    t0 = time.perf_counter()
    pred = model.predict(task, feats)
    return np.asarray(pred, dtype="float64"), time.perf_counter() - t0
