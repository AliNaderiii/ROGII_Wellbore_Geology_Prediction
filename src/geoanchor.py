"""Controlled GeoAnchor experiment — arms A–E.

This module implements the *idea-level* patterns studied from four public
ROGII notebooks ("ROGII GeoAnchor", "hahaha det agi", "rogii-shift-275" and
"Well-Level GBDT Gate") **inside this repository's own leakage-safe
architecture**. No public notebook code or artifacts are copied or imported;
only the following whitelisted ideas are implemented, from scratch:

1. Affine calibration of horizontal GR into Typewell GR space
   (``AffineCalibrationFeatureGenerator``, arm B).
2. Multi-branch GR/Typewell alignment via a constant datum-shift scan
   (``MultiBranchFeatureGenerator``, arm C).
3. Bimodal branch uncertainty features — branch separation, cost gap and a
   trust-shrunk effective branch probability (same generator, arms C/D).
4. Prefix pseudo-holdout validation — every candidate correction is checked
   empirically against ``TVT_input`` on a nested masked window that lies
   strictly inside the visible prefix (``nested_pseudo_task``), never against
   a hidden label.
5. Well-level confidence gating — a HistGradientBoosting gate
   (``WellLevelGate``) trained **only** from fold-training wells, with strict
   inner cross-fitting producing the OOF prefix diagnostics it learns from.
6. Ridge Default stays the stable anchor prediction: the arm-E gated model
   predicts its own Ridge Default output whenever the gate declines, and the
   fold-level kill switch restores it for the whole fold (arm E).
7. PF/Beam as optional candidate corrections only — the repository's existing
   target-free ``pf_shift``/``beam_shift`` tracks are offered as bounded
   corrections to the anchor, never as replacements or ensemble branches
   (``generate_candidate_corrections``).

Explicitly NOT used here (per the experiment's pre-registration): train target
lookup for test rows, any duplicate-well shortcut, Typewell Geology, formation
markers, external artifacts, Koolbox, public-leaderboard scores, hidden TVT as
a feature, and public-LB-tuned constants. Every numeric constant below is an
a-priori algorithmic sanity bound documented at its definition; every
*decision* threshold is tuned per fold from fold-training wells only (OOF).

Arms
----
A  ``ridge_default``             Ridge Default (unchanged anchor)
B  ``ridge_affine_cal``          Ridge + affine GR calibration features
C  ``ridge_multibranch``         Ridge + multi-branch / bimodal features
D  ``ridge_affine_multibranch``  Ridge + B and C feature sets
E  ``ridge_gated_gbdt``          Ridge anchor + well-level GBDT gate applying
                                 bounded PF/Beam candidate corrections when —
                                 and only when — every safety rule passes

A correction is applied only when ALL of the following hold (see
``WellLevelGate`` / ``GatedRidgeAnchor``):

* the candidate improves a visible-prefix pseudo-holdout (target-free);
* alignment confidence exceeds the (fold-OOF-tuned) threshold;
* branch disagreement is acceptable (fold-OOF-tuned cap);
* worst-tail risk does not increase on the pseudo-holdout; and
* the fold's pooled OOF policy does not degrade the fold validation metric —
  otherwise the gate is disabled for the entire fold (kill switch).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.baselines import BaselineModel, RidgeBaseline
from src.features import (
    TypewellReference,
    _rolling,
    interpolate_within_well,
)
from src.manifest import assert_safe_features
from src.particle_filter import PathFeatureOutput
from src.tasks import MIN_PREFIX_ROWS, InferenceTask, WellTask
from src.validation import (
    CrossFitLeakage,
    assert_no_blocked_wells,
    evaluate_models,
    make_group_folds,
)

GEOANCHOR_VERSION = "geoanchor-controlled-experiment-v1"

# --------------------------------------------------------------------------
# Arm registry
# --------------------------------------------------------------------------

ARM_A = "ridge_default"
ARM_B = "ridge_affine_cal"
ARM_C = "ridge_multibranch"
ARM_D = "ridge_affine_multibranch"
ARM_E = "ridge_gated_gbdt"

ARM_ORDER = (ARM_A, ARM_B, ARM_C, ARM_D, ARM_E)
DEFAULT_ARM = ARM_A

ARM_LABELS = {
    ARM_A: "A. Ridge Default (stable anchor)",
    ARM_B: "B. Ridge + affine GR calibration features",
    ARM_C: "C. Ridge + bimodal branch features",
    ARM_D: "D. Ridge + affine calibration + branch uncertainty",
    ARM_E: "E. Ridge anchor + well-level GBDT gate for candidate corrections",
}

#: A-priori bound on any applied candidate correction (ft of TVT). Matches the
#: repository's established GR/typewell search radius (25.0 in
#: ``GRTypewellMatching``). It is a physical sanity bound, not a tuned knob.
CORRECTION_CAP_FT = 25.0

#: Minimum fraction of observed (non-fallback) rows in the early prediction
#: region for a PF/Beam track to qualify as a candidate correction. A track
#: built almost entirely on the geometry prior carries no GR evidence; the
#: bound was fixed a priori, not tuned.
CORRECTION_MIN_OBSERVED_FRAC = 0.25

#: PF/Beam *soft* failure: rows inside GR outages follow the geometry prior
#: while measured rows still inform the track. That track may be a candidate.
#: Any other failure reason is a hard failure (no usable track at all).
_PATH_SOFT_FAILURES = frozenset({"", "partial_missing_or_invalid_horizontal_gr"})

#: Nested pseudo-holdout construction limits (mirror src.tasks masked mode).
NESTED_MIN_PREFIX = MIN_PREFIX_ROWS
NESTED_MIN_PREDICT = 25


# ==========================================================================
# Idea 1 — affine calibration of horizontal GR into Typewell GR space
# ==========================================================================


@dataclass(frozen=True)
class AffineCalibrationConfig:
    """A-priori sanity bounds for the prefix-only affine GR calibration.

    ``alpha``/``beta`` bounds exist only to reject unphysical fits (a negative
    or exploding gain, a calibration offset larger than any observed GR); they
    were fixed before any data was seen and are never tuned.
    """

    min_prefix_rows: int = 40
    alpha_min: float = 0.25
    alpha_max: float = 4.0
    beta_abs_max: float = 500.0


@dataclass
class AffineCalibration:
    """Result of fitting ``G_hw ≈ alpha * G_tw(TVT_input) + beta`` on the prefix.

    Reads ``tvt_known`` strictly below ``task.start``; the hidden region never
    contributes to ``alpha``/``beta``.
    """

    ok: bool
    alpha: float = np.nan
    beta: float = np.nan
    fit_rmse_z: float = np.nan
    prefix_corr: float = np.nan
    failure_reason: str = ""
    mu_tw: float = np.nan
    sd_tw: float = np.nan


def fit_prefix_affine_calibration(
    task: InferenceTask,
    ref: TypewellReference | None = None,
    *,
    config: AffineCalibrationConfig | None = None,
) -> AffineCalibration:
    """Prefix-only affine map of the lateral GR onto the typewell GR.

    Least squares on rows where ``TVT_input`` is known, comparing the measured
    horizontal GR with the typewell GR sampled at the *known* TVT (a reference
    coordinate, never the hidden label). Quality bounds reject nonsensical
    fits so downstream features degrade to neutral values instead of carrying
    an exploded calibration.
    """
    config = config or AffineCalibrationConfig()
    ref = ref or TypewellReference(task.tw_tvt, task.tw_gr)
    if not ref.ok:
        return AffineCalibration(ok=False, failure_reason="missing_or_invalid_typewell")
    s = task.start
    gr_obs = np.asarray(task.gr[:s], dtype="float64")
    known = np.isfinite(task.tvt_known[:s]) & np.isfinite(gr_obs)
    if int(known.sum()) < config.min_prefix_rows:
        return AffineCalibration(ok=False, failure_reason="insufficient_prefix_gr_rows")
    tvt_known = task.tvt_known[:s][known]
    ref_gr = np.interp(tvt_known, ref.grid, ref.gr)
    obs = gr_obs[known]
    ok = np.isfinite(ref_gr)
    if int(ok.sum()) < config.min_prefix_rows:
        return AffineCalibration(ok=False, failure_reason="typewell_gr_not_defined_on_prefix_tvt")
    ref_gr, obs = ref_gr[ok], obs[ok]
    A = np.column_stack([ref_gr, np.ones_like(ref_gr)])
    try:
        alpha, beta = np.linalg.lstsq(A, obs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return AffineCalibration(ok=False, failure_reason="affine_lstsq_failed")
    alpha, beta = float(alpha), float(beta)
    if not (
        np.isfinite(alpha)
        and np.isfinite(beta)
        and config.alpha_min <= alpha <= config.alpha_max
        and abs(beta) <= config.beta_abs_max
    ):
        return AffineCalibration(
            ok=False, alpha=alpha, beta=beta, failure_reason="affine_bounds_rejected"
        )
    resid = obs - (alpha * ref_gr + beta)
    fit_rmse_raw = float(np.sqrt(np.mean(resid**2)))
    sd_tw = float(ref.sd) if ref.sd > 1e-9 else 1.0
    fit_rmse_z = fit_rmse_raw / sd_tw
    if float(np.std(obs)) > 1e-9 and float(np.std(ref_gr)) > 1e-9:
        prefix_corr = float(np.corrcoef(obs, ref_gr)[0, 1])
    else:
        prefix_corr = np.nan
    return AffineCalibration(
        ok=True,
        alpha=alpha,
        beta=beta,
        fit_rmse_z=float(fit_rmse_z),
        prefix_corr=prefix_corr,
        failure_reason="",
        mu_tw=float(ref.mu),
        sd_tw=sd_tw,
    )


def calibrated_gr_signal(
    task: InferenceTask, cal: AffineCalibration
) -> tuple[np.ndarray, np.ndarray]:
    """Full-well calibrated signal ``((GR - beta)/alpha - mu_tw)/sd_tw``.

    Returns ``(signal, was_missing)``. Within-well interpolation fills GR
    outages (per the repository's documented within-well-only rule); the
    ``was_missing`` mask lets callers down-weight interpolated rows.
    """
    filled, missing = interpolate_within_well(task.gr)
    if not cal.ok:
        return np.zeros(task.n_rows, dtype="float64"), missing
    sd = cal.sd_tw if np.isfinite(cal.sd_tw) and cal.sd_tw > 1e-9 else 1.0
    signal = ((filled - cal.beta) / max(cal.alpha, 1e-12) - cal.mu_tw) / sd
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0), missing


AFFINE_CAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "acal_alpha",
    "acal_beta",
    "acal_fit_rmse",
    "acal_prefix_corr",
    "acal_ok",
    "acal_gr_tw_z",
    "acal_roll_mean_51",
    "acal_roll_std_51",
    "acal_gr_grad",
)


class AffineCalibrationFeatureGenerator:
    """Arm-B feature set: the calibration coefficients and the calibrated log.

    Prediction-region columns (all broadcast scalars plus row-level calibrated
    GR statistics). Every quantity is computable at inference time from GR,
    Typewell GR/TVT and the visible ``TVT_input`` prefix only.
    """

    feature_columns = AFFINE_CAL_FEATURE_COLUMNS

    def __init__(self, config: AffineCalibrationConfig | None = None) -> None:
        self.config = config or AffineCalibrationConfig()

    def generate(self, task: InferenceTask) -> PathFeatureOutput:
        assert_safe_features(self.feature_columns, context="arm-B affine calibration frame")
        ref = TypewellReference(task.tw_tvt, task.tw_gr)
        cal = fit_prefix_affine_calibration(task, ref, config=self.config)
        signal, missing = calibrated_gr_signal(task, cal)
        if cal.ok:
            roll_mean = _rolling(signal, 51, "mean")
            roll_std = _rolling(signal, 51, "std")
            # Local per-foot gradient of the calibrated log; interpolated
            # (imputed) rows contribute nothing, so mark-free gradient is an
            # honest continuation rather than an invented observation.
            grad = np.gradient(_rolling(signal, 51, "mean"), task.md)
            grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            roll_mean = np.zeros(task.n_rows)
            roll_std = np.zeros(task.n_rows)
            grad = np.zeros(task.n_rows)
        sl = slice(task.start, task.stop)
        scalars = {
            "acal_alpha": cal.alpha if cal.ok else 0.0,
            "acal_beta": cal.beta if cal.ok else 0.0,
            "acal_fit_rmse": cal.fit_rmse_z if cal.ok else 0.0,
            "acal_prefix_corr": (
                cal.prefix_corr if cal.ok and np.isfinite(cal.prefix_corr) else 0.0
            ),
            "acal_ok": 1.0 if cal.ok else 0.0,
        }
        frame = pd.DataFrame(
            {
                "acal_alpha": scalars["acal_alpha"],
                "acal_beta": scalars["acal_beta"],
                "acal_fit_rmse": scalars["acal_fit_rmse"],
                "acal_prefix_corr": scalars["acal_prefix_corr"],
                "acal_ok": scalars["acal_ok"],
                "acal_gr_tw_z": signal[sl],
                "acal_roll_mean_51": roll_mean[sl],
                "acal_roll_std_51": roll_std[sl],
                "acal_gr_grad": grad[sl],
            },
            index=np.arange(task.start, task.stop),
            dtype="float64",
        )
        diagnostics = {
            "affine_ok": bool(cal.ok),
            "affine_failure_reason": cal.failure_reason,
            "affine_alpha": float(cal.alpha) if cal.ok else np.nan,
            "affine_beta": float(cal.beta) if cal.ok else np.nan,
            "affine_fit_rmse_z": float(cal.fit_rmse_z) if cal.ok else np.nan,
            "affine_prefix_corr": float(cal.prefix_corr)
            if np.isfinite(cal.prefix_corr)
            else np.nan,
        }
        return PathFeatureOutput(frame=frame, diagnostics=diagnostics)


# ==========================================================================
# Ideas 2+3 — multi-branch datum scan with bimodal uncertainty
# ==========================================================================


@dataclass(frozen=True)
class MultiBranchConfig:
    """A-priori constants for the constant-datum GR/Typewell alignment scan.

    The scan asks a single, tightly-bounded question: "is there a constant
    TVT offset of the anchor trajectory for which the measured hidden GR
    matches the typewell GR better?" All constants are algorithmic sanity
    choices fixed before any run; none was tuned on any leaderboard.
    """

    search: float = 15.0          # ft of TVT scanned around the anchor
    step: float = 0.75            # scan grid step (ft)
    window: int = 600             # hidden GR rows contributing to the cost
    prefix_tail: int = 600        # prefix rows used for the trust diagnostic
    min_gr_rows: int = 50         # measured hidden GR rows required to scan
    min_prefix_tail_rows: int = 100
    min_branch_sep: float = 3.0   # ft separating two candidate branch minima
    bimodal_cost_margin: float = 0.10  # branch 2 must cost <= J1 * (1 + margin)
    clip: float = 6.0             # clipped-residual scale (z units)
    min_scale: float = 0.25       # floor of the prefix residual scale (z units)


@dataclass
class MultiBranchResult:
    ok: bool
    failure_reason: str = ""
    shift1: float = 0.0
    shift2: float = 0.0
    sep: float = 0.0
    bimodal: bool = False
    cost_gap: float = 0.0
    w1: float = 0.5
    confidence: float = 0.0
    prefix_trust: float = 0.0
    j_curve: np.ndarray | None = None
    deltas: np.ndarray | None = None


def _scan_cost_curve(
    signal_rows: np.ndarray,
    tvt_rows: np.ndarray,
    deltas: np.ndarray,
    ref: TypewellReference,
    scale: float,
    clip: float,
) -> np.ndarray:
    """Mean clipped squared residual of the signal against the shifted typewell.

    ``tvt_rows`` holds the base-path TVT of each contributing row (known prefix
    TVT, or the hold-last anchor on hidden rows). For each candidate datum the
    typewell GR is sampled at ``tvt_rows + delta``. This is a GR-to-GR
    comparison and reads no TVT label on hidden rows.
    """
    n = signal_rows.size
    out = np.full(deltas.size, np.nan)
    if n == 0:
        return out
    for i, d in enumerate(deltas):
        want = np.interp(tvt_rows + d, ref.grid, ref.gr_z)
        r = np.clip((signal_rows - want) / scale, -clip, clip)
        out[i] = float(np.mean(r * r))
    return out


def multibranch_scan(
    task: InferenceTask,
    *,
    cal: AffineCalibration | None = None,
    ref: TypewellReference | None = None,
    config: MultiBranchConfig | None = None,
) -> MultiBranchResult:
    """Constant datum scan with a two-branch (bimodal) uncertainty summary.

    Branches are the two best-separated cost minima. The effective branch
    probability is shrunk toward 0.5 by the **prefix trust** diagnostic: a
    well where the same scan, run wholly inside the visible prefix (where the
    true shift is zero by construction), hallucinates a large shift is not
    trusted to produce a sharp branch decision on the hidden suffix.
    """
    config = config or MultiBranchConfig()
    ref = ref or TypewellReference(task.tw_tvt, task.tw_gr)
    if not ref.ok:
        return MultiBranchResult(ok=False, failure_reason="missing_or_invalid_typewell")
    anchor = task.anchor_tvt
    if not np.isfinite(anchor):
        return MultiBranchResult(ok=False, failure_reason="no_visible_anchor")
    if cal is None or not cal.ok:
        cal_local = fit_prefix_affine_calibration(task, ref)
    else:
        cal_local = cal
    if not cal_local.ok:
        return MultiBranchResult(ok=False, failure_reason="affine_calibration_unusable")

    signal, missing = calibrated_gr_signal(task, cal_local)
    s = task.start

    # --- noise scale and trust from the visible prefix only --------------
    known = np.isfinite(task.tvt_known[:s]) & ~(missing[:s])
    want_prefix = np.interp(task.tvt_known[:s][known], ref.grid, ref.gr_z)
    sig_prefix = signal[:s][known]
    if want_prefix.size < config.min_prefix_tail_rows:
        return MultiBranchResult(ok=False, failure_reason="insufficient_prefix_observations")
    resid_p = sig_prefix - want_prefix
    med = float(np.median(resid_p))
    scale = float(np.median(np.abs(resid_p - med))) * 1.4826
    if not np.isfinite(scale) or scale < config.min_scale:
        scale = config.min_scale

    deltas = np.arange(-config.search, config.search + 0.5 * config.step, config.step)

    # Trust: rerun the same scan on the prefix tail, where truth == zero shift.
    tail_rows = np.flatnonzero(known)[-config.prefix_tail :]
    if tail_rows.size >= config.min_prefix_tail_rows:
        j_p = _scan_cost_curve(
            signal[:s][tail_rows],
            task.tvt_known[:s][tail_rows],
            deltas,
            ref,
            scale,
            config.clip,
        )
        if np.isfinite(j_p).any():
            d_hat = float(deltas[int(np.nanargmin(j_p))])
            prefix_trust = float(np.clip(1.0 - abs(d_hat) / max(0.5 * config.search, 1e-9), 0.0, 1.0))
        else:
            prefix_trust = 0.0
    else:
        prefix_trust = 0.0

    # --- hidden-region scan against the hold-last anchor ------------------
    hi = min(task.stop, s + config.window)
    rows = np.arange(s, hi)
    measured = ~(missing[s:hi])
    rows = rows[measured]
    if rows.size < config.min_gr_rows:
        return MultiBranchResult(ok=False, failure_reason="insufficient_hidden_gr_rows")
    j = _scan_cost_curve(
        signal[rows], np.full(rows.size, anchor), deltas, ref, scale, config.clip
    )
    finite = np.isfinite(j)
    if not finite.any():
        return MultiBranchResult(ok=False, failure_reason="scan_cost_all_nan")

    i1 = int(np.nanargmin(j))
    j1 = float(j[i1])
    shift1 = float(deltas[i1])
    pool = j[finite]
    typical = float(np.median(pool))
    temperature = max(typical, 1e-9)
    confidence = float(np.clip(1.0 - j1 / temperature, 0.0, 1.0))

    sep_mask = np.abs(deltas - shift1) >= config.min_branch_sep
    cand2 = sep_mask & finite
    if cand2.any():
        i2 = int(np.nanargmin(np.where(cand2, j, np.inf)))
        j2 = float(j[i2])
        shift2 = float(deltas[i2])
        bimodal = bool(j2 <= j1 * (1.0 + config.bimodal_cost_margin))
    else:
        j2, shift2, bimodal = np.inf, shift1, False

    sep = float(abs(shift2 - shift1)) if bimodal else 0.0
    if bimodal:
        w1_raw = float(np.exp(-(j1 - j1) / temperature))
        w2_raw = float(np.exp(-(j2 - j1) / temperature))
        p1 = w1_raw / (w1_raw + w2_raw)
        cost_gap = float((j2 - j1) / temperature)
        w1 = prefix_trust * p1 + (1.0 - prefix_trust) * 0.5
    else:
        shift2 = shift1
        cost_gap = 0.0
        w1 = 0.5 + 0.5 * prefix_trust * confidence

    return MultiBranchResult(
        ok=True,
        failure_reason="",
        shift1=shift1,
        shift2=shift2,
        sep=sep,
        bimodal=bimodal,
        cost_gap=cost_gap,
        w1=float(np.clip(w1, 0.0, 1.0)),
        confidence=confidence,
        prefix_trust=prefix_trust,
        j_curve=j,
        deltas=deltas,
    )


MB_FEATURE_COLUMNS: tuple[str, ...] = (
    "mb_shift1",
    "mb_shift2",
    "mb_shift_hedged",
    "mb_sep",
    "mb_bimodal",
    "mb_cost_gap",
    "mb_w1",
    "mb_confidence",
    "mb_prefix_trust",
    "mb_ok",
)


class MultiBranchFeatureGenerator:
    """Arm-C feature set: datum branches and their bimodal uncertainty."""

    feature_columns = MB_FEATURE_COLUMNS

    def __init__(self, config: MultiBranchConfig | None = None) -> None:
        self.config = config or MultiBranchConfig()

    def generate(self, task: InferenceTask) -> PathFeatureOutput:
        assert_safe_features(self.feature_columns, context="arm-C multibranch frame")
        res = multibranch_scan(task, config=self.config)
        hedged = res.w1 * res.shift1 + (1.0 - res.w1) * res.shift2 if res.ok else 0.0
        values = {
            "mb_shift1": res.shift1 if res.ok else 0.0,
            "mb_shift2": res.shift2 if res.ok else 0.0,
            "mb_shift_hedged": hedged,
            "mb_sep": res.sep if res.ok else 0.0,
            "mb_bimodal": 1.0 if (res.ok and res.bimodal) else 0.0,
            "mb_cost_gap": res.cost_gap if res.ok else 0.0,
            "mb_w1": res.w1 if res.ok else 0.5,
            "mb_confidence": res.confidence if res.ok else 0.0,
            "mb_prefix_trust": res.prefix_trust if res.ok else 0.0,
            "mb_ok": 1.0 if res.ok else 0.0,
        }
        frame = pd.DataFrame(
            {k: np.full(task.n_predict, v, dtype="float64") for k, v in values.items()},
            index=np.arange(task.start, task.stop),
        )
        diagnostics = {
            "mb_ok": bool(res.ok),
            "mb_failure_reason": res.failure_reason,
            "mb_shift1": float(res.shift1),
            "mb_sep": float(res.sep),
            "mb_bimodal": bool(res.bimodal),
            "mb_confidence": float(res.confidence),
            "mb_prefix_trust": float(res.prefix_trust),
            "mb_cost_gap": float(res.cost_gap),
            "mb_w1": float(res.w1),
        }
        return PathFeatureOutput(frame=frame, diagnostics=diagnostics)


# ==========================================================================
# Ridge variants (arms B–D)
# ==========================================================================


class GeoAnchorRidge(RidgeBaseline):
    """Ridge Default plus extra feature generators (arms B/C/D only).

    The base matrix is exactly the selected default (no alignment, no
    spatial); the generators append their own frames, which the manifest
    re-validates before the model ever sees them.
    """

    def __init__(self, *, extra_generators: Iterable = (), name: str, **kw):
        kw.setdefault("alignment_features", False)
        super().__init__(**kw)
        self.extra_generators = tuple(extra_generators)
        self.name = name

    def _features(self, task: InferenceTask, feats) -> pd.DataFrame:
        X = super()._features(task, feats)
        for gen in self.extra_generators:
            out = gen.generate(task)
            X = pd.concat([X.reset_index(drop=True), out.frame.reset_index(drop=True)], axis=1)
        from src.features import validate_feature_frame

        validate_feature_frame(X)
        return X


def make_arm_factory(arm: str):
    """A zero-argument factory for arms A–D (E is built fold-scoped)."""
    if arm not in (ARM_A, ARM_B, ARM_C, ARM_D):
        raise KeyError(f"make_arm_factory only serves arms A-D; got {arm!r}")

    def factory(*, spatial=None):
        if arm == ARM_A:
            return RidgeBaseline(alignment_features=False, spatial=spatial)
        generators = []
        if arm in (ARM_B, ARM_D):
            generators.append(AffineCalibrationFeatureGenerator())
        if arm in (ARM_C, ARM_D):
            generators.append(MultiBranchFeatureGenerator())
        return GeoAnchorRidge(extra_generators=generators, name=arm, spatial=spatial)

    return factory


# ==========================================================================
# Idea 4 — nested prefix pseudo-holdout
# ==========================================================================


@dataclass
class NestedPseudoTask:
    """A masked task nested strictly inside the parent's visible prefix.

    Truth comes from ``TVT_input`` rows the parent treated as visible, so the
    check is target-free: no ``TVT`` label and no parent-scored row is read.
    """

    inputs: InferenceTask
    truth: np.ndarray


def nested_pseudo_task(
    task: InferenceTask,
    *,
    min_prefix: int = NESTED_MIN_PREFIX,
    min_predict: int = NESTED_MIN_PREDICT,
) -> NestedPseudoTask | None:
    """Move the boundary one level deeper into the visible prefix.

    The masked window mirrors the parent's prediction span, clipped so at
    least ``min_prefix`` visible rows survive before the nested start — the
    same construction ``src.tasks.make_task(mode="masked")`` applies at the
    real boundary. Returns ``None`` when the prefix cannot host a check
    (such wells are skipped by the gate rather than guessed at).
    """
    base_start = int(task.start)
    budget = base_start - min_prefix
    if budget < min_predict:
        return None
    masked_len = int(np.clip(task.n_predict, min_predict, budget))
    start = base_start - masked_len
    tvt_known = np.asarray(task.tvt_known, dtype="float64").copy()
    truth = tvt_known[start:base_start].copy()
    tvt_known[start:] = np.nan
    nested = replace(
        task,
        start=int(start),
        stop=int(base_start),
        tvt_known=tvt_known,
        mode=f"nested_{task.mode}",
    )
    nested.assert_no_target()
    return NestedPseudoTask(inputs=nested, truth=truth)


def _finite_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(truth)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2)))


def _tail_mean_se(pred: np.ndarray, truth: np.ndarray, frac: float = 0.10) -> float:
    """Worst-tail risk: mean of the top ``frac`` per-row squared errors."""
    m = np.isfinite(pred) & np.isfinite(truth)
    if int(m.sum()) < 10:
        return np.nan
    se = (pred[m] - truth[m]) ** 2
    k = max(1, int(np.ceil(frac * se.size)))
    return float(np.mean(np.sort(se)[-k:]))


# ==========================================================================
# Idea 7 — PF/Beam candidate corrections (bounded)
# ==========================================================================

CANDIDATE_FAMILIES = ("pf", "beam")
CANDIDATES = ("pf", "beam", "pf_beam_mean")


class MemoizedPathGenerator:
    """In-process memoization of a deterministic target-free path generator.

    PF/Beam outputs are pure functions of the task boundary and generator
    config — fold identity does not enter the computation. Memoizing on
    (well, boundary, mode) across folds is therefore numerically identical to
    recomputing, and lets the gate reuse inner/outer artifacts within a run.
    Only the slim arrays the gate needs are retained.
    """

    def __init__(self, delegate, memo: dict, family: str):
        self.delegate = delegate
        self.memo = memo
        self.family = family
        self.feature_columns = delegate.feature_columns

    def generate(self, task: InferenceTask) -> PathFeatureOutput:
        key = (
            GEOANCHOR_VERSION,
            self.family,
            task.well_id,
            task.mode,
            int(task.start),
            int(task.stop),
            int(task.n_rows),
        )
        hit = self.memo.get(key)
        if hit is None:
            out = self.delegate.generate(task)
            # Only the shift track and the scalar diagnostics are retained:
            # correction candidates and gate features need nothing else.
            hit = (
                np.asarray(out.frame[f"{self.family}_shift"], dtype="float64"),
                dict(out.diagnostics),
            )
            self.memo[key] = hit
        shift, diag = hit
        frame = pd.DataFrame(
            {f"{self.family}_shift": shift},
            index=np.arange(task.start, task.stop),
        )
        return PathFeatureOutput(frame=frame, diagnostics=dict(diag))


@dataclass
class CandidateCorrection:
    """One bounded candidate correction of the anchor prediction."""

    name: str
    prediction: np.ndarray | None
    confidence: float
    disagreement: float
    available: bool
    failure_reason: str


def _family_confidence(diag: dict) -> float:
    v = diag.get("confidence_mean", np.nan)
    return float(v) if np.isfinite(v) else 0.0


def generate_candidate_corrections(
    task: InferenceTask,
    base_pred: np.ndarray,
    *,
    pf,
    beam,
    mb: MultiBranchResult | None = None,
    correction_cap: float = CORRECTION_CAP_FT,
    observation_rows: int = 600,
) -> dict[str, CandidateCorrection]:
    """Bounded anchor-referenced corrections from the PF/Beam shift tracks.

    ``pf_shift``/``beam_shift`` are already anchor-relative, so the candidate
    trajectory is ``anchor + shift`` and the applied correction is clipped to
    ±``correction_cap`` around the anchor prediction — the bounded final move
    the studied notebooks apply, re-derived here with repo-consistent limits.
    """
    anchor = task.anchor_tvt
    anchor = anchor if np.isfinite(anchor) else 0.0
    tracks: dict[str, np.ndarray] = {}
    confs: dict[str, float] = {}
    observers = np.arange(task.start, min(task.stop, task.start + observation_rows))

    pf_out = pf.generate(task)
    beam_out = beam.generate(task)
    pf_diag = dict(pf_out.diagnostics)
    beam_diag = dict(beam_out.diagnostics)

    pf_track = anchor + np.asarray(pf_out.frame["pf_shift"], dtype="float64")
    beam_track = anchor + np.asarray(beam_out.frame["beam_shift"], dtype="float64")

    def _family_ok(diag: dict) -> bool:
        """A family qualifies when its track carries genuine GR evidence.

        Hard generator failures (missing typewell/anchor, all-missing GR)
        are excluded outright. A soft (partial-outage) track must additionally
        have enough measured rows to have corrected the geometry prior.
        """
        if str(diag.get("failure_reason", "")) not in _PATH_SOFT_FAILURES:
            return False
        try:
            observed = 1.0 - float(diag.get("fallback_fraction", 1.0))
        except (TypeError, ValueError):
            observed = 0.0
        if observed < CORRECTION_MIN_OBSERVED_FRAC:
            return False
        return _family_confidence(diag) > 0.0

    pf_ok = _family_ok(pf_diag)
    beam_ok = _family_ok(beam_diag)
    if pf_ok:
        tracks["pf"] = pf_track
        confs["pf"] = _family_confidence(pf_diag)
    if beam_ok:
        tracks["beam"] = beam_track
        confs["beam"] = _family_confidence(beam_diag)

    n_obs = len(observers)
    if pf_ok and beam_ok and n_obs:
        idx = observers - task.start
        disagreement = float(np.mean(np.abs(pf_track[idx] - beam_track[idx])))
    elif mb is not None and mb.ok:
        # Single family available: the GR datum scan's own branch separation
        # is the disagreement evidence. If the scan also failed, no objective
        # disagreement estimate exists and the cap cannot pass.
        disagreement = float(mb.sep)
    else:
        disagreement = float("inf")

    out: dict[str, CandidateCorrection] = {}
    for name in CANDIDATES:
        if name == "pf_beam_mean":
            avail = sorted(tracks)
            if not avail:
                out[name] = CandidateCorrection(
                    name, None, 0.0, disagreement, False, "no_pf_or_beam_track"
                )
                continue
            track = np.mean([tracks[a] for a in avail], axis=0)
            confidence = float(np.mean([confs[a] for a in avail]))
        else:
            if name not in tracks:
                reason = pf_diag.get("failure_reason") or beam_diag.get("failure_reason")
                out[name] = CandidateCorrection(
                    name, None, 0.0, disagreement, False, str(reason or "track_unavailable")
                )
                continue
            track = tracks[name]
            confidence = confs[name]
        corrected = base_pred + np.clip(track - base_pred, -correction_cap, correction_cap)
        corrected = np.where(np.isfinite(corrected), corrected, base_pred)
        out[name] = CandidateCorrection(
            name, corrected, confidence, disagreement, True, ""
        )
    return out


# ==========================================================================
# Idea 5 — the well-level GBDT gate
# ==========================================================================

GATE_FEATURE_COLUMNS: tuple[str, ...] = (
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
    "gate_cand_pf",
    "gate_cand_beam",
    "gate_cand_mean",
)


@dataclass(frozen=True)
class GateConfig:
    """Gate hyperparameters. Everything the gate *decides* is tuned per fold

    from fold-training wells (OOF); the values here are the fixed search grid
    and the GBDT's own capacity-limiting constants, fixed a priori.
    """

    inner_splits: int = 5       # cross-fitting depth for OOF gate examples
    tune_splits: int = 3        # sub-folds used for threshold selection
    max_correction: float = CORRECTION_CAP_FT
    margins: tuple[float, ...] = (0.0, 0.05)
    confidence_levels: tuple[float, ...] = (0.0, 0.5)  # quantiles of OOF conf
    disagreement_levels: tuple[float, ...] = (1.0, 0.5)  # quantiles of OOF disagreement
    max_iter: int = 150
    max_depth: int = 3
    min_examples: int = 40
    seed: int = 0


@dataclass
class GateThresholds:
    margin: float = 0.0
    conf_thr: float = 0.0
    sep_cap: float = np.inf
    tuned_on: int = 0
    reason: str = "default"


@dataclass
class GateFitInfo:
    """Per-fold gate training bookkeeping (reported, never fed back)."""

    protocol: str = ""
    fold: int = -1
    n_oof_wells: int = 0
    n_examples: int = 0
    n_pseudo_skipped: int = 0
    killed: bool = False
    kill_reason: str = ""
    margin: float = 0.0
    conf_thr: float = 0.0
    sep_cap: float = np.inf
    pooled_oof_delta: float = np.nan
    oof_activation_rate: float = np.nan
    fit_seconds: float = 0.0


def _well_prefix_scalars(task: InferenceTask) -> dict[str, float]:
    s = task.start
    tvt = task.tvt_known[:s]
    known = np.isfinite(tvt)
    vals = tvt[known]
    if vals.size > 1:
        std = float(np.std(vals))
        rng = float(np.max(vals) - np.min(vals))
    else:
        std, rng = 0.0, 0.0
    slope = 0.0
    if known.sum() > 3 and task.anchor_row >= 0:
        md = task.md[:s]
        w = known & (md >= md[task.anchor_row] - 300.0)
        if w.sum() > 3:
            x, y = md[w], tvt[w]
            xd = x - x.mean()
            denom = float((xd * xd).sum())
            if denom > 0:
                slope = float((xd * (y - y.mean())).sum() / denom)
    return {"std": std, "range": rng, "slope300": slope}


def gate_feature_row(
    task: InferenceTask,
    *,
    cal: AffineCalibration,
    mb: MultiBranchResult,
    pf_out: PathFeatureOutput,
    beam_out: PathFeatureOutput,
    candidate: str,
) -> dict[str, float]:
    """One gate design row = one (well, candidate) at one boundary."""
    prefix = _well_prefix_scalars(task)
    gr_miss_suffix = float(np.mean(~np.isfinite(task.gr[task.start : task.stop])))
    gr_miss_prefix = float(np.mean(~np.isfinite(task.gr[: task.start])))
    anchor = task.anchor_tvt
    pf_track = (anchor if np.isfinite(anchor) else 0.0) + np.asarray(
        pf_out.frame["pf_shift"], dtype="float64"
    )
    beam_track = (anchor if np.isfinite(anchor) else 0.0) + np.asarray(
        beam_out.frame["beam_shift"], dtype="float64"
    )
    # Fallback *fractions* (0 = fully GR-informed), not the any-row status:
    # a track with a short outage is still informative.
    pf_fallback = float(pf_out.diagnostics.get("fallback_fraction", 1.0))
    beam_fallback = float(beam_out.diagnostics.get("fallback_fraction", 1.0))
    both_informed = (
        pf_fallback <= 1.0 - CORRECTION_MIN_OBSERVED_FRAC
        and beam_fallback <= 1.0 - CORRECTION_MIN_OBSERVED_FRAC
    )
    if both_informed:
        disagreement = float(np.mean(np.abs(pf_track - beam_track)))
    elif mb.ok:
        disagreement = float(mb.sep)
    else:
        disagreement = np.inf
    row = {
        "gate_prefix_len": float(task.prefix_len),
        "gate_suffix_len": float(task.n_predict),
        "gate_gr_missing_suffix": float(gr_miss_suffix),
        "gate_prefix_gr_missing": float(gr_miss_prefix),
        "gate_tvt_std_prefix": prefix["std"],
        "gate_tvt_range_prefix": prefix["range"],
        "gate_tvt_slope_300": prefix["slope300"],
        "gate_anchor": float(anchor) if np.isfinite(anchor) else 0.0,
        "gate_acal_alpha": float(cal.alpha) if cal.ok else 0.0,
        "gate_acal_beta": float(cal.beta / max(cal.sd_tw, 1e-9)) if cal.ok else 0.0,
        "gate_acal_fit_rmse": float(cal.fit_rmse_z) if cal.ok else 0.0,
        "gate_acal_prefix_corr": float(cal.prefix_corr)
        if cal.ok and np.isfinite(cal.prefix_corr)
        else 0.0,
        "gate_mb_shift1": float(mb.shift1) if mb.ok else 0.0,
        "gate_mb_sep": float(mb.sep) if mb.ok else 0.0,
        "gate_mb_cost_gap": float(mb.cost_gap) if mb.ok else 0.0,
        "gate_mb_confidence": float(mb.confidence) if mb.ok else 0.0,
        "gate_mb_bimodal": 1.0 if (mb.ok and mb.bimodal) else 0.0,
        "gate_mb_prefix_trust": float(mb.prefix_trust) if mb.ok else 0.0,
        "gate_pf_confidence": _family_confidence(pf_out.diagnostics),
        "gate_pf_spread": float(pf_out.diagnostics.get("branch_spread_mean", 0.0) or 0.0),
        "gate_pf_fallback": float(pf_fallback),
        "gate_beam_confidence": _family_confidence(beam_out.diagnostics),
        "gate_beam_spread": float(beam_out.diagnostics.get("branch_spread_mean", 0.0) or 0.0),
        "gate_beam_fallback": float(beam_fallback),
        "gate_track_disagreement": float(disagreement) if np.isfinite(disagreement) else 1e6,
        "gate_cand_pf": 1.0 if candidate == "pf" else 0.0,
        "gate_cand_beam": 1.0 if candidate == "beam" else 0.0,
        "gate_cand_mean": 1.0 if candidate == "pf_beam_mean" else 0.0,
    }
    return {k: float(v) for k, v in row.items()}


class WellLevelGate:
    """Well-level GBDT gate over candidate corrections.

    Trained **only** from the fold-training wells of one outer fold:

    1. Inner GroupKFold cross-fits a Ridge Default over the training wells,
       so each training well's pseudo-holdout is scored by a model that never
       saw it (OOF prefix diagnostics).
    2. For every training well, a nested pseudo task (target-free
       ``TVT_input`` truth) yields, per candidate, the anchor-vs-correction
       RMSE delta, empirical improvement and tail-risk labels.
    3. A small GBDT regressor learns expected delta from the well/candidate
       gate features; thresholds (margin, confidence, disagreement) are tuned
       on tuning sub-folds carved from the same training wells.
    4. Kill switch: if the tuned policy cannot beat the anchor on pooled OOF,
       the gate is disabled for the whole fold.

    At no point can a fold-validation well (or any hidden label) reach step
    1–4: the runner asserts train/validation disjointness before ``fit`` and
    this class only ever receives the training list.
    """

    def __init__(
        self,
        *,
        pf,
        beam,
        config: GateConfig | None = None,
        protocol: str = "",
        fold: int = -1,
    ) -> None:
        self.pf = pf
        self.beam = beam
        self.config = config or GateConfig()
        self.protocol = protocol
        self.fold = fold
        self.thresholds = GateThresholds()
        self.killed = False
        self.kill_reason = ""
        self.model = None
        self.info = GateFitInfo(protocol=protocol, fold=fold)

    # ----------------------------------------------------------- training --
    def _anchor_model(self, tasks: list[WellTask]) -> RidgeBaseline:
        m = RidgeBaseline(alignment_features=False)
        m.fit(tasks)
        return m

    def _examples_for_task(
        self, task_outer: WellTask, anchor_model: RidgeBaseline
    ) -> list[dict]:
        """OOF example rows for one *training* well at its nested boundary."""
        inp_outer = task_outer.inputs()
        nested = nested_pseudo_task(inp_outer)
        if nested is None:
            return []
        ref = TypewellReference(inp_outer.tw_tvt, inp_outer.tw_gr)
        cal = fit_prefix_affine_calibration(inp_outer, ref)
        mb = multibranch_scan(inp_outer, cal=cal, ref=ref)
        pf_outer = self.pf.generate(inp_outer)
        beam_outer = self.beam.generate(inp_outer)
        base_outer = anchor_model.predict(inp_outer)
        cands_outer = generate_candidate_corrections(
            inp_outer, base_outer, pf=self.pf, beam=self.beam, mb=mb,
            correction_cap=self.config.max_correction,
        )
        pseudo = nested.inputs
        base_pseudo = anchor_model.predict(pseudo)
        cands_pseudo = generate_candidate_corrections(
            pseudo, base_pseudo, pf=self.pf, beam=self.beam, mb=None,
            correction_cap=self.config.max_correction,
        )
        truth = nested.truth
        n_pseudo_points = int(np.isfinite(truth).sum())
        base_rmse = _finite_rmse(base_pseudo, truth)
        base_tail = _tail_mean_se(base_pseudo, truth)
        rows: list[dict] = []
        for name in CANDIDATES:
            cand_o = cands_outer[name]
            cand_p = cands_pseudo[name]
            feats = gate_feature_row(
                inp_outer,
                cal=cal,
                mb=mb,
                pf_out=pf_outer,
                beam_out=beam_outer,
                candidate=name,
            )
            example = {
                "well_id": inp_outer.well_id,
                "candidate": name,
                "features": feats,
                "confidence": cand_o.confidence,
                "disagreement": cand_o.disagreement,
                "outer_available": cand_o.available,
                "pseudo_available": cand_p.available,
                "base_rmse_pseudo": base_rmse,
                "base_tail_pseudo": base_tail,
                "n_pseudo_points": n_pseudo_points,
            }
            if cand_p.available and np.isfinite(base_rmse):
                cand_rmse = _finite_rmse(cand_p.prediction, truth)
                cand_tail = _tail_mean_se(cand_p.prediction, truth)
                example["delta_rmse_pseudo"] = base_rmse - cand_rmse
                example["tail_delta_pseudo"] = (
                    base_tail - cand_tail
                    if np.isfinite(base_tail) and np.isfinite(cand_tail)
                    else 0.0
                )
                example["pseudo_improved"] = bool(base_rmse - cand_rmse > 1e-12)
                example["pseudo_tail_ok"] = bool(cand_tail <= base_tail + 1e-12)
            else:
                example["delta_rmse_pseudo"] = np.nan
                example["tail_delta_pseudo"] = np.nan
                example["pseudo_improved"] = False
                example["pseudo_tail_ok"] = False
            rows.append(example)
        return rows

    def _build_oof_examples(self, train_tasks: list[WellTask]) -> list[dict]:
        ids = [t.well_id for t in train_tasks]
        assert_no_blocked_wells(ids, context="gate training wells")
        inner = make_group_folds(ids, n_splits=self.config.inner_splits, seed=self.config.seed + 11)
        by_id = {t.well_id: t for t in train_tasks}
        examples: list[dict] = []
        skipped = 0
        for fold in inner:
            train_inner = [by_id[w] for w in fold.train_ids if w in by_id]
            valid_inner = [by_id[w] for w in fold.valid_ids if w in by_id]
            if not train_inner:
                continue
            anchor_inner = self._anchor_model(train_inner)
            for task in valid_inner:
                try:
                    rows = self._examples_for_task(task, anchor_inner)
                except Exception:
                    rows = []
                if not rows:
                    skipped += 1
                examples.extend(rows)
        self.info.n_pseudo_skipped += skipped
        return examples

    def _eligible_mask(self, frame: pd.DataFrame, thr: GateThresholds, predicted=None) -> np.ndarray:
        if predicted is None:
            predicted = np.zeros(len(frame))
        conf = frame["confidence"].to_numpy(dtype="float64")
        dis = frame["disagreement"].to_numpy(dtype="float64")
        ok = (
            frame["pseudo_improved"].to_numpy(dtype=bool)
            & frame["pseudo_tail_ok"].to_numpy(dtype=bool)
            & (conf >= thr.conf_thr)
            & (dis <= thr.sep_cap)
            & (np.asarray(predicted, dtype="float64") > thr.margin)
        )
        return ok

    def _policy_pooled_delta(
        self, examples: list[dict], thr: GateThresholds, predicted: np.ndarray
    ) -> tuple[float, float]:
        """Pooled OOF RMSE delta (policy − anchor) plus activation rate.

        Negative means the policy improved on the anchor. Wells with no
        eligible candidate fall back to the anchor (their squared errors are
        identical). Pooling weights by pseudo point count so long wells
        dominate, exactly like the global point-level metric.
        """
        frame = pd.DataFrame(
            {
                "confidence": [e["confidence"] for e in examples],
                "disagreement": [e["disagreement"] for e in examples],
                "pseudo_improved": [e["pseudo_improved"] for e in examples],
                "pseudo_tail_ok": [e["pseudo_tail_ok"] for e in examples],
            }
        )
        eligible = self._eligible_mask(frame, thr, predicted)
        by_well: dict[str, list[int]] = {}
        for i, e in enumerate(examples):
            by_well.setdefault(e["well_id"], []).append(i)

        se_pol = se_base = 0.0
        den = 0
        n_act = 0
        for _w, idxs in by_well.items():
            e0 = examples[idxs[0]]
            base_rmse = e0.get("base_rmse_pseudo", np.nan)
            if not np.isfinite(base_rmse):
                continue
            chosen = None
            best_pred = -np.inf
            for i in idxs:
                if not eligible[i]:
                    continue
                if float(predicted[i]) > best_pred:
                    best_pred = float(predicted[i])
                    chosen = i
            n_points = max(int(e0.get("n_pseudo_points") or 1), 1)
            den += n_points
            if chosen is not None:
                cand_rmse = base_rmse - float(examples[chosen].get("delta_rmse_pseudo", 0.0))
                se_pol += n_points * max(cand_rmse, 0.0) ** 2
                n_act += 1
            else:
                se_pol += n_points * base_rmse**2
            se_base += n_points * base_rmse**2
        if den == 0:
            return np.nan, 0.0
        delta = float(np.sqrt(se_pol / den) - np.sqrt(se_base / den))
        n_wells = sum(
            1 for idxs in by_well.values()
            if np.isfinite(examples[idxs[0]].get("base_rmse_pseudo", np.nan))
        )
        return delta, (n_act / max(n_wells, 1))

    def _fit_gbdt(self, X: pd.DataFrame, y: np.ndarray):
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_depth=self.config.max_depth,
            max_iter=self.config.max_iter,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=self.config.seed,
        )
        model.fit(X, y)
        return model

    def fit(self, train_tasks: list[WellTask], *, validation_ids: Iterable[str] = ()) -> "WellLevelGate":
        t0 = time.perf_counter()
        train_ids = {t.well_id for t in train_tasks}
        overlap = train_ids & {str(w) for w in validation_ids}
        if overlap:
            raise CrossFitLeakage(
                f"gate training: {len(overlap)} validation well(s) reached gate.fit, "
                f"e.g. {sorted(overlap)[:5]}"
            )
        examples = self._build_oof_examples(train_tasks)
        usable = [
            e
            for e in examples
            if e["outer_available"] and np.isfinite(e.get("delta_rmse_pseudo", np.nan))
        ]
        self.info.n_oof_wells = len({e["well_id"] for e in usable})
        self.info.n_examples = len(usable)
        if len(usable) < self.config.min_examples:
            self.killed = True
            self.kill_reason = "insufficient_oof_examples"
            self.info.killed = True
            self.info.kill_reason = self.kill_reason
            self.info.fit_seconds = time.perf_counter() - t0
            return self

        X = pd.DataFrame([e["features"] for e in usable], columns=GATE_FEATURE_COLUMNS)
        assert_safe_features(X.columns, context="gate design matrix")
        y = np.asarray([e["delta_rmse_pseudo"] for e in usable], dtype="float64")

        # ---- threshold tuning on tuning sub-folds of the training wells ---
        well_ids = sorted({e["well_id"] for e in usable})
        tune_folds = make_group_folds(
            well_ids, n_splits=self.config.tune_splits, seed=self.config.seed + 23
        )
        conf_pool = np.asarray([e["confidence"] for e in usable], dtype="float64")
        conf_pool = conf_pool[np.isfinite(conf_pool)]
        dis_pool = np.asarray(
            [e["disagreement"] if np.isfinite(e["disagreement"]) else np.nan for e in usable],
            dtype="float64",
        )
        dis_pool = dis_pool[np.isfinite(dis_pool)]
        conf_options = sorted(
            {0.0}
            | {float(np.quantile(conf_pool, q)) for q in self.config.confidence_levels}
            if conf_pool.size else {0.0}
        )
        sep_options = sorted(
            {float("inf")}
            | {float(np.quantile(dis_pool, q)) for q in self.config.disagreement_levels}
            if dis_pool.size else {float("inf")}
        )
        example_idx_by_well: dict[str, list[int]] = {}
        for i, e in enumerate(usable):
            example_idx_by_well.setdefault(e["well_id"], []).append(i)

        pred_cache: dict[tuple, np.ndarray] = {}
        for fold in tune_folds:
            tr_idx = [i for w in fold.train_ids for i in example_idx_by_well.get(w, [])]
            va_idx = [i for w in fold.valid_ids for i in example_idx_by_well.get(w, [])]
            if not tr_idx or not va_idx:
                continue
            sub = self._fit_gbdt(X.iloc[tr_idx], y[tr_idx])
            pred_cache[tuple(sorted(fold.valid_ids))] = (va_idx, sub.predict(X.iloc[va_idx]))

        best: tuple[float, GateThresholds] | None = None
        for margin in self.config.margins:
            for conf_thr in conf_options:
                for sep_cap in sep_options:
                    thr = GateThresholds(
                        margin=margin, conf_thr=conf_thr, sep_cap=sep_cap,
                        tuned_on=len(usable), reason="tuned_subcf",
                    )
                    deltas, ok = [], True
                    for key, (va_idx, va_pred) in pred_cache.items():
                        sub_examples = [usable[i] for i in va_idx]
                        delta, _act = self._policy_pooled_delta(sub_examples, thr, va_pred)
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
            self.info.killed = True
            self.info.kill_reason = self.kill_reason
            self.info.fit_seconds = time.perf_counter() - t0
            return self

        self.thresholds = best[1]
        self.model = self._fit_gbdt(X, y)
        pred_all = self.model.predict(X)
        pooled, act = self._policy_pooled_delta(usable, self.thresholds, pred_all)
        self.info.pooled_oof_delta = float(pooled) if np.isfinite(pooled) else np.nan
        self.info.oof_activation_rate = float(act)
        if not np.isfinite(pooled) or pooled >= 0.0:
            # Rule: the correction must not degrade the fold metric — checked
            # on the fold-training wells' pooled OOF, never on validation.
            self.killed = True
            self.kill_reason = "kill_switch_pooled_oof_degraded"
        self.info.killed = self.killed
        self.info.kill_reason = self.kill_reason
        self.info.margin = self.thresholds.margin
        self.info.conf_thr = self.thresholds.conf_thr
        self.info.sep_cap = self.thresholds.sep_cap
        self.info.fit_seconds = time.perf_counter() - t0
        return self

    # ------------------------------------------------------------ apply ----
    def predict_improvements(
        self,
        task: InferenceTask,
        *,
        cal: AffineCalibration,
        mb: MultiBranchResult,
        pf_out: PathFeatureOutput,
        beam_out: PathFeatureOutput,
    ) -> dict[str, float]:
        """GBDT expected improvement per candidate, from target-free features."""
        if self.model is None:
            return {c: -np.inf for c in CANDIDATES}
        rows = [
            gate_feature_row(
                task, cal=cal, mb=mb, pf_out=pf_out, beam_out=beam_out,
                candidate=c,
            )
            for c in CANDIDATES
        ]
        X = pd.DataFrame(rows, columns=GATE_FEATURE_COLUMNS)
        assert_safe_features(X.columns, context="gate inference row")
        pred = self.model.predict(X)
        return {c: float(p) for c, p in zip(CANDIDATES, pred)}


@dataclass
class GateDecision:
    outcome: str            # "applied_<candidate>" or "fallback"
    candidate: str | None
    reason: str
    confidence: float
    disagreement: float
    predicted_improvement: float
    n_eligible: int
    pseudo_delta: float = np.nan


class GatedRidgeAnchor(BaselineModel):
    """Arm E — Ridge Default anchor with the well-level GBDT gate.

    ``predict`` returns the Ridge Default trajectory whenever the gate
    declines (any failed rule, any failed check, any exception) — Ridge
    Default is not only the baseline, it is the fallback of the gated model
    itself. A correction is applied only when every rule passes:

    1. the candidate beats the anchor on a visible-prefix pseudo-holdout;
    2. its alignment confidence clears the fold-OOF threshold;
    3. the PF/Beam (or GR-scan) branch disagreement clears the fold-OOF cap;
    4. worst-tail risk on the pseudo-holdout does not increase; and
    5. the gate was not kill-switched during fold-OOF training.
    """

    name = ARM_E
    needs_alignment = False

    def __init__(
        self,
        *,
        pf,
        beam,
        gate_config: GateConfig | None = None,
        protocol: str = "",
        fold: int = -1,
        gate_log: list | None = None,
    ) -> None:
        self.pf = pf
        self.beam = beam
        self.protocol = protocol
        self.fold = fold
        self.gate_config = gate_config or GateConfig()
        self.anchor_model: RidgeBaseline | None = None
        self.gate: WellLevelGate | None = None
        self.gate_log = gate_log if gate_log is not None else []
        self._last_diagnostics: dict = {}

    def fit(self, tasks: list[WellTask], **kw) -> "GatedRidgeAnchor":
        self.anchor_model = RidgeBaseline(alignment_features=False)
        self.anchor_model.fit(tasks)
        self.gate = WellLevelGate(
            pf=self.pf,
            beam=self.beam,
            config=self.gate_config,
            protocol=self.protocol,
            fold=self.fold,
        )
        # The gate trains on tasks' training wells only; the caller passes
        # exactly the fold-training list (asserted by the driver as well).
        # A gate-training failure is never fatal: the gate is kill-switched
        # and the anchor carries the fold, so arm E is always scorable.
        if len(tasks) < max(2, self.gate_config.inner_splits):
            self.gate.killed = True
            self.gate.kill_reason = "insufficient_training_wells"
            self.gate.info.killed = True
            self.gate.info.kill_reason = self.gate.kill_reason
            return self
        try:
            self.gate.fit(tasks, validation_ids=kw.get("validation_ids", ()))
        except CrossFitLeakage:
            raise  # a leakage assertion must never degrade to a fallback
        except Exception as exc:
            self.gate.killed = True
            self.gate.kill_reason = f"gate_training_failed:{type(exc).__name__}"
            self.gate.info.killed = True
            self.gate.info.kill_reason = self.gate.kill_reason
        return self

    def _log(self, task: InferenceTask, decision: GateDecision) -> None:
        self.gate_log.append(
            {
                "protocol": self.protocol,
                "fold": self.fold,
                "well_id": task.well_id,
                "outcome": decision.outcome,
                "candidate": decision.candidate or "",
                "reason": decision.reason,
                "confidence": float(decision.confidence),
                "disagreement": float(decision.disagreement)
                if np.isfinite(decision.disagreement)
                else np.nan,
                "predicted_improvement": float(decision.predicted_improvement)
                if np.isfinite(decision.predicted_improvement)
                else np.nan,
                "pseudo_delta": float(decision.pseudo_delta)
                if np.isfinite(decision.pseudo_delta)
                else np.nan,
                "n_eligible": int(decision.n_eligible),
                "gate_killed": bool(self.gate.killed) if self.gate else True,
            }
        )

    def predict(self, task: InferenceTask, feats=None) -> np.ndarray:
        base = self.anchor_model.predict(task, feats) if self.anchor_model else None
        if base is None:
            base = np.full(task.n_predict, self._anchor(task))
        self._last_diagnostics = {}
        gate = self.gate
        if gate is None or gate.killed:
            self._log(
                task,
                GateDecision("fallback", None, gate.kill_reason if gate else "gate_unfitted", 0.0, np.inf, np.nan, 0),
            )
            return base
        ref = TypewellReference(task.tw_tvt, task.tw_gr)
        cal = fit_prefix_affine_calibration(task, ref)
        mb = multibranch_scan(task, cal=cal, ref=ref)
        pf_out = self.pf.generate(task)
        beam_out = self.beam.generate(task)
        cands = generate_candidate_corrections(
            task, base, pf=self.pf, beam=self.beam, mb=mb,
            correction_cap=gate.config.max_correction,
        )
        predicted = gate.predict_improvements(
            task, cal=cal, mb=mb, pf_out=pf_out, beam_out=beam_out
        )

        # Rule 1 + 4 evidence: nested visible-prefix pseudo-holdout.
        nested = nested_pseudo_task(task)
        pseudo_stats: dict[str, tuple[float, bool]] = {}
        if nested is not None:
            base_pseudo = self.anchor_model.predict(nested.inputs)
            cands_pseudo = generate_candidate_corrections(
                nested.inputs, base_pseudo, pf=self.pf, beam=self.beam, mb=None,
                correction_cap=gate.config.max_correction,
            )
            truth = nested.truth
            base_rmse = _finite_rmse(base_pseudo, truth)
            base_tail = _tail_mean_se(base_pseudo, truth)
            for name, cand_p in cands_pseudo.items():
                if not cand_p.available or not np.isfinite(base_rmse):
                    pseudo_stats[name] = (-np.inf, False)
                    continue
                cand_rmse = _finite_rmse(cand_p.prediction, truth)
                cand_tail = _tail_mean_se(cand_p.prediction, truth)
                delta = base_rmse - cand_rmse
                tail_ok = bool(
                    np.isfinite(base_tail)
                    and np.isfinite(cand_tail)
                    and cand_tail <= base_tail + 1e-12
                )
                pseudo_stats[name] = (float(delta), tail_ok)

        thr = gate.thresholds
        eligible: list[str] = []
        reasons: dict[str, str] = {}
        for name in CANDIDATES:
            cand = cands[name]
            if not cand.available:
                reasons[name] = f"candidate_unavailable:{cand.failure_reason}"
                continue
            if nested is None:
                reasons[name] = "pseudo_holdout_unavailable"
                continue
            delta, tail_ok = pseudo_stats[name]
            if not np.isfinite(delta) or delta <= 1e-12:
                reasons[name] = "pseudo_holdout_not_improved"      # rule 1
                continue
            if not tail_ok:
                reasons[name] = "worst_tail_risk_increased"        # rule 4
                continue
            if cand.confidence < thr.conf_thr:
                reasons[name] = "alignment_confidence_below_threshold"  # rule 2
                continue
            if cand.disagreement > thr.sep_cap:
                reasons[name] = "branch_disagreement_unacceptable"      # rule 3
                continue
            if not np.isfinite(predicted[name]) or predicted[name] <= thr.margin:
                reasons[name] = "gbdt_expected_gain_below_margin"
                continue
            eligible.append(name)

        self._last_diagnostics = {
            "alignment_confidence_mean": float(mb.confidence) if mb.ok else 0.0,
            "alignment_confidence_p10": float(mb.prefix_trust) if mb.ok else 0.0,
            "alignment_ok": bool(mb.ok),
            "alignment_failure_reason": "" if mb.ok else mb.failure_reason,
            "fallback_points": 0,
            "fallback_fraction": 0.0,
        }
        if not eligible:
            chosen_reason = (
                ";".join(f"{k}:{v}" for k, v in reasons.items()) or "no_candidates"
            )
            self._log(
                task,
                GateDecision(
                    "fallback", None, chosen_reason,
                    float(max((c.confidence for c in cands.values()), default=0.0)),
                    float(max((c.disagreement for c in cands.values() if np.isfinite(c.disagreement)), default=np.nan)),
                    float(max((p for p in predicted.values() if np.isfinite(p)), default=np.nan)),
                    0,
                ),
            )
            self._last_diagnostics["fallback_points"] = int(task.n_predict)
            self._last_diagnostics["fallback_fraction"] = 1.0
            return base

        chosen = max(eligible, key=lambda c: predicted[c])
        cand = cands[chosen]
        decision = GateDecision(
            f"applied_{chosen}", chosen, "all_rules_passed",
            cand.confidence, cand.disagreement, predicted[chosen], len(eligible),
            pseudo_delta=pseudo_stats[chosen][0],
        )
        self._log(task, decision)
        # Arm E cannot override the repository's model-status gate for
        # rejected models: only PF/Beam candidate features are used, both of
        # which remain CANDIDATE generators, applied as bounded corrections
        # behind the fold-OOF kill switch. Nothing here routes a REJECTED
        # model.
        self._last_diagnostics["fallback_points"] = 0
        self._last_diagnostics["fallback_fraction"] = 0.0
        return cand.prediction

    def prediction_diagnostics(self, task, feats, pred) -> dict:
        out = dict(self._last_diagnostics)
        return out


# ==========================================================================
# Driver
# ==========================================================================


@dataclass
class GeoAnchorRun:
    protocol: str
    well_results: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    fold_records: list = field(default_factory=list)
    gate_logs: list = field(default_factory=list)
    gate_fit_infos: list = field(default_factory=list)


def run_geoanchor_protocol(
    *,
    protocol: str,
    mode: str,
    folds,
    task_builder,
    arms: tuple[str, ...] = ARM_ORDER,
    memo: dict | None = None,
    seed: int = 0,
    device: str = "cpu",
    path_cache=None,
    dataset_version: str = "rogii-mounted-v1",
    gate_config: GateConfig | None = None,
    inner_gate_log: list | None = None,
    verbose: bool = False,
) -> GeoAnchorRun:
    """Cross-fitted evaluation of all arms on one protocol.

    Arms A–D and E share the same folds, tasks and scoring path
    (``evaluate_models``), so every arm within a protocol is paired. The
    per-fold ordering — disjointness assertion, fit, gate training, scoring —
    keeps fold-validation wells out of every fitting step, including the gate.

    ``memo`` is a process-local PF/Beam artifact memo (see
    ``MemoizedPathGenerator``); it is keyed by boundary only and carries no
    fold-specific or target-derived information.
    """
    from src.beam_search import BeamSearchFeatureGenerator
    from src.particle_filter import ParticleFilterFeatureGenerator

    run = GeoAnchorRun(protocol=protocol)
    memo = memo if memo is not None else {}
    arms = tuple(arms)

    for fold in folds:
        t0 = time.perf_counter()
        assert_no_blocked_wells(fold.train_ids, context=f"geoanchor {protocol} fold {fold.index} train")
        assert_no_blocked_wells(fold.valid_ids, context=f"geoanchor {protocol} fold {fold.index} valid")

        train_tasks, sk_train = task_builder(fold.train_ids, mode)
        valid_tasks, sk_valid = task_builder(fold.valid_ids, mode)
        for wid, reason in sk_train + sk_valid:
            run.failures.append({"stage": "task", "model": "", "well_id": wid, "error": reason})
        if not train_tasks or not valid_tasks:
            continue

        train_ids = {t.well_id for t in train_tasks}
        valid_ids = {t.well_id for t in valid_tasks}
        overlap = train_ids & valid_ids
        if overlap:
            raise CrossFitLeakage(
                f"geoanchor {protocol} fold {fold.index}: {len(overlap)} well(s) are both "
                f"fitted and scored, e.g. {sorted(overlap)[:5]}."
            )

        models: dict[str, BaselineModel] = {}
        for arm in arms:
            if arm == ARM_E:
                continue
            model = make_arm_factory(arm)()
            try:
                model.fit(train_tasks)
                models[arm] = model
            except Exception as exc:
                run.failures.append(
                    {"stage": "fit", "model": arm, "well_id": "", "error": f"{type(exc).__name__}: {exc}"}
                )

        if ARM_E in arms:
            def _pf():
                return MemoizedPathGenerator(
                    ParticleFilterFeatureGenerator(
                        cache=path_cache,
                        dataset_version=dataset_version,
                        fold_id=fold.index,
                        protocol=protocol,
                        device=device,
                    ),
                    memo,
                    "pf",
                )

            def _beam():
                return MemoizedPathGenerator(
                    BeamSearchFeatureGenerator(
                        cache=path_cache,
                        dataset_version=dataset_version,
                        fold_id=fold.index,
                        protocol=protocol,
                        device=device,
                    ),
                    memo,
                    "beam",
                )

            gated = GatedRidgeAnchor(
                pf=_pf(),
                beam=_beam(),
                gate_config=gate_config or GateConfig(seed=seed),
                protocol=protocol,
                fold=fold.index,
                gate_log=run.gate_logs,
            )
            try:
                gated.fit(train_tasks, validation_ids=valid_ids)
                models[ARM_E] = gated
                run.gate_fit_infos.append(dict(vars(gated.gate.info)))
            except Exception as exc:
                run.failures.append(
                    {"stage": "fit", "model": ARM_E, "well_id": "", "error": f"{type(exc).__name__}: {exc}"}
                )

        run.well_results += evaluate_models(
            models,
            valid_tasks,
            protocol,
            fold.index,
            verbose=verbose,
            failures=run.failures,
            cache_context={"dataset_version": dataset_version, "fold": fold.index, "protocol": protocol},
        )
        run.fold_records.append(
            {
                "protocol": protocol,
                "fold": fold.index,
                "n_train_wells": len(train_tasks),
                "n_valid_wells": len(valid_tasks),
                "n_arms_fitted": len(models),
                "seconds": time.perf_counter() - t0,
            }
        )
        if verbose:
            print(
                f"      fold {fold.index}: {len(train_tasks)} train / {len(valid_tasks)} valid, "
                f"{len(models)}/{len(arms)} arms in {time.perf_counter() - t0:.1f}s"
            )
    return run
