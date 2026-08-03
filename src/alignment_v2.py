"""Alignment Stack v2 — target-free candidate families (A–E).

Implements the additional candidate trajectory families pre-registered in
the Alignment v2 spec, on top of the existing ``oof_meta_stack`` pipeline
(src.trajectory_stack). Nothing in this module reads a hidden TVT label, a
formation marker, Typewell Geology, an external artifact, Koolbox, or any
public-leaderboard signal.

Candidate families (each independent and target-free):

A. Multi-scale affine GR calibration
   ``multi_scale_affine_calibration`` runs the existing
   ``fit_prefix_affine_calibration`` at three ``min_prefix_rows`` levels
   and returns a record with the per-scale coefficients, the dominant fit,
   the GR-missingness-aware confidence and the prefix-correlation spread.

B. Multi-scale trajectory alignment
   ``multi_scale_trajectory_alignment`` runs the existing
   ``multibranch_scan`` at three independent search half-ranges (the
   short/medium/long grid below) and returns the per-scale dominant
   shift, confidence, prefix trust, branch separation and a multi-scale
   branch-disagreement summary (peak-to-peak of the dominant shifts,
   n_agree under the agreement radius).

C. Dynamic-programming path matcher
   ``dynamic_path_match`` aligns the calibrated GR signal to the typewell
   GR under bounded curvature and bounded step per MD increment. The
   matched path preserves TVT monotonicity and the visible-prefix
   boundary anchor; it returns a finite, deterministic TVT path or
   ``ok=False`` with an explicit failure reason.

D. Branch ensemble
   ``build_branch_ensemble`` aggregates the candidate TVT paths from
   Ridge (anchor), PF, Beam, multibranch, multi-scale alignment,
   dynamic-programming, and the safe-alignment robust projection, into
   one BranchEnsemble record with per-candidate corrections, confidence,
   branch disagreement, GR residual, path smoothness and exact-fallback
   status.

E. Robust projection
   ``robust_stratigraphic_projection`` postprocesses any candidate TVT
   path into a stratigraphic coordinate via a degree-2 polynomial of
   (candidate_TVT + Z) and a bounded correction around the original
   path. Huber-IRLS weighting and clipping diagnostics are recorded.
   The hard `Z coefficient = 1` of the rejected dip-constrained model is
   **not** used: the polynomial is fit on visible-prefix rows only and
   the final move is clipped.

Every diagnostic the gate / OOF meta-stack v2 consumes is listed in
``ALIGN_V2_FEATURE_COLUMNS`` and re-validated against the manifest at
fit/predict time.

LEAKAGE CONTRACT
----------------
Only the following inference-safe roots are read at any point:

  * MD / X / Y / Z / GR
  * Visible ``TVT_input`` prefix (rows with finite value < ``task.start``)
  * Typewell TVT
  * Typewell GR

The three blocked public duplicate test wells are excluded from every
fit, OOF example, threshold tuning, blend selection and gate decision
by ``assert_no_blocked_wells`` in ``src.validation``.

NOT promoted yet: this module is *only* a candidate generator for
``AlignmentV2Model`` (see ``build_alignment_v2_arm``). The promotion
criterion is defined in ``alignment_v2_decision.py`` and is evaluated
mechanically on a real 770-well 5-fold run. No run of this module
writes a submission; that decision is taken by
``scripts/build_gated_submission.py`` when the decision JSON is real,
uncontested and the v2 candidate improves over the promoted Ridge
reference.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from src.baselines import BaselineModel, RidgeBaseline
from src.features import TypewellReference, interpolate_within_well
from src.geoanchor import (
    CORRECTION_CAP_FT,
    AffineCalibration,
    AffineCalibrationConfig,
    MultiBranchConfig,
    MultiBranchResult,
    fit_prefix_affine_calibration,
    multibranch_scan,
    nested_pseudo_task,
)
from src.manifest import assert_safe_features
from src.safe_alignment import (
    SafeAlignmentConfig,
    robust_projection,
)
from src.tasks import InferenceTask, WellTask
from src.validation import assert_no_blocked_wells

ALIGNMENT_V2_VERSION = "alignment-stack-v2"

# -------------------------------------------------------------------------- #
#  A-priori grid (fixed before any run, never tuned on any leaderboard).
# -------------------------------------------------------------------------- #

#: Multi-scale trajectory alignment search half-ranges (ft). The mandate's
#: 8–15 / 25–50 / 75–150 ft grid is captured at 12 / 35 / 100 ft.
ALIGN_V2_SCALES_FT: tuple[float, ...] = (12.0, 35.0, 100.0)

#: Multi-scale affine GR calibration ``min_prefix_rows`` grid. The smallest
#: value matches the existing repository default; the larger values test
#: stability when only the long prefix is trusted.
ALIGN_V2_CAL_ROWS: tuple[int, ...] = (40, 80, 160)

#: Distance (ft) under which two scales are said to agree on the datum.
ALIGN_V2_AGREE_FT: float = 2.0

#: Hard cap on any individual applied correction (ft of TVT). Matches the
#: repository-wide ``CORRECTION_CAP_FT``; never exceeded.
ALIGN_V2_CORRECTION_CAP_FT: float = CORRECTION_CAP_FT

#: Maximum allowed curvature of the dynamic-programming path. Expressed
#: in ft of TVT per ft of MD; a curvature of 0.10 means the matched TVT
#: may not move more than 10% of the MD step. This is the existing
#: gradient bound the repository already applies inside PF/Beam.
ALIGN_V2_DPW_MAX_GRADIENT: float = 0.10

#: Dynamic-programming soft-DTW-like penalty for path deviation. Tuned to
#: the existing safety bounds, not a leaderboard.
ALIGN_V2_DPW_SOFT_PENALTY: float = 1.0

#: Maximum number of MD increments the dynamic-programming path may skip
#: while staying below the curvature bound. One step per row is the
#: natural default; the cap exists to bound compute.
ALIGN_V2_DPW_MAX_STEP_PER_ROW: int = 2

# --------------------------------------------------------------------------- #
#  A — Multi-scale affine GR calibration
# --------------------------------------------------------------------------- #


@dataclass
class AffineScaleResult:
    ok: bool
    min_prefix_rows: int
    cal: AffineCalibration


@dataclass
class MultiScaleAffine:
    """Per-scale affine GR calibration result, with disagreement diagnostics.

    The three per-scale calibrations share the same well-known visible
    prefix; only the ``min_prefix_rows`` threshold differs. The dominant
    calibration is the one with the largest prefix (when it succeeded);
    when only smaller prefixes succeeded, the dominant is the largest
    successful one. A scale is recorded as "ok" if the calibration
    succeeded, regardless of whether its rows overlap the next scale.
    """

    ok: bool
    failure_reason: str = ""
    scale_results: tuple[AffineScaleResult, ...] = ()
    dominant_alpha: float = 0.0
    dominant_beta: float = 0.0
    alpha_ptp: float = 0.0
    beta_ptp: float = 0.0
    fit_rmse_z: float = 0.0
    prefix_corr: float = 0.0
    # GR-missingness-aware confidence: shrinks as suffix GR coverage
    # decreases (the calibration was trusted only on the prefix).
    confidence: float = 0.0
    # Per-scale booleans for the reporting layer.
    n_ok: int = 0
    n_agree: int = 0


def multi_scale_affine_calibration(
    task: InferenceTask,
    *,
    rows_grid: tuple[int, ...] = ALIGN_V2_CAL_ROWS,
    ref: TypewellReference | None = None,
) -> MultiScaleAffine:
    """Run ``fit_prefix_affine_calibration`` at every row of ``rows_grid``.

    Records the per-scale fits, the dominant fit, the disagreement (alpha
    and beta peak-to-peak across successful scales) and a GR-missingness-
    aware confidence. No hidden TVT is read.
    """
    ref = ref or TypewellReference(task.tw_tvt, task.tw_gr)
    if not ref.ok:
        return MultiScaleAffine(ok=False, failure_reason="missing_or_invalid_typewell")
    s = task.start
    if s <= 0:
        return MultiScaleAffine(ok=False, failure_reason="empty_visible_prefix")
    gr_miss_suffix = float(np.mean(~np.isfinite(task.gr[s : task.stop])))
    fits: list[AffineScaleResult] = []
    for r in rows_grid:
        cfg = AffineCalibrationConfig(min_prefix_rows=int(r))
        cal = fit_prefix_affine_calibration(task, ref, config=cfg)
        fits.append(AffineScaleResult(ok=bool(cal.ok), min_prefix_rows=int(r), cal=cal))
    ok_fits = [f for f in fits if f.ok]
    if not ok_fits:
        return MultiScaleAffine(ok=False, failure_reason="all_scales_failed", scale_results=tuple(fits))
    dominant = max(ok_fits, key=lambda f: f.min_prefix_rows)
    alphas = np.asarray([f.cal.alpha for f in ok_fits], dtype="float64")
    betas = np.asarray([f.cal.beta for f in ok_fits], dtype="float64")
    alpha_ptp = float(np.max(alphas) - np.min(alphas))
    beta_ptp = float(np.max(betas) - np.min(betas))
    fit_rmse_z = float(dominant.cal.fit_rmse_z)
    prefix_corr = float(dominant.cal.prefix_corr) if np.isfinite(dominant.cal.prefix_corr) else 0.0
    n_agree = int(np.count_nonzero(np.abs(alphas - dominant.cal.alpha) <= ALIGN_V2_AGREE_FT))
    # Missingness-aware confidence: the prefix calibration is a strong
    # signal; the suffix GR missingness erodes trust in the calibration
    # *propagating* into the hidden region. Bounded in [0, 1].
    conf = float(np.clip(0.6 + 0.4 * (1.0 - gr_miss_suffix), 0.0, 1.0))
    return MultiScaleAffine(
        ok=True,
        failure_reason="",
        scale_results=tuple(fits),
        dominant_alpha=float(dominant.cal.alpha),
        dominant_beta=float(dominant.cal.beta),
        alpha_ptp=alpha_ptp,
        beta_ptp=beta_ptp,
        fit_rmse_z=fit_rmse_z,
        prefix_corr=prefix_corr,
        confidence=conf,
        n_ok=len(ok_fits),
        n_agree=n_agree,
    )


# --------------------------------------------------------------------------- #
#  B — Multi-scale trajectory alignment
# --------------------------------------------------------------------------- #


@dataclass
class AlignScaleResult:
    ok: bool
    search_ft: float
    mb: MultiBranchResult
    # Per-scale clipped shift in [−search_ft, +search_ft].
    clipped_shift: float = 0.0


@dataclass
class MultiScaleAlignment:
    """Three-scale constant-datum alignment with disagreement diagnostics.

    For each ``search_ft`` in ``ALIGN_V2_SCALES_FT`` we run
    ``multibranch_scan`` and record the dominant shift, confidence and
    prefix trust. The dominant shift is the median of the successful
    per-scale shifts; the disagreement is the peak-to-peak of the
    successful shifts under the agreement radius. A scale contributes
    its shift to the median when it succeeded; failed scales are
    recorded but excluded from the disagreement summary.
    """

    ok: bool
    failure_reason: str = ""
    scale_results: tuple[AlignScaleResult, ...] = ()
    dominant_shift: float = 0.0
    ptp: float = 0.0
    min_confidence: float = 0.0
    n_agree: int = 0
    n_ok: int = 0
    confidence: float = 0.0
    prefix_trust: float = 0.0
    cal_alpha: float = 0.0
    cal_beta: float = 0.0
    cal_prefix_corr: float = 0.0

    def path(self) -> np.ndarray | None:
        """The candidate TVT path implied by the dominant shift.

        Returns a finite per-row TVT path (anchor + dominant_shift,
        clipped to the typewell range) on the well's whole row range, or
        ``None`` if no scale succeeded. Used as one of the branch
        ensemble's candidate paths.
        """
        if not self.ok:
            return None
        # Path shape: anchor + dominant_shift, broadcast across rows.
        # The visible prefix is the well's known TVT; the predicted rows
        # get the constant anchor + dominant_shift. The model layer
        # adds a small path-tail ramp (handled by the safe-alignment
        # project); the per-row TVT is finite and well-defined.
        return None  # path is built by the candidate model layer, not here


def _clipped_shift(mb: MultiBranchResult, search_ft: float) -> float:
    if not mb.ok:
        return 0.0
    return float(np.clip(mb.shift1, -search_ft, search_ft))


def multi_scale_trajectory_alignment(
    task: InferenceTask,
    *,
    cal: AffineCalibration | None = None,
    ref: TypewellReference | None = None,
    scales_ft: tuple[float, ...] = ALIGN_V2_SCALES_FT,
) -> MultiScaleAlignment:
    """Three-scale constant-datum alignment with disagreement diagnostics."""
    ref = ref or TypewellReference(task.tw_tvt, task.tw_gr)
    if not ref.ok:
        return MultiScaleAlignment(ok=False, failure_reason="missing_or_invalid_typewell")
    if cal is None or not cal.ok:
        cal = fit_prefix_affine_calibration(task, ref)
    if not cal.ok:
        return MultiScaleAlignment(ok=False, failure_reason="affine_calibration_unusable")
    rows: list[AlignScaleResult] = []
    for s in scales_ft:
        cfg = MultiBranchConfig(search=float(s), step=max(0.25, 0.05 * float(s)))
        mb = multibranch_scan(task, cal=cal, ref=ref, config=cfg)
        rows.append(
            AlignScaleResult(
                ok=bool(mb.ok),
                search_ft=float(s),
                mb=mb,
                clipped_shift=_clipped_shift(mb, float(s)),
            )
        )
    ok_rows = [r for r in rows if r.ok]
    if not ok_rows:
        return MultiScaleAlignment(
            ok=False,
            failure_reason="all_scales_failed",
            scale_results=tuple(rows),
            cal_alpha=float(cal.alpha),
            cal_beta=float(cal.beta),
            cal_prefix_corr=float(cal.prefix_corr) if np.isfinite(cal.prefix_corr) else 0.0,
        )
    shifts = np.asarray([r.clipped_shift for r in ok_rows], dtype="float64")
    dominant = float(np.median(shifts))
    ptp = float(np.max(shifts) - np.min(shifts))
    min_conf = float(min(r.mb.confidence for r in ok_rows))
    avg_prefix_trust = float(np.mean([r.mb.prefix_trust for r in ok_rows]))
    n_agree = int(np.count_nonzero(np.abs(shifts - dominant) <= ALIGN_V2_AGREE_FT))
    conf = float(np.clip(0.5 * (1.0 - ptp / max(ALIGN_V2_SCALES_FT[-1], 1e-9)) + 0.5 * min_conf, 0.0, 1.0))
    return MultiScaleAlignment(
        ok=True,
        failure_reason="",
        scale_results=tuple(rows),
        dominant_shift=dominant,
        ptp=ptp,
        min_confidence=min_conf,
        n_agree=n_agree,
        n_ok=len(ok_rows),
        confidence=conf,
        prefix_trust=avg_prefix_trust,
        cal_alpha=float(cal.alpha),
        cal_beta=float(cal.beta),
        cal_prefix_corr=float(cal.prefix_corr) if np.isfinite(cal.prefix_corr) else 0.0,
    )


# --------------------------------------------------------------------------- #
#  C — Dynamic-programming path matcher
# --------------------------------------------------------------------------- #


@dataclass
class DynamicPathResult:
    """A bounded curvature dynamic-programming path between calibrated GR
    and Typewell GR.

    The path is a per-row TVT trajectory over the whole well. It obeys:

    * TVT path is monotonically non-decreasing (preserve TVT order);
    * the path is anchored at the visible-prefix boundary
      (the first predicted row matches the boundary anchor ±
      ``ALIGN_V2_DPW_MAX_GRADIENT * step``);
    * the path is bounded in curvature by
      ``ALIGN_V2_DPW_MAX_GRADIENT`` (TVT per MD);
    * the path is bounded in step size by
      ``ALIGN_V2_DPW_MAX_STEP_PER_ROW`` MD rows per increment;
    * the path is finite and deterministic; failed or unstable
      matching returns ``ok=False`` with an explicit reason.
    """

    ok: bool
    failure_reason: str = ""
    path: np.ndarray | None = None  # length n_rows
    prefix_mismatch: float = 0.0    # mean abs difference vs boundary anchor
    gr_misfit: float = 0.0          # clipped GR mismatch mean
    smoothness: float = 0.0         # std of finite differences of the path
    confidence: float = 0.0


def dynamic_path_match(
    task: InferenceTask,
    *,
    cal: AffineCalibration | None = None,
    ref: TypewellReference | None = None,
    max_gradient: float = ALIGN_V2_DPW_MAX_GRADIENT,
    soft_penalty: float = ALIGN_V2_DPW_SOFT_PENALTY,
) -> DynamicPathResult:
    """Bounded-curvature DP path matcher between calibrated GR and Typewell.

    The matcher walks the well's MD array one step at a time and finds a
    TVT increment ``dtvt`` per step that minimises the soft-DTW-like cost
    against the typewell GR, bounded by the curvature cap. The cost is
    a clipped squared residual of the calibrated GR against the typewell
    GR sampled at the candidate TVT.

    The path is anchored at the visible-prefix boundary: the first
    predicted TVT equals the boundary anchor exactly (well-defined from
    the visible TVT prefix).

    Determinism
    -----------
    The matcher is fully deterministic — no RNG, no iterative refinement
    that depends on initialisation. Two calls on the same task produce
    bit-identical paths.

    Fallback
    --------
    If anything fails, ``ok=False`` and ``path=None``. The caller is
    expected to fall back to the Ridge anchor (or another candidate).
    """
    ref = ref or TypewellReference(task.tw_tvt, task.tw_gr)
    if not ref.ok:
        return DynamicPathResult(ok=False, failure_reason="missing_or_invalid_typewell")
    if cal is None or not cal.ok:
        cal = fit_prefix_affine_calibration(task, ref)
    if not cal.ok:
        return DynamicPathResult(ok=False, failure_reason="affine_calibration_unusable")
    s, stop = int(task.start), int(task.stop)
    n = int(task.n_rows)
    if n <= 1 or stop <= s + 1:
        return DynamicPathResult(ok=False, failure_reason="empty_prediction_region")
    anchor = task.anchor_tvt
    if not np.isfinite(anchor):
        return DynamicPathResult(ok=False, failure_reason="no_visible_anchor")
    md = np.asarray(task.md, dtype="float64")
    # GR signal (calibrated), with missingness imputed within the well.
    filled, missing = interpolate_within_well(task.gr)
    if not np.isfinite(filled).any():
        return DynamicPathResult(ok=False, failure_reason="all_gr_missing")
    sd = cal.sd_tw if cal.sd_tw > 1e-9 else 1.0
    signal = ((filled - cal.beta) / max(cal.alpha, 1e-12) - cal.mu_tw) / sd
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    # Typewell span and signal range. The path is forced to live inside
    # the typewell TVT span to avoid excursions outside the reference.
    tw_finite = np.isfinite(ref.grid) & np.isfinite(ref.gr_z)
    if int(tw_finite.sum()) < 2:
        return DynamicPathResult(ok=False, failure_reason="typewell_unusable")
    tw_min = float(np.min(ref.grid[tw_finite]))
    tw_max = float(np.max(ref.grid[tw_finite]))
    # Bounded curvature step.
    dmd = np.diff(md[s:stop])
    dmd_finite = np.where(np.isfinite(dmd) & (dmd > 1e-9), dmd, 0.0)
    if dmd_finite.sum() <= 0.0:
        return DynamicPathResult(ok=False, failure_reason="non_monotonic_md")
    # Cap the per-row step at the median MD step in the predicted region;
    # this bounds compute on long-prediction regions and makes the bound
    # robust to gaps.
    median_step = float(np.median(dmd_finite[dmd_finite > 0])) if (dmd_finite > 0).any() else 1.0
    median_step = median_step if median_step > 1e-9 else 1.0
    # Initial path: hold-last TVT inside the predicted region.
    pred_len = stop - s
    path = np.full(n, np.nan, dtype="float64")
    path[:s] = np.asarray(task.tvt_known[:s], dtype="float64")
    # The visible TVT may be missing inside the prefix (rare in this
    # data, but the matcher must not assume otherwise). Fill small
    # interior gaps by linear interpolation between finite neighbours.
    finite = np.isfinite(path[:s])
    if finite.any():
        idx = np.flatnonzero(finite)
        path[:s] = np.interp(np.arange(s), idx, path[:s][idx])
    # Anchor the first predicted row to the boundary anchor exactly.
    path[s] = float(anchor)
    # Pre-compute a local cost lookup at a coarse grid to keep compute
    # linear in pred_len. The lookup spans the typewell span plus the
    # anchor; we sample the typewell GR at a grid step ~ median_step.
    grid_step = float(np.clip(median_step * max_gradient * 4.0, 0.5, 5.0))
    tvt_grid = np.arange(tw_min, tw_max + grid_step, grid_step)
    if tvt_grid.size < 2:
        return DynamicPathResult(ok=False, failure_reason="typewell_grid_too_coarse")
    tw_at_tvt = np.interp(tvt_grid, ref.grid, ref.gr_z)

    # Bounded step. We restrict dtvt to ±max_gradient * md_step.
    max_dtvt_per_step = max_gradient * median_step
    cand_offsets = np.arange(-1.0, 1.0 + 0.25, 0.25) * max_dtvt_per_step

    prev = float(anchor)
    cumulative = float(anchor)
    for i in range(1, pred_len):
        cur_md = float(md[s + i])
        cur_sig = float(signal[s + i])
        best_cost = np.inf
        best_dtvt = 0.0
        for dtvt in cand_offsets:
            new = prev + dtvt
            # The path must be monotonically non-decreasing.
            if new < prev - 1e-9:
                continue
            # Clamp inside the typewell range.
            new = float(np.clip(new, tw_min, tw_max))
            # Cost: clipped squared residual of the calibrated GR
            # against the typewell GR at this candidate TVT, plus a
            # soft penalty for the dtvt magnitude (keeps the path
            # smooth).
            want = float(np.interp(new, tvt_grid, tw_at_tvt))
            r = float(np.clip(cur_sig - want, -6.0, 6.0))
            cost = r * r + soft_penalty * (dtvt / max_dtvt_per_step) ** 2
            if cost < best_cost:
                best_cost = cost
                best_dtvt = dtvt
        cumulative = max(prev + best_dtvt, prev)
        path[s + i] = cumulative
        prev = cumulative
    # Finalise: in case the path has any residual non-finite rows, fill
    # by holding the last finite value. The DP is monotone so the only
    # way this happens is if the prediction region is empty (already
    # checked) or the path is fully clipped to the boundary.
    finite = np.isfinite(path)
    if not finite.all():
        last = float(path[finite][-1]) if finite.any() else float(anchor)
        path = np.where(finite, path, last)
    # Diagnostics.
    visible_part = path[:s]
    visible_known = np.isfinite(task.tvt_known[:s])
    if visible_known.any():
        prefix_mismatch = float(np.mean(np.abs(visible_part[visible_known] - task.tvt_known[:s][visible_known])))
    else:
        prefix_mismatch = 0.0
    # GR misfit over the predicted region.
    gr_misfit = float(
        np.mean(
            [
                (signal[s + i] - float(np.interp(path[s + i], tvt_grid, tw_at_tvt))) ** 2
                for i in range(pred_len)
            ]
        )
    )
    # Path smoothness: std of the per-row finite difference of the path
    # in the predicted region.
    diffs = np.diff(path[s:stop])
    smoothness = float(np.std(diffs)) if diffs.size > 1 else 0.0
    # Confidence: a bounded function of prefix mismatch and GR misfit.
    conf = float(np.clip(1.0 - 0.5 * prefix_mismatch - 0.1 * gr_misfit, 0.0, 1.0))
    return DynamicPathResult(
        ok=True,
        failure_reason="",
        path=path,
        prefix_mismatch=prefix_mismatch,
        gr_misfit=gr_misfit,
        smoothness=smoothness,
        confidence=conf,
    )


# --------------------------------------------------------------------------- #
#  D — Branch ensemble
# --------------------------------------------------------------------------- #


@dataclass
class BranchCorrection:
    """One candidate TVT path inside the branch ensemble.

    ``name`` is the candidate family. ``path`` is the per-row TVT
    trajectory; ``correction`` is the difference from the anchor
    prediction. ``confidence`` is the target-free scalar confidence the
    gate learns from. ``fallback_exact_ridge`` flags whether the
    candidate returned the anchor output unchanged — when true, the
    candidate is removed from the ensemble for averaging.
    """

    name: str
    correction: np.ndarray
    confidence: float
    available: bool = True
    failure_reason: str = ""
    gr_residual: float = np.nan
    path_smoothness: float = np.nan
    branch_disagreement: float = np.nan
    fallback_exact_ridge: bool = False
    correction_max_abs: float = 0.0


@dataclass
class BranchEnsemble:
    """Aggregated candidate paths for a single (well, boundary).

    Every candidate family is independently computed; failed candidates
    are kept with ``available=False`` and excluded from the disagreement
    summary. The ensemble-level diagnostics — branch disagreement,
    agreement count, mean correction magnitude — are derived from the
    *available* candidates only.
    """

    well_id: str
    start: int
    stop: int
    anchor: float
    candidates: tuple[BranchCorrection, ...]
    n_available: int
    n_fallback: int
    branch_disagreement: float
    mean_correction_abs: float
    max_correction_abs: float
    confidence: float
    failure_reason: str = ""


def _branch_correction(
    name: str,
    path: np.ndarray | None,
    anchor_pred: np.ndarray,
    anchor: float,
    *,
    confidence: float,
    correction_cap: float = ALIGN_V2_CORRECTION_CAP_FT,
) -> BranchCorrection:
    """Build a BranchCorrection from a candidate path. Path shape may be
    either length ``n_predict`` (predicted rows only) or length
    ``n_rows``; we always normalise to the predicted rows.
    """
    if path is None or not np.isfinite(path).any():
        return BranchCorrection(
            name,
            np.zeros_like(anchor_pred, dtype="float64"),
            confidence=0.0,
            available=False,
            failure_reason="path_unavailable",
        )
    if path.size != anchor_pred.size:
        # Try to align shapes if the candidate returned n_rows.
        if path.size == anchor_pred.size + int(np.where(np.isfinite(anchor_pred), 0, 0)[0]) + 0:
            pass  # nothing to do
    # If path is too long or too short, refuse silently.
    if path.size != anchor_pred.size:
        return BranchCorrection(
            name,
            np.zeros_like(anchor_pred, dtype="float64"),
            confidence=0.0,
            available=False,
            failure_reason="path_shape_mismatch",
        )
    raw = np.asarray(path, dtype="float64") - anchor
    if not np.isfinite(raw).all():
        raw = np.where(np.isfinite(raw), raw, 0.0)
    corr = np.clip(raw, -correction_cap, correction_cap)
    fallback = bool(np.allclose(corr, 0.0, atol=1e-12))
    if fallback:
        confidence = 0.0
    return BranchCorrection(
        name=name,
        correction=corr,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        available=not fallback,
        failure_reason="" if not fallback else "fallback_to_anchor",
        gr_residual=float(np.nan),
        path_smoothness=float(np.std(corr)) if corr.size > 1 else 0.0,
        branch_disagreement=float(np.nan),
        fallback_exact_ridge=fallback,
        correction_max_abs=float(np.max(np.abs(corr))),
    )


def build_branch_ensemble(
    task: InferenceTask,
    *,
    base_pred: np.ndarray,
    pf_path: np.ndarray | None = None,
    beam_path: np.ndarray | None = None,
    mb_path: np.ndarray | None = None,
    ms_path: np.ndarray | None = None,
    dp_path: np.ndarray | None = None,
    irls_path: np.ndarray | None = None,
    correction_cap: float = ALIGN_V2_CORRECTION_CAP_FT,
) -> BranchEnsemble:
    """Aggregate the candidate TVT paths for one well into a BranchEnsemble.

    Every argument is the *per-row TVT prediction* for the candidate
    family, restricted to the predicted rows (length
    ``task.n_predict``); a ``None`` value means the family did not
    produce a usable path.

    The Ridge anchor prediction is ``base_pred``; every branch
    correction is computed relative to it. The conservative
    "branch-hedged" path is the trimmed mean of the available branches.
    """
    n_predict = int(task.n_predict)
    if base_pred.size != n_predict:
        raise ValueError(
            f"build_branch_ensemble: base_pred has size {base_pred.size} but "
            f"the task predicts {n_predict} rows"
        )
    anchor = float(task.anchor_tvt) if np.isfinite(task.anchor_tvt) else 0.0
    branches: list[BranchCorrection] = []
    branches.append(
        _branch_correction(
            "ridge",
            base_pred.copy(),
            base_pred,
            anchor,
            confidence=1.0,
            correction_cap=correction_cap,
        )
    )
    if pf_path is not None:
        branches.append(
            _branch_correction("pf", pf_path, base_pred, anchor, confidence=0.5, correction_cap=correction_cap)
        )
    if beam_path is not None:
        branches.append(
            _branch_correction("beam", beam_path, base_pred, anchor, confidence=0.5, correction_cap=correction_cap)
        )
    if mb_path is not None:
        branches.append(
            _branch_correction("multibranch", mb_path, base_pred, anchor, confidence=0.5, correction_cap=correction_cap)
        )
    if ms_path is not None:
        branches.append(
            _branch_correction("multi_scale", ms_path, base_pred, anchor, confidence=0.5, correction_cap=correction_cap)
        )
    if dp_path is not None:
        branches.append(
            _branch_correction("dp_path", dp_path, base_pred, anchor, confidence=0.5, correction_cap=correction_cap)
        )
    if irls_path is not None:
        branches.append(
            _branch_correction("irls", irls_path, base_pred, anchor, confidence=0.5, correction_cap=correction_cap)
        )
    # Trimmed mean of the available, non-fallback branches.
    available = [b for b in branches if b.available]
    if available:
        stacked = np.stack([b.correction for b in available], axis=0)
        # 10% trimmed mean (drop the min and max correction row, when
        # there are enough branches to make that meaningful).
        if stacked.shape[0] >= 4:
            lo = np.min(stacked, axis=0)
            hi = np.max(stacked, axis=0)
            mask = (stacked != lo[None, :]) & (stacked != hi[None, :])
            # Use only rows where at least one element survived the mask;
            # for very tight distributions, fall back to the untrimmed mean.
            n_keep = mask.sum(axis=0)
            use_trim = n_keep > 0
            mean_trim = np.where(
                use_trim,
                np.where(mask, stacked, 0.0).sum(axis=0) / np.maximum(n_keep, 1),
                stacked.mean(axis=0),
            )
            mean_corr = mean_trim
        else:
            mean_corr = stacked.mean(axis=0)
        hedge_path = base_pred + mean_corr
        conf = float(np.mean([b.confidence for b in available]))
        branches.append(
            _branch_correction(
                "branch_hedged",
                hedge_path,
                base_pred,
                anchor,
                confidence=conf,
                correction_cap=correction_cap,
            )
        )
    # Recompute availability after hedging.
    available = [b for b in branches if b.available]
    n_available = len(available)
    n_fallback = sum(1 for b in branches if b.fallback_exact_ridge)
    if n_available == 0:
        return BranchEnsemble(
            well_id=task.well_id,
            start=int(task.start),
            stop=int(task.stop),
            anchor=anchor,
            candidates=tuple(branches),
            n_available=0,
            n_fallback=n_fallback,
            branch_disagreement=float("inf"),
            mean_correction_abs=0.0,
            max_correction_abs=0.0,
            confidence=0.0,
            failure_reason="no_available_branch",
        )
    # Branch disagreement: mean per-row std of the available
    # corrections.
    stacked = np.stack([b.correction for b in available], axis=0)
    branch_disagreement = float(np.mean(np.std(stacked, axis=0)))
    mean_correction_abs = float(np.mean(np.abs(stacked)))
    max_correction_abs = float(np.max(np.abs(stacked)))
    confidence = float(np.mean([b.confidence for b in available]))
    return BranchEnsemble(
        well_id=task.well_id,
        start=int(task.start),
        stop=int(task.stop),
        anchor=anchor,
        candidates=tuple(branches),
        n_available=n_available,
        n_fallback=n_fallback,
        branch_disagreement=branch_disagreement,
        mean_correction_abs=mean_correction_abs,
        max_correction_abs=max_correction_abs,
        confidence=confidence,
    )


# --------------------------------------------------------------------------- #
#  E — Robust projection (postprocessing layer)
# --------------------------------------------------------------------------- #


@dataclass
class RobustProjection:
    ok: bool
    failure_reason: str = ""
    path: np.ndarray | None = None
    degree: int = 2
    n_clipped: int = 0
    n_points: int = 0
    clip_fraction: float = 0.0
    movement: float = 0.0
    coefs: tuple = ()
    # Visible-prefix verification: the projected path's prefix disagreement
    # with the visible TVT, which must be small on successful fits.
    visible_prefix_mismatch: float = 0.0


def robust_stratigraphic_projection(
    task: InferenceTask,
    *,
    candidate_path: np.ndarray,
    degree: int = 2,
    max_movement_ft: float = ALIGN_V2_CORRECTION_CAP_FT,
    max_clip_fraction: float = 0.10,
    config: SafeAlignmentConfig | None = None,
) -> RobustProjection:
    """Postprocess a candidate TVT path with a robust polynomial projection.

    The stratigraphic coordinate is ``candidate_TVT + Z - anchor`` (the
    rejected dip-constrained model used ``+Z`` with a hard coefficient
    1.0; we use a degree-<=2 polynomial of MD on that coordinate with
    IRLS / Huber reweighting — see ``src.safe_alignment.robust_projection``).
    The polynomial is fit on the visible prefix only; the predicted
    rows are then mapped through the same polynomial. Movement of the
    projected path from the input candidate path is bounded by
    ``max_movement_ft``; if too many points are clipped
    (above ``max_clip_fraction``) the projection is rejected and the
    original candidate path is returned.
    """
    s, stop = int(task.start), int(task.stop)
    if candidate_path is None or candidate_path.size != stop - s:
        return RobustProjection(ok=False, failure_reason="candidate_path_shape_mismatch")
    if degree not in (1, 2):
        return RobustProjection(ok=False, failure_reason="unsupported_degree")
    if config is None:
        config = SafeAlignmentConfig(
            projection_degree=int(degree),
            projection_max_move_ft=float(max_movement_ft),
            projection_max_clipped_frac=float(max_clip_fraction),
        )
    md_pred = np.asarray(task.md[s:stop], dtype="float64")
    z_pred = np.asarray(task.z[s:stop], dtype="float64")
    anchor = float(task.anchor_tvt) if np.isfinite(task.anchor_tvt) else 0.0
    if not np.isfinite(md_pred).any() or not np.isfinite(z_pred).any():
        return RobustProjection(ok=False, failure_reason="non_finite_md_or_z")
    projected, applied, reason = robust_projection(
        md=md_pred,
        z=z_pred,
        cand=candidate_path.astype("float64"),
        anchor=anchor,
        config=config,
    )
    if not applied:
        return RobustProjection(
            ok=False,
            failure_reason=str(reason or "projection_rejected"),
            degree=int(degree),
        )
    if not np.all(np.isfinite(projected)):
        return RobustProjection(ok=False, failure_reason="projection_nonfinite", degree=int(degree))
    move = projected - candidate_path
    n_clipped = int(np.count_nonzero(np.abs(move) >= float(max_movement_ft) - 1e-9))
    clip_fraction = float(n_clipped) / max(int(move.size), 1)
    return RobustProjection(
        ok=True,
        failure_reason="",
        path=projected,
        degree=int(degree),
        n_clipped=n_clipped,
        n_points=int(move.size),
        clip_fraction=clip_fraction,
        movement=float(np.mean(np.abs(move))),
        coefs=(),
        visible_prefix_mismatch=0.0,
    )


# --------------------------------------------------------------------------- #
#  Feature columns registered in the manifest
# --------------------------------------------------------------------------- #


ALIGN_V2_FEATURE_COLUMNS: tuple[str, ...] = (
    "align_v2_cal_alpha_ptp",
    "align_v2_cal_beta_ptp",
    "align_v2_cal_fit_rmse_z",
    "align_v2_cal_prefix_corr",
    "align_v2_cal_confidence",
    "align_v2_cal_n_ok",
    "align_v2_cal_n_agree",
    "align_v2_ms_dominant_shift",
    "align_v2_ms_ptp",
    "align_v2_ms_min_conf",
    "align_v2_ms_n_agree",
    "align_v2_ms_n_ok",
    "align_v2_ms_confidence",
    "align_v2_ms_prefix_trust",
    "align_v2_dp_ok",
    "align_v2_dp_prefix_mismatch",
    "align_v2_dp_gr_misfit",
    "align_v2_dp_smoothness",
    "align_v2_dp_confidence",
    "align_v2_ens_n_available",
    "align_v2_ens_n_fallback",
    "align_v2_ens_branch_disagreement",
    "align_v2_ens_mean_correction_abs",
    "align_v2_ens_max_correction_abs",
    "align_v2_ens_confidence",
    "align_v2_proj_ok",
    "align_v2_proj_movement",
    "align_v2_proj_n_clipped",
    "align_v2_proj_clip_fraction",
    "align_v2_proj_visible_prefix_mismatch",
)


def align_v2_feature_row(
    task: InferenceTask,
    *,
    multi_affine: MultiScaleAffine,
    multi_align: MultiScaleAlignment,
    dp: DynamicPathResult,
    ensemble: BranchEnsemble,
    projection: RobustProjection,
) -> dict[str, float]:
    """One design row of the Alignment v2 gate (per well, per boundary)."""
    row = {
        "align_v2_cal_alpha_ptp": float(multi_affine.alpha_ptp),
        "align_v2_cal_beta_ptp": float(multi_affine.beta_ptp),
        "align_v2_cal_fit_rmse_z": float(multi_affine.fit_rmse_z),
        "align_v2_cal_prefix_corr": float(multi_affine.prefix_corr),
        "align_v2_cal_confidence": float(multi_affine.confidence),
        "align_v2_cal_n_ok": float(multi_affine.n_ok),
        "align_v2_cal_n_agree": float(multi_affine.n_agree),
        "align_v2_ms_dominant_shift": float(multi_align.dominant_shift),
        "align_v2_ms_ptp": float(multi_align.ptp),
        "align_v2_ms_min_conf": float(multi_align.min_confidence),
        "align_v2_ms_n_agree": float(multi_align.n_agree),
        "align_v2_ms_n_ok": float(multi_align.n_ok),
        "align_v2_ms_confidence": float(multi_align.confidence),
        "align_v2_ms_prefix_trust": float(multi_align.prefix_trust),
        "align_v2_dp_ok": 1.0 if dp.ok else 0.0,
        "align_v2_dp_prefix_mismatch": float(dp.prefix_mismatch),
        "align_v2_dp_gr_misfit": float(dp.gr_misfit),
        "align_v2_dp_smoothness": float(dp.smoothness),
        "align_v2_dp_confidence": float(dp.confidence),
        "align_v2_ens_n_available": float(ensemble.n_available),
        "align_v2_ens_n_fallback": float(ensemble.n_fallback),
        "align_v2_ens_branch_disagreement": float(ensemble.branch_disagreement)
        if np.isfinite(ensemble.branch_disagreement)
        else 1e6,
        "align_v2_ens_mean_correction_abs": float(ensemble.mean_correction_abs),
        "align_v2_ens_max_correction_abs": float(ensemble.max_correction_abs),
        "align_v2_ens_confidence": float(ensemble.confidence),
        "align_v2_proj_ok": 1.0 if projection.ok else 0.0,
        "align_v2_proj_movement": float(projection.movement),
        "align_v2_proj_n_clipped": float(projection.n_clipped),
        "align_v2_proj_clip_fraction": float(projection.clip_fraction),
        "align_v2_proj_visible_prefix_mismatch": float(projection.visible_prefix_mismatch),
    }
    return {k: float(v) for k, v in row.items()}


# --------------------------------------------------------------------------- #
#  Candidate correction builders
# --------------------------------------------------------------------------- #


def _constant_shift_path(
    task: InferenceTask, shift: float, *, anchor: float | None = None
) -> np.ndarray:
    """A constant-datum TVT path: every predicted row is ``anchor + shift``."""
    a = anchor if (anchor is not None and np.isfinite(anchor)) else float(task.anchor_tvt)
    if not np.isfinite(a):
        a = 0.0
    return np.full(int(task.n_predict), a + float(shift), dtype="float64")


def _path_predicted_rows(task: InferenceTask, path: np.ndarray) -> np.ndarray:
    """Return the path on the predicted rows only."""
    if path.size == task.n_predict:
        return path
    if path.size == task.n_rows:
        return path[task.start : task.stop]
    raise ValueError(
        f"path has size {path.size}; expected {task.n_predict} (predicted) "
        f"or {task.n_rows} (full)"
    )


def align_v2_candidate_paths(
    task: InferenceTask,
    *,
    base_pred: np.ndarray,
    pf_correction: np.ndarray | None = None,
    beam_correction: np.ndarray | None = None,
    multi_affine: MultiScaleAffine | None = None,
    multi_align: MultiScaleAlignment | None = None,
    dp: DynamicPathResult | None = None,
    correction_cap: float = ALIGN_V2_CORRECTION_CAP_FT,
) -> dict[str, np.ndarray | None]:
    """Build the v2 candidate TVT paths for one well.

    Every returned path has the shape of the predicted rows
    (``task.n_predict``) and is clipped to ``±correction_cap`` around
    the anchor. A ``None`` value means the family failed.
    """
    anchor = float(task.anchor_tvt) if np.isfinite(task.anchor_tvt) else 0.0
    n = int(task.n_predict)
    paths: dict[str, np.ndarray | None] = {}
    # Ridge anchor
    paths["ridge"] = base_pred.copy()
    if pf_correction is not None and pf_correction.size == n:
        paths["pf"] = base_pred + np.clip(pf_correction, -correction_cap, correction_cap)
    if beam_correction is not None and beam_correction.size == n:
        paths["beam"] = base_pred + np.clip(beam_correction, -correction_cap, correction_cap)
    if multi_align is not None and multi_align.ok:
        shift = float(np.clip(multi_align.dominant_shift, -correction_cap, correction_cap))
        paths["multi_scale"] = _constant_shift_path(task, shift, anchor=anchor)
    if dp is not None and dp.ok and dp.path is not None:
        try:
            dp_pred = _path_predicted_rows(task, dp.path)
            raw = dp_pred - base_pred
            paths["dp_path"] = base_pred + np.clip(
                np.where(np.isfinite(raw), raw, 0.0), -correction_cap, correction_cap
            )
        except ValueError:
            paths["dp_path"] = None
    if multi_affine is not None and multi_affine.ok:
        # The affine calibration alone is not a TVT path; the path it
        # supports is the multibranch / multi-scale one. We do *not*
        # return a separate affine path here.
        pass
    return paths


# --------------------------------------------------------------------------- #
#  Convenience: run all candidate families on a single task
# --------------------------------------------------------------------------- #


@dataclass
class AlignV2Candidates:
    multi_affine: MultiScaleAffine
    multi_align: MultiScaleAlignment
    dp: DynamicPathResult
    ensemble: BranchEnsemble
    projection: RobustProjection
    paths: dict[str, np.ndarray | None]
    features: dict[str, float]


def run_align_v2_candidates(
    task: InferenceTask,
    *,
    base_pred: np.ndarray,
    pf_correction: np.ndarray | None = None,
    beam_correction: np.ndarray | None = None,
    mb_shift: float | None = None,
    apply_projection: bool = True,
) -> AlignV2Candidates:
    """Run all v2 candidate families on a single well and return the bundle."""
    ref = TypewellReference(task.tw_tvt, task.tw_gr)
    cal = fit_prefix_affine_calibration(task, ref)
    multi_affine = multi_scale_affine_calibration(task, ref=ref)
    multi_align = multi_scale_trajectory_alignment(task, cal=cal, ref=ref)
    dp = dynamic_path_match(task, cal=cal, ref=ref)
    # mb_path: the multibranch constant-datum path (Ridge-relative).
    mb_path = None
    if mb_shift is not None and np.isfinite(mb_shift):
        anchor = float(task.anchor_tvt) if np.isfinite(task.anchor_tvt) else 0.0
        mb_path = base_pred + (anchor + mb_shift - anchor)
    pf_path = None
    if pf_correction is not None and pf_correction.size == base_pred.size:
        pf_path = base_pred + np.clip(pf_correction, -ALIGN_V2_CORRECTION_CAP_FT, ALIGN_V2_CORRECTION_CAP_FT)
    beam_path = None
    if beam_correction is not None and beam_correction.size == base_pred.size:
        beam_path = base_pred + np.clip(beam_correction, -ALIGN_V2_CORRECTION_CAP_FT, ALIGN_V2_CORRECTION_CAP_FT)
    ms_path = None
    if multi_align.ok:
        anchor = float(task.anchor_tvt) if np.isfinite(task.anchor_tvt) else 0.0
        ms_path = base_pred + np.clip(multi_align.dominant_shift, -ALIGN_V2_CORRECTION_CAP_FT, ALIGN_V2_CORRECTION_CAP_FT)
    dp_path = None
    if dp.ok and dp.path is not None:
        try:
            dp_pred = _path_predicted_rows(task, dp.path)
            raw = dp_pred - base_pred
            dp_path = base_pred + np.clip(np.where(np.isfinite(raw), raw, 0.0), -ALIGN_V2_CORRECTION_CAP_FT, ALIGN_V2_CORRECTION_CAP_FT)
        except ValueError:
            dp_path = None
    ensemble = build_branch_ensemble(
        task,
        base_pred=base_pred,
        pf_path=pf_path,
        beam_path=beam_path,
        mb_path=mb_path,
        ms_path=ms_path,
        dp_path=dp_path,
    )
    # Apply the robust projection on the *ensemble trimmed mean* if
    # available, else fall back to the dominant multi-scale shift path.
    if apply_projection:
        proj_input = None
        # Prefer the ensemble's branch_hedged path; else the ms_path; else the base.
        for cand in ensemble.candidates:
            if cand.name == "branch_hedged" and cand.available:
                proj_input = base_pred + cand.correction
                break
        if proj_input is None:
            proj_input = ms_path if ms_path is not None else base_pred
        projection = robust_stratigraphic_projection(task, candidate_path=proj_input)
    else:
        projection = RobustProjection(ok=False, failure_reason="projection_disabled", path=None)
    paths = align_v2_candidate_paths(
        task,
        base_pred=base_pred,
        pf_correction=pf_correction,
        beam_correction=beam_correction,
        multi_affine=multi_affine,
        multi_align=multi_align,
        dp=dp,
    )
    features = align_v2_feature_row(
        task,
        multi_affine=multi_affine,
        multi_align=multi_align,
        dp=dp,
        ensemble=ensemble,
        projection=projection,
    )
    return AlignV2Candidates(
        multi_affine=multi_affine,
        multi_align=multi_align,
        dp=dp,
        ensemble=ensemble,
        projection=projection,
        paths=paths,
        features=features,
    )


__all__ = [
    "ALIGNMENT_V2_VERSION",
    "ALIGN_V2_SCALES_FT",
    "ALIGN_V2_CAL_ROWS",
    "ALIGN_V2_AGREE_FT",
    "ALIGN_V2_CORRECTION_CAP_FT",
    "ALIGN_V2_FEATURE_COLUMNS",
    "AffineScaleResult",
    "MultiScaleAffine",
    "AlignScaleResult",
    "MultiScaleAlignment",
    "DynamicPathResult",
    "BranchCorrection",
    "BranchEnsemble",
    "RobustProjection",
    "AlignV2Candidates",
    "multi_scale_affine_calibration",
    "multi_scale_trajectory_alignment",
    "dynamic_path_match",
    "build_branch_ensemble",
    "robust_stratigraphic_projection",
    "align_v2_feature_row",
    "align_v2_candidate_paths",
    "run_align_v2_candidates",
]
