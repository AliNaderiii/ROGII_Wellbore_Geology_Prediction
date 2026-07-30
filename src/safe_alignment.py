"""Safe alignment candidate pipeline — staged experiment A–F.

Target architecture (see reports/safe_alignment_protocol.md):

    Ridge anchor
    + PF/Beam candidate trajectories          (stage B)
    + affine GR heel calibration              (stage C)
    + bimodal branch detection / hedging      (stage D)
    + robust IRLS projection                  (stage E)
    + visible-prefix self-verification + tail/disagreement guard (stage F)

Every stage is a strict superset of the previous one and every stage falls
back to the *exact* Ridge Default prediction whenever any component fails,
any guard declines, or anything is non-finite. Stage A *is* Ridge Default.

Leakage contract (non-negotiable)
---------------------------------
Inference reads only: MD, X, Y, Z, GR (and its missingness), the visible
``TVT_input`` prefix, Typewell TVT and Typewell GR. The hidden TVT label is
structurally unreachable (``InferenceTask`` carries no target); the three
public duplicate test wells are excluded from every fold by
``src.validation.BLOCKED_WELL_IDS``; nothing here reads Typewell Geology,
formation markers, external artifacts, Koolbox, or any public-LB signal.

All numeric constants below are a-priori algorithmic sanity bounds documented
at their definition. The only tuned quantities — the stage-F confidence
threshold and warmup length — are selected per fold from **fold-training
wells only** (their own targets, which are legitimately visible in-fold),
over a small fixed grid; nothing is ever tuned on the public leaderboard or
on the blocked test wells.

Ideas re-implemented independently from the studied public notebooks (never
copied): anchor blending, heel affine calibration, multi-branch datum
scanning with conservative hedging, robust projection in a stratigraphic
coordinate, multi-cut visible-prefix verification, warmup smoothing, and
tail/disagreement guards.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from src.baselines import BaselineModel, RidgeBaseline
from src.features import TypewellReference, interpolate_within_well
from src.geoanchor import (
    CORRECTION_CAP_FT,
    CandidateCorrection,
    MemoizedPathGenerator,
    _finite_rmse,
    _tail_mean_se,
    fit_prefix_affine_calibration,
    generate_candidate_corrections,
    multibranch_scan,
    nested_pseudo_task,
)
from src.tasks import MIN_PREFIX_ROWS, InferenceTask, WellTask

SAFE_ALIGNMENT_VERSION = "safe-alignment-v1"

# --------------------------------------------------------------------------
# Stage registry
# --------------------------------------------------------------------------

STAGE_A = "ridge_default"
STAGE_B = "safe_b_anchor_blend"
STAGE_C = "safe_c_affine_cal"
STAGE_D = "safe_d_branch_guard"
STAGE_E = "safe_e_projection"
STAGE_F = "safe_f_verified"

STAGE_ORDER = (STAGE_A, STAGE_B, STAGE_C, STAGE_D, STAGE_E, STAGE_F)
STAGE_LEVEL = {name: i for i, name in enumerate(STAGE_ORDER)}

STAGE_LABELS = {
    STAGE_A: "A. Ridge Default (exact fallback / anchor)",
    STAGE_B: "B. Ridge + bounded PF/Beam anchor blend",
    STAGE_C: "C. B + affine GR heel calibration trust",
    STAGE_D: "D. C + multi-branch / bimodal hedging guard",
    STAGE_E: "E. D + robust IRLS stratigraphic projection",
    STAGE_F: "F. E + multi-cut prefix verification + tail/disagreement guard",
}

#: Stage G (OOF residual GBDT) is deliberately not implemented here: the
#: mandate authorises it only when LightGBM/CatBoost is already available AND
#: an earlier candidate shows a real improvement. The runner reports this
#: gate honestly instead of running an ungated booster.
STAGE_G_STATUS = "gated_off:requires_real_B_to_F_evidence_and_lightgbm_or_catboost"


@dataclass(frozen=True)
class SafeAlignmentConfig:
    """A-priori bounds. Nothing here was tuned on any leaderboard."""

    #: Hard cap on any applied correction (ft of TVT). Matches the
    #: repository-wide CORRECTION_CAP_FT (25 ft, the GR/typewell search
    #: radius of ``GRTypewellMatching``).
    correction_cap_ft: float = CORRECTION_CAP_FT
    #: Reduced cap used when the affine heel calibration is unusable
    #: (stage >= C): half of the full cap, a conservative a-priori choice.
    reduced_cap_ft: float = CORRECTION_CAP_FT / 2.0
    #: Affine-fit quality gate: prefix fit RMSE (typewell-σ units) above
    #: which the calibration is treated as low-trust.
    affine_fit_rmse_z_max: float = 2.0
    #: Trust multiplier applied to corrections when calibration is low-trust.
    low_trust_factor: float = 0.5
    #: Bimodal ambiguity guard (stage >= D): a second branch within this
    #: normalised cost gap of the best branch, separated by more than
    #: ``ambiguous_sep_ft``, marks the well ambiguous.
    ambiguous_cost_gap: float = 0.05
    ambiguous_sep_ft: float = 6.0
    #: Robust projection (stage >= E).
    projection_degree: int = 2
    projection_min_rows: int = 50
    projection_max_move_ft: float = 5.0
    projection_irls_iters: int = 5
    projection_huber_c: float = 1.345
    #: Fraction of clip-saturated rows above which the projection fit is
    #: declared unstable and rejected.
    projection_max_clipped_frac: float = 0.5
    #: Warmup ramp length (rows) applied to the final correction; the
    #: stage-F fold-train tuner may pick a different value from the grid.
    warmup_rows: int = 200
    warmup_grid: tuple = (100, 200, 400)
    #: Stage-F confidence threshold grid (candidate confidence in [0, 1]).
    conf_grid: tuple = (0.0, 0.25, 0.5)
    #: Stage-F PF/Beam (or branch-separation) disagreement cap, ft. Two
    #: thirds of the multibranch scan half-range (15 ft), fixed a priori.
    disagreement_cap_ft: float = 10.0
    #: Stage-F visible-prefix verification cuts (fractions of the prefix).
    #: May only be changed with training-well OOF evidence.
    verify_cuts: tuple = (0.50, 0.65, 0.75)
    #: Minimum number of valid cuts that must improve (when >= 2 are valid).
    min_cuts_improved: int = 2
    #: Maximum fold-train wells used by the stage-F threshold tuner.
    tune_max_wells: int = 40
    #: Minimum rows for a pseudo-holdout window.
    min_cut_predict: int = 25
    seed: int = 0


# --------------------------------------------------------------------------
# Pseudo-holdout cuts inside the visible prefix
# --------------------------------------------------------------------------


def pseudo_task_at(task: InferenceTask, cut_row: int, *, min_predict: int = 25):
    """A pseudo-task whose boundary sits at ``cut_row`` inside the prefix.

    Predicts ``cut_row .. task.start``; truth is the ``TVT_input`` rows the
    parent treated as visible. Entirely target-free. Returns ``None`` when
    the window is unusable.
    """
    base_start = int(task.start)
    cut_row = int(cut_row)
    if cut_row < MIN_PREFIX_ROWS or base_start - cut_row < min_predict:
        return None
    tvt_known = np.asarray(task.tvt_known, dtype="float64").copy()
    truth = tvt_known[cut_row:base_start].copy()
    if not np.isfinite(truth).any():
        return None
    tvt_known[cut_row:] = np.nan
    if not np.isfinite(tvt_known[:cut_row]).any():
        return None
    nested = replace(
        task,
        start=cut_row,
        stop=base_start,
        tvt_known=tvt_known,
        mode=f"cut{cut_row}_{task.mode}",
    )
    nested.assert_no_target()
    return nested, truth


# --------------------------------------------------------------------------
# Robust projection (stage E)
# --------------------------------------------------------------------------


def robust_projection(
    md: np.ndarray,
    z: np.ndarray,
    cand: np.ndarray,
    anchor: float,
    config: SafeAlignmentConfig,
) -> tuple[np.ndarray, bool, str]:
    """IRLS low-degree projection of a candidate track in U = cand + Z - anchor.

    ``U`` is a target-free stratigraphic-style coordinate: the candidate's
    TVT plus true vertical position, referenced to the visible anchor. A
    Huber-weighted polynomial (degree <= 2) smooths ``U`` along MD; the
    smoothed trajectory is mapped back and the movement is bounded by
    ``projection_max_move_ft``. Unstable fits are rejected (candidate kept).

    Returns ``(projected_track, applied, reason)``.
    """
    n = cand.size
    if n < config.projection_min_rows:
        return cand, False, "projection_too_few_rows"
    z_filled, z_missing = interpolate_within_well(z)
    if z_missing.all():
        return cand, False, "projection_no_z"
    m = np.isfinite(md) & np.isfinite(cand)
    if int(m.sum()) < config.projection_min_rows:
        return cand, False, "projection_too_few_finite_rows"
    span = float(np.nanmax(md[m]) - np.nanmin(md[m]))
    if not np.isfinite(span) or span <= 0:
        return cand, False, "projection_degenerate_md"

    t = (md - float(np.nanmin(md[m]))) / span
    u = cand + z_filled - anchor
    w = np.where(m, 1.0, 0.0)

    coeffs = None
    for _ in range(config.projection_irls_iters):
        try:
            coeffs = np.polyfit(t[m], u[m], config.projection_degree, w=np.sqrt(w[m]))
        except (np.linalg.LinAlgError, ValueError):
            return cand, False, "projection_fit_failed"
        resid = u - np.polyval(coeffs, t)
        r = resid[m]
        scale = float(np.median(np.abs(r - np.median(r)))) * 1.4826
        if not np.isfinite(scale) or scale < 1e-9:
            break  # already an (almost) exact fit
        c = config.projection_huber_c * scale
        w_new = np.where(m, np.minimum(1.0, c / np.maximum(np.abs(resid), 1e-12)), 0.0)
        if np.allclose(w_new[m], w[m], atol=1e-6):
            w = w_new
            break
        w = w_new
    if coeffs is None or not np.all(np.isfinite(coeffs)):
        return cand, False, "projection_nonfinite_fit"

    proj_u = np.polyval(coeffs, t)
    proj = proj_u - z_filled + anchor
    move = proj - cand
    move = np.where(np.isfinite(move), move, 0.0)
    clipped = np.abs(move) > config.projection_max_move_ft
    if float(np.mean(clipped)) > config.projection_max_clipped_frac:
        return cand, False, "projection_unstable_fit"
    move = np.clip(move, -config.projection_max_move_ft, config.projection_max_move_ft)
    out = cand + move
    if not np.all(np.isfinite(out)):
        return cand, False, "projection_nonfinite_output"
    return out, True, ""


# --------------------------------------------------------------------------
# Decision bookkeeping
# --------------------------------------------------------------------------


@dataclass
class StageDecision:
    """One well's decision record — the runner aggregates these."""

    well_id: str
    stage: str
    protocol: str
    fold: int
    outcome: str  # "applied" | "fallback"
    candidate: str = ""
    reason: str = ""
    confidence: float = np.nan
    disagreement: float = np.nan
    pseudo_delta: float = np.nan
    correction_mean_abs: float = 0.0
    correction_max_abs: float = 0.0
    cap_used: float = np.nan
    trust_factor: float = np.nan
    cal_ok: bool | None = None
    mb_ok: bool | None = None
    mb_bimodal: bool | None = None
    ambiguity_guard: bool = False
    projection_applied: bool = False
    projection_reason: str = ""
    n_cuts_valid: int = 0
    n_cuts_improved: int = 0
    tail_guard_failed: bool = False
    conf_threshold: float = np.nan
    warmup_rows: int = 0
    seconds: float = 0.0


@dataclass
class _Bundle:
    """Target-free per-boundary artifacts shared by real and pseudo cuts."""

    base: np.ndarray
    cands: dict
    cal_ok: bool
    cal_low_trust: bool
    mb: Any
    cap: float
    trust: float
    ambiguous: bool


# --------------------------------------------------------------------------
# The staged model
# --------------------------------------------------------------------------


class SafeAlignmentModel(BaselineModel):
    """Ridge Default anchor plus the staged, guarded alignment correction.

    ``predict`` returns the anchor's exact output whenever anything declines.
    """

    needs_alignment = False

    def __init__(
        self,
        stage: str,
        *,
        pf: MemoizedPathGenerator,
        beam: MemoizedPathGenerator,
        config: SafeAlignmentConfig | None = None,
        protocol: str = "",
        fold: int = -1,
        decision_log: list | None = None,
        tune: bool = True,
        anchor_model: RidgeBaseline | None = None,
    ) -> None:
        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
        self.stage = stage
        self.level = STAGE_LEVEL[stage]
        self.name = stage
        self.pf = pf
        self.beam = beam
        self.config = config or SafeAlignmentConfig()
        self.protocol = protocol
        self.fold = fold
        self.decision_log = decision_log if decision_log is not None else []
        self.tune = tune
        #: Shared anchor: when provided, all stages of a fold use the *same*
        #: fitted Ridge Default instance, so the fallback is bit-identical to
        #: the scored ridge_default arm and Ridge is fitted once per fold.
        self.anchor_model: RidgeBaseline | None = anchor_model
        # Stage-F selected thresholds (fold-train tuned; defaults otherwise).
        self.conf_thr: float = float(self.config.conf_grid[0])
        self.warmup: int = int(self.config.warmup_rows)
        self._last_diagnostics: dict = {}

    # ------------------------------------------------------------- fitting

    def fit(self, tasks: list[WellTask], **kw) -> "SafeAlignmentModel":
        if self.anchor_model is None:
            self.anchor_model = RidgeBaseline(alignment_features=False)
        if self.anchor_model.model is None:  # fit the shared anchor only once
            self.anchor_model.fit(tasks)
        if self.stage == STAGE_F and self.tune:
            try:
                self._tune_on_fold_train(tasks)
            except Exception:
                # Tuning is best-effort; defaults are safe a-priori values.
                self.conf_thr = float(self.config.conf_grid[0])
                self.warmup = int(self.config.warmup_rows)
        return self

    def _tune_on_fold_train(self, tasks: list[WellTask]) -> None:
        """Select (conf_thr, warmup) from fold-training wells only.

        Uses fold-train targets (legitimately in-fold) on a deterministic
        subsample; the expensive per-well artifacts are computed once and the
        small grid is applied cheaply on top.
        """
        cfg = self.config
        scored = [t for t in tasks if t.target is not None]
        if len(scored) < 8:
            return
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(len(scored))[: cfg.tune_max_wells]
        artifacts = []
        for i in order:
            wt = scored[int(i)]
            inp = wt.inputs()
            inp.assert_no_target()
            try:
                base, corr, dec = self._decide(inp)
            except Exception:
                continue
            truth = np.asarray(wt.scored(), dtype="float64")
            if truth.size != base.size:
                continue
            artifacts.append((base, corr, dec, truth))
        if len(artifacts) < 8:
            return
        best = None
        for conf_thr in cfg.conf_grid:
            for warmup in cfg.warmup_grid:
                sse, n = 0.0, 0
                for base, corr, dec, truth in artifacts:
                    if corr is None or dec.confidence < conf_thr:
                        pred = base
                    else:
                        pred = base + corr * _ramp(base.size, warmup)
                    m = np.isfinite(pred) & np.isfinite(truth)
                    if m.any():
                        d = pred[m] - truth[m]
                        sse += float(np.sum(d * d))
                        n += int(m.sum())
                if n == 0:
                    continue
                key = (sse / n, conf_thr, warmup)  # deterministic tie-break
                if best is None or key < best:
                    best = key
        if best is not None:
            self.conf_thr = float(best[1])
            self.warmup = int(best[2])

    # ---------------------------------------------------------- components

    def _bundle(self, task: InferenceTask, base: np.ndarray) -> _Bundle:
        """Candidates + calibration + branch state for one boundary."""
        cfg = self.config
        cap = cfg.correction_cap_ft
        trust = 1.0
        cal_ok = False
        cal_low_trust = True
        mb = None
        ambiguous = False

        ref = TypewellReference(task.tw_tvt, task.tw_gr)
        cal = None
        if self.level >= STAGE_LEVEL[STAGE_C]:
            cal = fit_prefix_affine_calibration(task, ref)
            cal_ok = bool(cal.ok)
            if cal_ok and np.isfinite(cal.fit_rmse_z) and cal.fit_rmse_z <= cfg.affine_fit_rmse_z_max:
                cal_low_trust = False
                trust = 1.0
            else:
                # Calibration failed or is low quality: keep the uncalibrated
                # candidate but shrink and cap it (never expose raw affine
                # values as unrestricted features).
                trust = cfg.low_trust_factor
                if not cal_ok:
                    cap = cfg.reduced_cap_ft
        if self.level >= STAGE_LEVEL[STAGE_D]:
            mb = multibranch_scan(task, cal=cal if (cal is not None and cal.ok) else None, ref=ref)

        cands = generate_candidate_corrections(
            task,
            base,
            pf=self.pf,
            beam=self.beam,
            mb=mb if (mb is not None and mb.ok) else None,
            correction_cap=cap,
        )

        if self.level >= STAGE_LEVEL[STAGE_D] and mb is not None and mb.ok:
            # Conservative hedged datum candidate: shift the anchor prediction
            # by the trust-shrunk branch mixture. w1 is already shrunk toward
            # 0.5 by the prefix-trust diagnostic inside multibranch_scan.
            hedged_shift = mb.w1 * mb.shift1 + (1.0 - mb.w1) * mb.shift2
            hedged = base + np.clip(hedged_shift, -cap, cap)
            cands["mb_hedged"] = CandidateCorrection(
                name="mb_hedged",
                prediction=hedged,
                confidence=float(np.clip(mb.confidence * max(mb.prefix_trust, 0.0), 0.0, 1.0)),
                disagreement=float(mb.sep) if mb.bimodal else 0.0,
                available=True,
                failure_reason="",
            )
            ambiguous = bool(
                mb.bimodal
                and mb.cost_gap < cfg.ambiguous_cost_gap
                and mb.sep > cfg.ambiguous_sep_ft
            )

        return _Bundle(
            base=base,
            cands=cands,
            cal_ok=cal_ok,
            cal_low_trust=cal_low_trust,
            mb=mb,
            cap=cap,
            trust=trust,
            ambiguous=ambiguous,
        )

    def _finalize_correction(
        self, task: InferenceTask, bundle: _Bundle, cand_name: str
    ) -> tuple[np.ndarray | None, bool, str]:
        """Trust scaling, ambiguity hedging and (stage E) robust projection.

        Returns ``(correction, projection_applied, projection_reason)`` or
        ``(None, ...)`` when the candidate is unusable.
        """
        cfg = self.config
        cand = bundle.cands.get(cand_name)
        if cand is None or not cand.available or cand.prediction is None:
            return None, False, "candidate_unavailable"
        corr = np.asarray(cand.prediction, dtype="float64") - bundle.base
        corr = np.where(np.isfinite(corr), corr, 0.0)
        if self.level >= STAGE_LEVEL[STAGE_C]:
            corr = corr * bundle.trust
        if self.level >= STAGE_LEVEL[STAGE_D] and bundle.ambiguous and bundle.mb is not None:
            # Bimodal, near-tied branches: hedge by halving the correction and
            # capping it at half the branch separation.
            hedge_cap = max(0.5 * float(bundle.mb.sep), 0.0)
            corr = np.clip(0.5 * corr, -hedge_cap, hedge_cap)
        corr = np.clip(corr, -cfg.correction_cap_ft, cfg.correction_cap_ft)

        proj_applied, proj_reason = False, ""
        if self.level >= STAGE_LEVEL[STAGE_E]:
            sl = slice(task.start, task.stop)
            anchor = task.anchor_tvt if np.isfinite(task.anchor_tvt) else 0.0
            track = bundle.base + corr
            proj_track, proj_applied, proj_reason = robust_projection(
                np.asarray(task.md[sl], dtype="float64"),
                np.asarray(task.z[sl], dtype="float64"),
                track,
                anchor,
                cfg,
            )
            if proj_applied:
                corr = np.clip(
                    proj_track - bundle.base,
                    -cfg.correction_cap_ft,
                    cfg.correction_cap_ft,
                )
        if not np.all(np.isfinite(corr)):
            return None, proj_applied, "nonfinite_correction"
        return corr, proj_applied, proj_reason

    def _evaluate_cut(
        self, nested: InferenceTask, truth: np.ndarray, cand_name: str
    ) -> tuple[float, bool] | None:
        """(rmse_delta, tail_ok) of the stage policy on one pseudo cut."""
        base_p = self.anchor_model.predict(nested)
        bundle_p = self._bundle(nested, np.asarray(base_p, dtype="float64"))
        corr_p, _, _ = self._finalize_correction(nested, bundle_p, cand_name)
        if corr_p is None:
            return None
        pred_p = bundle_p.base + corr_p * _ramp(corr_p.size, self.warmup)
        base_rmse = _finite_rmse(bundle_p.base, truth)
        cand_rmse = _finite_rmse(pred_p, truth)
        if not (np.isfinite(base_rmse) and np.isfinite(cand_rmse)):
            return None
        base_tail = _tail_mean_se(bundle_p.base, truth)
        cand_tail = _tail_mean_se(pred_p, truth)
        tail_ok = bool(
            not np.isfinite(base_tail)
            or (np.isfinite(cand_tail) and cand_tail <= base_tail + 1e-12)
        )
        return float(base_rmse - cand_rmse), tail_ok

    # ----------------------------------------------------------- decisions

    def _decide(self, task: InferenceTask) -> tuple[np.ndarray, np.ndarray | None, StageDecision]:
        """Full target-free decision for one boundary.

        Returns ``(base, correction_or_None, decision)``. The correction is
        pre-warmup; ``predict`` applies the ramp. ``None`` means exact Ridge.
        """
        t0 = time.perf_counter()
        cfg = self.config
        base = np.asarray(self.anchor_model.predict(task), dtype="float64")
        dec = StageDecision(
            well_id=task.well_id,
            stage=self.stage,
            protocol=self.protocol,
            fold=self.fold,
            outcome="fallback",
            conf_threshold=self.conf_thr,
            warmup_rows=self.warmup,
        )
        if self.stage == STAGE_A:
            dec.reason = "stage_a_is_ridge_default"
            dec.seconds = time.perf_counter() - t0
            return base, None, dec

        try:
            bundle = self._bundle(task, base)
        except Exception as exc:
            dec.reason = f"bundle_failed:{type(exc).__name__}"
            dec.seconds = time.perf_counter() - t0
            return base, None, dec
        dec.cal_ok = bundle.cal_ok if self.level >= STAGE_LEVEL[STAGE_C] else None
        dec.mb_ok = bool(bundle.mb.ok) if bundle.mb is not None else None
        dec.mb_bimodal = bool(bundle.mb.bimodal) if (bundle.mb is not None and bundle.mb.ok) else None
        dec.ambiguity_guard = bundle.ambiguous
        dec.cap_used = bundle.cap
        dec.trust_factor = bundle.trust

        available = [n for n, c in bundle.cands.items() if c.available and c.prediction is not None]
        if not available:
            dec.reason = "no_candidate_tracks"
            dec.seconds = time.perf_counter() - t0
            return base, None, dec

        # Primary selection cut: the mirrored nested pseudo-holdout.
        nested = nested_pseudo_task(task, min_predict=cfg.min_cut_predict)
        if nested is None:
            dec.reason = "pseudo_holdout_unavailable"
            dec.seconds = time.perf_counter() - t0
            return base, None, dec

        deltas: dict[str, tuple[float, bool]] = {}
        for name in available:
            try:
                got = self._evaluate_cut(nested.inputs, nested.truth, name)
            except Exception:
                got = None
            if got is not None:
                deltas[name] = got
        improved = {n: d for n, (d, _tail) in deltas.items() if np.isfinite(d) and d > 1e-12}
        if not improved:
            dec.reason = "pseudo_holdout_not_improved"
            dec.n_cuts_valid = 1 if deltas else 0
            dec.seconds = time.perf_counter() - t0
            return base, None, dec
        chosen = max(improved, key=improved.get)
        dec.candidate = chosen
        dec.pseudo_delta = improved[chosen]
        cand = bundle.cands[chosen]
        dec.confidence = float(cand.confidence)
        dec.disagreement = (
            float(cand.disagreement) if np.isfinite(cand.disagreement) else np.nan
        )

        corr, proj_applied, proj_reason = self._finalize_correction(task, bundle, chosen)
        dec.projection_applied = proj_applied
        dec.projection_reason = proj_reason
        if corr is None:
            dec.reason = f"finalize_failed:{proj_reason}"
            dec.seconds = time.perf_counter() - t0
            return base, None, dec

        # ----- stage F: multi-cut self-verification + tail/disagreement guard
        if self.level >= STAGE_LEVEL[STAGE_F]:
            if dec.confidence < self.conf_thr:
                dec.reason = "confidence_below_threshold"
                dec.seconds = time.perf_counter() - t0
                return base, None, dec
            if not np.isfinite(cand.disagreement) or cand.disagreement > cfg.disagreement_cap_ft:
                dec.reason = "disagreement_above_cap"
                dec.seconds = time.perf_counter() - t0
                return base, None, dec
            n_valid, n_improved, tail_failed = 0, 0, False
            # Primary nested cut counts as one verification cut.
            n_valid += 1
            primary_delta, primary_tail_ok = deltas[chosen]
            if primary_delta > 1e-12:
                n_improved += 1
            if not primary_tail_ok:
                tail_failed = True
            for frac in cfg.verify_cuts:
                cut_row = int(round(frac * task.start))
                built = pseudo_task_at(task, cut_row, min_predict=cfg.min_cut_predict)
                if built is None:
                    continue
                cut_task, cut_truth = built
                if cut_task.start == nested.inputs.start:
                    continue  # identical to the primary cut
                try:
                    got = self._evaluate_cut(cut_task, cut_truth, chosen)
                except Exception:
                    got = None
                if got is None:
                    continue
                d, tail_ok = got
                n_valid += 1
                if d > 1e-12:
                    n_improved += 1
                if not tail_ok:
                    tail_failed = True
            dec.n_cuts_valid = n_valid
            dec.n_cuts_improved = n_improved
            dec.tail_guard_failed = tail_failed
            if tail_failed:
                dec.reason = "worst_decile_tail_increased"
                dec.seconds = time.perf_counter() - t0
                return base, None, dec
            if n_valid < 2:
                # Self-verification needs corroboration: one cut alone is not
                # evidence enough to move away from Ridge Default.
                dec.reason = "insufficient_verification_cuts"
                dec.seconds = time.perf_counter() - t0
                return base, None, dec
            if n_improved < cfg.min_cuts_improved:
                dec.reason = "multi_cut_verification_failed"
                dec.seconds = time.perf_counter() - t0
                return base, None, dec

        dec.outcome = "applied"
        dec.reason = "all_guards_passed"
        dec.correction_mean_abs = float(np.mean(np.abs(corr)))
        dec.correction_max_abs = float(np.max(np.abs(corr)))
        dec.seconds = time.perf_counter() - t0
        return base, corr, dec

    # ---------------------------------------------------------- prediction

    def predict(self, task: InferenceTask, feats=None) -> np.ndarray:
        if self.anchor_model is None:
            return np.full(task.n_predict, self._anchor(task))
        if self.stage == STAGE_A:
            return self.anchor_model.predict(task, feats)
        try:
            base, corr, dec = self._decide(task)
        except Exception as exc:  # any failure => exact Ridge Default
            base = np.asarray(self.anchor_model.predict(task, feats), dtype="float64")
            dec = StageDecision(
                well_id=task.well_id,
                stage=self.stage,
                protocol=self.protocol,
                fold=self.fold,
                outcome="fallback",
                reason=f"decision_exception:{type(exc).__name__}",
                conf_threshold=self.conf_thr,
                warmup_rows=self.warmup,
            )
            corr = None
        self.decision_log.append(dec)
        self._last_diagnostics = {
            "gate_activation": dec.outcome == "applied",
            "gate_fallback_exact_ridge": dec.outcome != "applied",
            "gate_correction_magnitude": dec.correction_mean_abs,
            "gate_confidence_threshold": dec.conf_threshold,
            "gate_max_correction": dec.cap_used,
            "alignment_confidence_mean": dec.confidence,
            "alignment_ok": dec.outcome == "applied",
            "alignment_failure_reason": "" if dec.outcome == "applied" else dec.reason,
            "fallback_points": 0 if dec.outcome == "applied" else int(task.n_predict),
            "fallback_fraction": 0.0 if dec.outcome == "applied" else 1.0,
        }
        if corr is None:
            return base
        pred = base + corr * _ramp(corr.size, self.warmup)
        pred = np.where(np.isfinite(pred), pred, base)
        return pred

    def prediction_diagnostics(self, task, feats, pred) -> dict:
        return dict(self._last_diagnostics)


def _ramp(n: int, warmup: int) -> np.ndarray:
    """Warmup ramp 0->1 over ``warmup`` rows (bounded to half the suffix).

    Guarantees the first-row correction is at most cap/warmup_eff — never an
    abrupt jump at the boundary — while long suffixes still receive the full
    correction after the warmup region.
    """
    if n <= 0:
        return np.zeros(0)
    w_eff = int(min(max(warmup, 1), max(10, n // 2))) if n >= 20 else max(1, n // 2)
    return np.clip((np.arange(n, dtype="float64") + 1.0) / float(w_eff), 0.0, 1.0)


# --------------------------------------------------------------------------
# Factories used by the experiment runner
# --------------------------------------------------------------------------


def build_stage_models(
    stages: tuple[str, ...],
    *,
    memo: dict,
    protocol: str,
    fold: int,
    config: SafeAlignmentConfig | None = None,
    decision_log: list | None = None,
    device: str = "cpu",
    dataset_version: str = "rogii-mounted-v1",
    path_cache=None,
    tune: bool = True,
) -> dict[str, BaselineModel]:
    """Instantiate one model per requested stage, sharing PF/Beam memoization."""
    from src.beam_search import BeamSearchFeatureGenerator
    from src.particle_filter import ParticleFilterFeatureGenerator

    def _pf():
        return MemoizedPathGenerator(
            ParticleFilterFeatureGenerator(
                cache=path_cache,
                dataset_version=dataset_version,
                fold_id=fold,
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
                fold_id=fold,
                protocol=protocol,
                device=device,
            ),
            memo,
            "beam",
        )

    models: dict[str, BaselineModel] = {}
    shared_anchor = RidgeBaseline(alignment_features=False)
    for stage in stages:
        if stage == STAGE_A:
            # Stage A *is* the shared anchor instance: the fallback of every
            # later stage is bit-identical to the scored ridge_default arm.
            shared_anchor.name = STAGE_A
            models[stage] = shared_anchor
            continue
        models[stage] = SafeAlignmentModel(
            stage,
            pf=_pf(),
            beam=_beam(),
            config=config,
            protocol=protocol,
            fold=fold,
            decision_log=decision_log,
            tune=tune,
            anchor_model=shared_anchor,
        )
    return models
