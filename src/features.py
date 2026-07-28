"""Feature construction from an ``InferenceTask``.

Every function here takes an ``InferenceTask`` — an object that structurally
cannot expose the TVT label — so no feature can be target-derived. The output
column names are all registered in ``src.manifest`` and are re-checked with
``assert_safe_features`` before a model consumes them.

Design rules
------------
* GR is interpolated **within a single well only**. Per-well tool calibration
  differs, so a global fill would be meaningless.
* Prefix statistics read ``tvt_known`` strictly below ``task.start``.
* Everything is anchored: features describe movement *since the boundary*, so
  the model learns a residual and hold-last is the zero prediction.
* Alignment features come from normalized cross-correlation of the lateral GR
  against the typewell GR — a physical measurement that never reads TVT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.manifest import assert_safe_features
from src.tasks import InferenceTask

GR_SMOOTH_WINDOW = 51
DIP_WINDOW = 51


# --------------------------------------------------------------- utilities --

def interpolate_within_well(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation inside one well, edges held. Returns (filled, was_missing)."""
    v = np.asarray(values, dtype="float64")
    missing = ~np.isfinite(v)
    if missing.all():
        return np.zeros_like(v), missing
    idx = np.arange(v.size)
    good = ~missing
    filled = np.interp(idx, idx[good], v[good])
    return filled, missing


def _rolling(v: np.ndarray, window: int, fn: str) -> np.ndarray:
    if v.size == 0:
        return v
    s = pd.Series(v)
    r = s.rolling(window, center=True, min_periods=1)
    out = getattr(r, fn)()
    return out.to_numpy(dtype="float64")


def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Least-squares slope, NaN when underdetermined."""
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 2:
        return np.nan
    x, y = xs[m], ys[m]
    xd = x - x.mean()
    denom = float((xd * xd).sum())
    if denom <= 0:
        return np.nan
    return float((xd * (y - y.mean())).sum() / denom)


def _safe(v: float, default: float = 0.0) -> float:
    return float(v) if np.isfinite(v) else default


# ------------------------------------------------------------ prefix stats --

def prefix_stats(task: InferenceTask) -> dict:
    """Scalar summary of everything known before the boundary."""
    s = task.start
    md, tvt = task.md[:s], task.tvt_known[:s]
    known = np.isfinite(tvt)
    out = {
        "tvt_last": task.anchor_tvt,
        "prefix_len": float(s),
        "tvt_std_prefix": float(np.nanstd(tvt[known])) if known.sum() > 1 else 0.0,
        "tvt_range_prefix": (
            float(np.nanmax(tvt[known]) - np.nanmin(tvt[known])) if known.sum() > 1 else 0.0
        ),
    }
    anchor_md = md[task.anchor_row] if task.anchor_row >= 0 else (md[-1] if s else np.nan)
    for span in (100, 300, 1000):
        window = known & (md >= anchor_md - span)
        out[f"tvt_slope_{span}"] = _safe(_slope(md[window], tvt[window]))
    return out


# ------------------------------------------------------------ gr / typewell --

def gr_features(task: InferenceTask) -> dict:
    gr_filled, missing = interpolate_within_well(task.gr)
    s = task.start
    pre = gr_filled[:s]
    pre_valid = pre[np.isfinite(pre) & ~missing[:s]] if s else np.array([])
    mu = float(pre_valid.mean()) if pre_valid.size else float(np.mean(gr_filled))
    sd = float(pre_valid.std()) if pre_valid.size > 1 else 1.0
    sd = sd if sd > 1e-9 else 1.0
    return {
        "gr_filled": gr_filled,
        "gr_is_missing": missing.astype("float64"),
        "gr_roll_mean_51": _rolling(gr_filled, GR_SMOOTH_WINDOW, "mean"),
        "gr_roll_std_51": _rolling(gr_filled, GR_SMOOTH_WINDOW, "std"),
        "gr_z": (gr_filled - mu) / sd,
        "gr_missing_frac_well": float(missing.mean()),
        "_gr_mu": mu,
        "_gr_sd": sd,
    }


class TypewellReference:
    """Typewell GR resampled onto a uniform TVT grid (built once per well)."""

    def __init__(self, tvt: np.ndarray | None, gr: np.ndarray | None, step: float = 0.5):
        self.ok = False
        self.step = step
        if tvt is None or gr is None:
            return
        m = np.isfinite(tvt) & np.isfinite(gr)
        if m.sum() < 5:
            return
        t, g = tvt[m], gr[m]
        order = np.argsort(t)
        t, g = t[order], g[order]
        uniq, inv = np.unique(t, return_inverse=True)
        if uniq.size < 5:
            return
        g = np.bincount(inv, weights=g) / np.bincount(inv)
        self.tvt_min, self.tvt_max = float(uniq[0]), float(uniq[-1])
        n = int(np.floor((self.tvt_max - self.tvt_min) / step)) + 1
        if n < 5:
            return
        self.grid = self.tvt_min + step * np.arange(n)
        self.gr = np.interp(self.grid, uniq, g)
        self.mu = float(self.gr.mean())
        self.sd = float(self.gr.std()) or 1.0
        self.gr_z = (self.gr - self.mu) / self.sd
        self.dgr = np.gradient(self.gr, step)
        self.ok = True

    def gr_at(self, tvt: float) -> float:
        if not self.ok or not np.isfinite(tvt):
            return np.nan
        return float(np.interp(tvt, self.grid, self.gr))

    def dgr_at(self, tvt: float) -> float:
        if not self.ok or not np.isfinite(tvt):
            return np.nan
        return float(np.interp(tvt, self.grid, self.dgr))

    def stats(self) -> dict:
        if not self.ok:
            return {"tw_tvt_min": np.nan, "tw_tvt_max": np.nan, "tw_gr_std": np.nan}
        return {
            "tw_tvt_min": self.tvt_min,
            "tw_tvt_max": self.tvt_max,
            "tw_gr_std": self.sd,
        }


# ----------------------------------------------------------- geometry ------

def geometry_features(task: InferenceTask) -> dict:
    row = task.anchor_row if task.anchor_row >= 0 else max(task.start - 1, 0)
    md, x, y, z = task.md, task.x, task.y, task.z
    md0 = md[row]
    x0 = x[row] if np.isfinite(x[row]) else np.nan
    y0 = y[row] if np.isfinite(y[row]) else np.nan
    z0 = z[row] if np.isfinite(z[row]) else np.nan

    dmd = md - md0
    dx = x - x0
    dy = y - y0
    dz = z - z0
    lateral = np.sqrt(np.nan_to_num(dx) ** 2 + np.nan_to_num(dy) ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        dz_per_ft = np.where(np.abs(dmd) > 1e-9, dz / dmd, 0.0)

    z_filled, _ = interpolate_within_well(z)
    local_dz = np.gradient(_rolling(z_filled, DIP_WINDOW, "mean"), md)

    x_f, _ = interpolate_within_well(x)
    y_f, _ = interpolate_within_well(y)
    hx = np.gradient(_rolling(x_f, DIP_WINDOW, "mean"), md)
    hy = np.gradient(_rolling(y_f, DIP_WINDOW, "mean"), md)
    norm = np.hypot(hx, hy)
    norm = np.where(norm > 1e-12, norm, 1.0)

    return {
        "dmd": dmd,
        "log1p_dmd": np.log1p(np.clip(dmd, 0, None)),
        "dx": np.nan_to_num(dx),
        "dy": np.nan_to_num(dy),
        "dz": np.nan_to_num(dz),
        "lateral_disp": lateral,
        "dz_per_ft": np.nan_to_num(dz_per_ft),
        "local_dz_dmd": np.nan_to_num(local_dz),
        "heading_sin": np.nan_to_num(hy / norm),
        "heading_cos": np.nan_to_num(hx / norm),
    }


def typewell_gr_prefix_correlation(
    task: InferenceTask,
    ref: TypewellReference,
    gr_z: np.ndarray,
    *,
    gr_missing: np.ndarray,
) -> float:
    """Actual horizontal-GR/Typewell-GR Pearson r on visible TVT_input rows.

    This is an inference-time diagnostic, not a feature.  The visible prefix
    provides the Typewell TVT coordinate for each measured horizontal GR row;
    both logs are then compared after the prefix-only robust calibration.  It
    is intentionally separate from ``align_score`` (match discriminability)
    and from the hidden-label scorer.
    """
    if not ref.ok:
        return np.nan
    s = task.start
    signal = calibrate_gr_to_reference(task, ref, gr_z)
    known = np.isfinite(task.tvt_known[:s])
    expected = np.interp(task.tvt_known[:s], ref.grid, ref.gr_z)
    missing = np.asarray(gr_missing[:s], dtype=bool)
    m = known & np.isfinite(signal[:s]) & np.isfinite(expected) & ~missing
    if int(m.sum()) < 10:
        return np.nan
    a, b = signal[:s][m], expected[m]
    if float(np.std(a)) <= 1e-9 or float(np.std(b)) <= 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------- ncc alignment --

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two equal-length windows."""
    if a.size != b.size or a.size < 3:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.nan
    return float(a @ b / (na * nb))


def align_window(
    ref: TypewellReference,
    gr_win_z: np.ndarray,
    md_win: np.ndarray,
    centre_tvt: float,
    *,
    search: float = 12.0,
    gradients=(0.0,),
    continuity: float = 8.0,
    decimate: int = 4,
) -> tuple[float, float, float]:
    """Best (tvt, score, gradient) for one GR window against the typewell.

    A candidate is a straight TVT path through the window: a constant offset
    plus a gradient in TVT per foot of MD. The window's GR is compared with the
    reference GR sampled along that path.

    Scoring is deliberately **not** pure NCC. Correlation demeans both series,
    so a candidate path that is nearly flat in TVT — the common case in a
    horizontal well, where a 200 ft window may cross only a foot of section —
    produces a near-constant reference segment with no variance, and the search
    then prefers spurious steep gradients purely because they manufacture
    variation. The cost is therefore a level-matching MSE in shared z-space
    plus a continuity penalty on movement away from the seed; the shape
    correlation at the winning path is returned separately as the confidence.

    The candidate bank is evaluated by integer-grid gather rather than repeated
    interpolation, which is what makes a full-field run affordable.
    """
    if not ref.ok or gr_win_z.size < 5 or not np.isfinite(centre_tvt):
        return np.nan, np.nan, np.nan
    lo = max(ref.tvt_min, centre_tvt - search)
    hi = min(ref.tvt_max, centre_tvt + search)
    if hi <= lo:
        return np.nan, np.nan, np.nan

    d = max(1, int(decimate))
    obs = np.asarray(gr_win_z[::d], dtype="float64")
    dmd = np.asarray(md_win[::d], dtype="float64")
    dmd = dmd - dmd[-1]
    good = np.isfinite(obs)
    if good.sum() < 5:
        return np.nan, np.nan, np.nan
    obs, dmd = obs[good], dmd[good]

    step = ref.step
    n_grid = ref.gr_z.size
    off_idx = np.arange(
        int(round((lo - ref.tvt_min) / step)),
        int(round((hi - ref.tvt_min) / step)) + 1,
    )
    if off_idx.size == 0:
        return np.nan, np.nan, np.nan
    offsets = ref.tvt_min + off_idx * step

    best_cost, best = np.inf, (np.nan, np.nan, 0.0)
    all_mse: list[np.ndarray] = []
    for grad in gradients:
        # index of (grad * dmd) relative to the offset, on the reference grid
        rel = np.rint((grad * dmd) / step).astype(np.int64)
        idx = off_idx[:, None] + rel[None, :]          # (n_offsets, n_samples)
        inside = (idx >= 0) & (idx < n_grid)
        ok_rows = inside.all(axis=1)
        if not ok_rows.any():
            continue
        cand = ref.gr_z[np.clip(idx[ok_rows], 0, n_grid - 1)]
        mse = np.mean((cand - obs[None, :]) ** 2, axis=1)
        all_mse.append(mse)
        move = offsets[ok_rows] - centre_tvt
        cost = mse + (move / max(continuity, 1e-6)) ** 2
        j = int(np.argmin(cost))
        if cost[j] < best_cost:
            best_cost = float(cost[j])
            best = (float(offsets[ok_rows][j]), float(mse[j]), float(grad))

    if not np.isfinite(best[0]):
        return np.nan, np.nan, np.nan

    # Confidence = peak distinctiveness, not shape correlation.
    #
    # Correlation is the wrong confidence measure here: over a 200 ft window a
    # horizontal well may cross a foot of section, so both series are nearly
    # constant and the correlation of two flat lines is noise. What actually
    # distinguishes a trustworthy pick is whether the winning depth is clearly
    # better than the alternatives — i.e. whether a marker bed pins it. That is
    # the ratio of the best misfit to the typical misfit across the search
    # window, which is ~0 in a featureless interval and approaches 1 at a sharp,
    # unambiguous match.
    pool = np.concatenate(all_mse) if all_mse else np.array([np.nan])
    typical = float(np.median(pool))
    if not np.isfinite(typical) or typical <= 1e-9:
        score = 0.0
    else:
        score = float(np.clip(1.0 - best[1] / typical, 0.0, 1.0))
    return best[0], score, best[2]


def calibrate_gr_to_reference(
    task: InferenceTask, ref: TypewellReference, gr_z: np.ndarray
) -> np.ndarray:
    """Map the lateral GR into reference-GR space using the prefix only.

    The lateral and the typewell are different tools in different holes, so
    their GR values are not directly comparable. Where TVT is known (the
    prefix) the correct reference sample can be looked up for each observation,
    which determines a robust affine map ``a * gr + b``. This is the
    log-calibration a geosteerer performs by hand before correlating.

    Reads ``tvt_known`` strictly below ``task.start``; the hidden region is
    never consulted.
    """
    if not ref.ok:
        return gr_z
    s = task.start
    known = np.isfinite(task.tvt_known[:s]) & np.isfinite(gr_z[:s])
    if known.sum() < 50:
        return gr_z
    obs = gr_z[:s][known]
    want = np.interp(task.tvt_known[:s][known], ref.grid, ref.gr_z)
    o_med, w_med = float(np.median(obs)), float(np.median(want))
    o_scale = float(np.median(np.abs(obs - o_med)))
    w_scale = float(np.median(np.abs(want - w_med)))
    if o_scale <= 1e-6 or w_scale <= 1e-6:
        return gr_z
    a = float(np.clip(w_scale / o_scale, 0.2, 5.0))
    return (gr_z - o_med) * a + w_med


#: Version tag for the established NCC alignment track. Part of the cache key,
#: so changing the algorithm invalidates every stored artifact.
NCC_ALIGNMENT_VERSION = "ncc-gr-typewell-v1"
NCC_ALIGNMENT_WINDOW = 201
NCC_ALIGNMENT_STRIDE = 50
NCC_ALIGNMENT_SEARCH = 12.0


def cached_alignment_features(
    task: InferenceTask,
    ref: TypewellReference,
    gr_z: np.ndarray,
    *,
    gr_missing: np.ndarray,
    cache=None,
    cache_context: dict | None = None,
) -> dict:
    """Target-free persistent cache for the established NCC alignment track.

    Stores only derived tracks, scores, shifts and gradients. The cache API
    rejects target-like keys, and the task boundary is part of the key, so a
    real suffix and a same-well mask can never share an artifact.
    """
    if cache is None:
        return alignment_features(task, ref, gr_z, gr_missing=gr_missing)
    from src.cache import cache_key

    context = cache_context or {}
    key = cache_key(
        dataset_version=context.get("dataset_version", "rogii-mounted-v1"),
        well_id=task.well_id,
        fold_id=context.get("fold", "feature"),
        protocol=context.get("protocol", task.mode),
        feature_config={
            "name": NCC_ALIGNMENT_VERSION,
            "n_rows": task.n_rows,
            "start": task.start,
            "stop": task.stop,
        },
        alignment_config={
            "window": NCC_ALIGNMENT_WINDOW,
            "stride": NCC_ALIGNMENT_STRIDE,
            "search": NCC_ALIGNMENT_SEARCH,
        },
        device_profile={"feature_execution": "cpu"},
        code_version=NCC_ALIGNMENT_VERSION,
    )
    needed = ("align_tvt", "align_score", "align_shift", "align_gradient")
    hit = cache.get(key)
    if hit is not None and all(k in hit for k in needed) and len(hit["align_tvt"]) == task.n_rows:
        out = {k: np.asarray(hit[k], dtype="float64") for k in needed}
        out["_align_bias"] = float(np.asarray(hit["bias"]).ravel()[0]) if "bias" in hit else 0.0
        out["_align_ok"] = bool(np.asarray(hit["ok"]).ravel()[0]) if "ok" in hit else True
        out["_align_gr_correlation"] = (
            float(np.asarray(hit["gr_correlation"]).ravel()[0]) if "gr_correlation" in hit else np.nan
        )
        out["_align_cache_hit"] = True
        return out

    computed = alignment_features(task, ref, gr_z, gr_missing=gr_missing)
    try:
        cache.put(
            key,
            align_tvt=np.asarray(computed["align_tvt"], dtype="float64"),
            align_score=np.asarray(computed["align_score"], dtype="float64"),
            align_shift=np.asarray(computed["align_shift"], dtype="float64"),
            align_gradient=np.asarray(computed["align_gradient"], dtype="float64"),
            bias=np.asarray([computed["_align_bias"]], dtype="float64"),
            ok=np.asarray([bool(computed["_align_ok"])], dtype=bool),
            gr_correlation=np.asarray([computed["_align_gr_correlation"]], dtype="float64"),
        )
    except (ValueError, OSError):  # a cache write must never fail the run
        pass
    computed["_align_cache_hit"] = False
    return computed


def alignment_features(
    task: InferenceTask,
    ref: TypewellReference,
    gr_z: np.ndarray,
    *,
    gr_missing: np.ndarray | None = None,
    window: int = 201,
    stride: int = 50,
    search: float = 12.0,
    gradients=(-0.004, -0.002, -0.001, 0.0, 0.001, 0.002, 0.004),
) -> dict:
    """Windowed alignment over the whole well, calibrated on the known prefix.

    Two calibrations make this work on real logs:

    1. **Amplitude.** The lateral GR and the typewell GR are different tools in
       different holes. Before matching, the lateral log is mapped into
       reference-GR space by a robust affine fit ``a * gr + b`` computed on the
       *prefix only*, where TVT is known and the correct reference sample can
       therefore be looked up. This is the log-calibration step a geosteerer
       performs by hand.
    2. **Depth.** The median residual between the windowed solution and the
       known prefix TVT is removed as a constant bias.

    Both read the prefix exclusively, so neither can see the hidden region.
    """
    n = task.n_rows
    nan = np.full(n, np.nan)
    if not ref.ok or n < window:
        return {
            "align_tvt": nan.copy(),
            "align_score": np.zeros(n),
            "align_shift": np.zeros(n),
            "align_gradient": np.zeros(n),
            "_align_bias": 0.0,
            "_align_ok": False,
            "_align_gr_correlation": np.nan,
        }

    anchor = task.anchor_tvt
    s = task.start
    signal = calibrate_gr_to_reference(task, ref, gr_z)
    if gr_missing is None:
        gr_missing = np.zeros(n, dtype=bool)
    gr_missing = np.asarray(gr_missing, dtype=bool)

    centres = list(range(window // 2, n, stride))
    if centres[-1] != n - 1:
        centres.append(n - 1)

    raw_tvt = np.full(len(centres), np.nan)
    raw_score = np.zeros(len(centres))
    raw_grad = np.zeros(len(centres))

    # Walk forward so each window starts its search from the previous answer:
    # the trajectory is continuous, so this both speeds the search and stops it
    # jumping between repeated GR patterns.
    running = anchor
    for i, c in enumerate(centres):
        lo = max(0, c - window // 2)
        hi = min(n, lo + window)
        lo = max(0, hi - window)
        seed = running if np.isfinite(running) else anchor
        tvt, score, grad = align_window(
            ref, signal[lo:hi], task.md[lo:hi], seed, search=search, gradients=gradients
        )
        # A window made mostly of interpolated GR carries no real observation,
        # so its match is an artefact of the interpolation. Down-weight the
        # confidence in proportion to how much of the window was actually
        # measured; the pick itself is kept so the track stays continuous.
        measured = 1.0 - float(gr_missing[lo:hi].mean())
        conf = (score if np.isfinite(score) else 0.0) * measured
        raw_tvt[i] = tvt
        raw_score[i] = conf
        raw_grad[i] = grad if np.isfinite(grad) else 0.0
        if np.isfinite(tvt):
            running = tvt

    cent = np.asarray(centres, dtype="float64")
    good = np.isfinite(raw_tvt)
    if good.sum() < 2:
        return {
            "align_tvt": nan.copy(),
            "align_score": np.zeros(n),
            "align_shift": np.zeros(n),
            "align_gradient": np.zeros(n),
            "_align_bias": 0.0,
            "_align_ok": False,
            "_align_gr_correlation": np.nan,
        }

    rows = np.arange(n, dtype="float64")
    align_tvt = np.interp(rows, cent[good], raw_tvt[good])
    align_score = np.interp(rows, cent, raw_score)
    align_grad = np.interp(rows, cent, raw_grad)

    # Bias correction: compare against KNOWN TVT on the prefix only.
    known = np.isfinite(task.tvt_known[:s])
    bias = 0.0
    if known.sum() > 10:
        resid = task.tvt_known[:s][known] - align_tvt[:s][known]
        resid = resid[np.isfinite(resid)]
        if resid.size:
            bias = float(np.median(resid))
    align_tvt = align_tvt + bias

    # A report-only, actual GR-to-Typewell-GR Pearson correlation on the
    # predicted rows. It is distinct from `align_score`, whose documented role
    # is match discriminability. This uses only the hidden-region GR signal and
    # the alignment track, never its TVT label.
    sl = slice(task.start, task.stop)
    observed_gr = signal[sl]
    reference_gr = np.interp(align_tvt[sl], ref.grid, ref.gr_z)
    corr_mask = (
        np.isfinite(observed_gr)
        & np.isfinite(reference_gr)
        & ~gr_missing[sl]
    )
    gr_correlation = np.nan
    if int(corr_mask.sum()) >= 10:
        a, b = observed_gr[corr_mask], reference_gr[corr_mask]
        if float(np.std(a)) > 1e-9 and float(np.std(b)) > 1e-9:
            gr_correlation = float(np.corrcoef(a, b)[0, 1])

    return {
        "align_tvt": align_tvt,
        "align_score": align_score,
        "align_shift": align_tvt - (anchor if np.isfinite(anchor) else 0.0),
        "align_gradient": align_grad,
        "_align_bias": bias,
        "_align_ok": True,
        "_align_gr_correlation": gr_correlation,
    }


# -------------------------------------------------- dip-constrained GR alignment --
#
# This is intentionally a *separate* alignment path.  It is not added to
# FEATURE_COLUMNS, so the established Ridge baseline cannot accidentally use
# it.  The controlled experiment in baselines.py consumes it directly.

DIP_ALIGNMENT_VERSION = "dip-gr-typewell-v1"
DIP_ALIGNMENT_WINDOW = 201
DIP_ALIGNMENT_STRIDE = 50
DIP_ALIGNMENT_SEARCH = 12.0
DIP_ALIGNMENT_MIN_PREFIX_ROWS = 30


def _empty_dip_alignment(n: int, reason: str, *, fallback: np.ndarray | None = None) -> dict:
    """Return a diagnostic-rich, target-free failed alignment bundle."""
    track = np.full(n, np.nan)
    fallback = np.zeros(n, dtype="float64") if fallback is None else fallback
    return {
        "track": track,
        "confidence": np.zeros(n, dtype="float64"),
        "dip_prediction": np.asarray(fallback, dtype="float64"),
        "expected_gradient": np.zeros(n, dtype="float64"),
        "ok": False,
        "failure_reason": reason,
        "dip_r2": np.nan,
        "cache_hit": False,
    }


def dip_constrained_prediction(task: InferenceTask) -> dict:
    """Fit a locally planar stratigraphic surface from the *visible* prefix.

    The fitted quantity is ``TVT_input + Z``.  We regress it on centered X/Y
    coordinates (in 1,000-ft units) and then evaluate the plane along the
    planned trajectory.  Re-anchoring at the last visible TVT_input keeps this
    geometry-only fallback continuous at Prediction Start.

    No ``TVT`` label or typewell geology is read here.  The function receives
    an ``InferenceTask``, which structurally has no target field.
    """
    n, s = task.n_rows, task.start
    anchor = task.anchor_tvt
    anchor = anchor if np.isfinite(anchor) else 0.0
    default = np.full(n, anchor, dtype="float64")
    if s < DIP_ALIGNMENT_MIN_PREFIX_ROWS or task.anchor_row < 0:
        return {
            "ok": False, "prediction": default, "gradient": np.zeros(n),
            "r2": np.nan, "reason": "insufficient_visible_prefix",
        }

    known = np.isfinite(task.tvt_known[:s])
    m = known & np.isfinite(task.x[:s]) & np.isfinite(task.y[:s]) & np.isfinite(task.z[:s])
    if int(m.sum()) < DIP_ALIGNMENT_MIN_PREFIX_ROWS:
        return {
            "ok": False, "prediction": default, "gradient": np.zeros(n),
            "r2": np.nan, "reason": "insufficient_visible_dip_support",
        }

    # Scaling avoids a poorly conditioned solve at field-coordinate magnitudes.
    x0 = float(task.x[task.anchor_row])
    y0 = float(task.y[task.anchor_row])
    z0 = float(task.z[task.anchor_row])
    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(z0)):
        return {
            "ok": False, "prediction": default, "gradient": np.zeros(n),
            "r2": np.nan, "reason": "invalid_anchor_geometry",
        }
    px = (task.x[:s][m] - x0) / 1000.0
    py = (task.y[:s][m] - y0) / 1000.0
    surface = task.tvt_known[:s][m] + task.z[:s][m]
    A = np.column_stack([px, py, np.ones_like(px)])
    # A tiny ridge makes an along-track trajectory identifiable even when X/Y
    # are nearly collinear.  It shrinks the unidentifiable cross-track dip,
    # rather than inventing a large one.
    gram = A.T @ A
    penalty = np.diag([1e-4, 1e-4, 0.0])
    try:
        coef = np.linalg.solve(gram + penalty, A.T @ surface)
    except np.linalg.LinAlgError:
        return {
            "ok": False, "prediction": default, "gradient": np.zeros(n),
            "r2": np.nan, "reason": "dip_plane_fit_failed",
        }
    fitted = A @ coef
    denom = float(np.sum((surface - np.mean(surface)) ** 2))
    r2 = float(1.0 - np.sum((surface - fitted) ** 2) / denom) if denom > 1e-12 else 0.0

    x_f, _ = interpolate_within_well(task.x)
    y_f, _ = interpolate_within_well(task.y)
    z_f, _ = interpolate_within_well(task.z)
    surface_delta = coef[0] * (x_f - x0) / 1000.0 + coef[1] * (y_f - y0) / 1000.0
    pred = anchor + surface_delta - (z_f - z0)
    md_f, _ = interpolate_within_well(task.md)
    try:
        gradient = np.gradient(pred, md_f)
    except (ValueError, np.linalg.LinAlgError):  # malformed MD is a fallback case
        gradient = np.zeros(n, dtype="float64")
    gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    return {"ok": True, "prediction": pred, "gradient": gradient, "r2": r2, "reason": ""}


def dip_constrained_alignment_features(
    task: InferenceTask,
    ref: TypewellReference,
    gr_z: np.ndarray,
    *,
    gr_missing: np.ndarray | None = None,
    window: int = DIP_ALIGNMENT_WINDOW,
    stride: int = DIP_ALIGNMENT_STRIDE,
    search: float = DIP_ALIGNMENT_SEARCH,
) -> dict:
    """Align horizontal GR to Typewell GR under a prefix-derived dip prior.

    Candidate TVT gradients are centered on the apparent dip implied by the
    visible ``TVT_input``/X/Y/Z trajectory.  The matching signal is strictly
    horizontal GR versus Typewell GR.  The known prefix is used only for the
    same amplitude and constant-bias calibrations used by the existing NCC
    feature; it never exposes a hidden label.
    """
    n = task.n_rows
    dip = dip_constrained_prediction(task)
    fallback = dip["prediction"]
    if not dip["ok"]:
        return _empty_dip_alignment(n, dip["reason"], fallback=fallback)
    if not ref.ok:
        return _empty_dip_alignment(n, "missing_or_invalid_typewell", fallback=fallback)
    if n < window:
        return _empty_dip_alignment(n, "well_shorter_than_alignment_window", fallback=fallback)
    if gr_missing is None:
        gr_missing = np.zeros(n, dtype=bool)
    gr_missing = np.asarray(gr_missing, dtype=bool)
    if bool(gr_missing.all()):
        return _empty_dip_alignment(n, "horizontal_gr_all_missing", fallback=fallback)

    signal = calibrate_gr_to_reference(task, ref, gr_z)
    half = window // 2
    centres = list(range(half, n, max(1, stride)))
    if not centres:
        return _empty_dip_alignment(n, "no_alignment_windows", fallback=fallback)
    if centres[-1] != n - 1:
        centres.append(n - 1)

    raw_track = np.full(len(centres), np.nan)
    raw_score = np.zeros(len(centres), dtype="float64")
    raw_grad = np.zeros(len(centres), dtype="float64")
    for i, c in enumerate(centres):
        lo = max(0, c - half)
        hi = min(n, lo + window)
        lo = max(0, hi - window)
        expected = float(dip["gradient"][c])
        # The offsets are deliberately narrow.  The GR match can correct a
        # local plane error, but cannot choose a geologically impossible slope.
        gradients = tuple(np.clip(expected + np.asarray((-0.004, -0.002, -0.001, 0.0, 0.001, 0.002, 0.004)), -0.04, 0.04))
        tvt, score, grad = align_window(
            ref,
            signal[lo:hi],
            task.md[lo:hi],
            float(fallback[c]),
            search=search,
            gradients=gradients,
        )
        measured = 1.0 - float(gr_missing[lo:hi].mean())
        raw_track[i] = tvt
        raw_score[i] = (score if np.isfinite(score) else 0.0) * measured
        raw_grad[i] = grad if np.isfinite(grad) else expected

    good = np.isfinite(raw_track)
    if int(good.sum()) < 2:
        return _empty_dip_alignment(n, "no_valid_gr_typewell_match", fallback=fallback)
    rows = np.arange(n, dtype="float64")
    cent = np.asarray(centres, dtype="float64")
    track = np.interp(rows, cent[good], raw_track[good])
    confidence = np.interp(rows, cent, raw_score)
    gradient = np.interp(rows, cent, raw_grad)

    # Calibrate only a constant level offset against visible TVT_input.  The
    # movement in the hidden region stays entirely GR/dip-derived.
    known = np.isfinite(task.tvt_known[: task.start])
    if int(known.sum()) > 10:
        residual = task.tvt_known[: task.start][known] - track[: task.start][known]
        residual = residual[np.isfinite(residual)]
        if residual.size:
            track = track + float(np.median(residual))

    return {
        "track": track,
        "confidence": np.clip(confidence, 0.0, 1.0),
        "dip_prediction": fallback,
        "expected_gradient": dip["gradient"],
        "ok": True,
        "failure_reason": "",
        "dip_r2": float(dip["r2"]),
        "cache_hit": False,
    }


def cached_dip_constrained_alignment_features(
    task: InferenceTask,
    ref: TypewellReference,
    gr_z: np.ndarray,
    *,
    gr_missing: np.ndarray,
    cache=None,
    cache_context: dict | None = None,
) -> dict:
    """Target-free persistent cache for the expensive dip/GR alignment.

    Only derived tracks, confidences, dip fallbacks and diagnostic codes are
    persisted.  The cache API rejects target-like keys, and this function never
    receives a ``WellTask.target``.  The task boundary is part of the key, so a
    real suffix and a same-well mask cannot share an alignment artifact.
    """
    if cache is None:
        return dip_constrained_alignment_features(task, ref, gr_z, gr_missing=gr_missing)
    from src.cache import cache_key

    context = cache_context or {}
    key = cache_key(
        dataset_version=context.get("dataset_version", "rogii-mounted-v1"),
        well_id=task.well_id,
        fold_id=context.get("fold", "feature"),
        protocol=context.get("protocol", task.mode),
        feature_config={
            "name": DIP_ALIGNMENT_VERSION,
            "n_rows": task.n_rows,
            "start": task.start,
            "stop": task.stop,
        },
        alignment_config={
            "window": DIP_ALIGNMENT_WINDOW,
            "stride": DIP_ALIGNMENT_STRIDE,
            "search": DIP_ALIGNMENT_SEARCH,
        },
        device_profile={"feature_execution": "cpu"},
        code_version=DIP_ALIGNMENT_VERSION,
    )
    hit = cache.get(key)
    needed = {"track", "confidence", "dip_prediction", "expected_gradient", "ok", "reason", "dip_r2"}
    if hit is not None and needed.issubset(hit) and len(hit["track"]) == task.n_rows:
        return {
            "track": np.asarray(hit["track"], dtype="float64"),
            "confidence": np.asarray(hit["confidence"], dtype="float64"),
            "dip_prediction": np.asarray(hit["dip_prediction"], dtype="float64"),
            "expected_gradient": np.asarray(hit["expected_gradient"], dtype="float64"),
            "ok": bool(np.asarray(hit["ok"]).ravel()[0]),
            "failure_reason": str(np.asarray(hit["reason"]).ravel()[0]),
            "dip_r2": float(np.asarray(hit["dip_r2"]).ravel()[0]),
            "cache_hit": True,
        }
    out = dip_constrained_alignment_features(task, ref, gr_z, gr_missing=gr_missing)
    cache.put(
        key,
        track=np.asarray(out["track"], dtype="float64"),
        confidence=np.asarray(out["confidence"], dtype="float64"),
        dip_prediction=np.asarray(out["dip_prediction"], dtype="float64"),
        expected_gradient=np.asarray(out["expected_gradient"], dtype="float64"),
        ok=np.asarray([int(bool(out["ok"]))], dtype="int8"),
        reason=np.asarray([str(out["failure_reason"])]),
        dip_r2=np.asarray([out["dip_r2"]], dtype="float64"),
    )
    return out

# ------------------------------------------------------------ assembly -----

ROW_FEATURES = [
    "dmd",
    "log1p_dmd",
    "dx",
    "dy",
    "dz",
    "lateral_disp",
    "dz_per_ft",
    "local_dz_dmd",
    "heading_sin",
    "heading_cos",
    "gr_filled",
    "gr_is_missing",
    "gr_roll_mean_51",
    "gr_roll_std_51",
    "gr_z",
    "align_tvt",
    "align_score",
    "align_shift",
    "align_gradient",
]

#: The four established GR/typewell alignment columns.  The real 770-well
#: ablation removed them from the default Ridge matrix, but they remain in the
#: complete feature catalogue for explicit diagnostics and opt-in experiments.
ALIGNMENT_FEATURES = [
    "align_tvt",
    "align_score",
    "align_shift",
    "align_gradient",
]

SCALAR_FEATURES = [
    "tvt_last",
    "tvt_slope_100",
    "tvt_slope_300",
    "tvt_slope_1000",
    "tvt_std_prefix",
    "tvt_range_prefix",
    "prefix_len",
    "gr_missing_frac_well",
    "tw_gr_at_tvt_last",
    "tw_dgr_dtvt_at_tvt_last",
    "tw_tvt_min",
    "tw_tvt_max",
    "tw_gr_std",
]

FEATURE_COLUMNS = ROW_FEATURES + SCALAR_FEATURES


def feature_columns(*, alignment_features: bool = True) -> list[str]:
    """Return the complete or no-alignment base feature catalogue.

    ``FEATURE_COLUMNS`` intentionally still contains the implemented alignment
    capability.  :class:`RidgeBaseline` now explicitly requests ``False`` by
    default after the real 770-well decision; diagnostic branches request
    ``True`` without restoring alignment to the default.
    """
    if alignment_features:
        return list(FEATURE_COLUMNS)
    drop = set(ALIGNMENT_FEATURES)
    return [c for c in FEATURE_COLUMNS if c not in drop]


class WellFeatures:
    """Per-well feature bundle, computed once and reused by every model.

    ``dip_alignment`` is opt-in and remains outside ``FEATURE_COLUMNS``.  It
    exists for the isolated, REJECTED dip-constrained diagnostic model only.
    """

    __slots__ = (
        "task", "ref", "gr", "geom", "prefix", "align", "dip_align",
        "typewell_gr_prefix_correlation", "_frame",
    )

    def __init__(
        self,
        task: InferenceTask,
        *,
        alignment: bool = True,
        dip_alignment: bool = False,
        alignment_cache=None,
        cache_context: dict | None = None,
    ):
        self.task = task
        self.ref = TypewellReference(task.tw_tvt, task.tw_gr)
        self.gr = gr_features(task)
        self.typewell_gr_prefix_correlation = typewell_gr_prefix_correlation(
            task,
            self.ref,
            self.gr["gr_z"],
            gr_missing=self.gr["gr_is_missing"] > 0.5,
        )
        self.geom = geometry_features(task)
        self.prefix = prefix_stats(task)
        if alignment:
            self.align = cached_alignment_features(
                task,
                self.ref,
                self.gr["gr_z"],
                gr_missing=self.gr["gr_is_missing"] > 0.5,
                cache=alignment_cache,
                cache_context=cache_context,
            )
        else:
            n = task.n_rows
            self.align = {
                "align_tvt": np.full(n, np.nan),
                "align_score": np.zeros(n),
                "align_shift": np.zeros(n),
                "align_gradient": np.zeros(n),
                "_align_bias": 0.0,
                "_align_ok": False,
                "_align_gr_correlation": np.nan,
            }
        if dip_alignment:
            self.dip_align = cached_dip_constrained_alignment_features(
                task,
                self.ref,
                self.gr["gr_z"],
                gr_missing=self.gr["gr_is_missing"] > 0.5,
                cache=alignment_cache,
                cache_context=cache_context,
            )
        else:
            self.dip_align = _empty_dip_alignment(task.n_rows, "not_requested")
        self._frame: pd.DataFrame | None = None

    @property
    def scalars(self) -> dict:
        anchor = self.task.anchor_tvt
        out = dict(self.prefix)
        out["gr_missing_frac_well"] = self.gr["gr_missing_frac_well"]
        out.update(self.ref.stats())
        out["tw_gr_at_tvt_last"] = self.ref.gr_at(anchor)
        out["tw_dgr_dtvt_at_tvt_last"] = self.ref.dgr_at(anchor)
        return out

    def frame(self, rows: slice | np.ndarray | None = None) -> pd.DataFrame:
        """Feature matrix for the requested rows (default: prediction region)."""
        if self._frame is None:
            data = {}
            for name in ROW_FEATURES:
                src = self.geom if name in self.geom else (
                    self.gr if name in self.gr else self.align
                )
                data[name] = np.asarray(src[name], dtype="float64")
            n = self.task.n_rows
            for name, value in self.scalars.items():
                data[name] = np.full(n, _safe(value, np.nan))
            frame = pd.DataFrame(data)
            # align_tvt is NaN when there is no usable typewell; fall back to
            # the anchor so downstream models see a finite, honest value and
            # align_score (= 0) tells them not to trust it.
            anchor = self.task.anchor_tvt
            frame["align_tvt"] = frame["align_tvt"].fillna(anchor if np.isfinite(anchor) else 0.0)
            frame["align_shift"] = frame["align_shift"].fillna(0.0)
            frame = frame.replace([np.inf, -np.inf], np.nan)
            self._frame = frame[FEATURE_COLUMNS]
        if rows is None:
            rows = slice(self.task.start, self.task.stop)
        return self._frame.iloc[rows]


def build_features(
    task: InferenceTask,
    *,
    alignment: bool = True,
    dip_alignment: bool = False,
    alignment_cache=None,
    cache_context: dict | None = None,
) -> WellFeatures:
    """Build target-free features for one inference task.

    The optional persistent cache applies solely to the dip-constrained
    alignment track.  It stores no labels and keys on the simulated boundary.
    """
    return WellFeatures(
        task,
        alignment=alignment,
        dip_alignment=dip_alignment,
        alignment_cache=alignment_cache,
        cache_context=cache_context,
    )


def validate_feature_frame(frame: pd.DataFrame) -> None:
    """Manifest check: refuse anything not cleared for inference."""
    assert_safe_features(frame.columns, context="feature matrix")
