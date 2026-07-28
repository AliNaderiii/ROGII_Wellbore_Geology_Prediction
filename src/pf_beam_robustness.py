"""Paired-error and robustness analysis for Particle Filter + Beam Ridge.

Consumes the cross-fitted per-well table emitted by
``scripts/run_validation.py --particle-filter --beam-search``
(``particle_beam_wells.csv`` / ``well_level_validation.csv``).

Nothing is estimated from aggregate RMSEs alone.  When the well-level table is
absent, callers must treat every well-/fold-/bootstrap-level quantity as
unavailable rather than inventing a number that looks like a completed
analysis.

Protocols are reported separately and are never averaged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.validation import (
    GR_BINS,
    GR_LABELS,
    PREFIX_BINS,
    PREFIX_LABELS,
    PROTOCOL_A,
    PROTOCOL_B,
    SUFFIX_BINS,
    SUFFIX_LABELS,
)

REAL_PROTOCOLS = (PROTOCOL_A, PROTOCOL_B)

DEFAULT_MODEL = "ridge_default"
CANDIDATE_MODEL = "ridge_particle_beam"
ALL_BRANCHES = (
    "ridge_default",
    "ridge_particle_filter",
    "ridge_beam_search",
    "ridge_particle_beam",
)

#: Owner-supplied completed-run global point-level RMSE (770 eligible wells,
#: zero task/fit/predict failures).  Used only for the authored decision
#: narrative when the per-well artifact is not mounted in this checkout.
OWNER_GLOBAL_RMSE: dict[str, dict[str, float]] = {
    PROTOCOL_B: {
        "ridge_default": 14.423,
        "ridge_particle_filter": 14.429,
        "ridge_beam_search": 14.432,
        "ridge_particle_beam": 14.419,
    },
    PROTOCOL_A: {
        "ridge_default": 29.486,
        "ridge_particle_filter": 29.406,
        "ridge_beam_search": 29.406,
        "ridge_particle_beam": 29.388,
    },
}
OWNER_N_WELLS = 770
OWNER_N_FAILURES = 0
OWNER_SOURCE = (
    "real 770-well PF/Beam validation, both protocols, cross-fitted by well ID, "
    "zero task/fit/predict failures (run-owner aggregate)"
)

REAL_BANNER = "REAL KAGGLE VALIDATION"
SYNTHETIC_BANNER = "SYNTHETIC — NOT A COMPETITION RESULT"
PUBLIC_LB_BANNER = "PUBLIC LEADERBOARD"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0
CONCENTRATION_TOP_K = 10
# Share of total SSE improvement attributable to the best-K wells above which
# the gain is treated as concentrated rather than broad.
CONCENTRATION_SSE_SHARE = 0.50


def _md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df is None or df.empty:
        return "_No computed rows were available._\n"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "" if pd.isna(v) else f"{float(v):.{digits}f}"
            )
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda v: "yes" if v else "no")
        else:
            out[col] = out[col].astype(str)
    header = "| " + " | ".join(out.columns) + " |"
    rule = "|" + "|".join("---" for _ in out.columns) + "|"
    body = ["| " + " | ".join(row) + " |" for row in out.to_numpy()]
    return "\n".join([header, rule, *body]) + "\n"


def _weighted_rmse(group: pd.DataFrame) -> float:
    n = float(group["n_points"].sum())
    return float(np.sqrt(group["sse"].sum() / n)) if n else np.nan


def _bool_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    column = frame[name]
    if pd.api.types.is_bool_dtype(column):
        return column.fillna(False).astype(bool)
    truthy = {"true", "1", "yes", "t"}
    return column.map(
        lambda value: (
            False
            if value is None or (not isinstance(value, str) and pd.isna(value))
            else (
                str(value).strip().lower() in truthy
                if isinstance(value, str)
                else bool(value)
            )
        )
    ).astype(bool)


def load_well_table(path: str | Path) -> pd.DataFrame:
    """Load a particle-beam / well-level validation CSV and normalise dtypes."""
    frame = pd.read_csv(path)
    required = {
        "model", "protocol", "well_id", "n_points", "sse", "rmse",
        "prefix_len", "suffix_len", "gr_missing_frac",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = frame.copy()
    out["model"] = out["model"].astype(str)
    out["protocol"] = out["protocol"].astype(str)
    out["well_id"] = out["well_id"].astype(str)
    if "fold" not in out.columns:
        out["fold"] = -1
    for col in ("n_points", "sse", "rmse", "prefix_len", "suffix_len", "gr_missing_frac"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def pair_default_vs_candidate(
    well: pd.DataFrame,
    *,
    default: str = DEFAULT_MODEL,
    candidate: str = CANDIDATE_MODEL,
) -> pd.DataFrame:
    """One row per (protocol, well) with RMSE/SSE deltas (candidate − default).

    Negative ``delta_rmse`` means the candidate is better on that well.
    Only wells scored by both models within a protocol are retained.
    """
    if well is None or well.empty:
        return pd.DataFrame()
    frame = well[well["protocol"].isin(REAL_PROTOCOLS)].copy()
    frame = frame[frame["model"].isin({default, candidate})]
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for protocol, group in frame.groupby("protocol", sort=False):
        wide = group.pivot_table(
            index="well_id",
            columns="model",
            values=["rmse", "sse", "n_points", "fold", "prefix_len", "suffix_len", "gr_missing_frac"],
            aggfunc="first",
        )
        if not ({default, candidate} <= set(group["model"])):
            continue
        # MultiIndex columns: (metric, model)
        have_default = wide[("rmse", default)].notna()
        have_candidate = wide[("rmse", candidate)].notna()
        keep = have_default & have_candidate
        wide = wide.loc[keep]
        if wide.empty:
            continue
        for well_id, row in wide.iterrows():
            n_def = float(row[("n_points", default)])
            n_cand = float(row[("n_points", candidate)])
            # Point counts must match for a fair paired comparison.
            n_points = int(n_def) if n_def == n_cand else int(min(n_def, n_cand))
            d_rmse = float(row[("rmse", candidate)] - row[("rmse", default)])
            d_sse = float(row[("sse", candidate)] - row[("sse", default)])
            rows.append(
                {
                    "protocol": protocol,
                    "well_id": str(well_id),
                    "fold": int(row[("fold", default)])
                    if pd.notna(row[("fold", default)])
                    else int(row[("fold", candidate)])
                    if pd.notna(row[("fold", candidate)])
                    else -1,
                    "n_points": n_points,
                    "prefix_len": int(row[("prefix_len", default)])
                    if pd.notna(row[("prefix_len", default)])
                    else np.nan,
                    "suffix_len": int(row[("suffix_len", default)])
                    if pd.notna(row[("suffix_len", default)])
                    else np.nan,
                    "gr_missing_frac": float(row[("gr_missing_frac", default)])
                    if pd.notna(row[("gr_missing_frac", default)])
                    else np.nan,
                    "rmse_default": float(row[("rmse", default)]),
                    "rmse_candidate": float(row[("rmse", candidate)]),
                    "sse_default": float(row[("sse", default)]),
                    "sse_candidate": float(row[("sse", candidate)]),
                    "delta_rmse": d_rmse,
                    "delta_sse": d_sse,
                    "improved": bool(d_rmse < 0.0),
                    "degraded": bool(d_rmse > 0.0),
                    "unchanged": bool(d_rmse == 0.0),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # Attach PF/Beam diagnostics from the candidate branch when present.
    cand = well[well["model"] == candidate].copy()
    diag_cols = [
        c
        for c in cand.columns
        if c.startswith("particle_") or c.startswith("beam_")
    ]
    if diag_cols:
        keep = ["protocol", "well_id", *diag_cols]
        meta = cand[keep].drop_duplicates(["protocol", "well_id"])
        out = out.merge(meta, on=["protocol", "well_id"], how="left")
    return out.sort_values(["protocol", "delta_rmse"]).reset_index(drop=True)


def well_level_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Improved / degraded counts, mean/median delta, worst-10 delta per protocol."""
    if paired is None or paired.empty:
        return pd.DataFrame()
    rows = []
    for protocol, g in paired.groupby("protocol", sort=False):
        n = int(len(g))
        improved = int(g["improved"].sum())
        degraded = int(g["degraded"].sum())
        unchanged = int(g["unchanged"].sum())
        worst10 = g.nlargest(min(10, n), "delta_rmse")
        best10 = g.nsmallest(min(10, n), "delta_rmse")
        # Concentration: share of total SSE reduction coming from the K wells
        # with the largest SSE reductions (most negative delta_sse).
        sse_gain = g["delta_sse"].clip(upper=0.0)  # only improvements
        total_gain = float((-sse_gain).sum())
        top_k = g.nsmallest(min(CONCENTRATION_TOP_K, n), "delta_sse")
        top_k_gain = float((-top_k["delta_sse"].clip(upper=0.0)).sum())
        share = top_k_gain / total_gain if total_gain > 0 else np.nan
        rows.append(
            {
                "protocol": protocol,
                "n_wells": n,
                "n_improved": improved,
                "pct_improved": 100.0 * improved / n if n else np.nan,
                "n_degraded": degraded,
                "pct_degraded": 100.0 * degraded / n if n else np.nan,
                "n_unchanged": unchanged,
                "pct_unchanged": 100.0 * unchanged / n if n else np.nan,
                "mean_well_delta_rmse": float(g["delta_rmse"].mean()),
                "median_well_delta_rmse": float(g["delta_rmse"].median()),
                "p10_well_delta_rmse": float(g["delta_rmse"].quantile(0.10)),
                "p90_well_delta_rmse": float(g["delta_rmse"].quantile(0.90)),
                "worst10_mean_delta_rmse": float(worst10["delta_rmse"].mean()),
                "best10_mean_delta_rmse": float(best10["delta_rmse"].mean()),
                "global_point_rmse_default": float(
                    np.sqrt(g["sse_default"].sum() / g["n_points"].sum())
                )
                if g["n_points"].sum()
                else np.nan,
                "global_point_rmse_candidate": float(
                    np.sqrt(g["sse_candidate"].sum() / g["n_points"].sum())
                )
                if g["n_points"].sum()
                else np.nan,
                "global_point_delta_rmse": (
                    float(np.sqrt(g["sse_candidate"].sum() / g["n_points"].sum()))
                    - float(np.sqrt(g["sse_default"].sum() / g["n_points"].sum()))
                )
                if g["n_points"].sum()
                else np.nan,
                "top10_sse_improvement_share": share,
                "improvement_concentrated": bool(
                    np.isfinite(share) and share >= CONCENTRATION_SSE_SHARE
                ),
            }
        )
    return pd.DataFrame(rows)


def fold_deltas(
    well: pd.DataFrame,
    *,
    default: str = DEFAULT_MODEL,
    candidate: str = CANDIDATE_MODEL,
) -> pd.DataFrame:
    """Per-fold global RMSE for default and candidate, with deltas."""
    if well is None or well.empty or "fold" not in well.columns:
        return pd.DataFrame()
    frame = well[
        well["protocol"].isin(REAL_PROTOCOLS)
        & well["model"].isin({default, candidate})
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (protocol, fold), group in frame.groupby(["protocol", "fold"], sort=False):
        d = group[group["model"] == default]
        c = group[group["model"] == candidate]
        # Pair on common wells inside the fold.
        common = set(d["well_id"]) & set(c["well_id"])
        if not common:
            continue
        d = d[d["well_id"].isin(common)]
        c = c[c["well_id"].isin(common)]
        rd = _weighted_rmse(d)
        rc = _weighted_rmse(c)
        rows.append(
            {
                "protocol": protocol,
                "fold": int(fold),
                "n_wells": int(len(common)),
                "n_points": int(d["n_points"].sum()),
                "rmse_default": rd,
                "rmse_candidate": rc,
                "delta_rmse": rc - rd,
                "candidate_better": bool(rc < rd),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["protocol", "fold"]).reset_index(drop=True)
    # Fold-stability flag: candidate better (or equal) on every fold of a protocol.
    flags = []
    for protocol, g in out.groupby("protocol", sort=False):
        n_folds = int(g["fold"].nunique())
        n_better = int(g["candidate_better"].sum())
        n_not_worse = int((g["delta_rmse"] <= 0.0).sum())
        flags.append(
            {
                "protocol": protocol,
                "n_folds": n_folds,
                "n_folds_candidate_better": n_better,
                "n_folds_candidate_not_worse": n_not_worse,
                "stable_across_folds": bool(n_folds > 0 and n_not_worse == n_folds),
                "mean_fold_delta_rmse": float(g["delta_rmse"].mean()),
            }
        )
    stability = pd.DataFrame(flags)
    out = out.merge(stability, on="protocol", how="left")
    return out


def bootstrap_global_rmse_delta(
    paired: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Point-level global RMSE delta CI by resampling wells with replacement.

    Each bootstrap draw resamples wells (keeping their n_points and SSE), then
    recomputes weighted global RMSE for default and candidate.  This is a
    well-cluster bootstrap of the *point-level* metric, not an i.i.d. point
    bootstrap (points inside a well are dependent).
    """
    if paired is None or paired.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for protocol, g in paired.groupby("protocol", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        sse_d = g["sse_default"].to_numpy(dtype="float64")
        sse_c = g["sse_candidate"].to_numpy(dtype="float64")
        pts = g["n_points"].to_numpy(dtype="float64")
        obs_d = float(np.sqrt(sse_d.sum() / pts.sum())) if pts.sum() else np.nan
        obs_c = float(np.sqrt(sse_c.sum() / pts.sum())) if pts.sum() else np.nan
        obs_delta = obs_c - obs_d
        deltas = np.empty(n_boot, dtype="float64")
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            p = pts[idx].sum()
            if p <= 0:
                deltas[b] = np.nan
                continue
            deltas[b] = np.sqrt(sse_c[idx].sum() / p) - np.sqrt(sse_d[idx].sum() / p)
        finite = deltas[np.isfinite(deltas)]
        rows.append(
            {
                "protocol": protocol,
                "metric": "global_point_rmse_delta",
                "resampling_unit": "well",
                "n_wells": n,
                "n_bootstrap": int(len(finite)),
                "observed_delta": obs_delta,
                "bootstrap_mean": float(np.mean(finite)) if finite.size else np.nan,
                "ci_low_2.5": float(np.quantile(finite, 0.025)) if finite.size else np.nan,
                "ci_high_97.5": float(np.quantile(finite, 0.975)) if finite.size else np.nan,
                "frac_bootstrap_negative": float(np.mean(finite < 0)) if finite.size else np.nan,
                "frac_bootstrap_positive": float(np.mean(finite > 0)) if finite.size else np.nan,
                "ci_excludes_zero": bool(
                    finite.size
                    and (
                        (np.quantile(finite, 0.025) > 0)
                        or (np.quantile(finite, 0.975) < 0)
                    )
                ),
                "strongly_negative_ci": bool(
                    finite.size and np.quantile(finite, 0.975) < 0
                ),
                # "Strongly negative" for the *delta* means the candidate is
                # confidently worse (delta = cand − default > 0 with CI above 0).
                # For the decision rule we care about the opposite: CI not
                # strongly positive (candidate not confidently worse).  The
                # rule phrase "paired CI is not strongly negative" is
                # interpreted on the *improvement* scale (−delta): see
                # ``decide_candidate``.
                "note": (
                    "delta = ridge_particle_beam − ridge_default; "
                    "negative favours the candidate"
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_paired_well_delta(
    paired: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED + 1,
) -> pd.DataFrame:
    """Paired bootstrap CI for the *mean well-level* RMSE delta."""
    if paired is None or paired.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for protocol, g in paired.groupby("protocol", sort=False):
        deltas = g["delta_rmse"].to_numpy(dtype="float64")
        n = deltas.size
        if n == 0:
            continue
        obs = float(np.mean(deltas))
        boot = np.empty(n_boot, dtype="float64")
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[b] = float(np.mean(deltas[idx]))
        finite = boot[np.isfinite(boot)]
        rows.append(
            {
                "protocol": protocol,
                "metric": "mean_well_rmse_delta",
                "resampling_unit": "well",
                "n_wells": n,
                "n_bootstrap": int(len(finite)),
                "observed_delta": obs,
                "bootstrap_mean": float(np.mean(finite)) if finite.size else np.nan,
                "ci_low_2.5": float(np.quantile(finite, 0.025)) if finite.size else np.nan,
                "ci_high_97.5": float(np.quantile(finite, 0.975)) if finite.size else np.nan,
                "frac_bootstrap_negative": float(np.mean(finite < 0)) if finite.size else np.nan,
                "frac_bootstrap_positive": float(np.mean(finite > 0)) if finite.size else np.nan,
                "ci_excludes_zero": bool(
                    finite.size
                    and (
                        (np.quantile(finite, 0.025) > 0)
                        or (np.quantile(finite, 0.975) < 0)
                    )
                ),
                "strongly_negative_ci": bool(
                    finite.size and np.quantile(finite, 0.975) < 0
                ),
                "note": (
                    "delta = ridge_particle_beam − ridge_default per well, "
                    "then mean; negative favours the candidate"
                ),
            }
        )
    return pd.DataFrame(rows)


def stratified_delta(
    paired: pd.DataFrame,
    column: str,
    bins,
    labels,
    name: str,
) -> pd.DataFrame:
    """Mean/global delta RMSE inside strata of a well-level covariate."""
    if paired is None or paired.empty or column not in paired.columns:
        return pd.DataFrame()
    local = paired.copy()
    local["stratum"] = pd.cut(local[column], bins=bins, labels=labels)
    rows = []
    for (protocol, stratum), g in local.groupby(
        ["protocol", "stratum"], observed=False, sort=False
    ):
        if g.empty:
            continue
        n_pts = float(g["n_points"].sum())
        rows.append(
            {
                "protocol": protocol,
                "stratify_by": name,
                "stratum": str(stratum),
                "n_wells": int(len(g)),
                "n_points": int(n_pts),
                "mean_well_delta_rmse": float(g["delta_rmse"].mean()),
                "median_well_delta_rmse": float(g["delta_rmse"].median()),
                "global_point_delta_rmse": (
                    float(np.sqrt(g["sse_candidate"].sum() / n_pts))
                    - float(np.sqrt(g["sse_default"].sum() / n_pts))
                )
                if n_pts
                else np.nan,
                "n_improved": int(g["improved"].sum()),
                "n_degraded": int(g["degraded"].sum()),
            }
        )
    return pd.DataFrame(rows)


def all_stratified_deltas(paired: pd.DataFrame) -> pd.DataFrame:
    parts = [
        stratified_delta(paired, "gr_missing_frac", GR_BINS, GR_LABELS, "gr_missingness"),
        stratified_delta(paired, "suffix_len", SUFFIX_BINS, SUFFIX_LABELS, "hidden_suffix_length"),
        stratified_delta(paired, "prefix_len", PREFIX_BINS, PREFIX_LABELS, "prefix_length"),
    ]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def long_well_concentration(paired: pd.DataFrame) -> pd.DataFrame:
    """Whether the global gain is driven by the longest-suffix wells."""
    if paired is None or paired.empty or "suffix_len" not in paired.columns:
        return pd.DataFrame()
    rows = []
    for protocol, g in paired.groupby("protocol", sort=False):
        g = g.copy()
        q75 = float(g["suffix_len"].quantile(0.75))
        long = g[g["suffix_len"] >= q75]
        rest = g[g["suffix_len"] < q75]
        total_gain = float((-g["delta_sse"].clip(upper=0.0)).sum())
        long_gain = float((-long["delta_sse"].clip(upper=0.0)).sum())
        rows.append(
            {
                "protocol": protocol,
                "suffix_len_p75": q75,
                "n_long_wells": int(len(long)),
                "n_other_wells": int(len(rest)),
                "long_mean_delta_rmse": float(long["delta_rmse"].mean()) if len(long) else np.nan,
                "other_mean_delta_rmse": float(rest["delta_rmse"].mean()) if len(rest) else np.nan,
                "long_sse_improvement_share": (
                    long_gain / total_gain if total_gain > 0 else np.nan
                ),
                "concentrated_in_long_wells": bool(
                    total_gain > 0 and (long_gain / total_gain) >= CONCENTRATION_SSE_SHARE
                ),
            }
        )
    return pd.DataFrame(rows)


def generator_diagnostics(well: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PF/Beam confidence and fallback rates from well rows."""
    if well is None or well.empty:
        return pd.DataFrame()
    frame = well[well["protocol"].isin(REAL_PROTOCOLS)].copy()
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (model, protocol), g in frame.groupby(["model", "protocol"], sort=False):
        row = {
            "model": model,
            "protocol": protocol,
            "n_wells": int(g["well_id"].nunique()),
        }
        for prefix in ("particle", "beam"):
            conf = f"{prefix}_confidence_mean"
            conf_p10 = f"{prefix}_confidence_p10"
            fallback = f"{prefix}_fallback_fraction"
            status = f"{prefix}_fallback_status"
            reason = f"{prefix}_failure_reason"
            if conf in g.columns and g[conf].notna().any():
                row[f"{prefix}_confidence_mean"] = float(
                    pd.to_numeric(g[conf], errors="coerce").mean()
                )
                row[f"{prefix}_confidence_p10_mean"] = (
                    float(pd.to_numeric(g[conf_p10], errors="coerce").mean())
                    if conf_p10 in g.columns
                    else np.nan
                )
                row[f"{prefix}_fallback_fraction_mean"] = (
                    float(pd.to_numeric(g[fallback], errors="coerce").mean())
                    if fallback in g.columns
                    else np.nan
                )
                if status in g.columns:
                    row[f"{prefix}_fallback_well_rate"] = float(
                        _bool_series(g, status).mean()
                    )
                else:
                    row[f"{prefix}_fallback_well_rate"] = np.nan
                if reason in g.columns:
                    reasons = g[reason].fillna("").astype(str)
                    nonempty = reasons[reasons.str.strip() != ""]
                    row[f"{prefix}_n_failure_reasons"] = int(nonempty.size)
                    row[f"{prefix}_top_failure_reason"] = (
                        nonempty.value_counts().index[0] if len(nonempty) else ""
                    )
                else:
                    row[f"{prefix}_n_failure_reasons"] = 0
                    row[f"{prefix}_top_failure_reason"] = ""
            else:
                row[f"{prefix}_confidence_mean"] = np.nan
                row[f"{prefix}_confidence_p10_mean"] = np.nan
                row[f"{prefix}_fallback_fraction_mean"] = np.nan
                row[f"{prefix}_fallback_well_rate"] = np.nan
                row[f"{prefix}_n_failure_reasons"] = 0
                row[f"{prefix}_top_failure_reason"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def owner_aggregate_table() -> pd.DataFrame:
    """The run-owner global RMSE table (no per-well support)."""
    rows = []
    for protocol, models in OWNER_GLOBAL_RMSE.items():
        base = models[DEFAULT_MODEL]
        for model, rmse in models.items():
            rows.append(
                {
                    "source": "owner_aggregate",
                    "validation": REAL_BANNER,
                    "protocol": protocol,
                    "model": model,
                    "n_wells": OWNER_N_WELLS,
                    "n_failures": OWNER_N_FAILURES,
                    "global_rmse": rmse,
                    "delta_vs_default": rmse - base,
                }
            )
    return pd.DataFrame(rows)


def decide_candidate(
    *,
    well_summary: pd.DataFrame | None,
    fold_table: pd.DataFrame | None,
    global_boot: pd.DataFrame | None,
    paired_boot: pd.DataFrame | None,
    long_wells: pd.DataFrame | None,
    owner_only: bool = False,
) -> dict:
    """Apply the pre-registered robustness decision rule.

    Keep ``ridge_particle_beam`` as the *next candidate* only if:

    1. the global improvement is stable across all five folds (both protocols), and
    2. the paired confidence interval is not strongly against the candidate, and
    3. the improvement is not concentrated in a small number of wells.

    ``ridge_default`` is always preserved as the fallback.  The candidate is
    never promoted to a final submission from this function.
    """
    decision = {
        "candidate": CANDIDATE_MODEL,
        "fallback": DEFAULT_MODEL,
        "keep_as_next_candidate": False,
        "use_as_final": False,
        "preserve_default_fallback": True,
        "delete_pf_beam_code": False,
        "owner_only_aggregates": bool(owner_only),
        "reasons": [],
        "protocol_notes": {},
    }

    if owner_only or well_summary is None or well_summary.empty:
        # Aggregates alone cannot establish fold stability, paired CI, or
        # concentration.  The small unseen-well gain is recorded but is not
        # enough under the rule.
        unseen = OWNER_GLOBAL_RMSE[PROTOCOL_B]
        masked = OWNER_GLOBAL_RMSE[PROTOCOL_A]
        d_unseen = unseen[CANDIDATE_MODEL] - unseen[DEFAULT_MODEL]
        d_masked = masked[CANDIDATE_MODEL] - masked[DEFAULT_MODEL]
        decision["reasons"] = [
            (
                f"Owner-supplied global RMSE only: unseen_well delta "
                f"{d_unseen:+.3f} ({unseen[DEFAULT_MODEL]:.3f} → "
                f"{unseen[CANDIDATE_MODEL]:.3f}); same_well_masked delta "
                f"{d_masked:+.3f} ({masked[DEFAULT_MODEL]:.3f} → "
                f"{masked[CANDIDATE_MODEL]:.3f})."
            ),
            (
                "Per-well table was not available in this checkout, so fold "
                "stability, paired bootstrap CI, improved/degraded well counts, "
                "and concentration cannot be verified."
            ),
            (
                "Under the pre-registered rule, ridge_particle_beam is NOT "
                "kept as the next candidate until those robustness checks pass "
                "on the cross-fitted well-level artifact."
            ),
            (
                "ridge_default remains the default and the fallback. PF/Beam "
                "code is retained. No final submission is authorised."
            ),
        ]
        decision["protocol_notes"] = {
            PROTOCOL_B: {
                "global_delta": d_unseen,
                "stable_across_folds": None,
                "paired_ci_ok": None,
                "not_concentrated": None,
            },
            PROTOCOL_A: {
                "global_delta": d_masked,
                "stable_across_folds": None,
                "paired_ci_ok": None,
                "not_concentrated": None,
            },
        }
        return decision

    # ---- full well-level path -------------------------------------------
    ok_protocols = []
    for protocol in REAL_PROTOCOLS:
        notes: dict = {"protocol": protocol}
        ws = well_summary[well_summary["protocol"] == protocol]
        if ws.empty:
            notes["available"] = False
            decision["protocol_notes"][protocol] = notes
            decision["reasons"].append(f"{protocol}: no paired wells.")
            continue
        notes["available"] = True
        row = ws.iloc[0]
        notes["global_delta"] = float(row["global_point_delta_rmse"])
        notes["pct_improved"] = float(row["pct_improved"])
        notes["pct_degraded"] = float(row["pct_degraded"])
        notes["mean_well_delta"] = float(row["mean_well_delta_rmse"])
        notes["median_well_delta"] = float(row["median_well_delta_rmse"])
        notes["improvement_concentrated"] = bool(row["improvement_concentrated"])
        notes["top10_sse_share"] = float(row["top10_sse_improvement_share"])

        # Fold stability
        stable = False
        if fold_table is not None and not fold_table.empty:
            ft = fold_table[fold_table["protocol"] == protocol]
            if not ft.empty:
                stable = bool(ft["stable_across_folds"].iloc[0])
                notes["n_folds"] = int(ft["n_folds"].iloc[0])
                notes["n_folds_not_worse"] = int(ft["n_folds_candidate_not_worse"].iloc[0])
        notes["stable_across_folds"] = stable

        # Paired CI: not strongly against the candidate.
        # On the delta = cand − default scale, "strongly against" means the
        # lower CI bound is clearly positive (candidate worse).  Equivalently,
        # on the improvement scale (−delta), the CI is not strongly negative.
        paired_ok = False
        if paired_boot is not None and not paired_boot.empty:
            pb = paired_boot[paired_boot["protocol"] == protocol]
            if not pb.empty:
                lo = float(pb["ci_low_2.5"].iloc[0])
                hi = float(pb["ci_high_97.5"].iloc[0])
                notes["paired_ci_low"] = lo
                notes["paired_ci_high"] = hi
                # Strongly against candidate if entire CI is positive.
                strongly_against = lo > 0.0
                paired_ok = not strongly_against
        notes["paired_ci_ok"] = paired_ok

        if global_boot is not None and not global_boot.empty:
            gb = global_boot[global_boot["protocol"] == protocol]
            if not gb.empty:
                notes["global_ci_low"] = float(gb["ci_low_2.5"].iloc[0])
                notes["global_ci_high"] = float(gb["ci_high_97.5"].iloc[0])
                notes["global_ci_excludes_zero"] = bool(gb["ci_excludes_zero"].iloc[0])

        concentrated = bool(row["improvement_concentrated"])
        if long_wells is not None and not long_wells.empty:
            lw = long_wells[long_wells["protocol"] == protocol]
            if not lw.empty and bool(lw["concentrated_in_long_wells"].iloc[0]):
                concentrated = True
                notes["concentrated_in_long_wells"] = True
        notes["not_concentrated"] = not concentrated

        improves = float(row["global_point_delta_rmse"]) < 0.0
        notes["improves_global"] = improves
        protocol_ok = bool(improves and stable and paired_ok and not concentrated)
        notes["passes_rule"] = protocol_ok
        decision["protocol_notes"][protocol] = notes
        if protocol_ok:
            ok_protocols.append(protocol)
        else:
            missing = []
            if not improves:
                missing.append("no global improvement")
            if not stable:
                missing.append("not stable across folds")
            if not paired_ok:
                missing.append("paired CI does not support the candidate")
            if concentrated:
                missing.append("improvement concentrated in few/long wells")
            decision["reasons"].append(
                f"{protocol}: fails robustness rule ({', '.join(missing)})."
            )

    # Require both protocols under the same spirit as other promotion rules.
    if set(ok_protocols) == set(REAL_PROTOCOLS):
        decision["keep_as_next_candidate"] = True
        decision["reasons"].append(
            "Both protocols pass fold stability, paired-CI and concentration checks. "
            "ridge_particle_beam may be kept as the next candidate; ridge_default "
            "remains the fallback. Not authorised as a final submission."
        )
    else:
        decision["keep_as_next_candidate"] = False
        decision["reasons"].append(
            "ridge_particle_beam is NOT kept as the next candidate. "
            "ridge_default remains the default and the fallback. "
            "PF/Beam implementations are retained for diagnostics. "
            "No final submission is authorised."
        )
    return decision


def _empty_csv_schema(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def write_robustness_reports(
    reports_dir: str | Path,
    well: pd.DataFrame | None = None,
    *,
    failures: pd.DataFrame | None = None,
    environment: dict | None = None,
    owner_aggregates_ok: bool = True,
) -> list[Path]:
    """Write the five required PF/Beam robustness artifacts.

    If ``well`` is None or empty and ``owner_aggregates_ok`` is True, the
    decision and failure-analysis markdown files are written from the
    owner-supplied global RMSE table, and the three CSV files receive an
    explicit unavailable schema rather than fabricated well-level numbers.
    """
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    env = environment or {}
    n_failures = int(len(failures)) if failures is not None else OWNER_N_FAILURES

    has_wells = well is not None and not well.empty
    paired = pair_default_vs_candidate(well) if has_wells else pd.DataFrame()
    summary = well_level_summary(paired) if not paired.empty else pd.DataFrame()
    folds = fold_deltas(well) if has_wells else pd.DataFrame()
    global_boot = (
        bootstrap_global_rmse_delta(paired) if not paired.empty else pd.DataFrame()
    )
    paired_boot = (
        bootstrap_paired_well_delta(paired) if not paired.empty else pd.DataFrame()
    )
    strat = all_stratified_deltas(paired) if not paired.empty else pd.DataFrame()
    long_wells = long_well_concentration(paired) if not paired.empty else pd.DataFrame()
    diag = generator_diagnostics(well) if has_wells else pd.DataFrame()
    owner = owner_aggregate_table() if owner_aggregates_ok else pd.DataFrame()

    decision = decide_candidate(
        well_summary=summary if not summary.empty else None,
        fold_table=folds if not folds.empty else None,
        global_boot=global_boot if not global_boot.empty else None,
        paired_boot=paired_boot if not paired_boot.empty else None,
        long_wells=long_wells if not long_wells.empty else None,
        owner_only=not has_wells or paired.empty,
    )

    written: list[Path] = []

    # ---- 1. paired well deltas ------------------------------------------
    well_csv = root / "pf_beam_paired_well_deltas.csv"
    if paired.empty:
        schema = _empty_csv_schema(
            [
                "protocol", "well_id", "fold", "n_points", "prefix_len", "suffix_len",
                "gr_missing_frac", "rmse_default", "rmse_candidate", "sse_default",
                "sse_candidate", "delta_rmse", "delta_sse", "improved", "degraded",
                "unchanged", "availability",
            ]
        )
        # One explanatory row so the file is not mistaken for a corrupt empty.
        schema = pd.DataFrame(
            [
                {
                    "protocol": PROTOCOL_B,
                    "well_id": "",
                    "fold": np.nan,
                    "n_points": np.nan,
                    "prefix_len": np.nan,
                    "suffix_len": np.nan,
                    "gr_missing_frac": np.nan,
                    "rmse_default": OWNER_GLOBAL_RMSE[PROTOCOL_B][DEFAULT_MODEL],
                    "rmse_candidate": OWNER_GLOBAL_RMSE[PROTOCOL_B][CANDIDATE_MODEL],
                    "sse_default": np.nan,
                    "sse_candidate": np.nan,
                    "delta_rmse": (
                        OWNER_GLOBAL_RMSE[PROTOCOL_B][CANDIDATE_MODEL]
                        - OWNER_GLOBAL_RMSE[PROTOCOL_B][DEFAULT_MODEL]
                    ),
                    "delta_sse": np.nan,
                    "improved": np.nan,
                    "degraded": np.nan,
                    "unchanged": np.nan,
                    "availability": (
                        "UNAVAILABLE_per_well — only owner global RMSE is known; "
                        "re-run analysis on particle_beam_wells.csv to populate"
                    ),
                },
                {
                    "protocol": PROTOCOL_A,
                    "well_id": "",
                    "fold": np.nan,
                    "n_points": np.nan,
                    "prefix_len": np.nan,
                    "suffix_len": np.nan,
                    "gr_missing_frac": np.nan,
                    "rmse_default": OWNER_GLOBAL_RMSE[PROTOCOL_A][DEFAULT_MODEL],
                    "rmse_candidate": OWNER_GLOBAL_RMSE[PROTOCOL_A][CANDIDATE_MODEL],
                    "sse_default": np.nan,
                    "sse_candidate": np.nan,
                    "delta_rmse": (
                        OWNER_GLOBAL_RMSE[PROTOCOL_A][CANDIDATE_MODEL]
                        - OWNER_GLOBAL_RMSE[PROTOCOL_A][DEFAULT_MODEL]
                    ),
                    "delta_sse": np.nan,
                    "improved": np.nan,
                    "degraded": np.nan,
                    "unchanged": np.nan,
                    "availability": (
                        "UNAVAILABLE_per_well — only owner global RMSE is known; "
                        "re-run analysis on particle_beam_wells.csv to populate"
                    ),
                },
            ]
        )
        schema.to_csv(well_csv, index=False)
    else:
        paired.to_csv(well_csv, index=False)
    written.append(well_csv)

    # ---- 2. fold deltas -------------------------------------------------
    fold_csv = root / "pf_beam_fold_deltas.csv"
    if folds.empty:
        pd.DataFrame(
            [
                {
                    "protocol": protocol,
                    "fold": np.nan,
                    "n_wells": OWNER_N_WELLS,
                    "n_points": np.nan,
                    "rmse_default": OWNER_GLOBAL_RMSE[protocol][DEFAULT_MODEL],
                    "rmse_candidate": OWNER_GLOBAL_RMSE[protocol][CANDIDATE_MODEL],
                    "delta_rmse": (
                        OWNER_GLOBAL_RMSE[protocol][CANDIDATE_MODEL]
                        - OWNER_GLOBAL_RMSE[protocol][DEFAULT_MODEL]
                    ),
                    "candidate_better": (
                        OWNER_GLOBAL_RMSE[protocol][CANDIDATE_MODEL]
                        < OWNER_GLOBAL_RMSE[protocol][DEFAULT_MODEL]
                    ),
                    "n_folds": 5,
                    "n_folds_candidate_better": np.nan,
                    "n_folds_candidate_not_worse": np.nan,
                    "stable_across_folds": np.nan,
                    "mean_fold_delta_rmse": np.nan,
                    "availability": (
                        "UNAVAILABLE_per_fold — re-run on particle_beam_wells.csv"
                    ),
                }
                for protocol in REAL_PROTOCOLS
            ]
        ).to_csv(fold_csv, index=False)
    else:
        folds.to_csv(fold_csv, index=False)
    written.append(fold_csv)
    # Alias requested by the robustness brief (same bytes as pf_beam_*).
    fold_alias = root / "particle_beam_fold_deltas.csv"
    fold_alias.write_bytes(fold_csv.read_bytes())
    written.append(fold_alias)

    # ---- 3. bootstrap CI ------------------------------------------------
    boot_csv = root / "pf_beam_bootstrap_ci.csv"
    if global_boot.empty and paired_boot.empty:
        pd.DataFrame(
            [
                {
                    "protocol": protocol,
                    "metric": metric,
                    "resampling_unit": "well",
                    "n_wells": OWNER_N_WELLS,
                    "n_bootstrap": 0,
                    "observed_delta": (
                        OWNER_GLOBAL_RMSE[protocol][CANDIDATE_MODEL]
                        - OWNER_GLOBAL_RMSE[protocol][DEFAULT_MODEL]
                    )
                    if metric == "global_point_rmse_delta"
                    else np.nan,
                    "bootstrap_mean": np.nan,
                    "ci_low_2.5": np.nan,
                    "ci_high_97.5": np.nan,
                    "frac_bootstrap_negative": np.nan,
                    "frac_bootstrap_positive": np.nan,
                    "ci_excludes_zero": np.nan,
                    "strongly_negative_ci": np.nan,
                    "availability": (
                        "UNAVAILABLE — bootstrap requires per-well SSE/n_points; "
                        "observed_delta is the owner global RMSE difference only"
                    ),
                    "note": (
                        "delta = ridge_particle_beam − ridge_default; "
                        "negative favours the candidate"
                    ),
                }
                for protocol in REAL_PROTOCOLS
                for metric in ("global_point_rmse_delta", "mean_well_rmse_delta")
            ]
        ).to_csv(boot_csv, index=False)
    else:
        boot = pd.concat(
            [b for b in (global_boot, paired_boot) if not b.empty],
            ignore_index=True,
        )
        boot.to_csv(boot_csv, index=False)
    written.append(boot_csv)
    boot_alias = root / "particle_beam_bootstrap_ci.csv"
    boot_alias.write_bytes(boot_csv.read_bytes())
    written.append(boot_alias)

    # Optional supporting tables (not in the required five, but useful).
    if not summary.empty:
        summary.to_csv(root / "pf_beam_well_summary.csv", index=False)
        written.append(root / "pf_beam_well_summary.csv")
    if not strat.empty:
        strat.to_csv(root / "pf_beam_stratified_deltas.csv", index=False)
        written.append(root / "pf_beam_stratified_deltas.csv")
    if not diag.empty:
        diag.to_csv(root / "pf_beam_generator_diagnostics.csv", index=False)
        written.append(root / "pf_beam_generator_diagnostics.csv")
    if not owner.empty:
        owner.to_csv(root / "pf_beam_owner_aggregates.csv", index=False)
        written.append(root / "pf_beam_owner_aggregates.csv")
    (root / "pf_beam_decision.json").write_text(
        json.dumps(decision, indent=2, default=str), encoding="utf-8"
    )
    written.append(root / "pf_beam_decision.json")

    # ---- 4. decision markdown -------------------------------------------
    decision_md = root / "pf_beam_real_decision.md"
    decision_md.write_text(
        _render_decision_md(
            decision=decision,
            summary=summary,
            folds=folds,
            global_boot=global_boot,
            paired_boot=paired_boot,
            strat=strat,
            long_wells=long_wells,
            diag=diag,
            owner=owner,
            n_failures=n_failures,
            has_wells=has_wells and not paired.empty,
            env=env,
        ),
        encoding="utf-8",
    )
    written.append(decision_md)

    # ---- 5. failure analysis markdown -----------------------------------
    failure_md = root / "pf_beam_failure_analysis.md"
    failure_md.write_text(
        _render_failure_md(
            decision=decision,
            summary=summary,
            folds=folds,
            global_boot=global_boot,
            paired_boot=paired_boot,
            strat=strat,
            long_wells=long_wells,
            diag=diag,
            failures=failures,
            owner=owner,
            has_wells=has_wells and not paired.empty,
            n_failures=n_failures,
        ),
        encoding="utf-8",
    )
    written.append(failure_md)

    return written


def _render_decision_md(
    *,
    decision: dict,
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    global_boot: pd.DataFrame,
    paired_boot: pd.DataFrame,
    strat: pd.DataFrame,
    long_wells: pd.DataFrame,
    diag: pd.DataFrame,
    owner: pd.DataFrame,
    n_failures: int,
    has_wells: bool,
    env: dict,
) -> str:
    keep = decision["keep_as_next_candidate"]
    status = (
        "KEEP as next candidate (not final)"
        if keep
        else "DO NOT keep as next candidate — preserve ridge_default"
    )
    lines = [
        "# PF + Beam robustness decision\n",
        f"**Decision: {status}.**\n",
        "## Evidence classification — read this before quoting anything\n",
        "Findings are separated into three classes and **must not be conflated**.\n",
        "### A. Real Kaggle validation\n",
        (
            "Established by the completed 770-well PF/Beam run, both protocols, "
            "cross-fitted by well ID, zero task/fit/predict failures. Global "
            "point-level RMSE figures below are run-owner aggregates from that "
            f"run ({OWNER_SOURCE}).\n"
        ),
        "### B. Synthetic verification\n",
        (
            "Harness checks under `reports/synthetic_validation/` and "
            "`reports/synthetic_ablation/`. Banner-stamped "
            f"`{SYNTHETIC_BANNER}`. **Not competition results.** No synthetic "
            "RMSE is used in this decision.\n"
        ),
        "### C. Public leaderboard results\n",
        (
            f"**{PUBLIC_LB_BANNER}: none.** No submission was created from the "
            "PF/Beam experiment, and no public-leaderboard score is claimed or "
            "available for this branch.\n"
        ),
        "## Pre-registered decision rule\n",
        (
            "1. Keep `ridge_particle_beam` as the **next candidate** only if the "
            "global improvement is stable across all five folds **and** the "
            "paired confidence interval is not strongly against the candidate.\n"
            "2. Do **not** use it as final if the improvement is caused only by "
            "a small number of wells (or only by long wells).\n"
            "3. Preserve `ridge_default` as the fallback in every case.\n"
            "4. Do not delete PF or Beam code.\n"
            "5. Do not start external artifacts.\n"
            "6. Do not use the direct dip-constrained alignment model.\n"
            "7. Do not create a final submission from this analysis.\n"
        ),
        "## Real Kaggle validation — owner global RMSE\n",
        (
            "Protocols are separate and are **never averaged**. "
            f"Failures recorded for the completed run: **{n_failures}**.\n"
        ),
        _md_table(owner) if not owner.empty else "_No owner aggregates._\n",
        (
            f"| Protocol | ridge_default | ridge_particle_beam | "
            f"delta (cand − default) |\n"
            f"|---|---:|---:|---:|\n"
            f"| `unseen_well` | "
            f"{OWNER_GLOBAL_RMSE[PROTOCOL_B][DEFAULT_MODEL]:.3f} | "
            f"{OWNER_GLOBAL_RMSE[PROTOCOL_B][CANDIDATE_MODEL]:.3f} | "
            f"{OWNER_GLOBAL_RMSE[PROTOCOL_B][CANDIDATE_MODEL] - OWNER_GLOBAL_RMSE[PROTOCOL_B][DEFAULT_MODEL]:+.3f} |\n"
            f"| `same_well_masked` | "
            f"{OWNER_GLOBAL_RMSE[PROTOCOL_A][DEFAULT_MODEL]:.3f} | "
            f"{OWNER_GLOBAL_RMSE[PROTOCOL_A][CANDIDATE_MODEL]:.3f} | "
            f"{OWNER_GLOBAL_RMSE[PROTOCOL_A][CANDIDATE_MODEL] - OWNER_GLOBAL_RMSE[PROTOCOL_A][DEFAULT_MODEL]:+.3f} |\n"
        ),
        (
            "Full four-branch owner table (global point-level RMSE):\n\n"
            "| Protocol | ridge_default | +PF | +Beam | +PF+Beam |\n"
            "|---|---:|---:|---:|---:|\n"
            f"| `unseen_well` | 14.423 | 14.429 | 14.432 | **14.419** |\n"
            f"| `same_well_masked` | 29.486 | 29.406 | 29.406 | **29.388** |\n"
        ),
        (
            "On the aggregates alone, the combined PF+Beam branch is the best "
            "of the four under both protocols. The unseen-well gain is very "
            "small (−0.004 RMSE). No statistical significance is claimed from "
            "these two scalars.\n"
        ),
    ]

    if has_wells:
        lines += [
            "## Per-well paired deltas (computed)\n",
            _md_table(summary),
            "### Fold-level RMSE deltas\n",
            _md_table(
                folds.drop(
                    columns=[c for c in folds.columns if c.startswith("n_folds") and c != "fold"],
                    errors="ignore",
                )
                if not folds.empty
                else folds
            ),
            "### Bootstrap confidence intervals\n",
            (
                "Well-cluster bootstrap of the global point-level RMSE delta and "
                "the mean well-level RMSE delta. "
                f"{BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED}. "
                "These are descriptive intervals, not a hypothesis test; "
                "exclusion of zero is reported without a p-value claim.\n"
            ),
            _md_table(
                pd.concat(
                    [b for b in (global_boot, paired_boot) if not b.empty],
                    ignore_index=True,
                )
                if not global_boot.empty or not paired_boot.empty
                else pd.DataFrame()
            ),
            "### Stratified deltas\n",
            _md_table(strat) if not strat.empty else "_Unavailable._\n",
            "### Long-well concentration\n",
            _md_table(long_wells) if not long_wells.empty else "_Unavailable._\n",
            "### PF / Beam generator diagnostics\n",
            _md_table(diag) if not diag.empty else "_Unavailable._\n",
        ]
    else:
        lines += [
            "## Per-well / fold / bootstrap analysis\n",
            (
                "**Unavailable in this checkout.** The cross-fitted "
                "`particle_beam_wells.csv` (per-well SSE, n_points, fold, "
                "prefix/suffix length, GR missingness, PF/Beam diagnostics) "
                "was not present. Without it the following quantities cannot "
                "be computed and are **not fabricated**:\n\n"
                "1. Per-well RMSE delta (`ridge_particle_beam − ridge_default`)\n"
                "2. Number and percentage of wells improved / degraded\n"
                "3. Fold-level RMSE deltas and five-fold stability\n"
                "4. Bootstrap CI for the global RMSE delta\n"
                "5. Paired bootstrap CI over wells\n"
                "6. Mean and median well-level delta; worst-10 delta\n"
                "7. Error by GR missingness / hidden suffix length / prefix length\n"
                "8. PF and Beam confidence and fallback rates\n"
                "9. Whether the gain is concentrated in a few long wells\n\n"
                "The CSV companions "
                "`pf_beam_paired_well_deltas.csv`, "
                "`pf_beam_fold_deltas.csv`, and "
                "`pf_beam_bootstrap_ci.csv` record the owner global deltas "
                "and mark well-/fold-/bootstrap-level fields as "
                "`UNAVAILABLE`. Re-run:\n\n"
                "```bash\n"
                "python scripts/analyze_pf_beam_robustness.py \\\n"
                "  --reports-dir /path/to/particle_beam_reports\n"
                "```\n\n"
                "against the completed run's `particle_beam_wells.csv` to "
                "populate them without retraining.\n"
            ),
        ]

    lines += [
        "## Decision\n",
        f"**keep_as_next_candidate = `{decision['keep_as_next_candidate']}`**\n",
        f"**use_as_final = `{decision['use_as_final']}`**\n",
        f"**preserve_default_fallback = `{decision['preserve_default_fallback']}`**\n",
        f"**delete_pf_beam_code = `{decision['delete_pf_beam_code']}`**\n",
        "### Reasons\n",
    ]
    for reason in decision["reasons"]:
        lines.append(f"- {reason}")
    lines += [
        "\n### Applied outcome\n",
        (
            "- **Default predictor:** `ridge_default` "
            "(`RidgeBaseline(alignment_features=False, spatial=None)`).\n"
            "- **Next candidate:** not promoted from this analysis"
            if not keep
            else "- **Next candidate:** `ridge_particle_beam` may remain under evaluation"
        ),
        (
            ".\n- **PF and Beam code:** retained (`src/particle_filter.py`, "
            "`src/beam_search.py`, Ridge opt-in flags).\n"
            "- **Direct dip-constrained alignment:** still REJECTED "
            "(`src/model_status.py`).\n"
            "- **External artifacts:** not used.\n"
            "- **Final submission:** not created.\n"
        ),
        "## Synthetic verification\n",
        (
            f"Status: harness-only under `reports/synthetic_*` "
            f"({SYNTHETIC_BANNER}). Not used for this decision.\n"
        ),
        "## Public leaderboard results\n",
        (
            f"Status: **none** ({PUBLIC_LB_BANNER}). No PF/Beam submission "
            "exists; no LB number is reported.\n"
        ),
    ]
    if env:
        env_rows = pd.DataFrame(
            [
                {
                    "key": k,
                    "value": json.dumps(v) if isinstance(v, (dict, list)) else str(v),
                }
                for k, v in env.items()
            ]
        )
        lines += ["## Run environment (if supplied)\n", _md_table(env_rows)]
    return "\n".join(lines)


def _render_failure_md(
    *,
    decision: dict,
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    global_boot: pd.DataFrame,
    paired_boot: pd.DataFrame,
    strat: pd.DataFrame,
    long_wells: pd.DataFrame,
    diag: pd.DataFrame,
    failures: pd.DataFrame | None,
    owner: pd.DataFrame,
    has_wells: bool,
    n_failures: int,
) -> str:
    lines = [
        "# PF + Beam failure and robustness analysis\n",
        "## Evidence classification\n",
        "| Class | Scope | Used for decision? |\n"
        "|---|---|---|\n"
        f"| **A. Real Kaggle validation** | Completed 770-well PF/Beam run; "
        f"owner global RMSE; well-level table when mounted | **Yes** |\n"
        f"| **B. Synthetic verification** | `reports/synthetic_*` harness | **No** |\n"
        f"| **C. Public leaderboard** | No PF/Beam submission | **No — none exist** |\n",
        "## 0. What failed?\n",
        (
            f"Task / fit / predict failures on the completed real run: "
            f"**{n_failures}**. There is no crash-level failure to debug. "
            "The question is whether the small global RMSE gain of "
            "`ridge_particle_beam` over `ridge_default` is robust enough to "
            "keep the combined branch as the next candidate.\n"
        ),
        "## 1. Real Kaggle validation — global point-level RMSE\n",
        _md_table(owner) if not owner.empty else "_No owner table._\n",
        (
            "Combined PF+Beam is best on both protocols in the owner table. "
            "Unseen-well improvement is −0.004 RMSE (14.423 → 14.419). "
            "Same-well masked improvement is −0.098 RMSE (29.486 → 29.388). "
            "PF-only and Beam-only each *worsen* unseen-well slightly "
            "(+0.006 / +0.009). No significance claim is made from aggregates "
            "alone.\n"
        ),
        "## 2. Per-well RMSE delta\n",
    ]
    if has_wells and not summary.empty:
        lines += [
            _md_table(summary),
            (
                "`delta_rmse = rmse(ridge_particle_beam) − rmse(ridge_default)`. "
                "Negative means the candidate is better on that well / in that "
                "global aggregate.\n"
            ),
        ]
    else:
        lines += [
            (
                "**Unavailable.** `particle_beam_wells.csv` was not present in "
                "this checkout. Improved/degraded well counts, mean/median "
                "well delta and worst-10 delta are not invented.\n"
            ),
        ]

    lines += ["## 3. Fold-level stability\n"]
    if has_wells and not folds.empty:
        lines += [
            _md_table(folds),
            (
                "Stable means the candidate is not worse than the default on "
                "every fold of that protocol.\n"
            ),
        ]
    else:
        lines += [
            (
                "**Unavailable** without per-fold well rows. The decision rule "
                "requires stability across all five folds; that check is "
                "recorded as unmet until the well-level artifact is analysed.\n"
            ),
        ]

    lines += ["## 4. Bootstrap confidence intervals\n"]
    if has_wells and (not global_boot.empty or not paired_boot.empty):
        boot = pd.concat(
            [b for b in (global_boot, paired_boot) if not b.empty],
            ignore_index=True,
        )
        lines += [
            _md_table(boot),
            (
                "Intervals are well-cluster bootstrap quantiles. They are "
                "**not** p-values. A CI that includes zero means the observed "
                "delta is compatible with no effect under well resampling; "
                "that is reported descriptively.\n"
            ),
        ]
    else:
        lines += [
            (
                "**Unavailable** without per-well SSE and point counts. The "
                "CSV `pf_beam_bootstrap_ci.csv` stores the owner observed "
                "global delta with `n_bootstrap = 0` and empty CI bounds so "
                "the absence is machine-readable.\n"
            ),
        ]

    lines += [
        "## 5. Error by GR missingness / suffix length / prefix length\n",
    ]
    if has_wells and not strat.empty:
        lines += [_md_table(strat)]
    else:
        lines += [
            (
                "**Unavailable** without per-well covariates from the "
                "cross-fitted table.\n"
            ),
        ]

    lines += ["## 6. Concentration in a few / long wells\n"]
    if has_wells and not long_wells.empty:
        lines += [_md_table(long_wells)]
        if not summary.empty and "top10_sse_improvement_share" in summary.columns:
            lines += [
                (
                    f"Top-{CONCENTRATION_TOP_K} wells' share of total SSE "
                    "improvement (threshold for 'concentrated': "
                    f"{CONCENTRATION_SSE_SHARE:.0%}):\n"
                ),
                _md_table(
                    summary[
                        [
                            "protocol",
                            "top10_sse_improvement_share",
                            "improvement_concentrated",
                        ]
                    ]
                ),
            ]
    else:
        lines += [
            (
                "**Unavailable.** Cannot test whether the −0.004 unseen-well "
                "gain is carried by a handful of long wells without the "
                "per-well table.\n"
            ),
        ]

    lines += ["## 7. PF and Beam confidence / fallback rates\n"]
    if has_wells and not diag.empty:
        lines += [_md_table(diag)]
    else:
        lines += [
            (
                "**Unavailable** in this checkout. The completed run reported "
                "zero task/fit/predict failures; generator-level confidence "
                "and fallback fractions live on "
                "`particle_beam_diagnostics.csv` from the runner and should be "
                "joined when that file is mounted.\n"
            ),
        ]

    if failures is not None and len(failures):
        lines += [
            "## 8. Recorded failures\n",
            _md_table(failures.head(50)),
        ]
    else:
        lines += [
            "## 8. Recorded failures\n",
            f"None (n_failures = {n_failures}).\n",
        ]

    lines += [
        "## 9. Decision linkage\n",
        (
            f"`keep_as_next_candidate = {decision['keep_as_next_candidate']}`\n\n"
            "Reasons:\n"
        ),
    ]
    for reason in decision["reasons"]:
        lines.append(f"- {reason}")
    lines += [
        "\n## 10. Synthetic verification\n",
        (
            f"{SYNTHETIC_BANNER}. See `reports/synthetic_validation/` and "
            "`reports/synthetic_ablation/`. Not used here.\n"
        ),
        "## 11. Public leaderboard results\n",
        (
            f"{PUBLIC_LB_BANNER}: **no PF/Beam submission has been filed.** "
            "No LB score is available or claimed.\n"
        ),
        "## 12. What was deliberately not done\n",
        (
            "- No retrain of Ridge or regeneration of PF/Beam features.\n"
            "- No final submission.\n"
            "- No promotion of `ridge_particle_beam` over `ridge_default`.\n"
            "- No deletion of PF/Beam code.\n"
            "- No external artifacts.\n"
            "- No use of the rejected direct dip-constrained alignment model.\n"
            "- No fabricated bootstrap CI, fold table, or improved-well "
            "percentage from aggregates alone.\n"
            "- No claim of statistical significance.\n"
        ),
    ]
    return "\n".join(lines)


def resolve_well_table(reports_dir: str | Path) -> Path | None:
    """Locate the best available well-level PF/Beam artifact."""
    root = Path(reports_dir)
    for name in (
        "particle_beam_wells.csv",
        "well_level_validation.csv",
        "pf_beam_paired_well_deltas.csv",
    ):
        path = root / name
        if path.exists():
            # The paired-deltas file is already paired; only the first two are
            # raw multi-model tables.
            if name == "pf_beam_paired_well_deltas.csv":
                # Usable only if it already has real well rows (not the
                # unavailable placeholder).
                try:
                    peek = pd.read_csv(path, nrows=5)
                except Exception:
                    continue
                if "availability" in peek.columns:
                    cont = peek["availability"].astype(str)
                    if cont.str.contains("UNAVAILABLE", na=False).all():
                        continue
                # Paired form is not a multi-model table; skip as input source.
                continue
            return path
    return None
