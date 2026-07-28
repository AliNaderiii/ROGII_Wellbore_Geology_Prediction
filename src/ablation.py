"""Ridge alignment/spatial feature ablation — a 2x2 factorial, per protocol.

Four branches are fitted and scored through the *existing* Ridge model. No
new model is introduced and the shipped baseline is not modified: branch B is
byte-for-byte the current Ridge feature set, and the other branches are
reached only through the explicit ``alignment_features`` switch and the
existing fold-train-only spatial prior.

    A  ridge_no_align            alignment features removed, no spatial
    B  ridge_baseline            current Ridge baseline (reference)
    C  ridge_spatial_only        alignment features removed, + spatial
    D  ridge_align_spatial       alignment features + spatial

Every branch is cross-fitted by well ID under both protocols, using the same
folds, so the four numbers within a protocol are paired. Protocols are never
averaged together.

Deltas are reported against branch B, the current Ridge baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.baselines import RidgeBaseline
from src.spatial import SpatialConfig, SpatialPrior
from src.validation import (
    PROTOCOL_A,
    PROTOCOL_B,
    Fold,
    assert_no_blocked_wells,
    evaluate_models,
    CrossFitLeakage,
)

#: Branch identifiers, in report order. ``BASELINE_BRANCH`` is the reference
#: every delta is taken against.
BRANCH_A = "ridge_no_align"
BRANCH_B = "ridge_baseline"
BRANCH_C = "ridge_spatial_only"
BRANCH_D = "ridge_align_spatial"
BASELINE_BRANCH = BRANCH_B

BRANCH_ORDER = (BRANCH_A, BRANCH_B, BRANCH_C, BRANCH_D)

BRANCH_LABELS = {
    BRANCH_A: "A. Ridge without alignment features",
    BRANCH_B: "B. Ridge with alignment features (current baseline)",
    BRANCH_C: "C. Ridge with spatial features",
    BRANCH_D: "D. Ridge with alignment and spatial features",
}

#: (alignment_features, spatial) per branch.
BRANCH_SPEC = {
    BRANCH_A: (False, False),
    BRANCH_B: (True, False),
    BRANCH_C: (False, True),
    BRANCH_D: (True, True),
}


@dataclass
class AblationRun:
    well_results: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    fold_records: list = field(default_factory=list)


def branch_factory(branch: str):
    """A zero-argument Ridge factory configured for ``branch``."""
    if branch not in BRANCH_SPEC:
        raise KeyError(f"unknown ablation branch {branch!r}; known: {list(BRANCH_ORDER)}")
    alignment_features, _ = BRANCH_SPEC[branch]

    def factory(*, spatial=None):
        return RidgeBaseline(spatial=spatial, alignment_features=alignment_features)

    return factory


def branch_uses_spatial(branch: str) -> bool:
    return BRANCH_SPEC[branch][1]


def branch_uses_alignment(branch: str) -> bool:
    return BRANCH_SPEC[branch][0]


def run_ablation_protocol(
    *,
    protocol: str,
    mode: str,
    folds: list[Fold],
    task_builder,
    branches=BRANCH_ORDER,
    spatial_config: SpatialConfig | None = None,
    verbose: bool = False,
    alignment_cache=None,
    cache_context: dict | None = None,
) -> AblationRun:
    """Fit and score all four branches on the same folds, one protocol."""
    run = AblationRun()
    spatial_config = spatial_config or SpatialConfig()
    branches = tuple(branches)

    for fold in folds:
        t0 = time.perf_counter()
        assert_no_blocked_wells(fold.train_ids, context=f"ablation {protocol} fold {fold.index} train")
        assert_no_blocked_wells(fold.valid_ids, context=f"ablation {protocol} fold {fold.index} valid")

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
                f"ablation {protocol} fold {fold.index}: {len(overlap)} well(s) are both "
                f"fitted and scored, e.g. {sorted(overlap)[:5]}."
            )

        prior = None
        if any(branch_uses_spatial(b) for b in branches):
            # Donors are fold-train wells only; the fold's validation wells are
            # excluded here and again by well_id at query time.
            prior = SpatialPrior(spatial_config).fit(train_tasks)
            prior.assert_disjoint(valid_ids)

        models = {}
        for branch in branches:
            factory = branch_factory(branch)
            spatial = prior if branch_uses_spatial(branch) else None
            model = factory(spatial=spatial)
            try:
                model.fit(train_tasks)
            except Exception as exc:
                run.failures.append(
                    {"stage": "fit", "model": branch, "well_id": "", "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            models[branch] = model

        if models:
            run.well_results += evaluate_models(
                models, valid_tasks, protocol, fold.index,
                verbose=verbose, failures=run.failures,
                alignment_cache=alignment_cache,
                cache_context={**(cache_context or {}), "fold": fold.index, "protocol": protocol},
            )

        run.fold_records.append(
            {
                "protocol": protocol,
                "fold": fold.index,
                "n_train_wells": len(train_tasks),
                "n_valid_wells": len(valid_tasks),
                "n_branches_fitted": len(models),
                "seconds": time.perf_counter() - t0,
            }
        )
        if verbose:
            print(
                f"      fold {fold.index}: {len(train_tasks)} train / {len(valid_tasks)} valid, "
                f"{len(models)}/{len(branches)} branches in {time.perf_counter() - t0:.1f}s"
            )
    return run


def _weighted_rmse(group: pd.DataFrame) -> float:
    n = float(group["n_points"].sum())
    return float(np.sqrt(group["sse"].sum() / n)) if n else np.nan


def summarize_ablation(well_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (protocol, branch) with the delta against branch B.

    Only wells scored by *every* branch within a protocol enter the comparison,
    so a branch cannot look better by having quietly dropped a hard well.
    """
    if well_df is None or well_df.empty:
        return pd.DataFrame()
    frame = well_df[well_df["model"].isin(BRANCH_ORDER)].copy()
    if frame.empty:
        return pd.DataFrame()

    rows = []
    for protocol, group in frame.groupby("protocol", sort=False):
        present = [b for b in BRANCH_ORDER if b in set(group["model"])]
        counts = group.groupby("well_id")["model"].nunique()
        common = set(counts[counts == len(present)].index)
        paired = group[group["well_id"].isin(common)]
        base = paired[paired["model"] == BASELINE_BRANCH]
        base_rmse = _weighted_rmse(base) if len(base) else np.nan
        base_median = float(base["rmse"].median()) if len(base) else np.nan
        for branch in present:
            g = paired[paired["model"] == branch]
            if g.empty:
                continue
            global_rmse = _weighted_rmse(g)
            worst = g.nlargest(min(10, len(g)), "rmse")
            rows.append(
                {
                    "protocol": protocol,
                    "branch": branch,
                    "label": BRANCH_LABELS[branch],
                    "alignment_features": branch_uses_alignment(branch),
                    "spatial_features": branch_uses_spatial(branch),
                    "n_wells": int(g["well_id"].nunique()),
                    "n_points": int(g["n_points"].sum()),
                    "global_rmse": global_rmse,
                    "median_well_rmse": float(g["rmse"].median()),
                    "worst10_well_rmse": float(worst["rmse"].mean()),
                    "delta_global_rmse_vs_baseline": global_rmse - base_rmse,
                    "pct_change_vs_baseline": (
                        100.0 * (global_rmse - base_rmse) / base_rmse
                        if np.isfinite(base_rmse) and base_rmse
                        else np.nan
                    ),
                    "delta_median_well_rmse_vs_baseline": float(g["rmse"].median()) - base_median,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {b: i for i, b in enumerate(BRANCH_ORDER)}
    out["_o"] = out["branch"].map(order)
    return out.sort_values(["protocol", "_o"]).drop(columns="_o").reset_index(drop=True)


def alignment_feature_verdict(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-protocol A-vs-B and C-vs-D isolation of the alignment features.

    Two independent contrasts answer the same question: does adding the
    alignment features to Ridge lower RMSE, with and without spatial features?
    A negative delta means the alignment features help.
    """
    if summary is None or summary.empty:
        return pd.DataFrame()
    rows = []
    for protocol, group in summary.groupby("protocol", sort=False):
        by = group.set_index("branch")["global_rmse"].to_dict()
        for label, without, with_ in (
            ("no_spatial", BRANCH_A, BRANCH_B),
            ("with_spatial", BRANCH_C, BRANCH_D),
        ):
            if without not in by or with_ not in by:
                continue
            delta = by[with_] - by[without]
            rows.append(
                {
                    "protocol": protocol,
                    "contrast": f"{with_} - {without}",
                    "spatial_context": label,
                    "global_rmse_without_alignment": by[without],
                    "global_rmse_with_alignment": by[with_],
                    "delta_global_rmse": delta,
                    "alignment_features_help": bool(delta < 0),
                }
            )
    return pd.DataFrame(rows)


#: Pre-registered tolerance for "material degradation" in the secondary
#: per-well metrics, as a fraction of the branch-B value. Fixed before the real
#: results were seen; a branch may not be kept if it inflates median or
#: worst-10 well RMSE by more than this even while global RMSE improves.
MATERIAL_DEGRADATION_TOLERANCE = 0.02


def preregistered_decision(summary: pd.DataFrame) -> pd.DataFrame:
    """Apply the decision rule fixed before the real results were inspected.

    Alignment features (contrast A->B and C->D):
        keep only if global RMSE improves in **both** protocols and neither
        median nor worst-10 well RMSE degrades materially.

    Spatial features (contrast A->C and B->D):
        keep if global RMSE improves, **or** worst-10 well RMSE improves
        consistently across both protocols (the rule explicitly allows a
        worst-well-only justification), and the runtime cost is acceptable.

    Returns one row per (feature group, protocol, contrast) plus the aggregate
    verdict rows, so the decision can be audited rather than trusted.
    """
    empty = pd.DataFrame(
        columns=[
            "feature_group", "protocol", "contrast", "context",
            "global_rmse_without", "global_rmse_with", "delta_global_rmse", "pct_global_rmse",
            "median_well_rmse_without", "median_well_rmse_with", "delta_median_well_rmse",
            "pct_median_well_rmse",
            "worst10_well_rmse_without", "worst10_well_rmse_with", "delta_worst10_well_rmse",
            "pct_worst10_well_rmse",
            "improves_global", "improves_worst10",
            "material_median_degradation", "material_worst10_degradation",
        ]
    )
    if summary is None or summary.empty:
        return empty
    rows = []
    metrics = ("global_rmse", "median_well_rmse", "worst10_well_rmse")
    contrasts = (
        ("alignment", BRANCH_A, BRANCH_B, "no_spatial"),
        ("alignment", BRANCH_C, BRANCH_D, "with_spatial"),
        ("spatial", BRANCH_A, BRANCH_C, "no_alignment"),
        ("spatial", BRANCH_B, BRANCH_D, "with_alignment"),
    )
    for protocol, group in summary.groupby("protocol", sort=False):
        indexed = group.set_index("branch")
        for feature_group, without, with_, context in contrasts:
            if without not in indexed.index or with_ not in indexed.index:
                continue
            row = {
                "feature_group": feature_group,
                "protocol": protocol,
                "contrast": f"{with_} - {without}",
                "context": context,
            }
            for metric in metrics:
                base = float(indexed.loc[without, metric])
                cand = float(indexed.loc[with_, metric])
                row[f"{metric}_without"] = base
                row[f"{metric}_with"] = cand
                row[f"delta_{metric}"] = cand - base
                row[f"pct_{metric}"] = 100.0 * (cand - base) / base if base else np.nan
            row["improves_global"] = bool(row["delta_global_rmse"] < 0)
            row["improves_worst10"] = bool(row["delta_worst10_well_rmse"] < 0)
            # A degradation sitting exactly on the tolerance is *not* material.
            # The relative epsilon stops a decision flipping on ~1e-16 of
            # floating-point noise when the delta lands on the boundary.
            tol = MATERIAL_DEGRADATION_TOLERANCE
            for metric in ("median_well_rmse", "worst10_well_rmse"):
                base = abs(row[f"{metric}_without"])
                budget = tol * base
                row[f"material_{'median' if metric.startswith('median') else 'worst10'}_degradation"] = bool(
                    row[f"delta_{metric}"] > budget + 1e-9 * max(base, 1.0)
                )
            rows.append(row)
    return pd.DataFrame(rows) if rows else empty


def preregistered_verdict(decision: pd.DataFrame) -> dict:
    """Collapse the per-contrast decision table into keep/remove per group."""
    out: dict = {}
    if decision is None or decision.empty:
        for group in ("alignment", "spatial"):
            out[group] = {
                "decision": "undetermined",
                "reason": "no contrasts were computed",
                "protocols_covered": [],
                "n_contrasts": 0,
            }
        return out

    both = {PROTOCOL_A, PROTOCOL_B}
    for group, sub in decision.groupby("feature_group", sort=False):
        protocols = sorted(set(sub["protocol"]))
        covered = both <= set(protocols)
        improves_global_everywhere = bool(sub["improves_global"].all())
        no_material = not bool(
            sub["material_median_degradation"].any() or sub["material_worst10_degradation"].any()
        )
        if group == "alignment":
            keep = covered and improves_global_everywhere and no_material
            if keep:
                reason = (
                    "global RMSE improved in every contrast under both protocols with no "
                    "material median or worst-10 degradation"
                )
            elif not covered:
                reason = "the rule requires both protocols; only " + ", ".join(protocols) + " was covered"
            elif not improves_global_everywhere:
                reason = "global RMSE did not improve in every contrast under both protocols"
            else:
                reason = "global RMSE improved but median or worst-10 well RMSE degraded materially"
        else:
            # Spatial may also be justified by a consistent worst-well gain.
            worst_consistent = bool(sub["improves_worst10"].all())
            keep = covered and (improves_global_everywhere or worst_consistent) and no_material
            if keep and improves_global_everywhere:
                reason = "global RMSE improved in every contrast under both protocols"
            elif keep:
                reason = (
                    "global RMSE did not improve everywhere, but worst-10 well RMSE improved "
                    "consistently across both protocols with no material degradation"
                )
            elif not covered:
                reason = "the rule requires both protocols; only " + ", ".join(protocols) + " was covered"
            else:
                reason = (
                    "neither a consistent global improvement nor a consistent worst-10 "
                    "improvement was observed"
                )
        out[group] = {
            "decision": "keep_as_features" if keep else "remove_from_next_baseline",
            "reason": reason,
            "protocols_covered": protocols,
            "n_contrasts": int(len(sub)),
            "n_improving_global": int(sub["improves_global"].sum()),
            "n_improving_worst10": int(sub["improves_worst10"].sum()),
            "any_material_degradation": not no_material,
        }
    return out


def alignment_feature_recommendation(verdict: pd.DataFrame) -> dict:
    """Turn the contrasts into the keep/remove decision the task asks for.

    Keep the alignment features only if they lower RMSE in **every** computed
    contrast under **both** protocols; a mixed result is not evidence of value.
    """
    if verdict is None or verdict.empty:
        return {
            "decision": "undetermined",
            "reason": "no ablation contrasts were computed",
            "protocols_covered": [],
            "n_contrasts": 0,
            "n_helping": 0,
        }
    helps = verdict["alignment_features_help"].astype(bool)
    protocols = sorted(set(verdict["protocol"]))
    both_protocols = {PROTOCOL_A, PROTOCOL_B} <= set(protocols)
    all_help = bool(helps.all()) and both_protocols
    return {
        "decision": "keep_as_features" if all_help else "remove_from_next_baseline",
        "reason": (
            "alignment features lowered global RMSE in every contrast under both protocols"
            if all_help
            else "alignment features did not lower global RMSE in every contrast under both protocols"
        ),
        "protocols_covered": protocols,
        "n_contrasts": int(len(verdict)),
        "n_helping": int(helps.sum()),
        "max_delta_global_rmse": float(verdict["delta_global_rmse"].max()),
        "min_delta_global_rmse": float(verdict["delta_global_rmse"].min()),
    }
