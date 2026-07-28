"""The six REAL KAGGLE VALIDATION reports for the A/B/C/D Ridge ablation.

Every table is computed from the per-well rows the runner produced. Nothing is
estimated, defaulted, or carried over from another run: if a stage did not
execute, its table says so rather than showing a plausible number.

Protocols are reported separately throughout and are never averaged.

Files written:

    real_alignment_ablation_results.csv    one row per (protocol, branch)
    real_alignment_ablation_summary.md     headline + runtime/memory/cache
    real_alignment_feature_comparison.md   A->B and C->D alignment contrasts
    real_protocol_comparison.md            same_well_masked vs unseen_well
    real_spatial_ablation.md               A->C and B->D spatial contrasts
    real_well_level_ablation.csv           one row per (protocol, branch, well)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ablation import (
    BASELINE_BRANCH,
    BRANCH_LABELS,
    BRANCH_ORDER,
    BRANCH_SPEC,
    preregistered_decision,
    preregistered_verdict,
    summarize_ablation,
)
from src.validation import (
    GR_BINS,
    GR_LABELS,
    PREFIX_BINS,
    PREFIX_LABELS,
    SUFFIX_BINS,
    SUFFIX_LABELS,
)

REAL_BANNER = "REAL KAGGLE VALIDATION"
SYNTHETIC_BANNER = "SYNTHETIC — NOT A COMPETITION RESULT"

#: The audited eligible universe: 773 discovered train wells minus the three
#: visible public test wells. A run that does not match this is not the real
#: full validation and must not be banner-stamped as one.
AUDITED_ELIGIBLE_WELLS = 770
AUDITED_DISCOVERED_WELLS = 773


def is_real_run(environment: dict | None) -> bool:
    """Whether this run may carry the REAL KAGGLE VALIDATION banner.

    Deliberately conservative and evidence-based: the banner is earned by the
    discovered well counts matching the audited real mount, not by a flag the
    caller passes. A synthetic field or a partial mount therefore cannot be
    stamped as real even by accident — which is the whole point of keeping the
    two report families separate.
    """
    env = environment or {}
    return (
        int(env.get("n_train_wells_discovered", -1)) == AUDITED_DISCOVERED_WELLS
        and int(env.get("n_eligible_wells", -1)) == AUDITED_ELIGIBLE_WELLS
    )


def banner_for(environment: dict | None) -> str:
    return REAL_BANNER if is_real_run(environment) else SYNTHETIC_BANNER


def file_prefix(environment: dict | None) -> str:
    """Filename prefix matching the data source.

    A real run emits the exact ``real_*`` filenames the brief requires. Anything
    else emits ``synthetic_*`` — so a file called ``real_...`` on disk is always
    a real result, and a synthetic run cannot leave behind a file whose *name*
    contradicts its banner.
    """
    return "real_" if is_real_run(environment) else "synthetic_"


def banner_block(environment: dict | None) -> str:
    """Header stamped on every file, reflecting the *actual* data source."""
    env = environment or {}
    if is_real_run(env):
        subset = ""
        evaluated = int(env.get("n_wells_evaluated", 0))
        if evaluated and evaluated != AUDITED_ELIGIBLE_WELLS:
            subset = (
                f"> \n> **Subset run:** {evaluated} of {AUDITED_ELIGIBLE_WELLS} eligible wells "
                f"(`--max-wells {env.get('max_wells')}`). Not the full validation.\n"
            )
        return (
            f"> # {REAL_BANNER}\n"
            "> \n"
            "> Computed from the real ROGII competition mount "
            f"({AUDITED_DISCOVERED_WELLS} train wells discovered, "
            f"{AUDITED_ELIGIBLE_WELLS} eligible after excluding the three visible public "
            "test wells). Synthetic harness output lives under "
            "`reports/synthetic_validation/` and `reports/synthetic_ablation/` and is "
            "never mixed with these files.\n" + subset
        )
    return (
        f"> # {SYNTHETIC_BANNER}\n"
        "> \n"
        "> **This is not a competition result.** The discovered well counts do not match "
        f"the audited real mount ({env.get('n_train_wells_discovered', 'unknown')} train "
        f"wells discovered, {env.get('n_eligible_wells', 'unknown')} eligible; the real "
        f"mount has {AUDITED_DISCOVERED_WELLS}/{AUDITED_ELIGIBLE_WELLS}). These files were "
        "produced by the harness against a synthetic or partial field to verify that it "
        "runs, and their numbers must not be quoted as validation results.\n"
    )


def _md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df is None or df.empty:
        return "_No computed rows were available._\n"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.{digits}f}")
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


def branch_metrics(well: pd.DataFrame, runtime: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full per-(protocol, branch) metric table required by the brief.

    Scored point counts are computed **within** a (protocol, branch) group, so
    a point is never double-counted across branches: each branch predicts the
    same rows, and summing across branches would multiply the total by four.
    """
    if well is None or well.empty:
        return pd.DataFrame()
    frame = well[well["model"].isin(BRANCH_ORDER)].copy()
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
        for branch in present:
            g = paired[paired["model"] == branch]
            if g.empty:
                continue
            worst = g.nlargest(min(10, len(g)), "rmse")
            global_rmse = _weighted_rmse(g)
            row = {
                "protocol": protocol,
                "branch": branch,
                "label": BRANCH_LABELS[branch],
                "alignment_features": BRANCH_SPEC[branch][0],
                "spatial_features": BRANCH_SPEC[branch][1],
                "n_wells_evaluated": int(g["well_id"].nunique()),
                "n_points_evaluated": int(g["n_points"].sum()),
                "global_rmse": global_rmse,
                "mean_well_rmse": float(g["rmse"].mean()),
                "median_well_rmse": float(g["rmse"].median()),
                "p90_well_rmse": float(g["rmse"].quantile(0.90)),
                "worst10_well_rmse": float(worst["rmse"].mean()),
                "worst_well_rmse": float(g["rmse"].max()),
                "worst_well_id": str(g.loc[g["rmse"].idxmax(), "well_id"]),
                "max_abs_error": float(g["max_abs_error"].max()),
                "mean_bias": float(g["bias"].mean()),
                "predict_seconds": float(g["predict_seconds"].sum()),
                "delta_global_rmse_vs_baseline": global_rmse - base_rmse,
                "pct_change_vs_baseline": (
                    100.0 * (global_rmse - base_rmse) / base_rmse
                    if np.isfinite(base_rmse) and base_rmse
                    else np.nan
                ),
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if runtime is not None and not runtime.empty:
        out = out.merge(runtime, on=["protocol", "branch"], how="left")
    order = {b: i for i, b in enumerate(BRANCH_ORDER)}
    out["_o"] = out["branch"].map(order)
    return out.sort_values(["protocol", "_o"]).drop(columns="_o").reset_index(drop=True)


def _stratum(well: pd.DataFrame, column: str, bins, labels, name: str) -> pd.DataFrame:
    if well is None or well.empty or column not in well.columns:
        return pd.DataFrame()
    local = well[well["model"].isin(BRANCH_ORDER)].copy()
    if local.empty:
        return pd.DataFrame()
    local["stratum"] = pd.cut(local[column], bins=bins, labels=labels)
    rows = []
    for (protocol, branch, stratum), g in local.groupby(
        ["protocol", "model", "stratum"], observed=True, sort=False
    ):
        n = float(g["n_points"].sum())
        if not n:
            continue
        rows.append(
            {
                "protocol": protocol,
                "branch": branch,
                "stratify_by": name,
                "stratum": str(stratum),
                "n_wells": int(g["well_id"].nunique()),
                "n_points": int(n),
                "global_rmse": float(np.sqrt(g["sse"].sum() / n)),
                "median_well_rmse": float(g["rmse"].median()),
                "worst_well_rmse": float(g["rmse"].max()),
            }
        )
    return pd.DataFrame(rows)


def stratified_ablation(well: pd.DataFrame) -> pd.DataFrame:
    """RMSE by GR missingness, hidden suffix length and prefix length."""
    parts = [
        _stratum(well, "gr_missing_frac", GR_BINS, GR_LABELS, "gr_missingness"),
        _stratum(well, "suffix_len", SUFFIX_BINS, SUFFIX_LABELS, "hidden_suffix_length"),
        _stratum(well, "prefix_len", PREFIX_BINS, PREFIX_LABELS, "prefix_length"),
    ]
    parts = [p for p in parts if not p.empty]
    if parts:
        return pd.concat(parts, ignore_index=True)
    # Preserve the schema so an empty run still writes a parseable CSV.
    return pd.DataFrame(
        columns=[
            "protocol", "branch", "stratify_by", "stratum",
            "n_wells", "n_points", "global_rmse", "median_well_rmse", "worst_well_rmse",
        ]
    )


def protocol_comparison(well: pd.DataFrame) -> pd.DataFrame:
    """Descriptive per-protocol table. No metric is averaged across protocols."""
    if well is None or well.empty:
        return pd.DataFrame()
    frame = well[well["model"] == BASELINE_BRANCH].copy()
    if frame.empty:
        frame = well[well["model"].isin(BRANCH_ORDER)].copy()
    rows = []
    for protocol, g in frame.groupby("protocol", sort=False):
        n = float(g["n_points"].sum())
        rows.append(
            {
                "protocol": protocol,
                "reference_branch": BASELINE_BRANCH,
                "n_wells": int(g["well_id"].nunique()),
                "n_scored_points": int(n),
                "prefix_min": int(g["prefix_len"].min()),
                "prefix_median": float(g["prefix_len"].median()),
                "suffix_min": int(g["suffix_len"].min()),
                "suffix_median": float(g["suffix_len"].median()),
                "gr_missing_median": float(g["gr_missing_frac"].median()),
                "global_rmse": float(np.sqrt(g["sse"].sum() / n)) if n else np.nan,
                "median_well_rmse": float(g["rmse"].median()),
                "p90_well_rmse": float(g["rmse"].quantile(0.90)),
                "worst10_well_rmse": float(g.nlargest(min(10, len(g)), "rmse")["rmse"].mean()),
                "scored_exact_suffix_all": bool(g.get("scored_exact_suffix", pd.Series(dtype=bool)).all())
                if "scored_exact_suffix" in g
                else None,
            }
        )
    return pd.DataFrame(rows)


def _decision_section(decision: pd.DataFrame, group: str) -> pd.DataFrame:
    if decision is None or decision.empty:
        return pd.DataFrame()
    sub = decision[decision["feature_group"] == group]
    cols = [
        "protocol", "contrast", "context",
        "global_rmse_without", "global_rmse_with", "delta_global_rmse", "pct_global_rmse",
        "median_well_rmse_without", "median_well_rmse_with", "delta_median_well_rmse",
        "worst10_well_rmse_without", "worst10_well_rmse_with", "delta_worst10_well_rmse",
        "improves_global", "improves_worst10",
        "material_median_degradation", "material_worst10_degradation",
    ]
    return sub[[c for c in cols if c in sub.columns]].reset_index(drop=True)


def write_real_ablation_reports(
    reports_dir,
    well: pd.DataFrame,
    *,
    environment: dict | None = None,
    failures: pd.DataFrame | None = None,
    runtime: pd.DataFrame | None = None,
) -> list[Path]:
    """Write all six REAL KAGGLE VALIDATION artifacts."""
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    env = environment or {}
    # The banner is derived from the observed well counts, never passed in, so
    # a synthetic or partial run cannot be labelled as real validation.
    banner = banner_block(env)
    banner_label = banner_for(env)
    prefix = file_prefix(env)

    metrics = branch_metrics(well, runtime=runtime)
    summary = summarize_ablation(well)
    decision = preregistered_decision(summary)
    verdict = preregistered_verdict(decision)
    strat = stratified_ablation(well)
    protocols = protocol_comparison(well)

    n_failures = int(len(failures)) if failures is not None else 0
    failure_note = (
        f"{n_failures} task/fit/predict failures were recorded"
        if n_failures
        else "No task, fit or predict failure was recorded"
    )

    written: list[Path] = []

    # -- 1. results csv -----------------------------------------------------
    results_csv = root / f"{prefix}alignment_ablation_results.csv"
    enriched = metrics.copy()
    env_columns = ("runtime_seconds", "peak_rss_mb", "cache_hits", "cache_misses", "cache_writes")
    if enriched.empty:
        # An empty run must still produce a *parseable* file with the real
        # schema. Writing a header-less empty CSV would make the absence of
        # results indistinguishable from a corrupt file downstream.
        enriched = pd.DataFrame(
            columns=[
                "validation", "protocol", "branch", "label",
                "alignment_features", "spatial_features",
                "n_wells_evaluated", "n_points_evaluated",
                "global_rmse", "mean_well_rmse", "median_well_rmse", "p90_well_rmse",
                "worst10_well_rmse", "worst_well_rmse", "worst_well_id",
                "max_abs_error", "mean_bias", "predict_seconds",
                "delta_global_rmse_vs_baseline", "pct_change_vs_baseline",
                *env_columns, "failure_count",
            ]
        )
    else:
        enriched.insert(0, "validation", banner_label)
        for key in env_columns:
            if key in env:
                enriched[key] = env[key]
        enriched["failure_count"] = n_failures
    enriched.to_csv(results_csv, index=False)
    written.append(results_csv)

    # -- 2. well-level csv --------------------------------------------------
    well_csv = root / f"{prefix}well_level_ablation.csv"
    well_out = (
        well[well["model"].isin(BRANCH_ORDER)].copy()
        if well is not None and not well.empty
        else pd.DataFrame(columns=["validation", "protocol", "branch", "well_id"])
    )
    if "model" in well_out.columns:
        well_out.insert(0, "validation", banner_label)
        well_out = well_out.rename(columns={"model": "branch"})
    well_out.to_csv(well_csv, index=False)
    written.append(well_csv)

    strat_csv = root / f"{prefix}ablation_stratified.csv"
    strat.to_csv(strat_csv, index=False)
    written.append(strat_csv)

    # -- 3. headline summary ------------------------------------------------
    env_rows = pd.DataFrame(
        [{"key": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else str(v)}
         for k, v in env.items()]
    )
    display_cols = [
        "protocol", "branch", "n_wells_evaluated", "n_points_evaluated",
        "global_rmse", "mean_well_rmse", "median_well_rmse", "p90_well_rmse",
        "worst10_well_rmse", "worst_well_rmse", "worst_well_id",
        "delta_global_rmse_vs_baseline", "pct_change_vs_baseline",
    ]
    if "fit_seconds" in metrics.columns:
        display_cols += ["fit_seconds"]
    display_cols += ["predict_seconds"]
    display = metrics[[c for c in display_cols if c in metrics.columns]] if not metrics.empty else metrics
    lines = [
        banner,
        "# Real alignment / spatial ablation — summary\n",
        "Four Ridge configurations, cross-fitted by well ID, both protocols reported "
        "separately and **never averaged**. Branch B is the current, unmodified Ridge "
        "baseline and every delta is taken against it.\n",
        "| branch | alignment features | spatial features |\n|---|---|---|\n"
        + "\n".join(
            f"| {BRANCH_LABELS[b]} | {'yes' if BRANCH_SPEC[b][0] else 'no'} | "
            f"{'yes' if BRANCH_SPEC[b][1] else 'no'} |"
            for b in BRANCH_ORDER
        )
        + "\n",
        "## Headline metrics\n",
        _md_table(display),
        "Only wells scored by every branch within a protocol enter the comparison, so a "
        "branch cannot look better by having dropped a hard well. Point counts are "
        "per (protocol, branch) and are not summed across branches.\n",
        f"## Failures\n\n{failure_note}.\n",
        "## Run environment\n",
        _md_table(env_rows) if not env_rows.empty else "_Not recorded._\n",
        "## Stratified RMSE\n",
        "### By GR missingness\n",
        _md_table(strat[strat["stratify_by"] == "gr_missingness"].drop(columns="stratify_by")
                  if not strat.empty else strat),
        "### By hidden suffix length\n",
        _md_table(strat[strat["stratify_by"] == "hidden_suffix_length"].drop(columns="stratify_by")
                  if not strat.empty else strat),
        "### By prefix length\n",
        _md_table(strat[strat["stratify_by"] == "prefix_length"].drop(columns="stratify_by")
                  if not strat.empty else strat),
    ]
    summary_md = root / f"{prefix}alignment_ablation_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    written.append(summary_md)

    # -- 4. alignment feature comparison ------------------------------------
    align = _decision_section(decision, "alignment")
    align_verdict = verdict.get("alignment", {})
    lines = [
        banner,
        "# Real alignment-feature comparison\n",
        "Two paired contrasts isolate the four GR/typewell alignment features "
        "(`align_tvt`, `align_score`, `align_shift`, `align_gradient`), holding the "
        "spatial setting fixed:\n",
        "- **A → B** adds the alignment features without spatial features.\n"
        "- **C → D** adds the alignment features with spatial features.\n",
        "A negative `delta_global_rmse` means the alignment features lowered RMSE.\n",
        "## Contrasts\n",
        _md_table(align),
        "## Pre-registered decision rule\n",
        "Keep the alignment features in the next baseline **only if** they improve "
        "global RMSE in **both** protocols **and** do not materially degrade median or "
        "worst-10 well RMSE (tolerance: 2% of the branch-B value). Otherwise remove "
        "them. This rule was fixed before the real results were inspected.\n",
        "## Decision\n",
        f"**{align_verdict.get('decision', 'undetermined').replace('_', ' ').upper()}** — "
        f"{align_verdict.get('reason', 'no contrasts were computed')}.\n",
        f"Contrasts computed: {align_verdict.get('n_contrasts', 0)}; "
        f"improving global RMSE: {align_verdict.get('n_improving_global', 0)}; "
        f"protocols covered: {', '.join(align_verdict.get('protocols_covered', [])) or 'none'}.\n",
    ]
    align_md = root / f"{prefix}alignment_feature_comparison.md"
    align_md.write_text("\n".join(lines), encoding="utf-8")
    written.append(align_md)

    # -- 5. spatial ablation ------------------------------------------------
    spatial = _decision_section(decision, "spatial")
    spatial_verdict = verdict.get("spatial", {})
    lines = [
        banner,
        "# Real spatial-feature ablation\n",
        "Two paired contrasts isolate the offset-well spatial features, holding the "
        "alignment setting fixed:\n",
        "- **A → C** adds spatial features without alignment features.\n"
        "- **B → D** adds spatial features with alignment features.\n",
        "Donors are fold-train wells only, rebuilt inside every fold; the queried well "
        "is excluded from its own neighbour set by well ID at query time, and a "
        "fold-level `assert_disjoint` guard refuses to run if any validation well "
        "could donate.\n",
        "## Contrasts\n",
        _md_table(spatial),
        "## Pre-registered decision rule\n",
        "Keep the spatial features **only if** they improve the global metric **or** "
        "give a consistent worst-well improvement across both protocols, without "
        "unacceptable runtime or leakage risk. Otherwise remove them.\n",
        "## Decision\n",
        f"**{spatial_verdict.get('decision', 'undetermined').replace('_', ' ').upper()}** — "
        f"{spatial_verdict.get('reason', 'no contrasts were computed')}.\n",
        f"Contrasts computed: {spatial_verdict.get('n_contrasts', 0)}; "
        f"improving global RMSE: {spatial_verdict.get('n_improving_global', 0)}; "
        f"improving worst-10: {spatial_verdict.get('n_improving_worst10', 0)}.\n",
    ]
    spatial_md = root / f"{prefix}spatial_ablation.md"
    spatial_md.write_text("\n".join(lines), encoding="utf-8")
    written.append(spatial_md)

    # -- 6. protocol comparison ---------------------------------------------
    lines = [
        banner,
        "# Real protocol comparison\n",
        "`same_well_masked` and `unseen_well` are reported separately and are **never "
        "averaged**. No metric on this page is averaged across the two protocols, and no combined ranking is "
        "produced: the protocols answer different questions and their scored rows come "
        "from different sources (`TVT_input` for the masked boundary, the `TVT` label "
        "for the real hidden suffix).\n",
        "## Per-protocol support and reference-branch metrics\n",
        _md_table(protocols),
        "## Per-branch global RMSE, by protocol\n",
        _md_table(
            metrics.pivot(index="branch", columns="protocol", values="global_rmse").reset_index()
            if not metrics.empty
            else metrics
        ),
        "A branch may rank differently under the two protocols. That is a real finding "
        "about the protocols, not a tie to be broken by averaging.\n",
    ]
    protocol_md = root / f"{prefix}protocol_comparison.md"
    protocol_md.write_text("\n".join(lines), encoding="utf-8")
    written.append(protocol_md)

    decision_csv = root / f"{prefix}ablation_decision.csv"
    decision.to_csv(decision_csv, index=False)
    written.append(decision_csv)
    return written
