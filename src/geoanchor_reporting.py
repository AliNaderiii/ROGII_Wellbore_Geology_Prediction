"""Reporting for the controlled GeoAnchor experiment (arms A–E).

Every number here is computed from the run's well-level results; nothing is
carried forward or assumed. Report *naming* follows the repository's
evidence-based convention: real competition mounts (773 discovered / 770
eligible wells) produce ``real_geoanchor_*`` files, everything else produces
``synthetic_geoanchor_*`` files stamped SYNTHETIC — NOT A COMPETITION RESULT.

Metrics reported per arm and protocol (never averaged across protocols):

* global point-level RMSE, mean/median well RMSE, P90, worst-10 RMSE,
  worst single well, max abs error, bias;
* fold stability: per-fold pooled RMSE and delta-vs-default sign consistency;
* per-well improved/degraded/unchanged counts relative to Ridge Default;
* well-cluster bootstrap confidence intervals for the delta vs Ridge Default
  (global RMSE delta and mean well-level delta);
* GR missingness and hidden-suffix-length stratifications;
* gate activation/fallback rates (arm E), including per-reason fallback;
* runtime seconds and peak RSS memory.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.geoanchor import ARM_A, ARM_E, ARM_LABELS, ARM_ORDER
from src.pf_beam_robustness import (
    bootstrap_global_rmse_delta,
    bootstrap_paired_well_delta,
    fold_deltas,
    pair_default_vs_candidate,
)
from src.real_ablation_reporting import banner_block, file_prefix, is_real_run
from src.validation import stratified_report, summarize

#: Pre-registered tolerance for "material degradation" in secondary metrics,
#: matching the repository's existing 2% ablation rule.
MATERIAL_DEGRADATION_TOLERANCE = 0.02


# --------------------------------------------------------------------------
# Core tables
# --------------------------------------------------------------------------


def arm_summary_table(well_df: pd.DataFrame) -> pd.DataFrame:
    """Global / mean / median / worst-10 metrics per (arm, protocol)+labels."""
    summary = summarize(well_df)
    if summary.empty:
        return summary
    summary = summary[summary["model"].isin(ARM_ORDER)].copy()
    summary["arm_label"] = summary["model"].map(ARM_LABELS)
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    summary["_o"] = summary["model"].map(order)
    return summary.sort_values(["protocol", "_o"]).drop(columns="_o").reset_index(drop=True)


def paired_tables(well_df: pd.DataFrame) -> pd.DataFrame:
    """Per-well paired deltas of every candidate arm against Ridge Default."""
    frames = []
    for arm in ARM_ORDER:
        if arm == ARM_A:
            continue
        paired = pair_default_vs_candidate(well_df, default=ARM_A, candidate=arm)
        if paired is None or paired.empty:
            continue
        paired = paired.copy()
        paired["candidate_arm"] = arm
        frames.append(paired)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def improved_degraded_counts(paired: pd.DataFrame) -> pd.DataFrame:
    """Per-arm per-protocol counts improved/degraded/unchanged vs default."""
    if paired is None or paired.empty:
        return pd.DataFrame()
    rows = []
    for (protocol, arm), g in paired.groupby(["protocol", "candidate_arm"], sort=False):
        rows.append(
            {
                "protocol": protocol,
                "candidate_arm": arm,
                "n_paired_wells": int(len(g)),
                "n_improved": int(g["improved"].sum()),
                "n_degraded": int(g["degraded"].sum()),
                "n_unchanged": int(g["unchanged"].sum()),
                "frac_improved": float(g["improved"].mean()),
            }
        )
    return pd.DataFrame(rows)


def fold_stability_table(well_df: pd.DataFrame) -> pd.DataFrame:
    """Per-fold pooled RMSE per arm plus stability flags for the delta."""
    frames = []
    for arm in ARM_ORDER:
        if arm == ARM_A:
            continue
        fd = fold_deltas(well_df, default=ARM_A, candidate=arm)
        if fd is None or fd.empty:
            continue
        fd = fd.copy()
        fd["candidate_arm"] = arm
        frames.append(fd)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bootstrap_ci_table(paired: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Well-cluster bootstrap CIs for the delta vs Ridge Default."""
    rows = []
    for arm, g in paired.groupby("candidate_arm", sort=False):
        for fn in (bootstrap_global_rmse_delta, bootstrap_paired_well_delta):
            ci = fn(g, n_boot=n_boot, seed=seed)
            if ci is None or ci.empty:
                continue
            ci = ci.copy()
            ci.insert(1, "candidate_arm", arm)
            rows.append(ci)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def stratified_table(well_df: pd.DataFrame) -> pd.DataFrame:
    """RMSE by hidden suffix length and GR missingness."""
    full = stratified_report(well_df[well_df["model"].isin(ARM_ORDER)])
    if full.empty:
        return full
    keep = {"hidden_suffix_length", "gr_missingness"}
    return full[full["stratify_by"].isin(keep)].reset_index(drop=True)


# --------------------------------------------------------------------------
# Gate tables
# --------------------------------------------------------------------------


def gate_stats_table(gate_logs: pd.DataFrame, gate_infos: pd.DataFrame) -> pd.DataFrame:
    """Gate activation/fallback rates plus training diagnostics per fold."""
    rows = []
    if gate_logs is not None and not gate_logs.empty:
        for (protocol, fold), g in gate_logs.groupby(["protocol", "fold"], sort=False):
            applied = g["outcome"].str.startswith("applied")
            rows.append(
                {
                    "protocol": protocol,
                    "fold": int(fold),
                    "n_wells": int(len(g)),
                    "n_activated": int(applied.sum()),
                    "activation_rate": float(applied.mean()),
                    "fallback_rate": float((~applied).mean()),
                    "gate_killed": bool(g["gate_killed"].any()),
                    "n_fallback_pseudo_holdout_not_improved": int(
                        g["reason"].str.contains("pseudo_holdout_not_improved").sum()
                    ),
                    "n_fallback_tail_risk": int(
                        g["reason"].str.contains("worst_tail_risk_increased").sum()
                    ),
                    "n_fallback_low_confidence": int(
                        g["reason"].str.contains("alignment_confidence_below_threshold").sum()
                    ),
                    "n_fallback_disagreement": int(
                        g["reason"].str.contains("branch_disagreement_unacceptable").sum()
                    ),
                    "n_fallback_below_margin": int(
                        g["reason"].str.contains("gbdt_expected_gain_below_margin").sum()
                    ),
                    "n_fallback_candidate_unavailable": int(
                        g["reason"].str.contains("candidate_unavailable").sum()
                    ),
                    "n_applied_pf": int((g["candidate"] == "pf").sum()),
                    "n_applied_beam": int((g["candidate"] == "beam").sum()),
                    "n_applied_pf_beam_mean": int((g["candidate"] == "pf_beam_mean").sum()),
                }
            )
    stats = pd.DataFrame(rows)
    if gate_infos is not None and not gate_infos.empty:
        info_cols = [
            "protocol", "fold", "n_oof_wells", "n_examples", "n_pseudo_skipped",
            "killed", "kill_reason", "margin", "conf_thr", "sep_cap",
            "pooled_oof_delta", "oof_activation_rate", "fit_seconds",
        ]
        infos = gate_infos[[c for c in info_cols if c in gate_infos.columns]].copy()
        if stats.empty:
            return infos
        return stats.merge(infos, on=["protocol", "fold"], how="outer").sort_values(
            ["protocol", "fold"]
        ).reset_index(drop=True)
    return stats


def gate_protocol_totals(gate_logs: pd.DataFrame) -> pd.DataFrame:
    """Protocol-level activation/fallback totals for the decision document."""
    if gate_logs is None or gate_logs.empty:
        return pd.DataFrame()
    rows = []
    for protocol, g in gate_logs.groupby("protocol", sort=False):
        applied = g["outcome"].str.startswith("applied")
        reasons = g.loc[~applied, "reason"].str.split(";").explode()
        reasons = reasons[reasons.astype(str).str.len() > 0]
        top = reasons.value_counts().head(8)
        rows.append(
            {
                "protocol": protocol,
                "n_wells": int(len(g)),
                "n_activated": int(applied.sum()),
                "activation_rate": float(applied.mean()),
                "fallback_rate": float((~applied).mean()),
                "mean_predicted_improvement_activated": float(
                    g.loc[applied, "predicted_improvement"].mean()
                ) if applied.any() else np.nan,
                "top_fallback_reasons": "; ".join(
                    f"{name} ({count})" for name, count in top.items()
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pre-registered decision rule
# --------------------------------------------------------------------------


def decision_table(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    fold_stability: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    gate_logs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply the pre-registered promotion rule arm by arm.

    An arm is CARRIED (candidate for a future *real-mount* confirmation) only
    when, versus Ridge Default, under **both** protocols:

    1. global point-level RMSE improves;
    2. the well-cluster bootstrap CI for the global delta is **not strongly
       against** the arm (its 2.5% bound lies below zero, i.e. the arm is not
       confidently worse — the repository's established PF/Beam criterion);
    3. neither median nor worst-10 well RMSE degrades materially (2%);
    4. the delta is fold-stable (candidate not worse on every fold);
    5. (arm E only) the gate activated on at least one well and activated on
       at most half of the wells (a gate that fires everywhere is not a gate).

    Ridge Default remains the fallback in every case: a CARRIED arm is not a
    new default and never overrides ``src.model_status``.
    """
    rows = []
    if summary is None or summary.empty:
        return pd.DataFrame()
    for protocol, group in summary.groupby("protocol", sort=False):
        indexed = group.set_index("model")
        if ARM_A not in indexed.index:
            continue
        for arm in ARM_ORDER:
            if arm == ARM_A or arm not in indexed.index:
                continue
            base = indexed.loc[ARM_A]
            cand = indexed.loc[arm]
            delta_global = float(cand["global_rmse"] - base["global_rmse"])
            improves_global = bool(delta_global < 0)
            med_budget = MATERIAL_DEGRADATION_TOLERANCE * abs(float(base["median_well_rmse"]))
            w10_budget = MATERIAL_DEGRADATION_TOLERANCE * abs(float(base["worst10_well_rmse"]))
            delta_med = float(cand["median_well_rmse"] - base["median_well_rmse"])
            delta_w10 = float(cand["worst10_well_rmse"] - base["worst10_well_rmse"])
            no_material = bool(
                delta_med <= med_budget + 1e-9 and delta_w10 <= w10_budget + 1e-9
            )
            ci_ok = None
            if bootstrap_ci is not None and not bootstrap_ci.empty:
                sub = bootstrap_ci[
                    (bootstrap_ci["candidate_arm"] == arm)
                    & (bootstrap_ci["protocol"] == protocol)
                    & (bootstrap_ci["metric"] == "global_point_rmse_delta")
                ]
                if not sub.empty:
                    ci_ok = bool(float(sub.iloc[0]["ci_low_2.5"]) < 0.0)
            fold_ok = None
            if fold_stability is not None and not fold_stability.empty:
                sub = fold_stability[
                    (fold_stability["candidate_arm"] == arm)
                    & (fold_stability["protocol"] == protocol)
                ]
                if not sub.empty:
                    fold_ok = bool(sub.iloc[0]["stable_across_folds"])
            rows.append(
                {
                    "protocol": protocol,
                    "candidate_arm": arm,
                    "delta_global_rmse": delta_global,
                    "delta_median_well_rmse": delta_med,
                    "delta_worst10_well_rmse": delta_w10,
                    "improves_global": improves_global,
                    "bootstrap_ci_not_against": ci_ok,
                    "no_material_degradation": no_material,
                    "fold_stable": fold_ok,
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    verdicts = []
    for arm, g in table.groupby("candidate_arm", sort=False):
        both = set(g["protocol"]) == {"same_well_masked", "unseen_well"}
        criteria = [
            bool(g["improves_global"].all()),
            bool(g["bootstrap_ci_not_against"].fillna(False).all()),
            bool(g["no_material_degradation"].all()),
            bool(g["fold_stable"].fillna(False).all()),
        ]
        verdict = bool(both and all(criteria))
        if arm == ARM_E and gate_logs is not None and not gate_logs.empty:
            act = []
            for _p, gl in gate_logs.groupby("protocol", sort=False):
                rate = float(gl["outcome"].str.startswith("applied").mean())
                act.append(0.0 < rate <= 0.5)
            verdict = verdict and bool(act and all(act))
        verdicts.append(
            {
                "candidate_arm": arm,
                "verdict": "CARRIED_for_real_mount_confirmation" if verdict else "NOT_CARRIED",
                "both_protocols_covered": both,
                "improves_global_both": bool(g["improves_global"].all()),
                "bootstrap_ok_both": bool(g["bootstrap_ci_not_against"].fillna(False).all()),
                "no_material_degradation_both": bool(g["no_material_degradation"].all()),
                "fold_stable_both": bool(g["fold_stable"].fillna(False).all()),
            }
        )
    verdict_df = pd.DataFrame(verdicts)
    return table.merge(verdict_df, on="candidate_arm", how="left")


# --------------------------------------------------------------------------
# Markdown decision document
# --------------------------------------------------------------------------


def _md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df is None or df.empty:
        return "_No rows were computed._\n"
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


def render_decision_md(
    *,
    environment: dict,
    summary: pd.DataFrame,
    decision: pd.DataFrame,
    fold_stability: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    counts: pd.DataFrame,
    gate_totals: pd.DataFrame,
    gate_stats: pd.DataFrame,
    failures: pd.DataFrame,
    runtime: dict,
) -> str:
    banner = banner_block(environment)
    lines = [
        f"# GeoAnchor Controlled Experiment — Decision",
        "",
        banner,
        "",
        "Pre-registered in `reports/geoanchor_experiment.md`. Arms A–D vary only the",
        "Ridge feature set; arm E adds the well-level GBDT gate over bounded PF/Beam",
        "candidate corrections. Ridge Default is the anchor *and* the fallback of arm",
        "E itself. **No final submission was created by this experiment.**",
        "",
        "## Arm metrics",
        "",
        _md_table(
            summary[
                [
                    "protocol", "model", "n_wells", "n_points", "global_rmse",
                    "mean_well_rmse", "median_well_rmse", "p90_well_rmse",
                    "worst10_well_rmse", "worst_well_rmse", "max_abs_error",
                    "mean_bias", "predict_seconds",
                ]
            ]
            if summary is not None and not summary.empty
            else summary
        ),
        "## Pre-registered decision",
        "",
        _md_table(decision),
        "## Fold stability (delta vs Ridge Default; negative favours the arm)",
        "",
        _md_table(fold_stability),
        "## Bootstrap confidence intervals (well-cluster resampling)",
        "",
        _md_table(
            bootstrap_ci[
                [
                    "protocol", "candidate_arm", "metric", "n_wells",
                    "observed_delta", "ci_low_2.5", "ci_high_97.5",
                    "ci_excludes_zero", "frac_bootstrap_negative",
                ]
            ]
            if bootstrap_ci is not None and not bootstrap_ci.empty
            else bootstrap_ci
        ),
        "## Per-well improved / degraded counts vs Ridge Default",
        "",
        _md_table(counts),
        "## Gate behaviour (arm E)",
        "",
        _md_table(gate_totals),
        "",
        _md_table(gate_stats),
        "## Runtime and memory",
        "",
        _md_table(
            pd.DataFrame(
                [
                    {
                        "protocol_or_total": k,
                        "seconds": v["seconds"],
                        "peak_rss_mb": v.get("peak_rss_mb", np.nan),
                    }
                    for k, v in runtime.items()
                ]
            )
        ),
        "## Failures",
        "",
        (
            f"{len(failures)} task/fit/predict failures were recorded."
            if failures is not None and not failures.empty
            else "No task, fit or prediction failures were recorded."
        ),
        "",
        "## Honesty notes",
        "",
        "- Every number above was computed in this run from well-level results; nothing",
        "  was copied from another run or from public-leaderboard information.",
        "- CARRIED is not promotion: it only permits a confirmation run on the real",
        "  competition mount. Ridge Default remains the fallback in all cases.",
        "- The three visible public test wells were excluded from every fold, fit,",
        "  gate-training set and table by the hard guard in `src.validation`.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# File writing
# --------------------------------------------------------------------------


def report_paths(reports_dir: Path, environment: dict) -> dict[str, Path]:
    prefix = file_prefix(environment)
    base = Path(reports_dir)
    return {
        "summary": base / f"{prefix}geoanchor_arm_summary.csv",
        "well_level": base / f"{prefix}geoanchor_well_level.csv",
        "paired": base / f"{prefix}geoanchor_paired_well_deltas.csv",
        "counts": base / f"{prefix}geoanchor_improved_degraded_counts.csv",
        "fold_stability": base / f"{prefix}geoanchor_fold_stability.csv",
        "bootstrap": base / f"{prefix}geoanchor_bootstrap_ci.csv",
        "stratified": base / f"{prefix}geoanchor_stratified.csv",
        "gate_stats": base / f"{prefix}geoanchor_gate_stats.csv",
        "gate_wells": base / f"{prefix}geoanchor_gate_well_decisions.csv",
        "decision_md": base / f"{prefix}geoanchor_decision.md",
        "environment": base / f"{prefix}geoanchor_run_environment.json",
        "failures": base / f"{prefix}geoanchor_failures.csv",
    }


def write_reports(
    *,
    reports_dir: Path,
    environment: dict,
    well_results,
    fold_records,
    failures,
    gate_logs,
    gate_infos,
    protocol_seconds: dict,
    peak_rss_mb: float,
    n_boot: int = 2000,
) -> dict[str, Path]:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = report_paths(reports_dir, environment)

    well_df = pd.DataFrame([vars(r) for r in well_results]) if well_results else pd.DataFrame()
    fold_df = pd.DataFrame(fold_records)
    fail_df = pd.DataFrame(failures)
    gate_log_df = pd.DataFrame(gate_logs)
    gate_info_df = pd.DataFrame(gate_infos)

    summary = arm_summary_table(well_df) if not well_df.empty else pd.DataFrame()
    paired = paired_tables(well_df) if not well_df.empty else pd.DataFrame()
    counts = improved_degraded_counts(paired)
    folds_t = fold_stability_table(well_df) if not well_df.empty else pd.DataFrame()
    ci = bootstrap_ci_table(paired, n_boot=n_boot) if not paired.empty else pd.DataFrame()
    strat = stratified_table(well_df) if not well_df.empty else pd.DataFrame()
    gstats = gate_stats_table(gate_log_df, gate_info_df)
    gtotals = gate_protocol_totals(gate_log_df)
    decision = decision_table(summary, paired, folds_t, ci, gate_log_df)

    runtime = {
        **{
            p: {"seconds": round(float(s), 3), "peak_rss_mb": round(float(peak_rss_mb), 1)}
            for p, s in protocol_seconds.items()
        },
        "total": {
            "seconds": round(float(sum(protocol_seconds.values())), 3),
            "peak_rss_mb": round(float(peak_rss_mb), 1),
        },
    }

    summary.to_csv(paths["summary"], index=False)
    well_df.to_csv(paths["well_level"], index=False)
    paired.to_csv(paths["paired"], index=False)
    counts.to_csv(paths["counts"], index=False)
    folds_t.to_csv(paths["fold_stability"], index=False)
    ci.to_csv(paths["bootstrap"], index=False)
    strat.to_csv(paths["stratified"], index=False)
    gstats.to_csv(paths["gate_stats"], index=False)
    gate_log_df.to_csv(paths["gate_wells"], index=False)
    fail_df.to_csv(paths["failures"], index=False)

    decision_md = render_decision_md(
        environment=environment,
        summary=summary,
        decision=decision,
        fold_stability=folds_t,
        bootstrap_ci=ci,
        counts=counts,
        gate_totals=gtotals,
        gate_stats=gstats,
        failures=fail_df,
        runtime=runtime,
    )
    paths["decision_md"].write_text(decision_md)

    env = dict(environment)
    env.update(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runtime_seconds_total": runtime["total"]["seconds"],
            "peak_rss_mb": runtime["total"]["peak_rss_mb"],
            "n_bootstrap": int(n_boot),
            "is_real_run": is_real_run(environment),
            "fold_records": fold_df.to_dict(orient="records"),
            "generated_unix": int(time.time()),
        }
    )
    paths["environment"].write_text(json.dumps(env, indent=2, default=str))
    paths["tables"] = {
        "summary": summary,
        "paired": paired,
        "counts": counts,
        "fold_stability": folds_t,
        "bootstrap_ci": ci,
        "stratified": strat,
        "gate_stats": gstats,
        "gate_totals": gtotals,
        "decision": decision,
        "well_level": well_df,
    }
    return paths
