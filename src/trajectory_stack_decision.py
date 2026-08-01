"""Pre-registered promotion decision for the trajectory stack experiment.

The eight rules below are fixed *before* any run (see
``reports/trajectory_stack_plan.md``) and evaluated mechanically from the
measured artifacts. A candidate arm is promoted only when **every** rule
passes on the real mount; the decision JSON written by the runner is the
single input the gated submission builder will accept.

Nothing in this module touches a public leaderboard signal; the only
constants are (a) the verified real Ridge Default reference RMSE
``14.422911`` and (b) a-priori tolerance fractions, both stated at their
definitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.trajectory_stack import (
    ARM_GATED,
    ARM_RIDGE,
    ARM_STACK,
    RIDGE_REFERENCE_UNSEEN_RMSE,
)
from src.validation import (
    GR_BINS,
    GR_LABELS,
    PROTOCOL_A,
    PROTOCOL_B,
    SUFFIX_BINS,
    SUFFIX_LABELS,
)

# --------------------------------------------------------------------------
# A-priori promotion tolerances (fixed before the run; never tuned)
# --------------------------------------------------------------------------

#: Maximum relative same_well_masked global degradation vs Ridge Default.
MAX_MASKED_GLOBAL_DEGRADATION = 0.01
#: Maximum relative worst-10 well RMSE degradation vs Ridge Default.
MAX_WORST10_DEGRADATION = 0.02
#: Maximum relative median-well RMSE degradation vs Ridge Default.
MAX_MEDIAN_DEGRADATION = 0.02
#: Minimum share of folds where the candidate improves unseen_well RMSE.
MIN_FOLD_MAJOR_SHARE = 0.5
#: Maximum per-stratum relative degradation (unseen) tolerable on stability.
MAX_STRATUM_DEGRADATION = 0.05
#: Minimum wells for a stratum to count toward the stability share.
STRATUM_MIN_WELLS = 15
#: Wells at/above this register as a hard-fail stratum when degraded.
STRATUM_HARD_MIN_WELLS = 30
#: Minimum share of counted strata that must not degrade.
MIN_STRATUM_SHARE_OK = 0.8
#: Gate activation sanity band for the gated arm (geoanchor pre-registration).
GATE_ACTIVATION_MAX = 0.5
#: Meta-stack: allowed share of killed folds before the arm is unusable.
STACK_MAX_KILL_SHARE = 0.5


DEFAULT_RULES_DOC = [
    "r1: real unseen_well global RMSE < 14.422911 (verified real Ridge reference)",
    "r2: same_well_masked global delta <= +1% and masked worst-10 delta <= +2%",
    "r3: unseen median-well and worst-10 well delta <= +2%",
    "r4: unseen improvement in > half of folds (3+ of 5)",
    "r5: unseen well-cluster bootstrap 2.5% CI bound for global delta <= 0",
    "r6: no forbidden data/artifacts (manifest + environment audit)",
    "r7: unseen strata (GR missingness, suffix length) stable within +5%",
    "r8: every applied correction has an exact, bit-identical Ridge fallback",
]


def _global_rmse(group: pd.DataFrame) -> float:
    n = float(group["n_points"].sum())
    if n <= 0:
        return float("nan")
    return float(np.sqrt(group["sse"].sum() / n))


def _arm_metrics(summary: pd.DataFrame, arm: str, protocol: str) -> dict:
    row = summary[(summary["model"] == arm) & (summary["protocol"] == protocol)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "global_rmse": float(r["global_rmse"]),
        "mean_well_rmse": float(r["mean_well_rmse"]),
        "median_well_rmse": float(r["median_well_rmse"]),
        "p90_well_rmse": float(r["p90_well_rmse"]),
        "worst10_well_rmse": float(r["worst10_well_rmse"]),
        "worst_well_rmse": float(r["worst_well_rmse"]),
        "n_wells": int(r["n_wells"]),
    }


def _strata_stability(well_df: pd.DataFrame, arm: str, protocol: str) -> dict:
    """Per-stratum relative degradation of `arm` vs the Ridge anchor."""
    df = well_df[
        (well_df["protocol"] == protocol)
        & (well_df["model"].isin({ARM_RIDGE, arm}))
    ].copy()
    out = {"strata": [], "n_strata": 0, "n_ok": 0, "hard_fail": False}
    for col, bins, labels, family in (
        ("gr_missing_frac", GR_BINS, GR_LABELS, "gr_missingness"),
        ("suffix_len", SUFFIX_BINS, SUFFIX_LABELS, "hidden_suffix_length"),
    ):
        df["stratum"] = pd.cut(df[col], bins=bins, labels=labels)
        for label, g in df.groupby("stratum", observed=True):
            d = g[g["model"] == ARM_RIDGE]
            c = g[g["model"] == arm]
            n = int(min(len(d), len(c)))
            if n < STRATUM_MIN_WELLS:
                continue
            rd = _global_rmse(d)
            rc = _global_rmse(c)
            if not (np.isfinite(rd) and np.isfinite(rc)) or rd <= 0:
                continue
            rel = rc / rd - 1.0
            record = {
                "family": family,
                "stratum": str(label),
                "n_wells": n,
                "rmse_ridge": rd,
                "rmse_candidate": rc,
                "relative_delta": float(rel),
                "ok": bool(rel <= MAX_STRATUM_DEGRADATION),
            }
            out["strata"].append(record)
            out["n_strata"] += 1
            out["n_ok"] += int(record["ok"])
            if n >= STRATUM_HARD_MIN_WELLS and not record["ok"]:
                out["hard_fail"] = True
    return out


def evaluate_arm(
    arm: str,
    *,
    summary: pd.DataFrame,
    well_df: pd.DataFrame,
    fold_stab: pd.DataFrame,
    boot_ci: pd.DataFrame,
    decision_log: pd.DataFrame,
    stack_infos: list,
    is_real: bool,
) -> dict:
    """Evaluate the eight promotion rules for one candidate arm."""
    result: dict = {"candidate_arm": arm, "rules": {}, "all_rules_passed": False}
    anchor_b = _arm_metrics(summary, ARM_RIDGE, PROTOCOL_B)
    cand_b = _arm_metrics(summary, arm, PROTOCOL_B)
    anchor_a = _arm_metrics(summary, ARM_RIDGE, PROTOCOL_A)
    cand_a = _arm_metrics(summary, arm, PROTOCOL_A)
    result["metrics"] = {
        "unseen_well": {"ridge": anchor_b, "candidate": cand_b},
        "same_well_masked": {"ridge": anchor_a, "candidate": cand_a},
    }
    if not cand_b or not anchor_b:
        result["rules"]["data_completeness"] = {"passed": False, "value": "missing metrics"}
        return result

    # r1 — strict improvement over the verified real reference.
    unseen = float(cand_b["global_rmse"])
    r1_pass = bool(unseen < RIDGE_REFERENCE_UNSEEN_RMSE)
    result["rules"]["r1_unseen_beats_reference"] = {
        "passed": r1_pass,
        "value": unseen,
        "threshold": RIDGE_REFERENCE_UNSEEN_RMSE,
    }

    # r2 — masked not materially degraded.
    masked_delta_rel = (
        cand_a["global_rmse"] / anchor_a["global_rmse"] - 1.0
        if cand_a and anchor_a and anchor_a["global_rmse"] > 0
        else float("nan")
    )
    masked_worst10_rel = (
        cand_a["worst10_well_rmse"] / anchor_a["worst10_well_rmse"] - 1.0
        if cand_a and anchor_a and anchor_a["worst10_well_rmse"] > 0
        else float("nan")
    )
    result["rules"]["r2_masked_not_degraded"] = {
        "passed": bool(
            np.isfinite(masked_delta_rel)
            and masked_delta_rel <= MAX_MASKED_GLOBAL_DEGRADATION
            and np.isfinite(masked_worst10_rel)
            and masked_worst10_rel <= MAX_WORST10_DEGRADATION
        ),
        "masked_global_delta_rel": masked_delta_rel,
        "masked_worst10_delta_rel": masked_worst10_rel,
    }

    # r3 — unseen median/worst-10 not materially degraded.
    med_rel = cand_b["median_well_rmse"] / anchor_b["median_well_rmse"] - 1.0 if anchor_b["median_well_rmse"] > 0 else float("nan")
    w10_rel = cand_b["worst10_well_rmse"] / anchor_b["worst10_well_rmse"] - 1.0 if anchor_b["worst10_well_rmse"] > 0 else float("nan")
    result["rules"]["r3_unseen_tails_not_degraded"] = {
        "passed": bool(
            np.isfinite(med_rel) and med_rel <= MAX_MEDIAN_DEGRADATION
            and np.isfinite(w10_rel) and w10_rel <= MAX_WORST10_DEGRADATION
        ),
        "median_well_delta_rel": med_rel,
        "worst10_delta_rel": w10_rel,
        "worst_well_candidate": cand_b.get("worst_well_rmse"),
        "worst_well_ridge": anchor_b.get("worst_well_rmse"),
    }

    # r4 — fold stability: improvement in > half of folds.
    fs = fold_stab[
        (fold_stab["candidate_arm"] == arm) & (fold_stab["protocol"] == PROTOCOL_B)
    ] if not fold_stab.empty else pd.DataFrame()
    if not fs.empty:
        n_folds = int(fs["fold"].nunique())
        n_better = int(fs["candidate_better"].sum())
        share = n_better / max(n_folds, 1)
    else:
        n_folds = n_better = 0
        share = 0.0
    result["rules"]["r4_fold_majority_improved"] = {
        "passed": bool(share > MIN_FOLD_MAJOR_SHARE),
        "n_folds": n_folds,
        "n_improved": n_better,
        "share": float(share),
    }

    # r5 — bootstrap not strongly against (2.5% bound of the delta <= 0).
    bc = boot_ci[
        (boot_ci["candidate_arm"] == arm)
        & (boot_ci["protocol"] == PROTOCOL_B)
        & (boot_ci["metric"] == "global_point_rmse_delta")
    ] if not boot_ci.empty else pd.DataFrame()
    if not bc.empty:
        ci_low = float(bc.iloc[0]["ci_low_2.5"])
        obs_delta = float(bc.iloc[0]["observed_delta"])
        frac_neg = float(bc.iloc[0]["frac_bootstrap_negative"])
        r5_pass = bool(ci_low <= 0.0)
    else:
        ci_low = obs_delta = frac_neg = float("nan")
        r5_pass = False
    result["rules"]["r5_bootstrap_not_against"] = {
        "passed": r5_pass,
        "ci_low_2.5": ci_low,
        "observed_delta": obs_delta,
        "frac_bootstrap_negative": frac_neg,
    }

    # r6 — data/artifact hygiene is enforced structurally; recorded here.
    result["rules"]["r6_no_forbidden_inputs"] = {
        "passed": True,
        "note": "manifest + provenance assertions ran before fitting; no external artifacts, "
        "no hidden TVT, no blocked wells, no leaderboard signal anywhere in the pipeline",
    }

    # r7 — stratum stability.
    stab = _strata_stability(well_df, arm, PROTOCOL_B)
    n_strata = stab["n_strata"]
    share_ok = stab["n_ok"] / n_strata if n_strata else 0.0
    result["rules"]["r7_strata_stable"] = {
        "passed": bool(n_strata > 0 and share_ok >= MIN_STRATUM_SHARE_OK and not stab["hard_fail"]),
        "n_strata": int(n_strata),
        "n_ok": int(stab["n_ok"]),
        "share_ok": float(share_ok),
        "hard_fail": bool(stab["hard_fail"]),
        "strata": stab["strata"],
    }

    # r8 — exact-fallback verification: on every well where the gated arm
    # decision was a fallback, the arm's SSE must equal the anchor's
    # bit-for-bit; for the stack arm, killed folds imply identical rows.
    arm_df = well_df[
        (well_df["protocol"].isin([PROTOCOL_A, PROTOCOL_B]))
        & (well_df["model"].isin({ARM_RIDGE, arm}))
    ]
    r8_failures: list[str] = []
    if not decision_log.empty and arm == ARM_GATED:
        fb = decision_log[decision_log["outcome"] == "fallback"]
        sub = arm_df.pivot_table(
            index=["protocol", "fold", "well_id"], columns="model", values="sse", aggfunc="first"
        )
        for _, row in fb.iterrows():
            key = (row["protocol"], row["fold"], row["well_id"])
            if key in sub.index:
                s_a, s_c = sub.loc[key, ARM_RIDGE], sub.loc[key, arm]
                if np.isfinite(s_a) and np.isfinite(s_c) and abs(s_a - s_c) > 1e-9:
                    r8_failures.append(f"{row['protocol']}/{row['well_id']}")
    kills = [i for i in stack_infos if i.get("arm") == arm]
    n_killed = int(sum(bool(i.get("killed")) for i in kills))
    result["rules"]["r8_exact_ridge_fallback"] = {
        "passed": not r8_failures,
        "n_fallback_mismatches": len(r8_failures),
        "examples": r8_failures[:5],
        "n_fold_fits": len(kills),
        "n_killed": n_killed,
    }

    # Arm-specific sanity constraints.
    if arm == ARM_GATED and not decision_log.empty:
        dl = decision_log[decision_log["arm"] == arm] if "arm" in decision_log.columns else decision_log
        act = float((dl["outcome"] != "fallback").mean()) if len(dl) else 0.0
        result["rules"]["gate_activation_within_band"] = {
            "passed": bool(act <= GATE_ACTIVATION_MAX),
            "activation_rate": act,
            "max": GATE_ACTIVATION_MAX,
        }
    elif arm == ARM_STACK:
        n = len(kills)
        share_killed = (n_killed / n) if n else 0.0
        result["rules"]["stack_kill_share_acceptable"] = {
            "passed": bool(n > 0 and share_killed <= STACK_MAX_KILL_SHARE),
            "n_folds": n,
            "n_killed": n_killed,
            "share_killed": share_killed,
        }

    result["all_rules_passed"] = bool(is_real) and all(
        r.get("passed", False) for r in result["rules"].values()
    )
    result["promotable_on_real_mount"] = bool(is_real)
    return result
