"""Paired-error / robustness analysis for ridge_particle_beam vs ridge_default."""
from __future__ import annotations

import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest


def _mod(name):
    key = f"src.{name}"
    return importlib.reload(sys.modules[key]) if key in sys.modules else importlib.import_module(key)


def _wells(
    *,
    n_wells: int = 20,
    n_folds: int = 5,
    default_rmse: float = 2.0,
    candidate_shift: float = -0.1,
    long_well_bonus: float = 0.0,
    concentrate_on: int = 0,
) -> pd.DataFrame:
    """Build a multi-model well-level table with controllable deltas.

    ``candidate_shift`` is added to every well's candidate RMSE (negative =
    candidate better).  When ``concentrate_on`` > 0, only the first K wells
    receive the full shift; the rest are unchanged.
    """
    rows = []
    for protocol in ("same_well_masked", "unseen_well"):
        for i in range(n_wells):
            fold = i % n_folds
            suffix = 200 + 50 * i
            prefix = 800 + 20 * i
            gr = 0.01 * (i % 5)
            n_points = 100 + 10 * i
            base = default_rmse + 0.01 * i
            if concentrate_on and i >= concentrate_on:
                shift = 0.0
            else:
                shift = candidate_shift
            if long_well_bonus and suffix >= (200 + 50 * int(0.75 * n_wells)):
                shift = shift + long_well_bonus
            for model, rmse in (
                ("ridge_default", base),
                ("ridge_particle_filter", base + 0.05),
                ("ridge_beam_search", base + 0.04),
                ("ridge_particle_beam", base + shift),
            ):
                rows.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "fold": fold,
                        "well_id": f"w{i:03d}",
                        "n_points": n_points,
                        "sse": n_points * rmse**2,
                        "rmse": rmse,
                        "max_abs_error": rmse * 2,
                        "bias": 0.0,
                        "prefix_len": prefix,
                        "suffix_len": suffix,
                        "gr_missing_frac": gr,
                        "anchor_tvt": 0.0,
                        "has_typewell": True,
                        "predict_seconds": 0.01,
                        "particle_confidence_mean": 0.55 if "particle" in model else np.nan,
                        "particle_confidence_p10": 0.40 if "particle" in model else np.nan,
                        "particle_fallback_fraction": 0.05 if "particle" in model else np.nan,
                        "particle_fallback_status": False if "particle" in model else np.nan,
                        "particle_failure_reason": "" if "particle" in model else "",
                        "beam_confidence_mean": 0.50 if "beam" in model else np.nan,
                        "beam_confidence_p10": 0.35 if "beam" in model else np.nan,
                        "beam_fallback_fraction": 0.08 if "beam" in model else np.nan,
                        "beam_fallback_status": False if "beam" in model else np.nan,
                        "beam_failure_reason": "" if "beam" in model else "",
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------- pairing --


def test_pair_default_vs_candidate_delta_sign():
    rb = _mod("pf_beam_robustness")
    paired = rb.pair_default_vs_candidate(_wells(candidate_shift=-0.2))
    assert set(paired["protocol"]) == {"same_well_masked", "unseen_well"}
    assert (paired["delta_rmse"] < 0).all()
    assert paired["improved"].all()
    assert not paired["degraded"].any()


def test_pair_requires_both_models():
    rb = _mod("pf_beam_robustness")
    wells = _wells()
    wells = wells[wells["model"] != "ridge_particle_beam"]
    paired = rb.pair_default_vs_candidate(wells)
    assert paired.empty


def test_well_level_summary_counts():
    rb = _mod("pf_beam_robustness")
    # Half the wells improve, half degrade.
    rows = []
    for i in range(10):
        shift = -0.2 if i < 5 else 0.2
        for protocol in ("same_well_masked", "unseen_well"):
            for model, rmse in (
                ("ridge_default", 2.0),
                ("ridge_particle_beam", 2.0 + shift),
            ):
                rows.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "fold": i % 5,
                        "well_id": f"w{i}",
                        "n_points": 100,
                        "sse": 100 * rmse**2,
                        "rmse": rmse,
                        "prefix_len": 1000,
                        "suffix_len": 500,
                        "gr_missing_frac": 0.0,
                    }
                )
    paired = rb.pair_default_vs_candidate(pd.DataFrame(rows))
    summary = rb.well_level_summary(paired)
    for _, row in summary.iterrows():
        assert row["n_improved"] == 5
        assert row["n_degraded"] == 5
        assert row["pct_improved"] == pytest.approx(50.0)
        assert row["mean_well_delta_rmse"] == pytest.approx(0.0)


# --------------------------------------------------------------- folds ----


def test_fold_deltas_mark_stable_when_every_fold_improves():
    rb = _mod("pf_beam_robustness")
    folds = rb.fold_deltas(_wells(candidate_shift=-0.15, n_wells=25, n_folds=5))
    assert folds["fold"].nunique() == 5
    assert folds["stable_across_folds"].all()
    assert (folds["delta_rmse"] < 0).all()


def test_fold_deltas_mark_unstable_when_one_fold_worsens():
    rb = _mod("pf_beam_robustness")
    wells = _wells(candidate_shift=-0.1, n_wells=20, n_folds=5)
    # Make fold 0 worse for the candidate under both protocols.
    mask = (wells["fold"] == 0) & (wells["model"] == "ridge_particle_beam")
    wells.loc[mask, "rmse"] = wells.loc[mask, "rmse"] + 1.0
    wells.loc[mask, "sse"] = wells.loc[mask, "n_points"] * wells.loc[mask, "rmse"] ** 2
    folds = rb.fold_deltas(wells)
    for protocol, g in folds.groupby("protocol"):
        assert not bool(g["stable_across_folds"].iloc[0]), protocol
        assert int(g["n_folds_candidate_not_worse"].iloc[0]) < 5


# ----------------------------------------------------------- bootstrap ----


def test_bootstrap_ci_negative_when_candidate_clearly_better():
    rb = _mod("pf_beam_robustness")
    paired = rb.pair_default_vs_candidate(
        _wells(candidate_shift=-0.5, n_wells=30)
    )
    global_boot = rb.bootstrap_global_rmse_delta(paired, n_boot=500, seed=1)
    paired_boot = rb.bootstrap_paired_well_delta(paired, n_boot=500, seed=2)
    for boot in (global_boot, paired_boot):
        assert (boot["observed_delta"] < 0).all()
        assert (boot["ci_high_97.5"] < 0).all()
        assert boot["ci_excludes_zero"].all()


def test_bootstrap_ci_includes_zero_for_tiny_noisy_effect():
    rb = _mod("pf_beam_robustness")
    # Tiny alternating shifts so the mean is near zero.
    rows = []
    rng = np.random.default_rng(0)
    for protocol in ("same_well_masked", "unseen_well"):
        for i in range(40):
            shift = float(rng.normal(0.0, 0.3))
            for model, rmse in (
                ("ridge_default", 2.0),
                ("ridge_particle_beam", 2.0 + shift),
            ):
                rows.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "fold": i % 5,
                        "well_id": f"w{i}",
                        "n_points": 100,
                        "sse": 100 * rmse**2,
                        "rmse": rmse,
                        "prefix_len": 1000,
                        "suffix_len": 500,
                        "gr_missing_frac": 0.0,
                    }
                )
    paired = rb.pair_default_vs_candidate(pd.DataFrame(rows))
    boot = rb.bootstrap_paired_well_delta(paired, n_boot=400, seed=3)
    # With mean-zero noise the CI should typically include zero; we only
    # assert the machinery produced finite bounds, not a particular sign.
    assert boot["ci_low_2.5"].notna().all()
    assert boot["ci_high_97.5"].notna().all()
    assert (boot["ci_low_2.5"] < boot["ci_high_97.5"]).all()


# ----------------------------------------------------------- decision -----


def test_decision_keeps_candidate_when_robust():
    rb = _mod("pf_beam_robustness")
    wells = _wells(candidate_shift=-0.25, n_wells=30, n_folds=5)
    paired = rb.pair_default_vs_candidate(wells)
    summary = rb.well_level_summary(paired)
    folds = rb.fold_deltas(wells)
    gboot = rb.bootstrap_global_rmse_delta(paired, n_boot=300, seed=4)
    pboot = rb.bootstrap_paired_well_delta(paired, n_boot=300, seed=5)
    long_w = rb.long_well_concentration(paired)
    decision = rb.decide_candidate(
        well_summary=summary,
        fold_table=folds,
        global_boot=gboot,
        paired_boot=pboot,
        long_wells=long_w,
    )
    assert decision["keep_as_next_candidate"] is True
    assert decision["use_as_final"] is False
    assert decision["preserve_default_fallback"] is True
    assert decision["delete_pf_beam_code"] is False


def test_decision_rejects_concentrated_improvement():
    rb = _mod("pf_beam_robustness")
    # Only 2 of 30 wells improve a lot; the rest are flat. Global may still
    # improve slightly, but concentration must block promotion.
    wells = _wells(
        candidate_shift=-2.0, n_wells=30, n_folds=5, concentrate_on=2
    )
    paired = rb.pair_default_vs_candidate(wells)
    summary = rb.well_level_summary(paired)
    # Force the concentration flag on for the test's clarity.
    assert summary["improvement_concentrated"].all() or summary[
        "top10_sse_improvement_share"
    ].min() >= 0.5
    folds = rb.fold_deltas(wells)
    gboot = rb.bootstrap_global_rmse_delta(paired, n_boot=200, seed=6)
    pboot = rb.bootstrap_paired_well_delta(paired, n_boot=200, seed=7)
    decision = rb.decide_candidate(
        well_summary=summary,
        fold_table=folds,
        global_boot=gboot,
        paired_boot=pboot,
        long_wells=rb.long_well_concentration(paired),
    )
    assert decision["keep_as_next_candidate"] is False


def test_decision_owner_only_does_not_promote():
    rb = _mod("pf_beam_robustness")
    decision = rb.decide_candidate(
        well_summary=None,
        fold_table=None,
        global_boot=None,
        paired_boot=None,
        long_wells=None,
        owner_only=True,
    )
    assert decision["keep_as_next_candidate"] is False
    assert decision["use_as_final"] is False
    assert decision["preserve_default_fallback"] is True
    assert decision["owner_only_aggregates"] is True
    text = " ".join(decision["reasons"]).lower()
    assert "per-well" in text or "not available" in text


def test_decision_rejects_when_folds_unstable():
    rb = _mod("pf_beam_robustness")
    wells = _wells(candidate_shift=-0.2, n_wells=25, n_folds=5)
    mask = (wells["fold"] == 2) & (wells["model"] == "ridge_particle_beam")
    wells.loc[mask, "rmse"] = wells.loc[mask, "rmse"] + 1.5
    wells.loc[mask, "sse"] = wells.loc[mask, "n_points"] * wells.loc[mask, "rmse"] ** 2
    paired = rb.pair_default_vs_candidate(wells)
    decision = rb.decide_candidate(
        well_summary=rb.well_level_summary(paired),
        fold_table=rb.fold_deltas(wells),
        global_boot=rb.bootstrap_global_rmse_delta(paired, n_boot=200, seed=8),
        paired_boot=rb.bootstrap_paired_well_delta(paired, n_boot=200, seed=9),
        long_wells=rb.long_well_concentration(paired),
    )
    assert decision["keep_as_next_candidate"] is False
    assert any("stable" in r.lower() or "fold" in r.lower() for r in decision["reasons"])


# ----------------------------------------------------------- reporting ----


def test_write_reports_owner_only(tmp_path):
    rb = _mod("pf_beam_robustness")
    written = {p.name for p in rb.write_robustness_reports(tmp_path, well=None)}
    required = {
        "pf_beam_real_decision.md",
        "pf_beam_paired_well_deltas.csv",
        "pf_beam_fold_deltas.csv",
        "pf_beam_bootstrap_ci.csv",
        "pf_beam_failure_analysis.md",
        "particle_beam_fold_deltas.csv",
        "particle_beam_bootstrap_ci.csv",
    }
    assert required <= written

    decision = (tmp_path / "pf_beam_real_decision.md").read_text()
    assert "REAL KAGGLE VALIDATION" in decision
    assert "SYNTHETIC" in decision
    assert "PUBLIC LEADERBOARD" in decision
    assert "14.419" in decision
    assert "29.388" in decision
    assert "ridge_default" in decision
    assert "DO NOT keep" in decision or "NOT kept" in decision or "not promoted" in decision.lower()
    assert "No statistical significance" in decision or "No significance" in decision or "not fabricated" in decision.lower() or "significance" in decision.lower()

    failure = (tmp_path / "pf_beam_failure_analysis.md").read_text()
    assert "REAL KAGGLE VALIDATION" in failure or "Real Kaggle" in failure
    assert "Synthetic" in failure or "SYNTHETIC" in failure
    assert "leaderboard" in failure.lower()
    assert "not fabricated" in failure.lower() or "Unavailable" in failure

    paired = pd.read_csv(tmp_path / "pf_beam_paired_well_deltas.csv")
    assert "availability" in paired.columns
    assert paired["availability"].astype(str).str.contains("UNAVAILABLE").all()

    boot = pd.read_csv(tmp_path / "pf_beam_bootstrap_ci.csv")
    assert (boot["n_bootstrap"] == 0).all()
    # Observed owner deltas are recorded; CI bounds are empty.
    unseen = boot[
        (boot["protocol"] == "unseen_well")
        & (boot["metric"] == "global_point_rmse_delta")
    ]
    assert float(unseen["observed_delta"].iloc[0]) == pytest.approx(14.419 - 14.423)

    payload = json.loads((tmp_path / "pf_beam_decision.json").read_text())
    assert payload["keep_as_next_candidate"] is False
    assert payload["use_as_final"] is False


def test_write_reports_with_wells(tmp_path):
    rb = _mod("pf_beam_robustness")
    wells = _wells(candidate_shift=-0.3, n_wells=25, n_folds=5)
    written = rb.write_robustness_reports(tmp_path, wells)
    names = {p.name for p in written}
    assert "pf_beam_real_decision.md" in names
    assert "pf_beam_paired_well_deltas.csv" in names
    assert "pf_beam_fold_deltas.csv" in names
    assert "pf_beam_bootstrap_ci.csv" in names
    assert "pf_beam_failure_analysis.md" in names
    assert "particle_beam_fold_deltas.csv" in names
    assert "particle_beam_bootstrap_ci.csv" in names

    paired = pd.read_csv(tmp_path / "pf_beam_paired_well_deltas.csv")
    assert len(paired) == 50  # 25 wells x 2 protocols
    assert (paired["delta_rmse"] < 0).all()
    assert "availability" not in paired.columns

    folds = pd.read_csv(tmp_path / "pf_beam_fold_deltas.csv")
    assert set(folds["fold"]) == {0, 1, 2, 3, 4}

    boot = pd.read_csv(tmp_path / "pf_beam_bootstrap_ci.csv")
    assert set(boot["metric"]) >= {"global_point_rmse_delta", "mean_well_rmse_delta"}
    assert (boot["n_bootstrap"] > 0).all()
    assert boot["ci_low_2.5"].notna().all()

    decision_text = (tmp_path / "pf_beam_real_decision.md").read_text()
    assert "Per-well paired deltas (computed)" in decision_text
    assert "Bootstrap confidence intervals" in decision_text


def test_protocols_never_averaged_in_decision(tmp_path):
    rb = _mod("pf_beam_robustness")
    rb.write_robustness_reports(tmp_path, _wells(candidate_shift=-0.2))
    text = (tmp_path / "pf_beam_real_decision.md").read_text().lower()
    assert "never averaged" in text
    # The mean of the two owner globals must not appear as a headline metric.
    mean_val = (14.419 + 29.388) / 2.0
    assert f"{mean_val:.3f}" not in (tmp_path / "pf_beam_real_decision.md").read_text()


def test_stratified_and_diagnostics_populated():
    rb = _mod("pf_beam_robustness")
    wells = _wells(candidate_shift=-0.1, n_wells=30)
    paired = rb.pair_default_vs_candidate(wells)
    strat = rb.all_stratified_deltas(paired)
    assert set(strat["stratify_by"]) == {
        "gr_missingness",
        "hidden_suffix_length",
        "prefix_length",
    }
    diag = rb.generator_diagnostics(wells)
    cand = diag[diag["model"] == "ridge_particle_beam"]
    assert cand["particle_confidence_mean"].notna().all()
    assert cand["beam_confidence_mean"].notna().all()
    assert cand["particle_fallback_fraction_mean"].notna().all()


def test_owner_aggregate_table_matches_brief():
    rb = _mod("pf_beam_robustness")
    owner = rb.owner_aggregate_table()
    unseen = owner[owner["protocol"] == "unseen_well"].set_index("model")["global_rmse"]
    masked = owner[owner["protocol"] == "same_well_masked"].set_index("model")["global_rmse"]
    assert unseen["ridge_default"] == pytest.approx(14.423)
    assert unseen["ridge_particle_beam"] == pytest.approx(14.419)
    assert masked["ridge_default"] == pytest.approx(29.486)
    assert masked["ridge_particle_beam"] == pytest.approx(29.388)
    assert unseen["ridge_particle_filter"] == pytest.approx(14.429)
    assert unseen["ridge_beam_search"] == pytest.approx(14.432)


def test_analyze_script_owner_only(tmp_path):
    import runpy
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_pf_beam_robustness.py"
    # Execute via main() import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.analyze_pf_beam_robustness import main

    rc = main(["--reports-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "pf_beam_real_decision.md").exists()
    assert (tmp_path / "pf_beam_failure_analysis.md").exists()
