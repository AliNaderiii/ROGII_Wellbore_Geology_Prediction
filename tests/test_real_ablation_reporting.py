"""The six REAL KAGGLE VALIDATION reports and the pre-registered decision rule.

The most important property tested here is that a synthetic or partial run
**cannot** be stamped as real validation: the banner is derived from the
observed well counts, not from a flag the caller passes.
"""
from __future__ import annotations

import importlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest


def _mod(name):
    key = f"src.{name}"
    return importlib.reload(sys.modules[key]) if key in sys.modules else importlib.import_module(key)


def _wells(values: dict, *, n_wells: int = 4) -> pd.DataFrame:
    """Per-well rows whose weighted RMSE equals the requested value exactly."""
    rows = []
    for (protocol, branch), rmse in values.items():
        for i in range(n_wells):
            rows.append(
                {
                    "protocol": protocol, "model": branch, "well_id": f"w{i}",
                    "fold": i % 2, "n_points": 100, "sse": 100 * rmse**2, "rmse": rmse,
                    "max_abs_error": rmse * 2, "bias": 0.0,
                    "prefix_len": 1000 + 500 * i, "suffix_len": 400 + 300 * i,
                    "gr_missing_frac": 0.02 * i, "anchor_tvt": 0.0,
                    "has_typewell": True, "predict_seconds": 0.01,
                    "scored_exact_suffix": True,
                }
            )
    return pd.DataFrame(rows)


def _grid(**overrides) -> dict:
    ablation = _mod("ablation")
    values = {}
    for protocol in ("same_well_masked", "unseen_well"):
        values[(protocol, ablation.BRANCH_A)] = 2.0
        values[(protocol, ablation.BRANCH_B)] = 1.5
        values[(protocol, ablation.BRANCH_C)] = 1.8
        values[(protocol, ablation.BRANCH_D)] = 1.4
    values.update(overrides)
    return values


REAL_ENV = {"n_train_wells_discovered": 773, "n_eligible_wells": 770, "n_wells_evaluated": 770}
SYNTH_ENV = {"n_train_wells_discovered": 40, "n_eligible_wells": 40, "n_wells_evaluated": 40}


# ----------------------------------------------------- banner correctness --

def test_only_the_audited_real_mount_earns_the_real_banner():
    real = _mod("real_ablation_reporting")
    assert real.is_real_run(REAL_ENV)
    assert real.banner_for(REAL_ENV) == real.REAL_BANNER
    for env in (SYNTH_ENV, {}, None,
                {"n_train_wells_discovered": 773, "n_eligible_wells": 100},
                {"n_train_wells_discovered": 100, "n_eligible_wells": 770}):
        assert not real.is_real_run(env)
        assert real.banner_for(env) == real.SYNTHETIC_BANNER


def test_synthetic_run_is_never_labelled_real(tmp_path):
    """The exact mixing hazard the brief calls out."""
    real = _mod("real_ablation_reporting")
    real.write_real_ablation_reports(tmp_path, _wells(_grid()), environment=SYNTH_ENV)
    for name in (
        "synthetic_alignment_ablation_summary.md",
        "synthetic_alignment_feature_comparison.md",
        "synthetic_protocol_comparison.md",
        "synthetic_spatial_ablation.md",
    ):
        text = (tmp_path / name).read_text()
        assert real.SYNTHETIC_BANNER in text
        assert "not a competition result" in text.lower()
        assert not text.startswith(f"> # {real.REAL_BANNER}")
    results = pd.read_csv(tmp_path / "synthetic_alignment_ablation_results.csv")
    assert set(results["validation"]) == {real.SYNTHETIC_BANNER}
    # And nothing named `real_*` was left behind.
    assert not any(p.name.startswith("real_") for p in tmp_path.iterdir())


def test_real_run_is_labelled_real(tmp_path):
    real = _mod("real_ablation_reporting")
    real.write_real_ablation_reports(tmp_path, _wells(_grid()), environment=REAL_ENV)
    text = (tmp_path / "real_alignment_ablation_summary.md").read_text()
    assert real.REAL_BANNER in text
    assert "773" in text and "770" in text
    results = pd.read_csv(tmp_path / "real_alignment_ablation_results.csv")
    assert set(results["validation"]) == {real.REAL_BANNER}


def test_a_real_subset_run_is_flagged_as_a_subset(tmp_path):
    real = _mod("real_ablation_reporting")
    env = {**REAL_ENV, "n_wells_evaluated": 100, "max_wells": 100}
    real.write_real_ablation_reports(tmp_path, _wells(_grid()), environment=env)
    text = (tmp_path / "real_alignment_ablation_summary.md").read_text()
    assert real.REAL_BANNER in text
    assert "Subset run" in text
    assert "Not the full validation" in text


# ----------------------------------------------------------- the six files --

def test_all_six_required_reports_are_written(tmp_path):
    real = _mod("real_ablation_reporting")
    written = {p.name for p in real.write_real_ablation_reports(
        tmp_path, _wells(_grid()), environment=REAL_ENV)}
    required = {
        "real_alignment_ablation_results.csv",
        "real_alignment_ablation_summary.md",
        "real_alignment_feature_comparison.md",
        "real_protocol_comparison.md",
        "real_spatial_ablation.md",
        "real_well_level_ablation.csv",
    }
    assert required <= written


def test_results_csv_carries_every_required_metric(tmp_path):
    real = _mod("real_ablation_reporting")
    env = {**REAL_ENV, "runtime_seconds": 12.5, "peak_rss_mb": 900.0,
           "cache_hits": 7, "cache_misses": 3, "cache_writes": 3}
    real.write_real_ablation_reports(tmp_path, _wells(_grid()), environment=env,
                                     failures=pd.DataFrame([{"stage": "fit"}]))
    df = pd.read_csv(tmp_path / "real_alignment_ablation_results.csv")
    required = {
        "global_rmse", "mean_well_rmse", "median_well_rmse", "p90_well_rmse",
        "worst10_well_rmse", "worst_well_id", "n_wells_evaluated", "n_points_evaluated",
        "runtime_seconds", "peak_rss_mb", "failure_count",
        "cache_hits", "cache_misses", "cache_writes",
    }
    assert required <= set(df.columns)
    assert set(df["failure_count"]) == {1}
    assert len(df) == 8  # 4 branches x 2 protocols


def test_stratifications_are_all_present(tmp_path):
    real = _mod("real_ablation_reporting")
    real.write_real_ablation_reports(tmp_path, _wells(_grid()), environment=REAL_ENV)
    strat = pd.read_csv(tmp_path / "real_ablation_stratified.csv")
    assert set(strat["stratify_by"]) == {
        "gr_missingness", "hidden_suffix_length", "prefix_length"
    }
    text = (tmp_path / "real_alignment_ablation_summary.md").read_text()
    for heading in ("By GR missingness", "By hidden suffix length", "By prefix length"):
        assert heading in text


def test_points_are_not_double_counted_across_branches(tmp_path):
    real = _mod("real_ablation_reporting")
    wells = _wells(_grid())
    real.write_real_ablation_reports(tmp_path, wells, environment=REAL_ENV)
    df = pd.read_csv(tmp_path / "real_alignment_ablation_results.csv")
    # Each of 4 wells contributes 100 points to each branch. A per-branch row
    # must show 400, not 1600 (which is what summing across branches gives).
    assert set(df["n_points_evaluated"]) == {400}


def test_protocols_are_never_averaged(tmp_path):
    real = _mod("real_ablation_reporting")
    values = _grid()
    ablation = _mod("ablation")
    values[("unseen_well", ablation.BRANCH_B)] = 6.0  # make the protocols very different
    real.write_real_ablation_reports(tmp_path, _wells(values), environment=REAL_ENV)
    text = (tmp_path / "real_protocol_comparison.md").read_text()
    assert "not averaged" in text.lower() or "never averaged" in text.lower()
    df = pd.read_csv(tmp_path / "real_alignment_ablation_results.csv")
    same = df[(df["protocol"] == "same_well_masked") & (df["branch"] == ablation.BRANCH_B)]
    unseen = df[(df["protocol"] == "unseen_well") & (df["branch"] == ablation.BRANCH_B)]
    assert float(same["global_rmse"].iloc[0]) == pytest.approx(1.5)
    assert float(unseen["global_rmse"].iloc[0]) == pytest.approx(6.0)
    # The mean (3.75) must appear nowhere as a metric.
    assert not np.isclose(df["global_rmse"], 3.75).any()


def test_reports_emit_no_pandas_futurewarning(tmp_path):
    real = _mod("real_ablation_reporting")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        real.write_real_ablation_reports(
            tmp_path, _wells(_grid()), environment=REAL_ENV,
            failures=pd.DataFrame(columns=["stage", "model", "well_id", "error"]),
        )


def test_empty_input_produces_no_fabricated_numbers(tmp_path):
    real = _mod("real_ablation_reporting")
    real.write_real_ablation_reports(tmp_path, pd.DataFrame(), environment=REAL_ENV)
    results = pd.read_csv(tmp_path / "real_alignment_ablation_results.csv")
    assert results.empty
    text = (tmp_path / "real_alignment_ablation_summary.md").read_text()
    assert "No computed rows were available" in text


# ----------------------------------------- the pre-registered decision rule --

def test_rule_keeps_alignment_when_both_protocols_improve_cleanly():
    ablation = _mod("ablation")
    summary = ablation.summarize_ablation(_wells(_grid()))
    verdict = ablation.preregistered_verdict(ablation.preregistered_decision(summary))
    assert verdict["alignment"]["decision"] == "keep_as_features"


def test_rule_removes_alignment_when_only_one_protocol_improves():
    ablation = _mod("ablation")
    values = _grid()
    values[("same_well_masked", ablation.BRANCH_B)] = 2.5  # worse than A (2.0)
    summary = ablation.summarize_ablation(_wells(values))
    verdict = ablation.preregistered_verdict(ablation.preregistered_decision(summary))
    assert verdict["alignment"]["decision"] == "remove_from_next_baseline"


def test_rule_removes_alignment_on_material_worst10_degradation():
    """Global RMSE improves, but the worst wells get materially worse.

    Uses 30 wells so worst-10 is a genuine tail statistic rather than the mean
    of every well. Branch B trades 27 much-better wells against 3 much-worse
    ones: point-weighted global RMSE falls, but the tail the rule protects
    rises past the 2% tolerance.
    """
    ablation = _mod("ablation")
    flat = [2.0] * 30
    tail_heavy = [0.5] * 27 + [6.0] * 3
    rows = []
    for protocol in ("same_well_masked", "unseen_well"):
        for branch, per_well in (
            (ablation.BRANCH_A, flat),
            (ablation.BRANCH_B, tail_heavy),
            (ablation.BRANCH_C, flat),
            (ablation.BRANCH_D, tail_heavy),
        ):
            for i, rmse in enumerate(per_well):
                rows.append({
                    "protocol": protocol, "model": branch, "well_id": f"w{i}", "fold": 0,
                    "n_points": 100, "sse": 100 * rmse**2, "rmse": rmse,
                    "max_abs_error": rmse, "bias": 0.0, "prefix_len": 1000,
                    "suffix_len": 500, "gr_missing_frac": 0.0, "anchor_tvt": 0.0,
                    "has_typewell": True, "predict_seconds": 0.0,
                })
    summary = ablation.summarize_ablation(pd.DataFrame(rows))
    decision = ablation.preregistered_decision(summary)
    align = decision[decision["feature_group"] == "alignment"]
    assert align["improves_global"].all(), "fixture must improve global RMSE"
    assert align["material_worst10_degradation"].all(), "fixture must inflate the tail"
    verdict = ablation.preregistered_verdict(decision)
    assert verdict["alignment"]["decision"] == "remove_from_next_baseline"
    assert "degraded materially" in verdict["alignment"]["reason"]


def test_rule_allows_spatial_on_a_consistent_worst_well_gain():
    """Spatial may be kept for worst-well behaviour even without a global win.

    Median well RMSE is held exactly equal so the only movement is in the tail,
    which is the case the rule's worst-well clause exists to permit.
    """
    ablation = _mod("ablation")
    rows = []
    for protocol in ("same_well_masked", "unseen_well"):
        for branch, per_well in (
            (ablation.BRANCH_A, [1.0, 1.0, 1.0, 3.0]),
            (ablation.BRANCH_B, [1.0, 1.0, 1.0, 3.0]),
            # Spatial: identical typical wells, clearly better tail.
            (ablation.BRANCH_C, [1.0, 1.0, 1.0, 2.0]),
            (ablation.BRANCH_D, [1.0, 1.0, 1.0, 2.0]),
        ):
            for i, rmse in enumerate(per_well):
                rows.append({
                    "protocol": protocol, "model": branch, "well_id": f"w{i}", "fold": 0,
                    "n_points": 100, "sse": 100 * rmse**2, "rmse": rmse,
                    "max_abs_error": rmse, "bias": 0.0, "prefix_len": 1000,
                    "suffix_len": 500, "gr_missing_frac": 0.0, "anchor_tvt": 0.0,
                    "has_typewell": True, "predict_seconds": 0.0,
                })
    decision = ablation.preregistered_decision(ablation.summarize_ablation(pd.DataFrame(rows)))
    spatial = decision[decision["feature_group"] == "spatial"]
    assert spatial["improves_worst10"].all()
    assert not spatial["material_median_degradation"].any()
    verdict = ablation.preregistered_verdict(decision)
    assert verdict["spatial"]["decision"] == "keep_as_features"


def test_rule_requires_both_protocols():
    ablation = _mod("ablation")
    values = {(p, b): v for (p, b), v in _grid().items() if p == "unseen_well"}
    summary = ablation.summarize_ablation(_wells(values))
    verdict = ablation.preregistered_verdict(ablation.preregistered_decision(summary))
    for group in ("alignment", "spatial"):
        assert verdict[group]["decision"] == "remove_from_next_baseline"
        assert "both protocols" in verdict[group]["reason"]


def test_rule_is_undetermined_without_data():
    ablation = _mod("ablation")
    verdict = ablation.preregistered_verdict(pd.DataFrame())
    assert verdict["alignment"]["decision"] == "undetermined"
    assert verdict["spatial"]["decision"] == "undetermined"


def test_decision_is_persisted_for_audit(tmp_path):
    real = _mod("real_ablation_reporting")
    real.write_real_ablation_reports(tmp_path, _wells(_grid()), environment=REAL_ENV)
    decision = pd.read_csv(tmp_path / "real_ablation_decision.csv")
    assert set(decision["feature_group"]) == {"alignment", "spatial"}
    assert {"improves_global", "material_worst10_degradation"} <= set(decision.columns)
    assert len(decision) == 8  # 2 groups x 2 contrasts x 2 protocols


# ------------------------------------------------- filename/banner coupling --

def test_only_a_real_run_emits_real_prefixed_filenames(tmp_path):
    """A file named `real_*` on disk must always be a real result."""
    real = _mod("real_ablation_reporting")
    assert real.file_prefix(REAL_ENV) == "real_"
    assert real.file_prefix(SYNTH_ENV) == "synthetic_"

    synth_dir, real_dir = tmp_path / "s", tmp_path / "r"
    real.write_real_ablation_reports(synth_dir, _wells(_grid()), environment=SYNTH_ENV)
    real.write_real_ablation_reports(real_dir, _wells(_grid()), environment=REAL_ENV)

    assert not any(p.name.startswith("real_") for p in synth_dir.iterdir())
    assert all(p.name.startswith("real_") for p in real_dir.iterdir())
    # The six required names appear verbatim for a real run.
    names = {p.name for p in real_dir.iterdir()}
    assert {
        "real_alignment_ablation_results.csv",
        "real_alignment_ablation_summary.md",
        "real_alignment_feature_comparison.md",
        "real_protocol_comparison.md",
        "real_spatial_ablation.md",
        "real_well_level_ablation.csv",
    } <= names


def test_filename_and_banner_never_disagree(tmp_path):
    real = _mod("real_ablation_reporting")
    for env in (REAL_ENV, SYNTH_ENV):
        out = tmp_path / ("real" if real.is_real_run(env) else "synth")
        real.write_real_ablation_reports(out, _wells(_grid()), environment=env)
        expected = real.banner_for(env)
        for path in out.glob("*.md"):
            assert expected in path.read_text(), f"{path.name} banner disagrees with its prefix"
