"""Leakage and feature-safety preflight for every A/B/C/D ablation branch.

These tests assert the preflight *catches* leakage, not merely that it runs on
clean input. Each poison case injects one class of forbidden column into the
design matrix a branch would consume and requires the preflight to fail and
`assert_preflight` to raise.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest


def _mod(name):
    key = f"src.{name}"
    return importlib.reload(sys.modules[key]) if key in sys.modules else importlib.import_module(key)


def _reload_chain(*names):
    return tuple(_mod(name) for name in names)


def _task(mount, well_id="TRW006", mode="real"):
    data = _mod("data")
    tasks = _mod("tasks")
    return tasks.make_task(data.load_well(data.discover_wells("train")[well_id]), mode)


def _prior(mount, donors=("TRW007", "TRW008", "TRW009")):
    spatial = _mod("spatial")
    return spatial.SpatialPrior(spatial.SpatialConfig()).fit([_task(mount, w) for w in donors])


# ------------------------------------------------------------- clean pass --

def test_preflight_passes_on_every_branch_with_a_clean_matrix(mount):
    pf = _mod("ablation_preflight")
    report = pf.run_preflight(_task(mount).inputs(), spatial=_prior(mount))
    assert report.passed, [f"{c.branch}: {c.check} -> {c.detail}" for c in report.failures()]
    pf.assert_preflight(report)


def test_preflight_covers_all_four_branches(mount):
    pf, ablation = _reload_chain("ablation_preflight", "ablation")
    report = pf.run_preflight(_task(mount).inputs(), spatial=_prior(mount))
    branches = set(report.checks_frame()["branch"]) - {"all"}
    assert branches == set(ablation.BRANCH_ORDER)
    assert set(report.provenance["branch"]) == set(ablation.BRANCH_ORDER)


def test_preflight_records_every_required_checklist_item(mount):
    pf = _mod("ablation_preflight")
    report = pf.run_preflight(_task(mount).inputs(), spatial=_prior(mount))
    checks = set(report.checks_frame()["check"])
    required = {
        "TVT is never in X",
        "hidden TVT_input is never in X",
        "Typewell Geology and formation markers never reach the Test feature matrix",
        "every feature root is a test-available raw column",
        "manifest whitelist accepts the matrix",
        "alignment features are target-free",
        "public test wells are never used for tuning",
        "spatial features use only fold-training donor wells",
        "query wells are excluded from their own neighbour set",
        "no public leaderboard result is used as a training label",
    }
    assert required <= checks


def test_every_feature_root_is_a_permitted_raw_column(mount):
    pf, manifest = _reload_chain("ablation_preflight", "manifest")
    report = pf.run_preflight(_task(mount).inputs(), spatial=_prior(mount))
    allowed = set(manifest.SAFE_RAW_INFERENCE_SOURCES)
    prov = report.provenance
    manifest_rows = prov[prov["kind"] != "spatial"]
    roots = set()
    for value in manifest_rows["raw_roots"]:
        roots |= {r for r in str(value).split("|") if r}
    assert roots <= allowed
    assert "TVT" not in roots
    assert not (roots & set(manifest.TRAIN_ONLY_MARKERS))
    assert "Typewell Geology" not in roots


# ------------------------------------------------------------ poison cases --

@pytest.mark.parametrize(
    "poison,expect_check",
    [
        ("TVT", "TVT is never in X"),
        ("BUDA", "Typewell Geology and formation markers never reach the Test feature matrix"),
        ("ANCC", "Typewell Geology and formation markers never reach the Test feature matrix"),
        ("Typewell Geology", "Typewell Geology and formation markers never reach the Test feature matrix"),
        ("mystery_unaudited_column", "every feature root is a test-available raw column"),
    ],
)
def test_preflight_rejects_a_leaked_column(mount, monkeypatch, poison, expect_check):
    pf = _mod("ablation_preflight")
    original = pf._columns_for_branch

    def poisoned(branch, task, feats, spatial=None):
        return original(branch, task, feats, spatial=spatial) + [poison]

    monkeypatch.setattr(pf, "_columns_for_branch", poisoned)
    report = pf.run_preflight(_task(mount).inputs())
    assert not report.passed
    failed = {c.check for c in report.failures()}
    assert expect_check in failed
    with pytest.raises(pf.PreflightFailure):
        pf.assert_preflight(report)


def test_preflight_failure_message_names_the_offending_branch(mount, monkeypatch):
    pf = _mod("ablation_preflight")
    original = pf._columns_for_branch
    monkeypatch.setattr(
        pf, "_columns_for_branch",
        lambda b, t, f, spatial=None: original(b, t, f, spatial=spatial) + ["TVT"],
    )
    report = pf.run_preflight(_task(mount).inputs())
    with pytest.raises(pf.PreflightFailure) as exc:
        pf.assert_preflight(report)
    message = str(exc.value)
    assert "ridge_baseline" in message
    assert "no branch was trained" in message


def test_hidden_tvt_input_check_is_real(mount):
    """The hidden-region check must read the actual task array."""
    pf = _mod("ablation_preflight")
    task = _task(mount, mode="masked")
    inp = task.inputs()
    hidden = np.asarray(inp.tvt_known[inp.start : inp.stop])
    assert not np.isfinite(hidden).any(), "fixture precondition: hidden region must be NaN"
    report = pf.run_preflight(inp)
    check = [c for c in report.checks if c.check == "hidden TVT_input is never in X"]
    assert check and all(c.passed for c in check)
    assert "0 finite values" in check[0].detail


def test_branch_configuration_mismatch_is_detected(mount, monkeypatch):
    """A branch that silently gained alignment features must fail the check."""
    pf, ablation = _reload_chain("ablation_preflight", "ablation")
    features = _mod("features")
    original = pf._columns_for_branch

    def wrong(branch, task, feats, spatial=None):
        cols = original(branch, task, feats, spatial=spatial)
        if branch == ablation.BRANCH_A:  # must NOT have alignment features
            cols = cols + list(features.ALIGNMENT_FEATURES)
        return cols

    monkeypatch.setattr(pf, "_columns_for_branch", wrong)
    report = pf.run_preflight(_task(mount).inputs())
    failed = [c for c in report.failures() if c.check == "branch matrix matches its declared configuration"]
    assert failed and failed[0].branch == ablation.BRANCH_A


def test_spatial_check_fails_when_the_query_well_is_a_donor(mount):
    """Self-donation is the exact spatial leak the guard exists to stop."""
    pf = _mod("ablation_preflight")
    task = _task(mount, "TRW006")
    # Fit the prior on a donor set that (wrongly) includes the queried well.
    prior = _prior(mount, donors=("TRW006", "TRW007", "TRW008"))
    report = pf.run_preflight(task.inputs(), spatial=prior)
    failed = {c.check for c in report.failures()}
    assert "query wells are excluded from their own neighbour set" in failed
    with pytest.raises(pf.PreflightFailure):
        pf.assert_preflight(report)


# ---------------------------------------------------------------- writing --

def test_preflight_report_is_written(mount, tmp_path):
    pf = _mod("ablation_preflight")
    report = pf.run_preflight(_task(mount).inputs(), spatial=_prior(mount))
    written = {p.name for p in pf.write_preflight(report, tmp_path)}
    assert written == {
        "real_ablation_preflight.csv",
        "real_ablation_preflight_checks.csv",
        "real_ablation_preflight.md",
    }
    text = (tmp_path / "real_ablation_preflight.md").read_text()
    assert "ALL CHECKS PASSED" in text
    prov = pd.read_csv(tmp_path / "real_ablation_preflight.csv")
    assert {"branch", "feature", "raw_roots"} <= set(prov.columns)


def test_written_preflight_marks_failures_visibly(mount, tmp_path, monkeypatch):
    pf = _mod("ablation_preflight")
    original = pf._columns_for_branch
    monkeypatch.setattr(
        pf, "_columns_for_branch",
        lambda b, t, f, spatial=None: original(b, t, f, spatial=spatial) + ["TVT"],
    )
    report = pf.run_preflight(_task(mount).inputs())
    pf.write_preflight(report, tmp_path)
    text = (tmp_path / "real_ablation_preflight.md").read_text()
    assert "FAILED" in text
    assert "**FAIL**" in text
