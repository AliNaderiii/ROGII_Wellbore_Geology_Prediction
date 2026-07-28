"""Ridge alignment/spatial feature ablation: wiring, pairing and the verdict.

These tests assert the historical ablation is a *valid* comparison — same
folds, same Ridge model and protocols kept separate — and that its completed
selection is now the no-alignment default. The implementation remains opt-in.
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
    """Reload modules dependency-first and return them.

    ``src.ablation`` binds ``SpatialPrior`` and ``RidgeBaseline`` by value at
    import time, so reloading a dependency *after* ``src.ablation`` would leave
    the ablation holding the previous class object and any spy patched onto the
    new one would never fire.
    """
    return tuple(_mod(name) for name in names)


# ------------------------------------------------- the feature-set switch --

def test_baseline_feature_set_is_untouched():
    """The shipped Ridge feature list must not change because of the ablation."""
    features = _mod("features")
    assert features.feature_columns() == list(features.FEATURE_COLUMNS)
    assert features.feature_columns(alignment_features=True) == list(features.FEATURE_COLUMNS)
    assert set(features.ALIGNMENT_FEATURES) <= set(features.FEATURE_COLUMNS)


def test_dropping_alignment_features_removes_exactly_those_four():
    features = _mod("features")
    full = features.feature_columns(alignment_features=True)
    narrow = features.feature_columns(alignment_features=False)
    assert set(full) - set(narrow) == set(features.ALIGNMENT_FEATURES)
    assert len(narrow) == len(full) - 4
    # Order of the surviving columns is preserved.
    assert narrow == [c for c in full if c not in set(features.ALIGNMENT_FEATURES)]


def test_ridge_defaults_to_real_770_well_selected_matrix(mount):
    """The post-ablation default excludes alignment; opt-in restores it."""
    features = _mod("features")
    baselines = _mod("baselines")
    data = _mod("data")
    tasks = _mod("tasks")
    files = data.discover_wells("train")
    task = tasks.make_task(data.load_well(files["TRW001"]), "real")
    inp = task.inputs()
    feats = features.build_features(inp)

    default = baselines.BASELINES["ridge"]()
    assert default.alignment_features is False
    cols = list(default._features(inp, feats).columns)
    assert cols == features.feature_columns(alignment_features=False)
    assert not (set(features.ALIGNMENT_FEATURES) & set(cols))

    opt_in = baselines.BASELINES["ridge"](alignment_features=True)
    assert list(opt_in._features(inp, feats).columns) == list(features.FEATURE_COLUMNS)


def test_narrow_ridge_still_fits_and_predicts(mount):
    baselines = _mod("baselines")
    data = _mod("data")
    tasks = _mod("tasks")
    files = data.discover_wells("train")
    train = [tasks.make_task(data.load_well(files[w]), "real") for w in ("TRW006", "TRW007")]
    model = baselines.BASELINES["ridge"](alignment_features=False).fit(train)
    inp = tasks.make_task(data.load_well(files["TRW008"]), "real").inputs()
    pred = model.predict(inp)
    assert pred.shape == (inp.n_predict,)
    assert np.isfinite(pred).all()
    assert not (set(model.feature_names_) & set(_mod("features").ALIGNMENT_FEATURES))


# ------------------------------------------------------- branch definitions --

def test_the_four_branches_form_a_clean_2x2():
    ablation = _mod("ablation")
    assert len(ablation.BRANCH_ORDER) == 4
    spec = {ablation.BRANCH_SPEC[b] for b in ablation.BRANCH_ORDER}
    assert spec == {(False, False), (True, False), (False, True), (True, True)}
    # Branch B is the reference and is the alignment-only, no-spatial cell.
    assert ablation.BASELINE_BRANCH == ablation.BRANCH_B
    assert ablation.BRANCH_SPEC[ablation.BRANCH_B] == (True, False)


def test_branch_factories_build_the_configured_ridge():
    ablation = _mod("ablation")
    for branch in ablation.BRANCH_ORDER:
        model = ablation.branch_factory(branch)()
        assert model.name == "ridge"
        assert model.alignment_features is ablation.branch_uses_alignment(branch)
        assert model.uses_spatial is False  # no prior passed here
    with pytest.raises(KeyError):
        ablation.branch_factory("not_a_branch")


# ------------------------------------------------------------ the live run --

def _folds_and_builder(mount, n_splits=2):
    data = _mod("data")
    tasks = _mod("tasks")
    validation = _mod("validation")
    files = data.discover_wells("train")
    ids = [w for w in ("TRW006", "TRW007", "TRW008", "TRW009")]

    def task_builder(well_ids, mode):
        out, skipped = [], []
        for wid in well_ids:
            try:
                out.append(tasks.make_task(data.load_well(files[wid]), mode))
            except Exception as exc:
                skipped.append((wid, str(exc)))
        return out, skipped

    return validation.make_group_folds(ids, n_splits=n_splits, seed=0), task_builder


def test_ablation_runs_all_branches_under_both_protocols(mount):
    ablation = _mod("ablation")
    validation = _mod("validation")
    folds, builder = _folds_and_builder(mount)
    rows = []
    for protocol, mode in ((validation.PROTOCOL_A, "masked"), (validation.PROTOCOL_B, "real")):
        run = ablation.run_ablation_protocol(
            protocol=protocol, mode=mode, folds=folds, task_builder=builder
        )
        rows += run.well_results
    well = pd.DataFrame([r.__dict__ for r in rows])
    assert set(well["model"]) == set(ablation.BRANCH_ORDER)
    assert set(well["protocol"]) == {validation.PROTOCOL_A, validation.PROTOCOL_B}
    # Every branch scored every well in each protocol — that is what makes the
    # deltas paired rather than a comparison of different well sets.
    for protocol, group in well.groupby("protocol"):
        counts = group.groupby("well_id")["model"].nunique()
        assert set(counts) == {4}


def test_ablation_never_scores_a_well_it_was_fitted_on(mount, monkeypatch):
    """Cross-fitting spy: fitted IDs and scored IDs must stay disjoint."""
    baselines, validation, ablation = _reload_chain("baselines", "validation", "ablation")
    folds, builder = _folds_and_builder(mount)

    fitted: list[set] = []
    original_fit = baselines.RidgeBaseline.fit

    def spy_fit(self, tasks, **kw):
        fitted.append({t.well_id for t in tasks})
        return original_fit(self, tasks, **kw)

    monkeypatch.setattr(baselines.RidgeBaseline, "fit", spy_fit)
    run = ablation.run_ablation_protocol(
        protocol=validation.PROTOCOL_B, mode="real", folds=folds, task_builder=builder
    )
    scored_by_fold: dict[int, set] = {}
    for r in run.well_results:
        scored_by_fold.setdefault(r.fold, set()).add(r.well_id)
    assert fitted
    for fold in folds:
        scored = scored_by_fold.get(fold.index, set())
        assert not (scored & set(fold.train_ids))
        assert scored <= set(fold.valid_ids)


def test_spatial_branches_exclude_validation_wells_as_donors(mount, monkeypatch):
    spatial, validation, ablation = _reload_chain("spatial", "validation", "ablation")
    folds, builder = _folds_and_builder(mount)
    seen = []
    original = spatial.SpatialPrior.assert_disjoint

    def spy(self, well_ids):
        seen.append(set(well_ids))
        return original(self, well_ids)

    monkeypatch.setattr(spatial.SpatialPrior, "assert_disjoint", spy)
    ablation.run_ablation_protocol(
        protocol=validation.PROTOCOL_B, mode="real", folds=folds, task_builder=builder,
        branches=(ablation.BRANCH_C, ablation.BRANCH_D),
    )
    assert seen, "the donor guard was never invoked for a spatial branch"


def test_no_spatial_prior_is_built_when_no_branch_needs_one(mount, monkeypatch):
    spatial, validation, ablation = _reload_chain("spatial", "validation", "ablation")
    folds, builder = _folds_and_builder(mount)
    built = []
    original = spatial.SpatialPrior.fit

    def spy(self, tasks):
        built.append(len(tasks))
        return original(self, tasks)

    monkeypatch.setattr(spatial.SpatialPrior, "fit", spy)
    ablation.run_ablation_protocol(
        protocol=validation.PROTOCOL_B, mode="real", folds=folds, task_builder=builder,
        branches=(ablation.BRANCH_A, ablation.BRANCH_B),
    )
    assert built == []


# ------------------------------------------------------------- the summary --

def _summary_frame(values: dict) -> pd.DataFrame:
    """Per-well rows whose weighted RMSE equals the requested value exactly."""
    rows = []
    for (protocol, branch), rmse in values.items():
        for well in ("w0", "w1"):
            rows.append(
                {
                    "protocol": protocol, "model": branch, "well_id": well,
                    "n_points": 100, "sse": 100 * rmse**2, "rmse": rmse,
                }
            )
    return pd.DataFrame(rows)


def test_summary_deltas_are_taken_against_branch_b():
    ablation = _mod("ablation")
    values = {
        ("same_well_masked", ablation.BRANCH_A): 2.0,
        ("same_well_masked", ablation.BRANCH_B): 3.0,
        ("same_well_masked", ablation.BRANCH_C): 4.0,
        ("same_well_masked", ablation.BRANCH_D): 5.0,
    }
    summary = ablation.summarize_ablation(_summary_frame(values))
    by = summary.set_index("branch")["delta_global_rmse_vs_baseline"].to_dict()
    assert by[ablation.BRANCH_B] == pytest.approx(0.0)
    assert by[ablation.BRANCH_A] == pytest.approx(-1.0)
    assert by[ablation.BRANCH_C] == pytest.approx(1.0)
    assert by[ablation.BRANCH_D] == pytest.approx(2.0)


def test_summary_keeps_protocols_separate():
    ablation = _mod("ablation")
    values = {
        ("same_well_masked", ablation.BRANCH_A): 2.0,
        ("same_well_masked", ablation.BRANCH_B): 3.0,
        ("unseen_well", ablation.BRANCH_A): 9.0,
        ("unseen_well", ablation.BRANCH_B): 6.0,
    }
    summary = ablation.summarize_ablation(_summary_frame(values))
    same = summary[summary["protocol"] == "same_well_masked"].set_index("branch")
    unseen = summary[summary["protocol"] == "unseen_well"].set_index("branch")
    # Each protocol is referenced to its own branch-B value; a shared or
    # averaged reference would give -1.5 / +1.5 here.
    assert same.loc[ablation.BRANCH_A, "delta_global_rmse_vs_baseline"] == pytest.approx(-1.0)
    assert unseen.loc[ablation.BRANCH_A, "delta_global_rmse_vs_baseline"] == pytest.approx(3.0)


def test_summary_compares_only_wells_scored_by_every_branch():
    ablation = _mod("ablation")
    rows = []
    for branch in ablation.BRANCH_ORDER:
        wells = ("w0", "w1") if branch != ablation.BRANCH_A else ("w0",)
        for well in wells:
            rows.append(
                {"protocol": "unseen_well", "model": branch, "well_id": well,
                 "n_points": 100, "sse": 100.0, "rmse": 1.0}
            )
    summary = ablation.summarize_ablation(pd.DataFrame(rows))
    # w1 is dropped from every branch, not just the branch that missed it.
    assert set(summary["n_wells"]) == {1}


# -------------------------------------------------------------- the verdict --

def test_verdict_isolates_the_alignment_features_in_both_contrasts():
    ablation = _mod("ablation")
    values = {
        ("unseen_well", ablation.BRANCH_A): 2.0,   # no align, no spatial
        ("unseen_well", ablation.BRANCH_B): 1.5,   # + align
        ("unseen_well", ablation.BRANCH_C): 1.8,   # no align, + spatial
        ("unseen_well", ablation.BRANCH_D): 1.4,   # + align, + spatial
    }
    verdict = ablation.alignment_feature_verdict(ablation.summarize_ablation(_summary_frame(values)))
    assert set(verdict["spatial_context"]) == {"no_spatial", "with_spatial"}
    assert verdict["delta_global_rmse"].tolist() == pytest.approx([-0.5, -0.4])
    assert verdict["alignment_features_help"].all()


def test_recommendation_keeps_features_only_when_both_protocols_improve():
    ablation = _mod("ablation")
    values = {}
    for protocol in ("same_well_masked", "unseen_well"):
        values[(protocol, ablation.BRANCH_A)] = 2.0
        values[(protocol, ablation.BRANCH_B)] = 1.5
        values[(protocol, ablation.BRANCH_C)] = 1.8
        values[(protocol, ablation.BRANCH_D)] = 1.4
    summary = ablation.summarize_ablation(_summary_frame(values))
    rec = ablation.alignment_feature_recommendation(ablation.alignment_feature_verdict(summary))
    assert rec["decision"] == "keep_as_features"
    assert rec["n_helping"] == rec["n_contrasts"] == 4


def test_recommendation_removes_features_on_a_mixed_result():
    ablation = _mod("ablation")
    values = {
        ("same_well_masked", ablation.BRANCH_A): 2.0,
        ("same_well_masked", ablation.BRANCH_B): 1.5,   # helps
        ("same_well_masked", ablation.BRANCH_C): 1.8,
        ("same_well_masked", ablation.BRANCH_D): 1.9,   # hurts
        ("unseen_well", ablation.BRANCH_A): 2.0,
        ("unseen_well", ablation.BRANCH_B): 1.5,
        ("unseen_well", ablation.BRANCH_C): 1.8,
        ("unseen_well", ablation.BRANCH_D): 1.4,
    }
    summary = ablation.summarize_ablation(_summary_frame(values))
    rec = ablation.alignment_feature_recommendation(ablation.alignment_feature_verdict(summary))
    assert rec["decision"] == "remove_from_next_baseline"


def test_recommendation_requires_both_protocols():
    """A one-protocol improvement is not evidence enough to keep the features."""
    ablation = _mod("ablation")
    values = {
        ("unseen_well", ablation.BRANCH_A): 2.0,
        ("unseen_well", ablation.BRANCH_B): 1.5,
        ("unseen_well", ablation.BRANCH_C): 1.8,
        ("unseen_well", ablation.BRANCH_D): 1.4,
    }
    summary = ablation.summarize_ablation(_summary_frame(values))
    rec = ablation.alignment_feature_recommendation(ablation.alignment_feature_verdict(summary))
    assert rec["decision"] == "remove_from_next_baseline"
    assert rec["protocols_covered"] == ["unseen_well"]


def test_empty_input_yields_an_undetermined_decision_not_a_default():
    ablation = _mod("ablation")
    assert ablation.summarize_ablation(pd.DataFrame()).empty
    rec = ablation.alignment_feature_recommendation(pd.DataFrame())
    assert rec["decision"] == "undetermined"


# ------------------------------------------------------------- the report --

def test_ablation_report_is_written_from_per_well_rows(tmp_path):
    ablation = _mod("ablation")
    real = _mod("real_reporting")
    values = {}
    for protocol in ("same_well_masked", "unseen_well"):
        values[(protocol, ablation.BRANCH_A)] = 2.0
        values[(protocol, ablation.BRANCH_B)] = 1.5
        values[(protocol, ablation.BRANCH_C)] = 1.8
        values[(protocol, ablation.BRANCH_D)] = 1.4
    _summary_frame(values).to_csv(tmp_path / "alignment_spatial_ablation_wells.csv", index=False)

    written = {p.name for p in real.write_alignment_spatial_ablation(tmp_path)}
    assert {
        "alignment_spatial_ablation.csv",
        "alignment_feature_verdict.csv",
        "alignment_spatial_ablation.md",
    } == written
    text = (tmp_path / "alignment_spatial_ablation.md").read_text()
    assert "former Ridge baseline" in text
    assert "Keep" in text
    assert "never averaged" in text


def test_no_ablation_report_is_invented_when_the_run_did_not_happen(tmp_path):
    real = _mod("real_reporting")
    assert real.write_alignment_spatial_ablation(tmp_path) == []
    assert not (tmp_path / "alignment_spatial_ablation.md").exists()


# --------------------------------------------------------- CLI + caching --

def _cli(mount, tmp_path, *extra, reports="rep", cache="cache"):
    """Invoke the ablation entrypoint in-process and return its exit code."""
    import scripts.run_feature_ablation as runner

    return runner.main([
        "--n-splits", "2", "--max-wells", "4", "--quiet",
        "--reports-dir", str(tmp_path / reports),
        "--cache-dir", str(tmp_path / cache),
        *extra,
    ])


def test_cli_supports_every_required_option():
    """The brief fixes the exact flags the Kaggle command line uses."""
    import scripts.run_feature_ablation as runner

    # Read the CLI's own --help output rather than a hand-kept list.
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        runner.main(["--help"])
    help_text = buf.getvalue()
    for flag in ("--max-wells", "--n-splits", "--cache-dir", "--reports-dir",
                 "--device", "--clear-cache", "--spatial"):
        assert flag in help_text, f"{flag} is not exposed by the CLI"


def test_cli_writes_all_six_required_reports(mount, tmp_path):
    assert _cli(mount, tmp_path) == 0
    out = tmp_path / "rep"
    for name in (
        "synthetic_alignment_ablation_results.csv",
        "synthetic_alignment_ablation_summary.md",
        "synthetic_alignment_feature_comparison.md",
        "synthetic_protocol_comparison.md",
        "synthetic_spatial_ablation.md",
        "synthetic_well_level_ablation.csv",
        "synthetic_ablation_preflight.md",
        "synthetic_ablation_run_environment.json",
    ):
        assert (out / name).exists(), f"{name} was not written"


def test_cli_run_on_a_fixture_is_never_labelled_real(mount, tmp_path):
    """A non-audited mount must not produce REAL KAGGLE VALIDATION reports."""
    real = _mod("real_ablation_reporting")
    assert _cli(mount, tmp_path) == 0
    text = (tmp_path / "rep" / "synthetic_alignment_ablation_summary.md").read_text()
    assert real.SYNTHETIC_BANNER in text
    assert not text.startswith(f"> # {real.REAL_BANNER}")


def test_cli_cache_is_written_then_hit(mount, tmp_path):
    import json

    assert _cli(mount, tmp_path, reports="r1") == 0
    first = json.loads((tmp_path / "r1" / "synthetic_ablation_run_environment.json").read_text())
    assert first["cache_writes"] > 0
    assert first["cache_hits"] == 0

    assert _cli(mount, tmp_path, reports="r2") == 0
    second = json.loads((tmp_path / "r2" / "synthetic_ablation_run_environment.json").read_text())
    assert second["cache_hits"] > 0
    assert second["cache_writes"] == 0


def test_cli_clear_cache_forces_recomputation(mount, tmp_path):
    import json

    assert _cli(mount, tmp_path, reports="r1") == 0
    assert _cli(mount, tmp_path, "--clear-cache", reports="r2") == 0
    second = json.loads((tmp_path / "r2" / "synthetic_ablation_run_environment.json").read_text())
    assert second["cache_hits"] == 0
    assert second["cache_writes"] > 0


def test_cli_no_spatial_restricts_to_branches_a_and_b(mount, tmp_path):
    ablation = _mod("ablation")
    assert _cli(mount, tmp_path, "--no-spatial") == 0
    results = pd.read_csv(tmp_path / "rep" / "synthetic_alignment_ablation_results.csv")
    assert set(results["branch"]) == {ablation.BRANCH_A, ablation.BRANCH_B}


def test_cli_records_runtime_and_peak_memory(mount, tmp_path):
    import json

    assert _cli(mount, tmp_path) == 0
    env = json.loads((tmp_path / "rep" / "synthetic_ablation_run_environment.json").read_text())
    assert env["runtime_seconds"] > 0
    assert env["peak_rss_mb"] > 0
    assert env["preflight_checks_passed"] == env["preflight_checks_total"]
    assert env["device_selected"] in {"cpu", "gpu"}
    # The fixture deliberately contains wells too short to mask a suffix, so a
    # non-zero count is expected here; what matters is that every recorded
    # failure is a task-construction skip, never a fit or predict error.
    failures = pd.read_csv(tmp_path / "rep" / "alignment_ablation_failures.csv")
    assert env["failure_count"] == len(failures)
    if len(failures):
        assert set(failures["stage"]) == {"task"}


def test_cli_expect_wells_guards_against_a_partial_mount(mount, tmp_path):
    """A mismatched eligible count must abort before any model is fitted."""
    with pytest.raises(SystemExit) as exc:
        _cli(mount, tmp_path, "--expect-wells", "770")
    assert "expected 770 eligible wells" in str(exc.value)
    assert not (tmp_path / "rep" / "synthetic_alignment_ablation_results.csv").exists()


def test_cached_alignment_features_never_store_a_target(mount, tmp_path):
    """The cache API must refuse target-like keys outright."""
    cache_mod = _mod("cache")
    cache = cache_mod.FeatureCache(tmp_path / "c")
    with pytest.raises(ValueError):
        cache.put("k", tvt=np.zeros(3))
    with pytest.raises(ValueError):
        cache.put("k", target_values=np.zeros(3))


def test_cached_alignment_matches_the_uncached_computation(mount, tmp_path):
    """A cache hit must return exactly what recomputation would."""
    features = _mod("features")
    cache_mod = _mod("cache")
    data = _mod("data")
    tasks = _mod("tasks")
    files = data.discover_wells("train")
    inp = tasks.make_task(data.load_well(files["TRW006"]), "real").inputs()
    ref = features.TypewellReference(inp.tw_tvt, inp.tw_gr)
    gr = features.gr_features(inp)
    missing = gr["gr_is_missing"] > 0.5

    direct = features.alignment_features(inp, ref, gr["gr_z"], gr_missing=missing)
    cache = cache_mod.FeatureCache(tmp_path / "c")
    ctx = {"dataset_version": "test", "fold": 0, "protocol": "unseen_well"}
    miss = features.cached_alignment_features(
        inp, ref, gr["gr_z"], gr_missing=missing, cache=cache, cache_context=ctx)
    hit = features.cached_alignment_features(
        inp, ref, gr["gr_z"], gr_missing=missing, cache=cache, cache_context=ctx)
    assert miss["_align_cache_hit"] is False
    assert hit["_align_cache_hit"] is True
    for key in ("align_tvt", "align_score", "align_shift", "align_gradient"):
        np.testing.assert_allclose(direct[key], hit[key], equal_nan=True)
    assert hit["_align_ok"] == direct["_align_ok"]


def test_alignment_cache_key_separates_the_two_protocols(mount, tmp_path):
    """A masked boundary and a real suffix must never share an artifact."""
    features = _mod("features")
    cache_mod = _mod("cache")
    data = _mod("data")
    tasks = _mod("tasks")
    files = data.discover_wells("train")
    well = data.load_well(files["TRW006"])
    cache = cache_mod.FeatureCache(tmp_path / "c")
    for mode, protocol in (("real", "unseen_well"), ("masked", "same_well_masked")):
        inp = tasks.make_task(well, mode).inputs()
        ref = features.TypewellReference(inp.tw_tvt, inp.tw_gr)
        gr = features.gr_features(inp)
        out = features.cached_alignment_features(
            inp, ref, gr["gr_z"], gr_missing=gr["gr_is_missing"] > 0.5,
            cache=cache, cache_context={"dataset_version": "t", "protocol": protocol},
        )
        assert out["_align_cache_hit"] is False, f"{mode} wrongly reused another boundary"
