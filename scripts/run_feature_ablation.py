"""Run the A/B/C/D Ridge feature ablation under both validation protocols.

    python scripts/run_feature_ablation.py                     # full run
    python scripts/run_feature_ablation.py --max-wells 60      # quick pass
    python scripts/run_feature_ablation.py --label "SYNTHETIC ..."

Branches, all fitted through the **existing** Ridge model:

    A  ridge_no_align         alignment features removed, no spatial
    B  ridge_baseline         the current Ridge baseline (delta reference)
    C  ridge_spatial_only     alignment features removed, + spatial
    D  ridge_align_spatial    alignment features + spatial

The shipped Ridge baseline is not modified: branch B uses exactly
``FEATURE_COLUMNS``, and the narrower branches are reached only through the
explicit ``alignment_features=False`` switch added for this ablation.

Both protocols (``same_well_masked`` and ``unseen_well``) are always run and
are never averaged together. Outputs (into REPORTS_DIR):

    alignment_spatial_ablation_wells.csv   one row per (branch, protocol, well)
    alignment_spatial_ablation.csv         one row per (protocol, branch) + delta vs B
    alignment_feature_verdict.csv          paired A-vs-B and C-vs-D contrasts
    alignment_spatial_ablation.md          the report and the keep/remove decision
    alignment_ablation_failures.csv        every task/fit/predict failure
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:  # loose-file execution
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap
bootstrap()

import numpy as np
import pandas as pd

from src.ablation import (
    BRANCH_ORDER,
    BRANCH_SPEC,
    alignment_feature_verdict,
    preregistered_decision,
    preregistered_verdict,
    run_ablation_protocol,
    summarize_ablation,
)
from src.ablation_preflight import assert_preflight, run_preflight, write_preflight
from src.cache import FeatureCache
from src.data import discover_wells, load_well
from src.model_status import assert_not_rejected
from src.paths import ensure_reports_dir, require_competition_data
from src.real_ablation_reporting import (
    file_prefix as report_file_prefix,
    write_real_ablation_reports,
)
from src.real_reporting import write_alignment_spatial_ablation
from src.resources import detect_resources
from src.spatial import SpatialConfig, SpatialPrior
# Imported as a module, not as symbols. A Kaggle notebook cell (or a test
# fixture) that reloads ``src.tasks`` replaces the class object; a symbol bound
# at import time would then no longer match the exception actually raised, and
# a routine "prefix too short to mask" skip would abort the whole run.
import src.tasks as tasks_module
from src.validation import (
    PROTOCOL_A,
    PROTOCOL_B,
    assert_no_blocked_wells,
    filter_blocked,
    make_group_folds,
)


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-wells", type=int, default=None,
                    help="evaluate a random subset of this many eligible wells "
                         "(omit for the full eligible universe)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto",
                    help="compute device profile; recorded in the run environment "
                         "and used as part of the feature-cache key")
    ap.add_argument("--cache-dir", default=None,
                    help="target-free feature cache directory; defaults to "
                         "/kaggle/working/feature_ablation_cache on Kaggle and to "
                         "<reports-dir>/../feature_ablation_cache elsewhere")
    ap.add_argument("--clear-cache", action="store_true",
                    help="delete every cached artifact before running")
    ap.add_argument("--spatial", dest="spatial", action="store_true", default=None,
                    help="explicitly include the spatial branches C and D (the default)")
    ap.add_argument("--no-spatial", dest="spatial", action="store_false",
                    help="restrict the run to the non-spatial branches A and B")
    ap.add_argument("--spatial-k", type=int, default=12)
    ap.add_argument("--spatial-radius", type=float, default=6000.0)
    ap.add_argument("--branches", default=",".join(BRANCH_ORDER))
    ap.add_argument("--expect-wells", type=int, default=None,
                    help="fail unless exactly this many eligible wells are found "
                         "(use 770 for the audited real mount)")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--label", default="", help="banner text stamped on the report (e.g. SYNTHETIC)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    verbose = not args.quiet
    if args.reports_dir:
        os.environ["ROGII_REPORTS_DIR"] = args.reports_dir
        import importlib

        import src.paths

        importlib.reload(src.paths)
        reports_dir = src.paths.ensure_reports_dir()
    else:
        reports_dir = ensure_reports_dir()

    require_competition_data()

    branches = tuple(b.strip() for b in args.branches.split(",") if b.strip())
    unknown = [b for b in branches if b not in BRANCH_ORDER]
    if unknown:
        raise SystemExit(f"unknown branch(es): {unknown}; known: {list(BRANCH_ORDER)}")
    # --spatial / --no-spatial narrows the branch list rather than silently
    # changing what a branch means.
    if args.spatial is False:
        branches = tuple(b for b in branches if not BRANCH_SPEC[b][1])
        if not branches:
            raise SystemExit("--no-spatial removed every requested branch")
    # Every branch is a Ridge configuration. A rejected model must not be able
    # to enter the ablation and thereby the promotion discussion.
    assert_not_rejected(branches, context="feature ablation branches")

    resources = detect_resources(args.device)
    os.environ["ROGII_LIGHTGBM_DEVICE"] = resources.selected
    cache_dir = args.cache_dir
    if cache_dir is None:
        kaggle_working = Path("/kaggle/working")
        cache_dir = (
            kaggle_working / "feature_ablation_cache"
            if kaggle_working.exists()
            else reports_dir.parent / "feature_ablation_cache"
        )
    cache = FeatureCache(cache_dir)
    if args.clear_cache:
        cache.clear()
    dataset_version = os.environ.get("ROGII_DATASET_VERSION", "rogii-mounted-v1")

    t_start = time.perf_counter()
    files = discover_wells("train")

    def loader(well_id):
        entry = files.get(well_id)
        if entry is None or entry.horizontal is None:
            return None
        try:
            return load_well(entry)
        except Exception:
            return None

    def task_builder(ids, mode):
        tasks, skipped = [], []
        for wid in ids:
            well = loader(wid)
            if well is None:
                skipped.append((wid, "load_failed"))
                continue
            try:
                tasks.append(tasks_module.make_task(well, mode))
            except tasks_module.TaskConstructionError as exc:
                skipped.append((wid, str(exc).split(":")[-1].strip()))
        return tasks, skipped

    discovered = sorted(files)
    universe = filter_blocked(discovered)
    assert_no_blocked_wells(universe, context="ablation universe")
    excluded = sorted(set(discovered) - set(universe))
    # The eligible universe is the audited count *after* removing the three
    # visible public test wells. Checking it here stops a partial mount from
    # silently producing a smaller, non-comparable run.
    if args.expect_wells is not None and len(universe) != args.expect_wells:
        raise SystemExit(
            f"expected {args.expect_wells} eligible wells, found {len(universe)} "
            f"({len(discovered)} discovered, {len(excluded)} blocked). Refusing to run: "
            "a partial mount would produce misleading metrics."
        )
    n_eligible = len(universe)
    if args.max_wells:
        rng = np.random.default_rng(args.seed)
        pick = rng.permutation(len(universe))[: args.max_wells]
        universe = sorted(universe[i] for i in pick)
    assert_no_blocked_wells(universe, context="ablation subsampled universe")

    folds = make_group_folds(universe, n_splits=args.n_splits, seed=args.seed)
    spatial_config = SpatialConfig(k=args.spatial_k, radius=args.spatial_radius)

    if verbose:
        print("=" * 72)
        print("Ridge alignment / spatial feature ablation — cross-fitted by well ID")
        if args.label:
            print(f"*** {args.label} ***")
        print("=" * 72)
        print(f"reports dir : {reports_dir}")
        print(f"cache dir   : {cache.directory}")
        print(f"device      : {resources.selected} ({resources.gpu_name or 'no GPU'})")
        print(f"CPU/RAM     : {resources.cpu_count} cores / {resources.ram_mb or 'unknown'} MB")
        print(f"discovered  : {len(discovered)} train wells")
        print(f"blocked     : {excluded or 'none'}")
        print(f"eligible    : {n_eligible} wells")
        print(f"evaluated   : {len(universe)} wells"
              + (f" (--max-wells {args.max_wells})" if args.max_wells else " (full universe)"))
        print(f"branches    : {list(branches)}")
        print(f"folds       : {args.n_splits}")

    # ---------------------------------------------------- preflight ------
    # Verify the leakage checklist against the real per-branch design matrices
    # before a single model is fitted.
    probe_tasks, _ = task_builder(folds[0].train_ids[: min(8, len(folds[0].train_ids))], "real")
    if not probe_tasks:
        raise SystemExit("Could not build a probe task for the preflight; refusing to train.")
    probe_prior = None
    if any(BRANCH_SPEC[b][1] for b in branches) and len(probe_tasks) > 1:
        probe_prior = SpatialPrior(spatial_config).fit(probe_tasks[1:])
    preflight = run_preflight(probe_tasks[0].inputs(), spatial=probe_prior, branches=branches)
    # The filename prefix is decided by the same evidence rule the reports use,
    # so a synthetic or partial run never leaves a file named `real_*`.
    file_prefix = report_file_prefix(
        {"n_train_wells_discovered": len(discovered), "n_eligible_wells": n_eligible}
    )
    preflight_files = write_preflight(
        preflight, reports_dir, label=args.label, prefix=file_prefix
    )
    if verbose:
        checks = preflight.checks_frame()
        print(f"\npreflight   : {int(checks['passed'].sum())}/{len(checks)} checks passed")
    assert_preflight(preflight)

    well_rows, failures, fold_records = [], [], []
    for protocol, mode in ((PROTOCOL_A, "masked"), (PROTOCOL_B, "real")):
        if verbose:
            print(f"\n[{protocol}]")
        run = run_ablation_protocol(
            protocol=protocol,
            mode=mode,
            folds=folds,
            task_builder=task_builder,
            branches=branches,
            spatial_config=spatial_config,
            verbose=verbose,
            alignment_cache=cache,
            cache_context={"dataset_version": dataset_version},
        )
        well_rows += run.well_results
        failures += run.failures
        fold_records += run.fold_records

    if not well_rows:
        raise SystemExit(
            "No ablation results were produced. Check that the competition data is "
            "mounted and that wells form usable tasks."
        )

    well_df = pd.DataFrame([r.__dict__ for r in well_rows])
    assert_no_blocked_wells(well_df["well_id"], context="ablation well-level results")
    wells_csv = reports_dir / "alignment_spatial_ablation_wells.csv"
    well_df.to_csv(wells_csv, index=False)
    failures_df = pd.DataFrame(failures, columns=["stage", "model", "well_id", "error"])
    failures_df.to_csv(reports_dir / "alignment_ablation_failures.csv", index=False)
    folds_df = pd.DataFrame(fold_records)
    folds_df.to_csv(reports_dir / "alignment_ablation_folds.csv", index=False)

    written = write_alignment_spatial_ablation(reports_dir)
    if args.label:
        report = reports_dir / "alignment_spatial_ablation.md"
        if report.exists():
            report.write_text(
                f"> **{args.label}**\n\n" + report.read_text(encoding="utf-8"), encoding="utf-8"
            )

    summary = summarize_ablation(well_df)
    verdict = alignment_feature_verdict(summary)
    decision = preregistered_decision(summary)
    prereg = preregistered_verdict(decision)

    runtime_total = time.perf_counter() - t_start
    environment = {
        "validation": "REAL KAGGLE VALIDATION" if not args.label else args.label,
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "runtime_seconds": round(runtime_total, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "n_train_wells_discovered": len(discovered),
        "n_blocked_wells_excluded": len(excluded),
        "blocked_well_ids": sorted(excluded),
        "n_eligible_wells": n_eligible,
        "n_wells_evaluated": len(universe),
        "max_wells": args.max_wells,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "branches": list(branches),
        "spatial_k": args.spatial_k,
        "spatial_radius": args.spatial_radius,
        "device_requested": args.device,
        "device_selected": resources.selected,
        "gpu_name": resources.gpu_name,
        "cpu_count": resources.cpu_count,
        "ram_mb": resources.ram_mb,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "failure_count": int(len(failures_df)),
        "preflight_checks_passed": int(preflight.checks_frame()["passed"].sum()),
        "preflight_checks_total": int(len(preflight.checks_frame())),
        **cache.report(),
    }
    (reports_dir / f"{file_prefix}ablation_run_environment.json").write_text(
        json.dumps(environment, indent=2, default=str), encoding="utf-8"
    )

    real_written = write_real_ablation_reports(
        reports_dir, well_df, environment=environment, failures=failures_df
    )
    written = list(written) + list(preflight_files) + list(real_written)

    if verbose:
        print(f"\nCompleted in {runtime_total:.1f}s, peak RSS {environment['peak_rss_mb']} MB")
        print(f"cache: {cache.stats.hits} hits / {cache.stats.misses} misses / "
              f"{cache.stats.writes} writes")
        print(f"failures: {len(failures_df)}")
        print("\nDelta vs the current Ridge baseline (branch B):")
        cols = ["protocol", "branch", "n_wells", "global_rmse", "delta_global_rmse_vs_baseline"]
        print(summary[cols].to_string(index=False))
        print("\nAlignment-feature contrasts:")
        print(verdict.to_string(index=False))
        print("\nPre-registered decision:")
        for group, item in prereg.items():
            print(f"  {group:10s} {item['decision']:28s} {item['reason']}")
        print("\nWritten:")
        for path in [wells_csv, *written]:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
