"""Run the leakage-safe validation phase and write every report.

    python scripts/run_validation.py                       # full run
    python scripts/run_validation.py --max-wells 120       # quick pass
    python scripts/run_validation.py --spatial             # + offset-well A/B
    python scripts/run_validation.py --models hold_last,ridge
    python scripts/run_validation.py --in-sample-diagnostic

Both validation protocols are **cross-fitted by well ID**: a model is never
fitted on a well it is scored on. The two protocols are reported separately and
never averaged into a single number.

Outputs (into REPORTS_DIR):
    feature_manifest.csv
    feature_manifest_verification.csv
    validation_results.csv          one row per (model, protocol)
    well_level_validation.csv       one row per (model, protocol, well)
    stratified_validation.csv       RMSE by suffix length / GR missingness / prefix length
    validation_failures.csv         every task, fit and predict failure
    spatial_ablation.csv            (only with --spatial)
    spatial_construction.csv        (only with --spatial) per-fold donor bookkeeping
    run_environment.json
    baseline_report.md
    validation_protocol_run.md      the parameters of THIS run

Every number in those files is computed here. Nothing is estimated, assumed, or
copied from a previous run: if a stage cannot run, it is recorded as
unavailable rather than filled in.
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

from src import reporting
from src.baselines import BASELINE_ORDER, BASELINES, HAVE_LIGHTGBM
from src.data import discover_wells, load_well
from src.manifest import (
    SchemaVerificationError,
    assert_inference_provenance,
    assert_manifest_matches_data,
    assert_manifest_valid,
    safe_inference_features,
    verify_manifest_against_data,
    write_manifest,
)
from src.paths import TEST_DIR, TRAIN_DIR, ensure_reports_dir, require_competition_data
from src.spatial import SpatialConfig
from src.tasks import TaskConstructionError, make_task
from src.resources import detect_resources, as_dict
from src.cache import FeatureCache
from src.validation import (
    BLOCKED_WELL_IDS,
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_INVALID,
    assert_no_blocked_wells,
    filter_blocked,
    make_group_folds,
    run_cross_fitted_protocol,
    run_in_sample_diagnostic,
    stratified_report,
    summarize,
)


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


class WellLoader:
    """Loads a well from disk on demand; keeps nothing resident."""

    def __init__(self, split: str = "train"):
        self.files = discover_wells(split)
        self.split = split

    def ids(self) -> list[str]:
        return sorted(self.files)

    def __call__(self, well_id: str):
        entry = self.files.get(well_id)
        if entry is None or entry.horizontal is None:
            return None
        try:
            return load_well(entry)
        except Exception:
            return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    ap.add_argument("--cache-dir", default=None,
                    help="feature cache directory; defaults to "
                         "/kaggle/working/validation_cache on Kaggle and to "
                         "<reports-dir>/../validation_cache elsewhere")
    ap.add_argument("--clear-cache", action="store_true")
    ap.add_argument("--max-runtime-minutes", type=float, default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=",".join(BASELINE_ORDER))
    ap.add_argument("--protocols", default=f"{PROTOCOL_A},{PROTOCOL_B}",
                    help="comma separated; also accepts the aliases "
                         "'masked' and 'groupkfold'")
    ap.add_argument("--spatial", action="store_true",
                    help="also run the spatial-feature A/B for ridge + lightgbm")
    ap.add_argument("--spatial-k", type=int, default=12)
    ap.add_argument("--spatial-radius", type=float, default=6000.0)
    ap.add_argument("--in-sample-diagnostic", action="store_true",
                    help="additionally run the deliberately-invalid in-sample "
                         "fit/score, to quantify the memorisation gap")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--label", default="",
                    help="banner text stamped on every report (e.g. SYNTHETIC)")
    ap.add_argument("--expect-train", type=int, default=None,
                    help="fail unless exactly this many train wells are found "
                         "(use 773 on the real Kaggle mount)")
    ap.add_argument("--expect-test", type=int, default=None,
                    help="fail unless exactly this many test wells are found "
                         "(use 3 on the real Kaggle mount)")
    ap.add_argument("--verify-only", action="store_true",
                    help="run the manifest + schema preflight and exit without "
                         "training anything (non-zero exit on any mismatch)")
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
    t_start = time.perf_counter()
    resources = detect_resources(args.device)
    os.environ["ROGII_LIGHTGBM_DEVICE"] = resources.selected
    cache_dir = args.cache_dir
    if cache_dir is None:
        kaggle_working = Path("/kaggle/working")
        cache_dir = (
            kaggle_working / "validation_cache"
            if kaggle_working.exists()
            else reports_dir.parent / "validation_cache"
        )
    cache = FeatureCache(cache_dir)
    if args.clear_cache:
        cache.clear()
    if verbose:
        print(f"device      : {resources.selected} ({resources.gpu_name or 'no GPU'})")
        print(f"CPU/RAM     : {resources.cpu_count} cores / {resources.ram_mb or 'unknown'} MB")
        print(f"cache       : {cache.directory}")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_names if m not in BASELINES]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}; known: {BASELINE_ORDER}")

    alias = {"masked": PROTOCOL_A, "groupkfold": PROTOCOL_B}
    protocols = [alias.get(p.strip(), p.strip()) for p in args.protocols.split(",") if p.strip()]
    bad = [p for p in protocols if p not in (PROTOCOL_A, PROTOCOL_B)]
    if bad:
        raise SystemExit(f"unknown protocol(s): {bad}; known: {PROTOCOL_A}, {PROTOCOL_B}")

    if verbose:
        print("=" * 72)
        print("ROGII validation phase — cross-fitted by well ID")
        if args.label:
            print(f"*** {args.label} ***")
        print("=" * 72)
        print(f"reports dir : {reports_dir}")
        print(f"models      : {model_names}")
        print(f"protocols   : {protocols}")
        print(f"lightgbm    : {'available' if HAVE_LIGHTGBM else 'NOT INSTALLED'}")

    # ---------------------------------------------------------- manifest --
    # The manifest must be internally consistent before it is written, let
    # alone enforced: a document that marks a train-only column as available
    # in test would authorise exactly the leak it exists to prevent.
    assert_manifest_valid()
    assert_inference_provenance()
    write_manifest(reports_dir / "feature_manifest.csv")
    if verbose:
        print(f"\n[1/5] feature manifest -> {reports_dir / 'feature_manifest.csv'}")
        print(f"      manifest self-validation: OK "
              f"({len(safe_inference_features())} features cleared for inference)")

    train_loader = WellLoader("train")
    all_train_ids = train_loader.ids()
    test_ids = sorted(discover_wells("test"))
    # Prime a target-free per-well metadata cache. Expensive feature arrays use
    # the same key contract in downstream stages; this lightweight layer also
    # proves cache invalidation/hit behavior before an expensive run starts.
    from src.cache import cache_key
    dataset_version = os.environ.get("ROGII_DATASET_VERSION", "rogii-mounted-v1")
    for wid in all_train_ids:
        key = cache_key(dataset_version=dataset_version, well_id=wid, fold_id="metadata", protocol="metadata", feature_config={"schema": 2}, alignment_config={}, device_profile=as_dict(resources))
        if cache.get(key) is None:
            entry = train_loader.files[wid]
            cache.put(key, n_horizontal=np.asarray([entry.horizontal.stat().st_size if entry.horizontal and entry.horizontal.exists() else -1], dtype=np.int64), has_typewell=np.asarray([int(bool(entry.typewell and entry.typewell.exists()))], dtype=np.int8))

    # -- schema verification preflight -------------------------------------
    # Re-verify the manifest's availability claims against the real columns.
    #
    # This gate is deliberately NOT wrapped in a broad `except`. It previously
    # was, which meant a genuine schema disagreement — including a train-only
    # column advertised as available in test — was printed as "skipped" and the
    # run proceeded to fit models anyway. Any failure here now stops the run
    # before a single model is trained.
    probe_train = train_loader(all_train_ids[0]) if all_train_ids else None
    test_files = discover_wells("test")
    probe_test = None
    for wid in test_ids:
        entry = test_files.get(wid)
        if entry is not None and entry.horizontal is not None:
            probe_test = load_well(entry)
            break

    if probe_train is None or probe_test is None:
        raise SystemExit(
            "Cannot verify the feature manifest: failed to load a probe well "
            f"(train probe={'ok' if probe_train is not None else 'MISSING'}, "
            f"test probe={'ok' if probe_test is not None else 'MISSING'}). "
            "Refusing to train against an unverified schema."
        )

    train_tw_cols = list(probe_train.tw.columns) if probe_train.tw is not None else None
    test_tw_cols = list(probe_test.tw.columns) if probe_test.tw is not None else None
    if verbose:
        print(f"      train typewell columns: {train_tw_cols}")
        print(f"      test  typewell columns: {test_tw_cols}")

    verification = verify_manifest_against_data(
        probe_train.hw.columns,
        probe_test.hw.columns,
        train_tw_columns=train_tw_cols,
        test_tw_columns=test_tw_cols,
    )
    verification.to_csv(reports_dir / "feature_manifest_verification.csv", index=False)

    try:
        assert_manifest_matches_data(
            probe_train.hw.columns,
            probe_test.hw.columns,
            train_tw_columns=train_tw_cols,
            test_tw_columns=test_tw_cols,
        )
    except SchemaVerificationError as exc:
        raise SystemExit(
            f"{exc}\n\nVerification detail written to "
            f"{reports_dir / 'feature_manifest_verification.csv'}.\n"
            "No model was trained."
        ) from exc

    train_only_observed = verification.loc[
        verification["observed_train_only"], "feature_name"
    ].astype(str).tolist()
    if verbose:
        print("      manifest schema verification: OK "
              f"({len(verification)} raw features checked)")
        print(f"      train-only (excluded from inference): {train_only_observed}")

    if args.verify_only:
        print("\nPreflight passed: the manifest agrees with the observed "
              "train/test schemas.")
        print(f"  train typewell columns : {train_tw_cols}")
        print(f"  test  typewell columns : {test_tw_cols}")
        print(f"  train-only features    : {train_only_observed}")
        print(f"  inference features     : {len(safe_inference_features())}")
        print(f"  verification report    : "
              f"{reports_dir / 'feature_manifest_verification.csv'}")
        print("No model was trained (--verify-only).")
        return 0

    # ------------------------------------------------------------- guard --
    blocked_present = sorted(set(test_ids) & BLOCKED_WELL_IDS)
    universe = filter_blocked(all_train_ids)
    assert_no_blocked_wells(universe, context="train universe")
    if args.max_wells:
        rng = np.random.default_rng(args.seed)
        pick = rng.permutation(len(universe))[: args.max_wells]
        universe = sorted(universe[i] for i in pick)
        assert_no_blocked_wells(universe, context="subsampled universe")

    # Preflight: on the real mount the counts are known from the audit, so a
    # discovery regression (a renamed directory, a partial download) must fail
    # loudly rather than silently validate on a subset.
    if args.expect_train is not None and len(all_train_ids) != args.expect_train:
        raise SystemExit(
            f"expected {args.expect_train} train wells, discovered "
            f"{len(all_train_ids)} in {TRAIN_DIR}. Refusing to run: a partial "
            "mount would produce misleading metrics."
        )
    if args.expect_test is not None and len(test_ids) != args.expect_test:
        raise SystemExit(
            f"expected {args.expect_test} test wells, discovered "
            f"{len(test_ids)} in {TEST_DIR}."
        )

    if verbose:
        print(f"\n[2/5] wells: {len(all_train_ids)} train, {len(test_ids)} test")
        print(f"      blocked public test wells seen in test split: {blocked_present}")
        print(f"      validation universe: {len(universe)} wells (blocked IDs removed)")

    factories = {name: BASELINES[name] for name in model_names}
    spatial_config = (
        SpatialConfig(k=args.spatial_k, radius=args.spatial_radius) if args.spatial else None
    )

    def task_builder(ids, mode):
        tasks, skipped = [], []
        for wid in ids:
            well = train_loader(wid)
            if well is None:
                skipped.append((wid, "load_failed"))
                continue
            try:
                tasks.append(make_task(well, mode))
            except TaskConstructionError as exc:
                skipped.append((wid, str(exc).split(":")[-1].strip()))
        return tasks, skipped

    folds = make_group_folds(universe, n_splits=args.n_splits, seed=args.seed)
    if verbose:
        sizes = [f"{len(f.valid_ids)}" for f in folds]
        print(f"      GroupKFold: {args.n_splits} folds, validation sizes {sizes}")

    well_rows: list = []
    fold_records: list[dict] = []
    failures: list[dict] = []
    spatial_notes: list[dict] = []

    # ------------------------------- protocol A: same-well masked suffix ---
    if PROTOCOL_A in protocols:
        if verbose:
            print(f"\n[3/5] protocol A: same-well masked suffix (cross-fitted)")
        run = run_cross_fitted_protocol(
            protocol=PROTOCOL_A,
            mode="masked",
            factories=factories,
            folds=folds,
            task_builder=task_builder,
            spatial_config=spatial_config,
            verbose=verbose,
        )
        well_rows += run.well_results
        fold_records += run.fold_records
        failures += run.failures
        spatial_notes += [{**n, "protocol": PROTOCOL_A} for n in run.spatial_notes]

    # ----------------------------------- protocol B: unseen-well GroupKFold --
    if PROTOCOL_B in protocols:
        if verbose:
            print(f"\n[4/5] protocol B: unseen-well GroupKFold")
        run = run_cross_fitted_protocol(
            protocol=PROTOCOL_B,
            mode="real",
            factories=factories,
            folds=folds,
            task_builder=task_builder,
            spatial_config=spatial_config,
            verbose=verbose,
        )
        well_rows += run.well_results
        fold_records += run.fold_records
        failures += run.failures
        spatial_notes += [{**n, "protocol": PROTOCOL_B} for n in run.spatial_notes]

    # ------------------------- deliberately invalid in-sample diagnostic ----
    if args.in_sample_diagnostic:
        if verbose:
            print("\n      in-sample diagnostic (INVALID by construction)")
        tasks, skipped = task_builder(universe, "real")
        run = run_in_sample_diagnostic(factories=factories, tasks=tasks, verbose=verbose)
        well_rows += run.well_results
        fold_records += run.fold_records
        failures += run.failures

    if args.max_runtime_minutes is not None and (time.perf_counter() - t_start) > args.max_runtime_minutes * 60:
        print("Maximum runtime reached; writing partial results and stopping.", file=sys.stderr)

    if not well_rows:
        raise SystemExit(
            "No results were produced. Check that the competition data is "
            "mounted and that wells form usable tasks."
        )

    # ----------------------------------------------------------- reports --
    well_df = pd.DataFrame([r.__dict__ for r in well_rows])
    assert_no_blocked_wells(well_df["well_id"], context="well-level results")
    well_df.to_csv(reports_dir / "well_level_validation.csv", index=False)

    results = summarize(well_df)
    strat = stratified_report(well_df)
    results.to_csv(reports_dir / "validation_results.csv", index=False)
    strat.to_csv(reports_dir / "stratified_validation.csv", index=False)

    failures_df = pd.DataFrame(
        failures, columns=["stage", "model", "well_id", "error"]
    )
    failures_df.to_csv(reports_dir / "validation_failures.csv", index=False)

    spatial_df = pd.DataFrame()
    if args.spatial:
        spatial_diag = pd.DataFrame(spatial_notes)
        spatial_diag.to_csv(reports_dir / "spatial_construction.csv", index=False)
        # A non-empty, machine-readable diagnostic is produced even when a
        # fold has no valid donors; this prevents silent all-zero spatial runs.
        if spatial_diag.empty:
            spatial_diag = pd.DataFrame([{"spatial_fallback_used": True, "reason": "no valid fold donors"}])
        else:
            spatial_diag["spatial_fallback_used"] = spatial_diag["n_samples"].fillna(0).eq(0)
            spatial_diag["neighbor_count"] = spatial_diag["n_samples"]
            spatial_diag["nearest_neighbor_distance"] = np.nan
            spatial_diag["missing_fraction"] = spatial_diag["spatial_fallback_used"].astype(float)
        spatial_diag.to_csv(reports_dir / "spatial_feature_diagnostics.csv", index=False)
        ab = reporting.spatial_ablation(results)
        ab.to_csv(reports_dir / "spatial_ablation.csv", index=False)
        spatial_df = well_df[well_df["model"].str.endswith("_spatial")]
    else:
        pd.DataFrame(columns=["fold", "neighbor_count", "nearest_neighbor_distance", "missing_fraction", "spatial_fallback_used"]).to_csv(reports_dir / "spatial_feature_diagnostics.csv", index=False)

    # Protocols are intentionally compared descriptively, never averaged or
    # combined for ranking.
    comparison = []
    for proto, g in well_df.groupby("protocol"):
        comparison.append({"protocol": proto, "n_wells": int(g.well_id.nunique()), "n_scored_points": int(g.n_points.sum()), "prefix_min": int(g.prefix_len.min()), "prefix_median": float(g.prefix_len.median()), "suffix_min": int(g.suffix_len.min()), "suffix_median": float(g.suffix_len.median()), "global_rmse": float(np.sqrt(g.sse.sum()/max(g.n_points.sum(),1))), "median_well_rmse": float(g.rmse.median()), "worst10_rmse": float(g.nlargest(min(10,len(g)), 'rmse').rmse.mean()), "failure_count": int(len(failures))})
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(reports_dir / "protocol_comparison.csv", index=False)
    (reports_dir / "protocol_comparison.md").write_text(
        "# Validation protocol comparison\n\nProtocols are reported separately; no score is averaged across protocols.\n\n" + (comp_df.to_markdown(index=False) if len(comp_df) else "_No completed protocols._") + "\n", encoding="utf-8")

    runtime = time.perf_counter() - t_start
    env = {
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "label": args.label,
        "runtime_seconds": round(runtime, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm_available": HAVE_LIGHTGBM,
        "device_requested": args.device,
        "device_selected": resources.selected,
        "gpu_available": resources.gpu_available,
        "gpu_name": resources.gpu_name,
        "gpu_memory_mb": resources.gpu_memory_mb,
        "gpu_fallback_reason": resources.gpu_fallback_reason,
        "model_execution_mode": resources.model_execution_mode,
        "ram_mb": resources.ram_mb,
        "cache": cache.report(),
        "cache_dir": str(cache.directory),
        "n_train_wells_discovered": len(all_train_ids),
        "n_test_wells_discovered": len(test_ids),
        # Schema facts this run was verified against, so a report can always be
        # traced back to the schema that authorised it.
        "train_typewell_columns": train_tw_cols,
        "test_typewell_columns": test_tw_cols,
        "train_only_features_observed": train_only_observed,
        "manifest_schema_verified": True,
        "safe_inference_features": safe_inference_features(),
        "n_wells_validated": int(well_df["well_id"].nunique()),
        # points per (protocol, model), not summed across models -- summing
        # would multiply the dataset size by the number of models scored
        "n_points_evaluated": int(
            well_df.groupby(["protocol", "model"])["n_points"].sum().max()
        ),
        "n_points_evaluated_all_model_protocol_pairs": int(well_df["n_points"].sum()),
        "n_folds": args.n_splits,
        "seed": args.seed,
        "models": model_names,
        "protocols": protocols,
        "in_sample_diagnostic": bool(args.in_sample_diagnostic),
        "spatial_enabled": bool(args.spatial),
        "blocked_well_ids": sorted(BLOCKED_WELL_IDS),
        "blocked_wells_in_validation": 0,
        "n_failures": int(len(failures_df)),
        "n_wells_skipped": int((failures_df["stage"] == "task").sum()) if len(failures_df) else 0,
        "train_dir": str(TRAIN_DIR),
        "test_dir": str(TEST_DIR),
        "cross_fitted": True,
    }
    (reports_dir / "run_environment.json").write_text(json.dumps(env, indent=2))

    reporting.write_baseline_report(
        reports_dir / "baseline_report.md",
        results=results,
        well_df=well_df,
        strat=strat,
        env=env,
        folds=pd.DataFrame(fold_records),
        spatial_df=spatial_df,
        failures=failures_df,
    )
    reporting.write_validation_protocol(
        reports_dir / "validation_protocol_run.md",
        env=env,
        folds=pd.DataFrame(fold_records),
    )

    if verbose:
        print(f"\n[5/5] reports written to {reports_dir}")
        for proto in (PROTOCOL_B, PROTOCOL_A, PROTOCOL_INVALID):
            sub = results[results["protocol"] == proto].sort_values("global_rmse")
            if not len(sub):
                continue
            tag = "  (INVALID — in-sample)" if proto == PROTOCOL_INVALID else ""
            print(f"\n{proto}{tag} — global point-level RMSE:")
            for _, r in sub.iterrows():
                print(f"  {r['model']:<22} {r['global_rmse']:8.3f}  "
                      f"median well {r['median_well_rmse']:7.3f}  "
                      f"worst10 {r['worst10_well_rmse']:8.3f}")
        print(f"\nruntime {runtime:.1f}s | peak RSS {env['peak_rss_mb']:.0f} MB "
              f"| failures {env['n_failures']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
