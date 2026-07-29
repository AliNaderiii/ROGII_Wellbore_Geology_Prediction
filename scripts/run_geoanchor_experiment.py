"""Run the controlled GeoAnchor experiment (arms A–E) and write every report.

    python scripts/run_geoanchor_experiment.py --max-wells 120        # quick pass
    python scripts/run_geoanchor_experiment.py --expect-wells 770     # full real run
    ROGII_COMPETITION_ROOT=/path/to/synthetic \\
        python scripts/run_geoanchor_experiment.py

Pre-registration: ``reports/geoanchor_experiment.md``. The experiment never
creates a submission. The three visible public test wells are hard-blocked
from every fold, fit, gate-training set and report.

Report naming is evidence-based: files are named ``real_geoanchor_*`` only
when the discovered well counts match the audited real mount (773 train
wells, 770 eligible after excluding the three blocked IDs); every other run
writes ``synthetic_geoanchor_*`` files banner-stamped
``SYNTHETIC — NOT A REAL KAGGLE COMPETITION RESULT``.

Outputs (into REPORTS_DIR, or --reports-dir):
    {prefix}geoanchor_arm_summary.csv            global/mean/median/worst-10 per arm & protocol
    {prefix}geoanchor_well_level.csv             one row per (arm, protocol, well)
    {prefix}geoanchor_paired_well_deltas.csv     paired deltas vs Ridge Default
    {prefix}geoanchor_improved_degraded_counts.csv
    {prefix}geoanchor_fold_stability.csv
    {prefix}geoanchor_bootstrap_ci.csv
    {prefix}geoanchor_stratified.csv             GR missingness + hidden suffix length
    {prefix}geoanchor_gate_stats.csv             activation/fallback rates + fold training info
    {prefix}geoanchor_gate_well_decisions.csv    per-well gate decisions (arm E)
    {prefix}geoanchor_decision.md                pre-registered decision, computed numbers
    {prefix}geoanchor_run_environment.json
    {prefix}geoanchor_failures.csv
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
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

from src.data import discover_wells, load_well
from src.geoanchor import ARM_ORDER, GateConfig, GEOANCHOR_VERSION, run_geoanchor_protocol
from src.geoanchor_reporting import write_reports
from src.manifest import (
    SchemaVerificationError,
    assert_inference_provenance,
    assert_manifest_valid,
    safe_inference_features,
    write_manifest,
)
from src.paths import ensure_reports_dir
from src.real_ablation_reporting import (
    AUDITED_DISCOVERED_WELLS,
    AUDITED_ELIGIBLE_WELLS,
    is_real_run,
)
from src.tasks import TaskConstructionError, make_task
from src.validation import (
    BLOCKED_WELL_IDS,
    PROTOCOL_A,
    PROTOCOL_B,
    filter_blocked,
    make_group_folds,
)
from src.resources import detect_resources


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


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=5,
                    help="cross-fitting depth for the gate's OOF examples")
    ap.add_argument("--tune-splits", type=int, default=3,
                    help="sub-folds used by the gate's threshold tuning")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-wells", type=int, default=None)
    ap.add_argument("--device", choices=("auto", "cpu", "gpu"), default="cpu")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    verbose = not args.quiet

    t_start = time.perf_counter()
    reports_dir = ensure_reports_dir() if args.reports_dir is None else Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Feature whitelist: validated BEFORE any model is fitted.
    assert_manifest_valid()
    provenance = assert_inference_provenance()
    cleared = safe_inference_features()
    if verbose:
        print(f"[1/5] manifest valid: {len(cleared)} cleared features; "
              f"provenance roots verified across {len(provenance)} entries")

    loader = WellLoader("train")
    all_train_ids = loader.ids()
    universe = filter_blocked(all_train_ids)
    if args.max_wells is not None:
        universe = universe[: args.max_wells]
    blocked_present = sorted(set(all_train_ids) & set(BLOCKED_WELL_IDS))
    if args.expect_wells is not None and len(universe) != args.expect_wells:
        raise RuntimeError(
            f"expected {args.expect_wells} eligible train wells, found {len(universe)} "
            f"(discovered {len(all_train_ids)}). Refusing to run on an unexpected universe."
        )
    if len(universe) < args.n_splits * 2:
        raise RuntimeError(
            f"need at least {args.n_splits * 2} eligible wells, got {len(universe)}"
        )
    if verbose:
        print(f"[2/5] wells: {len(all_train_ids)} train discovered; "
              f"{len(universe)} eligible after blocking {blocked_present or 'none'}")

    data_source = (
        "real_kaggle"
        if (
            len(all_train_ids) == AUDITED_DISCOVERED_WELLS
            and len(universe) == AUDITED_ELIGIBLE_WELLS
        )
        else ("synthetic_or_partial" if len(universe) < 100 else "synthetic_field")
    )
    environment = {
        "data_source": data_source,
        "n_train_wells_discovered": len(all_train_ids),
        "n_eligible_wells": len(universe),
        "n_wells_evaluated": len(universe),
        "blocked_wells_in_validation": 0,
        "blocked_public_test_wells": sorted(BLOCKED_WELL_IDS),
        "max_wells": args.max_wells,
        "n_splits": args.n_splits,
        "inner_splits": args.inner_splits,
        "tune_splits": args.tune_splits,
        "seed": args.seed,
        "device": args.device,
        "label": args.label,
        "geoanchor_version": GEOANCHOR_VERSION,
        "git_commit": _git_commit(),
        "experiment": "geoanchor controlled experiment, arms A-E",
        "final_submission_created": False,
    }
    environment["is_real_run"] = is_real_run(environment)

    def task_builder(ids, mode):
        tasks, skipped = [], []
        for wid in ids:
            well = loader(wid)
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
        sizes = [len(f.valid_ids) for f in folds]
        print(f"      GroupKFold: {args.n_splits} folds, validation sizes {sizes}")

    write_manifest(reports_dir / "feature_manifest.csv")

    memo: dict = {}
    gate_config = GateConfig(
        inner_splits=args.inner_splits, tune_splits=args.tune_splits, seed=args.seed
    )
    well_rows: list = []
    fold_records: list = []
    failures: list = []
    gate_logs: list = []
    gate_infos: list = []
    protocol_seconds: dict = {}

    for i, (protocol, mode) in enumerate(
        ((PROTOCOL_A, "masked"), (PROTOCOL_B, "real"))
    ):
        if verbose:
            print(f"[{3 + i}/5] {protocol} (cross-fitted, arms A–E)")
        t0 = time.perf_counter()
        run = run_geoanchor_protocol(
            protocol=protocol,
            mode=mode,
            folds=folds,
            task_builder=task_builder,
            arms=ARM_ORDER,
            memo=memo,
            seed=args.seed,
            device=args.device,
            path_cache=None,
            gate_config=gate_config,
            verbose=verbose,
        )
        protocol_seconds[protocol] = time.perf_counter() - t0
        well_rows += run.well_results
        fold_records += run.fold_records
        failures += run.failures
        gate_logs += run.gate_logs
        gate_infos += run.gate_fit_infos
        if verbose:
            print(f"      {protocol}: {len(run.well_results)} well-scores in "
                  f"{protocol_seconds[protocol]:.1f}s")

    if verbose:
        print("[5/5] writing reports")
    paths = write_reports(
        reports_dir=reports_dir,
        environment=environment,
        well_results=well_rows,
        fold_records=fold_records,
        failures=failures,
        gate_logs=gate_logs,
        gate_infos=gate_infos,
        protocol_seconds=protocol_seconds,
        peak_rss_mb=peak_rss_mb(),
        n_boot=args.n_bootstrap,
    )
    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"\nWrote {len([p for p in paths.values() if isinstance(p, Path)])} report files "
              f"to {reports_dir} in {elapsed:.1f}s (peak RSS {peak_rss_mb():.0f} MB)")
        for name, path in paths.items():
            if isinstance(path, Path):
                print(f"  {name}: {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
