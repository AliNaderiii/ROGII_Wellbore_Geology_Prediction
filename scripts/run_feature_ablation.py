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
import os
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
    alignment_feature_recommendation,
    alignment_feature_verdict,
    run_ablation_protocol,
    summarize_ablation,
)
from src.data import discover_wells, load_well
from src.model_status import assert_not_rejected
from src.paths import ensure_reports_dir, require_competition_data
from src.real_reporting import write_alignment_spatial_ablation
from src.spatial import SpatialConfig
from src.tasks import TaskConstructionError, make_task
from src.validation import (
    PROTOCOL_A,
    PROTOCOL_B,
    assert_no_blocked_wells,
    filter_blocked,
    make_group_folds,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spatial-k", type=int, default=12)
    ap.add_argument("--spatial-radius", type=float, default=6000.0)
    ap.add_argument("--branches", default=",".join(BRANCH_ORDER))
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
    # Every branch is a Ridge configuration. A rejected model must not be able
    # to enter the ablation and thereby the promotion discussion.
    assert_not_rejected(branches, context="feature ablation branches")

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
                tasks.append(make_task(well, mode))
            except TaskConstructionError as exc:
                skipped.append((wid, str(exc).split(":")[-1].strip()))
        return tasks, skipped

    universe = filter_blocked(sorted(files))
    assert_no_blocked_wells(universe, context="ablation universe")
    if args.max_wells:
        rng = np.random.default_rng(args.seed)
        pick = rng.permutation(len(universe))[: args.max_wells]
        universe = sorted(universe[i] for i in pick)

    folds = make_group_folds(universe, n_splits=args.n_splits, seed=args.seed)
    spatial_config = SpatialConfig(k=args.spatial_k, radius=args.spatial_radius)

    if verbose:
        print("=" * 72)
        print("Ridge alignment / spatial feature ablation — cross-fitted by well ID")
        if args.label:
            print(f"*** {args.label} ***")
        print("=" * 72)
        print(f"reports dir : {reports_dir}")
        print(f"wells       : {len(universe)}")
        print(f"branches    : {list(branches)}")
        print(f"folds       : {args.n_splits}")

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
    pd.DataFrame(failures, columns=["stage", "model", "well_id", "error"]).to_csv(
        reports_dir / "alignment_ablation_failures.csv", index=False
    )
    pd.DataFrame(fold_records).to_csv(reports_dir / "alignment_ablation_folds.csv", index=False)

    written = write_alignment_spatial_ablation(reports_dir)
    if args.label:
        report = reports_dir / "alignment_spatial_ablation.md"
        if report.exists():
            report.write_text(
                f"> **{args.label}**\n\n" + report.read_text(encoding="utf-8"), encoding="utf-8"
            )

    summary = summarize_ablation(well_df)
    verdict = alignment_feature_verdict(summary)
    recommendation = alignment_feature_recommendation(verdict)

    if verbose:
        print(f"\nCompleted in {time.perf_counter() - t_start:.1f}s")
        print("\nDelta vs the current Ridge baseline (branch B):")
        cols = ["protocol", "branch", "n_wells", "global_rmse", "delta_global_rmse_vs_baseline"]
        print(summary[cols].to_string(index=False))
        print("\nAlignment-feature contrasts:")
        print(verdict.to_string(index=False))
        print(f"\nDecision: {recommendation['decision']} — {recommendation['reason']}")
        print("\nWritten:")
        for path in [wells_csv, *written]:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
