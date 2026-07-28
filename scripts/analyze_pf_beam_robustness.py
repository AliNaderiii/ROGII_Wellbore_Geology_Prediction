"""Paired-error / robustness analysis for the PF + Beam Ridge experiment.

Usage (against a completed particle-beam validation output directory)::

    python scripts/analyze_pf_beam_robustness.py \\
        --reports-dir /kaggle/working/particle_beam_reports

The script prefers ``particle_beam_wells.csv`` and falls back to
``well_level_validation.csv``.  It never retrains.  When neither well-level
table is present it still writes the five required report files from the
owner-supplied 770-well global RMSE aggregates, marking every well-/fold-/
bootstrap-level field as unavailable rather than fabricating it.

Outputs (into ``--reports-dir``, default ``REPORTS_DIR``):

    pf_beam_real_decision.md
    pf_beam_failure_analysis.md
    pf_beam_paired_well_deltas.csv
    pf_beam_fold_deltas.csv
    pf_beam_bootstrap_ci.csv
    particle_beam_fold_deltas.csv      (alias of pf_beam_fold_deltas.csv)
    particle_beam_bootstrap_ci.csv     (alias of pf_beam_bootstrap_ci.csv)

The validation runner also writes ``particle_beam_wells.csv`` (the per-well
input this analyzer consumes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd

from src.paths import ensure_reports_dir
from src.pf_beam_robustness import (
    load_well_table,
    resolve_well_table,
    write_robustness_reports,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="directory with particle_beam_wells.csv / well_level_validation.csv "
        "(defaults to REPORTS_DIR); reports are written here too",
    )
    parser.add_argument(
        "--wells-csv",
        default=None,
        help="explicit path to a multi-model well-level CSV "
        "(overrides auto-discovery under --reports-dir)",
    )
    parser.add_argument(
        "--failures-csv",
        default=None,
        help="optional particle_beam_failures.csv / validation_failures.csv",
    )
    parser.add_argument(
        "--environment-json",
        default=None,
        help="optional particle_beam_run_environment.json / run_environment.json",
    )
    parser.add_argument(
        "--allow-owner-only",
        action="store_true",
        default=True,
        help="when no well-level table is found, still write the decision from "
        "owner aggregates (default: on)",
    )
    parser.add_argument(
        "--require-wells",
        action="store_true",
        help="fail if no well-level table is found (disables owner-only mode)",
    )
    args = parser.parse_args(argv)

    root = Path(args.reports_dir) if args.reports_dir else ensure_reports_dir()
    root.mkdir(parents=True, exist_ok=True)

    well = None
    wells_path = Path(args.wells_csv) if args.wells_csv else resolve_well_table(root)
    if wells_path is not None and wells_path.exists():
        well = load_well_table(wells_path)
        # Keep only the PF/Beam experiment branches when the full baseline
        # table was supplied.
        keep = {
            "ridge_default",
            "ridge_particle_filter",
            "ridge_beam_search",
            "ridge_particle_beam",
            "ridge",  # tolerate a plain ridge column name as default
        }
        if "model" in well.columns:
            present = set(well["model"].astype(str))
            if "ridge" in present and "ridge_default" not in present:
                well = well.copy()
                well.loc[well["model"] == "ridge", "model"] = "ridge_default"
            well = well[well["model"].isin(keep - {"ridge"} | {"ridge_default"})].copy()
        print(f"loaded well-level table: {wells_path} ({len(well)} rows)")
    else:
        msg = (
            f"No particle_beam_wells.csv or well_level_validation.csv under {root}."
        )
        if args.require_wells:
            print(msg, file=sys.stderr)
            return 2
        print(msg + " Writing owner-aggregate decision only.")

    failures = None
    fail_path = (
        Path(args.failures_csv)
        if args.failures_csv
        else next(
            (
                p
                for p in (
                    root / "particle_beam_failures.csv",
                    root / "validation_failures.csv",
                )
                if p.exists()
            ),
            None,
        )
    )
    if fail_path is not None:
        failures = pd.read_csv(fail_path)
        print(f"loaded failures: {fail_path} ({len(failures)} rows)")

    env = None
    env_path = (
        Path(args.environment_json)
        if args.environment_json
        else next(
            (
                p
                for p in (
                    root / "particle_beam_run_environment.json",
                    root / "run_environment.json",
                )
                if p.exists()
            ),
            None,
        )
    )
    if env_path is not None:
        env = json.loads(env_path.read_text(encoding="utf-8"))
        print(f"loaded environment: {env_path}")

    written = write_robustness_reports(
        root,
        well,
        failures=failures,
        environment=env,
        owner_aggregates_ok=True,
    )
    print("Wrote PF/Beam robustness reports:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
