"""Create real protocol/error reports from a completed validation output.

Usage (on the same reports directory written by the real runner)::

    python scripts/analyze_real_validation.py --reports-dir /kaggle/working/reports

The script requires ``well_level_validation.csv``. It refuses to invent
point counts, distributions, target ranges or worst-well diagnostics from an
aggregate RMSE alone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap

bootstrap()

from src.paths import ensure_reports_dir
from src.real_reporting import write_real_analysis


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="completed real validation output directory (defaults to REPORTS_DIR)",
    )
    args = parser.parse_args(argv)
    root = Path(args.reports_dir) if args.reports_dir else ensure_reports_dir()
    written = write_real_analysis(root)
    print("Computed real-analysis reports:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
