"""Run the full audit chain (sections 1-7) and the pipeline smoke test (8).

    python scripts/run_all_audits.py
"""
from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STEPS = [
    ("1. competition data", "scripts/audit_competition_data.py"),
    ("2. task presentation", "scripts/audit_task_presentation.py"),
    ("3. sample submission", "scripts/audit_submission.py"),
    ("4-6. external resources & leakage", "scripts/audit_external_resources.py"),
]


def main() -> None:
    failed = []
    for label, script in STEPS:
        print(f"\n=== {label} ===")
        rc = subprocess.call([sys.executable, str(ROOT / script)])
        if rc != 0:
            failed.append(label)

    print("\n=== 8. baseline pipeline smoke test ===")
    try:
        from src.data import iter_wells, validate_split, well_metadata

        meta_tr = well_metadata("train")
        meta_te = well_metadata("test")
        meta = __import__("pandas").concat([meta_tr, meta_te], ignore_index=True)
        out = ROOT / "reports" / "well_metadata.csv"
        meta.to_csv(out, index=False)
        print(f"train wells={len(meta_tr)} test wells={len(meta_te)} -> {out}")

        issues = __import__("pandas").concat(
            [validate_split("train"), validate_split("test")], ignore_index=True
        )
        ipath = ROOT / "reports" / "pipeline_validation_issues.csv"
        issues.to_csv(ipath, index=False)
        print(f"validation issues: {len(issues)} -> {ipath}")
    except Exception as exc:
        failed.append(f"8. pipeline ({type(exc).__name__}: {exc})")

    print("\n" + ("ALL STEPS OK" if not failed else "FAILED: " + ", ".join(map(str, failed))))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
