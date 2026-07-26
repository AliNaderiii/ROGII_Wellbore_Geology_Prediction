"""Orchestrate the full audit chain.

Command line:

    python scripts/run_all_audits.py

Kaggle Notebook cell:

    import sys; sys.path.insert(0, "/kaggle/working/ROGII_Wellbore_Geology_Prediction")
    from scripts.run_all_audits import run_all
    run_all()

Notes
-----
* Never uses ``__file__`` for the repo root — that is undefined in a notebook
  cell. The root is located by walking up from the current working directory
  looking for the ``src/`` package, with an env-var override.
* Audits run in-process (imported), not as subprocesses, so this works when
  the code has been imported into a notebook.
* Writes every report to REPORTS_DIR (/kaggle/working/reports by default).
* Fails fast with a clear message when the competition data is not mounted.
* Trains nothing, fabricates nothing, generates no synthetic data.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
from pathlib import Path

# --------------------------------------------------------- repo bootstrap --

def find_repo_root() -> Path:
    """Locate the repository root without relying on __file__."""
    override = os.environ.get("ROGII_REPO_ROOT")
    if override:
        return Path(override).resolve()

    candidates: list[Path] = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    # when run as a script, the script's directory is a good hint too
    if "__file__" in globals():
        here = Path(globals()["__file__"]).resolve()
        candidates = [here.parent.parent, *candidates]
    for base in candidates:
        if (base / "src" / "paths.py").exists():
            return base
    raise RuntimeError(
        "Could not locate the repository root (no src/paths.py found walking up "
        f"from {Path.cwd()}). Set ROGII_REPO_ROOT to the checkout directory."
    )


def _bootstrap() -> Path:
    root = find_repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# --------------------------------------------------------------- the steps --

STEPS = [
    # (label, module, callable, required)
    ("competition data audit", "scripts.audit_competition_data", "main", True),
    ("task presentation audit", "scripts.audit_task_presentation", "main", False),
    ("sample submission audit", "scripts.audit_submission", "main", False),
    ("external artifact + leakage audit", "scripts.audit_external_resources", "main", False),
]


def run_all(*, skip_existing: bool = False, verbose: bool = True) -> dict:
    """Run every audit. Returns a structured result dict."""
    root = _bootstrap()

    from src.paths import (
        COMPETITION_ROOT,
        EXTERNAL_RESOURCES,
        REPORTS_DIR,
        SAMPLE_SUBMISSION,
        TASK_PPTX,
        available,
        ensure_reports_dir,
        require_competition_data,
    )

    # Hard gate: no competition data, no audit.
    require_competition_data()

    reports_dir = ensure_reports_dir()
    if verbose:
        print(f"repo root   : {root}")
        print(f"competition : {COMPETITION_ROOT}")
        print(f"reports dir : {reports_dir}\n")

    before = {p.name: p.stat().st_mtime for p in reports_dir.glob("**/*") if p.is_file()}

    results: list[dict] = []
    for label, module_name, func_name, required in STEPS:
        # Skip optional steps whose inputs are simply not mounted.
        if module_name.endswith("audit_task_presentation") and not available(TASK_PPTX):
            results.append({"step": label, "status": "SKIPPED", "detail": f"missing {TASK_PPTX}"})
            if verbose:
                print(f"[SKIP] {label}: presentation not mounted")
            continue
        if module_name.endswith("audit_submission") and not available(SAMPLE_SUBMISSION):
            results.append({"step": label, "status": "SKIPPED", "detail": f"missing {SAMPLE_SUBMISSION}"})
            if verbose:
                print(f"[SKIP] {label}: sample_submission.csv not mounted")
            continue
        if module_name.endswith("audit_external_resources") and not any(
            available(p) for p in EXTERNAL_RESOURCES.values()
        ):
            results.append({"step": label, "status": "SKIPPED",
                            "detail": "no auxiliary datasets mounted"})
            if verbose:
                print(f"[SKIP] {label}: no auxiliary datasets mounted")
            continue

        if verbose:
            print(f"=== {label} ===")
        t0 = time.time()
        try:
            module = importlib.import_module(module_name)
            importlib.reload(module)  # avoid a stale module in a long notebook session
            getattr(module, func_name)()
            results.append({"step": label, "status": "OK", "seconds": round(time.time() - t0, 1)})
        except Exception as exc:
            results.append({
                "step": label,
                "status": "FAILED" if required else "FAILED (optional)",
                "detail": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.time() - t0, 1),
            })
            if verbose:
                print(f"[ERROR] {label}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            if required:
                break

    # ---- decision table (created once, never silently overwritten) --------
    dt = reports_dir / "decision_table.md"
    if not dt.exists():
        dt.write_text(_decision_table_template(), encoding="utf-8")
        results.append({"step": "decision table", "status": "CREATED", "detail": str(dt)})
    else:
        results.append({"step": "decision table", "status": "EXISTS (left untouched)", "detail": str(dt)})

    # ---- final listing ----------------------------------------------------
    generated = sorted(p for p in reports_dir.glob("**/*") if p.is_file())
    if verbose:
        print("\n" + "=" * 60)
        print("STEP SUMMARY")
        for r in results:
            extra = f" — {r['detail']}" if r.get("detail") else ""
            print(f"  {r['status']:<24} {r['step']}{extra}")
        print("\nREPORTS GENERATED")
        for p in generated:
            flag = "" if p.name in before and p.stat().st_mtime == before[p.name] else " *"
            print(f"  {p.relative_to(reports_dir)}  ({p.stat().st_size:,} bytes){flag}")
        print(f"\n  {len(generated)} file(s) in {reports_dir}   (* = written this run)")

    failed = [r for r in results if r["status"].startswith("FAILED")]
    return {
        "repo_root": str(root),
        "reports_dir": str(reports_dir),
        "steps": results,
        "reports": [str(p) for p in generated],
        "ok": not failed,
    }


def _decision_table_template() -> str:
    return """# Section 7 — Resource Decision Table

Generated skeleton. **Fill this in from the audit reports in this directory** —
do not leave the "NEEDS FURTHER REVIEW" rows unresolved before modelling.

| Resource | Purpose | Allowed to use? | Why? | Required attribution | Potential leakage risk | Required preprocessing | Expected benefit | Final decision |
|---|---|---|---|---|---|---|---|---|
| `train/*__horizontal_well.csv` | Supervised signal MD/X/Y/Z/GR -> TVT | Yes | Official competition data | None | None | Per-well GR interpolation; preserve row order; GroupKFold by well | Core of the model | **USE** |
| `train/*__typewell.csv` | Stratigraphic TVT->GR->Geology reference | Yes | Official competition data | None | None | Interpolate GR onto a TVT grid | High, target-free prior | **USE** |
| `test/*` | Inference inputs | Yes | Official competition data | None | None | Identical loader path to train | Required | **USE** |
| `sample_submission.csv` | Output contract | Yes | Official | None | None | Drive the validator from it | Required | **USE** |
| Formation marker columns | Distance-to-marker features | Conditional | Check `input_inventory.md` for presence in test | None | Train/serve skew if train-only | Impute via fold-trained structural model | High if correct | **USE WITH CARE** |
| `koolbox-offline` | Offline wheels | See `koolbox_audit.md` | Not required by the baseline | Package + version + license | Low | pip --no-index if adopted | None for baseline | **DO NOT USE** |
| `rogii-claude-models-pub` | Third-party models | See `artifact_compatibility.md` | Provenance unverified | Dataset owner + URL | Unknown, treat as high | Schema + provenance proof first | Unknown | **NEEDS FURTHER REVIEW** |
| `wellbore-geology-prediction-artifacts` | Third-party models/OOF | See leakage audit | Provenance unverified | Dataset owner + URL | Unknown, treat as high | Confirm training wells subset of train | Unknown | **NEEDS FURTHER REVIEW** |
"""


def main() -> int:
    try:
        result = run_all()
    except Exception as exc:
        print(f"\nAUDIT ABORTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("\n" + ("ALL AUDITS OK" if result["ok"] else "SOME AUDITS FAILED"))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
