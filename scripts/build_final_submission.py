#!/usr/bin/env python3
"""
Final Kaggle inference and submission pipeline — Ridge Default.

Active final model:
  Ridge Default
  alignment_features=False
  spatial=None

Forbidden (must never enter final pipeline):
  - Particle Filter
  - Beam Search
  - Direct dip-constrained alignment
  - GR imputation (rolling/bounded-fill scheme rejected in real 770-well run)
  - GR quality scalar model
  - Spatial features
  - External artifacts / Koolbox / Typewell Geology / Formation markers
  - Train-target lookup / Public duplicate test-well shortcut
  - Hidden TVT values as inference features

This script:
  1. Loads real Kaggle data from /kaggle/input/competitions/rogii-wellbore-geology-prediction
     (respecting ROGII_COMPETITION_ROOT override).
  2. Discovers train/test wells dynamically.
  3. Excludes three visible public duplicate test wells from training.
  4. Trains final Ridge Default model on all eligible training wells (770).
  5. Uses only inference-safe features (MD, X, Y, Z, GR, visible TVT_input-derived,
     Typewell TVT, Typewell GR) via the existing validated feature factory.
  6. Preserves original row order, predicts only hidden suffix, never overwrites
     visible prefix.
  7. Matches submission IDs exactly to sample_submission.csv, preserving order.
  8. Validates submission (columns, row count, ID order, no duplicate/missing,
     no NaN/inf, finite numeric TVT).
  9. Writes submission.csv and submission_audit.json under --output-dir.

Fails closed on:
  - Missing Kaggle mount
  - Wrong train/test counts when --expect-* supplied
  - Forbidden feature entering inference matrix
  - Submission ID mismatch
  - Non-finite predictions
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import resource
import platform
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

# Bootstrap repository root for imports (works inside Kaggle notebook cells)
try:
    from scripts._bootstrap import bootstrap
except ImportError:
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import pandas as pd

from src.paths import (
    COMPETITION_ROOT,
    TRAIN_DIR,
    TEST_DIR,
    SAMPLE_SUBMISSION,
    KAGGLE_COMPETITION_ROOT,
    KAGGLE_WORKING,
    require_competition_data,
    CompetitionDataMissing,
)
from src.data import discover_wells, load_well
from src.tasks import make_task, TaskConstructionError
from src.baselines import RidgeBaseline
from src.validation import BLOCKED_WELL_IDS, filter_blocked, assert_no_blocked_wells
from src.manifest import verify_manifest_against_data, assert_manifest_matches_data
from src.submission import validate_submission, audit_sample_submission
from src.features import feature_columns
from src.resources import detect_resources

# Constants
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/final_submission")
ALLOWED_FEATURES = set(feature_columns(alignment_features=False))

# Forbidden substrings that must never appear in final feature list
# (alignment, particle, beam, spatial, dip, geology, formation markers are excluded by ALLOWED_FEATURES,
# but we add an explicit guard)
FORBIDDEN_FEATURE_SUBSTRINGS = [
    "align_tvt",
    "align_score",
    "align_shift",
    "align_gradient",
    "pf_",
    "beam_",
    "nbr_",
    "dip_",
    "geology",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
]

ID_SPLIT_RE = re.compile(r"^(.*?)[_\-:](\d+)$")


def split_id(raw: str) -> tuple[str, str | None]:
    m = ID_SPLIT_RE.match(str(raw))
    if m:
        return m.group(1).rstrip("_-:"), m.group(2)
    return str(raw), None


def peak_rss_mb() -> float:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024
    except Exception:
        return float("nan")


def get_git_info() -> dict:
    info = {"commit": "unknown", "branch": "unknown", "author_name": "unknown", "author_email": "unknown"}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        if commit:
            info["commit"] = commit
    except Exception:
        pass
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        if branch:
            info["branch"] = branch
    except Exception:
        pass
    try:
        name = subprocess.check_output(["git", "config", "user.name"], stderr=subprocess.DEVNULL, text=True).strip()
        if name:
            info["author_name"] = name
    except Exception:
        pass
    try:
        email = subprocess.check_output(["git", "config", "user.email"], stderr=subprocess.DEVNULL, text=True).strip()
        if email:
            info["author_email"] = email
    except Exception:
        pass
    return info


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto", help="Device selector; Ridge is CPU-safe, GPU request is ignored with warning")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Writable output directory for submission.csv and audit")
    ap.add_argument("--expect-train", type=int, default=None, help="Fail unless exactly this many train wells are found (smoke test: 773 on real mount)")
    ap.add_argument("--expect-test", type=int, default=None, help="Optional: fail unless exactly this many test wells are found (visible public smoke test: 3). Not required for final hidden rerun")
    ap.add_argument("--clear-cache", action="store_true", help="Clear feature cache if cache-dir is set (no-op for Ridge Default)")
    ap.add_argument("--cache-dir", type=str, default=None, help="Optional cache directory (unused for Ridge Default but accepted for CLI compatibility)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device handling (Ridge is CPU)
    try:
        resources = detect_resources(args.device)
        print(f"[device] requested={args.device} selected={resources.selected} gpu={resources.gpu_name or 'none'}")
        if args.device == "gpu":
            print("[warning] Ridge Default is CPU-only; GPU request ignored, running on CPU")
    except Exception as e:
        print(f"[device] resource detection failed: {e}; continuing on CPU")

    # Cache handling (no-op for Ridge, but support clear)
    if args.cache_dir and args.clear_cache:
        cache_path = Path(args.cache_dir).expanduser()
        if cache_path.exists():
            import shutil
            print(f"[cache] clearing {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)

    # 1. Fail closed if Kaggle mount absent
    try:
        require_competition_data(need_sample_submission=True)
    except CompetitionDataMissing as exc:
        print(f"[fatal] Competition data missing: {exc}", file=sys.stderr)
        return 2

    if not TRAIN_DIR.exists() or not TEST_DIR.exists() or not SAMPLE_SUBMISSION.exists():
        print(f"[fatal] Expected competition layout not found: TRAIN_DIR={TRAIN_DIR} TEST_DIR={TEST_DIR} SAMPLE={SAMPLE_SUBMISSION}", file=sys.stderr)
        return 2

    # 2. Discover wells dynamically
    train_files = discover_wells("train")
    test_files = discover_wells("test")
    n_train_discovered = len(train_files)
    n_test_discovered = len(test_files)

    print(f"[discovery] train={n_train_discovered} test={n_test_discovered}")
    print(f"[discovery] train dir={TRAIN_DIR} test dir={TEST_DIR}")

    if args.expect_train is not None and n_train_discovered != args.expect_train:
        print(f"[fatal] expected {args.expect_train} train wells, discovered {n_train_discovered} in {TRAIN_DIR}. Refusing to run.", file=sys.stderr)
        return 2
    if args.expect_test is not None and n_test_discovered != args.expect_test:
        print(f"[fatal] expected {args.expect_test} test wells, discovered {n_test_discovered} in {TEST_DIR}.", file=sys.stderr)
        return 2

    # 3. Blocked IDs guard
    blocked_present_train = sorted(set(train_files.keys()) & set(BLOCKED_WELL_IDS))
    blocked_present_test = sorted(set(test_files.keys()) & set(BLOCKED_WELL_IDS))
    print(f"[guard] blocked IDs in train: {blocked_present_train or 'none'} (will be excluded)")
    print(f"[guard] blocked IDs in test: {blocked_present_test or 'none'} (visible public examples)")

    eligible_train_ids = filter_blocked(list(train_files.keys()))
    n_eligible = len(eligible_train_ids)
    print(f"[guard] eligible train wells after excluding blocked: {n_eligible}")

    try:
        assert_no_blocked_wells(eligible_train_ids, context="final training universe")
    except Exception as exc:
        print(f"[fatal] Blocked well guard triggered: {exc}", file=sys.stderr)
        return 2

    # 4. Manifest verification (fail closed if schema mismatch)
    probe_train = None
    probe_test = None
    if eligible_train_ids:
        try:
            entry = train_files.get(eligible_train_ids[0])
            if entry:
                probe_train = load_well(entry)
        except Exception:
            pass
    if test_files:
        try:
            first_test_id = sorted(test_files.keys())[0]
            probe_test = load_well(test_files[first_test_id])
        except Exception:
            pass

    if probe_train is None or probe_test is None:
        print(f"[fatal] Cannot verify manifest: probe_train={probe_train is not None} probe_test={probe_test is not None}", file=sys.stderr)
        return 2

    try:
        train_tw_cols = list(probe_train.tw.columns) if probe_train.tw is not None else None
        test_tw_cols = list(probe_test.tw.columns) if probe_test.tw is not None else None
        verification = verify_manifest_against_data(
            probe_train.hw.columns,
            probe_test.hw.columns,
            train_tw_columns=train_tw_cols,
            test_tw_columns=test_tw_cols,
        )
        assert_manifest_matches_data(
            probe_train.hw.columns,
            probe_test.hw.columns,
            train_tw_columns=train_tw_cols,
            test_tw_columns=test_tw_cols,
        )
        print(f"[manifest] schema verification OK ({len(verification)} raw features checked)")
    except Exception as exc:
        print(f"[fatal] Manifest verification failed: {exc}", file=sys.stderr)
        return 2

    # 5. Build training tasks (real mode, residual target)
    training_tasks = []
    training_failures = []
    for wid in sorted(eligible_train_ids):
        entry = train_files.get(wid)
        if entry is None:
            continue
        try:
            well = load_well(entry)
            task = make_task(well, mode="real")
            if task.target is None:
                training_failures.append((wid, "no_target"))
                continue
            training_tasks.append(task)
        except TaskConstructionError as e:
            training_failures.append((wid, str(e)))
        except Exception as e:
            training_failures.append((wid, f"{type(e).__name__}: {e}"))

    print(f"[training] built {len(training_tasks)} tasks from {n_eligible} eligible wells; failures={len(training_failures)}")
    if training_failures:
        for wid, reason in training_failures[:10]:
            print(f"  failure {wid}: {reason}")
    if not training_tasks:
        print("[fatal] No training tasks built", file=sys.stderr)
        return 2

    # 6. Train final Ridge Default model
    # Active validated baseline: Ridge Default, alignment_features=False, spatial=None
    model = RidgeBaseline(alpha=10.0, alignment_features=False, spatial=None)

    try:
        model.fit(training_tasks)
    except Exception as exc:
        print(f"[fatal] Model training failed: {exc}", file=sys.stderr)
        return 2

    feature_list = getattr(model, "feature_names_", None)
    if not feature_list:
        # fallback to allowed features if fit produced no list (should not happen)
        feature_list = sorted(ALLOWED_FEATURES)
    else:
        feature_list = list(feature_list)

    print(f"[model] Ridge Default trained: alpha=10.0 alignment_features=False spatial=None")
    print(f"[model] feature count={len(feature_list)} features={feature_list[:10]}...")

    # Guard: forbidden feature check
    # Ensure no alignment, particle, beam, spatial, dip, geology, formation markers
    for f in feature_list:
        for forbidden in FORBIDDEN_FEATURE_SUBSTRINGS:
            if forbidden.lower() in f.lower():
                print(f"[fatal] Forbidden feature '{forbidden}' found in feature list via '{f}'. Failing closed.", file=sys.stderr)
                return 2

    # Ensure feature list is subset of allowed
    extra = set(feature_list) - ALLOWED_FEATURES
    if extra:
        print(f"[fatal] Feature list contains extra/forbidden features not in allowed Ridge Default set: {sorted(extra)}", file=sys.stderr)
        return 2

    # 7. Test inference: discover test wells, predict hidden suffix
    # Load sample submission to get contract
    try:
        sample_df = pd.read_csv(SAMPLE_SUBMISSION)
    except Exception as exc:
        print(f"[fatal] Could not read sample_submission {SAMPLE_SUBMISSION}: {exc}", file=sys.stderr)
        return 2

    if list(sample_df.columns) != ["id", "tvt"]:
        print(f"[fatal] Sample submission columns mismatch: got {list(sample_df.columns)} expected ['id','tvt']", file=sys.stderr)
        return 2

    ordered_sample_ids = sample_df["id"].astype(str).tolist()
    n_submission_rows = len(ordered_sample_ids)

    # Parse sample IDs into well groups
    sample_well_to_ids = defaultdict(list)  # well_part -> list of (pos, id, tail_int)
    for pos, sid in enumerate(ordered_sample_ids):
        well_part, tail = split_id(sid)
        tail_int = int(tail) if tail is not None and tail.isdigit() else None
        sample_well_to_ids[well_part].append((pos, sid, tail_int))

    # Build mapping of well -> sorted IDs in sample order
    # Preserve sample order per well (order of appearance in sample file)
    sample_well_ordered_ids = {}
    for well_part, lst in sample_well_to_ids.items():
        # sort by original position to preserve sample order, but also typically tail ascending
        lst_sorted = sorted(lst, key=lambda x: x[0])
        sample_well_ordered_ids[well_part] = [sid for _, sid, _ in lst_sorted]

    sample_wells_set = set(sample_well_ordered_ids.keys())
    test_wells_set = set(test_files.keys())
    print(f"[inference] sample distinct wells={len(sample_wells_set)} test discovered={len(test_wells_set)}")

    # Dynamic matching: every sample well must have a test file
    missing_test_files = sorted(sample_wells_set - test_wells_set)
    if missing_test_files:
        print(f"[fatal] Sample references wells with no test file: {missing_test_files[:10]}", file=sys.stderr)
        return 2

    # For hidden rerun compatibility, we do NOT fail if test dir has extra wells not in sample,
    # but we warn and ignore them unless they are required. However if we have extra test wells
    # that are not in sample, we should still log.
    extra_test_files = sorted(test_wells_set - sample_wells_set)
    if extra_test_files:
        print(f"[warning] Test directory contains wells not referenced in sample_submission: {extra_test_files[:10]} (ignored)")

    # Predict per test well
    id_to_pred = {}
    fallback_count = 0
    total_visible_rows = 0
    total_hidden_rows = 0
    inference_failures = []

    for well_part in sorted(sample_wells_set):  # iterate over sample wells to preserve determinism
        test_file_entry = test_files.get(well_part)
        if test_file_entry is None:
            # Should have been caught above
            inference_failures.append((well_part, "no_test_file"))
            continue
        try:
            well_data = load_well(test_file_entry)
            # hidden mask from well_data.hw
            hidden_mask = well_data.hw["is_hidden"].to_numpy() if "is_hidden" in well_data.hw.columns else None
            visible_mask = well_data.hw["is_visible"].to_numpy() if "is_visible" in well_data.hw.columns else None
            if hidden_mask is None:
                # fallback: identify via TVT_input first gap
                from src.data import identify_visible_prefix
                visible, _ = identify_visible_prefix(well_data.hw, well_data.roles)
                hidden_mask = ~visible
                visible_mask = visible

            n_visible = int(visible_mask.sum()) if visible_mask is not None else 0
            n_hidden = int(hidden_mask.sum())
            total_visible_rows += n_visible
            total_hidden_rows += n_hidden

            # Build InferenceTask
            task = make_task(well_data, mode="real")
            inf_task = task.inputs()

            # Safety: ensure InferenceTask has no target
            inf_task.assert_no_target()

            # Predict
            pred = model.predict(inf_task)
            pred = np.asarray(pred, dtype=np.float64)

            if pred.shape[0] != n_hidden:
                # For robustness, check against task.n_predict
                if pred.shape[0] != inf_task.n_predict:
                    print(f"[fatal] Prediction length mismatch for well {well_part}: pred={pred.shape[0]} hidden={n_hidden} task_n={inf_task.n_predict}", file=sys.stderr)
                    return 2

            # Check non-finite
            if not np.all(np.isfinite(pred)):
                # Count non-finite as fallback and fail closed per spec: "Any non-finite prediction exists" -> fail
                # We treat as fatal, not silent fallback
                n_bad = int(np.sum(~np.isfinite(pred)))
                print(f"[fatal] Non-finite predictions for well {well_part}: {n_bad} bad values", file=sys.stderr)
                return 2

            # Map predictions to sample IDs for this well
            sample_ids_for_well = sample_well_ordered_ids[well_part]
            if len(sample_ids_for_well) != n_hidden:
                print(f"[fatal] Row count mismatch for well {well_part}: sample has {len(sample_ids_for_well)} rows but hidden suffix is {n_hidden}", file=sys.stderr)
                return 2

            # Preserve original row order: hidden rows are in file order, sample IDs for this well are in sample order
            # For real data, sample order per well corresponds to increasing row index (file order), so positional mapping is correct
            for sid, p in zip(sample_ids_for_well, pred):
                if sid in id_to_pred:
                    print(f"[fatal] Duplicate sample ID {sid} encountered", file=sys.stderr)
                    return 2
                id_to_pred[sid] = float(p)

        except TaskConstructionError as e:
            inference_failures.append((well_part, f"task_construction: {e}"))
            print(f"[error] Task construction failed for {well_part}: {e}", file=sys.stderr)
        except Exception as e:
            inference_failures.append((well_part, f"{type(e).__name__}: {e}"))
            print(f"[error] Inference failed for {well_part}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    if inference_failures:
        print(f"[fatal] Inference failures: {inference_failures[:10]}", file=sys.stderr)
        return 2

    # Ensure we have predictions for all sample IDs
    missing_ids = [sid for sid in ordered_sample_ids if sid not in id_to_pred]
    if missing_ids:
        print(f"[fatal] Missing predictions for {len(missing_ids)} sample IDs, e.g. {missing_ids[:5]}", file=sys.stderr)
        return 2

    # Build final submission in exact sample order
    preds_ordered = [id_to_pred[sid] for sid in ordered_sample_ids]
    submission_df = pd.DataFrame({"id": ordered_sample_ids, "tvt": preds_ordered})

    # Final safety: no NaN/inf
    if submission_df["tvt"].isna().any():
        print("[fatal] Submission contains NaN", file=sys.stderr)
        return 2
    arr = pd.to_numeric(submission_df["tvt"], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(arr)):
        print("[fatal] Submission contains non-finite values", file=sys.stderr)
        return 2

    # Validate using existing validator (fail closed)
    validation_report = validate_submission(submission_df, SAMPLE_SUBMISSION, require_exact_order=True)
    print(validation_report)
    if not validation_report.passed:
        print(f"[fatal] Submission validation FAILED: {validation_report.to_dict()}", file=sys.stderr)
        return 2

    # Write submission.csv
    submission_path = output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"[output] Wrote {submission_path} ({len(submission_df)} rows)")

    # Compute stats
    pred_min = float(np.min(arr)) if arr.size else float("nan")
    pred_max = float(np.max(arr)) if arr.size else float("nan")
    pred_mean = float(np.mean(arr)) if arr.size else float("nan")
    pred_std = float(np.std(arr)) if arr.size else float("nan")
    runtime = time.perf_counter() - t_start
    rss_mb = peak_rss_mb()
    git_info = get_git_info()

    # Audit
    audit = {
        "repository_commit_sha": git_info["commit"],
        "repository_branch": git_info["branch"],
        "git_author_name": git_info["author_name"],
        "git_author_email": git_info["author_email"],
        "data_source": "real_kaggle",
        "dataset_root": str(COMPETITION_ROOT),
        "n_train_wells_discovered": n_train_discovered,
        "n_eligible_training_wells": n_eligible,
        "n_train_tasks_built": len(training_tasks),
        "n_training_failures": len(training_failures),
        "n_test_wells_discovered": n_test_discovered,
        "n_test_wells_used": len(sample_wells_set),
        "n_submission_rows": int(len(submission_df)),
        "n_predicted_hidden_rows": int(total_hidden_rows),
        "n_visible_prefix_rows": int(total_visible_rows),
        "feature_list": feature_list,
        "feature_manifest_version": "reports/feature_manifest.csv",
        "model_name": "ridge",
        "model_configuration": {
            "final_model": "Ridge Default",
            "alignment_features": False,
            "spatial": None,
            "particle_filter": None,
            "beam_search": None,
        },
        "ridge_hyperparameters": {"alpha": 10.0},
        "target_definition": "TVT",
        "target_transformation": "residual = TVT - anchor_tvt (last known visible TVT_input); final_prediction = anchor_tvt + predicted_residual",
        "runtime_seconds": float(runtime),
        "peak_rss_mb": float(rss_mb) if np.isfinite(rss_mb) else None,
        "prediction_min": pred_min,
        "prediction_max": pred_max,
        "prediction_mean": pred_mean,
        "prediction_std": pred_std,
        "fallback_count": int(fallback_count),
        "n_missing_predictions": 0,
        "submission_validation_status": "PASS" if validation_report.passed else "FAIL",
        "submission_validation_report": validation_report.to_dict(),
        "public_duplicate_test_wells_excluded": sorted(list(BLOCKED_WELL_IDS)),
        "external_artifacts_used": False,
        "synthetic_generator_used": False,
        "data_source_detailed": "real_kaggle",
        "final_model": "Ridge Default",
        "alignment_features": False,
        "spatial_features": False,
        "final_submission_authorized": bool(validation_report.passed),
        "forbidden_features_check": "PASS",
        "allowed_features": sorted(list(ALLOWED_FEATURES)),
        # Additional explicit fields requested in first spec
        "number_of_train_wells": n_train_discovered,
        "number_of_eligible_training_wells": n_eligible,
        "number_of_test_wells": n_test_discovered,
        "number_of_submission_rows": int(len(submission_df)),
        "model_config": {"alignment_features": False, "spatial": None},
        "validation_status": "PASS" if validation_report.passed else "FAIL",
        "device_requested": args.device,
        "output_dir": str(output_dir),
        "expect_train": args.expect_train,
        "expect_test": args.expect_test,
    }

    # Explicit flags required
    audit["data_source"] = "real_kaggle"
    audit["synthetic_generator_used"] = False
    audit["final_model"] = "Ridge Default"

    audit_path = output_dir / "submission_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"[output] Wrote {audit_path}")
    print(f"[audit] final_submission_authorized={audit['final_submission_authorized']} data_source={audit['data_source']} model={audit['final_model']}")

    # Final check: if not authorized, fail
    if not audit["final_submission_authorized"]:
        print("[fatal] Final submission not authorized", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
