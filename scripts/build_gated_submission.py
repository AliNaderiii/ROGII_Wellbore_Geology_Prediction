#!/usr/bin/env python3
"""
Promotion-gated final Kaggle submission builder for the trajectory stack.

This script writes a submission **only** when a real-mount experiment has
already promoted a candidate arm. It requires:

    --require-promotion reports/real_trajectory_stack_decision.json

and refuses (exit 2) when the decision file is missing, is not a real-run
decision, or promotes nothing. In every other respect it follows the exact
same safety contract as the validated ``build_final_submission.py``:

  1. Loads the real Kaggle mount (ROGII_COMPETITION_ROOT override honoured).
  2. Discovers train/test wells dynamically.
  3. Excludes the three blocked public duplicate wells from training.
  4. Verifies the feature manifest against the observed schema.
  5. Fits the promoted arm on all eligible training wells in ``real`` mode.
  6. Predicts through the SAME inference code path validated in the
     experiment; every internal guard failure inside the arm returns the
     exact Ridge anchor prediction (bit-identical to Ridge Default output).
  7. Derives submission IDs and row count exclusively from the active
     sample_submission.csv (never --expect-test for the hidden finale).
  8. Validates: exact columns, exact row count, exact ID order, no
     duplicates/missing, finite predictions, no constant placeholder.
  9. Writes both /kaggle/working/final_submission/submission.csv and
     /kaggle/working/submission.csv, plus submission_audit.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict

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
    KAGGLE_WORKING,
    require_competition_data,
    CompetitionDataMissing,
)
from src.cache import FeatureCache
from src.data import discover_wells, load_well
from src.tasks import make_task, TaskConstructionError
from src.baselines import RidgeBaseline
from src.geoanchor import MemoizedPathGenerator
from src.manifest import verify_manifest_against_data, assert_manifest_matches_data
from src.submission import validate_submission
from src.validation import BLOCKED_WELL_IDS, filter_blocked, assert_no_blocked_wells
from src.trajectory_stack import (
    ARM_GATED,
    ARM_LABELS,
    ARM_ORDER,
    ARM_RIDGE,
    STACK_CANDIDATES,
    TRAJECTORY_STACK_VERSION,
    StackConfig,
    TrajectoryGateConfig,
    build_stack_models,
)
from scripts.build_final_submission import split_id

DEFAULT_OUTPUT_DIR = Path("/kaggle/working/final_submission")


def peak_rss_mb() -> float:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024
    except Exception:
        return float("nan")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_info() -> dict:
    info = {"commit": "unknown", "branch": "unknown"}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass
    return info


def load_promotion_decision(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"[fatal] promotion decision file not found: {path}")
    try:
        decision = json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"[fatal] promotion decision file is not valid JSON: {exc}")
    if decision.get("schema") != "trajectory-stack-promotion/v1":
        raise SystemExit("[fatal] decision file is not a trajectory-stack promotion record")
    if not decision.get("is_real_run"):
        raise SystemExit(
            "[fatal] the decision comes from a synthetic/partial run; no arm may be "
            "promoted from it. Run scripts/run_trajectory_stack_experiment.py "
            "--expect-wells 770 on the real mount first."
        )
    if not decision.get("promoted") or not decision.get("promoted_arm"):
        raise SystemExit(
            "[fatal] the real-run decision promotes NO candidate; Ridge Default stands. "
            "Use scripts/build_final_submission.py instead."
        )
    arm = decision["promoted_arm"]
    if arm not in ARM_ORDER or arm == ARM_RIDGE:
        raise SystemExit(f"[fatal] decision promotes unknown arm {arm!r}")
    ref = decision.get("reference", {}).get("unseen_well_rmse")
    if ref is None:
        raise SystemExit("[fatal] decision file lacks the Ridge reference RMSE")
    from src.trajectory_stack import RIDGE_REFERENCE_UNSEEN_RMSE

    if abs(float(ref) - RIDGE_REFERENCE_UNSEEN_RMSE) > 8:  # audit sanity, not tuning
        raise SystemExit(
            "[fatal] decision reference inconsistent with the verified Ridge reference"
        )
    return decision


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-promotion", required=True,
                    help="path to real_trajectory_stack_decision.json")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--mirror-dir", default=str(KAGGLE_WORKING),
                    help="directory for the second, flat submission.csv copy")
    ap.add_argument("--expect-train", type=int, default=None,
                    help="visible public smoke test only (773 on the real mount)")
    ap.add_argument("--expect-test", type=int, default=None,
                    help="visible public smoke test only (3); NEVER for the hidden finale")
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--tune-splits", type=int, default=3)
    ap.add_argument("--boost-max-iter", type=int, default=400)
    ap.add_argument("--boost-estop-rounds", type=int, default=50)
    ap.add_argument("--boost-threads", type=int, default=4)
    ap.add_argument("--path-cache", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="defaults to the seed recorded in the promotion decision")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()

    decision = load_promotion_decision(Path(args.require_promotion))
    arm = str(decision["promoted_arm"])
    seed = int(args.seed if args.seed is not None
               else decision.get("environment_key", {}).get("seed", 0))
    val_splits = int(decision.get("environment_key", {}).get("n_splits", 5))
    print(f"[promotion] decision OK: arm={arm} from a REAL run "
          f"(outer folds={val_splits}, seed={seed})")
    print(f"[promotion] reference: {decision.get('reference', {})}")
    ev = decision.get("arms", {}).get(arm, {})
    unseen = ev.get("metrics", {}).get("unseen_well", {}).get("candidate", {}).get("global_rmse")
    print(f"[promotion] validated unseen_well global RMSE of arm: {unseen}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = Path(args.mirror_dir).expanduser().resolve()

    try:
        require_competition_data(need_sample_submission=True)
    except CompetitionDataMissing as exc:
        print(f"[fatal] Competition data missing: {exc}", file=sys.stderr)
        return 2
    if not TRAIN_DIR.exists() or not TEST_DIR.exists() or not SAMPLE_SUBMISSION.exists():
        print("[fatal] Expected competition layout not found", file=sys.stderr)
        return 2

    train_files = discover_wells("train")
    test_files = discover_wells("test")
    print(f"[discovery] train={len(train_files)} test={len(test_files)}")
    if args.expect_train is not None and len(train_files) != args.expect_train:
        print(f"[fatal] expected {args.expect_train} train wells, got {len(train_files)}", file=sys.stderr)
        return 2
    if args.expect_test is not None and len(test_files) != args.expect_test:
        print(f"[fatal] expected {args.expect_test} test wells, got {len(test_files)} (smoke-only flag)", file=sys.stderr)
        return 2

    eligible_train_ids = filter_blocked(list(train_files.keys()))
    # The promotion decision must have been produced on THIS universe: a
    # decision from a synthetic field or a partial run can never license a
    # submission against the real mount (and vice versa).
    env_key = decision.get("environment_key", {})
    want_train = env_key.get("n_train_wells_discovered")
    want_eligible = env_key.get("n_eligible_wells")
    if (
        want_train is not None
        and want_eligible is not None
        and (int(want_train) != len(train_files) or int(want_eligible) != len(eligible_train_ids))
    ):
        print(
            f"[fatal] promotion decision universe ({want_train}/{want_eligible}) does not "
            f"match the discovered mount ({len(train_files)}/{len(eligible_train_ids)}); "
            "re-run the experiment on this mount before submitting.",
            file=sys.stderr,
        )
        return 2
    try:
        assert_no_blocked_wells(eligible_train_ids, context="final training universe")
    except Exception as exc:
        print(f"[fatal] Blocked well guard triggered: {exc}", file=sys.stderr)
        return 2
    print(f"[guard] eligible train wells: {len(eligible_train_ids)} "
          f"(blocked: {sorted(set(train_files) & set(BLOCKED_WELL_IDS)) or 'none in train'})")

    probe_train = load_well(train_files[eligible_train_ids[0]]) if eligible_train_ids else None
    probe_test = load_well(test_files[sorted(test_files)[0]]) if test_files else None
    if probe_train is None or probe_test is None:
        print("[fatal] Cannot verify manifest: probes failed", file=sys.stderr)
        return 2
    try:
        verification = verify_manifest_against_data(
            probe_train.hw.columns,
            probe_test.hw.columns,
            train_tw_columns=list(probe_train.tw.columns) if probe_train.tw is not None else None,
            test_tw_columns=list(probe_test.tw.columns) if probe_test.tw is not None else None,
        )
        assert_manifest_matches_data(
            probe_train.hw.columns,
            probe_test.hw.columns,
            train_tw_columns=list(probe_train.tw.columns) if probe_train.tw is not None else None,
            test_tw_columns=list(probe_test.tw.columns) if probe_test.tw is not None else None,
        )
        print(f"[manifest] schema verification OK ({len(verification)} raw features checked)")
    except Exception as exc:
        print(f"[fatal] Manifest verification failed: {exc}", file=sys.stderr)
        return 2

    # ----- training tasks --------------------------------------------------
    training_tasks, training_failures = [], []
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
    print(f"[training] built {len(training_tasks)} tasks; failures={len(training_failures)}")
    if not training_tasks:
        print("[fatal] No training tasks built", file=sys.stderr)
        return 2

    # ----- fit the promoted arm (shared anchor instance) --------------------
    memo: dict = {}
    path_cache = FeatureCache(args.path_cache) if args.path_cache else None

    def _pf():
        from src.particle_filter import ParticleFilterFeatureGenerator

        return MemoizedPathGenerator(
            ParticleFilterFeatureGenerator(
                cache=path_cache, dataset_version="rogii-mounted-v1",
                fold_id="final", protocol="unseen_well", device="cpu",
            ),
            memo,
            "pf",
        )

    def _beam():
        from src.beam_search import BeamSearchFeatureGenerator

        return MemoizedPathGenerator(
            BeamSearchFeatureGenerator(
                cache=path_cache, dataset_version="rogii-mounted-v1",
                fold_id="final", protocol="unseen_well", device="cpu",
            ),
            memo,
            "beam",
        )

    anchor = RidgeBaseline(alignment_features=False)
    anchor.name = ARM_RIDGE
    t_anchor = time.perf_counter()
    try:
        anchor.fit(training_tasks)
    except Exception as exc:
        print(f"[fatal] Ridge anchor fit failed: {exc}", file=sys.stderr)
        return 2
    print(f"[model] Ridge anchor fitted in {time.perf_counter() - t_anchor:.1f}s")

    arm_decisions: list = []
    models = build_stack_models(
        (arm,),
        anchor_model=anchor,
        pf_factory=_pf,
        beam_factory=_beam,
        stack_config=StackConfig(
            inner_splits=args.inner_splits, tune_splits=args.tune_splits,
            boost_max_iter=args.boost_max_iter, boost_estop_rounds=args.boost_estop_rounds,
            boost_threads=args.boost_threads, seed=seed,
        ),
        gate_config=TrajectoryGateConfig(
            inner_splits=args.inner_splits, tune_splits=args.tune_splits,
            boost_max_iter=args.boost_max_iter, boost_estop_rounds=args.boost_estop_rounds,
            boost_threads=args.boost_threads, seed=seed,
        ),
        protocol="final",
        fold=-1,
        decision_log=arm_decisions,
        boost_kw=dict(seed=seed, max_iter=args.boost_max_iter,
                      estop_rounds=args.boost_estop_rounds, thread_count=args.boost_threads),
    )
    model = models[arm]
    t_fit = time.perf_counter()
    try:
        model.fit(training_tasks)
    except Exception as exc:
        print(f"[fatal] promoted arm fit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    fit_seconds = time.perf_counter() - t_fit
    arm_info: dict = {}
    if arm == "oof_meta_stack":
        arm_info = dict(model.stack.info)
    elif arm == ARM_GATED:
        arm_info = dict(vars(model.info))
        arm_info.update({
            "killed": model.killed,
            "kill_reason": model.kill_reason,
            "thresholds": {
                "margin": model.thresholds.margin,
                "conf_thr": model.thresholds.conf_thr,
                "sep_cap": model.thresholds.sep_cap,
                "shrink": model.thresholds.shrink,
                "warmup": model.thresholds.warmup,
            },
            "oof_skills": dict(model._oof_skills),
        })
        if model.killed:
            print(f"[gate] KILL SWITCH active at final fit ({model.kill_reason}); "
                  "predictions are the exact Ridge Default output")
    else:
        if hasattr(model, "info") and hasattr(model.info, "library"):
            arm_info = asdict(model.info)
    print(f"[model] {arm} fitted in {fit_seconds:.1f}s")

    # ----- sample contract ---------------------------------------------------
    try:
        sample_df = pd.read_csv(SAMPLE_SUBMISSION)
    except Exception as exc:
        print(f"[fatal] Could not read sample_submission {SAMPLE_SUBMISSION}: {exc}", file=sys.stderr)
        return 2
    if list(sample_df.columns) != ["id", "tvt"]:
        print(f"[fatal] Sample submission columns mismatch: {list(sample_df.columns)}", file=sys.stderr)
        return 2
    ordered_sample_ids = sample_df["id"].astype(str).tolist()
    sample_well_to_ids: dict[str, list] = defaultdict(list)
    for pos, sid in enumerate(ordered_sample_ids):
        well_part, tail = split_id(sid)
        tail_int = int(tail) if tail is not None and tail.isdigit() else None
        sample_well_to_ids[well_part].append((pos, sid, tail_int))
    sample_well_ordered_ids = {
        w: [sid for _, sid, _ in sorted(lst, key=lambda x: x[0])]
        for w, lst in sample_well_to_ids.items()
    }
    sample_wells_set = set(sample_well_ordered_ids)
    missing_test_files = sorted(sample_wells_set - set(test_files))
    if missing_test_files:
        print(f"[fatal] Sample wells with no test file: {missing_test_files[:10]}", file=sys.stderr)
        return 2
    extra_test_files = sorted(set(test_files) - sample_wells_set)
    if extra_test_files:
        print(f"[warning] test wells not in sample (ignored): {extra_test_files[:10]}")

    # ----- inference ----------------------------------------------------------
    id_to_pred: dict[str, float] = {}
    total_hidden = 0
    inference_failures = []
    gate_applied = 0
    for well_part in sorted(sample_wells_set):
        entry = test_files.get(well_part)
        try:
            well_data = load_well(entry)
            task = make_task(well_data, mode="real")
            inf_task = task.inputs()
            inf_task.assert_no_target()
            pred = np.asarray(model.predict(inf_task), dtype="float64")
            if pred.shape[0] != inf_task.n_predict:
                print(f"[fatal] prediction length mismatch for {well_part}", file=sys.stderr)
                return 2
            if not np.all(np.isfinite(pred)):
                print(f"[fatal] non-finite predictions for {well_part}", file=sys.stderr)
                return 2
            diag = model.prediction_diagnostics(inf_task, None, pred) if hasattr(model, "prediction_diagnostics") else {}
            if diag.get("gate_activation") and not diag.get("gate_fallback_exact_ridge"):
                gate_applied += 1
            n_hidden = int(inf_task.n_predict)
            total_hidden += n_hidden
            sample_ids_for_well = sample_well_ordered_ids[well_part]
            if len(sample_ids_for_well) != n_hidden:
                print(f"[fatal] row count mismatch for {well_part}: sample {len(sample_ids_for_well)} vs hidden {n_hidden}", file=sys.stderr)
                return 2
            for sid, p in zip(sample_ids_for_well, pred):
                if sid in id_to_pred:
                    print(f"[fatal] duplicate sample ID {sid}", file=sys.stderr)
                    return 2
                id_to_pred[sid] = float(p)
        except TaskConstructionError as e:
            inference_failures.append((well_part, f"task_construction: {e}"))
        except Exception as e:
            inference_failures.append((well_part, f"{type(e).__name__}: {e}"))

    if inference_failures:
        print(f"[fatal] Inference failures: {inference_failures[:10]}", file=sys.stderr)
        return 2
    missing_ids = [sid for sid in ordered_sample_ids if sid not in id_to_pred]
    if missing_ids:
        print(f"[fatal] missing predictions for {len(missing_ids)} IDs", file=sys.stderr)
        return 2

    preds_ordered = [id_to_pred[sid] for sid in ordered_sample_ids]
    submission_df = pd.DataFrame({"id": ordered_sample_ids, "tvt": preds_ordered})
    arr = submission_df["tvt"].to_numpy(dtype="float64")
    if not np.all(np.isfinite(arr)):
        print("[fatal] non-finite values in submission", file=sys.stderr)
        return 2
    if float(np.std(arr)) < 1e-9:
        print("[fatal] constant placeholder submission rejected", file=sys.stderr)
        return 2

    validation_report = validate_submission(submission_df, SAMPLE_SUBMISSION, require_exact_order=True)
    print(validation_report)
    if not validation_report.passed:
        print(f"[fatal] Submission validation FAILED: {validation_report.to_dict()}", file=sys.stderr)
        return 2

    submission_path = output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    mirror_path = mirror_dir / "submission.csv"
    if mirror_path.resolve() != submission_path.resolve():
        submission_df.to_csv(mirror_path, index=False)
    print(f"[output] wrote {submission_path} ({len(submission_df)} rows)")
    print(f"[output] wrote {mirror_path}")

    git = _git_info()
    audit = {
        "repository_commit_sha": git["commit"],
        "repository_branch": git["branch"],
        "data_source": "real_kaggle",
        "dataset_root": str(COMPETITION_ROOT),
        "promotion_decision_file": str(Path(args.require_promotion).resolve()),
        "promoted_arm": arm,
        "arm_label": ARM_LABELS.get(arm, arm),
        "arm_validation_unseen_rmse": unseen,
        "reference_ridge_unseen_rmse": decision.get("reference", {}).get("unseen_well_rmse"),
        "trajectory_stack_version": TRAJECTORY_STACK_VERSION,
        "gate_candidates": list(STACK_CANDIDATES),
        "final_model_configuration": {
            "anchor": "Ridge Default (alpha=10.0, alignment_features=False, spatial=None)",
            "arm": arm,
            "armed_with_exact_ridge_fallback": True,
            "inner_splits": args.inner_splits,
            "tune_splits": args.tune_splits,
            "boost_max_iter": args.boost_max_iter,
            "boost_estop_rounds": args.boost_estop_rounds,
            "boost_threads": args.boost_threads,
            "seed": seed,
        },
        "arm_fit_info": arm_info,
        "n_train_wells_discovered": len(train_files),
        "n_eligible_training_wells": len(eligible_train_ids),
        "n_train_tasks_built": len(training_tasks),
        "n_training_failures": len(training_failures),
        "n_test_wells_discovered": len(test_files),
        "n_test_wells_used": len(sample_wells_set),
        "n_submission_rows": int(len(submission_df)),
        "n_predicted_hidden_rows": int(total_hidden),
        "n_wells_with_applied_correction": int(gate_applied),
        "prediction_min": float(np.min(arr)),
        "prediction_max": float(np.max(arr)),
        "prediction_mean": float(np.mean(arr)),
        "prediction_std": float(np.std(arr)),
        "runtime_seconds": float(time.perf_counter() - t_start),
        "peak_rss_mb": peak_rss_mb(),
        "python": platform.python_version(),
        "public_duplicate_test_wells_excluded": sorted(BLOCKED_WELL_IDS),
        "external_artifacts_used": False,
        "synthetic_generator_used": False,
        "submission_validation_status": "PASS",
        "submission_validation_report": validation_report.to_dict(),
        "submission_sha256": _sha256(submission_path),
        "expect_train_used_for_smoke_only": args.expect_train,
        "expect_test_used_for_smoke_only": args.expect_test,
        "ids_from_active_sample_submission": True,
    }
    audit_path = output_dir / "submission_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str))
    print(f"[output] wrote {audit_path}")
    print(f"[done] promoted arm {arm}; runtime {audit['runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
