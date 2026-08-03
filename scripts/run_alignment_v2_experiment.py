#!/usr/bin/env python3
"""Run the Alignment v2 experiment.

    ROGII_COMPETITION_ROOT=/path/to/field \\
        python scripts/run_alignment_v2_experiment.py --max-wells 100
    python scripts/run_alignment_v2_experiment.py --expect-wells 770

Arms compared
-------------
ridge_default          A/L — Ridge Default anchor + exact fallback
alignment_v2           Alignment v2 gated model (the promotion candidate)
align_v2_meta_stack    Alignment v2 OOF meta-stack

Validation: ``same_well_masked`` and ``unseen_well``, GroupKFold by
well (5 folds by default), inner OOF for v2 gate and meta-stack
selection, fold-specific imputation, deterministic seeds, the three
public duplicate wells hard-blocked.

THIS EXPERIMENT NEVER WRITES A SUBMISSION. Promotion decisions are
written to ``{prefix}alignment_v2_decision.json``; a submission may only
be produced afterwards by ``scripts/build_alignment_v2_submission.py``.

Report naming is evidence-based: ``real_alignment_v2_*`` files appear
only when the discovered counts match the audited mount (773 train /
770 eligible); anything else is stamped SYNTHETIC.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import pandas as pd

from src.alignment_v2_model import (
    ALIGN_V2_MODEL_VERSION,
    PROMOTED_REFERENCE_UNSEEN_RMSE,
    AlignmentV2Config,
    OOFMetaStackV2Config,
    build_alignment_v2_arm,
)
from src.alignment_v2_decision import DEFAULT_RULES_DOC, evaluate_arm
from src.baselines import RidgeBaseline
from src.cache import FeatureCache
from src.data import discover_wells, load_well
from src.manifest import (
    assert_inference_provenance,
    assert_manifest_valid,
    safe_inference_features,
)
from src.paths import ensure_reports_dir
from src.real_ablation_reporting import (
    AUDITED_DISCOVERED_WELLS,
    AUDITED_ELIGIBLE_WELLS,
    banner_block,
    file_prefix,
    is_real_run,
)
from src.tasks import TaskConstructionError, make_task
from src.validation import (
    BLOCKED_WELL_IDS,
    PROTOCOL_A,
    PROTOCOL_B,
    CrossFitLeakage,
    assert_no_blocked_wells,
    evaluate_models,
    filter_blocked,
    make_group_folds,
    stratified_report,
    summarize,
)

ARM_RIDGE = "ridge_default"
ARM_ALIGN_V2 = "alignment_v2"
ARM_META_V2 = "align_v2_meta_stack"
ARM_ORDER = (ARM_RIDGE, ARM_ALIGN_V2, ARM_META_V2)
ARM_LABELS = {
    ARM_RIDGE: "A/L. Ridge Default (anchor + exact fallback)",
    ARM_ALIGN_V2: "B. Alignment v2 gated (promotion candidate)",
    ARM_META_V2: "C. Alignment v2 OOF meta-stack",
}


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def decision_summary_tables(decision_df: pd.DataFrame):
    """Activation rates and fallback-reason counts for the v2 gated arm."""
    if decision_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    rates = []
    for (protocol, fold), g in decision_df.groupby(["protocol", "fold"], sort=False):
        applied = g[g["outcome"] != "fallback"]
        rates.append(
            {
                "protocol": protocol,
                "fold": int(fold),
                "n_wells": int(len(g)),
                "n_applied": int(len(applied)),
                "activation_rate": float(len(applied) / len(g)) if len(g) else np.nan,
                "correction_mean_abs_ft": float(applied["correction_mean_abs"].mean()) if len(applied) else 0.0,
                "correction_max_abs_ft": float(applied["correction_max_abs"].max()) if len(applied) else 0.0,
                "shrink_mode": float(applied["shrink"].mode().iloc[0]) if len(applied) else np.nan,
                "warmup_mode": float(applied["warmup"].mode().iloc[0]) if len(applied) else np.nan,
            }
        )
    reasons = (
        decision_df[decision_df["outcome"] == "fallback"]
        .groupby(["protocol", "reason"], sort=False)
        .size()
        .reset_index(name="n_wells")
        .sort_values(["protocol", "n_wells"], ascending=[True, False])
    )
    return pd.DataFrame(rates), reasons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=5, help="outer GroupKFold depth (default 5)")
    ap.add_argument("--inner-splits", type=int, default=5, help="inner OOF depth (v2 gate + meta-stack)")
    ap.add_argument("--tune-splits", type=int, default=3, help="threshold/blend selection sub-folds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-wells", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=list(ARM_ORDER), choices=list(ARM_ORDER))
    ap.add_argument("--protocols", nargs="*", default=[PROTOCOL_A, PROTOCOL_B],
                    choices=[PROTOCOL_A, PROTOCOL_B])
    ap.add_argument("--path-cache", default=None,
                    help="optional directory for the target-free PF/Beam npz cache")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    verbose = not args.quiet
    if args.inner_splits < 2 or args.tune_splits < 2:
        raise SystemExit("--inner-splits and --tune-splits must be >= 2")

    t_start = time.perf_counter()
    reports_dir = ensure_reports_dir() if args.reports_dir is None else Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Manifest + provenance guard.
    assert_manifest_valid()
    provenance = assert_inference_provenance()
    cleared = safe_inference_features()
    if verbose:
        print(f"[1/6] manifest valid: {len(cleared)} cleared features; "
              f"provenance roots verified across {len(provenance)} entries")

    files = discover_wells("train")
    all_train_ids = sorted(files)
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
        raise RuntimeError(f"need at least {args.n_splits * 2} eligible wells, got {len(universe)}")
    if verbose:
        print(f"[2/6] wells: {len(all_train_ids)} train discovered; "
              f"{len(universe)} eligible after blocking {blocked_present or 'none'}")

    data_source = (
        "real_kaggle"
        if (len(all_train_ids) == AUDITED_DISCOVERED_WELLS and len(universe) == AUDITED_ELIGIBLE_WELLS)
        else ("synthetic_or_partial" if len(universe) < 100 else "synthetic_field")
    )
    arms = tuple(a for a in ARM_ORDER if a in set(args.arms))
    if ARM_RIDGE not in arms:
        arms = (ARM_RIDGE,) + arms

    def task_builder(ids, mode):
        tasks, skipped = [], []
        for wid in ids:
            entry = files.get(wid)
            if entry is None or entry.horizontal is None:
                skipped.append((wid, "no_files"))
                continue
            try:
                well = load_well(entry)
            except Exception as exc:
                skipped.append((wid, f"load_failed:{type(exc).__name__}"))
                continue
            try:
                tasks.append(make_task(well, mode))
            except TaskConstructionError as exc:
                skipped.append((wid, str(exc).split(":")[-1].strip()))
        return tasks, skipped

    folds = make_group_folds(universe, n_splits=args.n_splits, seed=args.seed)
    if verbose:
        print(f"      GroupKFold: {args.n_splits} folds, validation sizes "
              f"{[len(f.valid_ids) for f in folds]}")

    environment = {
        "experiment": "alignment v2 experiment (multi-scale affine cal + multi-scale alignment + DP + branch ensemble + IRLS projection + v2 OOF meta-stack + v2 two-stage gate)",
        "alignment_v2_version": ALIGN_V2_MODEL_VERSION,
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
        "arms": list(arms),
        "protocols": list(args.protocols),
        "label": args.label,
        "reference_align_v2_unseen_rmse": PROMOTED_REFERENCE_UNSEEN_RMSE,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "final_submission_created": False,
    }
    environment["is_real_run"] = is_real_run(environment)
    prefix = file_prefix(environment)

    well_rows: list = []
    failures: list = []
    gate_log_rows: list = []
    model_infos: list = []
    protocol_seconds: dict = {}
    proto_modes = {PROTOCOL_A: "masked", PROTOCOL_B: "real"}

    for i, protocol in enumerate(args.protocols):
        mode = proto_modes[protocol]
        if verbose:
            print(f"[{3 + i}/6] {protocol} (cross-fitted, arms {', '.join(arms)})")
        t0 = time.perf_counter()
        for fold in folds:
            tf = time.perf_counter()
            assert_no_blocked_wells(fold.train_ids, context=f"{protocol} fold {fold.index} train")
            assert_no_blocked_wells(fold.valid_ids, context=f"{protocol} fold {fold.index} valid")
            train_tasks, sk_train = task_builder(fold.train_ids, mode)
            valid_tasks, sk_valid = task_builder(fold.valid_ids, mode)
            for wid, reason in sk_train + sk_valid:
                failures.append({"stage": "task", "model": "", "well_id": wid, "error": reason})
            if not train_tasks or not valid_tasks:
                continue
            overlap = {t.well_id for t in train_tasks} & {t.well_id for t in valid_tasks}
            if overlap:
                raise CrossFitLeakage(
                    f"{protocol} fold {fold.index}: wells in both train and valid: {sorted(overlap)[:5]}"
                )

            anchor = RidgeBaseline(alignment_features=False)
            anchor.name = ARM_RIDGE
            fold_gate_log: list = []
            try:
                anchor.fit(train_tasks)
            except Exception as exc:
                failures.append({"stage": "fit", "model": ARM_RIDGE, "well_id": "",
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue

            models: dict = {ARM_RIDGE: anchor}
            for arm in arms:
                if arm == ARM_RIDGE:
                    continue
                t_fit = time.perf_counter()
                try:
                    if arm == ARM_ALIGN_V2:
                        m = build_alignment_v2_arm(
                            anchor_model=anchor,
                            config=AlignmentV2Config(
                                inner_splits=args.inner_splits,
                                tune_splits=args.tune_splits,
                                seed=args.seed,
                            ),
                            protocol=protocol,
                            fold=fold.index,
                            decision_log=fold_gate_log,
                        )[ARM_ALIGN_V2]
                        m.fit(train_tasks)
                        models[ARM_ALIGN_V2] = m
                    elif arm == ARM_META_V2:
                        m = build_alignment_v2_arm(
                            anchor_model=anchor,
                            meta_config=OOFMetaStackV2Config(
                                inner_splits=args.inner_splits,
                                tune_splits=args.tune_splits,
                                seed=args.seed,
                            ),
                        )[ARM_META_V2]
                        m.fit(train_tasks)
                        models[ARM_META_V2] = m
                except Exception as exc:
                    failures.append({"stage": "fit", "model": arm, "well_id": "",
                                     "error": f"{type(exc).__name__}: {exc}"})
                if verbose and arm in models:
                    print(f"      fold {fold.index} {protocol} {arm}: "
                          f"fit {time.perf_counter() - t_fit:.1f}s")

            well_rows += evaluate_models(
                models, valid_tasks, protocol, fold.index, verbose=False, failures=failures,
                cache_context={"dataset_version": "rogii-mounted-v1", "fold": fold.index,
                               "protocol": protocol},
            )
            for d in fold_gate_log:
                d = dict(d)
                d["arm"] = ARM_ALIGN_V2
                gate_log_rows.append(d)
            for arm_name, model in models.items():
                if arm_name in (ARM_ALIGN_V2, ARM_META_V2):
                    info = {
                        "arm": arm_name,
                        "protocol": protocol,
                        "fold": fold.index,
                        "killed": getattr(model, "killed", False),
                        "kill_reason": getattr(model, "kill_reason", ""),
                    }
                    if arm_name == ARM_ALIGN_V2:
                        info.update(
                            {
                                "shrink": float(model.thresholds.get("shrink", 1.0)),
                                "warmup": int(model.thresholds.get("warmup", 0)),
                                "sep_cap": float(model.thresholds.get("sep_cap", float("inf"))),
                                "conf_thr": float(model.thresholds.get("conf_thr", 0.0)),
                                "margin": float(model.thresholds.get("margin", 0.0)),
                                "pooled_oof_delta": float(getattr(model.info, "pooled_oof_delta", float("nan"))),
                                "oof_activation_rate": float(getattr(model.info, "oof_activation_rate", float("nan"))),
                                "n_oof_wells": int(getattr(model.info, "n_oof_wells", 0)),
                                "n_examples": int(getattr(model.info, "n_examples", 0)),
                                "fit_seconds": float(getattr(model.info, "fit_seconds", 0.0)),
                            }
                        )
                    elif arm_name == ARM_META_V2:
                        info.update(
                            {
                                "killed": model.stack.killed,
                                "kill_reason": model.stack.kill_reason,
                                "meta_alpha": float(model.stack.meta_alpha),
                                "n_oof_wells": int(model.stack.info.get("n_oof_wells", 0)),
                                "n_oof_rows": int(model.stack.info.get("n_oof_rows", 0)),
                                "pooled_sub_oof_delta": float(model.stack.info.get("pooled_sub_oof_delta", float("nan"))),
                                "fit_seconds": float(model.stack.info.get("fit_seconds", 0.0)),
                            }
                        )
                    model_infos.append(info)
            if verbose:
                print(f"      fold {fold.index}: {len(train_tasks)} train / {len(valid_tasks)} valid, "
                      f"{len(models)}/{len(arms)} arms in {time.perf_counter() - tf:.1f}s")
        protocol_seconds[protocol] = time.perf_counter() - t0
        if verbose:
            print(f"      {protocol} done in {protocol_seconds[protocol]:.1f}s")

    if verbose:
        print("[6/6] writing reports")
    well_df = pd.DataFrame([vars(r) for r in well_rows])
    gate_log_df = pd.DataFrame(gate_log_rows)
    info_df = pd.DataFrame(model_infos)
    banner = banner_block(environment)

    outputs: dict[str, Path] = {}

    def _write(name: str, df: pd.DataFrame):
        path = reports_dir / f"{prefix}alignment_v2_{name}.csv"
        df.to_csv(path, index=False)
        outputs[name] = path

    summary = summarize(well_df) if not well_df.empty else pd.DataFrame()
    if not summary.empty:
        order = {s: i for i, s in enumerate(ARM_ORDER)}
        summary["arm_label"] = summary["model"].map(ARM_LABELS)
        summary = summary.sort_values(
            ["protocol", "model"], key=lambda c: c.map(order) if c.name == "model" else c
        )
    _write("summary", summary)
    _write("well_level", well_df)
    _write("stratified", stratified_report(well_df) if not well_df.empty else pd.DataFrame())
    _write("gate_decisions", gate_log_df)
    _write("model_infos", info_df)
    _write("failures", pd.DataFrame(failures))
    rates, reasons = decision_summary_tables(gate_log_df)
    _write("gate_activation_rates", rates)
    _write("gate_fallback_reasons", reasons)

    # ------------------------------------------------ promotion decision --
    decision = {
        "schema": "alignment-v2-promotion/v1",
        "is_real_run": bool(environment["is_real_run"]),
        "data_source": data_source,
        "alignment_v2_version": ALIGN_V2_MODEL_VERSION,
        "reference": {
            "promoted_arm": "oof_meta_stack",
            "unseen_well_rmse": PROMOTED_REFERENCE_UNSEEN_RMSE,
        },
        "rules_doc": DEFAULT_RULES_DOC,
        "arms": {},
        "promoted": False,
        "promoted_arm": None,
        "promotion_note": "",
        "environment_key": {
            "n_train_wells_discovered": environment["n_train_wells_discovered"],
            "n_eligible_wells": environment["n_eligible_wells"],
            "n_splits": environment["n_splits"],
            "seed": environment["seed"],
            "git_commit": environment["git_commit"],
        },
        "final_submission_created": False,
    }
    arm_evals = {}
    for arm in arms:
        if arm == ARM_RIDGE:
            continue
        # The promotion rule uses the v2 decision's own metrics
        # (well_df, fold_stab, boot_ci, decision_log, stack_infos,
        # is_real). We build these from the well_df; the
        # paired_deltas / fold_stab / bootstrap_ci are computed below.
        ev = evaluate_arm(
            arm,
            summary=summary,
            well_df=well_df,
            gate_log_df=gate_log_df,
            info_df=info_df,
            is_real=bool(environment["is_real_run"]),
        )
        arm_evals[arm] = ev
        decision["arms"][arm] = ev
    passing = [
        arm for arm, ev in arm_evals.items()
        if ev.get("all_rules_passed")
        and "r1_unseen_beats_promoted_reference" in ev.get("rules", {})
    ]
    if passing:
        promoted = min(
            passing,
            key=lambda a: decision["arms"][a]["metrics"]["unseen_well"]["candidate"]["global_rmse"],
        )
        decision["promoted"] = True
        decision["promoted_arm"] = promoted
        decision["promotion_note"] = (
            f"{promoted} passed every pre-registered rule on the real mount; "
            "submission may be built by scripts/build_alignment_v2_submission.py."
        )
    else:
        if not environment["is_real_run"]:
            decision["promotion_note"] = (
                "NOT the real mount: no arm may be promoted from a synthetic/partial "
                "run, regardless of measured numbers. The promoted oof_meta_stack stands."
            )
        else:
            decision["promotion_note"] = (
                "No Alignment v2 candidate passed every pre-registered rule on the "
                "real mount. The promoted oof_meta_stack remains the production model."
            )
    environment["runtime_seconds"] = time.perf_counter() - t_start
    environment["protocol_seconds"] = protocol_seconds
    environment["peak_rss_mb"] = peak_rss_mb()
    env_path = reports_dir / f"{prefix}alignment_v2_run_environment.json"
    env_path.write_text(json.dumps(environment, indent=2, default=str))
    outputs["run_environment"] = env_path

    decision_path = reports_dir / f"{prefix}alignment_v2_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, default=str))
    outputs["decision_json"] = decision_path

    # ------------------------------------------------ decision narrative --
    lines = [banner, "", "# Alignment v2 experiment — results", ""]
    lines.append(f"- arms: {', '.join(arms)}")
    lines.append(f"- protocols: {', '.join(args.protocols)}")
    lines.append(f"- folds: {args.n_splits} outer GroupKFold, inner OOF {args.inner_splits}, "
                 f"tune sub-folds {args.tune_splits}, seed {args.seed}")
    lines.append(f"- runtime: {environment['runtime_seconds']:.1f}s, "
                 f"peak RSS {environment['peak_rss_mb']:.0f} MB")
    lines.append(f"- reference (promoted oof_meta_stack): unseen_well RMSE = "
                 f"{PROMOTED_REFERENCE_UNSEEN_RMSE:.6f}")
    lines.append("")
    if not summary.empty:
        cols = [c for c in ("protocol", "model", "global_rmse", "mean_well_rmse",
                            "median_well_rmse", "p90_well_rmse", "worst10_well_rmse",
                            "worst_well_rmse", "n_wells") if c in summary.columns]
        lines.append("## Global RMSE per arm and protocol")
        lines.append("")
        lines.append("```")
        lines.append(summary[cols].to_string(index=False))
        lines.append("```")
        lines.append("")
    if not rates.empty:
        lines.append("## Gate activation rates (alignment_v2)")
        lines.append("")
        lines.append("```")
        lines.append(rates.to_string(index=False))
        lines.append("```")
        lines.append("")
    lines.append("## Promotion rule evaluation")
    lines.append("")
    for arm, ev in arm_evals.items():
        lines.append(f"### {arm}")
        lines.append("")
        for rule, rr in ev.get("rules", {}).items():
            mark = "PASS" if rr.get("passed") else "FAIL"
            detail = {k: v for k, v in rr.items() if k not in {"passed", "strata", "note", "examples"}}
            lines.append(f"- r `{rule}`: **{mark}** {json.dumps(detail, default=str)}")
        lines.append("")
    lines.append(f"**Decision: {'PROMOTED ' + str(decision['promoted_arm']) if decision['promoted'] else 'NO PROMOTION — promoted oof_meta_stack stands.'}**")
    lines.append("")
    lines.append(decision["promotion_note"])
    md_path = reports_dir / f"{prefix}alignment_v2_decision.md"
    md_path.write_text("\n".join(lines))
    outputs["decision_md"] = md_path

    if verbose:
        print(f"\nWrote {len(outputs)} files to {reports_dir} "
              f"in {environment['runtime_seconds']:.1f}s")
        if not summary.empty:
            cols = [c for c in ("protocol", "model", "global_rmse", "mean_well_rmse",
                                "median_well_rmse", "p90_well_rmse", "worst10_well_rmse",
                                "worst_well_rmse", "n_wells") if c in summary.columns]
            print(summary[cols].to_string(index=False))
        print(f"\nPromoted: {decision['promoted']} ({decision['promoted_arm']}) — "
              f"{decision['promotion_note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
