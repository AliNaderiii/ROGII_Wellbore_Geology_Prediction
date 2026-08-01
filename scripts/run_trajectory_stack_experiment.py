#!/usr/bin/env python3
"""Run the trajectory stack experiment (arms A/F/G/H + gated candidate).

    ROGII_COMPETITION_ROOT=/path/to/field \
        python scripts/run_trajectory_stack_experiment.py --max-wells 100   # smoke
    python scripts/run_trajectory_stack_experiment.py --expect-wells 770    # real

Arms (every arm shares one fitted Ridge anchor instance, so a fallback is
bit-identical to the scored ``ridge_default`` arm):

    ridge_default       A/L — Ridge Default anchor + exact fallback
    lgbm_residual       F — LightGBM anchored-residual (evidence arm)
    catboost_residual   G — CatBoost anchored-residual (evidence arm)
    oof_meta_stack      H — kill-switched OOF Ridge meta-stack
    gated_trajectory    the promotion candidate: Ridge + guarded correction

Validation: ``same_well_masked`` and ``unseen_well``, GroupKFold by well
(5 folds by default), strict outer cross-fitting, inner OOF for gate and
blend selection, fold-specific imputation and scaling, deterministic seeds.
The three public duplicate wells are hard-blocked from every fit, gate
training, threshold selection and report.

THIS EXPERIMENT NEVER WRITES A SUBMISSION. Promotion decisions are written
to ``{prefix}trajectory_stack_decision.json``; a submission may only be
produced afterwards by ``scripts/build_gated_submission.py``.

Report naming is evidence-based: ``real_trajectory_stack_*`` files appear
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
from dataclasses import asdict
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:  # loose-file execution
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap

bootstrap()

import numpy as np
import pandas as pd

from src.baselines import RidgeBaseline
from src.device import DEVICE_CHOICES, resolve_device
from src.cache import FeatureCache
from src.data import discover_wells, load_well
from src.geoanchor import MemoizedPathGenerator
from src.geoanchor_reporting import (
    bootstrap_ci_table,
    fold_stability_table,
    improved_degraded_counts,
)
from src.manifest import (
    assert_inference_provenance,
    assert_manifest_valid,
    safe_inference_features,
)
from src.paths import ensure_reports_dir
from src.pf_beam_robustness import fold_deltas, pair_default_vs_candidate
from src.real_ablation_reporting import (
    AUDITED_DISCOVERED_WELLS,
    AUDITED_ELIGIBLE_WELLS,
    banner_block,
    file_prefix,
    is_real_run,
)
from src.tasks import TaskConstructionError, make_task
from src.trajectory_stack import (
    ARM_GATED,
    ARM_LABELS,
    ARM_ORDER,
    ARM_RIDGE,
    ARM_STACK,
    RIDGE_REFERENCE_UNSEEN_RMSE,
    STACK_CANDIDATES,
    TRAJECTORY_STACK_VERSION,
    HAVE_CATBOOST,
    HAVE_LIGHTGBM,
    StackConfig,
    TrajectoryGateConfig,
    build_stack_models,
)
from src.trajectory_stack_decision import DEFAULT_RULES_DOC, evaluate_arm
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


def decision_summary_tables(decision_df: pd.DataFrame):
    """Activation rates and fallback-reason counts for the gated arm."""
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
    ap.add_argument("--inner-splits", type=int, default=5, help="inner OOF depth (gate + stack)")
    ap.add_argument("--tune-splits", type=int, default=3, help="threshold/blend selection sub-folds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-wells", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=list(ARM_ORDER), choices=list(ARM_ORDER))
    ap.add_argument("--protocols", nargs="*", default=[PROTOCOL_A, PROTOCOL_B],
                    choices=[PROTOCOL_A, PROTOCOL_B])
    ap.add_argument("--skip-lightgbm", action="store_true")
    ap.add_argument("--skip-catboost", action="store_true")
    ap.add_argument("--boost-max-iter", type=int, default=400)
    ap.add_argument("--boost-estop-rounds", type=int, default=50)
    ap.add_argument("--boost-threads", type=int, default=4)
    ap.add_argument("--device", choices=list(DEVICE_CHOICES), default="auto",
                    help="where the LightGBM/CatBoost residual models train: "
                         "'cpu' forces CPU, 'gpu' asks for GPU, 'auto' (default) "
                         "uses GPU only when the installed library really supports "
                         "it and otherwise falls back to CPU with a logged reason. "
                         "Ridge, PF/Beam and the gate are unaffected.")
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

    # Device resolution never raises for a device reason: a missing or broken
    # GPU stack downgrades to CPU with the exact reason recorded.
    device = resolve_device(args.device, log=(print if verbose else None))

    t_start = time.perf_counter()
    reports_dir = ensure_reports_dir() if args.reports_dir is None else Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Feature whitelist: validated BEFORE any model is fitted.
    assert_manifest_valid()
    provenance = assert_inference_provenance()
    cleared = safe_inference_features()
    if verbose:
        print(f"[1/6] manifest valid: {len(cleared)} cleared features; "
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
        arms = (ARM_RIDGE,) + arms  # the paired anchor is mandatory
    # --skip-* removes both the standalone evidence arm and its use inside
    # the stack/gate (documented as one coherent library switch).
    if args.skip_lightgbm and "lgbm_residual" in arms:
        arms = tuple(a for a in arms if a != "lgbm_residual")
        if verbose:
            print("      --skip-lightgbm: excluded the lgbm_residual arm and its stack/gate tracks")
    if args.skip_catboost and "catboost_residual" in arms:
        arms = tuple(a for a in arms if a != "catboost_residual")
        if verbose:
            print("      --skip-catboost: excluded the catboost_residual arm and its stack/gate tracks")
    environment = {
        "experiment": "trajectory stack experiment (boosters, OOF meta-stack, gated stack)",
        "trajectory_stack_version": TRAJECTORY_STACK_VERSION,
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
        "gate_candidates": list(STACK_CANDIDATES),
        "protocols": list(args.protocols),
        "boost_max_iter": args.boost_max_iter,
        "boost_estop_rounds": args.boost_estop_rounds,
        "boost_threads": args.boost_threads,
        **device.as_report(),
        "use_lightgbm": not args.skip_lightgbm and HAVE_LIGHTGBM,
        "use_catboost": not args.skip_catboost and HAVE_CATBOOST,
        "label": args.label,
        "reference_ridge_unseen_rmse": RIDGE_REFERENCE_UNSEEN_RMSE,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "final_submission_created": False,
    }
    environment["is_real_run"] = is_real_run(environment)
    prefix = file_prefix(environment)

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
        print(f"      GroupKFold: {args.n_splits} folds, validation sizes "
              f"{[len(f.valid_ids) for f in folds]}")

    path_cache = FeatureCache(args.path_cache) if args.path_cache else None
    stack_config = StackConfig(
        inner_splits=args.inner_splits,
        tune_splits=args.tune_splits,
        boost_max_iter=args.boost_max_iter,
        boost_estop_rounds=args.boost_estop_rounds,
        boost_threads=args.boost_threads,
        use_lightgbm=not args.skip_lightgbm,
        use_catboost=not args.skip_catboost,
        device=device,
        seed=args.seed,
    )
    gate_config = TrajectoryGateConfig(
        inner_splits=args.inner_splits,
        tune_splits=args.tune_splits,
        boost_max_iter=args.boost_max_iter,
        boost_estop_rounds=args.boost_estop_rounds,
        boost_threads=args.boost_threads,
        use_lightgbm=not args.skip_lightgbm,
        use_catboost=not args.skip_catboost,
        device=device,
        seed=args.seed,
    )
    boost_kw = dict(
        seed=args.seed,
        max_iter=args.boost_max_iter,
        estop_rounds=args.boost_estop_rounds,
        thread_count=args.boost_threads,
        device=device,
    )

    memo: dict = {}
    well_rows: list = []
    failures: list = []
    fold_records: list = []
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

            def _pf(fold_index=fold.index, proto=protocol):
                from src.particle_filter import ParticleFilterFeatureGenerator

                return MemoizedPathGenerator(
                    ParticleFilterFeatureGenerator(
                        cache=path_cache,
                        dataset_version="rogii-mounted-v1",
                        fold_id=fold_index,
                        protocol=proto,
                        device="cpu",
                    ),
                    memo,
                    "pf",
                )

            def _beam(fold_index=fold.index, proto=protocol):
                from src.beam_search import BeamSearchFeatureGenerator

                return MemoizedPathGenerator(
                    BeamSearchFeatureGenerator(
                        cache=path_cache,
                        dataset_version="rogii-mounted-v1",
                        fold_id=fold_index,
                        protocol=proto,
                        device="cpu",
                    ),
                    memo,
                    "beam",
                )

            anchor = RidgeBaseline(alignment_features=False)
            anchor.name = ARM_RIDGE
            fold_gate_log: list = []
            t_anchor = time.perf_counter()
            try:
                anchor.fit(train_tasks)
            except Exception as exc:
                failures.append({"stage": "fit", "model": ARM_RIDGE, "well_id": "",
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            anchor_seconds = time.perf_counter() - t_anchor

            models = build_stack_models(
                arms,
                anchor_model=anchor,
                pf_factory=_pf,
                beam_factory=_beam,
                stack_config=stack_config,
                gate_config=gate_config,
                protocol=protocol,
                fold=fold.index,
                decision_log=fold_gate_log,
                boost_kw=boost_kw,
            )
            fitted = {ARM_RIDGE: anchor}
            for name in arms:
                if name == ARM_RIDGE or name not in models:
                    continue
                t_fit = time.perf_counter()
                try:
                    models[name].fit(train_tasks)
                    fitted[name] = models[name]
                except Exception as exc:
                    failures.append({"stage": "fit", "model": name, "well_id": "",
                                     "error": f"{type(exc).__name__}: {exc}"})
                if verbose:
                    print(f"      fold {fold.index} {protocol} {name}: "
                          f"fit {time.perf_counter() - t_fit:.1f}s")

            well_rows += evaluate_models(
                fitted, valid_tasks, protocol, fold.index, verbose=False, failures=failures,
                cache_context={"dataset_version": "rogii-mounted-v1", "fold": fold.index,
                               "protocol": protocol},
            )
            for d in fold_gate_log:
                d = dict(d)
                d["arm"] = ARM_GATED
                gate_log_rows.append(d)
            for name, model in fitted.items():
                if name == ARM_STACK:
                    info = dict(model.stack.info)
                    info.update({"arm": name, "protocol": protocol, "fold": fold.index,
                                 "fit_seconds": info.get("fit_seconds", 0.0)})
                    info.update(device.as_report())
                    model_infos.append(info)
                elif name == ARM_GATED:
                    info = dict(vars(model.info))
                    info.update({
                        "arm": name,
                        "killed": model.killed,
                        "kill_reason": model.kill_reason,
                        "shrink": model.thresholds.shrink,
                        "warmup": model.thresholds.warmup,
                        "sep_cap": model.thresholds.sep_cap,
                        "conf_thr": model.thresholds.conf_thr,
                        "margin": model.thresholds.margin,
                        "oof_skills": dict(model._oof_skills),
                    })
                    info.update({"protocol": protocol, "fold": fold.index})
                    info.update(device.as_report())
                    model_infos.append(info)
                else:
                    learner = models.get(name)
                    if learner is not None and hasattr(learner, "info") and hasattr(learner.info, "library"):
                        info = asdict(learner.info)
                        info.update({"arm": name, "protocol": protocol, "fold": fold.index})
                        # BoostFitInfo already carries the five device keys
                        # for the device the learner actually trained on.
                        for k, v in device.as_report().items():
                            info.setdefault(k, v)
                        model_infos.append(info)
            fold_records.append(
                {
                    "protocol": protocol,
                    "fold": fold.index,
                    "n_train_wells": len(train_tasks),
                    "n_valid_wells": len(valid_tasks),
                    "n_arms_fitted": len(fitted),
                    "anchor_fit_seconds": anchor_seconds,
                    "seconds": time.perf_counter() - tf,
                }
            )
            if verbose:
                print(f"      fold {fold.index}: {len(train_tasks)} train / {len(valid_tasks)} valid, "
                      f"{len(fitted)}/{len(arms)} arms in {time.perf_counter() - tf:.1f}s")
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
        path = reports_dir / f"{prefix}trajectory_stack_{name}.csv"
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

    paired_frames = []
    for arm in arms:
        if arm == ARM_RIDGE:
            continue
        p = pair_default_vs_candidate(well_df, default=ARM_RIDGE, candidate=arm)
        if p is not None and not p.empty:
            p = p.copy()
            p["candidate_arm"] = arm
            paired_frames.append(p)
    paired = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    _write("paired_well_deltas", paired)
    _write("improved_degraded", improved_degraded_counts(paired))

    fold_frames = []
    for arm in arms:
        if arm == ARM_RIDGE:
            continue
        fd = fold_deltas(well_df, default=ARM_RIDGE, candidate=arm)
        if fd is not None and not fd.empty:
            fd = fd.copy()
            fd["candidate_arm"] = arm
            fold_frames.append(fd)
    fold_stab = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    _write("fold_stability", fold_stab)

    ci_df = (
        bootstrap_ci_table(paired, n_boot=args.n_bootstrap, seed=args.seed)
        if not paired.empty
        else pd.DataFrame()
    )
    if not ci_df.empty and "note" in ci_df.columns:
        ci_df["note"] = ci_df["note"].str.replace("ridge_particle_beam", "candidate_arm", regex=False)
    _write("bootstrap_ci", ci_df)
    _write("stratified", stratified_report(well_df) if not well_df.empty else pd.DataFrame())

    rates, reasons = decision_summary_tables(gate_log_df)
    _write("gate_activation_rates", rates)
    _write("gate_fallback_reasons", reasons)
    _write("gate_decisions", gate_log_df)
    _write("model_infos", info_df)
    _write("failures", pd.DataFrame(failures))
    _write("fold_records", pd.DataFrame(fold_records))

    # ------------------------------------------------ promotion decision --
    decision = {
        "schema": "trajectory-stack-promotion/v1",
        "is_real_run": bool(environment["is_real_run"]),
        "data_source": data_source,
        "trajectory_stack_version": TRAJECTORY_STACK_VERSION,
        "reference": {
            "model": "Ridge Default",
            "unseen_well_rmse": RIDGE_REFERENCE_UNSEEN_RMSE,
            "same_well_masked_rmse": 29.486086,
            "public_lb": 14.813,
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
        ev = evaluate_arm(
            arm,
            summary=summary,
            well_df=well_df,
            fold_stab=fold_stab,
            boot_ci=ci_df,
            decision_log=gate_log_df,
            stack_infos=model_infos,
            is_real=bool(environment["is_real_run"]),
        )
        arm_evals[arm] = ev
        decision["arms"][arm] = ev
    passing = [
        arm
        for arm, ev in arm_evals.items()
        if ev.get("all_rules_passed")
        and "r1_unseen_beats_reference" in ev.get("rules", {})
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
            "submission may be built by scripts/build_gated_submission.py."
        )
    else:
        if not environment["is_real_run"]:
            decision["promotion_note"] = (
                "NOT the real mount: no arm may be promoted from a synthetic/partial "
                "run, regardless of measured numbers. Ridge Default stands."
            )
        else:
            decision["promotion_note"] = (
                "No candidate passed every pre-registered rule on the real mount. "
                "Ridge Default remains the promoted model."
            )
    environment["runtime_seconds"] = time.perf_counter() - t_start
    environment["protocol_seconds"] = protocol_seconds
    environment["peak_rss_mb"] = peak_rss_mb()
    environment["path_cache"] = (
        path_cache.report() if path_cache is not None else None
    )
    env_path = reports_dir / f"{prefix}trajectory_stack_run_environment.json"
    env_path.write_text(json.dumps(environment, indent=2, default=str))
    outputs["run_environment"] = env_path

    decision_path = reports_dir / f"{prefix}trajectory_stack_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, default=str))
    outputs["decision_json"] = decision_path

    # ------------------------------------------------ decision narrative --
    lines = [banner, "", "# Trajectory stack experiment — results", ""]
    lines.append(f"- arms: {', '.join(arms)}")
    lines.append(f"- gate candidates: {', '.join(STACK_CANDIDATES)}")
    lines.append(f"- protocols: {', '.join(args.protocols)}")
    lines.append(f"- folds: {args.n_splits} outer GroupKFold, inner OOF {args.inner_splits}, "
                 f"tune sub-folds {args.tune_splits}, seed {args.seed}")
    lines.append(f"- runtime: {environment['runtime_seconds']:.1f}s, "
                 f"peak RSS {environment['peak_rss_mb']:.0f} MB")
    lines.append(f"- {device.summary_line()}")
    lines.append(f"- lightgbm: {'yes' if HAVE_LIGHTGBM else 'NO (arm unavailable)'}, "
                 f"catboost: {'yes' if HAVE_CATBOOST else 'NO (arm unavailable)'}")
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
        lines.append("## Gate activation rates (gated_trajectory)")
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
    lines.append(f"**Decision: {'PROMOTED ' + str(decision['promoted_arm']) if decision['promoted'] else 'NO PROMOTION — Ridge Default stands.'}**")
    lines.append("")
    lines.append(decision["promotion_note"])
    md_path = reports_dir / f"{prefix}trajectory_stack_decision.md"
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
