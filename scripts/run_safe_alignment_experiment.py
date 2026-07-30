#!/usr/bin/env python3
"""Run the staged safe-alignment experiment (stages A–F) and write reports.

    python scripts/run_safe_alignment_experiment.py --max-wells 60          # smoke
    python scripts/run_safe_alignment_experiment.py --expect-wells 770 \\
        --n-splits 3                                                        # discovery
    python scripts/run_safe_alignment_experiment.py --expect-wells 770 \\
        --n-splits 5 --stages safe_f_verified                               # confirmation

Stages (each a superset of the previous; every stage falls back to the exact
Ridge Default prediction on any guard failure):

    A ridge_default        Ridge Default (identical instance = exact fallback)
    B safe_b_anchor_blend  Ridge + bounded PF/Beam anchor blend
    C safe_c_affine_cal    B + affine GR heel calibration trust
    D safe_d_branch_guard  C + multi-branch/bimodal hedging guard
    E safe_e_projection    D + robust IRLS stratigraphic projection
    F safe_f_verified      E + multi-cut prefix verification + tail guard

Stage G (OOF residual GBDT) is reported as gated-off unless LightGBM is
available AND an earlier stage shows real evidence; this runner never trains
an ungated booster.

This experiment NEVER writes a submission. The three visible public test
wells are hard-blocked from every fold by ``src.validation``. Report naming
is evidence-based: ``real_safe_alignment_*`` files appear only when the
discovered counts match the audited mount (773 train / 770 eligible);
anything else is stamped SYNTHETIC.
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

from src.data import discover_wells, load_well
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
from src.pf_beam_robustness import pair_default_vs_candidate
from src.real_ablation_reporting import (
    AUDITED_DISCOVERED_WELLS,
    AUDITED_ELIGIBLE_WELLS,
    banner_block,
    file_prefix,
    is_real_run,
)
from src.safe_alignment import (
    SAFE_ALIGNMENT_VERSION,
    STAGE_A,
    STAGE_G_STATUS,
    STAGE_LABELS,
    STAGE_ORDER,
    SafeAlignmentConfig,
    build_stage_models,
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

try:
    import lightgbm  # noqa: F401

    HAVE_LIGHTGBM = True
except Exception:
    HAVE_LIGHTGBM = False


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


def run_protocol(
    *,
    protocol: str,
    mode: str,
    folds,
    task_builder,
    stages,
    memo: dict,
    config: SafeAlignmentConfig,
    device: str,
    tune: bool,
    verbose: bool,
):
    well_results, failures, fold_records, decisions = [], [], [], []
    for fold in folds:
        t0 = time.perf_counter()
        assert_no_blocked_wells(fold.train_ids, context=f"safe-alignment {protocol} fold {fold.index} train")
        assert_no_blocked_wells(fold.valid_ids, context=f"safe-alignment {protocol} fold {fold.index} valid")
        train_tasks, sk_train = task_builder(fold.train_ids, mode)
        valid_tasks, sk_valid = task_builder(fold.valid_ids, mode)
        for wid, reason in sk_train + sk_valid:
            failures.append({"stage": "task", "model": "", "well_id": wid, "error": reason})
        if not train_tasks or not valid_tasks:
            continue
        overlap = {t.well_id for t in train_tasks} & {t.well_id for t in valid_tasks}
        if overlap:
            raise CrossFitLeakage(
                f"safe-alignment {protocol} fold {fold.index}: wells in both train "
                f"and valid, e.g. {sorted(overlap)[:5]}"
            )
        fold_decisions: list = []
        models = build_stage_models(
            stages,
            memo=memo,
            protocol=protocol,
            fold=fold.index,
            config=config,
            decision_log=fold_decisions,
            device=device,
            tune=tune,
        )
        for name in stages:  # fit in stage order (anchor first, shared)
            try:
                models[name].fit(train_tasks)
            except Exception as exc:
                failures.append({"stage": "fit", "model": name, "well_id": "", "error": f"{type(exc).__name__}: {exc}"})
                models.pop(name, None)
        well_results += evaluate_models(
            models, valid_tasks, protocol, fold.index, verbose=False, failures=failures
        )
        decisions += [asdict(d) for d in fold_decisions]
        dt = time.perf_counter() - t0
        fold_records.append(
            {
                "protocol": protocol,
                "fold": fold.index,
                "n_train_wells": len(train_tasks),
                "n_valid_wells": len(valid_tasks),
                "n_stages_fitted": len(models),
                "seconds": dt,
            }
        )
        if verbose:
            print(
                f"      fold {fold.index}: {len(train_tasks)} train / {len(valid_tasks)} valid, "
                f"{len(models)}/{len(stages)} stages in {dt:.1f}s"
            )
    return well_results, failures, fold_records, decisions


def decision_tables(decision_df: pd.DataFrame):
    """Activation rates and fallback-reason counts per stage & protocol."""
    if decision_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    rates = []
    for (protocol, stage), g in decision_df.groupby(["protocol", "stage"], sort=False):
        applied = g[g["outcome"] == "applied"]
        rates.append(
            {
                "protocol": protocol,
                "stage": stage,
                "n_wells": int(len(g)),
                "n_applied": int(len(applied)),
                "activation_rate": float(len(applied) / len(g)) if len(g) else np.nan,
                "correction_mean_abs_ft": float(applied["correction_mean_abs"].mean()) if len(applied) else 0.0,
                "correction_max_abs_ft": float(applied["correction_max_abs"].max()) if len(applied) else 0.0,
                "confidence_mean": float(applied["confidence"].mean()) if len(applied) else np.nan,
                "confidence_p10": float(applied["confidence"].quantile(0.10)) if len(applied) else np.nan,
                "disagreement_mean_ft": float(applied["disagreement"].mean()) if len(applied) else np.nan,
                "ambiguity_guard_rate": float(g["ambiguity_guard"].mean()),
                "projection_applied_rate": float(g["projection_applied"].mean()),
                "mean_decision_seconds": float(g["seconds"].mean()),
            }
        )
    reasons = (
        decision_df[decision_df["outcome"] != "applied"]
        .groupby(["protocol", "stage", "reason"], sort=False)
        .size()
        .reset_index(name="n_wells")
        .sort_values(["protocol", "stage", "n_wells"], ascending=[True, True, False])
    )
    return pd.DataFrame(rates), reasons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=3,
                    help="3 for rapid discovery, 5 for final confirmation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-wells", type=int, default=None)
    ap.add_argument("--stages", nargs="*", default=list(STAGE_ORDER),
                    choices=list(STAGE_ORDER))
    ap.add_argument("--device", choices=("auto", "cpu", "gpu"), default="cpu")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--no-tune", action="store_true",
                    help="skip the stage-F fold-train threshold tuner (faster smoke runs)")
    ap.add_argument("--protocols", nargs="*", default=[PROTOCOL_A, PROTOCOL_B],
                    choices=[PROTOCOL_A, PROTOCOL_B])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    verbose = not args.quiet

    t_start = time.perf_counter()
    reports_dir = ensure_reports_dir() if args.reports_dir is None else Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    stages = tuple(s for s in STAGE_ORDER if s in set(args.stages))
    if STAGE_A not in stages:
        stages = (STAGE_A,) + stages  # the paired anchor is mandatory

    assert_manifest_valid()
    assert_inference_provenance()
    cleared = safe_inference_features()
    if verbose:
        print(f"[1/5] manifest valid: {len(cleared)} cleared features")

    loader = WellLoader("train")
    all_train_ids = loader.ids()
    universe = filter_blocked(all_train_ids)
    if args.max_wells is not None:
        universe = universe[: args.max_wells]
    if args.expect_wells is not None and len(universe) != args.expect_wells:
        raise RuntimeError(
            f"expected {args.expect_wells} eligible train wells, found {len(universe)} "
            f"(discovered {len(all_train_ids)}). Refusing to run on an unexpected universe."
        )
    if len(universe) < args.n_splits * 2:
        raise RuntimeError(f"need at least {args.n_splits * 2} eligible wells, got {len(universe)}")
    if verbose:
        print(f"[2/5] wells: {len(all_train_ids)} discovered, {len(universe)} eligible "
              f"(blocked: {sorted(set(all_train_ids) & set(BLOCKED_WELL_IDS)) or 'none present'})")

    data_source = (
        "real_kaggle"
        if (
            len(all_train_ids) == AUDITED_DISCOVERED_WELLS
            and len(universe) == AUDITED_ELIGIBLE_WELLS
        )
        else ("synthetic_or_partial" if len(universe) < 100 else "synthetic_field")
    )
    environment = {
        "experiment": "safe alignment staged experiment A-F",
        "safe_alignment_version": SAFE_ALIGNMENT_VERSION,
        "data_source": data_source,
        "n_train_wells_discovered": len(all_train_ids),
        "n_eligible_wells": len(universe),
        "n_wells_evaluated": len(universe),
        "blocked_public_test_wells": sorted(BLOCKED_WELL_IDS),
        "max_wells": args.max_wells,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "stages": list(stages),
        "protocols": list(args.protocols),
        "device": args.device,
        "tuning_enabled": not args.no_tune,
        "stage_g_status": STAGE_G_STATUS,
        "lightgbm_available": HAVE_LIGHTGBM,
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
    config = SafeAlignmentConfig(seed=args.seed)
    memo: dict = {}
    all_results, all_failures, all_folds, all_decisions = [], [], [], []
    protocol_seconds = {}
    proto_modes = {PROTOCOL_A: "masked", PROTOCOL_B: "real"}
    for i, protocol in enumerate(args.protocols):
        if verbose:
            print(f"[{3 + i}/5] {protocol} (cross-fitted, stages {', '.join(stages)})")
        t0 = time.perf_counter()
        wr, fl, fr, dc = run_protocol(
            protocol=protocol,
            mode=proto_modes[protocol],
            folds=folds,
            task_builder=task_builder,
            stages=stages,
            memo=memo,
            config=config,
            device=args.device,
            tune=not args.no_tune,
            verbose=verbose,
        )
        protocol_seconds[protocol] = time.perf_counter() - t0
        all_results += wr
        all_failures += fl
        all_folds += fr
        all_decisions += dc

    if verbose:
        print("[5/5] writing reports")
    well_df = pd.DataFrame([vars(r) for r in all_results])
    decision_df = pd.DataFrame(all_decisions)
    banner = banner_block(environment)

    outputs: dict[str, Path] = {}

    def _write(name: str, df: pd.DataFrame):
        path = reports_dir / f"{prefix}safe_alignment_{name}.csv"
        df.to_csv(path, index=False)
        outputs[name] = path

    summary = summarize(well_df) if not well_df.empty else pd.DataFrame()
    if not summary.empty:
        summary = summary[summary["model"].isin(stages)].copy()
        summary["stage_label"] = summary["model"].map(STAGE_LABELS)
        order = {s: i for i, s in enumerate(STAGE_ORDER)}
        summary = summary.sort_values(
            ["protocol", "model"], key=lambda c: c.map(order) if c.name == "model" else c
        )
    _write("summary", summary)
    _write("well_level", well_df)

    paired_frames = []
    for stage in stages:
        if stage == STAGE_A:
            continue
        p = pair_default_vs_candidate(well_df, default=STAGE_A, candidate=stage)
        if p is not None and not p.empty:
            p = p.copy()
            p["candidate_arm"] = stage
            paired_frames.append(p)
    paired = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    _write("paired_well_deltas", paired)
    _write("improved_degraded", improved_degraded_counts(paired))

    # Fold stability / bootstrap reuse the geoanchor helpers via model names.
    fold_frames = []
    from src.pf_beam_robustness import fold_deltas

    for stage in stages:
        if stage == STAGE_A:
            continue
        fd = fold_deltas(well_df, default=STAGE_A, candidate=stage)
        if fd is not None and not fd.empty:
            fd = fd.copy()
            fd["candidate_arm"] = stage
            fold_frames.append(fd)
    _write("fold_stability", pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame())
    ci_df = (
        bootstrap_ci_table(paired, n_boot=args.n_bootstrap, seed=args.seed)
        if not paired.empty
        else pd.DataFrame()
    )
    if not ci_df.empty and "note" in ci_df.columns:
        ci_df["note"] = ci_df["note"].str.replace("ridge_particle_beam", "candidate_stage", regex=False)
    _write("bootstrap_ci", ci_df)
    _write("stratified", stratified_report(well_df) if not well_df.empty else pd.DataFrame())

    rates, reasons = decision_tables(decision_df)
    _write("activation_rates", rates)
    _write("fallback_reasons", reasons)
    _write("decisions", decision_df)
    _write("failures", pd.DataFrame(all_failures))
    _write("fold_records", pd.DataFrame(all_folds))

    environment["runtime_seconds"] = time.perf_counter() - t_start
    environment["protocol_seconds"] = protocol_seconds
    environment["peak_rss_mb"] = peak_rss_mb()
    env_path = reports_dir / f"{prefix}safe_alignment_run_environment.json"
    env_path.write_text(json.dumps(environment, indent=2, default=str))
    outputs["run_environment"] = env_path

    # ---- decision narrative -------------------------------------------
    lines = [banner, "", "# Safe alignment staged experiment — results", ""]
    lines.append(f"- stages: {', '.join(stages)}")
    lines.append(f"- protocols: {', '.join(args.protocols)}")
    lines.append(f"- folds: {args.n_splits}, seed {args.seed}")
    lines.append(f"- stage G (OOF residual GBDT): {STAGE_G_STATUS} "
                 f"(lightgbm available: {HAVE_LIGHTGBM})")
    lines.append(f"- runtime: {environment['runtime_seconds']:.1f}s, "
                 f"peak RSS {environment['peak_rss_mb']:.0f} MB")
    lines.append("")
    if not summary.empty:
        lines.append("## Global RMSE per stage and protocol")
        lines.append("")
        cols = [c for c in ("protocol", "model", "global_rmse", "mean_well_rmse",
                            "median_well_rmse", "p90_well_rmse", "worst10_well_rmse",
                            "worst_well_rmse", "n_wells") if c in summary.columns]
        sub = summary[cols]
        lines.append("```")
        lines.append(sub.to_string(index=False))
        lines.append("```")
        lines.append("")
    if not rates.empty:
        lines.append("## Activation rates")
        lines.append("")
        lines.append("```")
        lines.append(rates.to_string(index=False))
        lines.append("```")
        lines.append("")
    lines.append("## Promotion rule (pre-registered)")
    lines.append("")
    lines.append("A stage may be considered for a real submission only if real "
                 "unseen_well RMSE improves over the verified 14.4229 baseline, "
                 "same_well_masked and worst-10 do not materially degrade, the "
                 "improvement holds in most folds and the bootstrap CI is not "
                 "strongly against it. Otherwise the submission remains Ridge "
                 "Default (public LB 14.813).")
    md_path = reports_dir / f"{prefix}safe_alignment_report.md"
    md_path.write_text("\n".join(lines))
    outputs["report"] = md_path

    if verbose:
        print(f"\nWrote {len(outputs)} files to {reports_dir} "
              f"in {environment['runtime_seconds']:.1f}s")
        if not summary.empty:
            cols = [c for c in ("protocol", "model", "global_rmse", "mean_well_rmse",
                                "median_well_rmse", "p90_well_rmse", "worst10_well_rmse",
                                "worst_well_rmse", "n_wells") if c in summary.columns]
            print(summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
