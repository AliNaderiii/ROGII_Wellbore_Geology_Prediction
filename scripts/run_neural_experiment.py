"""Run the leakage-safe PyTorch/hybrid experiment; never create a submission.

This is intentionally a separate runner from the established Ridge validation
runner.  It makes the first neural pass easy to audit: every model is trained
inside the existing strict GroupKFold driver, both protocols are reported
separately, and the output is diagnostic only.  A candidate submission is not
an allowed side effect of this script.

Example on the complete Kaggle mount::

    python scripts/run_neural_experiment.py \
      --expect-train 773 --expect-test 3 --device auto \
      --reports-dir /kaggle/working/neural_reports

A quick synthetic smoke run is useful for plumbing only and must be labelled
synthetic in the report directory; its metrics are not competition metrics.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
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

from src.baselines import RidgeBaseline
from src.data import discover_wells, load_well
from src.reporting import _md_table
from src.hybrid import ConservativeRidgeNeuralGate, RidgeNeuralBlend
from src.manifest import (
    assert_inference_provenance,
    assert_manifest_matches_data,
    assert_manifest_valid,
    safe_inference_features,
    verify_manifest_against_data,
    write_manifest,
)
from src.neural import HAVE_TORCH, NeuralConfig, NeuralResidualModel
from src.paths import REPORTS_DIR, SAMPLE_SUBMISSION, TEST_DIR, TRAIN_DIR, ensure_reports_dir, require_competition_data
from src.tasks import TaskConstructionError, make_task
from src.validation import (
    BLOCKED_WELL_IDS,
    PROTOCOL_A,
    PROTOCOL_B,
    assert_no_blocked_wells,
    filter_blocked,
    make_group_folds,
    run_cross_fitted_protocol,
    stratified_report,
    summarize,
)


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def _bootstrap_delta(well: pd.DataFrame, candidate: str, anchor: str, protocol: str, *, seed: int, n_boot: int) -> dict:
    """Well-cluster bootstrap CI for candidate minus Ridge global RMSE."""
    c = well[(well.model == candidate) & (well.protocol == protocol)].set_index("well_id")
    a = well[(well.model == anchor) & (well.protocol == protocol)].set_index("well_id")
    common = sorted(set(c.index) & set(a.index))
    if not common:
        return {"candidate": candidate, "anchor": anchor, "protocol": protocol, "n_wells": 0, "observed_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_candidate_better": np.nan}
    c, a = c.loc[common], a.loc[common]
    observed = float(np.sqrt(c.sse.sum() / c.n_points.sum()) - np.sqrt(a.sse.sum() / a.n_points.sum()))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        pick = rng.integers(0, len(common), len(common))
        cc, aa = c.iloc[pick], a.iloc[pick]
        draws[i] = np.sqrt(cc.sse.sum() / max(cc.n_points.sum(), 1)) - np.sqrt(aa.sse.sum() / max(aa.n_points.sum(), 1))
    return {
        "candidate": candidate, "anchor": anchor, "protocol": protocol,
        "n_wells": len(common), "observed_delta": observed,
        "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
        "p_candidate_better": float(np.mean(draws < 0.0)),
    }


def _paired_well_deltas(well: pd.DataFrame, anchor: str = "ridge_default") -> pd.DataFrame:
    rows = []
    for protocol in sorted(well.protocol.unique()):
        base = well[(well.model == anchor) & (well.protocol == protocol)].set_index("well_id")
        for model in sorted(set(well.model) - {anchor}):
            cand = well[(well.model == model) & (well.protocol == protocol)].set_index("well_id")
            common = sorted(set(base.index) & set(cand.index))
            for wid in common:
                rows.append({
                    "model": model, "protocol": protocol, "well_id": wid,
                    "anchor_rmse": float(base.loc[wid, "rmse"]),
                    "candidate_rmse": float(cand.loc[wid, "rmse"]),
                    "delta_rmse_candidate_minus_anchor": float(cand.loc[wid, "rmse"] - base.loc[wid, "rmse"]),
                    "anchor_sse": float(base.loc[wid, "sse"]), "candidate_sse": float(cand.loc[wid, "sse"]),
                    "n_points": int(cand.loc[wid, "n_points"]),
                })
    return pd.DataFrame(rows)


def _decision_report(path: Path, results: pd.DataFrame, bootstrap: pd.DataFrame, env: dict, *, torch_note: str) -> None:
    lines = [
        "# Neural / hybrid experiment decision report",
        "",
        "**Diagnostic only. No Kaggle submission was created by this runner.**",
        "",
        "The authoritative ranking protocol is `unseen_well`; `same_well_masked` is a separate continuation stress test. Public leaderboard values were not read or used.",
        "",
        f"PyTorch status: {torch_note}",
        f"Data source: `{env['competition_root']}`; train wells discovered={env['n_train_wells_discovered']}, test wells discovered={env['n_test_wells_discovered']}; public duplicates in validation={env['blocked_wells_in_validation']}.",
        "",
        "## Model comparison",
        "",
    ]
    if len(results):
        show = [c for c in ("model", "protocol", "n_wells", "n_points", "global_rmse", "mean_well_rmse", "median_well_rmse", "p90_well_rmse", "worst10_well_rmse", "worst_well_rmse", "mean_bias") if c in results]
        lines.append(_md_table(results[show]))
    else:
        lines.append("_No model produced a score._")
    lines += ["", "## Paired well bootstrap (candidate minus Ridge; negative is better)", ""]
    if len(bootstrap):
        lines.append(_md_table(bootstrap))
    else:
        lines.append("_Unavailable: no paired results._")
    lines += [
        "", "## Promotion decision", "",
        "The code does not auto-promote. A candidate is **REJECTED / diagnostic** unless the complete real-data report proves improvement over Ridge Default on unseen wells, no material same-well or worst-tail degradation, fold stability, strata stability, non-adverse bootstrap evidence, and deterministic provenance. If any criterion is unavailable, Ridge Default remains the exact production fallback.",
        "", "## Reproducibility", "", f"```json\n{json.dumps(env, indent=2)}\n```", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--models", default="ridge_default,neural_mlp,neural_gru,neural_tcn")
    ap.add_argument("--protocols", default=f"{PROTOCOL_A},{PROTOCOL_B}")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--max-wells", type=int, default=None)
    ap.add_argument("--expect-train", type=int, default=None)
    ap.add_argument("--expect-test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    ap.add_argument("--max-epochs", type=int, default=24)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-sequence-rows", type=int, default=1024)
    ap.add_argument("--no-pseudo", action="store_true", help="diagnostic ablation: use real suffix examples only")
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    require_competition_data(need_sample_submission=True)
    reports_dir = Path(args.reports_dir) if args.reports_dir else ensure_reports_dir() / "neural_experiment"
    reports_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    train_files = discover_wells("train")
    test_files = discover_wells("test")
    all_train_ids = sorted(train_files)
    test_ids = sorted(test_files)
    if args.expect_train is not None and len(all_train_ids) != args.expect_train:
        raise SystemExit(f"expected {args.expect_train} train wells, discovered {len(all_train_ids)}")
    if args.expect_test is not None and len(test_ids) != args.expect_test:
        raise SystemExit(f"expected {args.expect_test} test wells, discovered {len(test_ids)}")

    # Schema/provenance smoke check only. No test task is loaded for tuning or
    # model selection; the test split is reserved for the authorized final path.
    train_probe = load_well(train_files[all_train_ids[0]]) if all_train_ids else None
    test_probe = load_well(test_files[test_ids[0]]) if test_ids else None
    if train_probe is None or test_probe is None:
        raise SystemExit("could not load train/test schema probes")
    assert_manifest_valid()
    assert_inference_provenance()
    verification = verify_manifest_against_data(
        train_probe.hw.columns, test_probe.hw.columns,
        train_tw_columns=list(train_probe.tw.columns) if train_probe.tw is not None else None,
        test_tw_columns=list(test_probe.tw.columns) if test_probe.tw is not None else None,
    )
    verification.to_csv(reports_dir / "feature_manifest_verification.csv", index=False)
    assert_manifest_matches_data(
        train_probe.hw.columns, test_probe.hw.columns,
        train_tw_columns=list(train_probe.tw.columns) if train_probe.tw is not None else None,
        test_tw_columns=list(test_probe.tw.columns) if test_probe.tw is not None else None,
    )
    write_manifest(reports_dir / "feature_manifest.csv")

    universe = filter_blocked(all_train_ids)
    assert_no_blocked_wells(universe, context="neural validation universe")
    if args.max_wells is not None:
        universe = universe[: int(args.max_wells)]
    folds = make_group_folds(universe, n_splits=args.n_splits, seed=args.seed)

    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    allowed = {"ridge_default", "neural_mlp", "neural_gru", "neural_tcn", "ridge_neural_blend", "ridge_neural_gated"}
    unknown = sorted(set(model_names) - allowed)
    if unknown:
        raise SystemExit(f"unknown neural model(s): {unknown}; allowed={sorted(allowed)}")
    config = NeuralConfig(
        seed=args.seed, device=args.device, max_epochs=args.max_epochs,
        patience=args.patience, batch_size=args.batch_size,
        max_sequence_rows=args.max_sequence_rows,
    )

    def task_builder(ids, mode):
        tasks, skipped = [], []
        for wid in ids:
            try:
                well = load_well(train_files[wid])
                tasks.append(make_task(well, mode))
            except (TaskConstructionError, Exception) as exc:
                skipped.append((wid, f"{type(exc).__name__}: {exc}"))
        return tasks, skipped

    def factories_for(_fold=None, _protocol=None):
        out = {}
        if "ridge_default" in model_names:
            out["ridge_default"] = lambda: RidgeBaseline(alpha=10.0, alignment_features=False)
        for name, arch in (("neural_mlp", "mlp"), ("neural_gru", "gru"), ("neural_tcn", "tcn")):
            if name in model_names:
                cfg = NeuralConfig(**vars(config))
                cfg.architecture = arch
                cfg.seed = args.seed + (0 if arch == "mlp" else 101 if arch == "gru" else 202)
                if args.no_pseudo:
                    cfg.max_pseudo_examples_per_well = 0
                out[name] = lambda cfg=cfg, arch=arch: NeuralResidualModel(config=cfg, architecture=arch)
        if "ridge_neural_blend" in model_names:
            out["ridge_neural_blend"] = lambda: RidgeNeuralBlend(config, inner_splits=min(3, args.n_splits), seed=args.seed)
        if "ridge_neural_gated" in model_names:
            out["ridge_neural_gated"] = lambda: ConservativeRidgeNeuralGate(config, inner_splits=min(3, args.n_splits), seed=args.seed)
        return out

    protocols = []
    for p in args.protocols.split(","):
        p = {"masked": PROTOCOL_A, "groupkfold": PROTOCOL_B}.get(p.strip(), p.strip())
        if p not in (PROTOCOL_A, PROTOCOL_B):
            raise SystemExit(f"unknown protocol {p}")
        protocols.append(p)

    well_rows, fold_records, failures = [], [], []
    for protocol, mode in ((PROTOCOL_A, "masked"), (PROTOCOL_B, "real")):
        if protocol not in protocols:
            continue
        run = run_cross_fitted_protocol(
            protocol=protocol, mode=mode, factories=factories_for(),
            folds=folds, task_builder=task_builder, verbose=True,
            factory_builder=factories_for,
        )
        well_rows.extend(run.well_results)
        fold_records.extend(run.fold_records)
        failures.extend(run.failures)
    if not well_rows:
        raise SystemExit("no neural validation results were produced")

    well = pd.DataFrame([r.__dict__ for r in well_rows])
    assert_no_blocked_wells(well.well_id, context="neural result table")
    results = summarize(well)
    strat = stratified_report(well)
    deltas = _paired_well_deltas(well)
    boot = pd.DataFrame([
        _bootstrap_delta(well, model, "ridge_default", protocol, seed=args.seed + i, n_boot=args.bootstrap_reps)
        for i, model in enumerate(sorted(set(well.model) - {"ridge_default"}))
        for protocol in (PROTOCOL_A, PROTOCOL_B) if {model, "ridge_default"} <= set(well[well.protocol == protocol].model)
    ])
    folds_df = pd.DataFrame(fold_records)
    # Fold-level global RMSE makes stability visible instead of hiding it in a
    # single pooled score.
    fold_rows = []
    for (model, protocol, fold), g in well.groupby(["model", "protocol", "fold"], sort=False):
        fold_rows.append({"model": model, "protocol": protocol, "fold": int(fold), "n_wells": int(g.well_id.nunique()), "n_points": int(g.n_points.sum()), "global_rmse": float(np.sqrt(g.sse.sum() / max(g.n_points.sum(), 1))), "mean_well_rmse": float(g.rmse.mean()), "worst10_well_rmse": float(g.nlargest(min(10, len(g)), "rmse").rmse.mean())})
    fold_metrics = pd.DataFrame(fold_rows)

    well.to_csv(reports_dir / "neural_well_level_validation.csv", index=False)
    results.to_csv(reports_dir / "neural_validation_results.csv", index=False)
    strat.to_csv(reports_dir / "neural_stratified_validation.csv", index=False)
    deltas.to_csv(reports_dir / "neural_paired_well_deltas.csv", index=False)
    boot.to_csv(reports_dir / "neural_bootstrap_ci.csv", index=False)
    fold_metrics.to_csv(reports_dir / "neural_fold_metrics.csv", index=False)
    pd.DataFrame(failures, columns=["stage", "model", "well_id", "error"]).to_csv(reports_dir / "neural_validation_failures.csv", index=False)
    training_reports = []
    for record in fold_records:
        for model_name, report in (record.get("model_fit_reports") or {}).items():
            training_reports.append({"protocol": record.get("protocol"), "fold": record.get("fold"), "model": model_name, "report": report})
    (reports_dir / "neural_training_reports.json").write_text(json.dumps(training_reports, indent=2), encoding="utf-8")

    env = {
        "experiment": "leakage_safe_neural_hybrid_v1",
        "label": args.label,
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "pytorch_available": HAVE_TORCH,
        "pytorch_version": getattr(__import__("torch"), "__version__", None) if HAVE_TORCH else None,
        "device_requested": args.device,
        "competition_root": str(TRAIN_DIR.parent), "train_dir": str(TRAIN_DIR), "test_dir": str(TEST_DIR),
        "n_train_wells_discovered": len(all_train_ids), "n_test_wells_discovered": len(test_ids),
        "n_wells_validation_universe": len(universe), "n_wells_validated": int(well.well_id.nunique()),
        "n_points_evaluated": int(well.groupby(["protocol", "model"]).n_points.sum().max()),
        "n_folds": args.n_splits, "seed": args.seed, "models": model_names, "protocols": protocols,
        "blocked_well_ids": sorted(BLOCKED_WELL_IDS), "blocked_wells_in_validation": 0,
        "public_test_used_for_selection": False, "public_leaderboard_used_for_selection": False,
        "training_target_usage": "TVT hidden suffix and visible-prefix TVT_input pseudo-holdouts only inside fold-training fit; no target reaches InferenceTask.",
        "inference_safe_raw_sources": safe_inference_features(),
        "manifest_schema_verified": True, "cross_fitted": True,
        "runtime_seconds": round(time.perf_counter() - t0, 3), "peak_rss_mb": round(peak_rss_mb(), 1),
        "n_failures": len(failures), "submission_created": False,
        "final_submission_authorized": False,
    }
    (reports_dir / "neural_run_environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    _decision_report(reports_dir / "neural_decision_report.md", results, boot, env, torch_note=f"available={HAVE_TORCH}")
    print(f"Neural diagnostic reports written to {reports_dir}")
    print(results.to_string(index=False))
    print("No submission was created; Ridge Default remains the production fallback until the promotion rule passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
