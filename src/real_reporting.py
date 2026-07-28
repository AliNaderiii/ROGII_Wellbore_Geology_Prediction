"""Computed protocol and error-analysis reports for a real validation run.

This module never contains a competition metric literal.  It reads the
well-level results emitted by the cross-fitted runner and derives every table
from those rows.  It deliberately emits separate rows/tables for
``same_well_masked`` and ``unseen_well``; there is no combined score or average.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.ablation import (
    BRANCH_LABELS,
    BRANCH_ORDER,
    BRANCH_SPEC,
    alignment_feature_recommendation,
    alignment_feature_verdict,
    summarize_ablation,
)
from src.model_status import status_of, status_table
from src.reporting import spatial_ablation
from src.validation import (
    GR_BINS,
    GR_LABELS,
    PREFIX_BINS,
    PREFIX_LABELS,
    PROTOCOL_A,
    PROTOCOL_B,
    SUFFIX_BINS,
    SUFFIX_LABELS,
)

REAL_PROTOCOLS = (PROTOCOL_A, PROTOCOL_B)


def _md_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df is None or df.empty:
        return "_No computed rows were available._\n"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
            )
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda value: "yes" if value else "no")
        else:
            out[col] = out[col].astype(str)
    header = "| " + " | ".join(out.columns) + " |"
    rule = "|" + "|".join("---" for _ in out.columns) + "|"
    body = ["| " + " | ".join(row) + " |" for row in out.to_numpy()]
    return "\n".join([header, rule, *body]) + "\n"


def _bool_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Read ``name`` as a strict boolean Series, treating missing as ``False``.

    A CSV round-trip stores ``alignment_ok`` / ``alignment_cache_hit`` as
    ``object`` whenever any row is blank, and ``Series.fillna`` on an object
    column silently downcasts — which pandas deprecated (``FutureWarning``:
    "Downcasting object dtype arrays on .fillna, .ffill, .bfill is
    deprecated"). Converting explicitly is both warning-free and unambiguous
    about how a blank is scored: a well with no recorded flag is not counted
    as a success.
    """
    if name not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    column = frame[name]
    if pd.api.types.is_bool_dtype(column):
        return column.astype(bool)
    truthy = {"true", "1", "yes", "t"}
    coerced = column.map(
        lambda value: (
            False
            if value is None or (not isinstance(value, str) and pd.isna(value))
            else (str(value).strip().lower() in truthy if isinstance(value, str) else bool(value))
        )
    )
    return coerced.astype(bool)


def _weighted_rmse(group: pd.DataFrame) -> float:
    n = float(group["n_points"].sum())
    return float(np.sqrt(group["sse"].sum() / n)) if n else np.nan


def _require_columns(df: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _enrich_missing_metadata_from_mounted_tasks(well: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct score-only diagnostics for a historical result schema.

    Earlier completed runs did not persist target range, curvature or Typewell
    GR-correlation columns.  When the real mount is still available, these can
    be recomputed from each task without fitting or predicting a model.  A
    missing mount is harmless: the caller leaves the fields unavailable rather
    than substituting an estimate.
    """
    needed = {
        "target_min", "target_max", "target_range",
        "trajectory_curvature_deg_per_1000ft", "typewell_gr_correlation",
    }
    if needed.issubset(well.columns):
        return well
    try:
        from src.data import discover_wells, load_well
        from src.features import build_features
        from src.tasks import TaskConstructionError, make_task
        from src.validation import _trajectory_curvature_deg_per_1000ft

        files = discover_wells("train")
        cache: dict[str, object] = {}
        rows = []
        pairs = well[well["protocol"].isin(REAL_PROTOCOLS)][["protocol", "well_id"]].drop_duplicates()
        for item in pairs.itertuples(index=False):
            protocol, well_id = str(item.protocol), str(item.well_id)
            try:
                if well_id not in cache:
                    cache[well_id] = load_well(files[well_id])
                task = make_task(cache[well_id], "masked" if protocol == PROTOCOL_A else "real")
            except (KeyError, TaskConstructionError, OSError, ValueError):
                continue
            truth = np.asarray(task.target, dtype="float64") if task.target is not None else np.array([])
            truth = truth[np.isfinite(truth)]
            inp = task.inputs()
            # alignment=False avoids an unnecessary expensive hidden-track
            # search. This prefix-only correlation is the same diagnostic now
            # persisted by Ridge score rows.
            feats = build_features(inp, alignment=False)
            rows.append(
                {
                    "protocol": protocol,
                    "well_id": well_id,
                    "target_min": float(truth.min()) if truth.size else np.nan,
                    "target_max": float(truth.max()) if truth.size else np.nan,
                    "target_range": float(truth.max() - truth.min()) if truth.size else np.nan,
                    "trajectory_curvature_deg_per_1000ft": _trajectory_curvature_deg_per_1000ft(task),
                    "typewell_gr_correlation": float(feats.typewell_gr_prefix_correlation),
                    "metadata_source": "reconstructed_from_mounted_tasks",
                }
            )
        if not rows:
            return well
        meta = pd.DataFrame(rows)
        out = well.merge(meta, on=["protocol", "well_id"], how="left", suffixes=("", "_reconstructed"))
        for column in needed:
            reconstructed = f"{column}_reconstructed"
            if reconstructed in out.columns:
                if column in well.columns:
                    out[column] = out[column].where(out[column].notna(), out[reconstructed])
                else:
                    out[column] = out[reconstructed]
                out = out.drop(columns=[reconstructed])
        return out
    except Exception:
        # Report generation must not turn a completed metric artifact into a
        # failure merely because the optional data mount was detached.
        return well


def _ridge_rows(well: pd.DataFrame) -> pd.DataFrame:
    rows = well[(well["model"] == "ridge") & well["protocol"].isin(REAL_PROTOCOLS)].copy()
    if rows.empty or set(rows["protocol"]) != set(REAL_PROTOCOLS):
        raise ValueError(
            "The well-level results contain no Ridge rows for both real protocols. "
            "Run `scripts/run_validation.py --models ridge --protocols "
            "same_well_masked,unseen_well --real-analysis` first."
        )
    return rows


def _numeric_summary(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    rows = []
    for protocol, group in frame.groupby("protocol", sort=False):
        v = pd.to_numeric(group[column], errors="coerce").dropna()
        rows.append(
            {
                "protocol": protocol,
                "measure": label,
                "n_wells": int(v.size),
                "min": float(v.min()) if len(v) else np.nan,
                "p25": float(v.quantile(0.25)) if len(v) else np.nan,
                "median": float(v.median()) if len(v) else np.nan,
                "p75": float(v.quantile(0.75)) if len(v) else np.nan,
                "p90": float(v.quantile(0.90)) if len(v) else np.nan,
                "max": float(v.max()) if len(v) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _histogram(frame: pd.DataFrame, column: str, bins, labels, name: str) -> pd.DataFrame:
    rows = []
    local = frame.copy()
    local["stratum"] = pd.cut(local[column], bins=bins, labels=labels)
    for (protocol, stratum), group in local.groupby(["protocol", "stratum"], observed=False, sort=False):
        rows.append(
            {
                "protocol": protocol,
                "distribution": name,
                "stratum": str(stratum),
                "n_wells": int(len(group)),
                "n_scored_points": int(group["n_points"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _error_by(frame: pd.DataFrame, column: str, bins, labels, name: str) -> pd.DataFrame:
    rows = []
    local = frame.copy()
    local["stratum"] = pd.cut(local[column], bins=bins, labels=labels)
    for (protocol, stratum), group in local.groupby(["protocol", "stratum"], observed=False, sort=False):
        rows.append(
            {
                "model": "ridge",
                "protocol": protocol,
                "stratify_by": name,
                "stratum": str(stratum),
                "n_wells": int(len(group)),
                "n_points": int(group["n_points"].sum()),
                "global_rmse": _weighted_rmse(group),
                "median_well_rmse": float(group["rmse"].median()),
                "mean_well_rmse": float(group["rmse"].mean()),
                "worst_well_rmse": float(group["rmse"].max()),
            }
        )
    return pd.DataFrame(rows)


def _protocol_metrics(ridge: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol, group in ridge.groupby("protocol", sort=False):
        worst = group.nlargest(min(10, len(group)), "rmse")
        rows.append(
            {
                "protocol": protocol,
                "model": "ridge",
                "n_wells": int(group["well_id"].nunique()),
                "scored_points": int(group["n_points"].sum()),
                "hidden_suffix_rows": int(group["suffix_len"].sum()),
                "global_rmse": _weighted_rmse(group),
                "median_well_rmse": float(group["rmse"].median()),
                "worst10_well_rmse": float(worst["rmse"].mean()),
                "score_equals_entire_hidden_suffix": bool(
                    (group["n_points"] == group["suffix_len"]).all()
                ),
            }
        )
    return pd.DataFrame(rows)


def _target_range(ridge: pd.DataFrame) -> pd.DataFrame:
    needed = {"target_min", "target_max", "target_range"}
    if not needed.issubset(ridge.columns):
        return pd.DataFrame(
            [{
                "protocol": "unavailable",
                "reason": "This historical well_level_validation.csv predates target-range reporting; rerun validation to compute it.",
            }]
        )
    rows = []
    for protocol, group in ridge.groupby("protocol", sort=False):
        vals = pd.to_numeric(group["target_range"], errors="coerce").dropna()
        rows.append(
            {
                "protocol": protocol,
                "target_TVT_min": float(pd.to_numeric(group["target_min"], errors="coerce").min()),
                "target_TVT_max": float(pd.to_numeric(group["target_max"], errors="coerce").max()),
                "per_well_target_range_median": float(vals.median()) if len(vals) else np.nan,
                "per_well_target_range_p90": float(vals.quantile(0.90)) if len(vals) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _add_tail_flags(ridge: pd.DataFrame) -> pd.DataFrame:
    """Attach reproducible tail flags without pretending they are causal tests."""
    out = ridge.copy()
    out["high_gr_missingness"] = out["gr_missing_frac"] >= 0.20
    out["long_hidden_suffix"] = False
    out["high_trajectory_curvature"] = False
    out["weak_typewell_alignment"] = False
    out["large_ridge_residual"] = False
    for protocol, idx in out.groupby("protocol").groups.items():
        g = out.loc[idx]
        out.loc[idx, "long_hidden_suffix"] = g["suffix_len"] >= g["suffix_len"].quantile(0.75)
        out.loc[idx, "large_ridge_residual"] = g["rmse"] >= g["rmse"].quantile(0.75)
        if "trajectory_curvature_deg_per_1000ft" in out.columns:
            curve = pd.to_numeric(g["trajectory_curvature_deg_per_1000ft"], errors="coerce")
            if curve.notna().any():
                out.loc[idx, "high_trajectory_curvature"] = curve >= curve.quantile(0.75)
        if "typewell_gr_correlation" in out.columns:
            corr = pd.to_numeric(g["typewell_gr_correlation"], errors="coerce")
            # A same-log correlation below 0.20 is conventionally weak. NaN
            # means the correlation was unavailable (e.g. no usable typewell).
            out.loc[idx, "weak_typewell_alignment"] = corr.isna() | (corr < 0.20)
        elif "alignment_confidence_mean" in out.columns:
            # Compatibility path for historical reports produced before the
            # direct GR-correlation diagnostic was added.
            conf = pd.to_numeric(g["alignment_confidence_mean"], errors="coerce")
            out.loc[idx, "weak_typewell_alignment"] = conf.isna() | (conf < 0.35)
    return out


def _alignment_ablation(well: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    alignment = well[
        (well["model"] == "dip_constrained_alignment")
        & well["protocol"].isin(REAL_PROTOCOLS)
    ].copy()
    ridge = well[(well["model"] == "ridge") & well["protocol"].isin(REAL_PROTOCOLS)].copy()
    if alignment.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for protocol, group in alignment.groupby("protocol", sort=False):
        comparator = ridge[ridge["protocol"] == protocol]
        rr = _weighted_rmse(comparator) if len(comparator) else np.nan
        n = float(group["n_points"].sum())
        fallback = float(group["fallback_points"].sum()) if "fallback_points" in group else np.nan
        fail = int((~_bool_column(group, "alignment_ok")).sum())
        rows.append(
            {
                "protocol": protocol,
                "alignment_global_rmse": _weighted_rmse(group),
                "ridge_global_rmse": rr,
                "delta_vs_ridge": _weighted_rmse(group) - rr,
                "n_wells": int(len(group)),
                "n_points": int(n),
                "mean_alignment_confidence": float(pd.to_numeric(group.get("alignment_confidence_mean"), errors="coerce").mean()),
                "alignment_failure_wells": int(fail),
                "fallback_points": int(fallback),
                "fallback_fraction": fallback / n if n else np.nan,
                "cache_hit_wells": int(_bool_column(group, "alignment_cache_hit").sum()),
            }
        )
    return pd.DataFrame(rows), alignment


def write_real_analysis(reports_dir: str | Path) -> list[Path]:
    """Write the requested real-run protocol and error-analysis artifacts.

    The input is the cross-fitted well-level table.  If the table is absent,
    this fails loudly rather than creating a report whose uncomputed fields
    could be mistaken for real validation results.
    """
    root = Path(reports_dir)
    well_path = root / "well_level_validation.csv"
    if not well_path.exists():
        raise FileNotFoundError(
            f"{well_path} does not exist. Real analysis requires the raw per-well "
            "validation output; aggregate RMSEs alone cannot supply point counts, "
            "distributions, or worst-well diagnostics."
        )
    well = pd.read_csv(well_path)
    well = _enrich_missing_metadata_from_mounted_tasks(well)
    _require_columns(
        well,
        {"model", "protocol", "well_id", "n_points", "sse", "rmse", "prefix_len", "suffix_len", "gr_missing_frac"},
        well_path,
    )
    ridge = _ridge_rows(well)
    ridge = _add_tail_flags(ridge)

    metrics = _protocol_metrics(ridge)
    suffix_dist = _numeric_summary(ridge, "suffix_len", "hidden suffix length (rows)")
    prefix_dist = _numeric_summary(ridge, "prefix_len", "visible prefix length (rows)")
    target = _target_range(ridge)
    distributions = pd.concat(
        [
            _histogram(ridge, "suffix_len", SUFFIX_BINS, SUFFIX_LABELS, "hidden_suffix_length"),
            _histogram(ridge, "prefix_len", PREFIX_BINS, PREFIX_LABELS, "prefix_length"),
        ],
        ignore_index=True,
    )
    gr_error = _error_by(ridge, "gr_missing_frac", GR_BINS, GR_LABELS, "gr_missingness")
    suffix_error = _error_by(ridge, "suffix_len", SUFFIX_BINS, SUFFIX_LABELS, "hidden_suffix_length")
    prefix_error = _error_by(ridge, "prefix_len", PREFIX_BINS, PREFIX_LABELS, "prefix_length")

    # The per-well CSV is intentionally the source of truth for the tail
    # analysis, including flags instead of a prose-only assertion.
    error_columns = [
        "protocol", "fold", "well_id", "n_points", "sse", "rmse", "max_abs_error", "bias",
        "prefix_len", "suffix_len", "gr_missing_frac", "has_typewell", "target_min", "target_max",
        "target_range", "scored_exact_suffix", "trajectory_curvature_deg_per_1000ft",
        "alignment_confidence_mean", "alignment_confidence_p10", "typewell_gr_correlation", "alignment_ok",
        "alignment_failure_reason", "high_gr_missingness", "long_hidden_suffix",
        "high_trajectory_curvature", "weak_typewell_alignment", "large_ridge_residual",
    ]
    error_columns = [c for c in error_columns if c in ridge.columns]
    error_out = root / "error_analysis_real.csv"
    ridge[error_columns].sort_values(["protocol", "rmse"], ascending=[True, False]).to_csv(error_out, index=False)

    gr_out = root / "gr_missingness_error_real.csv"
    gr_error.to_csv(gr_out, index=False)
    suffix_out = root / "suffix_length_error_real.csv"
    suffix_error.to_csv(suffix_out, index=False)
    prefix_out = root / "prefix_length_error_real.csv"
    prefix_error.to_csv(prefix_out, index=False)

    # A separate top-20 is kept for each protocol: concatenating rankings or
    # selecting one combined top-20 would violate protocol separation.
    worst = (
        ridge.sort_values(["protocol", "rmse"], ascending=[True, False])
        .groupby("protocol", group_keys=False)
        .head(20)
        .copy()
    )
    worst["rank_within_protocol"] = worst.groupby("protocol").cumcount() + 1
    worst_columns = ["protocol", "rank_within_protocol", *[c for c in error_columns if c != "protocol"]]
    worst_columns = list(dict.fromkeys(c for c in worst_columns if c in worst.columns))
    worst_out = root / "worst_wells_real.csv"
    worst[worst_columns].to_csv(worst_out, index=False)

    # Protocol narrative: all values interpolate directly from the computed
    # tables above.  The explanation is limited to protocol mechanics and does
    # not claim an unobserved geological cause.
    lines = [
        "# Real validation protocol comparison\n",
        "**Scope.** This report is derived from the completed cross-fitted per-well output. "
        "`same_well_masked` and `unseen_well` remain separate throughout; this report never averages them "
        "in a metric, well rank, or distribution.\n",
        "## Ridge headline and exact scoring support\n",
        _md_table(metrics),
        "`hidden_suffix_rows` is the number of rows requested by each task; `scored_points` "
        "is the number with finite prediction and truth. `score_equals_entire_hidden_suffix` "
        "is true only when those counts match for every Ridge well in that protocol.\n",
        "## Why the two RMSEs can differ\n",
        "Both rows are cross-fitted by Well ID, so the gap is **not** an in-sample versus "
        "out-of-sample comparison. In `same_well_masked`, the boundary is moved earlier in the "
        "visible prefix and the truth is the masked `TVT_input` interval. In `unseen_well`, the "
        "boundary and truth are the real hidden suffix and label, respectively. The mask mirrors "
        "the real suffix only subject to the 200-row-prefix clip, so the available context and "
        "target interval can differ. The distributions below quantify that difference; they are "
        "the evidence for interpreting an RMSE gap, rather than a claim that one protocol is easier.\n",
        "## Hidden suffix length distribution\n",
        _md_table(suffix_dist),
        _md_table(distributions[distributions["distribution"] == "hidden_suffix_length"]),
        "## Visible prefix length distribution\n",
        _md_table(prefix_dist),
        _md_table(distributions[distributions["distribution"] == "prefix_length"]),
        "## Target TVT range in the scored region\n",
        _md_table(target),
        "## Exact scoring-region audit\n",
        "For `same_well_masked`, the task constructor masks `[start, real_prediction_start)` and "
        "uses only `TVT_input` there; for `unseen_well`, it predicts `[real_prediction_start, n)` "
        "and scores only the held-out `TVT` label there. `InferenceTask.assert_no_target()` also "
        "requires `TVT_input` to be NaN from `start` onward. The computed boolean above is the "
        "row-count confirmation for the completed run.\n",
        "## Error versus GR missingness\n",
        _md_table(gr_error),
        "## Error versus hidden suffix length\n",
        _md_table(suffix_error),
        "## Error versus prefix length\n",
        _md_table(prefix_error),
        "## Worst wells: separate top-20 lists\n",
        "`worst_wells_real.csv` contains 20 Ridge wells ranked independently within each protocol. "
        "It also records whether each is in the high-GR-missingness bin (≥20%), upper-quartile "
        "suffix/curvature group, has weak or unavailable visible-prefix Typewell GR correlation (<0.20), "
        "and is upper-quartile Ridge RMSE. These are descriptive flags, not causal attributions.\n",
    ]
    protocol_out = root / "protocol_comparison_real.md"
    protocol_out.write_text("\n".join(lines), encoding="utf-8")

    # Spatial diagnostics and ablation report.  A missing feature diagnostic
    # stays explicitly unavailable; it is not converted into an assumed zero.
    result_path = root / "validation_results.csv"
    result = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    spatial = spatial_ablation(result) if len(result) else pd.DataFrame()
    diag_path = root / "spatial_feature_diagnostics.csv"
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    spatial_lines = [
        "# Spatial Ridge ablation — real validation\n",
        "Spatial and non-spatial Ridge are paired **within each protocol**. No cross-protocol "
        "average is calculated. The donor guard excludes all validation wells, and query-time "
        "self-exclusion removes the queried well from every neighbour list.\n",
        "## RMSE A/B\n",
        _md_table(spatial[spatial["model"] == "ridge"] if len(spatial) else spatial),
    ]
    if len(diag) and {"feature", "n_populated", "n_prediction_rows", "non_constant"}.issubset(diag.columns):
        summary = (
            diag.groupby(["protocol", "feature"], as_index=False)
            .agg(
                n_prediction_rows=("n_prediction_rows", "sum"),
                n_populated=("n_populated", "sum"),
                any_non_constant=("non_constant", "any"),
                max_unique_finite=("n_unique_finite", "max"),
            )
        )
        summary["populated_fraction"] = summary["n_populated"] / summary["n_prediction_rows"].replace(0, np.nan)
        spatial_lines += [
            "## Feature population and variation\n",
            _md_table(summary),
            "A feature is populated when it has a finite value (or `nbr_n > 0` for neighbour count); "
            "`any_non_constant` is measured on actual validation rows.\n",
        ]
    else:
        spatial_lines += [
            "## Feature population and variation\n",
            "_Unavailable: this run predates validation-row spatial diagnostics or did not run `--spatial`. "
            "No conclusion about all-zero or constant spatial features is made._\n",
        ]
    if len(spatial):
        same = spatial[(spatial["protocol"] == PROTOCOL_A) & (spatial["model"] == "ridge")]
        if len(same):
            delta = float(same.iloc[0]["delta_global_rmse"])
            direction = "did not improve" if delta >= 0 else "improved"
            spatial_lines += [
                "## Interpretation\n",
                f"On `same_well_masked`, spatial Ridge {direction} global RMSE by {delta:+.3f}. "
                "This small aggregate delta does not establish significance. If the diagnostic table shows "
                "populated, non-constant features, the likely limitation is signal usefulness rather than an "
                "empty feature matrix: XY-nearest TVT donors can be noisy after structural variation, and Ridge "
                "already receives trajectory/GR/typewell information.\n",
            ]
    else:
        spatial_lines += ["## Interpretation\n", "_No paired Ridge spatial result was available; no performance explanation is asserted._\n"]
    spatial_out = root / "spatial_ablation_real.md"
    spatial_out.write_text("\n".join(spatial_lines), encoding="utf-8")

    # Controlled alignment A/B report, emitted only when the experiment was
    # actually run.  Its absence is a non-result, never a fabricated metric.
    ablation, alignment_rows = _alignment_ablation(well)
    produced = [protocol_out, error_out, gr_out, suffix_out, prefix_out, worst_out, spatial_out]
    if len(ablation):
        alignment_csv = root / "dip_constrained_alignment_ablation.csv"
        ablation.to_csv(alignment_csv, index=False)
        failed = alignment_rows[~_bool_column(alignment_rows, "alignment_ok")].copy()
        if len(failed):
            reasons = failed["alignment_failure_reason"].astype("string").fillna("")
            reasons = reasons.mask(reasons.str.strip() == "", "unspecified").astype(str)
            reason_counts = (
                failed.assign(alignment_failure_reason=reasons)
                .groupby(["protocol", "alignment_failure_reason"], as_index=False)
                .agg(failure_wells=("well_id", "nunique"), scored_points=("n_points", "sum"))
            )
        else:
            reason_counts = pd.DataFrame(columns=["protocol", "alignment_failure_reason", "failure_wells", "scored_points"])
        alignment_md = root / "dip_constrained_alignment_real.md"
        alignment_md.write_text(
            "# Dip-Constrained GR/Typewell Alignment — controlled A/B\n\n"
            "This is a direct alignment diagnostic compared only with its Ridge reference, "
            "under each protocol independently. It uses horizontal GR, Typewell GR/TVT reference, "
            "MD/X/Y/Z and visible `TVT_input`; it does not use the `TVT` label, Typewell Geology, "
            "formation markers, external artifacts, Ridge stacking, Particle Filter, Beam Search, "
            "or an ensemble. Alignment tracks are cached target-free by well and boundary.\n\n"
            "## Computed A/B and alignment diagnostics\n\n"
            + _md_table(ablation)
            + "\n`alignment_failure_wells` counts unusable dip/GR/typewell alignments. `fallback_points` "
            "counts predicted rows where the model used its visible-prefix X/Y/Z dip projection because "
            "the match failed or confidence was below the fixed threshold.\n\n"
            "## Alignment failure reasons\n\n"
            + _md_table(reason_counts)
            + "\n## Promotion status\n\n"
            + _md_table(pd.DataFrame(status_table()))
            + "\n"
            + _rejection_paragraph("dip_constrained_alignment"),
            encoding="utf-8",
        )
        produced += [alignment_csv, alignment_md]

    produced += write_alignment_spatial_ablation(root)
    return produced


def _rejection_paragraph(model: str) -> str:
    status = status_of(model)
    if not status.is_rejected:
        return f"`{model}` is currently **{status.status}**. {status.reason}\n"
    return (
        f"`{model}` is **{status.status}**. {status.reason} "
        f"Evidence: {status.source_run}. It is blocked from every final-predictor and "
        "ensemble path by `src.model_status.assert_not_rejected`.\n"
    )


def write_alignment_spatial_ablation(reports_dir: str | Path) -> list[Path]:
    """Write the A/B/C/D Ridge feature ablation report, when it was run.

    Reads ``alignment_spatial_ablation_wells.csv`` — the per-well output of
    ``scripts/run_feature_ablation.py``. Absence means the ablation was not
    run; no table is invented in that case.
    """
    root = Path(reports_dir)
    wells_path = root / "alignment_spatial_ablation_wells.csv"
    if not wells_path.exists():
        return []
    wells = pd.read_csv(wells_path)
    summary = summarize_ablation(wells)
    if summary.empty:
        return []
    verdict = alignment_feature_verdict(summary)
    recommendation = alignment_feature_recommendation(verdict)

    summary_csv = root / "alignment_spatial_ablation.csv"
    summary.to_csv(summary_csv, index=False)
    verdict_csv = root / "alignment_feature_verdict.csv"
    verdict.to_csv(verdict_csv, index=False)

    keep = recommendation["decision"] == "keep_as_features"
    action = (
        "**Keep** the alignment features, as residual/features only — never as a direct "
        "predictor or an ensemble branch."
        if keep
        else "**Remove** the alignment features from the next baseline."
    )
    lines = [
        "# Ridge alignment / spatial feature ablation\n",
        "A 2x2 factorial run through the **existing** Ridge model. Branch B is the former "
        "baseline and historical delta reference; the real decision selected branch A. All four branches share "
        "the same folds and are cross-fitted by well ID; the two protocols are reported "
        "separately and never averaged.\n",
        "| branch | alignment features | spatial features |\n|---|---|---|\n"
        + "\n".join(
            f"| {BRANCH_LABELS[b]} | {'yes' if BRANCH_SPEC[b][0] else 'no'} | "
            f"{'yes' if BRANCH_SPEC[b][1] else 'no'} |"
            for b in BRANCH_ORDER
        )
        + "\n",
        "## Historical delta against the former Ridge baseline (branch B)\n",
        _md_table(summary.drop(columns=["label"], errors="ignore")),
        "Only wells scored by every branch within a protocol enter the comparison, so a branch "
        "cannot look better by having dropped a hard well.\n",
        "## Isolating the alignment features\n",
        _md_table(verdict),
        "Each row is a paired contrast holding the spatial setting fixed. A negative "
        "`delta_global_rmse` means the alignment features lowered RMSE.\n",
        "## Decision\n",
        f"{recommendation['n_helping']} of {recommendation['n_contrasts']} contrasts favour the "
        f"alignment features, covering {', '.join(recommendation['protocols_covered'])}. "
        f"{recommendation['reason'].capitalize()}.\n",
        f"{action}\n",
    ]
    report = root / "alignment_spatial_ablation.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return [summary_csv, verdict_csv, report]
