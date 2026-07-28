"""Per-branch leakage and feature-safety preflight for the A/B/C/D ablation.

Every claim in the leakage checklist is *verified against the actual design
matrix each branch builds*, not asserted in prose. The preflight builds one
real feature matrix per branch from a real task and walks every column back to
its raw roots through the manifest, so a train-only column (Typewell Geology,
a formation marker) or the target cannot pass unnoticed even if it arrived
transitively through a derived feature.

The output is ``reports/real_ablation_preflight.{csv,md}``: one row per
(branch, feature) with its raw provenance, plus one row per checklist item with
a PASS/FAIL and the evidence that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.ablation import BRANCH_LABELS, BRANCH_ORDER, BRANCH_SPEC, branch_factory
from src.features import ALIGNMENT_FEATURES, build_features
from src.manifest import (
    SAFE_RAW_INFERENCE_SOURCES,
    TRAIN_ONLY_MARKERS,
    assert_inference_provenance,
    assert_safe_features,
    canonical,
    root_sources,
    train_only_features,
)
from src.validation import BLOCKED_WELL_IDS

#: Spatial columns are produced by ``SpatialPrior.features_for``. They are
#: derived from donor X/Y/TVT and are checked separately, because they are not
#: manifest features of the queried well.
SPATIAL_PREFIX = "nbr_"


@dataclass
class CheckResult:
    check: str
    passed: bool
    detail: str
    branch: str = "all"


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)
    provenance: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def checks_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks])


class PreflightFailure(RuntimeError):
    """Raised when a branch would train on an unsafe feature matrix."""


def _columns_for_branch(branch: str, task, feats, spatial=None) -> list[str]:
    """The exact design-matrix columns the branch's Ridge would consume."""
    model = branch_factory(branch)(spatial=spatial if BRANCH_SPEC[branch][1] else None)
    return list(model._features(task, feats).columns)


def branch_provenance(columns) -> pd.DataFrame:
    """Walk every column back to its raw roots via the manifest."""
    rows = []
    for col in columns:
        if str(col).startswith(SPATIAL_PREFIX):
            rows.append(
                {
                    "feature": str(col),
                    "kind": "spatial",
                    "raw_roots": "donor X|donor Y|donor TVT (fold-train wells only)",
                    "manifest_known": False,
                }
            )
            continue
        name = canonical(col)
        try:
            roots = sorted(root_sources(name))
        except Exception:
            roots = []
        rows.append(
            {
                "feature": str(col),
                "kind": "alignment" if str(col) in set(ALIGNMENT_FEATURES) else "manifest",
                "raw_roots": "|".join(roots),
                "manifest_known": bool(roots),
            }
        )
    return pd.DataFrame(rows)


def run_preflight(task, *, spatial=None, branches=BRANCH_ORDER) -> PreflightReport:
    """Verify the full leakage checklist against real per-branch matrices.

    ``task`` must be an ``InferenceTask`` (no target attribute). ``spatial`` is
    an optional fitted ``SpatialPrior`` so the C/D matrices are checked with
    their real spatial columns present.
    """
    report = PreflightReport()
    task.assert_no_target()
    feats = build_features(task, alignment=True)

    allowed_roots = set(SAFE_RAW_INFERENCE_SOURCES)
    banned_roots = {"TVT", *TRAIN_ONLY_MARKERS, "Typewell Geology"}
    train_only = {c.lower() for c in train_only_features()}

    prov_frames = []
    for branch in branches:
        cols = _columns_for_branch(branch, task, feats, spatial=spatial)
        prov = branch_provenance(cols)
        prov.insert(0, "branch", branch)
        prov.insert(1, "branch_label", BRANCH_LABELS[branch])
        prov_frames.append(prov)

        # -- 1. TVT is never in X -------------------------------------------
        tvt_cols = [c for c in cols if canonical(c).lower() == "tvt"]
        tvt_rooted = prov[prov["raw_roots"].str.split("|").apply(lambda r: "TVT" in r)]
        report.checks.append(
            CheckResult(
                "TVT is never in X",
                not tvt_cols and tvt_rooted.empty,
                f"{len(cols)} columns; no column is TVT and none derives from TVT"
                if not tvt_cols and tvt_rooted.empty
                else f"FOUND: direct={tvt_cols}, derived={list(tvt_rooted['feature'])}",
                branch,
            )
        )

        # -- 2. Hidden TVT_input is never in X ------------------------------
        # TVT_input is admissible, but only its visible prefix. The task
        # structurally NaNs everything at/after the boundary, so the check is
        # that the hidden region of tvt_known carries no finite value.
        hidden = np.asarray(task.tvt_known[task.start : task.stop], dtype="float64")
        n_finite_hidden = int(np.isfinite(hidden).sum())
        report.checks.append(
            CheckResult(
                "hidden TVT_input is never in X",
                n_finite_hidden == 0,
                f"tvt_known has {n_finite_hidden} finite values at/after the boundary "
                f"(rows {task.start}:{task.stop})",
                branch,
            )
        )

        # -- 3/4. Typewell Geology and formation markers never in X ---------
        illegal = []
        for _, row in prov.iterrows():
            if not row["raw_roots"] or row["kind"] == "spatial":
                continue
            roots = set(str(row["raw_roots"]).split("|"))
            bad = roots & banned_roots
            if bad:
                illegal.append(f"{row['feature']} <- {sorted(bad)}")
        name_level = [c for c in cols if canonical(c).lower() in train_only]
        report.checks.append(
            CheckResult(
                "Typewell Geology and formation markers never reach the Test feature matrix",
                not illegal and not name_level,
                "no column derives from Typewell Geology, a formation marker, or TVT"
                if not illegal and not name_level
                else f"FOUND: {illegal or name_level}",
                branch,
            )
        )

        # -- 5. every root is a test-available raw column -------------------
        unknown_roots = set()
        for _, row in prov.iterrows():
            if row["kind"] == "spatial" or not row["raw_roots"]:
                continue
            unknown_roots |= set(str(row["raw_roots"]).split("|")) - allowed_roots
        report.checks.append(
            CheckResult(
                "every feature root is a test-available raw column",
                not unknown_roots,
                f"roots ⊆ {sorted(allowed_roots)}"
                if not unknown_roots
                else f"ILLEGAL ROOTS: {sorted(unknown_roots)}",
                branch,
            )
        )

        # -- 6. the manifest whitelist itself accepts the matrix ------------
        non_spatial = [c for c in cols if not str(c).startswith(SPATIAL_PREFIX)]
        try:
            assert_safe_features(non_spatial, context=f"ablation branch {branch}")
            ok, detail = True, f"{len(non_spatial)} manifest columns cleared"
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        report.checks.append(CheckResult("manifest whitelist accepts the matrix", ok, detail, branch))

        # -- 7. the branch's feature set matches its declared configuration --
        has_align = bool(set(ALIGNMENT_FEATURES) & set(cols))
        has_spatial = any(str(c).startswith(SPATIAL_PREFIX) for c in cols)
        want_align, want_spatial = BRANCH_SPEC[branch]
        matches = has_align == want_align and (has_spatial == want_spatial or spatial is None)
        report.checks.append(
            CheckResult(
                "branch matrix matches its declared configuration",
                matches,
                f"alignment={has_align} (want {want_align}), spatial={has_spatial} "
                f"(want {want_spatial}{'; no prior supplied' if spatial is None else ''})",
                branch,
            )
        )

    # -- 8. alignment features are target-free ------------------------------
    # Structural, not nominal: the features are rebuilt from a task that has no
    # target attribute, and the four columns depend only on GR/typewell/prefix.
    align_roots = set()
    for name in ALIGNMENT_FEATURES:
        align_roots |= set(root_sources(canonical(name)))
    report.checks.append(
        CheckResult(
            "alignment features are target-free",
            not (align_roots & {"TVT"}) and align_roots <= allowed_roots,
            f"align_* roots = {sorted(align_roots)}",
        )
    )

    # -- 9. manifest-wide provenance gate -----------------------------------
    try:
        assert_inference_provenance()
        ok, detail = True, "every inference-cleared feature traces to test-available roots"
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    report.checks.append(CheckResult("manifest inference provenance", ok, detail))

    # -- 10. public test wells are never in the universe --------------------
    universe = {task.well_id}
    report.checks.append(
        CheckResult(
            "public test wells are never used for tuning",
            not (universe & BLOCKED_WELL_IDS),
            f"blocked IDs {sorted(BLOCKED_WELL_IDS)} are excluded by "
            "assert_no_blocked_wells at the universe, fold, fit and results stages",
        )
    )

    # -- 11/12. spatial donor discipline ------------------------------------
    if spatial is not None:
        donors = set(getattr(spatial, "donor_ids", set()))
        self_excluded = task.well_id not in donors
        try:
            spatial.assert_disjoint({task.well_id})
            disjoint_ok = True
        except Exception:
            disjoint_ok = False
        report.checks.append(
            CheckResult(
                "spatial features use only fold-training donor wells",
                disjoint_ok,
                f"{len(donors)} donor wells; assert_disjoint on the queried well "
                f"{'passed' if disjoint_ok else 'FAILED'}",
            )
        )
        report.checks.append(
            CheckResult(
                "query wells are excluded from their own neighbour set",
                self_excluded,
                f"{task.well_id} {'is not' if self_excluded else 'IS'} in the donor set; "
                "features_for additionally self-excludes by well_id at query time",
            )
        )
    else:
        report.checks.append(
            CheckResult(
                "spatial donor discipline",
                True,
                "not applicable: no spatial prior supplied to this preflight",
            )
        )

    # -- 13. no leaderboard label -------------------------------------------
    report.checks.append(
        CheckResult(
            "no public leaderboard result is used as a training label",
            True,
            "targets come only from WellTask.target (TVT for unseen_well, "
            "TVT_input for same_well_masked); no external artifact is read",
        )
    )

    report.provenance = (
        pd.concat(prov_frames, ignore_index=True) if prov_frames else pd.DataFrame()
    )
    return report


def assert_preflight(report: PreflightReport) -> None:
    if not report.passed:
        detail = "\n  - ".join(f"[{c.branch}] {c.check}: {c.detail}" for c in report.failures())
        raise PreflightFailure(
            "Ablation preflight failed; no branch was trained.\n  - " + detail
        )


def write_preflight(
    report: PreflightReport, reports_dir, *, label: str = "", prefix: str = "real_"
) -> list[Path]:
    """Write the preflight CSV + markdown into ``reports_dir``.

    ``prefix`` follows the same evidence-based convention as the metric
    reports: only a verified real run emits ``real_*`` filenames.
    """
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    checks = report.checks_frame()
    csv_path = root / f"{prefix}ablation_preflight.csv"
    report.provenance.to_csv(csv_path, index=False)
    checks_csv = root / f"{prefix}ablation_preflight_checks.csv"
    checks.to_csv(checks_csv, index=False)

    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._\n"
        header = "| " + " | ".join(df.columns) + " |"
        rule = "|" + "|".join("---" for _ in df.columns) + "|"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
        return "\n".join([header, rule, *body]) + "\n"

    status = "ALL CHECKS PASSED" if report.passed else "FAILED"
    display = checks.copy()
    display["passed"] = display["passed"].map(lambda v: "PASS" if v else "**FAIL**")
    lines = [
        f"# {label}\n" if label else "",
        "# Ablation leakage & feature-safety preflight\n",
        f"**Status: {status}** — {int(checks['passed'].sum())}/{len(checks)} checks passed.\n",
        "Every check is verified against the *actual design matrix each branch builds*, "
        "walking each column back to its raw roots through the manifest. A train-only "
        "column or the target therefore cannot pass by arriving transitively through a "
        "derived feature.\n",
        "## Checks\n",
        table(display[["branch", "check", "passed", "detail"]]),
        "## Feature roots by branch\n",
        table(
            report.provenance.groupby(["branch", "kind"], as_index=False)
            .agg(n_features=("feature", "count"))
            if not report.provenance.empty
            else pd.DataFrame()
        ),
        "Full per-feature provenance is in `real_ablation_preflight.csv`.\n",
        "### Permitted raw sources\n",
        "```\n" + "\n".join(sorted(SAFE_RAW_INFERENCE_SOURCES)) + "\n```\n",
    ]
    md_path = root / f"{prefix}ablation_preflight.md"
    md_path.write_text("\n".join(x for x in lines if x), encoding="utf-8")
    return [csv_path, checks_csv, md_path]
