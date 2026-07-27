"""Feature manifest — the single source of truth for what may enter a model.

The manifest is *code*, not a spreadsheet, so that the same object which
documents a decision also enforces it:

    reports/feature_manifest.csv  <- rendered from MANIFEST
    assert_safe_features(cols)    <- raises on anything not marked USE

Every entry records the audited availability of the column in train and test,
whether it survives past ``Prediction Start``, whether it is derived from the
target, and the resulting decision. The availability facts encoded here come
from the completed feature audit on the real Kaggle data (773 train wells,
3 visible public test wells); they are stated as *audited findings*, and
``verify_manifest_against_data`` re-checks them against a live mount so the
document can never silently drift from reality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Audited constants
# --------------------------------------------------------------------------

#: Formation-top marker columns present in train and ABSENT from test.
TRAIN_ONLY_MARKERS: tuple[str, ...] = (
    "ANCC",
    "ASTNL",
    "ASTNU",
    "BUDA",
    "EGFDL",
    "EGFDU",
)

#: Geological stacking order (shallowest -> deepest), used for ordering checks
#: only. Distinct from the alphabetical audit listing above.
MARKER_STACKING_ORDER: tuple[str, ...] = (
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
)

DECISIONS = {
    "USE",
    "USE_PREFIX_ONLY",
    "TARGET",
    "REJECT",
    "DEFER",
    "USE_ALIGNMENT_ONLY",
}


@dataclass(frozen=True)
class FeatureSpec:
    """One row of the feature manifest."""

    feature_name: str
    source: str
    train_availability: str
    test_availability: str
    available_after_prediction_start: str
    target_derived: str
    safe_for_inference: str
    decision: str
    leakage_risk: str
    notes: str
    tier: str = "raw"
    used_by: str = ""

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"{self.feature_name}: unknown decision {self.decision!r}")


# --------------------------------------------------------------------------
# Tier 1 — raw columns
# --------------------------------------------------------------------------

_HW = "horizontal well CSV (<well_id>__horizontal_well.csv)"
_TW = "typewell CSV (<well_id>__typewell.csv)"

RAW_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        feature_name="MD",
        source=_HW,
        train_availability="yes (all 773 wells)",
        test_availability="yes (all 3 public test wells)",
        available_after_prediction_start="yes",
        target_derived="no",
        safe_for_inference="yes",
        decision="USE",
        leakage_risk="none",
        notes=(
            "Measured depth. Audited monotonic increasing, exactly 1 ft step, "
            "no duplicate values. Defines the row grid and the prefix/suffix "
            "boundary; used as the along-hole coordinate for every extrapolation."
        ),
        used_by="all",
    ),
    FeatureSpec(
        feature_name="X",
        source=_HW,
        train_availability="yes",
        test_availability="yes",
        available_after_prediction_start="yes",
        target_derived="no",
        safe_for_inference="yes",
        decision="USE",
        leakage_risk="none",
        notes=(
            "Surface easting of the trajectory point. Known for the whole "
            "wellbore at inference time (the trajectory is surveyed before the "
            "geology is interpreted), so it is legitimately available past "
            "Prediction Start. Drives lateral displacement and offset-well "
            "neighbour search."
        ),
        used_by="ridge, lightgbm, spatial",
    ),
    FeatureSpec(
        feature_name="Y",
        source=_HW,
        train_availability="yes",
        test_availability="yes",
        available_after_prediction_start="yes",
        target_derived="no",
        safe_for_inference="yes",
        decision="USE",
        leakage_risk="none",
        notes="Surface northing. Same reasoning as X.",
        used_by="ridge, lightgbm, spatial",
    ),
    FeatureSpec(
        feature_name="Z",
        source=_HW,
        train_availability="yes",
        test_availability="yes",
        available_after_prediction_start="yes",
        target_derived="no",
        safe_for_inference="yes",
        decision="USE",
        leakage_risk="none",
        notes=(
            "True vertical depth/elevation of the trajectory point. The single "
            "most important geometric input: TVT changes are the difference "
            "between the wellbore's vertical movement and the stratigraphic "
            "surface's vertical movement, so dZ is the observable half of dTVT."
        ),
        used_by="geometric projection, ridge, lightgbm",
    ),
    FeatureSpec(
        feature_name="GR",
        source=_HW,
        train_availability="yes (145 wells with high missingness; worst 80.1%)",
        test_availability="yes",
        available_after_prediction_start="yes",
        target_derived="no",
        safe_for_inference="yes",
        decision="USE",
        leakage_risk="none",
        notes=(
            "Gamma ray measured along the lateral. The only stratigraphic "
            "observation available inside the hidden suffix, and therefore the "
            "only feature that can correct a drifting geometric extrapolation. "
            "Interpolate WITHIN a well only — per-well tool calibration differs. "
            "145 wells have high missingness, so every GR-dependent model must "
            "degrade to a geometric fallback rather than emit NaN."
        ),
        used_by="gr matching, ncc, ridge, lightgbm",
    ),
    FeatureSpec(
        feature_name="TVT_input",
        source=_HW,
        train_availability="yes (visible prefix only)",
        test_availability="yes (visible prefix only)",
        available_after_prediction_start="no (NaN by construction)",
        target_derived="yes (it is the target, revealed on the prefix)",
        safe_for_inference="yes, strictly before Prediction Start",
        decision="USE_PREFIX_ONLY",
        leakage_risk=(
            "high if read past Prediction Start; none on the prefix (NaN there "
            "by construction in both splits)"
        ),
        notes=(
            "The known part of the TVT curve. Audited clean prefix/suffix "
            "structure with no internal gaps. It is the anchor for every "
            "baseline: all models predict the residual TVT - TVT_last so that "
            "hold-last is the zero prediction. Any feature derived from it must "
            "read only rows strictly before Prediction Start; "
            "``prefix_only_view`` enforces this."
        ),
        used_by="all (anchor)",
    ),
    FeatureSpec(
        feature_name="TVT",
        source=_HW,
        train_availability="yes (full curve, including the hidden suffix)",
        test_availability="no (this is what is scored)",
        available_after_prediction_start="train only — it is the label",
        target_derived="yes — it IS the target",
        safe_for_inference="no",
        decision="TARGET",
        leakage_risk="critical",
        notes=(
            "Never a feature under any transformation, lag, or aggregate. Train "
            "wells carry the complete curve including the hidden region, which "
            "makes accidental inclusion silent and catastrophic: it would score "
            "near zero in validation and fail on the leaderboard. "
            "``assert_safe_features`` and ``WellData.assert_no_target_leakage`` "
            "both refuse feature frames containing it."
        ),
        used_by="supervision only",
    ),
    FeatureSpec(
        feature_name="Typewell TVT",
        source=_TW,
        train_availability="yes (all wells have typewell files; 1,567,045 rows)",
        test_availability="yes (all 3 public test wells)",
        available_after_prediction_start="yes",
        target_derived="no",
        safe_for_inference="yes",
        decision="USE",
        leakage_risk=(
            "none — it is the typewell's own stratigraphic axis, not the "
            "horizontal well's label"
        ),
        notes=(
            "The vertical stratigraphic coordinate of the offset/type well. "
            "Provides the domain of the reference GR curve, i.e. the candidate "
            "set the alignment searches over, and the physical bounds "
            "[min, max] any prediction should respect. Distinct from the TVT "
            "target: this is a coordinate of a different well."
        ),
        used_by="gr matching, ncc, ridge, lightgbm",
    ),
    FeatureSpec(
        feature_name="Typewell GR",
        source=_TW,
        train_availability="yes",
        test_availability="yes",
        available_after_prediction_start="yes",
        target_derived="no",
        safe_for_inference="yes",
        decision="USE",
        leakage_risk="none",
        notes=(
            "Reference gamma-ray log as a function of Typewell TVT. Together "
            "with the lateral GR it defines the inverse problem the whole task "
            "reduces to: find the TVT whose reference GR best explains the GR "
            "observed in the lateral. Resampled onto a uniform TVT grid once "
            "per well and reused by the matching and cross-correlation models."
        ),
        used_by="gr matching, ncc, ridge, lightgbm",
    ),
    FeatureSpec(
        feature_name="Typewell Geology",
        source=_TW,
        train_availability="yes",
        test_availability="yes",
        available_after_prediction_start="yes",
        target_derived="no",
        safe_for_inference="yes, but deliberately restricted for now",
        decision="USE_ALIGNMENT_ONLY",
        leakage_risk=(
            "low directly; medium indirectly — formation labels are defined by "
            "the same stratigraphic surfaces that define TVT, so an "
            "unrestricted categorical encoding can smuggle in structural "
            "information that is not available with the same fidelity at "
            "inference"
        ),
        notes=(
            "Formation label of the typewell as a function of Typewell TVT. "
            "Used to interpret and sanity-check alignment (does the matched TVT "
            "land in a plausible formation?) and to enforce the "
            "ANCC->ASTNU->ASTNL->EGFDU->EGFDL->BUDA stacking order in "
            "post-processing. NOT one-hot encoded into the baseline feature "
            "matrix; promote only after a fold-safe A/B shows a gain."
        ),
        used_by="interpretation, post-processing checks",
    ),
)

MARKER_FEATURES: tuple[FeatureSpec, ...] = tuple(
    FeatureSpec(
        feature_name=name,
        source=_HW,
        train_availability="yes",
        test_availability="NO — column absent from the test horizontal wells",
        available_after_prediction_start="train only",
        target_derived=(
            "partially — formation tops are picked from the same structural "
            "interpretation that defines TVT"
        ),
        safe_for_inference="no",
        decision="REJECT",
        leakage_risk=(
            "critical — train-only column: silent train/serve skew plus "
            "target-adjacent structural information"
        ),
        notes=(
            f"{name} formation-top marker. Present in train, absent in test, so "
            "a model that consumes it raw cannot be evaluated honestly and "
            "cannot run at inference. Rejected as a raw feature. The only "
            "admissible future use is as the supervision target of a separate "
            "fold-trained structural-surface model whose *predictions* are then "
            "used as features — that is out of scope until the baselines are "
            f"reported. Stacking position: {MARKER_STACKING_ORDER.index(name) + 1} "
            f"of {len(MARKER_STACKING_ORDER)}."
        ),
        tier="raw",
        used_by="none (rejected)",
    )
    for name in TRAIN_ONLY_MARKERS
)


# --------------------------------------------------------------------------
# Tier 2 — derived features (built only from tier-1 USE / USE_PREFIX_ONLY)
# --------------------------------------------------------------------------

def _derived(
    name: str,
    parents: str,
    notes: str,
    *,
    decision: str = "USE",
    risk: str = "none",
    after_ps: str = "yes",
    used_by: str = "ridge, lightgbm",
) -> FeatureSpec:
    return FeatureSpec(
        feature_name=name,
        source=f"derived from {parents}",
        train_availability="yes (computed)",
        test_availability="yes (computed)",
        available_after_prediction_start=after_ps,
        target_derived="no",
        safe_for_inference="yes" if decision.startswith("USE") else "no",
        decision=decision,
        leakage_risk=risk,
        notes=notes,
        tier="derived",
        used_by=used_by,
    )


DERIVED_FEATURES: tuple[FeatureSpec, ...] = (
    _derived(
        "dmd",
        "MD",
        "MD - MD at Prediction Start: feet of hidden suffix already drilled. "
        "The primary uncertainty axis — error grows with it.",
    ),
    _derived(
        "log1p_dmd",
        "MD",
        "log(1 + dmd). Compresses the long tail so a linear model can express "
        "sub-linear error growth.",
    ),
    _derived(
        "dz",
        "Z",
        "Z - Z at the last known TVT row. The observable vertical movement of "
        "the bit; TVT changes only where dz differs from the structural surface.",
    ),
    _derived("dx", "X", "X - X at the last known TVT row."),
    _derived("dy", "Y", "Y - Y at the last known TVT row."),
    _derived(
        "lateral_disp",
        "X, Y",
        "sqrt(dx^2 + dy^2): horizontal step-out from the prediction start. The "
        "lever arm on which structural dip acts.",
    ),
    _derived(
        "dz_per_ft",
        "Z, MD",
        "dz / dmd: mean wellbore vertical gradient since the prediction start.",
    ),
    _derived(
        "local_dz_dmd",
        "Z, MD",
        "Locally smoothed dZ/dMD (51 ft window): instantaneous wellbore "
        "inclination relative to horizontal.",
    ),
    _derived(
        "heading_sin",
        "X, Y",
        "sin of the local trajectory azimuth. Combined with dip, azimuth sets "
        "the sign and magnitude of apparent dip along the hole.",
    ),
    _derived(
        "heading_cos",
        "X, Y",
        "cos of the local trajectory azimuth.",
    ),
    _derived(
        "tvt_last",
        "TVT_input (prefix only)",
        "Last known TVT value. The anchor: all models predict TVT - tvt_last.",
        after_ps="constant, captured before Prediction Start",
        risk="none — read strictly from the prefix",
        used_by="all",
    ),
    _derived(
        "tvt_slope_100",
        "TVT_input, MD (prefix only)",
        "Least-squares dTVT/dMD over the final 100 ft of the visible prefix.",
        after_ps="constant, captured before Prediction Start",
        risk="none — read strictly from the prefix",
    ),
    _derived(
        "tvt_slope_300",
        "TVT_input, MD (prefix only)",
        "Least-squares dTVT/dMD over the final 300 ft of the visible prefix. "
        "Less noisy, slower to react.",
        after_ps="constant, captured before Prediction Start",
        risk="none — read strictly from the prefix",
    ),
    _derived(
        "tvt_slope_1000",
        "TVT_input, MD (prefix only)",
        "Least-squares dTVT/dMD over the final 1000 ft of the visible prefix: "
        "the regional dip trend.",
        after_ps="constant, captured before Prediction Start",
        risk="none — read strictly from the prefix",
    ),
    _derived(
        "tvt_std_prefix",
        "TVT_input (prefix only)",
        "Standard deviation of TVT over the visible prefix: how structurally "
        "restless this well has been so far.",
        after_ps="constant, captured before Prediction Start",
        risk="none — read strictly from the prefix",
    ),
    _derived(
        "tvt_range_prefix",
        "TVT_input (prefix only)",
        "max - min of TVT over the visible prefix.",
        after_ps="constant, captured before Prediction Start",
        risk="none — read strictly from the prefix",
    ),
    _derived(
        "prefix_len",
        "TVT_input (prefix only)",
        "Number of rows before Prediction Start: how much evidence the anchor "
        "and slope estimates rest on.",
        after_ps="constant, captured before Prediction Start",
        risk="none — read strictly from the prefix",
    ),
    _derived(
        "gr_filled",
        "GR",
        "GR interpolated WITHIN the well (linear, then edge-held). Never "
        "interpolated across wells: per-well calibration baselines differ.",
    ),
    _derived(
        "gr_is_missing",
        "GR",
        "1 where GR was originally NaN. Lets the model discount gr_filled "
        "inside a tool outage instead of trusting an interpolated value.",
    ),
    _derived(
        "gr_roll_mean_51",
        "GR",
        "51 ft centred rolling mean of gr_filled: the smoothed log character "
        "actually used for correlation.",
    ),
    _derived(
        "gr_roll_std_51",
        "GR",
        "51 ft rolling standard deviation: log texture. Low values mean a "
        "featureless interval where alignment is unreliable.",
    ),
    _derived(
        "gr_z",
        "GR (prefix statistics)",
        "GR standardised by the well's own prefix mean/std, making the lateral "
        "log comparable to the normalised typewell log.",
    ),
    _derived(
        "gr_missing_frac_well",
        "GR",
        "Fraction of the well's rows with missing GR. Also a reporting stratum: "
        "RMSE is broken out by this variable.",
        after_ps="constant per well",
    ),
    _derived(
        "tw_gr_at_tvt_last",
        "Typewell GR, Typewell TVT, TVT_input",
        "Reference GR at the anchor TVT: where in the type section the bit was "
        "when supervision stopped.",
        after_ps="constant, captured before Prediction Start",
    ),
    _derived(
        "tw_dgr_dtvt_at_tvt_last",
        "Typewell GR, Typewell TVT, TVT_input",
        "Reference dGR/dTVT at the anchor. Large magnitude means a nearby "
        "marker bed and a well-conditioned alignment; near zero means the "
        "inverse problem is locally degenerate.",
        after_ps="constant, captured before Prediction Start",
    ),
    _derived(
        "tw_tvt_min",
        "Typewell TVT",
        "Lower bound of the reference section: predictions outside it are "
        "unphysical and are clipped in post-processing.",
    ),
    _derived(
        "tw_tvt_max",
        "Typewell TVT",
        "Upper bound of the reference section.",
    ),
    _derived(
        "tw_gr_std",
        "Typewell GR",
        "Variability of the reference log: how much character is available to "
        "correlate against.",
    ),
    _derived(
        "align_tvt",
        "GR, Typewell GR, Typewell TVT, MD",
        "TVT estimated by windowed normalized cross-correlation of the lateral "
        "GR against the typewell GR, bias-corrected on the prefix. A physical "
        "measurement, not a label: it never reads TVT.",
        used_by="ncc, gr matching, ridge, lightgbm",
    ),
    _derived(
        "align_score",
        "GR, Typewell GR",
        "Peak normalized correlation of the winning alignment in [-1, 1]. The "
        "model's own confidence; low values must trigger the geometric fallback.",
        used_by="ncc, gr matching, ridge, lightgbm",
    ),
    _derived(
        "align_shift",
        "align_tvt, TVT_input",
        "align_tvt - tvt_last: the stratigraphic move the correlation claims "
        "has happened since the anchor.",
        used_by="ridge, lightgbm",
    ),
    _derived(
        "align_gradient",
        "GR, Typewell GR, MD",
        "Best-fitting local dTVT/dMD from the 2-D (offset, gradient) alignment "
        "search: an apparent-dip estimate measured from the logs.",
        used_by="ridge, lightgbm",
    ),
)


# --------------------------------------------------------------------------
# Tier 3 — spatial / offset-well features (fold-safe by construction)
# --------------------------------------------------------------------------

def _spatial(name: str, notes: str) -> FeatureSpec:
    return FeatureSpec(
        feature_name=name,
        source="offset wells (X, Y of the target well; TVT of NEIGHBOUR wells)",
        train_availability="yes (computed per fold)",
        test_availability="yes (neighbours drawn from the train split)",
        available_after_prediction_start="yes",
        target_derived=(
            "yes, but only from OTHER wells' labels — never the well being "
            "predicted"
        ),
        safe_for_inference="yes, only when built fold-safely",
        decision="USE",
        leakage_risk=(
            "high if built globally — must be rebuilt inside every fold from "
            "fold-train wells only; the guard in src/spatial.py enforces this"
        ),
        notes=notes,
        tier="spatial",
        used_by="ridge, lightgbm (spatial variant)",
    )


SPATIAL_FEATURES: tuple[FeatureSpec, ...] = (
    _spatial(
        "nbr_n",
        "Number of offset wells contributing to the estimate at this row.",
    ),
    _spatial(
        "nbr_dist_min",
        "Distance (ft) to the nearest contributing offset-well sample.",
    ),
    _spatial(
        "nbr_tvt_wmean",
        "Inverse-distance-weighted mean TVT of the k nearest offset-well "
        "samples in the X/Y plane. The structural prior: what TVT other wells "
        "recorded at this map position.",
    ),
    _spatial(
        "nbr_tvt_std",
        "Spread of neighbour TVT values: disagreement between offset wells, "
        "i.e. local structural complexity.",
    ),
    _spatial(
        "nbr_shift",
        "nbr_tvt_wmean - tvt_last: the structural move the offset wells imply, "
        "expressed in the same anchored residual space the models predict in.",
    ),
    _spatial(
        "nbr_grad_along",
        "Component of the locally fitted neighbour TVT plane along the current "
        "trajectory heading — a directly estimated apparent dip.",
    ),
)


MANIFEST: tuple[FeatureSpec, ...] = (
    RAW_FEATURES + MARKER_FEATURES + DERIVED_FEATURES + SPATIAL_FEATURES
)

MANIFEST_COLUMNS = [
    "feature_name",
    "tier",
    "source",
    "train_availability",
    "test_availability",
    "available_after_prediction_start",
    "target_derived",
    "safe_for_inference",
    "decision",
    "leakage_risk",
    "used_by",
    "notes",
]


# --------------------------------------------------------------------------
# Rendering + enforcement
# --------------------------------------------------------------------------

def manifest_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(f) for f in MANIFEST])[MANIFEST_COLUMNS]


def write_manifest(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_frame().to_csv(out, index=False)
    return out


def _names(decisions: set[str]) -> list[str]:
    return [f.feature_name for f in MANIFEST if f.decision in decisions]


def safe_inference_features() -> list[str]:
    """Feature names that may appear in a model input matrix."""
    return _names({"USE"})


def prefix_only_features() -> list[str]:
    return _names({"USE_PREFIX_ONLY"})


def rejected_features() -> list[str]:
    """Names that must never appear in a model input matrix."""
    return _names({"REJECT", "TARGET", "USE_ALIGNMENT_ONLY"})


def target_features() -> list[str]:
    return _names({"TARGET"})


class FeatureLeakage(RuntimeError):
    """Raised when a rejected or target-derived column reaches a model."""


_ALIASES = {
    "tvt": "TVT",
    "tvt_target": "TVT",
    "target": "TVT",
    "tvt_input": "TVT_input",
    "typewell_tvt": "Typewell TVT",
    "typewell_gr": "Typewell GR",
    "typewell_geology": "Typewell Geology",
    "geology": "Typewell Geology",
}


def canonical(column: str) -> str:
    key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
    if key in _ALIASES:
        return _ALIASES[key]
    upper = key.upper()
    if upper in TRAIN_ONLY_MARKERS:
        return upper
    for spec in MANIFEST:
        if spec.feature_name.lower().replace(" ", "_") == key:
            return spec.feature_name
    return str(column)


def assert_safe_features(columns, *, context: str = "model input") -> None:
    """Raise ``FeatureLeakage`` if any column is not cleared for inference.

    Unknown columns are rejected too: the manifest is a whitelist, so a feature
    nobody has audited cannot reach a model by accident.
    """
    allowed = set(safe_inference_features())
    banned = {c.lower(): c for c in rejected_features()}
    unknown: list[str] = []
    leaks: list[str] = []
    for col in columns:
        name = canonical(col)
        if name in allowed:
            continue
        if name.lower() in banned:
            leaks.append(f"{col!r} -> {name} (decision={_decision_of(name)})")
        else:
            unknown.append(str(col))
    problems = []
    if leaks:
        problems.append("rejected/target columns: " + ", ".join(sorted(leaks)))
    if unknown:
        problems.append("columns absent from the manifest: " + ", ".join(sorted(unknown)))
    if problems:
        raise FeatureLeakage(
            f"{context}: refusing to proceed.\n  - " + "\n  - ".join(problems)
        )


def _decision_of(name: str) -> str:
    for spec in MANIFEST:
        if spec.feature_name == name:
            return spec.decision
    return "UNKNOWN"


# --------------------------------------------------------------------------
# Re-verification against a live mount
# --------------------------------------------------------------------------

def verify_manifest_against_data(
    train_columns, test_columns, *, tw_columns=None
) -> pd.DataFrame:
    """Re-check the manifest's availability claims against observed columns.

    Returns one row per raw manifest entry with ``claim``, ``observed`` and
    ``agrees``. Any ``agrees == False`` means the audit findings encoded here no
    longer describe the data and modelling must stop.
    """
    train = {canonical(c) for c in train_columns}
    test = {canonical(c) for c in test_columns}
    if tw_columns is not None:
        train |= {f"Typewell {canonical(c)}" for c in tw_columns}
        test |= {f"Typewell {canonical(c)}" for c in tw_columns}

    rows = []
    for spec in MANIFEST:
        if spec.tier != "raw":
            continue
        claim_train = not spec.train_availability.lower().startswith("no")
        claim_test = not spec.test_availability.lower().startswith("no")
        obs_train = spec.feature_name in train
        obs_test = spec.feature_name in test
        rows.append(
            {
                "feature_name": spec.feature_name,
                "claim_in_train": claim_train,
                "observed_in_train": obs_train,
                "claim_in_test": claim_test,
                "observed_in_test": obs_test,
                "agrees": bool(claim_train == obs_train and claim_test == obs_test),
                "decision": spec.decision,
            }
        )
    return pd.DataFrame(rows)
