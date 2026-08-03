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

Schema audit (authoritative, re-verified against the real Kaggle mount)
-----------------------------------------------------------------------
::

    TRAIN  <well>__typewell.csv  columns: ['TVT', 'GR', 'Geology']
    TEST   <well>__typewell.csv  columns: ['TVT', 'GR']

``Geology`` is therefore **train-only**. It is admissible for train-side EDA,
geological interpretation and error analysis, and for nothing else: it must
never reach the test feature matrix, the alignment features, post-processing,
calibration or an ensemble. The manifest encodes that as
``decision=TRAIN_ANALYSIS_ONLY`` and ``validate_manifest`` refuses to render or
enforce a manifest that says otherwise.
"""
from __future__ import annotations

import re
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

#: Observed typewell schemas, from the Kaggle schema audit. These are the
#: reference values the verification path compares a live mount against.
AUDITED_TRAIN_TYPEWELL_COLUMNS: tuple[str, ...] = ("TVT", "GR", "Geology")
AUDITED_TEST_TYPEWELL_COLUMNS: tuple[str, ...] = ("TVT", "GR")

#: Observed horizontal-well schemas, from the same audit.
AUDITED_TRAIN_HW_COLUMNS: tuple[str, ...] = (
    "MD", "X", "Y", "Z", "GR", "TVT_input", "TVT", *TRAIN_ONLY_MARKERS,
)
AUDITED_TEST_HW_COLUMNS: tuple[str, ...] = ("MD", "X", "Y", "Z", "GR", "TVT_input")

#: The only raw columns that may reach a final inference feature matrix, either
#: directly or as the provenance of a derived feature. Anything else is either
#: train-only (markers, Typewell Geology), the target (TVT), or unaudited.
SAFE_RAW_INFERENCE_SOURCES: tuple[str, ...] = (
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "TVT_input",          # visible prefix only
    "Typewell TVT",
    "Typewell GR",
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

#: Decisions whose holders may enter a model input matrix.
INFERENCE_DECISIONS = {"USE"}

#: Decisions that forbid a column from ever reaching an inference matrix.
#: ``TRAIN_ANALYSIS_ONLY`` is for train-only columns (Typewell Geology) that
#: remain useful for EDA and error analysis but are absent from test.
NON_INFERENCE_DECISIONS = {
    "TARGET",
    "REJECT",
    "DEFER",
    "TRAIN_ANALYSIS_ONLY",
    "USE_PREFIX_ONLY",
}

DECISIONS = INFERENCE_DECISIONS | NON_INFERENCE_DECISIONS


class ManifestInconsistency(RuntimeError):
    """Raised when the manifest contradicts itself or the audited schemas.

    Distinct from ``FeatureLeakage``: this fires on the *document*, before any
    data is touched, so a mis-stated availability claim is caught at import
    time rather than surviving into a run.
    """


#: Leading tokens that mean "absent" / "not permitted" in a manifest field.
_NEGATIVE_PREFIXES = ("no", "false", "never", "none")


def _is_no(text: str) -> bool:
    """True when an availability/safety field states absence or denial.

    These fields are prose, not booleans — "yes (all 773 wells)", "NO — column
    absent from the test horizontal wells", "false, but train-only geological
    metadata" — so the verdict is read from the leading token. ``"none"`` is
    included because a rejected feature may record ``used_by="none"``-style
    phrasing; ``"nope"``-like words are not special-cased on purpose, the
    vocabulary is deliberately small and checked by tests.
    """
    head = str(text).strip().lower()
    return any(
        head == p or head.startswith(p + " ") or head.startswith(p + ",")
        or head.startswith(p + ".") or head.startswith(p + ";")
        or head.startswith(p + "(") or head.startswith(p + " —")
        or head.startswith(p + "-")
        for p in _NEGATIVE_PREFIXES
    )


def _is_yes(text: str) -> bool:
    return not _is_no(text)


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
    #: Canonical names of the manifest entries this feature is computed from.
    #: Empty for raw columns, which are read straight off disk.
    parents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"{self.feature_name}: unknown decision {self.decision!r}")
        # Self-consistency, enforced per row so an inconsistent entry cannot be
        # constructed at all — not merely reported later.
        if self.decision in INFERENCE_DECISIONS and _is_no(self.test_availability):
            raise ManifestInconsistency(
                f"{self.feature_name}: decision={self.decision} requires the "
                f"column to exist in test, but test_availability="
                f"{self.test_availability!r}. A train-only feature must never "
                "be marked usable at inference."
            )
        if self.decision in INFERENCE_DECISIONS and _is_no(self.safe_for_inference):
            raise ManifestInconsistency(
                f"{self.feature_name}: decision={self.decision} contradicts "
                f"safe_for_inference={self.safe_for_inference!r}."
            )
        if self.decision == "TRAIN_ANALYSIS_ONLY":
            if _is_yes(self.test_availability):
                raise ManifestInconsistency(
                    f"{self.feature_name}: TRAIN_ANALYSIS_ONLY is reserved for "
                    "train-only columns, but test_availability="
                    f"{self.test_availability!r}."
                )
            if _is_yes(self.safe_for_inference):
                raise ManifestInconsistency(
                    f"{self.feature_name}: TRAIN_ANALYSIS_ONLY must have "
                    f"safe_for_inference=no, got {self.safe_for_inference!r}."
                )

    # -- convenience predicates used by the verification paths -------------
    @property
    def in_train(self) -> bool:
        return _is_yes(self.train_availability)

    @property
    def in_test(self) -> bool:
        return _is_yes(self.test_availability)

    @property
    def train_only(self) -> bool:
        return self.in_train and not self.in_test

    @property
    def claims_inference_safe(self) -> bool:
        return self.decision in INFERENCE_DECISIONS


# --------------------------------------------------------------------------
# Name canonicalisation
# --------------------------------------------------------------------------
# Defined before the manifest entries because provenance parsing (`_derived`)
# canonicalises parent names at construction time.

_ALIASES = {
    "tvt": "TVT",
    "tvt_target": "TVT",
    "target": "TVT",
    "tvt_input": "TVT_input",
    "typewell_tvt": "Typewell TVT",
    "typewell_gr": "Typewell GR",
    "typewell_geology": "Typewell Geology",
    "geology": "Typewell Geology",
    # A bare typewell "Geology" header, and the shapes it arrives in after a
    # `Typewell `-prefixed lookup. All resolve to the train-only entry so the
    # column cannot slip in under a different spelling.
    "formation": "Typewell Geology",
    "facies": "Typewell Geology",
    "tw_geology": "Typewell Geology",
    "typewell_formation": "Typewell Geology",
    "typewell_facies": "Typewell Geology",
}


def canonical(column: str) -> str:
    """Resolve a column spelling to its canonical manifest feature name."""
    key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
    if key in _ALIASES:
        return _ALIASES[key]
    upper = key.upper()
    if upper in TRAIN_ONLY_MARKERS:
        return upper
    for spec in globals().get("MANIFEST", ()):
        if spec.feature_name.lower().replace(" ", "_") == key:
            return spec.feature_name
    return str(column)


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
        test_availability="no",
        available_after_prediction_start="no",
        target_derived="false, but train-only geological metadata",
        safe_for_inference="false",
        decision="TRAIN_ANALYSIS_ONLY",
        leakage_risk="HIGH for final inference",
        notes=(
            "The Geology column is present in train Typewell files "
            "(['TVT', 'GR', 'Geology']) but absent from test Typewell files "
            "(['TVT', 'GR']). It may be used only for train-side EDA, "
            "geological interpretation, and error analysis. It must never "
            "enter the final Test feature matrix, alignment features, "
            "post-processing, calibration, or ensemble. Corrected from the "
            "earlier USE_ALIGNMENT_ONLY entry, which wrongly claimed test "
            "availability and would have produced silent train/serve skew: "
            "any stacking-order or formation-plausibility check built on it "
            "would run in validation and be unavailable at inference."
        ),
        used_by="train-side EDA and error analysis only (never inference)",
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

def _parse_parents(parents: str) -> tuple[str, ...]:
    """Split a provenance string into canonical parent feature names.

    ``"Typewell GR, Typewell TVT, TVT_input"`` -> those three names;
    parenthetical qualifiers ("(prefix only)", "(prefix statistics)") are
    descriptive and stripped. Making provenance machine-readable is what lets
    ``validate_manifest`` prove no derived feature descends from a train-only
    column such as Typewell Geology.
    """
    cleaned = re.sub(r"\([^)]*\)", " ", str(parents))
    out: list[str] = []
    for chunk in cleaned.split(","):
        name = chunk.strip()
        if name:
            out.append(canonical(name))
    return tuple(out)


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
        safe_for_inference="yes" if decision in INFERENCE_DECISIONS else "no",
        decision=decision,
        leakage_risk=risk,
        notes=notes,
        tier="derived",
        used_by=used_by,
        parents=_parse_parents(parents),
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
    # Particle Filter and Beam Search remain opt-in Ridge feature generators.
    # Their common provenance is intentionally explicit: Typewell TVT is the
    # reference coordinate (not the horizontal-well target), and TVT_input is
    # read only from the structurally visible prefix.
    *tuple(
        _derived(
            name,
            "GR, Typewell GR, Typewell TVT, MD, X, Y, Z, TVT_input (prefix only)",
            notes,
            risk="none — InferenceTask hides horizontal-well TVT and hidden TVT_input",
            used_by=used_by,
        )
        for name, notes, used_by in (
            ("pf_track", "Particle-filter weighted TVT-coordinate track; diagnostic feature only.", "ridge particle-filter opt-in"),
            ("pf_shift", "Particle-filter track minus the last visible TVT_input anchor.", "ridge particle-filter opt-in"),
            ("pf_gradient", "Particle-filter weighted local path gradient.", "ridge particle-filter opt-in"),
            ("pf_confidence", "Particle concentration after the horizontal-GR observation update.", "ridge particle-filter opt-in"),
            ("pf_branch_spread", "Weighted P90-P10 spread of particle TVT-coordinate branches.", "ridge particle-filter opt-in"),
            ("pf_path_smoothness", "Absolute change in particle-filter path gradient.", "ridge particle-filter opt-in"),
            ("pf_gr_misfit", "Weighted horizontal-GR versus Typewell-GR mismatch.", "ridge particle-filter opt-in"),
            ("pf_fallback", "One when the particle update uses only the prefix/geometry fallback.", "ridge particle-filter opt-in"),
            ("beam_track", "Beam-weighted TVT-coordinate track; diagnostic feature only.", "ridge beam-search opt-in"),
            ("beam_shift", "Beam track minus the last visible TVT_input anchor.", "ridge beam-search opt-in"),
            ("beam_gradient", "Beam-weighted local path gradient.", "ridge beam-search opt-in"),
            ("beam_confidence", "Cost-gap and branch-entropy confidence of the retained beam.", "ridge beam-search opt-in"),
            ("beam_branch_spread", "Weighted P90-P10 spread of retained beam branches.", "ridge beam-search opt-in"),
            ("beam_path_smoothness", "Absolute change in the beam-search path gradient.", "ridge beam-search opt-in"),
            ("beam_gr_misfit", "Beam-weighted horizontal-GR versus Typewell-GR mismatch.", "ridge beam-search opt-in"),
            ("beam_fallback", "One when beam expansion uses only the prefix/geometry fallback.", "ridge beam-search opt-in"),
        )
    ),
    # Controlled GeoAnchor experiment features (arms B/C/D). Raw roots are
    # strictly GR, Typewell GR, Typewell TVT, MD, and the visible TVT_input
    # prefix: the affine calibration fits alpha/beta on prefix rows only, and
    # the multi-branch datum scan compares GR-to-GR without reading any TVT
    # label on hidden rows. Used only by the leakage-gated experiment in
    # src/geoanchor.py; none of these columns is in the default Ridge matrix.
    *tuple(
        _derived(
            name,
            "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)",
            notes,
            risk="none — alpha/beta fitted strictly on prefix rows with a-priori sanity bounds",
            used_by=used_by,
        )
        for name, notes, used_by in (
            ("acal_alpha", "Prefix-only affine gain mapping horizontal GR onto Typewell GR.", "geoanchor arms B, D"),
            ("acal_beta", "Prefix-only affine offset of the horizontal-to-Typewell GR calibration.", "geoanchor arms B, D"),
            ("acal_fit_rmse", "Prefix residual RMS of the affine calibration, in Typewell-GR z units.", "geoanchor arms B, D"),
            ("acal_prefix_corr", "Pearson r between calibrated horizontal GR and Typewell GR on the visible prefix; a calibration-quality diagnostic.", "geoanchor arms B, D"),
            ("acal_ok", "One when the prefix affine calibration passed its sanity bounds.", "geoanchor arms B, D"),
            ("acal_gr_tw_z", "Horizontal GR mapped into Typewell GR space and z-scored (within-well interpolation of outages).", "geoanchor arms B, D"),
            ("acal_roll_mean_51", "Rolling 51-row mean of the calibrated horizontal GR.", "geoanchor arms B, D"),
            ("acal_roll_std_51", "Rolling 51-row std of the calibrated horizontal GR.", "geoanchor arms B, D"),
            ("acal_gr_grad", "Local per-foot gradient of the calibrated horizontal GR log.", "geoanchor arms B, D"),
        )
    ),
    *tuple(
        _derived(
            name,
            "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)",
            notes,
            risk="none — datum scan scores GR versus Typewell GR only; no hidden label is readable",
            used_by=used_by,
        )
        for name, notes, used_by in (
            ("mb_shift1", "Best constant TVT datum shift of the GR/Typewell alignment scan (ft).", "geoanchor arms C, D"),
            ("mb_shift2", "Second-best separated datum branch (equals mb_shift1 when the scan is unimodal).", "geoanchor arms C, D"),
            ("mb_shift_hedged", "Trust-shrunk bimodal posterior-mean datum shift: w1*shift1 + (1-w1)*shift2.", "geoanchor arms C, D"),
            ("mb_sep", "Separation between the two branch minima (0 when unimodal).", "geoanchor arms C, D"),
            ("mb_bimodal", "One when the scan carries two plausible branch minima.", "geoanchor arms C, D"),
            ("mb_cost_gap", "Cost gap between the branch minima, normalised by the median scan cost.", "geoanchor arms C, D"),
            ("mb_w1", "Effective probability of branch 1, shrunk toward 0.5 by the prefix-trust diagnostic.", "geoanchor arms C, D"),
            ("mb_confidence", "Peak distinctiveness of the winning datum: 1 - J1/median(J), in [0, 1].", "geoanchor arms C, D"),
            ("mb_prefix_trust", "Prefix self-check of the scan: how close its best shift is to zero where the truth is visible.", "geoanchor arms C, D"),
            ("mb_ok", "One when the datum scan ran with sufficient measured GR and a usable typewell.", "geoanchor arms C, D"),
        )
    ),
    # Arm-E well-level gate design rows (well-level GBDT inputs). All are
    # target-free diagnostics of the candidate corrections, calibration and
    # prefix; candidate identity flags are one-hot indicators, not measured
    # data (provenanced to MD only as a harmless root).
    *tuple(
        _derived(
            name,
            parents,
            notes,
            risk="none — computed from InferenceTask only, per boundary, cross-fitted by well",
            used_by="geoanchor arm E gate",
        )
        for name, parents, notes in (
            ("gate_prefix_len", "MD, TVT_input (prefix only)", "Visible prefix length at the boundary."),
            ("gate_suffix_len", "MD", "Rows to predict past the boundary."),
            ("gate_gr_missing_suffix", "GR", "GR missing fraction in the prediction region."),
            ("gate_prefix_gr_missing", "GR", "GR missing fraction in the visible prefix."),
            ("gate_tvt_std_prefix", "TVT_input (prefix only)", "Std of the visible TVT_input prefix."),
            ("gate_tvt_range_prefix", "TVT_input (prefix only)", "Range of the visible TVT_input prefix."),
            ("gate_tvt_slope_300", "MD, TVT_input (prefix only)", "Least-squares prefix TVT slope over the last 300 ft."),
            ("gate_anchor", "TVT_input (prefix only)", "Last known TVT before the boundary."),
            ("gate_acal_alpha", "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)", "Affine calibration gain (prefix-only fit)."),
            ("gate_acal_beta", "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)", "Affine calibration offset normalised by Typewell GR std."),
            ("gate_acal_fit_rmse", "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)", "Prefix affine fit residual RMS in z units."),
            ("gate_acal_prefix_corr", "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)", "Prefix correlation of calibrated GR with Typewell GR."),
            ("gate_mb_shift1", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Winning datum shift of the GR scan."),
            ("gate_mb_sep", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Branch separation of the GR scan (0 when unimodal)."),
            ("gate_mb_cost_gap", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Normalised cost gap between scan branches."),
            ("gate_mb_confidence", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Datum-scan peak distinctiveness in [0, 1]."),
            ("gate_mb_bimodal", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "One when the scan is bimodal."),
            ("gate_mb_prefix_trust", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Prefix-trust diagnostic of the datum scan."),
            ("gate_pf_confidence", "GR, Typewell GR, Typewell TVT, MD, X, Y, Z, TVT_input (prefix only)", "Particle-filter mean confidence at the boundary."),
            ("gate_pf_spread", "GR, Typewell GR, Typewell TVT, MD, X, Y, Z, TVT_input (prefix only)", "Particle-filter mean branch spread."),
            ("gate_pf_fallback", "GR, Typewell GR, Typewell TVT, MD, X, Y, Z, TVT_input (prefix only)", "Fraction of prediction rows where the particle filter used its geometry fallback (GR outage)."),
            ("gate_beam_confidence", "GR, Typewell GR, Typewell TVT, MD, X, Y, Z, TVT_input (prefix only)", "Beam-search mean confidence at the boundary."),
            ("gate_beam_spread", "GR, Typewell GR, Typewell TVT, MD, X, Y, Z, TVT_input (prefix only)", "Beam-search mean branch spread."),
            ("gate_beam_fallback", "GR, Typewell GR, Typewell TVT, MD, X, Y, Z, TVT_input (prefix only)", "Fraction of prediction rows where beam search used its geometry fallback (GR outage)."),
            ("gate_track_disagreement", "GR, Typewell GR, Typewell TVT, MD, X, Y, Z, TVT_input (prefix only)", "Mean |PF track - Beam track| over the early prediction rows; GR-scan separation when only one family is available."),
            ("gate_cand_pf", "MD", "One-hot indicator: the PF candidate (identity flag, not a measurement)."),
            ("gate_cand_beam", "MD", "One-hot indicator: the Beam candidate (identity flag, not a measurement)."),
            ("gate_cand_mean", "MD", "One-hot indicator: the PF/Beam-mean candidate (identity flag, not a measurement)."),
        )
    ),
    # Trajectory-stack gate design rows (src/trajectory_stack.py). Same
    # provenance discipline as the arm-E gate rows: every entry is a
    # target-free diagnostic computed per boundary from the allowed roots;
    # the multi-scale rows summarise GR/Typewell-GR datum scans at fixed
    # half-ranges; the oof-skill scalars are fold-training OOF residual
    # quality of the cross-fitted booster analogues (a fixed fold-level
    # scalar at inference, never a validation-derived value).
    *tuple(
        _derived(
            name,
            parents,
            notes,
            risk="none — computed from InferenceTask only, per boundary, cross-fitted by well",
            used_by="trajectory stack gate / meta-stack",
        )
        for name, parents, notes in (
            ("gate_ms_ptp", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Peak-to-peak range of the datum-scan shift across the a-priori scan half-ranges {8, 15, 25} ft (multi-scale disagreement)."),
            ("gate_ms_dominant_shift", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Median datum-scan shift across the multi-scale half-ranges."),
            ("gate_ms_n_agree", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Number of scan scales agreeing with the dominant shift within 1.5 ft."),
            ("gate_ms_min_conf", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Minimum datum-scan confidence across the multi-scale half-ranges."),
            ("gate_lgbm_oof_skill", "MD, X, Y, Z, GR, TVT_input (prefix only), Typewell TVT, Typewell GR", "Fold-training inner-OOF residual RMSE skill of the cross-fitted LightGBM analogue vs the anchor, clipped to [0, 1]; a fixed fold-level scalar at inference."),
            ("gate_cat_oof_skill", "MD, X, Y, Z, GR, TVT_input (prefix only), Typewell TVT, Typewell GR", "Fold-training inner-OOF residual RMSE skill of the cross-fitted CatBoost analogue vs the anchor, clipped to [0, 1]; a fixed fold-level scalar at inference."),
            ("gate_cand_mb", "MD", "One-hot indicator: the multi-branch hedged candidate (identity flag, not a measurement)."),
            ("gate_cand_lgbm", "MD", "One-hot indicator: the LightGBM residual candidate (identity flag, not a measurement)."),
            ("gate_cand_cat", "MD", "One-hot indicator: the CatBoost residual candidate (identity flag, not a measurement)."),
            ("meta_res_ridge", "MD, X, Y, Z, GR, TVT_input (prefix only), Typewell TVT, Typewell GR", "OOF Ridge anchored-residual prediction feeding the meta-stack (cross-fitted by well; fold-fitted at inference)."),
            ("meta_res_lgbm", "MD, X, Y, Z, GR, TVT_input (prefix only), Typewell TVT, Typewell GR", "OOF LightGBM anchored-residual prediction feeding the meta-stack (cross-fitted by well; fold-fitted at inference)."),
            ("meta_res_cat", "MD, X, Y, Z, GR, TVT_input (prefix only), Typewell TVT, Typewell GR", "OOF CatBoost anchored-residual prediction feeding the meta-stack (cross-fitted by well; fold-fitted at inference)."),
            ("meta_dmd", "MD", "Feet drilled past the boundary for the stacked row (same construction as dmd, scoped to the meta-stack design)."),
            ("meta_log1p_dmd", "MD", "log(1 + dmd) for the stacked row (same construction as log1p_dmd, scoped to the meta-stack design)."),
        )
    ),
    # Alignment Stack v2 design rows (src/alignment_v2.py). All are
    # target-free diagnostics computed per boundary from the allowed
    # roots (MD, X, Y, Z, GR, Typewell TVT, Typewell GR, visible
    # TVT_input prefix). The candidate identity / availability flags
    # are one-hot indicators and are not measured data (provenanced to
    # MD only as a harmless root).
    *tuple(
        _derived(
            name,
            parents,
            notes,
            risk="none — computed from InferenceTask only, per boundary, cross-fitted by well",
            used_by="alignment v2 gate / OOF meta-stack v2",
        )
        for name, parents, notes in (
            ("align_v2_cal_alpha_ptp", "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)", "Peak-to-peak of the affine alpha across the multi-scale min_prefix_rows grid {40, 80, 160}."),
            ("align_v2_cal_beta_ptp", "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)", "Peak-to-peak of the affine beta across the multi-scale min_prefix_rows grid."),
            ("align_v2_cal_fit_rmse_z", "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)", "Dominant (largest-scale) prefix affine fit residual RMS in z units."),
            ("align_v2_cal_prefix_corr", "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)", "Prefix correlation of calibrated GR with Typewell GR from the dominant affine scale."),
            ("align_v2_cal_confidence", "GR, TVT_input (prefix only)", "GR-missingness-aware affine calibration confidence, shrunk by suffix GR coverage."),
            ("align_v2_cal_n_ok", "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)", "Number of affine scales that passed their sanity bounds (0..3)."),
            ("align_v2_cal_n_agree", "GR, Typewell GR, Typewell TVT, TVT_input (prefix only)", "Number of successful scales whose alpha agrees with the dominant within 2.0."),
            ("align_v2_ms_dominant_shift", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Median datum-scan shift across the multi-scale half-ranges {12, 35, 100} ft."),
            ("align_v2_ms_ptp", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Peak-to-peak of the per-scale dominant shifts; multi-scale branch disagreement."),
            ("align_v2_ms_min_conf", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Minimum datum-scan confidence across the multi-scale half-ranges."),
            ("align_v2_ms_n_agree", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Number of scan scales agreeing with the dominant shift within the agreement radius."),
            ("align_v2_ms_n_ok", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Number of multi-scale scan scales that produced a usable shift (0..3)."),
            ("align_v2_ms_confidence", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Aggregate multi-scale confidence in [0, 1]."),
            ("align_v2_ms_prefix_trust", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Average prefix-trust diagnostic across the multi-scale scans."),
            ("align_v2_dp_ok", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "One when the bounded-curvature dynamic-programming path matcher succeeded."),
            ("align_v2_dp_prefix_mismatch", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Mean absolute visible-prefix mismatch of the DP path against the known TVT."),
            ("align_v2_dp_gr_misfit", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Clipped GR misfit of the DP path against the Typewell GR on the predicted region."),
            ("align_v2_dp_smoothness", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Standard deviation of the per-row finite differences of the DP path."),
            ("align_v2_dp_confidence", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Bounded confidence in [0, 1] combining prefix mismatch and GR misfit."),
            ("align_v2_ens_n_available", "MD", "Number of available, non-fallback branches in the v2 ensemble (candidate identity / count flag)."),
            ("align_v2_ens_n_fallback", "MD", "Number of branches that fell back to the anchor (candidate identity / count flag)."),
            ("align_v2_ens_branch_disagreement", "MD", "Mean per-row std of the available branch corrections; the ensemble's internal disagreement."),
            ("align_v2_ens_mean_correction_abs", "MD", "Mean absolute correction magnitude of the available branches."),
            ("align_v2_ens_max_correction_abs", "MD", "Max absolute correction magnitude across the available branches."),
            ("align_v2_ens_confidence", "MD", "Mean confidence across the available branches, in [0, 1]."),
            ("align_v2_proj_ok", "MD, Z, TVT_input (prefix only)", "One when the robust stratigraphic projection succeeded and was applied."),
            ("align_v2_proj_movement", "MD, Z, TVT_input (prefix only)", "Mean absolute movement of the projected path from the input candidate path."),
            ("align_v2_proj_n_clipped", "MD, Z, TVT_input (prefix only)", "Number of rows where the projection's movement was clipped to the cap."),
            ("align_v2_proj_clip_fraction", "MD, Z, TVT_input (prefix only)", "Fraction of predicted rows where the projection's movement was clipped."),
            ("align_v2_proj_visible_prefix_mismatch", "MD, Z, TVT_input (prefix only)", "Visible-prefix verification of the projected path (0 on a well-formed fit)."),
        )
    ),
    # OOF meta-stack v2 design rows (src.alignment_v2_model.py). These
    # are summary scalars tiled across the predicted rows; each
    # comes from the allowed roots only and is never the well's
    # hidden TVT label. The candidate-correction means (multi_scale,
    # dp_path, irls, branch_hedged) are mean corrections of the
    # candidate paths, expressed in ft of TVT.
    *tuple(
        _derived(
            name,
            parents,
            notes,
            risk="none — summary scalar per boundary, never a per-row label",
            used_by="OOF meta-stack v2",
        )
        for name, parents, notes in (
            ("v2_dmd", "MD", "Feet drilled past the boundary (same as dmd, scoped to the v2 meta-stack design)."),
            ("v2_log1p_dmd", "MD", "log(1 + dmd) (same as log1p_dmd, scoped to the v2 meta-stack design)."),
            ("v2_corr_multi_scale", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Mean correction of the multi-scale alignment candidate relative to the Ridge anchor."),
            ("v2_corr_dp", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Mean correction of the dynamic-programming path candidate relative to the Ridge anchor."),
            ("v2_corr_irls", "GR, Typewell GR, Typewell TVT, MD, Z, TVT_input (prefix only)", "Mean correction of the robust IRLS stratigraphic projection candidate relative to the Ridge anchor."),
            ("v2_corr_branch_hedged", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Mean correction of the branch-hedged ensemble candidate relative to the Ridge anchor."),
            ("v2_disagreement", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Branch disagreement of the v2 ensemble (mean per-row std of the available branches)."),
            ("v2_confidence", "GR, Typewell GR, Typewell TVT, MD, TVT_input (prefix only)", "Aggregate v2 ensemble confidence in [0, 1]."),
            ("v2_gr_miss_suffix", "GR", "GR missing fraction in the prediction region (v2 meta-stack design)."),
            ("v2_gr_miss_prefix", "GR", "GR missing fraction in the visible prefix (v2 meta-stack design)."),
            ("v2_suffix_len", "MD", "Rows to predict past the boundary (v2 meta-stack design)."),
            ("v2_prefix_len", "MD, TVT_input (prefix only)", "Visible prefix length at the boundary (v2 meta-stack design)."),
        )
    ),
    # One-hot candidate identity flags for the v2 two-stage gate (the
    # gate's GBDT needs to know which candidate each row is about).
    # These are identity flags, not measured data; their manifest root
    # is MD only as a harmless anchor (no actual measurement comes
    # from MD).
    *tuple(
        _derived(
            name,
            "MD",
            "One-hot candidate identity flag for the v2 two-stage gate (identity, not a measurement).",
            risk="none — identity flag, never a measured data column",
            used_by="alignment v2 gate",
        )
        for name in (
            "v2_cand_multi_scale",
            "v2_cand_dp_path",
            "v2_cand_irls",
            "v2_cand_branch_hedged",
        )
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
        parents=("X", "Y"),
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
# Manifest self-validation
# --------------------------------------------------------------------------

def validate_manifest(manifest: "tuple[FeatureSpec, ...] | None" = None) -> list[str]:
    """Check the manifest against its own invariants. Returns problem strings.

    Invariants:

    1. No feature is marked usable at inference while being absent from test.
       This is the exact defect the Kaggle schema audit found in the
       ``Typewell Geology`` row.
    2. A train-only feature (``in train``, not ``in test``) carries a
       non-inference decision and ``safe_for_inference`` = no.
    3. No derived or spatial feature descends from a parent that is itself
       barred from inference — a safe-looking child cannot launder a
       train-only parent.
    4. Every parent named in a provenance string exists in the manifest.
    5. ``safe_for_inference`` never contradicts ``decision``.

    ``assert_manifest_valid`` raises on a non-empty result; both
    ``manifest_frame`` and ``assert_safe_features`` call it, so the document
    cannot be rendered or enforced while inconsistent.
    """
    entries = MANIFEST if manifest is None else tuple(manifest)
    by_name = {spec.feature_name: spec for spec in entries}
    problems: list[str] = []

    for spec in entries:
        # 1 + 5 — decision vs. availability / safety
        if spec.claims_inference_safe and not spec.in_test:
            problems.append(
                f"{spec.feature_name}: decision={spec.decision} but "
                f"test_availability={spec.test_availability!r} — a train-only "
                "feature is marked available in test."
            )
        if spec.claims_inference_safe and _is_no(spec.safe_for_inference):
            problems.append(
                f"{spec.feature_name}: decision={spec.decision} contradicts "
                f"safe_for_inference={spec.safe_for_inference!r}."
            )
        # 2 — train-only columns must be barred and labelled unsafe
        if spec.train_only:
            if spec.decision in INFERENCE_DECISIONS:
                problems.append(
                    f"{spec.feature_name}: train-only (train yes / test no) but "
                    f"decision={spec.decision}."
                )
            if _is_yes(spec.safe_for_inference):
                problems.append(
                    f"{spec.feature_name}: train-only but safe_for_inference="
                    f"{spec.safe_for_inference!r}."
                )
        # 3 + 4 — provenance
        for parent in spec.parents:
            pspec = by_name.get(parent)
            if pspec is None:
                problems.append(
                    f"{spec.feature_name}: provenance names {parent!r}, which "
                    "is not a manifest entry."
                )
                continue
            if spec.claims_inference_safe and not pspec.in_test:
                problems.append(
                    f"{spec.feature_name}: decision={spec.decision} but its "
                    f"parent {parent!r} is absent from test "
                    f"(test_availability={pspec.test_availability!r}). A "
                    "derived feature cannot be safer than its inputs."
                )
            if spec.claims_inference_safe and pspec.decision in {
                "TARGET", "REJECT", "TRAIN_ANALYSIS_ONLY"
            }:
                problems.append(
                    f"{spec.feature_name}: decision={spec.decision} but its "
                    f"parent {parent!r} carries decision={pspec.decision}."
                )
    return problems


def assert_manifest_valid(manifest: "tuple[FeatureSpec, ...] | None" = None) -> None:
    """Raise ``ManifestInconsistency`` if the manifest violates its invariants."""
    problems = validate_manifest(manifest)
    if problems:
        raise ManifestInconsistency(
            "Feature manifest is internally inconsistent; refusing to "
            "proceed.\n  - " + "\n  - ".join(problems)
        )


# Fail at import time rather than mid-run.
assert_manifest_valid()


# --------------------------------------------------------------------------
# Rendering + enforcement
# --------------------------------------------------------------------------

def manifest_frame() -> pd.DataFrame:
    """Render the manifest, refusing to emit an inconsistent document."""
    assert_manifest_valid()
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
    return _names(set(INFERENCE_DECISIONS))


def prefix_only_features() -> list[str]:
    return _names({"USE_PREFIX_ONLY"})


def rejected_features() -> list[str]:
    """Names that must never appear in a model input matrix."""
    return _names(set(NON_INFERENCE_DECISIONS))


def train_only_features() -> list[str]:
    """Columns present in train and absent from test.

    These are admissible for train-side EDA and error analysis only. The list
    is derived from the availability fields, not hardcoded, so it tracks the
    manifest automatically.
    """
    return [f.feature_name for f in MANIFEST if f.train_only]


def train_analysis_only_features() -> list[str]:
    """Train-only columns explicitly retained for analysis (not inference)."""
    return _names({"TRAIN_ANALYSIS_ONLY"})


def target_features() -> list[str]:
    return _names({"TARGET"})


class FeatureLeakage(RuntimeError):
    """Raised when a rejected or target-derived column reaches a model."""


def assert_safe_features(columns, *, context: str = "model input") -> None:
    """Raise ``FeatureLeakage`` if any column is not cleared for inference.

    Unknown columns are rejected too: the manifest is a whitelist, so a feature
    nobody has audited cannot reach a model by accident. The manifest itself is
    validated first, so enforcement can never be carried out against an
    inconsistent document (e.g. one that marks a train-only column as usable).
    """
    assert_manifest_valid()
    allowed = set(safe_inference_features())
    banned = {c.lower(): c for c in rejected_features()}
    train_only = {c.lower() for c in train_only_features()}
    unknown: list[str] = []
    leaks: list[str] = []
    train_only_leaks: list[str] = []
    for col in columns:
        name = canonical(col)
        if name in allowed:
            continue
        if name.lower() in train_only:
            train_only_leaks.append(
                f"{col!r} -> {name} (decision={_decision_of(name)}, "
                "train-only: absent from test)"
            )
        elif name.lower() in banned:
            leaks.append(f"{col!r} -> {name} (decision={_decision_of(name)})")
        else:
            unknown.append(str(col))
    problems = []
    if train_only_leaks:
        problems.append(
            "TRAIN-ONLY columns (absent from the test schema, so they cannot "
            "exist at inference): " + ", ".join(sorted(train_only_leaks))
        )
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

def canonical_typewell(column: str) -> str:
    """Resolve a *typewell* column header to its manifest feature name.

    Typewell entries are namespaced (``Typewell TVT``, ``Typewell GR``,
    ``Typewell Geology``) because the same headers appear in the horizontal
    files with a different meaning: a typewell's ``TVT`` is its own
    stratigraphic axis, not the prediction target.

    Naively prefixing ``canonical(c)`` was a real defect: ``canonical("Geology")``
    already returns ``"Typewell Geology"``, so the prefix produced
    ``"Typewell Typewell Geology"`` and the column was scored as absent from
    *both* splits. That masked the true train/test asymmetry behind a
    both-sides-missing mismatch.
    """
    name = canonical(column)
    if name.startswith("Typewell "):
        return name
    return f"Typewell {name}"


def observed_schema(
    train_columns,
    test_columns,
    *,
    train_tw_columns=None,
    test_tw_columns=None,
    tw_columns=None,
) -> tuple[set[str], set[str]]:
    """Canonical feature-name sets observed in the train and test schemas."""
    train = {canonical(c) for c in train_columns}
    test = {canonical(c) for c in test_columns}
    # Typewell schemas are observed independently per split: reusing the train
    # schema for both sides is exactly how a train-only Geology column came to
    # be recorded as present in test.
    if train_tw_columns is None and test_tw_columns is None:
        train_tw_columns = test_tw_columns = tw_columns
    if train_tw_columns is not None:
        train |= {canonical_typewell(c) for c in train_tw_columns}
    if test_tw_columns is not None:
        test |= {canonical_typewell(c) for c in test_tw_columns}
    return train, test


def verify_manifest_against_data(
    train_columns, test_columns, *, tw_columns=None,
    train_tw_columns=None, test_tw_columns=None
) -> pd.DataFrame:
    """Re-check the manifest's availability claims against observed columns.

    Returns one row per raw manifest entry with ``claim``, ``observed``,
    ``agrees`` and — critically — ``train_only_but_marked_available``, which is
    the specific failure the Kaggle schema audit caught: a column observed only
    in train while the manifest advertises it as usable at inference.

    Any ``agrees == False``, or any ``train_only_but_marked_available == True``,
    means the encoded audit findings no longer describe the data and modelling
    must stop.
    """
    train, test = observed_schema(
        train_columns,
        test_columns,
        train_tw_columns=train_tw_columns,
        test_tw_columns=test_tw_columns,
        tw_columns=tw_columns,
    )

    rows = []
    for spec in MANIFEST:
        if spec.tier != "raw":
            continue
        claim_train = spec.in_train
        claim_test = spec.in_test
        obs_train = spec.feature_name in train
        obs_test = spec.feature_name in test
        observed_train_only = bool(obs_train and not obs_test)
        rows.append(
            {
                "feature_name": spec.feature_name,
                "claim_in_train": claim_train,
                "observed_in_train": obs_train,
                "claim_in_test": claim_test,
                "observed_in_test": obs_test,
                "agrees": bool(claim_train == obs_train and claim_test == obs_test),
                "decision": spec.decision,
                "observed_train_only": observed_train_only,
                # The headline safety check: a column that physically exists
                # only in train must not be cleared for the inference matrix.
                "train_only_but_marked_available": bool(
                    observed_train_only
                    and (spec.claims_inference_safe or claim_test)
                ),
                "safe_for_inference": spec.safe_for_inference,
            }
        )
    return pd.DataFrame(rows)


class SchemaVerificationError(RuntimeError):
    """Raised when observed schemas contradict the manifest's claims."""


def assert_manifest_matches_data(
    train_columns, test_columns, *, tw_columns=None,
    train_tw_columns=None, test_tw_columns=None,
) -> pd.DataFrame:
    """Verify the manifest against observed schemas, raising on any mismatch.

    This is the fail-loud entry point the validation runner calls before any
    model is fitted. It raises on:

    * an availability claim that disagrees with the observed columns, and
    * any column observed in train but not test while marked available in
      test or cleared for inference.

    Returns the verification frame when everything agrees.
    """
    assert_manifest_valid()
    frame = verify_manifest_against_data(
        train_columns,
        test_columns,
        tw_columns=tw_columns,
        train_tw_columns=train_tw_columns,
        test_tw_columns=test_tw_columns,
    )
    problems: list[str] = []

    flagged = frame[frame["train_only_but_marked_available"]]
    for _, row in flagged.iterrows():
        problems.append(
            f"{row['feature_name']}: observed in TRAIN only, but the manifest "
            f"reports test_availability={row['claim_in_test']} / "
            f"decision={row['decision']}. A train-only feature must be "
            "TRAIN_ANALYSIS_ONLY or REJECT and never enter the test matrix."
        )

    disagreements = frame[(~frame["agrees"]) & (~frame["train_only_but_marked_available"])]
    for _, row in disagreements.iterrows():
        problems.append(
            f"{row['feature_name']}: manifest claims train="
            f"{row['claim_in_train']}/test={row['claim_in_test']}, observed "
            f"train={row['observed_in_train']}/test={row['observed_in_test']}."
        )

    if problems:
        raise SchemaVerificationError(
            "Feature manifest disagrees with the observed train/test "
            "schemas; refusing to proceed to modelling.\n  - "
            + "\n  - ".join(problems)
        )
    return frame


def assert_audited_schemas() -> pd.DataFrame:
    """Verify the manifest against the recorded Kaggle schema audit.

    Runs without a data mount, so the invariant is checkable in CI and in the
    test suite: the manifest must agree with

        train typewell: ['TVT', 'GR', 'Geology']
        test  typewell: ['TVT', 'GR']
    """
    return assert_manifest_matches_data(
        AUDITED_TRAIN_HW_COLUMNS,
        AUDITED_TEST_HW_COLUMNS,
        train_tw_columns=AUDITED_TRAIN_TYPEWELL_COLUMNS,
        test_tw_columns=AUDITED_TEST_TYPEWELL_COLUMNS,
    )


# --------------------------------------------------------------------------
# Provenance: which raw columns does the inference matrix ultimately rest on?
# --------------------------------------------------------------------------

def root_sources(feature_name: str, _seen: "set[str] | None" = None) -> set[str]:
    """Raw (tier-1) columns a feature ultimately derives from.

    Walks the ``parents`` graph to its roots, so ``align_shift`` resolves to
    ``{GR, Typewell GR, Typewell TVT, MD, TVT_input}`` rather than to its
    immediate parents. Cycles are impossible in a well-formed manifest but are
    guarded against anyway.
    """
    _seen = set() if _seen is None else _seen
    if feature_name in _seen:
        return set()
    _seen.add(feature_name)
    spec = next((s for s in MANIFEST if s.feature_name == feature_name), None)
    if spec is None:
        return {feature_name}
    if not spec.parents:
        return {spec.feature_name}
    roots: set[str] = set()
    for parent in spec.parents:
        roots |= root_sources(parent, _seen)
    return roots


def inference_feature_provenance() -> dict[str, set[str]]:
    """``{feature -> root raw columns}`` for every inference-cleared feature."""
    return {name: root_sources(name) for name in safe_inference_features()}


def assert_inference_provenance() -> dict[str, set[str]]:
    """Prove the inference feature set rests only on test-available columns.

    Requirement: the final inference matrix may contain only MD, X, Y, Z, GR,
    the visible-prefix part of TVT_input, Typewell TVT, Typewell GR, and safe
    features derived from those. This walks each cleared feature back to its
    raw roots and rejects anything outside that set — which is what keeps
    Typewell Geology and the formation markers out transitively, not just by
    name.
    """
    allowed = set(SAFE_RAW_INFERENCE_SOURCES)
    provenance = inference_feature_provenance()
    problems: list[str] = []
    for name, roots in sorted(provenance.items()):
        illegal = sorted(roots - allowed)
        if illegal:
            problems.append(
                f"{name}: derives from {illegal}, which are not permitted "
                "sources for an inference feature."
            )
    if problems:
        raise ManifestInconsistency(
            "Inference feature provenance check failed.\n  - "
            + "\n  - ".join(problems)
            + f"\n\nPermitted raw sources: {sorted(allowed)}"
        )
    return provenance


def assert_inference_matrix(columns, *, context: str = "final inference matrix") -> None:
    """Full safety gate for a matrix that will be used at inference time.

    Stricter than ``assert_safe_features``: besides the whitelist check it
    re-verifies the manifest against the audited Kaggle schemas and proves
    every cleared feature's provenance traces back only to columns that exist
    in test. A matrix is cleared only when the document it is checked against
    still matches reality.
    """
    assert_audited_schemas()
    assert_inference_provenance()
    assert_safe_features(columns, context=context)
