"""Promotion status registry for candidate models.

A model is only allowed into a final predictor or an ensemble branch if this
registry says so.  The registry is evidence-bound: every ``REJECTED`` entry
records the completed validation run that produced the rejection, per protocol,
so the decision can be re-checked rather than remembered.

The recorded numbers are *validation diagnostics computed after prediction*.
They are not features and nothing in ``src.features`` or ``src.baselines``
reads this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Declared locally rather than imported from ``src.validation`` so this module
# stays dependency-free and importable from any layer (including a model
# module) without a cycle.  ``tests`` assert the two spellings agree.
PROTOCOL_A = "same_well_masked"
PROTOCOL_B = "unseen_well"

#: Registry statuses.  ``CANDIDATE`` is the default for anything unlisted:
#: absence of evidence is never promotion.
APPROVED = "APPROVED"
CANDIDATE = "CANDIDATE"
REJECTED = "REJECTED"


@dataclass(frozen=True)
class ProtocolEvidence:
    """One protocol's completed-run comparison against the Ridge baseline."""

    protocol: str
    n_wells: int
    ridge_global_rmse: float
    model_global_rmse: float
    mean_confidence: float
    fallback_fraction: float

    @property
    def delta_vs_ridge(self) -> float:
        """Positive means the candidate is worse than Ridge."""
        return self.model_global_rmse - self.ridge_global_rmse

    @property
    def worse_than_ridge(self) -> bool:
        return self.delta_vs_ridge > 0.0


@dataclass(frozen=True)
class ModelStatus:
    """A promotion decision plus the evidence that forced it."""

    model: str
    status: str
    reason: str
    source_run: str
    evidence: tuple[ProtocolEvidence, ...] = field(default_factory=tuple)

    @property
    def is_rejected(self) -> bool:
        return self.status == REJECTED

    def evidence_for(self, protocol: str) -> ProtocolEvidence:
        for item in self.evidence:
            if item.protocol == protocol:
                return item
        raise KeyError(f"{self.model}: no recorded evidence for protocol {protocol!r}")


#: Completed real validation run of the isolated dip-constrained A/B.
#:
#: 770 eligible wells, both protocols, cross-fitted by well ID.  The direct
#: alignment trajectory is decisively worse than Ridge under both protocols, so
#: it is rejected as a final predictor and as an ensemble branch.  The Ridge
#: baseline itself is unchanged by this decision.
DIP_ALIGNMENT_RUN = "real validation, 770 eligible wells, both protocols, cross-fitted by well ID"

MODEL_STATUS: dict[str, ModelStatus] = {
    "dip_constrained_alignment": ModelStatus(
        model="dip_constrained_alignment",
        status=REJECTED,
        reason=(
            "Direct dip-constrained GR/typewell alignment is decisively worse than the "
            "Ridge baseline under both validation protocols (+248.202 RMSE on "
            "same_well_masked, +82.104 RMSE on unseen_well). Mean alignment confidence is "
            "low (0.158 / 0.320) and the fallback path dominates same_well_masked (67.5% of "
            "predicted rows). It must not be used as a final predictor or as an ensemble "
            "branch in its current form."
        ),
        source_run=DIP_ALIGNMENT_RUN,
        evidence=(
            ProtocolEvidence(
                protocol=PROTOCOL_A,
                n_wells=770,
                ridge_global_rmse=29.452,
                model_global_rmse=277.654,
                mean_confidence=0.1577,
                fallback_fraction=0.675,
            ),
            ProtocolEvidence(
                protocol=PROTOCOL_B,
                n_wells=770,
                ridge_global_rmse=14.441,
                model_global_rmse=96.545,
                mean_confidence=0.3197,
                fallback_fraction=0.340,
            ),
        ),
    ),
}


class RejectedModelError(RuntimeError):
    """Raised when a rejected model is routed into a final/ensemble path."""


def status_of(model: str) -> ModelStatus:
    """Status for ``model``; unlisted models are ``CANDIDATE``, never approved."""
    known = MODEL_STATUS.get(model)
    if known is not None:
        return known
    return ModelStatus(
        model=model,
        status=CANDIDATE,
        reason="No promotion decision has been recorded for this model.",
        source_run="",
    )


def rejected_models() -> list[str]:
    return sorted(name for name, s in MODEL_STATUS.items() if s.is_rejected)


def is_rejected(model: str) -> bool:
    return status_of(model).is_rejected


def assert_not_rejected(models, *, context: str) -> None:
    """Fail loudly if any rejected model reaches a final or ensemble path.

    ``models`` may be model names or objects carrying a ``name`` attribute.
    """
    names = []
    for item in models:
        names.append(str(getattr(item, "name", item)))
    bad = [n for n in names if is_rejected(n)]
    if bad:
        detail = "; ".join(f"{n}: {status_of(n).reason}" for n in sorted(set(bad)))
        raise RejectedModelError(
            f"{context}: rejected model(s) {sorted(set(bad))} may not be promoted. {detail}"
        )


def status_table() -> list[dict]:
    """Flat rows for the report layer; one row per (model, protocol) evidence."""
    rows = []
    for name in sorted(MODEL_STATUS):
        s = MODEL_STATUS[name]
        if not s.evidence:
            rows.append({"model": name, "status": s.status, "protocol": "", "source_run": s.source_run})
            continue
        for e in s.evidence:
            rows.append(
                {
                    "model": name,
                    "status": s.status,
                    "protocol": e.protocol,
                    "n_wells": e.n_wells,
                    "ridge_global_rmse": e.ridge_global_rmse,
                    "model_global_rmse": e.model_global_rmse,
                    "delta_vs_ridge": e.delta_vs_ridge,
                    "mean_confidence": e.mean_confidence,
                    "fallback_fraction": e.fallback_fraction,
                    "source_run": e.protocol and s.source_run,
                }
            )
    return rows
