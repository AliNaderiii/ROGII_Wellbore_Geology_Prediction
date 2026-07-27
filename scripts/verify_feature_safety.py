"""Feature-safety verification — the gate that must pass before training.

    python scripts/verify_feature_safety.py              # audited schemas only
    python scripts/verify_feature_safety.py --mount      # + the live mount
    python scripts/verify_feature_safety.py --json

Answers one question with evidence rather than assertion:

    Can any column that does not exist in the test split reach the final
    inference feature matrix?

It checks, in order:

1. **Manifest self-consistency** — no entry is marked usable at inference while
   being absent from test; no train-only entry claims to be inference-safe.
2. **Schema agreement** — the manifest's availability claims match the audited
   Kaggle schemas (and, with ``--mount``, the columns actually on disk).
3. **Provenance** — every inference-cleared feature traces back only to
   ``MD, X, Y, Z, GR, TVT_input (visible prefix), Typewell TVT, Typewell GR``.
4. **Exclusions** — ``Typewell Geology`` and the six formation markers are
   absent from the inference feature set, under every spelling the loader
   recognises.
5. **The real feature matrix** — the columns ``src.features`` actually builds
   are run through ``assert_inference_matrix``.

Exits non-zero on any failure. Trains nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:  # loose-file execution
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap
bootstrap()

from src.manifest import (  # noqa: E402
    AUDITED_TEST_TYPEWELL_COLUMNS,
    AUDITED_TRAIN_TYPEWELL_COLUMNS,
    SAFE_RAW_INFERENCE_SOURCES,
    TRAIN_ONLY_MARKERS,
    FeatureLeakage,
    ManifestInconsistency,
    SchemaVerificationError,
    assert_audited_schemas,
    assert_inference_provenance,
    assert_manifest_matches_data,
    assert_manifest_valid,
    assert_safe_features,
    canonical,
    inference_feature_provenance,
    manifest_frame,
    safe_inference_features,
    train_only_features,
    verify_manifest_against_data,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return passed


# --------------------------------------------------------------------------


def check_manifest_self_consistency() -> None:
    print("== 1. manifest self-consistency ==")
    try:
        assert_manifest_valid()
        check("manifest invariants hold", True,
              "no train-only feature is marked available in test")
    except ManifestInconsistency as exc:
        check("manifest invariants hold", False, str(exc))


def check_audited_schemas() -> None:
    print("\n== 2. schema agreement (audited Kaggle schemas) ==")
    print(f"    train typewell columns: {list(AUDITED_TRAIN_TYPEWELL_COLUMNS)}")
    print(f"    test  typewell columns: {list(AUDITED_TEST_TYPEWELL_COLUMNS)}")
    try:
        frame = assert_audited_schemas()
        check("manifest agrees with the audited schemas", True,
              f"{len(frame)} raw features verified")
    except (SchemaVerificationError, ManifestInconsistency) as exc:
        check("manifest agrees with the audited schemas", False, str(exc))
        return

    # The specific defect the audit caught, asserted explicitly.
    row = frame.set_index("feature_name").loc["Typewell Geology"]
    check(
        "Typewell Geology recorded as train-only",
        bool(row["observed_in_train"]) and not bool(row["observed_in_test"]),
        f"observed train={row['observed_in_train']}, test={row['observed_in_test']}, "
        f"decision={row['decision']}",
    )
    check(
        "no train-only feature is marked available in test",
        not bool(frame["train_only_but_marked_available"].any()),
        "checked " + ", ".join(
            frame.loc[frame["observed_train_only"], "feature_name"].astype(str)
        ),
    )


def check_provenance() -> None:
    print("\n== 3. inference feature provenance ==")
    try:
        provenance = assert_inference_provenance()
        roots = sorted({r for rs in provenance.values() for r in rs})
        check("every inference feature derives only from test-available columns",
              True, f"roots: {roots}")
    except ManifestInconsistency as exc:
        check("every inference feature derives only from test-available columns",
              False, str(exc))
        return
    check(
        "permitted raw sources are exactly the approved eight",
        set(roots) <= set(SAFE_RAW_INFERENCE_SOURCES),
        f"allowed: {list(SAFE_RAW_INFERENCE_SOURCES)}",
    )


def check_exclusions() -> None:
    print("\n== 4. explicit exclusions ==")
    safe = set(safe_inference_features())

    check("Typewell Geology absent from the inference feature set",
          "Typewell Geology" not in safe)

    spellings = ["Geology", "geology", "GEOLOGY", "Typewell Geology",
                 "typewell_geology", "tw_geology", "Formation", "facies"]
    rejected = []
    for spelling in spellings:
        try:
            assert_safe_features([spelling], context="exclusion probe")
        except FeatureLeakage:
            rejected.append(spelling)
    check("every Geology spelling is rejected by the feature gate",
          len(rejected) == len(spellings),
          f"{len(rejected)}/{len(spellings)} rejected "
          f"({[s for s in spellings if s not in rejected]} slipped through)"
          if len(rejected) != len(spellings) else f"{spellings}")

    marker_leaks = [m for m in TRAIN_ONLY_MARKERS if m in safe]
    check("formation markers absent from the inference feature set",
          not marker_leaks,
          f"markers: {list(TRAIN_ONLY_MARKERS)}" if not marker_leaks
          else f"LEAKED: {marker_leaks}")

    marker_rejected = []
    for marker in TRAIN_ONLY_MARKERS:
        try:
            assert_safe_features([marker], context="exclusion probe")
        except FeatureLeakage:
            marker_rejected.append(marker)
    check("every formation marker is rejected by the feature gate",
          len(marker_rejected) == len(TRAIN_ONLY_MARKERS),
          f"{len(marker_rejected)}/{len(TRAIN_ONLY_MARKERS)} rejected")

    for name in ("TVT", "TVT_input"):
        try:
            assert_safe_features([name], context="exclusion probe")
            ok = False
        except FeatureLeakage:
            ok = True
        check(f"{name} is rejected as a row feature", ok)


def check_real_feature_matrix() -> None:
    print("\n== 5. the feature matrix the code actually builds ==")
    from src.features import FEATURE_COLUMNS
    from src.manifest import assert_inference_matrix
    from src.spatial import SPATIAL_COLUMNS

    try:
        assert_inference_matrix(FEATURE_COLUMNS, context="src.features.FEATURE_COLUMNS")
        check("src.features.FEATURE_COLUMNS passes the inference gate", True,
              f"{len(FEATURE_COLUMNS)} columns")
    except (FeatureLeakage, ManifestInconsistency, SchemaVerificationError) as exc:
        check("src.features.FEATURE_COLUMNS passes the inference gate", False, str(exc))

    combined = list(FEATURE_COLUMNS) + list(SPATIAL_COLUMNS)
    try:
        assert_inference_matrix(combined, context="features + spatial")
        check("feature matrix with spatial columns passes the inference gate",
              True, f"{len(combined)} columns")
    except (FeatureLeakage, ManifestInconsistency, SchemaVerificationError) as exc:
        check("feature matrix with spatial columns passes the inference gate",
              False, str(exc))

    canon = {canonical(c) for c in combined}
    check("Typewell Geology not present in the built matrix",
          "Typewell Geology" not in canon)
    check("no formation marker present in the built matrix",
          not (canon & set(TRAIN_ONLY_MARKERS)))


def check_live_mount() -> None:
    print("\n== 6. live mount verification ==")
    from src.data import discover_wells, load_well
    from src.paths import require_competition_data

    require_competition_data()
    train_ids = sorted(discover_wells("train"))
    test_ids = sorted(discover_wells("test"))
    if not train_ids or not test_ids:
        check("probe wells discovered", False,
              f"{len(train_ids)} train, {len(test_ids)} test")
        return

    tr = load_well(train_ids[0], "train")
    te = load_well(test_ids[0], "test")
    train_tw = list(tr.tw.columns) if tr.tw is not None else None
    test_tw = list(te.tw.columns) if te.tw is not None else None
    print(f"    train typewell columns ({tr.well_id}): {train_tw}")
    print(f"    test  typewell columns ({te.well_id}): {test_tw}")

    try:
        frame = assert_manifest_matches_data(
            tr.hw.columns, te.hw.columns,
            train_tw_columns=train_tw, test_tw_columns=test_tw,
        )
        check("manifest agrees with the mounted schemas", True,
              f"{len(frame)} raw features verified")
        observed_train_only = frame.loc[
            frame["observed_train_only"], "feature_name"
        ].astype(str).tolist()
        print(f"    train-only on this mount: {observed_train_only}")
    except (SchemaVerificationError, ManifestInconsistency) as exc:
        check("manifest agrees with the mounted schemas", False, str(exc))


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mount", action="store_true",
                    help="additionally verify against the mounted competition data")
    ap.add_argument("--json", action="store_true",
                    help="emit a machine-readable summary on stdout")
    args = ap.parse_args(argv)

    print("=" * 72)
    print("FEATURE SAFETY VERIFICATION")
    print("=" * 72)

    check_manifest_self_consistency()
    check_audited_schemas()
    check_provenance()
    check_exclusions()
    check_real_feature_matrix()
    if args.mount:
        check_live_mount()

    print("\n" + "-" * 72)
    print("Final safe inference feature list:")
    provenance = inference_feature_provenance()
    for name in safe_inference_features():
        print(f"  {name:<26} <- {', '.join(sorted(provenance[name]))}")
    print("-" * 72)
    print(f"Train-only (never at inference): {train_only_features()}")

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 72)
    print(f"FEATURE SAFETY: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
    print("No model was trained.")

    if args.json:
        print(json.dumps({
            "passed": not failed,
            "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in RESULTS],
            "safe_inference_features": safe_inference_features(),
            "train_only_features": train_only_features(),
            "train_typewell_columns": list(AUDITED_TRAIN_TYPEWELL_COLUMNS),
            "test_typewell_columns": list(AUDITED_TEST_TYPEWELL_COLUMNS),
        }, indent=2))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
