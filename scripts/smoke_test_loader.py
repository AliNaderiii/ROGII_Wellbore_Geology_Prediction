"""Smoke tests against the real mounted competition data.

    python scripts/smoke_test_loader.py
    python scripts/smoke_test_loader.py --expect-train 773 --expect-test 3

Runs the 12 required checks. Loads only — never trains, never writes a model.
Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap
bootstrap()

import numpy as np
import pandas as pd

from src.data import (
    discover_wells,
    load_well,
    summarize_well,
    validate_split,
    well_metadata,
)
from src.paths import SAMPLE_SUBMISSION, available, describe_paths, require_competition_data
from src.submission import audit_sample_submission, validate_submission

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {detail}")
    return passed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-train", type=int, default=None,
                    help="expected number of train wells (e.g. 773)")
    ap.add_argument("--expect-test", type=int, default=None,
                    help="expected number of test wells (e.g. 3)")
    ap.add_argument("--full-scan", action="store_true",
                    help="scan every train well (slow); default samples 40")
    args = ap.parse_args(argv)

    require_competition_data()
    print(describe_paths(), "\n")

    # ---- 1 & 2: discovery -------------------------------------------------
    print("== discovery ==")
    train_ids = sorted(discover_wells("train"))
    test_ids = sorted(discover_wells("test"))
    check("1. train wells discovered",
          len(train_ids) > 0 if args.expect_train is None else len(train_ids) == args.expect_train,
          f"found {len(train_ids)}" + (f", expected {args.expect_train}" if args.expect_train else ""))
    check("2. test wells discovered",
          len(test_ids) > 0 if args.expect_test is None else len(test_ids) == args.expect_test,
          f"found {len(test_ids)}: {test_ids[:5]}")
    if not train_ids:
        return _finish()

    # ---- 3: load one train well ------------------------------------------
    print("\n== single-well loading ==")
    tr = load_well(train_ids[0], "train")
    s_tr = summarize_well(tr)
    check("3. train well loads", s_tr["n_rows"] > 0,
          f"{tr.well_id}: {s_tr['n_rows']:,} rows x {s_tr['n_columns']} cols, "
          f"markers={s_tr['n_markers_present']}, typewell={s_tr['has_typewell']}")

    # ---- 4: load one test well -------------------------------------------
    s_te = None
    if test_ids:
        te = load_well(test_ids[0], "test")
        s_te = summarize_well(te)
        check("4. test well loads", s_te["n_rows"] > 0,
              f"{te.well_id}: {s_te['n_rows']:,} rows, markers={s_te['n_markers_present']}")
    else:
        check("4. test well loads", False, "no test wells discovered")

    # ---- scan for edge cases ---------------------------------------------
    limit = None if args.full_scan else min(40, len(train_ids))
    print(f"\n== scanning {'all' if limit is None else limit} train wells (streamed) ==")
    meta = well_metadata("train", limit=limit)
    print(f"  scanned {len(meta)} wells")

    # ---- 5: high GR missingness ------------------------------------------
    if "gr_missing_frac" in meta and meta["gr_missing_frac"].notna().any():
        row = meta.loc[meta["gr_missing_frac"].idxmax()]
        check("5. high-GR-missingness well loads", True,
              f"{row['well_id']}: {row['gr_missing_frac']:.1%} missing, "
              f"longest contiguous gap {row.get('gr_longest_gap')}")
    else:
        check("5. high-GR-missingness well loads", False, "no GR column resolved")

    # ---- 6: MD monotonic --------------------------------------------------
    print("\n== grid integrity ==")
    if "md_monotonic" in meta:
        bad = meta.loc[~meta["md_monotonic"].astype(bool), "well_id"].tolist()
        check("6. MD monotonic increasing", not bad,
              "all scanned wells" if not bad else f"{len(bad)} non-monotonic: {bad[:5]}")
    else:
        check("6. MD monotonic increasing", False, "not evaluated")

    # ---- 7: one-foot MD step ---------------------------------------------
    if "md_step_is_one_foot" in meta:
        bad = meta.loc[~meta["md_step_is_one_foot"].astype(bool), "well_id"].tolist()
        med = meta["md_step_median"].median()
        check("7. MD step is one foot", not bad,
              f"median step {med:.4g}" + ("" if not bad else f"; {len(bad)} deviate: {bad[:5]}"))
    else:
        check("7. MD step is one foot", False, "not evaluated")

    dups = meta.loc[meta.get("md_duplicates", pd.Series(0, index=meta.index)) > 0, "well_id"].tolist()
    check("7b. no duplicate MD values", not dups,
          "none" if not dups else f"{len(dups)} wells: {dups[:5]}")

    # ---- 8 & 9: target presence ------------------------------------------
    print("\n== target availability ==")
    check("8. train target present", bool(s_tr["has_target_column"]),
          f"column resolved: {tr.roles.get('tvt')}, {s_tr['n_target_known']:,} known values")
    if s_te is not None:
        ok = (not s_te["has_target_column"]) or (not s_te["target_available_on_hidden"])
        check("9. test target absent", ok,
              "no TVT column in test horizontal file" if not s_te["has_target_column"]
              else "TVT column present but empty on hidden rows")
    else:
        check("9. test target absent", False, "no test well loaded")

    # ---- 10: masks --------------------------------------------------------
    print("\n== prefix / suffix masks ==")
    ok_masks = True
    for w, lbl in [(tr, "train")] + ([(te, "test")] if test_ids else []):
        vis, hid = w.visible_mask, w.hidden_mask
        good = (
            bool((vis ^ hid).all())
            and int(vis.sum()) == w.region_info["n_visible"]
            and int(hid.sum()) == w.region_info["n_hidden"]
            and not bool(vis[w.region_info["prediction_start_row"]:].any())
        )
        ok_masks &= good
        print(f"    {lbl} {w.well_id}: visible={int(vis.sum()):,} hidden={int(hid.sum()):,} "
              f"start={w.region_info['prediction_start_row']} "
              f"({w.region_info['prediction_start_source']}) "
              f"clean={w.region_info.get('clean_prefix_split')}")
    check("10. visible/hidden masks correct", ok_masks, "complementary and consistent")

    clean = meta.get("clean_prefix_split")
    if clean is not None:
        gaps = meta.loc[clean == False, "well_id"].tolist()  # noqa: E712
        check("10b. no internal TVT_input gaps", not gaps,
              "all scanned wells have a clean prefix/suffix split"
              if not gaps else f"{len(gaps)} wells with internal gaps: {gaps[:5]}")

    # ---- 11: sample submission -------------------------------------------
    print("\n== submission contract ==")
    if available(SAMPLE_SUBMISSION):
        spec = audit_sample_submission()
        print(f"    {spec.describe()}")
        rep = validate_submission(SAMPLE_SUBMISSION, SAMPLE_SUBMISSION)
        check("11. sample_submission valid", rep.passed,
              f"{len(rep.errors)} error(s), {len(rep.warnings)} warning(s) "
              "(placeholder warnings expected)")
        unmatched = [w for w in spec.wells if w not in set(test_ids)]
        check("11b. submission wells match test wells", not unmatched,
              f"{len(spec.wells)} well(s) in submission" if not unmatched
              else f"unmatched: {unmatched[:5]}")
    else:
        check("11. sample_submission valid", False, f"not mounted: {SAMPLE_SUBMISSION}")

    # ---- structural issues -------------------------------------------------
    print("\n== structural issues ==")
    issues = pd.concat(
        [validate_split("train", limit=limit), validate_split("test")], ignore_index=True
    )
    if len(issues):
        print(issues["issue"].value_counts().to_string())
    else:
        print("  none")
    leaks = issues[issues["issue"].str.startswith("TEST_LEAK")]
    check("no test-label leakage detected", leaks.empty,
          "clean" if leaks.empty else f"!! {len(leaks)} well(s): {leaks['well_id'].tolist()}")

    return _finish()


def _finish() -> int:
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 62)
    print(f"SMOKE TESTS: {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    print("No model was trained.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
