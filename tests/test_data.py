"""Tests for the data loader (src/data.py). Synthetic fixtures only."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _data():
    import src.data
    return importlib.reload(src.data)


# ------------------------------------------------------------- discovery ---

def test_discover_wells_train_and_test(mount):
    d = _data()
    train = d.discover_wells("train")
    test = d.discover_wells("test")
    assert set(train) == {"TRW001", "TRW002", "TRW003", "TRW004", "TRW005"}
    assert set(test) == {"TSW001", "TSW002"}
    # no hardcoded ids: everything came off the filesystem
    assert all(w.horizontal is not None for w in train.values())


def test_missing_competition_directory_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ROGII_COMPETITION_ROOT", str(tmp_path / "nope"))
    import src.paths
    importlib.reload(src.paths)
    with pytest.raises(src.paths.CompetitionDataMissing) as exc:
        src.paths.require_competition_data()
    assert "not mounted" in str(exc.value)
    assert "competition root not found" in str(exc.value)


def test_missing_horizontal_file_raises(mount):
    d = _data()
    comp = mount / "input" / "competitions" / "rogii-wellbore-geology-prediction"
    (comp / "train" / "TRW001__horizontal_well.csv").unlink()
    wells = d.discover_wells("train")
    assert wells["TRW001"].horizontal is None
    with pytest.raises(FileNotFoundError):
        d.load_well("TRW001", "train")


def test_missing_typewell_is_tolerated_but_reported(mount):
    d = _data()
    well = d.load_well("TRW005", "train")       # fixture has no typewell
    assert well.tw is None
    assert d.summarize_well(well)["has_typewell"] is False
    with pytest.raises(FileNotFoundError):
        d.load_well("TRW005", "train", require_typewell=True)


def test_wrong_columns_raises_on_missing_md(mount, tmp_path):
    d = _data()
    bad = tmp_path / "bad"
    bad.mkdir()
    pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}).to_csv(
        bad / "BAD001__horizontal_well.csv", index=False
    )
    with pytest.raises(ValueError, match="MD column"):
        d.load_well("BAD001", "train", directory=bad)


# ------------------------------------------------------- prefix / suffix ---

def test_clean_prefix_suffix_split(mount):
    d = _data()
    well = d.load_well("TRW001", "train")
    info = well.region_info
    assert info["prediction_start_row"] == 50
    assert info["n_visible"] == 50
    assert info["n_hidden"] == 150
    assert info["clean_prefix_split"] is True
    assert info["internal_tvt_input_gap"] is False
    assert well.visible_mask.sum() == 50
    assert (well.visible_mask | well.hidden_mask).all()
    assert not (well.visible_mask & well.hidden_mask).any()


def test_internal_tvt_input_gap_is_flagged(mount):
    d = _data()
    well = d.load_well("TRW004", "train")
    info = well.region_info
    # the first gap is the internal hole at row 10, not the true boundary
    assert info["prediction_start_row"] == 10
    assert info["clean_prefix_split"] is False
    assert info["known_after_prediction_start"] > 0
    issues = d.validate_split("train")
    assert "tvt_input_internal_gap" in set(issues.loc[issues.well_id == "TRW004", "issue"])


def test_identify_helpers_are_complementary(mount):
    d = _data()
    hw = d.load_horizontal_well(d.discover_wells("train")["TRW001"])
    from src.columns import resolve_roles
    roles = resolve_roles(hw.columns)
    vis, _ = d.identify_visible_prefix(hw, roles)
    hid, _ = d.identify_hidden_suffix(hw, roles)
    assert np.array_equal(vis, ~hid)


def test_prediction_start_column_takes_precedence(mount, tmp_path):
    d = _data()
    dd = tmp_path / "ps"
    dd.mkdir()
    n = 100
    pd.DataFrame({
        "MD": np.arange(n, dtype=float) + 100,
        "GR": 1.0,
        "TVT_input": [1.0] * 40 + [np.nan] * 60,
        "Prediction Start": [150.0] * n,   # note the space in the name
    }).to_csv(dd / "PS001__horizontal_well.csv", index=False)
    well = d.load_well("PS001", "train", directory=dd)
    assert well.region_info["prediction_start_source"] == "prediction_start_md"
    assert well.region_info["prediction_start_row"] == 50


# -------------------------------------------------------------- loading ----

def test_single_train_well_loading(mount):
    d = _data()
    well = d.load_well("TRW001", "train")
    assert well.split == "train"
    assert len(well.hw) == 200
    assert well.has_target
    assert set(well.markers) == {"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}
    assert well.roles["md"] == "MD" and well.roles["gr"] == "GR"
    # original row order preserved
    assert well.hw["row_index"].tolist() == list(range(200))
    assert well.hw["MD"].is_monotonic_increasing


def test_test_well_loading_without_target_or_markers(mount):
    d = _data()
    well = d.load_well("TSW001", "test")
    assert well.split == "test"
    assert not well.has_target
    assert well.markers == {}
    s = d.summarize_well(well)
    assert s["has_target_column"] is False
    assert s["target_available_on_hidden"] is False
    assert s["n_markers_present"] == 0
    # no spurious leak flag on a clean test split
    issues = d.validate_split("test")
    assert "TEST_LEAK_target_on_hidden_rows" not in set(issues["issue"])


def test_high_gr_missingness_is_detected(mount):
    d = _data()
    s = d.summarize_well(d.load_well("TRW002", "train"))
    assert s["gr_missing_frac"] > 0.5
    assert s["gr_high_missingness"] is True
    assert s["gr_longest_gap"] == 130     # contiguous outage, not scattered
    issues = d.validate_split("train")
    assert "high_gr_missingness" in set(issues.loc[issues.well_id == "TRW002", "issue"])


def test_long_hidden_suffix(mount):
    d = _data()
    well = d.load_well("TRW003", "train")
    s = d.summarize_well(well)
    assert s["n_rows"] == 400
    assert s["n_visible"] == 20
    assert s["n_hidden"] == 380
    assert s["visible_fraction"] == pytest.approx(0.05)


def test_iter_wells_streams_and_metadata_builds(mount):
    d = _data()
    ids = [w.well_id for w in d.iter_wells("train")]
    assert ids == sorted(ids) and len(ids) == 5
    meta = d.well_metadata("train")
    assert len(meta) == 5
    assert {"well_id", "n_rows", "n_visible", "n_hidden", "gr_missing_frac"} <= set(meta.columns)
    assert d.well_metadata("train", limit=2).shape[0] == 2


def test_load_horizontal_well_preserves_row_order_exactly(mount):
    d = _data()
    files = d.discover_wells("train")["TRW001"]
    raw = pd.read_csv(files.horizontal)
    hw = d.load_horizontal_well(files)
    assert np.allclose(hw["MD"].to_numpy(), raw["MD"].to_numpy())
    assert list(hw.columns)[0] == "row_index"


# ------------------------------------------------ Kaggle-environment specifics --

def test_is_hidden_column_materialised(mount):
    d = _data()
    well = d.load_well("TRW001", "train")
    assert "is_visible" in well.hw.columns and "is_hidden" in well.hw.columns
    assert (well.hw["is_visible"] ^ well.hw["is_hidden"]).all()
    assert np.array_equal(well.hidden_mask, ~well.visible_mask)


def test_md_step_is_one_foot(mount):
    d = _data()
    s = d.summarize_well(d.load_well("TRW001", "train"))
    assert s["md_step_median"] == pytest.approx(1.0)
    assert s["md_step_is_one_foot"] is True
    assert s["md_uniform_spacing"] is True
    assert s["md_duplicates"] == 0
    assert s["md_has_gaps"] is False


def test_non_one_foot_spacing_is_flagged(mount, tmp_path):
    d = _data()
    dd = tmp_path / "spacing"
    dd.mkdir()
    n = 50
    pd.DataFrame({
        "MD": np.arange(n, dtype=float) * 0.5,     # half-foot grid
        "GR": 1.0,
        "TVT_input": [1.0] * 10 + [np.nan] * 40,
    }).to_csv(dd / "SP001__horizontal_well.csv", index=False)
    s = d.summarize_well(d.load_well("SP001", "train", directory=dd))
    assert s["md_step_is_one_foot"] is False
    issues = d.validate_split("train", directory=dd)
    assert "md_step_not_one_foot" in set(issues["issue"])


def test_duplicate_md_is_flagged(mount, tmp_path):
    d = _data()
    dd = tmp_path / "dupmd"
    dd.mkdir()
    md = np.arange(50, dtype=float)
    md[10] = md[9]                                  # repeated depth reading
    pd.DataFrame({
        "MD": md, "GR": 1.0, "TVT_input": [1.0] * 10 + [np.nan] * 40,
    }).to_csv(dd / "DM001__horizontal_well.csv", index=False)
    s = d.summarize_well(d.load_well("DM001", "train", directory=dd))
    assert s["md_duplicates"] == 1
    issues = d.validate_split("train", directory=dd)
    assert "duplicate_md_values" in set(issues["issue"])


def test_inference_features_exclude_target(mount):
    """Train wells carry the full TVT curve; it must never become a feature."""
    d = _data()
    well = d.load_well("TRW001", "train")
    assert "TVT" in well.hw.columns              # label is present on disk
    feats = well.inference_features()
    assert "TVT" not in feats.columns
    well.assert_no_target_leakage(feats)
    with pytest.raises(ValueError, match="target column"):
        well.assert_no_target_leakage(well.hw)


def test_tvt_input_is_nan_on_hidden_region(mount):
    """TVT_input is safe as a feature precisely because it stops at the boundary."""
    d = _data()
    well = d.load_well("TRW001", "train")
    ti = pd.to_numeric(well.hw["TVT_input"], errors="coerce")
    assert ti[well.visible_mask].notna().all()
    assert ti[well.hidden_mask].isna().all()


def test_target_accessor_is_explicit(mount):
    d = _data()
    well = d.load_well("TRW001", "train")
    assert len(well.target("hidden")) == well.region_info["n_hidden"]
    assert len(well.target("visible")) == well.region_info["n_visible"]
    assert len(well.target("all")) == len(well.hw)
    assert d.load_well("TSW001", "test").target() is None
