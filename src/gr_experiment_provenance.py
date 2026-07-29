"""Provenance and validation scope for GR quality / gated PF/Beam experiment.

This module implements the strict scope distinction required by the brief:

- REAL_KAGGLE_SUBSET: real mount verified, data_source=real_kaggle, n_wells_evaluated < 770
- REAL_KAGGLE_FULL: real mount verified, n_wells_evaluated == 770, all eligible evaluated, 3 public test wells excluded
- SYNTHETIC: synthetic generator used, Kaggle mount absent, counts don't match audited dataset

Never label a 100-well real subset as full 770-well validation.

Also provides banner strings, file naming, metric/target definitions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

# Audited counts
AUDITED_ELIGIBLE_WELLS = 770
AUDITED_DISCOVERED_TRAIN = 773
AUDITED_DISCOVERED_TEST = 3

# Banners required by task
REAL_SUBSET_BANNER = "REAL KAGGLE SUBSET VALIDATION"
REAL_FULL_BANNER = "REAL KAGGLE FULL VALIDATION"
SYNTHETIC_BANNER = "SYNTHETIC — NOT A REAL KAGGLE COMPETITION RESULT"

# Metric / target definitions – must be logged and stamped in reports
METRIC_DEFINITION = (
    "Absolute hidden-suffix TVT RMSE — primary metric is point-weighted global RMSE "
    "computed on the hidden suffix where both prediction and truth are finite. "
    "Additional per-protocol diagnostics: Mean Well RMSE, Median Well RMSE, Worst-10 Well RMSE. "
    "No residual RMSE. No averaging across protocols."
)
TARGET_DEFINITION = (
    "Target = TVT (True Vertical Thickness). "
    "Protocol same_well_masked: truth from TVT_input masked interval inside visible prefix (simulated boundary). "
    "Protocol unseen_well: truth from TVT label on real hidden suffix [real_prediction_start, n_rows). "
    "Visible-prefix TVT_input is input; hidden TVT_input stays NaN in InferenceTask."
)

# File naming maps
REAL_SUBSET_FILES = {
    "gr_quality": "real_subset_gr_quality.csv",
    "gr_imputation_ablation": "real_subset_gr_imputation_ablation.csv",
    "gated_pf_beam_ablation": "real_subset_gated_pf_beam_ablation.csv",
    "gr_quality_analysis": "real_subset_gr_quality_analysis.md",
    "gated_model_decision": "real_subset_gated_model_decision.md",
    "well_level": "real_subset_well_level_gr_experiment.csv",
}
REAL_FULL_FILES = {
    "gr_quality": "real_full_gr_quality.csv",
    "gr_imputation_ablation": "real_full_gr_imputation_ablation.csv",
    "gated_pf_beam_ablation": "real_full_gated_pf_beam_ablation.csv",
    "gr_quality_analysis": "real_full_gr_quality_analysis.md",
    "gated_model_decision": "real_full_gated_model_decision.md",
    "well_level": "real_full_well_level_gr_experiment.csv",
}
SYNTHETIC_FILES = {
    # Keep historic names under synthetic dir for backward compat, but ensure they are under that dir
    "gr_quality": "gr_quality_synthetic.csv",
    "gr_imputation_ablation": "gr_imputation_ablation.csv",
    "gated_pf_beam_ablation": "gated_pf_beam_ablation.csv",
    "gr_quality_analysis": "gr_quality_analysis.md",
    "gated_model_decision": "gated_model_decision.md",
    "well_level": "well_level_gr_experiment.csv",
}

def determine_validation_scope(
    data_source: str,
    n_wells_evaluated: int,
    n_eligible_wells: int,
    n_train_discovered: int,
    n_test_discovered: int,
    max_wells_arg: int,
) -> Tuple[str, str, str]:
    """Return (scope_code, banner, file_prefix_key).

    scope_code in {"REAL_KAGGLE_SUBSET", "REAL_KAGGLE_FULL", "SYNTHETIC"}
    banner is the exact string to stamp.
    file_prefix_key is "real_subset", "real_full", "synthetic"
    """
    if data_source != "real_kaggle":
        return "SYNTHETIC", SYNTHETIC_BANNER, "synthetic"

    # Real mount verified
    if max_wells_arg == 0:
        # Caller explicitly requests full run; require exactly 770
        if n_wells_evaluated == AUDITED_ELIGIBLE_WELLS and n_eligible_wells == AUDITED_ELIGIBLE_WELLS:
            return "REAL_KAGGLE_FULL", REAL_FULL_BANNER, "real_full"
        else:
            # Incomplete full – caller asked for full but didn't get 770
            # This should be fail-closed upstream, but scope remains FULL for banner purposes
            return "REAL_KAGGLE_FULL", REAL_FULL_BANNER, "real_full"
    else:
        # Subset request
        if n_wells_evaluated < AUDITED_ELIGIBLE_WELLS:
            return "REAL_KAGGLE_SUBSET", REAL_SUBSET_BANNER, "real_subset"
        elif n_wells_evaluated == AUDITED_ELIGIBLE_WELLS:
            # If they asked for, say, 770 via max-wells 770 (not 0), treat as full
            return "REAL_KAGGLE_FULL", REAL_FULL_BANNER, "real_full"
        else:
            # More than 770 shouldn't happen; treat as synthetic for safety
            return "SYNTHETIC", SYNTHETIC_BANNER, "synthetic"

def banner_block_for_scope(
    scope_code: str,
    env_meta: dict,
    n_wells_loaded: int,
    n_wells_evaluated: int,
) -> str:
    """Construct the markdown banner block stamped on every md file."""
    data_source = env_meta.get("data_source", "unknown")
    n_train = env_meta.get("n_train_wells_discovered", "unknown")
    n_test = env_meta.get("n_test_wells_discovered", "unknown")
    n_elig = env_meta.get("n_eligible_wells", "unknown")

    if scope_code == "REAL_KAGGLE_SUBSET":
        return (
            f"> # {REAL_SUBSET_BANNER}\n"
            f"> \n"
            f"> Real Kaggle competition mount verified (train={n_train}, eligible={n_elig} after excluding 3 public test wells, "
            f"test={n_test}). This is a **robust subset** run: {n_wells_evaluated} of {AUDITED_ELIGIBLE_WELLS} eligible wells "
            f"evaluated (n_wells_loaded={n_wells_loaded}). Synthetic harness output lives under "
            f"`reports/synthetic_gr_experiment/` and is never mixed with these files.\n"
            f"> \n"
            f"> data_source={data_source} | validation_scope={scope_code} | n_wells_loaded={n_wells_loaded} | "
            f"n_wells_evaluated={n_wells_evaluated} | n_train_discovered={n_train} | n_test_discovered={n_test} | "
            f"n_eligible={n_elig}\n"
            f"> \n"
            f"> Metric: {METRIC_DEFINITION}\n"
            f"> Target: {TARGET_DEFINITION}\n"
        )
    elif scope_code == "REAL_KAGGLE_FULL":
        return (
            f"> # {REAL_FULL_BANNER}\n"
            f"> \n"
            f"> Computed from the real ROGII competition mount (773 train wells discovered, "
            f"770 eligible after excluding the three visible public test wells, 3 test wells). "
            f"All eligible wells evaluated (n_wells_loaded={n_wells_loaded}, n_wells_evaluated={n_wells_evaluated}). "
            f"Synthetic harness output lives under `reports/synthetic_validation/` and `reports/synthetic_ablation/`.\n"
            f"> \n"
            f"> data_source={data_source} | validation_scope={scope_code} | n_wells_loaded={n_wells_loaded} | "
            f"n_wells_evaluated={n_wells_evaluated} | n_train_discovered={n_train} | n_test_discovered={n_test} | "
            f"n_eligible={n_elig}\n"
            f"> \n"
            f"> Metric: {METRIC_DEFINITION}\n"
            f"> Target: {TARGET_DEFINITION}\n"
        )
    else:
        return (
            f"> # {SYNTHETIC_BANNER}\n"
            f"> \n"
            f"> **This is not a competition result.** The discovered well counts do not match "
            f"the audited real mount ({n_train} train discovered, {n_elig} eligible; real mount has "
            f"{AUDITED_DISCOVERED_TRAIN}/{AUDITED_ELIGIBLE_WELLS}). These files were produced by the harness against a "
            f"synthetic or partial field to verify that it runs, and their numbers must not be quoted as validation results.\n"
            f"> \n"
            f"> data_source={data_source} | validation_scope={scope_code} | n_wells_loaded={n_wells_loaded} | "
            f"n_wells_evaluated={n_wells_evaluated} | n_train_discovered={n_train} | n_test_discovered={n_test} | "
            f"n_eligible={n_elig}\n"
            f"> \n"
            f"> Metric: {METRIC_DEFINITION}\n"
            f"> Target: {TARGET_DEFINITION}\n"
        )

def expected_filenames_for_scope(scope_code: str) -> dict:
    if scope_code == "REAL_KAGGLE_SUBSET":
        return REAL_SUBSET_FILES
    elif scope_code == "REAL_KAGGLE_FULL":
        return REAL_FULL_FILES
    else:
        return SYNTHETIC_FILES

def provenance_log_lines(
    env_meta: dict,
    n_wells_loaded: int,
    n_wells_evaluated: int,
    scope_code: str,
    max_wells_arg: int,
) -> list[str]:
    lines = [
        f"data_source={env_meta.get('data_source')}",
        f"validation_scope={scope_code}",
        f"n_wells_loaded={n_wells_loaded}",
        f"n_wells_evaluated={n_wells_evaluated}",
        f"n_train_wells_discovered={env_meta.get('n_train_wells_discovered')}",
        f"n_test_wells_discovered={env_meta.get('n_test_wells_discovered')}",
        f"n_eligible_wells={env_meta.get('n_eligible_wells')}",
        f"max_wells_arg={max_wells_arg}",
        f"metric_definition={METRIC_DEFINITION}",
        f"target_definition={TARGET_DEFINITION}",
    ]
    return lines
