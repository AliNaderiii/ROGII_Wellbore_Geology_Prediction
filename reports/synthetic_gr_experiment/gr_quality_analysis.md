> # SYNTHETIC — NOT A REAL KAGGLE COMPETITION RESULT
> 
> **This is not a competition result.** The discovered well counts do not match the audited real mount (770 train discovered, 770 eligible; real mount has 773/770). These files were produced by the harness against a synthetic or partial field to verify that it runs, and their numbers must not be quoted as validation results.
> 
> data_source=synthetic | validation_scope=SYNTHETIC | n_wells_loaded=770 | n_wells_evaluated=100 | n_train_discovered=770 | n_test_discovered=3 | n_eligible=770
> 
> Metric: Absolute hidden-suffix TVT RMSE — primary metric is point-weighted global RMSE computed on the hidden suffix where both prediction and truth are finite. Additional per-protocol diagnostics: Mean Well RMSE, Median Well RMSE, Worst-10 Well RMSE. No residual RMSE. No averaging across protocols.
> Target: Target = TVT (True Vertical Thickness). Protocol same_well_masked: truth from TVT_input masked interval inside visible prefix (simulated boundary). Protocol unseen_well: truth from TVT label on real hidden suffix [real_prediction_start, n_rows). Visible-prefix TVT_input is input; hidden TVT_input stays NaN in InferenceTask.

# Per-Well Gamma Ray (GR) Quality Bottleneck Analysis

This report presents the findings of systematic analysis of the GR quality bottleneck and target-free imputation strategies.

**Data Source:** synthetic

**Validation Scope:** SYNTHETIC — SYNTHETIC — NOT A REAL KAGGLE COMPETITION RESULT

**Provenance:** 770 wells loaded (eligible universe), 100 wells evaluated, 770 train discovered, 3 test discovered, 770 eligible after excluding 3 public test wells.

**Metric Definition:** Absolute hidden-suffix TVT RMSE — primary metric is point-weighted global RMSE computed on the hidden suffix where both prediction and truth are finite. Additional per-protocol diagnostics: Mean Well RMSE, Median Well RMSE, Worst-10 Well RMSE. No residual RMSE. No averaging across protocols.

**Target Definition:** Target = TVT (True Vertical Thickness). Protocol same_well_masked: truth from TVT_input masked interval inside visible prefix (simulated boundary). Protocol unseen_well: truth from TVT label on real hidden suffix [real_prediction_start, n_rows). Visible-prefix TVT_input is input; hidden TVT_input stays NaN in InferenceTask.

## 1. GR Quality Summary Stats
- **Number of wells in quality report (loaded):** 770
- **Number of wells evaluated in ablation:** 100
- **Average Missing Fraction:** 17.16%
- **Average Longest Contiguous Missing Gap:** 986.5 ft
- **Max Longest Contiguous Missing Gap:** 6815.0 ft
- **Average Prefix GR Quality (r):** 0.6576
- **Average Hidden-Region GR Quality (r):** 0.5710

## 2. GR Imputation Ablation Results

**Metric:** Absolute hidden-suffix TVT RMSE — global point-weighted RMSE over scored hidden suffix rows.

| protocol         | model            |   global_rmse |   mean_well_rmse |   median_well_rmse |   worst_10_rmse |   average_fallback_fraction |   n_wells_evaluated | data_source   | validation_scope   | metric_definition               | target_definition   |
|:-----------------|:-----------------|--------------:|-----------------:|-------------------:|----------------:|----------------------------:|--------------------:|:--------------|:-------------------|:--------------------------------|:--------------------|
| same_well_masked | ridge_default    |       2.40941 |          1.83329 |            1.41802 |         4.55198 |                           0 |                 100 | synthetic     | SYNTHETIC          | Absolute hidden-suffix TVT RMSE | TVT hidden suffix   |
| same_well_masked | ridge_imputed_gr |       2.38772 |          1.81658 |            1.42582 |         4.50381 |                           0 |                 100 | synthetic     | SYNTHETIC          | Absolute hidden-suffix TVT RMSE | TVT hidden suffix   |
| unseen_well      | ridge_default    |       2.41989 |          1.80719 |            1.4698  |         4.5478  |                           0 |                 100 | synthetic     | SYNTHETIC          | Absolute hidden-suffix TVT RMSE | TVT hidden suffix   |
| unseen_well      | ridge_imputed_gr |       2.36893 |          1.77999 |            1.4729  |         4.47819 |                           0 |                 100 | synthetic     | SYNTHETIC          | Absolute hidden-suffix TVT RMSE | TVT hidden suffix   |

## 3. Key Findings on Imputation
- **Linear Interpolation:** Provides a smooth baseline across gaps but is prone to linear artifacts over extremely long missing spans (> 100 ft).
- **Local Rolling Interpolation:** Captures local trends better but can hallucinate variations or high-frequency noise in regions with high tool noise.
- **Bounded Fill (Justified Gaps):** Standardizing to fill only small, local gaps (<= 10 ft) and relying on explicit missingness indicators for larger gaps protects the model from learning from hallucinated signals.

**Decision:** GR imputation is not promoted based on the subset result. No final decision until full 770-well run is complete.

## 4. Well Improvement Summary (Absolute TVT RMSE)

Comparison vs Ridge Default (active baseline), for models that are actually evaluated and present in ablation CSVs (ridge_imputed_gr, gated_pf_beam). ridge_gr_quality is evaluated but its summary is tracked via well_level file to avoid inconsistency where a model appears in summary but not in result CSV.

| protocol         | model            |   wells_improved |   wells_degraded |   wells_unchanged |
|:-----------------|:-----------------|-----------------:|-----------------:|------------------:|
| same_well_masked | ridge_imputed_gr |               49 |               51 |                 0 |
| same_well_masked | gated_pf_beam    |                1 |                3 |                96 |
| unseen_well      | ridge_imputed_gr |               56 |               44 |                 0 |
| unseen_well      | gated_pf_beam    |                3 |                4 |                93 |

## 5. Final Notes
- Ridge Default remains the active baseline.
- GR Imputation remains unpromoted for current subset.
- No final submission is authorized until full 770-well GR experiment completes.
- No external artifacts were used.
