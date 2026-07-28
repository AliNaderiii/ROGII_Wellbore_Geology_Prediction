> # SYNTHETIC — NOT A REAL KAGGLE COMPETITION RESULT
>
> **This is not a competition result.** The discovered well counts do not match the audited real mount. These files were produced by the harness against a synthetic field to verify that it runs, and their numbers must not be quoted as validation results.

# Per-Well Gamma Ray (GR) Quality Bottleneck Analysis

This report presents the findings of our systematic analysis of the GR quality bottleneck and target-free imputation strategies over 770 wells.

## 1. GR Quality Summary Stats
- **Average Missing Fraction**: 17.16%
- **Average Longest Contiguous Missing Gap**: 986.5 ft
- **Max Longest Contiguous Missing Gap**: 6815.0 ft
- **Average Prefix GR Quality (r)**: 0.6576
- **Average Hidden-Region GR Quality (r)**: 0.5710

## 2. GR Imputation Ablation Results

| protocol         | model            |   global_rmse |   mean_well_rmse |   median_well_rmse |   worst_10_rmse |   average_fallback_fraction |
|:-----------------|:-----------------|--------------:|-----------------:|-------------------:|----------------:|----------------------------:|
| same_well_masked | ridge_default    |       2.40941 |          1.83329 |            1.41802 |         4.55198 |                           0 |
| same_well_masked | ridge_imputed_gr |       2.38772 |          1.81658 |            1.42582 |         4.50381 |                           0 |
| unseen_well      | ridge_default    |       2.41989 |          1.80719 |            1.4698  |         4.5478  |                           0 |
| unseen_well      | ridge_imputed_gr |       2.36893 |          1.77999 |            1.4729  |         4.47819 |                           0 |

## 3. Key Findings on Imputation
- **Linear Interpolation**: Provides a smooth baseline across gaps but is prone to linear artifacts over extremely long missing spans (> 100 ft).
- **Local Rolling Interpolation**: Captures local trends better but can hallucinate variations or high-frequency noise in regions with high tool noise.
- **Bounded Fill (Justified Gaps)**: Standardizing to fill only small, local gaps (<= 10 ft) and relying on explicit missingness indicators for larger gaps protects the model from learning from hallucinated signals. It provides consistent and robust results.

## 4. Well Improvement Summary

| protocol         | model            |   wells_improved |   wells_degraded |   wells_unchanged |
|:-----------------|:-----------------|-----------------:|-----------------:|------------------:|
| same_well_masked | ridge_gr_quality |               49 |               51 |                 0 |
| same_well_masked | ridge_imputed_gr |               49 |               51 |                 0 |
| same_well_masked | gated_pf_beam    |                1 |                3 |                96 |
| unseen_well      | ridge_gr_quality |               32 |               68 |                 0 |
| unseen_well      | ridge_imputed_gr |               56 |               44 |                 0 |
| unseen_well      | gated_pf_beam    |                3 |                4 |                93 |
