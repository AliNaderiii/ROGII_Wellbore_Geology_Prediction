> **SYNTHETIC PIPELINE VERIFICATION ONLY — NOT A COMPETITION RESULT**

# Ridge alignment / spatial feature ablation

A 2x2 factorial run through the **existing** Ridge model. Branch B is the current, unmodified Ridge baseline and every delta is taken against it. All four branches share the same folds and are cross-fitted by well ID; the two protocols are reported separately and never averaged.

| branch | alignment features | spatial features |
|---|---|---|
| A. Ridge without alignment features | no | no |
| B. Ridge with alignment features (current baseline) | yes | no |
| C. Ridge with spatial features | no | yes |
| D. Ridge with alignment and spatial features | yes | yes |

## Delta against the current Ridge baseline

| protocol | branch | alignment_features | spatial_features | n_wells | n_points | global_rmse | median_well_rmse | worst10_well_rmse | delta_global_rmse_vs_baseline | pct_change_vs_baseline | delta_median_well_rmse_vs_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_no_align | no | no | 40 | 94674 | 2.122 | 1.240 | 3.205 | -0.017 | -0.775 | -0.148 |
| same_well_masked | ridge_baseline | yes | no | 40 | 94674 | 2.138 | 1.388 | 3.278 | 0.000 | 0.000 | 0.000 |
| same_well_masked | ridge_spatial_only | no | yes | 40 | 94674 | 2.108 | 1.348 | 3.147 | -0.030 | -1.402 | -0.040 |
| same_well_masked | ridge_align_spatial | yes | yes | 40 | 94674 | 2.138 | 1.406 | 3.220 | -0.000 | -0.003 | 0.018 |
| unseen_well | ridge_no_align | no | no | 40 | 97528 | 1.791 | 1.412 | 2.754 | 0.098 | 5.759 | 0.100 |
| unseen_well | ridge_baseline | yes | no | 40 | 97528 | 1.694 | 1.312 | 2.530 | 0.000 | 0.000 | 0.000 |
| unseen_well | ridge_spatial_only | no | yes | 40 | 97528 | 1.633 | 1.447 | 2.426 | -0.061 | -3.614 | 0.135 |
| unseen_well | ridge_align_spatial | yes | yes | 40 | 97528 | 1.654 | 1.381 | 2.462 | -0.040 | -2.359 | 0.069 |

Only wells scored by every branch within a protocol enter the comparison, so a branch cannot look better by having dropped a hard well.

## Isolating the alignment features

| protocol | contrast | spatial_context | global_rmse_without_alignment | global_rmse_with_alignment | delta_global_rmse | alignment_features_help |
|---|---|---|---|---|---|---|
| same_well_masked | ridge_baseline - ridge_no_align | no_spatial | 2.122 | 2.138 | 0.017 | no |
| same_well_masked | ridge_align_spatial - ridge_spatial_only | with_spatial | 2.108 | 2.138 | 0.030 | no |
| unseen_well | ridge_baseline - ridge_no_align | no_spatial | 1.791 | 1.694 | -0.098 | yes |
| unseen_well | ridge_align_spatial - ridge_spatial_only | with_spatial | 1.633 | 1.654 | 0.021 | no |

Each row is a paired contrast holding the spatial setting fixed. A negative `delta_global_rmse` means the alignment features lowered RMSE.

## Decision

1 of 4 contrasts favour the alignment features, covering same_well_masked, unseen_well. Alignment features did not lower global rmse in every contrast under both protocols.

**Remove** the alignment features from the next baseline.
