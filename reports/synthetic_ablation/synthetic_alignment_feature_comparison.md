> # SYNTHETIC — NOT A COMPETITION RESULT
> 
> **This is not a competition result.** The discovered well counts do not match the audited real mount (40 train wells discovered, 40 eligible; the real mount has 773/770). These files were produced by the harness against a synthetic or partial field to verify that it runs, and their numbers must not be quoted as validation results.

# Real alignment-feature comparison

Two paired contrasts isolate the four GR/typewell alignment features (`align_tvt`, `align_score`, `align_shift`, `align_gradient`), holding the spatial setting fixed:

- **A → B** adds the alignment features without spatial features.
- **C → D** adds the alignment features with spatial features.

A negative `delta_global_rmse` means the alignment features lowered RMSE.

## Contrasts

| protocol | contrast | context | global_rmse_without | global_rmse_with | delta_global_rmse | pct_global_rmse | median_well_rmse_without | median_well_rmse_with | delta_median_well_rmse | worst10_well_rmse_without | worst10_well_rmse_with | delta_worst10_well_rmse | improves_global | improves_worst10 | material_median_degradation | material_worst10_degradation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_baseline - ridge_no_align | no_spatial | 2.1215 | 2.1381 | 0.0166 | 0.7813 | 1.2399 | 1.3880 | 0.1482 | 3.2054 | 3.2783 | 0.0729 | no | no | yes | yes |
| same_well_masked | ridge_align_spatial - ridge_spatial_only | with_spatial | 2.1081 | 2.1380 | 0.0299 | 1.4190 | 1.3483 | 1.4061 | 0.0578 | 3.1466 | 3.2205 | 0.0739 | no | no | yes | yes |
| unseen_well | ridge_baseline - ridge_no_align | no_spatial | 1.7914 | 1.6939 | -0.0976 | -5.4458 | 1.4118 | 1.3120 | -0.0998 | 2.7545 | 2.5296 | -0.2248 | yes | yes | no | no |
| unseen_well | ridge_align_spatial - ridge_spatial_only | with_spatial | 1.6327 | 1.6539 | 0.0212 | 1.3015 | 1.4466 | 1.3814 | -0.0653 | 2.4256 | 2.4624 | 0.0368 | no | no | no | no |

## Pre-registered decision rule

Keep the alignment features in the next baseline **only if** they improve global RMSE in **both** protocols **and** do not materially degrade median or worst-10 well RMSE (tolerance: 2% of the branch-B value). Otherwise remove them. This rule was fixed before the real results were inspected.

## Decision

**REMOVE FROM NEXT BASELINE** — global RMSE did not improve in every contrast under both protocols.

Contrasts computed: 4; improving global RMSE: 1; protocols covered: same_well_masked, unseen_well.
