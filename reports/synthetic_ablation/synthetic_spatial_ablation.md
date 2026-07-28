> # SYNTHETIC — NOT A COMPETITION RESULT
> 
> **This is not a competition result.** The discovered well counts do not match the audited real mount (40 train wells discovered, 40 eligible; the real mount has 773/770). These files were produced by the harness against a synthetic or partial field to verify that it runs, and their numbers must not be quoted as validation results.

# Real spatial-feature ablation

Two paired contrasts isolate the offset-well spatial features, holding the alignment setting fixed:

- **A → C** adds spatial features without alignment features.
- **B → D** adds spatial features with alignment features.

Donors are fold-train wells only, rebuilt inside every fold; the queried well is excluded from its own neighbour set by well ID at query time, and a fold-level `assert_disjoint` guard refuses to run if any validation well could donate.

## Contrasts

| protocol | contrast | context | global_rmse_without | global_rmse_with | delta_global_rmse | pct_global_rmse | median_well_rmse_without | median_well_rmse_with | delta_median_well_rmse | worst10_well_rmse_without | worst10_well_rmse_with | delta_worst10_well_rmse | improves_global | improves_worst10 | material_median_degradation | material_worst10_degradation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_spatial_only - ridge_no_align | no_alignment | 2.1215 | 2.1081 | -0.0134 | -0.6320 | 1.2399 | 1.3483 | 0.1084 | 3.2054 | 3.1466 | -0.0588 | yes | yes | yes | no |
| same_well_masked | ridge_align_spatial - ridge_baseline | with_alignment | 2.1381 | 2.1380 | -0.0001 | -0.0033 | 1.3880 | 1.4061 | 0.0181 | 3.2783 | 3.2205 | -0.0578 | yes | yes | no | no |
| unseen_well | ridge_spatial_only - ridge_no_align | no_alignment | 1.7914 | 1.6327 | -0.1588 | -8.8630 | 1.4118 | 1.4466 | 0.0349 | 2.7545 | 2.4256 | -0.3289 | yes | yes | yes | no |
| unseen_well | ridge_align_spatial - ridge_baseline | with_alignment | 1.6939 | 1.6539 | -0.0400 | -2.3595 | 1.3120 | 1.3814 | 0.0694 | 2.5296 | 2.4624 | -0.0673 | yes | yes | yes | no |

## Pre-registered decision rule

Keep the spatial features **only if** they improve the global metric **or** give a consistent worst-well improvement across both protocols, without unacceptable runtime or leakage risk. Otherwise remove them.

## Decision

**REMOVE FROM NEXT BASELINE** — neither a consistent global improvement nor a consistent worst-10 improvement was observed.

Contrasts computed: 4; improving global RMSE: 4; improving worst-10: 4.
