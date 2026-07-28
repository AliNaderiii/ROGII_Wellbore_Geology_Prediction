> # SYNTHETIC — NOT A COMPETITION RESULT
> 
> **This is not a competition result.** The discovered well counts do not match the audited real mount (40 train wells discovered, 40 eligible; the real mount has 773/770). These files were produced by the harness against a synthetic or partial field to verify that it runs, and their numbers must not be quoted as validation results.

# Real protocol comparison

`same_well_masked` and `unseen_well` are reported separately and are **never averaged**. No metric on this page is averaged across the two protocols, and no combined ranking is produced: the protocols answer different questions and their scored rows come from different sources (`TVT_input` for the masked boundary, the `TVT` label for the real hidden suffix).

## Per-protocol support and reference-branch metrics

| protocol | reference_branch | n_wells | n_scored_points | prefix_min | prefix_median | suffix_min | suffix_median | gr_missing_median | global_rmse | median_well_rmse | p90_well_rmse | worst10_well_rmse | scored_exact_suffix_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_baseline | 40 | 94674 | 200 | 1655.0000 | 1291 | 2352.5000 | 0.1057 | 2.1381 | 1.3880 | 2.8229 | 3.2783 | yes |
| unseen_well | ridge_baseline | 40 | 97528 | 1682 | 3831.0000 | 1291 | 2381.0000 | 0.1057 | 1.6939 | 1.3120 | 2.3343 | 2.5296 | yes |

## Per-branch global RMSE, by protocol

| branch | same_well_masked | unseen_well |
|---|---|---|
| ridge_align_spatial | 2.1380 | 1.6539 |
| ridge_baseline | 2.1381 | 1.6939 |
| ridge_no_align | 2.1215 | 1.7914 |
| ridge_spatial_only | 2.1081 | 1.6327 |

A branch may rank differently under the two protocols. That is a real finding about the protocols, not a tie to be broken by averaging.
