# GeoAnchor Controlled Experiment — Decision

> # SYNTHETIC — NOT A REAL KAGGLE COMPETITION RESULT
> 
> **This is not a competition result.** The discovered well counts do not match the audited real mount (240 train wells discovered, 240 eligible; the real mount has 773/770). These files were produced by the harness against a synthetic or partial field to verify that it runs, and their numbers must not be quoted as validation results.


Pre-registered in `reports/geoanchor_experiment.md`. Arms A–D vary only the
Ridge feature set; arm E adds the well-level GBDT gate over bounded PF/Beam
candidate corrections. Ridge Default is the anchor *and* the fallback of arm
E itself. **No final submission was created by this experiment.**

## Arm metrics

| protocol | model | n_wells | n_points | global_rmse | mean_well_rmse | median_well_rmse | p90_well_rmse | worst10_well_rmse | worst_well_rmse | max_abs_error | mean_bias | predict_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_default | 240 | 581811 | 2.2433 | 1.7272 | 1.3277 | 3.6193 | 5.5717 | 8.2533 | 10.8754 | 0.0086 | 3.7556 |
| same_well_masked | ridge_affine_cal | 240 | 581811 | 2.2596 | 1.7506 | 1.4074 | 3.5775 | 5.5241 | 8.1924 | 12.0134 | 0.0173 | 5.6627 |
| same_well_masked | ridge_multibranch | 240 | 581811 | 2.2359 | 1.7411 | 1.4367 | 3.6212 | 5.0937 | 6.6014 | 11.2772 | -0.0033 | 5.8122 |
| same_well_masked | ridge_affine_multibranch | 240 | 581811 | 2.2470 | 1.7566 | 1.4218 | 3.6539 | 5.1057 | 6.9104 | 12.8895 | 0.0093 | 7.4706 |
| same_well_masked | ridge_gated_gbdt | 240 | 581811 | 2.9045 | 1.7473 | 0.9490 | 4.0015 | 10.4920 | 19.7024 | 21.6464 | -0.1781 | 17.7485 |
| unseen_well | ridge_default | 240 | 603192 | 2.2420 | 1.7158 | 1.3494 | 3.4861 | 6.0939 | 9.9323 | 44.3694 | 0.0465 | 3.6845 |
| unseen_well | ridge_affine_cal | 240 | 603192 | 2.2628 | 1.7329 | 1.3582 | 3.4240 | 5.9572 | 9.6813 | 41.6157 | 0.0539 | 5.6052 |
| unseen_well | ridge_multibranch | 240 | 603192 | 2.2810 | 1.7716 | 1.3568 | 3.4915 | 6.1737 | 9.4483 | 44.2164 | 0.0252 | 5.8399 |
| unseen_well | ridge_affine_multibranch | 240 | 603192 | 2.2933 | 1.7845 | 1.3254 | 3.3927 | 6.0440 | 8.8991 | 43.7282 | 0.0384 | 7.5935 |
| unseen_well | ridge_gated_gbdt | 240 | 603192 | 2.7956 | 1.8209 | 1.1804 | 3.9748 | 8.9475 | 13.7629 | 44.3694 | 0.0142 | 17.5358 |

## Pre-registered decision

| protocol | candidate_arm | delta_global_rmse | delta_median_well_rmse | delta_worst10_well_rmse | improves_global | bootstrap_ci_not_against | no_material_degradation | fold_stable | verdict | both_protocols_covered | improves_global_both | bootstrap_ok_both | no_material_degradation_both | fold_stable_both |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_affine_cal | 0.0163 | 0.0797 | -0.0476 | no | yes | no | no | NOT_CARRIED | yes | no | yes | no | no |
| same_well_masked | ridge_multibranch | -0.0074 | 0.1089 | -0.4779 | yes | yes | no | no | NOT_CARRIED | yes | no | no | no | no |
| same_well_masked | ridge_affine_multibranch | 0.0037 | 0.0941 | -0.4660 | no | yes | no | no | NOT_CARRIED | yes | no | yes | no | no |
| same_well_masked | ridge_gated_gbdt | 0.6612 | -0.3787 | 4.9203 | no | no | no | no | NOT_CARRIED | yes | no | no | no | no |
| unseen_well | ridge_affine_cal | 0.0208 | 0.0089 | -0.1367 | no | yes | yes | no | NOT_CARRIED | yes | no | yes | no | no |
| unseen_well | ridge_multibranch | 0.0390 | 0.0074 | 0.0798 | no | no | yes | no | NOT_CARRIED | yes | no | no | no | no |
| unseen_well | ridge_affine_multibranch | 0.0513 | -0.0239 | -0.0498 | no | yes | yes | no | NOT_CARRIED | yes | no | yes | no | no |
| unseen_well | ridge_gated_gbdt | 0.5537 | -0.1689 | 2.8537 | no | no | no | no | NOT_CARRIED | yes | no | no | no | no |

## Fold stability (delta vs Ridge Default; negative favours the arm)

| protocol | fold | n_wells | n_points | rmse_default | rmse_candidate | delta_rmse | candidate_better | n_folds | n_folds_candidate_better | n_folds_candidate_not_worse | stable_across_folds | mean_fold_delta_rmse | candidate_arm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same_well_masked | 0 | 48 | 109757 | 2.0847 | 2.2056 | 0.1209 | no | 5 | 2 | 2 | no | 0.0193 | ridge_affine_cal |
| same_well_masked | 1 | 48 | 112536 | 2.3792 | 2.3598 | -0.0194 | yes | 5 | 2 | 2 | no | 0.0193 | ridge_affine_cal |
| same_well_masked | 2 | 48 | 117471 | 2.3556 | 2.3896 | 0.0339 | no | 5 | 2 | 2 | no | 0.0193 | ridge_affine_cal |
| same_well_masked | 3 | 48 | 117547 | 2.3906 | 2.3318 | -0.0588 | yes | 5 | 2 | 2 | no | 0.0193 | ridge_affine_cal |
| same_well_masked | 4 | 48 | 124500 | 1.9844 | 2.0043 | 0.0199 | no | 5 | 2 | 2 | no | 0.0193 | ridge_affine_cal |
| unseen_well | 0 | 48 | 114439 | 2.5163 | 2.4962 | -0.0201 | yes | 5 | 3 | 3 | no | 0.0197 | ridge_affine_cal |
| unseen_well | 1 | 48 | 117832 | 2.0256 | 2.0593 | 0.0336 | no | 5 | 3 | 3 | no | 0.0197 | ridge_affine_cal |
| unseen_well | 2 | 48 | 121162 | 2.4589 | 2.5459 | 0.0870 | no | 5 | 3 | 3 | no | 0.0197 | ridge_affine_cal |
| unseen_well | 3 | 48 | 120342 | 1.9562 | 1.9547 | -0.0014 | yes | 5 | 3 | 3 | no | 0.0197 | ridge_affine_cal |
| unseen_well | 4 | 48 | 129417 | 2.2077 | 2.2073 | -0.0004 | yes | 5 | 3 | 3 | no | 0.0197 | ridge_affine_cal |
| same_well_masked | 0 | 48 | 109757 | 2.0847 | 2.0259 | -0.0588 | yes | 5 | 3 | 3 | no | -0.0098 | ridge_multibranch |
| same_well_masked | 1 | 48 | 112536 | 2.3792 | 2.3142 | -0.0650 | yes | 5 | 3 | 3 | no | -0.0098 | ridge_multibranch |
| same_well_masked | 2 | 48 | 117471 | 2.3556 | 2.4685 | 0.1129 | no | 5 | 3 | 3 | no | -0.0098 | ridge_multibranch |
| same_well_masked | 3 | 48 | 117547 | 2.3906 | 2.3340 | -0.0566 | yes | 5 | 3 | 3 | no | -0.0098 | ridge_multibranch |
| same_well_masked | 4 | 48 | 124500 | 1.9844 | 2.0028 | 0.0184 | no | 5 | 3 | 3 | no | -0.0098 | ridge_multibranch |
| unseen_well | 0 | 48 | 114439 | 2.5163 | 2.5090 | -0.0073 | yes | 5 | 1 | 1 | no | 0.0398 | ridge_multibranch |
| unseen_well | 1 | 48 | 117832 | 2.0256 | 2.1217 | 0.0961 | no | 5 | 1 | 1 | no | 0.0398 | ridge_multibranch |
| unseen_well | 2 | 48 | 121162 | 2.4589 | 2.4777 | 0.0188 | no | 5 | 1 | 1 | no | 0.0398 | ridge_multibranch |
| unseen_well | 3 | 48 | 120342 | 1.9562 | 1.9635 | 0.0073 | no | 5 | 1 | 1 | no | 0.0398 | ridge_multibranch |
| unseen_well | 4 | 48 | 129417 | 2.2077 | 2.2919 | 0.0842 | no | 5 | 1 | 1 | no | 0.0398 | ridge_multibranch |
| same_well_masked | 0 | 48 | 109757 | 2.0847 | 2.1507 | 0.0660 | no | 5 | 2 | 2 | no | 0.0045 | ridge_affine_multibranch |
| same_well_masked | 1 | 48 | 112536 | 2.3792 | 2.2849 | -0.0943 | yes | 5 | 2 | 2 | no | 0.0045 | ridge_affine_multibranch |
| same_well_masked | 2 | 48 | 117471 | 2.3556 | 2.4832 | 0.1275 | no | 5 | 2 | 2 | no | 0.0045 | ridge_affine_multibranch |
| same_well_masked | 3 | 48 | 117547 | 2.3906 | 2.2752 | -0.1154 | yes | 5 | 2 | 2 | no | 0.0045 | ridge_affine_multibranch |
| same_well_masked | 4 | 48 | 124500 | 1.9844 | 2.0228 | 0.0384 | no | 5 | 2 | 2 | no | 0.0045 | ridge_affine_multibranch |
| unseen_well | 0 | 48 | 114439 | 2.5163 | 2.4571 | -0.0592 | yes | 5 | 2 | 2 | no | 0.0512 | ridge_affine_multibranch |
| unseen_well | 1 | 48 | 117832 | 2.0256 | 2.1672 | 0.1416 | no | 5 | 2 | 2 | no | 0.0512 | ridge_affine_multibranch |
| unseen_well | 2 | 48 | 121162 | 2.4589 | 2.5628 | 0.1039 | no | 5 | 2 | 2 | no | 0.0512 | ridge_affine_multibranch |
| unseen_well | 3 | 48 | 120342 | 1.9562 | 1.9533 | -0.0029 | yes | 5 | 2 | 2 | no | 0.0512 | ridge_affine_multibranch |
| unseen_well | 4 | 48 | 129417 | 2.2077 | 2.2804 | 0.0727 | no | 5 | 2 | 2 | no | 0.0512 | ridge_affine_multibranch |
| same_well_masked | 0 | 48 | 109757 | 2.0847 | 3.2881 | 1.2035 | no | 5 | 0 | 0 | no | 0.6420 | ridge_gated_gbdt |
| same_well_masked | 1 | 48 | 112536 | 2.3792 | 2.8153 | 0.4361 | no | 5 | 0 | 0 | no | 0.6420 | ridge_gated_gbdt |
| same_well_masked | 2 | 48 | 117471 | 2.3556 | 2.3572 | 0.0015 | no | 5 | 0 | 0 | no | 0.6420 | ridge_gated_gbdt |
| same_well_masked | 3 | 48 | 117547 | 2.3906 | 3.4671 | 1.0765 | no | 5 | 0 | 0 | no | 0.6420 | ridge_gated_gbdt |
| same_well_masked | 4 | 48 | 124500 | 1.9844 | 2.4769 | 0.4925 | no | 5 | 0 | 0 | no | 0.6420 | ridge_gated_gbdt |
| unseen_well | 0 | 48 | 114439 | 2.5163 | 3.0992 | 0.5829 | no | 5 | 1 | 1 | no | 0.4939 | ridge_gated_gbdt |
| unseen_well | 1 | 48 | 117832 | 2.0256 | 3.0748 | 1.0491 | no | 5 | 1 | 1 | no | 0.4939 | ridge_gated_gbdt |
| unseen_well | 2 | 48 | 121162 | 2.4589 | 2.7660 | 0.3071 | no | 5 | 1 | 1 | no | 0.4939 | ridge_gated_gbdt |
| unseen_well | 3 | 48 | 120342 | 1.9562 | 1.5376 | -0.4185 | yes | 5 | 1 | 1 | no | 0.4939 | ridge_gated_gbdt |
| unseen_well | 4 | 48 | 129417 | 2.2077 | 3.1567 | 0.9490 | no | 5 | 1 | 1 | no | 0.4939 | ridge_gated_gbdt |

## Bootstrap confidence intervals (well-cluster resampling)

| protocol | candidate_arm | metric | n_wells | observed_delta | ci_low_2.5 | ci_high_97.5 | ci_excludes_zero | frac_bootstrap_negative |
|---|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_affine_cal | global_point_rmse_delta | 240 | 0.0163 | -0.0355 | 0.0708 | no | 0.2750 |
| unseen_well | ridge_affine_cal | global_point_rmse_delta | 240 | 0.0208 | -0.0169 | 0.0619 | no | 0.1430 |
| same_well_masked | ridge_affine_cal | mean_well_rmse_delta | 240 | 0.0234 | -0.0221 | 0.0668 | no | 0.1450 |
| unseen_well | ridge_affine_cal | mean_well_rmse_delta | 240 | 0.0171 | -0.0156 | 0.0547 | no | 0.1525 |
| same_well_masked | ridge_multibranch | global_point_rmse_delta | 240 | -0.0074 | -0.0768 | 0.0568 | no | 0.5770 |
| unseen_well | ridge_multibranch | global_point_rmse_delta | 240 | 0.0390 | 0.0003 | 0.0830 | yes | 0.0235 |
| same_well_masked | ridge_multibranch | mean_well_rmse_delta | 240 | 0.0139 | -0.0394 | 0.0645 | no | 0.2905 |
| unseen_well | ridge_multibranch | mean_well_rmse_delta | 240 | 0.0558 | 0.0197 | 0.0968 | yes | 0.0010 |
| same_well_masked | ridge_affine_multibranch | global_point_rmse_delta | 240 | 0.0037 | -0.0841 | 0.0937 | no | 0.4565 |
| unseen_well | ridge_affine_multibranch | global_point_rmse_delta | 240 | 0.0513 | -0.0101 | 0.1141 | no | 0.0450 |
| same_well_masked | ridge_affine_multibranch | mean_well_rmse_delta | 240 | 0.0294 | -0.0382 | 0.0937 | no | 0.1850 |
| unseen_well | ridge_affine_multibranch | mean_well_rmse_delta | 240 | 0.0686 | 0.0199 | 0.1224 | yes | 0.0030 |
| same_well_masked | ridge_gated_gbdt | global_point_rmse_delta | 240 | 0.6612 | 0.0615 | 1.2640 | yes | 0.0125 |
| unseen_well | ridge_gated_gbdt | global_point_rmse_delta | 240 | 0.5537 | 0.0898 | 1.0967 | yes | 0.0050 |
| same_well_masked | ridge_gated_gbdt | mean_well_rmse_delta | 240 | 0.0202 | -0.2284 | 0.2758 | no | 0.4375 |
| unseen_well | ridge_gated_gbdt | mean_well_rmse_delta | 240 | 0.1051 | -0.0860 | 0.3388 | no | 0.1580 |

## Per-well improved / degraded counts vs Ridge Default

| protocol | candidate_arm | n_paired_wells | n_improved | n_degraded | n_unchanged | frac_improved |
|---|---|---|---|---|---|---|
| same_well_masked | ridge_affine_cal | 240 | 112 | 128 | 0 | 0.4667 |
| unseen_well | ridge_affine_cal | 240 | 118 | 122 | 0 | 0.4917 |
| same_well_masked | ridge_multibranch | 240 | 112 | 128 | 0 | 0.4667 |
| unseen_well | ridge_multibranch | 240 | 109 | 131 | 0 | 0.4542 |
| same_well_masked | ridge_affine_multibranch | 240 | 123 | 117 | 0 | 0.5125 |
| unseen_well | ridge_affine_multibranch | 240 | 105 | 135 | 0 | 0.4375 |
| same_well_masked | ridge_gated_gbdt | 240 | 90 | 41 | 109 | 0.3750 |
| unseen_well | ridge_gated_gbdt | 240 | 60 | 39 | 141 | 0.2500 |

## Gate behaviour (arm E)

| protocol | n_wells | n_activated | activation_rate | fallback_rate | mean_predicted_improvement_activated | top_fallback_reasons |
|---|---|---|---|---|---|---|
| same_well_masked | 240 | 131 | 0.5458 | 0.4542 | 0.7389 | pf:pseudo_holdout_unavailable (59); beam:pseudo_holdout_unavailable (59); pf_beam_mean:pseudo_holdout_unavailable (59); pf:candidate_unavailable:partial_missing_or_invalid_horizontal_gr (25); beam:candidate_unavailable:partial_missing_or_invalid_horizontal_gr (25); pf_beam_mean:candidate_unavailable:no_pf_or_beam_track (25); pf:pseudo_holdout_not_improved (19); pf_beam_mean:pseudo_holdout_not_improved (18) |
| unseen_well | 240 | 99 | 0.4125 | 0.5875 | 1.0271 | pf:pseudo_holdout_not_improved (87); pf_beam_mean:pseudo_holdout_not_improved (86); beam:pseudo_holdout_not_improved (84); beam:gbdt_expected_gain_below_margin (45); pf_beam_mean:gbdt_expected_gain_below_margin (42); pf:gbdt_expected_gain_below_margin (41); pf:candidate_unavailable:partial_missing_or_invalid_horizontal_gr (10); beam:candidate_unavailable:partial_missing_or_invalid_horizontal_gr (10) |


| protocol | fold | n_wells | n_activated | activation_rate | fallback_rate | gate_killed | n_fallback_pseudo_holdout_not_improved | n_fallback_tail_risk | n_fallback_low_confidence | n_fallback_disagreement | n_fallback_below_margin | n_fallback_candidate_unavailable | n_applied_pf | n_applied_beam | n_applied_pf_beam_mean | n_oof_wells | n_examples | n_pseudo_skipped | killed | kill_reason | margin | conf_thr | sep_cap | pooled_oof_delta | oof_activation_rate | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same_well_masked | 0 | 48 | 28 | 0.5833 | 0.4167 | no | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 27 | 1 | 123 | 369 | 46 | no |  | 0.0000 | 0.0000 | 0.6557 | -0.7998 | 0.8618 | 35.4965 |
| same_well_masked | 1 | 48 | 23 | 0.4792 | 0.5208 | no | 5 | 1 | 0 | 1 | 0 | 3 | 0 | 21 | 2 | 126 | 378 | 45 | no |  | 0.0000 | 0.0000 | 0.5634 | -0.8750 | 0.8889 | 17.2881 |
| same_well_masked | 2 | 48 | 26 | 0.5417 | 0.4583 | no | 2 | 1 | 0 | 0 | 3 | 6 | 0 | 25 | 1 | 123 | 369 | 50 | no |  | 0.0000 | 0.0000 | 0.6557 | -0.8817 | 0.8780 | 17.5138 |
| same_well_masked | 3 | 48 | 24 | 0.5000 | 0.5000 | no | 9 | 0 | 0 | 0 | 0 | 6 | 0 | 22 | 2 | 122 | 366 | 52 | no |  | 0.0000 | 0.0000 | 0.6557 | -0.9457 | 0.9180 | 17.8976 |
| same_well_masked | 4 | 48 | 30 | 0.6250 | 0.3750 | no | 0 | 0 | 0 | 0 | 2 | 7 | 0 | 29 | 1 | 122 | 366 | 51 | no |  | 0.0000 | 0.0000 | 0.6557 | -0.9046 | 0.8689 | 17.4960 |
| unseen_well | 0 | 48 | 22 | 0.4583 | 0.5417 | no | 17 | 0 | 0 | 0 | 10 | 0 | 19 | 3 | 0 | 165 | 495 | 0 | no |  | 0.0000 | 0.0000 | 0.7241 | -1.0611 | 0.6545 | 49.4286 |
| unseen_well | 1 | 48 | 24 | 0.5000 | 0.5000 | no | 14 | 1 | 0 | 0 | 10 | 0 | 4 | 19 | 1 | 165 | 495 | 0 | no |  | 0.0000 | 0.0000 | 0.7241 | -1.0584 | 0.6667 | 19.4790 |
| unseen_well | 2 | 48 | 18 | 0.3750 | 0.6250 | no | 17 | 0 | 0 | 0 | 7 | 6 | 3 | 15 | 0 | 171 | 513 | 0 | no |  | 0.0000 | 0.0000 | 0.7241 | -0.9303 | 0.6608 | 20.2591 |
| unseen_well | 3 | 48 | 18 | 0.3750 | 0.6250 | no | 20 | 1 | 0 | 0 | 8 | 4 | 1 | 17 | 0 | 170 | 510 | 0 | no |  | 0.0000 | 0.0000 | 0.7241 | -1.0066 | 0.6824 | 20.4273 |
| unseen_well | 4 | 48 | 17 | 0.3542 | 0.6458 | no | 20 | 1 | 0 | 1 | 11 | 0 | 4 | 13 | 0 | 169 | 507 | 0 | no |  | 0.0000 | 0.0000 | 0.6859 | -0.9962 | 0.6450 | 20.5449 |

## Runtime and memory

| protocol_or_total | seconds | peak_rss_mb |
|---|---|---|
| same_well_masked | 271.8390 | 712.6000 |
| unseen_well | 298.6880 | 712.6000 |
| total | 570.5270 | 712.6000 |

## Failures

No task, fit or prediction failures were recorded.

## Honesty notes

- Every number above was computed in this run from well-level results; nothing
  was copied from another run or from public-leaderboard information.
- CARRIED is not promotion: it only permits a confirmation run on the real
  competition mount. Ridge Default remains the fallback in all cases.
- The three visible public test wells were excluded from every fold, fit,
  gate-training set and table by the hard guard in `src.validation`.
