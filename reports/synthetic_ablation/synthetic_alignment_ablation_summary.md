> # SYNTHETIC — NOT A COMPETITION RESULT
> 
> **This is not a competition result.** The discovered well counts do not match the audited real mount (40 train wells discovered, 40 eligible; the real mount has 773/770). These files were produced by the harness against a synthetic or partial field to verify that it runs, and their numbers must not be quoted as validation results.

# Real alignment / spatial ablation — summary

Four Ridge configurations, cross-fitted by well ID, both protocols reported separately and **never averaged**. Branch B is the current, unmodified Ridge baseline and every delta is taken against it.

| branch | alignment features | spatial features |
|---|---|---|
| A. Ridge without alignment features | no | no |
| B. Ridge with alignment features (current baseline) | yes | no |
| C. Ridge with spatial features | no | yes |
| D. Ridge with alignment and spatial features | yes | yes |

## Headline metrics

| protocol | branch | n_wells_evaluated | n_points_evaluated | global_rmse | mean_well_rmse | median_well_rmse | p90_well_rmse | worst10_well_rmse | worst_well_rmse | worst_well_id | delta_global_rmse_vs_baseline | pct_change_vs_baseline | predict_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_no_align | 40 | 94674 | 2.1215 | 1.6140 | 1.2399 | 2.5555 | 3.2054 | 8.5023 | tr00023 | -0.0166 | -0.7752 | 0.3665 |
| same_well_masked | ridge_baseline | 40 | 94674 | 2.1381 | 1.6359 | 1.3880 | 2.8229 | 3.2783 | 8.0709 | tr00023 | 0.0000 | 0.0000 | 0.2531 |
| same_well_masked | ridge_spatial_only | 40 | 94674 | 2.1081 | 1.6445 | 1.3483 | 2.4960 | 3.1466 | 6.5925 | tr00023 | -0.0300 | -1.4024 | 0.5798 |
| same_well_masked | ridge_align_spatial | 40 | 94674 | 2.1380 | 1.6645 | 1.4061 | 2.5620 | 3.2205 | 6.2232 | tr00023 | -0.0001 | -0.0033 | 0.5813 |
| unseen_well | ridge_no_align | 40 | 97528 | 1.7914 | 1.6032 | 1.4118 | 2.5680 | 2.7545 | 3.8831 | tr0001d | 0.0976 | 5.7595 | 0.3659 |
| unseen_well | ridge_baseline | 40 | 97528 | 1.6939 | 1.5015 | 1.3120 | 2.3343 | 2.5296 | 3.7336 | tr0001d | 0.0000 | 0.0000 | 0.2512 |
| unseen_well | ridge_spatial_only | 40 | 97528 | 1.6327 | 1.4604 | 1.4466 | 2.3860 | 2.4256 | 3.1959 | tr0001d | -0.0612 | -3.6140 | 0.5889 |
| unseen_well | ridge_align_spatial | 40 | 97528 | 1.6539 | 1.4735 | 1.3814 | 2.3669 | 2.4624 | 3.0322 | tr00008 | -0.0400 | -2.3595 | 0.5912 |

Only wells scored by every branch within a protocol enter the comparison, so a branch cannot look better by having dropped a hard well. Point counts are per (protocol, branch) and are not summed across branches.

## Failures

No task, fit or predict failure was recorded.

## Run environment

| key | value |
|---|---|
| validation | SYNTHETIC PIPELINE VERIFICATION ONLY — NOT A COMPETITION RESULT |
| timestamp_utc | 2026-07-28T08:34:25.261270+00:00 |
| runtime_seconds | 111.23 |
| peak_rss_mb | 227.3 |
| n_train_wells_discovered | 40 |
| n_blocked_wells_excluded | 0 |
| blocked_well_ids | [] |
| n_eligible_wells | 40 |
| n_wells_evaluated | 40 |
| max_wells | None |
| n_splits | 5 |
| seed | 0 |
| branches | ["ridge_no_align", "ridge_baseline", "ridge_spatial_only", "ridge_align_spatial"] |
| spatial_k | 12 |
| spatial_radius | 6000.0 |
| device_requested | auto |
| device_selected | cpu |
| gpu_name |  |
| cpu_count | 2 |
| ram_mb | None |
| python | 3.11.2 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| failure_count | 0 |
| preflight_checks_passed | 30 |
| preflight_checks_total | 30 |
| cache_dir | /tmp/synth_abl_cache |
| cache_hits | 0 |
| cache_misses | 80 |
| cache_writes | 80 |
| cache_size_bytes | 3662045 |

## Stratified RMSE

### By GR missingness

| protocol | branch | stratum | n_wells | n_points | global_rmse | median_well_rmse | worst_well_rmse |
|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_no_align | 50-80% | 11 | 26979 | 2.9014 | 1.5568 | 8.5023 |
| same_well_masked | ridge_baseline | 50-80% | 11 | 26979 | 2.7913 | 1.7326 | 8.0709 |
| same_well_masked | ridge_spatial_only | 50-80% | 11 | 26979 | 2.4238 | 1.3968 | 6.5925 |
| same_well_masked | ridge_align_spatial | 50-80% | 11 | 26979 | 2.3475 | 1.8090 | 6.2232 |
| same_well_masked | ridge_no_align | 5-20% | 14 | 33409 | 1.6451 | 1.1953 | 3.6033 |
| same_well_masked | ridge_baseline | 5-20% | 14 | 33409 | 1.6926 | 1.1572 | 4.1563 |
| same_well_masked | ridge_spatial_only | 5-20% | 14 | 33409 | 2.0717 | 1.2528 | 5.4995 |
| same_well_masked | ridge_align_spatial | 5-20% | 14 | 33409 | 2.1086 | 1.2262 | 5.8213 |
| same_well_masked | ridge_no_align | <5% | 13 | 31208 | 1.8591 | 1.5966 | 3.3376 |
| same_well_masked | ridge_baseline | <5% | 13 | 31208 | 2.0060 | 1.6580 | 3.4341 |
| same_well_masked | ridge_spatial_only | <5% | 13 | 31208 | 1.9376 | 1.6133 | 3.3972 |
| same_well_masked | ridge_align_spatial | <5% | 13 | 31208 | 2.0753 | 1.7626 | 3.7772 |
| same_well_masked | ridge_no_align | 20-50% | 1 | 1787 | 0.5307 | 0.5307 | 0.5307 |
| same_well_masked | ridge_baseline | 20-50% | 1 | 1787 | 0.7621 | 0.7621 | 0.7621 |
| same_well_masked | ridge_spatial_only | 20-50% | 1 | 1787 | 0.8612 | 0.8612 | 0.8612 |
| same_well_masked | ridge_align_spatial | 20-50% | 1 | 1787 | 0.6401 | 0.6401 | 0.6401 |
| same_well_masked | ridge_no_align | >80% | 1 | 1291 | 0.4148 | 0.4148 | 0.4148 |
| same_well_masked | ridge_baseline | >80% | 1 | 1291 | 0.4607 | 0.4607 | 0.4607 |
| same_well_masked | ridge_spatial_only | >80% | 1 | 1291 | 0.5381 | 0.5381 | 0.5381 |
| same_well_masked | ridge_align_spatial | >80% | 1 | 1291 | 0.5718 | 0.5718 | 0.5718 |
| unseen_well | ridge_no_align | 50-80% | 11 | 28770 | 2.1245 | 2.0371 | 3.6783 |
| unseen_well | ridge_baseline | 50-80% | 11 | 28770 | 1.8741 | 2.0467 | 2.5431 |
| unseen_well | ridge_spatial_only | 50-80% | 11 | 28770 | 1.8300 | 1.5937 | 2.5281 |
| unseen_well | ridge_align_spatial | 50-80% | 11 | 28770 | 1.9175 | 1.9607 | 2.6899 |
| unseen_well | ridge_no_align | 5-20% | 14 | 34419 | 1.5990 | 1.1501 | 3.3410 |
| unseen_well | ridge_baseline | 5-20% | 14 | 34419 | 1.5794 | 1.1891 | 3.3388 |
| unseen_well | ridge_spatial_only | 5-20% | 14 | 34419 | 1.5272 | 1.1363 | 2.9840 |
| unseen_well | ridge_align_spatial | 5-20% | 14 | 34419 | 1.5142 | 1.0549 | 3.0322 |
| unseen_well | ridge_no_align | <5% | 13 | 31261 | 1.5473 | 1.4072 | 2.4938 |
| unseen_well | ridge_baseline | <5% | 13 | 31261 | 1.5346 | 1.3124 | 2.2403 |
| unseen_well | ridge_spatial_only | <5% | 13 | 31261 | 1.4881 | 1.4774 | 2.1886 |
| unseen_well | ridge_align_spatial | <5% | 13 | 31261 | 1.4819 | 1.3073 | 2.2098 |
| unseen_well | ridge_no_align | 20-50% | 1 | 1787 | 0.6803 | 0.6803 | 0.6803 |
| unseen_well | ridge_baseline | 20-50% | 1 | 1787 | 0.8569 | 0.8569 | 0.8569 |
| unseen_well | ridge_spatial_only | 20-50% | 1 | 1787 | 0.7221 | 0.7221 | 0.7221 |
| unseen_well | ridge_align_spatial | 20-50% | 1 | 1787 | 0.9410 | 0.9410 | 0.9410 |
| unseen_well | ridge_no_align | >80% | 1 | 1291 | 3.8831 | 3.8831 | 3.8831 |
| unseen_well | ridge_baseline | >80% | 1 | 1291 | 3.7336 | 3.7336 | 3.7336 |
| unseen_well | ridge_spatial_only | >80% | 1 | 1291 | 3.1959 | 3.1959 | 3.1959 |
| unseen_well | ridge_align_spatial | >80% | 1 | 1291 | 3.0302 | 3.0302 | 3.0302 |

### By hidden suffix length

| protocol | branch | stratum | n_wells | n_points | global_rmse | median_well_rmse | worst_well_rmse |
|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_no_align | 2k-4k | 25 | 69116 | 2.3875 | 1.4727 | 8.5023 |
| same_well_masked | ridge_baseline | 2k-4k | 25 | 69116 | 2.3878 | 1.5749 | 8.0709 |
| same_well_masked | ridge_spatial_only | 2k-4k | 25 | 69116 | 2.3608 | 1.5455 | 6.5925 |
| same_well_masked | ridge_align_spatial | 2k-4k | 25 | 69116 | 2.3771 | 1.4722 | 6.2232 |
| same_well_masked | ridge_no_align | 1k-2k | 15 | 25558 | 1.1217 | 0.9282 | 1.7819 |
| same_well_masked | ridge_baseline | 1k-2k | 15 | 25558 | 1.2310 | 1.0241 | 1.9697 |
| same_well_masked | ridge_spatial_only | 1k-2k | 15 | 25558 | 1.1794 | 0.9845 | 1.6710 |
| same_well_masked | ridge_align_spatial | 1k-2k | 15 | 25558 | 1.2852 | 1.1593 | 1.9313 |
| unseen_well | ridge_no_align | 2k-4k | 25 | 69361 | 1.6882 | 1.4163 | 2.7429 |
| unseen_well | ridge_baseline | 2k-4k | 25 | 69361 | 1.6556 | 1.4160 | 2.5431 |
| unseen_well | ridge_spatial_only | 2k-4k | 25 | 69361 | 1.6477 | 1.4523 | 2.5281 |
| unseen_well | ridge_align_spatial | 2k-4k | 25 | 69361 | 1.6375 | 1.4879 | 2.5874 |
| unseen_well | ridge_no_align | 1k-2k | 14 | 24129 | 2.0218 | 1.2201 | 3.8831 |
| unseen_well | ridge_baseline | 1k-2k | 14 | 24129 | 1.7369 | 1.0153 | 3.7336 |
| unseen_well | ridge_spatial_only | 1k-2k | 14 | 24129 | 1.5953 | 1.2251 | 3.1959 |
| unseen_well | ridge_align_spatial | 1k-2k | 14 | 24129 | 1.6848 | 1.1886 | 3.0322 |
| unseen_well | ridge_no_align | >4k | 1 | 4038 | 2.0328 | 2.0328 | 2.0328 |
| unseen_well | ridge_baseline | >4k | 1 | 4038 | 2.0467 | 2.0467 | 2.0467 |
| unseen_well | ridge_spatial_only | >4k | 1 | 4038 | 1.5937 | 1.5937 | 1.5937 |
| unseen_well | ridge_align_spatial | >4k | 1 | 4038 | 1.7460 | 1.7460 | 1.7460 |

### By prefix length

| protocol | branch | stratum | n_wells | n_points | global_rmse | median_well_rmse | worst_well_rmse |
|---|---|---|---|---|---|---|---|
| same_well_masked | ridge_no_align | 1k-2k | 10 | 22978 | 1.3933 | 0.9582 | 2.4354 |
| same_well_masked | ridge_baseline | 1k-2k | 10 | 22978 | 1.3793 | 0.9851 | 2.2184 |
| same_well_masked | ridge_spatial_only | 1k-2k | 10 | 22978 | 1.4954 | 1.1214 | 2.2042 |
| same_well_masked | ridge_align_spatial | 1k-2k | 10 | 22978 | 1.4835 | 1.2521 | 2.2598 |
| same_well_masked | ridge_no_align | 2k-4k | 16 | 37407 | 2.8740 | 1.7754 | 8.5023 |
| same_well_masked | ridge_baseline | 2k-4k | 16 | 37407 | 2.9104 | 1.7383 | 8.0709 |
| same_well_masked | ridge_spatial_only | 2k-4k | 16 | 37407 | 2.8359 | 1.6500 | 6.5925 |
| same_well_masked | ridge_align_spatial | 2k-4k | 16 | 37407 | 2.9022 | 1.7684 | 6.2232 |
| same_well_masked | ridge_no_align | <1k | 14 | 34289 | 1.4544 | 1.1345 | 2.5167 |
| same_well_masked | ridge_baseline | <1k | 14 | 34289 | 1.4514 | 1.1572 | 2.4771 |
| same_well_masked | ridge_spatial_only | <1k | 14 | 34289 | 1.4136 | 1.2217 | 2.2188 |
| same_well_masked | ridge_align_spatial | <1k | 14 | 34289 | 1.3991 | 1.2240 | 2.2592 |
| unseen_well | ridge_no_align | 4k-8k | 19 | 51115 | 1.7753 | 1.8505 | 2.7429 |
| unseen_well | ridge_baseline | 4k-8k | 19 | 51115 | 1.7525 | 1.8644 | 2.5431 |
| unseen_well | ridge_spatial_only | 4k-8k | 19 | 51115 | 1.7351 | 1.6751 | 2.5281 |
| unseen_well | ridge_align_spatial | 4k-8k | 19 | 51115 | 1.7268 | 1.5320 | 2.5874 |
| unseen_well | ridge_no_align | 2k-4k | 20 | 44398 | 1.8418 | 1.3107 | 3.8831 |
| unseen_well | ridge_baseline | 2k-4k | 20 | 44398 | 1.6542 | 1.2117 | 3.7336 |
| unseen_well | ridge_spatial_only | 2k-4k | 20 | 44398 | 1.5262 | 1.2526 | 3.1959 |
| unseen_well | ridge_align_spatial | 2k-4k | 20 | 44398 | 1.5919 | 1.2667 | 3.0322 |
| unseen_well | ridge_no_align | 1k-2k | 1 | 2015 | 0.8012 | 0.8012 | 0.8012 |
| unseen_well | ridge_baseline | 1k-2k | 1 | 2015 | 0.8206 | 0.8206 | 0.8206 |
| unseen_well | ridge_spatial_only | 1k-2k | 1 | 2015 | 1.1487 | 1.1487 | 1.1487 |
| unseen_well | ridge_align_spatial | 1k-2k | 1 | 2015 | 0.9601 | 0.9601 | 0.9601 |
