# PF + Beam failure and robustness analysis

## Evidence classification

| Class | Scope | Used for decision? |
|---|---|---|
| **A. Real Kaggle validation** | Completed 770-well PF/Beam run; owner global RMSE; well-level table when mounted | **Yes** |
| **B. Synthetic verification** | `reports/synthetic_*` harness | **No** |
| **C. Public leaderboard** | No PF/Beam submission | **No — none exist** |

## 0. What failed?

Task / fit / predict failures on the completed real run: **0**. There is no crash-level failure to debug. The question is whether the small global RMSE gain of `ridge_particle_beam` over `ridge_default` is robust enough to keep the combined branch as the next candidate.

## 1. Real Kaggle validation — global point-level RMSE

| source | validation | protocol | model | n_wells | n_failures | global_rmse | delta_vs_default |
|---|---|---|---|---|---|---|---|
| owner_aggregate | REAL KAGGLE VALIDATION | unseen_well | ridge_default | 770 | 0 | 14.4230 | 0.0000 |
| owner_aggregate | REAL KAGGLE VALIDATION | unseen_well | ridge_particle_filter | 770 | 0 | 14.4290 | 0.0060 |
| owner_aggregate | REAL KAGGLE VALIDATION | unseen_well | ridge_beam_search | 770 | 0 | 14.4320 | 0.0090 |
| owner_aggregate | REAL KAGGLE VALIDATION | unseen_well | ridge_particle_beam | 770 | 0 | 14.4190 | -0.0040 |
| owner_aggregate | REAL KAGGLE VALIDATION | same_well_masked | ridge_default | 770 | 0 | 29.4860 | 0.0000 |
| owner_aggregate | REAL KAGGLE VALIDATION | same_well_masked | ridge_particle_filter | 770 | 0 | 29.4060 | -0.0800 |
| owner_aggregate | REAL KAGGLE VALIDATION | same_well_masked | ridge_beam_search | 770 | 0 | 29.4060 | -0.0800 |
| owner_aggregate | REAL KAGGLE VALIDATION | same_well_masked | ridge_particle_beam | 770 | 0 | 29.3880 | -0.0980 |

Combined PF+Beam is best on both protocols in the owner table. Unseen-well improvement is −0.004 RMSE (14.423 → 14.419). Same-well masked improvement is −0.098 RMSE (29.486 → 29.388). PF-only and Beam-only each *worsen* unseen-well slightly (+0.006 / +0.009). No significance claim is made from aggregates alone.

## 2. Per-well RMSE delta

**Unavailable.** `particle_beam_wells.csv` was not present in this checkout. Improved/degraded well counts, mean/median well delta and worst-10 delta are not invented.

## 3. Fold-level stability

**Unavailable** without per-fold well rows. The decision rule requires stability across all five folds; that check is recorded as unmet until the well-level artifact is analysed.

## 4. Bootstrap confidence intervals

**Unavailable** without per-well SSE and point counts. The CSV `pf_beam_bootstrap_ci.csv` stores the owner observed global delta with `n_bootstrap = 0` and empty CI bounds so the absence is machine-readable.

## 5. Error by GR missingness / suffix length / prefix length

**Unavailable** without per-well covariates from the cross-fitted table.

## 6. Concentration in a few / long wells

**Unavailable.** Cannot test whether the −0.004 unseen-well gain is carried by a handful of long wells without the per-well table.

## 7. PF and Beam confidence / fallback rates

**Unavailable** in this checkout. The completed run reported zero task/fit/predict failures; generator-level confidence and fallback fractions live on `particle_beam_diagnostics.csv` from the runner and should be joined when that file is mounted.

## 8. Recorded failures

None (n_failures = 0).

## 9. Decision linkage

`keep_as_next_candidate = False`

Reasons:

- Owner-supplied global RMSE only: unseen_well delta -0.004 (14.423 → 14.419); same_well_masked delta -0.098 (29.486 → 29.388).
- Per-well table was not available in this checkout, so fold stability, paired bootstrap CI, improved/degraded well counts, and concentration cannot be verified.
- Under the pre-registered rule, ridge_particle_beam is NOT kept as the next candidate until those robustness checks pass on the cross-fitted well-level artifact.
- ridge_default remains the default and the fallback. PF/Beam code is retained. No final submission is authorised.

## 10. Synthetic verification

SYNTHETIC — NOT A COMPETITION RESULT. See `reports/synthetic_validation/` and `reports/synthetic_ablation/`. Not used here.

## 11. Public leaderboard results

PUBLIC LEADERBOARD: **no PF/Beam submission has been filed.** No LB score is available or claimed.

## 12. What was deliberately not done

- No retrain of Ridge or regeneration of PF/Beam features.
- No final submission.
- No promotion of `ridge_particle_beam` over `ridge_default`.
- No deletion of PF/Beam code.
- No external artifacts.
- No use of the rejected direct dip-constrained alignment model.
- No fabricated bootstrap CI, fold table, or improved-well percentage from aggregates alone.
- No claim of statistical significance.
