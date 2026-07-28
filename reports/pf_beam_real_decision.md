# PF + Beam robustness decision

**Decision: DO NOT keep as next candidate — preserve ridge_default.**

## Evidence classification — read this before quoting anything

Findings are separated into three classes and **must not be conflated**.

### A. Real Kaggle validation

Established by the completed 770-well PF/Beam run, both protocols, cross-fitted by well ID, zero task/fit/predict failures. Global point-level RMSE figures below are run-owner aggregates from that run (real 770-well PF/Beam validation, both protocols, cross-fitted by well ID, zero task/fit/predict failures (run-owner aggregate)).

### B. Synthetic verification

Harness checks under `reports/synthetic_validation/` and `reports/synthetic_ablation/`. Banner-stamped `SYNTHETIC — NOT A COMPETITION RESULT`. **Not competition results.** No synthetic RMSE is used in this decision.

### C. Public leaderboard results

**PUBLIC LEADERBOARD: none.** No submission was created from the PF/Beam experiment, and no public-leaderboard score is claimed or available for this branch.

## Pre-registered decision rule

1. Keep `ridge_particle_beam` as the **next candidate** only if the global improvement is stable across all five folds **and** the paired confidence interval is not strongly against the candidate.
2. Do **not** use it as final if the improvement is caused only by a small number of wells (or only by long wells).
3. Preserve `ridge_default` as the fallback in every case.
4. Do not delete PF or Beam code.
5. Do not start external artifacts.
6. Do not use the direct dip-constrained alignment model.
7. Do not create a final submission from this analysis.

## Real Kaggle validation — owner global RMSE

Protocols are separate and are **never averaged**. Failures recorded for the completed run: **0**.

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

| Protocol | ridge_default | ridge_particle_beam | delta (cand − default) |
|---|---:|---:|---:|
| `unseen_well` | 14.423 | 14.419 | -0.004 |
| `same_well_masked` | 29.486 | 29.388 | -0.098 |

Full four-branch owner table (global point-level RMSE):

| Protocol | ridge_default | +PF | +Beam | +PF+Beam |
|---|---:|---:|---:|---:|
| `unseen_well` | 14.423 | 14.429 | 14.432 | **14.419** |
| `same_well_masked` | 29.486 | 29.406 | 29.406 | **29.388** |

On the aggregates alone, the combined PF+Beam branch is the best of the four under both protocols. The unseen-well gain is very small (−0.004 RMSE). No statistical significance is claimed from these two scalars.

## Per-well / fold / bootstrap analysis

**Unavailable in this checkout.** The cross-fitted `particle_beam_wells.csv` (per-well SSE, n_points, fold, prefix/suffix length, GR missingness, PF/Beam diagnostics) was not present. Without it the following quantities cannot be computed and are **not fabricated**:

1. Per-well RMSE delta (`ridge_particle_beam − ridge_default`)
2. Number and percentage of wells improved / degraded
3. Fold-level RMSE deltas and five-fold stability
4. Bootstrap CI for the global RMSE delta
5. Paired bootstrap CI over wells
6. Mean and median well-level delta; worst-10 delta
7. Error by GR missingness / hidden suffix length / prefix length
8. PF and Beam confidence and fallback rates
9. Whether the gain is concentrated in a few long wells

The CSV companions `pf_beam_paired_well_deltas.csv`, `pf_beam_fold_deltas.csv`, and `pf_beam_bootstrap_ci.csv` record the owner global deltas and mark well-/fold-/bootstrap-level fields as `UNAVAILABLE`. Re-run:

```bash
python scripts/analyze_pf_beam_robustness.py \
  --reports-dir /path/to/particle_beam_reports
```

against the completed run's `particle_beam_wells.csv` to populate them without retraining.

## Decision

**keep_as_next_candidate = `False`**

**use_as_final = `False`**

**preserve_default_fallback = `True`**

**delete_pf_beam_code = `False`**

### Reasons

- Owner-supplied global RMSE only: unseen_well delta -0.004 (14.423 → 14.419); same_well_masked delta -0.098 (29.486 → 29.388).
- Per-well table was not available in this checkout, so fold stability, paired bootstrap CI, improved/degraded well counts, and concentration cannot be verified.
- Under the pre-registered rule, ridge_particle_beam is NOT kept as the next candidate until those robustness checks pass on the cross-fitted well-level artifact.
- ridge_default remains the default and the fallback. PF/Beam code is retained. No final submission is authorised.

### Applied outcome

- **Default predictor:** `ridge_default` (`RidgeBaseline(alignment_features=False, spatial=None)`).
- **Next candidate:** not promoted from this analysis
.
- **PF and Beam code:** retained (`src/particle_filter.py`, `src/beam_search.py`, Ridge opt-in flags).
- **Direct dip-constrained alignment:** still REJECTED (`src/model_status.py`).
- **External artifacts:** not used.
- **Final submission:** not created.

## Synthetic verification

Status: harness-only under `reports/synthetic_*` (SYNTHETIC — NOT A COMPETITION RESULT). Not used for this decision.

## Public leaderboard results

Status: **none** (PUBLIC LEADERBOARD). No PF/Beam submission exists; no LB number is reported.
