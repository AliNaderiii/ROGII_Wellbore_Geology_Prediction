> # REAL KAGGLE FULL VALIDATION
>
> Computed from the real ROGII competition mount (773 train wells discovered,
> 770 eligible after excluding the three visible public test wells, 3 test wells).
> All eligible wells evaluated (n_wells_loaded=770, n_wells_evaluated=770).
> Failures: 0.
> Synthetic harness output lives under `reports/synthetic_validation/` and `reports/synthetic_ablation/`.
>
> data_source=real_kaggle | validation_scope=REAL_KAGGLE_FULL | n_wells_loaded=770 |
> n_wells_evaluated=770 | n_train_discovered=773 | n_test_discovered=3 |
> n_eligible=770
>
> Metric: Absolute hidden-suffix TVT RMSE — primary metric is point-weighted global RMSE
> computed on the hidden suffix where both prediction and truth are finite.
> Additional per-protocol diagnostics: Mean Well RMSE, Median Well RMSE, Worst-10 Well RMSE.
> No residual RMSE. No averaging across protocols.
> Target: Target = TVT (True Vertical Thickness). Protocol same_well_masked: truth from TVT_input masked interval inside visible prefix (simulated boundary). Protocol unseen_well: truth from TVT label on real hidden suffix [real_prediction_start, n_rows). Visible-prefix TVT_input is input; hidden TVT_input stays NaN in InferenceTask.

# Per-Well Gamma Ray (GR) Quality Bottleneck Analysis — Full 770-Well Real Run

This report documents the completed full 770-well real-Kaggle GR experiment
(`data_source=real_kaggle`, `validation_scope=REAL_KAGGLE_FULL`) covering GR
imputation and GR quality scalar features.

**Data Source:** real_kaggle

**Validation Scope:** REAL_KAGGLE_FULL — REAL KAGGLE FULL VALIDATION

**Provenance:** 773 train wells discovered; 770 eligible wells after excluding
the three visible public test wells; 770 wells loaded; 770 wells evaluated;
0 failures.

**Metric Definition:** Absolute hidden-suffix TVT RMSE — primary metric is
point-weighted global RMSE computed on the hidden suffix where both prediction
and truth are finite. Additional per-protocol diagnostics reported alongside
when available: Mean Well RMSE, Median Well RMSE, Worst-10 Well RMSE. Metrics
are kept clearly distinguished and are never averaged across protocols. No
residual RMSE.

**Target Definition:** Target = TVT (True Vertical Thickness). Protocol
`same_well_masked`: truth from `TVT_input` masked interval inside the visible
prefix (simulated boundary). Protocol `unseen_well`: truth from the `TVT`
label on the real hidden suffix `[real_prediction_start, n_rows)`.
Visible-prefix `TVT_input` is input; hidden `TVT_input` stays NaN in
`InferenceTask`.

## 1. Run status

The full 770-well GR experiment is **complete**.

- data_source: real_kaggle
- validation_scope: REAL_KAGGLE_FULL
- train wells discovered: 773
- eligible wells: 770
- wells evaluated: 770
- failures: 0

Stale placeholder language indicating "no final decision until full 770 run"
has been removed: the full run has completed and the decisions below are
final for this branch.

## 2. GR Imputation ablation (Absolute hidden-suffix TVT RMSE)

Global point-weighted RMSE over the scored hidden suffix rows. Per-well
diagnostics (Mean Well RMSE, Median Well RMSE, Worst-10 Well RMSE) are not
supplied by the run-owner aggregate and are not fabricated; only the
absolute hidden-suffix TVT RMSE (global) is reported here.

| protocol | model | global_rmse (Absolute hidden-suffix TVT RMSE) |
|---|---|---:|
| same_well_masked | Ridge Default | 29.4861 |
| same_well_masked | Ridge Imputed GR | 29.4872 |
| unseen_well | Ridge Default | 14.4229 |
| unseen_well | Ridge Imputed GR | 14.4230 |

Deltas (candidate − Ridge Default):

| protocol | Ridge Default | Ridge Imputed GR | Delta |
|---|---:|---:|---:|
| same_well_masked | 29.4861 | 29.4872 | +0.0011 |
| unseen_well | 14.4229 | 14.4230 | +0.0001 |

**Decision: GR imputation is REJECTED.** Imputing GR with the rolling /
bounded-fill scheme does not improve absolute hidden-suffix TVT RMSE on
either protocol; deltas are +0.0011 on `same_well_masked` and +0.0001 on
`unseen_well`. The Ridge Default remains unchanged and GR imputation is not
promoted.

## 3. GR quality scalar features ablation (Absolute hidden-suffix TVT RMSE)

Companion CSV: `reports/real_full_gr_quality_features_ablation.csv`.

Global point-weighted RMSE (Absolute hidden-suffix TVT RMSE). Per-well
diagnostics (Mean/Median/Worst-10 Well RMSE) are not part of the
run-owner aggregate for this comparison and are not fabricated.

| protocol | model | global_rmse (Absolute hidden-suffix TVT RMSE) |
|---|---|---:|
| same_well_masked | Ridge Default | 29.4861 |
| same_well_masked | Ridge GR Quality | 29.2005 |
| unseen_well | Ridge Default | 14.4229 |
| unseen_well | Ridge GR Quality | 14.4707 |

Deltas (candidate − Ridge Default):

| protocol | Ridge Default | Ridge GR Quality | Delta |
|---|---:|---:|---:|
| same_well_masked | 29.4861 | 29.2005 | −0.2856 |
| unseen_well | 14.4229 | 14.4707 | +0.0478 |

### Decision

**GR quality scalar features are REJECTED as a default.** The scalar GR
quality features (valid/missing fraction, longest contiguous missing gap,
prefix GR quality, number of valid segments, signal variance/stability)
improve the `same_well_masked` absolute hidden-suffix TVT RMSE by 0.2856 but
**worsen `unseen_well` absolute hidden-suffix TVT RMSE by 0.0478**. The
pre-registered promotion rule requires generalisation across both
protocols. Because `unseen_well` is the protocol that exercises
out-of-well generalisation, and the default must not regress on it,
`ridge_gr_quality` is **not** made the default.

The GR-quality feature branch remains available as an explicit diagnostic
(`RidgeWithGRQuality`) and is not deleted.

## 4. Distinction between reported RMSE quantities

The following quantities are distinct and must not be conflated when this
report is quoted:

1. **Absolute hidden-suffix TVT RMSE (global, point-weighted):** the primary
   metric. Computed as `sqrt(sum(sse) / sum(n_points))` over all scored
   hidden-suffix rows across all evaluated wells, weighting every depth
   point equally. This is the number used for promotion decisions and is
   what appears in Sections 2 and 3 above.
2. **Mean Well RMSE:** arithmetic mean of per-well RMSE across evaluated
   wells. Wells are weighted equally regardless of depth count. Not
   available from the run-owner aggregate supplied for this report and is
   not fabricated.
3. **Median Well RMSE:** median of per-well RMSE across evaluated wells.
   Not available from the run-owner aggregate supplied for this report and
   is not fabricated.
4. **Worst-10 Well RMSE:** mean RMSE of the ten wells with the highest
   per-well RMSE. Not available from the run-owner aggregate supplied for
   this report and is not fabricated.

When future runs emit Mean / Median / Worst-10 Well RMSE they must be
reported under their own headings and never substituted for the primary
Absolute hidden-suffix TVT RMSE.

## 5. Final decisions carried forward

- **GR imputation:** REJECTED.
- **GR quality scalar features:** REJECTED as a default (worsen unseen-well
  validation).
- **Ridge Default** (`RidgeBaseline(alignment_features=False, spatial=None)`)
  **remains the active baseline.**
- The Ridge Default implementation is **not changed** by this report.
- PF and Beam code are retained separately; the Gated PF/Beam decision is
  recorded in `real_full_gated_model_decision.md` and
  `synthetic_gr_experiment/gated_model_decision.md`.
- No final submission is created.
- No external artifacts are used.
