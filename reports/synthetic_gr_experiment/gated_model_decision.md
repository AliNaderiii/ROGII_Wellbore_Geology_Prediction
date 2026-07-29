> # SYNTHETIC — NOT A REAL KAGGLE COMPETITION RESULT
>
> **This is not a competition result.** The discovered well counts do not match the audited real mount (770 train discovered, 770 eligible; real mount has 773/770). These files were produced by the harness against a synthetic or partial field to verify that it runs, and their numbers must not be quoted as validation results.
>
> data_source=synthetic | validation_scope=SYNTHETIC | n_wells_loaded=770 | n_wells_evaluated=100 | n_train_discovered=770 | n_test_discovered=3 | n_eligible=770
>
> Metric: Absolute hidden-suffix TVT RMSE — primary metric is point-weighted global RMSE computed on the hidden suffix where both prediction and truth are finite. Additional per-protocol diagnostics: Mean Well RMSE, Median Well RMSE, Worst-10 Well RMSE. No residual RMSE. No averaging across protocols.
> Target: Target = TVT (True Vertical Thickness). Protocol same_well_masked: truth from TVT_input masked interval inside visible prefix (simulated boundary). Protocol unseen_well: truth from TVT label on real hidden suffix [real_prediction_start, n_rows). Visible-prefix TVT_input is input; hidden TVT_input stays NaN in InferenceTask.

# Gated PF/Beam Model Evaluation and Promotion Decision

**Data Source:** synthetic

**Validation Scope:** SYNTHETIC — SYNTHETIC — NOT A REAL KAGGLE COMPETITION RESULT

**Provenance:** 770 loaded, 100 evaluated, 770 train discovered, 3 test discovered, 770 eligible.

**Metric Definition:** Absolute hidden-suffix TVT RMSE — primary metric is point-weighted global RMSE computed on the hidden suffix where both prediction and truth are finite. Additional per-protocol diagnostics: Mean Well RMSE, Median Well RMSE, Worst-10 Well RMSE. No residual RMSE. No averaging across protocols.

**Target Definition:** Target = TVT (True Vertical Thickness). Protocol same_well_masked: truth from TVT_input masked interval inside visible prefix (simulated boundary). Protocol unseen_well: truth from TVT label on real hidden suffix [real_prediction_start, n_rows). Visible-prefix TVT_input is input; hidden TVT_input stays NaN in InferenceTask.

We evaluated the confidence-gated PF/Beam residual model under both cross-fitted validation protocols in the synthetic harness only. The numbers below are synthetic and must not be quoted as competition results.

**The full real-Kaggle 770-well GR experiment is complete** (see
`reports/real_full_gr_quality_analysis.md` and
`reports/real_full_gated_model_decision.md`). Decisions carried forward from
that completed real run:

- **GR imputation is REJECTED.**
- **GR quality scalar features are REJECTED as a default** (they worsen
  `unseen_well` validation).
- **Gated PF/Beam is REJECTED.**
- **Ridge Default remains the active baseline.**

## 1. Gated PF/Beam Ablation Results (Synthetic, Absolute TVT RMSE)

| protocol         | model         |   global_rmse |   mean_well_rmse |   median_well_rmse |   worst_10_rmse |   average_fallback_fraction |   n_wells_evaluated | data_source   | validation_scope   | metric_definition               | target_definition   |
|:-----------------|:--------------|--------------:|-----------------:|-------------------:|----------------:|----------------------------:|--------------------:|:--------------|:-------------------|:--------------------------------|:--------------------|
| same_well_masked | gated_pf_beam |       2.40935 |          1.83347 |            1.41802 |         4.55105 |                    0.149871 |                 100 | synthetic     | SYNTHETIC          | Absolute hidden-suffix TVT RMSE | TVT hidden suffix   |
| same_well_masked | ridge_default |       2.40941 |          1.83329 |            1.41802 |         4.55198 |                    0        |                 100 | synthetic     | SYNTHETIC          | Absolute hidden-suffix TVT RMSE | TVT hidden suffix   |
| unseen_well      | gated_pf_beam |       2.4205  |          1.80762 |            1.4698  |         4.54855 |                    0.147849 |                 100 | synthetic     | SYNTHETIC          | Absolute hidden-suffix TVT RMSE | TVT hidden suffix   |
| unseen_well      | ridge_default |       2.41989 |          1.80719 |            1.4698  |         4.5478  |                    0        |                 100 | synthetic     | SYNTHETIC          | Absolute hidden-suffix TVT RMSE | TVT hidden suffix   |

## 2. Decision and Verdict

- **Protocol A (same_well_masked) Change:** -0.0001 RMSE (Absolute TVT, synthetic)
- **Protocol B (unseen_well) Change:** +0.0006 RMSE (Absolute TVT, synthetic)

**VERDICT: REJECTED (Do not promote)**

Reason (synthetic harness): Gated PF/Beam residual model does not show
consistent robust improvements over baseline Ridge Default in this synthetic
harness run.

Reason (completed real 770-well run, authoritative): see
`reports/real_full_gated_model_decision.md`. Gated PF/Beam worsens the
primary Absolute hidden-suffix TVT RMSE under **both** protocols on real
data (+16.5736 on `same_well_masked`, +0.3250 on `unseen_well`) with
measured fallback fractions 0.781749 and 0.845084 respectively; it is
rejected on real evidence.

## 3. Analysis of Gating and Fallback (Measured, not approximated) — Synthetic Harness

- **Measured Fallback Fractions (synthetic harness only):**
  - same_well_masked: 0.149871
  - unseen_well: 0.147849
- **Note:** Do NOT describe fallback as approximately 99%; use measured
  values from the actual run. The full real-Kaggle run's measured fallback
  fractions are 0.781749 (`same_well_masked`) and 0.845084 (`unseen_well`)
  and are reported in `reports/real_full_gated_model_decision.md` — those
  are the real-run values.
- **Robustness:** Enforcing a strict confidence gate prevents the model
  from trusting poor or ambiguous alignment trajectories, protecting the
  default Ridge predictions on difficult wells.

## 4. Final Decision

The full 770-well real-Kaggle GR experiment has completed. The decisions
below are final for this branch:

- **GR imputation is REJECTED.**
- **GR quality scalar features are REJECTED as a default** because they
  worsen `unseen_well` validation.
- **Gated PF/Beam is REJECTED.**
- **Ridge Default remains the active baseline**; its implementation is not
  changed.
- No final submission is authorized.
- No external artifacts are used.
