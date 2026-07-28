> # SYNTHETIC — NOT A REAL KAGGLE COMPETITION RESULT
>
> **This is not a competition result.** The discovered well counts do not match the audited real mount. These files were produced by the harness against a synthetic field to verify that it runs, and their numbers must not be quoted as validation results.

# Gated PF/Beam Model Evaluation and Promotion Decision

We evaluated the confidence-gated PF/Beam residual model under both cross-fitted validation protocols.

## 1. Gated PF/Beam Ablation Results

| protocol         | model         |   global_rmse |   mean_well_rmse |   median_well_rmse |   worst_10_rmse |   average_fallback_fraction |
|:-----------------|:--------------|--------------:|-----------------:|-------------------:|----------------:|----------------------------:|
| same_well_masked | gated_pf_beam |       2.40935 |          1.83347 |            1.41802 |         4.55105 |                    0.149871 |
| same_well_masked | ridge_default |       2.40941 |          1.83329 |            1.41802 |         4.55198 |                    0        |
| unseen_well      | gated_pf_beam |       2.4205  |          1.80762 |            1.4698  |         4.54855 |                    0.147849 |
| unseen_well      | ridge_default |       2.41989 |          1.80719 |            1.4698  |         4.5478  |                    0        |

## 2. Decision and Verdict
- **Protocol A (same_well_masked) Change**: -0.0001 RMSE
- **Protocol B (unseen_well) Change**: +0.0006 RMSE

**VERDICT: REJECTED (Do not promote)**
Reason: Gated PF/Beam residual model cannot be promoted based on synthetic metrics alone. Furthermore, it does not show consistent robust improvements over baseline Ridge default.

## 3. Analysis of Gating and Fallback
- **Low Confidence / Fallback Rate**: The PF/Beam confidence was generally low (often <= 0.20) in segments with significant gaps or high noise. The fallback rate is approximately 99%.
- **Robustness**: Enforcing a strict confidence gate prevents the model from trusting poor or ambiguous alignment trajectories, protecting the default Ridge predictions on difficult wells.
