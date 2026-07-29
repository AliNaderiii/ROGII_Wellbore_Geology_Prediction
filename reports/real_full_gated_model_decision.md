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

# Gated PF/Beam Model Evaluation and Promotion Decision — Full 770-Well Real Run

**Data Source:** real_kaggle

**Validation Scope:** REAL_KAGGLE_FULL — REAL KAGGLE FULL VALIDATION

**Provenance:** 773 train wells discovered; 770 eligible wells after excluding
the three visible public test wells; 770 wells loaded; 770 wells evaluated;
0 failures. The full 770-well GR experiment is **complete**.

**Metric Definition:** Absolute hidden-suffix TVT RMSE — primary metric is
point-weighted global RMSE computed on the hidden suffix where both
prediction and truth are finite. Additional per-protocol diagnostics
reported alongside when available: Mean Well RMSE, Median Well RMSE,
Worst-10 Well RMSE. Metrics are kept clearly distinguished and are never
averaged across protocols. No residual RMSE.

**Target Definition:** Target = TVT (True Vertical Thickness). Protocol
`same_well_masked`: truth from `TVT_input` masked interval inside the visible
prefix (simulated boundary). Protocol `unseen_well`: truth from the `TVT`
label on the real hidden suffix `[real_prediction_start, n_rows)`.
Visible-prefix `TVT_input` is input; hidden `TVT_input` stays NaN in
`InferenceTask`.

We evaluated the confidence-gated PF/Beam residual model under both
cross-fitted validation protocols on the full 770 eligible wells.

## 1. Gated PF/Beam Ablation Results (Absolute hidden-suffix TVT RMSE)

Primary metric (global, point-weighted). Per-well diagnostics (Mean Well
RMSE, Median Well RMSE, Worst-10 Well RMSE) are not supplied by the
run-owner aggregate and are not fabricated; only the absolute hidden-suffix
TVT RMSE (global) is reported below.

| protocol | model | global_rmse (Absolute hidden-suffix TVT RMSE) | average_fallback_fraction | n_wells_evaluated |
|---|---|---:|---:|---:|
| same_well_masked | Ridge Default | 29.4861 | 0.000000 | 770 |
| same_well_masked | Gated PF/Beam | 46.0597 | 0.781749 | 770 |
| unseen_well | Ridge Default | 14.4229 | 0.000000 | 770 |
| unseen_well | Gated PF/Beam | 14.7479 | 0.845084 | 770 |

| Protocol | Ridge Default | Gated PF/Beam | Delta (cand − default) | Measured Fallback Fraction |
|---|---:|---:|---:|---:|
| same_well_masked | 29.4861 | 46.0597 | +16.5736 | 0.781749 |
| unseen_well | 14.4229 | 14.7479 | +0.3250 | 0.845084 |

## 2. RMSE quantity clarification

The numbers above are **Absolute hidden-suffix TVT RMSE** (global,
point-weighted). The following related but distinct quantities are not
available from the run-owner aggregate supplied for this decision and are
not fabricated:

- **Mean Well RMSE:** per-well RMSE averaged across wells (equal well
  weighting).
- **Median Well RMSE:** median of per-well RMSE across wells.
- **Worst-10 Well RMSE:** mean of the 10 highest per-well RMSE values.

These three quantities must be reported separately from — and never
substituted for — the primary Absolute hidden-suffix TVT RMSE if and when
they become available from a per-well artifact.

## 3. Decision and Verdict

- **Protocol A (`same_well_masked`) change:** +16.5736 RMSE (Absolute
  hidden-suffix TVT RMSE).
- **Protocol B (`unseen_well`) change:** +0.3250 RMSE (Absolute hidden-suffix
  TVT RMSE).

**VERDICT: Gated PF/Beam is REJECTED.**

### Reasons

- Gated PF/Beam **worsens** Absolute hidden-suffix TVT RMSE under **both**
  protocols relative to Ridge Default: +16.5736 on `same_well_masked` and
  +0.3250 on `unseen_well`.
- The model triggers its fallback path on a large majority of scored rows
  under both protocols, but even with that fallback in place it still
  degrades global RMSE — the non-fallback PF/Beam trajectories are harmful
  on net.
- The pre-registered promotion rule requires stable improvement across
  protocols and forbids regressing `unseen_well`; Gated PF/Beam fails that
  rule decisively.

## 4. Analysis of Gating and Fallback (Measured, not approximated)

Fallback fractions reported below are the **exact measured**
`average_fallback_fraction` values from the full 770-well run. They are
**not** described as "approximately 99%" and are not approximated.

- **Measured fallback fractions:**
  - `same_well_masked`: **0.781749**
  - `unseen_well`: **0.845084**
- **Note:** Do NOT describe fallback as approximately 99%; use the exact
  measured values above when citing this run.
- **Interpretation:** the confidence gate allows PF/Beam to override Ridge
  on roughly 21.8% of `same_well_masked` rows and 15.5% of `unseen_well`
  rows; those overrides are responsible for the large `same_well_masked`
  degradation (+16.5736) and the `unseen_well` degradation (+0.3250).
- Enforcing the strict confidence gate already prevents trusting poor or
  ambiguous alignment trajectories on most rows, but even with that guard
  in place the model is net-harmful.

## 5. Final Decisions

- **Gated PF/Beam:** REJECTED. It is not promoted and is not used as a
  final predictor or ensemble branch.
- **GR imputation:** REJECTED (see `real_full_gr_quality_analysis.md`).
- **GR quality scalar features:** REJECTED as a default (worsen
  `unseen_well` validation; see `real_full_gr_quality_analysis.md`).
- **Ridge Default** (`RidgeBaseline(alignment_features=False,
  spatial=None)`) **remains the active baseline.**
- The Ridge Default implementation is **not changed**.
- PF and Beam source code is retained for diagnostic use; no code is
  deleted.
- No final submission is created from this analysis.
- No external artifacts are used.
