# Real 770-Well Ridge Feature Decision

**Decision status: APPLIED.** These are aggregate results from the completed
real-data A/B/C/D run supplied by the run owner. The validation protocol,
public-test isolation, feature-safety rules, and folds are unchanged.

## Completed result

Global point-level RMSE; protocols are separate and are never averaged.

| Protocol | A. Ridge, no alignment/spatial | B. + alignment | C. + spatial | D. + alignment and spatial |
|---|---:|---:|---:|---:|
| `same_well_masked` | **29.486** | 29.452 | 29.569 | 29.531 |
| `unseen_well` | **14.423** | 14.441 | 14.582 | 14.580 |

The bold cells identify the selected default, not the lowest cell in each
individual row. The pre-registered rule requires a capability to generalise
across both protocols.

## Pre-registered decision

1. **Alignment features: remove from the next default Ridge.** Adding the
   established alignment columns improves `same_well_masked` by 0.034 RMSE in
   the non-spatial contrast, but worsens `unseen_well` by 0.018. It therefore
   does not improve both protocols and fails the pre-registered promotion rule.
2. **Spatial features: remove from the next default Ridge.** Relative to the
   no-alignment default, spatial features worsen `same_well_masked` by 0.083
   and `unseen_well` by 0.159 RMSE. The with-alignment spatial branches also
   worsen both protocols relative to alignment without spatial.
3. **Selected default:** `RidgeBaseline(alignment_features=False, spatial=None)`.
4. **Capabilities retained:** established alignment remains available with
   the explicit `alignment_features=True` constructor option and
   `--alignment-features` CLI flag; spatial remains available through the
   explicit `--spatial` diagnostic flag. Their implementations are not
   deleted.
5. **Direct `dip_constrained_alignment`: REJECTED.** This decision is unchanged
   and remains enforced by `src/model_status.py`.

## Constraints carried forward

- No change to `same_well_masked` or `unseen_well` validation construction.
- No public test well in fitting, validation, tuning, or promotion.
- Horizontal-well `TVT` is supervision/scoring only, never an inference
  feature.
- Only visible-prefix `TVT_input` is allowed; hidden `TVT_input` stays NaN in
  `InferenceTask`.
- Typewell Geology and train-only formation markers stay out of Test inference.
- No external artifact was used to make or apply this decision.
