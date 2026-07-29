# Leakage-safe neural experiment protocol

Status: **implemented, but not promoted**. This document describes the first
controlled PyTorch pass. It is not a claim about the Kaggle leaderboard.

## Scope

`src/neural.py` provides three small residual sequence models:

- `neural_mlp`
- `neural_gru`
- `neural_tcn`

All predict

```text
residual = TVT - last_visible_TVT_input
prediction = last_visible_TVT_input + residual
```

`src/hybrid.py` provides an inner-GroupKFold OOF Ridge/neural blend and a
conservative Ridge-anchor gate. The existing `src.geoanchor.py` remains the
separate target-free affine/multi-branch alignment candidate; the neural pass
does not silently add alignment columns to Ridge Default.

## Leakage controls

1. `InferenceTask` is the only object passed to `predict`; it has no target
   attribute and asserts that `tvt_known` is NaN in the prediction region.
2. Sequence features use the existing manifest-cleared, alignment-free frame.
   They derive only from MD/X/Y/Z/GR, visible `TVT_input`, Typewell TVT and
   Typewell GR. Typewell Geology, formation markers, hidden TVT and full TVT
   are rejected.
3. Real hidden suffix labels and visible-prefix nested pseudo-holdout labels
   are read only in `fit` for fold-training wells.
4. Pseudo examples are contiguous suffixes, not random row masks. The example
   builder makes deterministic long-horizon and shorter nested boundaries.
5. Scalers are fit inside each fold; neural early stopping uses a deterministic
   inner well split. No random row-level split is used.
6. The public duplicate IDs (`000d7d20`, `00bbac68`, `00e12e8b`) are rejected by
   the repository hard guard in sequence construction and by the outer
   validation driver.
7. Padding is right-padded with an explicit mask. Masks are applied to the
   supervised, boundary, first-difference and second-difference losses; TCN
   blocks mask their activations after every block.
8. A failed or rejected hybrid returns the exact Ridge anchor prediction.

## Run instructions

The first pass deliberately creates no submission:

```bash
python scripts/run_neural_experiment.py \
  --expect-train 773 --expect-test 3 \
  --device auto --reports-dir /kaggle/working/neural_reports
```

The competition root is discovered by `src.paths.py`; set
`ROGII_COMPETITION_ROOT` only for local fixtures. The script loads one train
and one test well for schema/provenance smoke checks. Test wells are not used
for fitting, validation, threshold selection, blending, or calibration.

Useful bounded diagnostics:

```bash
python scripts/run_neural_experiment.py \
  --models ridge_default,neural_mlp,neural_gru,neural_tcn,ridge_neural_blend \
  --max-epochs 24 --patience 5 --device auto
```

`--models ridge_neural_gated` enables the conservative gate. It is more
expensive because its policy is fit from inner OOF predictions. `--no-pseudo`
is an explicitly labelled ablation, not the recommended production setting.

Kaggle's standard image supplies Python, NumPy, pandas, scikit-learn and
PyTorch. The code detects CUDA, uses it only when available and falls back to
CPU. Deterministic seeds and deterministic PyTorch kernels are requested.

## Required artifacts

The runner writes only diagnostic reports under its selected report directory:

- `neural_validation_results.csv`
- `neural_well_level_validation.csv`
- `neural_fold_metrics.csv`
- `neural_stratified_validation.csv`
- `neural_paired_well_deltas.csv`
- `neural_bootstrap_ci.csv`
- `neural_validation_failures.csv`
- `neural_training_reports.json` (epochs, parameter counts, scaler rows and every loss component)
- `neural_run_environment.json`
- `neural_decision_report.md`
- `feature_manifest.csv` and `feature_manifest_verification.csv`

No `submission.csv` is written. A submission is authorized only by a later,
separate promotion step after the complete real-data report passes every
criterion in the user protocol. Public LB values must not be read for that
step.

## Interpretation rule

A synthetic run is a plumbing check only. It cannot support promotion. On the
real 770-well validation run, report global RMSE, mean/median/P90/worst-10/worst
well RMSE, bias, max error, per-fold values, paired well bootstrap intervals,
GR-missingness/suffix-length/prefix-length strata, training epochs and
parameter counts, correction magnitudes, fallback/gate rates, runtime and
peak memory. If a candidate fails any promotion condition, retain it as a
diagnostic experiment and keep Ridge Default unchanged.
