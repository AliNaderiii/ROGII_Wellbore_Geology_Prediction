# Safe alignment staged experiment — pre-registration & execution protocol

Status: **pre-registered, awaiting real-mount execution.**
No real competition numbers exist for this pipeline yet; nothing in this
document is a leaderboard claim. The verified baseline remains:

| Item | Value |
|---|---|
| Model | Ridge Default (`alignment_features=False`, no spatial) |
| Public LB | **14.813** |
| Internal `unseen_well` RMSE | **14.4229** |
| Internal `same_well_masked` RMSE | **29.4861** |
| Eligible training wells | 770 (773 discovered − 3 blocked public IDs) |
| Blocked public duplicate IDs | `000d7d20`, `00bbac68`, `00e12e8b` |
| Rollback submission | `scripts/build_final_submission.py` (Ridge Default) |

## 1. Architecture

Implemented in `src/safe_alignment.py`, driven by
`scripts/run_safe_alignment_experiment.py`. Six paired stages, each a strict
superset of the previous, each falling back to the **bit-exact Ridge Default
prediction** (the stages share one fitted `RidgeBaseline` instance per fold)
whenever any guard declines or any component fails:

| Stage | Name | Adds |
|---|---|---|
| A | `ridge_default` | the anchor itself (identical instance = exact fallback) |
| B | `safe_b_anchor_blend` | bounded PF/Beam candidate corrections (±25 ft cap), selected by a nested visible-prefix pseudo-holdout |
| C | `safe_c_affine_cal` | heel affine GR calibration `GR_hw ≈ α·GR_tw(TVT_input)+β` (α∈[0.25,4], |β|≤500, ≥40 prefix rows); failed/low-quality fits shrink the correction (×0.5) and halve the cap instead of feeding raw affine values to Ridge |
| D | `safe_d_branch_guard` | multi-branch datum scan (±15 ft, 0.75 ft step), a trust-shrunk hedged branch candidate, and an ambiguity guard that halves + sep-caps corrections when two branches are near-tied (`cost_gap < 0.05`, `sep > 6 ft`) |
| E | `safe_e_projection` | robust IRLS (Huber, c=1.345, deg ≤ 2) projection in the target-free coordinate `U = candidate_TVT + Z − anchor`; movement bounded to ±5 ft; unstable fits (>50 % rows clipped) rejected |
| F | `safe_f_verified` | multi-cut visible-prefix self-verification (cuts 0.50/0.65/0.75 of the prefix + the mirrored nested cut; ≥2 valid cuts must improve), worst-decile tail guard, PF/Beam disagreement cap (10 ft), and a fold-train-tuned confidence threshold + warmup ramp (grids {0, 0.25, 0.5} × {100, 200, 400} rows) |
| G | OOF residual GBDT | **gated off** — runs only if LightGBM/CatBoost is already present *and* stages B–F show real evidence; never an external booster |

Warmup: corrections ramp linearly from `1/warmup` at the first suffix row to
1.0, so no abrupt first-row jump is possible (tested).

## 2. Leakage contract

Inference inputs: MD, X, Y, Z, GR (+missingness), visible `TVT_input`
prefix, Typewell TVT, Typewell GR — nothing else. Enforced structurally:

* `InferenceTask` carries no target; `assert_no_target()` on every boundary,
  including every pseudo cut (`pseudo_task_at` re-asserts).
* `make_group_folds` / `Fold.__post_init__` raise on any blocked public ID;
  the three duplicate wells can never enter training, tuning, OOF,
  calibration or threshold selection.
* The manifest whitelist (`assert_manifest_valid`,
  `assert_inference_provenance`) runs before any model is fitted.
* The only tuned decisions (stage-F confidence threshold, warmup length) are
  selected from **fold-training wells only** over fixed a-priori grids.
  Nothing is tuned on the public LB or on test wells.
* No Koolbox, external artifacts, formation markers, Typewell Geology, or
  duplicate-target lookup anywhere in the pipeline.

Tests: `tests/test_safe_alignment.py` (22 tests: blocked-well guards,
pseudo-cut construction, bit-exact fallback under missing typewell / all-GR
missing / internal exception, correction cap, warmup shape, projection
stability, decision logging).

## 3. Validation design

Both protocols, never averaged together: `same_well_masked` (truth from
`TVT_input`, boundary moved inside the prefix) and `unseen_well`
(GroupKFold by well, truth from the TVT label). Rapid discovery = 3 folds,
seed 0, PF/Beam memoized per boundary. Confirmation = 5 folds + bootstrap
CIs + paired per-well deltas + GR-missingness/suffix-length strata + fold
stability. Reports are evidence-named: `real_safe_alignment_*` only when
discovery matches the audited 773/770 mount.

## 4. Kaggle execution instructions

```bash
# 0. one-time sanity (visible public smoke only; never for the hidden rerun)
python scripts/smoke_test_loader.py --expect-train 773 --expect-test 3

# 1. rapid discovery (3 folds, one seed) — target < ~40 min on Kaggle CPU
python scripts/run_safe_alignment_experiment.py \
    --expect-wells 770 --n-splits 3 --seed 0 \
    --reports-dir /kaggle/working/reports

# 2. confirmation of the most promising stage only (5 folds, full bootstrap)
python scripts/run_safe_alignment_experiment.py \
    --expect-wells 770 --n-splits 5 --seed 0 \
    --stages ridge_default safe_f_verified \
    --n-bootstrap 2000 --reports-dir /kaggle/working/reports
```

Outputs (per run): `*_summary.csv`, `*_well_level.csv`,
`*_paired_well_deltas.csv`, `*_improved_degraded.csv`, `*_fold_stability.csv`,
`*_bootstrap_ci.csv`, `*_stratified.csv`, `*_activation_rates.csv`,
`*_fallback_reasons.csv`, `*_decisions.csv`, `*_failures.csv`,
`*_fold_records.csv`, `*_run_environment.json` (runtime + peak RSS),
`*_report.md`. The experiment **never writes a submission**.

## 5. Promotion rule (pre-registered, unchanged from the mandate)

A stage may be considered for one real submission only if, on the real
770-well run:

1. `unseen_well` global RMSE improves over **14.4229**;
2. `same_well_masked` does not materially degrade (>2 % tolerance rule);
3. worst-10 well RMSE does not materially degrade;
4. the improvement appears in most folds;
5. bootstrap evidence is not strongly against it;
6. no forbidden feature/artifact is used (manifest + tests);
7. stable across GR-missingness and suffix-length strata;
8. every correction demonstrably falls back to exact Ridge.

If any criterion fails: **submit Ridge Default** (public LB 14.813). The
final submission builder for a promoted stage may be created only after the
real run passes this rule; until then `scripts/build_final_submission.py`
(Ridge Default) is the only submission path and the rollback.

## 6. Synthetic harness verification (NOT a competition result)

⚠️ SYNTHETIC — pipeline verification only, 40-well synthetic field
(`scripts/make_synthetic_field.py`), 3 folds, seed 0, 83 s runtime.

* All six stages ran in both protocols with zero fit/predict failures.
* Stage F activation: 17.5 % (masked) / 27.5 % (unseen); all other wells
  returned bit-exact Ridge, with per-reason fallback counts recorded.
* Stage F `unseen_well`: global RMSE 2.936 vs Ridge 2.961; mean-well delta
  −0.204 with 97 % of bootstrap mass in favour (CI −0.428 … +0.024).
* Stage F `same_well_masked`: +0.011 global (within noise), tail guard and
  disagreement cap observed firing.
* Unguarded stages B/C degraded the unseen global RMSE on this field —
  which is exactly why the stage-F guards exist and why promotion requires
  the guarded stage, not the raw blend.

These numbers verify the machinery only. Real evidence requires the
770-well Kaggle run above.
