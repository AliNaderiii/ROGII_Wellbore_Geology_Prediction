# Trajectory Stack Pipeline — Working Plan (2026-08-01 session)

## Verified environment facts (this sandbox)

- No Kaggle competition mount is reachable from this sandbox (no
  `/kaggle/input`, no Kaggle credentials). All real-data claims below must
  therefore be re-measured by executing the same code on the Kaggle mount;
  nothing in this session fabricates real validation numbers.
- Python stack installed locally: numpy 2.4.6, pandas 3.0.5, scikit-learn
  1.9.0, scipy 1.17.1, lightgbm 4.7.0, catboost 1.2.10, pytest 9.1.1.
- `python -m pytest tests -q` at session start: **387 passed, 1 skipped**.

## Verified reference numbers (carried forward from repo evidence)

- Ridge Default real validation: `unseen_well` 14.422911,
  `same_well_masked` 29.486086, real Public LB 14.813 (770 eligible wells,
  770 - excluded IDs 000d7d20, 00bbac68, 00e12e8b).
- Safe F (safe alignment stage F): unseen_well 14.489825 → REJECTED.
- Gated PF/Beam (770-well real): unseen 14.7479, masked 46.0597 → REJECTED.
- Alignment features: unseen 14.441 (+0.018) → removed from default.
- Spatial features: unseen 14.582 (+0.159) → removed from default.
- A safe 45-feature tree pipeline (session brief): Ridge 14.732,
  LightGBM 15.196, CatBoost 15.022, fixed blend ~14.53 — all lose to
  14.422911 (the Ridge anchor itself degrades on the expanded matrix).

## Design decision

Do not change any validated component (`RidgeBaseline`,
`scripts/build_final_submission.py`, `src/geoanchor.py`,
`src/safe_alignment.py`). Add one new module and one new runner:
implemented arms

- A/L `ridge_default` — exact anchor and fallback (shared instance, so a
  fallback is bit-identical to the scored arm A).
- F `lgbm_residual` — LightGBM on the default 28-feature no-alignment
  matrix, residual target `tvt - anchor`, fold-specific medians,
  well-disjoint inner early-stopping holdout. Evidence arm (ungated).
- G `cat_boost_residual` — CatBoost on the same matrix/protocol.
  Evidence arm (ungated). Honest "unavailable" handling if the package is
  missing.
- H `oof_meta_stack` — inner GroupKFold(5) OOF stacking of
  {Ridge, LGBM, CatBoost} residuals with a small meta-design
  (OOF residuals + dmd + log1p_dmd), meta-Ridge with alpha tuned on
  tune-subfolds, fold kill switch (pooled sub-OOF delta), correction cap
  vs the anchor (±25 ft a-priori bound).
- D multi-scale datum scan at half-ranges {8, 15, 25} ft (a-priori grid)
  feeding consistent-shift diagnostics into the gate design rows.
- I/J/K `gated_trajectory` — Ridge anchor + guarded correction over
  candidates {pf, beam, pf_beam_mean, mb_hedged, lgbm_row, cat_row}, all
  7 gate rules from the brief, warmup ramp, correction cap, exact Ridge
  fallback on any failure.

Rules 1–7 enforced: pseudo-holdout improvement, fold-OOF confidence
threshold, branch-disagreement cap, worst-decile tail non-increase,
bounded correction, fold-OOF policy non-degradation kill switch, finiteness.

## Promotion rule (pre-registered, machine-checked)

An arm may be promoted only if, on the real mount (is_real_run = true):

1. real unseen_well global RMSE < 14.422911
2. same_well_masked delta ≤ +1% relative
3. worst-10 well RMSE delta ≤ +2% relative
4. improved folds ≥ majority of 5
5. bootstrap CI for the global delta not strongly against (2.5% bound ≤ 0)
6. gate activation ≤ 50% of wells (sanity), stack not killed in majority folds
7. no forbidden data/artifact (enforced structurally + manifest)

If none pass → Ridge Default stands; no submission is written.

## Measured sandbox smoke evidence (SYNTHETIC ONLY — not validation)

40 synthetic wells (scripts/make_synthetic_field.py, seed 7), 3 outer folds,
both protocols, boost iterations capped at 120: total **246 s**, peak per-fold
times ~14 s anchor+boosters, ~17 s meta-stack, ~22 s gate; peak RSS < 2 GB.
A 10-well/2-fold sanity run completes in ~35 s.

100 synthetic wells (seed 11), **5 outer folds, both protocols, full defaults**
(boost 400 iters, early-stop 50, 4 threads, inner 5, tune 3), 2 vCPU / 3 GB
sandbox: total **1313 s** (same_well_masked 622 s, unseen_well 690 s), peak
RSS **485 MB**, zero failures. Extrapolated real-run budget (770 wells × 5
folds × 2 protocols at defaults, Kaggle 4-core CPU): training wells per fold
grow 80 → 616 (7.7×), giving **≈ 2–6 h**; use `--path-cache` so a relaxed
rerun reuses target-free PF/Beam artifacts, and `--skip-lightgbm` /
`--skip-catboost` remain available as runtime relief valves.

Synthetic arm metrics from that 100-well run (harness sanity only — synthetic
fields are crack-driven and learnable, so directions here say **nothing**
about the real mount; `is_real_run=false` and the decision JSON correctly
reports `promoted=false`):

| arm | masked RMSE | unseen RMSE |
|---|---|---|
| ridge_default (anchor) | 3.059 | 3.115 |
| lgbm_residual | 2.512 | 2.784 |
| catboost_residual | 2.523 | 2.796 |
| oof_meta_stack | 2.518 | 2.498 |
| gated_trajectory | 2.904 | 3.349 |

Boosters and the meta-stack mechanically improve over their anchor on the
synthetic field (intended mechanism direction); the gate fires on 25–80 % of
wells per fold and lands slightly worse than the anchor here — exactly why
promotion is decided only by the pre-registered 8-rule gate on the real
mount, never from synthetic numbers.

Guard behaviour measured in the sandbox: killed gates and stacks return the
bit-identical Ridge anchor output; the submission builder refuses synthetic
decisions and universe-mismatched decisions; a fabricated real-format
decision drives the full 9031-row contract path with validation PASS.

## Execution plan for the Kaggle session (outside this sandbox)

1. `pip install lightgbm catboost` (stock Kaggle image has both). On a GPU
   session add `--device gpu` to steps 2-4 (or leave the `auto` default,
   which probes the GPU and falls back to CPU with a logged reason); see the
   GPU runbook in `AUDIT.md` for the exact commands and the five device keys
   written to `run_environment.json`.
2. Smoke: `python scripts/run_trajectory_stack_experiment.py --max-wells 100`
   (validates runtime/memory; writes synthetic_* only — 100 wells is not the
   real universe).
3. Real: `python scripts/run_trajectory_stack_experiment.py --expect-wells 770`
   (5 folds × 2 protocols; writes `real_trajectory_stack_*` + decision JSON).
4. Only if the decision JSON says promoted:
   `python scripts/build_gated_submission.py --require-promotion <json>`,
   producing `/kaggle/working/final_submission/submission.csv`,
   `/kaggle/working/submission.csv` and `submission_audit.json`.
   Otherwise run the validated `scripts/build_final_submission.py`
   (Ridge Default).
5. Hidden-test finale: rerun step 4 without `--expect-test 3` (IDs and row
   counts are derived from the active `sample_submission.csv` only).
