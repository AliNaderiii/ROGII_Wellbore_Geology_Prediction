# PF + Beam / Gated PF/Beam robustness decision

**Decision: REJECTED — preserve ridge_default.**

This report covers two related PF/Beam evaluations:

1. The **plain** PF / Beam / PF+Beam branches that always emit their
   trajectory (prior real 770-well four-branch comparison).
2. The **confidence-gated** PF/Beam residual branch evaluated in the
   completed REAL_KAGGLE_FULL GR experiment (770 wells, 0 failures).

The completed real GR experiment is the authoritative evidence for the
gated variant and is reported in detail in
`real_full_gated_model_decision.md`.

## Evidence classification — read this before quoting anything

Findings are separated into three classes and **must not be conflated**.

### A. Real Kaggle validation

Established by completed 770-well real-Kaggle runs, both protocols,
cross-fitted by well ID, zero task/fit/predict failures. Global
point-level RMSE figures below are run-owner aggregates from those runs.

### B. Synthetic verification

Harness checks under `reports/synthetic_validation/` and
`reports/synthetic_ablation/`. Banner-stamped
`SYNTHETIC — NOT A COMPETITION RESULT`. **Not competition results.** No
synthetic RMSE is used in this decision.

### C. Public leaderboard results

**PUBLIC LEADERBOARD: none.** No submission was created from the PF/Beam
or gated PF/Beam experiments, and no public-leaderboard score is claimed
or available for this branch.

## Pre-registered decision rule

1. Promote a candidate only if the global improvement is stable across
   all five folds **and** the paired confidence interval is not strongly
   against the candidate.
2. Do **not** use it as final if the improvement is caused only by a
   small number of wells (or only by long wells).
3. Preserve `ridge_default` as the fallback in every case.
4. Do not delete PF or Beam code.
5. Do not start external artifacts.
6. Do not use the direct dip-constrained alignment model.
7. Do not create a final submission from this analysis.

## Completed full 770-well run status

The full real-Kaggle 770-well GR experiment is **complete**:

- data_source: real_kaggle
- validation_scope: REAL_KAGGLE_FULL
- train wells discovered: 773
- eligible wells: 770
- wells evaluated: 770
- failures: 0

Stale "no final decision until full 770 run" placeholder language has been
removed. The decisions recorded below are final for this branch.

## Real Kaggle validation — Gated PF/Beam (authoritative, full 770)

Primary metric: **Absolute hidden-suffix TVT RMSE** (global,
point-weighted). Protocols are separate and are **never averaged**.
Failures recorded for the completed run: **0**.

| Protocol | Ridge Default | Gated PF/Beam | Delta (cand − default) | Measured Fallback Fraction |
|---|---:|---:|---:|---:|
| `same_well_masked` | 29.4861 | 46.0597 | +16.5736 | 0.781749 |
| `unseen_well` | 14.4229 | 14.7479 | +0.3250 | 0.845084 |

Gated PF/Beam **worsens** the primary Absolute hidden-suffix TVT RMSE
under **both** protocols. The confidence gate triggers fallback on the
exact measured fractions above (0.781749 / 0.845084); fallback is
**never** described as "approximately 99%" and is not approximated.

Related per-protocol diagnostics (Mean Well RMSE, Median Well RMSE,
Worst-10 Well RMSE) are not supplied by the run-owner aggregate and are
not fabricated; those quantities must be reported separately from the
primary metric if and when they become available from a per-well
artifact.

## Real Kaggle validation — plain PF / Beam / PF+Beam (owner global RMSE)

The four-branch plain (non-gated) comparison previously reported
global-level improvements for the combined PF+Beam branch
(`ridge_particle_beam`), but per-well diagnostics were unavailable in
this checkout. With the completed gated run showing a decisive
regression even after a strict confidence gate, and given that the
plain-branch unseen-well gain was −0.004 RMSE with no per-well/fold/
bootstrap verification, the plain PF/Beam branches are likewise not
promoted. They remain available as explicit diagnostics and are not
deleted.

## Decision

**keep_as_next_candidate = `False`**

**use_as_final = `False`**

**preserve_default_fallback = `True`**

**delete_pf_beam_code = `False`**

Specifically:

- **Gated PF/Beam:** REJECTED (+16.5736 `same_well_masked`, +0.3250
  `unseen_well`; measured fallback fractions 0.781749 / 0.845084).
- **Plain PF / Beam / PF+Beam:** not promoted; retained for diagnostics.
- **GR imputation:** REJECTED (see `real_full_gr_quality_analysis.md`).
- **GR quality scalar features:** REJECTED as a default (worsen
  `unseen_well`; see `real_full_gr_quality_analysis.md` and
  `real_full_gr_quality_features_ablation.csv`).
- **Active baseline:** `ridge_default`
  (`RidgeBaseline(alignment_features=False, spatial=None)`). The Ridge
  Default implementation is **not changed**.
- **Direct dip-constrained alignment:** still REJECTED
  (`src/model_status.py`).
- **External artifacts:** not used.
- **Final submission:** not created.

## Synthetic verification

Status: harness-only under `reports/synthetic_*`
(SYNTHETIC — NOT A COMPETITION RESULT). Not used for this decision.

## Public leaderboard results

Status: **none** (PUBLIC LEADERBOARD). No PF/Beam or gated-PF/Beam
submission exists; no LB number is reported.
