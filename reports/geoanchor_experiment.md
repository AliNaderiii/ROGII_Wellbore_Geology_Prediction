# GeoAnchor Controlled Experiment — Protocol and Pre-Registration

This document is **authored** and tracked in git, like
`reports/validation_protocol.md`. It fixes the design of the controlled
GeoAnchor experiment (arms A–E) *before* results are inspected, records which
public-notebook ideas are being exercised and why, and states the decision
rule used in the run's generated decision file. Machine-generated outputs of
a run live next to it only as `real_geoanchor_*` (real mount) or
`synthetic_geoanchor_*` (everything else) files, banner-stamped accordingly.

## 1. What is studied, and where it comes from

Four public Kaggle notebooks for the ROGII Wellbore Geology Prediction
competition were studied **for ideas only** (read July 2026; all four share a
common "Contact-Gated Stratigraphic Alignment" lineage, and the fourth adds a
meta-layer on top):

| Notebook | Study focus |
|---|---|
| `ROGII GeoAnchor` (lucifer19) | anchor + guarded corrections; GR heel calibration; bimodal hedge |
| `hahaha det agi` (tamerlanomralinov) | same lineage, visualised; uncertainty-aware alignment narrative |
| `rogii-shift-275` (zhexinjiang) | small bounded final moves; visible-prefix calibration profiles |
| `Well-Level GBDT Gate` (blacklions) | well-level GBDT meta-gate over OOF diagnostics; anchor restore on any failed check |

No notebook code is copied, and none of their artifacts, pretrained models,
mounted datasets or Koolbox utilities are imported or loaded. The following
idea patterns — and only these — were re-implemented from scratch inside this
repository's leakage-safe architecture:

1. **Affine calibration of horizontal GR into Typewell GR space.**
   On the visible prefix, fit `G_hw ≈ α·G_tw(TVT_input) + β` by least squares
   (min 40 rows; a-priori sanity bounds `0.25 ≤ α ≤ 4`, `|β| ≤ 500`), map the
   hidden GR back through `((G−β)/α − μ_tw)/σ_tw`. Implemented in
   `src/geoanchor.py::fit_prefix_affine_calibration`.
2. **Multi-branch GR/Typewell alignment.** A bounded constant-datum scan
   (`Δ ∈ [−15, +15] ft`, step 0.75) of the calibrated hidden GR against the
   typewell GR, using clipped residuals scaled by a prefix-derived robust
   scale, seeded at the hold-last anchor. Implemented in
   `src/geoanchor.py::multibranch_scan`.
3. **Bimodal branch uncertainty features.** When the scan has two separated
   (`≥3 ft`) minima within 10% cost of each other, expose branch separation,
   normalised cost gap, and an effective branch probability shrunk toward 0.5
   by a prefix-trust diagnostic (`p_eff = τ·p_scan + (1−τ)·½`).
4. **Prefix pseudo-holdout validation.** Corrections are validated
   *empirically* on a nested masked window strictly inside the visible prefix
   whose truth is `TVT_input` — never a hidden label
   (`src/geoanchor.py::nested_pseudo_task`, mirroring the established
   `src/tasks.py` masked construction).
5. **Well-level confidence gating.** A well-level GBDT
   (`HistGradientBoostingRegressor`, depth 3, L2 1.0) predicts, per
   (well, candidate), the expected RMSE gain of applying a candidate
   correction, from target-free well diagnostics
   (`src/geoanchor.py::WellLevelGate`).
6. **Ridge Default as the stable anchor prediction.** Arm E predicts the
   Ridge Default trajectory whenever the gate declines, and arm A (Ridge
   Default) is the experiment's reference. Nothing here changes
   `src/model_status.py` — no model is promoted by this experiment.
7. **PF/Beam as optional candidate corrections only.** The repository's
   existing target-free `pf_shift`/`beam_shift` tracks are offered as bounded
   (`±25 ft`, the established GR/typewell search radius) corrections to the
   anchor, never as predictions or ensemble branches in their own right
   (`src/geoanchor.py::generate_candidate_corrections`).

### Explicitly not used (and why they are out of scope)

- Train target lookup for Test (any same-ID "the answer is in train" move).
- Public duplicate-well shortcuts / same-well contact reconstruction read
  from a labelled duplicate well.
- Typewell Geology in Test inference (train-only column; the manifest forbids
  it — all new features are registered in `src/manifest.py` with provenance
  walked back to `MD, X, Y, Z, GR, TVT_input, Typewell TVT, Typewell GR`).
- Formation markers (absent from Test).
- External artifacts, mounted models, Koolbox.
- Public leaderboard scores as training labels or tuning targets.
- Hidden TVT as a feature.
- Hardcoded public-LB-tuned weights (e.g. per-well manual shifts or notebook
  spacing constants). Every numeric constant in `src/geoanchor.py` is an
  a-priori sanity bound stated at its definition; every *decision* parameter
  (gate margin, alignment-confidence threshold, disagreement cap) is tuned
  per fold from fold-training wells only, via OOF diagnostics.

## 2. Arms

| Arm | Name | Contents |
|---|---|---|
| A | `ridge_default` | Ridge Default, unchanged (the anchor) |
| B | `ridge_affine_cal` | A + affine-calibration features (idea 1) |
| C | `ridge_multibranch` | A + multi-branch/bimodal features (ideas 2–3) |
| D | `ridge_affine_multibranch` | A + B + C feature sets |
| E | `ridge_gated_gbdt` | A + well-level GBDT gate over bounded PF/Beam candidate corrections (ideas 4, 5, 7) |

## 3. Validation design

- Both protocols of the repository: `same_well_masked` (truth from
  `TVT_input`) and `unseen_well` (truth from the `TVT` label of held-out
  wells). Reported separately, never averaged.
- GroupKFold over well IDs (default 5 folds, seed 0) shared across all arms.
- **Strict cross-fitting everywhere.** A fold's arm models are fitted on
  fold-train wells only; the harness raises `CrossFitLeakage` on overlap.
  Within arm E, the gate is trained only from fold-training wells: an inner
  GroupKFold (5 folds) cross-fits Ridge Default over the training wells so
  every training well's pseudo-holdout is scored by a model that never saw
  it, producing the OOF prefix diagnostics the gate learns from. Gate
  threshold tuning adds a further 3-way sub-split of the same training wells.
- The three visible public test wells are hard-blocked (unchanged guard in
  `src/validation.py`), and the gate raises if a validation well reaches its
  `fit`.

### Gate application rules (arm E)

A candidate correction is applied to a fold-validation well only when **all**
of the following hold; otherwise the well receives the Ridge Default
prediction and the decision is logged:

1. **Pseudo-holdout improvement** — the candidate beats the anchor on the
   well's own nested visible-prefix pseudo-holdout (target-free).
2. **Alignment confidence** — candidate confidence ≥ the fold-OOF-tuned
   threshold.
3. **Branch disagreement acceptable** — PF-vs-Beam track disagreement (or the
   GR-scan branch separation when only one family is available) ≤ the
   fold-OOF-tuned cap.
4. **Worst-tail risk does not increase** — top-decile pseudo-holdout squared
   error of the candidate ≤ the anchor's.
5. **Fold non-degradation (kill switch)** — the tuned policy must improve the
   pooled OOF metric over the anchor on the *fold-training* wells; otherwise
   the gate is disabled for the entire fold (all wells fall back, and the
   fallback records the kill reason).

## 4. Metrics reported

Global point-level RMSE; mean/median well RMSE; P90; worst-10 RMSE; worst
single well; max abs error; bias; fold stability (per-fold pooled RMSE and
delta-sign consistency); per-well improved/degraded/unchanged counts vs
Ridge Default; well-cluster bootstrap confidence intervals for the delta vs
Ridge Default (global-RMSE and mean-well-RMSE scales); GR missingness and
hidden-suffix-length stratifications; gate activation rate, gate fallback
rate and per-rule fallback counts; runtime seconds; peak RSS memory. No
submission is created.

## 5. Pre-registered decision rule

An arm is **CARRIED_for_real_mount_confirmation** only when, versus Ridge
Default, under **both** protocols:

1. global point-level RMSE improves;
2. the well-cluster bootstrap CI for the global delta is not strongly against
   the arm (2.5% bound below zero — the arm is not confidently worse);
3. neither median nor worst-10 well RMSE degrades materially (>2%);
4. the delta is fold-stable (arm not worse on every fold);
5. (arm E) the gate activated on at least one and at most half of the wells.

CARRIED is not promotion: it only licenses a confirmation run on the real
competition mount. Ridge Default remains the fallback in every case, and the
`src/model_status.py` registry is untouched (the REJECTED direct alignment
model is not used anywhere in this experiment; PF/Beam remain candidate
feature generators).

## 6. Honesty constraints

- Every reported number is computed in the run from well-level results;
  nothing is carried forward, and no leaderboard information exists anywhere
  in the pipeline.
- Report files are named `real_geoanchor_*` **only** when the discovered well
  counts match the audited real mount (773 train / 770 eligible). Every other
  run writes `synthetic_geoanchor_*` files stamped
  `SYNTHETIC — NOT A REAL KAGGLE COMPETITION RESULT`.
- A stage that cannot run is reported in words (e.g. killed gates list their
  kill reason); a failed model fold falls back to the Ridge Default anchor
  loudly in the gate log rather than disappearing from the tables.
