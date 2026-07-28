# Section 7 — Resource Decision Table

Decisions are deliberately conservative: a resource is promoted to **USE**
only once the corresponding audit report records positive evidence, not
merely the absence of red flags. Re-run `scripts/run_all_audits.py` on Kaggle
and revise this table against the generated reports before modelling.

| Resource | Purpose | Allowed to use? | Why? | Required attribution | Potential leakage risk | Required preprocessing | Expected benefit | Final decision |
|---|---|---|---|---|---|---|---|---|
| `train/*__horizontal_well.csv` | Primary supervised signal (MD/X/Y/Z/GR → TVT) | Yes | Official competition data | None | None (labels are the intended target) | Per-well GR interpolation; preserve row order; GroupKFold by well | Core of the model | **USE** |
| `train/*__typewell.csv` | Stratigraphic reference: TVT→GR→Geology lookup | Yes | Official competition data | None | None | Interpolate GR onto a TVT grid to build `GR_vs_tw` | High — target-free prior | **USE** |
| `test/*` | Inference inputs | Yes | Official competition data | None | None | Same loader path as train; marker columns may be absent | Required | **USE** |
| `sample_submission.csv` | Output contract (ids, order, dtype) | Yes | Official | None | None | Drive the validator from it; never hardcode ids | Required | **USE** |
| Per-well `*.png` plots | Visual QC of trajectory/logs | Yes | Official | None | None | None — inspection only | Low, diagnostic | **USE ONLY FOR ANALYSIS** |
| `AI_wellbore_geology_prediction_task_en.pptx` | Target/metric/constraint definition | Yes | Official | None | None | Transcribe verbatim | High — defines correctness | **USE** |
| Formation-top marker columns (ANCC…BUDA) in train HW | Distance-to-marker features | Conditional | Prior EDA reports them **absent from the test** horizontal file — confirm in `input_inventory.md` | None | **Train-only feature → silent train/serve skew** | Must be predicted by a fold-trained structural-surface model, never consumed raw at inference | High if imputed correctly, catastrophic if not | **USE WITH CARE (imputed only)** |
| Established NCC alignment columns | Optional Ridge diagnostic | Yes, opt-in only | Real 770-well A/B/C/D: 29.486→29.452 masked but 14.423→14.441 unseen | None | Low when built from `InferenceTask`, but no cross-protocol benefit | `--alignment-features`; never default | Diagnostic only | **REMOVE FROM DEFAULT; RETAIN IMPLEMENTATION** |
| Fold-train offset-well spatial columns | Optional Ridge diagnostic | Yes, opt-in only | Real 770-well A/B/C/D: 29.486→29.569 masked and 14.423→14.582 unseen | None | Fold leakage unless donors are strictly train-only | Existing `SpatialPrior` guards plus `--spatial` | Negative in completed run | **REMOVE FROM DEFAULT; RETAIN IMPLEMENTATION** |
| Particle Filter feature generator | Candidate Ridge features | Conditional | Real 770-well run complete; PF-only unseen 14.429 (worse than default 14.423); combined PF+Beam best on aggregates but robustness gate not passed without per-well CI/fold checks in-repo | None | Hidden TVT_input or fold/cache mixing if boundary keys are wrong | Per-well/fold cache; visible prefix only; NumPy CPU path | Small / unconfirmed | **DIAGNOSTIC ONLY — NOT PROMOTED; CODE RETAINED** |
| Beam Search feature generator | Candidate Ridge features | Conditional | Real 770-well run complete; Beam-only unseen 14.432 (worse than default); see `pf_beam_real_decision.md` | None | Same as PF | Per-well/fold cache; visible prefix only; bounded beam | Small / unconfirmed | **DIAGNOSTIC ONLY — NOT PROMOTED; CODE RETAINED** |
| `ridge_particle_beam` (PF+Beam Ridge) | Next-candidate Ridge branch | No (not yet) | Owner globals: unseen 14.419 (−0.004) / masked 29.388 (−0.098); zero failures; paired CI, fold stability and concentration **not verified** in this checkout (no well-level artifact). Rule: do not keep as next candidate until those pass | None | Same as PF/Beam feature generators | `scripts/analyze_pf_beam_robustness.py` on `particle_beam_wells.csv` | Unconfirmed robustness | **NOT NEXT CANDIDATE YET; `ridge_default` REMAINS FALLBACK** |
| `datasets/phongnguyn23021656/koolbox-offline` | Offline wheel bundle | Not needed | Baseline has no missing dependency; rules confirmation pending | Package name + version + PyPI URL + license, if adopted | Low (code, not data) | `pip install --no-index --find-links` if ever adopted; keep a pure-Python fallback | None for the baseline | **DO NOT USE** |
| `datasets/fleongg/rogii-claude-models-pub` | Third-party pretrained models | Not yet | Training origin, feature schema and output semantics undocumented; cannot prove hidden labels were unused | Dataset owner + Kaggle URL, if adopted | **Unknown → treat as high** | Static pickle scan done; needs schema + provenance proof and an offline load test | Unknown | **NEEDS FURTHER REVIEW** |
| `datasets/ravaghi/wellbore-geology-prediction-artifacts` | Third-party models / OOF / calibration | Not yet | Same as above; any calibration constant fitted on test wells is disqualifying | Dataset owner + Kaggle URL, if adopted | **Unknown → treat as high** | Confirm training wells ⊆ train split before any use | Unknown | **NEEDS FURTHER REVIEW** |
| Repo `ROGII_EDA_Portfolio.html` (own prior EDA) | Domain hypotheses to re-verify | Yes | Own work in this repository | Self | None | Re-verify every quoted statistic against the freshly generated reports | Medium — hypothesis source | **USE ONLY FOR ANALYSIS** |

## Promotion criteria for the two artifact datasets

Move from *NEEDS FURTHER REVIEW* to *USE WITH ATTRIBUTION* only when all hold:

1. A README/metadata file in the dataset states the training wells, and they
   are a subset of the train split.
2. The feature schema (names **and** order) is recoverable from the artifact.
3. Loading and predicting works offline with the stock Kaggle image.
4. `external_artifact_leakage_audit.md` reports LOW risk for that file.
5. Its blended OOF score improves over the self-trained baseline under the
   same GroupKFold split.

Failing any of these, the artifact stays out of the final model.

## Standing constraints

- No third-party source code is copied without attribution.
- No competition data is redistributed in this repository.
- No private or restricted artifact is uploaded to GitHub.
- Generated reports are git-ignored; only this table and the code are tracked.
