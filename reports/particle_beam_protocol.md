# Particle Filter / Beam Search Feature Protocol

## Status and scope

Particle Filter (PF) and Beam Search (BS) are **candidate target-free feature
generators**. They do not replace Ridge, are not registered in `BASELINES`, and
cannot create a submission. Their first experiment is the following paired
Ridge comparison on 100 real train wells:

| Branch | Ridge base | PF features | BS features |
|---|---|---:|---:|
| A | no alignment, no spatial | no | no |
| B | same as A | yes | no |
| C | same as A | no | yes |
| D | same as A | yes | yes |

Each branch uses the same folds and scored wells. Results from
`same_well_masked` and `unseen_well` are reported separately and never
averaged. No final ensemble or submission is permitted at this stage.

## Allowed inputs

Both generators receive only `InferenceTask` and use:

- horizontal-well GR;
- Typewell GR indexed by **Typewell TVT** (the coordinate of a different well,
  not the horizontal-well target);
- MD, X, Y and Z; and
- horizontal-well `TVT_input` strictly before Prediction Start.

They must not read horizontal-well `TVT`, finite `TVT_input` at or after the
boundary, Typewell Geology, or formation markers. `InferenceTask.assert_no_target`
and the manifest whitelist enforce these constraints. Typewell Geology may be
present on a train-side task object for analysis compatibility, but neither
implementation accesses it; Test inference therefore has no train/serve
dependency on that column.

## Algorithms

### Particle Filter

A prefix-only ridge calibration maps increments of MD/X/Y/Z to visible
`TVT_input` increments. This gives a geometry transition prior. A deterministic
bank of particles then evolves along the hidden suffix. Horizontal GR is
compared with Typewell GR sampled at each candidate Typewell-TVT coordinate.
The filter exports the weighted track, shift from the visible anchor, gradient,
particle concentration, P90-P10 branch spread, path smoothness, GR mismatch,
and fallback indicator.

### Beam Search

The same target-free geometry prior seeds a bounded beam. At each decimated
row, candidate gradients branch around the geometry transition; cumulative
cost combines horizontal-GR/Typewell-GR mismatch, transition departure and
curvature. The beam exports its weighted track, shift, gradient, entropy/cost-gap
confidence, branch spread, path smoothness, GR mismatch, and fallback indicator.

The algorithms do not expose their tracks as direct predictions. Ridge may
learn to use or ignore them.

## Cross-fitting and cache safety

- The existing `make_group_folds` and `run_cross_fitted_protocol` paths are
  unchanged.
- One model is fitted only on fold-train wells and scored only on disjoint
  fold-validation wells under both protocols.
- Generator cache keys include dataset version, well ID, fold ID, protocol,
  task boundary, algorithm version, and full generator configuration.
- Cached payloads contain feature arrays and target-free diagnostics only.
  `FeatureCache` rejects target-like payload names.
- A real and masked boundary cannot share an artifact.
- The three public test IDs remain hard-blocked from the validation universe.

PF/BS do not learn cross-well target parameters. Their prefix calibration is
within-well and reads visible `TVT_input` only. “Cross-fitted” refers to the
Ridge model consuming these per-task features and to fold-scoped generation
and caching; no validation-well target can enter its own features or the model
that scores it.

## Diagnostics and failure contract

Every well and branch reports:

- mean and P10 confidence;
- mean and P90 branch spread;
- path smoothness;
- fallback status and fallback fraction;
- explicit failure reason; and
- cache-hit status.

Missing/invalid Typewell GR/TVT, no visible anchor, insufficient visible
`TVT_input`, or all-missing horizontal GR produces a finite geometry/anchor
fallback frame with confidence zero. Partial GR outage marks affected rows as
fallback rather than treating interpolated GR as observed.

## Runtime and device contract

Both generators are bounded (`64` particles; beam width `24`; stride `8` by
default), vectorised NumPy implementations. They run one well at a time and
cache compressed feature arrays. This is the CPU-safe path intended for the
Kaggle runtime budget.

`--device auto|cpu|gpu` is accepted and reported. PF/BS execute through the
same portable NumPy CPU path for all three choices, so a GPU is neither
required nor a source of different results. Other models may still use the
resource selector independently.

## First real run

With the official competition mount available:

```bash
python scripts/run_validation.py \
  --particle-filter \
  --beam-search \
  --max-wells 100 \
  --n-splits 5 \
  --expect-train 773 \
  --expect-test 3 \
  --cache-dir /kaggle/working/particle_beam_cache \
  --reports-dir /kaggle/working/particle_beam_reports \
  --device auto
```

The runner writes:

- `particle_beam_results.csv` — paired A/B/C/D scores by protocol, including
  delta from A;
- `particle_beam_wells.csv` — paired well-level metrics;
- `particle_beam_diagnostics.csv` — required confidence/spread/smoothness/
  fallback/failure fields;
- `particle_beam_failures.csv`;
- `particle_beam_ablation.md`; and
- `particle_beam_run_environment.json`.

`--max-wells` samples only from the already public-test-filtered train
universe. `--cache-dir`, `--reports-dir`, and `--device` are explicit CLI
controls; no external artifact is loaded.

## Execution status in this checkout

The requested 100-real-well run was not fabricated or replaced with synthetic
data: the official competition mount is absent in this checkout. Therefore no
PF/BS RMSE is recorded here. Run the command above on the official mount; its
schema preflight and `--expect-train/--expect-test` guards will fail before
training on a partial or wrong dataset.

## Promotion rule

The default remains branch A. A PF/BS branch may be considered for a later
baseline only after it beats A under leakage-safe validation in **both**
protocols, without an unacceptable tail, failure-rate, or runtime regression.
Until then:

- no PF/BS final ensemble;
- no PF/BS final submission; and
- no claim that a lower score from only one protocol is sufficient.
