# Neural phase 1 status

Date of implementation: 2026-07-29 (UTC).

## What changed

Implemented a small, leakage-safe PyTorch track in `src/neural.py`, an
inner-OOF Ridge/neural blend and conservative gate in `src/hybrid.py`, a
cross-fitted experiment runner at `scripts/run_neural_experiment.py`, and
structural tests in `tests/test_neural_safety.py`.

Ridge Default remains immutable: `RidgeBaseline(alpha=10.0,
alignment_features=False, spatial=None)` is still the fallback and the
comparison anchor.

## Data and execution status

The initial checkout did not have the competition mount at
`/kaggle/input/competitions/rogii-wellbore-geology-prediction`, so a complete
real 773-well / 3-test-well experiment could not be run in this environment.
The system Python initially had neither PyTorch nor the scientific Python
stack. A local ignored virtual environment was used only to run tests and a
synthetic plumbing check; it is not a repository artifact and contains no
competition data.

The real Kaggle run remains authorized through:

```bash
python scripts/run_neural_experiment.py --expect-train 773 --expect-test 3
```

## Checks

- Existing suite before changes: **359 passed, 1 skipped**.
- Existing suite plus neural safety tests after changes: **365 passed, 1
  skipped**.
- Synthetic neural runner smoke test: completed for Ridge Default, MLP, GRU
  and TCN; both protocols and paired/bootstrap report paths were exercised.
- Public duplicate IDs are hard-rejected in neural sequence construction and
  outer validation.
- Typewell Geology and horizontal TVT are excluded from neural inference
  features.
- Nested pseudo-holdouts are contiguous and prefix-only.
- Padding, deterministic tiny GRU output, finite prediction and exact Ridge
  fallback tests pass.
- No submission was created.

## Comparison against the verified Ridge reference

The supplied completed real-data reference remains **Ridge Default: 14.4229
unseen_well and 29.4861 same_well_masked**. This phase produced no new real-data
neural score, so there is no scientifically valid real candidate delta to
report and the first promotion criterion is untested.

On one 12-well synthetic fixture with one training epoch and a deliberately
small sequence cap, the GRU and MLP/TCN diagnostic models were worse than the
synthetic Ridge anchor. This is not competition evidence and is not used for
selection. It is a useful warning that a neural residual must earn promotion
under the full real protocol rather than being assumed beneficial.

## Decision

**Rejected for promotion pending the real experiment.** The implementation is
accepted as a diagnostic candidate, not as a production model. No Public LB
score was used, no test target was used, and no claim of top-leaderboard or
top-three performance is made.

## Next scientifically justified step

Run the complete real experiment with fixed seeds and both protocols. Inspect
fold stability, tail and missing-GR strata, paired well bootstrap intervals,
loss-component scales, gate activation/fallback, correction magnitudes and
runtime. Promote only if the stated promotion rule is met; otherwise preserve
Ridge Default and investigate alignment-specific candidates separately.
