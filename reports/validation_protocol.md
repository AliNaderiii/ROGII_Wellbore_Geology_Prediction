# Validation Protocol

The protocol the validation phase implements, and the guards that enforce it.

This document is **authored** and tracked in git. The runner
(`scripts/run_validation.py`) also emits `validation_protocol_run.md` alongside
the metric tables, recording the exact parameters of a particular run. This
file describes the design; that one records the execution.

---

## 0. Why in-sample evaluation is invalid

**This section exists because the harness got it wrong once.** An early version
fitted the Protocol A models on the *same wells* it then scored them on, and
reported a LightGBM RMSE of **0.062** — a number that looked like a
breakthrough and was in fact meaningless. It is documented here rather than
quietly deleted, because the failure is subtle and easy to reintroduce.

### The mechanism

A gradient-boosted model with hundreds of trees has enough capacity to
memorise the individual trajectory of a specific well. Wells are long
(thousands of rows) and each has a distinctive structural signature. Given
features that identify the well implicitly — its X/Y position, its typewell
statistics, its prefix slope — the model can effectively learn a lookup table:
*"in this region of feature space, the answer is this particular curve."*

Score that model on the same well and you measure how well it recalls its
training set. You learn nothing about whether it can predict a well it has
never seen, which is the only thing the leaderboard tests.

### The evidence

The `INVALID_in_sample` diagnostic quantifies the gap directly (synthetic
field, 40 wells — the ratio is the point, not the magnitude):

| model | in-sample RMSE | honest unseen-well RMSE | optimism factor |
|---|---|---|---|
| lightgbm | 0.084 | 2.493 | **~30×** |
| ridge | 1.193 | 2.016 | ~1.7× |
| geom_projection | 0.945 | 0.945 | 1.0× |
| hold_last | 4.538 | 4.538 | 1.0× |

The pattern is exactly what the mechanism predicts: optimism scales with model
capacity. The parameter-free baselines have nothing to memorise and show a
factor of 1.0, which doubles as a consistency check on the diagnostic itself.
A harness that reported only the LightGBM in-sample figure would have claimed a
model 30× better than it is.

### The fix

Both protocols now route through a single function,
`src.validation.run_cross_fitted_protocol`, which fits on fold-train wells and
scores on fold-validation wells. It raises `CrossFitLeakage` if those sets
intersect. Using one driver for both protocols is deliberate: it makes it
impossible to fix the leak in one protocol and leave it in the other.

The in-sample computation is retained, but only as an explicitly named
diagnostic (`INVALID_in_sample`) that is excluded from ranking, from the
stratified tables and from model selection.

**Rule: a validation number is only meaningful if the model has never seen the
well it is scored on.** Protocol A is not an exception — see below.

---

## 1. Two protocols, because there are two ways to fail

A single validation number would conflate two independent failure modes.
Continuation skill and generalisation to an unseen well break for different
reasons, so both are measured, **reported separately, and never averaged into
one number**.

### Protocol A — same-well masked suffix (`same_well_masked`)

The prediction boundary is simulated **earlier**, inside the visible prefix.
The masked window is sized to mirror the well's real hidden suffix, clipped so
at least 200 rows of prefix survive as context. Predictions are scored **only
after the simulated Prediction Start**.

| | |
|---|---|
| Inputs | rows strictly before the simulated boundary |
| Truth | `TVT_input` on the masked window |
| `TVT` label read? | **Never** |
| Fitting | **fold-train wells only — cross-fitted by well ID** |

The name refers to where the *task* comes from: the truth is drawn from the
same well's own visible prefix, so no labels are needed. It does **not** mean
the model is fitted on that well. Those are independent choices, and conflating
them is precisely the bug described in §0.

Because it needs no labels, this protocol is the only honest way to sanity-check
behaviour on the three public test wells — the model would be scored on a
region whose answer is already public, so nothing hidden is consumed. (Even
then, the guards forbid those wells from entering any fitting or tuning step.)

### Protocol B — unseen-well GroupKFold (`unseen_well`)

| | |
|---|---|
| Split | `GroupKFold` over well IDs (default 5 folds) |
| Inputs | fold-validation well's real visible prefix |
| Truth | the **real hidden suffix**, from the `TVT` label |
| Fitting | fold-train wells only |

Every model parameter — the linear-extrapolation damping factor, the geometric
`beta`/`lambda`, the matching continuity weight, the correlation threshold,
ridge and LightGBM fits — is re-derived inside each fold from fold-train wells
alone. Nothing is selected using the wells being scored.

A well appears in exactly one fold's validation set and never spans folds; the
`Fold` dataclass asserts train/validation disjointness at construction, and a
test asserts the folds form a partition of the universe.

This is the protocol that resembles the leaderboard, so it alone decides the
ranking. **No baseline may be described as competitive on any other basis.**

## 2. Metrics

| Metric | Definition | Why it is reported |
|---|---|---|
| **Global point-level RMSE** | `sqrt(Σ SSE / Σ points)` pooled over all wells | The competition metric. Long wells dominate, exactly as on the leaderboard |
| Mean well RMSE | Unweighted mean of per-well RMSE | Every well counts equally; diverges from global when errors correlate with length |
| **Median well RMSE** | 50th percentile of per-well RMSE | The typical well, robust to the tail |
| P90 well RMSE | 90th percentile | Early warning of tail behaviour |
| **Worst-10-well RMSE** | Mean RMSE of the ten worst wells | The tail that decides a squared-error leaderboard |
| Worst single well | Max per-well RMSE, plus its ID | Names the specific failure to investigate |
| Max abs error | Largest single-point error | Detects divergence |
| Bias | Mean signed error | Detects systematic shallow/deep drift |
| Wells evaluated | Count of scored wells | Denominator for every well-level metric |
| Points evaluated | Count of scored rows, per model | Denominator for global RMSE |
| Runtime | Wall clock, total and per fold | Feasibility on the 773-well set |
| Memory | Peak RSS | Confirms one-well-resident streaming |
| Failure count | Task + fit + predict failures | A skipped well must never be silently absent |

Global RMSE decides; the rest explain *why*, and warn when a good aggregate
rests on a fragile tail.

### Stratifications

RMSE is additionally broken out by:

- **hidden suffix length** — `<500`, `500-1k`, `1k-2k`, `2k-4k`, `>4k` rows.
  Error growth with extrapolation distance is the key discriminator between
  geometric and GR-driven models.
- **GR missingness** — `<5%`, `5-20%`, `20-50%`, `50-80%`, `>80%`. The audit
  found 145 wells with high missingness (worst ≈ 80.1%); a model that is
  excellent on clean wells and catastrophic here is not deployable.
- **prefix length** — `<1k`, `1k-2k`, `2k-4k`, `4k-8k`, `>8k` rows. Short
  prefixes give a poorly determined anchor and slope.

## 3. Public test well exclusion

The three visible public test wells are

```
000d7d20    00bbac68    00e12e8b
```

`src.validation.assert_no_blocked_wells` raises `BlockedWellError` if any of
them reaches:

1. the validation universe,
2. any fold's **train** list,
3. any fold's **validation** list,
4. the model-fitting entry point,
5. the spatial prior's donor set,
6. the results tables written to disk.

`filter_blocked` strips the IDs from any candidate universe before folds are
built; the assertions then *prove* the removal worked rather than assuming it.
**The guard has no disable flag.** `run_environment.json` records
`blocked_wells_in_validation`, which must read `0`.

Test coverage: the guard is asserted for each of the three IDs individually, at
fold construction, after fold construction (injection), and for the results
table. See `tests/test_leakage_guards.py`.

## 4. Feature admission is a whitelist

Features are admitted from `reports/feature_manifest.csv`, which is generated
from `src/manifest.py` — the manifest is executable, so the document and the
enforcement cannot drift apart.

| Decision | Meaning |
|---|---|
| `USE` | May enter a model matrix |
| `USE_PREFIX_ONLY` | `TVT_input`: readable strictly before the boundary, for the anchor and prefix slopes. NaN past the boundary by construction |
| `USE_ALIGNMENT_ONLY` | `Typewell Geology`: interpretation and stacking-order checks only, not an unrestricted categorical |
| `REJECT` | The six train-only formation markers |
| `TARGET` | `TVT` |

`assert_safe_features` rejects anything not marked `USE`, **including columns
absent from the manifest**. An unaudited feature therefore cannot reach a model
by accident — the failure mode is a loud exception, not a silent inclusion.

`verify_manifest_against_data` re-checks the manifest's availability claims
against the columns actually present in the mounted train and test files, and
the runner writes the result to `feature_manifest_verification.csv`. If the
organisers ever ship markers in the test set, the claim/observation mismatch is
reported rather than silently inherited.

### Structural guarantee

Beyond the whitelist, target access is made *impossible* rather than merely
forbidden. Models receive an `InferenceTask`, a frozen dataclass that has no
target field:

```python
def predict(self, task: InferenceTask, feats=None) -> np.ndarray
```

There is nothing on the object to read. `WellTask.target` — the truth — is held
by the harness and handed only to the scorer. `InferenceTask.assert_no_target()`
additionally verifies that `tvt_known` contains no finite value inside the
prediction region, catching a mis-built boundary.

## 5. Spatial feature construction (exact method)

Applies when `--spatial` is supplied; reported in `spatial_ablation.csv`.

1. **Donors**: fold-train wells only. The prior is rebuilt from scratch inside
   every fold.
2. **Samples**: `(X, Y, TVT)` at every 25th row where TVT is known. For a
   fold-train well this is the full curve (`source="label"`); `source="prefix"`
   restricts to `TVT_input` for a strictly conservative variant.
3. **Index**: `scipy.spatial.cKDTree` on `(X, Y)`, with an exact brute-force
   fallback when SciPy is absent.
4. **Query**: for each predicted row, the `k` nearest donor samples within
   `radius` (defaults: `k=12`, 6000 ft), **excluding every sample whose well ID
   equals the queried well**. Queries run on a 25-row stride and are
   interpolated between.
5. **Reduction**: inverse-distance weights `w = 1/(d + 50 ft)` produce
   `nbr_tvt_wmean`, `nbr_tvt_std`, `nbr_dist_min`, `nbr_n`; a weighted plane fit
   gives `nbr_grad_along`; `nbr_shift = nbr_tvt_wmean − anchor`.
6. **Fold guard (leave-one-validation-well-out)**:
   `SpatialPrior.assert_disjoint(valid_ids)` raises `SpatialLeakage` before any
   prediction if a validation well appears in the donor set. Combined with the
   query-time self-exclusion below, a validation well contributes neither its
   own target values nor any of its rows to its own features.

### Why self-exclusion is applied at query time, not by refusing donors

Self-exclusion happens inside the neighbour search, for **both** fold-train and
fold-validation wells. That is deliberate. When features are built for a
fold-train well, that well legitimately sits in the prior — it is one of the
donors — and the correct protection is to drop its own samples from its own
neighbour list.

This keeps the feature definition *identical* (leave-one-well-out) during
training and validation. Without it, training rows would see a perfect
self-match at distance zero, the model would learn to depend on a signal that
cannot exist at inference, and validation would look excellent while the
leaderboard collapsed.

One implementation hazard, recorded because it failed silently: when the query
well is a donor, its own samples are by construction the *nearest* ones — they
lie on the same trajectory. A fixed over-fetch of `3k` candidates returned
nothing but self-matches, all of which were discarded, yielding zero
neighbours and all-empty features with no error raised. The fetch size must
account for the well's own sample count.

## 6. Honesty constraints on reporting

- Every number in the generated reports is computed in that run. There are no
  placeholders and no carried-forward values.
- A stage that cannot run is reported as unavailable **in words**. If LightGBM
  is not installed, the report says so and no substitute is scored in its place.
- A model that raises on a well is skipped and logged rather than silently
  filled with a fallback prediction, so a broken model cannot masquerade as a
  mediocre one.
- Prediction length is asserted against the task's row count, so a
  silently truncated prediction fails loudly instead of scoring on a subset.

## 7. What is never used as a feature

Confirmed by test, not merely by intent:

| Never in X | Enforcement |
|---|---|
| `TVT` (the target) | `InferenceTask` has no target attribute; `assert_safe_features` rejects the column name; `WellData.assert_no_target_leakage` |
| Hidden `TVT_input` | NaN past the boundary by construction; `assert_no_target()` verifies no finite value inside the prediction region |
| Train-only markers (ANCC, ASTNL, ASTNU, BUDA, EGFDL, EGFDU) | `assert_safe_features` raises; each of the six is tested individually |
| Unaudited columns | The manifest is a whitelist — unknown names are rejected |
| Validation-well targets in its own spatial features | Fold-train donors only, plus query-time self-exclusion by well ID |
| Public leaderboard results | No leaderboard value is read anywhere in the codebase |
| External pretrained artifacts | Out of scope this phase; `reports/decision_table.md` keeps them at NEEDS FURTHER REVIEW |

## 8. Scope boundary

Per the plan, this phase implements **only** the seven approved baselines.
Particle filters, beam search, DTW ensembles and external pretrained artifacts
are explicitly out of scope until these results are reviewed and approved.
