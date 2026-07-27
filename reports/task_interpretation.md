# Task Interpretation

What the ROGII wellbore geology problem actually is, in physical terms, and
what each of its features means for how a model must be built. Every dataset
figure quoted here comes from the completed feature audit on the real Kaggle
data; nothing is estimated.

---

## 1. TVT as a geological trajectory

**TVT (True Vertical Thickness) is not a depth. It is a position within the
rock column.**

A well has several depth-like coordinates and they answer different questions:

| Coordinate | Question it answers |
|---|---|
| MD | How much pipe is in the hole? |
| TVD / Z | How far below the surface is the bit? |
| **TVT** | **Where is the bit inside the stratigraphic section?** |

TVT measures the bit's offset from a stratigraphic reference surface, measured
perpendicular to bedding. A constant TVT means the bit is tracking a single
bed: it may be climbing or dropping thousands of feet in Z while staying in the
same rock. That is precisely what a geosteering crew is trying to achieve.

So the target sequence `TVT(MD)` is a **trajectory through stratigraphy**, and
this has three consequences that shape everything downstream:

1. **It is continuous.** Rock does not teleport. Except at faults, TVT is a
   smooth function of MD. Predictions that jump discontinuously between
   adjacent rows are physically impossible, whatever the metric says.
2. **It is anchored.** At the last visible row the answer is known exactly.
   Every prediction is a *continuation* of a known state, not a fresh estimate.
   This is why all models here predict the residual `TVT - TVT_last`: it makes
   hold-last the zero prediction, so any learned signal is measured as an
   improvement over doing nothing.
3. **Its increments are the physics.** The quantity with a mechanical
   explanation is `dTVT/dMD` — the difference between where the bit is going
   and where the rock is going. Modelling the level directly discards that.

## 2. Horizontal well and Typewell: the inverse problem

Each well ships as a pair of files, and the pairing is the heart of the task.

**The horizontal well** is the lateral being drilled. It records MD, position
(X, Y, Z) and a gamma-ray log along a path that is nearly parallel to bedding.
Because it runs *along* the layers rather than through them, it may spend
5,000 ft inside a single formation. Its GR log therefore reads as a slowly
varying signal punctuated by excursions whenever the bit drifts across a bed
boundary.

**The typewell** is a nearby vertical (or near-vertical) reference well. It
penetrates the entire section perpendicular to bedding, so its GR log is a
complete stratigraphic *fingerprint*: `Typewell GR` as a function of
`Typewell TVT`, labelled by `Typewell Geology`. Audit: **1,567,045 typewell
rows across 773 wells, and every well has a typewell file** — the reference is
never missing.

This gives the task its structure:

> The typewell tells you what the rock **looks like** at every stratigraphic
> position. The lateral tells you what the rock **looks like right now**.
> Predicting TVT means inverting the first with the second.

Formally, given a reference log `g(TVT)` and an observation `GR_obs` at some
MD, find the `TVT` such that `g(TVT) ≈ GR_obs`. It is a **log-correlation
inverse problem**, not a generic tabular regression — and it is ill-posed
pointwise, because `g` is not injective: many depths in a section share the
same GR value. That is why the solution must be a *trajectory* constrained by
continuity, and why single-point matching is the weakest of the GR baselines.

Two practical corrections are mandatory before the two logs can be compared at
all, both implemented in `calibrate_gr_to_reference`:

- **Amplitude.** The lateral and the typewell are different tools in different
  holes. Their GR values are not on a common scale. A robust affine map is
  fitted on the visible prefix, where TVT is known and the correct reference
  sample can therefore be looked up.
- **Depth.** Residual bias between the matched track and the known prefix TVT
  is removed as a constant.

Both read the prefix exclusively. Neither can see the hidden region.

## 3. GR-based stratigraphic alignment

Alignment is what a geosteerer does by hand: slide the lateral's GR response
along the type section until the wiggles line up.

Mechanically, for a window of the lateral log we search candidate TVT paths and
score how well the reference log sampled along that path reproduces what was
measured. The implementation searches a 2-D family — a constant **offset** plus
a **gradient** (`dTVT/dMD`) — because a horizontal well crossing dipping beds
produces a log that is not merely shifted but *stretched*.

Three engineering findings from building this, each of which changed the
result materially:

**Correlation is the wrong scoring function here.** Normalized cross-correlation
demeans both series. Over a 200 ft window a horizontal well may cross barely a
foot of section, so both the observation and the candidate are nearly constant,
and the correlation of two flat lines is noise. Worse, the search then *prefers*
steep spurious gradients purely because they manufacture variance. Scoring on
level-matching MSE in calibrated GR space, with a continuity penalty, fixed a
divergence in which the alignment wandered ±150 ft away from a true range of
just 36 ft.

**Confidence must mean "is this pick distinctive?", not "do the shapes
correlate?"** The reported `align_score` is the improvement of the winning
misfit over the typical misfit across the search window. It approaches 1 at a
sharp marker bed that pins the answer, and collapses toward 0 in a featureless
interval where many depths explain the data equally well. It is further scaled
by the fraction of the window that was actually *measured*, so a match made
entirely of interpolated GR is correctly reported as worthless.

**Trust the movement, not the level.** The alignment's absolute level inherits
calibration error, but its *increment* since the boundary is the trustworthy
part — and the level at the boundary is already known exactly from the anchor.
Re-referencing the track to the anchor and adding only its increment improved
the NCC baseline's global RMSE by roughly a quarter in the synthetic harness.

## 4. Increasing, decreasing and constant TVT

The sign of `dTVT/dMD` is a direct readout of the drilling situation, and the
three regimes have different error characteristics:

| Regime | Physical meaning | Modelling consequence |
|---|---|---|
| **Constant** | Bit is holding the target zone; wellbore and bedding are parallel | The majority regime. Hold-last is near-optimal, and this is why hold-last is a genuinely strong baseline rather than a straw man |
| **Increasing** | Bit is moving *up-section* relative to bedding — either climbing, or the formation is dropping away beneath it | Persistent while a steering correction is underway; the slope is autocorrelated over hundreds of feet |
| **Decreasing** | Bit is moving *down-section* | Same, mirrored |

Two failure modes follow directly, and both are visible in the baseline
results:

- **Trend persistence is real but must be damped.** A slope measured over the
  last 300 ft genuinely predicts the next few hundred feet, but extrapolated
  over 3,000 ft it diverges badly, because steering corrections are applied
  precisely to *stop* the trend. `LinearExtrapolation` therefore fits a damping
  factor on fold-train wells rather than trusting the raw slope.
- **Faults break continuity.** A fault produces a genuine step change of a few
  feet. No smooth model predicts it, and it will dominate the error on affected
  wells. This is an irreducible error floor, not a bug to be tuned away.

## 5. Geological dip and azimuth

Bedding is a surface with an orientation: **dip** (steepest tilt) and **dip
azimuth** (compass direction of that tilt). The well cuts through this surface
along its own heading, and what matters is not true dip but **apparent dip** —
the component of dip resolved along the wellbore's direction:

```
apparent_dip ≈ true_dip × cos(well_azimuth − dip_azimuth)
```

Two wells in the same field with identical true dip experience *opposite*
apparent dip if drilled in opposing directions. A well drilled exactly along
strike (perpendicular to dip azimuth) sees near-zero apparent dip and holds
constant TVT with no steering effort at all.

This is why `heading_sin` / `heading_cos` are in the feature set rather than a
raw azimuth angle: the relationship is trigonometric and periodic, and a
tree-based model splitting on degrees would have to rediscover that the values
359 and 1 are adjacent.

It also motivates the decomposition used by `GeometricProjection`:

```
TVT(md) = TVT_anchor − (dZ(md) − dip × dMD)
```

The bit's vertical movement `dZ` is **known** past the prediction boundary —
the trajectory is surveyed before the geology is interpreted. So the only
unknown is the structural term. The model estimates apparent dip from the
prefix as `d(TVT + Z)/dMD` and fits the transfer coefficient on fold-train
wells. This is the cheapest physically-grounded model available, and in the
synthetic harness it is the strongest baseline by a wide margin. That margin is
partly an artefact of how the synthetic field is generated (TVT is constructed
as `surface − Z`), which is exactly why the real-mount run is the one that
decides — see the caveat in §10.

## 6. Offset-well spatial priors

Geological structure is spatially correlated. Two wells 500 ft apart on the
same pad penetrate nearly the same surface. So a well's TVT at map position
(X, Y) is informative about a *different* well's TVT at the same position —
and, critically, **X and Y are known for the entire hidden suffix**.

This is a legitimate and valuable prior, but it is also the single easiest
place in this task to leak, so the construction is stated exactly:

- Donors are **fold-train wells only**, rebuilt from scratch inside every fold.
- The queried well's own samples are **always excluded by well ID**, so the
  feature is leave-one-well-out whether the well is a donor (fold-train) or not
  (fold-validation). This matters: if training rows could see their own well's
  samples, the model would learn to rely on a self-match that cannot exist at
  inference, and validation would look excellent while the leaderboard
  collapsed.
- A fold-level guard (`assert_disjoint`) additionally refuses to run if any
  validation well appears in the donor set.

A subtle bug worth recording: the first implementation under-fetched
neighbours. When the query well is itself a donor, its own samples are by
construction the nearest ones — they lie on the same trajectory — so a fixed
over-fetch returned nothing but self-matches, which were then all discarded,
silently yielding zero neighbours and all-empty features. The fetch size must
account for the well's own sample count.

## 7. The long hidden suffix

The prediction region is not a short gap. It is a **long extrapolation** with a
one-sided constraint: known state on the left, nothing on the right.

Audit findings that define the shape of the problem: **5,092,255 train
horizontal rows**, MD monotonic at exactly **1 ft steps**, no duplicates, and a
**clean prefix/suffix structure with no internal `TVT_input` gaps**. So the
boundary is unambiguous and every predicted row is a fixed physical distance
from the anchor.

Error necessarily grows with distance from the anchor, and the growth rate is
what distinguishes the models:

- Hold-last has **zero** error at the boundary and grows with however far the
  formation moves.
- Linear extrapolation is better near the anchor and worse far away — the
  classic bias/variance crossover, and the reason for fitted damping.
- **GR alignment is the only family whose error does not necessarily grow with
  distance**, because it consumes a fresh measurement at every row. That single
  property is the strategic argument for investing in alignment: geometry
  extrapolates, but only GR *observes*.

This is why RMSE is reported stratified by hidden suffix length. A model that
wins overall while being worse in the `>4k` stratum is the wrong model to build
on, and the aggregate number alone would hide that.

## 8. GR missingness

Audit: **145 wells have high GR missingness, worst ≈ 80.1%**. Gaps are
contiguous tool outages, not random dropout.

Consequences:

1. **Interpolate within a well only.** Per-well tool calibration differs, so a
   global fill mixes incompatible baselines. `interpolate_within_well` is
   per-well by construction.
2. **Interpolated GR is not evidence.** Filling a 700 ft gap produces a smooth
   line that correlates against *something* in the type section. The
   `gr_is_missing` flag and the measurement-fraction scaling of `align_score`
   exist so that a model can tell the difference between an observation and an
   inference.
3. **Graceful degradation is a hard requirement.** On an 80%-missing well the
   GR models must fall back to geometry rather than emit confident nonsense.
   Verified: on the two synthetic wells with 74% and 80% missing GR, the
   alignment confidence correctly collapses to 0.00, which routes the
   prediction back to the anchor.
4. **Report by stratum.** A model that is excellent on clean wells and
   catastrophic on the 145 GR-poor wells may still look fine on average and
   will not be robust. `stratified_validation.csv` breaks RMSE out by GR
   missingness for exactly this reason.

## 9. Global point-level RMSE

The competition metric is RMSE over **all predicted points pooled**, not the
mean of per-well RMSEs. The difference is not cosmetic:

```
global = sqrt( Σ_wells Σ_rows (pred − true)²  /  Σ_wells n_rows )
```

- **Long wells dominate.** A 6,000-row well contributes six times the weight of
  a 1,000-row well. Optimising mean well RMSE optimises the wrong objective.
- **It is quadratic in error.** One well with 30 ft of error contributes as much
  as nine hundred wells with 1 ft. The tail *is* the score.
- **It is unbounded above.** A single diverging well can dominate the
  leaderboard, which is why every model here clips to the typewell's physical
  TVT range and degrades to the anchor when unconfident.

Because a single number can hide all of this, the harness reports global RMSE
alongside mean, median, P90, worst-10 and worst-single well RMSE, plus the
three stratifications. Global RMSE decides; the others explain *why*, and warn
when a good aggregate is built on a fragile tail.

## 10. What this implies for model selection

Ranking the baselines is not the goal — understanding *where each one breaks*
is, because that determines what the next model should do.

- **Hold-last** is the honesty benchmark. Any model that cannot beat it has
  learned nothing.
- **Geometry** (`GeometricProjection`) uses `dZ`, which is genuinely known past
  the boundary. It is cheap, robust, and independent of GR — so it is the
  correct fallback for the 145 GR-poor wells.
- **GR alignment** is the only family that ingests new information inside the
  hidden region. It should therefore dominate at long suffix lengths *provided*
  GR is present, and must be gated on confidence where it is not.
- **Learned models** (ridge, LightGBM) do not replace the physics; they arbitrate
  between it. Their most valuable input is `align_score`, because that is what
  lets them learn *when to trust the correlation*.

**One caveat on the synthetic numbers.** The end-to-end results produced in
this sandbox come from a synthetic field (`scripts/make_synthetic_field.py`),
because the Kaggle mount is not available here. In that field TVT is generated
as `surface − Z`, which hands `GeometricProjection` a nearly exact relationship
and almost certainly overstates it. Those numbers prove the *harness* is
correct — the guards fire, the folds are clean, the models rank and run — and
nothing more. They live under `reports/synthetic_validation/`, banner-stamped,
and are never mixed with real results. The real ranking is whatever
`scripts/run_validation.py` produces on the real mount.

**And one on validation itself.** Both protocols are cross-fitted by well ID: a
model is never fitted on a well it is scored on. An earlier version of the
harness violated this in the masked protocol and reported a LightGBM RMSE of
0.062 that was pure memorisation. The in-sample diagnostic retained in
`reports/validation_protocol.md` §0 measures that illusion at roughly **30×**
for LightGBM and **1.0×** for the parameter-free baselines — capacity is
exactly what gets inflated. No baseline may be called competitive on anything
but Protocol B.
