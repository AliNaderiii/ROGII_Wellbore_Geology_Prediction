# Dip-Constrained Alignment — Failure Analysis

**Status: REJECTED.** The direct dip-constrained GR/typewell alignment model
must not be used as a final predictor or as an ensemble branch in its current
form. This is enforced in code by `src/model_status.py` and asserted by
`tests/test_direct_alignment_rejection.py`.

---

## 0. The result being explained

Completed real validation run: **770 eligible wells**, both protocols,
cross-fitted by well ID, Ridge baseline unchanged.

| Protocol | Ridge | Dip-constrained alignment | Delta |
|---|---|---|---|
| `same_well_masked` | 29.452 | 277.654 | **+248.202 RMSE** |
| `unseen_well` | 14.441 | 96.545 | **+82.104 RMSE** |

| Protocol | Mean confidence | Fallback fraction |
|---|---|---|
| `same_well_masked` | 0.1577 | 67.5% |
| `unseen_well` | 0.3197 | 34.0% |

The two protocols are analysed separately throughout and are never averaged.

### What the aggregate numbers already prove, before any code is read

The model is a confidence blend,
`pred = w·aligned + (1−w)·fallback`, with `w = 0` on every row whose
confidence is below `min_confidence = 0.20`. Those rows are exactly the
reported fallback fraction `f`. Decomposing the total MSE and granting the
model the most generous possible assumption — that its *non-fallback* rows are
as accurate as Ridge — bounds the fallback branch from below:

| Protocol | f | Total MSE | Ridge-equivalent share | **Implied fallback-branch RMSE** |
|---|---|---|---|---|
| `same_well_masked` | 0.675 | 77,091.7 | 281.9 | **≥ 337.3 ft (11.5× Ridge)** |
| `unseen_well` | 0.340 | 9,320.9 | 137.6 | **≥ 164.4 ft (11.4× Ridge)** |

Two conclusions follow immediately and do not depend on any further analysis:

1. **The failure is in the fallback branch, not the GR correlation.** Even if
   the GR alignment were perfect on every row it touched, the fallback rows
   alone would still produce RMSE in the hundreds of feet. If the fallback had
   merely been Ridge-quality, the totals would have been 29.45 and 14.44 —
   i.e. indistinguishable from Ridge.
2. **The fallback branch is not a degradation, it is a divergence.** A model
   that gave up and returned the anchor (hold-last) would score in the same
   order of magnitude as Ridge. Producing 337 ft of error requires *actively
   extrapolating in the wrong direction*, not declining to predict.

The ratio structure confirms the mechanism. Between protocols, Ridge degrades
2.04× (14.441 → 29.452) but the alignment model degrades 2.88× (96.545 →
277.654) — and `same_well_masked` is precisely the protocol where the fallback
share doubles, from 34.0% to 67.5%. The excess degradation tracks the fallback
share, not the GR data.

### Provenance of the numbers in this document — read this first

Two kinds of number appear below, and they are **not** interchangeable:

| Kind | Source | Status |
|---|---|---|
| The RMSE / confidence / fallback figures in the two tables above, and everything derived from them in §0 and §8 | The completed real 770-well validation run | **Real competition results** |
| Every diagnostic tagged `Q1`…`Q9b` in §1–§9 | `scripts/diagnose_dip_alignment.py` on the 40-well **synthetic** field | **Mechanism evidence, not competition results** |

The synthetic diagnostics are used to establish *which mechanisms are present
and how they behave*, not to quantify the real error. That distinction is load-
bearing: the synthetic generator constructs `TVT = surface − Z` by definition,
which is precisely the relationship the dip-constrained model assumes, so the
synthetic field **flatters** this model and cannot reproduce the real failure
magnitude. Where a synthetic diagnostic shows a defect anyway — the clipped GR
gain, the 2%-of-a-cycle window, the −0.75 empirical `dTVT/dZ` — that defect is
present under conditions deliberately favourable to the model, which makes it
a lower bound on the real problem, not an upper one.

Conclusions about *magnitude* are therefore taken from the real run (§0, §8);
conclusions about *mechanism* are taken from the diagnostics. Where the two are
combined, it is stated explicitly.

**Reproduce, real mount:**

```bash
python scripts/diagnose_dip_alignment.py
```

**Reproduce, synthetic field (the numbers quoted below):**

```bash
python scripts/make_synthetic_field.py --n-train 40 --seed 0
ROGII_COMPETITION_ROOT=/tmp/rogii_synthetic/competition \
ROGII_REPORTS_DIR=reports/synthetic_ablation \
python scripts/diagnose_dip_alignment.py
```

Both write `dip_alignment_diagnostics_wells.csv` and
`dip_alignment_diagnostics_summary.csv`; every `Q`-tag below is a labelled row
in that summary. All diagnostics are computed from an `InferenceTask`; the
target is read only in the clearly marked post-prediction error block, and a
test (`test_diagnostics_never_read_the_target_into_a_feature`) enforces it.

---

## 1. Why the direct alignment trajectory fails

The fallback — `src.features.dip_constrained_prediction` — is the branch
producing the error, and it fails for a specific, structural reason.

It fits a plane to the quantity **`TVT_input + Z`** over the visible prefix,
then evaluates:

```python
surface_delta = coef[0]*(x_f - x0)/1000 + coef[1]*(y_f - y0)/1000
pred = anchor + surface_delta - (z_f - z0)
```

There are three compounding problems in those two lines.

### 1a. The Z term is subtracted with a hard-coded unit coefficient

The prediction asserts `dTVT/dZ = −1` exactly. It is not fitted, not damped,
and not bounded. This is the one modelling choice that separates this model
from the established `GeometricProjection` baseline, which *fits* the same
transfer coefficient on fold-train wells — and which
`reports/task_interpretation.md` §5 already identifies as the reason that
baseline works at all.

Measured on the synthetic field, the empirical prefix coefficient is
**−0.75, not −1.0** (`Q1b`). A 25% coefficient error is multiplied by the full
Z excursion across the predicted region. On real laterals that lever arm is
hundreds to thousands of feet — in the build section, more — so a coefficient
error of this size alone generates errors of exactly the observed magnitude.
Nothing downstream can recover it: the GR blend re-references to this track,
and the typewell clip only bounds it at the reference-section edge.

### 1b. `TVT + Z` is a difference of large numbers

The plane is fitted to `TVT + Z` but only the TVT part is wanted, and Z is
subtracted back off at evaluation time. When Z varies more than TVT — always
true in a lateral, where TVT moves feet while Z moves tens to hundreds — the
fit is driven by the wellbore's own vertical shape rather than by structure.
Any part of the Z signal the planar X/Y term cannot reproduce leaks into the
TVT prediction **one-for-one**, because the coefficient on that leak is exactly
1. Measured: `std(Z)/std(TVT) ≈ 1.19` even on the deliberately gentle synthetic
field, with a non-planar Z residual of 0.21–0.22× the entire TVT signal's
standard deviation (`Q1`).

### 1c. The prediction is a pure extrapolation of a 3-parameter plane

The plane has three degrees of freedom fitted on the prefix, then evaluated
along the hidden trajectory. The hidden region extrapolates a median of
**1.37× the fitted along-track span** under `same_well_masked` versus 0.58×
under `unseen_well` (`Q4b`) — the masked boundary is moved *earlier inside the
prefix*, so it both shortens the fitted support and lengthens the reach. A
linear extrapolation error grows linearly in that ratio, which is the direct
mechanism behind the 2.88× protocol degradation.

This is also why the failure is worst exactly where confidence is lowest: the
same short prefix that starves the plane fit also starves the GR alignment, so
the model falls back to the diverging branch precisely when that branch is at
its least reliable. The two failure modes are positively correlated, not
independent.

**Verdict:** the trajectory fails because the geometric fallback is an
unregularised, unfitted, unbounded extrapolation whose dominant term (`−1·ΔZ`)
is asserted rather than estimated. This is not a tuning problem.

---

## 2. Is the TVT + Z coordinate convention correct?

**Probably yes as a sign, but the convention is not the failure, and the code
does not verify it.**

Measured (`Q2`): `TVT + Z` is the flatter of the two candidate surfaces in
**85%** of wells under both protocols. So the sign is more often right than
wrong.

But the diagnostic also shows why this test is weak: the **median |R² margin|
between the two sign choices is only 0.041–0.087**. A three-parameter plane
with a free intercept absorbs *either* sign at high R², because the fit trades
the sign error off against the X/Y coefficients. High `dip_r2` is therefore not
evidence that the convention is right — the model reports R² ≈ 0.99 on wells
where the convention test is nearly a coin flip.

Two concrete gaps:

- The code hard-codes `+Z` and never tests `−Z`. If the mount's Z is depth-
  positive-down rather than elevation, the sign is inverted for every well
  simultaneously — and the R² diagnostic would not reveal it.
- Even with the correct sign, §1a shows the *magnitude* (the unit coefficient)
  is wrong, and that error is larger than the sign question.

**Verdict:** convention is not the primary cause. Sign should be verified
empirically on the real mount before this code is reused; it currently is not.

---

## 3. Are the dip sign and gradient direction correct?

**Yes. This part of the model is working.**

Measured (`Q3`): the projected gradient's sign agrees with the observed prefix
`dTVT/dMD` in **92.5–95%** of wells, with a **median gradient ratio of 0.971** —
i.e. the projected apparent dip is within 3% of the observed prefix dip in
magnitude, as well as correct in sign.

The candidate gradient bank is also constructed defensibly: offsets of
±0.001…±0.004 around the plane-implied gradient, clipped to ±0.04. That is a
narrow, geologically sane band that cannot select an impossible slope.

**Verdict:** dip sign and gradient direction are correct and are **not** the
cause of the failure. This component should be retained if the model is ever
rebuilt.

---

## 4. Is the local apparent dip identifiable from the visible prefix?

**The along-track component: yes. The cross-track component: no — but it does
not matter, and this is an important negative result.**

A lateral is nearly a straight line in map view, so the X/Y design matrix is
severely rank-deficient in the cross-track direction. Measured (`Q4`): median
along-track/cross-track singular ratio of **227–413**, with a median
perpendicular span of only **4–20 ft** against thousands of feet along track.
The `1e-4` ridge penalty in the solve exists to handle this and does so
correctly: it shrinks the unidentifiable cross-track coefficient rather than
inventing a large one.

The consequence is small because the hidden trajectory continues along the
*same* line, so the cross-track lever arm stays tiny. Median implied TVT error
from the entirely unidentified cross-track dip: **0.004 ft**.

**Verdict:** cross-track dip is genuinely unidentifiable, the penalty handles it
correctly, and it contributes ~0.004 ft — five orders of magnitude below the
observed error. **Not the cause.** The along-track apparent dip *is*
identifiable (see §3, ratio 0.971). The problem is not what the prefix can
identify; it is what the model does with it beyond the prefix (§1c).

---

## 5. Is GR amplitude calibration stable?

**No. This is a real, measurable defect — and it is why confidence is low.**

`calibrate_gr_to_reference` fits a robust affine map `a·gr + b` on the prefix,
with the gain clipped to `[0.2, 5.0]`.

Measured (`Q5`):

- The gain is **clipped in 50–57.5% of wells**. The clip is not a rare
  safety net; it is the common path.
- The **median raw gain is 0.139–0.198** — below the 0.2 floor, so the median
  well is clipped.
- Split-half stability across the prefix: **median ratio 1.42–1.90**, i.e. the
  gain estimated on the first half of the prefix differs from the second half
  by 42–90%. The calibration is not stationary even within a single well.

When the gain is clipped, the calibrated lateral GR is systematically
mis-scaled against the typewell. The matching cost is a level-matching MSE in
shared z-space, so a scale error translates directly into a level error, and
the returned confidence (a shape correlation at the winning path) collapses.
That is consistent with the reported mean confidence of **0.1577** under
`same_well_masked` — below the `min_confidence = 0.20` gate, which is precisely
why 67.5% of rows fall back.

**Verdict:** GR amplitude calibration is unstable and is the direct cause of
the low confidence and therefore of the high fallback fraction. It is the
*trigger*; the fallback branch (§1) is the *damage*.

---

## 6. Are Horizontal GR and Typewell GR resolutions compatible?

**No — and this is the deepest problem with the alignment concept as
implemented.**

The two logs live in different coordinates, and the mismatch is severe:

| Quantity | Measured |
|---|---|
| Alignment window | 201 rows ≈ **201 ft MD** |
| TVT traversed per window | **≈ 1.37 ft** (`Q7`) |
| Typewell grid step | 0.5 ft TVT |
| Typewell GR dominant wavelength | **≈ 70 ft TVT** (`Q6`) |
| **GR cycles visible per window** | **≈ 0.0196** (`Q6`) |

A 201 ft window crosses about **1.4 ft of section**, which is **~2% of one GR
cycle**. The window therefore sees a nearly flat piece of the reference log
with almost no variation to correlate against.

`align_window`'s own docstring anticipates exactly this and is why the cost
function is a level-matching MSE plus a continuity penalty rather than pure
NCC — pure correlation would demean a near-constant segment and prefer
spurious steep gradients. That mitigation is correct, but it does not create
information that is not there. With 2% of a cycle in view, the match is
determined by GR *level*, which is exactly the quantity §5 shows is
mis-calibrated in half the wells.

This is the fundamental tension: the horizontal well is sampled densely in MD
and sparsely in TVT, while the typewell is the opposite. Correlating them
window-by-window in MD is the wrong parameterisation.

**Verdict:** resolutions are incompatible as used. Not fixable by tuning the
window length alone (see §7).

---

## 7. Is the alignment window too short or too long?

**Both, simultaneously — which is why no single window length fixes it.**

- **Too short in TVT** (§6): 201 ft MD buys only 1.37 ft of section, ~2% of a
  GR cycle. Too little stratigraphic signal to identify a match.
- **Too long in MD**: 201 ft of lateral is long enough that the true `dTVT/dMD`
  is not constant within the window, yet the candidate bank models each window
  as a *straight* TVT path (constant offset + constant gradient). Real steering
  changes within the window are unrepresentable.
- **Search range mismatched to both**: the search half-width is **±12 ft** of
  TVT while the window physically traverses **1.37 ft**. The model is free to
  place the window anywhere in a **24 ft** band on the basis of a 1.4 ft
  observation — roughly a **17:1** ratio of freedom to evidence. That is what
  makes the match under-determined and the confidence low.

Lengthening the window increases TVT coverage but worsens the straight-path
assumption and the non-stationary calibration. Shortening it does the reverse.
The parameterisation, not the parameter, is wrong.

**Verdict:** window length is genuinely mis-specified, but retuning it will not
fix the model, because the ±12 ft search over a 1.4 ft observation and the
diverging geometric fallback both remain.

---

## 8. Is fallback behaviour dominating the result?

**Yes. This is the single largest contributor and it is quantitatively
decisive.**

- `same_well_masked`: **67.5%** of predicted rows fall back; mean confidence
  **0.1577**, below the 0.20 gate.
- `unseen_well`: **34.0%** fall back; mean confidence **0.3197**.

From §0, the fallback branch's implied RMSE is **≥ 337 ft** (masked) and
**≥ 164 ft** (unseen) — about **11.4–11.5× Ridge in both protocols**. That
consistency across two protocols with very different fallback shares is strong
evidence that the fallback branch has a stable, protocol-independent error
scale, and that the protocol difference in the *total* is driven almost
entirely by how often that branch is used.

The blend makes this worse rather than better. Confidence gates *toward* the
fallback: low confidence means weight 0 on the GR track and weight 1 on the
diverging plane. So the model's own uncertainty signal routes it into its worst
branch. A confidence measure should degrade toward something safe; here it
degrades toward something unbounded. Falling back to the **anchor**
(hold-last), or to Ridge, would have scored in Ridge's order of magnitude.

**Verdict:** fallback behaviour dominates the result. Fixing only the GR
alignment while leaving this fallback in place would not rescue the model.

---

## 9. Does the alignment output have a systematic bias?

**There is meaningful signed bias, but the error is predominantly variance —
so bias correction would not save the model.**

Measured (`Q9`): median |bias| is **65–67% of the per-well RMSE**. That is a
substantial systematic component per well, and it is consistent with §1a: a
wrong `dTVT/dZ` coefficient produces a *drift* proportional to ΔZ, which reads
as bias within a well.

However, the *mean signed* bias across wells is small (0.015–0.121 ft), so the
per-well biases have inconsistent sign and largely cancel in aggregate. There
is no global offset to subtract. The error is dominated by well-to-well
variance in the extrapolated drift, not by a common shift.

The typewell clip (`Q9b`, median clip fraction 0.0 on the synthetic field) is
not currently masking the problem, but on real wells with a diverging plane it
would pin predictions to the reference-section edge — which converts an
unbounded error into a large bounded one, not into a correct one.

**Verdict:** bias is real and per-well but not globally correctable. A constant
de-biasing step would not materially change the result.

---

## 10. Are `align_tvt`, `align_score`, `align_shift`, `align_gradient` still
useful as Ridge features?

These four are the **established NCC alignment features** (`alignment_features`
in `src/features.py`), which are a *different* code path from the rejected
dip-constrained model. Rejecting the direct model says nothing about them, so
they were tested independently — see the ablation in §11.

The distinction matters and is worth stating: as *features*, these columns are
consumed by a fitted Ridge model that can learn to down-weight them, and
`align_score` explicitly tells the model when not to trust `align_tvt`. As a
*direct predictor*, the same alignment has no such governor. That is why the
question has to be settled by the ablation rather than inferred from the direct
model's failure.

---

## 11. Ablation: Ridge with and without alignment and spatial features

Four branches through the **existing, unmodified** Ridge model. Branch B is the
current baseline and every delta is taken against it. All branches share the
same folds, are cross-fitted by well ID, and are scored on the identical well
set within each protocol.

| Branch | Alignment features | Spatial features |
|---|---|---|
| **A** `ridge_no_align` | no | no |
| **B** `ridge_baseline` (reference) | **yes** | no |
| **C** `ridge_spatial_only` | no | yes |
| **D** `ridge_align_spatial` | yes | yes |

**Run it:**

```bash
python scripts/run_feature_ablation.py --n-splits 5
```

Writes `alignment_spatial_ablation.csv`, `alignment_feature_verdict.csv` and
`alignment_spatial_ablation.md` into `REPORTS_DIR`, both protocols, deltas
against branch B.

### Status of the ablation numbers

The A/B/C/D ablation **has not yet been run against the real 770-well mount** —
the mount is not present in the environment where this analysis was written.
Rather than quote a number that was not computed, the runner is delivered and
verified end to end on the synthetic field, and this section is completed from
the real run.

The synthetic verification output is in `reports/synthetic_ablation/` and is
banner-stamped. **Those figures are not competition results** — the synthetic
generator constructs `TVT = surface − Z` by definition, which distorts exactly
the geometric relationships under test here. They establish only that the
harness runs, that the branches are paired and cross-fitted, and that the
verdict logic fires.

### Decision rule, fixed in advance

`src.ablation.alignment_feature_recommendation` applies a rule chosen before
seeing the real result:

- **Keep** the alignment features — as residual/features only, never as a
  direct predictor or ensemble branch — if and only if they lower global RMSE
  in **every** contrast (A→B and C→D) under **both** protocols.
- **Remove** them from the next baseline otherwise.

Requiring both contrasts and both protocols is deliberate: a mixed result means
the features are not carrying reliable signal, and a one-protocol improvement
is consistent with noise. On the synthetic run the rule returned
`remove_from_next_baseline` (3 of 4 contrasts against), which confirms the rule
fires; it is not a statement about the real data.

Until the real ablation is run, the Ridge baseline stays exactly as it is. The
current run is **not** authority to remove the features.

---

## Summary of causes, ranked by contribution

"real" = the 770-well run; "synth" = mechanism evidence from the synthetic
field (see the provenance note in §0).

| # | Cause | Verdict | Evidence | Source |
|---|---|---|---|---|
| 1 | Geometric fallback diverges (`−1·ΔZ` hard-coded, unfitted) | **Primary** | Implied fallback RMSE ≥ 337 / ≥ 164 ft (11.4–11.5× Ridge); empirical `dTVT/dZ` = −0.75 | real + synth |
| 2 | Fallback dominates via a confidence gate that routes *into* the bad branch | **Primary** | 67.5% / 34.0% fallback; protocol degradation tracks fallback share (2.88× vs Ridge's 2.04×) | real |
| 3 | GR/typewell resolution incompatibility (~2% of a GR cycle per window) | **Major** | 1.37 ft TVT per 201 ft window vs 70 ft wavelength | synth |
| 4 | GR amplitude calibration unstable | **Major (trigger)** | Gain clipped in 50–57.5% of wells; split-half ratio 1.42–1.90; consistent with real mean confidence 0.1577 < 0.20 gate | synth + real |
| 5 | Window/search mis-specification (±12 ft search on 1.4 ft of evidence) | **Contributing** | 17:1 freedom-to-evidence ratio | synth |
| 6 | Extrapolation reach under the masked protocol | **Contributing** | 1.37× fitted span vs 0.58× | synth |
| 7 | Per-well bias | **Minor** | 65–67% of per-well RMSE, but cancels across wells | synth |
| 8 | `TVT + Z` sign convention | **Not the cause** (unverified on real) | Correct in 85% of wells; R² margin only 0.041–0.087 | synth |
| 9 | Dip sign / gradient direction | **Correct** | 92.5–95% sign agreement; ratio 0.971 | synth |
| 10 | Cross-track dip unidentifiability | **Not the cause** | 0.004 ft implied error; penalty handles it | synth |

### If this model is ever rebuilt

Keep: the dip sign/gradient projection (§3) and the cross-track ridge penalty
(§4). Replace: the fallback must fit its `dTVT/dZ` transfer coefficient on
fold-train wells, as `GeometricProjection` already does, and low confidence
must degrade toward the anchor or toward Ridge — never toward an unbounded
extrapolation (§8). Re-parameterise: correlate in TVT rather than window-by-
window in MD (§6). None of this is authorised by this analysis; it is recorded
so the next attempt does not repeat the same three mistakes.

## What was deliberately not done

- Particle Filter and Beam Search were **not** started.
- No external artifacts were used.
- The Ridge baseline was **not** changed. The `alignment_features` switch added
  to `_LearnedBaseline` defaults to `True`, reproducing `FEATURE_COLUMNS`
  exactly; only the ablation passes `False`.
- Hidden TVT values were **not** used as features anywhere. Target values are
  read only in post-prediction validation diagnostics.
- The direct dip-constrained model remains **REJECTED** and is blocked from
  final/ensemble paths by `src.model_status.assert_not_rejected`.
