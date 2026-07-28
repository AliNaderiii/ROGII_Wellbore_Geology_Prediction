# Dip-Constrained Alignment — Failure Analysis

**Status: REJECTED.** The direct dip-constrained GR/typewell alignment model
must not be used as a final predictor or as an ensemble branch in its current
form. This is enforced in code by `src/model_status.py` and asserted by
`tests/test_direct_alignment_rejection.py`.

---

## Evidence classification — read this before quoting anything

Findings are separated into three classes and **must not be conflated**. Class
B is *not* a root cause; it is a mechanism hypothesis awaiting real-data
confirmation.

### A. Confirmed from real Kaggle data

Established by the completed 770-well run, both protocols, cross-fitted by
well ID.

| # | Finding | Evidence |
|---|---|---|
| A1 | The direct model is decisively worse than Ridge under **both** protocols | +248.202 RMSE (`same_well_masked`), +82.104 RMSE (`unseen_well`) |
| A2 | Mean alignment confidence is low, and below the 0.20 gate on `same_well_masked` | 0.1577 / 0.3197 |
| A3 | The fallback path dominates, and dominates far more under the masked protocol | 67.5% / 34.0% of predicted rows |
| A4 | The failure is concentrated in the **fallback branch**, not the GR correlation | Implied fallback RMSE ≥ 337 ft / ≥ 164 ft — 11.4–11.5× Ridge under both protocols (derivation in §0) |
| A5 | The masked protocol degrades this model ~1.4× more than it degrades Ridge, tracking the fallback share | 2.88× vs Ridge's 2.04× |

A4 and A5 are arithmetic consequences of A1–A3 and the model's own blend rule.
They require no assumption about the code and are therefore also class A.

### B. Hypotheses supported only by synthetic diagnostics

Measured on the 40-well synthetic field via
`scripts/diagnose_dip_alignment.py`. **None of these is a confirmed real-data
root cause.** The synthetic generator builds `TVT = surface − Z` by definition
— exactly the relationship this model assumes — so it *flatters* the model.
A defect visible there is a lower bound on the real problem, but its real
magnitude, and in some cases whether it is present at all, is unverified.

| # | Hypothesis | Synthetic measurement | Status |
|---|---|---|---|
| B1 | The hard-coded unit `dTVT/dZ` coefficient is the dominant error term | empirical coefficient −0.75, not −1.0 | **UNCONFIRMED — the leading hypothesis, not a finding** |
| B2 | `TVT + Z` is a difference of large numbers, leaking non-planar Z into TVT 1:1 | `std(Z)/std(TVT)` ≈ 1.19; non-planar Z residual 0.21–0.22× the TVT signal | Unconfirmed |
| B3 | GR amplitude calibration is unstable and clips | gain clipped in 50–57.5% of wells; split-half ratio 1.42–1.90 | Unconfirmed, but *consistent with* the real A2 |
| B4 | GR/typewell resolutions are incompatible | ~1.37 ft TVT per 201 ft window vs a ~70 ft GR wavelength → ~2% of a cycle | Unconfirmed |
| B5 | The ±12 ft search is mismatched to ~1.4 ft of evidence | ~17:1 freedom-to-evidence ratio | Unconfirmed |
| B6 | Extrapolation reach explains the protocol gap | 1.37× fitted span (masked) vs 0.58× (unseen) | Unconfirmed, but *consistent with* the real A5 |
| B7 | Dip sign and gradient direction are **correct** | 92.5–95% sign agreement; median ratio 0.971 | Unconfirmed (a negative result — needs real confirmation before relying on it) |
| B8 | Cross-track dip is unidentifiable but harmless | 0.004 ft implied TVT error | Unconfirmed |
| B9 | The `TVT + Z` sign convention is usually right but weakly determined | correct in 85% of wells; R² margin only 0.041–0.087 | Unconfirmed |

### C. Unresolved questions

| # | Question | What would settle it |
|---|---|---|
| C1 | Is B1 true on real data? Is the empirical real `dTVT/dZ` materially different from −1? | Fold-safe real diagnostic: fit the coefficient on fold-train prefixes, evaluate on held-out wells, compare against −1 (see §12) |
| C2 | What is the real `std(Z)/std(TVT)` ratio, and does non-planar Z genuinely dominate? | Same real diagnostic run |
| C3 | Is the real Z column elevation (negative-down) or depth (positive-down)? The code hard-codes `+Z` and never tests it. | Real per-well comparison of `std(TVT+Z)` vs `std(TVT−Z)`, reported per well rather than in aggregate |
| C4 | What is the real typewell GR wavelength and the real TVT traversed per window? | Real diagnostic; the synthetic 70 ft wavelength is an artefact of the generator |
| C5 | Which of B1–B6 dominates the real error, and in what proportion? | Requires an ablation *of the fallback itself* — deliberately not run, since the model is rejected |
| C6 | Would replacing the fallback with a fitted-coefficient projection (or with Ridge) rescue the model? | A separate real-data experiment, required before any un-rejection |
| C7 | Is the real failure uniform across wells, or driven by a subpopulation (e.g. high-curvature or build-section wells)? | Per-well real diagnostics joined to the real well-level results |

**Bottom line.** What is *confirmed* is that the model fails, that it fails
through its fallback branch, and that the fallback branch is ~11.5× worse than
Ridge. *Why* the fallback diverges is, at present, a hypothesis (B1/B2) with
supporting synthetic evidence and no real-data confirmation.

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

### 1a. The Z term is subtracted with a hard-coded unit coefficient — **hypothesis B1, UNCONFIRMED**

> **This subsection is the leading hypothesis, not a confirmed real-data root
> cause.** The code fact is verifiable by reading the source; the *magnitude of
> its contribution on real data* is not yet measured. See C1.

The **code fact** (verifiable, not a measurement): the prediction asserts
`dTVT/dZ = −1` exactly. It is not fitted, not damped, and not bounded. This is
the one modelling choice separating this model from the established
`GeometricProjection` baseline, which *fits* the same transfer coefficient on
fold-train wells — and which `reports/task_interpretation.md` §5 already
identifies as the reason that baseline works at all.

The **synthetic measurement** (class B): the empirical prefix coefficient is
−0.75, not −1.0 (`Q1b`). *If* a comparable discrepancy holds on real data, the
error is multiplied by the full Z excursion across the predicted region — on
real laterals hundreds to thousands of feet, more in the build section — and
would generate errors of the observed order. Nothing downstream could recover
it: the GR blend re-references to this track, and the typewell clip only bounds
it at the reference-section edge.

**What is not established:** that the real empirical coefficient differs from
−1 at all, or by how much. The synthetic field's coefficient is a property of
its generator. §12 specifies the fold-safe real diagnostic that would settle
this; until it runs, B1 must not be reported as the cause.

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

**Run it on Kaggle — validation pass, then the full run:**

```bash
# 1. 100-well pass: validates runtime, memory and output shape
python scripts/run_feature_ablation.py \
  --n-splits 5 \
  --max-wells 100 \
  --cache-dir /kaggle/working/feature_ablation_cache \
  --reports-dir /kaggle/working/feature_ablation_reports

# 2. full run: all 770 eligible wells
python scripts/run_feature_ablation.py \
  --n-splits 5 \
  --cache-dir /kaggle/working/feature_ablation_cache \
  --reports-dir /kaggle/working/feature_ablation_reports \
  --expect-wells 770
```

`--expect-wells 770` is optional but recommended for the full run: it aborts
before fitting anything if the mount is partial, rather than silently
producing a smaller, non-comparable result.

Writes the six required reports into `--reports-dir`, both protocols, deltas
against branch B, plus `real_ablation_preflight.md` (the per-branch leakage
check) and `real_ablation_run_environment.json` (runtime, peak RSS, cache
statistics, device, well counts).

### Status of the ablation numbers — NOT YET RUN ON REAL DATA

The A/B/C/D ablation **has not been run against the real 770-well mount.** The
Kaggle mount is not present in this environment: `/kaggle/input/...` does not
exist, there is no network access, and no Kaggle credentials are configured.
Every attempt to reach the data was made and is recorded.

Rather than quote a number that was not computed, the runner is delivered,
both required command forms are verified to execute end to end, and this
section will be completed from the real run's output.

**The banner is enforced in code, not by convention.**
`src.real_ablation_reporting.is_real_run` grants the `REAL KAGGLE VALIDATION`
header only when the discovered counts match the audited mount exactly (773
train wells, 770 eligible). Any other run — synthetic, partial, subset — is
stamped `SYNTHETIC — NOT A COMPETITION RESULT` instead. A synthetic run
therefore *cannot* be mislabelled as real, even by passing the wrong flag,
and `test_synthetic_run_is_never_labelled_real` asserts it.

The synthetic verification output is in `reports/synthetic_ablation/` and is
banner-stamped. **Those figures are not competition results** — the synthetic
generator constructs `TVT = surface − Z` by definition, which distorts exactly
the geometric relationships under test here. They establish only that the
harness runs, that the branches are paired and cross-fitted, and that the
verdict logic fires.

### Pre-registered decision rule

`src.ablation.preregistered_decision` / `preregistered_verdict` implement a
rule fixed **before** any real result was inspected:

**Alignment features** (contrasts A→B and C→D) — keep in the next baseline
only if they improve global RMSE in **both** protocols **and** do not
materially degrade median or worst-10 well RMSE. "Material" is a
pre-registered 2% of the branch-B value (`MATERIAL_DEGRADATION_TOLERANCE`).
Otherwise remove.

**Spatial features** (contrasts A→C and B→D) — keep if they improve the global
metric **or** give a consistent worst-well improvement across both protocols,
without material degradation elsewhere and without unacceptable runtime or
leakage risk. Otherwise remove.

**Direct `dip_constrained_alignment`** — never promoted to a final prediction
branch unless a *separate* real-data experiment proves it beats Ridge. It is
currently REJECTED and blocked in code.

Two design choices are deliberate. Requiring both protocols means a
one-protocol improvement — which is consistent with noise — is not sufficient.
Adding the median/worst-10 guard means a branch cannot be kept on a global-RMSE
win that was bought by inflating the tail, which is the failure mode a
point-weighted metric hides. The spatial clause is deliberately *looser*: it
admits a worst-well-only justification, because tail behaviour is the thing an
offset-well prior is most likely to help.

The rule and its tolerance are unit-tested against synthetic fixtures covering
keep, remove, one-protocol-only, tail-degradation and worst-well-only cases, so
the decision logic is verified independently of any data.

**Until the real ablation runs, the Ridge baseline stays exactly as it is.**
No synthetic result is authority to add or remove a feature.

---

## Summary of causes, ranked by contribution

**Class A = confirmed on real data. Class B = synthetic hypothesis, unconfirmed.**
Only rows 1 and 2 are class A; every "cause" below them is a candidate
explanation, not an established one.

| # | Cause | Verdict | Evidence | Class |
|---|---|---|---|---|
| 1 | The failure is concentrated in the fallback branch | **CONFIRMED** | Implied fallback RMSE ≥ 337 / ≥ 164 ft (11.4–11.5× Ridge) | **A** |
| 2 | Fallback dominates via a confidence gate that routes *into* the bad branch | **CONFIRMED** | 67.5% / 34.0% fallback; protocol degradation tracks fallback share (2.88× vs Ridge's 2.04×) | **A** |
| 3 | *Why* the fallback diverges: hard-coded `−1·ΔZ`, unfitted | **HYPOTHESIS (B1)** | Code fact + synthetic `dTVT/dZ` = −0.75 | B |
| 4 | `TVT + Z` is a difference of large numbers | **HYPOTHESIS (B2)** | `std(Z)/std(TVT)` ≈ 1.19 (synthetic) | B |
| 5 | GR amplitude calibration unstable | **HYPOTHESIS (B3)**, consistent with real A2 | Gain clipped in 50–57.5% of wells; split-half ratio 1.42–1.90 | B |
| 6 | GR/typewell resolution incompatibility (~2% of a GR cycle per window) | **HYPOTHESIS (B4)** | 1.37 ft TVT per 201 ft window vs 70 ft wavelength | B |
| 7 | Window/search mis-specification (±12 ft search on 1.4 ft of evidence) | **HYPOTHESIS (B5)** | 17:1 freedom-to-evidence ratio | B |
| 8 | Extrapolation reach under the masked protocol | **HYPOTHESIS (B6)**, consistent with real A5 | 1.37× fitted span vs 0.58× | B |
| 9 | Dip sign / gradient direction are correct | **HYPOTHESIS (B7)** | 92.5–95% sign agreement; ratio 0.971 | B |
| 10 | Cross-track dip unidentifiable but harmless | **HYPOTHESIS (B8)** | 0.004 ft implied error | B |
| 11 | `TVT + Z` sign convention | **HYPOTHESIS (B9)**, weakly determined | Correct in 85% of wells; R² margin only 0.041–0.087 | B |

### If this model is ever rebuilt

Only after C1–C6 are settled on real data. Provisionally: keep the dip
sign/gradient projection (B7) and the cross-track ridge penalty (B8). Replace
the fallback so it *fits* its `dTVT/dZ` transfer coefficient on fold-train
wells, as `GeometricProjection` already does, and so low confidence degrades
toward the anchor or toward Ridge rather than toward an unbounded extrapolation
(A4/A2). Re-parameterise the correlation to work in TVT rather than
window-by-window in MD (B4). None of this is authorised by this analysis; it is
recorded so the next attempt does not repeat the same mistakes.

---

## 12. Fold-safe real diagnostic for the Z coefficient, dip sign and convention

Specification for the experiment that would move B1, B2, B7 and B9 from class B
to class A. It is **not** part of the production model and its output must be
reported separately.

**Constraints, all enforced structurally:**

1. **Never use hidden TVT as an inference feature.** The coefficient is fitted
   on the *visible prefix only* (`tvt_known[:start]`), which is `TVT_input` and
   is NaN past the boundary by construction. `InferenceTask` has no `target`
   attribute, so a diagnostic that tried to read the label would raise.
2. **Fold-safe.** Fit the transfer coefficient on fold-**train** wells and
   evaluate it on held-out wells, using the same `make_group_folds` splits as
   the ablation. A coefficient fitted per-well and evaluated on that same well
   would be in-sample and would overstate its quality.
3. **Target used only after prediction.** The held-out `TVT` is read solely to
   score the already-formed prediction, in the same clearly-marked
   post-prediction block `scripts/diagnose_dip_alignment.py` already uses, and
   is asserted by `test_diagnostics_never_read_the_target_into_a_feature`.
4. **Reported separately** from the production model, in its own file, and
   never merged into the Ridge baseline decision.

**Measurements, per protocol:**

| Quantity | Method | Settles |
|---|---|---|
| Real empirical `dTVT/dZ` | Regress prefix `TVT_input − anchor` on `Z − Z_anchor`, pooled over fold-train wells; report the distribution and compare with the asserted −1 | C1 / B1 |
| Contribution of the coefficient error | Score the fallback with the *fitted* coefficient vs the hard-coded −1 on held-out wells; the RMSE difference is B1's real contribution | C1, C5 |
| Real `std(Z)/std(TVT)` and non-planar Z residual | Per-well, prefix only | C2 / B2 |
| Z sign convention | Per-well `std(TVT+Z)` vs `std(TVT−Z)`, reported as a distribution rather than an aggregate, since a single global answer would hide a mixed convention | C3 / B9 |
| Real typewell GR wavelength and TVT-per-window | FFT of the typewell GR on its TVT grid; prefix `dTVT/dMD` × window length | C4 / B4 |
| Dip sign agreement | Projected gradient vs observed prefix `dTVT/dMD`, sign and ratio | B7 |

`scripts/diagnose_dip_alignment.py` already computes the per-well quantities
(`Q1`, `Q1b`, `Q2`, `Q3`, `Q6`, `Q7`) target-free; what it does **not** yet do
is the fold-safe fitted-vs-hardcoded comparison, which is the part that would
actually confirm B1. That comparison is deliberately left unimplemented while
the model is rejected — building it would only be worthwhile as the first step
of a rebuild (C6).

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
- The real A/B/C/D ablation was **not** run: the Kaggle mount is absent from
  this environment and there is no network access. No real ablation number is
  reported anywhere in this document or in the repository.
- No synthetic figure is presented as a real finding. The `REAL KAGGLE
  VALIDATION` banner is granted by observed well counts (773/770), not by a
  caller-supplied flag, so a synthetic run cannot claim it.
