# SYNTHETIC PIPELINE VERIFICATION ONLY — NOT A COMPETITION RESULT

# Ablation leakage & feature-safety preflight

**Status: ALL CHECKS PASSED** — 30/30 checks passed.

Every check is verified against the *actual design matrix each branch builds*, walking each column back to its raw roots through the manifest. A train-only column or the target therefore cannot pass by arriving transitively through a derived feature.

## Checks

| branch | check | passed | detail |
|---|---|---|---|
| ridge_no_align | TVT is never in X | PASS | 28 columns; no column is TVT and none derives from TVT |
| ridge_no_align | hidden TVT_input is never in X | PASS | tvt_known has 0 finite values at/after the boundary (rows 2413:3942) |
| ridge_no_align | Typewell Geology and formation markers never reach the Test feature matrix | PASS | no column derives from Typewell Geology, a formation marker, or TVT |
| ridge_no_align | every feature root is a test-available raw column | PASS | roots ⊆ ['GR', 'MD', 'TVT_input', 'Typewell GR', 'Typewell TVT', 'X', 'Y', 'Z'] |
| ridge_no_align | manifest whitelist accepts the matrix | PASS | 28 manifest columns cleared |
| ridge_no_align | branch matrix matches its declared configuration | PASS | alignment=False (want False), spatial=False (want False) |
| ridge_baseline | TVT is never in X | PASS | 32 columns; no column is TVT and none derives from TVT |
| ridge_baseline | hidden TVT_input is never in X | PASS | tvt_known has 0 finite values at/after the boundary (rows 2413:3942) |
| ridge_baseline | Typewell Geology and formation markers never reach the Test feature matrix | PASS | no column derives from Typewell Geology, a formation marker, or TVT |
| ridge_baseline | every feature root is a test-available raw column | PASS | roots ⊆ ['GR', 'MD', 'TVT_input', 'Typewell GR', 'Typewell TVT', 'X', 'Y', 'Z'] |
| ridge_baseline | manifest whitelist accepts the matrix | PASS | 32 manifest columns cleared |
| ridge_baseline | branch matrix matches its declared configuration | PASS | alignment=True (want True), spatial=False (want False) |
| ridge_spatial_only | TVT is never in X | PASS | 34 columns; no column is TVT and none derives from TVT |
| ridge_spatial_only | hidden TVT_input is never in X | PASS | tvt_known has 0 finite values at/after the boundary (rows 2413:3942) |
| ridge_spatial_only | Typewell Geology and formation markers never reach the Test feature matrix | PASS | no column derives from Typewell Geology, a formation marker, or TVT |
| ridge_spatial_only | every feature root is a test-available raw column | PASS | roots ⊆ ['GR', 'MD', 'TVT_input', 'Typewell GR', 'Typewell TVT', 'X', 'Y', 'Z'] |
| ridge_spatial_only | manifest whitelist accepts the matrix | PASS | 28 manifest columns cleared |
| ridge_spatial_only | branch matrix matches its declared configuration | PASS | alignment=False (want False), spatial=True (want True) |
| ridge_align_spatial | TVT is never in X | PASS | 38 columns; no column is TVT and none derives from TVT |
| ridge_align_spatial | hidden TVT_input is never in X | PASS | tvt_known has 0 finite values at/after the boundary (rows 2413:3942) |
| ridge_align_spatial | Typewell Geology and formation markers never reach the Test feature matrix | PASS | no column derives from Typewell Geology, a formation marker, or TVT |
| ridge_align_spatial | every feature root is a test-available raw column | PASS | roots ⊆ ['GR', 'MD', 'TVT_input', 'Typewell GR', 'Typewell TVT', 'X', 'Y', 'Z'] |
| ridge_align_spatial | manifest whitelist accepts the matrix | PASS | 32 manifest columns cleared |
| ridge_align_spatial | branch matrix matches its declared configuration | PASS | alignment=True (want True), spatial=True (want True) |
| all | alignment features are target-free | PASS | align_* roots = ['GR', 'MD', 'TVT_input', 'Typewell GR', 'Typewell TVT'] |
| all | manifest inference provenance | PASS | every inference-cleared feature traces to test-available roots |
| all | public test wells are never used for tuning | PASS | blocked IDs ['000d7d20', '00bbac68', '00e12e8b'] are excluded by assert_no_blocked_wells at the universe, fold, fit and results stages |
| all | spatial features use only fold-training donor wells | PASS | 7 donor wells; assert_disjoint on the queried well passed |
| all | query wells are excluded from their own neighbour set | PASS | tr00000 is not in the donor set; features_for additionally self-excludes by well_id at query time |
| all | no public leaderboard result is used as a training label | PASS | targets come only from WellTask.target (TVT for unseen_well, TVT_input for same_well_masked); no external artifact is read |

## Feature roots by branch

| branch | kind | n_features |
|---|---|---|
| ridge_align_spatial | alignment | 4 |
| ridge_align_spatial | manifest | 28 |
| ridge_align_spatial | spatial | 6 |
| ridge_baseline | alignment | 4 |
| ridge_baseline | manifest | 28 |
| ridge_no_align | manifest | 28 |
| ridge_spatial_only | manifest | 28 |
| ridge_spatial_only | spatial | 6 |

Full per-feature provenance is in `real_ablation_preflight.csv`.

### Permitted raw sources

```
GR
MD
TVT_input
Typewell GR
Typewell TVT
X
Y
Z
```
