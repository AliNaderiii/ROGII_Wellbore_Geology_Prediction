# ROGII Wellbore Geology Prediction — Audit & Baseline Pipeline

Audit-first scaffolding for the competition. **No modelling happens until the
audit reports are generated and reviewed** — that ordering is enforced by the
repository layout: `src/data.py` is a loader only, and nothing trains.

## Important: where the data is

The audit code targets the Kaggle mounts:

```
/kaggle/input/competitions/rogii-wellbore-geology-prediction/{train,test,sample_submission.csv}
/kaggle/input/datasets/phongnguyn23021656/koolbox-offline
/kaggle/input/datasets/fleongg/rogii-claude-models-pub
/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts
```

Reports are written to `/kaggle/working/reports`, **not** into the repository,
and are not committed: competition data must not be redistributed. The
scripts are the deliverable; run them on Kaggle to produce the findings. They
were verified end to end against synthetic fixtures (`tests/conftest.py`) that
reproduce the documented file layout, including deliberately planted faults
(non-monotonic MD, duplicate MD, half-foot spacing, missing typewell, an
internal `TVT_input` gap, and a leaky `submission_blend.csv` referencing a
test well) — all of which the audits and validators detect.

## Run

```bash
python scripts/run_all_audits.py           # all audits -> /kaggle/working/reports
python scripts/smoke_test_loader.py        # data-loader smoke test (loads, never trains)
python -m src.submission --submission submission.csv \
  --sample-submission /kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv
python scripts/smoke_test_loader.py --expect-train 773 --expect-test 3 --full-scan
python scripts/run_feature_ablation.py     # historical A/B/C/D Ridge feature ablation
python scripts/run_validation.py --particle-filter --beam-search --max-wells 100
python scripts/diagnose_dip_alignment.py   # REJECTED dip-alignment diagnostics
python -m pytest tests -q                  # 285 tests, synthetic fixtures only
```

### Real A/B/C/D ablation on Kaggle

```bash
# 1. 100-well pass: validate runtime, memory and output shape
python scripts/run_feature_ablation.py \
  --n-splits 5 --max-wells 100 \
  --cache-dir /kaggle/working/feature_ablation_cache \
  --reports-dir /kaggle/working/feature_ablation_reports

# 2. full run: all 770 eligible wells (773 discovered - 3 public test wells)
python scripts/run_feature_ablation.py \
  --n-splits 5 --expect-wells 770 \
  --cache-dir /kaggle/working/feature_ablation_cache \
  --reports-dir /kaggle/working/feature_ablation_reports
```

Historical ablation CLI: `--max-wells --n-splits --cache-dir --reports-dir
--device --clear-cache --spatial/--no-spatial --spatial-k --spatial-radius
--branches --expect-wells --seed --label --quiet`.

The main validation CLI additionally provides `--particle-filter`,
`--beam-search`, and `--alignment-features`; `--spatial` is also an explicit
opt-in there. PF/Beam runs write diagnostics only and never a submission.

**Report naming is evidence-based.** `real_*` filenames and the
`REAL KAGGLE VALIDATION` banner are emitted **only** when the discovered well
counts match the audited mount exactly (773 train / 770 eligible), as decided
by `src.real_ablation_reporting.is_real_run`. Any other run — synthetic,
partial or `--max-wells` subset — writes `synthetic_*` files banner-stamped
`SYNTHETIC — NOT A COMPETITION RESULT`. A file named `real_*` in this
repository is therefore always a real competition result.

A per-branch leakage preflight (`src/ablation_preflight.py`) runs before any
model is fitted and aborts the run on failure. It walks every design-matrix
column back to its raw roots through the manifest, so TVT, a formation marker
or Typewell Geology cannot pass even when reached transitively via a derived
feature.

From a Kaggle Notebook cell:

```python
import sys; sys.path.insert(0, "/path/to/checkout")
from scripts.run_all_audits import run_all
run_all()
```

The orchestrator resolves the repo root from `__file__` when run as a script
and by walking up from the CWD when imported into a notebook cell (where
`__file__` is undefined); `ROGII_REPO_ROOT` overrides both.

### Path resolution

`COMPETITION_ROOT` is resolved in strict priority order:

1. `$ROGII_COMPETITION_ROOT` — override for local dev, CI and tests
2. `/kaggle/input/competitions/rogii-wellbore-geology-prediction` — Kaggle
3. `<repo>/data/...` — local development fallback

`REPORTS_DIR` defaults to `/kaggle/working/reports` when `/kaggle/working`
exists, and to `<repo>/reports` otherwise, so a local run never tries to write
into a non-existent `/kaggle`. `$ROGII_REPORTS_DIR` overrides it.
`describe_paths()` prints the resolved table with an OK/MISSING flag per path,
and the orchestrator emits it on every run.

Offline-safe: only `pandas`/`numpy` are required; `python-pptx` is optional
(a stdlib `zipfile` + `ElementTree` fallback extracts slide text without it).

## Layout

```
src/paths.py       canonical mount paths (single source of truth)
src/discovery.py   dynamic well discovery — no hardcoded well IDs
src/columns.py     case/separator-insensitive column-role resolution
src/submission.py  reusable submission validator (section 3)
src/data.py        baseline loader: visible/hidden regions, well metadata (section 8)
scripts/_bootstrap.py  notebook-safe sys.path setup (no __file__)
scripts/           the five audit entrypoints
reports/           generated output (git-ignored) + decision_table.md
scripts/smoke_test_loader.py   loader smoke test on the real mount
tests/             pytest suite + synthetic fixtures (conftest.py)
```

## Safety posture

- **Pickles are never loaded.** Third-party model files are inspected
  statically with `pickletools.genops`, which recovers class names without
  executing the payload.
- **Nothing is pip-installed.** Wheels are parsed for METADATA only.
- **Every third-party artifact starts at NEEDS FURTHER REVIEW** and is only
  promoted with explicit on-disk evidence (see `reports/decision_table.md`).
- **Leakage detector** flags files that reference test well IDs, carry
  target-like columns, or look like precomputed submissions.
- Reports and `submission.csv` are git-ignored so competition data is never
  redistributed.

## Baseline pipeline contract (section 8)

`src.data` guarantees:

- wells discovered dynamically from the filesystem, train and test through the
  identical code path;
- original row order preserved (`row_index` materialised, never sorted);
- `is_visible` marks the known-TVT prefix, its complement the hidden suffix —
  derived from an explicit `Prediction Start` column when present, otherwise
  from the first `TVT_input` gap, with the source recorded in metadata;
- target availability validated rather than assumed
  (`validate_split` flags a test well that unexpectedly carries labels);
- one well in memory at a time via `iter_wells`.

## Known modelling hazards carried forward

1. Formation-top marker columns appear in train but are reported absent from
   test — verify in `input_inventory.md`; if confirmed, they must be *imputed*
   by a fold-trained structural model, never read raw at inference.
2. GR gaps are contiguous tool outages: interpolate **within each well**, never
   globally, since per-well calibration baselines differ.
3. Split with GroupKFold at the well level; row-level splits leak.
4. Anchor on the last known TVT of the prefix and model the residual.
6. **Train wells carry the full TVT curve, including the hidden region.** Those
   values are the label. Use `well.inference_features()` (drops TVT) for model
   inputs and `well.target(region)` when you deliberately want the label;
   `well.assert_no_target_leakage(frame)` fails loudly if TVT survives into a
   feature matrix. `TVT_input` is safe: it is NaN past the boundary by
   construction.
7. The 3 visible test wells are for pipeline validation only — never tune on
   them.
5. Enforce ANCC→ASTNU→ASTNL→EGFDU→EGFDL→BUDA ordering in post-processing.

## Model promotion status

The completed real 770-well A/B/C/D decision selected Ridge without alignment
and without spatial features (29.486 / 14.423 RMSE for `same_well_masked` /
`unseen_well`). Alignment and spatial implementations remain available only
through explicit diagnostic opt-ins. Particle Filter and Beam Search are
candidate target-free Ridge features, not models or ensemble branches; see
`reports/particle_beam_protocol.md`.

`src/model_status.py` is the single source of truth for whether a model may
enter a final predictor or an ensemble branch. Nothing is approved by default:
an unlisted model is `CANDIDATE`, never `APPROVED`.

| Model | Status | Evidence |
|---|---|---|
| `dip_constrained_alignment` | **REJECTED** | Real 770-well run, both protocols: +248.202 RMSE (`same_well_masked`), +82.104 RMSE (`unseen_well`) vs Ridge |

`assert_not_rejected()` raises on any attempt to route a rejected model into a
final/ensemble path; rejection does **not** block running the model as a
diagnostic, so the failure stays reproducible. The full analysis is in
`reports/dip_alignment_failure_analysis.md`.
