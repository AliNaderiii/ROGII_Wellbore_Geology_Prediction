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

Those mounts **do not exist in the development sandbox this code was written
in**, so the reports in `reports/` are not checked in with real numbers. The
scripts are the deliverable; run them on Kaggle to produce the findings. They
were verified end to end against a synthetic fixture (`tests/make_mock_mount.py`)
that reproduces the documented file layout, including deliberately planted
faults (non-monotonic MD, missing typewell, a leaky `submission_blend.csv`
referencing a test well) — all of which the audits detect.

## Run

```bash
python scripts/run_all_audits.py          # sections 1-8
# or individually
python scripts/audit_competition_data.py  # 1
python scripts/audit_task_presentation.py # 2
python scripts/audit_submission.py        # 3
python scripts/audit_external_resources.py# 4,5,6
```

Offline-safe: only `pandas`/`numpy` are required; `python-pptx` is optional
(a stdlib `zipfile` + `ElementTree` fallback extracts slide text without it).

## Layout

```
src/paths.py       canonical mount paths (single source of truth)
src/discovery.py   dynamic well discovery — no hardcoded well IDs
src/columns.py     case/separator-insensitive column-role resolution
src/submission.py  reusable submission validator (section 3)
src/data.py        baseline loader: visible/hidden regions, well metadata (section 8)
scripts/           the five audit entrypoints
reports/           generated output (git-ignored) + decision_table.md
tests/             synthetic mount fixture
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
5. Enforce ANCC→ASTNU→ASTNL→EGFDU→EGFDL→BUDA ordering in post-processing.
