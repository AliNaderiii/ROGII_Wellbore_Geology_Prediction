# reports/

This directory holds **generated** audit output. It is intentionally almost
empty in git: reports are produced from the mounted Kaggle data, and the
competition data must not be redistributed.

Run on Kaggle (or anywhere the mounts exist):

```bash
python scripts/run_all_audits.py
```

which writes:

| File | Section | Produced by |
|---|---|---|
| `input_inventory.md` | 1 | `audit_competition_data.py` |
| `dataset_schema.csv` | 1 | `audit_competition_data.py` |
| `well_summary.csv` | 1 | `audit_competition_data.py` |
| `data_quality_initial.md` | 1 | `audit_competition_data.py` |
| `task_presentation_summary.md` | 2 | `audit_task_presentation.py` |
| `task_presentation_images/` | 2 | `audit_task_presentation.py` |
| `sample_submission_audit.md` | 3 | `audit_submission.py` |
| `koolbox_audit.md` | 4 | `audit_external_resources.py` |
| `artifact_inventory.csv` | 5 | `audit_external_resources.py` |
| `artifact_compatibility.md` | 5 | `audit_external_resources.py` |
| `external_artifact_leakage_audit.md` | 6 | `audit_external_resources.py` |
| `well_metadata.csv` | 8 | `run_all_audits.py` |
| `pipeline_validation_issues.csv` | 8 | `run_all_audits.py` |

`decision_table.md` (section 7) is hand-maintained and **is** tracked, because
it records judgement calls rather than machine output. Update it after each
audit run.
