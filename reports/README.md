# reports/

This directory holds both **authored** documents (tracked in git) and
**generated** output (git-ignored, because it is derived from competition data
that must not be redistributed).

## Tracked — authored, no competition data

| File | Contents |
|---|---|
| `task_interpretation.md` | What the task is physically: TVT as a stratigraphic trajectory, the horizontal/typewell inverse problem, dip and azimuth, GR missingness, the metric |
| `validation_protocol.md` | The validation design, **including §0 on why in-sample evaluation is invalid** |
| `feature_manifest.csv` | Every feature's availability, target-derived status, decision and leakage risk. Generated from `src/manifest.py`, which also *enforces* it |
| `decision_table.md` | Section 7 resource decisions (external artifacts etc.) |
| `real_770_ablation_decision.md` | Completed real 770-well A/B/C/D decision: default Ridge has no alignment or spatial features |
| `particle_beam_protocol.md` | Target-free, fold-scoped PF/Beam feature protocol and 100-real-well run contract |
| `pf_beam_real_decision.md` | Robustness decision for `ridge_particle_beam` vs `ridge_default` (owner aggregates + rule) |
| `pf_beam_failure_analysis.md` | Paired-error / failure analysis; separates real, synthetic, and public-LB evidence |
| `pf_beam_paired_well_deltas.csv`, `pf_beam_fold_deltas.csv`, `pf_beam_bootstrap_ci.csv` | Paired well/fold/bootstrap tables (or UNAVAILABLE placeholders when well-level artifacts are absent) |
| `synthetic_validation/` | Harness-verification run on a **synthetic** field — banner-stamped, never a competition result |
| `geoanchor_experiment.md` | Pre-registration of the controlled GeoAnchor experiment (arms A–E: affine GR calibration, multi-branch alignment, bimodal uncertainty, prefix pseudo-holdout, well-level GBDT gate for PF/Beam corrections): ideas mapped to the studied public notebooks, forbidden list, gate rules, decision rule |
| `neural_protocol.md` | Leakage-safe PyTorch MLP/GRU/TCN residual protocol, fold-safe training, hybrid gate and Kaggle execution instructions |
| `neural_phase1_status.md` | Phase-1 implementation status; explicitly separates unavailable real-data evidence from synthetic plumbing evidence |
| `safe_alignment_protocol.md` | Pre-registration of the staged safe-alignment experiment (stages A–F: Ridge anchor, bounded PF/Beam blend, affine heel calibration, branch/bimodal hedging, robust IRLS projection, multi-cut prefix verification + tail guard), promotion rule, Kaggle execution instructions and synthetic harness verification |
| `synthetic_geoanchor/` | Executed GeoAnchor A–E run on a **synthetic** field — banner-stamped `synthetic_geoanchor_*` tables/JSON, never a competition result |

## Generated — git-ignored, written by the runner

```bash
python scripts/run_validation.py --n-splits 5 --spatial
```

Writes to `REPORTS_DIR` (`/kaggle/working/reports` on Kaggle, `<repo>/reports`
locally):

| File | Contents |
|---|---|
| `validation_results.csv` | One row per (model, protocol): global / mean / median / P90 / worst-10 RMSE, well and point counts |
| `well_level_validation.csv` | One row per (model, protocol, well) |
| `stratified_validation.csv` | RMSE by hidden suffix length, GR missingness, prefix length |
| `validation_failures.csv` | Every task, fit and predict failure |
| `spatial_ablation.csv` | With vs. without offset-well features (`--spatial`) |
| `spatial_construction.csv` | Per-fold donor counts and neighbour parameters (`--spatial`) |
| `baseline_report.md` | The full narrative report |
| `validation_protocol_run.md` | Parameters of that specific execution |
| `run_environment.json` | Versions, runtime, peak memory, guard status |
| `feature_manifest_verification.csv` | Manifest claims re-checked against the observed columns |
| `protocol_comparison_real.md` | Exact, separately reported Ridge protocol counts, distributions, ranges and scoring-region audit (`--real-analysis`) |
| `error_analysis_real.csv` | Ridge per-well error, target-range, curvature, GR/typewell-alignment and tail flags (`--real-analysis`) |
| `gr_missingness_error_real.csv`, `suffix_length_error_real.csv`, `prefix_length_error_real.csv` | Separate protocol-stratified Ridge error tables (`--real-analysis`) |
| `worst_wells_real.csv` | Top-20 Ridge wells **within each protocol**, never a combined rank (`--real-analysis`) |
| `spatial_ablation_real.md` | Per-protocol spatial Ridge A/B plus actual population/non-constant diagnostics (`--real-analysis`) |
| `dip_constrained_alignment_ablation.csv`, `dip_constrained_alignment_real.md` | Isolated Ridge vs dip-constrained GR/typewell A/B with confidence/failure/fallback counts (`--dip-alignment-experiment`) |
| `particle_beam_results.csv`, `particle_beam_wells.csv` | Paired Ridge A/B/C/D comparison for opt-in PF/Beam features (`--particle-filter --beam-search`) |
| `particle_beam_diagnostics.csv`, `particle_beam_failures.csv` | Per-well confidence, branch spread, smoothness, fallback/cache status and explicit failures |
| `particle_beam_ablation.md`, `particle_beam_run_environment.json` | Narrative comparison and reproducibility/runtime record; never a submission |
| `pf_beam_real_decision.md`, `pf_beam_failure_analysis.md`, `pf_beam_*_{deltas,ci}.csv` | Post-run paired-error robustness (`scripts/analyze_pf_beam_robustness.py`); no retrain |
| `synthetic_geoanchor_*` | GeoAnchor arms A–E paired comparison, gate diagnostics, fold stability, bootstrap CIs, stratified tables and decision report (`scripts/run_geoanchor_experiment.py`); prefixed `synthetic_` unless the audited 773/770 real well counts are discovered |

Audit-phase outputs (`input_inventory.md`, `dataset_schema.csv`,
`well_summary.csv`, `data_quality_initial.md`, `sample_submission_audit.md`,
`koolbox_audit.md`, `artifact_inventory.csv`, `artifact_compatibility.md`,
`external_artifact_leakage_audit.md`, `well_metadata.csv`,
`pipeline_validation_issues.csv`) come from `scripts/run_all_audits.py`.

## Real vs. synthetic

The two are kept in **separate directories** so a synthetic figure can never be
mistaken for a competition figure:

- `reports/` — real runs. No banner.
- `reports/synthetic_validation/` — synthetic runs. Every report begins with
  `⚠️ SYNTHETIC PIPELINE VERIFICATION ONLY`, applied by the `--label` flag.

## Preflight on the real mount

```bash
python scripts/run_validation.py --expect-train 773 --expect-test 3
```

Refuses to run unless discovery finds exactly the audited well counts, so a
partial or misconfigured mount fails loudly instead of silently validating on
a subset.
