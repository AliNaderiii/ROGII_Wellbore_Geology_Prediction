"""Target-free beam-search features for Ridge diagnostics.

Beam search explores a small bank of smooth stratigraphic paths.  It is not a
standalone TVT predictor and is not registered as a baseline model: its track,
uncertainty and failure indicators are optional columns consumed by Ridge.
Inputs are limited to horizontal GR/MD/X/Y/Z, Typewell GR/TVT and visible
prefix TVT_input through :class:`src.tasks.InferenceTask`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

import numpy as np
import pandas as pd

from src.cache import cache_key
from src.manifest import assert_safe_features
from src.particle_filter import (
    PathFeatureOutput,
    _interpolate_samples,
    fallback_output,
    path_smoothness,
    prepare_path_inputs,
    sampled_rows,
    weighted_quantile,
)
from src.tasks import InferenceTask

BEAM_SEARCH_VERSION = "target-free-beam-search-v1"

BEAM_FEATURE_COLUMNS: tuple[str, ...] = (
    "beam_track",
    "beam_shift",
    "beam_gradient",
    "beam_confidence",
    "beam_branch_spread",
    "beam_path_smoothness",
    "beam_gr_misfit",
    "beam_fallback",
)


@dataclass(frozen=True)
class BeamSearchConfig:
    """Bounded beam configuration chosen for Kaggle CPU runtime."""

    beam_width: int = 24
    branch_factor: int = 7
    stride: int = 8
    observation_sigma: float = 0.75
    gradient_step: float = 0.002
    max_abs_gradient: float = 0.08
    transition_weight: float = 0.20
    curvature_weight: float = 0.10
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.beam_width < 2:
            raise ValueError("beam_width must be >= 2")
        if self.branch_factor < 3 or self.branch_factor % 2 == 0:
            raise ValueError("branch_factor must be an odd integer >= 3")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.observation_sigma <= 0 or self.gradient_step <= 0 or self.temperature <= 0:
            raise ValueError("sigma, gradient_step and temperature must be positive")


class BeamSearchFeatureGenerator:
    """Generate cached beam tracks, branch spread and confidence features."""

    feature_columns = BEAM_FEATURE_COLUMNS

    def __init__(
        self,
        config: BeamSearchConfig | None = None,
        *,
        cache=None,
        dataset_version: str = "rogii-mounted-v1",
        fold_id: Any = "inference",
        protocol: str = "inference",
        device: str = "auto",
    ) -> None:
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be one of: auto, cpu, gpu")
        self.config = config or BeamSearchConfig()
        self.cache = cache
        self.dataset_version = str(dataset_version)
        self.fold_id = fold_id
        self.protocol = str(protocol)
        self.device = device

    def _key(self, task: InferenceTask) -> str:
        return cache_key(
            dataset_version=self.dataset_version,
            well_id=task.well_id,
            fold_id=self.fold_id,
            protocol=self.protocol,
            feature_config={
                "name": BEAM_SEARCH_VERSION,
                "boundary_mode": task.mode,
                "n_rows": task.n_rows,
                "start": task.start,
                "stop": task.stop,
                "config": asdict(self.config),
            },
            alignment_config={},
            device_profile={"feature_execution": "numpy_cpu_portable"},
            code_version=BEAM_SEARCH_VERSION,
        )

    def _from_cache(self, task: InferenceTask, hit: dict[str, np.ndarray]) -> PathFeatureOutput | None:
        needed = {f"f_{c}" for c in self.feature_columns} | {"diagnostics_json"}
        if not needed <= set(hit):
            return None
        if any(np.asarray(hit[f"f_{c}"]).size != task.n_predict for c in self.feature_columns):
            return None
        frame = pd.DataFrame(
            {c: np.asarray(hit[f"f_{c}"], dtype="float64") for c in self.feature_columns}
        )
        try:
            diagnostics = json.loads(str(np.asarray(hit["diagnostics_json"]).ravel()[0]))
        except Exception:
            return None
        diagnostics["cache_hit"] = True
        diagnostics["requested_device"] = self.device
        diagnostics["execution_device"] = "cpu"
        assert_safe_features(frame.columns, context="cached beam-search feature matrix")
        return PathFeatureOutput(frame=frame, diagnostics=diagnostics)

    def _put(self, key: str, output: PathFeatureOutput) -> None:
        if self.cache is None:
            return
        payload = {f"f_{c}": output.frame[c].to_numpy(dtype="float64") for c in self.feature_columns}
        payload["diagnostics_json"] = np.asarray(
            [json.dumps(output.diagnostics, sort_keys=True, default=str)]
        )
        self.cache.put(key, **payload)

    def generate(self, task: InferenceTask) -> PathFeatureOutput:
        task.assert_no_target()
        key = self._key(task)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                cached = self._from_cache(task, hit)
                if cached is not None:
                    return cached

        prepared = prepare_path_inputs(task)
        fatal = prepared.failure_reason in {
            "empty_well",
            "insufficient_visible_tvt_input",
            "missing_visible_tvt_anchor",
            "missing_or_invalid_typewell",
            "horizontal_gr_all_missing",
        }
        if fatal or task.n_predict == 0:
            output = fallback_output(
                task,
                prepared,
                self.feature_columns,
                "beam",
                prepared.failure_reason or "empty_prediction_region",
                requested_device=self.device,
            )
            self._put(key, output)
            return output

        cfg = self.config
        rows = sampled_rows(task, cfg.stride)
        n_steps = len(rows)
        track_s = np.zeros(n_steps)
        grad_s = np.zeros(n_steps)
        confidence_s = np.zeros(n_steps)
        spread_s = np.zeros(n_steps)
        misfit_s = np.zeros(n_steps)
        fallback_s = np.zeros(n_steps)

        first = int(rows[0])
        previous_row = max(task.anchor_row, task.start - 1)
        beam_track = np.asarray(
            [prepared.expected_track[previous_row]], dtype="float64"
        )
        beam_grad = np.asarray([prepared.expected_gradient[first]], dtype="float64")
        beam_cost = np.asarray([0.0], dtype="float64")
        offsets = np.linspace(
            -(cfg.branch_factor // 2),
            cfg.branch_factor // 2,
            cfg.branch_factor,
            dtype="float64",
        ) * cfg.gradient_step

        for j, row in enumerate(rows):
            row = int(row)
            delta_md = max(abs(float(prepared.md[row] - prepared.md[previous_row])), 1.0)
            expected_step = float(
                prepared.expected_track[row] - prepared.expected_track[previous_row]
            )
            geom_grad = expected_step / delta_md

            parent_n = beam_track.size
            parent_track = np.repeat(beam_track, cfg.branch_factor)
            parent_grad = np.repeat(beam_grad, cfg.branch_factor)
            parent_cost = np.repeat(beam_cost, cfg.branch_factor)
            branch_offset = np.tile(offsets, parent_n)

            candidate_grad = 0.75 * parent_grad + 0.25 * geom_grad + branch_offset
            candidate_grad = np.clip(
                candidate_grad, -cfg.max_abs_gradient, cfg.max_abs_gradient
            )
            candidate_track = parent_track + expected_step + (
                candidate_grad - geom_grad
            ) * delta_md
            candidate_track = np.clip(
                candidate_track, prepared.ref.tvt_min, prepared.ref.tvt_max
            )

            transition = ((candidate_grad - geom_grad) / (3.0 * cfg.gradient_step)) ** 2
            curvature = ((candidate_grad - parent_grad) / cfg.gradient_step) ** 2
            candidate_cost = (
                parent_cost
                + cfg.transition_weight * transition
                + cfg.curvature_weight * curvature
            )

            if prepared.gr_missing[row] or not np.isfinite(prepared.signal[row]):
                fallback_s[j] = 1.0
                gr_misfit = np.zeros_like(candidate_track)
            else:
                reference = np.interp(
                    candidate_track, prepared.ref.grid, prepared.ref.gr_z
                )
                gr_misfit = np.abs(reference - prepared.signal[row])
                candidate_cost += 0.5 * (gr_misfit / cfg.observation_sigma) ** 2

            keep_n = min(cfg.beam_width, candidate_cost.size)
            if keep_n < candidate_cost.size:
                keep = np.argpartition(candidate_cost, keep_n - 1)[:keep_n]
            else:
                keep = np.arange(candidate_cost.size)
            order = keep[np.argsort(candidate_cost[keep], kind="stable")]
            beam_track = candidate_track[order]
            beam_grad = candidate_grad[order]
            beam_cost = candidate_cost[order]
            kept_misfit = gr_misfit[order]

            relative = beam_cost - float(beam_cost.min())
            weights = np.exp(-relative / cfg.temperature)
            total = float(weights.sum())
            if not np.isfinite(total) or total <= 1e-15:
                weights = np.full(beam_track.size, 1.0 / beam_track.size)
                fallback_s[j] = 1.0
            else:
                weights /= total

            track_s[j] = float(np.sum(weights * beam_track))
            grad_s[j] = float(np.sum(weights * beam_grad))
            q10 = weighted_quantile(beam_track, weights, 0.10)
            q90 = weighted_quantile(beam_track, weights, 0.90)
            spread_s[j] = max(0.0, q90 - q10)
            misfit_s[j] = float(np.sum(weights * kept_misfit))

            if beam_track.size >= 2:
                gap = max(0.0, float(beam_cost[1] - beam_cost[0]))
                entropy = -float(np.sum(weights * np.log(np.clip(weights, 1e-15, 1.0))))
                entropy /= np.log(max(beam_track.size, 2))
                confidence_s[j] = np.clip(
                    (1.0 - entropy) * (1.0 - np.exp(-gap)), 0.0, 1.0
                )
            else:
                confidence_s[j] = 0.0
            if fallback_s[j]:
                confidence_s[j] = 0.0
            previous_row = row

        track = _interpolate_samples(rows, track_s, task)
        gradient = _interpolate_samples(rows, grad_s, task)
        confidence = np.clip(_interpolate_samples(rows, confidence_s, task), 0.0, 1.0)
        spread = np.maximum(_interpolate_samples(rows, spread_s, task), 0.0)
        misfit = np.maximum(_interpolate_samples(rows, misfit_s, task), 0.0)
        fallback = (_interpolate_samples(rows, fallback_s, task) > 0.5).astype("float64")
        smooth = path_smoothness(track, prepared.md[task.start : task.stop])
        anchor = float(task.anchor_tvt)

        frame = pd.DataFrame(
            {
                "beam_track": track,
                "beam_shift": track - anchor,
                "beam_gradient": gradient,
                "beam_confidence": confidence,
                "beam_branch_spread": spread,
                "beam_path_smoothness": smooth,
                "beam_gr_misfit": misfit,
                "beam_fallback": fallback,
            }
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame = frame[list(self.feature_columns)]
        assert_safe_features(frame.columns, context="beam-search feature matrix")

        failure_reason = ""
        if prepared.failure_reason:
            failure_reason = prepared.failure_reason
        elif bool(fallback.any()):
            failure_reason = "partial_missing_or_invalid_horizontal_gr"
        diagnostics = {
            "confidence_mean": float(np.mean(confidence)),
            "confidence_p10": float(np.quantile(confidence, 0.10)),
            "branch_spread_mean": float(np.mean(spread)),
            "branch_spread_p90": float(np.quantile(spread, 0.90)),
            "path_smoothness": float(np.mean(smooth)),
            "fallback_status": bool(fallback.any()),
            "fallback_fraction": float(np.mean(fallback)),
            "failure_reason": failure_reason,
            "cache_hit": False,
            "requested_device": self.device,
            "execution_device": "cpu",
            "geometry_missing_fraction": prepared.geometry_missing_fraction,
            "geometry_coefficients": prepared.geometry_coefficients.tolist(),
            "beam_width": cfg.beam_width,
            "branch_factor": cfg.branch_factor,
            "stride": cfg.stride,
            "n_generated_rows": task.n_predict,
        }
        output = PathFeatureOutput(frame=frame, diagnostics=diagnostics)
        self._put(key, output)
        return output


BeamSearch = BeamSearchFeatureGenerator


def beam_search_features(
    task: InferenceTask,
    *,
    config: BeamSearchConfig | None = None,
    cache=None,
    dataset_version: str = "rogii-mounted-v1",
    fold_id: Any = "inference",
    protocol: str = "inference",
    device: str = "auto",
) -> PathFeatureOutput:
    return BeamSearchFeatureGenerator(
        config,
        cache=cache,
        dataset_version=dataset_version,
        fold_id=fold_id,
        protocol=protocol,
        device=device,
    ).generate(task)
