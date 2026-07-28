"""Target-free particle-filter features for Ridge diagnostics.

The particle filter is deliberately a *feature generator*, not a predictor.
It receives only :class:`src.tasks.InferenceTask`, whose horizontal-well TVT
label is structurally unavailable.  Its observations are:

* horizontal GR, MD, X, Y and Z over the whole inference task;
* Typewell GR indexed by Typewell TVT; and
* horizontal-well ``TVT_input`` strictly before ``Prediction Start``.

No horizontal-well TVT label, Typewell Geology, or formation marker is read.
The implementation is NumPy-only and deterministic, so ``device='gpu'`` means
"GPU requested, portable CPU implementation used" rather than introducing a
GPU-only numerical path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from src.cache import cache_key
from src.features import (
    TypewellReference,
    calibrate_gr_to_reference,
    gr_features,
    interpolate_within_well,
)
from src.manifest import assert_safe_features
from src.tasks import InferenceTask

PARTICLE_FILTER_VERSION = "target-free-particle-filter-v1"

PARTICLE_FEATURE_COLUMNS: tuple[str, ...] = (
    "pf_track",
    "pf_shift",
    "pf_gradient",
    "pf_confidence",
    "pf_branch_spread",
    "pf_path_smoothness",
    "pf_gr_misfit",
    "pf_fallback",
)


@dataclass(frozen=True)
class ParticleFilterConfig:
    """Budgeted, deterministic particle-filter configuration."""

    n_particles: int = 64
    stride: int = 8
    observation_sigma: float = 0.75
    process_noise: float = 0.018
    gradient_noise: float = 0.0015
    resample_ess_fraction: float = 0.55
    max_abs_gradient: float = 0.08
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_particles < 8:
            raise ValueError("n_particles must be >= 8")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.observation_sigma <= 0 or self.process_noise < 0:
            raise ValueError("noise scales must be non-negative and observation_sigma > 0")
        if not 0 < self.resample_ess_fraction <= 1:
            raise ValueError("resample_ess_fraction must be in (0, 1]")


@dataclass
class PathFeatureOutput:
    """A prediction-region feature frame plus target-free diagnostics."""

    frame: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass
class PreparedPathInputs:
    """Shared target-free arrays used by particle and beam generators."""

    ref: TypewellReference
    signal: np.ndarray
    gr_missing: np.ndarray
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    expected_track: np.ndarray
    expected_gradient: np.ndarray
    geometry_coefficients: np.ndarray
    geometry_missing_fraction: float
    failure_reason: str


def _filled(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out, missing = interpolate_within_well(np.asarray(values, dtype="float64"))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), missing


def _geometry_prior(task: InferenceTask) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Fit a prefix-only dTVT model to increments of MD/X/Y/Z.

    Every geometry channel is present in the regression.  Ridge regularisation
    handles nearly collinear trajectory channels, while clipping prevents a
    short or noisy prefix from creating an unbounded continuation.
    """
    md, md_missing = _filled(task.md)
    x, x_missing = _filled(task.x)
    y, y_missing = _filled(task.y)
    z, z_missing = _filled(task.z)
    geom_missing = float(np.mean(np.column_stack([md_missing, x_missing, y_missing, z_missing])))

    n = task.n_rows
    if n == 0:
        return np.array([]), np.array([]), np.zeros(4), geom_missing, "empty_well"

    increments = np.column_stack([np.diff(md), np.diff(x), np.diff(y), np.diff(z)])
    known = np.isfinite(task.tvt_known)
    pair_known = known[1:] & known[:-1]
    pair_known &= np.arange(1, n) < task.start
    pair_known &= np.isfinite(increments).all(axis=1)
    dtvt = np.diff(np.asarray(task.tvt_known, dtype="float64"))
    pair_known &= np.isfinite(dtvt)

    reason = ""
    if int(pair_known.sum()) >= 8:
        a = increments[pair_known]
        b = dtvt[pair_known]
        scale = np.sqrt(np.mean(a * a, axis=0))
        scale = np.where(scale > 1e-9, scale, 1.0)
        az = a / scale
        # Fixed regularisation: this is a within-well physical calibration,
        # never a target-tuned hyperparameter.
        reg = 1e-2 * np.eye(az.shape[1])
        try:
            coef_z = np.linalg.solve(az.T @ az + reg, az.T @ b)
            coef = coef_z / scale
        except np.linalg.LinAlgError:  # pragma: no cover - solve is regularised
            coef = np.linalg.lstsq(az.T @ az + reg, az.T @ b, rcond=None)[0] / scale
    else:
        coef = np.zeros(4, dtype="float64")
        md_known = known & (np.arange(n) < task.start)
        if int(md_known.sum()) >= 2:
            idx = np.flatnonzero(md_known)
            dmd = md[idx[-1]] - md[idx[max(0, len(idx) - 25)]]
            dt = task.tvt_known[idx[-1]] - task.tvt_known[idx[max(0, len(idx) - 25)]]
            coef[0] = float(dt / dmd) if abs(dmd) > 1e-9 else 0.0
            reason = "short_prefix_geometry_slope_fallback"
        else:
            reason = "insufficient_visible_tvt_input"

    raw_step = increments @ coef
    dmd_abs = np.maximum(np.abs(increments[:, 0]), 1.0)
    max_step = 0.08 * dmd_abs
    raw_step = np.clip(np.nan_to_num(raw_step), -max_step, max_step)

    anchor_row = task.anchor_row
    anchor = task.anchor_tvt
    if anchor_row < 0 or not np.isfinite(anchor):
        anchor_row = max(task.start - 1, 0)
        anchor = 0.0
        reason = reason or "missing_visible_tvt_anchor"

    track = np.full(n, float(anchor), dtype="float64")
    for i in range(anchor_row + 1, n):
        track[i] = track[i - 1] + raw_step[i - 1]
    for i in range(anchor_row - 1, -1, -1):
        track[i] = track[i + 1] - raw_step[i]

    if task.tw_tvt is not None:
        finite = np.asarray(task.tw_tvt, dtype="float64")
        finite = finite[np.isfinite(finite)]
        if finite.size >= 2:
            track = np.clip(track, finite.min(), finite.max())

    gradient = np.zeros(n, dtype="float64")
    if n > 1:
        dmd = np.diff(md)
        with np.errstate(divide="ignore", invalid="ignore"):
            gradient[1:] = np.where(np.abs(dmd) > 1e-9, np.diff(track) / dmd, 0.0)
        gradient[0] = gradient[1]
    return track, gradient, coef, geom_missing, reason


def prepare_path_inputs(task: InferenceTask) -> PreparedPathInputs:
    """Build all allowed observations and enforce the hidden-prefix boundary."""
    task.assert_no_target()
    hidden = np.asarray(task.tvt_known[task.start : task.stop], dtype="float64")
    if np.isfinite(hidden).any():
        raise AssertionError("hidden TVT_input reached a path feature generator")

    # Access each required trajectory channel explicitly.  Typewell Geology is
    # intentionally never referenced here or anywhere else in this module.
    md, _ = _filled(task.md)
    x, _ = _filled(task.x)
    y, _ = _filled(task.y)
    z, _ = _filled(task.z)
    expected, gradient, coefficients, geom_missing, geometry_reason = _geometry_prior(task)

    ref = TypewellReference(task.tw_tvt, task.tw_gr)
    gr = gr_features(task)
    gr_missing = np.asarray(gr["gr_is_missing"], dtype="float64") > 0.5
    signal = calibrate_gr_to_reference(task, ref, np.asarray(gr["gr_z"], dtype="float64"))

    reason = geometry_reason
    if not ref.ok:
        reason = "missing_or_invalid_typewell"
    elif bool(gr_missing.all()):
        reason = "horizontal_gr_all_missing"
    elif task.anchor_row < 0 or not np.isfinite(task.anchor_tvt):
        reason = "missing_visible_tvt_anchor"

    return PreparedPathInputs(
        ref=ref,
        signal=signal,
        gr_missing=gr_missing,
        md=md,
        x=x,
        y=y,
        z=z,
        expected_track=expected,
        expected_gradient=gradient,
        geometry_coefficients=coefficients,
        geometry_missing_fraction=geom_missing,
        failure_reason=reason,
    )


def deterministic_seed(task: InferenceTask, config_seed: int, fold_id: Any, protocol: str) -> int:
    raw = f"{task.well_id}|{task.mode}|{task.start}|{task.stop}|{fold_id}|{protocol}".encode()
    digest = hashlib.sha256(raw).digest()
    return (int.from_bytes(digest[:8], "little") + int(config_seed)) % (2**32)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    if cumulative[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(cumulative, quantile * cumulative[-1], side="left")])


def path_smoothness(track: np.ndarray, md: np.ndarray) -> np.ndarray:
    """Absolute change in dTVT/dMD, interpolated as a row feature."""
    track = np.asarray(track, dtype="float64")
    md = np.asarray(md, dtype="float64")
    n = track.size
    if n < 2:
        return np.zeros(n, dtype="float64")
    dmd = np.gradient(md)
    dmd = np.where(np.abs(dmd) > 1e-9, dmd, 1.0)
    grad = np.gradient(track) / dmd
    smooth = np.abs(np.gradient(grad)) / np.maximum(np.abs(dmd), 1.0)
    return np.nan_to_num(smooth, nan=0.0, posinf=0.0, neginf=0.0)


def sampled_rows(task: InferenceTask, stride: int) -> np.ndarray:
    rows = np.arange(task.start, task.stop, max(1, stride), dtype=np.int64)
    if task.n_predict and (rows.size == 0 or rows[-1] != task.stop - 1):
        rows = np.append(rows, task.stop - 1)
    return rows


def _interpolate_samples(rows: np.ndarray, values: np.ndarray, task: InferenceTask) -> np.ndarray:
    target = np.arange(task.start, task.stop, dtype="float64")
    if target.size == 0:
        return np.array([], dtype="float64")
    if rows.size == 1:
        return np.full(target.size, float(values[0]))
    return np.interp(target, rows.astype("float64"), np.asarray(values, dtype="float64"))


def fallback_output(
    task: InferenceTask,
    prepared: PreparedPathInputs,
    columns: tuple[str, ...],
    prefix: str,
    reason: str,
    *,
    cache_hit: bool = False,
    requested_device: str = "auto",
) -> PathFeatureOutput:
    sl = slice(task.start, task.stop)
    track = np.asarray(prepared.expected_track[sl], dtype="float64")
    gradient = np.asarray(prepared.expected_gradient[sl], dtype="float64")
    smooth = path_smoothness(track, prepared.md[sl])
    anchor = task.anchor_tvt if np.isfinite(task.anchor_tvt) else 0.0
    values = {
        f"{prefix}_track": track,
        f"{prefix}_shift": track - anchor,
        f"{prefix}_gradient": gradient,
        f"{prefix}_confidence": np.zeros(task.n_predict),
        f"{prefix}_branch_spread": np.zeros(task.n_predict),
        f"{prefix}_path_smoothness": smooth,
        f"{prefix}_gr_misfit": np.zeros(task.n_predict),
        f"{prefix}_fallback": np.ones(task.n_predict),
    }
    frame = pd.DataFrame({name: values[name] for name in columns})
    assert_safe_features(frame.columns, context=f"{prefix} fallback feature matrix")
    diagnostics = {
        "confidence_mean": 0.0,
        "confidence_p10": 0.0,
        "branch_spread_mean": 0.0,
        "branch_spread_p90": 0.0,
        "path_smoothness": float(np.mean(smooth)) if smooth.size else 0.0,
        "fallback_status": True,
        "fallback_fraction": 1.0,
        "failure_reason": str(reason),
        "cache_hit": bool(cache_hit),
        "requested_device": requested_device,
        "execution_device": "cpu",
        "geometry_missing_fraction": prepared.geometry_missing_fraction,
        "n_generated_rows": task.n_predict,
    }
    return PathFeatureOutput(frame=frame, diagnostics=diagnostics)


class ParticleFilterFeatureGenerator:
    """Generate cached particle-filter tracks and uncertainty features."""

    feature_columns = PARTICLE_FEATURE_COLUMNS

    def __init__(
        self,
        config: ParticleFilterConfig | None = None,
        *,
        cache=None,
        dataset_version: str = "rogii-mounted-v1",
        fold_id: Any = "inference",
        protocol: str = "inference",
        device: str = "auto",
    ) -> None:
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be one of: auto, cpu, gpu")
        self.config = config or ParticleFilterConfig()
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
                "name": PARTICLE_FILTER_VERSION,
                "boundary_mode": task.mode,
                "n_rows": task.n_rows,
                "start": task.start,
                "stop": task.stop,
                "config": asdict(self.config),
            },
            alignment_config={},
            # Numerical execution is always NumPy CPU.  Excluding the requested
            # accelerator makes CPU/GPU requests share bit-identical artifacts.
            device_profile={"feature_execution": "numpy_cpu_portable"},
            code_version=PARTICLE_FILTER_VERSION,
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
        assert_safe_features(frame.columns, context="cached particle-filter feature matrix")
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
                "pf",
                prepared.failure_reason or "empty_prediction_region",
                requested_device=self.device,
            )
            self._put(key, output)
            return output

        cfg = self.config
        rng = np.random.default_rng(
            deterministic_seed(task, cfg.seed, self.fold_id, self.protocol)
        )
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
        expected_grad = float(prepared.expected_gradient[first])
        particles = np.full(
            cfg.n_particles, float(prepared.expected_track[previous_row])
        )
        particles += rng.normal(0.0, max(cfg.process_noise, 1e-9), cfg.n_particles)
        gradients = np.full(cfg.n_particles, expected_grad)
        gradients += rng.normal(0.0, cfg.gradient_noise, cfg.n_particles)
        weights = np.full(cfg.n_particles, 1.0 / cfg.n_particles)

        for j, row in enumerate(rows):
            row = int(row)
            delta_md = max(abs(float(prepared.md[row] - prepared.md[previous_row])), 1.0)
            expected_step = float(
                prepared.expected_track[row] - prepared.expected_track[previous_row]
            )
            geom_grad = expected_step / delta_md
            gradients = 0.82 * gradients + 0.18 * geom_grad
            gradients += rng.normal(
                0.0,
                cfg.gradient_noise * np.sqrt(max(delta_md / max(cfg.stride, 1), 1.0)),
                cfg.n_particles,
            )
            gradients = np.clip(gradients, -cfg.max_abs_gradient, cfg.max_abs_gradient)
            particles += expected_step + (gradients - geom_grad) * delta_md
            particles += rng.normal(0.0, cfg.process_noise * np.sqrt(delta_md), cfg.n_particles)
            particles = np.clip(particles, prepared.ref.tvt_min, prepared.ref.tvt_max)

            if prepared.gr_missing[row] or not np.isfinite(prepared.signal[row]):
                fallback_s[j] = 1.0
                weights.fill(1.0 / cfg.n_particles)
                misfit = np.zeros(cfg.n_particles)
            else:
                reference = np.interp(particles, prepared.ref.grid, prepared.ref.gr_z)
                misfit = np.abs(reference - prepared.signal[row])
                log_weight = -0.5 * (misfit / cfg.observation_sigma) ** 2
                # A weak geometry tether prevents repeated GR motifs from
                # launching an implausible branch without making geometry the
                # observation itself.
                scale = max(4.0, 0.02 * delta_md)
                log_weight -= 0.5 * (
                    (particles - prepared.expected_track[row]) / scale
                ) ** 2
                log_weight -= float(np.max(log_weight))
                weights = np.exp(log_weight)
                total = float(weights.sum())
                if not np.isfinite(total) or total <= 1e-15:
                    fallback_s[j] = 1.0
                    weights.fill(1.0 / cfg.n_particles)
                else:
                    weights /= total

            mean = float(np.sum(weights * particles))
            q10 = weighted_quantile(particles, weights, 0.10)
            q90 = weighted_quantile(particles, weights, 0.90)
            ess = 1.0 / float(np.sum(weights * weights))
            concentration = np.clip(1.0 - (ess - 1.0) / max(cfg.n_particles - 1, 1), 0.0, 1.0)
            track_s[j] = mean
            grad_s[j] = float(np.sum(weights * gradients))
            spread_s[j] = max(0.0, q90 - q10)
            confidence_s[j] = concentration if not fallback_s[j] else 0.0
            misfit_s[j] = float(np.sum(weights * misfit)) if misfit.size else 0.0

            if ess < cfg.resample_ess_fraction * cfg.n_particles:
                pick = systematic_resample(weights, rng)
                particles = particles[pick]
                gradients = gradients[pick]
                weights.fill(1.0 / cfg.n_particles)
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
                "pf_track": track,
                "pf_shift": track - anchor,
                "pf_gradient": gradient,
                "pf_confidence": confidence,
                "pf_branch_spread": spread,
                "pf_path_smoothness": smooth,
                "pf_gr_misfit": misfit,
                "pf_fallback": fallback,
            }
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame = frame[list(self.feature_columns)]
        assert_safe_features(frame.columns, context="particle-filter feature matrix")

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
            "n_particles": cfg.n_particles,
            "stride": cfg.stride,
            "n_generated_rows": task.n_predict,
        }
        output = PathFeatureOutput(frame=frame, diagnostics=diagnostics)
        self._put(key, output)
        return output


# Public short name plus concise functional API for notebooks and tests.
ParticleFilter = ParticleFilterFeatureGenerator


def particle_filter_features(
    task: InferenceTask,
    *,
    config: ParticleFilterConfig | None = None,
    cache=None,
    dataset_version: str = "rogii-mounted-v1",
    fold_id: Any = "inference",
    protocol: str = "inference",
    device: str = "auto",
) -> PathFeatureOutput:
    return ParticleFilterFeatureGenerator(
        config,
        cache=cache,
        dataset_version=dataset_version,
        fold_id=fold_id,
        protocol=protocol,
        device=device,
    ).generate(task)
