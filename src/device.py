"""Device (CPU/GPU) resolution for the boosted residual learners.

Scope is deliberately narrow: this module decides *where* LightGBM and
CatBoost run and returns the exact library parameters for that choice. It
never touches Ridge, PF/Beam, gate logic, feature provenance or promotion
rules — those stay bit-identical whatever device is selected.

Contract
--------
``resolve_device("auto"|"cpu"|"gpu")`` returns a :class:`DeviceResolution`
that always succeeds. GPU is used only when the installed library actually
supports it, proven by a tiny real fit (a probe), not by assumption. Any
probe failure is caught, its exact reason recorded in
``gpu_fallback_reason``, and the resolution falls back to CPU so an
experiment can never be killed by a missing/broken GPU stack.

The five reported keys (``device_requested``, ``device_selected``,
``lgbm_device``, ``catboost_device``, ``gpu_fallback_reason``) are written
into ``run_environment.json`` and the model reports.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = [
    "DEVICE_AUTO",
    "DEVICE_CPU",
    "DEVICE_GPU",
    "DEVICE_CHOICES",
    "DeviceResolution",
    "resolve_device",
    "probe_lightgbm_gpu",
    "probe_catboost_gpu",
]

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_GPU = "gpu"
DEVICE_CHOICES = (DEVICE_AUTO, DEVICE_CPU, DEVICE_GPU)

#: Rows/columns used by the GPU probes. Small enough to be instant, large
#: enough that LightGBM/CatBoost really initialise their GPU backend.
_PROBE_ROWS = 64
_PROBE_COLS = 4


def _exc_reason(prefix: str, exc: BaseException) -> str:
    """A single-line, exact-but-bounded description of a failure."""
    msg = " ".join(str(exc).split())
    if len(msg) > 400:
        msg = msg[:397] + "..."
    return f"{prefix}: {type(exc).__name__}: {msg}" if msg else f"{prefix}: {type(exc).__name__}"


def _probe_arrays():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(_PROBE_ROWS, _PROBE_COLS))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=_PROBE_ROWS)
    return X, y


def probe_lightgbm_gpu() -> tuple[bool, str]:
    """Return ``(True, "")`` if LightGBM can really train on a GPU here.

    Any import error, missing GPU build or CUDA/OpenCL initialisation error
    is caught and returned as the exact reason string.
    """
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, _exc_reason("lightgbm_import_failed", exc)
    try:
        X, y = _probe_arrays()
        params = {
            "objective": "regression",
            "device_type": "gpu",
            "gpu_use_dp": False,
            "verbosity": -1,
            "num_leaves": 4,
            "min_data_in_leaf": 1,
            "seed": 0,
        }
        lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=1)
    except Exception as exc:
        return False, _exc_reason("lightgbm_gpu_unavailable", exc)
    return True, ""


def probe_catboost_gpu() -> tuple[bool, str]:
    """Return ``(True, "")`` if CatBoost can really train on a GPU here."""
    try:
        from catboost import CatBoostRegressor
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, _exc_reason("catboost_import_failed", exc)
    try:
        from catboost.utils import get_gpu_device_count

        if int(get_gpu_device_count()) < 1:
            return False, "catboost_gpu_unavailable: get_gpu_device_count() == 0"
    except Exception as exc:
        # Not fatal on its own: fall through to the real fit probe below.
        probe_note = _exc_reason("catboost_device_count_failed", exc)
    else:
        probe_note = ""
    try:
        X, y = _probe_arrays()
        CatBoostRegressor(
            iterations=1,
            depth=2,
            task_type="GPU",
            devices="0",
            allow_writing_files=False,
            verbose=False,
            random_seed=0,
        ).fit(X, y)
    except Exception as exc:
        reason = _exc_reason("catboost_gpu_unavailable", exc)
        return False, f"{probe_note}; {reason}" if probe_note else reason
    return True, ""


@dataclass(frozen=True)
class DeviceResolution:
    """The outcome of device selection; safe to embed in JSON reports."""

    requested: str = DEVICE_CPU
    lgbm_device: str = DEVICE_CPU
    catboost_device: str = DEVICE_CPU
    gpu_fallback_reason: str = ""
    #: Per-library probe details (empty when no probe was needed).
    probe_notes: dict = field(default_factory=dict)

    @property
    def selected(self) -> str:
        """``gpu`` when at least one booster runs on the GPU, else ``cpu``."""
        return (
            DEVICE_GPU
            if DEVICE_GPU in (self.lgbm_device, self.catboost_device)
            else DEVICE_CPU
        )

    @property
    def used_gpu(self) -> bool:
        return self.selected == DEVICE_GPU

    # ---------------------------------------------------------- reporting --
    def as_report(self) -> dict:
        """The five keys required in run_environment.json / model reports."""
        return {
            "device_requested": self.requested,
            "device_selected": self.selected,
            "lgbm_device": self.lgbm_device,
            "catboost_device": self.catboost_device,
            "gpu_fallback_reason": self.gpu_fallback_reason,
        }

    def summary_line(self) -> str:
        base = (
            f"device: requested={self.requested} selected={self.selected} "
            f"(lightgbm={self.lgbm_device}, catboost={self.catboost_device})"
        )
        return f"{base}; GPU fallback reason: {self.gpu_fallback_reason}" if self.gpu_fallback_reason else base

    # ------------------------------------------------------ library params --
    def lgbm_params(self, thread_count: int) -> dict:
        """LightGBM parameters for the selected device.

        cpu: ``device_type="cpu"``, ``n_jobs=boost_threads``.
        gpu: ``device_type="gpu"``, ``gpu_use_dp=False``, deterministic
        behaviour where the library supports it.
        """
        if self.lgbm_device == DEVICE_GPU:
            return {
                "device_type": "gpu",
                "gpu_use_dp": False,
                # LightGBM honours `deterministic` on the GPU histogram
                # builder where supported; combined with force_row_wise it
                # removes the remaining multi-threaded build nondeterminism.
                "deterministic": True,
                "force_row_wise": True,
            }
        return {
            "device_type": "cpu",
            "n_jobs": int(thread_count),
            "deterministic": True,
        }

    def catboost_params(self, thread_count: int) -> dict:
        """CatBoost parameters for the selected device.

        cpu: ``task_type="CPU"``, ``thread_count=boost_threads``.
        gpu: ``task_type="GPU"``, ``devices="0"``, ``allow_writing_files=False``.
        """
        if self.catboost_device == DEVICE_GPU:
            return {
                "task_type": "GPU",
                "devices": "0",
                "allow_writing_files": False,
            }
        return {
            "task_type": "CPU",
            "thread_count": int(thread_count),
            "allow_writing_files": False,
        }


#: The resolution used when nothing has been resolved (pure-CPU default).
CPU_RESOLUTION = DeviceResolution(requested=DEVICE_CPU)


def _env_default_device() -> str | None:
    value = os.environ.get("ROGII_DEVICE", "").strip().lower()
    return value if value in DEVICE_CHOICES else None


def resolve_device(
    requested: str = DEVICE_AUTO,
    *,
    log=None,
    probe_lightgbm=probe_lightgbm_gpu,
    probe_catboost=probe_catboost_gpu,
) -> DeviceResolution:
    """Resolve ``auto``/``cpu``/``gpu`` into a concrete per-library device.

    Never raises for a device reason: a broken or absent GPU stack is
    logged and downgraded to CPU. ``log`` is an optional ``print``-like
    callable used to surface the exact fallback reason.
    """
    requested = (requested or DEVICE_AUTO).strip().lower()
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {requested!r}; choose from {DEVICE_CHOICES}")
    env_override = _env_default_device()
    if requested == DEVICE_AUTO and env_override is not None:
        requested_effective = env_override
    else:
        requested_effective = requested

    def _emit(msg: str) -> None:
        if log is not None:
            log(msg)

    if requested_effective == DEVICE_CPU:
        return DeviceResolution(requested=requested)

    lgbm_ok, lgbm_reason = False, ""
    cat_ok, cat_reason = False, ""
    try:
        lgbm_ok, lgbm_reason = probe_lightgbm()
    except Exception as exc:  # a probe must never kill the experiment
        lgbm_ok, lgbm_reason = False, _exc_reason("lightgbm_gpu_probe_crashed", exc)
    try:
        cat_ok, cat_reason = probe_catboost()
    except Exception as exc:
        cat_ok, cat_reason = False, _exc_reason("catboost_gpu_probe_crashed", exc)

    reasons = [r for r in (lgbm_reason, cat_reason) if r]
    fallback_reason = "; ".join(reasons)
    resolution = DeviceResolution(
        requested=requested,
        lgbm_device=DEVICE_GPU if lgbm_ok else DEVICE_CPU,
        catboost_device=DEVICE_GPU if cat_ok else DEVICE_CPU,
        gpu_fallback_reason=fallback_reason,
        probe_notes={"lightgbm": lgbm_reason, "catboost": cat_reason},
    )
    if fallback_reason:
        _emit(
            f"      GPU requested ({requested}) but falling back to CPU where needed: {fallback_reason}"
        )
    _emit(f"      {resolution.summary_line()}")
    return resolution
