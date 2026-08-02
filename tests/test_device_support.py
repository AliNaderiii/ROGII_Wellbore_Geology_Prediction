"""CPU/GPU device selection for the boosted residual models.

Pinned invariants:

1. ``cpu`` mode never probes a GPU and produces the documented CPU params
   (``device_type="cpu"`` / ``n_jobs``; ``task_type="CPU"`` / ``thread_count``).
2. ``--device`` parses to exactly ``{auto, cpu, gpu}``; anything else is a
   parser error, and ``gpu`` reaches the model configuration.
3. A GPU initialisation failure is caught, its exact reason recorded, and
   the run continues on CPU.
4. A no-GPU environment (libraries absent or GPU-less) resolves ``auto`` to
   CPU without raising.
5. The CPU fallback is deterministic: identical inputs give identical params
   and identical fitted predictions.
6. Device selection never changes Ridge / PF-Beam / gate / promotion code
   paths.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.device import (  # noqa: E402
    DEVICE_CHOICES,
    CPU_RESOLUTION,
    DeviceResolution,
    resolve_device,
)
from src import trajectory_stack as ts  # noqa: E402

REPORT_KEYS = (
    "device_requested",
    "device_selected",
    "lgbm_device",
    "catboost_device",
    "gpu_fallback_reason",
)


def _boom(reason="cuda driver missing"):
    def _probe():
        raise RuntimeError(reason)

    return _probe


# ---------------------------------------------------------------------------
# 1. CPU mode
# ---------------------------------------------------------------------------


def test_cpu_mode_never_probes_gpu():
    calls = []

    def _probe():
        calls.append(1)
        return True, ""

    res = resolve_device("cpu", probe_lightgbm=_probe, probe_catboost=_probe)
    assert calls == []
    assert res.selected == "cpu"
    assert (res.lgbm_device, res.catboost_device) == ("cpu", "cpu")
    assert res.gpu_fallback_reason == ""


def test_cpu_mode_library_parameters():
    res = resolve_device("cpu")
    lgbm = res.lgbm_params(thread_count=7)
    assert lgbm["device_type"] == "cpu"
    assert lgbm["n_jobs"] == 7
    assert "gpu_use_dp" not in lgbm

    cat = res.catboost_params(thread_count=7)
    assert cat["task_type"] == "CPU"
    assert cat["thread_count"] == 7


def test_cpu_report_keys_present_and_serialisable():
    report = resolve_device("cpu").as_report()
    assert set(report) == set(REPORT_KEYS)
    assert json.loads(json.dumps(report)) == report
    assert report["device_requested"] == "cpu"
    assert report["device_selected"] == "cpu"


def test_default_learner_is_cpu_and_carries_device_report():
    learner = ts.LightGBMResidual(seed=0)
    assert learner.device is CPU_RESOLUTION
    assert learner.effective_device == "cpu"
    info = learner.info
    for key in REPORT_KEYS:
        assert hasattr(info, key)
    assert info.lgbm_device == "cpu"


# ---------------------------------------------------------------------------
# 2. GPU flag parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["auto", "cpu", "gpu"])
def test_experiment_cli_accepts_device_values(value, monkeypatch):
    import scripts.run_trajectory_stack_experiment as exp

    seen = {}
    monkeypatch.setattr(
        exp, "resolve_device", lambda req, log=None: seen.setdefault("req", req) or CPU_RESOLUTION
    )
    # Parse only: force an early, deterministic exit after device resolution.
    monkeypatch.setattr(exp, "ensure_reports_dir", lambda: (_ for _ in ()).throw(SystemExit(99)))
    with pytest.raises(SystemExit):
        exp.main(["--device", value, "--quiet"])
    assert seen["req"] == value


def test_experiment_cli_rejects_unknown_device():
    import scripts.run_trajectory_stack_experiment as exp

    with pytest.raises(SystemExit):
        exp.main(["--device", "tpu"])


def test_builder_cli_exposes_same_device_choices():
    import scripts.build_gated_submission as bgs

    assert tuple(DEVICE_CHOICES) == ("auto", "cpu", "gpu")
    assert hasattr(bgs, "resolve_device")
    src = Path(bgs.__file__).read_text()
    assert "--device" in src
    assert "device.as_report()" in src


def test_gpu_request_reaches_library_parameters():
    res = DeviceResolution(requested="gpu", lgbm_device="gpu", catboost_device="gpu")
    lgbm = res.lgbm_params(thread_count=4)
    assert lgbm["device_type"] == "gpu"
    assert lgbm["gpu_use_dp"] is False
    assert lgbm["deterministic"] is True

    cat = res.catboost_params(thread_count=4)
    assert cat["task_type"] == "GPU"
    assert cat["devices"] == "0"
    assert cat["allow_writing_files"] is False
    assert "thread_count" not in cat

    assert res.selected == "gpu"
    assert res.as_report()["device_selected"] == "gpu"


def test_gpu_mode_selected_when_probes_succeed():
    res = resolve_device(
        "gpu",
        probe_lightgbm=lambda: (True, ""),
        probe_catboost=lambda: (True, ""),
    )
    assert res.as_report() == {
        "device_requested": "gpu",
        "device_selected": "gpu",
        "lgbm_device": "gpu",
        "catboost_device": "gpu",
        "gpu_fallback_reason": "",
    }


# ---------------------------------------------------------------------------
# 3. GPU failure fallback
# ---------------------------------------------------------------------------


def test_gpu_failure_falls_back_to_cpu_with_exact_reason():
    logs: list[str] = []
    res = resolve_device(
        "gpu",
        log=logs.append,
        probe_lightgbm=lambda: (False, "lightgbm_gpu_unavailable: LightGBMError: No OpenCL device found"),
        probe_catboost=lambda: (False, "catboost_gpu_unavailable: CatBoostError: no CUDA device"),
    )
    assert res.selected == "cpu"
    assert "No OpenCL device found" in res.gpu_fallback_reason
    assert "no CUDA device" in res.gpu_fallback_reason
    assert any("falling back to CPU" in m for m in logs)
    # And the params really are the CPU ones.
    assert res.lgbm_params(3)["device_type"] == "cpu"
    assert res.catboost_params(3)["task_type"] == "CPU"


def test_probe_exception_is_caught_not_propagated():
    res = resolve_device(
        "gpu",
        probe_lightgbm=_boom("cuda driver missing"),
        probe_catboost=_boom("libcudart.so.11 not found"),
    )
    assert res.selected == "cpu"
    assert "RuntimeError" in res.gpu_fallback_reason
    assert "cuda driver missing" in res.gpu_fallback_reason
    assert "libcudart.so.11 not found" in res.gpu_fallback_reason


def test_partial_gpu_failure_keeps_the_working_library_on_gpu():
    res = resolve_device(
        "gpu",
        probe_lightgbm=lambda: (True, ""),
        probe_catboost=lambda: (False, "catboost_gpu_unavailable: CatBoostError: no CUDA device"),
    )
    assert res.lgbm_device == "gpu"
    assert res.catboost_device == "cpu"
    assert res.selected == "gpu"
    assert "no CUDA device" in res.gpu_fallback_reason


def test_real_probes_never_raise_in_this_environment():
    from src.device import probe_catboost_gpu, probe_lightgbm_gpu

    for probe in (probe_lightgbm_gpu, probe_catboost_gpu):
        ok, reason = probe()
        assert isinstance(ok, bool)
        assert isinstance(reason, str)
        if not ok:
            assert reason  # a failure must always explain itself


# ---------------------------------------------------------------------------
# 4. No-GPU environments
# ---------------------------------------------------------------------------


def test_auto_without_any_gpu_resolves_to_cpu():
    res = resolve_device(
        "auto",
        probe_lightgbm=lambda: (False, "lightgbm_import_failed: ModuleNotFoundError: No module named 'lightgbm'"),
        probe_catboost=lambda: (False, "catboost_import_failed: ModuleNotFoundError: No module named 'catboost'"),
    )
    assert res.as_report()["device_requested"] == "auto"
    assert res.selected == "cpu"
    assert "No module named" in res.gpu_fallback_reason


def test_auto_on_this_machine_resolves_without_raising():
    res = resolve_device("auto")
    assert res.selected in ("cpu", "gpu")
    assert set(res.as_report()) == set(REPORT_KEYS)


def test_auto_honours_rogii_device_env_override(monkeypatch):
    monkeypatch.setenv("ROGII_DEVICE", "cpu")
    calls = []
    res = resolve_device(
        "auto",
        probe_lightgbm=lambda: calls.append(1) or (True, ""),
        probe_catboost=lambda: calls.append(1) or (True, ""),
    )
    assert res.selected == "cpu"
    assert calls == []


def test_unknown_device_string_is_a_value_error():
    with pytest.raises(ValueError):
        resolve_device("quantum")


# ---------------------------------------------------------------------------
# 5. Deterministic CPU fallback
# ---------------------------------------------------------------------------


def test_cpu_fallback_parameters_are_deterministic():
    a = resolve_device("gpu", probe_lightgbm=_boom(), probe_catboost=_boom())
    b = resolve_device("gpu", probe_lightgbm=_boom(), probe_catboost=_boom())
    assert a.as_report() == b.as_report()
    assert a.lgbm_params(4) == b.lgbm_params(4)
    assert a.catboost_params(4) == b.catboost_params(4)
    assert a.lgbm_params(4)["deterministic"] is True


def test_cpu_fallback_learner_matches_forced_cpu_learner():
    forced = resolve_device("cpu")
    fell_back = resolve_device("gpu", probe_lightgbm=_boom(), probe_catboost=_boom())
    assert forced.lgbm_params(4) == fell_back.lgbm_params(4)
    assert forced.catboost_params(4) == fell_back.catboost_params(4)
    assert forced.selected == fell_back.selected == "cpu"


def test_cpu_fallback_predictions_are_reproducible(mount, monkeypatch):
    """Ridge/anchor behaviour is unchanged and reproducible under fallback."""
    from src.baselines import RidgeBaseline
    from src.data import discover_wells, load_well
    from src.tasks import make_task

    files = discover_wells("train")
    tasks = []
    for wid in ("TRW001", "TRW006", "TRW007", "TRW008", "TRW009"):
        if wid not in files:
            continue
        try:
            tasks.append(make_task(load_well(files[wid]), "real"))
        except Exception:
            pass
    if len(tasks) < 4:
        pytest.skip("mount produced too few tasks")

    fell_back = resolve_device("gpu", probe_lightgbm=_boom(), probe_catboost=_boom())
    assert fell_back.selected == "cpu"

    preds = []
    for _ in range(2):
        model = RidgeBaseline(alignment_features=False)
        model.fit(tasks)
        preds.append(np.concatenate([model.predict(t.inputs()) for t in tasks]))
    assert np.array_equal(preds[0], preds[1])


# ---------------------------------------------------------------------------
# 6. Device never leaks into non-boosting components
# ---------------------------------------------------------------------------


def test_stack_and_gate_configs_default_to_cpu():
    assert ts.StackConfig().device is CPU_RESOLUTION
    assert ts.TrajectoryGateConfig().device is CPU_RESOLUTION


def test_build_residual_learners_propagates_device():
    gpu = DeviceResolution(requested="gpu", lgbm_device="gpu", catboost_device="gpu")
    learners = ts.build_residual_learners(seed=0, device=gpu)
    for learner in learners.values():
        assert learner.device is gpu
        assert learner.effective_device == "gpu"
    if not learners:
        pytest.skip("no boosting library installed in this environment")


def test_build_stack_models_keeps_ridge_arm_untouched():
    from src.baselines import RidgeBaseline

    anchor = RidgeBaseline(alignment_features=False)
    gpu = DeviceResolution(requested="gpu", lgbm_device="gpu", catboost_device="gpu")
    models = ts.build_stack_models(
        (ts.ARM_RIDGE,),
        anchor_model=anchor,
        pf_factory=lambda: None,
        beam_factory=lambda: None,
        boost_kw=dict(seed=0),
        device=gpu,
    )
    assert models[ts.ARM_RIDGE] is anchor
    assert not hasattr(anchor, "device")


# ---------------------------------------------------------------------------
# 7. Device parameters really reach the booster fit calls
# ---------------------------------------------------------------------------


class _StubBooster:
    best_iteration = 3
    best_score = {"valid_0": {"rmse": 1.0}}

    def predict(self, X):
        return np.zeros(len(X))


def test_lightgbm_fit_receives_gpu_parameters(monkeypatch):
    captured = {}

    class _StubLGB:
        @staticmethod
        def Dataset(X, label=None, **kw):
            return ("ds", len(X))

        @staticmethod
        def early_stopping(rounds, verbose=False):
            return "cb"

        @staticmethod
        def train(params, dtrain, **kw):
            captured.update(params)
            return _StubBooster()

    monkeypatch.setattr(ts, "_lgb", _StubLGB)
    gpu = DeviceResolution(requested="gpu", lgbm_device="gpu", catboost_device="gpu")
    learner = ts.LightGBMResidual(seed=0, thread_count=9, device=gpu)
    learner._fit_booster(np.zeros((10, 2)), np.zeros(10), None, None)
    assert captured["device_type"] == "gpu"
    assert captured["gpu_use_dp"] is False
    assert captured["deterministic"] is True

    captured.clear()
    cpu_learner = ts.LightGBMResidual(seed=0, thread_count=9, device=resolve_device("cpu"))
    cpu_learner._fit_booster(np.zeros((10, 2)), np.zeros(10), None, None)
    assert captured["device_type"] == "cpu"
    assert captured["n_jobs"] == 9


def test_catboost_fit_receives_gpu_parameters(monkeypatch):
    captured = {}

    class _StubModel:
        def __init__(self, **params):
            captured.update(params)

        def fit(self, train_pool, eval_set=None, use_best_model=False):
            return self

        def get_best_iteration(self):
            return 2

        def get_best_score(self):
            return {"validation": {"RMSE": 1.0}}

    monkeypatch.setattr(ts, "CatBoostRegressor", _StubModel)
    monkeypatch.setattr(ts, "Pool", lambda X, y=None: ("pool", len(X)))

    gpu = DeviceResolution(requested="gpu", lgbm_device="gpu", catboost_device="gpu")
    ts.CatBoostResidual(seed=0, thread_count=9, device=gpu)._fit_booster(
        np.zeros((10, 2)), np.zeros(10), None, None
    )
    assert captured["task_type"] == "GPU"
    assert captured["devices"] == "0"
    assert captured["allow_writing_files"] is False
    assert "thread_count" not in captured

    captured.clear()
    ts.CatBoostResidual(seed=0, thread_count=9, device=resolve_device("cpu"))._fit_booster(
        np.zeros((10, 2)), np.zeros(10), None, None
    )
    assert captured["task_type"] == "CPU"
    assert captured["thread_count"] == 9


def test_run_environment_carries_device_keys(monkeypatch, mount, tmp_path):
    """A real (synthetic-mount) run writes the five device keys."""
    import scripts.run_trajectory_stack_experiment as exp

    reports = tmp_path / "reports"
    rc = exp.main(
        [
            "--device", "cpu",
            "--max-wells", "8",
            "--n-splits", "2",
            "--inner-splits", "2",
            "--tune-splits", "2",
            "--arms", "ridge_default",
            "--protocols", "unseen_well",
            "--n-bootstrap", "10",
            "--reports-dir", str(reports),
            "--quiet",
        ]
    )
    assert rc == 0
    env_files = list(reports.glob("*trajectory_stack_run_environment.json"))
    assert env_files, "no run_environment.json written"
    env = json.loads(env_files[0].read_text())
    for key in REPORT_KEYS:
        assert key in env, key
    assert env["device_requested"] == "cpu"
    assert env["device_selected"] == "cpu"
    assert env["gpu_fallback_reason"] == ""
