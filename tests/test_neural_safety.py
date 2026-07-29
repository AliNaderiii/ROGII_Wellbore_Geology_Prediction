"""Unit tests for the first leakage-safe neural experiment pass.

The tests use only the repository's synthetic fixture.  They prove structure
and provenance; they are not evidence of competition performance.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest


def _mod(name):
    import importlib
    import sys
    key = f"src.{name}"
    return importlib.reload(sys.modules[key]) if key in sys.modules else importlib.import_module(key)


def _task(mount, well_id="TRW006", mode="real"):
    data = _mod("data")
    tasks = _mod("tasks")
    files = data.discover_wells("train")
    return tasks.make_task(data.load_well(files[well_id]), mode)


def test_neural_inference_features_have_no_hidden_tvt_or_typewell_geology(mount):
    neural = _mod("neural")
    task = _task(mount).inputs()
    frame = neural._safe_feature_frame(task)
    assert frame.shape == (task.n_predict, len(neural.SAFE_SEQUENCE_FEATURES))
    assert not any("tvt" == str(c).lower() for c in neural.SAFE_SEQUENCE_FEATURES)
    assert not any("geology" in str(c).lower() for c in neural.SAFE_SEQUENCE_FEATURES)
    task.assert_no_target()
    assert not hasattr(task, "target")


def test_nested_examples_are_contiguous_and_prefix_only(mount):
    neural = _mod("neural")
    task = _task(mount)
    config = neural.NeuralConfig(min_pseudo_suffix_rows=64, max_sequence_rows=128)
    examples = neural.build_sequence_examples([task], config)
    assert examples
    assert {e.label_source for e in examples} >= {"real_hidden_suffix", "visible_prefix_pseudo_holdout"}
    for example in examples:
        inp = example.task
        assert inp.n_predict == len(example.target_residual)
        assert not np.isfinite(inp.tvt_known[inp.start:inp.stop]).any()
        if example.label_source == "visible_prefix_pseudo_holdout":
            assert inp.stop <= task.inputs().start
            assert inp.start >= neural.NeuralConfig().min_prefix_rows


def test_padding_mask_excludes_nan_targets_and_padding(mount):
    neural = _mod("neural")
    task = _task(mount)
    config = neural.NeuralConfig(max_sequence_rows=128)
    examples = neural.build_sequence_examples([task], config)
    x1, y1, _ = neural.SequenceDataset(examples[:1], config)[0]
    # Make an explicit second item shorter and with one unavailable target.
    x2, y2 = x1[: max(2, len(x1) // 2)].copy(), y1[: max(2, len(y1) // 2)].copy()
    y2[-1] = np.nan
    batch = neural.collate_sequence_batch([(x1, y1, examples[0]), (x2, y2, examples[0])])
    assert batch.features.shape[0] == 2
    assert batch.features.shape[1] == len(x1)
    assert batch.mask[0].all()
    assert not batch.mask[1, len(x2) - 1]
    assert not batch.mask[1, len(x2):].any()
    assert np.all(batch.features[1, len(x2):] == 0.0)


def test_public_duplicate_cannot_enter_neural_sequence_construction(mount):
    neural = _mod("neural")
    with pytest.raises(_mod("validation").BlockedWellError):
        neural.build_sequence_examples([replace(_task(mount), inputs_=replace(_task(mount).inputs_, well_id="000d7d20"))])


def test_ridge_fallback_is_exact_for_rejected_gate(mount):
    neural_hybrid = _mod("hybrid")
    baselines = _mod("baselines")
    data = _mod("data")
    tasks_mod = _mod("tasks")
    files = data.discover_wells("train")
    train = [tasks_mod.make_task(data.load_well(files[w]), "real") for w in ("TRW006", "TRW007")]
    anchor = baselines.RidgeBaseline(alpha=10.0, alignment_features=False).fit(train)
    gate = neural_hybrid.ConservativeRidgeNeuralGate()
    gate.anchor_model = anchor
    # The default state is a rejected gate; no candidate is allowed through.
    inp = tasks_mod.make_task(data.load_well(files["TRW008"]), "real").inputs()
    expected = anchor.predict(inp)
    actual = gate.predict(inp)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not _mod("neural").HAVE_TORCH, reason="PyTorch is optional")
def test_tiny_gru_is_deterministic_and_anchored(mount):
    neural = _mod("neural")
    data = _mod("data")
    tasks_mod = _mod("tasks")
    files = data.discover_wells("train")
    train = [tasks_mod.make_task(data.load_well(files[w]), "real") for w in ("TRW006", "TRW007", "TRW008", "TRW009")]
    config = neural.NeuralConfig(
        architecture="gru", max_epochs=1, patience=1, batch_size=2,
        max_sequence_rows=64, min_pseudo_suffix_rows=32, seed=123, device="cpu",
    )
    a = neural.NeuralResidualModel(config=config).fit(train)
    b = neural.NeuralResidualModel(config=config).fit(train)
    inp = tasks_mod.make_task(data.load_well(files["TRW001"]), "real").inputs()
    np.testing.assert_array_equal(a.predict(inp), b.predict(inp))
    assert np.isfinite(a.predict(inp)).all()
    assert a.training_report["parameter_count"] < 5_000_000
