"""Leakage-safe sequence residual models for hidden-suffix TVT prediction.

This module is deliberately a small, optional PyTorch track.  It does not
replace :class:`src.baselines.RidgeBaseline`; Ridge remains the production
fallback and the comparison anchor.  The public model contract is the same as
the existing baselines: ``fit`` receives ``WellTask`` objects, while
``predict`` receives only an ``InferenceTask`` (which has no target field).

Important invariants
--------------------
* The supervised target is always ``TVT - last_visible_TVT_input``.
* Training examples are complete contiguous suffix holdouts, never random row
  masks.  Nested examples are made only inside the visible ``TVT_input``
  prefix; the real suffix uses the training target handed to ``fit``.
* Every inference matrix is built from ``InferenceTask`` and the existing
  manifest-cleared, alignment-free feature frame.  No ``TVT`` or
  ``Typewell Geology`` column is read.
* Padding is explicit.  Masks are applied after every temporal block and to
  every loss component, so padded zeros cannot become a sequence-length cue.
* PyTorch is optional at import time.  Kaggle's PyTorch installation is used
  when present; CPU inference/training remains supported.
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from src.baselines import BaselineModel
from src.features import build_features, feature_columns
from src.manifest import assert_safe_features
from src.tasks import InferenceTask, WellTask

try:  # PyTorch is present in the Kaggle image, but not required by the loader.
    import torch
    from torch import nn
    from torch.nn import functional as F

    HAVE_TORCH = True
except Exception:  # pragma: no cover - exercised on the minimal CPU CI image
    torch = None
    nn = None
    F = None
    HAVE_TORCH = False


SAFE_SEQUENCE_FEATURES = tuple(feature_columns(alignment_features=False))
# Alignment is an explicit separate candidate path in src.geoanchor.py.  The
# first neural pass must not quietly consume it.
assert "align_tvt" not in SAFE_SEQUENCE_FEATURES


class NeuralUnavailable(RuntimeError):
    """Raised only when a PyTorch model is requested without PyTorch."""


def require_torch() -> None:
    if not HAVE_TORCH:
        raise NeuralUnavailable(
            "PyTorch is not installed. The neural experiment is optional; "
            "install/use the standard Kaggle PyTorch image or keep Ridge Default."
        )


def set_deterministic(seed: int) -> None:
    """Set all RNGs used by this module and request deterministic kernels."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    if HAVE_TORCH:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        # These flags are safe for the small networks below.  We intentionally
        # do not enable TF32 because it can make CPU/GPU comparisons drift.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # older Kaggle torch
                torch.use_deterministic_algorithms(True)


def _finite_fill(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    out = np.asarray(values, dtype="float64").copy()
    out[~np.isfinite(out)] = fallback
    return out


def _safe_feature_frame(task: InferenceTask) -> np.ndarray:
    """Return only manifest-cleared, target-free rows for ``task``.

    ``build_features`` uses MD/X/Y/Z/GR, visible ``tvt_known`` prefix and the
    Typewell TVT/GR arrays.  With ``alignment=False`` it does not construct the
    established NCC alignment columns.  The final assertion is retained here,
    rather than relying solely on the feature module, as a provenance guard
    specific to the neural path.
    """
    task.assert_no_target()
    frame = build_features(task, alignment=False).frame()
    assert_safe_features(frame.columns, context="neural sequence features")
    if "TVT" in frame.columns or "TVT_input" in frame.columns:
        raise AssertionError("neural feature frame contains a horizontal TVT input")
    if any("geology" in str(c).lower() for c in frame.columns):
        raise AssertionError("neural feature frame contains Typewell Geology")
    columns = list(frame.columns)
    if tuple(columns) != SAFE_SEQUENCE_FEATURES:
        # Reindex is not a relaxation: all names were admitted above, and the
        # expected order is fixed for reproducibility.
        frame = frame.reindex(columns=list(SAFE_SEQUENCE_FEATURES))
    arr = frame.to_numpy(dtype="float64", copy=True)
    return arr


@dataclass(frozen=True)
class SequenceExample:
    """One contiguous suffix example; the target is never passed to a model."""

    well_id: str
    task: InferenceTask
    target_residual: np.ndarray
    label_source: str  # ``real_hidden_suffix`` or ``visible_prefix_pseudo_holdout``
    boundary: int

    @property
    def n_rows(self) -> int:
        return int(self.target_residual.size)


def _pseudo_task(task: InferenceTask, boundary: int, stop: int) -> tuple[InferenceTask, np.ndarray] | None:
    """Build a prefix-only nested holdout without touching hidden labels."""
    if boundary < 1 or stop <= boundary or stop > task.start:
        return None
    source = np.asarray(task.tvt_known, dtype="float64")
    target = source[boundary:stop].copy()
    if not np.isfinite(target).any():
        return None
    known = source.copy()
    known[boundary:] = np.nan
    pseudo = replace(task, start=int(boundary), stop=int(stop), tvt_known=known)
    pseudo.assert_no_target()
    return pseudo, target


def _nested_boundaries(task: InferenceTask, config: "NeuralConfig") -> list[tuple[int, int]]:
    """Deterministic long-horizon boundaries within the visible prefix."""
    prefix = int(task.start)
    if prefix < config.min_prefix_rows + config.min_pseudo_suffix_rows:
        return []
    available = prefix - config.min_prefix_rows
    requested = [
        min(available, max(config.min_pseudo_suffix_rows, task.n_predict)),
        min(available, max(config.min_pseudo_suffix_rows, task.n_predict // 2)),
    ]
    out: list[tuple[int, int]] = []
    for length in requested:
        length = int(length)
        if length < config.min_pseudo_suffix_rows:
            continue
        out.append((prefix - length, prefix))
    # Never duplicate a pseudo example when a short prefix collapses the grid.
    return list(dict.fromkeys(out))[: max(0, int(config.max_pseudo_examples_per_well))]


def build_sequence_examples(
    tasks: Sequence[WellTask],
    config: "NeuralConfig | None" = None,
    *,
    include_real: bool = True,
    include_pseudo: bool = True,
) -> list[SequenceExample]:
    """Construct contiguous examples from fold-training wells only.

    This function is intentionally side-effect free and accepts no test wells.
    ``WellTask.target`` is read only here, inside supervised fitting.  Pseudo
    targets are sliced from ``InferenceTask.tvt_known`` and therefore come
    exclusively from the visible prefix.
    """
    config = config or NeuralConfig()
    # Reuse the repository's hard public-test exclusion rather than copying a
    # second list that could drift.  This check also protects direct callers
    # that bypass the outer validation harness.
    from src.validation import assert_no_blocked_wells
    assert_no_blocked_wells([t.well_id for t in tasks], context="neural supervised sequence construction")
    out: list[SequenceExample] = []
    for wrapped in sorted(tasks, key=lambda t: str(t.well_id)):
        inp = wrapped.inputs()
        inp.assert_no_target()
        if include_real and wrapped.target is not None:
            target = np.asarray(wrapped.target, dtype="float64")
            if target.size == inp.n_predict and np.isfinite(target).any():
                residual = target - float(inp.anchor_tvt if np.isfinite(inp.anchor_tvt) else 0.0)
                out.append(
                    SequenceExample(
                        str(inp.well_id), inp, residual, "real_hidden_suffix", int(inp.start)
                    )
                )
        if include_pseudo:
            for boundary, stop in _nested_boundaries(inp, config):
                made = _pseudo_task(inp, boundary, stop)
                if made is None:
                    continue
                pseudo, target = made
                anchor = pseudo.anchor_tvt
                if not np.isfinite(anchor):
                    continue
                out.append(
                    SequenceExample(
                        str(inp.well_id), pseudo, target - anchor,
                        "visible_prefix_pseudo_holdout", int(boundary)
                    )
                )
    return out


@dataclass(frozen=True)
class Batch:
    features: np.ndarray
    targets: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    examples: tuple[SequenceExample, ...]


def _sample_rows(example: SequenceExample, config: "NeuralConfig") -> tuple[np.ndarray, np.ndarray]:
    x = _safe_feature_frame(example.task)
    y = np.asarray(example.target_residual, dtype="float64")
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n <= config.max_sequence_rows:
        return x, y
    # Uniform deterministic downsampling retains both the boundary and the
    # far end of a long hidden suffix without allocating a huge padded tensor.
    idx = np.linspace(0, n - 1, config.max_sequence_rows, dtype=int)
    return x[idx], y[idx]


class SequenceDataset:
    """Small deterministic dataset wrapper used by the manual batch iterator."""

    def __init__(self, examples: Sequence[SequenceExample], config: "NeuralConfig | None" = None):
        self.examples = tuple(examples)
        self.config = config or NeuralConfig()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, SequenceExample]:
        x, y = _sample_rows(self.examples[index], self.config)
        return x, y, self.examples[index]


def collate_sequence_batch(items: Sequence[tuple[np.ndarray, np.ndarray, SequenceExample]]) -> Batch:
    """Right-pad a batch and return an explicit boolean validity mask."""
    if not items:
        raise ValueError("cannot collate an empty sequence batch")
    n_features = int(items[0][0].shape[1])
    lengths = np.asarray([len(x) for x, _, _ in items], dtype="int64")
    max_len = int(lengths.max())
    features = np.zeros((len(items), max_len, n_features), dtype="float32")
    targets = np.zeros((len(items), max_len), dtype="float32")
    mask = np.zeros((len(items), max_len), dtype=bool)
    for i, (x, y, _) in enumerate(items):
        n = int(lengths[i])
        if x.shape != (n, n_features) or y.shape != (n,):
            raise ValueError("inconsistent sequence shapes")
        finite_x = np.isfinite(x)
        features[i, :n] = np.where(finite_x, x, 0.0).astype("float32")
        finite_y = np.isfinite(y)
        targets[i, :n] = np.where(finite_y, y, 0.0).astype("float32")
        mask[i, :n] = finite_y
    return Batch(features, targets, mask, lengths, tuple(item[2] for item in items))


class FoldSafeScaler:
    """Feature scaler fitted only on the examples supplied to ``fit``."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_rows_fit_: int = 0

    def fit(self, examples: Sequence[SequenceExample], config: "NeuralConfig | None" = None):
        config = config or NeuralConfig()
        chunks = []
        for ex in examples:
            x, _ = _sample_rows(ex, config)
            chunks.append(np.asarray(x, dtype="float64"))
        if not chunks:
            raise ValueError("cannot fit a scaler without sequence examples")
        x = np.concatenate(chunks, axis=0)
        self.mean_ = np.nanmean(x, axis=0)
        self.scale_ = np.nanstd(x, axis=0)
        self.mean_[~np.isfinite(self.mean_)] = 0.0
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-8)] = 1.0
        self.n_rows_fit_ = int(len(x))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FoldSafeScaler must be fitted first")
        out = (np.asarray(x, dtype="float64") - self.mean_) / self.scale_
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
            "n_rows_fit": self.n_rows_fit_,
        }


@dataclass
class NeuralConfig:
    architecture: str = "gru"
    hidden_size: int = 96
    embedding_size: int = 48
    layers: int = 1
    dropout: float = 0.12
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 24
    max_epochs: int = 24
    patience: int = 5
    min_delta: float = 1e-4
    max_sequence_rows: int = 1024
    min_prefix_rows: int = 200
    min_pseudo_suffix_rows: int = 64
    max_pseudo_examples_per_well: int = 2
    gradient_clip: float = 1.0
    lambda_boundary: float = 0.02
    lambda_smooth1: float = 0.01
    lambda_smooth2: float = 0.005
    lambda_geometry: float = 0.0
    inner_valid_fraction: float = 0.2
    seed: int = 17
    device: str = "auto"

    def resolved_device(self) -> str:
        require_torch()
        if self.device == "cpu":
            return "cpu"
        if self.device == "gpu":
            if not torch.cuda.is_available():
                raise NeuralUnavailable("device='gpu' requested but CUDA is unavailable")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"


if HAVE_TORCH:

    class MLPResidualNet(nn.Module):
        def __init__(self, n_features: int, config: NeuralConfig):
            super().__init__()
            h, e = config.hidden_size, config.embedding_size
            self.body = nn.Sequential(
                nn.Linear(n_features, e), nn.LayerNorm(e), nn.GELU(),
                nn.Dropout(config.dropout), nn.Linear(e, h), nn.GELU(),
                nn.Dropout(config.dropout), nn.Linear(h, 1),
            )

        def forward(self, x, mask=None):
            return self.body(x).squeeze(-1)


    class GRUResidualNet(nn.Module):
        def __init__(self, n_features: int, config: NeuralConfig):
            super().__init__()
            h, e = config.hidden_size, config.embedding_size
            self.embed = nn.Sequential(nn.Linear(n_features, e), nn.LayerNorm(e), nn.GELU())
            self.gru = nn.GRU(
                e, h, num_layers=config.layers, batch_first=True,
                dropout=config.dropout if config.layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(h, 1))

        def forward(self, x, mask=None):
            z = self.embed(x)
            if mask is not None:
                z = z * mask.unsqueeze(-1).to(z.dtype)
            z, _ = self.gru(z)
            if mask is not None:
                z = z * mask.unsqueeze(-1).to(z.dtype)
            return self.head(z).squeeze(-1)


    class _TCNBlock(nn.Module):
        def __init__(self, channels: int, dropout: float):
            super().__init__()
            self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
            self.norm = nn.GroupNorm(1, channels)
            self.drop = nn.Dropout(dropout)

        def forward(self, z, mask):
            residual = z
            z = self.conv(z)
            z = self.norm(z)
            z = F.gelu(z)
            z = self.drop(z)
            z = z + residual
            return z * mask.unsqueeze(1).to(z.dtype)


    class TCNResidualNet(nn.Module):
        def __init__(self, n_features: int, config: NeuralConfig):
            super().__init__()
            h, e = config.hidden_size, config.embedding_size
            self.embed = nn.Conv1d(n_features, e, kernel_size=1)
            self.blocks = nn.ModuleList([_TCNBlock(e, config.dropout) for _ in range(max(1, config.layers))])
            self.head = nn.Conv1d(e, 1, kernel_size=1)

        def forward(self, x, mask=None):
            if mask is None:
                mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            z = self.embed(x.transpose(1, 2))
            z = z * mask.unsqueeze(1).to(z.dtype)
            for block in self.blocks:
                z = block(z, mask)
            return self.head(z).squeeze(1)

else:  # pragma: no cover - makes imports and safety tests work without torch

    class MLPResidualNet:  # type: ignore[no-redef]
        pass

    class GRUResidualNet:  # type: ignore[no-redef]
        pass

    class TCNResidualNet:  # type: ignore[no-redef]
        pass


def _make_network(n_features: int, config: NeuralConfig):
    require_torch()
    arch = config.architecture.lower()
    if arch == "mlp":
        return MLPResidualNet(n_features, config)
    if arch == "gru":
        return GRUResidualNet(n_features, config)
    if arch == "tcn":
        return TCNResidualNet(n_features, config)
    raise ValueError(f"unknown neural architecture {config.architecture!r}")


def _group_split(examples: Sequence[SequenceExample], config: "NeuralConfig"):
    ids = sorted({e.well_id for e in examples})
    if len(ids) < 5 or config.inner_valid_fraction <= 0:
        return list(examples), []
    # Use the repository's canonical GroupKFold implementation for the inner
    # split as well as the outer split.  This is a well-level partition, never
    # a random row split.  The seed fixes which of the deterministic folds is
    # used for early-stopping diagnostics.
    from src.validation import make_group_folds
    desired = max(2, int(round(1.0 / config.inner_valid_fraction)))
    n_splits = min(desired, len(ids))
    inner_folds = make_group_folds(ids, n_splits=n_splits, seed=config.seed)
    valid_ids = set(inner_folds[0].valid_ids)
    train = [e for e in examples if e.well_id not in valid_ids]
    valid = [e for e in examples if e.well_id in valid_ids]
    return train, valid


def _batches(examples: Sequence[SequenceExample], config: NeuralConfig, *, shuffle: bool, epoch: int):
    order = np.arange(len(examples), dtype=int)
    if shuffle:
        rng = np.random.default_rng(config.seed + 1009 * epoch)
        rng.shuffle(order)
    ds = SequenceDataset(examples, config)
    for i in range(0, len(order), max(1, config.batch_size)):
        items = [ds[int(j)] for j in order[i : i + config.batch_size]]
        yield collate_sequence_batch(items)


def _loss_components(pred, target, mask, config: NeuralConfig) -> dict:
    """Masked supervised + continuity/smoothness loss with named components."""
    valid = mask.to(pred.dtype)
    denom = valid.sum().clamp_min(1.0)
    # Huber is less sensitive to a badly aligned/long-horizon training well.
    supervised = F.smooth_l1_loss(pred, target, reduction="none")
    supervised = (supervised * valid).sum() / denom

    first = pred[:, 0]
    first_mask = mask[:, 0].to(pred.dtype)
    boundary = ((first * first) * first_mask).sum() / first_mask.sum().clamp_min(1.0)

    if pred.shape[1] >= 2:
        m1 = (mask[:, 1:] & mask[:, :-1]).to(pred.dtype)
        smooth1 = (((pred[:, 1:] - pred[:, :-1]) ** 2) * m1).sum() / m1.sum().clamp_min(1.0)
    else:
        smooth1 = pred.new_zeros(())
    if pred.shape[1] >= 3:
        m2 = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).to(pred.dtype)
        smooth2 = (((pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]) ** 2) * m2).sum() / m2.sum().clamp_min(1.0)
    else:
        smooth2 = pred.new_zeros(())
    geometry = pred.new_zeros(())
    total = (
        supervised
        + config.lambda_boundary * boundary
        + config.lambda_smooth1 * smooth1
        + config.lambda_smooth2 * smooth2
        + config.lambda_geometry * geometry
    )
    return {
        "total": total,
        "supervised_residual": supervised,
        "boundary_continuity": boundary,
        "first_difference": smooth1,
        "second_difference": smooth2,
        "geometry_consistency": geometry,
    }


def _run_epoch(net, examples, scaler, config, device, *, train: bool, epoch: int) -> dict:
    require_torch()
    net.train(train)
    totals: dict[str, float] = {k: 0.0 for k in (
        "total", "supervised_residual", "boundary_continuity",
        "first_difference", "second_difference", "geometry_consistency",
    )}
    n_batches = 0
    optimizer = getattr(net, "_optimizer", None)
    for batch in _batches(examples, config, shuffle=train, epoch=epoch):
        x = torch.from_numpy(scaler.transform(batch.features.reshape(-1, batch.features.shape[-1])).reshape(batch.features.shape)).to(device)
        y = torch.from_numpy(batch.targets).to(device)
        mask = torch.from_numpy(batch.mask).to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = net(x, mask)
            comp = _loss_components(pred, y, mask, config)
            if train:
                comp["total"].backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.gradient_clip)
                optimizer.step()
        for key in totals:
            totals[key] += float(comp[key].detach().cpu())
        n_batches += 1
    if not n_batches:
        return totals
    return {k: v / n_batches for k, v in totals.items()}


def _parameter_count(net) -> int:
    return int(sum(p.numel() for p in net.parameters())) if HAVE_TORCH else 0


class NeuralResidualModel(BaselineModel):
    """Small MLP/GRU/TCN residual predictor with fold-safe early stopping."""

    needs_alignment = False
    uses_spatial = False

    def __init__(self, config: NeuralConfig | None = None, *, architecture: str | None = None):
        self.config = copy.deepcopy(config or NeuralConfig())
        if architecture is not None:
            self.config.architecture = architecture
        self.name = f"neural_{self.config.architecture.lower()}"
        self.net = None
        self.scaler = FoldSafeScaler()
        self.training_report: dict = {}
        self.feature_names_ = list(SAFE_SEQUENCE_FEATURES)
        self._fitted = False

    def fit(self, tasks: Sequence[WellTask], **kw):
        require_torch()
        set_deterministic(self.config.seed)
        examples = build_sequence_examples(tasks, self.config)
        if not examples:
            raise ValueError("neural model received no finite contiguous training examples")
        inner_train, inner_valid = _group_split(examples, self.config)
        # If the inner split is too small, use all examples for early stopping
        # diagnostics but never claim an independent validation result.
        if not inner_valid:
            inner_train, inner_valid = list(examples), []
        early_scaler = FoldSafeScaler().fit(inner_train, self.config)
        device = torch.device(self.config.resolved_device())
        net = _make_network(len(SAFE_SEQUENCE_FEATURES), self.config).to(device)
        net._optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        best_state = copy.deepcopy(net.state_dict())
        best_score = math.inf
        best_epoch = 1
        no_improve = 0
        history = []
        t0 = time.perf_counter()
        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = _run_epoch(net, inner_train, early_scaler, self.config, device, train=True, epoch=epoch)
            valid_metrics = (
                _run_epoch(net, inner_valid, early_scaler, self.config, device, train=False, epoch=epoch)
                if inner_valid else train_metrics
            )
            score = float(valid_metrics["supervised_residual"])
            history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
            if score + self.config.min_delta < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(net.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.config.patience:
                break

        # Refit from scratch on every outer-fold training well.  The inner
        # validation set selected only the epoch count; the final scaler and
        # weights see no outer validation rows.
        self.scaler = FoldSafeScaler().fit(examples, self.config)
        final_net = _make_network(len(SAFE_SEQUENCE_FEATURES), self.config).to(device)
        final_net._optimizer = torch.optim.AdamW(
            final_net.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        final_history = []
        for epoch in range(1, best_epoch + 1):
            final_history.append(_run_epoch(final_net, examples, self.scaler, self.config, device, train=True, epoch=epoch))
        self.net = final_net.eval()
        self._fitted = True
        self.training_report = {
            "architecture": self.config.architecture,
            "n_outer_training_wells": len({t.well_id for t in tasks}),
            "n_examples": len(examples),
            "n_real_examples": sum(e.label_source == "real_hidden_suffix" for e in examples),
            "n_pseudo_examples": sum(e.label_source == "visible_prefix_pseudo_holdout" for e in examples),
            "inner_validation_wells": sorted({e.well_id for e in inner_valid}),
            "inner_cross_fitted": bool(inner_valid),
            "selected_epochs": int(best_epoch),
            "epochs_run_for_early_stop": len(history),
            "parameter_count": _parameter_count(self.net),
            "device": str(device),
            "runtime_seconds": time.perf_counter() - t0,
            "scaler_fit_rows": self.scaler.n_rows_fit_,
            "loss_components_last_epoch": final_history[-1] if final_history else {},
            "loss_coefficients": {
                "lambda_boundary": self.config.lambda_boundary,
                "lambda_smooth1": self.config.lambda_smooth1,
                "lambda_smooth2": self.config.lambda_smooth2,
                "lambda_geometry": self.config.lambda_geometry,
            },
        }
        return self

    def predict(self, task: InferenceTask, feats=None) -> np.ndarray:
        task.assert_no_target()
        anchor = self._anchor(task)
        if not self._fitted or self.net is None:
            return np.full(task.n_predict, anchor, dtype="float64")
        x = _safe_feature_frame(task)
        device = next(self.net.parameters()).device
        out = np.empty(task.n_predict, dtype="float64")
        self.net.eval()
        with torch.no_grad():
            for start in range(0, task.n_predict, self.config.max_sequence_rows):
                stop = min(task.n_predict, start + self.config.max_sequence_rows)
                chunk = x[start:stop]
                tx = torch.from_numpy(self.scaler.transform(chunk)[None, ...]).to(device)
                mask = torch.ones((1, len(chunk)), dtype=torch.bool, device=device)
                residual = self.net(tx, mask).detach().cpu().numpy()[0]
                out[start:stop] = anchor + residual
        if not np.isfinite(out).all():
            return np.full(task.n_predict, anchor, dtype="float64")
        return self._clip_to_typewell(task, out)

    def prediction_diagnostics(self, task, feats, pred) -> dict:
        return {
            "neural_architecture": self.config.architecture,
            "neural_parameter_count": self.training_report.get("parameter_count", 0),
            "neural_selected_epochs": self.training_report.get("selected_epochs", 0),
            "neural_correction_mean_abs": float(np.mean(np.abs(np.asarray(pred) - self._anchor(task)))),
            "neural_fallback": False,
        }


def make_neural_factory(architecture: str, config: NeuralConfig | None = None):
    """Return a validation-harness factory without global mutable state."""
    def factory():
        return NeuralResidualModel(config=copy.deepcopy(config), architecture=architecture)
    return factory


__all__ = [
    "HAVE_TORCH", "NeuralUnavailable", "NeuralConfig", "NeuralResidualModel",
    "SequenceExample", "SequenceDataset", "Batch", "collate_sequence_batch",
    "FoldSafeScaler", "build_sequence_examples", "make_neural_factory",
    "SAFE_SEQUENCE_FEATURES", "set_deterministic", "require_torch",
]
