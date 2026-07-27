"""Leakage-safe validation protocols, metrics and guards.

Protocols
---------
Both protocols are **cross-fitted by well ID**. No model is ever fitted on a
well it is later scored on. See ``PROTOCOL_A`` / ``PROTOCOL_B`` below and
``reports/validation_protocol.md`` §1 for why in-sample evaluation is invalid.

``same_well_masked`` (Protocol A)
    The prediction boundary is simulated *earlier*, inside the visible prefix,
    so the truth is taken from ``TVT_input`` and the ``TVT`` label column is
    never read. Measures continuation skill. Because it needs no labels, it is
    the only protocol that could also be run on a public test well.

``unseen_well`` (Protocol B)
    The real competition boundary. Models are fitted on fold-train wells and
    scored on the **real hidden suffix** of fold-validation wells, against the
    ``TVT`` label. Measures generalisation to a well never seen in training.
    This is the protocol that resembles the leaderboard.

``INVALID_in_sample`` (diagnostic only)
    Deliberately fits and scores on the *same* wells. It exists solely to
    quantify how large the in-sample illusion is, and every reporting path
    prefixes it with ``INVALID_`` and excludes it from model selection. It is
    never a validation result.

Guards
------
``BLOCKED_WELL_IDS`` — the three visible public test wells. Every entry point
that could touch validation or hyperparameter selection calls
``assert_no_blocked_wells``, which raises ``BlockedWellError``. The guard is
applied to: the well universe, each fold's train and validation lists, model
fitting, spatial donor sets, and the metrics tables written to disk. There is
no flag to disable it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.baselines import BaselineModel
from src.features import build_features
from src.spatial import SpatialConfig, SpatialPrior
from src.tasks import TaskConstructionError, WellTask, make_task, task_descriptor

#: The visible public test wells. Never used for tuning, fitting or selection.
BLOCKED_WELL_IDS: frozenset[str] = frozenset({"000d7d20", "00bbac68", "00e12e8b"})

#: Canonical protocol names. Reported separately and never averaged together.
PROTOCOL_A = "same_well_masked"
PROTOCOL_B = "unseen_well"
PROTOCOL_INVALID = "INVALID_in_sample"

VALID_PROTOCOLS = (PROTOCOL_A, PROTOCOL_B)

PROTOCOL_LABELS = {
    PROTOCOL_A: "A — same-well masked suffix (truth from TVT_input, cross-fitted by well)",
    PROTOCOL_B: "B — unseen-well GroupKFold (real hidden suffix, truth from TVT label)",
    PROTOCOL_INVALID: "INVALID — in-sample diagnostic, NOT a validation result",
}


class BlockedWellError(RuntimeError):
    """Raised when a public test well reaches validation or tuning."""


def assert_no_blocked_wells(well_ids, *, context: str) -> None:
    """Hard guard. Raises if any blocked well ID appears in ``well_ids``."""
    ids = {str(w) for w in well_ids}
    hit = sorted(ids & BLOCKED_WELL_IDS)
    if hit:
        raise BlockedWellError(
            f"{context}: public test wells must never enter validation or "
            f"hyperparameter selection. Offending IDs: {hit}. "
            "These three wells are reserved for submission-pipeline checks only."
        )


def filter_blocked(well_ids) -> list[str]:
    """Drop blocked IDs from a candidate universe (and say nothing else)."""
    return [str(w) for w in well_ids if str(w) not in BLOCKED_WELL_IDS]


# ------------------------------------------------------------------ metrics --

def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    p = np.asarray(pred, dtype="float64")
    t = np.asarray(truth, dtype="float64")
    m = np.isfinite(p) & np.isfinite(t)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))


def _se(pred: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    p = np.asarray(pred, dtype="float64")
    t = np.asarray(truth, dtype="float64")
    m = np.isfinite(p) & np.isfinite(t)
    if not m.any():
        return 0.0, 0
    d = p[m] - t[m]
    return float(np.sum(d * d)), int(m.sum())


@dataclass
class WellResult:
    model: str
    protocol: str
    fold: int
    well_id: str
    n_points: int
    sse: float
    rmse: float
    max_abs_error: float
    bias: float
    prefix_len: int
    suffix_len: int
    gr_missing_frac: float
    anchor_tvt: float
    has_typewell: bool
    predict_seconds: float


def score_well(
    model_name: str,
    protocol: str,
    fold: int,
    task: WellTask,
    pred: np.ndarray,
    seconds: float,
) -> WellResult:
    truth = task.scored()
    sse, n = _se(pred, truth)
    d = np.asarray(pred, dtype="float64") - np.asarray(truth, dtype="float64")
    d = d[np.isfinite(d)]
    desc = task_descriptor(task)
    return WellResult(
        model=model_name,
        protocol=protocol,
        fold=fold,
        well_id=task.well_id,
        n_points=n,
        sse=sse,
        rmse=float(np.sqrt(sse / n)) if n else float("nan"),
        max_abs_error=float(np.max(np.abs(d))) if d.size else float("nan"),
        bias=float(np.mean(d)) if d.size else float("nan"),
        prefix_len=int(desc["prefix_len"]),
        suffix_len=int(desc["suffix_len"]),
        gr_missing_frac=float(desc["gr_missing_frac"]),
        anchor_tvt=float(desc["anchor_tvt"]),
        has_typewell=bool(desc["has_typewell"]),
        predict_seconds=float(seconds),
    )


# --------------------------------------------------------------- summaries --

SUFFIX_BINS = [0, 500, 1000, 2000, 4000, np.inf]
SUFFIX_LABELS = ["<500", "500-1k", "1k-2k", "2k-4k", ">4k"]
GR_BINS = [-0.001, 0.05, 0.2, 0.5, 0.8, 1.01]
GR_LABELS = ["<5%", "5-20%", "20-50%", "50-80%", ">80%"]
PREFIX_BINS = [0, 1000, 2000, 4000, 8000, np.inf]
PREFIX_LABELS = ["<1k", "1k-2k", "2k-4k", "4k-8k", ">8k"]


def summarize(well_df: pd.DataFrame) -> pd.DataFrame:
    """Global / mean / median / worst-10 metrics per (model, protocol)."""
    rows = []
    for (model, protocol), g in well_df.groupby(["model", "protocol"], sort=False):
        n = float(g["n_points"].sum())
        worst = g.nlargest(min(10, len(g)), "rmse")
        rows.append(
            {
                "model": model,
                "protocol": protocol,
                "n_wells": int(len(g)),
                "n_points": int(n),
                "global_rmse": float(np.sqrt(g["sse"].sum() / n)) if n else np.nan,
                "mean_well_rmse": float(g["rmse"].mean()),
                "median_well_rmse": float(g["rmse"].median()),
                "p90_well_rmse": float(g["rmse"].quantile(0.90)),
                "worst10_well_rmse": float(worst["rmse"].mean()),
                "worst_well_rmse": float(g["rmse"].max()),
                "worst_well_id": str(g.loc[g["rmse"].idxmax(), "well_id"]),
                "max_abs_error": float(g["max_abs_error"].max()),
                "mean_bias": float(g["bias"].mean()),
                "predict_seconds": float(g["predict_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _stratum(well_df: pd.DataFrame, col: str, bins, labels, name: str) -> pd.DataFrame:
    df = well_df.copy()
    df["stratum"] = pd.cut(df[col], bins=bins, labels=labels)
    out = []
    for (model, protocol, stratum), g in df.groupby(
        ["model", "protocol", "stratum"], observed=True, sort=False
    ):
        n = float(g["n_points"].sum())
        if n == 0:
            continue
        out.append(
            {
                "model": model,
                "protocol": protocol,
                "stratify_by": name,
                "stratum": str(stratum),
                "n_wells": int(len(g)),
                "n_points": int(n),
                "global_rmse": float(np.sqrt(g["sse"].sum() / n)),
                "median_well_rmse": float(g["rmse"].median()),
                "worst_well_rmse": float(g["rmse"].max()),
            }
        )
    return pd.DataFrame(out)


def stratified_report(well_df: pd.DataFrame) -> pd.DataFrame:
    parts = [
        _stratum(well_df, "suffix_len", SUFFIX_BINS, SUFFIX_LABELS, "hidden_suffix_length"),
        _stratum(well_df, "gr_missing_frac", GR_BINS, GR_LABELS, "gr_missingness"),
        _stratum(well_df, "prefix_len", PREFIX_BINS, PREFIX_LABELS, "prefix_length"),
    ]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----------------------------------------------------------------- folds ----

@dataclass
class Fold:
    index: int
    train_ids: list[str]
    valid_ids: list[str]

    def __post_init__(self) -> None:
        assert_no_blocked_wells(self.train_ids, context=f"fold {self.index} train")
        assert_no_blocked_wells(self.valid_ids, context=f"fold {self.index} valid")
        overlap = set(self.train_ids) & set(self.valid_ids)
        if overlap:
            raise RuntimeError(
                f"fold {self.index}: {len(overlap)} wells in both train and valid"
            )


def make_group_folds(well_ids, n_splits: int = 5, *, seed: int = 0) -> list[Fold]:
    """GroupKFold over well IDs. One well never spans two folds."""
    ids = filter_blocked(sorted({str(w) for w in well_ids}))
    assert_no_blocked_wells(ids, context="fold universe")
    if len(ids) < n_splits:
        raise ValueError(f"need >= {n_splits} wells, got {len(ids)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    folds = []
    for k in range(n_splits):
        valid = [ids[i] for i in order[k::n_splits]]
        train = [w for w in ids if w not in set(valid)]
        folds.append(Fold(index=k, train_ids=train, valid_ids=valid))
    return folds


# -------------------------------------------------------------- the harness --

@dataclass
class ValidationConfig:
    protocols: tuple[str, ...] = ("masked", "groupkfold")
    n_splits: int = 5
    max_wells: int | None = None
    seed: int = 0
    spatial: bool = False
    spatial_config: SpatialConfig = field(default_factory=SpatialConfig)
    models: tuple[str, ...] = ()
    verbose: bool = True


class TaskCache:
    """Loads each well once, keeps both task modes, one well resident at a time."""

    def __init__(self, loader):
        self.loader = loader

    def get(self, well_id: str, mode: str) -> WellTask | None:
        well = self.loader(well_id)
        if well is None:
            return None
        try:
            return make_task(well, mode)
        except TaskConstructionError:
            return None


def evaluate_models(
    models: dict[str, BaselineModel],
    tasks: list[WellTask],
    protocol: str,
    fold: int,
    *,
    verbose: bool = False,
    failures: list | None = None,
) -> list[WellResult]:
    """Score every model on every task, sharing feature computation."""
    results: list[WellResult] = []
    need_align = any(getattr(m, "needs_alignment", False) for m in models.values())
    for task in tasks:
        inp = task.inputs()
        inp.assert_no_target()
        feats = build_features(inp, alignment=need_align)
        for name, model in models.items():
            t0 = time.perf_counter()
            try:
                pred = np.asarray(model.predict(inp, feats), dtype="float64")
            except Exception as exc:  # a model failing must not kill the run
                if verbose:
                    print(f"    [{name}] {task.well_id}: {type(exc).__name__}: {exc}")
                if failures is not None:
                    failures.append(
                        {
                            "stage": "predict",
                            "model": name,
                            "well_id": task.well_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue
            dt = time.perf_counter() - t0
            if pred.size != inp.n_predict:
                raise RuntimeError(
                    f"{name}/{task.well_id}: predicted {pred.size} rows, "
                    f"expected {inp.n_predict}"
                )
            results.append(score_well(name, protocol, fold, task, pred, dt))
    return results


def fit_models(
    factories: dict[str, callable],
    train_tasks: list[WellTask],
    *,
    spatial: SpatialPrior | None = None,
    verbose: bool = False,
    failures: list | None = None,
) -> dict[str, BaselineModel]:
    """Fit one instance of each model on fold-train tasks only."""
    assert_no_blocked_wells([t.well_id for t in train_tasks], context="model fitting")
    models: dict[str, BaselineModel] = {}
    for name, factory in factories.items():
        try:
            model = factory(spatial=spatial) if spatial is not None else factory()
        except TypeError:
            model = factory()
        t0 = time.perf_counter()
        try:
            model.fit(train_tasks)
        except Exception as exc:
            if verbose:
                print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
            if failures is not None:
                failures.append(
                    {
                        "stage": "fit",
                        "model": name,
                        "well_id": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        if verbose:
            print(f"  fitted {name} in {time.perf_counter() - t0:.1f}s")
        models[name] = model
    return models


# ------------------------------------------------ the cross-fitted driver ---

class CrossFitLeakage(RuntimeError):
    """Raised when a scored well was also used to fit the model scoring it."""


@dataclass
class ProtocolRun:
    """Results plus the bookkeeping needed to prove the run was clean."""

    protocol: str
    well_results: list = field(default_factory=list)
    fold_records: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    spatial_notes: list = field(default_factory=list)

    @property
    def n_failures(self) -> int:
        return len(self.failures)


def run_cross_fitted_protocol(
    *,
    protocol: str,
    mode: str,
    factories: dict[str, callable],
    folds: list[Fold],
    task_builder,
    spatial_config: SpatialConfig | None = None,
    spatial_models: tuple[str, ...] = ("ridge", "lightgbm"),
    verbose: bool = False,
) -> ProtocolRun:
    """Fit on fold-train wells, score on fold-validation wells. Never both.

    This is the single code path used by *both* validation protocols; they
    differ only in ``mode`` (which boundary the task uses and therefore where
    the truth comes from), never in how fitting relates to scoring. Routing
    both through one driver is deliberate: it makes an in-sample evaluation
    impossible to reintroduce in one protocol but not the other.

    ``task_builder(well_ids, mode)`` must return ``(tasks, skipped)``.
    """
    run = ProtocolRun(protocol=protocol)

    for fold in folds:
        t0 = time.perf_counter()
        assert_no_blocked_wells(fold.train_ids, context=f"{protocol} fold {fold.index} train")
        assert_no_blocked_wells(fold.valid_ids, context=f"{protocol} fold {fold.index} valid")

        train_tasks, sk_train = task_builder(fold.train_ids, mode)
        valid_tasks, sk_valid = task_builder(fold.valid_ids, mode)
        for wid, reason in sk_train + sk_valid:
            run.failures.append(
                {"stage": "task", "model": "", "well_id": wid, "error": reason}
            )
        if not train_tasks or not valid_tasks:
            continue

        train_ids = {t.well_id for t in train_tasks}
        valid_ids = {t.well_id for t in valid_tasks}
        overlap = train_ids & valid_ids
        if overlap:
            raise CrossFitLeakage(
                f"{protocol} fold {fold.index}: {len(overlap)} well(s) are both "
                f"fitted and scored, e.g. {sorted(overlap)[:5]}. In-sample "
                "evaluation is invalid — see reports/validation_protocol.md §1."
            )

        models = fit_models(factories, train_tasks, verbose=verbose, failures=run.failures)
        run.well_results += evaluate_models(
            models, valid_tasks, protocol, fold.index,
            verbose=verbose, failures=run.failures,
        )

        # -- optional spatial variant, donors from fold-train wells only ----
        if spatial_config is not None:
            prior = SpatialPrior(spatial_config).fit(train_tasks)
            # leave-one-validation-well-out: no validation well may donate
            prior.assert_disjoint(valid_ids)
            sp_factories = {n: f for n, f in factories.items() if n in spatial_models}
            if sp_factories:
                sp_models = fit_models(
                    sp_factories, train_tasks, spatial=prior,
                    verbose=verbose, failures=run.failures,
                )
                sp_models = {f"{n}_spatial": m for n, m in sp_models.items()}
                run.well_results += evaluate_models(
                    sp_models, valid_tasks, protocol, fold.index,
                    verbose=verbose, failures=run.failures,
                )
                run.spatial_notes.append(
                    {"fold": fold.index, **prior.describe(),
                     "n_validation_wells_excluded": len(valid_ids)}
                )

        run.fold_records.append(
            {
                "protocol": protocol,
                "fold": fold.index,
                "n_train_wells": len(train_tasks),
                "n_valid_wells": len(valid_tasks),
                "seconds": time.perf_counter() - t0,
            }
        )
        if verbose:
            print(f"      fold {fold.index}: {len(train_tasks)} train / "
                  f"{len(valid_tasks)} valid in {time.perf_counter() - t0:.1f}s")
    return run


def run_in_sample_diagnostic(
    *,
    factories: dict[str, callable],
    tasks: list[WellTask],
    verbose: bool = False,
) -> ProtocolRun:
    """Fit and score on the SAME wells. Deliberately invalid.

    Reported only to quantify the size of the in-sample illusion — the gap
    between this and Protocol B is the memorisation a naive harness would have
    reported as skill. Results carry the ``INVALID_in_sample`` protocol name,
    which every selection path filters out.
    """
    run = ProtocolRun(protocol=PROTOCOL_INVALID)
    if not tasks:
        return run
    assert_no_blocked_wells([t.well_id for t in tasks], context="in-sample diagnostic")
    t0 = time.perf_counter()
    models = fit_models(factories, tasks, verbose=verbose, failures=run.failures)
    run.well_results += evaluate_models(
        models, tasks, PROTOCOL_INVALID, fold=-1, verbose=verbose, failures=run.failures
    )
    run.fold_records.append(
        {
            "protocol": PROTOCOL_INVALID,
            "fold": -1,
            "n_train_wells": len(tasks),
            "n_valid_wells": len(tasks),
            "seconds": time.perf_counter() - t0,
        }
    )
    return run
