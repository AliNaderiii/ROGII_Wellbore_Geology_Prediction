"""Fold-safe offset-well (spatial) features.

Neighbour construction — stated exactly, because this is the part that leaks
if it is done casually:

1. A ``SpatialPrior`` is built from an explicit list of **donor wells**. The
   harness passes fold-train wells only, so a validation well never donates
   to its own features, and the well being predicted is additionally excluded
   by ID at query time (belt and braces).

2. Each donor contributes samples ``(X, Y, TVT)`` taken from rows where TVT is
   known. Which rows count as "known" depends on the caller:
     * ``source="label"``  — the full TVT curve of a train well (permitted:
       these are other wells' labels, and they are available at inference for
       every train well);
     * ``source="prefix"`` — only the visible ``TVT_input`` prefix (the
       strictly conservative option; also the only one available if a donor
       were ever a test well).
   Donors are decimated to every ``stride``-th row (default 25 ft) to keep the
   index small; decimation is deterministic.

3. Samples are indexed in the X/Y plane with a KD-tree (scipy) or an exact
   brute-force fallback. For a query row we take the ``k`` nearest donor
   samples within ``radius`` feet, **excluding every sample from the queried
   well**, and reduce them with inverse-distance weights
   ``w = 1 / (dist + eps)``.

4. Outputs: ``nbr_n``, ``nbr_dist_min``, ``nbr_tvt_wmean``, ``nbr_tvt_std``,
   ``nbr_shift`` (= wmean - anchor) and ``nbr_grad_along`` (the component of a
   locally fitted TVT plane along the trajectory heading). Where no neighbour
   is found the features are NaN/0 and ``nbr_n`` is 0, which lets a model learn
   to ignore them rather than silently trust a fabricated value.

The prior is rebuilt inside every fold. ``SpatialPrior.donor_ids`` is checked
against the fold's validation IDs by ``src.validation`` and the run aborts if
they intersect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.tasks import InferenceTask, WellTask

try:
    from scipy.spatial import cKDTree

    HAVE_KDTREE = True
except Exception:  # pragma: no cover
    cKDTree = None
    HAVE_KDTREE = False

SPATIAL_COLUMNS = [
    "nbr_n",
    "nbr_dist_min",
    "nbr_tvt_wmean",
    "nbr_tvt_std",
    "nbr_shift",
    "nbr_grad_along",
]


class SpatialLeakage(RuntimeError):
    """Raised when a well would contribute to its own spatial features."""


@dataclass
class SpatialConfig:
    k: int = 12
    radius: float = 6000.0
    stride: int = 25
    source: str = "label"  # "label" (train wells) or "prefix" (conservative)
    eps: float = 50.0
    query_stride: int = 25


class SpatialPrior:
    """Inverse-distance TVT prior built from an explicit donor set."""

    def __init__(self, config: SpatialConfig | None = None):
        self.config = config or SpatialConfig()
        self.donor_ids: set[str] = set()
        self._xy = np.empty((0, 2))
        self._tvt = np.empty(0)
        self._well = np.empty(0, dtype=object)
        self._tree = None

    # -------------------------------------------------------------- build --
    def fit(self, tasks: list[WellTask]) -> "SpatialPrior":
        cfg = self.config
        xs, ys, ts, ws = [], [], [], []
        for t in tasks:
            inp = t.inputs()
            if cfg.source == "prefix":
                tvt = inp.tvt_known.copy()
            else:
                tvt = inp.tvt_known.copy()
                if t.target is not None:
                    tvt[inp.start : inp.stop] = t.target
            m = np.isfinite(tvt) & np.isfinite(inp.x) & np.isfinite(inp.y)
            idx = np.flatnonzero(m)[:: max(1, cfg.stride)]
            if idx.size == 0:
                continue
            xs.append(inp.x[idx])
            ys.append(inp.y[idx])
            ts.append(tvt[idx])
            ws.append(np.full(idx.size, inp.well_id, dtype=object))
            self.donor_ids.add(inp.well_id)
        if not xs:
            return self
        self._xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self._tvt = np.concatenate(ts)
        self._well = np.concatenate(ws)
        if HAVE_KDTREE and len(self._xy):
            self._tree = cKDTree(self._xy)
        return self

    @property
    def n_samples(self) -> int:
        return int(self._tvt.size)

    def assert_disjoint(self, well_ids) -> None:
        """Fold-level guard: no *validation* well may be a donor.

        Called once per fold by the harness with the fold's validation IDs.

        This is deliberately NOT called from ``features_for``. When features are
        built for a fold-*train* well, that well is legitimately in the prior —
        it is one of the donors — and the correct protection is the query-time
        self-exclusion in ``_neighbours``, which drops the well's own samples
        from its own neighbour list. That keeps the feature definition
        identical (leave-one-well-out) for training and validation rows, which
        is what stops the model learning to rely on a self-match that will not
        exist at inference.
        """
        overlap = self.donor_ids & set(well_ids)
        if overlap:
            raise SpatialLeakage(
                "spatial prior was built from wells that are being predicted: "
                + ", ".join(sorted(overlap)[:10])
            )

    # -------------------------------------------------------------- query --
    def _neighbours(self, xq: np.ndarray, yq: np.ndarray, exclude: str):
        cfg = self.config
        n = xq.size
        out = {
            "nbr_n": np.zeros(n),
            "nbr_dist_min": np.full(n, np.nan),
            "nbr_tvt_wmean": np.full(n, np.nan),
            "nbr_tvt_std": np.full(n, np.nan),
            "nbr_gx": np.zeros(n),
            "nbr_gy": np.zeros(n),
        }
        if self.n_samples == 0:
            return out
        pts = np.column_stack([np.nan_to_num(xq), np.nan_to_num(yq)])

        # Over-fetch, then drop the queried well's own samples. When the query
        # well is itself a donor (every fold-train row), its own samples are by
        # construction the closest ones — they are on the same trajectory — so
        # a fixed over-fetch can return nothing but self-matches and silently
        # produce empty features. Fetch enough to survive that: the well's own
        # sample count is a hard upper bound on how many must be discarded.
        n_own = int(np.count_nonzero(self._well == exclude))
        k = int(min(cfg.k + n_own + cfg.k * 2, self.n_samples))

        if self._tree is not None:
            dist, idx = self._tree.query(pts, k=k, distance_upper_bound=cfg.radius)
            if k == 1:
                dist, idx = dist[:, None], idx[:, None]
        else:  # exact fallback
            d = np.sqrt(
                ((pts[:, None, 0] - self._xy[None, :, 0]) ** 2)
                + ((pts[:, None, 1] - self._xy[None, :, 1]) ** 2)
            )
            idx = np.argsort(d, axis=1)[:, :k]
            dist = np.take_along_axis(d, idx, axis=1)
            dist = np.where(dist > cfg.radius, np.inf, dist)

        for i in range(n):
            di, ii = dist[i], idx[i]
            ok = np.isfinite(di) & (ii < self.n_samples)
            if not ok.any():
                continue
            di, ii = di[ok], ii[ok]
            own = self._well[ii] == exclude
            if own.any():
                di, ii = di[~own], ii[~own]
            if di.size == 0:
                continue
            di, ii = di[: cfg.k], ii[: cfg.k]
            tv = self._tvt[ii]
            w = 1.0 / (di + cfg.eps)
            wsum = float(w.sum())
            mean = float((w * tv).sum() / wsum)
            out["nbr_n"][i] = di.size
            out["nbr_dist_min"][i] = float(di.min())
            out["nbr_tvt_wmean"][i] = mean
            out["nbr_tvt_std"][i] = float(np.sqrt((w * (tv - mean) ** 2).sum() / wsum))
            if di.size >= 3:
                A = np.column_stack(
                    [
                        self._xy[ii, 0] - pts[i, 0],
                        self._xy[ii, 1] - pts[i, 1],
                        np.ones(di.size),
                    ]
                )
                try:
                    coef, *_ = np.linalg.lstsq(A * w[:, None], tv * w, rcond=None)
                    out["nbr_gx"][i], out["nbr_gy"][i] = float(coef[0]), float(coef[1])
                except np.linalg.LinAlgError:  # pragma: no cover
                    pass
        return out

    def features_for(self, task: InferenceTask) -> pd.DataFrame:
        """Spatial features for the task's prediction region.

        The queried well's own samples are excluded inside ``_neighbours``, so
        this is leave-one-well-out whether the well is a donor (fold-train) or
        not (fold-validation). See ``assert_disjoint`` for why the fold-level
        guard is not repeated here.
        """
        cfg = self.config
        sl = slice(task.start, task.stop)
        xq, yq = task.x[sl], task.y[sl]
        n = xq.size
        if n == 0:
            return pd.DataFrame(columns=SPATIAL_COLUMNS)

        step = max(1, cfg.query_stride)
        probe = np.arange(0, n, step)
        if probe[-1] != n - 1:
            probe = np.append(probe, n - 1)
        raw = self._neighbours(xq[probe], yq[probe], exclude=task.well_id)

        rows = np.arange(n, dtype="float64")
        dense = {
            key: np.interp(rows, probe.astype("float64"), val)
            for key, val in raw.items()
        }

        anchor = task.anchor_tvt
        anchor = anchor if np.isfinite(anchor) else 0.0
        # heading over the prediction region, for the along-hole dip component
        hx = np.gradient(np.nan_to_num(xq)) if n > 1 else np.zeros(n)
        hy = np.gradient(np.nan_to_num(yq)) if n > 1 else np.zeros(n)
        norm = np.hypot(hx, hy)
        norm = np.where(norm > 1e-12, norm, 1.0)

        return pd.DataFrame(
            {
                "nbr_n": dense["nbr_n"],
                "nbr_dist_min": dense["nbr_dist_min"],
                "nbr_tvt_wmean": dense["nbr_tvt_wmean"],
                "nbr_tvt_std": dense["nbr_tvt_std"],
                "nbr_shift": dense["nbr_tvt_wmean"] - anchor,
                "nbr_grad_along": dense["nbr_gx"] * (hx / norm)
                + dense["nbr_gy"] * (hy / norm),
            }
        )

    def describe(self) -> dict:
        return {
            "n_donor_wells": len(self.donor_ids),
            "n_samples": self.n_samples,
            "k": self.config.k,
            "radius_ft": self.config.radius,
            "donor_stride_rows": self.config.stride,
            "query_stride_rows": self.config.query_stride,
            "source": self.config.source,
            "index": "scipy.cKDTree" if self._tree is not None else "brute-force",
            "weighting": "inverse distance 1/(d + eps)",
            "eps_ft": self.config.eps,
            "self_exclusion": "by well_id at query time + fold-train donors only",
        }
