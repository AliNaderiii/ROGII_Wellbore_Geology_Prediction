"""Build a SYNTHETIC field that mimics the audited Kaggle layout.

Purpose: prove the validation harness runs end to end, that the guards fire,
and that the models rank sensibly — in an environment where the real
competition mount is not available.

**This is not the competition data and its metrics are not competition
metrics.** Reports produced from it are written to a directory that is clearly
marked synthetic, and `run_validation.py` on the real mount is what produces
the real numbers.

Structural properties reproduced from the audit findings:

* `<well>__horizontal_well.csv` + `<well>__typewell.csv` naming
* MD monotonic, exactly 1 ft steps, no duplicates
* clean `TVT_input` prefix / NaN suffix, no internal gaps
* train wells carry `TVT` plus the six train-only markers
* test wells carry neither the target nor the markers
* train typewells are `['TVT', 'GR', 'Geology']`; test typewells are
  `['TVT', 'GR']` — the Geology column is train-only, per the schema audit
* GR missingness in contiguous blocks, including wells above 50%
* wells laid out on a map with a dipping, folded structural surface, so
  offset-well (spatial) features carry genuine signal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def _structural_surface(x: np.ndarray, y: np.ndarray, seed_field: int = 7) -> np.ndarray:
    """A smooth regional surface: regional dip + folds. Shared by all wells."""
    rng = np.random.default_rng(seed_field)
    a = rng.uniform(-0.0015, 0.0015, 4)
    return (
        -0.0009 * x
        + 0.0004 * y
        + 18.0 * np.sin(x / 9000.0 + a[0])
        + 11.0 * np.cos(y / 7000.0 + a[1])
        + 5.0 * np.sin((x + y) / 4200.0 + a[2])
    )


def _landing_field(x: np.ndarray, y: np.ndarray, seed_field: int = 23) -> np.ndarray:
    """Spatially smooth part of TVT.

    Physically: operators land in the same zone across a pad, and the residual
    structural mis-estimate is correlated over kilometres. This is the component
    an offset-well prior can legitimately recover — it is why spatial features
    are worth testing at all.
    """
    rng = np.random.default_rng(seed_field)
    b = rng.uniform(-1.0, 1.0, 4)
    return (
        22.0 * np.sin(x / 15000.0 + b[0])
        + 16.0 * np.cos(y / 12000.0 + b[1])
        + 9.0 * np.sin((x - y) / 8000.0 + b[2])
        + 0.00035 * x
    )


def _typewell(
    rng: np.random.Generator, tvt_lo=-140.0, tvt_hi=140.0, *, with_geology: bool = True
) -> pd.DataFrame:
    """Reference GR log with distinctive marker beds, on a 0.25 ft TVT grid.

    ``with_geology`` reproduces the audited train/test asymmetry: train
    typewells are ``['TVT', 'GR', 'Geology']``, test typewells ``['TVT', 'GR']``.
    """
    tvt = np.arange(tvt_lo, tvt_hi, 0.25)
    gr = 70 + 18 * np.sin(tvt / 11.0) + 9 * np.sin(tvt / 3.1 + 1.2)
    for centre, amp, width in [(-95, 55, 4), (-40, -35, 6), (5, 60, 3),
                               (48, -30, 5), (96, 45, 4)]:
        gr += amp * np.exp(-0.5 * ((tvt - centre) / width) ** 2)
    gr += rng.normal(0, 1.5, tvt.size)
    frame = pd.DataFrame({"TVT": tvt, "GR": gr})
    if with_geology:
        edges = np.linspace(tvt_lo, tvt_hi, len(FORMATIONS) + 1)
        frame["Geology"] = [
            FORMATIONS[min(int(np.searchsorted(edges, t, "right")) - 1, 5)] for t in tvt
        ]
    return frame


def _well(
    well_id: str,
    rng: np.random.Generator,
    *,
    with_target: bool,
    n_rows: int | None = None,
    pad: tuple[float, float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = n_rows or int(rng.integers(3500, 9000))
    md0 = float(rng.uniform(8000, 12000))
    md = md0 + np.arange(n, dtype=float)  # exactly 1 ft, monotonic, unique

    # Wells are drilled from pads: several laterals a few hundred feet apart,
    # sub-parallel. This is what makes offset-well features meaningful, and a
    # uniformly scattered field would not test them at all.
    if pad is None:
        x0, y0 = rng.uniform(0, 60000), rng.uniform(0, 60000)
        heading = rng.uniform(0, 2 * np.pi)
    else:
        px, py, phead = pad
        x0 = px + rng.normal(0, 900)
        y0 = py + rng.normal(0, 900)
        heading = phead + rng.normal(0, 0.12)
    drift = np.cumsum(rng.normal(0, 0.0006, n))
    x = x0 + np.cumsum(np.cos(heading + drift))
    y = y0 + np.cumsum(np.sin(heading + drift))

    # TVT is built first, from three physically distinct components, then Z is
    # derived so that Z, TVT and the structural surface stay mutually
    # consistent (TVT = surface - Z, exactly as in the real task).
    #
    #  1. a spatially smooth landing/structure term  -> recoverable from offsets
    #  2. a slow per-well wander (steering + local structure the map misses)
    #  3. occasional faults: step changes of a few feet
    landing = _landing_field(x, y)
    wander = np.cumsum(rng.normal(0, 0.010, n))
    wander = wander - wander[0]
    wander += rng.uniform(-0.0025, 0.0025) * np.arange(n)  # per-well apparent dip

    faults = np.zeros(n)
    for _ in range(int(rng.integers(0, 3))):
        at = int(rng.integers(int(0.15 * n), n))
        faults[at:] += float(rng.normal(0, 6.0))

    tvt = landing + wander + faults + float(rng.uniform(-12, 12))
    tvt = np.clip(tvt, -130.0, 130.0)

    surface = _structural_surface(x, y)
    z = surface - tvt

    # Geology is a train-only typewell column (schema audit): train typewells
    # are ['TVT', 'GR', 'Geology'], test typewells ['TVT', 'GR'].
    tw = _typewell(rng, with_geology=with_target)
    gr = np.interp(tvt, tw["TVT"].to_numpy(), tw["GR"].to_numpy())
    gr = gr + rng.normal(0, 4.0, n) + rng.uniform(-6, 6)  # noise + calibration offset

    # contiguous GR outages
    for _ in range(int(rng.integers(0, 3))):
        s = int(rng.integers(0, max(1, n - 400)))
        gr[s : s + int(rng.integers(80, 700))] = np.nan
    if rng.random() < 0.19:  # the high-missingness cohort
        frac = rng.uniform(0.5, 0.8)
        s = int(rng.integers(0, max(1, int(n * (1 - frac)))))
        gr[s : s + int(n * frac)] = np.nan

    prefix = int(n * rng.uniform(0.45, 0.75))
    tvt_input = tvt.copy()
    tvt_input[prefix:] = np.nan

    hw = pd.DataFrame(
        {"MD": md, "X": x, "Y": y, "Z": z, "GR": gr, "TVT_input": tvt_input}
    )
    if with_target:
        hw["TVT"] = tvt
        base = float(rng.uniform(-120, -60))
        for i, f in enumerate(FORMATIONS):
            hw[f] = base + 40.0 * i + rng.normal(0, 0.2, n)
    return hw, tw


def build(root: Path, n_train: int = 60, n_test: int = 3, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    train, test = root / "train", root / "test"
    train.mkdir(parents=True, exist_ok=True)
    test.mkdir(parents=True, exist_ok=True)
    for p in list(train.glob("*.csv")) + list(test.glob("*.csv")):
        p.unlink()

    # lay out pads, then drill several wells from each
    n_pads = max(3, n_train // 6)
    pads = [
        (float(rng.uniform(0, 40000)), float(rng.uniform(0, 40000)),
         float(rng.uniform(0, 2 * np.pi)))
        for _ in range(n_pads)
    ]

    for i in range(n_train):
        wid = f"tr{i:05x}"
        hw, tw = _well(wid, rng, with_target=True, pad=pads[i % n_pads])
        hw.to_csv(train / f"{wid}__horizontal_well.csv", index=False)
        tw.to_csv(train / f"{wid}__typewell.csv", index=False)

    # The three real public test well IDs, so the blocked-ID guard is exercised
    # against the actual strings it must reject.
    test_ids = ["000d7d20", "00bbac68", "00e12e8b"][:n_test]
    rows = []
    for j, wid in enumerate(test_ids):
        hw, tw = _well(wid, rng, with_target=False, pad=pads[j % n_pads])
        hw.to_csv(test / f"{wid}__horizontal_well.csv", index=False)
        tw.to_csv(test / f"{wid}__typewell.csv", index=False)
        start = int(hw["TVT_input"].isna().to_numpy().argmax())
        rows += [{"id": f"{wid}_{r}", "tvt": 0.0} for r in range(start, len(hw))]
    pd.DataFrame(rows).to_csv(root / "sample_submission.csv", index=False)
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/rogii_synthetic/competition")
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    root = build(Path(a.root), a.n_train, a.n_test, a.seed)
    print(f"synthetic field written to {root}")
    print(f"  train wells: {len(list((root / 'train').glob('*__horizontal_well.csv')))}")
    print(f"  test wells : {len(list((root / 'test').glob('*__horizontal_well.csv')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
