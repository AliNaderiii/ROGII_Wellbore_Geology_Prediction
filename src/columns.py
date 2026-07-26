"""Column-role resolution.

Column names are resolved from the data itself (case-insensitive) rather than
hardcoded, so the loader keeps working if the organisers rename or reorder
anything. The canonical formation order below is the geological stacking
order reported in the repository EDA (ANCC shallowest -> BUDA deepest); it is
used only for ordering/consistency checks, never to invent data.
"""
from __future__ import annotations

import re

FORMATION_ORDER = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

ROLE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "md": ("md", "measured_depth", "depth"),
    "x": ("x", "easting"),
    "y": ("y", "northing"),
    "z": ("z", "tvd", "tvdss", "elevation"),
    "gr": ("gr", "gamma", "gamma_ray"),
    "tvt_input": ("tvt_input", "tvt_in", "input_tvt"),
    "tvt": ("tvt", "tvt_target", "target"),
    "geology": ("geology", "formation", "facies", "label"),
    "prediction_start": (
        "prediction_start",
        "predictionstart",
        "pred_start",
        "start_prediction",
        "is_prediction",
    ),
    "id": ("id", "row_id", "sample_id"),
}


def _norm(name) -> str:
    """Lowercase and collapse spaces/dashes/dots to underscores."""
    return re.sub(r"[\s\-.]+", "_", str(name).strip().lower())


def resolve_roles(columns) -> dict[str, str]:
    """Map canonical role -> actual column name present in `columns`."""
    lookup = {_norm(c): str(c) for c in columns}
    roles: dict[str, str] = {}
    for role, candidates in ROLE_CANDIDATES.items():
        for cand in candidates:
            if cand in lookup:
                roles[role] = lookup[cand]
                break
    # 'tvt' must not collide with 'tvt_input'
    if roles.get("tvt") and roles.get("tvt") == roles.get("tvt_input"):
        roles.pop("tvt")
    return roles


def marker_columns(columns) -> dict[str, str]:
    """Formation-top marker columns present in `columns`, in geological order."""
    lookup = {_norm(c).upper(): str(c) for c in columns}
    out: dict[str, str] = {}
    for name in FORMATION_ORDER:
        for key, actual in lookup.items():
            if key == name or key.startswith(name + "_") or key.endswith("_" + name):
                out[name] = actual
                break
    return out
