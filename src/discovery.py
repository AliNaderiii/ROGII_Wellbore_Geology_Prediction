"""Dynamic well discovery — no hardcoded well IDs anywhere.

File naming convention observed in the competition mount:

    <WELL_ID>__horizontal_well.csv
    <WELL_ID>__typewell.csv
    <WELL_ID>__*.png            (optional plots)

The discovery layer is deliberately tolerant: it also accepts a single
underscore separator and nested per-well directories, so a layout change on
Kaggle does not silently produce an empty well list.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HORIZONTAL_TOKENS = ("horizontal_well", "horizontalwell", "horizontal")
TYPEWELL_TOKENS = ("typewell", "type_well")

_SEP = re.compile(r"__|_(?=(?:horizontal|type))")


@dataclass
class WellFiles:
    well_id: str
    split: str
    horizontal: Path | None = None
    typewell: Path | None = None
    images: list[Path] = field(default_factory=list)
    others: list[Path] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.horizontal is not None and self.typewell is not None


def _classify(stem: str) -> tuple[str, str]:
    """Return (well_id, role) for a file stem."""
    low = stem.lower()
    for token in TYPEWELL_TOKENS:
        if low.endswith(token):
            return _SEP.split(stem)[0].rstrip("_"), "typewell"
    for token in HORIZONTAL_TOKENS:
        if low.endswith(token):
            return _SEP.split(stem)[0].rstrip("_"), "horizontal"
    return _SEP.split(stem)[0].rstrip("_"), "other"


def discover_wells(directory: Path, split: str) -> dict[str, WellFiles]:
    """Walk `directory` recursively and group files by well id."""
    wells: dict[str, WellFiles] = {}
    if not directory.exists():
        return wells
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        well_id, role = _classify(path.stem)
        if not well_id:
            continue
        well = wells.setdefault(well_id, WellFiles(well_id=well_id, split=split))
        suffix = path.suffix.lower()
        if suffix == ".csv" and role == "horizontal":
            well.horizontal = path
        elif suffix == ".csv" and role == "typewell":
            well.typewell = path
        elif suffix in {".png", ".jpg", ".jpeg"}:
            well.images.append(path)
        else:
            well.others.append(path)
    return wells


def well_id_prefix(well_id: str) -> str:
    """Leading alphabetic/numeric prefix, used for duplicate-prefix checks."""
    m = re.match(r"[A-Za-z0-9]+", well_id)
    return m.group(0) if m else well_id
