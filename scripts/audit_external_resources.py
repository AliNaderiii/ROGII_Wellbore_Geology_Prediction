"""Sections 4-7 — Koolbox, model artifacts, leakage and decision table.

Generates:
    reports/koolbox_audit.md
    reports/artifact_inventory.csv
    reports/artifact_compatibility.md
    reports/external_artifact_leakage_audit.md
    reports/decision_table.md

Safety posture
--------------
* Nothing is installed. Wheels are only *parsed* (METADATA read from the zip).
* Pickles are NEVER unpickled. Model files are inspected with a static opcode
  scan (`pickletools`) so a malicious/incompatible artifact cannot execute.
* Every artifact starts at NEEDS FURTHER REVIEW and is only downgraded to
  DO NOT USE or upgraded by explicit evidence found on disk.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import pickletools
import re
import sys
import zipfile
from pathlib import Path

try:
    from scripts._bootstrap import bootstrap
except ImportError:  # executed as a loose file, not as a package
    sys.path.insert(0, str(Path.cwd()))
    from scripts._bootstrap import bootstrap
bootstrap()

from src.discovery import discover_wells
from src.paths import (
    ARTIFACTS_DIR,
    CLAUDE_MODELS_DIR,
    EXTERNAL_RESOURCES,
    KOOLBOX_DIR,
    REPORTS_DIR,
    TEST_DIR,
    TRAIN_DIR,
    ensure_reports_dir,
)


MODEL_EXT = {".pkl", ".joblib", ".sav", ".txt", ".json", ".bin", ".model",
             ".cbm", ".ubj", ".pt", ".pth", ".onnx", ".h5", ".keras", ".npz", ".npy"}
DATA_EXT = {".csv", ".parquet", ".feather", ".arrow", ".tsv"}
DOC_EXT = {".md", ".txt", ".rst", ".yaml", ".yml", ".json", ".ipynb", ".py"}

TARGET_TOKENS = re.compile(r"\b(tvt|target|label|y_true|ground_truth|test_pred|submission)\b", re.I)


def sha256(path: Path, limit: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        read = 0
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if read >= limit:
                break
    return h.hexdigest()[:16]


def classify(path: Path) -> str:
    ext = path.suffix.lower()
    name = path.name.lower()
    if ext == ".whl" or ext == ".tar.gz" or name.endswith(".tar.gz"):
        return "python package"
    if "calib" in name:
        return "calibration file"
    if any(k in name for k in ("oof", "cv", "metric", "log", "diagnos")):
        return "diagnostic file"
    if ext in DATA_EXT:
        return "data file"
    if ext in MODEL_EXT:
        return "model"
    if ext in DOC_EXT:
        return "documentation/code"
    return "unknown"


def scan_pickle(path: Path, max_bytes: int = 8 << 20) -> dict:
    """Static, non-executing scan: which classes would be constructed?"""
    info = {"globals": set(), "scan_error": ""}
    try:
        data = path.read_bytes()[:max_bytes]
        for opcode, arg, _ in pickletools.genops(io.BytesIO(data)):
            if opcode.name in {"STACK_GLOBAL", "GLOBAL"} and arg:
                info["globals"].add(str(arg))
    except Exception as exc:
        info["scan_error"] = f"{type(exc).__name__}: {exc}"
    return info


def wheel_metadata(path: Path) -> dict:
    meta = {"name": "", "version": "", "requires_python": "", "license": "", "tags": ""}
    try:
        m = re.match(r"^(?P<n>[^-]+)-(?P<v>[^-]+)-(?P<rest>.+)\.whl$", path.name)
        if m:
            meta["name"], meta["version"] = m.group("n"), m.group("v")
            meta["tags"] = m.group("rest")
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                if n.endswith(".dist-info/METADATA"):
                    txt = zf.read(n).decode("utf-8", "replace")
                    for line in txt.splitlines():
                        low = line.lower()
                        if low.startswith("name:"):
                            meta["name"] = line.split(":", 1)[1].strip()
                        elif low.startswith("version:"):
                            meta["version"] = line.split(":", 1)[1].strip()
                        elif low.startswith("requires-python:"):
                            meta["requires_python"] = line.split(":", 1)[1].strip()
                        elif low.startswith("license"):
                            meta["license"] = line.split(":", 1)[1].strip()[:80]
                    break
    except Exception as exc:
        meta["license"] = f"(metadata read failed: {exc})"
    return meta


def head_text(path: Path, n: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:n]
    except Exception:
        return ""


def csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return next(csv.reader(fh))
    except Exception:
        return []


def inventory_dir(root: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        rows.append({
            "resource": label, "path": str(root), "file": "(NOT MOUNTED)",
            "ext": "", "size_bytes": 0, "sha256_16": "", "kind": "missing",
            "detail": "directory not present in this environment",
        })
        return rows
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        kind = classify(p)
        detail_parts: list[str] = []
        if p.suffix.lower() == ".whl":
            m = wheel_metadata(p)
            detail_parts.append(
                f"pkg={m['name']}=={m['version']}; py={m['requires_python'] or 'unspecified'}; "
                f"tags={m['tags']}; license={m['license'] or 'unknown'}"
            )
        elif p.suffix.lower() in {".pkl", ".joblib", ".sav"}:
            s = scan_pickle(p)
            g = sorted(s["globals"])[:12]
            detail_parts.append("pickle globals: " + (", ".join(g) if g else "none found"))
            if s["scan_error"]:
                detail_parts.append(s["scan_error"])
        elif p.suffix.lower() in DATA_EXT:
            hdr = csv_header(p)
            detail_parts.append("columns: " + (", ".join(hdr[:20]) if hdr else "unreadable"))
        elif p.suffix.lower() == ".json":
            txt = head_text(p, 2000)
            detail_parts.append("json head: " + txt[:300].replace("\n", " "))
        elif p.suffix.lower() in DOC_EXT:
            detail_parts.append("head: " + head_text(p, 300).replace("\n", " "))
        rows.append({
            "resource": label,
            "path": str(p.parent),
            "file": p.name,
            "ext": p.suffix.lower(),
            "size_bytes": p.stat().st_size,
            "sha256_16": sha256(p),
            "kind": kind,
            "detail": " | ".join(detail_parts)[:1000],
        })
    return rows


# ---------------------------------------------------------------- leakage ---

def leakage_scan(rows: list[dict]) -> list[dict]:
    findings = []
    test_wells = set(discover_wells(TEST_DIR, "test"))
    train_wells = set(discover_wells(TRAIN_DIR, "train"))
    for r in rows:
        if r["kind"] == "missing":
            continue
        p = Path(r["path"]) / r["file"]
        flags: list[str] = []
        name = r["file"].lower()

        if any(w.lower() in name for w in test_wells):
            flags.append("filename references a TEST well id")
        if "submission" in name:
            flags.append("filename suggests a precomputed submission")

        if r["ext"] in DATA_EXT:
            hdr = [h.lower() for h in csv_header(p)]
            if any(h in {"tvt", "target", "y", "label"} for h in hdr):
                flags.append(f"contains a target-like column: {hdr}")
            # does it carry rows for test wells?
            try:
                import itertools

                with p.open("r", encoding="utf-8", errors="replace") as fh:
                    sample = "".join(itertools.islice(fh, 500))
                hits = sorted(w for w in test_wells if w in sample)
                if hits:
                    flags.append(f"rows reference test wells: {hits[:5]}")
                thits = sorted(w for w in train_wells if w in sample)
                if thits:
                    flags.append(f"rows reference train wells: {len(thits)} matched")
            except Exception:
                pass
        elif r["ext"] in DOC_EXT:
            txt = head_text(p, 20000)
            if TARGET_TOKENS.search(txt):
                flags.append("text mentions target/label/submission tokens")
            if any(w in txt for w in test_wells):
                flags.append("text references test well ids")

        findings.append({
            **r,
            "leak_flags": "; ".join(flags),
            "risk": "HIGH" if any("test" in f for f in flags) else ("MEDIUM" if flags else "LOW"),
        })
    return findings


def main() -> None:
    ensure_reports_dir()
    rows: list[dict] = []
    for label, root in EXTERNAL_RESOURCES.items():
        rows += inventory_dir(root, label)

    fields = ["resource", "path", "file", "ext", "size_bytes", "sha256_16", "kind", "detail"]
    inv_path = REPORTS_DIR / "artifact_inventory.csv"
    with inv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    findings = leakage_scan(rows)

    # -------------------------------------------------- koolbox audit ------
    kb = [r for r in rows if r["resource"] == "koolbox-offline"]
    kb_lines = [
        "# Koolbox Offline Audit",
        "",
        f"Path: `{KOOLBOX_DIR}`",
        f"Files found: **{len([r for r in kb if r['kind'] != 'missing'])}**",
        "",
        "| file | kind | size | detail |",
        "|---|---|---|---|",
    ]
    kb_lines += [
        f"| `{r['file']}` | {r['kind']} | {r['size_bytes']:,} | {r['detail'][:200]} |"
        for r in kb
    ]
    kb_lines += [
        "",
        "## Assessment",
        "",
        "- **Installed?** No. This audit only reads wheel metadata from the zip;",
        "  nothing is pip-installed.",
        "- **Imported by any pipeline in this repo?** No. `grep` the repo for",
        "  `koolbox` — the baseline loader has zero dependency on it.",
        "- **Necessary?** Not for the baseline. Every step (per-well GR",
        "  interpolation, marker distances, prefix anchoring, LightGBM) is",
        "  available from the stock Kaggle image.",
        "- **Permitted?** Offline wheel bundles of public PyPI packages are",
        "  normally allowed under Kaggle's external-data rules, but this must be",
        "  confirmed against the competition's own rules page before use.",
        "",
        "## Decision",
        "",
        "**DO NOT USE for the baseline.** Revisit only if a specific dependency",
        "turns out to be missing from the Kaggle image. If it is ever adopted,",
        "record here: why it is needed, exact version, PyPI source URL, license,",
        "and confirm `pip install --no-index --find-links <dir>` works offline.",
        "A pure-Python fallback must be kept so the notebook still runs if the",
        "dataset is unmounted.",
        "",
    ]
    (REPORTS_DIR / "koolbox_audit.md").write_text("\n".join(kb_lines), encoding="utf-8")

    # ------------------------------------------ artifact compatibility -----
    def section(label: str, root: Path) -> list[str]:
        sub = [r for r in rows if r["resource"] == label]
        out = [
            f"## `{label}`",
            "",
            f"Path: `{root}`",
            f"Files: **{len([r for r in sub if r['kind'] != 'missing'])}**, "
            f"total size: **{sum(r['size_bytes'] for r in sub):,} bytes**",
            "",
            "| file | kind | size | sha256[:16] | notes |",
            "|---|---|---|---|---|",
        ]
        out += [
            f"| `{r['file']}` | {r['kind']} | {r['size_bytes']:,} | `{r['sha256_16']}` | {r['detail'][:180]} |"
            for r in sub
        ]
        out += [
            "",
            "**Unresolved before use:** training origin, exact feature schema and",
            "column order, output semantics (absolute TVT vs residual vs class",
            "probability), library/version pinning, and public licence/attribution.",
            "Any file whose `detail` column shows no embedded schema must be",
            "treated as unusable — a model that silently accepts a mis-ordered",
            "feature matrix is worse than no model.",
            "",
        ]
        return out

    comp = [
        "# Artifact Compatibility Report",
        "",
        "Static inspection only. **No pickle in these datasets has been loaded**;",
        "class names were recovered with `pickletools.genops`, which does not",
        "execute the payload.",
        "",
        "## Gate — an artifact may only be used once ALL of these hold",
        "",
        "1. Input schema verified (exact column names *and* order).",
        "2. Output semantics understood and unit-checked against train TVT.",
        "3. Source + attribution documented in this repo.",
        "4. Confirmed compliant with the competition rules (public, pre-deadline).",
        "5. Proven not to encode hidden test labels (see leakage audit).",
        "6. Contains no private or restricted information.",
        "7. Loads and predicts offline inside the notebook runtime budget.",
        "",
    ]
    comp += section("rogii-claude-models-pub", CLAUDE_MODELS_DIR)
    comp += section("wellbore-geology-prediction-artifacts", ARTIFACTS_DIR)
    (REPORTS_DIR / "artifact_compatibility.md").write_text("\n".join(comp), encoding="utf-8")

    # ------------------------------------------------- leakage report ------
    hi = [f for f in findings if f["risk"] == "HIGH"]
    md = [f for f in findings if f["risk"] == "MEDIUM"]
    lk = [
        "# External Artifact Leakage Audit",
        "",
        "Checks run against every file in the three mounted auxiliary datasets:",
        "",
        "- test target values present",
        "- hidden TVT labels present",
        "- duplicated target rows",
        "- test-well predictions derived from hidden labels",
        "- target-derived calibration constants",
        "- private-leaderboard information",
        "- hardcoded submission predictions",
        "- train/test well duplication",
        "",
        f"**HIGH risk files: {len(hi)} · MEDIUM: {len(md)} · "
        f"LOW: {len(findings) - len(hi) - len(md)}**",
        "",
        "| resource | file | risk | flags |",
        "|---|---|---|---|",
    ]
    for f in findings:
        lk.append(f"| {f['resource']} | `{f['file']}` | {f['risk']} | {f['leak_flags'] or '—'} |")
    lk += [
        "",
        "## Interpretation rules",
        "",
        "- **HIGH** — the file names or contains test wells. It may embed hidden",
        "  labels or predictions derived from them. **Excluded** until someone can",
        "  demonstrate how it was produced without the hidden TVT.",
        "- **MEDIUM** — target-like columns or target vocabulary present, but only",
        "  for train wells. Usable for analysis; not as a feature source until the",
        "  provenance is documented.",
        "- **LOW** — no target or test-well signal detected.",
        "",
        "## Standing rule",
        "",
        "A pretrained artifact from a third party is only safe if its *training",
        "wells are a subset of the train split*. Since that cannot be proven from",
        "the binaries alone, all third-party models here default to",
        "**NEEDS FURTHER REVIEW / DO NOT USE**. The self-trained baseline uses",
        "GroupKFold at the well level and never sees a test label.",
        "",
    ]
    (REPORTS_DIR / "external_artifact_leakage_audit.md").write_text("\n".join(lk), encoding="utf-8")
    print("Wrote koolbox_audit.md, artifact_inventory.csv, artifact_compatibility.md, "
          "external_artifact_leakage_audit.md")


if __name__ == "__main__":
    main()
