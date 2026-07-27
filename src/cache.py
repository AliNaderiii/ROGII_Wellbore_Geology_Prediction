"""Deterministic, target-free on-disk cache primitives for validation."""
from __future__ import annotations
import hashlib, json, shutil, time
from pathlib import Path
from dataclasses import dataclass
import numpy as np

CACHE_VERSION = "feature-cache-v2"
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

def cache_key(*, dataset_version, well_id, fold_id, protocol, feature_config, alignment_config, device_profile, code_version="pipeline-v2") -> str:
    payload = {"cache_version": CACHE_VERSION, "dataset_version": dataset_version, "well_id": str(well_id), "fold_id": fold_id, "protocol": protocol, "feature_config": feature_config, "alignment_config": alignment_config, "device_profile": device_profile, "code_version": code_version}
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()

class FeatureCache:
    """NPZ cache with atomic writes; callers decide which target-free arrays to store."""
    def __init__(self, directory, stats=None):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.stats = stats or CacheStats()
    def path(self, key): return self.directory / f"{key}.npz"
    def get(self, key):
        p = self.path(key)
        if not p.exists(): self.stats.misses += 1; return None
        try:
            with np.load(p, allow_pickle=False) as z: out = {k: z[k] for k in z.files}
            self.stats.hits += 1; return out
        except Exception:
            self.stats.misses += 1
            try: p.unlink()
            except OSError: pass
            return None
    def put(self, key, **arrays):
        if any("target" in str(k).lower() or k.lower() in {"y", "label", "tvt"} for k in arrays):
            raise ValueError("target values are forbidden in feature cache artifacts")
        p, tmp = self.path(key), self.path(key).with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays); tmp.replace(p); self.stats.writes += 1
    def clear(self):
        for p in self.directory.glob("*.npz"): p.unlink()
    def size_bytes(self): return sum(p.stat().st_size for p in self.directory.glob("*.npz"))
    def report(self): return {"cache_dir": str(self.directory), "cache_hits": self.stats.hits, "cache_misses": self.stats.misses, "cache_writes": self.stats.writes, "cache_size_bytes": self.size_bytes()}
