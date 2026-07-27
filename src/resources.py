"""Small, dependency-light runtime/resource detection utilities."""
from __future__ import annotations
import os, platform, subprocess
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class ResourceProfile:
    requested: str
    selected: str
    gpu_available: bool
    gpu_name: str
    gpu_memory_mb: float | None
    cpu_count: int
    ram_mb: float | None
    model_execution_mode: str
    gpu_fallback_reason: str = ""

def _nvidia():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=2).strip().splitlines()
        if out:
            name, mem = [x.strip() for x in out[0].split(",", 1)]
            return True, name, float(mem)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return False, "", None

def detect_resources(requested: str = "auto") -> ResourceProfile:
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be auto, cpu, or gpu")
    avail, name, mem = _nvidia()
    selected = "gpu" if requested == "gpu" and avail else ("gpu" if requested == "auto" and avail else "cpu")
    reason = "" if selected == "gpu" else ("GPU requested but nvidia-smi/GPU was unavailable" if requested == "gpu" else "GPU unavailable or CPU explicitly selected")
    ram = None
    try:
        import psutil
        ram = psutil.virtual_memory().total / (1024**2)
    except Exception:
        pass
    return ResourceProfile(requested, selected, avail, name, mem, os.cpu_count() or 1, ram, "gpu" if selected == "gpu" else "cpu", reason)

def as_dict(profile: ResourceProfile) -> dict:
    d = asdict(profile)
    d["platform"] = platform.platform()
    return d
