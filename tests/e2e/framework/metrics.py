"""System metrics collector for E2E test runs.

Collects Docker container CPU/memory, disk usage, and host load. All collection
is best-effort: if Docker or the container isn't reachable, the metric is
reported as ``None`` rather than raising — a missing metric must never fail a
scenario. The container name is configurable via ``KOBOI_CONTAINER`` (default:
``koboi``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

CONTAINER = os.environ.get("KOBOI_CONTAINER", "koboi")
DOCKER_BIN = shutil.which("docker")
WORKSPACE_ROOT = os.environ.get("KOBOI_WORKSPACE_ROOT", "./workspace")


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    """Run a command, return stdout or None on any failure/timeout."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def _resolve_container() -> str | None:
    """Find the running container id/name matching the koboi service.

    Handles docker-compose naming (``<project>-koboi-1``) by inspecting the
    compose service label. Falls back to the bare ``CONTAINER`` name.
    """
    if not DOCKER_BIN:
        return None
    # Fast path: explicit container name works.
    probe = _run([DOCKER_BIN, "inspect", CONTAINER])
    if probe is not None:
        return CONTAINER
    # Compose path: list containers, find the koboi service.
    raw = _run([DOCKER_BIN, "ps", "--format", "{{json .}}"])
    if not raw:
        return None
    for line in raw.splitlines():
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = c.get("Names", "")
        if "koboi" in name.lower():
            return name.split(",")[0]
    return None


def collect_docker_stats(container: str | None = None) -> dict:
    """Return ``{cpu_percent, memory_mb, memory_percent}`` from ``docker stats``."""
    if not DOCKER_BIN:
        return {"cpu_percent": None, "memory_mb": None, "memory_percent": None, "available": False}
    target = container or _resolve_container()
    if not target:
        return {"cpu_percent": None, "memory_mb": None, "memory_percent": None, "available": False}
    raw = _run([DOCKER_BIN, "stats", "--no-stream", "--format", "{{json .}}", target], timeout=15)
    if not raw:
        return {"cpu_percent": None, "memory_mb": None, "memory_percent": None, "available": False}
    try:
        s = json.loads(raw)
    except json.JSONDecodeError:
        return {"cpu_percent": None, "memory_mb": None, "memory_percent": None, "available": False}
    return {
        "cpu_percent": _pct(s.get("CPUPerc")),
        "memory_mb": _mem_mb(s.get("MemUsage")),
        "memory_percent": _pct(s.get("MemPerc")),
        "available": True,
    }


def _pct(val: str | None) -> float | None:
    if not val:
        return None
    try:
        return round(float(val.strip().rstrip("%")), 2)
    except (ValueError, AttributeError):
        return None


def _mem_mb(mem_usage: str | None) -> float | None:
    """Parse ``"180.5MiB / 1GiB"`` → 180.5 (MB)."""
    if not mem_usage:
        return None
    used = mem_usage.split("/")[0].strip()
    return _parse_size(used)


def _parse_size(s: str) -> float | None:
    try:
        if s.endswith("GiB"):
            return round(float(s[:-3].strip()) * 1024, 2)
        if s.endswith("MiB"):
            return round(float(s[:-3].strip()), 2)
        if s.endswith("KiB"):
            return round(float(s[:-3].strip()) / 1024, 4)
        if s.endswith("GB"):
            return round(float(s[:-2].strip()) * 1000, 2)
        if s.endswith("MB"):
            return round(float(s[:-2].strip()), 2)
        return round(float(s.strip()), 2)
    except (ValueError, IndexError):
        return None


def collect_disk_usage_mb() -> float | None:
    """Best-effort disk usage (MB) of the workspace/data dir inside the container."""
    if not DOCKER_BIN:
        return None
    target = _resolve_container()
    if not target:
        return None
    raw = _run([DOCKER_BIN, "exec", target, "sh", "-c", "du -sm /data 2>/dev/null | cut -f1"], timeout=10)
    if not raw:
        return None
    try:
        return round(float(raw.splitlines()[0]), 1)
    except (ValueError, IndexError):
        return None


def collect_host_load() -> dict:
    """Host load average (1/5/15 min) — a coarse saturation signal."""
    raw = _run(["sh", "-c", "cat /proc/loadavg 2>/dev/null || sysctl -n vm.loadavg 2>/dev/null"], timeout=5)
    if not raw:
        return {"load_1": None, "load_5": None, "load_15": None}
    # /proc/loadavg: "0.52 0.41 0.35 1/456 1234"
    # sysctl vm.loadavg: "{ 0.52 0.41 0.35 }"
    parts = raw.replace("{", "").replace("}", "").split()
    try:
        return {
            "load_1": round(float(parts[0]), 2),
            "load_5": round(float(parts[1]), 2),
            "load_15": round(float(parts[2]), 2),
        }
    except (ValueError, IndexError):
        return {"load_1": None, "load_5": None, "load_15": None}


def collect_system_metrics() -> dict:
    """One-shot snapshot of docker + host metrics for a scenario."""
    return {
        "docker": collect_docker_stats(),
        "disk_mb": collect_disk_usage_mb(),
        "host_load": collect_host_load(),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
