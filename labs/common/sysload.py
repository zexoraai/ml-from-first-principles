"""Record system contention alongside every benchmark.

WHY THIS EXISTS
---------------
This laptop is not a quiet benchmarking rig. At the time of the first measurements, 37 unrelated
containers were running (OpenSearch, ClamAV, ten-ish Postgres/pgvector instances, Redis, MinIO,
Temporal, LiteLLM, and several application services), all up for seven hours. They belong to the
user's other projects and are not ours to stop.

The effect is not subtle. The same 256x256 matmul measured 1.36 ms on one run and 19.17 ms on
another a few minutes later. A benchmark that reports the second number as "this CPU's
performance" is simply wrong, and one that reports the first without saying the machine was busy
is unreproducible.

So: every benchmark record carries a load snapshot. That converts an unexplainable outlier into an
annotated one, and it lets a reader decide how much weight a number deserves. Where numbers are
used for extrapolation, the run's contention state must be quoted with them.

`os.getloadavg()` inside the container reports the **WSL2 VM's** load, which includes every other
container, because they all share one kernel. That is exactly the quantity we want.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_snapshot", "describe_load"]


def _meminfo_mb() -> dict[str, float]:
    """Parse /proc/meminfo. Values are in kB in the file; returned in MiB."""
    out: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable", "MemFree"}:
                out[key] = float(rest.strip().split()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return out


def load_snapshot() -> dict[str, object]:
    """Capture contention indicators at a moment in time.

    Returns a dict with 1/5/15-minute load averages, the CPU count they should be compared
    against, a derived `load_per_cpu`, and memory availability.

    `load_per_cpu` is the number to look at: values near or above 1.0 mean the machine is
    saturated and any timing taken then is a lower bound on the hardware's capability, not a
    measurement of it.
    """
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        one = five = fifteen = float("nan")

    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    mem = _meminfo_mb()

    return {
        "loadavg_1m": round(one, 2),
        "loadavg_5m": round(five, 2),
        "loadavg_15m": round(fifteen, 2),
        "n_cpu_visible": n_cpu,
        "load_per_cpu_1m": round(one / n_cpu, 3) if n_cpu else None,
        "mem_total_mb": round(mem.get("MemTotal", float("nan")), 1),
        "mem_available_mb": round(mem.get("MemAvailable", float("nan")), 1),
        "contended": bool(one / n_cpu > 0.5) if n_cpu else None,
        "caveat": "load includes all other containers sharing the WSL2 kernel; this machine is "
                  "not an isolated benchmark host",
    }


def describe_load(snap: dict[str, object]) -> str:
    """One-line human summary suitable for printing above a results table."""
    return (
        f"loadavg {snap['loadavg_1m']}/{snap['loadavg_5m']}/{snap['loadavg_15m']} "
        f"over {snap['n_cpu_visible']} visible CPUs "
        f"(load/cpu={snap['load_per_cpu_1m']}), "
        f"mem_available={snap['mem_available_mb']:.0f} MiB, "
        f"{'CONTENDED -- timings are lower bounds' if snap['contended'] else 'relatively quiet'}"
    )
