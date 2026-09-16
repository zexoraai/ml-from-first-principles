"""Adaptive amortised timing for short operations.

WHY THIS EXISTS -- a real bug, kept as documentation
----------------------------------------------------
The first version of `scripts/bench_cpu.py` timed one call per sample:

    start = time.perf_counter(); fn(); samples.append(time.perf_counter() - start)

with warm-up and a median over repeats. That looks like a careful benchmark and it produced
numbers that were wrong by up to 300x:

    128x128 fp32 matmul   31.68 ms measured   ->    0.104 ms actual
    2048x2048 matmul     252.18 ms measured   ->   95.76 ms actual  (68 -> 179 GFLOP/s)
    0.53 M-param step   4979.70 ms measured   ->  100.45 ms actual

The tell was that a 27 M-parameter training step came out *faster* than a 0.53 M-parameter one.
A larger model performing strictly more arithmetic in less wall-clock time is impossible, so the
harness had to be wrong -- and reading the number as "this laptop is slow" would have propagated
a 50x error into every compute estimate downstream.

Diagnosis (`scripts/diag_threads.py`, recorded in evidence/env/thread_diagnosis.json): a large
fixed cost is paid on entry to each timed region -- consistent with PyTorch's intra-op thread pool
parking between calls and being woken again, which under a hypervisor is expensive. Back-to-back
calls inside one timed region keep the pool hot; isolated calls do not.

The fix is the standard one for microbenchmarks: choose the inner repeat count at run time so each
timed region lasts long enough that entry costs and clock granularity are negligible, then divide.

WHAT THIS FUNCTION DOES AND DOES NOT GUARANTEE
----------------------------------------------
It measures *steady-state throughput* of repeated calls. That is the right model for a training
loop, which is exactly what we extrapolate from. It is the wrong model for cold-start latency --
for a single-shot inference latency figure, measure a single shot and say so.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

__all__ = ["timed", "TimingResult"]

TimingResult = dict[str, float | int]


def timed(
    fn: Callable[[], object],
    *,
    warmup: int = 3,
    target_seconds: float = 0.30,
    samples: int = 5,
    max_inner: int = 1 << 20,
) -> TimingResult:
    """Time `fn` with an automatically chosen inner repeat count.

    Args:
        fn: zero-argument callable to measure. Must be safe to call repeatedly.
        warmup: calls made and discarded before any measurement. Absorbs lazy allocation,
            kernel selection, and first-touch page faults.
        target_seconds: approximate duration of each timed block. Larger is more accurate and
            slower; 0.3 s keeps the whole sweep tolerable on a mobile CPU.
        samples: number of timed blocks. The median is reported; min/max expose thermal
            throttling, which a mean would smear away.
        max_inner: safety cap so a pathologically fast `fn` cannot spin forever.

    Returns:
        dict with `per_call_s` (median), `min_s`, `max_s`, `inner` (calls per block), and
        `n_samples`. `min_s`/`max_s` are per-call figures from the fastest and slowest blocks.

    Reporting the spread is not decoration. This CPU is a 15 W mobile part in a laptop chassis; a
    long sweep will throttle, and a benchmark that hides that invites a reader to treat the median
    as a sustained rate when it is not.
    """
    if warmup < 0 or samples < 1:
        raise ValueError("warmup must be >= 0 and samples >= 1")

    for _ in range(warmup):
        fn()

    # --- calibrate `inner` so one block is long enough to measure reliably -------------------
    inner = 1
    elapsed = 0.0
    while inner < max_inner:
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        elapsed = time.perf_counter() - start
        if elapsed >= 0.02:
            break
        inner *= 4
    if elapsed > 0:
        inner = max(1, min(max_inner, int(inner * target_seconds / elapsed)))

    # --- measure -----------------------------------------------------------------------------
    per_call: list[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        per_call.append((time.perf_counter() - start) / inner)

    return {
        "per_call_s": statistics.median(per_call),
        "min_s": min(per_call),
        "max_s": max(per_call),
        "inner": inner,
        "n_samples": len(per_call),
    }
