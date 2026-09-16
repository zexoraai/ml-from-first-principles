"""Utilities shared by more than one project.

Per decision D-005, code is promoted here only once a **second** caller exists. Nothing is added
here in anticipation of future use. Current residents and who forced the promotion:

* `timing` -- `scripts/bench_cpu.py` and `scripts/diag_threads.py` both need an adaptive,
  amortised timer after the measurement bug documented in records/EXPERIMENTS.md (env-bench-01
  vs env-bench-02). Duplicating a benchmark harness across scripts is how two "identical"
  benchmarks end up disagreeing, so this one lives in a single place.
"""

__all__ = ["timing"]
