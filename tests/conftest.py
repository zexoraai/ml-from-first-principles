"""Shared pytest configuration.

Determinism is set up here rather than in each test so that a failure is always reproducible.
Single-threaded intraop execution is deliberate: it removes float non-determinism caused by
different reduction orders across thread counts, which would otherwise make tight numerical
tolerances flaky for reasons unrelated to correctness. Benchmarks re-enable threads explicitly.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

SEED = 1706  # arXiv:1706.03762


@pytest.fixture(autouse=True)
def _determinism() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
