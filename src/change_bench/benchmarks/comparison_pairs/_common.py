"""Shared constants, helpers, and the BenchmarkCase dataclass.

All comparison-pair modules import from here to avoid circular dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from change_bench.problems.base import BenchmarkProblem

# ---------------------------------------------------------------------------
# Benchmark case definition
# ---------------------------------------------------------------------------

BENCHMARK_SEED: int = 42


@dataclass
class BenchmarkCase:
    """A single benchmark case ready to be run.

    Uses a two-phase setup for memory efficiency:
    - ``prepare()`` generates data just-in-time (called once before timing loop)
    - ``setup(data)`` creates a fresh detector per run
    - ``func`` is the timed operation
    """

    package: str
    cpd_algorithm: str
    name: str
    n_samples: int
    n_changepoints: int
    data_dimension: int
    include_fit: bool
    min_segment_length: int
    prepare: Callable[[], np.ndarray]
    setup: Callable[[np.ndarray], tuple[tuple, dict]]
    func: Callable


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: Penalty used for all PELT-based pairs (same for skchange and ruptures).
PELT_PENALTY: float = 10.0

#: Bandwidth for moving-window pairs.
MW_BANDWIDTH: int = 25


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_prepare(problem: BenchmarkProblem, seed: int = BENCHMARK_SEED):
    """Create a prepare closure that generates data just-in-time."""

    def prepare() -> np.ndarray:
        return problem.generate(np.random.default_rng(seed))

    return prepare


def skchange_fit_predict(det, X):
    """Fit + predict for a skchange detector (the timed operation)."""
    det.fit(X)
    return det.predict_changepoints(X)


def skchange_predict_only(det, X):
    """Predict only (fit already done in setup)."""
    return det.predict_changepoints(X)
