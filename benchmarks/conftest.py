"""Shared pytest-benchmark fixtures for change-point benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from change_bench.problems.base import BenchmarkProblem, make_null_problems

# ---------------------------------------------------------------------------
# Reproducible RNG seed used across all benchmarks.
# ---------------------------------------------------------------------------

BENCHMARK_SEED: int = 42

# ---------------------------------------------------------------------------
# Problem batteries
# ---------------------------------------------------------------------------

#: Small battery for quick CI checks and smoke tests.
NULL_PROBLEMS_SMALL: list[BenchmarkProblem] = make_null_problems(
    n_samples_list=[500, 1000],
    distributions=["normal", "t", "gamma", "laplace", "exponential"],
    scale=1.0,
)

#: Full battery for thorough performance comparisons.
NULL_PROBLEMS_FULL: list[BenchmarkProblem] = make_null_problems(
    n_samples_list=[500, 1000, 5000, 10_000],
    distributions=["normal", "t", "gamma", "laplace", "exponential"],
    scale=1.0,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=NULL_PROBLEMS_SMALL,
    ids=lambda p: p.name,
)
def null_problem(request: pytest.FixtureRequest) -> BenchmarkProblem:
    """Parametrised fixture yielding each null-case problem in the small battery."""
    return request.param  # type: ignore[return-value]


@pytest.fixture()
def null_dataset(null_problem: BenchmarkProblem) -> np.ndarray:
    """Pre-generated dataset array for the current null problem.

    Data generation happens in fixture setup and is **not** counted in
    benchmark timings.
    """
    rng = np.random.default_rng(BENCHMARK_SEED)
    return null_problem.generate(rng)
