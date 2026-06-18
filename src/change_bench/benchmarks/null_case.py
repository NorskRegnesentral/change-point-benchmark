"""Null-case benchmarks: zero change-points, various distributions.

Run via the CLI::

    uv run bench --groups ruptures skchange --runs 10 -o results.parquet

Each factory function builds a list of :class:`BenchmarkCase` objects.
The CLI runner discovers and executes them using ``timeit``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import ruptures as rpt
from skchange.new_api.detectors import (
    CROPS,
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.new_api.detectors import (
    PELT as SkchangePELT,
)
from skchange.new_api.interval_scorers import (
    CUSUM,
    GaussianCost,
    L2Cost,
)

from change_bench.problems.base import BenchmarkProblem, make_null_problems

# ---------------------------------------------------------------------------
# Benchmark case definition
# ---------------------------------------------------------------------------

BENCHMARK_SEED: int = 42


@dataclass
class BenchmarkCase:
    """A single benchmark case ready to be run."""

    group: str
    name: str
    setup: Callable[[], tuple[tuple, dict]]
    func: Callable


# ---------------------------------------------------------------------------
# Problem batteries
# ---------------------------------------------------------------------------

NULL_PROBLEMS_SMALL: list[BenchmarkProblem] = make_null_problems(
    n_samples_list=[500, 1000],
    distributions=["normal", "t", "gamma", "laplace", "exponential"],
    scale=1.0,
)

NULL_PROBLEMS_FULL: list[BenchmarkProblem] = make_null_problems(
    n_samples_list=[500, 1000, 5000, 10_000],
    distributions=["normal", "t", "gamma", "laplace", "exponential"],
    scale=1.0,
)


# ---------------------------------------------------------------------------
# ruptures benchmarks
# ---------------------------------------------------------------------------


def _ruptures_cases(problems: list[BenchmarkProblem]) -> list[BenchmarkCase]:
    """Build benchmark cases for ruptures detectors."""
    cases: list[BenchmarkCase] = []

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)

        # PELT RBF
        def make_pelt_rbf_setup(d=data):
            def setup():
                algo = rpt.Pelt(model="rbf")
                algo.fit(d)
                return (algo,), {}
            return setup

        cases.append(BenchmarkCase(
            group="ruptures",
            name=f"pelt_rbf/{problem.name}",
            setup=make_pelt_rbf_setup(),
            func=lambda algo: algo.predict(pen=10),
        ))

        # PELT L2
        def make_pelt_l2_setup(d=data):
            def setup():
                algo = rpt.Pelt(model="l2")
                algo.fit(d)
                return (algo,), {}
            return setup

        cases.append(BenchmarkCase(
            group="ruptures",
            name=f"pelt_l2/{problem.name}",
            setup=make_pelt_l2_setup(),
            func=lambda algo: algo.predict(pen=10),
        ))

        # BinSeg RBF
        def make_binseg_rbf_setup(d=data):
            def setup():
                algo = rpt.Binseg(model="rbf")
                algo.fit(d)
                return (algo,), {}
            return setup

        cases.append(BenchmarkCase(
            group="ruptures",
            name=f"binseg_rbf/{problem.name}",
            setup=make_binseg_rbf_setup(),
            func=lambda algo: algo.predict(n_bkps=0),
        ))

        # Window RBF
        def make_window_rbf_setup(d=data):
            def setup():
                algo = rpt.Window(model="rbf")
                algo.fit(d)
                return (algo,), {}
            return setup

        cases.append(BenchmarkCase(
            group="ruptures",
            name=f"window_rbf/{problem.name}",
            setup=make_window_rbf_setup(),
            func=lambda algo: algo.predict(n_bkps=0),
        ))

    return cases


# ---------------------------------------------------------------------------
# skchange benchmarks
# ---------------------------------------------------------------------------


def _skchange_run(det, X):
    """Fit + predict for a skchange detector (the timed operation)."""
    det.fit(X)
    return det.predict_changepoints(X)


def _skchange_cases(problems: list[BenchmarkProblem]) -> list[BenchmarkCase]:
    """Build benchmark cases for skchange new_api detectors."""
    cases: list[BenchmarkCase] = []

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)

        # PELT L2Cost
        def make_pelt_l2_setup(d=data):
            def setup():
                det = SkchangePELT(cost=L2Cost())
                return (det, d), {}
            return setup

        cases.append(BenchmarkCase(
            group="skchange",
            name=f"pelt_l2/{problem.name}",
            setup=make_pelt_l2_setup(),
            func=_skchange_run,
        ))

        # PELT GaussianCost
        def make_pelt_gauss_setup(d=data):
            def setup():
                det = SkchangePELT(cost=GaussianCost())
                return (det, d), {}
            return setup

        cases.append(BenchmarkCase(
            group="skchange",
            name=f"pelt_gaussian/{problem.name}",
            setup=make_pelt_gauss_setup(),
            func=_skchange_run,
        ))

        # MovingWindow CUSUM
        def make_mw_cusum_setup(d=data):
            def setup():
                det = MovingWindow(change_score=CUSUM())
                return (det, d), {}
            return setup

        cases.append(BenchmarkCase(
            group="skchange",
            name=f"moving_window_cusum/{problem.name}",
            setup=make_mw_cusum_setup(),
            func=_skchange_run,
        ))

        # SeededBinarySegmentation CUSUM
        def make_sbs_cusum_setup(d=data):
            def setup():
                det = SeededBinarySegmentation(change_score=CUSUM())
                return (det, d), {}
            return setup

        cases.append(BenchmarkCase(
            group="skchange",
            name=f"seeded_binseg_cusum/{problem.name}",
            setup=make_sbs_cusum_setup(),
            func=_skchange_run,
        ))

        # CROPS L2Cost
        def make_crops_l2_setup(d=data):
            def setup():
                det = CROPS(cost=L2Cost())
                return (det, d), {}
            return setup

        cases.append(BenchmarkCase(
            group="skchange",
            name=f"crops_l2/{problem.name}",
            setup=make_crops_l2_setup(),
            func=_skchange_run,
        ))

    return cases


# ---------------------------------------------------------------------------
# Registry: maps group name -> factory function
# ---------------------------------------------------------------------------

BENCHMARK_GROUPS: dict[str, Callable[[list[BenchmarkProblem]], list[BenchmarkCase]]] = {
    "ruptures": _ruptures_cases,
    "skchange": _skchange_cases,
}


def collect_cases(
    groups: list[str] | None = None,
    problem_set: str = "small",
) -> list[BenchmarkCase]:
    """Collect benchmark cases, optionally filtered by group.

    Parameters
    ----------
    groups:
        List of group names to include. ``None`` means all groups.
    problem_set:
        ``"small"`` or ``"full"`` problem battery.
    """
    problems = NULL_PROBLEMS_SMALL if problem_set == "small" else NULL_PROBLEMS_FULL
    selected = groups if groups else list(BENCHMARK_GROUPS)

    cases: list[BenchmarkCase] = []
    for g in selected:
        if g not in BENCHMARK_GROUPS:
            raise ValueError(
                f"Unknown benchmark group {g!r}. "
                f"Available: {sorted(BENCHMARK_GROUPS)}"
            )
        cases.extend(BENCHMARK_GROUPS[g](problems))

    return cases

