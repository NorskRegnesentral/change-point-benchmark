"""Null-case benchmarks: zero change-points, various distributions.

Run via the CLI::

    uv run bench --runs 10 -o results.parquet
    uv run bench --runs 10 --pairs pelt_l2 moving_window -o results.parquet

Benchmarks are organised as **comparison pairs**: each pair contains one
skchange detector and its equivalent ruptures detector so that timing
differences are directly attributable to the implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import ruptures as rpt
from skchange.new_api.detectors import (
    PELT as SkchangePELT,
)
from skchange.new_api.detectors import (
    MovingWindow,
    SeededBinarySegmentation,
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
    pair: str
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
# Shared helpers
# ---------------------------------------------------------------------------


def _skchange_run(det, X):
    """Fit + predict for a skchange detector (the timed operation)."""
    det.fit(X)
    return det.predict_changepoints(X)


# ---------------------------------------------------------------------------
# Comparison pair: PELT + L2 cost
#   skchange: PELT(cost=L2Cost())
#   ruptures: KernelCPD(kernel="linear", min_size=1, jump=1)
# ---------------------------------------------------------------------------


def _pair_pelt_l2(problems: list[BenchmarkProblem]) -> list[BenchmarkCase]:
    """PELT with L2/linear-kernel cost — skchange vs ruptures."""
    pair_name = "pelt_l2"
    cases: list[BenchmarkCase] = []

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)

        # --- skchange side ---
        def make_sk_setup(d=data):
            def setup():
                det = SkchangePELT(cost=L2Cost())
                return (det, d), {}

            return setup

        cases.append(
            BenchmarkCase(
                group="skchange",
                pair=pair_name,
                name=f"pelt_l2/{problem.name}",
                setup=make_sk_setup(),
                func=_skchange_run,
            )
        )

        # --- ruptures side ---
        def make_rpt_setup(d=data):
            def setup():
                algo = rpt.KernelCPD(kernel="linear", min_size=1, jump=1)
                algo.fit(d)
                return (algo,), {}

            return setup

        cases.append(
            BenchmarkCase(
                group="ruptures",
                pair=pair_name,
                name=f"kernelcpd_linear/{problem.name}",
                setup=make_rpt_setup(),
                func=lambda algo: algo.predict(pen=10),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Comparison pair: PELT + Gaussian cost
#   skchange: PELT(cost=GaussianCost())
#   ruptures: Pelt(model="normal", min_size=1, jump=1)
# ---------------------------------------------------------------------------


def _pair_pelt_gaussian(problems: list[BenchmarkProblem]) -> list[BenchmarkCase]:
    """PELT with Gaussian/normal cost — skchange vs ruptures."""
    pair_name = "pelt_gaussian"
    cases: list[BenchmarkCase] = []

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)

        # --- skchange side ---
        def make_sk_setup(d=data):
            def setup():
                det = SkchangePELT(cost=GaussianCost())
                return (det, d), {}

            return setup

        cases.append(
            BenchmarkCase(
                group="skchange",
                pair=pair_name,
                name=f"pelt_gaussian/{problem.name}",
                setup=make_sk_setup(),
                func=_skchange_run,
            )
        )

        # --- ruptures side ---
        def make_rpt_setup(d=data):
            def setup():
                algo = rpt.Pelt(model="normal", min_size=1, jump=1)
                algo.fit(d)
                return (algo,), {}

            return setup

        cases.append(
            BenchmarkCase(
                group="ruptures",
                pair=pair_name,
                name=f"pelt_normal/{problem.name}",
                setup=make_rpt_setup(),
                func=lambda algo: algo.predict(pen=10),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Comparison pair: Moving Window + CUSUM / L2
#   skchange: MovingWindow(change_score=CUSUM())
#   ruptures: Window(model="l2", min_size=1, jump=1)
# ---------------------------------------------------------------------------


def _pair_moving_window(problems: list[BenchmarkProblem]) -> list[BenchmarkCase]:
    """Moving/sliding window with CUSUM/L2 — skchange vs ruptures."""
    pair_name = "moving_window"
    cases: list[BenchmarkCase] = []

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)

        # --- skchange side ---
        def make_sk_setup(d=data):
            def setup():
                det = MovingWindow(change_score=CUSUM())
                return (det, d), {}

            return setup

        cases.append(
            BenchmarkCase(
                group="skchange",
                pair=pair_name,
                name=f"moving_window_cusum/{problem.name}",
                setup=make_sk_setup(),
                func=_skchange_run,
            )
        )

        # --- ruptures side ---
        def make_rpt_setup(d=data):
            def setup():
                algo = rpt.Window(model="l2", min_size=1, jump=1)
                algo.fit(d)
                return (algo,), {}

            return setup

        cases.append(
            BenchmarkCase(
                group="ruptures",
                pair=pair_name,
                name=f"window_l2/{problem.name}",
                setup=make_rpt_setup(),
                func=lambda algo: algo.predict(n_bkps=0),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Comparison pair: Binary Segmentation + CUSUM / L2
#   skchange: SeededBinarySegmentation(change_score=CUSUM())
#   ruptures: Binseg(model="l2", min_size=1, jump=1)
# ---------------------------------------------------------------------------


def _pair_binseg(problems: list[BenchmarkProblem]) -> list[BenchmarkCase]:
    """Binary segmentation with CUSUM/L2 — skchange vs ruptures."""
    pair_name = "binseg"
    cases: list[BenchmarkCase] = []

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)

        # --- skchange side ---
        def make_sk_setup(d=data):
            def setup():
                det = SeededBinarySegmentation(change_score=CUSUM())
                return (det, d), {}

            return setup

        cases.append(
            BenchmarkCase(
                group="skchange",
                pair=pair_name,
                name=f"seeded_binseg_cusum/{problem.name}",
                setup=make_sk_setup(),
                func=_skchange_run,
            )
        )

        # --- ruptures side ---
        def make_rpt_setup(d=data):
            def setup():
                algo = rpt.Binseg(model="l2", min_size=1, jump=1)
                algo.fit(d)
                return (algo,), {}

            return setup

        cases.append(
            BenchmarkCase(
                group="ruptures",
                pair=pair_name,
                name=f"binseg_l2/{problem.name}",
                setup=make_rpt_setup(),
                func=lambda algo: algo.predict(n_bkps=0),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Registry: maps pair name -> factory function
# ---------------------------------------------------------------------------

BENCHMARK_PAIRS: dict[str, Callable[[list[BenchmarkProblem]], list[BenchmarkCase]]] = {
    "pelt_l2": _pair_pelt_l2,
    "pelt_gaussian": _pair_pelt_gaussian,
    "moving_window": _pair_moving_window,
    "binseg": _pair_binseg,
}


def collect_cases(
    groups: list[str] | None = None,
    pairs: list[str] | None = None,
    problem_set: str = "small",
) -> list[BenchmarkCase]:
    """Collect benchmark cases, optionally filtered by group and/or pair.

    Parameters
    ----------
    groups:
        Filter to only include cases from these groups (``"ruptures"``,
        ``"skchange"``). ``None`` means both.
    pairs:
        List of comparison-pair names to include. ``None`` means all pairs.
    problem_set:
        ``"small"`` or ``"full"`` problem battery.
    """
    problems = NULL_PROBLEMS_SMALL if problem_set == "small" else NULL_PROBLEMS_FULL
    selected_pairs = pairs if pairs else list(BENCHMARK_PAIRS)

    cases: list[BenchmarkCase] = []
    for p in selected_pairs:
        if p not in BENCHMARK_PAIRS:
            raise ValueError(
                f"Unknown benchmark pair {p!r}. Available: {sorted(BENCHMARK_PAIRS)}"
            )
        cases.extend(BENCHMARK_PAIRS[p](problems))

    # Optionally filter by group (ruptures / skchange)
    if groups:
        cases = [c for c in cases if c.group in groups]

    return cases
