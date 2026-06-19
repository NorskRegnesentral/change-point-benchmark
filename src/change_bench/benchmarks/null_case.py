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
    CostChangeScore,
    GaussianCost,
    L1Cost,
    L2Cost,
    LinearTrendCost,
)

from change_bench.problems.base import BenchmarkProblem, make_null_problems

# ---------------------------------------------------------------------------
# Benchmark case definition
# ---------------------------------------------------------------------------

BENCHMARK_SEED: int = 42


@dataclass
class BenchmarkCase:
    """A single benchmark case ready to be run."""

    package: str
    cpd_algorithm: str
    name: str
    n_samples: int
    n_changepoints: int
    data_dimension: int
    include_fit: bool
    min_segment_length: int
    setup: Callable[[], tuple[tuple, dict]]
    func: Callable


# ---------------------------------------------------------------------------
# Problem batteries
# ---------------------------------------------------------------------------

small_n_samples_list = [100, 250, 500, 750, 1000]
NULL_PROBLEMS_SMALL: list[BenchmarkProblem] = make_null_problems(
    n_samples_list=small_n_samples_list,
    distributions=["normal", "t", "gamma", "laplace", "exponential"],
    scale=1.0,
)

large_n_samples_list = [1500, 2500, 5000, 7500, 10_000]
NULL_PROBLEMS_FULL: list[BenchmarkProblem] = make_null_problems(
    n_samples_list=small_n_samples_list + large_n_samples_list,
    distributions=["normal", "t", "gamma", "laplace", "exponential"],
    scale=1.0,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: Penalty used for all PELT-based pairs (same for skchange and ruptures).
PELT_PENALTY: float = 10.0


def _skchange_run(det, X):
    """Fit + predict for a skchange detector (the timed operation)."""
    det.fit(X)
    return det.predict_changepoints(X)


def _skchange_predict_only(det, X):
    """Predict only (fit already done in setup)."""
    return det.predict_changepoints(X)


# ---------------------------------------------------------------------------
# Comparison pair: PELT + L2 cost
#   skchange: PELT(cost=L2Cost())
#   ruptures: KernelCPD(kernel="linear", min_size=1, jump=1)
# ---------------------------------------------------------------------------


def _pair_pelt_l2(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """PELT with L2/linear-kernel cost — skchange vs ruptures."""
    pair_name = "pelt_l2"
    cases: list[BenchmarkCase] = []
    sk_func = _skchange_run if include_fit else _skchange_predict_only

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)
        cfg = problem.dataset_config

        def make_sk_setup(d=data, fit=include_fit, msl=min_segment_length):
            def setup():
                det = SkchangePELT(
                    cost=L2Cost(),
                    penalty=PELT_PENALTY,
                    min_segment_length=msl,
                )
                if not fit:
                    det.fit(d)
                return (det, d), {}

            return setup

        def make_rpt_setup(d=data, fit=include_fit, msl=min_segment_length):
            def setup():
                algo = rpt.KernelCPD(kernel="linear", min_size=msl, jump=1)
                if not fit:
                    algo.fit(d)
                return (algo, d), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(pen=PELT_PENALTY)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_pelt_l2/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_sk_setup(),
                func=sk_func,
            )
        )

        cases.append(
            BenchmarkCase(
                package="ruptures",
                cpd_algorithm=pair_name,
                name=f"ruptures_kernelcpd_linear/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_rpt_setup(),
                func=rpt_func,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Comparison pair: PELT + Gaussian cost
#   skchange: PELT(cost=GaussianCost())
#   ruptures: Pelt(model="normal", min_size=1, jump=1)
# ---------------------------------------------------------------------------
def _pair_pelt_gaussian(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """PELT with Gaussian/normal cost — skchange vs ruptures."""
    pair_name = "pelt_gaussian"
    cases: list[BenchmarkCase] = []
    sk_func = _skchange_run if include_fit else _skchange_predict_only

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)
        cfg = problem.dataset_config

        def make_sk_setup(d=data, fit=include_fit, msl=min_segment_length):
            def setup():
                det = SkchangePELT(
                    cost=GaussianCost(),
                    penalty=PELT_PENALTY,
                    min_segment_length=msl,
                )
                if not fit:
                    det.fit(d)
                return (det, d), {}

            return setup

        def make_rpt_setup(d=data, fit=include_fit, msl=min_segment_length):
            def setup():
                algo = rpt.Pelt(model="normal", min_size=msl, jump=1)
                if not fit:
                    algo.fit(d)
                return (algo, d), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(pen=PELT_PENALTY)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_pelt_gaussian/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_sk_setup(),
                func=sk_func,
            )
        )

        cases.append(
            BenchmarkCase(
                package="ruptures",
                cpd_algorithm=pair_name,
                name=f"ruptures_pelt_normal/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_rpt_setup(),
                func=rpt_func,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Comparison pair: Moving Window + CUSUM / L2
#   skchange: MovingWindow(change_score=CUSUM())
#   ruptures: Window(model="l2", min_size=1, jump=1)
# ---------------------------------------------------------------------------
def _pair_moving_window(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Moving/sliding window with CUSUM/L2 — skchange vs ruptures."""
    pair_name = "moving_window"
    cases: list[BenchmarkCase] = []
    sk_func = _skchange_run if include_fit else _skchange_predict_only

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)
        cfg = problem.dataset_config

        def make_sk_setup(d=data, fit=include_fit):
            def setup():
                det = MovingWindow(change_score=CUSUM())
                if not fit:
                    det.fit(d)
                return (det, d), {}

            return setup

        def make_rpt_setup(d=data, fit=include_fit, msl=min_segment_length):
            def setup():
                algo = rpt.Window(model="l2", min_size=msl, jump=1)
                if not fit:
                    algo.fit(d)
                return (algo, d), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(n_bkps=0)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_moving_window_cusum/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_sk_setup(),
                func=sk_func,
            )
        )

        cases.append(
            BenchmarkCase(
                package="ruptures",
                cpd_algorithm=pair_name,
                name=f"ruptures_window_l2/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_rpt_setup(),
                func=rpt_func,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Comparison pair: Binary Segmentation + CUSUM / L2
#   skchange: SeededBinarySegmentation(change_score=CUSUM())
#   ruptures: Binseg(model="l2", min_size=1, jump=1)
# ---------------------------------------------------------------------------
def _pair_binseg(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Binary segmentation with CUSUM/L2 — skchange vs ruptures."""
    pair_name = "binseg"
    cases: list[BenchmarkCase] = []
    sk_func = _skchange_run if include_fit else _skchange_predict_only

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)
        cfg = problem.dataset_config

        def make_sk_setup(d=data, fit=include_fit):
            def setup():
                det = SeededBinarySegmentation(change_score=CUSUM())
                if not fit:
                    det.fit(d)
                return (det, d), {}

            return setup

        def make_rpt_setup(d=data, fit=include_fit, msl=min_segment_length):
            def setup():
                algo = rpt.Binseg(model="l2", min_size=msl, jump=1)
                if not fit:
                    algo.fit(d)
                return (algo, d), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(n_bkps=0)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_seeded_binseg_cusum/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_sk_setup(),
                func=sk_func,
            )
        )

        cases.append(
            BenchmarkCase(
                package="ruptures",
                cpd_algorithm=pair_name,
                name=f"ruptures_binseg_l2/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_rpt_setup(),
                func=rpt_func,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Comparison pair: MovingWindow + L2 cost (explicit bandwidth)
#   skchange: MovingWindow(change_score=CostChangeScore(L2Cost()), bandwidth=BW)
#   ruptures: Window(model="l2", width=2*BW, min_size=1, jump=1)
# ---------------------------------------------------------------------------
_MW_BANDWIDTH: int = 25


def _pair_moving_window_l2(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Moving window with L2 cost (fixed bandwidth) — skchange vs ruptures."""
    pair_name = "moving_window_l2"
    bw = _MW_BANDWIDTH
    cases: list[BenchmarkCase] = []
    sk_func = _skchange_run if include_fit else _skchange_predict_only

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)
        cfg = problem.dataset_config

        def make_sk_setup(d=data, bandwidth=bw, fit=include_fit):
            def setup():
                det = MovingWindow(
                    change_score=CostChangeScore(L2Cost()), bandwidth=bandwidth
                )
                if not fit:
                    det.fit(d)
                return (det, d), {}

            return setup

        def make_rpt_setup(
            d=data, width=2 * bw, fit=include_fit, msl=min_segment_length
        ):
            def setup():
                algo = rpt.Window(model="l2", width=width, min_size=msl, jump=1)
                if not fit:
                    algo.fit(d)
                return (algo, d), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(n_bkps=0)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_moving_window_l2/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_sk_setup(),
                func=sk_func,
            )
        )

        cases.append(
            BenchmarkCase(
                package="ruptures",
                cpd_algorithm=pair_name,
                name=f"ruptures_window_l2/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_rpt_setup(),
                func=rpt_func,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Comparison pair: MovingWindow + L1 cost (explicit bandwidth)
#   skchange: MovingWindow(change_score=CostChangeScore(L1Cost()), bandwidth=BW)
#   ruptures: Window(model="l1", width=2*BW, min_size=1, jump=1)
# ---------------------------------------------------------------------------


def _pair_moving_window_l1(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Moving window with L1 cost (fixed bandwidth) — skchange vs ruptures."""
    pair_name = "moving_window_l1"
    bw = _MW_BANDWIDTH
    cases: list[BenchmarkCase] = []
    sk_func = _skchange_run if include_fit else _skchange_predict_only

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)
        cfg = problem.dataset_config

        def make_sk_setup(d=data, bandwidth=bw, fit=include_fit):
            def setup():
                det = MovingWindow(
                    change_score=CostChangeScore(L1Cost()), bandwidth=bandwidth
                )
                if not fit:
                    det.fit(d)
                return (det, d), {}

            return setup

        def make_rpt_setup(
            d=data, width=2 * bw, fit=include_fit, msl=min_segment_length
        ):
            def setup():
                algo = rpt.Window(model="l1", width=width, min_size=msl, jump=1)
                if not fit:
                    algo.fit(d)
                return (algo, d), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(n_bkps=0)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_moving_window_l1/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_sk_setup(),
                func=sk_func,
            )
        )

        cases.append(
            BenchmarkCase(
                package="ruptures",
                cpd_algorithm=pair_name,
                name=f"ruptures_window_l1/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_rpt_setup(),
                func=rpt_func,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# NOTE: Not really a fair comparison, we solve for 'best linear trend'
#       in the segment, whilst ruptures draws a line from the start value
#       to the end value.
# Comparison pair: PELT + Linear Trend cost
#   skchange: PELT(cost=LinearTrendCost())
#   ruptures: Pelt(custom_cost=CostCLinear(), min_size=1, jump=1)
# ---------------------------------------------------------------------------
def _pair_pelt_linear_trend(
    problems: list[BenchmarkProblem],
    *,
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """PELT with linear-trend cost — skchange vs ruptures."""
    pair_name = "pelt_linear_trend"
    cases: list[BenchmarkCase] = []
    sk_func = _skchange_run if include_fit else _skchange_predict_only

    for problem in problems:
        rng = np.random.default_rng(BENCHMARK_SEED)
        data = problem.generate(rng)
        cfg = problem.dataset_config

        def make_sk_setup(d=data, fit=include_fit, msl=min_segment_length):
            def setup():
                det = SkchangePELT(
                    cost=LinearTrendCost(),
                    penalty=PELT_PENALTY,
                    min_segment_length=msl,
                )
                if not fit:
                    det.fit(d)
                return (det, d), {}

            return setup

        def make_rpt_setup(d=data, fit=include_fit, msl=min_segment_length):
            def setup():
                algo = rpt.Pelt(
                    custom_cost=rpt.costs.CostCLinear(), min_size=msl, jump=1
                )
                if not fit:
                    algo.fit(d)
                return (algo, d), {}

            return setup

        def rpt_func(algo, d, _fit=include_fit):
            if _fit:
                algo.fit(d)
            return algo.predict(pen=PELT_PENALTY)

        cases.append(
            BenchmarkCase(
                package="skchange",
                cpd_algorithm=pair_name,
                name=f"skchange_pelt_linear_trend/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_sk_setup(),
                func=sk_func,
            )
        )

        cases.append(
            BenchmarkCase(
                package="ruptures",
                cpd_algorithm=pair_name,
                name=f"ruptures_pelt_clinear/{problem.name}",
                n_samples=cfg.n_samples,
                n_changepoints=len(problem.true_changepoints),
                data_dimension=cfg.n_columns,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
                setup=make_rpt_setup(),
                func=rpt_func,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Registry: maps pair name -> factory function
# ---------------------------------------------------------------------------
BENCHMARK_PAIRS: dict[str, Callable[..., list[BenchmarkCase]]] = {
    "pelt_l2": _pair_pelt_l2,
    "pelt_gaussian": _pair_pelt_gaussian,
    # "pelt_linear_trend": _pair_pelt_linear_trend,
    "moving_window": _pair_moving_window,
    "moving_window_l2": _pair_moving_window_l2,
    "moving_window_l1": _pair_moving_window_l1,
    "binseg": _pair_binseg,
}


PAIR_CATEGORIES: dict[str, list[str]] = {
    "mean_change": [
        "pelt_l2",
        "moving_window",
        "moving_window_l2",
        "moving_window_l1",
        "binseg",
    ],
    "needs_min_segment_length": [
        "pelt_gaussian",
        # "pelt_linear_trend",
    ],
}


def collect_cases(
    packages: list[str] | None = None,
    pairs: list[str] | None = None,
    categories: list[str] | None = None,
    problem_set: str = "small",
    include_fit: bool = True,
    min_segment_length: int = 1,
) -> list[BenchmarkCase]:
    """Collect benchmark cases, optionally filtered by package, pair, or category.

    Parameters
    ----------
    packages:
        Filter to only include cases from these packages (``"ruptures"``,
        ``"skchange"``). ``None`` means both.
    pairs:
        List of comparison-pair names to include. ``None`` means all pairs.
    categories:
        Filter pairs by category (``"mean_change"``, ``"mean_variance"``).
        ``None`` means all categories.  When both *pairs* and *categories*
        are given, the union is used.
    problem_set:
        ``"small"`` or ``"full"`` problem battery.
    include_fit:
        If ``True`` (default), the timed operation includes both ``fit`` and
        ``predict``.  If ``False``, only ``predict`` is timed.
    min_segment_length:
        Minimum segment length for the detector (default: 1).  Maps to
        ``min_size`` in ruptures and ``min_segment_length`` in skchange PELT.
    """
    problems = NULL_PROBLEMS_SMALL if problem_set == "small" else NULL_PROBLEMS_FULL

    # Resolve which pairs to run
    selected_pairs: list[str] = []
    if categories:
        for cat in categories:
            if cat not in PAIR_CATEGORIES:
                raise ValueError(
                    f"Unknown category {cat!r}. Available: {sorted(PAIR_CATEGORIES)}"
                )
            selected_pairs.extend(PAIR_CATEGORIES[cat])
    if pairs:
        selected_pairs.extend(pairs)
    if not selected_pairs:
        selected_pairs = list(BENCHMARK_PAIRS)
    # Deduplicate while preserving order
    seen: set[str] = set()
    selected_pairs = [
        p
        for p in selected_pairs
        if not (p in seen or seen.add(p))  # type: ignore[func-returns-value]
    ]

    cases: list[BenchmarkCase] = []
    for p in selected_pairs:
        if p not in BENCHMARK_PAIRS:
            raise ValueError(
                f"Unknown benchmark pair {p!r}. Available: {sorted(BENCHMARK_PAIRS)}"
            )
        cases.extend(
            BENCHMARK_PAIRS[p](
                problems,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
            )
        )

    # Optionally filter by package (ruptures / skchange)
    if packages:
        cases = [c for c in cases if c.package in packages]

    return cases
