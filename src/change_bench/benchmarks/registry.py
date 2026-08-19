"""Benchmark registry and case collector.

Maps comparison-pair names to their factory functions and provides
:func:`collect_cases` to assemble benchmark batteries filtered by package,
pair name, category, and data dimensionality.

Run via the CLI::

    uv run bench --runs 10 -o results.parquet
    uv run bench --runs 10 --pairs pelt_l2 moving_window -o results.parquet
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from change_bench.benchmarks.comparison_pairs import (
    BenchmarkCase,
    pair_binseg_continuous_linear_trend,
    pair_binseg_esac,
    pair_binseg_l1,
    pair_binseg_l2_cusum,
    pair_binseg_linreg,
    pair_binseg_mv_gaussian,
    pair_binseg_poisson,
    pair_binseg_rank,
    pair_fpop_l2,
    pair_moving_window_esac,
    pair_moving_window_continuous_linear_trend,
    pair_moving_window_l1,
    pair_moving_window_l2,
    pair_moving_window_linreg,
    pair_moving_window_mv_gaussian,
    pair_moving_window_poisson,
    pair_moving_window_rank,
    pair_pelt_1d_gaussian,
    pair_pelt_l1,
    pair_pelt_l2,
    pair_pelt_linreg,
    pair_pelt_mv_gaussian,
    pair_pelt_poisson,
    pair_pelt_rank,
)
from change_bench.problems.base import BenchmarkProblem, make_null_problems


# ---------------------------------------------------------------------------
# Pair enum — the canonical identifier for every comparison pair.
# Uses StrEnum so values work as dict keys and CLI arguments directly.
# ---------------------------------------------------------------------------
class Pair(StrEnum):
    """Available comparison pairs."""

    PELT_L2 = "pelt_l2"
    FPOP_L2 = "fpop_l2"
    PELT_L1 = "pelt_l1"
    PELT_1D_GAUSSIAN = "pelt_1d_gaussian"
    PELT_POISSON = "pelt_poisson"
    PELT_RANK = "pelt_rank"
    PELT_LINREG = "pelt_linreg"
    MOVING_WINDOW_L2 = "moving_window_l2"
    MOVING_WINDOW_L1 = "moving_window_l1"
    MOVING_WINDOW_ESAC = "moving_window_esac"
    MOVING_WINDOW_POISSON = "moving_window_poisson"
    MOVING_WINDOW_RANK = "moving_window_rank"
    MOVING_WINDOW_LINREG = "moving_window_linreg"
    BINSEG_L2_CUSUM = "binseg_l2_cusum"
    BINSEG_L1 = "binseg_l1"
    BINSEG_ESAC = "binseg_esac"
    BINSEG_POISSON = "binseg_poisson"
    BINSEG_RANK = "binseg_rank"
    BINSEG_LINREG = "binseg_linreg"
    BINSEG_CONTINUOUS_LINEAR_TREND = "binseg_continuous_linear_trend"
    BINSEG_MV_GAUSSIAN = "binseg_mv_gaussian"
    PELT_MV_GAUSSIAN = "pelt_mv_gaussian"
    MOVING_WINDOW_MV_GAUSSIAN = "moving_window_mv_gaussian"
    MOVING_WINDOW_CONTINUOUS_LINEAR_TREND = (
        "moving_window_continuous_linear_trend"
    )


# ---------------------------------------------------------------------------
# Problem batteries
# ---------------------------------------------------------------------------

#: All supported null-case distributions.
ALL_DISTRIBUTIONS: list[str] = ["normal", "t", "gamma", "laplace", "exponential"]

#: Default distributions when none are specified.
DEFAULT_DISTRIBUTIONS: list[str] = ["normal"]


def _make_problems(
    n_samples_list: list[int],
    n_columns_list: list[int],
    distributions: list[str] | None = None,
) -> list[BenchmarkProblem]:
    """Create problem battery with given dimensions."""
    if distributions is None:
        distributions = DEFAULT_DISTRIBUTIONS
    return make_null_problems(
        n_samples_list=n_samples_list,
        distributions=distributions,
        scale=1.0,
        n_columns_list=n_columns_list,
    )


# ---------------------------------------------------------------------------
# Registry: maps pair name -> factory function
# ---------------------------------------------------------------------------
BENCHMARK_PAIRS: dict[Pair, Callable[..., list[BenchmarkCase]]] = {
    Pair.PELT_L2: pair_pelt_l2,
    Pair.FPOP_L2: pair_fpop_l2,
    Pair.PELT_L1: pair_pelt_l1,
    Pair.PELT_1D_GAUSSIAN: pair_pelt_1d_gaussian,
    Pair.PELT_POISSON: pair_pelt_poisson,
    Pair.PELT_RANK: pair_pelt_rank,
    Pair.PELT_LINREG: pair_pelt_linreg,
    Pair.MOVING_WINDOW_L2: pair_moving_window_l2,
    Pair.MOVING_WINDOW_L1: pair_moving_window_l1,
    Pair.MOVING_WINDOW_ESAC: pair_moving_window_esac,
    Pair.MOVING_WINDOW_POISSON: pair_moving_window_poisson,
    Pair.MOVING_WINDOW_RANK: pair_moving_window_rank,
    Pair.MOVING_WINDOW_LINREG: pair_moving_window_linreg,
    Pair.BINSEG_L2_CUSUM: pair_binseg_l2_cusum,
    Pair.BINSEG_L1: pair_binseg_l1,
    Pair.BINSEG_ESAC: pair_binseg_esac,
    Pair.BINSEG_POISSON: pair_binseg_poisson,
    Pair.BINSEG_RANK: pair_binseg_rank,
    Pair.BINSEG_LINREG: pair_binseg_linreg,
    Pair.BINSEG_CONTINUOUS_LINEAR_TREND: pair_binseg_continuous_linear_trend,
    Pair.BINSEG_MV_GAUSSIAN: pair_binseg_mv_gaussian,
    Pair.PELT_MV_GAUSSIAN: pair_pelt_mv_gaussian,
    Pair.MOVING_WINDOW_MV_GAUSSIAN: pair_moving_window_mv_gaussian,
    Pair.MOVING_WINDOW_CONTINUOUS_LINEAR_TREND: (
        pair_moving_window_continuous_linear_trend
    ),
}

#: Pairs that don't support data with more than one column (p > 1).
#: Pairs in this set will only receive univariate (p=1) problems.
NON_MULTIVARIATE_PAIRS: set[Pair] = {
    Pair.PELT_1D_GAUSSIAN,
    Pair.FPOP_L2,
    Pair.BINSEG_CONTINUOUS_LINEAR_TREND,
    Pair.MOVING_WINDOW_CONTINUOUS_LINEAR_TREND,
}

#: Pairs that ONLY make sense for multivariate data (p > 1).
#: Pairs in this set will only receive problems where p > 1.
MULTIVARIATE_ONLY_PAIRS: set[Pair] = {
    Pair.MOVING_WINDOW_ESAC,
    Pair.MOVING_WINDOW_RANK,
    Pair.BINSEG_ESAC,
    Pair.BINSEG_RANK,
    Pair.PELT_RANK,
    Pair.PELT_LINREG,
    Pair.MOVING_WINDOW_LINREG,
    Pair.BINSEG_LINREG,
}

PAIR_CATEGORIES: dict[str, list[Pair]] = {
    "mean_change": [
        Pair.PELT_L2,
        Pair.FPOP_L2,
        Pair.PELT_L1,
        Pair.PELT_POISSON,
        Pair.MOVING_WINDOW_L2,
        Pair.MOVING_WINDOW_L1,
        Pair.MOVING_WINDOW_POISSON,
        Pair.BINSEG_L2_CUSUM,
        Pair.BINSEG_L1,
        Pair.BINSEG_POISSON,
    ],
    "needs_min_segment_length": [
        Pair.PELT_1D_GAUSSIAN,
        # Pair.PELT_LINEAR_TREND,
        Pair.PELT_RANK,
        Pair.PELT_LINREG,
        Pair.MOVING_WINDOW_LINREG,
        Pair.BINSEG_LINREG,
    ],
    "multivariate": [
        Pair.MOVING_WINDOW_RANK,
    ],
    "mv_gaussian": [
        Pair.PELT_MV_GAUSSIAN,
        Pair.MOVING_WINDOW_MV_GAUSSIAN,
        Pair.BINSEG_MV_GAUSSIAN,
    ],
    "linear_regression": [
        Pair.PELT_LINREG,
        Pair.MOVING_WINDOW_LINREG,
        Pair.BINSEG_LINREG,
    ],
    "continuous_linear_trend": [
        Pair.MOVING_WINDOW_CONTINUOUS_LINEAR_TREND,
        Pair.BINSEG_CONTINUOUS_LINEAR_TREND,
    ],
    "multivariate_dimension": [
        Pair.PELT_L2,
        Pair.MOVING_WINDOW_L2,
        Pair.BINSEG_L2_CUSUM,
        Pair.MOVING_WINDOW_ESAC,
        Pair.BINSEG_ESAC,
        Pair.PELT_MV_GAUSSIAN,
        Pair.MOVING_WINDOW_MV_GAUSSIAN,
        Pair.BINSEG_MV_GAUSSIAN,
        Pair.PELT_RANK,
        Pair.MOVING_WINDOW_RANK,
        Pair.BINSEG_RANK,
    ],
}


def collect_cases(
    packages: list[str] | None = None,
    pairs: list[Pair] | None = None,
    categories: list[str] | None = None,
    *,
    n_samples_list: list[int],
    include_fit: bool = True,
    min_segment_length: int = 1,
    dimensions: list[int] | None = None,
    distributions: list[str] | None = None,
) -> list[BenchmarkCase]:
    """Collect benchmark cases, optionally filtered by package, pair, or category.

    Parameters
    ----------
    packages:
        Filter to only include cases from these packages (``"ruptures"``,
        ``"skchange"``). ``None`` means both.
    pairs:
        List of :class:`Pair` members to include. ``None`` means all pairs.
    categories:
        Filter pairs by category (``"mean_change"``, ``"needs_min_segment_length"``,
        ``"multivariate"``, ``"mv_gaussian"``).
        ``None`` means all categories.  When both *pairs* and *categories*
        are given, the union is used.
    n_samples_list:
        List of sample sizes to benchmark (required).
    include_fit:
        If ``True`` (default), the timed operation includes both ``fit`` and
        ``predict``.  If ``False``, only ``predict`` is timed.
    min_segment_length:
        Minimum segment length for the detector (default: 1).  Maps to
        ``min_size`` in ruptures and ``min_segment_length`` in skchange PELT.
    dimensions:
        List of data dimensionalities (number of columns) to benchmark.
        Default: ``[1]``.
    distributions:
        List of distribution names for the null-case data.  ``None`` uses
        the default (``["normal"]``).
    """
    if dimensions is None:
        dimensions = [1]

    # Generate problems with all requested dimensions
    problems = _make_problems(
        n_samples_list, n_columns_list=dimensions, distributions=distributions
    )

    # Resolve which pairs to run
    selected_pairs: list[Pair] = []
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
    seen: set[Pair] = set()
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
        # Filter problems by dimensionality support
        if p in NON_MULTIVARIATE_PAIRS:
            pair_problems = [
                prob for prob in problems if prob.dataset_config.n_columns == 1
            ]
        elif p in MULTIVARIATE_ONLY_PAIRS:
            pair_problems = [
                prob for prob in problems if prob.dataset_config.n_columns > 1
            ]
        else:
            pair_problems = problems

        if not pair_problems:
            continue

        cases.extend(
            BENCHMARK_PAIRS[p](
                pair_problems,
                include_fit=include_fit,
                min_segment_length=min_segment_length,
            )
        )

    # Optionally filter by package (ruptures / skchange)
    if packages:
        cases = [c for c in cases if c.package in packages]

    return cases
