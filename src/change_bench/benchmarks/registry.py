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

from change_bench.benchmarks.comparison_pairs import (
    BenchmarkCase,
    pair_binseg,
    pair_moving_window_l1,
    pair_moving_window_l2,
    pair_moving_window_rank,
    pair_pelt_1d_gaussian,
    pair_pelt_l2,
    pair_pelt_poisson,
    pair_pelt_rank,
)
from change_bench.problems.base import BenchmarkProblem, make_null_problems

# ---------------------------------------------------------------------------
# Problem batteries
# ---------------------------------------------------------------------------

small_n_samples_list = [100, 250, 500, 750, 1000]
large_n_samples_list = [1500, 2500, 5000, 7500, 10_000]

#: All supported null-case distributions.
ALL_DISTRIBUTIONS: list[str] = ["normal", "t", "gamma", "laplace", "exponential"]

#: Default distributions per problem set.
DEFAULT_DISTRIBUTIONS: dict[str, list[str]] = {
    "small": ["normal"],
    "full": ALL_DISTRIBUTIONS,
}


def _make_problems(
    problem_set: str,
    n_columns_list: list[int],
    distributions: list[str] | None = None,
) -> list[BenchmarkProblem]:
    """Create problem battery with given dimensions."""
    n_samples = (
        small_n_samples_list
        if problem_set == "small"
        else small_n_samples_list + large_n_samples_list
    )
    if distributions is None:
        distributions = DEFAULT_DISTRIBUTIONS.get(problem_set, ALL_DISTRIBUTIONS)
    return make_null_problems(
        n_samples_list=n_samples,
        distributions=distributions,
        scale=1.0,
        n_columns_list=n_columns_list,
    )


# ---------------------------------------------------------------------------
# Registry: maps pair name -> factory function
# ---------------------------------------------------------------------------
BENCHMARK_PAIRS: dict[str, Callable[..., list[BenchmarkCase]]] = {
    "pelt_l2": pair_pelt_l2,
    "pelt_1d_gaussian": pair_pelt_1d_gaussian,
    "pelt_poisson": pair_pelt_poisson,
    "pelt_rank": pair_pelt_rank,
    "moving_window_l2": pair_moving_window_l2,
    "moving_window_l1": pair_moving_window_l1,
    "moving_window_rank": pair_moving_window_rank,
    "binseg": pair_binseg,
}

#: Pairs that don't support data with more than one column (p > 1).
#: Pairs in this set will only receive univariate (p=1) problems.
NON_MULTIVARIATE_PAIRS: set[str] = {"pelt_1d_gaussian"}

#: Pairs that ONLY make sense for multivariate data (p > 1).
#: Pairs in this set will only receive problems where p > 1.
MULTIVARIATE_ONLY_PAIRS: set[str] = {"moving_window_rank", "pelt_rank"}

PAIR_CATEGORIES: dict[str, list[str]] = {
    "mean_change": [
        "pelt_l2",
        "pelt_poisson",
        "moving_window_l2",
        "moving_window_l1",
        "binseg",
    ],
    "needs_min_segment_length": [
        "pelt_1d_gaussian",
        # "pelt_linear_trend",
    ],
    "multivariate": [
        "moving_window_rank",
        "pelt_rank",
    ],
}


def collect_cases(
    packages: list[str] | None = None,
    pairs: list[str] | None = None,
    categories: list[str] | None = None,
    problem_set: str = "small",
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
        List of comparison-pair names to include. ``None`` means all pairs.
    categories:
        Filter pairs by category (``"mean_change"``, ``"needs_min_segment_length"``,
        ``"multivariate"``).
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
    dimensions:
        List of data dimensionalities (number of columns) to benchmark.
        Default: ``[1]``.
    distributions:
        List of distribution names for the null-case data.  ``None`` uses
        the default for the chosen *problem_set* (``"normal"`` only for
        ``"small"``, all distributions for ``"full"``).
    """
    if dimensions is None:
        dimensions = [1]

    # Generate problems with all requested dimensions
    problems = _make_problems(
        problem_set, n_columns_list=dimensions, distributions=distributions
    )

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
