"""Benchmark runner using timeit for timing, with statistics collection.

This module replaces the pytest-benchmark infrastructure with a standalone
runner that collects mean, std, median, and min timing results and stores
them in a Parquet file.
"""

from __future__ import annotations

import statistics
import timeit
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Timing results for a single benchmark case."""

    group: str
    pair: str
    name: str
    n_runs: int
    times: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times)

    @property
    def std(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def min(self) -> float:
        return min(self.times)

    def as_dict(self) -> dict:
        return {
            "group": self.group,
            "pair": self.pair,
            "name": self.name,
            "n_runs": self.n_runs,
            "mean_s": self.mean,
            "std_s": self.std,
            "median_s": self.median,
            "min_s": self.min,
        }


def run_benchmark(
    *,
    group: str,
    pair: str,
    name: str,
    setup: Callable[[], tuple[tuple, dict]],
    func: Callable,
    n_runs: int,
) -> BenchmarkResult:
    """Run a single benchmark case *n_runs* times and return statistics.

    Parameters
    ----------
    group:
        Logical grouping (e.g. ``"ruptures"`` or ``"skchange"``).
    pair:
        Name of the comparison pair this case belongs to.
    name:
        Human-readable benchmark name.
    setup:
        Callable returning ``(args, kwargs)`` to pass to *func*.
        Called once per run so each iteration gets a fresh setup.
    func:
        The function to time.  Called as ``func(*args, **kwargs)``.
    n_runs:
        Number of timed repetitions.
    """
    result = BenchmarkResult(group=group, pair=pair, name=name, n_runs=n_runs)

    for _ in range(n_runs):
        args, kwargs = setup()
        elapsed = timeit.timeit(lambda: func(*args, **kwargs), number=1)
        result.times.append(elapsed)

    return result
