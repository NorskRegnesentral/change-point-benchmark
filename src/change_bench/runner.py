"""Benchmark runner using timeit for timing, with statistics collection.

This module replaces the pytest-benchmark infrastructure with a standalone
runner that collects mean, std, median, and min timing results and stores
them in a Parquet file.
"""

from __future__ import annotations

import gc
import statistics
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Timing results for a single benchmark case."""

    package: str
    cpd_algorithm: str
    name: str
    n_samples: int
    n_changepoints: int
    data_dimension: int
    include_fit: bool
    min_segment_length: int
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
            "package": self.package,
            "cpd_algorithm": self.cpd_algorithm,
            "name": self.name,
            "n_samples": self.n_samples,
            "n_changepoints": self.n_changepoints,
            "data_dimension": self.data_dimension,
            "include_fit": self.include_fit,
            "min_segment_length": self.min_segment_length,
            "n_runs": self.n_runs,
            "mean_s": self.mean,
            "std_s": self.std,
            "median_s": self.median,
            "min_s": self.min,
        }


def run_benchmark(
    *,
    package: str,
    cpd_algorithm: str,
    name: str,
    n_samples: int,
    n_changepoints: int,
    data_dimension: int,
    include_fit: bool,
    min_segment_length: int,
    setup: Callable[[], tuple[tuple, dict]],
    func: Callable,
    n_runs: int,
) -> BenchmarkResult:
    """Run a single benchmark case *n_runs* times and return statistics.

    Parameters
    ----------
    package:
        Library package (e.g. ``"ruptures"`` or ``"skchange"``).
    cpd_algorithm:
        Name of the comparison algorithm pair.
    name:
        Human-readable benchmark name (e.g. distribution label).
    n_samples:
        Number of samples in the dataset.
    n_changepoints:
        Number of true change points in the dataset.
    data_dimension:
        Dimensionality of the time series (number of columns).
    include_fit:
        Whether the timed operation includes fitting.
    min_segment_length:
        Minimum segment length used by the detector.
    setup:
        Callable returning ``(args, kwargs)`` to pass to *func*.
        Called once per run so each iteration gets a fresh setup.
    func:
        The function to time.  Called as ``func(*args, **kwargs)``.
    n_runs:
        Number of timed repetitions.
    """
    result = BenchmarkResult(
        package=package,
        cpd_algorithm=cpd_algorithm,
        name=name,
        n_samples=n_samples,
        n_changepoints=n_changepoints,
        data_dimension=data_dimension,
        include_fit=include_fit,
        min_segment_length=min_segment_length,
        n_runs=n_runs,
    )

    for _ in range(n_runs):
        args, kwargs = setup()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        t0 = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if gc_was_enabled:
            gc.enable()
        result.times.append(elapsed)

    return result
